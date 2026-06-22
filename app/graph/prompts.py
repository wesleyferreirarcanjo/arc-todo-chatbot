from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient
from app.chatbot_settings import ChatbotRuntimeSettings
from app.graph.state import ChatGraphState
from app.history import trim_messages
from app.streaming import get_stream_handler
from app.task_id_resolver import is_friendly_task_id, is_uuid, normalize_friendly_task_id
from app.tools.todo_tools import TodoTools, execute_todo_tool

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are Arc Todo assistant. Help users manage tasks in Arc Todo.
You can answer questions about todos and perform actions using available tools.
When organization or project context is missing, ask the user to clarify or use list_organizations and list_projects first.
Keep responses concise and practical."""

MAX_PLANNED_ACTIONS = 5

PLANNER_PROMPT = """Analyze the latest user message and decide whether todo API tools are needed.

Return JSON only with this shape:
{"route":"direct"|"tools","actions":[{"intent":string,"tool_name":string,"tool_arguments":object}]}

You may plan up to 5 ordered actions when the user asks for multiple things in one message.
Example: create two tasks with descriptions -> one create_tasks action with both tasks and descriptions filled in.
Example: create a task and mark another selected task done -> two actions: create_task, then update_tasks.

Available tools:
- list_organizations {}
- list_projects {"organization_id": string}
- list_tasks {"organization_id": string|null, "project_id": string|null, "status": string|null, "criticity": string|null, "parent_task_id": string|null}
- get_task {"organization_id": string, "project_id": string, "task_id": string}
- get_tasks {"task_ids": string[]|null} — fetch multiple selected tasks; omit task_ids to use all taskIds from Selected task context
- create_task {"organization_id": string|null, "project_id": string|null, "title": string, "description": string|null, "status": "todo"|"in_progress"|"done", "criticity": "low"|"medium"|"high"|"critical", "due_date": string|null, "parent_task_id": string|null}
- create_tasks {"organization_id": string|null, "project_id": string|null, "tasks": [{"title": string, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null, "parent_task_id": string|null}]} — create multiple tasks in one request
- update_task {"organization_id": string, "project_id": string, "task_id": string, "title": string|null, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null, "parent_task_id": string|null}
- update_tasks {"task_ids": string[]|null, "title": string|null, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null, "parent_task_id": string|null, "tasks": [{"task_id": string, "title": string|null, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null, "parent_task_id": string|null}]|null} — for multiple selected tasks, prefer a tasks array with one entry per task_id when values differ (especially descriptions); shared fields may apply to all only when every task should get the same value
- move_task {"task_id": string, "target_project_id": string|null, "target_project_hint": string|null} — move one selected task to another project; use organization_id/project_id from Selected task context for the task's current location
- move_tasks {"task_ids": string[]|null, "target_project_id": string|null, "target_project_hint": string|null} — move multiple selected tasks to another project
- delete_task {"organization_id": string, "project_id": string, "task_id": string}
- delete_tasks {"task_ids": string[]|null} — delete multiple selected tasks; omit task_ids to use all taskIds from Selected task context
- list_knowledge {"scope": "general"|"organization"|"project"|"person", "organization_id": string|null, "project_id": string|null, "person_id": string|null, "file_name": string|null, "mime_type": string|null, "has_attachments": boolean|null}
- get_knowledge {"scope": string, "organization_id": string|null, "project_id": string|null, "person_id": string|null, "knowledge_id": string}
- create_knowledge {"scope": string, "organization_id": string|null, "project_id": string|null, "person_id": string|null, "title": string, "content": string}
- update_knowledge {"scope": string, "organization_id": string|null, "project_id": string|null, "person_id": string|null, "knowledge_id": string, "title": string|null, "content": string|null}
- list_persons {"organization_id": string|null}
- get_person {"person_id": string, "organization_id": string|null}
- trigger_rag_index_sync {} — queue a RAG index sync/reconcile job; use only when the user explicitly asks to refresh or reindex knowledge

Use knowledge/person tools when the user asks about documentation, notes, contacts, or people rather than tasks.
Use trigger_rag_index_sync only when the user explicitly asks to refresh, reindex, or sync the knowledge index.
When Retrieved knowledge context is present, prefer it for documentation answers and cite source filenames when helpful.

Task identifiers may be official UUID taskId values or friendly display IDs like arc-1 or #arc-1. Selected task context includes displayId when available.
Use move_tasks (not move_task) when the user wants to move more than one selected task.
When Selected task context is present, keep the task in its current organization_id/project_id and only change the destination project for move requests.
When Selected task context lists multiple tasks and the user asks to add or create descriptions, return update_tasks with a tasks array containing a distinct description for each task_id based on that task's title.
When the user asks to fix, change, or rewrite descriptions for selected tasks, call update_tasks immediately with appropriate per-task descriptions; do not ask for confirmation first.
When the user asks for a description (including creative or matching descriptions) and none is provided, infer a reasonable description from the task title and request — do not ask the user to supply it.
Use create_tasks (not create_task) when the user asks to create more than one task in the same project.
Tasks support one parent level: a parent task may have direct subtasks via parent_task_id. Subtasks cannot have subtasks.
When the user asks to add a subtask under a selected parent task, use create_task with parent_task_id from Selected task context.
When the user asks to create a parent with subtasks, use create_task for the parent first, then create_tasks with the same parent_task_id from the created parent id.
When the user asks to make one task a subtask of another, use update_task on the child with parent_task_id set to the parent task id.
When two tasks are selected and the user wants hierarchy, treat the parent as the task they call "parent"/"under"/"of", and the child as the task to attach.
Use update_task with parent_task_id null to detach a subtask from its parent.
Use route "direct" for greetings, general help, or when no API action is needed.
Ask the user a clarifying question only when organization/project scope, destructive intent, or target task identity is genuinely ambiguous.
Prefer provided organization_id and project_id context when present; omit organization_id/project_id from tool_arguments when context is already provided.
Never invent organization_id or project_id values from organization or project names in the user message — names like "arc-todo" are not UUIDs. Omit those fields and rely on context, or use list_organizations/list_projects first.
The API field for priority is criticity (low|medium|high|critical), not priority."""

RESPONSE_PROMPT = """Write a concise assistant reply for the user based on the conversation and verified action results.
Do not mention internal tool names unless helpful.
Never claim that tasks were created, updated, or deleted unless verified action results confirm it.
If an action failed or was partial, say so honestly.
If no verified action results are present for a mutation request, say you could not perform the action yet.
When Retrieved knowledge context is present, ground documentation answers in those excerpts and mention source filenames or titles when helpful.
If Retrieved knowledge context notes that retrieval failed, continue answering from live task data and mention that indexed knowledge was unavailable."""

MUTATION_TOOLS = {
    "create_task",
    "create_tasks",
    "update_task",
    "update_tasks",
    "move_task",
    "move_tasks",
    "delete_task",
    "delete_tasks",
}

UPDATE_TASK_FIELDS = ("title", "description", "status", "criticity", "due_date", "parent_task_id")

PLANNER_MUTATION_RETRY = (
    "\nIMPORTANT: The user is asking to create, update, or delete tasks. "
    'Return route "tools" with one or more actions in the actions array. '
    "Infer missing descriptions when requested. Do not answer directly."
)

CONFIRMATION_PATTERN = re.compile(
    r"^(?:yes|yep|yeah|sure|ok(?:ay)?|please|go ahead|do it|apply(?: them)?|"
    r"confirm(?:ed)?|sounds good|that works|looks good)[\s!.?]*$",
    re.I,
)
