from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient
from app.chatbot_settings import ChatbotRuntimeSettings
from app.graph.state import ChatGraphState
from app.task_id_resolver import is_friendly_task_id, normalize_friendly_task_id
from app.tools.todo_tools import TodoTools, execute_todo_tool

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
If no verified action results are present for a mutation request, say you could not perform the action yet."""

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


def build_model(runtime: ChatbotRuntimeSettings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=runtime.api_key,
        base_url=runtime.base_url.rstrip("/") + "/",
        model=runtime.model,
        temperature=runtime.temperature,
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if "priority" in normalized and "criticity" not in normalized:
        normalized["criticity"] = normalized.pop("priority")
    if "parent_id" in normalized and "parent_task_id" not in normalized:
        normalized["parent_task_id"] = normalized.pop("parent_id")
    raw_tasks = normalized.get("tasks")
    if isinstance(raw_tasks, list):
        normalized["tasks"] = [
            _normalize_tool_arguments(task) if isinstance(task, dict) else task
            for task in raw_tasks
        ]
    return normalized


def _normalize_planner_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("actions")
    normalized: list[dict[str, Any]] = []
    if isinstance(actions, list):
        for item in actions[:MAX_PLANNED_ACTIONS]:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            normalized.append(
                {
                    "intent": str(item.get("intent") or tool_name),
                    "tool_name": tool_name,
                    "tool_arguments": _normalize_tool_arguments(
                        item.get("tool_arguments") or {}
                    ),
                }
            )
    if normalized:
        return normalized

    tool_name = payload.get("tool_name")
    if isinstance(tool_name, str) and tool_name:
        return [
            {
                "intent": tool_name,
                "tool_name": tool_name,
                "tool_arguments": _normalize_tool_arguments(
                    payload.get("tool_arguments") or {}
                ),
            }
        ]
    return []


def _update_arguments_have_fields(tool_name: str, arguments: dict[str, Any]) -> bool:
    if tool_name == "update_task":
        return any(arguments.get(key) is not None for key in UPDATE_TASK_FIELDS)
    if tool_name == "update_tasks":
        raw_tasks = arguments.get("tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            return any(
                any(item.get(key) is not None for key in UPDATE_TASK_FIELDS)
                for item in raw_tasks
                if isinstance(item, dict)
            )
        return any(arguments.get(key) is not None for key in UPDATE_TASK_FIELDS)
    return True


def _format_task_context_line(task: dict[str, Any], fallback_title: str) -> str:
    title = task.get("title") or fallback_title
    status = task.get("status", "unknown")
    criticity = task.get("criticity", "unknown")
    due_date = task.get("dueDate") or task.get("due_date") or "none"
    description = (task.get("description") or "").strip() or "none"
    task_id = task.get("id", "unknown")
    display_id = task.get("displayId") or task.get("display_id")
    organization_id = task.get("organizationId") or task.get("organization_id")
    project_id = task.get("projectId") or task.get("project_id")
    parent_task_id = task.get("parentTaskId") or task.get("parent_task_id")
    subtasks = task.get("subtasks") or []
    subtask_progress = task.get("subtaskProgress") or task.get("subtask_progress")
    lines = [
        f"- taskId: {task_id}",
    ]
    if display_id:
        lines.append(f"  displayId: {display_id}")
    lines.extend([
        f"  title: {title}",
        f"  status: {status}",
        f"  criticity: {criticity}",
        f"  organizationId: {organization_id}",
        f"  projectId: {project_id}",
        f"  dueDate: {due_date}",
        f"  description: {description}",
    ])
    if parent_task_id:
        lines.append(f"  parentTaskId: {parent_task_id}")
    if isinstance(subtask_progress, dict) and subtask_progress.get("total"):
        lines.append(
            "  subtaskProgress: "
            f"{subtask_progress.get('done', 0)}/{subtask_progress.get('total', 0)} done"
        )
    if isinstance(subtasks, list) and subtasks:
        lines.append("  subtasks:")
        for subtask in subtasks[:8]:
            if not isinstance(subtask, dict):
                continue
            sub_title = subtask.get("title") or subtask.get("id") or "subtask"
            sub_status = subtask.get("status") or "unknown"
            lines.append(f"    - {sub_title} ({sub_status})")
    return "\n".join(lines)


async def _build_task_context_text(
    *,
    user_token: str,
    task_refs: list[dict[str, str]],
) -> str:
    if not task_refs:
        return ""

    client = ArcTodoClient(user_token=user_token)
    tools = TodoTools(client)
    lines: list[str] = []

    for ref in task_refs:
        task_id = ref.get("taskId") or ref.get("task_id")
        organization_id = ref.get("organizationId") or ref.get("organization_id")
        project_id = ref.get("projectId") or ref.get("project_id")
        title = ref.get("title") or task_id or "Task"

        if not task_id or not organization_id or not project_id:
            lines.append(
                f"- taskId: {task_id or 'unknown'}\n"
                f"  title: {title}\n"
                f"  note: missing organization or project scope"
            )
            continue

        try:
            task = await tools.get_task(
                organization_id=organization_id,
                project_id=project_id,
                task_id=task_id,
            )
            lines.append(_format_task_context_line(task, title))
        except Exception:
            lines.append(
                f"- taskId: {task_id}\n"
                f"  title: {title}\n"
                f"  organizationId: {organization_id}\n"
                f"  projectId: {project_id}\n"
                f"  note: task details could not be loaded"
            )

    return "Selected task context:\n" + "\n".join(lines)


def _ref_task_id(ref: dict[str, str]) -> str | None:
    return ref.get("taskId") or ref.get("task_id")


def _ref_display_id(ref: dict[str, str]) -> str | None:
    return ref.get("displayId") or ref.get("display_id")


def _normalize_task_identifier(value: str) -> str:
    if is_friendly_task_id(value):
        return normalize_friendly_task_id(value)
    return value


def _ref_matches_task_identifier(ref: dict[str, str], identifier: str) -> bool:
    if not identifier:
        return False

    normalized = _normalize_task_identifier(identifier)
    task_id = _ref_task_id(ref)
    if task_id and (task_id == identifier or task_id == normalized):
        return True

    display_id = _ref_display_id(ref)
    if display_id and _normalize_task_identifier(display_id) == normalized:
        return True

    return False


def _task_ref_lookup(task_refs: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for ref in task_refs:
        task_id = _ref_task_id(ref)
        if task_id:
            lookup[task_id] = ref
        display_id = _ref_display_id(ref)
        if display_id:
            lookup[display_id] = ref
            lookup[_normalize_task_identifier(display_id)] = ref
    return lookup


def _ref_organization_id(ref: dict[str, str]) -> str | None:
    return ref.get("organizationId") or ref.get("organization_id")


def _ref_project_id(ref: dict[str, str]) -> str | None:
    return ref.get("projectId") or ref.get("project_id")


def _apply_task_ref_source_scope(
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
) -> dict[str, Any]:
    if not task_refs:
        return arguments

    resolved = dict(arguments)
    task_id = resolved.get("task_id")
    if task_id:
        for ref in task_refs:
            if not _ref_matches_task_identifier(ref, task_id):
                continue
            org_id = _ref_organization_id(ref)
            project_id = _ref_project_id(ref)
            if _is_uuid(org_id):
                resolved["organization_id"] = org_id
            if _is_uuid(project_id):
                resolved["project_id"] = project_id
            break
        return resolved

    if len(task_refs) == 1:
        ref = task_refs[0]
        org_id = _ref_organization_id(ref)
        project_id = _ref_project_id(ref)
        task_ref_id = _ref_task_id(ref)
        if _is_uuid(org_id):
            resolved["organization_id"] = org_id
        if _is_uuid(project_id):
            resolved["project_id"] = project_id
        if task_ref_id and not resolved.get("task_id"):
            resolved["task_id"] = task_ref_id

    return resolved


def _looks_like_bulk_selected(message: str) -> bool:
    return bool(re.search(r"\b(all|these|selected|them)\b", message, re.I))


def _looks_like_task_mutation(message: str) -> bool:
    return bool(
        re.search(
            r"\b(create|add|make|new|update|delete|remove|mark|complete|move)\b.*\btasks?\b"
            r"|\btasks?\b.*\b(create|add|make|new|update|delete|remove|mark|complete|move)\b"
            r"|\bmove\b.*\bto\b",
            message,
            re.I,
        )
    )


def _looks_like_move_mutation(message: str) -> bool:
    return bool(re.search(r"\bmove\b", message, re.I))


def _extract_move_target_hint(message: str) -> str | None:
    for pattern in (
        r"\bmove\s+(?:this\s+)?(?:task\s+)?to\s+(.+?)\s*\.?\s*$",
        r"\bto\s+([a-z0-9][a-z0-9\s-]+)\s*\.?\s*$",
    ):
        match = re.search(pattern, message, re.I | re.M)
        if match:
            hint = match.group(1).strip()
            if len(hint) >= 2:
                return hint
    return None


def _looks_like_multi_create(message: str) -> bool:
    return bool(
        re.search(
            r"\b(two|both|multiple|another|second|2)\b.*\btasks?\b"
            r"|\btasks?\b.*\b(two|both|multiple|another|second|2)\b"
            r"|\banother task\b",
            message,
            re.I,
        )
    )


def _looks_like_update_mutation(message: str) -> bool:
    if re.search(r"\bcreate\b", message, re.I) and _looks_like_subtask_mutation(message):
        return False
    return bool(
        re.search(
            r"\b(update|edit|change|set|mark|complete|describe|fix|correct|rewrite)\b"
            r"|\b(?:add|create)\s+(?:a\s+)?(?:description|detail|note|comment|due\s*date)\b"
            r"|\b(description|details|notes)\s+(?:for|to|of)\b"
            r"|\bfix\s+(?:the\s+)?description\b",
            message,
            re.I,
        )
    )


CONFIRMATION_PATTERN = re.compile(
    r"^(?:yes|yep|yeah|sure|ok(?:ay)?|please|go ahead|do it|apply(?: them)?|"
    r"confirm(?:ed)?|sounds good|that works|looks good)[\s!.?]*$",
    re.I,
)


def _looks_like_mutation_confirmation(message: str) -> bool:
    return bool(CONFIRMATION_PATTERN.match(message.strip()))


def _find_last_assistant_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return ""


def _extract_proposed_descriptions_from_assistant(text: str) -> list[tuple[str, str]]:
    proposals: list[tuple[str, str]] = []
    for match in re.finditer(
        r"\d+\.\s+\*{0,2}([^*\n]+?)\*{0,2}\s*[—–-]\s*(?:\*{0,2})[\"']?(.+?)[\"']?(?:\*{0,2})?"
        r"(?=\n\d+\.|\n\n|Would you like|\Z)",
        text,
        re.I | re.S,
    ):
        title = match.group(1).strip()
        description = match.group(2).strip().strip("*").strip()
        if title and description and description.lower() not in {"description updated.", "updated."}:
            proposals.append((title, description))
    return proposals


def _title_matches_ref(proposal_title: str, ref_title: str) -> bool:
    left = _normalize_scope_name(proposal_title)
    right = _normalize_scope_name(ref_title)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) >= 10 and shorter[:10] in longer:
        return True
    if len(shorter) >= 6 and longer.startswith(shorter[:6]):
        return True
    return False


def _build_tasks_from_proposed_descriptions(
    proposals: list[tuple[str, str]],
    task_refs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for proposal_title, description in proposals:
        for ref in task_refs:
            task_id = _ref_task_id(ref)
            if not task_id or task_id in used_ids:
                continue
            ref_title = ref.get("title") or ""
            if not _title_matches_ref(proposal_title, ref_title):
                continue
            tasks.append({"task_id": task_id, "description": description})
            used_ids.add(task_id)
            break
    return tasks


def _resolve_confirmation_update_arguments(
    messages: list[dict[str, str]],
    task_refs: list[dict[str, str]],
) -> dict[str, Any] | None:
    last_assistant = _find_last_assistant_message(messages)
    if not last_assistant:
        return None
    proposals = _extract_proposed_descriptions_from_assistant(last_assistant)
    if not proposals:
        return None
    tasks = _build_tasks_from_proposed_descriptions(proposals, task_refs)
    if not tasks:
        return None
    return {"tasks": tasks}


def _filter_task_refs_by_message(
    task_refs: list[dict[str, str]],
    message: str,
) -> list[dict[str, str]]:
    if len(task_refs) <= 1:
        return task_refs

    message_norm = _normalize_scope_name(message)
    if not message_norm:
        return task_refs

    matched: list[dict[str, str]] = []
    for ref in task_refs:
        title = ref.get("title") or ""
        title_norm = _normalize_scope_name(title)
        if not title_norm:
            continue
        if title_norm in message_norm or message_norm in title_norm:
            matched.append(ref)
            continue
        if len(title_norm) >= 8 and title_norm[:8] in message_norm:
            matched.append(ref)
            continue
        if len(title_norm) >= 6 and title_norm[:6] in message_norm:
            matched.append(ref)

    return matched if matched else task_refs


def _effective_task_refs(
    task_refs: list[dict[str, str]],
    latest_user_message: str,
) -> list[dict[str, str]]:
    if len(task_refs) <= 1 or _looks_like_bulk_selected(latest_user_message):
        return task_refs
    return _filter_task_refs_by_message(task_refs, latest_user_message)


def _needs_mutation_tool_result(state: ChatGraphState) -> bool:
    latest = state.get("latest_user_message", "")
    if _looks_like_task_mutation(latest):
        return True
    if state.get("task_refs") and _looks_like_update_mutation(latest):
        return True
    if _looks_like_mutation_confirmation(latest) and state.get("task_refs"):
        last_assistant = _find_last_assistant_message(state.get("messages", []))
        if last_assistant and (
            _extract_proposed_descriptions_from_assistant(last_assistant)
            or re.search(r"would you like me to apply", last_assistant, re.I)
        ):
            return True
    return False


def _looks_like_subtask_mutation(message: str) -> bool:
    return bool(re.search(r"\b(?:subtasks?|sub-tasks?|sub tasks?)\b", message, re.I))


def _looks_like_create_parent_with_subtasks(message: str) -> bool:
    if not re.search(r"\bcreate\b", message, re.I):
        return False
    if not _looks_like_subtask_mutation(message):
        return False
    if _looks_like_reparent_mutation(message):
        return False
    return bool(
        re.search(
            r"(?:"
            r"create\s+(?:a\s+)?(?:parent\s+)?tasks?\s+.+\s+and\s+create\s+(?:the\s+)?subtasks?"
            r"|create\s+(?:a\s+)?(?:parent\s+)?tasks?\s+.+\s+with\s+subtasks?"
            r"|\bfor\s+(?:it|them|that)\b"
            r")",
            message,
            re.I,
        )
    )


def _clean_parsed_task_title(title: str) -> str:
    cleaned = title.strip(" .,\"'")
    cleaned = re.sub(
        r"\s+in\s+(?:my\s+)?[^,\n]+?\s+project\s*$",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+in\s+my\s+.+$", "", cleaned, flags=re.I)
    return cleaned.strip(" .,\"'")


def _split_subtask_titles(raw: str) -> list[str]:
    text = raw.strip()
    text = re.sub(r"\s+give\s+.+$", "", text, flags=re.I)
    text = re.sub(
        r"\s+(?:for\s+)?(?:testing(?:\s+purpose)?|each)\b.*$",
        "",
        text,
        flags=re.I,
    )
    if "," in text:
        parts = re.split(r"\s*,\s*", text)
    else:
        parts = re.split(r"\s+and\s+|\s+", text)
    return [part.strip(" .,\"'") for part in parts if part.strip(" .,\"'")]


def _parse_create_parent_with_subtasks(message: str) -> dict[str, Any] | None:
    if not _looks_like_create_parent_with_subtasks(message):
        return None

    patterns = (
        r"create\s+(?:a\s+)?(?:parent\s+)?tasks?\s+(.+?)\s+and\s+create\s+(?:the\s+)?subtasks?\s+(.+?)(?:\s+for\s+(?:it|them|that))?(?:\s+give|\s*$)",
        r"create\s+(?:a\s+)?(?:parent\s+)?tasks?\s+(.+?)\s+with\s+subtasks?\s+(.+?)(?:\s+give|\s*$)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.I | re.S)
        if not match:
            continue
        parent_title = _clean_parsed_task_title(match.group(1))
        subtask_titles = _split_subtask_titles(match.group(2))
        if parent_title and subtask_titles:
            return {
                "parent_title": parent_title,
                "subtask_titles": subtask_titles,
            }
    return None


def _build_create_parent_with_subtasks_actions(
    state: ChatGraphState,
    parsed: dict[str, Any],
) -> list[dict[str, Any]]:
    scope = {
        key: state[key]
        for key in ("organization_id", "project_id")
        if _is_uuid(state.get(key))
    }
    parent_title = parsed["parent_title"]
    return [
        {
            "intent": f"create parent task {parent_title}",
            "tool_name": "create_task",
            "tool_arguments": {**scope, "title": parent_title},
        },
        {
            "intent": f"create subtasks for {parent_title}",
            "tool_name": "create_tasks",
            "tool_arguments": {
                **scope,
                "_parent_from_previous": True,
                "tasks": [{"title": title} for title in parsed["subtask_titles"]],
            },
        },
    ]


def _inject_parent_from_previous(
    arguments: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not arguments.pop("_parent_from_previous", None):
        return arguments

    updated = dict(arguments)
    parent_id = None
    parent_title = None
    for prev in reversed(tool_results):
        if prev.get("tool_name") != "create_task" or not prev.get("success"):
            continue
        result = prev.get("tool_result") or {}
        parent_id = result.get("id")
        parent_title = result.get("title")
        if parent_id:
            break

    if not parent_id:
        return updated

    if parent_title:
        updated["_parent_title"] = parent_title
    tasks = updated.get("tasks")
    if isinstance(tasks, list):
        updated["tasks"] = [
            {**task, "parent_task_id": parent_id}
            if isinstance(task, dict) and not task.get("parent_task_id")
            else task
            for task in tasks
        ]
    return updated


def _create_tasks_have_parent(arguments: dict[str, Any]) -> bool:
    tasks = arguments.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return False
    return all(
        isinstance(task, dict) and task.get("parent_task_id")
        for task in tasks
    )


def _looks_like_reparent_mutation(message: str) -> bool:
    return bool(
        re.search(
            r"\b(?:subtask of|child of|under(?:\s+the)?\s+task|attach(?:\s+to)?|"
            r"make .+ (?:a )?subtask(?:\s+of)?|set parent|move under|detach from parent)\b",
            message,
            re.I,
        )
    )


def _looks_like_detach_subtask(message: str) -> bool:
    return bool(re.search(r"\b(?:detach|remove)\b.*\b(?:parent|subtask)\b", message, re.I))


def _resolve_reparent_arguments(
    message: str,
    task_refs: list[dict[str, str]],
) -> dict[str, Any] | None:
    if _looks_like_detach_subtask(message):
        refs = _effective_task_refs(task_refs, message)
        if len(refs) != 1:
            return None
        child_id = _ref_task_id(refs[0])
        if not child_id:
            return None
        ref = refs[0]
        return {
            "organization_id": ref.get("organizationId") or ref.get("organization_id"),
            "project_id": ref.get("projectId") or ref.get("project_id"),
            "task_id": child_id,
            "parent_task_id": None,
        }

    if not _looks_like_reparent_mutation(message):
        return None

    refs = _effective_task_refs(task_refs, message)
    if len(refs) < 2:
        return None

    child_ref = refs[0]
    parent_ref = refs[1]
    child_id = _ref_task_id(child_ref)
    parent_id = _ref_task_id(parent_ref)
    if not child_id or not parent_id or child_id == parent_id:
        return None

    return {
        "organization_id": child_ref.get("organizationId")
        or child_ref.get("organization_id"),
        "project_id": child_ref.get("projectId") or child_ref.get("project_id"),
        "task_id": child_id,
        "parent_task_id": parent_id,
    }


def _apply_subtask_parent_from_refs(
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    message: str,
) -> dict[str, Any]:
    if not _looks_like_subtask_mutation(message):
        return arguments
    if arguments.get("parent_task_id") or arguments.get("parent_id"):
        return arguments

    refs = _effective_task_refs(task_refs, message)
    if len(refs) != 1:
        return arguments

    parent_id = _ref_task_id(refs[0])
    if not parent_id:
        return arguments

    updated = dict(arguments)
    updated["parent_task_id"] = parent_id
    tasks = updated.get("tasks")
    if isinstance(tasks, list):
        updated["tasks"] = [
            {**task, "parent_task_id": parent_id}
            if isinstance(task, dict) and not task.get("parent_task_id")
            else task
            for task in tasks
        ]
    if not updated.get("organization_id"):
        updated["organization_id"] = refs[0].get("organizationId") or refs[0].get(
            "organization_id"
        )
    if not updated.get("project_id"):
        updated["project_id"] = refs[0].get("projectId") or refs[0].get("project_id")
    return updated


def _looks_like_create_mutation(message: str) -> bool:
    if _looks_like_update_mutation(message):
        return False
    if _looks_like_subtask_mutation(message):
        return True
    if re.search(r"\b(create|make|new)\s+(?:a\s+)?tasks?\b", message, re.I):
        return True
    if re.search(r"\badd\s+(?:a\s+)?tasks?\b", message, re.I):
        return True
    if re.search(r"\banother\s+task\b", message, re.I):
        return True
    if _looks_like_multi_create(message):
        return True
    return False


def _action_succeeded(
    tool_name: str,
    result: Any,
    error: str | None,
) -> tuple[bool, bool]:
    partial = False
    if tool_name not in MUTATION_TOOLS:
        return (error is None and result is not None, False)
    if error:
        return (False, False)
    if result is None:
        return (False, False)
    if not isinstance(result, dict):
        return (True, False)

    if tool_name == "create_task":
        return (bool(result.get("id")), False)
    if tool_name == "create_tasks":
        created = result.get("created") or []
        failed = result.get("failed") or []
        if created and failed:
            return (True, True)
        return (len(created) > 0 and not failed, False)
    if tool_name == "update_task":
        return (bool(result.get("id")), False)
    if tool_name == "update_tasks":
        updated = result.get("updated") or []
        failed = result.get("failed") or []
        if updated and failed:
            return (True, True)
        return (len(updated) > 0 and not failed, False)
    if tool_name == "move_task":
        return (bool(result.get("id")), False)
    if tool_name == "move_tasks":
        moved = result.get("moved") or []
        failed = result.get("failed") or []
        if moved and failed:
            return (True, True)
        return (len(moved) > 0 and not failed, False)
    if tool_name == "delete_task":
        return (True, False)
    if tool_name == "delete_tasks":
        deleted = result.get("deleted") or []
        failed = result.get("failed") or []
        if deleted and failed:
            return (True, True)
        return (len(deleted) > 0 and not failed, False)
    return (True, False)


def _mutation_succeeded(state: ChatGraphState) -> bool | None:
    if state.get("route") != "tools":
        return None

    tool_results = state.get("tool_results") or []
    if tool_results:
        mutation_results = [
            item for item in tool_results if item.get("tool_name") in MUTATION_TOOLS
        ]
        if not mutation_results:
            return None
        if all(item.get("success") for item in mutation_results):
            return True
        if any(item.get("success") for item in mutation_results):
            return False
        return False

    tool_name = state.get("tool_name")
    if tool_name not in MUTATION_TOOLS:
        return None
    success, _partial = _action_succeeded(
        tool_name,
        state.get("tool_result"),
        state.get("error"),
    )
    return success


def _build_mutation_failure_response(state: ChatGraphState) -> str:
    error = state.get("error")
    if error:
        return f"I couldn't complete that task action: {error}"

    tool_name = state.get("tool_name")
    result = state.get("tool_result")
    if isinstance(result, dict) and result.get("failed"):
        failed = result["failed"]
        details = "; ".join(
            str(item.get("error") or item.get("title") or item)
            for item in failed[:3]
        )
        return f"I couldn't complete that task action: {details}"

    catalog = state.get("scope_catalog") or {}
    candidates = catalog.get("candidates") or []
    if candidates:
        options = ", ".join(
            f"{candidate.get('project', {}).get('name')} in {candidate.get('organization', {}).get('name')}"
            for candidate in candidates[:8]
            if isinstance(candidate, dict)
        )
        if options:
            return (
                "I found multiple matching projects. Which one should I use? "
                f"Options: {options}."
            )

    if state.get("scope_status") == "not_found":
        return (
            "I couldn't find that organization or project. "
            "Please tell me the project name and organization."
        )

    if state.get("route") != "tools":
        return (
            "I couldn't create or change tasks because no todo action ran. "
            "Please try again, or select the organization and project first."
        )

    tool_name = state.get("tool_name")
    if tool_name and tool_name not in MUTATION_TOOLS:
        return (
            "I couldn't create the tasks because the assistant picked the wrong action. "
            "Please try again."
        )

    projects = catalog.get("projects") or []
    if projects:
        names = ", ".join(
            str(project.get("name") or project.get("id"))
            for project in projects[:8]
        )
        return (
            "I couldn't create the tasks because the project scope could not be resolved. "
            f"Available projects: {names}."
        )

    return "I couldn't complete that task action. Please try again."


def _validate_mutation_arguments(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name == "create_task":
        if not arguments.get("title"):
            return "Missing task title"
        if not _is_uuid(arguments.get("organization_id")) or not _is_uuid(
            arguments.get("project_id")
        ):
            return "Missing organization or project scope for task creation"
    if tool_name == "create_tasks":
        if not _is_uuid(arguments.get("organization_id")) or not _is_uuid(
            arguments.get("project_id")
        ):
            return "Missing organization or project scope for task creation"
        tasks = arguments.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return "No tasks to create"
    if tool_name in {"update_task", "update_tasks"}:
        if not _update_arguments_have_fields(tool_name, arguments):
            return "No fields to update"
    return None


def _extract_all_scope_hints(message: str) -> tuple[str | None, list[str]]:
    org_hint = None
    project_hints: list[str] = []

    for pattern in (
        r"\bin\s+([a-z0-9][a-z0-9-]*)\s+project\b",
        r"\bto\s+([a-z0-9][a-z0-9-]*)\b",
    ):
        match = re.search(pattern, message, re.I)
        if match and not _is_uuid(match.group(1)):
            org_hint = match.group(1)
            break

    project_match = re.search(r"\bin\s+my\s+(.+?)\s*\.?\s*$", message, re.I | re.M)
    if project_match:
        project_hints.append(f"my {project_match.group(1).strip()}")

    return org_hint, project_hints


def _extract_scope_hints(message: str) -> tuple[str | None, str | None]:
    org_hint, project_hints = _extract_all_scope_hints(message)
    project_hint = project_hints[0] if project_hints else None
    return org_hint, project_hint


def _project_hint_variants(hints: list[str]) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        for candidate in (hint, hint.removeprefix("my ").strip() if hint.lower().startswith("my ") else hint):
            norm = _normalize_scope_name(candidate)
            if norm and norm not in seen:
                seen.add(norm)
                variants.append(candidate)
    return variants


def _parse_create_task_titles(message: str) -> list[str]:
    if _looks_like_update_mutation(message):
        return []

    chunks = re.split(r"\banother\s+task\b", message, flags=re.I)
    titles: list[str] = []

    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        text = re.sub(
            r"^create\s+(?:a\s+)?tasks?\s+(?:(?:in|to)\s+[a-z0-9-]+\s*(?:project\s+)?)?",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"^create\s+(?:a\s+)?tasks?\s*", "", text, flags=re.I)
        text = re.sub(r"^create\s+(?:the\s+)?", "", text, flags=re.I)
        text = re.sub(r"\bin\s+my\s+.+$", "", text, flags=re.I | re.M).strip()
        text = text.strip(" .")
        if len(text) >= 3:
            titles.append(text)

    return titles


def _coerce_mutation_tool(
    state: ChatGraphState,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    message = state.get("latest_user_message", "")
    if not _looks_like_task_mutation(message):
        return tool_name, arguments
    if _looks_like_move_mutation(message):
        return tool_name, arguments
    if _looks_like_update_mutation(message):
        return tool_name, arguments
    if state.get("task_refs") and not _looks_like_create_mutation(message):
        return tool_name, arguments

    titles = _parse_create_task_titles(message)
    org_hint, project_hints = _extract_all_scope_hints(message)
    coerced = dict(arguments)

    if org_hint and not _is_uuid(coerced.get("organization_id")):
        coerced["organization_id"] = org_hint
    project_hint = project_hints[0] if project_hints else None
    if project_hint and not _is_uuid(coerced.get("project_id")):
        coerced["project_id"] = project_hint

    if _looks_like_create_parent_with_subtasks(message):
        return tool_name, arguments

    if tool_name in {"create_task", "create_tasks"}:
        if titles:
            if len(titles) > 1 or tool_name == "create_tasks":
                coerced["tasks"] = [{"title": title} for title in titles]
                return "create_tasks", coerced
            coerced["title"] = titles[0]
            return "create_task", coerced
        return tool_name, coerced

    if not titles:
        return tool_name, coerced

    coerced["tasks"] = [{"title": title} for title in titles]
    return "create_tasks", coerced


def _batch_task_ids(
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
) -> list[str]:
    task_ids = arguments.get("task_ids")
    refs = _effective_task_refs(task_refs, latest_user_message)
    if task_ids is None and refs:
        if len(refs) == 1:
            only_id = _ref_task_id(refs[0])
            return [only_id] if only_id else []
        if (
            _looks_like_bulk_selected(latest_user_message)
            or _looks_like_update_mutation(latest_user_message)
            or _looks_like_move_mutation(latest_user_message)
        ):
            return [tid for ref in refs if (tid := _ref_task_id(ref))]
    if task_ids is None:
        return []
    return list(task_ids)


def _batch_task_scopes(
    task_ids: list[str],
    task_refs: list[dict[str, str]],
) -> list[dict[str, str]]:
    ref_by_identifier = _task_ref_lookup(task_refs)

    tasks: list[dict[str, str]] = []
    for task_id in task_ids:
        ref = ref_by_identifier.get(task_id) or ref_by_identifier.get(
            _normalize_task_identifier(task_id),
        )
        if not ref:
            continue
        organization_id = _ref_organization_id(ref)
        project_id = _ref_project_id(ref)
        if not organization_id or not project_id:
            continue
        tasks.append(
            {
                "organization_id": organization_id,
                "project_id": project_id,
                "task_id": _ref_task_id(ref) or task_id,
            }
        )
    return tasks


SINGLE_TO_BATCH_TOOL = {
    "get_task": "get_tasks",
    "update_task": "update_tasks",
    "delete_task": "delete_tasks",
    "move_task": "move_tasks",
}

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)

SCOPE_TOOLS = {
    "create_task",
    "create_tasks",
    "update_task",
    "list_projects",
    "get_task",
    "delete_task",
}

EXISTING_TASK_TOOLS = {
    "get_task",
    "get_tasks",
    "update_task",
    "update_tasks",
    "move_task",
    "move_tasks",
    "delete_task",
    "delete_tasks",
}


def _is_uuid(value: str | None) -> bool:
    return bool(value and UUID_PATTERN.match(value))


def _normalize_scope_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _pick_uuid(*values: str | None) -> str | None:
    for value in values:
        if _is_uuid(value):
            return value
    return None


def _match_scope_name(items: list[dict[str, Any]], hint: str | None) -> str | None:
    if not hint or _is_uuid(hint):
        return None
    hint_norm = _normalize_scope_name(hint)
    if not hint_norm:
        return None

    matches: list[str] = []
    for item in items:
        item_id = item.get("id")
        if not _is_uuid(item_id):
            continue
        for key in ("name", "slug", "title"):
            candidate = item.get(key)
            if isinstance(candidate, str) and _normalize_scope_name(candidate) == hint_norm:
                matches.append(item_id)
                break

    if len(matches) == 1:
        return matches[0]
    return None


def _scope_item_labels(item: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ("name", "slug", "title"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value)
    return labels


def _best_scope_match(items: list[dict[str, Any]], hint: str | None) -> str | None:
    if not hint:
        return None
    if _is_uuid(hint):
        return hint

    exact = _match_scope_name(items, hint)
    if exact:
        return exact

    hint_norm = _normalize_scope_name(hint)
    if not hint_norm:
        return None

    best_id: str | None = None
    best_score = 0
    for item in items:
        item_id = item.get("id")
        if not _is_uuid(item_id):
            continue
        for label in _scope_item_labels(item):
            label_norm = _normalize_scope_name(label)
            if not label_norm:
                continue
            if hint_norm == label_norm:
                return item_id
            score = 0
            if hint_norm in label_norm or label_norm in hint_norm:
                score = 70 + min(len(hint_norm), len(label_norm))
            else:
                prefix = 0
                for left, right in zip(hint_norm, label_norm):
                    if left == right:
                        prefix += 1
                    else:
                        break
                if prefix >= 4:
                    score = 40 + prefix
            if score > best_score:
                best_score = score
                best_id = item_id

    return best_id if best_score >= 40 else None


def _catalog_from_scope_result(result: dict[str, Any]) -> dict[str, Any]:
    catalog: dict[str, Any] = {
        "status": result.get("status"),
        "organizations": [],
        "projects": [],
        "candidates": result.get("candidates") or [],
    }
    organization = result.get("organization")
    project = result.get("project")
    if isinstance(organization, dict):
        catalog["organizations"] = [
            {
                "id": organization.get("id"),
                "name": organization.get("name"),
                "slug": organization.get("slug"),
            }
        ]
    if isinstance(project, dict):
        catalog["projects"] = [
            {
                "id": project.get("id"),
                "name": project.get("name"),
            }
        ]
    for candidate in catalog["candidates"]:
        if not isinstance(candidate, dict):
            continue
        org = candidate.get("organization")
        proj = candidate.get("project")
        if isinstance(org, dict) and org not in catalog["organizations"]:
            catalog["organizations"].append(org)
        if isinstance(proj, dict) and proj not in catalog["projects"]:
            catalog["projects"].append(proj)
    return catalog


def _apply_resolved_scope(
    target: dict[str, Any],
    result: dict[str, Any],
) -> None:
    organization = result.get("organization") or {}
    project = result.get("project") or {}
    org_id = organization.get("id")
    project_id = project.get("id")
    if _is_uuid(org_id):
        target["organization_id"] = org_id
    if _is_uuid(project_id):
        target["project_id"] = project_id


async def _resolve_scope_via_api(
    tools: TodoTools,
    *,
    organization_hint: str | None,
    project_hint: str | None,
    message: str,
    state_org: str | None,
    state_proj: str | None,
) -> dict[str, Any]:
    organization_id = _pick_uuid(state_org)
    project_id = _pick_uuid(state_proj)
    if organization_id and project_id:
        return {
            "status": "resolved",
            "organization": {"id": organization_id},
            "project": {"id": project_id},
        }

    return await tools.resolve_scope(
        organization_hint=organization_hint,
        project_hint=project_hint,
        message=message,
    )


async def resolve_scope_arguments(
    tools: TodoTools,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state: ChatGraphState,
) -> dict[str, Any]:
    if tool_name not in SCOPE_TOOLS or tool_name in EXISTING_TASK_TOOLS:
        return arguments

    resolved = dict(arguments)
    org_hint = resolved.get("organization_id")
    proj_hint = resolved.get("project_id")
    if _is_uuid(org_hint):
        org_hint = None
    if _is_uuid(proj_hint):
        proj_hint = None

    scope_result = await _resolve_scope_via_api(
        tools,
        organization_hint=org_hint if isinstance(org_hint, str) else None,
        project_hint=proj_hint if isinstance(proj_hint, str) else None,
        message=state.get("latest_user_message", ""),
        state_org=state.get("organization_id"),
        state_proj=state.get("project_id"),
    )
    if scope_result.get("status") == "resolved":
        _apply_resolved_scope(resolved, scope_result)
    else:
        if org_hint and not _is_uuid(resolved.get("organization_id")):
            resolved.pop("organization_id", None)
        if proj_hint and not _is_uuid(resolved.get("project_id")):
            resolved.pop("project_id", None)

    if tool_name == "create_tasks":
        scope = {
            key: resolved[key]
            for key in ("organization_id", "project_id")
            if resolved.get(key)
        }
        tasks = resolved.get("tasks")
        if isinstance(tasks, list):
            resolved["tasks"] = [{**scope, **task} for task in tasks if isinstance(task, dict)]

    return resolved


async def scope_discovery_agent(state: ChatGraphState) -> ChatGraphState:
    message = state.get("latest_user_message", "")
    org_hint, project_hints = _extract_all_scope_hints(message)
    project_hint = project_hints[0] if project_hints else None

    client = ArcTodoClient(user_token=state["user_token"])
    tools = TodoTools(client)
    used_tools = list(state.get("used_tools", []))
    if "resolve_scope" not in used_tools:
        used_tools.append("resolve_scope")

    scope_result = await _resolve_scope_via_api(
        tools,
        organization_hint=org_hint,
        project_hint=project_hint,
        message=message,
        state_org=state.get("organization_id"),
        state_proj=state.get("project_id"),
    )
    scope_status = scope_result.get("status", "not_found")
    updates: ChatGraphState = {
        **state,
        "used_tools": used_tools,
        "scope_catalog": _catalog_from_scope_result(scope_result),
        "scope_status": scope_status,
        "scope_retried": bool(state.get("scope_retried")),
    }
    if scope_status == "resolved":
        _apply_resolved_scope(updates, scope_result)

    logger.info(
        "scope discovery org=%s project=%s org_hint=%s project_hints=%s status=%s",
        updates.get("organization_id"),
        updates.get("project_id"),
        org_hint,
        project_hints,
        scope_status,
    )

    if _looks_like_create_mutation(message) and scope_status == "resolved":
        parsed_parent = _parse_create_parent_with_subtasks(message)
        if parsed_parent:
            actions = _build_create_parent_with_subtasks_actions(updates, parsed_parent)
            updates["route"] = "tools"
            updates["tool_name"] = actions[0]["tool_name"]
            updates["tool_arguments"] = actions[0]["tool_arguments"]
            updates["actions"] = actions
        else:
            tool_name, tool_arguments = _coerce_mutation_tool(
                updates,
                "create_tasks",
                dict(updates.get("tool_arguments") or {}),
            )
            updates["route"] = "tools"
            updates["tool_name"] = tool_name
            updates["tool_arguments"] = tool_arguments
            updates["actions"] = [
                {
                    "intent": "create tasks",
                    "tool_name": tool_name,
                    "tool_arguments": tool_arguments,
                }
            ]

    return updates


def _needs_scope_retry(state: ChatGraphState) -> bool:
    if state.get("scope_retried"):
        return False
    if state.get("scope_status") in {"ambiguous", "not_found"}:
        return False

    tool_results = state.get("tool_results") or []
    if tool_results:
        failed_create = any(
            item.get("tool_name") in {"create_task", "create_tasks"}
            and not item.get("success")
            for item in tool_results
        )
        if not failed_create:
            return False
    elif state.get("tool_name") not in MUTATION_TOOLS:
        return False
    elif _mutation_succeeded(state) is True:
        return False

    error = (state.get("error") or "").lower()
    if "scope" in error:
        return True

    create_tools = {"create_task", "create_tasks"}
    if tool_results:
        if any(item.get("tool_name") in create_tools for item in tool_results):
            if not _is_uuid(state.get("organization_id")) or not _is_uuid(
                state.get("project_id")
            ):
                return True
        return False

    if state.get("tool_name") in create_tools:
        if not _is_uuid(state.get("organization_id")) or not _is_uuid(state.get("project_id")):
            return True
    return False


def resolve_delete_tasks_arguments(
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
) -> dict[str, Any]:
    task_ids = _batch_task_ids(
        arguments=arguments,
        task_refs=task_refs,
        latest_user_message=latest_user_message,
    )
    return {"tasks": _batch_task_scopes(task_ids, task_refs)}


def resolve_update_tasks_arguments(
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
) -> dict[str, Any]:
    per_task = _build_per_task_updates_from_arguments(
        arguments=arguments,
        task_refs=task_refs,
    )
    if per_task:
        return {"tasks": per_task}

    task_ids = _batch_task_ids(
        arguments=arguments,
        task_refs=task_refs,
        latest_user_message=latest_user_message,
    )
    updates = {
        key: arguments[key]
        for key in UPDATE_TASK_FIELDS
        if key in arguments and arguments[key] is not None
    }
    tasks = [
        {**scope, **updates}
        for scope in _batch_task_scopes(task_ids, task_refs)
    ]
    return {"tasks": tasks}


def _build_per_task_updates_from_arguments(
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
) -> list[dict[str, Any]] | None:
    raw_tasks = arguments.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return None

    ref_by_id = {
        task_id: ref
        for ref in task_refs
        if (task_id := _ref_task_id(ref))
    }
    merged: list[dict[str, Any]] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id") or item.get("taskId")
        if not isinstance(task_id, str) or task_id not in ref_by_id:
            continue
        scope = _batch_task_scopes([task_id], task_refs)[0]
        fields = {
            key: item[key]
            for key in UPDATE_TASK_FIELDS
            if key in item and item[key] is not None
        }
        if fields:
            merged.append({**scope, **fields})

    return merged or None


async def _generate_task_description_from_title(
    runtime: ChatbotRuntimeSettings,
    *,
    title: str,
    message: str,
) -> str:
    model = build_model(runtime)
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Write a concise, creative task description that matches the title "
                    "and user request. Return JSON only: "
                    '{"description":"<description>"}'
                )
            ),
            HumanMessage(content=f"User request:\n{message}\n\nTask title:\n{title}"),
        ]
    )
    payload = _extract_json(str(result.content))
    description = payload.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return f"Scope: {title}."


async def _maybe_generate_create_descriptions(
    runtime: ChatbotRuntimeSettings | None,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    if not runtime:
        return arguments
    wants_description = bool(
        re.search(r"\b(?:description|detail|note)\b", message, re.I)
    )
    if not wants_description:
        return arguments

    updated = dict(arguments)
    if tool_name == "create_task" and updated.get("title") and not updated.get(
        "description"
    ):
        updated["description"] = await _generate_task_description_from_title(
            runtime,
            title=str(updated["title"]),
            message=message,
        )
    if tool_name == "create_tasks":
        tasks = updated.get("tasks")
        if isinstance(tasks, list):
            enriched: list[dict[str, Any]] = []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                item = dict(task)
                if item.get("title") and not item.get("description"):
                    item["description"] = await _generate_task_description_from_title(
                        runtime,
                        title=str(item["title"]),
                        message=message,
                    )
                enriched.append(item)
            updated["tasks"] = enriched
    return updated


async def _generate_per_task_descriptions(
    runtime: ChatbotRuntimeSettings,
    *,
    message: str,
    task_refs: list[dict[str, str]],
    task_context_text: str = "",
) -> dict[str, str]:
    model = build_model(runtime)
    refs = _effective_task_refs(task_refs, message)
    task_lines = "\n".join(
        f"- task_id={task_id} title={ref.get('title') or 'Untitled'}"
        for ref in refs
        if (task_id := _ref_task_id(ref))
    )
    context_block = f"\n\n{task_context_text}" if task_context_text else ""
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Write a distinct task description for each task id based on its title "
                    "and the user request. Return JSON only in the shape "
                    '{"descriptions":{"<task_id>":"<description>"}}. '
                    "Each description must be specific to that task's title. "
                    "Ignore incorrect or copied current descriptions. "
                    "Do not reuse the same description text."
                )
            ),
            HumanMessage(
                content=f"User request:\n{message}\n\nTasks:\n{task_lines}{context_block}"
            ),
        ]
    )
    payload = _extract_json(str(result.content))
    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, dict):
        return {}
    return {
        str(task_id): str(description).strip()
        for task_id, description in descriptions.items()
        if description
    }


async def resolve_update_tasks_arguments_async(
    runtime: ChatbotRuntimeSettings | None,
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
    task_context_text: str = "",
) -> dict[str, Any]:
    per_task = _build_per_task_updates_from_arguments(
        arguments=arguments,
        task_refs=task_refs,
    )
    if per_task:
        return {"tasks": per_task}

    task_ids = _batch_task_ids(
        arguments=arguments,
        task_refs=task_refs,
        latest_user_message=latest_user_message,
    )
    scopes = _batch_task_scopes(task_ids, task_refs)
    updates = {
        key: arguments[key]
        for key in UPDATE_TASK_FIELDS
        if key in arguments and arguments[key] is not None
    }

    wants_descriptions = "description" in updates or bool(
        re.search(r"\b(?:description|detail|note)\b", latest_user_message, re.I)
    )
    if len(scopes) > 1 and runtime and wants_descriptions:
        generated = await _generate_per_task_descriptions(
            runtime,
            message=latest_user_message,
            task_refs=task_refs,
            task_context_text=task_context_text,
        )
        tasks: list[dict[str, Any]] = []
        for scope in scopes:
            task_id = scope["task_id"]
            description = generated.get(task_id)
            if not description:
                ref = next(
                    (ref for ref in task_refs if _ref_task_id(ref) == task_id),
                    None,
                )
                title = (ref or {}).get("title") or "Task"
                description = f"Scope: {title}."
            payload = {**scope, "description": description}
            for key, value in updates.items():
                if key != "description":
                    payload[key] = value
            tasks.append(payload)
        return {"tasks": tasks}

    if len(scopes) == 1 and runtime and wants_descriptions and not updates.get(
        "description"
    ):
        scope = scopes[0]
        task_id = scope["task_id"]
        ref = next((ref for ref in task_refs if _ref_task_id(ref) == task_id), None)
        title = (ref or {}).get("title") or "Task"
        description = await _generate_task_description_from_title(
            runtime,
            title=str(title),
            message=latest_user_message,
        )
        payload = {**scope, "description": description}
        for key, value in updates.items():
            if key != "description":
                payload[key] = value
        return {"tasks": [payload]}

    return {
        "tasks": [{**scope, **updates} for scope in scopes],
    }


def resolve_get_tasks_arguments(
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
) -> dict[str, Any]:
    task_ids = _batch_task_ids(
        arguments=arguments,
        task_refs=task_refs,
        latest_user_message=latest_user_message,
    )
    return {"tasks": _batch_task_scopes(task_ids, task_refs)}


async def resolve_move_tasks_arguments(
    tools: TodoTools,
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
) -> dict[str, Any]:
    task_ids = _batch_task_ids(
        arguments=arguments,
        task_refs=task_refs,
        latest_user_message=latest_user_message,
    )
    scopes = _batch_task_scopes(task_ids, task_refs)
    if not scopes:
        return {"tasks": []}

    target_hint = (
        arguments.get("target_project_hint")
        if isinstance(arguments.get("target_project_hint"), str)
        else None
    ) or _extract_move_target_hint(latest_user_message)
    target_project_id = arguments.get("target_project_id")
    if not _is_uuid(target_project_id):
        target_project_id = None

    if not target_project_id and target_hint:
        source_org_id = scopes[0].get("organization_id")
        scope_result = await tools.resolve_scope(
            organization_hint=None,
            project_hint=target_hint,
            message=latest_user_message,
        )
        if scope_result.get("status") == "resolved":
            project = scope_result.get("project") or {}
            organization = scope_result.get("organization") or {}
            resolved_project_id = project.get("id")
            resolved_org_id = organization.get("id")
            if _is_uuid(resolved_project_id):
                target_project_id = resolved_project_id
            if _is_uuid(resolved_org_id) and _is_uuid(source_org_id):
                if resolved_org_id != source_org_id:
                    raise ArcTodoApiError(
                        "Target project must be in the same organization as the selected task"
                    )

    if not target_project_id:
        raise ArcTodoApiError("Could not resolve target project for move")

    return {
        "tasks": [
            {**scope, "target_project_id": target_project_id}
            for scope in scopes
        ]
    }


BATCH_TOOL_RESOLVERS = {
    "get_tasks": resolve_get_tasks_arguments,
    "delete_tasks": resolve_delete_tasks_arguments,
}


async def context_agent(state: ChatGraphState) -> ChatGraphState:
    messages = state.get("messages", [])
    latest = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    task_context_text = await _build_task_context_text(
        user_token=state["user_token"],
        task_refs=state.get("task_refs", []),
    )
    return {
        **state,
        "latest_user_message": latest,
        "task_context_text": task_context_text,
        "used_tools": state.get("used_tools", []),
    }


def _format_action_success_line(action: dict[str, Any]) -> str:
    tool_name = action.get("tool_name")
    result = action.get("tool_result")
    intent = action.get("intent") or tool_name
    partial = action.get("partial")

    if not isinstance(result, dict):
        return f"- {intent}: completed"

    if tool_name == "create_task":
        title = result.get("title") or "task"
        task_id = result.get("id") or "unknown"
        parent_task_id = result.get("parentTaskId") or result.get("parent_task_id")
        if parent_task_id:
            return f"- Created subtask **{title}** under parent {parent_task_id} (id: {task_id})"
        return f"- Created **{title}** (id: {task_id})"
    if tool_name == "create_tasks":
        created = result.get("created") or []
        titles = ", ".join(
            f"**{item.get('title') or item.get('id')}**" for item in created[:5]
        )
        suffix = " (partial)" if partial else ""
        parent_task_id = action.get("tool_arguments", {}).get("parent_task_id")
        parent_title = action.get("tool_arguments", {}).get("_parent_title")
        if not parent_task_id:
            tasks = action.get("tool_arguments", {}).get("tasks") or []
            if isinstance(tasks, list) and tasks:
                first = tasks[0]
                if isinstance(first, dict):
                    parent_task_id = first.get("parent_task_id")
        if parent_task_id:
            parent_label = parent_title or parent_task_id
            return (
                f"- Created {len(created)} subtask(s) under **{parent_label}**{suffix}: "
                f"{titles}"
            )
        return f"- Created {len(created)} task(s){suffix}: {titles}"
    if tool_name == "update_task":
        title = result.get("title") or "task"
        parent_task_id = result.get("parentTaskId") or result.get("parent_task_id")
        if parent_task_id is None and action.get("tool_arguments", {}).get("parent_task_id") is None:
            return f"- Detached **{title}** from parent"
        if parent_task_id:
            return f"- Set **{title}** as subtask of {parent_task_id}"
        return f"- Updated **{title}**"
    if tool_name == "update_tasks":
        updated = result.get("updated") or []
        suffix = " (partial)" if partial else ""
        return f"- Updated {len(updated)} task(s){suffix}"
    if tool_name == "move_task":
        title = result.get("title") or "task"
        return f"- Moved **{title}**"
    if tool_name == "move_tasks":
        moved = result.get("moved") or []
        suffix = " (partial)" if partial else ""
        return f"- Moved {len(moved)} task(s){suffix}"
    if tool_name == "delete_task":
        return f"- Deleted task"
    if tool_name == "delete_tasks":
        deleted = result.get("deleted") or []
        suffix = " (partial)" if partial else ""
        return f"- Deleted {len(deleted)} task(s){suffix}"
    return f"- {intent}: completed"


def _format_action_failure_line(action: dict[str, Any]) -> str:
    intent = action.get("intent") or action.get("tool_name")
    error = action.get("error") or "action failed"
    result = action.get("tool_result")
    if isinstance(result, dict) and result.get("failed"):
        failed = result["failed"]
        details = "; ".join(
            str(item.get("error") or item.get("title") or item.get("task_id") or item)
            for item in failed[:3]
        )
        return f"- {intent}: failed ({details})"
    return f"- {intent}: failed ({error})"


def _build_verified_mutation_response(state: ChatGraphState) -> str | None:
    tool_results = state.get("tool_results") or []
    if not tool_results and state.get("tool_name") in MUTATION_TOOLS:
        success, partial = _action_succeeded(
            state.get("tool_name") or "",
            state.get("tool_result"),
            state.get("error"),
        )
        tool_results = [
            {
                "intent": state.get("tool_name"),
                "tool_name": state.get("tool_name"),
                "tool_result": state.get("tool_result"),
                "error": state.get("error"),
                "success": success,
                "partial": partial,
            }
        ]

    mutation_results = [
        item for item in tool_results if item.get("tool_name") in MUTATION_TOOLS
    ]
    if not mutation_results:
        return None

    success_lines = [
        _format_action_success_line(item) for item in mutation_results if item.get("success")
    ]
    failure_lines = [
        _format_action_failure_line(item)
        for item in mutation_results
        if not item.get("success")
    ]

    if success_lines and failure_lines:
        intro = (
            f"I completed {len(success_lines)} of "
            f"{len(mutation_results)} requested actions:"
        )
    elif success_lines:
        intro = "Done:"
    else:
        intro = "I couldn't complete the requested actions:"

    lines = [intro, *success_lines, *failure_lines]
    return "\n".join(lines)


async def planner_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    messages = state.get("messages", [])
    task_refs = state.get("task_refs", [])
    latest_user_message = state.get("latest_user_message", "")

    if _looks_like_mutation_confirmation(latest_user_message) and task_refs:
        apply_arguments = _resolve_confirmation_update_arguments(messages, task_refs)
        if apply_arguments:
            logger.info("planner confirmation apply update_tasks tasks=%s", len(apply_arguments.get("tasks", [])))
            return {
                **state,
                "route": "tools",
                "actions": [
                    {
                        "intent": "apply proposed descriptions",
                        "tool_name": "update_tasks",
                        "tool_arguments": apply_arguments,
                    }
                ],
                "tool_name": "update_tasks",
                "tool_arguments": apply_arguments,
            }

    model = build_model(runtime)
    context_bits = []
    if state.get("organization_id"):
        context_bits.append(f"organization_id={state['organization_id']}")
    if state.get("project_id"):
        context_bits.append(f"project_id={state['project_id']}")

    prompt = PLANNER_PROMPT
    if context_bits:
        prompt += "\nContext: " + ", ".join(context_bits)
    if state.get("task_context_text"):
        prompt += "\n\n" + state["task_context_text"]

    result = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state.get("latest_user_message", "")),
        ]
    )
    payload = _extract_json(str(result.content))
    route = payload.get("route", "direct")
    if route not in {"direct", "tools"}:
        route = "direct"

    latest_user_message = state.get("latest_user_message", "")
    if route == "direct" and (
        _looks_like_task_mutation(latest_user_message)
        or (task_refs and _looks_like_update_mutation(latest_user_message))
    ):
        retry = await model.ainvoke(
            [
                SystemMessage(content=prompt + PLANNER_MUTATION_RETRY),
                HumanMessage(content=latest_user_message),
            ]
        )
        retry_payload = _extract_json(str(retry.content))
        retry_route = retry_payload.get("route", route)
        if retry_route == "tools" and retry_payload.get("tool_name"):
            payload = retry_payload
            route = "tools"

        if retry_route == "tools" and (
            retry_payload.get("actions") or retry_payload.get("tool_name")
        ):
            payload = retry_payload
            route = "tools"

    actions = _normalize_planner_actions(payload)
    if route == "tools" and not actions:
        route = "direct"

    logger.info(
        "planner route=%s actions=%s org=%s project=%s",
        route,
        [action.get("tool_name") for action in actions],
        state.get("organization_id"),
        state.get("project_id"),
    )

    updates: ChatGraphState = {
        **state,
        "route": route,
        "actions": actions,
    }
    if actions:
        updates["tool_name"] = actions[0]["tool_name"]
        updates["tool_arguments"] = actions[0]["tool_arguments"]
    else:
        updates["tool_name"] = None
        updates["tool_arguments"] = {}
    return updates


async def _execute_single_tool(
    state: ChatGraphState,
    runtime: ChatbotRuntimeSettings | None,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    apply_heuristics: bool = True,
) -> tuple[str, dict[str, Any], Any, str | None]:
    arguments = _normalize_tool_arguments(dict(arguments or {}))
    if not _is_uuid(arguments.get("organization_id")) and _is_uuid(state.get("organization_id")):
        arguments["organization_id"] = state["organization_id"]
    elif not arguments.get("organization_id") and state.get("organization_id"):
        arguments["organization_id"] = state["organization_id"]
    if not _is_uuid(arguments.get("project_id")) and _is_uuid(state.get("project_id")):
        arguments["project_id"] = state["project_id"]
    elif not arguments.get("project_id") and state.get("project_id"):
        arguments["project_id"] = state["project_id"]

    task_id = arguments.get("task_id")
    if task_id and (
        not arguments.get("organization_id") or not arguments.get("project_id")
    ):
        for ref in state.get("task_refs", []):
            ref_task_id = ref.get("taskId") or ref.get("task_id")
            if ref_task_id != task_id:
                continue
            if not arguments.get("organization_id"):
                arguments["organization_id"] = ref.get("organizationId") or ref.get(
                    "organization_id"
                )
            if not arguments.get("project_id"):
                arguments["project_id"] = ref.get("projectId") or ref.get("project_id")
            break

    if not arguments.get("organization_id") and state.get("task_refs"):
        first_ref = state["task_refs"][0]
        arguments.setdefault(
            "organization_id",
            first_ref.get("organizationId") or first_ref.get("organization_id"),
        )
        arguments.setdefault(
            "project_id",
            first_ref.get("projectId") or first_ref.get("project_id"),
        )

    task_refs = state.get("task_refs", [])
    latest_user_message = state.get("latest_user_message", "")
    if task_refs and tool_name in EXISTING_TASK_TOOLS:
        arguments = _apply_task_ref_source_scope(arguments, task_refs)

    if apply_heuristics:
        reparent_arguments = _resolve_reparent_arguments(
            latest_user_message,
            task_refs,
        )
        if reparent_arguments:
            tool_name = "update_task"
            arguments = reparent_arguments

        if task_refs and _looks_like_move_mutation(latest_user_message):
            tool_name = "move_tasks"
            arguments = {
                key: value
                for key, value in arguments.items()
                if key in {"task_ids", "target_project_id", "target_project_hint"}
            }
            arguments.setdefault("task_ids", None)

        if task_refs and _looks_like_update_mutation(latest_user_message) and not _looks_like_move_mutation(
            latest_user_message
        ) and not _looks_like_reparent_mutation(latest_user_message):
            tool_name = "update_tasks"
            if isinstance(arguments.get("tasks"), list):
                arguments = {
                    "tasks": arguments["tasks"],
                    "task_ids": arguments.get("task_ids"),
                }
            else:
                arguments = {
                    key: arguments[key]
                    for key in UPDATE_TASK_FIELDS
                    if key in arguments and arguments[key] is not None
                }
                arguments.setdefault("task_ids", None)

        tool_name, arguments = _coerce_mutation_tool(state, tool_name, arguments)

    if task_refs and tool_name in EXISTING_TASK_TOOLS:
        arguments = _apply_task_ref_source_scope(arguments, task_refs)

    if apply_heuristics and (
        tool_name in SINGLE_TO_BATCH_TOOL
        and len(task_refs) > 1
        and (
            _looks_like_bulk_selected(latest_user_message)
            or _looks_like_move_mutation(latest_user_message)
            or (not arguments.get("task_id") and not arguments.get("task_ids"))
        )
    ):
        batch_arguments = dict(arguments)
        batch_arguments.setdefault("task_ids", None)
        tool_name = SINGLE_TO_BATCH_TOOL[tool_name]
        arguments = batch_arguments

    client = ArcTodoClient(user_token=state["user_token"])
    tools = TodoTools(client)
    try:
        if tool_name in {"move_task", "move_tasks"}:
            move_arguments = await resolve_move_tasks_arguments(
                tools,
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
            )
            if tool_name == "move_task":
                tasks = move_arguments.get("tasks") or []
                if len(tasks) != 1:
                    return tool_name, arguments, None, "Could not resolve selected task for move"
                task = tasks[0]
                arguments = {
                    "organization_id": task["organization_id"],
                    "project_id": task["project_id"],
                    "task_id": task["task_id"],
                    "target_project_id": task["target_project_id"],
                }
            else:
                arguments = move_arguments
        elif tool_name == "update_tasks":
            arguments = await resolve_update_tasks_arguments_async(
                runtime,
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
                task_context_text=state.get("task_context_text") or "",
            )
        elif tool_name in BATCH_TOOL_RESOLVERS:
            arguments = BATCH_TOOL_RESOLVERS[tool_name](
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
            )

        if tool_name in {"create_task", "create_tasks"}:
            arguments.pop("task_ids", None)
            arguments = _apply_subtask_parent_from_refs(
                arguments,
                task_refs,
                latest_user_message,
            )
            arguments = await _maybe_generate_create_descriptions(
                runtime,
                tool_name=tool_name,
                arguments=arguments,
                message=latest_user_message,
            )

        arguments = await resolve_scope_arguments(
            tools,
            tool_name=tool_name,
            arguments=arguments,
            state=state,
        )
        validation_error = _validate_mutation_arguments(tool_name, arguments)
        if validation_error:
            logger.warning(
                "mutation validation failed tool=%s error=%s args=%s",
                tool_name,
                validation_error,
                {
                    key: arguments.get(key)
                    for key in ("organization_id", "project_id", "title", "tasks")
                },
            )
            return tool_name, arguments, None, validation_error
        api_arguments = dict(arguments)
        api_arguments.pop("_parent_title", None)
        result = await execute_todo_tool(tools, tool_name, api_arguments)
        return tool_name, arguments, result, None
    except ArcTodoApiError as exc:
        return tool_name, arguments, None, str(exc)
    except Exception as exc:
        return tool_name, arguments, None, str(exc)


async def todo_tools_agent(
    state: ChatGraphState,
    runtime: ChatbotRuntimeSettings | None = None,
) -> ChatGraphState:
    actions = list(state.get("actions") or [])
    if not actions and state.get("tool_name"):
        actions = [
            {
                "intent": state.get("tool_name") or "action",
                "tool_name": state["tool_name"],
                "tool_arguments": dict(state.get("tool_arguments") or {}),
            }
        ]

    if not actions:
        return {**state, "error": "Planner did not select a tool"}

    apply_heuristics = len(actions) == 1
    tool_results: list[dict[str, Any]] = []
    used_tools = list(state.get("used_tools", []))
    last_tool_name: str | None = None
    last_tool_result: Any = None
    last_error: str | None = None

    for action in actions:
        tool_name = action["tool_name"]
        raw_arguments = dict(action.get("tool_arguments") or {})
        needs_parent = bool(raw_arguments.get("_parent_from_previous"))
        arguments = _inject_parent_from_previous(raw_arguments, tool_results)
        if needs_parent and not _create_tasks_have_parent(arguments):
            tool_results.append(
                {
                    "intent": action.get("intent") or tool_name,
                    "tool_name": tool_name,
                    "tool_arguments": arguments,
                    "tool_result": None,
                    "error": "Could not resolve parent task from previous create step",
                    "success": False,
                    "partial": False,
                }
            )
            last_tool_name = tool_name
            last_error = "Could not resolve parent task from previous create step"
            continue
        executed_name, executed_args, result, error = await _execute_single_tool(
            state,
            runtime,
            tool_name=tool_name,
            arguments=arguments,
            apply_heuristics=apply_heuristics,
        )
        success, partial = _action_succeeded(executed_name, result, error)
        tool_results.append(
            {
                "intent": action.get("intent") or executed_name,
                "tool_name": executed_name,
                "tool_arguments": executed_args,
                "tool_result": result,
                "error": error,
                "success": success,
                "partial": partial,
            }
        )
        if executed_name not in used_tools:
            used_tools.append(executed_name)
        last_tool_name = executed_name
        last_tool_result = result
        last_error = error
        logger.info(
            "tool executed tool=%s success=%s partial=%s",
            executed_name,
            success,
            partial,
        )

    any_success = any(item.get("success") for item in tool_results)
    return {
        **state,
        "route": "tools",
        "actions": actions,
        "tool_results": tool_results,
        "tool_name": last_tool_name,
        "tool_result": last_tool_result,
        "used_tools": used_tools,
        "error": None if any_success else last_error,
    }


async def response_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    if state.get("scope_status") in {"ambiguous", "not_found"} and _looks_like_create_mutation(
        state.get("latest_user_message", "")
    ):
        return {**state, "response": _build_mutation_failure_response(state)}

    if _needs_mutation_tool_result(state):
        verified = _build_verified_mutation_response(state)
        if verified:
            return {**state, "response": verified}

    model = build_model(runtime)
    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in state.get("messages", [])
    )
    tool_context = ""
    tool_results = state.get("tool_results") or []
    if tool_results:
        tool_context = (
            "\nVerified action results:\n"
            + json.dumps(tool_results, indent=2, default=str)
        )
    elif state.get("tool_result") is not None:
        tool_context = f"\nTool result:\n{json.dumps(state['tool_result'], indent=2, default=str)}"
    if state.get("error"):
        tool_context += f"\nError:\n{state['error']}"
    if state.get("task_context_text"):
        tool_context += f"\n\n{state['task_context_text']}"

    result = await model.ainvoke(
        [
            SystemMessage(content=RESPONSE_PROMPT),
            HumanMessage(content=f"Conversation:\n{conversation}{tool_context}"),
        ]
    )
    response_text = str(result.content).strip()

    if _needs_mutation_tool_result(state):
        succeeded = _mutation_succeeded(state)
        if succeeded is False or (
            succeeded is None and state.get("route") == "direct"
        ):
            response_text = _build_mutation_failure_response(state)
        else:
            verified = _build_verified_mutation_response(state)
            if verified:
                response_text = verified

    return {**state, "response": response_text}


def route_after_context(state: ChatGraphState) -> str:
    if _looks_like_create_mutation(state.get("latest_user_message", "")):
        return "scope_discovery_agent"
    return "planner_agent"


def route_after_scope_discovery(state: ChatGraphState) -> str:
    if not _looks_like_create_mutation(state.get("latest_user_message", "")):
        return "planner_agent"
    if state.get("scope_status") in {"ambiguous", "not_found"}:
        return "response_agent"
    if _is_uuid(state.get("organization_id")) and _is_uuid(state.get("project_id")):
        return "todo_tools_agent"
    return "response_agent"


def route_after_tools(state: ChatGraphState) -> str:
    if _needs_scope_retry(state):
        return "scope_discovery_agent"
    return "response_agent"


def route_after_planner(state: ChatGraphState) -> str:
    if state.get("route") == "tools" and (
        state.get("actions") or state.get("tool_name")
    ):
        return "todo_tools_agent"
    return "response_agent"
