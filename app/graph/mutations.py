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
from app.graph.heuristics import _extract_proposed_descriptions_from_assistant, _find_last_assistant_message, _looks_like_mutation_confirmation, _looks_like_task_mutation, _looks_like_update_mutation
from app.graph.llm import _update_arguments_have_fields
from app.graph.prompts import MUTATION_TOOLS
from app.graph.scope import _is_uuid

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
