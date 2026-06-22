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
from app.graph.prompts import CONFIRMATION_PATTERN
from app.graph.scope import _extract_all_scope_hints, _is_uuid, _normalize_scope_name

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
    from app.graph.task_refs import _ref_task_id

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
    from app.graph.task_refs import _effective_task_refs, _ref_task_id

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
    from app.graph.task_refs import _effective_task_refs, _ref_task_id

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
