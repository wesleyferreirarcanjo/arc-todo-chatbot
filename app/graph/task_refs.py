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
from app.graph.llm import _generate_per_task_descriptions, _generate_task_description_from_title
from app.graph.prompts import UPDATE_TASK_FIELDS
from app.graph.scope import _is_uuid, _normalize_scope_name

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
        f"  category: {task.get('category') or 'other'}",
        f"  organizationId: {organization_id}",
        f"  projectId: {project_id}",
        f"  dueDate: {due_date}",
        f"  description: {description}",
    ])
    metadata = task.get("metadata")
    if isinstance(metadata, dict) and metadata:
        lines.append(f"  metadata: {json.dumps(metadata, ensure_ascii=True)}")
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
    from app.graph import nodes

    if not task_refs:
        return ""

    client = nodes.ArcTodoClient(user_token=user_token)
    tools = nodes.TodoTools(client)
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
    from app.graph.heuristics import _looks_like_bulk_selected

    if len(task_refs) <= 1 or _looks_like_bulk_selected(latest_user_message):
        return task_refs
    return _filter_task_refs_by_message(task_refs, latest_user_message)

def _batch_task_ids(
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
) -> list[str]:
    from app.graph.heuristics import (
        _looks_like_bulk_selected,
        _looks_like_move_mutation,
        _looks_like_update_mutation,
    )

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

async def resolve_update_tasks_arguments_async(
    runtime: ChatbotRuntimeSettings | None,
    *,
    arguments: dict[str, Any],
    task_refs: list[dict[str, str]],
    latest_user_message: str = "",
    task_context_text: str = "",
) -> dict[str, Any]:
    from app.graph import nodes

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
        generated = await nodes._generate_per_task_descriptions(
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
        description = await nodes._generate_task_description_from_title(
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
    from app.graph.heuristics import _extract_move_target_hint

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
