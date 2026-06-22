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
from app.graph.prompts import MUTATION_TOOLS

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
