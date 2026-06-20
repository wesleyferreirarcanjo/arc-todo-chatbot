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
from app.tools.todo_tools import TodoTools, execute_todo_tool

SYSTEM_PROMPT = """You are Arc Todo assistant. Help users manage tasks in Arc Todo.
You can answer questions about todos and perform actions using available tools.
When organization or project context is missing, ask the user to clarify or use list_organizations and list_projects first.
Keep responses concise and practical."""

PLANNER_PROMPT = """Analyze the latest user message and decide whether todo API tools are needed.

Return JSON only with this shape:
{"route":"direct"|"tools","tool_name":string|null,"tool_arguments":object}

Available tools:
- list_organizations {}
- list_projects {"organization_id": string}
- list_tasks {"organization_id": string|null, "project_id": string|null, "status": string|null, "criticity": string|null}
- get_task {"organization_id": string, "project_id": string, "task_id": string}
- get_tasks {"task_ids": string[]|null} — fetch multiple selected tasks; omit task_ids to use all taskIds from Selected task context
- create_task {"organization_id": string|null, "project_id": string|null, "title": string, "description": string|null, "status": "todo"|"in_progress"|"done", "criticity": "low"|"medium"|"high"|"critical", "due_date": string|null}
- create_tasks {"organization_id": string|null, "project_id": string|null, "tasks": [{"title": string, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null}]} — create multiple tasks in one request
- update_task {"organization_id": string, "project_id": string, "task_id": string, "title": string|null, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null}
- update_tasks {"task_ids": string[]|null, "title": string|null, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null} — apply the same update fields to multiple selected tasks; omit task_ids to use all taskIds from Selected task context
- move_task {"task_id": string, "target_project_id": string|null, "target_project_hint": string|null} — move one selected task to another project; use organization_id/project_id from Selected task context for the task's current location
- move_tasks {"task_ids": string[]|null, "target_project_id": string|null, "target_project_hint": string|null} — move multiple selected tasks to another project
- delete_task {"organization_id": string, "project_id": string, "task_id": string}
- delete_tasks {"task_ids": string[]|null} — delete multiple selected tasks; omit task_ids to use all taskIds from Selected task context

Use get_tasks, update_tasks, move_tasks, or delete_tasks (not the single-task variants) when the user wants to act on more than one selected task.
Use move_tasks (not move_task) when the user wants to move more than one selected task.
When Selected task context is present, keep the task in its current organization_id/project_id and only change the destination project for move requests.
Use create_tasks (not create_task) when the user asks to create more than one task.
Use route "direct" for greetings, general help, or when no API action is needed.
Prefer provided organization_id and project_id context when present; omit organization_id/project_id from tool_arguments when context is already provided.
Never invent organization_id or project_id values from organization or project names in the user message — names like "arc-todo" are not UUIDs. Omit those fields and rely on context, or use list_organizations/list_projects first."""

RESPONSE_PROMPT = """Write a concise assistant reply for the user based on the conversation and any tool results.
Do not mention internal tool names unless helpful.
Never claim that tasks were created, updated, or deleted unless Tool result confirms it with task ids or a non-empty created/updated/deleted list.
If Error is present, explain that the action failed and include the error.
If no Tool result is present for a mutation request, say you could not perform the action yet."""

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

PLANNER_MUTATION_RETRY = (
    "\nIMPORTANT: The user is asking to create, update, or delete tasks. "
    'Return route "tools" with the appropriate tool and arguments. '
    "Do not answer directly."
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


def _format_task_context_line(task: dict[str, Any], fallback_title: str) -> str:
    title = task.get("title") or fallback_title
    status = task.get("status", "unknown")
    criticity = task.get("criticity", "unknown")
    due_date = task.get("dueDate") or task.get("due_date") or "none"
    description = (task.get("description") or "").strip() or "none"
    task_id = task.get("id", "unknown")
    organization_id = task.get("organizationId") or task.get("organization_id")
    project_id = task.get("projectId") or task.get("project_id")
    return (
        f"- taskId: {task_id}\n"
        f"  title: {title}\n"
        f"  status: {status}\n"
        f"  criticity: {criticity}\n"
        f"  organizationId: {organization_id}\n"
        f"  projectId: {project_id}\n"
        f"  dueDate: {due_date}\n"
        f"  description: {description}"
    )


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
            if _ref_task_id(ref) != task_id:
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
    return bool(
        re.search(
            r"\b(update|edit|change|set|mark|complete|describe)\b"
            r"|\b(?:add|create)\s+(?:a\s+)?(?:description|detail|note|comment|due\s*date)\b"
            r"|\b(description|details|notes)\s+(?:for|to)\b",
            message,
            re.I,
        )
    )


def _looks_like_create_mutation(message: str) -> bool:
    if _looks_like_update_mutation(message):
        return False
    if re.search(r"\b(create|make|new)\s+(?:a\s+)?tasks?\b", message, re.I):
        return True
    if re.search(r"\badd\s+(?:a\s+)?tasks?\b", message, re.I):
        return True
    if re.search(r"\banother\s+task\b", message, re.I):
        return True
    if _looks_like_multi_create(message):
        return True
    return False


def _mutation_succeeded(state: ChatGraphState) -> bool | None:
    if state.get("route") != "tools":
        return None

    tool_name = state.get("tool_name")
    if tool_name not in MUTATION_TOOLS:
        return None
    if state.get("error"):
        return False

    result = state.get("tool_result")
    if result is None:
        return False
    if not isinstance(result, dict):
        return True

    if tool_name == "create_task":
        return bool(result.get("id"))
    if tool_name == "create_tasks":
        return len(result.get("created", [])) > 0 and not result.get("failed")
    if tool_name == "update_task":
        return bool(result.get("id"))
    if tool_name == "update_tasks":
        return len(result.get("updated", [])) > 0 and not result.get("failed")
    if tool_name == "move_task":
        return bool(result.get("id"))
    if tool_name == "move_tasks":
        return len(result.get("moved", [])) > 0 and not result.get("failed")
    if tool_name == "delete_task":
        return True
    if tool_name == "delete_tasks":
        return len(result.get("deleted", [])) > 0 and not result.get("failed")
    return True


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
    if task_ids is None and task_refs:
        if len(task_refs) == 1:
            only_id = _ref_task_id(task_refs[0])
            return [only_id] if only_id else []
        if (
            _looks_like_bulk_selected(latest_user_message)
            or _looks_like_update_mutation(latest_user_message)
            or _looks_like_move_mutation(latest_user_message)
        ):
            return [tid for ref in task_refs if (tid := _ref_task_id(ref))]
    if task_ids is None:
        return []
    return list(task_ids)


def _batch_task_scopes(
    task_ids: list[str],
    task_refs: list[dict[str, str]],
) -> list[dict[str, str]]:
    ref_by_id = {
        tid: ref
        for ref in task_refs
        if (tid := _ref_task_id(ref))
    }

    tasks: list[dict[str, str]] = []
    for task_id in task_ids:
        ref = ref_by_id.get(task_id)
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
                "task_id": task_id,
            }
        )
    return tasks


UPDATE_TASK_FIELDS = ("title", "description", "status", "criticity", "due_date")

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
        tool_name, tool_arguments = _coerce_mutation_tool(
            updates,
            "create_tasks",
            dict(updates.get("tool_arguments") or {}),
        )
        updates["route"] = "tools"
        updates["tool_name"] = tool_name
        updates["tool_arguments"] = tool_arguments

    return updates


def _needs_scope_retry(state: ChatGraphState) -> bool:
    if state.get("scope_retried"):
        return False
    if state.get("scope_status") in {"ambiguous", "not_found"}:
        return False
    if state.get("tool_name") not in MUTATION_TOOLS:
        return False
    if _mutation_succeeded(state) is True:
        return False
    error = (state.get("error") or "").lower()
    if "scope" in error:
        return True
    if state.get("tool_name") in {"create_task", "create_tasks"}:
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
    "update_tasks": resolve_update_tasks_arguments,
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


async def planner_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
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
    if route == "direct" and _looks_like_task_mutation(latest_user_message):
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

    logger.info(
        "planner route=%s tool=%s org=%s project=%s",
        route,
        payload.get("tool_name"),
        state.get("organization_id"),
        state.get("project_id"),
    )

    return {
        **state,
        "route": route,
        "tool_name": payload.get("tool_name"),
        "tool_arguments": payload.get("tool_arguments") or {},
    }


async def todo_tools_agent(state: ChatGraphState) -> ChatGraphState:
    tool_name = state.get("tool_name")
    if not tool_name:
        return {**state, "error": "Planner did not select a tool"}

    arguments = dict(state.get("tool_arguments") or {})
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
    ):
        tool_name = "update_tasks"
        arguments = {
            key: arguments[key]
            for key in UPDATE_TASK_FIELDS
            if key in arguments and arguments[key] is not None
        }
        arguments.setdefault("task_ids", None)

    tool_name, arguments = _coerce_mutation_tool(state, tool_name, arguments)

    if task_refs and tool_name in EXISTING_TASK_TOOLS:
        arguments = _apply_task_ref_source_scope(arguments, task_refs)

    if (
        tool_name in SINGLE_TO_BATCH_TOOL
        and len(task_refs) > 1
        and (
            _looks_like_bulk_selected(latest_user_message)
            or _looks_like_move_mutation(latest_user_message)
            or (not arguments.get("task_id") and not arguments.get("task_ids"))
        )
    ):
        # ponytail: heuristic upgrade when planner picks single-task tool for bulk intent
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
                    return {
                        **state,
                        "tool_result": None,
                        "used_tools": list(state.get("used_tools", [])),
                        "error": "Could not resolve selected task for move",
                    }
                task = tasks[0]
                arguments = {
                    "organization_id": task["organization_id"],
                    "project_id": task["project_id"],
                    "task_id": task["task_id"],
                    "target_project_id": task["target_project_id"],
                }
            else:
                arguments = move_arguments
        elif tool_name in BATCH_TOOL_RESOLVERS:
            arguments = BATCH_TOOL_RESOLVERS[tool_name](
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
            )

        if tool_name in {"create_task", "create_tasks"}:
            arguments.pop("task_ids", None)

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
                {key: arguments.get(key) for key in ("organization_id", "project_id", "title", "tasks")},
            )
            return {
                **state,
                "tool_result": None,
                "used_tools": list(state.get("used_tools", [])),
                "error": validation_error,
            }
        result = await execute_todo_tool(tools, tool_name, arguments)
        logger.info(
            "tool executed tool=%s success=%s",
            tool_name,
            _mutation_succeeded({**state, "tool_result": result, "error": None}),
        )
    except ArcTodoApiError as exc:
        return {
            **state,
            "tool_result": None,
            "used_tools": list(state.get("used_tools", [])),
            "error": str(exc),
        }
    except Exception as exc:
        return {
            **state,
            "tool_result": None,
            "used_tools": list(state.get("used_tools", [])),
            "error": str(exc),
        }

    used_tools = list(state.get("used_tools", []))
    used_tools.append(tool_name)

    return {
        **state,
        "tool_name": tool_name,
        "tool_result": result,
        "used_tools": used_tools,
        "error": None,
    }


async def response_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    if state.get("scope_status") in {"ambiguous", "not_found"} and _looks_like_create_mutation(
        state.get("latest_user_message", "")
    ):
        return {**state, "response": _build_mutation_failure_response(state)}

    model = build_model(runtime)
    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in state.get("messages", [])
    )
    tool_context = ""
    if state.get("tool_result") is not None:
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

    if _looks_like_task_mutation(state.get("latest_user_message", "")):
        succeeded = _mutation_succeeded(state)
        if succeeded is False or (
            succeeded is None and state.get("route") == "direct"
        ):
            response_text = _build_mutation_failure_response(state)

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
    if state.get("route") == "tools" and state.get("tool_name"):
        return "todo_tools_agent"
    return "response_agent"
