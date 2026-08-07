from __future__ import annotations

import logging
import time
from typing import Any

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient

logger = logging.getLogger(__name__)

# ponytail: process-local TTL cache — fine for a single uvicorn worker; upgrade to Redis if multi-instance.
_CACHE_TTL_SECONDS = 300.0
_scope_cache: dict[str, tuple[float, dict[str, str | None]]] = {}


def _cache_get(key: str) -> dict[str, str | None] | None:
    entry = _scope_cache.get(key)
    if not entry:
        return None
    expires_at, value = entry
    if time.monotonic() >= expires_at:
        _scope_cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: dict[str, str | None]) -> None:
    _scope_cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, value)


async def build_scope_context(
    *,
    user_token: str,
    organization_id: str | None,
    project_id: str | None,
) -> dict[str, str | None]:
    """Fetch org name + project name/description with a short TTL cache."""
    result: dict[str, str | None] = {
        "organization_id": organization_id,
        "organization_name": None,
        "project_id": project_id,
        "project_name": None,
        "project_description": None,
    }
    if not organization_id and not project_id:
        return result

    cache_key = f"{organization_id or ''}:{project_id or ''}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = ArcTodoClient(user_token=user_token)
    try:
        if organization_id:
            org = await client.request("GET", f"/organizations/{organization_id}")
            if isinstance(org, dict):
                result["organization_name"] = org.get("name")
        if organization_id and project_id:
            project = await client.request(
                "GET",
                f"/organizations/{organization_id}/projects/{project_id}",
            )
            if isinstance(project, dict):
                result["project_name"] = project.get("name")
                description = project.get("description")
                result["project_description"] = (
                    str(description).strip() if description else None
                ) or None
    except ArcTodoApiError as exc:
        logger.warning("Failed to load scope context: %s", exc)

    _cache_set(cache_key, result)
    return result


def format_scope_context_text(scope: dict[str, Any] | None) -> str:
    if not scope:
        return ""
    bits: list[str] = []
    org_name = scope.get("organization_name")
    org_id = scope.get("organization_id")
    if org_name:
        bits.append(f'organization_name="{org_name}"')
    if org_id:
        bits.append(f"organization_id={org_id}")
    project_name = scope.get("project_name")
    project_id = scope.get("project_id")
    if project_name:
        bits.append(f'project_name="{project_name}"')
    if project_id:
        bits.append(f"project_id={project_id}")
    project_description = scope.get("project_description")
    if project_description:
        bits.append(f'project_description="{project_description}"')
    if not bits:
        return ""
    return "Current scope: " + ", ".join(bits)
