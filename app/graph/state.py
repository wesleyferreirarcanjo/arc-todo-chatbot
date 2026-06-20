from __future__ import annotations

from typing import Any, Literal, TypedDict


class ChatGraphState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_token: str
    organization_id: str | None
    project_id: str | None
    latest_user_message: str
    route: Literal["direct", "tools"]
    tool_name: str | None
    tool_arguments: dict[str, Any]
    tool_result: Any
    used_tools: list[str]
    response: str
    error: str | None
