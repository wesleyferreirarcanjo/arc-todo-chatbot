from __future__ import annotations

from typing import Any, Literal, TypedDict


class ActionPlanItem(TypedDict, total=False):
    intent: str
    tool_name: str
    tool_arguments: dict[str, Any]


class ActionResult(TypedDict, total=False):
    intent: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_result: Any
    error: str | None
    success: bool
    partial: bool


class ChatGraphState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_token: str
    organization_id: str | None
    project_id: str | None
    conversation_id: str | None
    task_refs: list[dict[str, str]]
    task_context_text: str
    latest_user_message: str
    route: Literal["direct", "tools"]
    actions: list[ActionPlanItem]
    tool_name: str | None
    tool_arguments: dict[str, Any]
    tool_result: Any
    tool_results: list[ActionResult]
    used_tools: list[str]
    scope_catalog: dict[str, Any]
    scope_retried: bool
    scope_status: Literal["resolved", "ambiguous", "not_found"]
    response: str
    error: str | None
    rag_chunks: list[dict[str, Any]]
    rag_context_text: str
    rag_error: str | None
