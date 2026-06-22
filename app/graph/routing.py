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
from app.graph.heuristics import _looks_like_create_mutation
from app.graph.scope import _is_uuid, _needs_scope_retry

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
