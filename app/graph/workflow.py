from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.chatbot_settings import ChatbotRuntimeSettings
from app.errors import from_exception
from app.graph.nodes import (
    context_agent,
    planner_agent,
    response_agent,
    route_after_context,
    route_after_planner,
    route_after_scope_discovery,
    route_after_tools,
    scope_discovery_agent,
    todo_tools_agent,
)
from app.graph.state import ChatGraphState
from app.history import trim_messages
from app.streaming import StreamEventHandler, bind_stream_handler, reset_stream_handler

logger = logging.getLogger(__name__)


def build_chat_graph(runtime: ChatbotRuntimeSettings):
    graph = StateGraph(ChatGraphState)

    async def planner_node(state: ChatGraphState) -> ChatGraphState:
        return await planner_agent(state, runtime)

    async def response_node(state: ChatGraphState) -> ChatGraphState:
        return await response_agent(state, runtime)

    async def tools_node(state: ChatGraphState) -> ChatGraphState:
        return await todo_tools_agent(state, runtime)

    graph.add_node("context_agent", context_agent)
    graph.add_node("scope_discovery_agent", scope_discovery_agent)
    graph.add_node("planner_agent", planner_node)
    graph.add_node("todo_tools_agent", tools_node)
    graph.add_node("response_agent", response_node)

    graph.add_edge(START, "context_agent")
    graph.add_conditional_edges(
        "context_agent",
        route_after_context,
        {
            "scope_discovery_agent": "scope_discovery_agent",
            "planner_agent": "planner_agent",
        },
    )
    graph.add_conditional_edges(
        "scope_discovery_agent",
        route_after_scope_discovery,
        {
            "todo_tools_agent": "todo_tools_agent",
            "planner_agent": "planner_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_conditional_edges(
        "planner_agent",
        route_after_planner,
        {
            "todo_tools_agent": "todo_tools_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_conditional_edges(
        "todo_tools_agent",
        route_after_tools,
        {
            "scope_discovery_agent": "scope_discovery_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_edge("response_agent", END)

    return graph.compile()


def _prepare_messages(
    messages: list[dict[str, str]],
    runtime: ChatbotRuntimeSettings,
) -> list[dict[str, str]]:
    trimmed = trim_messages(
        messages,
        max_messages=runtime.max_history_messages,
        max_tokens=runtime.max_history_tokens,
    )
    if len(trimmed) != len(messages):
        logger.info(
            "Trimmed conversation history from %s to %s messages",
            len(messages),
            len(trimmed),
        )
    return trimmed


def _build_initial_state(
    *,
    runtime: ChatbotRuntimeSettings,
    messages: list[dict[str, str]],
    user_token: str,
    organization_id: str | None,
    project_id: str | None,
    conversation_id: str | None,
    task_refs: list[dict[str, str]] | None,
) -> ChatGraphState:
    return {
        "messages": _prepare_messages(messages, runtime),
        "user_token": user_token,
        "organization_id": organization_id,
        "project_id": project_id,
        "conversation_id": conversation_id,
        "task_refs": task_refs or [],
        "used_tools": [],
    }


async def run_chat_workflow(
    *,
    runtime: ChatbotRuntimeSettings,
    messages: list[dict[str, str]],
    user_token: str,
    organization_id: str | None,
    project_id: str | None,
    conversation_id: str | None = None,
    task_refs: list[dict[str, str]] | None = None,
) -> ChatGraphState:
    graph = build_chat_graph(runtime)
    initial_state = _build_initial_state(
        runtime=runtime,
        messages=messages,
        user_token=user_token,
        organization_id=organization_id,
        project_id=project_id,
        conversation_id=conversation_id,
        task_refs=task_refs,
    )
    try:
        return await graph.ainvoke(initial_state)
    except Exception as exc:
        raise from_exception(exc, stage="workflow") from exc


async def run_chat_workflow_streaming(
    *,
    runtime: ChatbotRuntimeSettings,
    messages: list[dict[str, str]],
    user_token: str,
    organization_id: str | None,
    project_id: str | None,
    conversation_id: str | None = None,
    task_refs: list[dict[str, str]] | None = None,
    handler: StreamEventHandler,
) -> None:
    graph = build_chat_graph(runtime)
    initial_state = _build_initial_state(
        runtime=runtime,
        messages=messages,
        user_token=user_token,
        organization_id=organization_id,
        project_id=project_id,
        conversation_id=conversation_id,
        task_refs=task_refs,
    )
    token = bind_stream_handler(handler)
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        workflow_error = from_exception(exc, stage="workflow")
        await handler.emit_error(workflow_error.message, code=workflow_error.code)
        raise workflow_error from exc
    finally:
        reset_stream_handler(token)
