from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient
from app.chatbot_settings import ChatbotRuntimeSettings
from app.conversations import persist_conversation_turn, prepare_conversation_messages
from app.errors import from_exception
from app.graph.nodes import (
    context_agent,
    planner_agent,
    response_agent,
    retrieval_agent,
    route_after_context,
    route_after_planner,
    route_after_retrieval,
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
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("scope_discovery_agent", scope_discovery_agent)
    graph.add_node("planner_agent", planner_node)
    graph.add_node("todo_tools_agent", tools_node)
    graph.add_node("response_agent", response_node)

    graph.add_edge(START, "context_agent")
    graph.add_conditional_edges(
        "context_agent",
        route_after_context,
        {
            "retrieval_agent": "retrieval_agent",
        },
    )
    graph.add_conditional_edges(
        "retrieval_agent",
        route_after_retrieval,
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
        "rag_chunks": [],
        "rag_context_text": "",
        "rag_error": None,
        "rag_search_query": None,
        "rag_token_usage": None,
        "rag_index_status": None,
    }


async def _run_workflow_with_persistence(
    *,
    runtime: ChatbotRuntimeSettings,
    messages: list[dict[str, str]],
    user_token: str,
    organization_id: str | None,
    project_id: str | None,
    conversation_id: str | None,
    task_refs: list[dict[str, str]] | None,
) -> ChatGraphState:
    client = ArcTodoClient(user_token=user_token)
    prepared_messages, user_message_to_persist = await prepare_conversation_messages(
        client,
        conversation_id,
        messages,
    )
    graph = build_chat_graph(runtime)
    initial_state = _build_initial_state(
        runtime=runtime,
        messages=prepared_messages,
        user_token=user_token,
        organization_id=organization_id,
        project_id=project_id,
        conversation_id=conversation_id,
        task_refs=task_refs,
    )
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        raise from_exception(exc, stage="workflow") from exc

    if conversation_id:
        try:
            await persist_conversation_turn(
                client,
                conversation_id,
                user_message=user_message_to_persist,
                assistant_message=result.get("response") or "",
                used_tools=result.get("used_tools", []),
            )
        except ArcTodoApiError as exc:
            logger.warning(
                "Failed to persist conversation %s: %s",
                conversation_id,
                exc,
            )

    return result


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
    return await _run_workflow_with_persistence(
        runtime=runtime,
        messages=messages,
        user_token=user_token,
        organization_id=organization_id,
        project_id=project_id,
        conversation_id=conversation_id,
        task_refs=task_refs,
    )


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
    client = ArcTodoClient(user_token=user_token)
    prepared_messages, user_message_to_persist = await prepare_conversation_messages(
        client,
        conversation_id,
        messages,
    )
    graph = build_chat_graph(runtime)
    initial_state = _build_initial_state(
        runtime=runtime,
        messages=prepared_messages,
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

    if conversation_id:
        try:
            await persist_conversation_turn(
                client,
                conversation_id,
                user_message=user_message_to_persist,
                assistant_message=result.get("response") or "",
                used_tools=result.get("used_tools", []),
            )
        except ArcTodoApiError as exc:
            logger.warning(
                "Failed to persist conversation %s: %s",
                conversation_id,
                exc,
            )
