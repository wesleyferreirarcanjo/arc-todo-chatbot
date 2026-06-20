from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.chatbot_settings import ChatbotRuntimeSettings
from app.graph.nodes import (
    context_agent,
    planner_agent,
    response_agent,
    route_after_planner,
    todo_tools_agent,
)
from app.graph.state import ChatGraphState


def build_chat_graph(runtime: ChatbotRuntimeSettings):
    graph = StateGraph(ChatGraphState)

    async def planner_node(state: ChatGraphState) -> ChatGraphState:
        return await planner_agent(state, runtime)

    async def response_node(state: ChatGraphState) -> ChatGraphState:
        return await response_agent(state, runtime)

    graph.add_node("context_agent", context_agent)
    graph.add_node("planner_agent", planner_node)
    graph.add_node("todo_tools_agent", todo_tools_agent)
    graph.add_node("response_agent", response_node)

    graph.add_edge(START, "context_agent")
    graph.add_edge("context_agent", "planner_agent")
    graph.add_conditional_edges(
        "planner_agent",
        route_after_planner,
        {
            "todo_tools_agent": "todo_tools_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_edge("todo_tools_agent", "response_agent")
    graph.add_edge("response_agent", END)

    return graph.compile()


async def run_chat_workflow(
    *,
    runtime: ChatbotRuntimeSettings,
    messages: list[dict[str, str]],
    user_token: str,
    organization_id: str | None,
    project_id: str | None,
) -> ChatGraphState:
    graph = build_chat_graph(runtime)
    initial_state: ChatGraphState = {
        "messages": messages,
        "user_token": user_token,
        "organization_id": organization_id,
        "project_id": project_id,
        "used_tools": [],
    }
    return await graph.ainvoke(initial_state)
