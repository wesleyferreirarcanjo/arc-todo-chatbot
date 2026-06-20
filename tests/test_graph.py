import pytest

from app.graph.nodes import context_agent, route_after_planner


@pytest.mark.asyncio
async def test_context_agent_extracts_latest_user_message():
    state = await context_agent(
        {
            "user_token": "token",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "list tasks"},
            ],
        }
    )
    assert state["latest_user_message"] == "list tasks"


def test_route_after_planner_tools():
    assert route_after_planner({"route": "tools", "tool_name": "list_tasks"}) == "todo_tools_agent"


def test_route_after_planner_direct():
    assert route_after_planner({"route": "direct"}) == "response_agent"
