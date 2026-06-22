import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.workflow import run_chat_workflow


@pytest.mark.asyncio
async def test_run_chat_workflow_persists_conversation_turn():
    runtime = MagicMock()
    runtime.max_history_messages = 50
    runtime.max_history_tokens = 100000

    graph_result = {
        "response": "Here are your tasks.",
        "used_tools": ["list_tasks"],
    }

    with patch("app.graph.workflow.build_chat_graph") as build_graph, patch(
        "app.graph.workflow.ArcTodoClient"
    ) as client_cls, patch(
        "app.graph.workflow.prepare_conversation_messages",
        new=AsyncMock(
            return_value=(
                [{"role": "user", "content": "list tasks"}],
                {"role": "user", "content": "list tasks"},
            )
        ),
    ), patch(
        "app.graph.workflow.persist_conversation_turn",
        new=AsyncMock(),
    ) as persist:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value=graph_result)
        build_graph.return_value = graph
        client_cls.return_value = MagicMock()

        result = await run_chat_workflow(
            runtime=runtime,
            messages=[{"role": "user", "content": "list tasks"}],
            user_token="token",
            organization_id="org-1",
            project_id="proj-1",
            conversation_id="conv-1",
        )

    assert result["response"] == "Here are your tasks."
    persist.assert_awaited_once()
    kwargs = persist.await_args.kwargs
    assert kwargs["assistant_message"] == "Here are your tasks."
    assert kwargs["used_tools"] == ["list_tasks"]
