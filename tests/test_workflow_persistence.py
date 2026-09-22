import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.conversations import (
    merge_conversation_messages,
    persist_conversation_turn,
    prepare_conversation_messages,
)
from app.graph.workflow import run_chat_workflow


def test_merge_conversation_messages_appends_new_user_message():
    persisted = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    incoming = [{"role": "user", "content": "list tasks"}]

    merged, new_user = merge_conversation_messages(persisted, incoming)

    assert merged == persisted + incoming
    assert new_user == incoming[0]


def test_merge_conversation_messages_skips_duplicate_user_message():
    persisted = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "list tasks"},
    ]
    incoming = [{"role": "user", "content": "list tasks"}]

    merged, new_user = merge_conversation_messages(persisted, incoming)

    assert merged == persisted
    assert new_user is None


def test_merge_conversation_messages_strips_replayed_full_history():
    persisted = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    incoming = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "list tasks"},
    ]

    merged, new_user = merge_conversation_messages(persisted, incoming)

    assert merged == persisted + [{"role": "user", "content": "list tasks"}]
    assert new_user == {"role": "user", "content": "list tasks"}


@pytest.mark.asyncio
async def test_prepare_conversation_messages_loads_persisted_history():
    client = MagicMock()
    client.get_conversation = AsyncMock(
        return_value={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
    )

    merged, new_user = await prepare_conversation_messages(
        client,
        "conv-1",
        [{"role": "user", "content": "next question"}],
    )

    assert merged == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next question"},
    ]
    assert new_user == {"role": "user", "content": "next question"}
    client.get_conversation.assert_awaited_once_with("conv-1")


@pytest.mark.asyncio
async def test_persist_conversation_turn_saves_user_and_assistant():
    client = MagicMock()
    client.add_conversation_message = AsyncMock(return_value={"id": "msg-1"})

    await persist_conversation_turn(
        client,
        "conv-1",
        user_message={"role": "user", "content": "hello"},
        assistant_message="hi there",
        used_tools=["list_tasks"],
    )

    assert client.add_conversation_message.await_count == 2
    client.add_conversation_message.assert_any_await(
        "conv-1",
        role="user",
        content="hello",
    )
    client.add_conversation_message.assert_any_await(
        "conv-1",
        role="assistant",
        content="hi there",
        used_tools=["list_tasks"],
    )


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
