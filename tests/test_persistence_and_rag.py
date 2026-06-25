import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.conversations import (
    merge_conversation_messages,
    persist_conversation_turn,
    prepare_conversation_messages,
)
from app.tools.knowledge_tools import KnowledgeTools, execute_knowledge_tool


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
async def test_execute_knowledge_tool_list_persons():
    client = MagicMock()
    client.request = AsyncMock(return_value=[{"id": "p1", "name": "Ada"}])
    tools = KnowledgeTools(client)

    result = await execute_knowledge_tool(
        tools,
        "list_persons",
        {"organization_id": "org-1"},
    )

    assert result == [{"id": "p1", "name": "Ada"}]
    client.request.assert_awaited_once_with(
        "GET",
        "/organizations/org-1/persons",
    )


@pytest.mark.asyncio
async def test_execute_knowledge_tool_get_knowledge():
    client = MagicMock()
    client.request = AsyncMock(return_value={"id": "k1", "title": "Docs"})
    tools = KnowledgeTools(client)

    result = await execute_knowledge_tool(
        tools,
        "get_knowledge",
        {
            "scope": "project",
            "organization_id": "org-1",
            "project_id": "proj-1",
            "knowledge_id": "k1",
        },
    )

    assert result["title"] == "Docs"
    client.request.assert_awaited_once_with(
        "GET",
        "/organizations/org-1/projects/proj-1/knowledge/k1",
    )


@pytest.mark.asyncio
async def test_retrieval_agent_injects_chunks_and_fails_soft():
    from app.graph.nodes import retrieval_agent, route_after_context, route_after_retrieval

    with patch("app.rag_client.RagClient") as rag_cls:
        rag_cls.return_value.retrieve = AsyncMock(
            return_value={
                "chunks": [
                    {
                        "title": "Setup",
                        "sourceFilename": "setup.md",
                        "scope": "project",
                        "chunkIndex": 0,
                        "score": 0.91,
                        "text": "Install dependencies first.",
                    }
                ],
                "searchQuery": "how do I install?",
                "tokenUsage": {"totalTokens": 42},
                "indexStatus": {"queuedJobs": 0, "totalChunks": 10},
            }
        )
        state = await retrieval_agent(
            {
                "user_token": "token",
                "latest_user_message": "how do I install?",
                "messages": [{"role": "user", "content": "how do I install?"}],
                "organization_id": "org-1",
                "project_id": "proj-1",
            }
        )

    assert len(state["rag_chunks"]) == 1
    assert "Install dependencies first." in state["rag_context_text"]
    assert "scope=project" in state["rag_context_text"]
    assert state["rag_search_query"] == "how do I install?"
    assert state["rag_token_usage"]["totalTokens"] == 42
    assert state["rag_index_status"]["totalChunks"] == 10
    assert state["rag_error"] is None
    assert route_after_context({}) == "retrieval_agent"
    assert route_after_retrieval({"latest_user_message": "hello"}) == "planner_agent"


@pytest.mark.asyncio
async def test_retrieval_agent_builds_follow_up_query():
    from app.graph.agents import retrieval_agent

    with patch("app.rag_client.RagClient") as rag_cls:
        rag_cls.return_value.retrieve = AsyncMock(return_value={"chunks": []})
        await retrieval_agent(
            {
                "user_token": "token",
                "latest_user_message": "What about step 2?",
                "messages": [
                    {"role": "user", "content": "Explain deployment steps"},
                    {"role": "assistant", "content": "Step 1 is setup."},
                    {"role": "user", "content": "What about step 2?"},
                ],
            }
        )
        question = rag_cls.return_value.retrieve.await_args.kwargs["question"]
        assert "Explain deployment steps" in question
        assert question.endswith("What about step 2?")


@pytest.mark.asyncio
async def test_retrieval_agent_forwards_task_context_to_rag():
    from app.graph.agents import retrieval_agent

    task_context = (
        "Selected task context:\n"
        "- taskId: uuid-1\n"
        "  displayId: #arc-106\n"
        "  title: Improve RAG integration\n"
        "  status: in_progress\n"
        "  description: Connect tasks with knowledge search."
    )
    with patch("app.rag_client.RagClient") as rag_cls:
        rag_cls.return_value.retrieve = AsyncMock(return_value={"chunks": []})
        await retrieval_agent(
            {
                "user_token": "token",
                "latest_user_message": "How should I verify this?",
                "messages": [{"role": "user", "content": "How should I verify this?"}],
                "task_context_text": task_context,
            }
        )
        question = rag_cls.return_value.retrieve.await_args.kwargs["question"]
        assert "How should I verify this?" in question
        assert "#arc-106" in question
        assert "Improve RAG integration" in question
        assert "taskId" not in question


@pytest.mark.asyncio
async def test_retrieval_agent_handles_rag_failure():
    from app.graph.agents import retrieval_agent
    from app.rag_client import RagClientError

    with patch("app.rag_client.RagClient") as rag_cls:
        rag_cls.return_value.retrieve = AsyncMock(
            side_effect=RagClientError("RAG is disabled", 503)
        )
        state = await retrieval_agent(
            {
                "user_token": "token",
                "latest_user_message": "how do I install?",
            }
        )

    assert state["rag_chunks"] == []
    assert "unavailable" in state["rag_context_text"]
    assert state["rag_error"] == "RAG is disabled"
