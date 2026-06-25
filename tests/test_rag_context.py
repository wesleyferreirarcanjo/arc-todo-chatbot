from app.graph.rag_context import (
    build_rag_context_text,
    build_retrieval_query,
    compact_task_context_for_retrieval,
)


def test_build_retrieval_query_uses_latest_only_without_history():
    assert build_retrieval_query([], "how do I deploy?") == "how do I deploy?"


def test_build_retrieval_query_includes_prior_user_turns():
    messages = [
        {"role": "user", "content": "Tell me about deployment"},
        {"role": "assistant", "content": "Use Coolify."},
        {"role": "user", "content": "What about step 2?"},
    ]
    query = build_retrieval_query(messages, "What about step 2?")
    assert "Tell me about deployment" in query
    assert query.endswith("What about step 2?")


def test_compact_task_context_for_retrieval_strips_scope_ids():
    task_context = (
        "Selected task context:\n"
        "- taskId: uuid-1\n"
        "  displayId: #arc-106\n"
        "  title: Improve RAG integration\n"
        "  status: in_progress\n"
        "  criticity: medium\n"
        "  category: coding\n"
        "  organizationId: org-1\n"
        "  projectId: proj-1\n"
        "  dueDate: none\n"
        "  description: Connect tasks with knowledge search."
    )
    compact = compact_task_context_for_retrieval(task_context)
    assert "displayId: #arc-106" in compact
    assert "Improve RAG integration" in compact
    assert "Connect tasks with knowledge search." in compact
    assert "organizationId" not in compact
    assert "projectId" not in compact
    assert "taskId" not in compact


def test_compact_task_context_for_retrieval_bounds_length():
    long_description = "x" * 2000
    task_context = (
        "Selected task context:\n"
        f"  title: Long task\n"
        f"  description: {long_description}"
    )
    compact = compact_task_context_for_retrieval(task_context, max_chars=100)
    assert len(compact) <= 100
    assert compact.endswith("...")


def test_compact_task_context_for_retrieval_ignores_empty():
    assert compact_task_context_for_retrieval("") == ""
    assert compact_task_context_for_retrieval("Selected task context:") == ""


def test_build_retrieval_query_includes_task_context():
    task_context = (
        "Selected task context:\n"
        "- taskId: uuid-1\n"
        "  displayId: #arc-106\n"
        "  title: Improve RAG integration\n"
        "  status: in_progress\n"
        "  description: Connect tasks with knowledge search."
    )
    query = build_retrieval_query(
        [],
        "How should I verify this?",
        task_context_text=task_context,
    )
    assert query.startswith("How should I verify this?")
    assert "Selected task context:" in query
    assert "#arc-106" in query
    assert "Improve RAG integration" in query


def test_build_retrieval_query_without_task_context_unchanged():
    assert build_retrieval_query([], "how do I deploy?") == "how do I deploy?"
    assert build_retrieval_query([], "how do I deploy?", task_context_text="") == "how do I deploy?"


def test_build_rag_context_text_includes_metadata():
    text = build_rag_context_text(
        [
            {
                "title": "Setup",
                "sourceFilename": "setup.md",
                "scope": "project",
                "chunkIndex": 1,
                "score": 0.82,
                "knowledgeEntryId": "entry-1",
                "updatedAt": "2026-06-23T00:00:00+00:00",
                "text": "Install dependencies first.",
            }
        ]
    )
    assert "scope=project" in text
    assert "score=0.82" in text
    assert "entry=entry-1" in text
    assert "Install dependencies first." in text


def test_build_rag_context_text_notes_stale_index():
    text = build_rag_context_text(
        [],
        index_status={"queuedJobs": 3, "totalChunks": 0},
    )
    assert "3 index job(s) queued" in text
    assert "No indexed chunks matched" in text


def test_build_rag_context_text_handles_error():
    text = build_rag_context_text([], rag_error="RAG is disabled")
    assert "unavailable (RAG is disabled)" in text
