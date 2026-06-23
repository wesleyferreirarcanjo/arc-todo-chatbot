from app.graph.rag_context import build_rag_context_text, build_retrieval_query


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
