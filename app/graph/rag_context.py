from __future__ import annotations

from typing import Any


def build_retrieval_query(
    messages: list[dict[str, str]] | None,
    latest_user_message: str,
    *,
    max_prior_turns: int = 3,
) -> str:
    """Build a retrieval query from recent user turns plus the latest message."""
    # ponytail: fixed window of prior user turns; upgrade path is LLM query condensation
    latest = latest_user_message.strip()
    if not latest:
        return ""

    prior_user = [
        str(message.get("content") or "").strip()
        for message in (messages or [])
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    if prior_user and prior_user[-1] == latest:
        prior_user = prior_user[:-1]
    prior_user = prior_user[-max_prior_turns:]
    if not prior_user:
        return latest
    return "\n".join([*prior_user, latest])


def format_recent_conversation(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").capitalize()
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _chunk_header(index: int, chunk: dict[str, Any]) -> str:
    title = chunk.get("title") or chunk.get("sourceFilename") or "Untitled"
    source = chunk.get("sourceFilename") or chunk.get("title") or "unknown source"
    meta_bits: list[str] = []
    if chunk.get("scope"):
        meta_bits.append(f"scope={chunk['scope']}")
    if chunk.get("chunkIndex") is not None:
        meta_bits.append(f"chunk={chunk['chunkIndex']}")
    if chunk.get("score") is not None:
        meta_bits.append(f"score={float(chunk['score']):.2f}")
    if chunk.get("knowledgeEntryId"):
        meta_bits.append(f"entry={chunk['knowledgeEntryId']}")
    if chunk.get("updatedAt"):
        meta_bits.append(f"updated={chunk['updatedAt']}")
    if chunk.get("compressed"):
        meta_bits.append("summary")
    meta = f" [{', '.join(meta_bits)}]" if meta_bits else ""
    header = f"[{index}] {title} ({source}){meta}"
    if chunk.get("helperReason"):
        header += f"\nRelevance: {chunk['helperReason']}"
    return header


def build_rag_context_text(
    chunks: list[dict[str, Any]] | None,
    *,
    rag_error: str | None = None,
    index_status: dict[str, Any] | None = None,
) -> str:
    if rag_error:
        return f"Retrieved knowledge context: unavailable ({rag_error})."

    lines: list[str] = ["Retrieved knowledge context:"]
    queued_jobs = int((index_status or {}).get("queuedJobs") or 0)
    if queued_jobs > 0:
        lines.append(
            f"Note: {queued_jobs} index job(s) queued; excerpts may be stale until indexing completes."
        )

    if not chunks:
        if queued_jobs > 0:
            lines.append("No indexed chunks matched this query yet.")
        else:
            return ""
        return "\n\n".join(lines)

    for index, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{_chunk_header(index, chunk)}\n{text}")
    return "\n\n".join(lines)
