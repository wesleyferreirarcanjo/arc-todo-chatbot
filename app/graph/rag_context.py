from __future__ import annotations

from typing import Any


def build_rag_context_text(
    chunks: list[dict[str, Any]] | None,
    *,
    rag_error: str | None = None,
) -> str:
    if rag_error:
        return f"Retrieved knowledge context: unavailable ({rag_error})."

    if not chunks:
        return ""

    lines = ["Retrieved knowledge context:"]
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or chunk.get("sourceFilename") or "Untitled"
        source = chunk.get("sourceFilename") or chunk.get("title") or "unknown source"
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{index}] {title} ({source})\n{text}")
    return "\n\n".join(lines)
