from __future__ import annotations

from typing import Any


def estimate_tokens(text: str) -> int:
    # ponytail: chars/4 heuristic; upgrade path is tiktoken when accuracy matters
    return max(1, len(text) // 4)


def _message_tokens(message: dict[str, Any]) -> int:
    role = str(message.get("role", ""))
    content = str(message.get("content", ""))
    return estimate_tokens(f"{role}: {content}")


def trim_messages(
    messages: list[dict[str, str]],
    *,
    max_messages: int | None,
    max_tokens: int | None,
) -> list[dict[str, str]]:
    if not messages:
        return []

    trimmed = list(messages)
    latest_user_index = max(
        (index for index, message in enumerate(trimmed) if message.get("role") == "user"),
        default=len(trimmed) - 1,
    )

    if max_messages is not None and max_messages > 0 and len(trimmed) > max_messages:
        keep = trimmed[-max_messages:]
        if trimmed[latest_user_index] not in keep:
            keep[-1] = trimmed[latest_user_index]
        trimmed = keep

    if max_tokens is not None and max_tokens > 0:
        while trimmed and sum(_message_tokens(message) for message in trimmed) > max_tokens:
            if len(trimmed) == 1:
                break
            drop_index = 0
            if drop_index == latest_user_index and len(trimmed) > 1:
                drop_index = 1
            trimmed.pop(drop_index)
            if latest_user_index > drop_index:
                latest_user_index -= 1
            elif latest_user_index == drop_index:
                latest_user_index = max(
                    (
                        index
                        for index, message in enumerate(trimmed)
                        if message.get("role") == "user"
                    ),
                    default=0,
                )

    return trimmed
