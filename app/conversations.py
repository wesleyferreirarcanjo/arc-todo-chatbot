from __future__ import annotations

from typing import Any

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient

ChatMessageDict = dict[str, str]


def _normalize_api_message(message: dict[str, Any]) -> ChatMessageDict:
    role = str(message.get("role") or "user")
    if role not in {"user", "assistant", "system"}:
        role = "user"
    return {"role": role, "content": str(message.get("content") or "")}


def merge_conversation_messages(
    persisted: list[ChatMessageDict],
    incoming: list[ChatMessageDict],
) -> tuple[list[ChatMessageDict], ChatMessageDict | None]:
    """Merge persisted API history with the incoming request messages.

    Returns merged history and the new user message to persist after a successful turn.
    """
    if not incoming:
        return persisted, None

    merged = list(persisted)
    new_user_message: ChatMessageDict | None = None

    for message in incoming:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        if (
            merged
            and merged[-1]["role"] == role
            and merged[-1]["content"] == content
        ):
            continue
        merged.append({"role": role, "content": content})
        if role == "user":
            new_user_message = {"role": role, "content": content}

    return merged, new_user_message


async def load_conversation_messages(
    client: ArcTodoClient,
    conversation_id: str,
) -> list[ChatMessageDict]:
    data = await client.get_conversation(conversation_id)
    messages = data.get("messages") or []
    return [_normalize_api_message(message) for message in messages]


async def prepare_conversation_messages(
    client: ArcTodoClient,
    conversation_id: str | None,
    incoming_messages: list[ChatMessageDict],
) -> tuple[list[ChatMessageDict], ChatMessageDict | None]:
    if not conversation_id:
        return incoming_messages, None

    try:
        persisted = await load_conversation_messages(client, conversation_id)
    except ArcTodoApiError:
        persisted = []

    return merge_conversation_messages(persisted, incoming_messages)


async def persist_conversation_turn(
    client: ArcTodoClient,
    conversation_id: str,
    *,
    user_message: ChatMessageDict | None,
    assistant_message: str,
    used_tools: list[str] | None = None,
) -> None:
    if user_message:
        await client.add_conversation_message(
            conversation_id,
            role=user_message["role"],
            content=user_message["content"],
        )
    if assistant_message.strip():
        await client.add_conversation_message(
            conversation_id,
            role="assistant",
            content=assistant_message,
            used_tools=used_tools or [],
        )
