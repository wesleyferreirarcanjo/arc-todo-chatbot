from __future__ import annotations

import contextvars
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEventHandler:
    queue: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def emit_token(self, delta: str) -> None:
        if delta:
            self.queue.append(("token", {"delta": delta}))

    async def emit_done(self, message: str, used_tools: list[str]) -> None:
        self.queue.append(
            (
                "done",
                {
                    "message": message,
                    "usedTools": used_tools,
                },
            )
        )

    async def emit_error(self, message: str, *, code: str = "unexpected") -> None:
        self.queue.append(("error", {"code": code, "message": message}))


stream_handler_var: contextvars.ContextVar[StreamEventHandler | None] = contextvars.ContextVar(
    "stream_handler",
    default=None,
)


def get_stream_handler() -> StreamEventHandler | None:
    return stream_handler_var.get()


def bind_stream_handler(handler: StreamEventHandler | None) -> contextvars.Token[StreamEventHandler | None]:
    return stream_handler_var.set(handler)


def reset_stream_handler(token: contextvars.Token[StreamEventHandler | None]) -> None:
    stream_handler_var.reset(token)


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def iter_sse_events(
    handler: StreamEventHandler,
) -> AsyncIterator[str]:
    index = 0
    while index < len(handler.queue):
        event, data = handler.queue[index]
        index += 1
        yield format_sse(event, data)
        if event in {"done", "error"}:
            return
    while True:
        while index < len(handler.queue):
            event, data = handler.queue[index]
            index += 1
            yield format_sse(event, data)
            if event in {"done", "error"}:
                return
        await _wait_for_queue(handler, index)
        if index >= len(handler.queue):
            break


async def _wait_for_queue(handler: StreamEventHandler, index: int) -> None:
    import asyncio

    for _ in range(50):
        if index < len(handler.queue):
            return
        await asyncio.sleep(0.01)
