from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def map_limited(
    items: list[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    limit: int,
) -> list[R | BaseException]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(item: T) -> R | BaseException:
        async with semaphore:
            try:
                return await worker(item)
            except BaseException as exc:
                return exc

    return await asyncio.gather(*(run(item) for item in items))
