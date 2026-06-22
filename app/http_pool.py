from __future__ import annotations

import httpx

from app.config import settings

_shared_client: httpx.AsyncClient | None = None


def get_shared_http_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        # ponytail: lazy init for tests/CLI when FastAPI lifespan is not running
        _shared_client = create_shared_http_client()
    return _shared_client


def create_shared_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
        ),
    )


def set_shared_http_client(client: httpx.AsyncClient | None) -> None:
    global _shared_client
    _shared_client = client
