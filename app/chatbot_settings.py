from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import get_settings


class ChatbotSettingsError(Exception):
    pass


class ChatbotRuntimeSettings:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        enabled: bool,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.enabled = enabled


class ChatbotSettingsClient:
    def __init__(self) -> None:
        self._cache: ChatbotRuntimeSettings | None = None
        self._cache_expires_at = 0.0

    async def _service_headers(self) -> dict[str, str]:
        settings = get_settings()
        base_url = settings.arc_todo_api_base_url.rstrip("/")

        if settings.arc_todo_access_token:
            return {"Authorization": f"Bearer {settings.arc_todo_access_token}"}

        if settings.arc_todo_username and settings.arc_todo_password:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/auth/login",
                    json={
                        "username": settings.arc_todo_username,
                        "password": settings.arc_todo_password,
                    },
                )
                response.raise_for_status()
                token = response.json()["access_token"]
                return {"Authorization": f"Bearer {token}"}

        raise ChatbotSettingsError(
            "Cannot load chatbot settings without ARC_TODO_ACCESS_TOKEN or "
            "ARC_TODO_USERNAME/ARC_TODO_PASSWORD"
        )

    def _parse_runtime_settings(self, payload: dict[str, Any]) -> ChatbotRuntimeSettings:
        api_key = payload.get("apiKey")
        if not api_key:
            raise ChatbotSettingsError("Chatbot provider API key is not configured")

        return ChatbotRuntimeSettings(
            provider=payload.get("provider", "deepseek"),
            base_url=payload.get("baseUrl", "https://api.deepseek.com"),
            model=payload.get("model", "deepseek-chat"),
            api_key=api_key,
            temperature=float(payload.get("temperature", 0.2)),
            enabled=bool(payload.get("enabled", False)),
        )

    async def get_runtime_settings(self, *, force_refresh: bool = False) -> ChatbotRuntimeSettings:
        settings = get_settings()
        now = time.monotonic()
        if (
            not force_refresh
            and self._cache is not None
            and now < self._cache_expires_at
        ):
            return self._cache

        headers = await self._service_headers()
        base_url = settings.arc_todo_api_base_url.rstrip("/")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{base_url}/chatbot-settings/runtime",
                headers=headers,
            )
            if not response.is_success:
                raise ChatbotSettingsError(
                    f"Failed to load chatbot settings ({response.status_code})"
                )
            runtime = self._parse_runtime_settings(response.json())

        self._cache = runtime
        self._cache_expires_at = now + settings.chatbot_settings_cache_seconds
        return runtime


chatbot_settings_client = ChatbotSettingsClient()
