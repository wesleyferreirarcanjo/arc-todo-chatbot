from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings


class ArcTodoApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ArcTodoClient:
    def __init__(self, user_token: str | None = None) -> None:
        settings = get_settings()
        self._base_url = settings.arc_todo_api_base_url.rstrip("/")
        self._service_token = settings.arc_todo_access_token
        self._username = settings.arc_todo_username
        self._password = settings.arc_todo_password
        self._user_token = user_token
        self._cached_service_token: str | None = None

    async def _ensure_service_token(self, client: httpx.AsyncClient) -> None:
        if self._cached_service_token:
            return
        if self._service_token:
            self._cached_service_token = self._service_token
            return
        if not self._username or not self._password:
            raise ArcTodoApiError(
                "Missing credentials: set ARC_TODO_ACCESS_TOKEN or "
                "ARC_TODO_USERNAME and ARC_TODO_PASSWORD"
            )
        response = await client.post(
            f"{self._base_url}/auth/login",
            json={"username": self._username, "password": self._password},
        )
        if not response.is_success:
            await self._raise_api_error(response)
        self._cached_service_token = response.json()["access_token"]

    def _auth_headers(self, *, use_service_token: bool = False) -> dict[str, str]:
        if use_service_token:
            token = self._cached_service_token or self._service_token
        else:
            token = self._user_token
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    async def _raise_api_error(self, response: httpx.Response) -> None:
        message = f"Request failed ({response.status_code})"
        try:
            data = response.json()
            if isinstance(data.get("message"), list):
                message = ", ".join(data["message"])
            elif isinstance(data.get("message"), str):
                message = data["message"]
        except Exception:
            pass
        raise ArcTodoApiError(message, response.status_code)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        use_service_token: bool = False,
    ) -> Any:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if use_service_token:
                await self._ensure_service_token(client)
            response = await client.request(
                method,
                f"{self._base_url}{path}",
                params=params,
                json=json_body,
                headers=self._auth_headers(use_service_token=use_service_token),
            )
            if not response.is_success:
                await self._raise_api_error(response)
            if response.status_code == 204:
                return None
            if not response.content:
                return None
            return response.json()

    @staticmethod
    def format_result(data: Any) -> str:
        return json.dumps(data, indent=2, default=str)
