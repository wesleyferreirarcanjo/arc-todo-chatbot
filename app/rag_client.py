from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings
from app.http_pool import get_shared_http_client

logger = logging.getLogger(__name__)


class RagClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def select_retrieve_route(
    *,
    organization_id: str | None = None,
    project_id: str | None = None,
    person_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Return RAG retrieve path and scope ids for the request body."""
    if organization_id and project_id:
        return "/retrieve/project", {
            "organizationId": organization_id,
            "projectId": project_id,
        }
    if organization_id:
        return "/retrieve/organization", {"organizationId": organization_id}
    if person_id:
        scope: dict[str, str] = {"personId": person_id}
        if organization_id:
            scope["organizationId"] = organization_id
        return "/retrieve/person", scope
    return "/retrieve/general", {}


class RagClient:
    def __init__(
        self,
        user_token: str,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = settings.rag_api_base_url.rstrip("/")
        self._user_token = user_token
        self._timeout = settings.rag_timeout_seconds
        self._top_k = settings.rag_top_k
        self._max_context_tokens = settings.rag_max_context_tokens
        self._http_client = http_client

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        return get_shared_http_client()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._user_token}"}

    @staticmethod
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        text = response.text.strip()
        if not text:
            raise RagClientError(
                f"RAG returned an empty response ({response.status_code})",
                response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RagClientError(
                f"RAG returned invalid JSON ({response.status_code})",
                response.status_code,
            ) from exc
        return data if isinstance(data, dict) else {}

    async def retrieve(
        self,
        *,
        question: str,
        organization_id: str | None = None,
        project_id: str | None = None,
        person_id: str | None = None,
        top_k: int | None = None,
        max_context_tokens: int | None = None,
    ) -> dict[str, Any]:
        client = self._client()
        body: dict[str, Any] = {
            "question": question.strip(),
            "topK": top_k or self._top_k,
            "maxContextTokens": max_context_tokens or self._max_context_tokens,
        }
        path, scope_fields = select_retrieve_route(
            organization_id=organization_id,
            project_id=project_id,
            person_id=person_id,
        )
        body.update(scope_fields)

        try:
            response = await client.post(
                f"{self._base_url}{path}",
                json=body,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise RagClientError(str(exc)) from exc

        if response.status_code == 503:
            raise RagClientError("RAG is disabled", response.status_code)
        if not response.is_success:
            message = f"RAG request failed ({response.status_code})"
            try:
                data = self._parse_json(response)
                if isinstance(data.get("detail"), str):
                    message = data["detail"]
            except RagClientError:
                pass
            raise RagClientError(message, response.status_code)

        data = self._parse_json(response)
        return data if isinstance(data, dict) else {"chunks": []}

    async def trigger_index_sync(self) -> dict[str, Any]:
        client = self._client()
        try:
            response = await client.post(
                f"{self._base_url}/index/sync",
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise RagClientError(str(exc)) from exc
        if not response.is_success:
            raise RagClientError(
                f"RAG index sync failed ({response.status_code})",
                response.status_code,
            )
        data = self._parse_json(response)
        return data if isinstance(data, dict) else {"queuedJobs": 0}
