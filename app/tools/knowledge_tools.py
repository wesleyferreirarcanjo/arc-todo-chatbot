from __future__ import annotations

from typing import Any

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient
from app.rag_client import RagClient, RagClientError


def _knowledge_collection_path(
    *,
    scope: str,
    organization_id: str | None = None,
    project_id: str | None = None,
    person_id: str | None = None,
) -> str:
    if scope == "general":
        return "/knowledge"
    if scope == "organization":
        if not organization_id:
            raise ArcTodoApiError("organization_id is required for organization scope")
        return f"/organizations/{organization_id}/knowledge"
    if scope == "project":
        if not organization_id or not project_id:
            raise ArcTodoApiError("organization_id and project_id are required for project scope")
        return f"/organizations/{organization_id}/projects/{project_id}/knowledge"
    if scope == "person":
        if organization_id and person_id:
            return f"/organizations/{organization_id}/persons/{person_id}/knowledge"
        if person_id:
            return f"/persons/{person_id}/knowledge"
        raise ArcTodoApiError("person_id is required for person scope")
    raise ArcTodoApiError(f"Unsupported scope: {scope}")


def _scope_path_args(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": arguments.get("scope", "general"),
        "organization_id": arguments.get("organization_id"),
        "project_id": arguments.get("project_id"),
        "person_id": arguments.get("person_id"),
    }


def _knowledge_entry_path(arguments: dict[str, Any]) -> str:
    return f"{_knowledge_collection_path(**_scope_path_args(arguments))}/{arguments['knowledge_id']}"


KNOWLEDGE_TOOLS = {
    "list_knowledge",
    "get_knowledge",
    "create_knowledge",
    "update_knowledge",
    "list_persons",
    "get_person",
    "trigger_rag_index_sync",
}


class KnowledgeTools:
    def __init__(
        self,
        client: ArcTodoClient,
        *,
        rag_client: RagClient | None = None,
    ) -> None:
        self._client = client
        self._rag_client = rag_client

    async def list_knowledge(
        self,
        *,
        scope: str = "general",
        organization_id: str | None = None,
        project_id: str | None = None,
        person_id: str | None = None,
        file_name: str | None = None,
        mime_type: str | None = None,
        has_attachments: bool | None = None,
    ) -> Any:
        if scope == "general" and not any(
            [organization_id, project_id, person_id, file_name, mime_type, has_attachments]
        ):
            return await self._client.request("GET", "/knowledge")

        if scope == "general" and any(
            [scope, organization_id, project_id, person_id]
        ):
            params = {
                k: v
                for k, v in {
                    "scope": scope,
                    "organizationId": organization_id,
                    "projectId": project_id,
                    "personId": person_id,
                    "fileName": file_name,
                    "mimeType": mime_type,
                    "hasAttachments": "true" if has_attachments else None,
                }.items()
                if v is not None
            }
            return await self._client.request("GET", "/knowledge", params=params)

        return await self._client.request(
            "GET",
            _knowledge_collection_path(
                scope=scope,
                organization_id=organization_id,
                project_id=project_id,
                person_id=person_id,
            ),
        )

    async def get_knowledge(self, arguments: dict[str, Any]) -> Any:
        return await self._client.request("GET", _knowledge_entry_path(arguments))

    async def create_knowledge(self, arguments: dict[str, Any]) -> Any:
        return await self._client.request(
            "POST",
            _knowledge_collection_path(**_scope_path_args(arguments)),
            json_body={
                "title": arguments["title"],
                "content": arguments["content"],
            },
        )

    async def update_knowledge(self, arguments: dict[str, Any]) -> Any:
        body = {
            key: arguments[key]
            for key in ("title", "content")
            if arguments.get(key) is not None
        }
        return await self._client.request(
            "PATCH",
            _knowledge_entry_path(arguments),
            json_body=body,
        )

    async def list_persons(self, organization_id: str | None = None) -> Any:
        path = (
            f"/organizations/{organization_id}/persons"
            if organization_id
            else "/persons"
        )
        return await self._client.request("GET", path)

    async def get_person(
        self,
        person_id: str,
        organization_id: str | None = None,
    ) -> Any:
        path = (
            f"/organizations/{organization_id}/persons/{person_id}"
            if organization_id
            else f"/persons/{person_id}"
        )
        return await self._client.request("GET", path)

    async def trigger_rag_index_sync(self) -> Any:
        if not self._rag_client:
            raise ArcTodoApiError("RAG client is not configured")
        return await self._rag_client.trigger_index_sync()


async def execute_knowledge_tool(
    tools: KnowledgeTools,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    try:
        if tool_name == "list_knowledge":
            return await tools.list_knowledge(**arguments)
        if tool_name == "get_knowledge":
            return await tools.get_knowledge(arguments)
        if tool_name == "create_knowledge":
            return await tools.create_knowledge(arguments)
        if tool_name == "update_knowledge":
            return await tools.update_knowledge(arguments)
        if tool_name == "list_persons":
            return await tools.list_persons(arguments.get("organization_id"))
        if tool_name == "get_person":
            return await tools.get_person(
                arguments["person_id"],
                organization_id=arguments.get("organization_id"),
            )
        if tool_name == "trigger_rag_index_sync":
            return await tools.trigger_rag_index_sync()
        raise ArcTodoApiError(f"Unknown knowledge tool: {tool_name}")
    except KeyError as exc:
        raise ArcTodoApiError(f"Missing required argument for {tool_name}: {exc}") from exc
    except RagClientError as exc:
        raise ArcTodoApiError(str(exc)) from exc
