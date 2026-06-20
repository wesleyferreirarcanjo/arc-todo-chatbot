from __future__ import annotations

from typing import Any

from app.arc_todo_client import ArcTodoClient, ArcTodoApiError


class TodoTools:
    def __init__(self, client: ArcTodoClient) -> None:
        self._client = client

    async def list_organizations(self) -> Any:
        return await self._client.request("GET", "/organizations")

    async def list_projects(self, organization_id: str) -> Any:
        return await self._client.request(
            "GET",
            f"/organizations/{organization_id}/projects",
        )

    async def list_tasks(
        self,
        *,
        organization_id: str | None = None,
        project_id: str | None = None,
        status: str | None = None,
        criticity: str | None = None,
    ) -> Any:
        params: dict[str, str] = {}
        if organization_id:
            params["organizationId"] = organization_id
        if project_id:
            params["projectId"] = project_id
        if status:
            params["status"] = status
        if criticity:
            params["criticity"] = criticity
        return await self._client.request("GET", "/tasks", params=params or None)

    async def create_task(
        self,
        *,
        organization_id: str,
        project_id: str,
        title: str,
        description: str | None = None,
        status: str = "todo",
        criticity: str = "medium",
        due_date: str | None = None,
    ) -> Any:
        body: dict[str, Any] = {
            "title": title,
            "status": status,
            "criticity": criticity,
        }
        if description:
            body["description"] = description
        if due_date:
            body["dueDate"] = due_date
        return await self._client.request(
            "POST",
            f"/organizations/{organization_id}/projects/{project_id}/tasks",
            json_body=body,
        )

    async def update_task(
        self,
        *,
        organization_id: str,
        project_id: str,
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        criticity: str | None = None,
        due_date: str | None = None,
    ) -> Any:
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if description is not None:
            body["description"] = description
        if status is not None:
            body["status"] = status
        if criticity is not None:
            body["criticity"] = criticity
        if due_date is not None:
            body["dueDate"] = due_date
        return await self._client.request(
            "PATCH",
            f"/organizations/{organization_id}/projects/{project_id}/tasks/{task_id}",
            json_body=body,
        )

    async def delete_task(
        self,
        *,
        organization_id: str,
        project_id: str,
        task_id: str,
    ) -> Any:
        return await self._client.request(
            "DELETE",
            f"/organizations/{organization_id}/projects/{project_id}/tasks/{task_id}",
        )


async def execute_todo_tool(
    tools: TodoTools,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    try:
        if tool_name == "list_organizations":
            return await tools.list_organizations()
        if tool_name == "list_projects":
            return await tools.list_projects(arguments["organization_id"])
        if tool_name == "list_tasks":
            return await tools.list_tasks(
                organization_id=arguments.get("organization_id"),
                project_id=arguments.get("project_id"),
                status=arguments.get("status"),
                criticity=arguments.get("criticity"),
            )
        if tool_name == "create_task":
            return await tools.create_task(**arguments)
        if tool_name == "update_task":
            return await tools.update_task(**arguments)
        if tool_name == "delete_task":
            return await tools.delete_task(**arguments)
        raise ArcTodoApiError(f"Unknown tool: {tool_name}")
    except KeyError as exc:
        raise ArcTodoApiError(f"Missing required argument for {tool_name}: {exc}") from exc
