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

    async def get_task(
        self,
        *,
        organization_id: str,
        project_id: str,
        task_id: str,
    ) -> Any:
        return await self._client.request(
            "GET",
            f"/organizations/{organization_id}/projects/{project_id}/tasks/{task_id}",
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

    async def delete_tasks(
        self,
        *,
        tasks: list[dict[str, str]],
    ) -> Any:
        deleted: list[str] = []
        failed: list[dict[str, str]] = []
        for task in tasks:
            task_id = task["task_id"]
            try:
                await self.delete_task(
                    organization_id=task["organization_id"],
                    project_id=task["project_id"],
                    task_id=task_id,
                )
                deleted.append(task_id)
            except Exception as exc:
                failed.append({"task_id": task_id, "error": str(exc)})
        return {"deleted": deleted, "failed": failed}

    async def update_tasks(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> Any:
        updated: list[str] = []
        failed: list[dict[str, str]] = []
        results: list[dict[str, Any]] = []
        update_fields = ("title", "description", "status", "criticity", "due_date")
        for task in tasks:
            task_id = task["task_id"]
            try:
                payload = {
                    key: task[key]
                    for key in update_fields
                    if key in task and task[key] is not None
                }
                result = await self.update_task(
                    organization_id=task["organization_id"],
                    project_id=task["project_id"],
                    task_id=task_id,
                    **payload,
                )
                updated.append(task_id)
                results.append({"task_id": task_id, "result": result})
            except Exception as exc:
                failed.append({"task_id": task_id, "error": str(exc)})
        return {"updated": updated, "results": results, "failed": failed}

    async def get_tasks(
        self,
        *,
        tasks: list[dict[str, str]],
    ) -> Any:
        fetched: list[str] = []
        failed: list[dict[str, str]] = []
        results: list[Any] = []
        for task in tasks:
            task_id = task["task_id"]
            try:
                result = await self.get_task(
                    organization_id=task["organization_id"],
                    project_id=task["project_id"],
                    task_id=task_id,
                )
                fetched.append(task_id)
                results.append(result)
            except Exception as exc:
                failed.append({"task_id": task_id, "error": str(exc)})
        return {"fetched": fetched, "tasks": results, "failed": failed}


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
        if tool_name == "update_tasks":
            return await tools.update_tasks(tasks=arguments["tasks"])
        if tool_name == "delete_task":
            return await tools.delete_task(**arguments)
        if tool_name == "delete_tasks":
            return await tools.delete_tasks(tasks=arguments["tasks"])
        if tool_name == "get_task":
            return await tools.get_task(**arguments)
        if tool_name == "get_tasks":
            return await tools.get_tasks(tasks=arguments["tasks"])
        raise ArcTodoApiError(f"Unknown tool: {tool_name}")
    except KeyError as exc:
        raise ArcTodoApiError(f"Missing required argument for {tool_name}: {exc}") from exc
