from __future__ import annotations

from typing import Any

from app.arc_todo_client import ArcTodoClient, ArcTodoApiError
from app.concurrency import map_limited
from app.config import settings
from app.task_id_resolver import is_friendly_task_id, is_uuid


class TodoTools:
    def __init__(self, client: ArcTodoClient) -> None:
        self._client = client
        self._batch_concurrency = settings.todo_tools_batch_concurrency

    async def list_organizations(self) -> Any:
        return await self._client.request("GET", "/organizations")

    async def list_projects(self, organization_id: str) -> Any:
        return await self._client.request(
            "GET",
            f"/organizations/{organization_id}/projects",
        )

    async def resolve_scope(
        self,
        *,
        organization_hint: str | None = None,
        project_hint: str | None = None,
        message: str | None = None,
    ) -> Any:
        params: dict[str, str] = {}
        if organization_hint:
            params["organizationHint"] = organization_hint
        if project_hint:
            params["projectHint"] = project_hint
        if message:
            params["message"] = message
        return await self._client.request("GET", "/scope/resolve", params=params or None)

    async def resolve_task(self, *, identifier: str) -> Any:
        return await self._client.request(
            "GET",
            "/tasks/resolve",
            params={"identifier": identifier},
        )

    async def _resolve_task_scope(
        self,
        *,
        organization_id: str,
        project_id: str,
        task_id: str,
    ) -> tuple[str, str, str]:
        if is_uuid(task_id):
            return organization_id, project_id, task_id

        resolved = await self.resolve_task(identifier=task_id)
        return (
            resolved["organizationId"],
            resolved["projectId"],
            resolved["id"],
        )

    async def _resolve_parent_task_id(self, parent_task_id: str | None) -> str | None:
        if not parent_task_id or is_uuid(parent_task_id):
            return parent_task_id
        resolved = await self.resolve_task(identifier=parent_task_id)
        return resolved["id"]

    async def list_tasks(
        self,
        *,
        organization_id: str | None = None,
        project_id: str | None = None,
        status: str | None = None,
        criticity: str | None = None,
        category: str | None = None,
        parent_task_id: str | None = None,
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
        if category:
            params["category"] = category
        if parent_task_id:
            params["parentTaskId"] = await self._resolve_parent_task_id(parent_task_id)
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
        parent_task_id: str | None = None,
        category: str = "other",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        body: dict[str, Any] = {
            "title": title,
            "status": status,
            "criticity": criticity,
            "category": category,
        }
        if description:
            body["description"] = description
        if due_date:
            body["dueDate"] = due_date
        if parent_task_id:
            body["parentTaskId"] = await self._resolve_parent_task_id(parent_task_id)
        if metadata is not None:
            body["metadata"] = metadata
        return await self._client.request(
            "POST",
            f"/organizations/{organization_id}/projects/{project_id}/tasks",
            json_body=body,
        )

    async def create_tasks(
        self,
        *,
        organization_id: str,
        project_id: str,
        tasks: list[dict[str, Any]],
    ) -> Any:
        created: list[Any] = []
        failed: list[dict[str, str]] = []

        async def create_one(task: dict[str, Any]) -> tuple[Any | None, dict[str, str] | None]:
            title = task.get("title")
            if not title:
                return None, {"title": "", "error": "Missing title"}
            try:
                result = await self.create_task(
                    organization_id=organization_id,
                    project_id=project_id,
                    title=title,
                    description=task.get("description"),
                    status=task.get("status") or "todo",
                    criticity=task.get("criticity") or "medium",
                    due_date=task.get("due_date"),
                    parent_task_id=task.get("parent_task_id") or task.get("parent_id"),
                    category=task.get("category") or "other",
                    metadata=task.get("metadata"),
                )
                return result, None
            except Exception as exc:
                return None, {"title": title, "error": str(exc)}

        results = await map_limited(tasks, create_one, limit=self._batch_concurrency)
        for outcome in results:
            if isinstance(outcome, BaseException):
                failed.append({"title": "", "error": str(outcome)})
                continue
            result, error = outcome
            if error:
                failed.append(error)
            elif result is not None:
                created.append(result)
        return {"created": created, "failed": failed}

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
        new_project_id: str | None = None,
        parent_task_id: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
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
        if new_project_id is not None:
            body["projectId"] = new_project_id
        if parent_task_id is not None:
            body["parentTaskId"] = await self._resolve_parent_task_id(parent_task_id)
        if category is not None:
            body["category"] = category
        if metadata is not None:
            body["metadata"] = metadata
        organization_id, project_id, task_id = await self._resolve_task_scope(
            organization_id=organization_id,
            project_id=project_id,
            task_id=task_id,
        )
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
        organization_id, project_id, task_id = await self._resolve_task_scope(
            organization_id=organization_id,
            project_id=project_id,
            task_id=task_id,
        )
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
        organization_id, project_id, task_id = await self._resolve_task_scope(
            organization_id=organization_id,
            project_id=project_id,
            task_id=task_id,
        )
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

        async def delete_one(task: dict[str, str]) -> tuple[str | None, dict[str, str] | None]:
            task_id = task["task_id"]
            try:
                await self.delete_task(
                    organization_id=task["organization_id"],
                    project_id=task["project_id"],
                    task_id=task_id,
                )
                return task_id, None
            except Exception as exc:
                return None, {"task_id": task_id, "error": str(exc)}

        results = await map_limited(tasks, delete_one, limit=self._batch_concurrency)
        for outcome in results:
            if isinstance(outcome, BaseException):
                failed.append({"task_id": "", "error": str(outcome)})
                continue
            task_id, error = outcome
            if error:
                failed.append(error)
            elif task_id:
                deleted.append(task_id)
        return {"deleted": deleted, "failed": failed}

    async def update_tasks(
        self,
        *,
        tasks: list[dict[str, Any]],
    ) -> Any:
        updated: list[str] = []
        failed: list[dict[str, str]] = []
        results: list[dict[str, Any]] = []
        update_fields = (
            "title",
            "description",
            "status",
            "criticity",
            "due_date",
            "parent_task_id",
            "category",
            "metadata",
        )

        async def update_one(
            task: dict[str, Any],
        ) -> tuple[str | None, dict[str, Any] | None, dict[str, str] | None]:
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
                return task_id, {"task_id": task_id, "result": result}, None
            except Exception as exc:
                return None, None, {"task_id": task_id, "error": str(exc)}

        outcomes = await map_limited(tasks, update_one, limit=self._batch_concurrency)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                failed.append({"task_id": "", "error": str(outcome)})
                continue
            task_id, result_item, error = outcome
            if error:
                failed.append(error)
            elif task_id and result_item:
                updated.append(task_id)
                results.append(result_item)
        return {"updated": updated, "results": results, "failed": failed}

    async def move_task(
        self,
        *,
        organization_id: str,
        project_id: str,
        task_id: str,
        target_project_id: str,
    ) -> Any:
        return await self.update_task(
            organization_id=organization_id,
            project_id=project_id,
            task_id=task_id,
            new_project_id=target_project_id,
        )

    async def move_tasks(
        self,
        *,
        tasks: list[dict[str, str]],
    ) -> Any:
        moved: list[Any] = []
        failed: list[dict[str, str]] = []
        for task in tasks:
            task_id = task["task_id"]
            target_project_id = task.get("target_project_id")
            if not target_project_id:
                failed.append({"task_id": task_id, "error": "Missing target project"})
                continue
            try:
                result = await self.move_task(
                    organization_id=task["organization_id"],
                    project_id=task["project_id"],
                    task_id=task_id,
                    target_project_id=target_project_id,
                )
                moved.append(result)
            except Exception as exc:
                failed.append({"task_id": task_id, "error": str(exc)})
        return {"moved": moved, "failed": failed}

    async def get_tasks(
        self,
        *,
        tasks: list[dict[str, str]],
    ) -> Any:
        fetched: list[str] = []
        failed: list[dict[str, str]] = []
        results: list[Any] = []

        async def get_one(task: dict[str, str]) -> tuple[str | None, Any | None, dict[str, str] | None]:
            task_id = task["task_id"]
            try:
                result = await self.get_task(
                    organization_id=task["organization_id"],
                    project_id=task["project_id"],
                    task_id=task_id,
                )
                return task_id, result, None
            except Exception as exc:
                return None, None, {"task_id": task_id, "error": str(exc)}

        outcomes = await map_limited(tasks, get_one, limit=self._batch_concurrency)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                failed.append({"task_id": "", "error": str(outcome)})
                continue
            task_id, result, error = outcome
            if error:
                failed.append(error)
            elif task_id and result is not None:
                fetched.append(task_id)
                results.append(result)
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
        if tool_name == "resolve_scope":
            return await tools.resolve_scope(
                organization_hint=arguments.get("organization_hint"),
                project_hint=arguments.get("project_hint"),
                message=arguments.get("message"),
            )
        if tool_name == "list_tasks":
            return await tools.list_tasks(
                organization_id=arguments.get("organization_id"),
                project_id=arguments.get("project_id"),
                status=arguments.get("status"),
                criticity=arguments.get("criticity"),
                category=arguments.get("category"),
                parent_task_id=arguments.get("parent_task_id")
                or arguments.get("parent_id"),
            )
        if tool_name == "create_task":
            return await tools.create_task(**arguments)
        if tool_name == "create_tasks":
            tasks = list(arguments.get("tasks") or [])
            parent_task_id = arguments.get("parent_task_id") or arguments.get("parent_id")
            if parent_task_id:
                tasks = [
                    {**task, "parent_task_id": task.get("parent_task_id") or parent_task_id}
                    if isinstance(task, dict)
                    else task
                    for task in tasks
                ]
            return await tools.create_tasks(
                organization_id=arguments["organization_id"],
                project_id=arguments["project_id"],
                tasks=tasks,
            )
        if tool_name == "update_task":
            return await tools.update_task(**arguments)
        if tool_name == "update_tasks":
            return await tools.update_tasks(tasks=arguments["tasks"])
        if tool_name == "move_task":
            return await tools.move_task(**arguments)
        if tool_name == "move_tasks":
            return await tools.move_tasks(tasks=arguments["tasks"])
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
