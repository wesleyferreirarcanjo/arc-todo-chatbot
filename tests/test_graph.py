import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes import (
    BATCH_TOOL_RESOLVERS,
    context_agent,
    resolve_delete_tasks_arguments,
    resolve_get_tasks_arguments,
    resolve_scope_arguments,
    resolve_update_tasks_arguments,
    route_after_planner,
    todo_tools_agent,
    _is_uuid,
    _match_scope_name,
)
from app.tools.todo_tools import TodoTools, execute_todo_tool


@pytest.mark.asyncio
async def test_context_agent_extracts_latest_user_message():
    state = await context_agent(
        {
            "user_token": "token",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "list tasks"},
            ],
        }
    )
    assert state["latest_user_message"] == "list tasks"


def test_route_after_planner_tools():
    assert route_after_planner({"route": "tools", "tool_name": "list_tasks"}) == "todo_tools_agent"


def test_route_after_planner_direct():
    assert route_after_planner({"route": "direct"}) == "response_agent"


def test_resolve_delete_tasks_arguments_uses_all_refs_for_bulk_intent():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "One",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj2",
            "title": "Two",
        },
    ]

    result = resolve_delete_tasks_arguments(
        arguments={"task_ids": None},
        task_refs=task_refs,
        latest_user_message="please delete all selected",
    )

    assert result["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
        },
        {
            "organization_id": "org1",
            "project_id": "proj2",
            "task_id": "t2",
        },
    ]


def test_resolve_delete_tasks_arguments_filters_explicit_task_ids():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "One",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "Two",
        },
    ]

    result = resolve_delete_tasks_arguments(
        arguments={"task_ids": ["t2"]},
        task_refs=task_refs,
    )

    assert result["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t2",
        },
    ]


@pytest.mark.asyncio
async def test_todo_tools_agent_delete_tasks_deletes_every_ref():
    mock_tools = MagicMock()
    mock_tools.delete_tasks = AsyncMock(
        return_value={"deleted": ["t1", "t2"], "failed": []},
    )

    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
        return_value=mock_tools,
    ), patch(
        "app.graph.nodes.execute_todo_tool",
        new=AsyncMock(return_value={"deleted": ["t1", "t2"], "failed": []}),
    ) as execute_mock:
        state = await todo_tools_agent(
            {
                "user_token": "token",
                "tool_name": "delete_tasks",
                "tool_arguments": {"task_ids": None},
                "latest_user_message": "delete all of them",
                "task_refs": [
                    {
                        "taskId": "t1",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "One",
                    },
                    {
                        "taskId": "t2",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "Two",
                    },
                ],
                "used_tools": [],
            }
        )

    execute_mock.assert_awaited_once()
    assert execute_mock.await_args.args[1] == "delete_tasks"
    assert execute_mock.await_args.args[2]["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
        },
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t2",
        },
    ]
    assert state["tool_result"] == {"deleted": ["t1", "t2"], "failed": []}
    assert state["used_tools"] == ["delete_tasks"]


@pytest.mark.asyncio
async def test_delete_tasks_tool_calls_delete_task_for_each():
    client = MagicMock()
    client.request = AsyncMock(return_value={"ok": True})
    tools = TodoTools(client)

    result = await execute_todo_tool(
        tools,
        "delete_tasks",
        {
            "tasks": [
                {
                    "organization_id": "org1",
                    "project_id": "proj1",
                    "task_id": "t1",
                },
                {
                    "organization_id": "org1",
                    "project_id": "proj2",
                    "task_id": "t2",
                },
            ],
        },
    )

    assert result == {"deleted": ["t1", "t2"], "failed": []}
    assert client.request.await_count == 2
    client.request.assert_any_await(
        "DELETE",
        "/organizations/org1/projects/proj1/tasks/t1",
    )
    client.request.assert_any_await(
        "DELETE",
        "/organizations/org1/projects/proj2/tasks/t2",
    )


def test_resolve_update_tasks_arguments_applies_shared_fields():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "One",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "Two",
        },
    ]

    result = resolve_update_tasks_arguments(
        arguments={"task_ids": None, "status": "done"},
        task_refs=task_refs,
        latest_user_message="mark all of them as done",
    )

    assert result["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
            "status": "done",
        },
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t2",
            "status": "done",
        },
    ]


def test_resolve_get_tasks_arguments_uses_all_refs_for_bulk_intent():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "One",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj2",
            "title": "Two",
        },
    ]

    result = resolve_get_tasks_arguments(
        arguments={"task_ids": None},
        task_refs=task_refs,
        latest_user_message="show me all selected",
    )

    assert result["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
        },
        {
            "organization_id": "org1",
            "project_id": "proj2",
            "task_id": "t2",
        },
    ]


def test_batch_tool_resolvers_cover_selected_task_actions():
    assert set(BATCH_TOOL_RESOLVERS) == {"get_tasks", "update_tasks", "delete_tasks"}


@pytest.mark.asyncio
async def test_todo_tools_agent_upgrades_update_task_to_update_tasks():
    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
    ), patch(
        "app.graph.nodes.execute_todo_tool",
        new=AsyncMock(return_value={"updated": ["t1", "t2"], "results": [], "failed": []}),
    ) as execute_mock:
        await todo_tools_agent(
            {
                "user_token": "token",
                "tool_name": "update_task",
                "tool_arguments": {"status": "done"},
                "latest_user_message": "mark all selected as done",
                "task_refs": [
                    {
                        "taskId": "t1",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "One",
                    },
                    {
                        "taskId": "t2",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "Two",
                    },
                ],
                "used_tools": [],
            }
        )

    assert execute_mock.await_args.args[1] == "update_tasks"
    assert execute_mock.await_args.args[2]["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
            "status": "done",
        },
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t2",
            "status": "done",
        },
    ]


@pytest.mark.asyncio
async def test_update_tasks_tool_calls_update_task_for_each():
    client = MagicMock()
    client.request = AsyncMock(return_value={"id": "t1", "status": "done"})
    tools = TodoTools(client)

    result = await execute_todo_tool(
        tools,
        "update_tasks",
        {
            "tasks": [
                {
                    "organization_id": "org1",
                    "project_id": "proj1",
                    "task_id": "t1",
                    "status": "done",
                },
                {
                    "organization_id": "org1",
                    "project_id": "proj1",
                    "task_id": "t2",
                    "status": "done",
                },
            ],
        },
    )

    assert result["updated"] == ["t1", "t2"]
    assert result["failed"] == []
    assert client.request.await_count == 2
    client.request.assert_any_await(
        "PATCH",
        "/organizations/org1/projects/proj1/tasks/t1",
        json_body={"status": "done"},
    )


@pytest.mark.asyncio
async def test_create_tasks_tool_calls_create_task_for_each():
    client = MagicMock()
    client.request = AsyncMock(side_effect=[{"id": "t1"}, {"id": "t2"}])
    tools = TodoTools(client)

    result = await execute_todo_tool(
        tools,
        "create_tasks",
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "tasks": [
                {"title": "RAG system"},
                {"title": "Repository link connection"},
            ],
        },
    )

    assert len(result["created"]) == 2
    assert result["failed"] == []
    assert client.request.await_count == 2
    client.request.assert_any_await(
        "POST",
        "/organizations/org1/projects/proj1/tasks",
        json_body={"title": "RAG system", "status": "todo", "criticity": "medium"},
    )


@pytest.mark.asyncio
async def test_get_tasks_tool_calls_get_task_for_each():
    client = MagicMock()
    client.request = AsyncMock(side_effect=[{"id": "t1"}, {"id": "t2"}])
    tools = TodoTools(client)

    result = await execute_todo_tool(
        tools,
        "get_tasks",
        {
            "tasks": [
                {
                    "organization_id": "org1",
                    "project_id": "proj1",
                    "task_id": "t1",
                },
                {
                    "organization_id": "org1",
                    "project_id": "proj2",
                    "task_id": "t2",
                },
            ],
        },
    )

    assert result["fetched"] == ["t1", "t2"]
    assert result["tasks"] == [{"id": "t1"}, {"id": "t2"}]
    assert result["failed"] == []
    assert client.request.await_count == 2


def test_is_uuid():
    assert _is_uuid("4797da9c-f611-4bb8-b736-849a824c5fbc")
    assert not _is_uuid("arc-todo")


def test_match_scope_name_by_slug():
    orgs = [
        {"id": "4797da9c-f611-4bb8-b736-849a824c5fbc", "name": "Arc Todo", "slug": "arc-todo"},
    ]
    assert _match_scope_name(orgs, "arc-todo") == "4797da9c-f611-4bb8-b736-849a824c5fbc"


@pytest.mark.asyncio
async def test_resolve_scope_arguments_prefers_state_uuid_over_planner_slug():
    tools = MagicMock()
    tools.list_organizations = AsyncMock()
    tools.list_projects = AsyncMock()

    result = await resolve_scope_arguments(
        tools,
        tool_name="create_task",
        arguments={
            "organization_id": "arc-todo",
            "project_id": "backend",
            "title": "RAG system",
        },
        state={
            "organization_id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            "project_id": "8b2f0a44-1d63-4f7a-9c2e-111111111111",
        },
    )

    assert result["organization_id"] == "4797da9c-f611-4bb8-b736-849a824c5fbc"
    assert result["project_id"] == "8b2f0a44-1d63-4f7a-9c2e-111111111111"
    tools.list_organizations.assert_not_awaited()
    tools.list_projects.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_scope_arguments_resolves_organization_name():
    tools = MagicMock()
    tools.list_organizations = AsyncMock(
        return_value=[
            {
                "id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                "name": "Arc Todo",
                "slug": "arc-todo",
            }
        ]
    )
    tools.list_projects = AsyncMock(
        return_value=[{"id": "8b2f0a44-1d63-4f7a-9c2e-111111111111", "name": "Platform"}]
    )

    result = await resolve_scope_arguments(
        tools,
        tool_name="create_task",
        arguments={
            "organization_id": "arc-todo",
            "project_id": "Platform",
            "title": "RAG system",
        },
        state={},
    )

    assert result["organization_id"] == "4797da9c-f611-4bb8-b736-849a824c5fbc"
    assert result["project_id"] == "8b2f0a44-1d63-4f7a-9c2e-111111111111"


@pytest.mark.asyncio
async def test_todo_tools_agent_returns_error_instead_of_raising():
    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
    ), patch(
        "app.graph.nodes.resolve_scope_arguments",
        new=AsyncMock(side_effect=lambda tools, **kwargs: kwargs["arguments"]),
    ), patch(
        "app.graph.nodes.execute_todo_tool",
        new=AsyncMock(side_effect=Exception("Request failed (500)")),
    ):
        state = await todo_tools_agent(
            {
                "user_token": "token",
                "tool_name": "create_task",
                "tool_arguments": {
                    "organization_id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                    "project_id": "8b2f0a44-1d63-4f7a-9c2e-111111111111",
                    "title": "RAG system",
                },
                "used_tools": [],
            }
        )

    assert state["tool_result"] is None
    assert "Request failed (500)" in state["error"]
