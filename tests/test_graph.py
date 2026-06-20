import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes import (
    BATCH_TOOL_RESOLVERS,
    context_agent,
    _best_scope_match,
    _build_mutation_failure_response,
    _catalog_from_scope_result,
    _coerce_mutation_tool,
    _extract_all_scope_hints,
    resolve_delete_tasks_arguments,
    resolve_get_tasks_arguments,
    resolve_scope_arguments,
    resolve_update_tasks_arguments,
    route_after_context,
    route_after_planner,
    route_after_scope_discovery,
    route_after_tools,
    scope_discovery_agent,
    todo_tools_agent,
    _is_uuid,
    _looks_like_create_mutation,
    _looks_like_task_mutation,
    _match_scope_name,
    _mutation_succeeded,
    _parse_create_task_titles,
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
    tools.resolve_scope = AsyncMock(
        return_value={
            "status": "resolved",
            "organization": {
                "id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                "name": "Arc Todo",
                "slug": "arc-todo",
            },
            "project": {
                "id": "8b2f0a44-1d63-4f7a-9c2e-111111111111",
                "name": "Platform",
                "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            },
        }
    )

    result = await resolve_scope_arguments(
        tools,
        tool_name="create_task",
        arguments={
            "organization_id": "arc-todo",
            "project_id": "Platform",
            "title": "RAG system",
        },
        state={"latest_user_message": "create a task in arc-todo project"},
    )

    assert result["organization_id"] == "4797da9c-f611-4bb8-b736-849a824c5fbc"
    assert result["project_id"] == "8b2f0a44-1d63-4f7a-9c2e-111111111111"
    tools.resolve_scope.assert_awaited_once()


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


def test_looks_like_task_mutation():
    assert _looks_like_task_mutation("create a task to arc-todo")
    assert not _looks_like_task_mutation("hello there")


def test_mutation_succeeded_requires_create_tasks_result():
    assert _mutation_succeeded(
        {
            "route": "tools",
            "tool_name": "create_tasks",
            "tool_result": {"created": [{"id": "t1"}], "failed": []},
        }
    )
    assert not _mutation_succeeded(
        {
            "route": "tools",
            "tool_name": "create_tasks",
            "tool_result": {"created": [], "failed": [{"title": "x", "error": "nope"}]},
        }
    )
    assert not _mutation_succeeded({"route": "direct", "tool_name": "create_tasks"})


def test_build_mutation_failure_response_when_no_tool_ran():
    message = _build_mutation_failure_response({"route": "direct"})
    assert "no todo action ran" in message.lower()


def test_parse_create_task_titles_from_user_message():
    message = (
        "create a task to arc-todo create the rag system for the assisstant\n\n"
        "another task create the repositories link connection to task.\n\n"
        "in my system"
    )
    titles = _parse_create_task_titles(message)
    assert titles == [
        "rag system for the assisstant",
        "repositories link connection to task",
    ]


def test_coerce_mutation_tool_overrides_list_organizations():
    tool_name, arguments = _coerce_mutation_tool(
        {
            "latest_user_message": (
                "create a task to arc-todo create the rag system for the assisstant\n\n"
                "another task create the repositories link connection to task.\n\n"
                "in my system"
            ),
        },
        "list_organizations",
        {},
    )
    assert tool_name == "create_tasks"
    assert arguments["organization_id"] == "arc-todo"
    assert arguments["project_id"] == "my system"
    assert len(arguments["tasks"]) == 2


def test_extract_all_scope_hints_in_arc_todo_project():
    org_hint, project_hints = _extract_all_scope_hints(
        "create a task in arc-todo project create the rag system\n\nin my system"
    )
    assert org_hint == "arc-todo"
    assert project_hints == ["my system"]


def test_best_scope_match_fuzzy():
    projects = [
        {"id": "11111111-1111-4111-8111-111111111111", "name": "My System"},
        {"id": "22222222-2222-4222-8222-222222222222", "name": "Personal"},
    ]
    assert (
        _best_scope_match(projects, "my system")
        == "11111111-1111-4111-8111-111111111111"
    )


def test_create_mutation_routes_through_scope_discovery():
    assert route_after_context({"latest_user_message": "create a task in arc-todo project"}) == "scope_discovery_agent"
    assert route_after_scope_discovery(
        {
            "latest_user_message": "create a task in arc-todo project",
            "scope_status": "resolved",
            "organization_id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            "project_id": "8b2f0a44-1d63-4f7a-9c2e-111111111111",
        }
    ) == "todo_tools_agent"
    assert route_after_scope_discovery(
        {
            "latest_user_message": "create a task in arc-todo project",
            "scope_status": "ambiguous",
        }
    ) == "response_agent"
    assert _looks_like_create_mutation("create a task in arc-todo project")


@pytest.mark.asyncio
async def test_scope_discovery_agent_uses_api_resolver():
    mock_tools = MagicMock()
    mock_tools.resolve_scope = AsyncMock(
        return_value={
            "status": "resolved",
            "organization": {
                "id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                "name": "Arc Todo",
                "slug": "arc-todo",
            },
            "project": {
                "id": "11111111-1111-4111-8111-111111111111",
                "name": "My System",
                "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            },
        }
    )

    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
        return_value=mock_tools,
    ):
        state = await scope_discovery_agent(
            {
                "user_token": "token",
                "latest_user_message": (
                    "create a task in arc-todo project create the rag system\n\nin my system"
                ),
                "used_tools": [],
            }
        )

    assert state["scope_status"] == "resolved"
    assert state["organization_id"] == "4797da9c-f611-4bb8-b736-849a824c5fbc"
    assert state["project_id"] == "11111111-1111-4111-8111-111111111111"
    assert state["tool_name"] == "create_tasks"
    mock_tools.resolve_scope.assert_awaited_once()


def test_build_mutation_failure_response_for_ambiguous_scope():
    message = _build_mutation_failure_response(
        {
            "scope_status": "ambiguous",
            "scope_catalog": {
                "candidates": [
                    {
                        "organization": {"name": "Arc Todo"},
                        "project": {"name": "My System"},
                    },
                    {
                        "organization": {"name": "Personal Org"},
                        "project": {"name": "My System"},
                    },
                ]
            },
        }
    )
    assert "multiple matching projects" in message.lower()
    assert "My System in Arc Todo" in message


def test_catalog_from_scope_result_includes_candidates():
    catalog = _catalog_from_scope_result(
        {
            "status": "ambiguous",
            "candidates": [
                {
                    "organization": {"id": "org1", "name": "Arc Todo", "slug": "arc-todo"},
                    "project": {"id": "proj1", "name": "My System", "organizationId": "org1"},
                }
            ],
        }
    )
    assert catalog["status"] == "ambiguous"
    assert len(catalog["candidates"]) == 1
    assert catalog["organizations"][0]["slug"] == "arc-todo"
