import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes import (
    BATCH_TOOL_RESOLVERS,
    context_agent,
    _apply_task_ref_source_scope,
    _best_scope_match,
    _build_mutation_failure_response,
    _catalog_from_scope_result,
    _coerce_mutation_tool,
    _extract_all_scope_hints,
    _extract_move_target_hint,
    _looks_like_move_mutation,
    resolve_delete_tasks_arguments,
    resolve_get_tasks_arguments,
    resolve_move_tasks_arguments,
    resolve_scope_arguments,
    resolve_update_tasks_arguments,
    resolve_update_tasks_arguments_async,
    route_after_context,
    route_after_planner,
    route_after_scope_discovery,
    route_after_tools,
    scope_discovery_agent,
    todo_tools_agent,
    _is_uuid,
    _looks_like_create_mutation,
    _looks_like_task_mutation,
    _extract_proposed_descriptions_from_assistant,
    _filter_task_refs_by_message,
    _looks_like_mutation_confirmation,
    _looks_like_update_mutation,
    _resolve_confirmation_update_arguments,
    planner_agent,
    _match_scope_name,
    _mutation_succeeded,
    _normalize_planner_actions,
    _normalize_tool_arguments,
    _parse_create_task_titles,
    _validate_mutation_arguments,
    _build_verified_mutation_response,
    _action_succeeded,
    response_agent,
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
    assert set(BATCH_TOOL_RESOLVERS) == {"get_tasks", "delete_tasks"}


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


def test_resolve_update_tasks_arguments_uses_all_refs_for_description_update():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "make assistant smarter",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "repositories link connect",
        },
        {
            "taskId": "t3",
            "organizationId": "org1",
            "projectId": "proj2",
            "title": "rag system for the assistant",
        },
    ]

    result = resolve_update_tasks_arguments(
        arguments={"task_ids": None, "description": "Updated scope"},
        task_refs=task_refs,
        latest_user_message="add description for this task",
    )

    assert len(result["tasks"]) == 3
    assert all(task["description"] == "Updated scope" for task in result["tasks"])


def test_resolve_update_tasks_arguments_uses_per_task_payloads():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "make assistant smarter",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "repositories link connect",
        },
    ]

    result = resolve_update_tasks_arguments(
        arguments={
            "tasks": [
                {"task_id": "t1", "description": "History limit to 50 interactions."},
                {"task_id": "t2", "description": "Link repos to tasks."},
            ],
        },
        task_refs=task_refs,
    )

    assert result["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
            "description": "History limit to 50 interactions.",
        },
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t2",
            "description": "Link repos to tasks.",
        },
    ]


@pytest.mark.asyncio
async def test_resolve_update_tasks_arguments_async_generates_distinct_descriptions():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "make assistant smarter",
        },
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "repositories link connect",
        },
        {
            "taskId": "t3",
            "organizationId": "org1",
            "projectId": "proj2",
            "title": "rag system for the assistant",
        },
    ]
    runtime = MagicMock()

    with patch(
        "app.graph.nodes._generate_per_task_descriptions",
        new=AsyncMock(
            return_value={
                "t1": "Increase conversation history to 50 interactions or 100k tokens.",
                "t2": "Connect repository links to tasks.",
                "t3": "Add a RAG system for the assistant.",
            }
        ),
    ):
        result = await resolve_update_tasks_arguments_async(
            runtime,
            arguments={"task_ids": None, "description": "placeholder"},
            task_refs=task_refs,
            latest_user_message="create a description for this tasks",
        )

    descriptions = [task["description"] for task in result["tasks"]]
    assert len(descriptions) == 3
    assert len(set(descriptions)) == 3
    assert descriptions[0].startswith("Increase conversation history")
    assert descriptions[1].startswith("Connect repository links")
    assert descriptions[2].startswith("Add a RAG system")


def test_fix_description_matches_update_mutation():
    message = "fix the description of: repositories link connect rag system for the assiss"
    assert _looks_like_update_mutation(message)


def test_filter_task_refs_by_message_keeps_mentioned_subset():
    task_refs = [
        {"taskId": "t1", "title": "make assistant smarter"},
        {"taskId": "t2", "title": "repositories link connect"},
        {"taskId": "t3", "title": "rag system for the assistant"},
    ]
    filtered = _filter_task_refs_by_message(
        task_refs,
        "fix the description of: repositories link connect rag system for the assiss",
    )
    assert [ref["taskId"] for ref in filtered] == ["t2", "t3"]


def test_extract_proposed_descriptions_from_assistant():
    text = (
        "1. **repositories link connection to task** — "
        '*"Set up repository linking for tasks."*\n'
        "2. **rag system for the assisstant** — "
        '*"Implement a RAG system for the assistant."*\n\n'
        "Would you like me to apply these descriptions?"
    )
    proposals = _extract_proposed_descriptions_from_assistant(text)
    assert len(proposals) == 2
    assert "repository linking" in proposals[0][1]
    assert "RAG system" in proposals[1][1]


def test_resolve_confirmation_update_arguments():
    messages = [
        {"role": "user", "content": "please fix descriptions"},
        {
            "role": "assistant",
            "content": (
                "1. **repositories link connection to task** — "
                '*"Set up repository linking for tasks."*\n'
                "2. **rag system for the assisstant** — "
                '*"Implement a RAG system for the assistant."*\n\n'
                "Would you like me to apply these descriptions?"
            ),
        },
        {"role": "user", "content": "yes"},
    ]
    task_refs = [
        {
            "taskId": "t2",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "repositories link connect",
        },
        {
            "taskId": "t3",
            "organizationId": "org1",
            "projectId": "proj2",
            "title": "rag system for the assistant",
        },
    ]
    result = _resolve_confirmation_update_arguments(messages, task_refs)
    assert result is not None
    assert len(result["tasks"]) == 2
    assert result["tasks"][0]["task_id"] == "t2"
    assert "repository linking" in result["tasks"][0]["description"]


@pytest.mark.asyncio
async def test_planner_agent_confirmation_routes_to_update_tasks():
    runtime = MagicMock()
    with patch("app.graph.nodes.build_model"):
        state = await planner_agent(
            {
                "messages": [
                    {"role": "user", "content": "fix descriptions"},
                    {
                        "role": "assistant",
                        "content": (
                            "1. **repositories link connection to task** — "
                            '*"Set up repository linking for tasks."*\n'
                            "2. **rag system for the assisstant** — "
                            '*"Implement a RAG system for the assistant."*\n\n'
                            "Would you like me to apply these descriptions?"
                        ),
                    },
                    {"role": "user", "content": "yes"},
                ],
                "latest_user_message": "yes",
                "task_refs": [
                    {
                        "taskId": "t2",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "repositories link connect",
                    },
                    {
                        "taskId": "t3",
                        "organizationId": "org1",
                        "projectId": "proj2",
                        "title": "rag system for the assistant",
                    },
                ],
            },
            runtime,
        )

    assert state["route"] == "tools"
    assert state["tool_name"] == "update_tasks"
    assert len(state["tool_arguments"]["tasks"]) == 2
    assert _looks_like_mutation_confirmation("yes")


def test_create_mutation_routes_through_scope_discovery():
    assert route_after_context({"latest_user_message": "create a task in arc-todo project"}) == "scope_discovery_agent"
    assert route_after_context(
        {"latest_user_message": "add description for this task"}
    ) == "planner_agent"
    assert route_after_context(
        {
            "latest_user_message": (
                "make assistant smater and... repositories link connect... "
                "rag system for the assiss... create a description for this tasks"
            )
        }
    ) == "planner_agent"
    assert not _looks_like_create_mutation("add description for this task")
    assert _looks_like_update_mutation("add description for this task")
    assert _looks_like_update_mutation(
        "make assistant smater create a description for this tasks"
    )
    assert _parse_create_task_titles(
        "make assistant smater create a description for this tasks"
    ) == []
    tool_name, coerced = _coerce_mutation_tool(
        {
            "latest_user_message": (
                "make assistant smater create a description for this tasks"
            ),
            "task_refs": [{"taskId": "t1", "organizationId": "org1", "projectId": "p1", "title": "x"}],
        },
        "update_tasks",
        {"task_ids": None, "description": "smarter assistant"},
    )
    assert tool_name == "update_tasks"
    assert coerced["description"] == "smarter assistant"
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


def test_apply_task_ref_source_scope_overrides_wrong_project():
    arguments = _apply_task_ref_source_scope(
        {
            "task_id": "869f737f-412f-4ffb-b70c-c528c067e630",
            "organization_id": "org-target",
            "project_id": "proj-target",
        },
        [
            {
                "taskId": "869f737f-412f-4ffb-b70c-c528c067e630",
                "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                "projectId": "11111111-1111-4111-8111-111111111111",
                "title": "bug verify",
            }
        ],
    )
    assert arguments["organization_id"] == "4797da9c-f611-4bb8-b736-849a824c5fbc"
    assert arguments["project_id"] == "11111111-1111-4111-8111-111111111111"


def test_extract_move_target_hint():
    assert _extract_move_target_hint("arc-todo bug verify move to arc-todo") == "arc-todo"
    assert _looks_like_move_mutation("move to arc-todo")


@pytest.mark.asyncio
async def test_resolve_move_tasks_arguments_uses_task_ref_and_target_project():
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
                "id": "22222222-2222-4222-8222-222222222222",
                "name": "arc-todo",
                "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            },
        }
    )

    result = await resolve_move_tasks_arguments(
        tools,
        arguments={"task_ids": None},
        task_refs=[
            {
                "taskId": "869f737f-412f-4ffb-b70c-c528c067e630",
                "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                "projectId": "11111111-1111-4111-8111-111111111111",
                "title": "bug verify",
            }
        ],
        latest_user_message="arc-todo bug verify move to arc-todo",
    )

    assert result["tasks"] == [
        {
            "organization_id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            "project_id": "11111111-1111-4111-8111-111111111111",
            "task_id": "869f737f-412f-4ffb-b70c-c528c067e630",
            "target_project_id": "22222222-2222-4222-8222-222222222222",
        }
    ]


@pytest.mark.asyncio
async def test_todo_tools_agent_update_description_does_not_call_create_tasks():
    mock_tools = MagicMock()
    mock_tools.update_tasks = AsyncMock(
        return_value={"updated": ["t1", "t2", "t3"], "results": [], "failed": []},
    )

    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
        return_value=mock_tools,
    ), patch(
        "app.graph.nodes.execute_todo_tool",
        new=AsyncMock(return_value={"updated": ["t1", "t2", "t3"], "results": [], "failed": []}),
    ) as execute_mock, patch(
        "app.graph.nodes._generate_per_task_descriptions",
        new=AsyncMock(
            return_value={
                "t1": "Increase conversation history to 50 interactions or 100k tokens.",
                "t2": "Connect repository links to tasks.",
                "t3": "Add a RAG system for the assistant.",
            }
        ),
    ):
        state = await todo_tools_agent(
            {
                "user_token": "token",
                "tool_name": "create_tasks",
                "tool_arguments": {
                    "task_ids": None,
                    "description": "Updated descriptions",
                },
                "latest_user_message": (
                    "make assistant smater and... repositories link connect... "
                    "rag system for the assiss... create a description for this tasks"
                ),
                "task_refs": [
                    {
                        "taskId": "t1",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "make assistant smarter",
                    },
                    {
                        "taskId": "t2",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "repositories link connect",
                    },
                    {
                        "taskId": "t3",
                        "organizationId": "org1",
                        "projectId": "proj2",
                        "title": "rag system for the assistant",
                    },
                ],
                "used_tools": [],
            },
            runtime=MagicMock(),
        )

    execute_mock.assert_awaited_once()
    assert execute_mock.await_args.args[1] == "update_tasks"
    tasks = execute_mock.await_args.args[2]["tasks"]
    assert len(tasks) == 3
    descriptions = [task["description"] for task in tasks]
    assert len(set(descriptions)) == 3
    assert state["error"] is None


@pytest.mark.asyncio
async def test_todo_tools_agent_move_task_uses_source_scope_from_task_ref():
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
                "id": "22222222-2222-4222-8222-222222222222",
                "name": "arc-todo",
                "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            },
        }
    )

    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
        return_value=mock_tools,
    ), patch(
        "app.graph.nodes.execute_todo_tool",
        new=AsyncMock(return_value={"id": "869f737f-412f-4ffb-b70c-c528c067e630"}),
    ) as execute_mock:
        state = await todo_tools_agent(
            {
                "user_token": "token",
                "tool_name": "update_task",
                "tool_arguments": {
                    "task_id": "869f737f-412f-4ffb-b70c-c528c067e630",
                    "organization_id": "22222222-2222-4222-8222-222222222222",
                    "project_id": "22222222-2222-4222-8222-222222222222",
                },
                "latest_user_message": "arc-todo bug verify move to arc-todo",
                "task_refs": [
                    {
                        "taskId": "869f737f-412f-4ffb-b70c-c528c067e630",
                        "organizationId": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                        "projectId": "11111111-1111-4111-8111-111111111111",
                        "title": "bug verify",
                    }
                ],
                "used_tools": [],
            }
        )

    execute_mock.assert_awaited_once()
    assert execute_mock.await_args.args[1] == "move_tasks"
    assert execute_mock.await_args.args[2]["tasks"] == [
        {
            "organization_id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
            "project_id": "11111111-1111-4111-8111-111111111111",
            "task_id": "869f737f-412f-4ffb-b70c-c528c067e630",
            "target_project_id": "22222222-2222-4222-8222-222222222222",
        }
    ]
    assert state["error"] is None


def test_normalize_planner_actions_from_actions_array():
    actions = _normalize_planner_actions(
        {
            "route": "tools",
            "actions": [
                {
                    "intent": "create task one",
                    "tool_name": "create_task",
                    "tool_arguments": {"title": "One"},
                },
                {
                    "intent": "create task two",
                    "tool_name": "create_task",
                    "tool_arguments": {"title": "Two"},
                },
            ],
        }
    )
    assert len(actions) == 2
    assert actions[0]["tool_name"] == "create_task"
    assert actions[1]["tool_arguments"]["title"] == "Two"


def test_normalize_planner_actions_legacy_single_tool():
    actions = _normalize_planner_actions(
        {
            "route": "tools",
            "tool_name": "list_tasks",
            "tool_arguments": {},
        }
    )
    assert actions == [{"intent": "list_tasks", "tool_name": "list_tasks", "tool_arguments": {}}]


def test_normalize_tool_arguments_maps_priority_to_criticity():
    args = _normalize_tool_arguments({"priority": "high", "tasks": [{"priority": "low"}]})
    assert args["criticity"] == "high"
    assert args["tasks"][0]["criticity"] == "low"
    assert "priority" not in args


def test_validate_mutation_arguments_rejects_empty_update():
    error = _validate_mutation_arguments("update_tasks", {"tasks": []})
    assert error == "No fields to update"


def test_action_succeeded_detects_partial_batch_update():
    success, partial = _action_succeeded(
        "update_tasks",
        {"updated": ["t1"], "failed": [{"task_id": "t2", "error": "nope"}]},
        None,
    )
    assert success is True
    assert partial is True


def test_build_verified_mutation_response_reports_partial_success():
    message = _build_verified_mutation_response(
        {
            "route": "tools",
            "tool_results": [
                {
                    "intent": "create first task",
                    "tool_name": "create_task",
                    "tool_result": {"id": "t1", "title": "First"},
                    "error": None,
                    "success": True,
                    "partial": False,
                },
                {
                    "intent": "create second task",
                    "tool_name": "create_task",
                    "tool_result": None,
                    "error": "Missing task title",
                    "success": False,
                    "partial": False,
                },
            ],
        }
    )
    assert message is not None
    assert "1 of 2" in message
    assert "First" in message
    assert "failed" in message


@pytest.mark.asyncio
async def test_todo_tools_agent_executes_multiple_planned_actions():
    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
    ), patch(
        "app.graph.nodes._execute_single_tool",
        new=AsyncMock(
            side_effect=[
                ("create_task", {"title": "One"}, {"id": "t1", "title": "One"}, None),
                ("update_tasks", {"tasks": []}, {"updated": ["t2"], "failed": []}, None),
            ]
        ),
    ) as execute_mock:
        state = await todo_tools_agent(
            {
                "user_token": "token",
                "actions": [
                    {
                        "intent": "create task",
                        "tool_name": "create_task",
                        "tool_arguments": {"title": "One"},
                    },
                    {
                        "intent": "mark selected done",
                        "tool_name": "update_tasks",
                        "tool_arguments": {"status": "done", "task_ids": ["t2"]},
                    },
                ],
                "task_refs": [
                    {
                        "taskId": "t2",
                        "organizationId": "org1",
                        "projectId": "proj1",
                        "title": "Two",
                    }
                ],
                "latest_user_message": "create a task One and mark the selected task done",
                "used_tools": [],
            }
        )

    assert execute_mock.await_count == 2
    assert len(state["tool_results"]) == 2
    assert state["tool_results"][0]["success"] is True
    assert state["tool_results"][1]["success"] is True
    assert state["used_tools"] == ["create_task", "update_tasks"]


@pytest.mark.asyncio
async def test_todo_tools_agent_generates_create_description_when_requested():
    with patch("app.graph.nodes.ArcTodoClient"), patch(
        "app.graph.nodes.TodoTools",
    ), patch(
        "app.graph.nodes.resolve_scope_arguments",
        new=AsyncMock(side_effect=lambda tools, **kwargs: kwargs["arguments"]),
    ), patch(
        "app.graph.nodes._maybe_generate_create_descriptions",
        new=AsyncMock(
            side_effect=lambda runtime, **kwargs: {
                **kwargs["arguments"],
                "description": "Creative description for the task.",
            }
        ),
    ), patch(
        "app.graph.nodes.execute_todo_tool",
        new=AsyncMock(return_value={"id": "t1", "title": "Idea system"}),
    ) as execute_mock:
        state = await todo_tools_agent(
            {
                "user_token": "token",
                "actions": [
                    {
                        "intent": "create task with description",
                        "tool_name": "create_task",
                        "tool_arguments": {
                            "organization_id": "4797da9c-f611-4bb8-b736-849a824c5fbc",
                            "project_id": "11111111-1111-4111-8111-111111111111",
                            "title": "Idea system",
                        },
                    }
                ],
                "latest_user_message": (
                    "create a task Idea system and put a description as well for me"
                ),
                "used_tools": [],
            },
            runtime=MagicMock(),
        )

    assert state["tool_results"][0]["success"] is True
    assert execute_mock.await_args.args[2]["description"] == "Creative description for the task."


@pytest.mark.asyncio
async def test_response_agent_uses_verified_mutation_response():
    runtime = MagicMock()
    with patch("app.graph.nodes.build_model") as build_model:
        state = await response_agent(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "create a task with a description",
                    }
                ],
                "latest_user_message": "create a task with a description",
                "route": "tools",
                "tool_results": [
                    {
                        "intent": "create task",
                        "tool_name": "create_task",
                        "tool_result": {"id": "t1", "title": "Idea system"},
                        "error": None,
                        "success": True,
                        "partial": False,
                    }
                ],
            },
            runtime,
        )

    build_model.assert_not_called()
    assert "Idea system" in state["response"]
    assert "t1" in state["response"]


@pytest.mark.asyncio
async def test_resolve_update_tasks_arguments_async_generates_single_task_description():
    task_refs = [
        {
            "taskId": "t1",
            "organizationId": "org1",
            "projectId": "proj1",
            "title": "copy and planning verification",
        }
    ]
    runtime = MagicMock()

    with patch(
        "app.graph.nodes._generate_task_description_from_title",
        new=AsyncMock(return_value="Verify copy and planning flows in the task system."),
    ):
        result = await resolve_update_tasks_arguments_async(
            runtime,
            arguments={"task_ids": None},
            task_refs=task_refs,
            latest_user_message="add a description please",
        )

    assert result["tasks"] == [
        {
            "organization_id": "org1",
            "project_id": "proj1",
            "task_id": "t1",
            "description": "Verify copy and planning flows in the task system.",
        }
    ]
