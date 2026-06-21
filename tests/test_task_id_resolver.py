import pytest
import respx
from httpx import Response

from app.graph.nodes import (
    _batch_task_scopes,
    _format_task_context_line,
    _ref_matches_task_identifier,
)
from app.task_id_resolver import is_friendly_task_id, normalize_friendly_task_id
from app.tools.todo_tools import TodoTools


def test_is_friendly_task_id():
    assert is_friendly_task_id("arc-1")
    assert is_friendly_task_id("#arc-1")
    assert not is_friendly_task_id("arc-todo")
    assert normalize_friendly_task_id("#arc-1") == "arc-1"


def test_ref_matches_task_identifier_by_display_id():
    ref = {
        "taskId": "22222222-2222-2222-2222-222222222222",
        "displayId": "#arc-1",
        "organizationId": "org-id",
        "projectId": "proj-id",
    }
    assert _ref_matches_task_identifier(ref, "arc-1")
    assert _ref_matches_task_identifier(ref, "#arc-1")


def test_batch_task_scopes_resolves_friendly_task_ids():
    ref = {
        "taskId": "22222222-2222-2222-2222-222222222222",
        "displayId": "#arc-1",
        "organizationId": "org-id",
        "projectId": "proj-id",
    }
    scopes = _batch_task_scopes(["arc-1"], [ref])
    assert scopes == [
        {
            "organization_id": "org-id",
            "project_id": "proj-id",
            "task_id": "22222222-2222-2222-2222-222222222222",
        }
    ]


def test_format_task_context_line_includes_display_id():
    line = _format_task_context_line(
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "displayId": "#arc-1",
            "title": "Task",
            "status": "todo",
            "criticity": "medium",
        },
        "Task",
    )
    assert "displayId: #arc-1" in line


@pytest.mark.asyncio
@respx.mock
async def test_todo_tools_get_task_resolves_friendly_id(monkeypatch):
    monkeypatch.setenv("ARC_TODO_API_BASE_URL", "http://api.test")
    monkeypatch.setenv("ARC_TODO_ACCESS_TOKEN", "token-abc")

    org_id = "57df4a79-d87d-40e1-9fb0-2da29d8ebecf"
    project_id = "d576e04d-f683-4b88-a374-0aab28a4be10"
    task_id = "22222222-2222-2222-2222-222222222222"
    respx.get("http://api.test/tasks/resolve").mock(
        return_value=Response(
            200,
            json={
                "id": task_id,
                "displayId": "#arc-1",
                "organizationId": org_id,
                "projectId": project_id,
                "title": "Friendly task",
            },
        )
    )
    get_route = respx.get(
        f"http://api.test/organizations/{org_id}/projects/{project_id}/tasks/{task_id}"
    ).mock(return_value=Response(200, json={"id": task_id, "displayId": "#arc-1"}))

    from app.arc_todo_client import ArcTodoClient

    client = ArcTodoClient(user_token="token")
    tools = TodoTools(client)
    result = await tools.get_task(
        organization_id=org_id,
        project_id=project_id,
        task_id="arc-1",
    )

    assert get_route.called
    assert result["displayId"] == "#arc-1"
