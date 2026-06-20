import pytest
import respx
from httpx import Response

from app.arc_todo_client import ArcTodoClient


@pytest.mark.asyncio
@respx.mock
async def test_user_token_is_forwarded(monkeypatch):
    monkeypatch.setenv("ARC_TODO_API_BASE_URL", "http://api.test")

    route = respx.get("http://api.test/organizations").mock(
        return_value=Response(200, json=[{"id": "org-1", "name": "Acme"}])
    )

    client = ArcTodoClient(user_token="user-token")
    data = await client.request("GET", "/organizations")

    assert data[0]["id"] == "org-1"
    assert route.calls[0].request.headers["Authorization"] == "Bearer user-token"
