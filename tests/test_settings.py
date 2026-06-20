import pytest
import respx
from httpx import Response

from app.chatbot_settings import ChatbotSettingsClient


@pytest.mark.asyncio
@respx.mock
async def test_get_runtime_settings_with_access_token(monkeypatch):
    monkeypatch.setenv("ARC_TODO_API_BASE_URL", "http://api.test")
    monkeypatch.setenv("ARC_TODO_ACCESS_TOKEN", "token-abc")
    monkeypatch.delenv("ARC_TODO_USERNAME", raising=False)
    monkeypatch.delenv("ARC_TODO_PASSWORD", raising=False)

    route = respx.get("http://api.test/chatbot-settings/runtime").mock(
        return_value=Response(
            200,
            json={
                "provider": "deepseek",
                "baseUrl": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "apiKey": "sk-test",
                "temperature": 0.3,
                "enabled": True,
            },
        )
    )

    client = ChatbotSettingsClient()
    settings = await client.get_runtime_settings(force_refresh=True)

    assert settings.provider == "deepseek"
    assert settings.model == "deepseek-chat"
    assert settings.api_key == "sk-test"
    assert route.calls[0].request.headers["Authorization"] == "Bearer token-abc"
