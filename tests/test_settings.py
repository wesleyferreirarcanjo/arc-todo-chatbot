import pytest
import respx
from httpx import Response

from app.chatbot_settings import ChatbotSettingsClient, ChatbotSettingsError


@pytest.mark.asyncio
@respx.mock
async def test_get_runtime_settings_requires_access_token(monkeypatch):
    monkeypatch.setenv("ARC_TODO_API_BASE_URL", "http://api.test")
    monkeypatch.delenv("ARC_TODO_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("ARC_TODO_USERNAME", "admin")
    monkeypatch.setenv("ARC_TODO_PASSWORD", "secret")

    login = respx.post("http://api.test/auth/login").mock(
        return_value=Response(200, json={"access_token": "should-not-use"})
    )

    client = ChatbotSettingsClient()
    with pytest.raises(ChatbotSettingsError, match="ARC_TODO_ACCESS_TOKEN"):
        await client.get_runtime_settings(force_refresh=True)
    assert login.called is False


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
