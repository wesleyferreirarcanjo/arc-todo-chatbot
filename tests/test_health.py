from __future__ import annotations

import respx
from fastapi.testclient import TestClient
from httpx import Response

from app.main import app


@respx.mock
def test_health_detail_reports_checks(monkeypatch):
    monkeypatch.setenv("ARC_TODO_API_BASE_URL", "http://api.test")
    monkeypatch.setenv("ARC_TODO_ACCESS_TOKEN", "token-abc")

    respx.get("http://api.test/chatbot-settings/runtime").mock(
        return_value=Response(
            200,
            json={
                "provider": "deepseek",
                "baseUrl": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "apiKey": "sk-test",
                "temperature": 0.2,
                "enabled": True,
                "maxHistoryMessages": 50,
                "maxHistoryTokens": 100000,
            },
        )
    )

    with TestClient(app) as client:
        response = client.get("/health?detail=true")

    assert response.status_code == 200
    payload = response.json()
    assert "checks" in payload
    assert payload["checks"]["chatbotSettings"]["status"] == "ok"
