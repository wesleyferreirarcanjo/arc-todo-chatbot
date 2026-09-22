from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.chatbot_settings import ChatbotRuntimeSettings
from app.errors import disabled_error
from app.main import app, load_runtime_settings, workflow_http_exception
from app.naming import build_naming_prompt, parse_suggestions

RUNTIME = ChatbotRuntimeSettings(
    provider="deepseek",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    api_key="sk-test",
    temperature=0.2,
    enabled=True,
)

ORG = "11111111-1111-1111-1111-111111111111"
PROJ = "22222222-2222-2222-2222-222222222222"
SESS = "33333333-3333-3333-3333-333333333333"

BODY = {
    "organizationId": ORG,
    "projectId": PROJ,
    "sessionId": SESS,
    "title": "fieldlot",
    "namingGoal": "company",
    "productDescription": {
        "whatItIs": "A coffee subscription for small offices.",
        "audience": "Office managers",
        "languages": "Portuguese and English",
        "personality": "Warm and direct",
    },
    "count": 10,
    "avoid": ["Nova", "Rift"],
}


@dataclass
class FakeResult:
    content: str


class FakeModel:
    def __init__(self, content: str, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> FakeResult:
        self.calls.append(messages)
        if self.error:
            raise self.error
        return FakeResult(self.content)


def _enable_runtime():
    async def override():
        return RUNTIME

    app.dependency_overrides[load_runtime_settings] = override


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer user-jwt"}


@pytest.fixture
def client():
    _enable_runtime()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_generate_requires_auth():
    with TestClient(app) as test_client:
        response = test_client.post("/names/generate", json=BODY)
    assert response.status_code == 401


def test_generate_disabled_provider():
    async def disabled():
        raise workflow_http_exception(disabled_error())

    app.dependency_overrides[load_runtime_settings] = disabled
    with TestClient(app) as test_client:
        response = test_client.post("/names/generate", json=BODY, headers=_auth_headers())
    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "ERR-ARC-CHAT-02"


def test_generate_requires_product_sentence(client):
    body = {**BODY, "productDescription": {"audience": "Teams"}}
    response = client.post("/names/generate", json=body, headers=_auth_headers())
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "ERR-ARC-NAME-28"


def test_generate_rejects_count_over_20(client):
    response = client.post(
        "/names/generate",
        json={**BODY, "count": 21},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "ERR-ARC-NAME-28"


def test_generate_denies_nonmember(monkeypatch, client):
    from app.arc_todo_client import ArcTodoApiError

    class ForbiddenClient:
        def __init__(self, user_token: str | None = None, **kwargs) -> None:
            self.user_token = user_token

        async def get_name_session(self, *args, **kwargs):
            raise ArcTodoApiError("nope", 403)

    monkeypatch.setattr("app.main.ArcTodoClient", ForbiddenClient)
    response = client.post("/names/generate", json=BODY, headers=_auth_headers())
    assert response.status_code == 403
    assert response.json()["detail"]["error"]["code"] == "ERR-ARC-NAME-29"


def test_generate_returns_structured_suggestions_without_task_mutation(monkeypatch, client):
    api_calls: list[tuple[str, str]] = []
    fake = FakeModel(
        '{"suggestions":[{"name":"Luma","rationale":"warm light","family":"invented"},'
        '{"name":"Nova","rationale":"already seen"}]}'
    )

    class MemberClient:
        def __init__(self, user_token: str | None = None, **kwargs) -> None:
            assert user_token == "user-jwt"

        async def get_name_session(self, session_id):
            api_calls.append(("GET", session_id))
            return {"id": SESS}

        async def request(self, method: str, path: str, **kwargs):
            api_calls.append((method, path))
            raise AssertionError(f"unexpected API {method} {path}")

    monkeypatch.setattr("app.main.ArcTodoClient", MemberClient)
    monkeypatch.setattr("app.naming.build_model", lambda runtime: fake)

    response = client.post("/names/generate", json=BODY, headers=_auth_headers())
    assert response.status_code == 200
    payload = response.json()
    assert payload["usedTools"] == []
    assert payload["suggestions"] == [
        {"name": "Luma", "rationale": "warm light", "family": "invented"},
    ]
    assert api_calls == [("GET", SESS)]
    prompt = str(fake.calls[0][1].content)
    assert "A coffee subscription for small offices." in prompt
    assert "Office managers" in prompt
    assert "Portuguese and English" in prompt
    assert "Warm and direct" in prompt
    assert "Kind of name: Company/organization" in prompt
    assert "- Nova" in prompt
    assert "god-name" not in prompt.lower()
    assert fake.calls[0][0].content.startswith("You suggest names")


def test_generate_provider_error(monkeypatch, client):
    class MemberClient:
        def __init__(self, user_token: str | None = None, **kwargs) -> None:
            return None

        async def get_name_session(self, *args, **kwargs):
            return {"id": SESS}

    monkeypatch.setattr("app.main.ArcTodoClient", MemberClient)
    monkeypatch.setattr(
        "app.naming.build_model",
        lambda runtime: FakeModel("", error=RuntimeError("provider down")),
    )
    response = client.post("/names/generate", json=BODY, headers=_auth_headers())
    assert response.status_code == 502
    assert response.json()["detail"]["error"]["code"] == "ERR-ARC-NAME-30"


def test_generate_malformed_output(monkeypatch, client):
    class MemberClient:
        def __init__(self, user_token: str | None = None, **kwargs) -> None:
            return None

        async def get_name_session(self, *args, **kwargs):
            return {"id": SESS}

    monkeypatch.setattr("app.main.ArcTodoClient", MemberClient)
    monkeypatch.setattr(
        "app.naming.build_model",
        lambda runtime: FakeModel("I think a warm name would be nice for coffee."),
    )
    response = client.post("/names/generate", json=BODY, headers=_auth_headers())
    assert response.status_code == 502
    assert response.json()["detail"]["error"]["code"] == "ERR-ARC-NAME-30"


def test_parse_suggestions_drops_avoided_and_evidence_fields():
    parsed = parse_suggestions(
        '{"suggestions":['
        '{"name":"Helio","rationale":"sun","family":"metaphor","domain":"available","winner":true},'
        '{"name":"nova"}'
        "]}",
        count=10,
        avoid=["Nova"],
    )
    assert parsed == [{"name": "Helio", "rationale": "sun", "family": "metaphor"}]


def test_build_naming_prompt_includes_full_brief_and_count():
    prompt = build_naming_prompt(
        title="fieldlot",
        naming_goal="company",
        product_description={
            "whatItIs": "A loteadora CRM.",
            "audience": "Brokers in Brazil",
            "excludeWords": "lot, plot",
        },
        count=10,
        avoid=["Helio"],
        refinement="More like the shortlist favorite",
    )
    assert "Suggest exactly 10 fresh names" in prompt
    assert "What the product is: A loteadora CRM." in prompt
    assert "Primary audience: Brokers in Brazil" in prompt
    assert "Kind of name: Company/organization" in prompt
    assert "More like the shortlist favorite" in prompt
    assert "- Helio" in prompt
    assert "god-name" not in prompt.lower()
    assert "Do not include domain, trademark, score, rating, or winner fields." in prompt
