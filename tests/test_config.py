from app.config import Settings, get_settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    settings = Settings()
    assert settings.port == 8010
    assert "localhost:5173" in settings.cors_origin_list[0]


def test_get_settings_returns_instance():
    assert isinstance(get_settings(), Settings)
