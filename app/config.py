from pydantic_settings import BaseSettings, SettingsConfigDict


def get_settings() -> "Settings":
    return Settings()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8010
    arc_todo_api_base_url: str = "http://localhost:3000"
    arc_todo_username: str | None = None
    arc_todo_password: str | None = None
    arc_todo_access_token: str | None = None
    cors_origins: str = "http://localhost:5173"
    chatbot_settings_cache_seconds: int = 60
    http_timeout_seconds: float = 60.0
    http_max_connections: int = 20
    http_max_keepalive_connections: int = 10
    todo_tools_batch_concurrency: int = 5

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = get_settings()
