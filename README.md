# arc-todo-chatbot

Python FastAPI chatbot service for Arc Todo. Uses LangGraph to orchestrate a small multi-agent workflow and DeepSeek for language understanding. The chatbot calls `arc-todo-api` directly using the same REST endpoints as `arc-todo-mcp`.

## Endpoints

- `GET /health` — service health
- `POST /chat` — chat with the assistant

## Configuration

Runtime AI provider settings are loaded from `arc-todo-api` at `/chatbot-settings/runtime`.

Service configuration is provided through Coolify environment variables. See [coolify.md](./coolify.md).

## Development

This project is deployed through Coolify. Local startup is optional and not part of the verification flow.

```bash
pip install -e ".[dev]"
pytest
```
