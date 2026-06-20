# Coolify — arc-todo Chatbot

Python FastAPI chatbot service deployed in Coolify project **`arc-todo`** on server **`main`** (`72.60.59.203`).

## Project

| Field | Value |
| --- | --- |
| Coolify project name | `arc-todo` |
| Coolify project UUID | `qzmm8hhki6jz02yrrc21zung` |
| Environment | `production` (`oqofaco0eved39jqee22w7jo`) |
| Server UUID | `r9rokxstz1zlccajjxyenk93` |
| Destination UUID | `wchjqtdyj949s0ale2zofwgd` |

## This application

| Field | Value |
| --- | --- |
| Coolify resource name | `arc-todo-chatbot` |
| Application UUID | *(create in Coolify — add UUID here after provisioning)* |
| Repository | [wesleyferreirarcanjo/arc-todo-chatbot](https://github.com/wesleyferreirarcanjo/arc-todo-chatbot) |
| Branch | `main` |
| Build pack | Dockerfile |
| Public URL | *(assign in Coolify — add URL here after provisioning)* |
| Health check | `GET /health` → `{ "status": "ok" }` |

### Build / run

| Step | Command |
| --- | --- |
| Build | `docker build -f Dockerfile .` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Port | `8010` |

## Related resources

| Resource | UUID | Notes |
| --- | --- | --- |
| API `arc-todo-api` | `lmsx2avrg1k29ex12w6e3gce` | `http://lmsx2avrg1k29ex12w6e3gce.72.60.59.203.sslip.io` |
| Frontend `arc-todo-web` | `ifo33mi1s8efs8myb5g441vh` | Chat UI at `/chat`, settings at `/settings/chatbot` |
| MCP `arc-todo-mcp` | `qv9bek5he3ns8upu71rphbrc` | Shares the same Arc Todo API endpoints |
| PostgreSQL `arc-todo-postgres` | `bibl6ncxa3xkph2r8ubmbl4t` | Stores chatbot settings via API |

## Environment variables (production)

Secrets are stored in Coolify only. Do not commit real values.

| Variable | Purpose |
| --- | --- |
| `PORT` | `8010` |
| `ARC_TODO_API_BASE_URL` | Public or internal API URL |
| `ARC_TODO_USERNAME` | Service account username |
| `ARC_TODO_PASSWORD` | *(redacted — Coolify secret)* |
| `ARC_TODO_ACCESS_TOKEN` | Optional bearer token instead of username/password |
| `CORS_ORIGINS` | Frontend URL (`http://ifo33mi1s8efs8myb5g441vh.72.60.59.203.sslip.io`) |
| `CHATBOT_SETTINGS_CACHE_SECONDS` | `60` |

DeepSeek provider settings (`provider`, `baseUrl`, `model`, `apiKey`, `temperature`, `enabled`) are stored in PostgreSQL through `arc-todo-api` and loaded at runtime from `GET /chatbot-settings/runtime`.

## Deploy order

1. Ensure `arc-todo-postgres` is `running:healthy`.
2. Deploy / restart `arc-todo-api` so the `chatbot_settings` migration runs.
3. Deploy `arc-todo-web` with `VITE_CHAT_API_BASE_URL` set to this service URL.
4. Configure chatbot settings at `/settings/chatbot` in the web app.
5. Deploy / restart `arc-todo-chatbot`.

## Health-check verification checklist

- [ ] `GET <api-url>/health` returns `{ "status": "ok" }`
- [ ] `GET <chatbot-url>/health` returns `{ "status": "ok" }`
- [ ] Frontend loads from its Coolify URL and `/chat` route is available
- [ ] Chatbot settings page loads at `/settings/chatbot`

## Notes

- This service is not verified through local startup. Coolify deployment and health checks are the source of truth.
- The chatbot forwards the web user's bearer token to `arc-todo-api` for todo actions.
- Runtime AI settings are fetched from the API; do not put DeepSeek secrets in this service's env vars.
- Git source uses the Coolify deploy key (`private_key_uuid`: `lms2y9fjpybdznft4t7uf3td`).
- See [../arc-todo-api/coolify.md](../arc-todo-api/coolify.md), [../arc-todo-web/coolify.md](../arc-todo-web/coolify.md), and [../arc-todo-mcp/coolify.md](../arc-todo-mcp/coolify.md).
