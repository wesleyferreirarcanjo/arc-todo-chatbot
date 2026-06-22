from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.chatbot_settings import ChatbotSettingsError, chatbot_settings_client
from app.config import settings
from app.errors import WorkflowError, disabled_error, settings_error, validation_error
from app.graph.workflow import run_chat_workflow, run_chat_workflow_streaming
from app.http_pool import create_shared_http_client, get_shared_http_client, set_shared_http_client
from app.logging_context import bind_request_context, configure_logging, reset_request_context
from app.models import ChatRequest, ChatResponse
from app.streaming import StreamEventHandler, format_sse

logger = logging.getLogger(__name__)
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = create_shared_http_client()
    set_shared_http_client(client)
    try:
        yield
    finally:
        await client.aclose()
        set_shared_http_client(None)


app = FastAPI(title="Arc Todo Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


def workflow_http_exception(error: WorkflowError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail=error.to_payload(),
    )


async def load_runtime_settings():
    try:
        runtime = await chatbot_settings_client.get_runtime_settings()
    except ChatbotSettingsError as exc:
        raise workflow_http_exception(settings_error(str(exc))) from exc
    if not runtime.enabled:
        raise workflow_http_exception(disabled_error())
    return runtime


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    tokens = bind_request_context(route=request.url.path)
    try:
        return await call_next(request)
    finally:
        reset_request_context(tokens)


@app.get("/health")
async def health_check(detail: bool = Query(default=False)):
    if not detail:
        return {"status": "ok"}

    checks: dict[str, dict[str, str]] = {
        "chatbot": {"status": "ok"},
        "api": {"status": "unknown"},
        "chatbotSettings": {"status": "unknown"},
    }

    try:
        client = get_shared_http_client()
        api_response = await client.get(f"{settings.arc_todo_api_base_url.rstrip('/')}/health")
        checks["api"] = {
            "status": "ok" if api_response.is_success else "error",
            "httpStatus": str(api_response.status_code),
        }
    except Exception as exc:
        checks["api"] = {"status": "error", "message": str(exc)}

    try:
        await chatbot_settings_client.get_runtime_settings(force_refresh=True)
        checks["chatbotSettings"] = {"status": "ok"}
    except ChatbotSettingsError as exc:
        checks["chatbotSettings"] = {"status": "error", "message": str(exc)}

    overall = "ok" if all(item.get("status") == "ok" for item in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_token: str = Depends(extract_bearer_token),
):
    if not request.messages:
        raise workflow_http_exception(validation_error("At least one message is required"))

    runtime = await load_runtime_settings()
    tokens = bind_request_context(conversation_id=request.conversation_id, route="/chat")
    try:
        result = await run_chat_workflow(
            runtime=runtime,
            messages=[message.model_dump() for message in request.messages],
            user_token=user_token,
            organization_id=request.organization_id,
            project_id=request.project_id,
            conversation_id=request.conversation_id,
            task_refs=[task_ref.model_dump(by_alias=True) for task_ref in request.task_refs],
        )
    except WorkflowError as exc:
        logger.warning("Chat workflow failed: %s", exc.message)
        raise workflow_http_exception(exc) from exc
    finally:
        reset_request_context(tokens)

    return ChatResponse(
        message=result.get("response") or "I could not generate a response.",
        usedTools=result.get("used_tools", []),
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_token: str = Depends(extract_bearer_token),
):
    if not request.messages:
        raise workflow_http_exception(validation_error("At least one message is required"))

    runtime = await load_runtime_settings()

    async def event_generator():
        handler = StreamEventHandler()
        tokens = bind_request_context(
            conversation_id=request.conversation_id,
            route="/chat/stream",
        )
        workflow_task = asyncio.create_task(
            run_chat_workflow_streaming(
                runtime=runtime,
                messages=[message.model_dump() for message in request.messages],
                user_token=user_token,
                organization_id=request.organization_id,
                project_id=request.project_id,
                conversation_id=request.conversation_id,
                task_refs=[task_ref.model_dump(by_alias=True) for task_ref in request.task_refs],
                handler=handler,
            )
        )
        sent = 0
        try:
            while True:
                while sent < len(handler.queue):
                    event, data = handler.queue[sent]
                    sent += 1
                    yield format_sse(event, data)
                    if event in {"done", "error"}:
                        await workflow_task
                        return
                if workflow_task.done():
                    break
                await asyncio.sleep(0.01)

            if workflow_task.cancelled():
                return
            exc = workflow_task.exception()
            if exc:
                if isinstance(exc, WorkflowError):
                    yield format_sse("error", {"code": exc.code, "message": exc.message})
                else:
                    yield format_sse("error", {"code": "unexpected", "message": str(exc)})
                return

            while sent < len(handler.queue):
                event, data = handler.queue[sent]
                sent += 1
                yield format_sse(event, data)
        finally:
            reset_request_context(tokens)
            if not workflow_task.done():
                workflow_task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
