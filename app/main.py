from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.chatbot_settings import ChatbotSettingsError, chatbot_settings_client
from app.config import settings
from app.graph.workflow import run_chat_workflow
from app.models import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Arc Todo Chatbot")

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


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_token: str = Depends(extract_bearer_token),
):
    try:
        runtime = await chatbot_settings_client.get_runtime_settings()
    except ChatbotSettingsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not runtime.enabled:
        raise HTTPException(status_code=503, detail="Chatbot is disabled")

    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

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
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        message=result.get("response") or "I could not generate a response.",
        usedTools=result.get("used_tools", []),
    )
