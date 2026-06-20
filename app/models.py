from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    organization_id: str | None = Field(default=None, alias="organizationId")
    project_id: str | None = Field(default=None, alias="projectId")

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    message: str
    used_tools: list[str] = Field(default_factory=list, alias="usedTools")

    model_config = {"populate_by_name": True}
