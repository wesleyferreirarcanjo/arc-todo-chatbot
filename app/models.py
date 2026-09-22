from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class TaskRef(BaseModel):
    task_id: str = Field(alias="taskId")
    organization_id: str = Field(alias="organizationId")
    project_id: str = Field(alias="projectId")
    title: str
    display_id: str | None = Field(default=None, alias="displayId")

    model_config = {"populate_by_name": True}


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    organization_id: str | None = Field(default=None, alias="organizationId")
    project_id: str | None = Field(default=None, alias="projectId")
    conversation_id: str | None = Field(default=None, alias="conversationId")
    task_refs: list[TaskRef] = Field(default_factory=list, alias="taskRefs")

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    message: str
    used_tools: list[str] = Field(default_factory=list, alias="usedTools")

    model_config = {"populate_by_name": True}


class NamingGenerateRequest(BaseModel):
    organization_id: str | None = Field(default=None, alias="organizationId")
    project_id: str | None = Field(default=None, alias="projectId")
    session_id: str = Field(alias="sessionId")
    title: str = ""
    naming_goal: str | None = Field(default=None, alias="namingGoal")
    product_description: dict[str, str] = Field(
        default_factory=dict,
        alias="productDescription",
    )
    count: int = 10
    avoid: list[str] = Field(default_factory=list)
    refinement: str | None = None

    model_config = {"populate_by_name": True}


class NamingSuggestion(BaseModel):
    name: str
    rationale: str | None = None
    family: str | None = None


class NamingGenerateResponse(BaseModel):
    suggestions: list[NamingSuggestion]
    used_tools: list[str] = Field(default_factory=list, alias="usedTools")

    model_config = {"populate_by_name": True}
