from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.arc_todo_client import ArcTodoApiError


@dataclass(slots=True)
class WorkflowError(Exception):
    code: str
    stage: str
    message: str
    status_code: int = 502
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "stage": self.stage,
                "message": self.message,
            }
        }
        if self.details:
            payload["error"]["details"] = self.details
        return payload


def validation_error(message: str, *, stage: str = "request") -> WorkflowError:
    return WorkflowError(
        code="validation",
        stage=stage,
        message=message,
        status_code=400,
    )


def disabled_error(message: str = "Chatbot is disabled") -> WorkflowError:
    return WorkflowError(
        code="disabled",
        stage="settings",
        message=message,
        status_code=503,
    )


def settings_error(message: str) -> WorkflowError:
    return WorkflowError(
        code="api",
        stage="settings",
        message=message,
        status_code=503,
    )


def api_error(message: str, *, stage: str = "tools", status_code: int = 502) -> WorkflowError:
    return WorkflowError(
        code="api",
        stage=stage,
        message=message,
        status_code=status_code,
    )


def llm_error(message: str, *, stage: str = "response") -> WorkflowError:
    return WorkflowError(
        code="llm",
        stage=stage,
        message=message,
        status_code=502,
    )


def unexpected_error(message: str, *, stage: str = "workflow") -> WorkflowError:
    return WorkflowError(
        code="unexpected",
        stage=stage,
        message=message,
        status_code=502,
    )


def from_exception(exc: Exception, *, stage: str = "workflow") -> WorkflowError:
    if isinstance(exc, WorkflowError):
        return exc
    if isinstance(exc, ArcTodoApiError):
        status = exc.status_code or 502
        if status in {401, 403}:
            message = "Your session cannot access the todo API. Sign in again and retry."
        elif status == 404:
            message = "The requested todo resource was not found."
        else:
            message = str(exc)
        return api_error(message, stage=stage, status_code=min(status, 599))
    message = str(exc).strip() or "The chat workflow failed unexpectedly."
    return unexpected_error(message, stage=stage)
