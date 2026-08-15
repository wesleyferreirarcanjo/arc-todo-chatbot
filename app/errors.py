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


def validation_error(
    message: str = "Write a message before sending.",
    *,
    stage: str = "request",
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-CHAT-01",
        stage=stage,
        message=message,
        status_code=400,
    )


def disabled_error(
    message: str = "Chat is turned off right now. Ask an administrator to enable it.",
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-CHAT-02",
        stage="settings",
        message=message,
        status_code=503,
    )


def settings_error(
    message: str = "Chat settings could not be loaded. Try again in a moment.",
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-CHAT-03",
        stage="settings",
        message=message,
        status_code=503,
    )


def api_error(
    message: str = "Chat could not reach Arc Todo. Try again in a moment.",
    *,
    stage: str = "tools",
    status_code: int = 502,
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-CHAT-04",
        stage=stage,
        message=message,
        status_code=status_code,
    )


def llm_error(
    message: str = "The assistant could not reply just now. Try sending again.",
    *,
    stage: str = "response",
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-CHAT-05",
        stage=stage,
        message=message,
        status_code=502,
    )


def unexpected_error(
    message: str = "Chat hit an unexpected problem. Try again, or start a new conversation.",
    *,
    stage: str = "workflow",
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-CHAT-06",
        stage=stage,
        message=message,
        status_code=502,
    )


def auth_error(
    message: str = "Sign in again to keep chatting.",
) -> WorkflowError:
    return WorkflowError(
        code="ERR-ARC-AUTH-11",
        stage="auth",
        message=message,
        status_code=401,
    )


def from_exception(exc: Exception, *, stage: str = "workflow") -> WorkflowError:
    if isinstance(exc, WorkflowError):
        return exc
    if isinstance(exc, ArcTodoApiError):
        status = exc.status_code or 502
        if status in {401, 403}:
            message = "Your session cannot use chat tools right now. Sign in again and retry."
        elif status == 404:
            message = "Chat could not find that Arc Todo item. It may have been removed."
        else:
            message = str(exc).strip() or "Chat could not reach Arc Todo. Try again in a moment."
        return api_error(message, stage=stage, status_code=min(status, 599))
    message = str(exc).strip() or "Chat hit an unexpected problem. Try again, or start a new conversation."
    return unexpected_error(message, stage=stage)
