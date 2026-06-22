from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)
conversation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "conversation_id",
    default=None,
)
route_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "route",
    default=None,
)
tool_names_var: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "tool_names",
    default=None,
)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.conversation_id = conversation_id_var.get() or "-"
        record.route = route_var.get() or "-"
        tools = tool_names_var.get()
        record.tool_names = ",".join(tools) if tools else "-"
        return True


def configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(asctime)s %(levelname)s "
                "[req=%(request_id)s conv=%(conversation_id)s route=%(route)s tools=%(tool_names)s] "
                "%(name)s: %(message)s"
            ),
        )
    for handler in root.handlers:
        handler.addFilter(RequestContextFilter())


def bind_request_context(
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    route: str | None = None,
    tool_names: list[str] | None = None,
) -> dict[str, contextvars.Token[Any]]:
    tokens: dict[str, contextvars.Token[Any]] = {}
    tokens["request_id"] = request_id_var.set(request_id or str(uuid.uuid4()))
    if conversation_id is not None:
        tokens["conversation_id"] = conversation_id_var.set(conversation_id)
    if route is not None:
        tokens["route"] = route_var.set(route)
    if tool_names is not None:
        tokens["tool_names"] = tool_names_var.set(tool_names)
    return tokens


def reset_request_context(tokens: dict[str, contextvars.Token[Any]]) -> None:
    for var_name, token in reversed(list(tokens.items())):
        if var_name == "request_id":
            request_id_var.reset(token)
        elif var_name == "conversation_id":
            conversation_id_var.reset(token)
        elif var_name == "route":
            route_var.reset(token)
        elif var_name == "tool_names":
            tool_names_var.reset(token)
