from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.arc_todo_client import ArcTodoApiError, ArcTodoClient
from app.chatbot_settings import ChatbotRuntimeSettings
from app.graph.state import ChatGraphState
from app.history import trim_messages
from app.streaming import get_stream_handler
from app.task_id_resolver import is_friendly_task_id, is_uuid, normalize_friendly_task_id
from app.tools.todo_tools import TodoTools, execute_todo_tool

logger = logging.getLogger(__name__)
from app.graph.prompts import MAX_PLANNED_ACTIONS, UPDATE_TASK_FIELDS

def build_model(runtime: ChatbotRuntimeSettings) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=runtime.api_key,
        base_url=runtime.base_url.rstrip("/") + "/",
        model=runtime.model,
        temperature=runtime.temperature,
    )

def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

def _normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if "priority" in normalized and "criticity" not in normalized:
        normalized["criticity"] = normalized.pop("priority")
    if "parent_id" in normalized and "parent_task_id" not in normalized:
        normalized["parent_task_id"] = normalized.pop("parent_id")
    raw_tasks = normalized.get("tasks")
    if isinstance(raw_tasks, list):
        normalized["tasks"] = [
            _normalize_tool_arguments(task) if isinstance(task, dict) else task
            for task in raw_tasks
        ]
    return normalized

def _normalize_planner_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("actions")
    normalized: list[dict[str, Any]] = []
    if isinstance(actions, list):
        for item in actions[:MAX_PLANNED_ACTIONS]:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            normalized.append(
                {
                    "intent": str(item.get("intent") or tool_name),
                    "tool_name": tool_name,
                    "tool_arguments": _normalize_tool_arguments(
                        item.get("tool_arguments") or {}
                    ),
                }
            )
    if normalized:
        return normalized

    tool_name = payload.get("tool_name")
    if isinstance(tool_name, str) and tool_name:
        return [
            {
                "intent": tool_name,
                "tool_name": tool_name,
                "tool_arguments": _normalize_tool_arguments(
                    payload.get("tool_arguments") or {}
                ),
            }
        ]
    return []

def _update_arguments_have_fields(tool_name: str, arguments: dict[str, Any]) -> bool:
    if tool_name == "update_task":
        return any(arguments.get(key) is not None for key in UPDATE_TASK_FIELDS)
    if tool_name == "update_tasks":
        raw_tasks = arguments.get("tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            return any(
                any(item.get(key) is not None for key in UPDATE_TASK_FIELDS)
                for item in raw_tasks
                if isinstance(item, dict)
            )
        return any(arguments.get(key) is not None for key in UPDATE_TASK_FIELDS)
    return True

async def _generate_task_description_from_title(
    runtime: ChatbotRuntimeSettings,
    *,
    title: str,
    message: str,
) -> str:
    from app.graph import nodes

    model = nodes.build_model(runtime)
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Write a concise, creative task description that matches the title "
                    "and user request. Return JSON only: "
                    '{"description":"<description>"}'
                )
            ),
            HumanMessage(content=f"User request:\n{message}\n\nTask title:\n{title}"),
        ]
    )
    payload = _extract_json(str(result.content))
    description = payload.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return f"Scope: {title}."

async def _maybe_generate_create_descriptions(
    runtime: ChatbotRuntimeSettings | None,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    if not runtime:
        return arguments
    wants_description = bool(
        re.search(r"\b(?:description|detail|note)\b", message, re.I)
    )
    if not wants_description:
        return arguments

    updated = dict(arguments)
    if tool_name == "create_task" and updated.get("title") and not updated.get(
        "description"
    ):
        updated["description"] = await _generate_task_description_from_title(
            runtime,
            title=str(updated["title"]),
            message=message,
        )
    if tool_name == "create_tasks":
        tasks = updated.get("tasks")
        if isinstance(tasks, list):
            enriched: list[dict[str, Any]] = []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                item = dict(task)
                if item.get("title") and not item.get("description"):
                    item["description"] = await _generate_task_description_from_title(
                        runtime,
                        title=str(item["title"]),
                        message=message,
                    )
                enriched.append(item)
            updated["tasks"] = enriched
    return updated

async def _generate_per_task_descriptions(
    runtime: ChatbotRuntimeSettings,
    *,
    message: str,
    task_refs: list[dict[str, str]],
    task_context_text: str = "",
) -> dict[str, str]:
    from app.graph import nodes
    from app.graph.task_refs import _effective_task_refs, _ref_task_id

    model = nodes.build_model(runtime)
    refs = _effective_task_refs(task_refs, message)
    task_lines = "\n".join(
        f"- task_id={task_id} title={ref.get('title') or 'Untitled'}"
        for ref in refs
        if (task_id := _ref_task_id(ref))
    )
    context_block = f"\n\n{task_context_text}" if task_context_text else ""
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Write a distinct task description for each task id based on its title "
                    "and the user request. Return JSON only in the shape "
                    '{"descriptions":{"<task_id>":"<description>"}}. '
                    "Each description must be specific to that task's title. "
                    "Ignore incorrect or copied current descriptions. "
                    "Do not reuse the same description text."
                )
            ),
            HumanMessage(
                content=f"User request:\n{message}\n\nTasks:\n{task_lines}{context_block}"
            ),
        ]
    )
    payload = _extract_json(str(result.content))
    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, dict):
        return {}
    return {
        str(task_id): str(description).strip()
        for task_id, description in descriptions.items()
        if description
    }
