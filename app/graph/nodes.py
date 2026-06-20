from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.arc_todo_client import ArcTodoClient
from app.chatbot_settings import ChatbotRuntimeSettings
from app.graph.state import ChatGraphState
from app.tools.todo_tools import TodoTools, execute_todo_tool

SYSTEM_PROMPT = """You are Arc Todo assistant. Help users manage tasks in Arc Todo.
You can answer questions about todos and perform actions using available tools.
When organization or project context is missing, ask the user to clarify or use list_organizations and list_projects first.
Keep responses concise and practical."""

PLANNER_PROMPT = """Analyze the latest user message and decide whether todo API tools are needed.

Return JSON only with this shape:
{"route":"direct"|"tools","tool_name":string|null,"tool_arguments":object}

Available tools:
- list_organizations {}
- list_projects {"organization_id": string}
- list_tasks {"organization_id": string|null, "project_id": string|null, "status": string|null, "criticity": string|null}
- get_task {"organization_id": string, "project_id": string, "task_id": string}
- create_task {"organization_id": string, "project_id": string, "title": string, "description": string|null, "status": "todo"|"in_progress"|"done", "criticity": "low"|"medium"|"high"|"critical", "due_date": string|null}
- update_task {"organization_id": string, "project_id": string, "task_id": string, "title": string|null, "description": string|null, "status": string|null, "criticity": string|null, "due_date": string|null}
- delete_task {"organization_id": string, "project_id": string, "task_id": string}

Use route "direct" for greetings, general help, or when no API action is needed.
Prefer provided organization_id and project_id context when present."""

RESPONSE_PROMPT = """Write a concise assistant reply for the user based on the conversation and any tool results.
Do not mention internal tool names unless helpful."""


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


def _format_task_context_line(task: dict[str, Any], fallback_title: str) -> str:
    title = task.get("title") or fallback_title
    status = task.get("status", "unknown")
    criticity = task.get("criticity", "unknown")
    due_date = task.get("dueDate") or task.get("due_date") or "none"
    description = (task.get("description") or "").strip() or "none"
    task_id = task.get("id", "unknown")
    organization_id = task.get("organizationId") or task.get("organization_id")
    project_id = task.get("projectId") or task.get("project_id")
    return (
        f"- taskId: {task_id}\n"
        f"  title: {title}\n"
        f"  status: {status}\n"
        f"  criticity: {criticity}\n"
        f"  organizationId: {organization_id}\n"
        f"  projectId: {project_id}\n"
        f"  dueDate: {due_date}\n"
        f"  description: {description}"
    )


async def _build_task_context_text(
    *,
    user_token: str,
    task_refs: list[dict[str, str]],
) -> str:
    if not task_refs:
        return ""

    client = ArcTodoClient(user_token=user_token)
    tools = TodoTools(client)
    lines: list[str] = []

    for ref in task_refs:
        task_id = ref.get("taskId") or ref.get("task_id")
        organization_id = ref.get("organizationId") or ref.get("organization_id")
        project_id = ref.get("projectId") or ref.get("project_id")
        title = ref.get("title") or task_id or "Task"

        if not task_id or not organization_id or not project_id:
            lines.append(
                f"- taskId: {task_id or 'unknown'}\n"
                f"  title: {title}\n"
                f"  note: missing organization or project scope"
            )
            continue

        try:
            task = await tools.get_task(
                organization_id=organization_id,
                project_id=project_id,
                task_id=task_id,
            )
            lines.append(_format_task_context_line(task, title))
        except Exception:
            lines.append(
                f"- taskId: {task_id}\n"
                f"  title: {title}\n"
                f"  organizationId: {organization_id}\n"
                f"  projectId: {project_id}\n"
                f"  note: task details could not be loaded"
            )

    return "Selected task context:\n" + "\n".join(lines)


async def context_agent(state: ChatGraphState) -> ChatGraphState:
    messages = state.get("messages", [])
    latest = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    task_context_text = await _build_task_context_text(
        user_token=state["user_token"],
        task_refs=state.get("task_refs", []),
    )
    return {
        **state,
        "latest_user_message": latest,
        "task_context_text": task_context_text,
        "used_tools": state.get("used_tools", []),
    }


async def planner_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    model = build_model(runtime)
    context_bits = []
    if state.get("organization_id"):
        context_bits.append(f"organization_id={state['organization_id']}")
    if state.get("project_id"):
        context_bits.append(f"project_id={state['project_id']}")

    prompt = PLANNER_PROMPT
    if context_bits:
        prompt += "\nContext: " + ", ".join(context_bits)
    if state.get("task_context_text"):
        prompt += "\n\n" + state["task_context_text"]

    result = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state.get("latest_user_message", "")),
        ]
    )
    payload = _extract_json(str(result.content))
    route = payload.get("route", "direct")
    if route not in {"direct", "tools"}:
        route = "direct"

    return {
        **state,
        "route": route,
        "tool_name": payload.get("tool_name"),
        "tool_arguments": payload.get("tool_arguments") or {},
    }


async def todo_tools_agent(state: ChatGraphState) -> ChatGraphState:
    tool_name = state.get("tool_name")
    if not tool_name:
        return {**state, "error": "Planner did not select a tool"}

    arguments = dict(state.get("tool_arguments") or {})
    if state.get("organization_id") and "organization_id" not in arguments:
        arguments["organization_id"] = state["organization_id"]
    if state.get("project_id") and "project_id" not in arguments:
        arguments["project_id"] = state["project_id"]

    client = ArcTodoClient(user_token=state["user_token"])
    tools = TodoTools(client)
    result = await execute_todo_tool(tools, tool_name, arguments)
    used_tools = list(state.get("used_tools", []))
    used_tools.append(tool_name)

    return {
        **state,
        "tool_result": result,
        "used_tools": used_tools,
        "error": None,
    }


async def response_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    model = build_model(runtime)
    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in state.get("messages", [])
    )
    tool_context = ""
    if state.get("tool_result") is not None:
        tool_context = f"\nTool result:\n{json.dumps(state['tool_result'], indent=2, default=str)}"
    if state.get("error"):
        tool_context += f"\nError:\n{state['error']}"
    if state.get("task_context_text"):
        tool_context += f"\n\n{state['task_context_text']}"

    result = await model.ainvoke(
        [
            SystemMessage(content=RESPONSE_PROMPT),
            HumanMessage(content=f"Conversation:\n{conversation}{tool_context}"),
        ]
    )
    return {**state, "response": str(result.content).strip()}


def route_after_planner(state: ChatGraphState) -> str:
    if state.get("route") == "tools" and state.get("tool_name"):
        return "todo_tools_agent"
    return "response_agent"
