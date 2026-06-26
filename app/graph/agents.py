from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.chatbot_settings import ChatbotRuntimeSettings
from app.graph.state import ChatGraphState
from app.history import trim_messages
from app.streaming import get_stream_handler
from app.task_id_resolver import is_friendly_task_id, is_uuid, normalize_friendly_task_id

logger = logging.getLogger(__name__)
from app.graph.heuristics import _apply_subtask_parent_from_refs, _build_create_parent_with_subtasks_actions, _coerce_mutation_tool, _create_tasks_have_parent, _inject_parent_from_previous, _looks_like_bulk_selected, _looks_like_create_mutation, _looks_like_move_mutation, _looks_like_mutation_confirmation, _looks_like_reparent_mutation, _looks_like_task_mutation, _looks_like_update_mutation, _parse_create_parent_with_subtasks, _resolve_bug_flag_arguments, _resolve_confirmation_update_arguments, _resolve_reparent_arguments
from app.graph.llm import _extract_json, _maybe_generate_create_descriptions, _normalize_planner_actions, _normalize_tool_arguments
from app.graph.mutations import _action_succeeded, _build_mutation_failure_response, _build_verified_mutation_response, _mutation_succeeded, _needs_mutation_tool_result, _validate_mutation_arguments
from app.graph.prompts import PLANNER_MUTATION_RETRY, PLANNER_PROMPT, RESPONSE_PROMPT, UPDATE_TASK_FIELDS
from app.graph.scope import EXISTING_TASK_TOOLS, _apply_resolved_scope, _catalog_from_scope_result, _extract_all_scope_hints, _is_uuid, _resolve_scope_via_api, resolve_scope_arguments
from app.graph.task_refs import BATCH_TOOL_RESOLVERS, SINGLE_TO_BATCH_TOOL, _apply_task_ref_source_scope, _build_task_context_text, resolve_move_tasks_arguments, resolve_update_tasks_arguments_async

async def scope_discovery_agent(state: ChatGraphState) -> ChatGraphState:
    from app.graph import nodes

    message = state.get("latest_user_message", "")
    org_hint, project_hints = _extract_all_scope_hints(message)
    project_hint = project_hints[0] if project_hints else None

    client = nodes.ArcTodoClient(user_token=state["user_token"])
    tools = nodes.TodoTools(client)
    used_tools = list(state.get("used_tools", []))
    if "resolve_scope" not in used_tools:
        used_tools.append("resolve_scope")

    scope_result = await _resolve_scope_via_api(
        tools,
        organization_hint=org_hint,
        project_hint=project_hint,
        message=message,
        state_org=state.get("organization_id"),
        state_proj=state.get("project_id"),
    )
    scope_status = scope_result.get("status", "not_found")
    updates: ChatGraphState = {
        **state,
        "used_tools": used_tools,
        "scope_catalog": _catalog_from_scope_result(scope_result),
        "scope_status": scope_status,
        "scope_retried": bool(state.get("scope_retried")),
    }
    if scope_status == "resolved":
        _apply_resolved_scope(updates, scope_result)

    logger.info(
        "scope discovery org=%s project=%s org_hint=%s project_hints=%s status=%s",
        updates.get("organization_id"),
        updates.get("project_id"),
        org_hint,
        project_hints,
        scope_status,
    )

    if _looks_like_create_mutation(message) and scope_status == "resolved":
        parsed_parent = _parse_create_parent_with_subtasks(message)
        if parsed_parent:
            actions = _build_create_parent_with_subtasks_actions(updates, parsed_parent)
            updates["route"] = "tools"
            updates["tool_name"] = actions[0]["tool_name"]
            updates["tool_arguments"] = actions[0]["tool_arguments"]
            updates["actions"] = actions
        else:
            tool_name, tool_arguments = _coerce_mutation_tool(
                updates,
                "create_tasks",
                dict(updates.get("tool_arguments") or {}),
            )
            updates["route"] = "tools"
            updates["tool_name"] = tool_name
            updates["tool_arguments"] = tool_arguments
            updates["actions"] = [
                {
                    "intent": "create tasks",
                    "tool_name": tool_name,
                    "tool_arguments": tool_arguments,
                }
            ]

    return updates

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

async def retrieval_agent(state: ChatGraphState) -> ChatGraphState:
    from app.graph import nodes
    from app.graph.rag_context import build_rag_context_text, build_retrieval_query
    from app.rag_client import RagClient, RagClientError

    latest = state.get("latest_user_message", "").strip()
    if not latest:
        return {
            **state,
            "rag_chunks": [],
            "rag_context_text": "",
            "rag_error": None,
            "rag_search_query": None,
            "rag_token_usage": None,
            "rag_index_status": None,
        }

    question = build_retrieval_query(
        state.get("messages", []),
        latest,
        task_context_text=state.get("task_context_text"),
    )
    rag_client = RagClient(user_token=state["user_token"])
    try:
        result = await rag_client.retrieve(
            question=question,
            organization_id=state.get("organization_id"),
            project_id=state.get("project_id"),
        )
        chunks = result.get("chunks") or []
        rag_error = None
        rag_search_query = result.get("searchQuery")
        rag_token_usage = result.get("tokenUsage")
        rag_index_status = result.get("indexStatus")
    except RagClientError as exc:
        chunks = []
        rag_error = str(exc)
        rag_search_query = None
        rag_token_usage = None
        rag_index_status = None
        logger.warning("RAG retrieval failed: %s", rag_error)

    rag_context_text = build_rag_context_text(
        chunks,
        rag_error=rag_error,
        index_status=rag_index_status,
    )
    return {
        **state,
        "rag_chunks": chunks,
        "rag_context_text": rag_context_text,
        "rag_error": rag_error,
        "rag_search_query": rag_search_query,
        "rag_token_usage": rag_token_usage,
        "rag_index_status": rag_index_status,
    }

async def planner_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    from app.graph import nodes

    messages = state.get("messages", [])
    task_refs = state.get("task_refs", [])
    latest_user_message = state.get("latest_user_message", "")

    if _looks_like_mutation_confirmation(latest_user_message) and task_refs:
        apply_arguments = _resolve_confirmation_update_arguments(messages, task_refs)
        if apply_arguments:
            logger.info("planner confirmation apply update_tasks tasks=%s", len(apply_arguments.get("tasks", [])))
            return {
                **state,
                "route": "tools",
                "actions": [
                    {
                        "intent": "apply proposed descriptions",
                        "tool_name": "update_tasks",
                        "tool_arguments": apply_arguments,
                    }
                ],
                "tool_name": "update_tasks",
                "tool_arguments": apply_arguments,
            }

    model = nodes.build_model(runtime)
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
    if state.get("rag_context_text"):
        prompt += "\n\n" + state["rag_context_text"]

    planner_messages = trim_messages(messages, max_messages=6, max_tokens=2000)
    from app.graph.rag_context import format_recent_conversation

    conversation_text = format_recent_conversation(planner_messages)
    human_content = latest_user_message
    if conversation_text and conversation_text.strip() != f"User: {latest_user_message}".strip():
        human_content = f"Recent conversation:\n{conversation_text}\n\nLatest message:\n{latest_user_message}"

    result = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=human_content),
        ]
    )
    payload = _extract_json(str(result.content))
    route = payload.get("route", "direct")
    if route not in {"direct", "tools"}:
        route = "direct"

    latest_user_message = state.get("latest_user_message", "")
    if route == "direct" and (
        _looks_like_task_mutation(latest_user_message)
        or (task_refs and _looks_like_update_mutation(latest_user_message))
    ):
        retry = await model.ainvoke(
            [
                SystemMessage(content=prompt + PLANNER_MUTATION_RETRY),
                HumanMessage(content=latest_user_message),
            ]
        )
        retry_payload = _extract_json(str(retry.content))
        retry_route = retry_payload.get("route", route)
        if retry_route == "tools" and retry_payload.get("tool_name"):
            payload = retry_payload
            route = "tools"

        if retry_route == "tools" and (
            retry_payload.get("actions") or retry_payload.get("tool_name")
        ):
            payload = retry_payload
            route = "tools"

    actions = _normalize_planner_actions(payload)
    if route == "tools" and not actions:
        route = "direct"

    logger.info(
        "planner route=%s actions=%s org=%s project=%s",
        route,
        [action.get("tool_name") for action in actions],
        state.get("organization_id"),
        state.get("project_id"),
    )

    updates: ChatGraphState = {
        **state,
        "route": route,
        "actions": actions,
    }
    if actions:
        updates["tool_name"] = actions[0]["tool_name"]
        updates["tool_arguments"] = actions[0]["tool_arguments"]
    else:
        updates["tool_name"] = None
        updates["tool_arguments"] = {}
    return updates

async def _execute_single_tool(
    state: ChatGraphState,
    runtime: ChatbotRuntimeSettings | None,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    apply_heuristics: bool = True,
) -> tuple[str, dict[str, Any], Any, str | None]:
    from app.graph import nodes

    arguments = _normalize_tool_arguments(dict(arguments or {}))
    if not _is_uuid(arguments.get("organization_id")) and _is_uuid(state.get("organization_id")):
        arguments["organization_id"] = state["organization_id"]
    elif not arguments.get("organization_id") and state.get("organization_id"):
        arguments["organization_id"] = state["organization_id"]
    if not _is_uuid(arguments.get("project_id")) and _is_uuid(state.get("project_id")):
        arguments["project_id"] = state["project_id"]
    elif not arguments.get("project_id") and state.get("project_id"):
        arguments["project_id"] = state["project_id"]

    task_id = arguments.get("task_id")
    if task_id and (
        not arguments.get("organization_id") or not arguments.get("project_id")
    ):
        for ref in state.get("task_refs", []):
            ref_task_id = ref.get("taskId") or ref.get("task_id")
            if ref_task_id != task_id:
                continue
            if not arguments.get("organization_id"):
                arguments["organization_id"] = ref.get("organizationId") or ref.get(
                    "organization_id"
                )
            if not arguments.get("project_id"):
                arguments["project_id"] = ref.get("projectId") or ref.get("project_id")
            break

    if not arguments.get("organization_id") and state.get("task_refs"):
        first_ref = state["task_refs"][0]
        arguments.setdefault(
            "organization_id",
            first_ref.get("organizationId") or first_ref.get("organization_id"),
        )
        arguments.setdefault(
            "project_id",
            first_ref.get("projectId") or first_ref.get("project_id"),
        )

    task_refs = state.get("task_refs", [])
    latest_user_message = state.get("latest_user_message", "")
    if task_refs and tool_name in EXISTING_TASK_TOOLS:
        arguments = _apply_task_ref_source_scope(arguments, task_refs)

    if apply_heuristics:
        reparent_arguments = _resolve_reparent_arguments(
            latest_user_message,
            task_refs,
        )
        if reparent_arguments:
            tool_name = "update_task"
            arguments = reparent_arguments

        bug_arguments = _resolve_bug_flag_arguments(
            latest_user_message,
            task_refs,
        )
        if bug_arguments:
            tool_name = "update_task"
            arguments = bug_arguments

        if task_refs and _looks_like_move_mutation(latest_user_message):
            tool_name = "move_tasks"
            arguments = {
                key: value
                for key, value in arguments.items()
                if key in {"task_ids", "target_project_id", "target_project_hint"}
            }
            arguments.setdefault("task_ids", None)

        if task_refs and _looks_like_update_mutation(latest_user_message) and not _looks_like_move_mutation(
            latest_user_message
        ) and not _looks_like_reparent_mutation(latest_user_message):
            tool_name = "update_tasks"
            if isinstance(arguments.get("tasks"), list):
                arguments = {
                    "tasks": arguments["tasks"],
                    "task_ids": arguments.get("task_ids"),
                }
            else:
                arguments = {
                    key: arguments[key]
                    for key in UPDATE_TASK_FIELDS
                    if key in arguments and arguments[key] is not None
                }
                arguments.setdefault("task_ids", None)

        tool_name, arguments = _coerce_mutation_tool(state, tool_name, arguments)

    if task_refs and tool_name in EXISTING_TASK_TOOLS:
        arguments = _apply_task_ref_source_scope(arguments, task_refs)

    if apply_heuristics and (
        tool_name in SINGLE_TO_BATCH_TOOL
        and len(task_refs) > 1
        and (
            _looks_like_bulk_selected(latest_user_message)
            or _looks_like_move_mutation(latest_user_message)
            or (not arguments.get("task_id") and not arguments.get("task_ids"))
        )
    ):
        batch_arguments = dict(arguments)
        batch_arguments.setdefault("task_ids", None)
        tool_name = SINGLE_TO_BATCH_TOOL[tool_name]
        arguments = batch_arguments

    client = nodes.ArcTodoClient(user_token=state["user_token"])
    tools = nodes.TodoTools(client)
    try:
        if tool_name in {"move_task", "move_tasks"}:
            move_arguments = await nodes.resolve_move_tasks_arguments(
                tools,
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
            )
            if tool_name == "move_task":
                tasks = move_arguments.get("tasks") or []
                if len(tasks) != 1:
                    return tool_name, arguments, None, "Could not resolve selected task for move"
                task = tasks[0]
                arguments = {
                    "organization_id": task["organization_id"],
                    "project_id": task["project_id"],
                    "task_id": task["task_id"],
                    "target_project_id": task["target_project_id"],
                }
            else:
                arguments = move_arguments
        elif tool_name == "update_tasks":
            arguments = await nodes.resolve_update_tasks_arguments_async(
                runtime,
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
                task_context_text=state.get("task_context_text") or "",
            )
        elif tool_name in nodes.BATCH_TOOL_RESOLVERS:
            arguments = nodes.BATCH_TOOL_RESOLVERS[tool_name](
                arguments=arguments,
                task_refs=task_refs,
                latest_user_message=latest_user_message,
            )

        if tool_name in {"create_task", "create_tasks"}:
            arguments.pop("task_ids", None)
            arguments = _apply_subtask_parent_from_refs(
                arguments,
                task_refs,
                latest_user_message,
            )
            arguments = await nodes._maybe_generate_create_descriptions(
                runtime,
                tool_name=tool_name,
                arguments=arguments,
                message=latest_user_message,
            )

        arguments = await nodes.resolve_scope_arguments(
            tools,
            tool_name=tool_name,
            arguments=arguments,
            state=state,
        )
        validation_error = _validate_mutation_arguments(tool_name, arguments)
        if validation_error:
            logger.warning(
                "mutation validation failed tool=%s error=%s args=%s",
                tool_name,
                validation_error,
                {
                    key: arguments.get(key)
                    for key in ("organization_id", "project_id", "title", "tasks")
                },
            )
            return tool_name, arguments, None, validation_error
        api_arguments = dict(arguments)
        api_arguments.pop("_parent_title", None)
        from app.tools.knowledge_tools import KNOWLEDGE_TOOLS, KnowledgeTools, execute_knowledge_tool
        from app.rag_client import RagClient

        if tool_name in KNOWLEDGE_TOOLS:
            knowledge_tools = KnowledgeTools(
                client,
                rag_client=RagClient(user_token=state["user_token"]),
            )
            result = await execute_knowledge_tool(knowledge_tools, tool_name, api_arguments)
        else:
            result = await nodes.execute_todo_tool(tools, tool_name, api_arguments)
        return tool_name, arguments, result, None
    except nodes.ArcTodoApiError as exc:
        return tool_name, arguments, None, str(exc)
    except Exception as exc:
        return tool_name, arguments, None, str(exc)

async def todo_tools_agent(
    state: ChatGraphState,
    runtime: ChatbotRuntimeSettings | None = None,
) -> ChatGraphState:
    from app.graph import nodes

    actions = list(state.get("actions") or [])
    if not actions and state.get("tool_name"):
        actions = [
            {
                "intent": state.get("tool_name") or "action",
                "tool_name": state["tool_name"],
                "tool_arguments": dict(state.get("tool_arguments") or {}),
            }
        ]

    if not actions:
        return {**state, "error": "Planner did not select a tool"}

    apply_heuristics = len(actions) == 1
    tool_results: list[dict[str, Any]] = []
    used_tools = list(state.get("used_tools", []))
    last_tool_name: str | None = None
    last_tool_result: Any = None
    last_error: str | None = None

    for action in actions:
        tool_name = action["tool_name"]
        raw_arguments = dict(action.get("tool_arguments") or {})
        needs_parent = bool(raw_arguments.get("_parent_from_previous"))
        arguments = _inject_parent_from_previous(raw_arguments, tool_results)
        if needs_parent and not _create_tasks_have_parent(arguments):
            tool_results.append(
                {
                    "intent": action.get("intent") or tool_name,
                    "tool_name": tool_name,
                    "tool_arguments": arguments,
                    "tool_result": None,
                    "error": "Could not resolve parent task from previous create step",
                    "success": False,
                    "partial": False,
                }
            )
            last_tool_name = tool_name
            last_error = "Could not resolve parent task from previous create step"
            continue
        executed_name, executed_args, result, error = await nodes._execute_single_tool(
            state,
            runtime,
            tool_name=tool_name,
            arguments=arguments,
            apply_heuristics=apply_heuristics,
        )
        success, partial = _action_succeeded(executed_name, result, error)
        tool_results.append(
            {
                "intent": action.get("intent") or executed_name,
                "tool_name": executed_name,
                "tool_arguments": executed_args,
                "tool_result": result,
                "error": error,
                "success": success,
                "partial": partial,
            }
        )
        if executed_name not in used_tools:
            used_tools.append(executed_name)
        last_tool_name = executed_name
        last_tool_result = result
        last_error = error
        logger.info(
            "tool executed tool=%s success=%s partial=%s",
            executed_name,
            success,
            partial,
        )

    any_success = any(item.get("success") for item in tool_results)
    return {
        **state,
        "route": "tools",
        "actions": actions,
        "tool_results": tool_results,
        "tool_name": last_tool_name,
        "tool_result": last_tool_result,
        "used_tools": used_tools,
        "error": None if any_success else last_error,
    }

async def response_agent(state: ChatGraphState, runtime: ChatbotRuntimeSettings) -> ChatGraphState:
    from app.graph import nodes

    if state.get("scope_status") in {"ambiguous", "not_found"} and _looks_like_create_mutation(
        state.get("latest_user_message", "")
    ):
        response_text = _build_mutation_failure_response(state)
        handler = get_stream_handler()
        if handler:
            await handler.emit_done(response_text, state.get("used_tools", []))
        return {**state, "response": response_text}

    if _needs_mutation_tool_result(state):
        verified = _build_verified_mutation_response(state)
        if verified:
            handler = get_stream_handler()
            if handler:
                await handler.emit_done(verified, state.get("used_tools", []))
            return {**state, "response": verified}

    model = nodes.build_model(runtime)
    messages = trim_messages(
        state.get("messages", []),
        max_messages=runtime.max_history_messages,
        max_tokens=runtime.max_history_tokens,
    )
    conversation = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    tool_context = ""
    tool_results = state.get("tool_results") or []
    if tool_results:
        tool_context = (
            "\nVerified action results:\n"
            + json.dumps(tool_results, indent=2, default=str)
        )
    elif state.get("tool_result") is not None:
        tool_context = f"\nTool result:\n{json.dumps(state['tool_result'], indent=2, default=str)}"
    if state.get("error"):
        tool_context += f"\nError:\n{state['error']}"
    if state.get("task_context_text"):
        tool_context += f"\n\n{state['task_context_text']}"
    if state.get("rag_context_text"):
        tool_context += f"\n\n{state['rag_context_text']}"

    prompt_messages = [
        SystemMessage(content=RESPONSE_PROMPT),
        HumanMessage(content=f"Conversation:\n{conversation}{tool_context}"),
    ]
    handler = get_stream_handler()
    if handler:
        response_parts: list[str] = []
        async for chunk in model.astream(prompt_messages):
            delta = str(chunk.content or "")
            if delta:
                response_parts.append(delta)
                await handler.emit_token(delta)
        response_text = "".join(response_parts).strip()
    else:
        result = await model.ainvoke(prompt_messages)
        response_text = str(result.content).strip()

    if _needs_mutation_tool_result(state):
        succeeded = _mutation_succeeded(state)
        if succeeded is False or (
            succeeded is None and state.get("route") == "direct"
        ):
            response_text = _build_mutation_failure_response(state)
        else:
            verified = _build_verified_mutation_response(state)
            if verified:
                response_text = verified

    if handler:
        await handler.emit_done(response_text, state.get("used_tools", []))

    return {**state, "response": response_text}
