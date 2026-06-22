from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path("app/graph/nodes.py")
lines = SRC.read_text(encoding="utf-8").splitlines()
tree = ast.parse("\n".join(lines))

items: list[tuple[str, str, int, int]] = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        items.append(("func", node.name, node.lineno, node.end_lineno or node.lineno))
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                items.append(("assign", target.id, node.lineno, node.end_lineno or node.lineno))

groups: dict[str, set[str]] = {
    "prompts": {
        "SYSTEM_PROMPT",
        "MAX_PLANNED_ACTIONS",
        "PLANNER_PROMPT",
        "RESPONSE_PROMPT",
        "MUTATION_TOOLS",
        "UPDATE_TASK_FIELDS",
        "PLANNER_MUTATION_RETRY",
        "CONFIRMATION_PATTERN",
    },
    "llm": {
        "build_model",
        "_extract_json",
        "_normalize_tool_arguments",
        "_normalize_planner_actions",
        "_update_arguments_have_fields",
        "_generate_task_description_from_title",
        "_maybe_generate_create_descriptions",
        "_generate_per_task_descriptions",
    },
    "routing": {
        "route_after_context",
        "route_after_scope_discovery",
        "route_after_tools",
        "route_after_planner",
    },
    "agents": {
        "context_agent",
        "planner_agent",
        "todo_tools_agent",
        "response_agent",
        "_execute_single_tool",
        "scope_discovery_agent",
    },
    "mutations": {
        "_action_succeeded",
        "_mutation_succeeded",
        "_build_mutation_failure_response",
        "_validate_mutation_arguments",
        "_format_action_success_line",
        "_format_action_failure_line",
        "_build_verified_mutation_response",
        "_needs_mutation_tool_result",
    },
    "scope": {
        "_extract_all_scope_hints",
        "_extract_scope_hints",
        "_project_hint_variants",
        "_normalize_scope_name",
        "_pick_uuid",
        "_match_scope_name",
        "_scope_item_labels",
        "_best_scope_match",
        "_catalog_from_scope_result",
        "_apply_resolved_scope",
        "_resolve_scope_via_api",
        "resolve_scope_arguments",
        "_needs_scope_retry",
        "SCOPE_TOOLS",
        "EXISTING_TASK_TOOLS",
        "UUID_PATTERN",
        "_is_uuid",
    },
    "task_refs": {
        "_format_task_context_line",
        "_build_task_context_text",
        "_ref_task_id",
        "_ref_display_id",
        "_normalize_task_identifier",
        "_ref_matches_task_identifier",
        "_task_ref_lookup",
        "_ref_organization_id",
        "_ref_project_id",
        "_apply_task_ref_source_scope",
        "_filter_task_refs_by_message",
        "_effective_task_refs",
        "_batch_task_ids",
        "_batch_task_scopes",
        "SINGLE_TO_BATCH_TOOL",
        "BATCH_TOOL_RESOLVERS",
        "resolve_delete_tasks_arguments",
        "resolve_update_tasks_arguments",
        "_build_per_task_updates_from_arguments",
        "resolve_update_tasks_arguments_async",
        "resolve_get_tasks_arguments",
        "resolve_move_tasks_arguments",
    },
}

assigned = set().union(*groups.values())
groups["heuristics"] = {name for _, name, _, _ in items if name not in assigned}

HEADER = """from __future__ import annotations

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
"""

out_dir = pathlib.Path("app/graph")
for group, names in groups.items():
    chunks: list[str] = []
    for _, name, start, end in items:
        if name in names:
            chunks.append("\n".join(lines[start - 1 : end]))
    if not chunks:
        continue
    (out_dir / f"{group}.py").write_text(HEADER + "\n\n".join(chunks) + "\n", encoding="utf-8")
    print(f"{group}: {len(chunks)} symbols")
