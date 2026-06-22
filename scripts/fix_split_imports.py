from __future__ import annotations

import ast
import pathlib
import re

GRAPH_DIR = pathlib.Path("app/graph")
MODULES = [
    "prompts",
    "llm",
    "heuristics",
    "mutations",
    "scope",
    "task_refs",
    "agents",
    "routing",
]

HEADER_END = "logger = logging.getLogger(__name__)\n"

IMPORT_ALLOWLIST = {
    "llm": {"prompts"},
    "scope": {"prompts"},
    "task_refs": {"scope", "prompts", "llm", "heuristics"},
    "heuristics": {"prompts", "scope", "task_refs", "llm", "mutations"},
    "mutations": {"prompts", "llm", "scope", "heuristics", "task_refs"},
    "agents": {"prompts", "llm", "scope", "task_refs", "heuristics", "mutations"},
    "routing": {"heuristics", "scope"},
}

PUBLIC_SYMBOLS = {
    "build_model",
    "context_agent",
    "planner_agent",
    "response_agent",
    "scope_discovery_agent",
    "todo_tools_agent",
    "resolve_scope_arguments",
    "resolve_delete_tasks_arguments",
    "resolve_update_tasks_arguments",
    "resolve_update_tasks_arguments_async",
    "resolve_get_tasks_arguments",
    "resolve_move_tasks_arguments",
    "route_after_context",
    "route_after_scope_discovery",
    "route_after_tools",
    "route_after_planner",
}

BUILTIN_NAMES = set(dir(__builtins__)) | {
    "Any",
    "ChatGraphState",
    "ChatOpenAI",
    "ChatbotRuntimeSettings",
    "HumanMessage",
    "SystemMessage",
    "ArcTodoApiError",
    "ArcTodoClient",
    "TodoTools",
    "execute_todo_tool",
    "is_friendly_task_id",
    "is_uuid",
    "normalize_friendly_task_id",
    "trim_messages",
    "get_stream_handler",
    "json",
    "logging",
    "re",
    "logger",
}


def is_importable_symbol(name: str) -> bool:
    if name in PUBLIC_SYMBOLS:
        return True
    if name.startswith("_"):
        return True
    if name.isupper():
        return True
    return False


def module_symbols(path: pathlib.Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return defined, used


def main() -> None:
    symbol_to_module: dict[str, str] = {}
    module_defined: dict[str, set[str]] = {}
    module_used: dict[str, set[str]] = {}

    for module in MODULES:
        path = GRAPH_DIR / f"{module}.py"
        defined, used = module_symbols(path)
        module_defined[module] = defined
        module_used[module] = used
        for symbol in defined:
            symbol_to_module[symbol] = module

    for module in MODULES:
        path = GRAPH_DIR / f"{module}.py"
        content = path.read_text(encoding="utf-8")
        body_start = content.index(HEADER_END) + len(HEADER_END)
        header = content[:body_start]
        body = content[body_start:]
        body = re.sub(r"^from app\.graph\..*\n", "", body, flags=re.MULTILINE)
        body = re.sub(r"^logger = logging\.getLogger\(__name__\)\n", "", body)

        imports_by_module: dict[str, set[str]] = {}
        allowed = IMPORT_ALLOWLIST.get(module, set())
        for symbol in sorted(module_used[module]):
            if not is_importable_symbol(symbol):
                continue
            if symbol in module_defined[module] or symbol in BUILTIN_NAMES:
                continue
            owner = symbol_to_module.get(symbol)
            if owner and owner != module and owner in allowed:
                imports_by_module.setdefault(owner, set()).add(symbol)

        import_lines = [
            f"from app.graph.{owner} import {', '.join(sorted(symbols))}"
            for owner, symbols in sorted(imports_by_module.items())
        ]

        new_body = ("\n".join(import_lines) + "\n\n" if import_lines else "") + body.lstrip("\n")
        path.write_text(header + new_body, encoding="utf-8")
        print(f"fixed imports in {module}.py ({len(import_lines)} groups)")


if __name__ == "__main__":
    main()
