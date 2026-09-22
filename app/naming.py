from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.chatbot_settings import ChatbotRuntimeSettings
from app.errors import naming_llm_error
from app.graph.llm import build_model

logger = logging.getLogger(__name__)

NAMING_MIN_COUNT = 1
NAMING_DEFAULT_COUNT = 10
NAMING_MAX_COUNT = 20
NAME_MAX_LENGTH = 40

CANVAS_FIELDS: tuple[tuple[str, str], ...] = (
    ("whatItIs", "What the product is"),
    ("problem", "Problem it solves"),
    ("audience", "Primary audience"),
    ("platform", "Platform"),
    ("benefits", "Core benefits"),
    ("personality", "Brand personality"),
    ("countries", "Countries"),
    ("languages", "Languages"),
    ("competitors", "Avoid/competitors"),
    ("includeWords", "Include"),
    ("excludeWords", "Exclude"),
    ("preferredLength", "Preferred length"),
    ("oneLine", "One-line"),
    ("short", "Short"),
    ("full", "Full"),
)

GOAL_LABELS = {
    "public_product": "Public product/app",
    "company": "Company/organization",
    "feature": "Feature/module",
    "api": "API/developer tool",
    "internal_codename": "Internal codename",
    "campaign": "Campaign/project",
}

NAMING_SYSTEM_PROMPT = """You suggest names for a web tool, product, or company.
Return JSON only. Do not call tools. Do not look up tasks, conversations, or other projects.
Do not judge domain, trademark, language, search ranking, or legal clearance.
Do not include scores, ratings, votes, winners, or evidence fields.
Honor the brief's personality, length, and style. Do not force a god-name or invented-only style unless the brief asks for it.
"""


def clamp_count(count: int | None) -> int:
    if count is None:
        return NAMING_DEFAULT_COUNT
    return max(NAMING_MIN_COUNT, min(NAMING_MAX_COUNT, int(count)))


def format_canvas(desc: dict[str, str] | None) -> str:
    payload = desc or {}
    lines: list[str] = []
    for key, label in CANVAS_FIELDS:
        value = str(payload.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def goal_label(naming_goal: str | None) -> str:
    if not naming_goal:
        return GOAL_LABELS["public_product"]
    return GOAL_LABELS.get(naming_goal, naming_goal)


def build_naming_prompt(
    *,
    title: str,
    naming_goal: str | None,
    product_description: dict[str, str],
    count: int,
    avoid: list[str],
    refinement: str | None = None,
) -> str:
    canvas = format_canvas(product_description) or "(not provided)"
    avoid_block = (
        "Names to avoid (already seen or rejected in this session):\n"
        + "\n".join(f"- {name}" for name in avoid)
        if avoid
        else "Names to avoid: none yet."
    )
    extra = refinement.strip() if refinement and refinement.strip() else ""
    refinement_block = f"\nUser refinement: {extra}\n" if extra else ""
    return f"""Suggest exactly {count} fresh names for this brief. Return JSON only in this shape:
{{"suggestions":[{{"name":"Example","rationale":"short why it fits","family":"invented"}}]}}
family is optional: descriptive, suggestive, invented, compound, metaphor, or codename.
Do not include domain, trademark, score, rating, or winner fields.
Never reuse a name from the avoid list. Do not guess why a name was rejected.

Working name: {title.strip() or "(not provided)"}
Kind of name: {goal_label(naming_goal)}
Product context:
{canvas}
{refinement_block}
{avoid_block}
"""


def _extract_json(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    raw = (fenced.group(1) if fenced else text).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_obj = raw.find("{")
        start_arr = raw.find("[")
        starts = [index for index in (start_obj, start_arr) if index >= 0]
        if not starts:
            return None
        start = min(starts)
        slice_ = raw[start:]
        try:
            return json.loads(slice_)
        except json.JSONDecodeError:
            end = max(slice_.rfind("}"), slice_.rfind("]"))
            if end <= 0:
                return None
            try:
                return json.loads(slice_[: end + 1])
            except json.JSONDecodeError:
                return None


def _clean_name(value: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", value)
    cleaned = cleaned.strip().strip("\"'`").rstrip(".").strip()
    return cleaned


def _normalize_key(name: str) -> str:
    return name.casefold().strip()


def parse_suggestions(
    text: str,
    *,
    count: int,
    avoid: list[str],
) -> list[dict[str, str | None]]:
    blocked = {_normalize_key(name) for name in avoid if name.strip()}
    seen: set[str] = set()
    out: list[dict[str, str | None]] = []

    def add(name: str, rationale: str | None = None, family: str | None = None) -> None:
        cleaned = _clean_name(name)
        key = _normalize_key(cleaned)
        if (
            not key
            or len(cleaned) > NAME_MAX_LENGTH
            or key in blocked
            or key in seen
        ):
            return
        seen.add(key)
        item: dict[str, str | None] = {"name": cleaned}
        if rationale and rationale.strip():
            item["rationale"] = rationale.strip()
        if family and family.strip():
            item["family"] = family.strip()
        out.append(item)

    payload = _extract_json(text)
    if isinstance(payload, dict):
        rows = payload.get("suggestions")
        if not isinstance(rows, list):
            rows = payload.get("names")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, str):
                    add(row)
                elif isinstance(row, dict):
                    name = row.get("name")
                    if isinstance(name, str):
                        rationale = row.get("rationale")
                        family = row.get("family")
                        add(
                            name,
                            rationale if isinstance(rationale, str) else None,
                            family if isinstance(family, str) else None,
                        )
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, str):
                add(row)
            elif isinstance(row, dict) and isinstance(row.get("name"), str):
                rationale = row.get("rationale")
                family = row.get("family")
                add(
                    row["name"],
                    rationale if isinstance(rationale, str) else None,
                    family if isinstance(family, str) else None,
                )

    if not out:
        for line in text.splitlines():
            cleaned = _clean_name(line)
            if cleaned and len(cleaned) <= NAME_MAX_LENGTH:
                add(cleaned)

    return out[:count]


async def generate_name_suggestions(
    runtime: ChatbotRuntimeSettings,
    *,
    title: str,
    naming_goal: str | None,
    product_description: dict[str, str],
    count: int,
    avoid: list[str],
    refinement: str | None = None,
) -> list[dict[str, str | None]]:
    prompt = build_naming_prompt(
        title=title,
        naming_goal=naming_goal,
        product_description=product_description,
        count=count,
        avoid=avoid,
        refinement=refinement,
    )
    model = build_model(runtime)
    try:
        result = await model.ainvoke(
            [
                SystemMessage(content=NAMING_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:
        logger.warning("Naming LLM call failed: %s", exc)
        raise naming_llm_error() from exc

    content = str(getattr(result, "content", "") or "")
    suggestions = parse_suggestions(content, count=count, avoid=avoid)
    if not suggestions:
        raise naming_llm_error(
            "The assistant did not return usable names. Try generate again."
        )
    return suggestions
