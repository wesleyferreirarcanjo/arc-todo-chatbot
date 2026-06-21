from __future__ import annotations

import re

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)
FRIENDLY_TASK_ID_PATTERN = re.compile(r"^#?[a-z]{3}-\d+$", re.I)


def is_uuid(value: str | None) -> bool:
    return bool(value and UUID_PATTERN.match(value))


def is_friendly_task_id(value: str | None) -> bool:
    return bool(value and FRIENDLY_TASK_ID_PATTERN.match(value.strip()))


def normalize_friendly_task_id(value: str) -> str:
    return value.strip().lower().lstrip("#")
