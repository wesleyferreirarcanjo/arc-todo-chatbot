from __future__ import annotations

import pytest

from app.history import trim_messages


def test_trim_messages_keeps_latest_user_message():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest question"},
    ]

    trimmed = trim_messages(messages, max_messages=2, max_tokens=None)

    assert trimmed[-1]["content"] == "latest question"
    assert len(trimmed) == 2


def test_trim_messages_applies_token_limit():
    messages = [
        {"role": "user", "content": "x" * 400},
        {"role": "assistant", "content": "y" * 400},
        {"role": "user", "content": "keep me"},
    ]

    trimmed = trim_messages(messages, max_messages=50, max_tokens=120)

    assert trimmed[-1]["content"] == "keep me"
    from app.history import estimate_tokens

    assert sum(estimate_tokens(message["content"]) for message in trimmed) <= 120
