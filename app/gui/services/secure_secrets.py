from __future__ import annotations

import os
from pathlib import Path


def key_configured(provider: str, root: Path | None = None) -> bool:
    """The GUI intentionally exposes only a boolean and never persists a key."""

    variables = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    variable = variables.get(provider)
    if not variable:
        return False
    if os.getenv(variable):
        return True
    dotenv = (root / ".env") if root else None
    if not dotenv or not dotenv.is_file():
        return False
    try:
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            key, separator, value = raw.partition("=")
            if separator and key.strip() == variable and value.strip().strip("'\""):
                return True
    except OSError:
        return False
    return False
