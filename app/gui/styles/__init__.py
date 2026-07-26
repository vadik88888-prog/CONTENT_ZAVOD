from __future__ import annotations

from pathlib import Path


def load_theme() -> str:
    return (Path(__file__).with_name("theme.qss")).read_text(encoding="utf-8")
