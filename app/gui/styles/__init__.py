from __future__ import annotations

from pathlib import Path

from app.gui.styles.tokens import THEME_TOKENS


def load_theme() -> str:
    theme = (Path(__file__).with_name("theme.qss")).read_text(encoding="utf-8")
    for name, value in THEME_TOKENS.items():
        theme = theme.replace(f"@{name}@", value)
    if "@" in theme:
        unresolved = sorted({part.split("@", 1)[0] for part in theme.split("@")[1::2]})
        raise ValueError(f"Unresolved Desktop theme tokens: {', '.join(unresolved)}")
    return theme


__all__ = ["THEME_TOKENS", "load_theme"]
