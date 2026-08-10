from __future__ import annotations

import re

from PySide6.QtWidgets import QLabel, QSizePolicy


_LONG_TOKEN = re.compile(r"\S{29,}")


def break_long_tokens(value: object, *, chunk: int = 28) -> str:
    """Add invisible wrap opportunities without changing the source value.

    Qt's word wrapping preserves an uninterrupted word.  Persisted project
    names commonly contain underscores, while URLs and technical failures can
    be one very long token.  A zero-width space keeps the rendered/copied text
    readable and leaves the original value available in a tooltip.
    """

    text = str(value)
    if chunk < 1:
        raise ValueError("chunk must be positive")

    def split(match: re.Match[str]) -> str:
        token = match.group(0)
        return "\u200b".join(token[index:index + chunk] for index in range(0, len(token), chunk))

    return _LONG_TOKEN.sub(split, text)


def make_label_shrinkable(label: QLabel) -> QLabel:
    """Allow a wrapped label to follow its layout instead of widening it."""

    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return label


def set_responsive_text(label: QLabel, value: object, *, tooltip: bool = True) -> None:
    """Render hostile long tokens safely while preserving their full value."""

    text = str(value)
    label.setText(break_long_tokens(text))
    if tooltip:
        label.setToolTip(text)
