from __future__ import annotations

"""Resolve canonical, pre-rendered Settings previews by production identity."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from app.caption_presets import CAPTION_PRESET_DEFINITIONS
from app.creative_policy import creative_preset_definition


SETTINGS_PREVIEW_SCHEMA_VERSION = "friend-beta.settings-preview.2"


def settings_preview_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "settings-previews"


@lru_cache(maxsize=1)
def settings_preview_manifest() -> dict[str, Any]:
    path = settings_preview_root() / "manifest.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("schema_version") != SETTINGS_PREVIEW_SCHEMA_VERSION:
        return {}
    return raw


def settings_preview_path(style_id: str, caption_preset_id: str) -> Path | None:
    """Resolve one exact current style + caption artifact, never a name scan."""

    caption = CAPTION_PRESET_DEFINITIONS.get(caption_preset_id)  # type: ignore[arg-type]
    if caption is None:
        return None
    try:
        style = creative_preset_definition(style_id)  # type: ignore[arg-type]
    except KeyError:
        return None
    items = settings_preview_manifest().get("items")
    if not isinstance(items, list):
        return None
    record = next((
        item for item in items
        if isinstance(item, dict)
        and item.get("creative_style_id") == style_id
        and item.get("creative_style_version") == style.preset_version
        and item.get("caption_preset_id") == caption.preset_id
        and item.get("caption_preset_version") == caption.preset_version
    ), None)
    relative = record.get("path") if isinstance(record, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        return None
    root = settings_preview_root().resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_file() and path.stat().st_size > 0 else None
