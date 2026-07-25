from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str, fallback: str = "source") -> str:
    stem = Path(value).stem if value else fallback
    cleaned = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", stem).strip("._-")
    return (cleaned or fallback)[:80]


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        temporary = Path(file.name)
    temporary.replace(path)


def format_seconds(value: float | None) -> str:
    if value is None:
        return "н/д"
    minutes, seconds = divmod(max(0, value), 60)
    return f"{int(minutes)} мин {seconds:04.1f} с" if minutes else f"{seconds:.1f} с"
