"""Canonical registry for the final clips exposed to product surfaces.

Pipeline working directories can contain cached source media, intermediate files,
and legacy renderer outputs.  None of those are user-facing results.  The
registry intentionally records only a validated production-render result per
selected candidate so every consumer counts the same files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUCCESSFUL_RENDER_STATUSES = frozenset({"completed", "warning"})


@dataclass(frozen=True, slots=True)
class ClipResult:
    candidate_id: str
    output_file: str
    status: str = "completed"
    primary: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "output_file": self.output_file,
            "status": self.status,
            "primary": self.primary,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ClipResult | None":
        if not isinstance(value, dict):
            return None
        candidate_id = str(value.get("candidate_id") or "").strip()
        output_file = str(value.get("output_file") or "").strip()
        status = str(value.get("status") or "").strip()
        if not candidate_id or not output_file or status not in SUCCESSFUL_RENDER_STATUSES:
            return None
        return cls(candidate_id=candidate_id, output_file=output_file, status=status, primary=bool(value.get("primary", True)))


def primary_clip_results(production_render: dict[str, Any] | None) -> list[ClipResult]:
    """Return the sole list of product-visible clips from a render report."""

    if not isinstance(production_render, dict):
        return []
    results: list[ClipResult] = []
    raw_items = production_render.get("items", [])
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            result = ClipResult(
                candidate_id=str(item.get("candidate_id") or "").strip(),
                output_file=str(item.get("output_file") or "").strip(),
                status=str(item.get("status") or "").strip(),
            )
            if result.candidate_id and result.output_file and result.status in SUCCESSFUL_RENDER_STATUSES:
                results.append(result)
    if results:
        return _deduplicate(results)
    # Render-only runs have one report rather than a multi-candidate fanout.
    output_file = str(production_render.get("output_file") or "").strip()
    status = str(production_render.get("status") or "").strip()
    if output_file and status in SUCCESSFUL_RENDER_STATUSES:
        return [ClipResult("primary", output_file, status)]
    return []


def result_paths(results: list[ClipResult], output_directory: Path) -> list[Path]:
    """Resolve registry paths, retaining registry order and removing duplicates."""

    paths: list[Path] = []
    for result in results:
        path = Path(result.output_file)
        path = path if path.is_absolute() else output_directory / path
        if path not in paths:
            paths.append(path)
    return paths


def _deduplicate(results: list[ClipResult]) -> list[ClipResult]:
    seen: set[str] = set()
    unique: list[ClipResult] = []
    for result in results:
        key = str(Path(result.output_file))
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique
