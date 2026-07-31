"""Canonical registry for validated, product-visible production renders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUCCESSFUL_RENDER_STATUSES = frozenset({"completed", "warning"})
_RANGE_TOLERANCE_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class ClipResult:
    candidate_id: str
    output_file: str
    status: str = "completed"
    primary: bool = True
    clip_result_id: str = ""
    production_plan_id: str = ""
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    source_fingerprint: str = ""
    content_fingerprint: str = ""
    run_id: str = ""
    revision_id: str = ""
    artifact_id: str = ""
    artifact_checksum: str = ""
    quality_report_id: str = ""
    quality_report_path: str = ""
    quality_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_result_id": self.clip_result_id or _default_result_id(self),
            "candidate_id": self.candidate_id,
            "production_plan_id": self.production_plan_id,
            "output_file": self.output_file,
            "source_start_seconds": self.source_start_seconds,
            "source_end_seconds": self.source_end_seconds,
            "source_fingerprint": self.source_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "run_id": self.run_id,
            "revision_id": self.revision_id,
            "artifact_id": self.artifact_id,
            "artifact_checksum": self.artifact_checksum,
            "quality_report_id": self.quality_report_id,
            "quality_report_path": self.quality_report_path,
            "quality_status": self.quality_status,
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
        return cls(
            candidate_id=candidate_id,
            output_file=output_file,
            status=status,
            primary=bool(value.get("primary", True)),
            clip_result_id=str(value.get("clip_result_id") or "").strip(),
            production_plan_id=str(value.get("production_plan_id") or "").strip(),
            source_start_seconds=_as_float(value.get("source_start_seconds")),
            source_end_seconds=_as_float(value.get("source_end_seconds")),
            source_fingerprint=str(value.get("source_fingerprint") or "").strip(),
            content_fingerprint=str(value.get("content_fingerprint") or "").strip(),
            run_id=str(value.get("run_id") or "").strip(),
            revision_id=str(value.get("revision_id") or "").strip(),
            artifact_id=str(value.get("artifact_id") or "").strip(),
            artifact_checksum=str(value.get("artifact_checksum") or "").strip(),
            quality_report_id=str(value.get("quality_report_id") or "").strip(),
            quality_report_path=str(value.get("quality_report_path") or "").strip(),
            quality_status=str(value.get("quality_status") or "").strip(),
        )


def primary_clip_results(production_render: dict[str, Any] | None) -> list[ClipResult]:
    """Return one non-duplicated product result for every distinct render."""

    if not isinstance(production_render, dict):
        return []
    results: list[ClipResult] = []
    raw_items = production_render.get("items", [])
    if isinstance(raw_items, list):
        for item in raw_items:
            result = ClipResult.from_dict(item)
            if result is not None:
                results.append(result)
    if results:
        return unique_primary_results(results)
    direct = ClipResult.from_dict(production_render)
    if direct is not None:
        return [direct]
    output_file = str(production_render.get("output_file") or "").strip()
    status = str(production_render.get("status") or "").strip()
    if output_file and status in SUCCESSFUL_RENDER_STATUSES:
        return [ClipResult("primary", output_file, status)]
    return []


def unique_primary_results(results: Iterable[ClipResult]) -> list[ClipResult]:
    """Defensive last line: a final list cannot expose the same result twice.

    Every modern production result carries all of these keys.  Older reports
    remain readable: only keys that are present participate in de-duplication.
    """

    seen_paths: set[str] = set()
    seen_candidates: set[str] = set()
    seen_plans: set[str] = set()
    seen_content: set[str] = set()
    seen_ranges: list[tuple[float, float]] = []
    unique: list[ClipResult] = []
    for result in results:
        if not result.primary:
            continue
        path_key = str(Path(result.output_file)).replace("\\", "/").casefold()
        if path_key in seen_paths or result.candidate_id in seen_candidates:
            continue
        if result.production_plan_id and result.production_plan_id in seen_plans:
            continue
        if result.content_fingerprint and result.content_fingerprint in seen_content:
            continue
        source_range = _source_range(result)
        if source_range is not None and any(_same_range(source_range, prior) for prior in seen_ranges):
            continue
        seen_paths.add(path_key)
        seen_candidates.add(result.candidate_id)
        if result.production_plan_id:
            seen_plans.add(result.production_plan_id)
        if result.content_fingerprint:
            seen_content.add(result.content_fingerprint)
        if source_range is not None:
            seen_ranges.append(source_range)
        unique.append(result)
    return unique


def result_paths(results: Iterable[ClipResult], output_directory: Path) -> list[Path]:
    paths: list[Path] = []
    for result in unique_primary_results(results):
        path = Path(result.output_file)
        path = path if path.is_absolute() else output_directory / path
        if path not in paths:
            paths.append(path)
    return paths


def _default_result_id(result: ClipResult) -> str:
    return ":".join((result.candidate_id, result.production_plan_id or "legacy", result.output_file))


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _source_range(result: ClipResult) -> tuple[float, float] | None:
    if result.source_start_seconds is None or result.source_end_seconds is None:
        return None
    return result.source_start_seconds, result.source_end_seconds


def _same_range(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return (
        abs(first[0] - second[0]) <= _RANGE_TOLERANCE_SECONDS
        and abs(first[1] - second[1]) <= _RANGE_TOLERANCE_SECONDS
    )
