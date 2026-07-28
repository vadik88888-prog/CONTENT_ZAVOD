"""Run-scoped canonical output manifest for desktop results and recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.clip_results import ClipResult
from app.utils import utc_now, write_json


MANIFEST_VERSION = "4D.0"


def write_run_manifest(
    path: Path, *, run_id: str, source: dict[str, Any], started_at: str,
    requested_clip_count: int, production_render: dict[str, Any], results: list[ClipResult],
    run_directory: Path, project_id: str | None = None, content_understanding: dict[str, Any] | None = None,
    virality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = run_directory.resolve()
    seen_clip_results: set[str] = set()
    seen_candidates: set[str] = set()
    seen_plans: set[str] = set()
    seen_paths: set[Path] = set()
    for result in results:
        if result.run_id != run_id:
            raise ValueError(
                f"Canonical result run_id does not match manifest: {result.run_id or '<missing>'} != {run_id}"
            )
        if not result.revision_id:
            raise ValueError("Canonical result is missing revision_id.")
        candidate = Path(result.output_file).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Canonical result escapes current run directory: {candidate}") from error
        clip_result_id = result.clip_result_id or f"{result.candidate_id}:{result.production_plan_id}:{candidate}"
        if clip_result_id in seen_clip_results:
            raise ValueError(f"Canonical manifest contains duplicate clip_result_id: {clip_result_id}")
        if result.candidate_id in seen_candidates:
            raise ValueError(f"Canonical manifest contains duplicate candidate_id: {result.candidate_id}")
        if result.production_plan_id and result.production_plan_id in seen_plans:
            raise ValueError(f"Canonical manifest contains duplicate production_plan_id: {result.production_plan_id}")
        if candidate in seen_paths:
            raise ValueError(f"Canonical manifest contains duplicate output path: {candidate}")
        seen_clip_results.add(clip_result_id)
        seen_candidates.add(result.candidate_id)
        if result.production_plan_id:
            seen_plans.add(result.production_plan_id)
        seen_paths.add(candidate)
    status = str(production_render.get("status") or "failed")
    data = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "source_id": str(source.get("id") or ""),
        "project_id": project_id,
        "started_at": started_at,
        "finished_at": utc_now(),
        "requested_clip_count": requested_clip_count,
        "completed_clip_count": len(results),
        "status": status,
        "run_directory": str(root),
        "primary_results": [result.to_dict() for result in results],
        "result_paths": [result.output_file for result in results],
        "candidate_ids": [result.candidate_id for result in results],
        "production_plan_ids": [result.production_plan_id for result in results],
        "source_temporal_ranges": [
            {"start_seconds": result.source_start_seconds, "end_seconds": result.source_end_seconds}
            for result in results
        ],
        "content_fingerprints": [result.content_fingerprint for result in results],
        "media_fingerprints": [result.content_fingerprint for result in results],
        "subtitle_status": _subtitle_status(production_render),
        "audio_mode": production_render.get("audio_mode"),
        "composition_mode": production_render.get("resolution"),
        "content_understanding": content_understanding or {},
        "virality": virality or {},
    }
    write_json(path, data)
    return data


def is_run_scoped_path(path: Path, run_directory: Path) -> bool:
    try:
        path.resolve().relative_to(run_directory.resolve())
        return True
    except (OSError, ValueError):
        return False


def _subtitle_status(production_render: dict[str, Any]) -> dict[str, Any]:
    """Expose every rendered subtitle contract in the run manifest."""

    raw_items = production_render.get("items", []) if isinstance(production_render, dict) else []
    reports: list[tuple[str, dict[str, Any]]] = []
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        report = item.get("report")
        if isinstance(report, dict):
            reports.append((str(item.get("candidate_id") or report.get("candidate_id") or ""), report))
    if not reports and isinstance(production_render, dict):
        reports.append((str(production_render.get("candidate_id") or ""), production_render))
    items: list[dict[str, Any]] = []
    validations: list[str] = []
    for candidate_id, report in reports:
        quality = report.get("quality", {})
        validation = quality.get("status") if isinstance(quality, dict) else None
        if isinstance(validation, str):
            validations.append(validation)
        layout = report.get("subtitle_layout", {})
        cues = layout.get("cues", []) if isinstance(layout, dict) else []
        items.append({
            "candidate_id": candidate_id,
            "enabled": bool(report.get("subtitles_enabled", production_render.get("enabled"))),
            "cue_count": int(report.get("subtitle_cue_count") or 0),
            "validation": validation,
            "fallback_cue_count": sum(1 for cue in cues if isinstance(cue, dict) and cue.get("fallback_used")),
        })
    status = "failed" if "failed" in validations else "warning" if "warning" in validations else "passed" if validations else None
    return {
        "enabled": bool(items) and all(item["enabled"] for item in items),
        "validation": status,
        "items": items,
    }
