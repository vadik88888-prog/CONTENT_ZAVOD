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
    run_directory: Path,
) -> dict[str, Any]:
    root = run_directory.resolve()
    for result in results:
        candidate = Path(result.output_file).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Canonical result escapes current run directory: {candidate}") from error
    status = str(production_render.get("status") or "failed")
    data = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "source_id": str(source.get("id") or ""),
        "project_id": None,
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
        "subtitle_status": {
            "enabled": bool(production_render.get("enabled")),
            "validation": production_render.get("quality", {}).get("status") if isinstance(production_render.get("quality"), dict) else None,
        },
        "audio_mode": production_render.get("audio_mode"),
        "composition_mode": production_render.get("resolution"),
    }
    write_json(path, data)
    return data


def is_run_scoped_path(path: Path, run_directory: Path) -> bool:
    try:
        path.resolve().relative_to(run_directory.resolve())
        return True
    except (OSError, ValueError):
        return False
