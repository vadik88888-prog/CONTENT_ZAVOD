from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ai import sanitize_api_error
from app.config import AppConfig
from app.utils import write_json


def make_report(
    path: Path,
    source: dict[str, Any],
    metadata: dict[str, Any],
    config: AppConfig,
    state: dict[str, Any],
    selected_count: int,
    candidates_count: int,
    artifacts: list[str],
    warnings: list[str],
    errors: list[str],
    ai_usage: dict[str, Any] | None,
    gpu_used: bool,
    nvenc_used: bool,
    clip_intelligence: dict[str, Any] | None = None,
    content_transformation: dict[str, Any] | None = None,
    production_plan: dict[str, Any] | None = None,
    tts: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    production_render: dict[str, Any] | None = None,
    content_understanding: dict[str, Any] | None = None,
    virality: dict[str, Any] | None = None,
    primary_results: list[dict[str, Any]] | None = None,
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stages = state.get("stages", {})
    durations = {
        name: round(float(stage.get("duration_seconds", 0)), 3)
        for name, stage in stages.items()
    }
    total = round(sum(durations.values()), 3)
    usage = ai_usage or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    price = None
    if config.ai.input_token_price is not None or config.ai.output_token_price is not None:
        price = round(
            input_tokens * (config.ai.input_token_price or 0)
            + output_tokens * (config.ai.output_token_price or 0),
            8,
        )
    raw_errors = usage.get("api_errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = [raw_errors]
    ai = {
        "provider": str(usage.get("provider", "not-called")),
        "model": str(usage.get("model", config.ai.model)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": price,
        "retries": int(usage.get("retries", 0) or 0),
        "api_errors": [sanitize_api_error(item) for item in raw_errors],
    }
    report = {
        "source": source,
        "source_duration_seconds": metadata.get("duration"),
        "stages": stages,
        "timings_seconds": {**durations, "total": total},
        "gpu_used": gpu_used,
        "nvenc_used": nvenc_used,
        "candidates_count": candidates_count,
        "selected_clips_count": selected_count,
        "output_files": artifacts,
        "warnings": warnings,
        "errors": errors,
        "ai": ai,
        "clip_intelligence": clip_intelligence or {},
        "content_transformation": content_transformation or {"enabled": False, "status": "skipped"},
        "production_plan": production_plan or {"enabled": False, "status": "skipped"},
        "tts": tts or {"enabled": False, "status": "skipped"},
        "audio": audio or {"enabled": False, "status": "skipped"},
        "production_render": production_render or {"enabled": False, "status": "skipped"},
        "content_understanding": content_understanding or {"enabled": False, "status": "skipped"},
        "virality": virality or {"enabled": False, "status": "skipped"},
        "state_persistence": state.get("state_persistence", {"status": "saved"}),
        "primary_results": primary_results or [],
        "produced_clips_count": len(primary_results or []),
        "run": run or {},
    }
    write_json(path, report)
    return report
