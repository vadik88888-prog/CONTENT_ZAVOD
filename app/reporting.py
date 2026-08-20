from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ai import sanitize_api_error
from app.ai_cost import calculate_ai_cost_telemetry
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
    creative_preview: dict[str, Any] | None = None,
    content_understanding: dict[str, Any] | None = None,
    virality: dict[str, Any] | None = None,
    candidate_flow: dict[str, Any] | None = None,
    terminal: dict[str, Any] | None = None,
    primary_results: list[dict[str, Any]] | None = None,
    quality_gate: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    vision_ai_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stages = state.get("stages", {})
    durations = {
        name: round(float(stage.get("duration_seconds", 0)), 3)
        for name, stage in stages.items()
    }
    total = round(sum(durations.values()), 3)
    usage = dict(ai_usage or {})
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
    cache_write_input_tokens = int(usage.get("cache_write_input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    raw_errors = usage.get("api_errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = [raw_errors]
    sanitized_errors = [sanitize_api_error(item) for item in raw_errors]
    usage["api_errors"] = sanitized_errors
    sanitized_vision_usage: list[dict[str, Any]] = []
    for record in vision_ai_usage or []:
        if not isinstance(record, dict):
            continue
        safe_record = dict(record)
        record_errors = safe_record.get("api_errors", [])
        if not isinstance(record_errors, list):
            record_errors = [record_errors]
        safe_record["api_errors"] = [sanitize_api_error(item) for item in record_errors]
        sanitized_vision_usage.append(safe_record)
    ai_cost = calculate_ai_cost_telemetry(
        usage,
        sanitized_vision_usage,
        source_duration_seconds=metadata.get("duration"),
        default_provider=config.ai.provider,
        default_model=config.ai.model,
    )
    semantic_cost = ai_cost["semantic"]["cost_usd"]["total"]
    vision_cost = ai_cost["vision"]["cost_usd"]["total"]
    total_ai_cost = ai_cost["total_cost_usd"]
    model = usage.get("model") or config.ai.model
    ai = {
        "provider": str(usage.get("provider", "not-called")),
        "model": str(model),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "output_tokens": output_tokens,
        "semantic_cost_usd": semantic_cost,
        "vision_cost_usd": vision_cost,
        "calculated_cost_usd": total_ai_cost,
        # Backward-compatible Friend Beta summary; the detailed owner is ai_cost.
        "estimated_cost": total_ai_cost,
        "retries": int(usage.get("retries", 0) or 0),
        "api_errors": sanitized_errors,
    }
    for key in ("execution_state", "reason", "credential_presence", "credential_source"):
        if usage.get(key) is not None:
            ai[key] = usage[key]
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
        "ai_cost": ai_cost,
        "clip_intelligence": clip_intelligence or {},
        "content_transformation": content_transformation or {"enabled": False, "status": "skipped"},
        "production_plan": production_plan or {"enabled": False, "status": "skipped"},
        "tts": tts or {"enabled": False, "status": "skipped"},
        "audio": audio or {"enabled": False, "status": "skipped"},
        "production_render": production_render or {"enabled": False, "status": "skipped"},
        "creative_preview": creative_preview or {"enabled": False, "status": "skipped"},
        "content_understanding": content_understanding or {"enabled": False, "status": "skipped"},
        "virality": virality or {"enabled": False, "status": "skipped"},
        "candidate_flow": candidate_flow or {},
        "terminal": terminal or {"status": "completed", "error_code": None},
        "state_persistence": state.get("state_persistence", {"status": "saved"}),
        "primary_results": primary_results or [],
        "produced_clips_count": len(primary_results or []),
        "quality_gate": quality_gate,
        "run": run or {},
    }
    write_json(path, report)
    return report
