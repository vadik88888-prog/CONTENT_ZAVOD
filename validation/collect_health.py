"""Aggregate safe local report.json evidence into a Goal 3E health report.

This is validation tooling, not a pipeline stage: it never creates media, calls
providers, or alters source/output/work artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_GROUPS = {
    "analysis": {"metadata", "transcription", "transcript_features", "audio_features", "scene_detection", "candidates_v2", "local_scoring", "shortlist", "ai_ranking", "final_selection"},
    "transformation": {"transformation_source_context", "transformation_semantic_representation", "transformation_narrative_plan", "transformation_script_draft", "transformation_script_validation", "transformation_final_script", "transformation_result"},
    "production_plan": {"production_plan"},
    "tts": {"tts_generation"},
    "audio": {"audio_composition"},
    "video": {"production_render", "render"},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate local Content Factory validation reports.")
    parser.add_argument("--report", action="append", required=True, type=Path, help="Path to report.json; repeat for every run.")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parent / "metrics")
    arguments = parser.parse_args(argv)
    reports = [_read_report(path) for path in arguments.report]
    health = _health(reports)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    json_path = arguments.output_directory / "production-health.json"
    markdown_path = arguments.output_directory / "production-health.md"
    json_path.write_text(json.dumps(health, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(health), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Validation report not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Validation report is not JSON: {path}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"Validation report must be an object: {path}")
    return {"path": str(path.resolve()), "report": value}


def _health(items: list[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    totals = {name: 0.0 for name in STAGE_GROUPS}
    estimated_costs = {"ai": 0.0, "tts": 0.0, "other": 0.0}
    confirmed_actual_costs = {"tts": 0.0}
    cache = {"production_render_hits": 0, "production_render_misses": 0, "tts_hits": 0, "audio_dialogue_hits": 0, "audio_dialogue_misses": 0}
    valid_runs = 0
    for item in items:
        report = item["report"]
        timings = report.get("timings_seconds", {}) if isinstance(report.get("timings_seconds"), dict) else {}
        grouped = {
            group: round(sum(
                float(value or 0) for name, value in timings.items()
                if any(name == stage or name.startswith(stage + ":") for stage in stages)
            ), 3)
            for group, stages in STAGE_GROUPS.items()
        }
        for group, value in grouped.items():
            totals[group] += value
        ai = report.get("ai", {}) if isinstance(report.get("ai"), dict) else {}
        tts = report.get("tts", {}) if isinstance(report.get("tts"), dict) else {}
        audio = report.get("audio", {}) if isinstance(report.get("audio"), dict) else {}
        video = report.get("production_render", {}) if isinstance(report.get("production_render"), dict) else {}
        estimated_costs["ai"] += float(ai.get("estimated_cost", 0) or 0)
        estimated_costs["tts"] += float(tts.get("estimated_cost", 0) or 0)
        actual_tts_cost = tts.get("actual_cost")
        if isinstance(actual_tts_cost, (int, float)):
            confirmed_actual_costs["tts"] += float(actual_tts_cost)
        cache["production_render_hits"] += int(bool(video.get("cache_hit", False)))
        cache["production_render_misses"] += int(bool(video.get("enabled", False) and not video.get("cache_hit", False)))
        cache["tts_hits"] += int(tts.get("cache_hit_count", 0) or 0)
        audio_cache = audio.get("cache", {}) if isinstance(audio.get("cache"), dict) else {}
        cache["audio_dialogue_hits"] += int(audio_cache.get("dialogue_hit_count", 0) or 0)
        cache["audio_dialogue_misses"] += int(audio_cache.get("dialogue_miss_count", 0) or 0)
        healthy = video.get("validation") == "valid" and video.get("status") in {"completed", "warning"}
        valid_runs += int(healthy)
        runs.append({
            "report": item["path"], "source": report.get("source", {}).get("display_name"),
            "timings_seconds": grouped, "total_seconds": float(timings.get("total", 0) or 0),
            "ai": {"provider": ai.get("provider"), "estimated_cost": float(ai.get("estimated_cost", 0) or 0)},
            "tts": {
                "provider": tts.get("provider"),
                "estimated_cost": float(tts.get("estimated_cost", 0) or 0),
                "actual_cost": actual_tts_cost if isinstance(actual_tts_cost, (int, float)) else None,
            },
            "production_render": {"status": video.get("status"), "validation": video.get("validation"), "cache_hit": video.get("cache_hit")},
            "health_pass": healthy,
        })
    count = len(runs)
    average = {name: round(value / count, 3) if count else 0.0 for name, value in totals.items()}
    average_total = round(sum(item["total_seconds"] for item in runs) / count, 3) if count else 0.0
    estimated_total_cost = round(sum(estimated_costs.values()), 8)
    confirmed_actual_total = round(sum(confirmed_actual_costs.values()), 8)
    score = round(100 * valid_runs / count, 1) if count else 0.0
    return {
        "schema_version": "3E.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "run_count": count, "production_health_score": score,
        "pipeline_stability": {"valid_runs": valid_runs, "failed_or_unvalidated_runs": count - valid_runs},
        "average_runtime_seconds": {**average, "total": average_total},
        "cost_usd": {
            "estimated": {
                **{key: round(value, 8) for key, value in estimated_costs.items()},
                "total": estimated_total_cost,
                "average_per_short": round(estimated_total_cost / count, 8) if count else 0.0,
            },
            "confirmed_actual": {
                **{key: round(value, 8) for key, value in confirmed_actual_costs.items()},
                "total": confirmed_actual_total,
            },
        },
        "cache_metrics": cache, "runs": runs,
        "known_limitations": [
            "Metrics are only as representative as the supplied local validation reports.",
            "Synthetic fixtures validate technical stability, not semantic/content quality for licensed-real categories.",
            "Estimated costs are not invoices. Confirmed actual cost only includes explicit actual_cost values in supplied reports.",
        ],
    }


def _markdown(health: dict[str, Any]) -> str:
    average = health["average_runtime_seconds"]
    cache = health["cache_metrics"]
    return "\n".join([
        "# Production Health (Goal 3E)", "",
        f"- Runs: {health['run_count']}", f"- Health score: {health['production_health_score']:.1f}/100",
        f"- Valid production renders: {health['pipeline_stability']['valid_runs']}",
        f"- Average total runtime: {average['total']:.3f} s", f"- Estimated average cost per short: ${health['cost_usd']['estimated']['average_per_short']:.8f}",
        f"- Confirmed actual cost: ${health['cost_usd']['confirmed_actual']['total']:.8f}",
        f"- Production render cache: {cache['production_render_hits']} hits / {cache['production_render_misses']} misses", "",
        "## Average stage runtime (s)", "",
        *[f"- {name}: {value:.3f}" for name, value in average.items() if name != "total"], "",
        "## Known limitations", "",
        *[f"- {item}" for item in health["known_limitations"]], "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
