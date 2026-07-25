"""Render source-format fixtures against one existing valid plan/audio pair."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.audio_models import AudioProject
from app.config import load_config
from app.production_models import ProductionPlan
from app.sources import local_source
from app.utils import read_json
from app.video_composition import VideoCompositionService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate source resolutions and frame rates in isolated production render.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--audio-project", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parent / "artifacts" / "source-formats")
    parser.add_argument("--metrics", type=Path, default=Path(__file__).parent / "metrics" / "source-format-render.json")
    arguments = parser.parse_args(argv)
    root = Path.cwd()
    config = load_config(arguments.config.resolve())
    plan = ProductionPlan.model_validate(read_json(arguments.plan.resolve(), {}))
    audio = AudioProject.model_validate(read_json(arguments.audio_project.resolve(), {}))
    transcript = read_json(arguments.transcript.resolve(), {})
    if not isinstance(transcript, dict):
        raise SystemExit("Transcript must be a JSON object")
    service = VideoCompositionService(root, config)
    runs = []
    for path in arguments.source:
        source = local_source(str(path))
        target = arguments.output_directory / source.display_name
        project = service.compose(plan, audio, source, transcript, target / "work", target / "output")
        result = project.result
        runs.append({
            "source": str(source.path), "source_id": source.id,
            "status": result.status if result else "missing_result",
            "validation": result.validation.status if result else "missing_result",
            "duration_seconds": project.actual_duration_seconds,
            "sync_difference_ms": result.validation.sync_difference_ms if result else None,
            "output_file": result.output_file if result else None,
            "cache_hit": bool(result and result.cache_hit),
            "fallback_reasons": project.fallback_reasons,
        })
        print(source.path)
    report = {
        "schema_version": "3E.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "isolated_production_render_source_format", "run_count": len(runs), "runs": runs,
        "limitations": [
            "This intentionally reuses one already validated transcript, ProductionPlan and AudioProject.",
            "It validates the source decoder, crop/timeline mapping, subtitles, muxing and A/V sync—not transcript semantics.",
        ],
    }
    arguments.metrics.parent.mkdir(parents=True, exist_ok=True)
    arguments.metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(run["status"] in {"completed", "warning"} and run["validation"] != "invalid" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
