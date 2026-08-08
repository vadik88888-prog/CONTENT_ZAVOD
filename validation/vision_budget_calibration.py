"""Dry-run Goal 6B.1 vision budgets against persisted long-source evidence.

The script uses real transcript/audio/scene/StoryUnit artifacts for selection
and coverage.  A deterministic in-process provider plus synthetic frame bytes
exercise batching and the cache without external calls or billable work.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.multimodal_evidence import build_multimodal_timeline
from app.vision_intelligence import VisionGateway, _select_pass1_frames


class DryRunProvider:
    name = "goal-6b1-dry-run"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: str,
        pass_kind: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append([str(item["keyframe_id"]) for item in frames])
        observations = [
            {
                "keyframe_id": frame["keyframe_id"],
                "timestamp": frame["timestamp"],
                "scene_type": "UNKNOWN",
                "primary_subject": "scene",
                "normalized_center_x": None,
                "normalized_center_y": None,
                "visible_face_count": 0,
                "action": "unknown",
                "reaction": "unknown",
                "payoff_signal": "unknown",
                "on_screen_text": "",
                "composition_risk": "unknown",
                "confidence": 0.5,
                "missing_evidence": ["action", "composition", "payoff", "reaction", "subject", "text"],
            }
            for frame in frames
        ]
        return {"observations": observations}, {
            "input_tokens": 180 + 80 * len(frames),
            "output_tokens": 90 * len(frames),
            "request_id": f"dry-run-{len(self.calls):03d}",
        }


def _read(directory: Path, name: str) -> dict[str, Any]:
    value = json.loads((directory / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{directory / name} must contain an object")
    return value


def _coverage(
    timeline: dict[str, Any],
    content_map: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamps = [float(item["timestamp"]) for item in selected]
    scene_ids = sorted({
        str(scene["scene_id"])
        for scene in timeline["scenes"]
        if any(float(scene["start_seconds"]) <= timestamp < float(scene["end_seconds"]) for timestamp in timestamps)
    })
    story_ids = sorted({
        str(story["story_unit_id"])
        for story in content_map["story_units"]
        if any(float(story["start"]) <= timestamp < float(story["end"]) for timestamp in timestamps)
    })
    duration = max(0.001, float(timeline["source_duration_seconds"]))
    deciles = sorted({min(9, int(timestamp / duration * 10)) for timestamp in timestamps})
    return {
        "scene_count": len(scene_ids),
        "scene_total": len(timeline["scenes"]),
        "scene_ids": scene_ids,
        "story_unit_count": len(story_ids),
        "story_unit_total": len(content_map["story_units"]),
        "story_unit_ids": story_ids,
        "temporal_deciles": deciles,
    }


def run_case(label: str, content_type: str, work_directory: Path, source: Path) -> dict[str, Any]:
    transcript = _read(work_directory, "transcript.json")
    metadata = _read(work_directory, "metadata.json")
    content_map = _read(work_directory, "global_content_map.json")
    timeline = build_multimodal_timeline(
        source_id=str(transcript["source_id"]),
        source_duration_seconds=float(metadata["duration"]),
        transcript=transcript,
        audio_features=_read(work_directory, "audio_features.json"),
        scenes=_read(work_directory, "scene_boundaries.json"),
        visual_analysis=_read(work_directory, "visual_analysis.json"),
    )
    result: dict[str, Any] = {
        "label": label,
        "content_type": content_type,
        "source_path": str(source),
        "source_exists": source.is_file(),
        "source_id": transcript["source_id"],
        "duration_seconds": timeline["source_duration_seconds"],
        "found_keyframes": [
            {"keyframe_id": item["keyframe_id"], "timestamp": item["time_seconds"]}
            for item in timeline["keyframes"]
        ],
        "scene_total": len(timeline["scenes"]),
        "story_unit_total": len(content_map["story_units"]),
        "modes": {},
    }
    with tempfile.TemporaryDirectory(prefix="vision-6b1-") as temporary:
        for mode in ("fast", "standard", "maximum"):
            config = AppConfig(optional_visual_features=True)
            config.product_flow.processing_mode = mode
            provider = DryRunProvider()
            gateway = VisionGateway(
                config=config,
                cache_directory=Path(temporary) / mode,
                provider=provider,
                frame_loader=lambda _source, timestamp, _width: f"dry-frame:{timestamp:.3f}".encode(),
            )
            first = gateway.analyze_pass1(source=source, timeline=timeline, content_type=content_type)
            first_call_count = len(provider.calls)
            second = gateway.analyze_pass1(source=source, timeline=timeline, content_type=content_type)
            selected = list(first["diagnostics"]["selected_keyframes"])
            configured_cap = int(first["diagnostics"]["budget"]["configured_frame_limit"])
            legacy_selected = _select_pass1_frames(timeline, configured_cap)
            result["modes"][mode] = {
                "budget": first["diagnostics"]["budget"],
                "selected_keyframes": selected,
                "coverage": _coverage(timeline, content_map, selected),
                "legacy_fixed_cap_comparison": {
                    "cap": configured_cap,
                    "selected_count": len(legacy_selected),
                    "coverage": _coverage(timeline, content_map, legacy_selected),
                },
                "projected_uncached_usage": first["diagnostics"]["projected_uncached_usage"],
                "first_run_cache": {
                    "hits": first["diagnostics"]["cache_hits"],
                    "misses": first["diagnostics"]["cache_misses"],
                    "provider_calls": first_call_count,
                },
                "second_run_cache": {
                    "hits": second["diagnostics"]["cache_hits"],
                    "misses": second["diagnostics"]["cache_misses"],
                    "provider_calls": len(provider.calls) - first_call_count,
                },
                "analysis_stop_reason": first["diagnostics"]["analysis_stop_reason"],
                "second_run_stop_reason": second["diagnostics"]["analysis_stop_reason"],
                "external_provider_calls": 0,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", action="append", nargs=4, metavar=("LABEL", "CONTENT_TYPE", "WORK_DIRECTORY", "SOURCE"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": "6B.1-budget-calibration.1",
        "dry_run": True,
        "external_provider_calls": 0,
        "cases": [
            run_case(label, content_type, Path(work_directory), Path(source))
            for label, content_type, work_directory, source in args.case
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
