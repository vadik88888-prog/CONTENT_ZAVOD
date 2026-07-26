"""Deterministic product-quality checks layered on top of technical ffprobe validation."""

from __future__ import annotations

from typing import Any


def validate_output_quality(project: Any, subtitles_enabled: bool) -> dict[str, Any]:
    """Return explainable visual/subtitle findings without inspecting user content remotely."""

    warnings: list[str] = []
    errors: list[str] = []
    clips = list(getattr(getattr(project, "timeline", None), "clips", []) or [])
    if not clips:
        errors.append("Output has no visual timeline clips.")
    fills = [clip for clip in clips if getattr(clip, "clip_type", "") == "fill"]
    if clips and len(fills) / len(clips) > 0.5:
        warnings.append("More than half of the visual timeline uses neutral fallback fills.")
    subtitles = getattr(project, "subtitle_project", None)
    if subtitles_enabled and subtitles is not None:
        if not subtitles.cues and float(getattr(project, "target_duration_seconds", 0) or 0) > 0.5:
            warnings.append("Subtitles are enabled but no timed cues were produced.")
        for cue in subtitles.cues:
            if cue.line_count > subtitles.style.max_lines:
                errors.append(f"Subtitle cue {cue.cue_id} exceeds the configured line limit.")
                break
    reframe = getattr(project, "reframe_plan", None)
    fallback = getattr(reframe, "fallback_reason", None)
    return {
        "status": "failed" if errors else "warning" if warnings else "passed",
        "warnings": warnings,
        "errors": errors,
        "reframe_fallback": fallback,
        "fallback_fill_count": len(fills),
        "visual_clip_count": len(clips),
    }
