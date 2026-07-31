"""Deterministic product-quality checks layered on top of technical ffprobe validation."""

from __future__ import annotations

from typing import Any


def validate_output_quality(project: Any, subtitles_enabled: bool) -> dict[str, Any]:
    """Return explainable visual/subtitle findings without inspecting user content remotely."""

    warnings: list[str] = []
    errors: list[str] = []
    subtitle_quality: dict[str, Any] | None = None
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
            if cue.layout_state not in {"fitted", "fallback_fitted"}:
                errors.append(f"Subtitle cue {cue.cue_id} does not have a resolved layout.")
                break
            resolved_lines = list(cue.resolved_lines or [cue.text])
            if len(resolved_lines) > subtitles.style.max_lines:
                errors.append(f"Subtitle cue {cue.cue_id} exceeds the configured line limit.")
                break
            if cue.line_count != len(resolved_lines):
                errors.append(f"Subtitle cue {cue.cue_id} has an inconsistent resolved line count.")
                break
            if cue.fallback_used:
                warnings.append(f"Subtitle cue {cue.cue_id} uses the safe fitted fallback.")
        decision = getattr(subtitles, "quality_decision", None)
        if decision is None:
            warnings.append("Subtitle Quality V2 decision is unavailable for this legacy subtitle project.")
        else:
            subtitle_quality = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else {
                "status": getattr(decision, "status", "legacy_unassessed"),
                "reason_codes": list(getattr(decision, "reason_codes", []) or []),
                "severity": getattr(decision, "severity", "warning"),
            }
            decision_status = str(getattr(decision, "status", subtitle_quality.get("status", "legacy_unassessed")))
            codes = list(getattr(decision, "reason_codes", subtitle_quality.get("reason_codes", [])) or [])
            if decision_status == "blocked":
                errors.append(
                    "Subtitle Quality V2 blocks the resolved ASS layout: "
                    + (", ".join(str(code) for code in codes) or "unresolved subtitle safety failure")
                    + "."
                )
            elif decision_status in {"passed_with_warning", "legacy_unassessed"}:
                warnings.append(
                    "Subtitle Quality V2 requires attention: "
                    + (", ".join(str(code) for code in codes) or decision_status)
                    + "."
                )
    reframe = getattr(project, "reframe_plan", None)
    fallback = getattr(reframe, "fallback_reason", None)
    composition_segments = list(getattr(reframe, "composition_segments", []) or [])
    tracking_modes: dict[str, int] = {}
    tracking_validation: list[dict[str, Any]] = []
    composition_quality: list[dict[str, Any]] = []
    for segment in composition_segments:
        mode = str(getattr(segment, "tracking_mode", "none"))
        tracking_modes[mode] = tracking_modes.get(mode, 0) + 1
        validation_status = str(getattr(segment, "tracking_validation_status", "not_applicable"))
        if mode in {"face_tracking", "person_tracking", "active_speaker_tracking", "object_tracking"}:
            if bool(getattr(segment, "static_crop_sufficient", False)):
                errors.append(f"Composition segment {segment.segment_id} tracks despite a sufficient static crop.")
            if validation_status not in {"passed", "passed_with_warning"}:
                errors.append(f"Composition segment {segment.segment_id} has unvalidated dynamic tracking.")
        if validation_status == "failed_repaired":
            warnings.append(f"Composition segment {segment.segment_id} disabled tracking and applied its safe fallback.")
        composition_status = str(getattr(segment, "composition_quality_status", "passed"))
        decision = getattr(segment, "composition_quality_decision", None)
        decision_status = str(getattr(decision, "status", "evidence_unavailable"))
        decision_codes = list(getattr(decision, "reason_codes", []) or [])
        if composition_status in {"failed", "failed_repairable"}:
            errors.append(
                f"Composition segment {segment.segment_id} is blocked by content-aware composition: "
                f"{', '.join(decision_codes) or 'unresolved composition failure'}."
            )
        elif composition_status == "passed_with_warning":
            warnings.append(f"Composition segment {segment.segment_id} has composition-quality warnings.")
        if decision_status in {"fallback", "evidence_unavailable"}:
            warnings.append(
                f"Composition segment {segment.segment_id} uses explicit {decision_status} visual evidence handling."
            )
        composition_quality.append({
            "segment_id": segment.segment_id,
            "status": composition_status,
            "reasons": list(getattr(segment, "composition_quality_reasons", []) or []),
            "diagnostics": dict(getattr(segment, "composition_diagnostics", {}) or {}),
            "decision": decision.model_dump(mode="json") if decision is not None else None,
        })
        if mode in {"safe_fallback", "scene_wide", "group_framing"} or getattr(segment, "fallback_reason", None):
            tracking_validation.append({
                "segment_id": segment.segment_id,
                "tracking_mode": mode,
                "status": validation_status,
                "reason": getattr(segment, "tracking_reason", None),
                "fallback_reason": getattr(segment, "fallback_reason", None),
            })
    return {
        "status": "failed" if errors else "warning" if warnings else "passed",
        "warnings": warnings,
        "errors": errors,
        "reframe_fallback": fallback,
        "tracking": {
            "segment_count": len(composition_segments),
            "modes": tracking_modes,
            "required_count": sum(bool(getattr(segment, "tracking_required", False)) for segment in composition_segments),
            "repaired_count": sum(
                getattr(segment, "tracking_validation_status", "") == "failed_repaired"
                for segment in composition_segments
            ),
            "decisions_requiring_attention": tracking_validation,
        },
        "composition_quality": composition_quality,
        "subtitle_quality": subtitle_quality,
        "fallback_fill_count": len(fills),
        "visual_clip_count": len(clips),
    }
