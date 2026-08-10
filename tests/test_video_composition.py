from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.audio_service import AudioCompositionService
from app.cli import build_parser
from app.creative_contracts import (
    AttentionTarget,
    BeatRole,
    CanvasPlan,
    CompiledRenderPlan,
    CreativeIntent,
    CreativePolicy,
    EvidenceItem,
    Intensity,
    ImmutableProductionPlanLink,
    LayoutFamily,
    MotionDomain,
    MotionPurpose,
    NormalizedRect,
    OutputInterval,
    RenderParityManifest,
    ResolvedBeat,
    ResolvedCompositionTarget,
    ResolvedEmphasis,
    ResolvedMotionEvent,
    ResolvedSourceBRoll,
    SemanticClass,
    SourceBRollSemanticKind,
    SourceInterval,
    assert_preview_final_parity,
    compile_render_plan,
    source_output_map_from_legacy_timeline,
)
from app.composition_planning import TargetObservation
from app.errors import ProductionRenderError
from app.motion_planning import build_motion_plan
from app.pipeline import Pipeline, StageTracker
from app.production_subtitles import build_subtitle_project, resolve_subtitle_style, split_subtitle_text, write_production_ass
from app.sources import local_source
from app.source_broll_planning import SourceSceneEvidence
from app.video_composition import (
    VideoCompositionService,
    apply_composition_segments,
    build_composition_segments,
    build_reframe_plan,
    build_video_timeline,
    make_crop_plan,
    probe_media,
    production_render_report_section,
    _ffmpeg,
    _native_motion_filter,
    _source_range_for_audio,
    _timeline_filter,
    _split_timeline_at_scene_boundaries,
    _validate_tracking_decisions,
    _visual_filter,
)
from app.production_models import SourceSegmentRange
from app.video_models import (
    CanvasConfig, CompositionSegment, CropPlan, ReframeKeyframe, RenderValidation,
    SourceVideoClip, SubjectBounds, VideoTimeline,
)
from tests.test_audio_composition import _audio_config, _plan, _tts_result


def _source_video(path: Path, *, width: int = 320, height: int = 180) -> Path:
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("ffmpeg is required for production video composition tests")
    subprocess.run([
        executable, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"testsrc2=size={width}x{height}:rate=30", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ], check=True)
    return path


def _upstream(tmp_path: Path):
    config = _audio_config()
    config.production_render.enabled = True
    config.production_render.output_width = 180
    config.production_render.output_height = 320
    config.production_render.output_fps = 30
    config.production_render.video_bitrate = "500k"
    config.production_render.encoder = "cpu"
    config.validate()
    plan = _plan()
    source_path = _source_video(tmp_path / "source.mp4")
    source = local_source(str(source_path))
    transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Source dialogue remains audible."}]}
    audio = AudioCompositionService(tmp_path, config).compose(
        plan, source, transcript, _tts_result(tmp_path, config, plan), tmp_path / "work", tmp_path / "out",
    )
    return config, plan, source, transcript, audio


def _native_intent(plan, mapping) -> CreativeIntent:
    story = EvidenceItem(
        evidence_ref="story-native", evidence_kind="story_unit",
        source=SourceInterval.from_seconds(0.2, 0.8), confidence=0.96,
        artifact_fingerprint="1" * 64, provenance="regression:story",
    )
    visual = EvidenceItem(
        evidence_ref="visual-native", evidence_kind="visual",
        source=SourceInterval.from_seconds(0.2, 0.8), confidence=0.95,
        artifact_fingerprint="2" * 64, provenance="regression:target",
    )
    scene = EvidenceItem(
        evidence_ref="scene-native", evidence_kind="scene",
        source=SourceInterval.from_seconds(2.0, 2.6), confidence=0.94,
        artifact_fingerprint="3" * 64, provenance="regression:cutaway",
    )
    interval = OutputInterval(start_frame=6, end_frame=24)
    source = SourceInterval.from_seconds(0.2, 0.8)
    return CreativeIntent(
        intent_id="intent-native-regression",
        revision=1,
        production_plan=ImmutableProductionPlanLink.from_reference(plan.reference()),
        source_output_mapping=mapping,
        evidence_fingerprint="4" * 64,
        evidence_manifest=(story, visual, scene),
        proposal_hash="5" * 64,
        policy=CreativePolicy(
            preset_id="documentary", preset_version="1", platform="universal",
            caption_style_family="emphasis", intensity=Intensity.BALANCED,
            source_broll_enabled=True,
        ),
        confidence=0.94,
        provenance=("e2e-regression",),
        beats=(ResolvedBeat(
            decision_id="beat-native", source=source, output=interval,
            confidence=0.95, evidence_refs=("story-native",),
            role=BeatRole.ACTION, importance=0.9,
        ),),
        semantic_emphasis=(ResolvedEmphasis(
            decision_id="emphasis-native", source=source, output=interval,
            confidence=0.94, evidence_refs=("story-native",),
            text_span="Source", semantic_class=SemanticClass.ACTION, importance=0.9,
        ),),
        composition_targets=(ResolvedCompositionTarget(
            decision_id="target-native", source=source, output=interval,
            confidence=0.95, evidence_refs=("visual-native",),
            target=AttentionTarget.SUBJECT, target_ref="subject-native", priority=90,
            allowed_layouts=(LayoutFamily.SINGLE_SUBJECT,),
        ),),
        motion_events=(
            ResolvedMotionEvent(
                decision_id="motion-caption", source=SourceInterval.from_seconds(0.2, 0.4),
                output=OutputInterval(start_frame=6, end_frame=12),
                confidence=0.93, evidence_refs=("story-native",),
                purpose=MotionPurpose.HOOK, domain=MotionDomain.CAPTION,
                intensity=Intensity.BALANCED,
            ),
            ResolvedMotionEvent(
                decision_id="motion-composition", source=source, output=interval,
                confidence=0.94, evidence_refs=("visual-native",),
                purpose=MotionPurpose.EVIDENCE_REVEAL, domain=MotionDomain.COMPOSITION,
                intensity=Intensity.BALANCED,
            ),
        ),
        source_broll=(ResolvedSourceBRoll(
            decision_id="broll-native", source=source, output=interval,
            confidence=0.93, evidence_refs=("story-native",),
            source_cutaway=SourceInterval.from_seconds(2.0, 2.6),
            source_cutaway_evidence_refs=("scene-native",),
            story_unit_id="story-native", story_unit_evidence_ref="story-native",
            semantic_kind=SourceBRollSemanticKind.ACTION,
        ),),
    )


def test_video_models_validate_canvas_crop_and_timeline_ranges() -> None:
    canvas = CanvasConfig(width=180, height=320, fps=30)
    crop = CropPlan(strategy="center_crop", source_width=320, source_height=180, crop_width=100, crop_height=180, crop_x=110, crop_y=0)
    clip = SourceVideoClip(
        clip_id="clip-001", order=1, timeline_start_seconds=0, timeline_end_seconds=1,
        duration_seconds=1, source_path="source.mp4", source_start_seconds=0, source_end_seconds=1,
        visual_strategy="mapped_source", crop_plan=crop, status="ready",
    )
    assert VideoTimeline(clips=[clip], duration_seconds=1).duration_seconds == 1
    with pytest.raises(ValidationError):
        CanvasConfig(width=181, height=320, fps=30)
    with pytest.raises(ValidationError):
        CropPlan(strategy="center_crop", source_width=100, source_height=100, crop_width=80, crop_height=80, crop_x=30, crop_y=0)
    with pytest.raises(ValidationError):
        SourceVideoClip(**{**clip.model_dump(), "timeline_end_seconds": 0.5})


def test_narration_visual_mapping_prefers_validated_plan_range_over_full_transcript_segment() -> None:
    narration = next(segment for segment in _plan(narration=True, dialogue=False).segments if segment.segment_type == "narration")
    narration.source_ranges = [SourceSegmentRange(
        transcript_segment_id=0, source_start_seconds=0.5, source_end_seconds=1.5,
    )]

    assert _source_range_for_audio(None, narration, {0: (0.0, 3.0)})[:2] == (0.5, 1.5)


def test_missing_ffmpeg_and_ffprobe_fail_with_clear_render_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.video_composition.shutil.which", lambda _name: None)
    with pytest.raises(ProductionRenderError, match="ffprobe"):
        probe_media(Path("source.mp4"), require_video=True)
    with pytest.raises(ProductionRenderError, match="ffmpeg"):
        _ffmpeg()


def test_timeline_maps_actual_audio_dialogue_and_uses_deterministic_pause_fallback(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    timeline, fallbacks = build_video_timeline(
        plan, audio, transcript, source.path, probe_media(source.path, require_video=True),
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    dialogue = next(clip for clip in timeline.clips if clip.production_segment_id == "dialogue-001")
    assert dialogue.source_start_seconds == 1 and dialogue.source_end_seconds == 2
    assert timeline.duration_seconds == audio.timeline.duration_seconds
    assert fallbacks and all("silence" in reason for reason in fallbacks)
    assert [clip.order for clip in timeline.clips] == list(range(1, len(timeline.clips) + 1))


def test_prepared_source_clip_keeps_freeze_padding_duration(tmp_path: Path) -> None:
    config, _plan_value, source, _transcript, _audio = _upstream(tmp_path)
    canvas = CanvasConfig(width=180, height=320, fps=30)
    clip = SourceVideoClip(
        clip_id="visual-freeze", order=1, timeline_start_seconds=0, timeline_end_seconds=2,
        duration_seconds=2, source_path=str(source.path), source_start_seconds=0,
        source_end_seconds=1.4, visual_strategy="mapped_source",
        crop_plan=make_crop_plan(probe_media(source.path, require_video=True), canvas, config.production_render),
        freeze_duration_seconds=0.6, status="fallback", fallback_reason="source_clip_short_freeze",
    )
    destination = tmp_path / "freeze-padded.mp4"

    VideoCompositionService(tmp_path, config)._prepare_visual_clip(clip, canvas, destination)

    actual = float(probe_media(destination, require_video=True)["video_duration"])
    assert abs(actual - clip.duration_seconds) <= 0.15


def test_fit_background_filters_center_the_foreground_instead_of_bottom_aligning_it() -> None:
    canvas = CanvasConfig(width=180, height=320, fps=30)
    crop = CropPlan(strategy="fit_blur_background", source_width=320, source_height=180)
    clip = SourceVideoClip(
        clip_id="visual-centered-fit", order=1, timeline_start_seconds=0, timeline_end_seconds=1,
        duration_seconds=1, source_path="source.mp4", source_start_seconds=0, source_end_seconds=1,
        visual_strategy="mapped_source", crop_plan=crop, status="ready",
    )

    graph = _visual_filter(clip, canvas)

    assert "overlay=(W-w)/2:(H-h)/2" in graph


def test_missing_or_invalid_mapping_is_reported_without_random_clip_selection(tmp_path: Path) -> None:
    config, plan, source, _transcript, audio = _upstream(tmp_path)
    raw = plan.model_dump(mode="json")
    raw["segments"][1]["source_end_seconds"] = 99
    raw["dialogue_mappings"][0]["source_end_seconds"] = 99
    broken = type(plan).model_validate(raw)
    first, reasons_a = build_video_timeline(
        broken, audio, {}, source.path, probe_media(source.path, require_video=True),
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    second, reasons_b = build_video_timeline(
        broken, audio, {}, source.path, probe_media(source.path, require_video=True),
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    assert reasons_a == reasons_b and reasons_a
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert all(clip.status in {"ready", "fallback", "placeholder"} for clip in first.clips)


@pytest.mark.parametrize(("width", "height", "strategy"), [
    (320, 180, "center_crop"), (180, 320, "top_crop"), (240, 240, "fit_blur_background"),
])
def test_crop_plans_stay_inside_horizontal_vertical_and_square_sources(width: int, height: int, strategy: str) -> None:
    from app.config import ProductionRenderConfig

    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy=strategy)
    config.validate()
    plan = make_crop_plan(
        {"display_width": width, "display_height": height, "rotation": 0}, CanvasConfig(width=180, height=320, fps=30), config,
    )
    if plan.crop_width is not None:
        assert plan.crop_x + plan.crop_width <= width
        assert plan.crop_y + plan.crop_height <= height


def test_reframe_plan_uses_safe_fallback_and_smooths_subject_keyframes() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="center_crop")
    fallback = build_reframe_plan(
        {"display_width": 320, "display_height": 180}, canvas, config, VideoTimeline(clips=[], duration_seconds=0),
    )
    assert fallback.strategy == "center_crop"
    assert fallback.subject_detection_used is False
    manual_with_subject = build_reframe_plan(
        {
            "display_width": 320, "display_height": 180,
            "subject_keyframes": [
                {"time_seconds": 0, "normalized_x": 0.15, "normalized_y": 0.3, "confidence": 0.9},
                {"time_seconds": 1, "normalized_x": 0.95, "normalized_y": 0.9, "confidence": 0.9},
            ],
        }, canvas, config, VideoTimeline(clips=[], duration_seconds=0),
    )
    assert manual_with_subject.strategy == "center_crop"
    assert manual_with_subject.subject_detection_used is False
    auto = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    tracked = build_reframe_plan(
        {
            "display_width": 320, "display_height": 180,
            "subject_keyframes": [
                {"time_seconds": 0, "normalized_x": 0.15, "normalized_y": 0.3, "confidence": 0.9},
                {"time_seconds": 1, "normalized_x": 0.95, "normalized_y": 0.9, "confidence": 0.9},
            ],
        }, canvas, auto, VideoTimeline(clips=[], duration_seconds=0),
    )
    assert tracked.strategy == "subject_crop" and tracked.subject_detection_used
    assert tracked.keyframes[1].normalized_x - tracked.keyframes[0].normalized_x <= 0.16
    assert tracked.keyframes[1].normalized_y - tracked.keyframes[0].normalized_y <= 0.12


def test_safe_auto_reframe_preserves_unknown_landscape_and_uses_subject_when_confident() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {"display_width": 1920, "display_height": 1080, "rotation": 0}
    crop = make_crop_plan(source, canvas, config)
    reframe = build_reframe_plan(source, canvas, config, VideoTimeline(clips=[], duration_seconds=0))

    assert crop.strategy == "fit_blur_background"
    assert reframe.strategy == "blur_fallback"
    assert reframe.fallback_reason and "unsafe crop" in reframe.fallback_reason

    tracked_source = {
        **source,
        "subject_keyframes": [{"time_seconds": 0, "normalized_x": 0.8, "normalized_y": 0.4, "confidence": 0.9}],
    }
    tracked_crop = make_crop_plan(tracked_source, canvas, config)
    tracked_reframe = build_reframe_plan(tracked_source, canvas, config, VideoTimeline(clips=[], duration_seconds=0))
    assert tracked_crop.strategy == "manual_normalized_crop"
    assert tracked_reframe.strategy == "subject_crop" and tracked_reframe.subject_detection_used


def test_subject_anchor_changes_the_actual_horizontal_crop() -> None:
    from app.config import ProductionRenderConfig

    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    canvas = CanvasConfig(width=180, height=320, fps=30)
    crop = make_crop_plan(
        {"display_width": 1280, "display_height": 720, "rotation": 0, "subject_keyframes": [{"normalized_x": 0.8, "normalized_y": 0.4, "confidence": 0.9}]},
        canvas, config,
    )
    assert crop.strategy == "manual_normalized_crop"
    assert crop.crop_x is not None and crop.crop_x > (1280 - (720 * 9 / 16)) / 2


def _tracking_timeline(source_path: str, crop: CropPlan, *, duration: float = 3, speaker: str | None = "speaker-a") -> VideoTimeline:
    clip = SourceVideoClip(
        clip_id="tracking-source", order=1, timeline_start_seconds=0, timeline_end_seconds=duration,
        duration_seconds=duration, source_path=source_path, source_start_seconds=0, source_end_seconds=duration,
        production_segment_id="dialogue-tracking", speaker=speaker, visual_strategy="mapped_source",
        crop_plan=crop, status="ready",
    )
    return VideoTimeline(clips=[clip], duration_seconds=duration)


def test_composition_segments_prefer_static_crop_then_enable_safe_tracking() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {"display_width": 320, "display_height": 180, "rotation": 0}
    base_crop = make_crop_plan(source, canvas, config)
    static_timeline = _tracking_timeline("source.mp4", base_crop)
    static_source = {
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.48, "normalized_y": 0.38, "confidence": 0.9, "tracking_target": "primary_face"},
            {"time_seconds": 1.5, "normalized_x": 0.51, "normalized_y": 0.40, "confidence": 0.92, "tracking_target": "primary_face"},
            {"time_seconds": 3, "normalized_x": 0.53, "normalized_y": 0.39, "confidence": 0.91, "tracking_target": "primary_face"},
        ],
    }
    static = build_composition_segments(static_source, canvas, config, static_timeline)[0]
    assert static.tracking_mode == "static_subject_crop"
    assert static.static_crop_sufficient and not static.tracking_required

    moving_source = {
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.28, "normalized_y": 0.40, "confidence": 0.92, "tracking_target": "primary_face"},
            {"time_seconds": 1.5, "normalized_x": 0.45, "normalized_y": 0.41, "confidence": 0.92, "tracking_target": "primary_face"},
            {"time_seconds": 3, "normalized_x": 0.62, "normalized_y": 0.40, "confidence": 0.92, "tracking_target": "primary_face"},
        ],
    }
    plan = build_reframe_plan(moving_source, canvas, config, static_timeline)
    moving = plan.composition_segments[0]
    assert moving.tracking_mode == "face_tracking"
    assert moving.tracking_required and not moving.static_crop_sufficient
    assert moving.tracking_validation_status == "passed"
    assert moving.tracking_diagnostics["target_visible"]
    assert moving.tracking_diagnostics["target_in_safe_zone"]
    assert moving.target_crop and len(moving.target_crop.tracking_keyframes) >= 2
    applied = apply_composition_segments(static_timeline, plan.composition_segments)
    rendered_crop = applied.clips[0].crop_plan
    assert rendered_crop and rendered_crop.tracking_keyframes
    assert "if(lt(t\\," in _visual_filter(applied.clips[0], canvas)


def test_tracking_quality_validator_disables_a_crop_that_loses_its_target() -> None:
    canvas = CanvasConfig(width=180, height=320, fps=30)
    unsafe = CompositionSegment(
        segment_id="unsafe-tracker", visual_clip_id="source", start_seconds=0, end_seconds=1,
        source_start_seconds=0, source_end_seconds=1, strategy="face_crop",
        subject_bounds=[
            SubjectBounds(time_seconds=0, center_x=0.80, center_y=0.40, width=0.16, height=0.20, confidence=0.95, target="primary_face"),
            SubjectBounds(time_seconds=1, center_x=0.82, center_y=0.40, width=0.16, height=0.20, confidence=0.95, target="primary_face"),
        ],
        target_crop=CropPlan(
            strategy="manual_normalized_crop", source_width=320, source_height=180,
            crop_width=100, crop_height=180, crop_x=0, crop_y=0,
            tracking_keyframes=[
                ReframeKeyframe(time_seconds=0, normalized_x=0.10, normalized_y=0.40, confidence=0.95),
                ReframeKeyframe(time_seconds=1, normalized_x=0.10, normalized_y=0.40, confidence=0.95),
            ],
        ),
        confidence=0.95, tracking_mode="face_tracking", tracking_target="primary_face",
        tracking_required=True, tracking_confidence=0.95,
        tracking_reason="test", static_crop_sufficient=False, tracking_risk="low",
    )
    repaired = _validate_tracking_decisions([unsafe], canvas)[0]
    assert repaired.tracking_mode == "safe_fallback"
    assert repaired.tracking_validation_status == "failed_repaired"
    assert repaired.tracking_diagnostics["target_visible"] is False
    assert "fully visible" in repaired.tracking_reason


def test_tracking_engine_uses_group_or_safe_fallback_for_risky_scenes() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {"display_width": 320, "display_height": 180, "rotation": 0}
    timeline = _tracking_timeline("source.mp4", make_crop_plan(source, canvas, config), duration=1)
    group = build_composition_segments({
        **source,
        "composition_intent": {"schema_version": "6D.composition-intent.1", "multiple_subjects": {"value": True}},
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.45, "normalized_y": 0.4, "confidence": 0.9, "visible_face_count": 2, "active_speaker_confidence": 0.95},
            {"time_seconds": 1, "normalized_x": 0.55, "normalized_y": 0.4, "confidence": 0.9, "visible_face_count": 2, "active_speaker_confidence": 0.95},
        ],
    }, canvas, config, timeline)[0]
    assert group.tracking_mode == "group_framing"
    assert group.editorial_intent["multiple_subjects"]["value"] is True
    assert not group.tracking_required
    assert group.minimum_focus_hold_seconds == pytest.approx(1.25)

    unstable = build_reframe_plan({
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.15, "normalized_y": 0.4, "confidence": 0.95},
            {"time_seconds": 1, "normalized_x": 0.90, "normalized_y": 0.4, "confidence": 0.95},
        ],
    }, canvas, config, timeline).composition_segments[0]
    assert unstable.tracking_mode == "safe_fallback"
    assert unstable.fallback_strategy == "fit_with_blur"
    assert "below the safe threshold" in unstable.tracking_reason

    scene_changed = build_composition_segments({
        **source,
        "scene_boundaries": [{"timestamp": 0.5}],
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.28, "normalized_y": 0.4, "confidence": 0.95},
            {"time_seconds": 1, "normalized_x": 0.48, "normalized_y": 0.4, "confidence": 0.95},
        ],
    }, canvas, config, timeline)[0]
    assert scene_changed.tracking_mode == "scene_wide"
    assert scene_changed.scene_change_count == 1
    assert scene_changed.transition_type == "cut"

    tiny_face = build_composition_segments({
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.25, "normalized_y": 0.4, "normalized_width": 0.05, "normalized_height": 0.05, "confidence": 0.95, "tracking_target": "primary_face"},
            {"time_seconds": 1, "normalized_x": 0.45, "normalized_y": 0.4, "normalized_width": 0.05, "normalized_height": 0.05, "confidence": 0.95, "tracking_target": "primary_face"},
        ],
    }, canvas, config, timeline)[0]
    assert tiny_face.tracking_mode == "safe_fallback"
    assert "too small" in tiny_face.tracking_reason

    screen = build_composition_segments({
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.5, "normalized_y": 0.5, "confidence": 0.95, "tracking_target": "screen_region"},
        ],
    }, canvas, config, timeline)[0]
    assert screen.tracking_mode == "scene_wide"
    assert screen.tracking_target == "screen_region"


def test_content_aware_talking_head_persists_chest_up_quality_decision() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {
        "display_width": 320, "display_height": 180, "rotation": 0,
        "visual_evidence_status": "valid",
        "subject_keyframes": [
            {
                "time_seconds": 0, "normalized_x": 0.48, "normalized_y": 0.42,
                "normalized_width": 0.22, "normalized_height": 0.46, "confidence": 0.93,
                "tracking_target": "primary_face", "visible_face_count": 1,
                "scene_type": "TALKING_HEAD", "framing_observation": "chest_up",
                "eye_line_y": 0.34, "gesture_active": True, "gesture_area_visible": True,
            },
            {
                "time_seconds": 3, "normalized_x": 0.50, "normalized_y": 0.42,
                "normalized_width": 0.22, "normalized_height": 0.46, "confidence": 0.94,
                "tracking_target": "primary_face", "visible_face_count": 1,
                "scene_type": "TALKING_HEAD", "framing_observation": "chest_up",
                "eye_line_y": 0.34, "gesture_active": True, "gesture_area_visible": True,
            },
        ],
    }
    plan = build_reframe_plan(
        source, canvas, config, _tracking_timeline("source.mp4", make_crop_plan(source, canvas, config)),
    )
    segment = plan.composition_segments[0]
    decision = segment.composition_quality_decision
    assert segment.strategy == "subject_crop"
    assert decision.schema_version == "5D.0"
    assert decision.status == "passed"
    assert decision.scene_type == "TALKING_HEAD"
    assert decision.framing_intent == "CHEST_UP_PERSON"
    assert decision.selected_target == "primary_face"
    assert decision.evidence_status == "valid"
    assert {"face_visibility", "chest_shoulder_framing", "headroom_ratio", "gesture_area_visibility", "crop_stability"} <= decision.metrics.keys()


def test_content_aware_product_and_screen_decisions_protect_content_targets() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    product_source = {
        "display_width": 320, "display_height": 180, "rotation": 0, "visual_evidence_status": "valid",
        "subject_keyframes": [{
            "time_seconds": 1, "normalized_x": 0.50, "normalized_y": 0.48,
            "normalized_width": 0.16, "normalized_height": 0.40, "confidence": 0.94,
            "tracking_target": "important_object", "visible_face_count": 0,
            "scene_type": "HANDS_ON_DEMO", "framing_observation": "object", "eye_line_y": 0.48,
            "gesture_active": True, "gesture_area_visible": True,
        }],
    }
    timeline = _tracking_timeline("source.mp4", make_crop_plan(product_source, canvas, config))
    product = build_reframe_plan(product_source, canvas, config, timeline).composition_segments[0].composition_quality_decision
    assert product.status == "passed"
    assert product.scene_type == "HANDS_ON_DEMO"
    assert product.framing_intent == "PRODUCT_OR_HANDS"
    assert product.selected_target == "important_object"

    unsafe_screen_source = {
        "display_width": 320, "display_height": 180, "rotation": 0, "visual_evidence_status": "valid",
        "subject_keyframes": [{
            "time_seconds": 1, "normalized_x": 0.50, "normalized_y": 0.42,
            "normalized_width": 0.20, "normalized_height": 0.35, "confidence": 0.94,
            "tracking_target": "primary_face", "visible_face_count": 1,
            "scene_type": "PRESENTATION_SCREEN", "framing_observation": "head_shoulders", "eye_line_y": 0.35,
            "gesture_active": False, "gesture_area_visible": False,
        }],
    }
    blocked = build_reframe_plan(
        unsafe_screen_source, canvas, config,
        _tracking_timeline("source.mp4", make_crop_plan(unsafe_screen_source, canvas, config)),
    ).composition_segments[0].composition_quality_decision
    assert blocked.status == "blocked"
    assert "SCREEN_CONTENT_CROPPED" in blocked.reason_codes


def test_head_only_and_unavailable_visual_evidence_never_pass_composition() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    head_only_source = {
        "display_width": 320, "display_height": 180, "rotation": 0, "visual_evidence_status": "valid",
        "subject_keyframes": [{
            "time_seconds": 1, "normalized_x": 0.50, "normalized_y": 0.40,
            "normalized_width": 0.16, "normalized_height": 0.18, "confidence": 0.95,
            "tracking_target": "primary_face", "visible_face_count": 1,
            "scene_type": "TALKING_HEAD", "framing_observation": "head_only", "eye_line_y": 0.36,
            "gesture_active": False, "gesture_area_visible": False,
        }],
    }
    head_only = build_reframe_plan(
        head_only_source, canvas, config,
        _tracking_timeline("source.mp4", make_crop_plan(head_only_source, canvas, config)),
    ).composition_segments[0].composition_quality_decision
    assert head_only.status == "blocked"
    assert {"HEAD_ONLY_CROP", "CHEST_FRAMING_MISSING", "SHOULDERS_CROPPED"} <= set(head_only.reason_codes)

    unavailable_source = {"display_width": 320, "display_height": 180, "rotation": 0, "visual_evidence_status": "evidence_unavailable"}
    unavailable = build_reframe_plan(
        unavailable_source, canvas, config,
        _tracking_timeline("source.mp4", make_crop_plan(unavailable_source, canvas, config)),
    ).composition_segments[0].composition_quality_decision
    assert unavailable.status == "evidence_unavailable"
    assert {"UNKNOWN_SCENE_FALLBACK", "VISUAL_EVIDENCE_UNAVAILABLE"} <= set(unavailable.reason_codes)
    assert unavailable.fallback_provenance


def test_scene_boundaries_split_the_render_timeline_and_force_controlled_cuts() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto", transitions="short_crossfade")
    source = {"display_width": 320, "display_height": 180, "rotation": 0}
    clip = SourceVideoClip(
        clip_id="source-shot", order=1, timeline_start_seconds=0, timeline_end_seconds=3, duration_seconds=3,
        source_path="source.mp4", source_start_seconds=0, source_end_seconds=3,
        visual_strategy="mapped_source", crop_plan=make_crop_plan(source, canvas, config), status="ready",
    )
    split = _split_timeline_at_scene_boundaries(
        VideoTimeline(clips=[clip], duration_seconds=3),
        {**source, "scene_boundaries": [{"timestamp": 1}, {"timestamp": 2}]}, config,
    )
    assert [item.clip_id for item in split.clips] == ["source-shot-scene-01", "source-shot-scene-02", "source-shot-scene-03"]
    assert all(item.transition_type == "cut" for item in split.transitions)
    graph, label = _timeline_filter([item.duration_seconds for item in split.clips], split.transitions)
    assert "xfade" not in graph and "concat=n=2" in graph and label == "[mix2]"


def test_production_render_reads_scene_boundaries_and_renders_a_controlled_cut(tmp_path: Path) -> None:
    from app.utils import write_json

    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.transitions = "short_crossfade"
    write_json(tmp_path / "work" / "scene_boundaries.json", {"boundaries": [{"timestamp": 1.5}]})
    project = VideoCompositionService(tmp_path, config).compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "out",
    )
    assert project.result and project.result.validation.status == "valid"
    assert any(item.transition_type == "cut" for item in project.timeline.transitions)
    assert len(project.reframe_plan.composition_segments) == len(project.timeline.clips)
    assert production_render_report_section(project)["composition"]["summary"]["scene_transition_count"] == 1


def test_active_speaker_tracking_applies_hysteresis_before_switching_focus() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {"display_width": 320, "display_height": 180, "rotation": 0}
    crop = make_crop_plan(source, canvas, config)
    first = SourceVideoClip(
        clip_id="speaker-a", order=1, timeline_start_seconds=0, timeline_end_seconds=2, duration_seconds=2,
        source_path="source.mp4", source_start_seconds=0, source_end_seconds=1.95, speaker="speaker-a",
        visual_strategy="mapped_source", crop_plan=crop, status="ready",
    )
    second = SourceVideoClip(
        clip_id="speaker-b", order=2, timeline_start_seconds=2, timeline_end_seconds=4, duration_seconds=2,
        source_path="source.mp4", source_start_seconds=2.05, source_end_seconds=4, speaker="speaker-b",
        visual_strategy="mapped_source", crop_plan=crop, status="ready",
    )
    timeline = VideoTimeline(clips=[first, second], duration_seconds=4)
    plan = build_reframe_plan({
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.25, "normalized_y": 0.4, "confidence": 0.92, "visible_face_count": 2, "active_speaker_confidence": 0.92},
            {"time_seconds": 1.95, "normalized_x": 0.45, "normalized_y": 0.4, "confidence": 0.92, "visible_face_count": 2, "active_speaker_confidence": 0.92},
            {"time_seconds": 2.05, "normalized_x": 0.25, "normalized_y": 0.4, "confidence": 0.82, "visible_face_count": 2, "active_speaker_confidence": 0.82},
            {"time_seconds": 4, "normalized_x": 0.45, "normalized_y": 0.4, "confidence": 0.82, "visible_face_count": 2, "active_speaker_confidence": 0.82},
        ],
    }, canvas, config, timeline)
    first_segment, second_segment = plan.composition_segments
    assert first_segment.tracking_mode == "active_speaker_tracking"
    assert second_segment.tracking_mode == "group_framing"
    assert "hysteresis" in second_segment.tracking_reason


def test_tracking_filter_renders_a_smoothed_motion_crop(tmp_path: Path) -> None:
    from app.config import ProductionRenderConfig

    source_path = _source_video(tmp_path / "tracking-source.mp4")
    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {"display_width": 320, "display_height": 180, "rotation": 0}
    timeline = _tracking_timeline(str(source_path), make_crop_plan(source, canvas, config))
    reframe = build_reframe_plan({
        **source,
        "subject_keyframes": [
            {"time_seconds": 0, "normalized_x": 0.28, "normalized_y": 0.4, "confidence": 0.92},
            {"time_seconds": 1.5, "normalized_x": 0.45, "normalized_y": 0.4, "confidence": 0.92},
            {"time_seconds": 3, "normalized_x": 0.62, "normalized_y": 0.4, "confidence": 0.92},
        ],
    }, canvas, config, timeline)
    applied = apply_composition_segments(timeline, reframe.composition_segments)
    destination = tmp_path / "tracked.mp4"
    VideoCompositionService(tmp_path, _audio_config())._prepare_visual_clip(applied.clips[0], canvas, destination)
    assert abs(float(probe_media(destination, require_video=True)["video_duration"]) - 3) < 0.15


def test_rotation_metadata_is_reflected_in_safe_display_crop(tmp_path: Path) -> None:
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("ffmpeg is required for rotation metadata test")
    source = _source_video(tmp_path / "source.mp4", width=320, height=180)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run([
        executable, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-c", "copy",
        "-metadata:s:v:0", "rotate=90", str(rotated),
    ], check=True)
    info = probe_media(rotated, require_video=True)
    assert info["rotation"] in {0, 90, 270}
    if info["rotation"] in {90, 270}:
        assert info["display_width"] == info["height"] and info["display_height"] == info["width"]
    from app.config import ProductionRenderConfig
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="center_crop")
    config.validate()
    crop = make_crop_plan(info, CanvasConfig(width=180, height=320, fps=30), config)
    assert crop.crop_width is None or crop.crop_x + crop.crop_width <= crop.source_width


def test_subtitles_use_audio_timeline_split_long_text_and_escape_ass(tmp_path: Path) -> None:
    config, plan, _source, _transcript, audio = _upstream(tmp_path)
    raw = plan.model_dump(mode="json")
    raw["segments"][0]["text"] = "{Long} narration text is split deterministically into readable subtitle groups without rewriting the original words. " * 2
    raw["segments"][0]["word_count"] = len(raw["segments"][0]["text"].split())
    long_plan = type(plan).model_validate(raw)
    project = build_subtitle_project(long_plan, audio, config.production_render)
    assert len(split_subtitle_text(raw["segments"][0]["text"], config.production_render)) > 1
    assert len(split_subtitle_text("x" * 500, config.production_render)) > 1
    assert all(left.end_seconds <= right.start_seconds for left, right in zip(project.cues, project.cues[1:]))
    ass = tmp_path / "production-subtitles.ass"
    write_production_ass(project, ass, 180, 320)
    content = ass.read_text(encoding="utf-8-sig")
    assert r"\{" in content and "Dialogue:" in content
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Production,"))
    metrics = style_line.split(",")
    assert metrics[2] == "12" and metrics[16:20] == ["1", "0", "2", "12"]
    config.production_render.subtitle_font_family = "__missing_font__"
    _style, fallback, warning = resolve_subtitle_style(config.production_render)
    assert fallback and warning and "Arial" in warning


def test_dynamic_subtitles_use_word_level_highlighting_and_keep_cyrillic(tmp_path: Path) -> None:
    config, plan, _source, _transcript, audio = _upstream(tmp_path)
    config.production_render.subtitle_style = "dynamic"
    raw = plan.model_dump(mode="json")
    raw["segments"][0]["text"] = "Это заметный динамический заголовок"
    raw["segments"][0]["word_count"] = 4
    dynamic_plan = type(plan).model_validate(raw)
    project = build_subtitle_project(dynamic_plan, audio, config.production_render)
    ass = tmp_path / "dynamic.ass"
    write_production_ass(project, ass, 180, 320)
    content = ass.read_text(encoding="utf-8-sig")
    assert "&H004AD5FF" in content  # #FFD54A in ASS BGR order
    assert r"{\k" in content
    assert "ДИНАМИЧЕСКИЙ" in content


def test_dialogue_subtitles_use_source_word_timestamps_when_available(tmp_path: Path) -> None:
    config, plan, _source, _transcript, audio = _upstream(tmp_path)
    config.production_render.subtitle_style = "dynamic"
    transcript = {
        "words": [
            {"start": 1.0, "end": 1.2, "text": "Source"},
            {"start": 1.2, "end": 1.45, "text": "dialogue"},
            {"start": 1.45, "end": 1.7, "text": "remains"},
            {"start": 1.7, "end": 2.0, "text": "audible."},
        ]
    }

    project = build_subtitle_project(plan, audio, config.production_render, transcript)
    cues = [item for item in project.cues if item.source_type == "dialogue"]
    dialogue_clip = next(item for item in audio.timeline.clips if item.clip_type == "dialogue")
    assert [item.text for cue in cues for item in cue.word_timings] == ["Source", "dialogue", "remains", "audible."]
    assert cues[0].word_timings[0].start_seconds == pytest.approx(dialogue_clip.timeline_start_seconds)
    ass = tmp_path / "dialogue-timing.ass"
    write_production_ass(project, ass, 180, 320)
    assert r"{\k20}SOURCE" in ass.read_text(encoding="utf-8-sig")


def test_transition_filters_preserve_audio_timeline_duration() -> None:
    crossfade, label = _timeline_filter([1.0, 2.0, 1.0], "short_crossfade")
    assert "xfade" in crossfade and "tpad" in crossfade and label == "[vconcat]"
    fade, label = _timeline_filter([1.0], "fade_from_black")
    assert "fade=t=in" in fade and label == "[vfaded]"


def test_short_crossfade_renders_without_changing_final_duration(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.transitions = "short_crossfade"
    result = VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert result.result and result.result.validation.status == "valid"
    assert abs(result.actual_duration_seconds - audio.mix.duration_seconds) < 0.15


def test_render_cpu_mux_subtitles_cache_and_secret_free_report(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    first = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert first.status in {"completed", "warning"}
    assert first.result and first.result.validation.status == "valid"
    assert Path(first.result.output_file or "").is_file()
    assert first.metadata.ai_called is False and first.metadata.tts_regenerated is False and first.metadata.audio_remixed is False
    assert first.plan_reference == plan.reference()
    assert first.reframe_plan.plan_reference == plan.reference()
    assert first.subtitle_project and first.subtitle_project.plan_reference == plan.reference()
    second = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert second.result and second.result.cache_hit
    encode_manifest = next((tmp_path / "work" / "creative-render-cache" / "encode").glob("*.manifest.json"))
    encode_cache = encode_manifest.with_name(encode_manifest.name.removesuffix(".manifest.json") + ".mp4")
    encode_cache.write_bytes(b"corrupt")
    rebuilt = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert rebuilt.result and rebuilt.result.cache_hit is False
    assert rebuilt.metadata.cache_node_hits["base-visual"] is True
    assert rebuilt.metadata.cache_node_hits["encode"] is False
    report = production_render_report_section(second)
    assert report["cache_hit"] and "sk-" not in json.dumps(report)
    assert report["quality"]["status"] in {"passed", "warning"}
    assert report["subtitles_enabled"] is True
    layout = report["subtitle_layout"]
    assert layout["contract_version"] == "5E.0"
    assert layout["quality_decision"]["schema_version"] == "5E.0"
    assert layout["resolved_cue_count"] == report["subtitle_cue_count"]
    assert layout["final_validation"] == report["quality"]
    assert all({
        "original_text", "original_line_count", "resolved_lines", "resolved_font_size",
        "split_reason", "fallback_used", "layout_state",
    } <= cue.keys() for cue in layout["cues"])
    composition = report["composition"]
    assert len(composition["segments"]) == report["clip_count"]
    assert {
        "foreground_coverage_ratio", "blur_coverage_ratio", "subject_screen_ratio",
        "unused_visual_area_ratio", "scene_transition_count",
    } <= composition["summary"].keys()


def test_creative_preview_and_final_render_the_same_compiled_plan(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    final = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    compiled = CompiledRenderPlan.model_validate(json.loads(
        (tmp_path / "out" / "production-render" / "compiled-render-plan.json").read_text(encoding="utf-8")
    ))
    preview = service.compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "out",
        render_profile="creative_preview", compiled_plan=compiled,
    )
    preview_manifest = RenderParityManifest.model_validate(json.loads(
        (tmp_path / "out" / "creative-preview" / "parity-manifest.json").read_text(encoding="utf-8")
    ))
    final_manifest = RenderParityManifest.model_validate(json.loads(
        (tmp_path / "out" / "production-render" / "parity-manifest.json").read_text(encoding="utf-8")
    ))

    assert preview.canvas.fps == final.canvas.fps == 30
    assert (preview.canvas.width, preview.canvas.height) == (540, 960)
    assert preview.metadata.compiled_plan_hash == final.metadata.compiled_plan_hash == compiled.plan_hash
    assert preview.metadata.parity_signature == final.metadata.parity_signature == compiled.parity_signature
    assert_preview_final_parity(preview_manifest, final_manifest)
    approved = service.compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "out",
        compiled_plan=compiled,
    )
    assert approved.result and approved.result.cache_hit is True

    final_path = Path(approved.result.output_file or "")
    previous_final = final_path.read_bytes()
    Path(preview.result.output_file or "").write_bytes(b"corrupt preview")
    with pytest.raises(ProductionRenderError, match="PREVIEW_FINAL_PARITY_MISMATCH"):
        service.compose(
            plan, audio, source, transcript, tmp_path / "work", tmp_path / "out",
            compiled_plan=compiled,
        )
    assert final_path.read_bytes() == previous_final


def test_native_creative_decisions_change_rendered_output_end_to_end(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    source_info = probe_media(source.path, require_video=True)
    raw_timeline, _fallbacks = build_video_timeline(
        plan, audio, transcript, source.path, source_info,
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    mapping = source_output_map_from_legacy_timeline(raw_timeline)
    rich_intent = _native_intent(plan, mapping)
    calm_intent = rich_intent.model_copy(update={
        "intent_id": "intent-native-calm",
        "proposal_hash": "6" * 64,
        "policy": rich_intent.policy.model_copy(update={"source_broll_enabled": False}),
        "beats": (),
        "semantic_emphasis": (),
        "composition_targets": (),
        "motion_events": (),
        "source_broll": (),
    })
    observations = (
        TargetObservation(
            observation_id="observation-native-1", frame=8,
            target=AttentionTarget.SUBJECT, target_ref="subject-native",
            bounds={"x": 0.04, "y": 0.18, "width": 0.30, "height": 0.58},
            confidence=0.96, evidence_ref="visual-native", scene_id="scene-native",
        ),
        TargetObservation(
            observation_id="observation-native-2", frame=20,
            target=AttentionTarget.SUBJECT, target_ref="subject-native",
            bounds={"x": 0.08, "y": 0.18, "width": 0.30, "height": 0.58},
            confidence=0.95, evidence_ref="visual-native", scene_id="scene-native",
        ),
    )
    scenes = (SourceSceneEvidence(
        scene_id="scene-native", source_id=plan.reference().identity.source_id,
        source=SourceInterval.from_seconds(2.0, 2.6),
        semantic_kinds=(SourceBRollSemanticKind.ACTION,),
        story_unit_ids=("story-native",), beat_roles=(BeatRole.ACTION,),
        evidence_refs=("scene-native",), confidence=0.94,
        source_crop=NormalizedRect(x=0.40, y=0.20, width=0.36, height=0.55),
        source_target=AttentionTarget.OBJECT,
        identity_status="verified", attribution_status="verified",
        chronology_status="safe", causality_status="supported", rights_status="verified",
        payoff_signal="none", provenance=("e2e-regression",),
    ),)
    service = VideoCompositionService(tmp_path, config)

    calm = service.compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "calm",
        creative_intent=calm_intent, force_recompute=True,
    )
    preview = service.compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "rich",
        render_profile="creative_preview", creative_intent=rich_intent,
        target_observations=observations, source_scenes=scenes, force_recompute=True,
    )
    compiled = CompiledRenderPlan.model_validate(json.loads(
        (tmp_path / "rich" / "creative-preview" / "compiled-render-plan.json").read_text(encoding="utf-8")
    ))
    final = service.compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "rich",
        compiled_plan=compiled, force_recompute=True,
    )
    no_motion = build_motion_plan(
        rich_intent.model_copy(update={"motion_events": ()}),
        compiled.caption_plan,
        compiled.composition_plan,
        compiled.source_broll_plan,
    )
    motionless_compiled = compile_render_plan(
        rich_intent,
        compiled.caption_plan,
        compiled.composition_plan,
        no_motion,
        compiled.source_broll_plan,
        CanvasPlan(width=compiled.canvas.width, height=compiled.canvas.height),
    )
    motionless = service.compose(
        plan, audio, source, transcript, tmp_path / "work", tmp_path / "motionless",
        compiled_plan=motionless_compiled, force_recompute=True,
    )

    assert compiled.compatibility_mode == "native"
    assert compiled.caption_plan.schema_version == "7C.caption-plan.1"
    assert compiled.composition_plan.schema_version == "7D.composition-plan.1"
    assert compiled.source_broll_plan.schema_version == "7E.source-broll-plan.1"
    assert compiled.motion_plan.schema_version == "7F.motion-plan.1"
    assert compiled.source_broll_plan.segments
    assert any(item.primitive_id == "punch_in" for item in compiled.motion_plan.events)
    rendered_timeline = json.loads(
        (tmp_path / "rich" / "production-render" / "video-timeline.json").read_text(encoding="utf-8")
    )
    assert any(
        item["visual_strategy"] == "candidate_excerpt" and item["source_start_seconds"] >= 2
        for item in rendered_timeline["clips"]
    )
    assert any(
        item["crop_plan"]["strategy"] == "manual_normalized_crop"
        for item in rendered_timeline["clips"]
    )
    rendered_ass = (tmp_path / "rich" / "production-render" / "production-subtitles.ass").read_text(
        encoding="utf-8-sig"
    )
    assert "CaptionPlan: 7C.caption-plan.1" in rendered_ass
    assert "\\fad(" in rendered_ass
    assert preview.metadata.compiled_plan_hash == final.metadata.compiled_plan_hash == compiled.plan_hash
    assert preview.metadata.parity_signature == final.metadata.parity_signature == compiled.parity_signature
    assert Path(calm.result.output_file or "").read_bytes() != Path(final.result.output_file or "").read_bytes()
    assert _native_motion_filter(compiled.motion_plan, final.canvas) is None
    motionless_timeline = json.loads(
        (tmp_path / "motionless" / "production-render" / "video-timeline.json").read_text(encoding="utf-8")
    )
    assert motionless_timeline["clips"] == rendered_timeline["clips"]


def test_auto_encoder_falls_back_to_cpu_when_nvenc_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.encoder = "auto"
    monkeypatch.setattr("app.video_composition.nvenc_available", lambda: False)
    result = VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert result.result and result.result.encoder == "libx264" and result.result.hardware_fallback


def test_explicit_unavailable_nvenc_fails_without_creating_false_final(tmp_path: Path, monkeypatch) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.encoder = "nvenc"
    monkeypatch.setattr("app.video_composition.nvenc_available", lambda: False)
    with pytest.raises(ProductionRenderError, match="nvenc"):
        VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert not (tmp_path / "out" / "production-render" / "final-short.mp4").exists()


def test_subtitle_style_change_rerenders_without_changing_audio(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    first = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    audio_checksum = Path(audio.mix.mixed_audio_path or "").read_bytes()
    config.production_render.subtitle_style = "clean"
    second = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert first.result and second.result and second.result.cache_hit is False
    assert Path(audio.mix.mixed_audio_path or "").read_bytes() == audio_checksum
    assert second.metadata.cache_node_hits == {
        "captions": False,
        "composition": True,
        "broll": True,
        "motion": True,
        "base-visual": True,
        "composite": False,
        "encode": False,
        "qc": False,
    }


def test_subtitle_disabled_still_exports_ass_artifact_without_burning_it_into_mp4(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.subtitles_enabled = False
    result = VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert result.result and result.result.validation.status == "valid"
    root = tmp_path / "out" / "production-render"
    assert (root / "production-subtitles.ass").is_file()
    assert result.render_request.subtitles_enabled is False
    assert result.metadata.single_pass_encode is True
    assert result.result.encoder == "copy"


def test_atomic_invalid_final_keeps_previous_mp4(tmp_path: Path, monkeypatch) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    first = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    final = Path(first.result.output_file or "")
    checksum = final.read_bytes()
    monkeypatch.setattr(
        "app.video_composition.validate_final_video",
        lambda *args, **kwargs: RenderValidation(status="invalid", messages=["forced invalid"]),
    )
    with pytest.raises(ProductionRenderError, match="validation"):
        service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out", force_recompute=True)
    assert final.read_bytes() == checksum


def test_invalid_mixed_audio_and_cli_production_flags_are_clear(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    broken = audio.model_copy(update={"mix": audio.mix.model_copy(update={"mixed_audio_path": str(tmp_path / "missing.wav")})})
    with pytest.raises(ProductionRenderError, match="mixed_audio"):
        VideoCompositionService(tmp_path, config).compose(plan, broken, source, transcript, tmp_path / "work", tmp_path / "out")
    arguments = build_parser().parse_args([
        "process", "--input", "source.mp4", "--production-render-only", "--recompute-production-render",
        "--disable-subtitles", "--crop-strategy", "center_crop", "--subtitle-style", "clean", "--video-encoder", "cpu",
    ])
    assert arguments.production_render_only and arguments.recompute_production_render and arguments.disable_subtitles
    disabled = build_parser().parse_args(["process", "--input", "source.mp4", "--disable-production-render"])
    assert disabled.disable_production_render


def test_production_render_only_preserves_existing_legacy_mp4(tmp_path: Path) -> None:
    config, plan, source, transcript, _audio = _upstream(tmp_path)
    pipeline = Pipeline(tmp_path, config, production_render_only=True)
    prepared_source, work_directory, output_directory = pipeline._prepare_source(str(source.path), None)
    tts = _tts_result(tmp_path, config, plan)
    audio = AudioCompositionService(tmp_path, config).compose(
        plan, prepared_source, transcript, tts, work_directory, output_directory,
    )
    assert audio.export.path
    from app.utils import write_json
    write_json(output_directory / "production-plan.json", plan.model_dump(mode="json"))
    write_json(work_directory / "transcript.json", transcript)
    old_mp4 = output_directory / "legacy.mp4"
    old_mp4.write_bytes(b"legacy-video")
    write_json(output_directory / "report.json", {"output_files": [str(old_mp4)], "selected_clips_count": 1})
    result = pipeline._run_production_render_only(StageTracker(work_directory / "state.json"), prepared_source, work_directory, output_directory)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert old_mp4.read_bytes() == b"legacy-video"
    assert result.output_files and result.output_files[0].name == "final-short-01.mp4"
    assert report["production_render"]["ai_called"] is False
