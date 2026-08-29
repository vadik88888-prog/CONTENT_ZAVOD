from __future__ import annotations

import pytest

from app.composition_planning import (
    CompositionPlannerConfig,
    TargetObservation,
    _AtomicState,
    _calibrate_layout_timeline,
    _containment,
    build_composition_plan,
)
from app.creative_contracts import (
    AttentionTarget,
    CreativeIntent,
    CreativePolicy,
    EditMapSegment,
    EvidenceItem,
    ImmutableProductionIdentity,
    ImmutableProductionPlanLink,
    Intensity,
    LayoutFamily,
    MotionDomain,
    MotionPurpose,
    NormalizedRect,
    OutputInterval,
    ResolvedCompositionTarget,
    ResolvedMotionEvent,
    SourceInterval,
    SourceOutputTimeMap,
)


def _intent(
    targets: tuple[ResolvedCompositionTarget, ...],
    *,
    motion: tuple[ResolvedMotionEvent, ...] = (),
) -> CreativeIntent:
    refs = sorted({ref for item in (*targets, *motion) for ref in item.evidence_refs})
    evidence = tuple(
        EvidenceItem(
            evidence_ref=ref,
            evidence_kind="visual",
            source=SourceInterval.from_seconds(0, 10),
            confidence=0.95,
            artifact_fingerprint=(str(index + 1) * 64)[:64],
            provenance=f"vision:{ref}",
        )
        for index, ref in enumerate(refs)
    )
    return CreativeIntent(
        intent_id="intent-composition",
        revision=1,
        production_plan=ImmutableProductionPlanLink(
            plan_id="plan-001",
            plan_fingerprint="a" * 64,
            identity=ImmutableProductionIdentity(
                project_id="project-001",
                run_id="run-001",
                analysis_id="analysis-001",
                candidate_id="candidate-001",
                source_id="source-001",
            ),
        ),
        source_output_mapping=SourceOutputTimeMap(segments=(
            EditMapSegment(
                map_id="edit-001",
                source=SourceInterval.from_seconds(0, 10),
                output=OutputInterval(start_frame=0, end_frame=300),
            ),
        )),
        evidence_fingerprint="b" * 64,
        evidence_manifest=evidence,
        proposal_hash="c" * 64,
        policy=CreativePolicy(
            preset_id="clean-podcast",
            preset_version="1",
            platform="universal",
            intensity=Intensity.BALANCED,
        ),
        confidence=0.9,
        composition_targets=targets,
        motion_events=motion,
    )


def _target(
    decision_id: str,
    target: AttentionTarget,
    start: int,
    end: int,
    evidence_ref: str,
    *,
    target_ref: str | None,
    confidence: float = 0.92,
    priority: int = 50,
    layouts: tuple[LayoutFamily, ...] = (LayoutFamily.SINGLE_SUBJECT,),
) -> ResolvedCompositionTarget:
    return ResolvedCompositionTarget(
        decision_id=decision_id,
        source=SourceInterval.from_seconds(start / 30, end / 30),
        output=OutputInterval(start_frame=start, end_frame=end),
        confidence=confidence,
        evidence_refs=(evidence_ref,),
        target=target,
        target_ref=target_ref,
        priority=priority,
        allowed_layouts=layouts,
    )


def _observation(
    observation_id: str,
    frame: int,
    target: AttentionTarget,
    target_ref: str,
    evidence_ref: str,
    x: float,
    *,
    width: float = 0.16,
    confidence: float = 0.94,
) -> TargetObservation:
    return TargetObservation(
        observation_id=observation_id,
        frame=frame,
        target=target,
        target_ref=target_ref,
        bounds=NormalizedRect(x=x, y=0.2, width=width, height=0.56),
        confidence=confidence,
        evidence_ref=evidence_ref,
        scene_id="scene-001",
    )


def test_semantic_target_timeline_selects_single_screen_product_and_group_layouts() -> None:
    targets = (
        _target(
            "speaker-target", AttentionTarget.SPEAKER, 0, 120, "speaker-evidence",
            target_ref="speaker-a", layouts=(LayoutFamily.STABLE_SPEAKER, LayoutFamily.SINGLE_SUBJECT),
        ),
        _target(
            "screen-target", AttentionTarget.SCREEN, 120, 240, "screen-evidence",
            target_ref="screen-a", priority=90,
            layouts=(LayoutFamily.SCREEN_PRODUCT, LayoutFamily.SCREEN_PRIORITY),
        ),
        _target(
            "group-target", AttentionTarget.GROUP, 240, 300, "group-evidence",
            target_ref=None, layouts=(LayoutFamily.WIDE_GROUP, LayoutFamily.SPLIT),
        ),
    )
    observations = (
        _observation("speaker-1", 30, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.12),
        _observation("speaker-2", 90, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.14),
        _observation("screen-1", 150, AttentionTarget.SCREEN, "screen-a", "screen-evidence", 0.42, width=0.42),
        _observation("screen-2", 210, AttentionTarget.SCREEN, "screen-a", "screen-evidence", 0.40, width=0.44),
        _observation("group-1", 255, AttentionTarget.GROUP, "person-a", "group-evidence", 0.08),
        _observation("group-2", 285, AttentionTarget.GROUP, "person-b", "group-evidence", 0.72),
    )

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
    )

    assert plan.schema_version == "7D.composition-plan.1"
    assert plan.ordered_fallbacks == ("wider_crop", "stable_source")
    assert [segment.target for segment in plan.segments] == [
        AttentionTarget.SPEAKER, AttentionTarget.SCREEN, AttentionTarget.GROUP,
    ]
    assert [segment.layout for segment in plan.segments] == [
        LayoutFamily.STABLE_SPEAKER, LayoutFamily.SCREEN_PRIORITY, LayoutFamily.WIDE_GROUP,
    ]
    assert all(segment.layout != LayoutFamily.FIT_BACKGROUND for segment in plan.segments)
    assert all(segment.fallback != "fit_background" for segment in plan.segments)
    assert all(segment.geometry is not None for segment in plan.segments)
    assert all(segment.target_bounds is not None for segment in plan.segments)
    assert plan.quality_report.metrics.layout_switch_count == 2
    assert plan.quality_report.metrics.layout_switches_per_minute == 12.0

    strict = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(maximum_switches_per_minute=5.0),
    )
    assert any(
        item.code == "COMPOSITION_LAYOUT_SWITCH_RATE_HIGH"
        for item in strict.quality_report.findings
    )


def test_explicit_confident_speaker_handoff_bypasses_stale_crop_hold() -> None:
    targets = (
        _target("speaker-a-1", AttentionTarget.SPEAKER, 0, 30, "evidence-a", target_ref="speaker-a"),
        _target("speaker-b", AttentionTarget.SPEAKER, 30, 60, "evidence-b", target_ref="speaker-b"),
        _target("speaker-a-2", AttentionTarget.SPEAKER, 60, 90, "evidence-a", target_ref="speaker-a"),
    )
    observations = (
        _observation("a-1", 15, AttentionTarget.SPEAKER, "speaker-a", "evidence-a", 0.10),
        _observation("b-1", 45, AttentionTarget.SPEAKER, "speaker-b", "evidence-b", 0.70),
        _observation("a-2", 75, AttentionTarget.SPEAKER, "speaker-a", "evidence-a", 0.12),
    )
    config = CompositionPlannerConfig(minimum_hold_frames=45, switch_cooldown_frames=45)

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080, config=config,
    )

    assert plan.segments[0].target_ref == "speaker-a"
    assert plan.segments[1].target_ref == "speaker-b"
    assert plan.segments[1].movement_reason == "target_switch"
    assert plan.segments[2].target_ref == "speaker-a"
    assert all(segment.fallback != "fit_background" for segment in plan.segments)
    assert "TARGET_SWITCH_TIMELY:30" in plan.diagnostics
    assert "TARGET_SWITCH_TIMELY:60" in plan.diagnostics


def test_target_handoff_resets_only_when_smooth_track_would_clip_new_target() -> None:
    targets = (
        _target("speaker-left", AttentionTarget.SPEAKER, 0, 60, "evidence-left", target_ref="speaker-left"),
        _target("speaker-right", AttentionTarget.SPEAKER, 60, 120, "evidence-right", target_ref="speaker-right"),
    )
    observations = (
        _observation("left", 30, AttentionTarget.SPEAKER, "speaker-left", "evidence-left", 0.04),
        _observation("right", 90, AttentionTarget.SPEAKER, "speaker-right", "evidence-right", 0.80),
    )

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(minimum_hold_frames=120, switch_cooldown_frames=120),
    )

    handoff = next(item for item in plan.segments if item.output.start_frame == 60)
    assert handoff.target_ref == "speaker-right"
    assert handoff.movement_reason == "target_switch"
    assert handoff.crop_keyframes[0].crop == handoff.crop
    assert all(
        _containment(observations[1].bounds, keyframe.crop) >= 0.82
        for keyframe in handoff.crop_keyframes
    )
    assert "TARGET_SWITCH_TIMELY:60" in plan.diagnostics
    assert "CROP_RESET_FOR_TARGET_TRACK_SAFETY:60" in plan.diagnostics


def test_story_short_uses_actual_crop_and_blocks_an_uncontainable_group() -> None:
    target = _target(
        "story-group", AttentionTarget.GROUP, 120, 138, "story-group-evidence",
        target_ref=None, layouts=(LayoutFamily.WIDE_GROUP,),
    )
    observations = (
        _observation(
            "story-person-left", 126, AttentionTarget.GROUP, "person-left",
            "story-group-evidence", 0.05, width=0.22,
        ),
        _observation(
            "story-person-right", 132, AttentionTarget.GROUP, "person-right",
            "story-group-evidence", 0.73, width=0.22,
        ),
    )

    plan = build_composition_plan(
        _intent((target,)), observations, source_width=1920, source_height=1080,
    )

    assert plan.segments
    assert all(segment.layout != LayoutFamily.FIT_BACKGROUND for segment in plan.segments)
    assert all(segment.fallback != "fit_background" for segment in plan.segments)
    evidence_segment = next(segment for segment in plan.segments if segment.target == AttentionTarget.GROUP)
    assert evidence_segment.evidence_refs == ("story-group-evidence",)
    assert evidence_segment.target_bounds is not None
    assert evidence_segment.geometry is not None
    assert plan.quality_report.status == "BLOCKED"
    assert plan.quality_report.metrics.clipped_target_count == 1
    assert plan.quality_report.metrics.layout_switch_count == 0
    assert "SCENE_TARGET_CONTAINMENT_UNRESOLVED:0" in plan.diagnostics


def test_layout_edge_fragments_and_local_switch_bursts_are_calibrated() -> None:
    full = NormalizedRect(x=0, y=0, width=1, height=1)

    def state(start: int, end: int, layout: LayoutFamily) -> _AtomicState:
        return _AtomicState(
            output=OutputInterval(start_frame=start, end_frame=end),
            state=None,
            layout=layout,
            desired_crop=full,
            target=AttentionTarget.STABLE_SOURCE,
            target_ref=None,
            confidence=0,
            evidence_refs=(),
            fallback="fit_background",
            reason="safe_fallback",
            punch_event=None,
        )

    edge, edge_diagnostics = _calibrate_layout_timeline(
        (state(0, 5, LayoutFamily.WIDE_GROUP), state(5, 100, LayoutFamily.FIT_BACKGROUND)),
        CompositionPlannerConfig(maximum_local_layout_switches=10),
    )
    assert [item.layout for item in edge] == [
        LayoutFamily.FIT_BACKGROUND, LayoutFamily.FIT_BACKGROUND,
    ]
    assert "EDGE_LAYOUT_FRAGMENT_REMOVED:0" in edge_diagnostics

    burst, burst_diagnostics = _calibrate_layout_timeline(
        (
            state(0, 30, LayoutFamily.FIT_BACKGROUND),
            state(30, 60, LayoutFamily.WIDE_GROUP),
            state(60, 90, LayoutFamily.SCREEN_PRIORITY),
        ),
        CompositionPlannerConfig(
            minimum_layout_dwell_frames=10,
            minimum_edge_fragment_frames=10,
            local_burst_window_frames=90,
            maximum_local_layout_switches=1,
        ),
    )
    assert len({item.layout for item in burst}) == 1
    assert "LOCAL_LAYOUT_BURST_CALMED:0-90" in burst_diagnostics


def test_sparse_target_crop_is_held_without_extending_target_evidence() -> None:
    target = _target(
        "speaker-glimpse", AttentionTarget.SPEAKER, 8, 9, "speaker-evidence",
        target_ref="speaker-a", layouts=(LayoutFamily.STABLE_SPEAKER,),
    )
    observation = _observation(
        "speaker-frame", 8, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.55,
    )
    config = CompositionPlannerConfig(minimum_hold_frames=45)

    plan = build_composition_plan(
        _intent((target,)), (observation,), source_width=1920, source_height=1080,
        config=config,
    )

    assert [(item.output.start_frame, item.output.end_frame) for item in plan.segments] == [
        (0, 8), (8, 9), (9, 53), (53, 300),
    ]
    acquired = plan.segments[1]
    held = plan.segments[2]
    released = plan.segments[3]
    assert acquired.target == AttentionTarget.SPEAKER
    assert acquired.evidence_refs == ("speaker-evidence",)
    assert held.target == AttentionTarget.STABLE_SOURCE
    assert held.target_ref is None
    assert held.target_bounds is None
    assert held.evidence_refs == ()
    assert held.fallback == "stable_source"
    assert held.crop != NormalizedRect(x=0, y=0, width=1, height=1)
    assert held.geometry is not None
    assert held.geometry.target_regions == ()
    assert released.fallback == "stable_source"
    assert released.layout == acquired.layout
    assert any(item == "STABLE_CROP_HELD:9" for item in plan.diagnostics)


def test_sparse_podcast_observations_prefer_speaker_family_without_fit_background() -> None:
    targets = (
        _target(
            "speaker-glimpse", AttentionTarget.SPEAKER, 30, 31, "speaker-evidence",
            target_ref="speaker-a", layouts=(LayoutFamily.STABLE_SPEAKER, LayoutFamily.SINGLE_SUBJECT),
        ),
        _target(
            "group-glimpse", AttentionTarget.GROUP, 180, 181, "group-evidence",
            target_ref=None, layouts=(LayoutFamily.WIDE_GROUP, LayoutFamily.FIT_BACKGROUND),
        ),
    )
    observations = (
        _observation(
            "speaker-frame", 30, AttentionTarget.SPEAKER, "speaker-a",
            "speaker-evidence", 0.32, width=0.14,
        ),
        _observation(
            "group-frame", 180, AttentionTarget.GROUP, "group-a",
            "group-evidence", 0.36, width=0.20,
        ),
    )

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
    )

    assert {segment.layout for segment in plan.segments} == {LayoutFamily.STABLE_SPEAKER}
    assert plan.quality_report.metrics.layout_switch_count == 0
    assert any(
        item == "SCENE_LAYOUT_FAMILY_LOCKED:0:stable_speaker" for item in plan.diagnostics
    )
    expected_ratio = (9 / 16) / (1920 / 1080)
    assert all(abs(segment.crop.width / segment.crop.height - expected_ratio) < 1e-7 for segment in plan.segments)
    assert all(segment.fallback != "fit_background" for segment in plan.segments)


def test_gameplay_facecam_evidence_locks_stable_split_across_sparse_gaps() -> None:
    target = _target(
        "facecam-glimpse", AttentionTarget.SPEAKER, 90, 91, "facecam-evidence",
        target_ref="facecam-a",
        layouts=(LayoutFamily.SPLIT, LayoutFamily.SCREEN_PRIORITY, LayoutFamily.FIT_BACKGROUND),
    )
    observation = _observation(
        "facecam-frame", 90, AttentionTarget.SPEAKER, "facecam-a",
        "facecam-evidence", 0.05, width=0.18,
    ).model_copy(update={
        "bounds": NormalizedRect(x=0.05, y=0.55, width=0.18, height=0.28),
    })

    plan = build_composition_plan(
        _intent((target,)), (observation,), source_width=1920, source_height=1080,
    )

    assert {segment.layout for segment in plan.segments} == {LayoutFamily.SPLIT}
    assert plan.quality_report.metrics.layout_switch_count == 0
    assert all(segment.fallback != "fit_background" for segment in plan.segments)
    expected_facecam_ratio = ((9 / 16) / 0.35) / (1920 / 1080)
    assert all(
        abs(segment.crop.width / segment.crop.height - expected_facecam_ratio) < 1e-7
        for segment in plan.segments
    )


def test_gameplay_facecam_panel_region_is_used_without_source_bleed() -> None:
    target = _target(
        "facecam-panel", AttentionTarget.SPEAKER, 90, 91, "facecam-evidence",
        target_ref="facecam-a",
        layouts=(LayoutFamily.SPLIT, LayoutFamily.SCREEN_PRIORITY, LayoutFamily.FIT_BACKGROUND),
    )
    panel = NormalizedRect(
        x=0.0, y=0.34785714, width=0.25892857, height=0.25892857,
    )
    observation = _observation(
        "facecam-panel-frame", 90, AttentionTarget.SPEAKER, "facecam-a",
        "facecam-evidence", 0.0,
    ).model_copy(update={"bounds": panel})

    plan = build_composition_plan(
        _intent((target,)), (observation,), source_width=2560, source_height=1440,
    )

    assert {segment.layout for segment in plan.segments} == {LayoutFamily.SPLIT}
    assert {segment.crop for segment in plan.segments} == {panel}
    assert plan.quality_report.metrics.layout_switch_count == 0


def test_sparse_target_hold_resets_tracking_but_keeps_conversation_family_at_scene_cut() -> None:
    target = _target(
        "speaker-glimpse", AttentionTarget.SPEAKER, 8, 9, "speaker-evidence",
        target_ref="speaker-a", layouts=(LayoutFamily.STABLE_SPEAKER,),
    )
    before_cut = _observation(
        "speaker-frame", 8, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.55,
    )
    cut_marker = _observation(
        "new-scene-frame", 9, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.55,
    ).model_copy(update={"scene_id": "scene-002"})

    plan = build_composition_plan(
        _intent((target,)), (before_cut, cut_marker), source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(minimum_hold_frames=45),
    )

    after_cut = next(item for item in plan.segments if item.output.start_frame == 9)
    assert after_cut.movement_reason == "scene_reset"
    assert after_cut.fallback == "stable_source"
    assert after_cut.layout == LayoutFamily.STABLE_SPEAKER
    assert after_cut.crop != NormalizedRect(x=0, y=0, width=1, height=1)
    assert after_cut.crop_keyframes[0].frame == 9
    assert after_cut.crop_keyframes[0].crop == after_cut.crop
    assert after_cut.crop_keyframes[0].reason == "scene_reset"
    assert "STABLE_CROP_HELD:9" not in plan.diagnostics


def test_sparse_target_hold_keeps_conversation_family_at_edit_map_boundary() -> None:
    target = _target(
        "speaker-glimpse", AttentionTarget.SPEAKER, 8, 9, "speaker-evidence",
        target_ref="speaker-a", layouts=(LayoutFamily.STABLE_SPEAKER,),
    )
    intent = _intent((target,)).model_copy(update={
        "source_output_mapping": SourceOutputTimeMap(segments=(
            EditMapSegment(
                map_id="edit-before-cut",
                source=SourceInterval.from_seconds(0, 0.3),
                output=OutputInterval(start_frame=0, end_frame=9),
            ),
            EditMapSegment(
                map_id="edit-after-cut",
                source=SourceInterval.from_seconds(0.3, 10),
                output=OutputInterval(start_frame=9, end_frame=300),
            ),
        )),
    })

    plan = build_composition_plan(
        intent,
        (_observation(
            "speaker-frame", 8, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.55,
        ),),
        source_width=1920,
        source_height=1080,
        config=CompositionPlannerConfig(minimum_hold_frames=45),
    )

    after_cut = next(item for item in plan.segments if item.output.start_frame == 9)
    assert after_cut.movement_reason == "scene_reset"
    assert after_cut.fallback == "stable_source"
    assert after_cut.layout == LayoutFamily.STABLE_SPEAKER
    assert after_cut.crop != NormalizedRect(x=0, y=0, width=1, height=1)
    assert after_cut.crop_keyframes[0].frame == 9
    assert after_cut.crop_keyframes[0].crop == after_cut.crop
    assert after_cut.crop_keyframes[0].reason == "scene_reset"
    assert "STABLE_CROP_HELD:9" not in plan.diagnostics


def test_scene_family_lock_suppresses_observation_local_punch_in() -> None:
    target = _target(
        "product-target", AttentionTarget.PRODUCT, 0, 300, "product-evidence",
        target_ref="product-a", layouts=(LayoutFamily.SCREEN_PRODUCT,),
    )
    motion = ResolvedMotionEvent(
        decision_id="product-reveal",
        source=SourceInterval.from_seconds(3, 4),
        output=OutputInterval(start_frame=90, end_frame=120),
        confidence=0.9,
        evidence_refs=("product-evidence",),
        purpose=MotionPurpose.EVIDENCE_REVEAL,
        domain=MotionDomain.COMPOSITION,
        intensity=Intensity.BALANCED,
    )
    observations = (
        _observation("product-1", 30, AttentionTarget.PRODUCT, "product-a", "product-evidence", 0.42, width=0.22),
        _observation("product-2", 150, AttentionTarget.PRODUCT, "product-a", "product-evidence", 0.43, width=0.22),
    )

    plan = build_composition_plan(
        _intent((target,), motion=(motion,)), observations, source_width=1920, source_height=1080,
    )

    assert not any(segment.punch_in is not None for segment in plan.segments)
    assert all(
        segment.movement_reason != "editorial_punch_in"
        for segment in plan.segments
    )
    assert "SCENE_PUNCH_IN_SUPPRESSED:0" in plan.diagnostics


def test_low_confidence_and_untrusted_evidence_produce_calm_actual_crop() -> None:
    target = _target(
        "speaker-target", AttentionTarget.SPEAKER, 0, 300, "speaker-evidence",
        target_ref="speaker-a", confidence=0.9,
    )
    observations = (
        _observation(
            "weak", 60, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.15,
            confidence=0.4,
        ),
        _observation(
            "untrusted", 90, AttentionTarget.SPEAKER, "speaker-a", "missing-evidence", 0.75,
        ),
    )

    plan = build_composition_plan(
        _intent((target,)), observations, source_width=1920, source_height=1080,
    )

    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.target == AttentionTarget.STABLE_SOURCE
    assert segment.layout == LayoutFamily.WIDE_GROUP
    assert segment.crop == NormalizedRect(x=0.34179688, y=0, width=0.31640625, height=1)
    assert segment.fallback == "stable_source"
    assert segment.easing_id == "none"
    assert plan.quality_report.status == "PASS_WITH_WARNINGS"
    assert {finding.code for finding in plan.quality_report.findings} >= {
        "COMPOSITION_LOW_CONFIDENCE", "COMPOSITION_SAFE_FALLBACK",
    }


def test_widescreen_speaker_uses_a_target_crop_without_fit_blur() -> None:
    target = _target(
        "speaker-target", AttentionTarget.SPEAKER, 0, 300, "speaker-evidence",
        target_ref="speaker-a", layouts=(LayoutFamily.STABLE_SPEAKER,),
    )
    observation = _observation(
        "speaker-frame", 60, AttentionTarget.SPEAKER, "speaker-a",
        "speaker-evidence", 0.37, width=0.32,
    )

    plan = build_composition_plan(
        _intent((target,)), (observation,), source_width=1920, source_height=960,
    )

    assert plan.quality_report.status != "BLOCKED"
    assert all(segment.layout != LayoutFamily.FIT_BACKGROUND for segment in plan.segments)
    assert all(segment.fallback != "fit_background" for segment in plan.segments)
    assert any(segment.target == AttentionTarget.SPEAKER for segment in plan.segments)
    expected_ratio = (9 / 16) / (1920 / 960)
    assert all(
        abs(segment.crop.width / segment.crop.height - expected_ratio) < 1e-7
        for segment in plan.segments
    )


def test_same_target_reframe_waits_for_a_second_spatial_observation() -> None:
    targets = (
        _target("speaker-1", AttentionTarget.SPEAKER, 0, 90, "speaker-evidence", target_ref="speaker-a"),
        _target("speaker-2", AttentionTarget.SPEAKER, 90, 180, "speaker-evidence", target_ref="speaker-a"),
        _target("speaker-3", AttentionTarget.SPEAKER, 180, 270, "speaker-evidence", target_ref="speaker-a"),
    )
    observations = (
        _observation("speaker-left", 30, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.16),
        _observation("speaker-shift", 120, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.62),
        _observation("speaker-shift-confirmed", 210, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.63),
    )

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(minimum_hold_frames=1),
    )

    by_start = {segment.output.start_frame: segment for segment in plan.segments}
    assert by_start[90].target == AttentionTarget.STABLE_SOURCE
    assert by_start[90].fallback == "stable_source"
    assert by_start[90].crop.x == pytest.approx(by_start[0].crop.x)
    assert by_start[180].target_ref == "speaker-a"
    assert by_start[180].crop.x > by_start[0].crop.x
    assert "TARGET_REFRAME_HELD:90" in plan.diagnostics
    assert "TARGET_REFRAME_CONFIRMED:180" in plan.diagnostics


def test_same_target_reframe_does_not_confirm_from_distant_sparse_evidence() -> None:
    targets = (
        _target("speaker-1", AttentionTarget.SPEAKER, 0, 90, "speaker-evidence", target_ref="speaker-a"),
        _target("speaker-2", AttentionTarget.SPEAKER, 90, 180, "speaker-evidence", target_ref="speaker-a"),
        _target("speaker-3", AttentionTarget.SPEAKER, 180, 300, "speaker-evidence", target_ref="speaker-a"),
    )
    observations = (
        _observation("speaker-left", 30, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.16),
        _observation("speaker-shift", 120, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.62),
        _observation("speaker-late-repeat", 270, AttentionTarget.SPEAKER, "speaker-a", "speaker-evidence", 0.63),
    )

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(minimum_hold_frames=1),
    )

    by_start = {segment.output.start_frame: segment for segment in plan.segments}
    assert by_start[180].target == AttentionTarget.STABLE_SOURCE
    assert by_start[180].crop.x == pytest.approx(by_start[0].crop.x)
    assert "TARGET_REFRAME_CONFIRMED:180" not in plan.diagnostics


def test_crop_track_respects_velocity_acceleration_and_reports_safe_geometry() -> None:
    targets = (
        _target("left", AttentionTarget.SUBJECT, 0, 150, "left-evidence", target_ref="left"),
        _target("right", AttentionTarget.SUBJECT, 150, 300, "right-evidence", target_ref="right"),
    )
    observations = (
        _observation("left-1", 60, AttentionTarget.SUBJECT, "left", "left-evidence", 0.02),
        _observation("right-1", 210, AttentionTarget.SUBJECT, "right", "right-evidence", 0.80),
    )
    config = CompositionPlannerConfig(
        maximum_velocity_per_frame=0.01,
        maximum_acceleration_per_frame_sq=0.002,
    )

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080, config=config,
    )

    metrics = plan.quality_report.metrics
    assert metrics.max_velocity_per_frame <= config.maximum_velocity_per_frame
    assert metrics.max_acceleration_per_frame_sq <= config.maximum_acceleration_per_frame_sq
    assert metrics.clipped_target_count == 0
    assert metrics.unsafe_crop_count == 0
    assert metrics.jitter_event_count == 0
    assert plan.segments[-1].geometry is not None
    assert plan.segments[-1].geometry.output_coordinate_space == "normalized_output"
