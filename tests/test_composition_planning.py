from __future__ import annotations

import pytest

from app.composition_planning import (
    CompositionPlannerConfig,
    TargetObservation,
    _AtomicState,
    _TargetState,
    _calibrate_layout_timeline,
    _containment,
    _must_keep_core,
    _next_character_fallback_state,
    _quality_report,
    _safe_area_containment,
    _subject_safe_area_containment,
    build_composition_plan,
)
from app.creative_contracts import (
    AttentionTarget,
    CompositionCropKeyframe,
    CompositionSegmentPlan,
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
    duration_frames: int = 300,
) -> CreativeIntent:
    refs = sorted({ref for item in (*targets, *motion) for ref in item.evidence_refs})
    evidence = tuple(
        EvidenceItem(
            evidence_ref=ref,
            evidence_kind="visual",
            source=SourceInterval.from_seconds(0, duration_frames / 30),
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
                source=SourceInterval.from_seconds(0, duration_frames / 30),
                output=OutputInterval(start_frame=0, end_frame=duration_frames),
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
    config = CompositionPlannerConfig()
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
        config=config,
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
    group_segment = plan.segments[2]
    member_cores = tuple(
        _must_keep_core(observation.bounds, config)
        for observation in observations[-2:]
    )
    assert group_segment.target_bounds in member_cores
    assert any(
        _subject_safe_area_containment(core, group_segment.crop, config)
        >= config.minimum_target_containment
        for core in member_cores
    )
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


def test_short_a_b_a_handoff_suppresses_pan_ping_pong_without_hidden_snap() -> None:
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

    assert [segment.target_ref for segment in plan.segments[:3]] == ["speaker-a", "speaker-b", "speaker-a"]
    assert plan.segments[1].movement_reason == "target_switch"
    assert all(segment.camera_phase == "HOLD" for segment in plan.segments)
    assert len({segment.crop for segment in plan.segments[:3]}) == 1
    assert all(segment.fallback != "fit_background" for segment in plan.segments)
    assert plan.quality_report.status == "BLOCKED"
    assert "SHOT_REVERSAL_SUPPRESSED:30" in plan.diagnostics
    assert "TARGET_TRACK_ARRIVAL_INCOMPLETE:30" in plan.diagnostics
    assert "TARGET_SWITCH_TIMELY:30" in plan.diagnostics
    assert "TARGET_SWITCH_TIMELY:60" in plan.diagnostics


def test_confident_target_handoff_uses_a_bounded_continuous_move() -> None:
    targets = (
        _target("speaker-left", AttentionTarget.SPEAKER, 0, 60, "evidence-left", target_ref="speaker-left"),
        _target("speaker-right", AttentionTarget.SPEAKER, 60, 120, "evidence-right", target_ref="speaker-right"),
    )
    observations = (
        _observation("left", 30, AttentionTarget.SPEAKER, "speaker-left", "evidence-left", 0.04),
        _observation("right", 90, AttentionTarget.SPEAKER, "speaker-right", "evidence-right", 0.24),
    )
    config = CompositionPlannerConfig(minimum_hold_frames=120, switch_cooldown_frames=120)

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080,
        config=config,
    )

    handoff = next(item for item in plan.segments if item.output.start_frame == 60)
    assert handoff.target_ref == "speaker-right"
    assert handoff.subject_lock_state == "SWITCH_CONFIRMED"
    assert handoff.movement_reason == "target_switch"
    assert handoff.camera_phase == "MOVE"
    previous = next(item for item in plan.segments if item.output.end_frame == 60)
    assert handoff.crop_keyframes[0].crop == previous.crop
    assert handoff.crop_keyframes[-1].crop == handoff.crop
    assert _safe_area_containment(
        observations[1].bounds, handoff.crop_keyframes[-1].crop, config,
    ) >= 0.82
    centers = [key.crop.x + key.crop.width / 2 for key in handoff.crop_keyframes]
    assert centers == sorted(centers)
    assert all(key.velocity_per_frame <= config.maximum_velocity_per_frame for key in handoff.crop_keyframes)
    assert all(
        key.acceleration_per_frame_sq <= config.maximum_acceleration_per_frame_sq
        for key in handoff.crop_keyframes
    )
    safe_frames = [
        key.frame for key in handoff.crop_keyframes
        if _safe_area_containment(observations[1].bounds, key.crop, config)
        >= config.minimum_target_containment
    ]
    assert safe_frames
    assert safe_frames[0] <= handoff.output.start_frame + config.camera_move_frames
    assert all(
        _safe_area_containment(observations[1].bounds, key.crop, config)
        >= config.minimum_target_containment
        for key in handoff.crop_keyframes
        if key.frame >= safe_frames[0]
    )
    assert plan.quality_report.metrics.jitter_event_count == 0
    assert "TARGET_SWITCH_TIMELY:60" in plan.diagnostics


def test_target_handoff_that_misses_the_bounded_arrival_deadline_cannot_pass() -> None:
    targets = (
        _target(
            "speaker-left", AttentionTarget.SPEAKER, 0, 60,
            "evidence-left", target_ref="speaker-left",
        ),
        _target(
            "speaker-right", AttentionTarget.SPEAKER, 60, 180,
            "evidence-right", target_ref="speaker-right",
        ),
    )
    observations = (
        _observation(
            "left", 30, AttentionTarget.SPEAKER,
            "speaker-left", "evidence-left", 0.18,
        ),
        _observation(
            "right", 120, AttentionTarget.SPEAKER,
            "speaker-right", "evidence-right", 0.32,
        ),
    )
    config = CompositionPlannerConfig(
        minimum_hold_frames=180,
        switch_cooldown_frames=180,
        camera_move_frames=6,
        maximum_velocity_per_frame=0.003,
        maximum_acceleration_per_frame_sq=0.0006,
    )

    plan = build_composition_plan(
        _intent(targets, duration_frames=180), observations,
        source_width=1920, source_height=1080, config=config,
    )

    handoff = next(item for item in plan.segments if item.output.start_frame == 60)
    assert handoff.movement_reason == "target_switch"
    assert handoff.camera_phase == "MOVE"
    deadline = next(
        item for item in handoff.crop_keyframes
        if item.frame >= 60 + config.camera_move_frames
    )
    assert handoff.target_bounds is not None
    assert _safe_area_containment(
        handoff.target_bounds, deadline.crop, config,
    ) < config.minimum_target_containment
    assert _containment(
        handoff.target_bounds, deadline.crop,
    ) >= config.minimum_target_containment
    assert _safe_area_containment(
        handoff.target_bounds, handoff.crop_keyframes[-1].crop, config,
    ) >= config.minimum_target_containment
    assert "TARGET_TRACK_ARRIVAL_INCOMPLETE:60" in plan.diagnostics
    assert plan.quality_report.status == "BLOCKED"
    assert any(
        item.code == "COMPOSITION_TARGET_CLIPPED"
        for item in plan.quality_report.findings
    )


def test_target_handoff_does_not_create_a_safety_pan() -> None:
    targets = (
        _target("speaker-left", AttentionTarget.SPEAKER, 0, 60, "evidence-left", target_ref="speaker-left"),
        _target("speaker-right", AttentionTarget.SPEAKER, 60, 120, "evidence-right", target_ref="speaker-right"),
    )
    observations = (
        _observation("left", 30, AttentionTarget.SPEAKER, "speaker-left", "evidence-left", 0.28),
        _observation("right", 90, AttentionTarget.SPEAKER, "speaker-right", "evidence-right", 0.34),
    )
    config = CompositionPlannerConfig(minimum_hold_frames=120, switch_cooldown_frames=120)

    plan = build_composition_plan(
        _intent(targets), observations, source_width=1920, source_height=1080, config=config,
    )

    handoff = next(item for item in plan.segments if item.output.start_frame == 60)
    assert handoff.target_ref == "speaker-right"
    assert handoff.subject_lock_state == "SWITCH_CONFIRMED"
    assert handoff.camera_phase == "HOLD"
    assert len(handoff.crop_keyframes) == 1
    assert _containment(observations[1].bounds, handoff.crop) == pytest.approx(1)
    assert _safe_area_containment(observations[1].bounds, handoff.crop, config) >= 0.82


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
    assert evidence_segment.target_ref in {None, "person-left", "group"}
    assert evidence_segment.geometry is not None
    assert plan.quality_report.status in {"BLOCKED", "PASS_WITH_WARNINGS"}
    assert plan.quality_report.metrics.layout_switch_count == 0
    assert plan.diagnostics


def test_real_series_uncontainable_track_reports_the_worst_crop_keyframe() -> None:
    """Regression for candidate-chapter-056-story-001 at output frame 262.

    Its 1920x960 source admits at most a 0.28125-wide true 9:16 crop, while
    the persisted person box is 0.32 wide.  The first pan keyframe clips more
    of the person than the final keyframe, so checking only ``segment.crop``
    misreports both the severity and the frame to investigate.
    """
    bounds = NormalizedRect(x=0.203, y=0.164, width=0.32, height=0.52)
    first_crop = NormalizedRect(x=0.36875, y=0, width=0.28125, height=1)
    final_crop = NormalizedRect(x=0.34375, y=0, width=0.28125, height=1)
    segment = CompositionSegmentPlan(
        segment_id="composition-010",
        output=OutputInterval(start_frame=262, end_frame=266),
        layout=LayoutFamily.STABLE_SPEAKER,
        target=AttentionTarget.SPEAKER,
        target_ref="speaker-series",
        crop=final_crop,
        target_bounds=bounds,
        crop_keyframes=(
            CompositionCropKeyframe(frame=262, crop=first_crop),
            CompositionCropKeyframe(frame=265, crop=final_crop),
        ),
    )

    config = CompositionPlannerConfig()
    report = _quality_report(
        (segment,), (), 0, 1, _intent(()), config,
    )

    finding = next(item for item in report.findings if item.code == "COMPOSITION_TARGET_CLIPPED")
    assert report.status == "BLOCKED"
    assert finding.segment_id == "composition-010"
    assert finding.measured_value == pytest.approx(
        _subject_safe_area_containment(bounds, first_crop, config)
    )
    assert "output frame 262" in finding.message
    assert report.metrics.clipped_target_count == 1


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
        (0, 8), (8, 9), (9, 98), (98, 300),
    ]
    acquired = plan.segments[1]
    held = plan.segments[2]
    released = plan.segments[3]
    assert acquired.target == AttentionTarget.SPEAKER
    assert acquired.evidence_refs == ("speaker-evidence",)
    assert held.target == AttentionTarget.STABLE_SOURCE
    assert held.target_ref is None
    assert held.subject_lock_state == "EVIDENCE_GAP"
    assert held.fallback == "stable_source"
    assert held.crop != NormalizedRect(x=0, y=0, width=1, height=1)
    assert held.geometry is not None
    assert not held.geometry.target_regions
    assert released.target == AttentionTarget.STABLE_SOURCE
    assert released.target_ref is None
    assert released.evidence_refs == ()
    assert released.fallback == "wider_crop"
    assert released.layout == acquired.layout
    assert released.layout != LayoutFamily.FIT_BACKGROUND
    assert held.crop == released.crop
    assert held.crop.x == pytest.approx(acquired.crop.x, abs=1e-7)
    assert held.crop.y == pytest.approx(acquired.crop.y, abs=1e-7)
    assert held.crop.width == pytest.approx(acquired.crop.width, abs=1e-7)
    assert held.crop.height == pytest.approx(acquired.crop.height, abs=1e-7)
    assert released.crop.x > 0.4
    observation_center = observation.bounds.x + observation.bounds.width / 2
    assert released.crop.x <= observation_center <= released.crop.x + released.crop.width
    assert not released.geometry.target_regions
    assert "STALE_TARGET_RELEASED_TO_WIDE_CROP:98" in plan.diagnostics
    assert "SHOT_SUBJECT_LOCK:0:speaker-a" in plan.diagnostics


def test_expired_stale_crop_uses_the_next_trusted_character_without_claiming_identity() -> None:
    targets = (
        _target(
            "series-left", AttentionTarget.SPEAKER, 274, 278, "left-evidence",
            target_ref="coarse-scene-speaker",
        ),
        _target(
            "series-right", AttentionTarget.SPEAKER, 562, 566, "right-evidence",
            target_ref="coarse-scene-speaker",
        ),
    )
    observations = (
        _observation(
            "left-character", 275, AttentionTarget.SPEAKER,
            "coarse-scene-speaker", "left-evidence", 0.28,
        ),
        _observation(
            "right-character", 563, AttentionTarget.SPEAKER,
            "coarse-scene-speaker", "right-evidence", 0.48,
        ),
    )
    config = CompositionPlannerConfig(subject_occlusion_hold_frames=90)

    plan = build_composition_plan(
        _intent(targets, duration_frames=700), observations,
        source_width=854, source_height=428, config=config,
    )

    released = next(item for item in plan.segments if item.output.start_frame == 365)
    reacquired = next(item for item in plan.segments if item.output.start_frame == 562)
    future_center = observations[1].bounds.x + observations[1].bounds.width / 2
    assert released.target == AttentionTarget.STABLE_SOURCE
    assert released.target_ref is None
    assert released.evidence_refs == ()
    assert released.fallback == "wider_crop"
    assert released.camera_mode == "GROUP"
    assert released.camera_phase == "MOVE"
    previous = next(item for item in plan.segments if item.output.end_frame == 365)
    assert released.crop_keyframes[0].frame == 365
    assert released.crop_keyframes[0].crop == previous.crop
    assert released.crop_keyframes[-1].crop == released.crop
    assert released.crop.x <= future_center <= released.crop.x + released.crop.width
    assert released.crop.x == pytest.approx(reacquired.crop.x, abs=1e-7)
    assert released.crop.width == pytest.approx(reacquired.crop.width, abs=1e-7)
    centers = [key.crop.x + key.crop.width / 2 for key in released.crop_keyframes]
    assert centers == sorted(centers)
    assert len({(key.crop.y, key.crop.width, key.crop.height) for key in released.crop_keyframes}) == 1
    assert reacquired.movement_reason == "target_switch"
    assert plan.quality_report.metrics.jitter_event_count == 0
    assert 0 < plan.quality_report.metrics.max_velocity_per_frame <= config.maximum_velocity_per_frame
    assert "STALE_TARGET_RELEASED_TO_FUTURE_CHARACTER:365" in plan.diagnostics


def test_future_character_fallback_never_looks_across_its_observation_scene_cut() -> None:
    decision = _target(
        "future-speaker", AttentionTarget.SPEAKER, 190, 210, "future-evidence",
        target_ref="future-speaker",
    )
    observation = _observation(
        "future-character", 200, AttentionTarget.SPEAKER,
        "future-speaker", "future-evidence", 0.72,
    ).model_copy(update={"scene_id": "scene-after-cut"})
    future = _TargetState(
        decision=decision,
        target_ref="future-speaker",
        bounds=observation.bounds,
        confidence=observation.effective_confidence,
        observations=(observation,),
    )
    work = (
        (OutputInterval(start_frame=0, end_frame=100), None),
        (OutputInterval(start_frame=190, end_frame=210), future),
    )

    assert _next_character_fallback_state(
        work, 0, frozenset({0}), CompositionPlannerConfig().enter_confidence,
    ) == future
    assert _next_character_fallback_state(
        work, 0, frozenset({0, 200}), CompositionPlannerConfig().enter_confidence,
    ) is None


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
        source_cut_frames=(9,),
    )

    after_cut = next(item for item in plan.segments if item.output.start_frame == 9)
    assert after_cut.movement_reason == "scene_reset"
    assert after_cut.fallback == "stable_source"
    assert after_cut.layout == LayoutFamily.STABLE_SPEAKER
    assert after_cut.crop != NormalizedRect(x=0, y=0, width=1, height=1)
    assert after_cut.crop_keyframes[0].frame == 9
    assert after_cut.crop_keyframes[0].crop == after_cut.crop
    assert after_cut.crop_keyframes[0].reason == "scene_reset"
    assert "SOURCE_CUT_RESET:9" in plan.diagnostics
    assert "STABLE_CROP_HELD:9" not in plan.diagnostics


def test_contiguous_edit_map_partition_does_not_reset_tracking() -> None:
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

    after_partition = next(item for item in plan.segments if item.output.start_frame == 9)
    before_partition = next(item for item in plan.segments if item.output.end_frame == 9)
    assert after_partition.movement_reason == "safe_fallback"
    assert after_partition.crop_keyframes[0].reason == "static"
    assert after_partition.crop_keyframes[0].crop == before_partition.crop
    assert "SOURCE_CUT_RESET:9" not in plan.diagnostics


def test_discontinuous_edit_map_partition_resets_the_camera() -> None:
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
                source=SourceInterval.from_seconds(1, 10.7),
                output=OutputInterval(start_frame=9, end_frame=300),
            ),
        )),
    })

    plan = build_composition_plan(
        intent,
        (_observation(
            "speaker-frame", 8, AttentionTarget.SPEAKER,
            "speaker-a", "speaker-evidence", 0.55,
        ),),
        source_width=1920,
        source_height=1080,
        config=CompositionPlannerConfig(minimum_hold_frames=45),
    )

    after_cut = next(item for item in plan.segments if item.output.start_frame == 9)
    assert after_cut.movement_reason == "scene_reset"
    assert after_cut.crop_keyframes == (CompositionCropKeyframe(
        frame=9,
        crop=after_cut.crop,
        reason="scene_reset",
        camera_phase="HOLD",
    ),)
    assert "SOURCE_CUT_RESET:9" in plan.diagnostics


def test_source_cut_prefix_starts_on_the_first_trusted_shot_framing() -> None:
    targets = (
        _target(
            "speaker-a", AttentionTarget.SPEAKER, 0, 60, "evidence-a",
            target_ref="speaker-a",
        ),
        _target(
            "speaker-b", AttentionTarget.SPEAKER, 66, 120, "evidence-b",
            target_ref="speaker-b",
        ),
    )
    observations = (
        _observation(
            "person-a", 30, AttentionTarget.SPEAKER,
            "speaker-a", "evidence-a", 0.08,
        ),
        _observation(
            "person-b", 90, AttentionTarget.SPEAKER,
            "speaker-b", "evidence-b", 0.74,
        ),
    )

    plan = build_composition_plan(
        _intent(targets), observations,
        source_width=1920,
        source_height=1080,
        source_cut_frames=(60,),
    )

    before_cut = next(item for item in plan.segments if item.output.end_frame == 60)
    prefix = next(item for item in plan.segments if item.output.start_frame == 60)
    acquired = next(item for item in plan.segments if item.output.start_frame == 66)
    assert prefix.target == AttentionTarget.STABLE_SOURCE
    assert prefix.target_ref is None
    assert prefix.evidence_refs == ()
    assert prefix.camera_phase == "HOLD"
    assert prefix.crop_keyframes[0].reason == "scene_reset"
    assert prefix.crop == acquired.crop
    assert prefix.crop != before_cut.crop
    assert "SOURCE_CUT_RESET:60" in plan.diagnostics


def test_source_cut_prefix_never_reacquires_an_observation_from_the_previous_shot() -> None:
    targets = (
        _target(
            "speaker-a-spans-cut", AttentionTarget.SPEAKER, 0, 120, "evidence-a",
            target_ref="speaker-a", priority=50,
        ),
        _target(
            "speaker-b-after-cut", AttentionTarget.SPEAKER, 66, 120, "evidence-b",
            target_ref="speaker-b", priority=90,
        ),
    )
    observations = (
        _observation(
            "person-a-old-shot", 30, AttentionTarget.SPEAKER,
            "speaker-a", "evidence-a", 0.08, confidence=0.99,
        ),
        _observation(
            "person-b-new-shot", 90, AttentionTarget.SPEAKER,
            "speaker-b", "evidence-b", 0.74,
        ),
    )

    plan = build_composition_plan(
        _intent(targets), observations,
        source_width=1920,
        source_height=1080,
        source_cut_frames=(60,),
    )

    before_cut = next(item for item in plan.segments if item.output.end_frame == 60)
    prefix = next(item for item in plan.segments if item.output.start_frame == 60)
    acquired = next(item for item in plan.segments if item.output.start_frame == 66)
    assert before_cut.target_ref == "speaker-a"
    assert prefix.target == AttentionTarget.STABLE_SOURCE
    assert prefix.target_ref is None
    assert prefix.evidence_refs == ()
    assert prefix.crop_keyframes[0].reason == "scene_reset"
    assert prefix.crop == acquired.crop
    assert prefix.crop != before_cut.crop
    assert acquired.target_ref == "speaker-b"


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
        config=CompositionPlannerConfig(
            minimum_hold_frames=1,
            subject_occlusion_hold_frames=300,
        ),
    )

    by_start = {segment.output.start_frame: segment for segment in plan.segments}
    assert by_start[90].target_ref is None
    assert by_start[90].fallback == "wider_crop"
    assert by_start[180].target_ref == "speaker-a"
    assert "TARGET_REFRAME_CONFIRMED:180" in plan.diagnostics
    assert sum(item.camera_phase == "MOVE" for item in by_start.values()) <= 1
    assert all(item.camera_mode != "TRACKING" for item in by_start.values())


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
        config=CompositionPlannerConfig(
            minimum_hold_frames=1,
            subject_occlusion_hold_frames=300,
        ),
    )

    by_start = {segment.output.start_frame: segment for segment in plan.segments}
    assert by_start[90].target_ref is None
    assert by_start[180].target_ref is None
    assert by_start[90].fallback == by_start[180].fallback == "wider_crop"
    assert sum(item.camera_phase == "MOVE" for item in by_start.values()) <= 1
    assert plan.quality_report.metrics.jitter_event_count == 0
    assert plan.quality_report.metrics.target_switch_count == 0
    assert "TARGET_REFRAME_CONFIRMED:180" not in plan.diagnostics


def test_stale_same_ref_reacquisition_immediately_frames_the_new_person() -> None:
    targets = (
        _target(
            "speaker-old", AttentionTarget.SPEAKER, 0, 60, "old-evidence",
            target_ref="scene-level-speaker",
        ),
        _target(
            "speaker-new", AttentionTarget.SPEAKER, 150, 210, "new-evidence",
            target_ref="scene-level-speaker",
        ),
    )
    observations = (
        _observation(
            "old-person", 30, AttentionTarget.SPEAKER,
            "scene-level-speaker", "old-evidence", 0.08,
        ),
        _observation(
            "new-person", 180, AttentionTarget.SPEAKER,
            "scene-level-speaker", "new-evidence", 0.74,
        ),
    )
    config = CompositionPlannerConfig(
        minimum_hold_frames=240,
        switch_cooldown_frames=240,
        subject_occlusion_hold_frames=90,
    )

    plan = build_composition_plan(
        _intent(targets), observations,
        source_width=1920, source_height=1080, config=config,
    )

    handoff = next(item for item in plan.segments if item.output.start_frame == 150)
    assert handoff.target_ref == "scene-level-speaker"
    assert handoff.movement_reason == "target_switch"
    assert handoff.subject_lock_state == "SWITCH_CONFIRMED"
    assert handoff.camera_phase == "MOVE"
    previous = next(item for item in plan.segments if item.output.end_frame == 150)
    assert handoff.crop_keyframes[0].crop == previous.crop
    assert handoff.target_bounds is not None
    assert _subject_safe_area_containment(
        handoff.target_bounds, handoff.crop_keyframes[-1].crop, config,
    ) >= config.minimum_target_containment
    new_center = observations[1].bounds.x + observations[1].bounds.width / 2
    assert handoff.crop.x <= new_center <= handoff.crop.x + handoff.crop.width
    assert "STALE_TARGET_RELEASED:150" in plan.diagnostics
    assert "TARGET_REFRAME_HELD:150" not in plan.diagnostics


def test_same_position_after_a_long_gap_is_not_a_false_target_switch() -> None:
    targets = (
        _target(
            "speaker-old", AttentionTarget.SPEAKER, 0, 30, "old-evidence",
            target_ref="scene-level-speaker",
        ),
        _target(
            "speaker-return", AttentionTarget.SPEAKER, 150, 180, "new-evidence",
            target_ref="scene-level-speaker",
        ),
    )
    observations = (
        _observation(
            "old-position", 15, AttentionTarget.SPEAKER,
            "scene-level-speaker", "old-evidence", 0.30,
        ),
        _observation(
            "same-position", 165, AttentionTarget.SPEAKER,
            "scene-level-speaker", "new-evidence", 0.30,
        ),
    )

    plan = build_composition_plan(
        _intent(targets), observations,
        source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(subject_occlusion_hold_frames=90),
    )

    reacquired = next(item for item in plan.segments if item.output.start_frame == 150)
    assert reacquired.target_ref == "scene-level-speaker"
    assert reacquired.movement_reason == "target_acquired"
    assert plan.quality_report.metrics.target_switch_count == 0
    assert not any(
        item.startswith("STALE_TARGET_RELEASED:") for item in plan.diagnostics
    )


def test_conflicting_pending_reframes_keep_the_first_character_crop_until_release() -> None:
    targets = (
        _target(
            "speaker-a", AttentionTarget.SPEAKER, 0, 90, "evidence-a",
            target_ref="coarse-scene-speaker",
        ),
        _target(
            "speaker-b", AttentionTarget.SPEAKER, 90, 180, "evidence-b",
            target_ref="coarse-scene-speaker",
        ),
        _target(
            "speaker-c", AttentionTarget.SPEAKER, 180, 270, "evidence-c",
            target_ref="coarse-scene-speaker",
        ),
        _target(
            "speaker-d", AttentionTarget.SPEAKER, 420, 440, "evidence-d",
            target_ref="coarse-scene-speaker",
        ),
    )
    observations = (
        _observation(
            "person-a", 30, AttentionTarget.SPEAKER,
            "coarse-scene-speaker", "evidence-a", 0.08,
        ),
        _observation(
            "person-b", 120, AttentionTarget.SPEAKER,
            "coarse-scene-speaker", "evidence-b", 0.62,
        ),
        _observation(
            "person-c", 210, AttentionTarget.SPEAKER,
            "coarse-scene-speaker", "evidence-c", 0.32,
        ),
        _observation(
            "person-d", 425, AttentionTarget.SPEAKER,
            "coarse-scene-speaker", "evidence-d", 0.80,
        ),
    )
    gap_boundary = ResolvedMotionEvent(
        decision_id="gap-boundary",
        source=SourceInterval.from_seconds(12, 13),
        output=OutputInterval(start_frame=360, end_frame=390),
        confidence=0.9,
        evidence_refs=("evidence-d",),
        purpose=MotionPurpose.EVIDENCE_REVEAL,
        domain=MotionDomain.COMPOSITION,
        intensity=Intensity.BALANCED,
    )
    config = CompositionPlannerConfig(
        minimum_hold_frames=1,
        subject_occlusion_hold_frames=300,
    )

    plan = build_composition_plan(
        _intent(targets, motion=(gap_boundary,), duration_frames=450), observations,
        source_width=1920, source_height=1080, config=config,
    )

    by_start = {segment.output.start_frame: segment for segment in plan.segments}
    held = tuple(by_start[frame] for frame in (90, 180, 270, 330, 360, 390))
    assert len({item.crop for item in held}) == 1
    assert all(item.target_ref is None for item in held)
    assert sum(item.camera_phase == "MOVE" for item in held) <= 1
    assert plan.quality_report.metrics.jitter_event_count == 0
    person_b_center = observations[1].bounds.x + observations[1].bounds.width / 2
    assert held[0].crop.x <= person_b_center <= held[0].crop.x + held[0].crop.width
    assert by_start[420].movement_reason == "target_switch"
    assert plan.quality_report.metrics.target_switch_count == 1
    assert "TARGET_REFRAME_CONFLICT_SUPPRESSED:180" in plan.diagnostics
    assert "PENDING_REFRAME_RELEASED_TO_WIDE_CROP:330" in plan.diagnostics


def test_confirmed_target_after_sparse_gap_moves_continuously_inside_the_shot() -> None:
    targets = (
        _target(
            "speaker-a", AttentionTarget.SPEAKER, 0, 30, "evidence-a",
            target_ref="speaker-a",
        ),
        _target(
            "speaker-b", AttentionTarget.SPEAKER, 150, 180, "evidence-b",
            target_ref="speaker-b",
        ),
    )
    observations = (
        _observation(
            "person-a", 15, AttentionTarget.SPEAKER,
            "speaker-a", "evidence-a", 0.08,
        ),
        _observation(
            "person-b", 165, AttentionTarget.SPEAKER,
            "speaker-b", "evidence-b", 0.74,
        ),
    )

    plan = build_composition_plan(
        _intent(targets), observations,
        source_width=1920, source_height=1080,
        config=CompositionPlannerConfig(
            minimum_hold_frames=240, switch_cooldown_frames=240,
        ),
    )

    handoff = next(item for item in plan.segments if item.output.start_frame == 150)
    previous = next(item for item in plan.segments if item.output.end_frame == 150)
    assert handoff.target_ref == "speaker-b"
    assert handoff.movement_reason == "target_switch"
    assert handoff.camera_phase == "MOVE"
    assert len(handoff.crop_keyframes) > 1
    assert handoff.crop_keyframes[0].crop == previous.crop
    assert handoff.crop_keyframes[-1].crop == handoff.crop
    assert handoff.target_bounds is not None
    assert _subject_safe_area_containment(
        handoff.target_bounds, handoff.crop, CompositionPlannerConfig(),
    ) >= CompositionPlannerConfig().minimum_target_containment
    assert "SHOT_SUBJECT_LOCK:0:speaker-a" in plan.diagnostics
    assert "SHOT_SUBJECT_LOCK:150:speaker-b" not in plan.diagnostics
    assert "TARGET_SWITCH_TIMELY:150" in plan.diagnostics
    assert plan.quality_report.metrics.jitter_event_count == 0


def test_pending_series_reframe_uses_a_character_bearing_safe_crop() -> None:
    targets = (
        _target(
            "series-old", AttentionTarget.SPEAKER, 0, 60, "series-evidence",
            target_ref="coarse-scene-speaker",
        ),
        _target(
            "series-new", AttentionTarget.SPEAKER, 98, 102, "series-evidence",
            target_ref="coarse-scene-speaker",
        ),
    )
    old = _observation(
        "series-old-face", 32, AttentionTarget.SPEAKER,
        "coarse-scene-speaker", "series-evidence", 0.45, width=0.32,
    ).model_copy(update={
        "bounds": NormalizedRect(x=0.45, y=0.09, width=0.32, height=0.52),
    })
    new = _observation(
        "series-new-face", 99, AttentionTarget.SPEAKER,
        "coarse-scene-speaker", "series-evidence", 0.21, width=0.32,
    ).model_copy(update={
        "bounds": NormalizedRect(x=0.21, y=0.05, width=0.32, height=0.52),
    })
    config = CompositionPlannerConfig(subject_occlusion_hold_frames=90)

    plan = build_composition_plan(
        _intent(targets), (old, new),
        source_width=854, source_height=428, config=config,
    )

    pending = next(item for item in plan.segments if item.output.contains(
        OutputInterval(start_frame=99, end_frame=100),
    ))
    pending_core = NormalizedRect(
        x=0.2772, y=0.0656, width=0.1856, height=0.3224,
    )
    assert pending.target == AttentionTarget.STABLE_SOURCE
    assert pending.fallback == "wider_crop"
    assert pending.camera_mode == "GROUP"
    assert pending.camera_phase == "MOVE"
    previous = next(item for item in plan.segments if item.output.end_frame == pending.output.start_frame)
    assert pending.crop_keyframes[0].crop == previous.crop
    settled = next(item for item in plan.segments if item.output.start_frame == 122)
    assert _subject_safe_area_containment(
        pending_core, settled.crop, config,
    ) >= config.minimum_target_containment
    assert all(
        key.velocity_per_frame <= config.maximum_velocity_per_frame
        and key.acceleration_per_frame_sq <= config.maximum_acceleration_per_frame_sq
        for item in plan.segments if item.camera_phase == "MOVE"
        for key in item.crop_keyframes
    )
    assert plan.quality_report.metrics.jitter_event_count == 0
    assert all(item.fallback != "fit_background" for item in plan.segments)
    assert "PENDING_REFRAME_GROUP_CROP:98" in plan.diagnostics


def test_protected_wide_group_is_not_dropped_from_a_speaker_scene() -> None:
    targets = (
        _target(
            "series-speaker", AttentionTarget.SPEAKER, 0, 90, "speaker-evidence",
            target_ref="series-speaker", layouts=(LayoutFamily.STABLE_SPEAKER,),
        ),
        _target(
            "series-group", AttentionTarget.GROUP, 147, 151, "group-evidence",
            target_ref="series-group", layouts=(LayoutFamily.WIDE_GROUP,),
        ),
    )
    speaker = _observation(
        "speaker", 30, AttentionTarget.SPEAKER,
        "series-speaker", "speaker-evidence", 0.37, width=0.24,
    )
    group = _observation(
        "group", 148, AttentionTarget.GROUP,
        "series-group", "group-evidence", 0.01, width=0.82,
    ).model_copy(update={
        "bounds": NormalizedRect(x=0.01, y=0.22, width=0.82, height=0.62),
        "confidence": 0.81,
        "protected": True,
    })
    config = CompositionPlannerConfig()

    plan = build_composition_plan(
        _intent(targets), (speaker, group),
        source_width=854, source_height=428, config=config,
    )

    framed = next(item for item in plan.segments if item.output.start_frame == 147)
    assert framed.target == AttentionTarget.GROUP
    assert framed.target_ref == "series-group"
    assert framed.target_bounds is not None
    assert framed.fallback == "wider_crop"
    assert _subject_safe_area_containment(
        framed.target_bounds, framed.crop, config,
    ) >= config.minimum_target_containment
    group_center = group.bounds.x + group.bounds.width / 2
    assert framed.crop.x <= group_center <= framed.crop.x + framed.crop.width
    assert all(item.fallback != "fit_background" for item in plan.segments)


def test_static_top_edge_face_preserves_all_available_headroom() -> None:
    target = _target(
        "top-edge-speaker", AttentionTarget.SPEAKER, 0, 90, "edge-evidence",
        target_ref="top-edge-speaker", layouts=(LayoutFamily.STABLE_SPEAKER,),
    )
    observation = _observation(
        "top-edge-face", 30, AttentionTarget.SPEAKER,
        "top-edge-speaker", "edge-evidence", 0.28, width=0.32,
    ).model_copy(update={
        "bounds": NormalizedRect(x=0.28, y=0.0, width=0.32, height=0.52),
    })
    config = CompositionPlannerConfig()

    plan = build_composition_plan(
        _intent((target,)), (observation,),
        source_width=854, source_height=428, config=config,
    )

    framed = next(item for item in plan.segments if item.target_bounds is not None)
    assert framed.crop.y == 0
    assert framed.crop.height == 1
    assert framed.target_bounds is not None
    assert _subject_safe_area_containment(
        framed.target_bounds, framed.crop, config,
    ) >= config.minimum_target_containment
    assert plan.quality_report.status != "BLOCKED"


def test_crop_track_respects_velocity_acceleration_and_reports_safe_geometry() -> None:
    targets = (
        _target("left", AttentionTarget.SUBJECT, 0, 150, "left-evidence", target_ref="left"),
        _target("right", AttentionTarget.SUBJECT, 150, 300, "right-evidence", target_ref="right"),
    )
    observations = (
        _observation("left-1", 60, AttentionTarget.SUBJECT, "left", "left-evidence", 0.02),
        _observation("right-1", 210, AttentionTarget.SUBJECT, "right", "right-evidence", 0.24),
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
    for segment in plan.segments:
        if segment.target_bounds is None:
            continue
        assessed = (
            segment.crop_keyframes[-1:]
            if segment.movement_reason in {"target_acquired", "target_switch"}
            and segment.camera_phase == "MOVE"
            else segment.crop_keyframes
        )
        assert all(
            _safe_area_containment(segment.target_bounds, keyframe.crop, config)
            >= config.minimum_target_containment
            for keyframe in assessed
        )
