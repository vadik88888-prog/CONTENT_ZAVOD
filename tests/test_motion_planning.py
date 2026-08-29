from __future__ import annotations

import pytest

from app.creative_execution import CAPTION_RENDER_BACKEND_VERSION, caption_plan_with_motion
from app.creative_contracts import (
    BackendAssignment,
    CaptionCuePlan,
    CaptionPlan,
    CaptionWordPlan,
    CanvasPlan,
    CompositionPlan,
    CompositionPunchIn,
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
    ResolvedMotionEvent,
    SourceBRollPlan,
    SourceBRollSegmentPlan,
    SourceInterval,
    SourceOutputTimeMap,
    compile_render_plan,
)
from app.motion_planning import (
    MotionCapability,
    MotionCapabilityRegistry,
    build_motion_plan,
)


def _event(
    event_id: str,
    start: int,
    end: int,
    purpose: MotionPurpose,
    domain: MotionDomain,
    *,
    intensity: Intensity = Intensity.HIGH,
) -> ResolvedMotionEvent:
    return ResolvedMotionEvent(
        decision_id=event_id,
        source=SourceInterval.from_seconds(start / 30, end / 30),
        output=OutputInterval(start_frame=start, end_frame=end),
        confidence=0.95,
        evidence_refs=(f"evidence-{event_id}",),
        purpose=purpose,
        domain=domain,
        intensity=intensity,
    )


def _intent(
    events: tuple[ResolvedMotionEvent, ...],
    *,
    intensity: Intensity = Intensity.BALANCED,
    reduced_motion: bool = False,
    duration_frames: int = 300,
) -> CreativeIntent:
    evidence = tuple(
        EvidenceItem(
            evidence_ref=item.evidence_refs[0],
            evidence_kind="visual",
            source=SourceInterval.from_seconds(0, duration_frames / 30),
            confidence=0.95,
            artifact_fingerprint=(str(index + 1) * 64)[:64],
            provenance=f"vision:{item.decision_id}",
        )
        for index, item in enumerate(events)
    )
    return CreativeIntent(
        intent_id="intent-motion",
        revision=1,
        production_plan=ImmutableProductionPlanLink(
            plan_id="plan-motion",
            plan_fingerprint="a" * 64,
            identity=ImmutableProductionIdentity(
                project_id="project-1", run_id="run-1", analysis_id="analysis-1",
                candidate_id="candidate-1", source_id="source-1",
            ),
        ),
        source_output_mapping=SourceOutputTimeMap(segments=(EditMapSegment(
            map_id="edit-motion",
            source=SourceInterval.from_seconds(0, duration_frames / 30),
            output=OutputInterval(start_frame=0, end_frame=duration_frames),
        ),)),
        evidence_fingerprint="b" * 64,
        evidence_manifest=evidence,
        proposal_hash="c" * 64,
        policy=CreativePolicy(
            preset_id="editorial", preset_version="1", platform="universal",
            intensity=intensity, reduced_motion=reduced_motion,
        ),
        confidence=0.95,
        motion_events=events,
    )


def _cue(cue_id: str, start: int, end: int, text: str = "Readable caption") -> CaptionCuePlan:
    return CaptionCuePlan(
        cue_id=cue_id,
        output=OutputInterval(start_frame=start, end_frame=end),
        resolved_lines=(text,),
        lane="lower",
        typography_token_id="clean-v1",
        timing_mode="word",
        timing_confidence=0.95,
    )


def _single_word_cue(*, fallback_reason: str | None = None) -> CaptionCuePlan:
    output = OutputInterval(start_frame=0, end_frame=60)
    word = CaptionWordPlan(
        word_id="word-protected",
        text="Readable",
        output=output,
        timing_source="verified",
        confidence=0.95,
        source=SourceInterval.from_seconds(0, 2),
        mapping_segment_ids=("edit-motion",),
    )
    return _cue("protected-cue", 0, 60, "Readable").model_copy(update={
        "words": (word,),
        "display_mode": "single_spoken_word",
        "primitive_id": "static" if fallback_reason else "word_pop",
        "motion_duration_frames": 0 if fallback_reason else 9,
        "scale_keyframes": (100, 100, 100) if fallback_reason else (88, 112, 100),
        "fallback_reason": fallback_reason,
    })


def _captions(intent: CreativeIntent, *cues: CaptionCuePlan) -> CaptionPlan:
    return CaptionPlan(intent_id=intent.intent_id, cues=tuple(cues))


def _composition(
    intent: CreativeIntent,
    punches: tuple[ResolvedMotionEvent, ...] = (),
) -> CompositionPlan:
    segments = tuple(
        CompositionSegmentPlan(
            segment_id=f"segment-{event.decision_id}",
            output=event.output,
            layout=LayoutFamily.FIT_BACKGROUND,
            crop=NormalizedRect(x=0, y=0, width=1, height=1),
            movement_reason="editorial_punch_in",
            punch_in=CompositionPunchIn(
                event_id=event.decision_id,
                output=event.output,
                scale=1.06,
                evidence_refs=event.evidence_refs,
            ),
            evidence_refs=event.evidence_refs,
        )
        for event in punches
    )
    return CompositionPlan(intent_id=intent.intent_id, segments=segments)


def _broll(intent: CreativeIntent, event: ResolvedMotionEvent | None = None) -> SourceBRollPlan:
    segments = () if event is None else (SourceBRollSegmentPlan(
        segment_id=f"broll-{event.decision_id}",
        destination=event.output,
        source_cutaway=SourceInterval.from_seconds(20, 21),
        story_unit_id="story-1",
        evidence_refs=(event.evidence_refs[0],),
        transition="short_dissolve",
    ),)
    return SourceBRollPlan(intent_id=intent.intent_id, segments=segments)


def test_calm_timeline_has_no_periodic_or_random_motion() -> None:
    intent = _intent(())
    plan = build_motion_plan(
        intent, _captions(intent, _cue("cue-calm", 0, 300)),
        _composition(intent), _broll(intent),
    )

    assert plan.schema_version == "7F.motion-plan.1"
    assert plan.events == ()
    assert plan.quality_report.status == "PASS"
    assert plan.quality_report.metrics.requested_event_count == 0


def test_high_intensity_is_a_ceiling_not_an_animation_quota() -> None:
    intent = _intent((), intensity=Intensity.HIGH)

    plan = build_motion_plan(
        intent, _captions(intent, _cue("gameplay-calm", 0, 300)),
        _composition(intent), _broll(intent),
    )

    assert plan.events == ()
    assert plan.animation_budget.points_used == 0
    assert plan.animation_budget.animated_frames_used == 0


@pytest.mark.parametrize(
    "protected_cue",
    (
        _single_word_cue(),
        _cue("protected-cue", 0, 60).model_copy(update={
            "primitive_id": "static",
            "timing_mode": "phrase",
            "fallback_reason": "weak_timing",
        }),
        _single_word_cue(fallback_reason="short_timing"),
    ),
)
def test_generic_motion_cannot_replace_word_pop_or_timing_safe_fallback(
    protected_cue: CaptionCuePlan,
) -> None:
    event = _event("caption-hook", 0, 60, MotionPurpose.HOOK, MotionDomain.CAPTION)
    intent = _intent((event,))
    captions = _captions(intent, protected_cue)
    motion = build_motion_plan(intent, captions, _composition(intent), _broll(intent))

    effective = caption_plan_with_motion(captions, motion)

    assert motion.events
    assert effective.cues == captions.cues


def test_generic_motion_still_applies_to_normal_phrase_caption() -> None:
    event = _event("caption-hook", 0, 60, MotionPurpose.HOOK, MotionDomain.CAPTION)
    intent = _intent((event,))
    captions = _captions(intent, _cue("normal-cue", 0, 60))
    motion = build_motion_plan(intent, captions, _composition(intent), _broll(intent))

    effective = caption_plan_with_motion(captions, motion)

    assert motion.events
    assert effective.cues[0].primitive_id in {"fade", "scale", "slide"}
    assert effective.cues[0].primitive_id != captions.cues[0].primitive_id


def test_reduced_motion_is_reported_even_when_intent_compiler_removed_all_animation_requests() -> None:
    intent = _intent((), reduced_motion=True)

    plan = build_motion_plan(
        intent, _captions(intent, _cue("cue-calm", 0, 300)),
        _composition(intent), _broll(intent),
    )

    assert plan.events == ()
    assert plan.reduced_motion is True
    assert plan.quality_report.status == "PASS_WITH_WARNINGS"
    assert plan.quality_report.findings[0].code == "MOTION_REDUCED_MOTION_FALLBACK"


@pytest.mark.parametrize(
    ("intensity", "purpose", "expected"),
    [
        (Intensity.LOW, MotionPurpose.HOOK, "fade"),
        (Intensity.BALANCED, MotionPurpose.EVIDENCE_REVEAL, "scale"),
        (Intensity.HIGH, MotionPurpose.HOOK, "slide"),
    ],
)
def test_caption_fade_scale_slide_follow_editorial_intensity(intensity, purpose, expected) -> None:
    event = _event("caption-event", 30, 90, purpose, MotionDomain.CAPTION)
    intent = _intent((event,), intensity=intensity)

    first = build_motion_plan(
        intent, _captions(intent, _cue("cue-1", 30, 90)), _composition(intent), _broll(intent),
    )
    second = build_motion_plan(
        intent, _captions(intent, _cue("cue-1", 30, 90)), _composition(intent), _broll(intent),
    )

    assert first == second
    assert first.events[0].primitive_id == expected
    assert first.events[0].intensity == intensity
    assert first.events[0].target_plan_ids == ("cue-1",)
    assert first.animation_budget.points_used <= first.animation_budget.point_limit


def test_controlled_punch_in_uses_existing_composition_geometry() -> None:
    event = _event(
        "product-reveal", 90, 120, MotionPurpose.EVIDENCE_REVEAL, MotionDomain.COMPOSITION,
    )
    intent = _intent((event,))

    plan = build_motion_plan(
        intent, _captions(intent), _composition(intent, (event,)), _broll(intent),
    )

    motion = plan.events[0]
    assert motion.primitive_id == "punch_in"
    assert motion.scale_from == 1 and motion.scale_to == 1.06
    assert motion.output == event.output
    assert motion.duration_frames == 14


def test_gameplay_one_or_two_frame_motion_is_static_not_flicker() -> None:
    event = _event(
        "gameplay-hit", 120, 122, MotionPurpose.EVIDENCE_REVEAL, MotionDomain.COMPOSITION,
    )
    intent = _intent((event,), intensity=Intensity.HIGH)

    plan = build_motion_plan(
        intent, _captions(intent), _composition(intent, (event,)), _broll(intent),
    )

    assert len(plan.events) == 1
    assert plan.events[0].primitive_id == "static"
    assert plan.events[0].duration_frames == 0
    assert plan.events[0].fallback_reason == "short_event"
    assert plan.animation_budget.points_used == 0
    assert plan.animation_budget.animated_frames_used == 0
    assert "MOTION_SHORT_EVENT_FALLBACK" in {
        finding.code for finding in plan.quality_report.findings
    }


def test_broll_transition_policy_uses_cut_for_low_and_short_dissolve_otherwise() -> None:
    event = _event("cutaway", 60, 120, MotionPurpose.EVIDENCE_REVEAL, MotionDomain.TRANSITION)
    low_intent = _intent((event,), intensity=Intensity.LOW)
    balanced_intent = _intent((event,), intensity=Intensity.BALANCED)

    low = build_motion_plan(
        low_intent, _captions(low_intent), _composition(low_intent), _broll(low_intent, event),
    )
    balanced = build_motion_plan(
        balanced_intent, _captions(balanced_intent), _composition(balanced_intent),
        _broll(balanced_intent, event),
    )

    assert low.events[0].primitive_id == "static"
    assert low.events[0].duration_frames == 0
    assert balanced.events[0].primitive_id == "dissolve"
    assert balanced.events[0].duration_frames == 8


def test_dense_reading_suppresses_non_caption_motion() -> None:
    event = _event(
        "dense-punch", 90, 120, MotionPurpose.EVIDENCE_REVEAL, MotionDomain.COMPOSITION,
    )
    intent = _intent((event,))
    dense = _cue("dense-cue", 90, 120, "This caption is deliberately far too dense to animate behind")

    plan = build_motion_plan(
        intent, _captions(intent, dense), _composition(intent, (event,)), _broll(intent),
    )

    assert plan.events == ()
    assert plan.quality_report.status == "PASS_WITH_WARNINGS"
    assert plan.quality_report.metrics.readability_suppression_count == 1
    assert plan.quality_report.findings[0].code == "MOTION_READABILITY_SUPPRESSED"


def test_cooldown_keeps_higher_priority_payoff_and_suppresses_claim() -> None:
    claim = _event("claim", 30, 60, MotionPurpose.CLAIM_CHANGE, MotionDomain.CAPTION)
    payoff = _event("payoff", 70, 100, MotionPurpose.PAYOFF, MotionDomain.CAPTION)
    intent = _intent((claim, payoff))

    plan = build_motion_plan(
        intent, _captions(intent, _cue("claim-cue", 30, 60), _cue("payoff-cue", 70, 100)),
        _composition(intent), _broll(intent),
    )

    assert [item.event_id for item in plan.events] == ["payoff"]
    assert plan.quality_report.metrics.cooldown_suppression_count == 1


def test_global_animation_budget_suppresses_lower_priority_events() -> None:
    events = tuple(
        _event(
            f"event-{index}", start, start + 30,
            MotionPurpose.PAYOFF if index == 3 else MotionPurpose.EVIDENCE_REVEAL,
            MotionDomain.CAPTION,
        )
        for index, start in enumerate((0, 60, 120, 180))
    )
    intent = _intent(events, duration_frames=300)
    cues = tuple(_cue(f"cue-{index}", event.output.start_frame, event.output.end_frame) for index, event in enumerate(events))

    plan = build_motion_plan(
        intent, _captions(intent, *cues), _composition(intent), _broll(intent),
    )

    assert plan.animation_budget.animated_frames_used <= plan.animation_budget.animated_frame_limit
    assert plan.quality_report.metrics.budget_suppression_count >= 1
    assert "event-3" in {item.event_id for item in plan.events}


def test_concurrency_limit_suppresses_the_lowest_priority_conflicting_layer() -> None:
    caption = _event("hook", 60, 120, MotionPurpose.HOOK, MotionDomain.CAPTION)
    punch = _event("payoff", 60, 120, MotionPurpose.PAYOFF, MotionDomain.COMPOSITION)
    transition = _event("reveal", 60, 120, MotionPurpose.EVIDENCE_REVEAL, MotionDomain.TRANSITION)
    intent = _intent((caption, punch, transition), intensity=Intensity.HIGH)

    plan = build_motion_plan(
        intent, _captions(intent, _cue("hook-cue", 60, 120, "Hook")),
        _composition(intent, (punch,)), _broll(intent, transition),
    )

    assert {item.event_id for item in plan.events} == {"hook", "payoff"}
    assert plan.quality_report.metrics.max_concurrent_layers == 2
    assert plan.quality_report.metrics.concurrency_suppression_count == 1


def test_reduced_motion_uses_only_static_or_short_caption_fade() -> None:
    caption = _event("caption-payoff", 30, 90, MotionPurpose.PAYOFF, MotionDomain.CAPTION)
    punch = _event("composition-payoff", 120, 150, MotionPurpose.PAYOFF, MotionDomain.COMPOSITION)
    intent = _intent((caption, punch), reduced_motion=True, intensity=Intensity.HIGH)

    plan = build_motion_plan(
        intent, _captions(intent, _cue("payoff-cue", 30, 90)),
        _composition(intent, (punch,)), _broll(intent),
    )

    assert {item.primitive_id for item in plan.events} == {"fade", "static"}
    assert all(item.reduced_motion_fallback for item in plan.events)
    assert plan.quality_report.metrics.fallback_count == 1


def test_missing_registry_primitive_falls_back_instead_of_executing_unknown_effect() -> None:
    event = _event("payoff-scale", 30, 90, MotionPurpose.PAYOFF, MotionDomain.CAPTION)
    intent = _intent((event,))
    registry = MotionCapabilityRegistry(entries=(
        MotionCapability(
            primitive_id="static", backend_id="none",
            domains=(MotionDomain.CAPTION, MotionDomain.COMPOSITION, MotionDomain.TRANSITION),
            fallback_primitive_id="static",
        ),
        MotionCapability(
            primitive_id="fade", backend_id="libass", domains=(MotionDomain.CAPTION,),
            fallback_primitive_id="static",
        ),
    ))

    plan = build_motion_plan(
        intent, _captions(intent, _cue("payoff-cue", 30, 90)),
        _composition(intent), _broll(intent), registry=registry,
    )

    assert plan.events[0].primitive_id == "fade"
    assert plan.events[0].requested_primitive_id == "scale"
    assert plan.quality_report.findings[0].code == "MOTION_PRIMITIVE_FALLBACK"


def test_assessed_motion_plan_crosses_compiled_renderer_handoff_and_detects_stale_inputs() -> None:
    event = _event("caption-hook", 0, 60, MotionPurpose.HOOK, MotionDomain.CAPTION)
    intent = _intent((event,))
    captions = _captions(intent, _cue("hook-cue", 0, 60))
    composition = _composition(intent)
    broll = _broll(intent)
    motion = build_motion_plan(intent, captions, composition, broll)

    compiled = compile_render_plan(
        intent, captions, composition, motion, broll, CanvasPlan(width=1080, height=1920),
    )

    assert compiled.motion_plan.quality_report.status == "PASS"
    assert "cross_domain_animation_budget" in compiled.expected_quality_constraints

    stale_captions = captions.model_copy(update={"cues": (_cue("different-cue", 0, 60),)})
    with pytest.raises(ValueError, match="MOTION_CAPTION_PLAN_STALE"):
        compile_render_plan(
            intent, stale_captions, composition, motion, broll,
            CanvasPlan(width=1080, height=1920),
        )


def test_caption_backend_revision_invalidates_only_caption_render_descendants() -> None:
    intent = _intent(())
    captions = _captions(intent, _cue("caption-cache-cue", 0, 60))
    composition = _composition(intent)
    broll = _broll(intent)
    motion = build_motion_plan(intent, captions, composition, broll)

    def compile_with(version: str):
        return compile_render_plan(
            intent,
            captions,
            composition,
            motion,
            broll,
            CanvasPlan(width=1080, height=1920),
            backends=(BackendAssignment(
                domain="caption",
                backend_id="libass",
                backend_version=version,
            ),),
        )

    previous = compile_with("7C.libass-tier1.1")
    current = compile_with(CAPTION_RENDER_BACKEND_VERSION)
    previous_nodes = {node.node_id: node.cache_key for node in previous.render_graph_nodes}
    current_nodes = {node.node_id: node.cache_key for node in current.render_graph_nodes}

    assert CAPTION_RENDER_BACKEND_VERSION == "7C.libass-tier1.2"
    assert current_nodes["captions"] != previous_nodes["captions"]
    assert current_nodes["base-visual"] == previous_nodes["base-visual"]
    assert current_nodes["composite"] != previous_nodes["composite"]
    assert current_nodes["encode"] != previous_nodes["encode"]
