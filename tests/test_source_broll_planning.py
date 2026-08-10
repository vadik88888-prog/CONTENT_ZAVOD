from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.creative_contracts import (
    BeatProposal,
    BeatRole,
    AttentionTarget,
    CanvasPlan,
    CaptionPlan,
    CompositionPlan,
    CompositionSegmentPlan,
    CreativePolicy,
    CreativeProposal,
    EditMapSegment,
    EvidenceBundle,
    EvidenceItem,
    ImmutableProductionIdentity,
    ImmutableProductionPlanLink,
    LayoutFamily,
    MotionPlan,
    NormalizedRect,
    OutputInterval,
    SourceBRollProposal,
    SourceBRollSegmentPlan,
    SourceBRollSemanticKind,
    SourceInterval,
    SourceOutputTimeMap,
    compile_creative_intent,
    compile_render_plan,
)
from app.source_broll_planning import SourceBRollPlanner, SourceSceneEvidence


def _reference() -> ImmutableProductionPlanLink:
    return ImmutableProductionPlanLink(
        plan_id="plan-7e",
        plan_fingerprint="1" * 64,
        identity=ImmutableProductionIdentity(
            project_id="project-1", run_id="run-1", analysis_id="analysis-1",
            candidate_id="candidate-1", source_id="source-1",
        ),
    )


def _intent(
    kind: SourceBRollSemanticKind = SourceBRollSemanticKind.ACTION,
    *,
    duplicate: bool = False,
):
    story = EvidenceItem(
        evidence_ref="story-current", evidence_kind="story_unit",
        source=SourceInterval.from_seconds(4, 7), confidence=0.95,
        artifact_fingerprint="2" * 64, provenance="content-map:story-1",
    )
    scene = EvidenceItem(
        evidence_ref="scene-cutaway", evidence_kind="scene",
        source=SourceInterval.from_seconds(20, 21), confidence=0.94,
        artifact_fingerprint="3" * 64, provenance="timeline:scene-20",
    )
    visual = EvidenceItem(
        evidence_ref="visual-cutaway", evidence_kind="visual",
        source=SourceInterval.from_seconds(20, 21), confidence=0.93,
        artifact_fingerprint="4" * 64, provenance="vision:keyframe-20",
    )
    beats = [BeatProposal(
        decision_id="beat-action", source=SourceInterval.from_seconds(4, 5),
        confidence=0.92, evidence_refs=("story-current",), role=BeatRole.ACTION,
        importance=0.9,
    )]
    broll = [SourceBRollProposal(
        decision_id="broll-action", source=SourceInterval.from_seconds(4, 5),
        confidence=0.91, evidence_refs=("story-current",),
        source_cutaway=SourceInterval.from_seconds(20, 21),
        source_cutaway_evidence_refs=("scene-cutaway", "visual-cutaway"),
        story_unit_id="story-1", story_unit_evidence_ref="story-current",
        semantic_kind=kind,
    )]
    if duplicate:
        beats.append(BeatProposal(
            decision_id="beat-reaction", source=SourceInterval.from_seconds(6, 7),
            confidence=0.9, evidence_refs=("story-current",), role=BeatRole.REACTION,
            importance=0.85,
        ))
        broll.append(SourceBRollProposal(
            decision_id="broll-repeat", source=SourceInterval.from_seconds(6, 7),
            confidence=0.9, evidence_refs=("story-current",),
            source_cutaway=SourceInterval.from_seconds(20, 21),
            source_cutaway_evidence_refs=("scene-cutaway", "visual-cutaway"),
            story_unit_id="story-1", story_unit_evidence_ref="story-current",
            semantic_kind=kind,
        ))
    proposal = CreativeProposal(
        proposal_id="proposal-7e", production_plan=_reference(), revision=1,
        beats=tuple(beats), source_broll=tuple(broll),
    )
    evidence = EvidenceBundle(
        production_plan=_reference(), source_range=SourceInterval.from_seconds(0, 30),
        candidate_source_range=SourceInterval.from_seconds(0, 10),
        items=(story, scene, visual),
    )
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="map-1", source=SourceInterval.from_seconds(0, 10),
        output=OutputInterval(start_frame=0, end_frame=300),
    ),))
    return compile_creative_intent(
        proposal, evidence, mapping,
        CreativePolicy(
            preset_id="documentary", preset_version="1", platform="universal",
            source_broll_enabled=True,
        ),
    )


def _composition(intent) -> CompositionPlan:
    return CompositionPlan(
        intent_id=intent.intent_id,
        segments=(CompositionSegmentPlan(
            segment_id="composition-current", output=OutputInterval(start_frame=0, end_frame=300),
            layout=LayoutFamily.FIT_BACKGROUND,
            crop=NormalizedRect(x=0, y=0, width=1, height=1),
        ),),
    )


def _scene(kind: SourceBRollSemanticKind = SourceBRollSemanticKind.ACTION) -> SourceSceneEvidence:
    return SourceSceneEvidence(
        scene_id="scene-20", source_id="source-1",
        source=SourceInterval.from_seconds(20, 21), semantic_kinds=(kind,),
        story_unit_ids=("story-1",), beat_roles=(BeatRole.ACTION, BeatRole.REACTION),
        evidence_refs=("scene-cutaway", "visual-cutaway"), confidence=0.9,
        source_crop=NormalizedRect(x=0.40, y=0.20, width=0.35, height=0.55),
        source_target=AttentionTarget.OBJECT,
        identity_status="verified", attribution_status="verified",
        chronology_status="safe", causality_status="supported", rights_status="verified",
        payoff_signal="none", provenance=("vision:scene-20",),
    )


@pytest.mark.parametrize("kind", list(SourceBRollSemanticKind))
def test_each_supported_semantic_kind_requires_matching_scene_evidence(kind) -> None:
    intent = _intent(kind)
    plan = SourceBRollPlanner().plan(intent, [_scene(kind)], _composition(intent))

    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.semantic_kind == kind
    assert segment.beat_role == BeatRole.ACTION
    assert segment.audio_timeline == "a_roll_master" and segment.retain_source_audio is False
    assert segment.source_crop == _scene(kind).source_crop
    assert segment.source_target == AttentionTarget.OBJECT
    assert segment.fallback == "a_roll"
    assert segment.fallback_composition_segment_ids == ("composition-current",)
    assert segment.safety_checks is not None
    assert plan.quality_report.status == "PASS"


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"confidence": 0.4}, "SOURCE_BROLL_CONFIDENCE_LOW"),
        ({"attribution_status": "uncertain"}, "SOURCE_BROLL_ATTRIBUTION_UNCERTAIN"),
        ({"chronology_status": "uncertain"}, "SOURCE_BROLL_CHRONOLOGY_UNSAFE"),
        ({"causality_status": "uncertain"}, "SOURCE_BROLL_CAUSALITY_UNSAFE"),
        ({"rights_status": "uncertain"}, "SOURCE_BROLL_RIGHTS_UNCERTAIN"),
        ({"visible_speech_requires_sync": True}, "SOURCE_BROLL_LIP_SYNC_REQUIRED"),
    ],
)
def test_forbidden_or_unproven_scene_falls_back_to_a_roll(updates, code) -> None:
    intent = _intent()
    plan = SourceBRollPlanner().plan(
        intent, [_scene().model_copy(update=updates)], _composition(intent),
    )

    assert plan.segments == ()
    assert plan.fallback_policy == "a_roll_current_composition"
    assert plan.quality_report.status == "PASS_WITH_WARNINGS"
    assert plan.quality_report.metrics.a_roll_fallback_count == 1
    assert plan.quality_report.findings[0].code == code


def test_payoff_reveal_is_not_inserted_into_an_action_beat() -> None:
    intent = _intent()
    plan = SourceBRollPlanner().plan(
        intent, [_scene().model_copy(update={"payoff_signal": "reveal"})], _composition(intent),
    )

    assert plan.segments == ()
    assert plan.quality_report.metrics.premature_reveal_count == 1
    assert plan.quality_report.findings[0].code == "SOURCE_BROLL_PREMATURE_REVEAL"


def test_same_filler_range_is_selected_once_then_rejected() -> None:
    intent = _intent(duplicate=True)
    plan = SourceBRollPlanner().plan(intent, [_scene()], _composition(intent))

    assert [item.decision_id for item in plan.segments] == ["broll-action"]
    assert plan.quality_report.metrics.repeated_range_count == 1
    assert any(item.code == "SOURCE_BROLL_FILLER_RANGE_REPEATED" for item in plan.quality_report.findings)


def test_unmatched_scene_is_not_used_as_generic_filler() -> None:
    intent = _intent(SourceBRollSemanticKind.PRODUCT)
    plan = SourceBRollPlanner().plan(intent, [_scene(SourceBRollSemanticKind.CONTEXT)], _composition(intent))

    assert plan.segments == ()
    assert plan.quality_report.findings[0].code == "SOURCE_BROLL_RELEVANCE_UNPROVEN"


def test_planner_skips_unsafe_high_confidence_scene_for_safe_evidence() -> None:
    intent = _intent()
    unsafe = _scene().model_copy(update={
        "scene_id": "scene-unsafe", "confidence": 0.99, "attribution_status": "uncertain",
    })
    safe = _scene().model_copy(update={"scene_id": "scene-safe", "confidence": 0.88})

    plan = SourceBRollPlanner().plan(intent, [unsafe, safe], _composition(intent))

    assert plan.segments[0].source_scene_id == "scene-safe"
    assert plan.quality_report.status == "PASS"


def test_segment_contract_cannot_enable_broll_audio() -> None:
    with pytest.raises(ValidationError):
        SourceBRollSegmentPlan(
            segment_id="unsafe", destination=OutputInterval(start_frame=0, end_frame=30),
            source_cutaway=SourceInterval.from_seconds(1, 2), story_unit_id="story-1",
            evidence_refs=("scene-1",), retain_source_audio=True,
        )


def test_assessed_broll_plan_crosses_compiled_renderer_handoff() -> None:
    intent = _intent()
    composition = _composition(intent)
    broll = SourceBRollPlanner().plan(intent, [_scene()], composition)

    compiled = compile_render_plan(
        intent, CaptionPlan(intent_id=intent.intent_id), composition,
        MotionPlan(intent_id=intent.intent_id), broll, CanvasPlan(width=1080, height=1920),
    )

    assert compiled.source_broll_plan.segments[0].source_scene_id == "scene-20"
    assert "a_roll_master_audio" in compiled.expected_quality_constraints
