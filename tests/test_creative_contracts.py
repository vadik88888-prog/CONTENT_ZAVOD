from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.creative_contracts import (
    BackendAssignment,
    BeatProposal,
    CaptionCuePlan,
    CaptionPlan,
    CanvasPlan,
    CompositionPlan,
    CreativePolicy,
    CreativeProposal,
    CreativeProposalNormalizer,
    EditMapSegment,
    EmphasisProposal,
    EvidenceBundle,
    EvidenceItem,
    EvidenceResolver,
    Intensity,
    ImmutableProductionIdentity,
    ImmutableProductionPlanLink,
    LayoutFamily,
    MotionEventProposal,
    MotionEventPlan,
    MotionPlan,
    OutputInterval,
    SourceBRollPlan,
    SourceBRollProposal,
    SourceInterval,
    SourceOutputTimeMap,
    canonical_hash,
    compile_creative_intent,
    compile_legacy_render_plan,
    compile_render_plan,
    source_output_map_from_legacy_timeline,
)
from app.production_models import (
    ProductionPlan,
    ProductionPlanEnvelope,
    ProductionPlanIdentity,
    ProductionPlanInputFingerprints,
    ProductionPlanPreset,
    ProductionPlanTarget,
)
from app.video_models import CropPlan, SourceVideoClip, SubtitleCue, SubtitleProject, SubtitleStyle, VideoTimeline


def _reference() -> ImmutableProductionPlanLink:
    return ImmutableProductionPlanLink(
        plan_id="plan-001",
        plan_fingerprint="a" * 64,
        identity=ImmutableProductionIdentity(
            project_id="project-001",
            run_id="run-001",
            analysis_id="analysis-001",
            candidate_id="candidate-001",
            source_id="source-001",
        ),
    )


def _mapping() -> SourceOutputTimeMap:
    return SourceOutputTimeMap(segments=(
        EditMapSegment(
            map_id="edit-001",
            source=SourceInterval.from_seconds(4, 8),
            output=OutputInterval(start_frame=0, end_frame=120),
        ),
        EditMapSegment(
            map_id="edit-002",
            source=SourceInterval.from_seconds(10, 12),
            output=OutputInterval(start_frame=120, end_frame=180),
        ),
    ))


def _evidence() -> EvidenceBundle:
    return EvidenceBundle(
        production_plan=_reference(),
        source_range=SourceInterval.from_seconds(0, 60),
        candidate_source_range=SourceInterval.from_seconds(4, 12),
        items=(
            EvidenceItem(
                evidence_ref="story-hook",
                evidence_kind="story_unit",
                source=SourceInterval.from_seconds(4, 6),
                confidence=0.94,
                artifact_fingerprint="b" * 64,
                provenance="multimodal-timeline:story-unit-1",
            ),
            EvidenceItem(
                evidence_ref="visual-target",
                evidence_kind="visual",
                source=SourceInterval.from_seconds(10, 11),
                confidence=0.88,
                artifact_fingerprint="c" * 64,
                provenance="vision:observation-4",
            ),
        ),
    )


def _proposal() -> CreativeProposal:
    return CreativeProposal(
        proposal_id="proposal-001",
        production_plan=_reference(),
        revision=1,
        beats=(BeatProposal(
            decision_id="beat-hook",
            source=SourceInterval.from_seconds(4, 6),
            confidence=0.9,
            evidence_refs=("story-hook",),
            role="hook",
            importance=1,
        ),),
        emphasis=(EmphasisProposal(
            decision_id="emphasis-missing",
            source=SourceInterval.from_seconds(5, 5.5),
            confidence=0.7,
            evidence_refs=("not-in-catalog",),
            text_span="important",
            semantic_class="claim",
            importance=0.8,
        ),),
        motion=(MotionEventProposal(
            decision_id="motion-hook",
            source=SourceInterval.from_seconds(4, 5),
            confidence=0.8,
            evidence_refs=("story-hook",),
            purpose="hook",
            domain="composition",
            intensity="balanced",
        ),),
    )


def _policy(*, reduced_motion: bool = False) -> CreativePolicy:
    return CreativePolicy(
        preset_id="clean-podcast",
        preset_version="1",
        platform="universal",
        caption_style_family="clean",
        caption_density="balanced",
        intensity=Intensity.BALANCED,
        reduced_motion=reduced_motion,
        source_broll_enabled=False,
    )


def test_source_output_mapping_handles_cuts_reorder_and_rejects_ambiguous_destination() -> None:
    mapping = _mapping()

    assert mapping.map_interval(SourceInterval.from_seconds(4.5, 5.5)) == OutputInterval(
        start_frame=15,
        end_frame=45,
    )
    assert mapping.map_interval(SourceInterval.from_seconds(8, 10)) is None

    with pytest.raises(ValidationError, match="same destination frame"):
        SourceOutputTimeMap(segments=(
            mapping.segments[0],
            EditMapSegment(
                map_id="overlap",
                source=SourceInterval.from_seconds(20, 21),
                output=OutputInterval(start_frame=119, end_frame=149),
            ),
        ))

    crop = CropPlan(strategy="center_crop", source_width=1920, source_height=1080)
    timeline = VideoTimeline(
        clips=[
            SourceVideoClip(
                clip_id="second-source-range",
                order=1,
                timeline_start_seconds=0,
                timeline_end_seconds=2,
                duration_seconds=2,
                source_path="source.mp4",
                source_start_seconds=10,
                source_end_seconds=12,
                visual_strategy="mapped_source",
                crop_plan=crop,
                status="ready",
            ),
            SourceVideoClip(
                clip_id="first-source-range",
                order=2,
                timeline_start_seconds=2,
                timeline_end_seconds=6,
                duration_seconds=4,
                source_path="source.mp4",
                source_start_seconds=4,
                source_end_seconds=8,
                visual_strategy="mapped_source",
                crop_plan=crop,
                status="ready",
            ),
        ],
        duration_seconds=6,
    )
    adapted = source_output_map_from_legacy_timeline(timeline)
    assert adapted.segments[0].source == SourceInterval.from_seconds(10, 12)
    assert adapted.segments[1].output == OutputInterval(start_frame=60, end_frame=180)


def test_evidence_resolver_drops_missing_evidence_with_typed_safe_fallback() -> None:
    resolution = EvidenceResolver().resolve(_proposal(), _evidence(), _mapping())

    assert [beat.decision_id for beat in resolution.beats] == ["beat-hook"]
    assert resolution.beats[0].output == OutputInterval(start_frame=0, end_frame=60)
    assert resolution.emphasis == ()
    assert resolution.diagnostics[0].code == "MISSING_EVIDENCE"
    assert resolution.diagnostics[0].fallback == "drop_emphasis"

    intent = compile_creative_intent(_proposal(), _evidence(), _mapping(), _policy(reduced_motion=True))
    assert intent.motion_events == ()
    assert intent.production_plan == _reference()

    misplaced = _evidence().model_copy(update={
        "items": (
            _evidence().items[0].model_copy(update={"source": SourceInterval.from_seconds(10, 11)}),
            _evidence().items[1],
        ),
    })
    invalid_resolution = EvidenceResolver().resolve(_proposal(), misplaced, _mapping())
    assert invalid_resolution.beats == ()
    assert invalid_resolution.diagnostics[0].code == "EVIDENCE_OUTSIDE_DECISION"

    broll_proposal = _proposal().model_copy(update={
        "source_broll": (SourceBRollProposal(
            decision_id="broll-missing",
            source=SourceInterval.from_seconds(4, 5),
            confidence=0.8,
            evidence_refs=("story-hook",),
            source_cutaway=SourceInterval.from_seconds(20, 21),
            source_cutaway_evidence_refs=("missing-cutaway",),
            story_unit_id="story-unit-1",
            story_unit_evidence_ref="story-hook",
        ),),
    })
    broll_resolution = EvidenceResolver().resolve(broll_proposal, _evidence(), _mapping())
    assert broll_resolution.source_broll == ()
    assert broll_resolution.diagnostics[-1].fallback == "a_roll"


def test_untrusted_proposal_rejects_unknown_fields_enums_and_raw_renderer_values() -> None:
    payload = _proposal().model_dump(mode="json")
    payload["ffmpeg_filter"] = "crop=evil"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CreativeProposal.model_validate(payload)

    payload = _proposal().model_dump(mode="json")
    payload["motion"][0]["purpose"] = "execute_shell"
    with pytest.raises(ValidationError):
        CreativeProposal.model_validate(payload)

    raw = _proposal().model_dump(mode="json", exclude={"production_plan"})
    normalized = CreativeProposalNormalizer().normalize(raw, _reference())
    assert normalized.production_plan == _reference()
    raw["production_plan"] = _reference().model_copy(update={"plan_id": "forged"}).model_dump(mode="json")
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        CreativeProposalNormalizer().normalize(raw, _reference())


def test_identity_mismatch_is_rejected_before_intent_compilation() -> None:
    wrong = _evidence().model_copy(update={
        "production_plan": _reference().model_copy(update={"plan_id": "other-plan"}),
    })
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        EvidenceResolver().resolve(_proposal(), wrong, _mapping())


def test_compiled_plan_hash_and_parity_are_canonical_and_immutable() -> None:
    intent = compile_creative_intent(_proposal(), _evidence(), _mapping(), _policy())
    captions = CaptionPlan(
        intent_id=intent.intent_id,
        backend_id="libass",
        cues=(CaptionCuePlan(
            cue_id="cue-001",
            output=OutputInterval(start_frame=0, end_frame=60),
            resolved_lines=("A stable line",),
            lane="lower",
            typography_token_id="clean-v1",
            primitive_id="static",
        ),),
    )
    composition = CompositionPlan(intent_id=intent.intent_id)
    motion = MotionPlan(intent_id=intent.intent_id)
    broll = SourceBRollPlan(intent_id=intent.intent_id)
    backend = BackendAssignment(
            domain="caption",
            backend_id="libass",
            backend_version="0.17",
            deterministic=True,
        )

    def compile_with(caption_plan: CaptionPlan):
        return compile_render_plan(
            intent,
            caption_plan,
            composition,
            motion,
            broll,
            CanvasPlan(width=1080, height=1920),
            backends=(backend,),
        )

    first = compile_with(captions)
    second = compile_with(captions)

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.parity_signature == second.parity_signature
    assert [node.node_id for node in first.render_graph_nodes] == [
        "base-visual", "caption-overlay", "composite", "quality-check",
    ]
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    with pytest.raises(ValidationError, match="Instance is frozen"):
        first.plan_hash = "f" * 64  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Instance is frozen"):
        first.production_plan.identity.source_id = "mutated"  # type: ignore[misc]

    changed_captions = captions.model_copy(update={
        "cues": (captions.cues[0].model_copy(update={"resolved_lines": ("A changed line",)}),),
    })
    changed = compile_with(changed_captions)
    assert changed.plan_hash != first.plan_hash
    assert changed.parity_signature != first.parity_signature

    tampered = first.model_dump(mode="json")
    tampered["caption_plan"]["cues"][0]["resolved_lines"] = ["tampered after compilation"]
    with pytest.raises(ValidationError, match="COMPILED_PLAN_HASH_MISMATCH"):
        type(first).model_validate(tampered)

    forged_motion = MotionPlan(
        intent_id=intent.intent_id,
        events=(MotionEventPlan(
            event_id="forged-motion",
            output=OutputInterval(start_frame=0, end_frame=15),
            purpose="payoff",
            domain="composition",
            primitive_id="punch_in",
            intensity="balanced",
            evidence_refs=("story-hook",),
        ),),
    )
    with pytest.raises(ValueError, match="MOTION_PLAN_EVIDENCE_MISMATCH"):
        compile_render_plan(
            intent,
            captions,
            composition,
            forged_motion,
            broll,
            CanvasPlan(width=1080, height=1920),
            backends=(backend,),
        )


def _constructed_production_plan() -> ProductionPlan:
    envelope = ProductionPlanEnvelope(
        identity=ProductionPlanIdentity.model_validate(_reference().identity.model_dump(mode="json")),
        preset=ProductionPlanPreset(
            preset_id="documentary",
            preset_version="4B.1",
            platform="universal",
        ),
        target=ProductionPlanTarget(width=1080, height=1920, fps=30),
        input_fingerprints=ProductionPlanInputFingerprints(
            source_sha256="1" * 64,
            transcript_sha256="2" * 64,
            analysis_sha256="3" * 64,
            final_script_sha256="4" * 64,
            boundary_decision_sha256="5" * 64,
        ),
        created_at="2026-08-08T00:00:00Z",
    )
    # The adapter only needs the already-validated envelope and the established
    # ProductionPlan fingerprint method.  No synthetic second lifecycle is built.
    return ProductionPlan.model_construct(plan_id="plan-legacy", envelope=envelope)


def test_legacy_adapter_keeps_renderer_behavior_descriptive_and_effect_free() -> None:
    subtitle = SubtitleProject(
        project_id="subtitles-001",
        audio_project_id="audio-001",
        duration_seconds=6,
        style=SubtitleStyle(
            style_id="clean",
            font_family="Arial",
            font_size=56,
            text_color="#FFFFFF",
            highlight_color="#FF8800",
            outline_color="#000000",
            outline_width=2,
            shadow=1,
            background="transparent",
            bottom_margin=120,
        ),
        cues=(SubtitleCue(
            cue_id="cue-legacy",
            segment_id="segment-001",
            speaker="speaker",
            text="Legacy line",
            start_seconds=0,
            end_seconds=1,
            word_count=2,
            line_count=1,
            style_id="clean",
            source_type="dialogue",
            resolved_lines=("Legacy line",),
        ),),
    )
    compiled = compile_legacy_render_plan(
        _constructed_production_plan(),
        _mapping(),
        subtitle_project=subtitle,
    )

    assert compiled.compatibility_mode == "legacy_passthrough"
    assert compiled.production_plan.plan_id == "plan-legacy"
    assert compiled.motion_plan.events == ()
    assert compiled.source_broll_plan.segments == ()
    assert compiled.caption_plan.backend_id == "legacy_passthrough"
    assert compiled.caption_plan.cues[0].resolved_lines == ("Legacy line",)
    assert compiled.expected_quality_constraints[-1] == "preview_final_parity"

    changed_cue = subtitle.cues[0].model_copy(update={
        "text": "Changed legacy line",
        "resolved_lines": ["Changed legacy line"],
    })
    changed_subtitle = subtitle.model_copy(update={"cues": [changed_cue]})
    changed = compile_legacy_render_plan(
        _constructed_production_plan(),
        _mapping(),
        subtitle_project=changed_subtitle,
    )
    assert changed.parity_signature != compiled.parity_signature
