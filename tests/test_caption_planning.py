from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.caption_planning import (
    CaptionProtectedRegion,
    build_caption_plan,
    resolve_caption_font_manifest,
    write_caption_plan_ass,
)
from app.config import AppConfig
from app.creative_contracts import (
    BeatRole,
    CaptionPlan,
    CreativeIntent,
    CreativePolicy,
    EditMapSegment,
    EvidenceItem,
    ImmutableProductionIdentity,
    ImmutableProductionPlanLink,
    Intensity,
    NormalizedRect,
    OutputInterval,
    ResolvedBeat,
    ResolvedEmphasis,
    SemanticClass,
    SourceInterval,
    SourceOutputTimeMap,
)


def _reference() -> ImmutableProductionPlanLink:
    return ImmutableProductionPlanLink(
        plan_id="plan-caption-001",
        plan_fingerprint="a" * 64,
        identity=ImmutableProductionIdentity(
            project_id="project-caption", run_id="run-caption", analysis_id="analysis-caption",
            candidate_id="candidate-caption", source_id="source-caption",
        ),
    )


def _intent(intensity: Intensity = Intensity.HIGH) -> CreativeIntent:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="edit-caption-001", source=SourceInterval.from_seconds(0, 12),
        output=OutputInterval(start_frame=0, end_frame=360),
    ),))
    evidence = (
        EvidenceItem(
            evidence_ref="evidence-hook", evidence_kind="story_unit", source=SourceInterval.from_seconds(0, 1.5),
            confidence=0.96, artifact_fingerprint="1" * 64, provenance="brain:hook",
        ),
        EvidenceItem(
            evidence_ref="evidence-claim", evidence_kind="transcript", source=SourceInterval.from_seconds(1.5, 4),
            confidence=0.94, artifact_fingerprint="2" * 64, provenance="brain:emphasis",
        ),
        EvidenceItem(
            evidence_ref="evidence-payoff", evidence_kind="story_unit", source=SourceInterval.from_seconds(5, 8),
            confidence=0.97, artifact_fingerprint="3" * 64, provenance="brain:payoff",
        ),
    )
    return CreativeIntent(
        intent_id="intent-caption-001", revision=1, production_plan=_reference(),
        source_output_mapping=mapping, evidence_fingerprint="b" * 64,
        evidence_manifest=evidence, proposal_hash="c" * 64,
        policy=CreativePolicy(
            preset_id="editorial", preset_version="1", platform="tiktok",
            caption_style_family="emphasis", caption_density="balanced", intensity=intensity,
        ),
        confidence=0.95, provenance=("brain",),
        beats=(
            ResolvedBeat(
                decision_id="beat-hook", source=SourceInterval.from_seconds(0, 1.5),
                output=OutputInterval(start_frame=0, end_frame=45), confidence=0.96,
                evidence_refs=("evidence-hook",), role=BeatRole.HOOK, importance=0.95,
            ),
            ResolvedBeat(
                decision_id="beat-payoff", source=SourceInterval.from_seconds(5, 8),
                output=OutputInterval(start_frame=150, end_frame=240), confidence=0.97,
                evidence_refs=("evidence-payoff",), role=BeatRole.PAYOFF, importance=0.98,
            ),
        ),
        semantic_emphasis=(ResolvedEmphasis(
            decision_id="emphasis-three-steps", source=SourceInterval.from_seconds(2, 3),
            output=OutputInterval(start_frame=60, end_frame=90), confidence=0.94,
            evidence_refs=("evidence-claim",), text_span="три шага",
            semantic_class=SemanticClass.NUMBER, importance=0.96,
        ),),
    )


def _config():
    config = AppConfig().production_render
    config.output_width = 1080
    config.output_height = 1920
    config.subtitle_font_family = "Arial"
    config.subtitle_max_words_per_cue = 7
    return config


def _word_transcript(confidence: float = 0.98) -> dict:
    text = (
        "Это главный вопрос. Сделай ровно три шага и проверь результат. "
        "Потом всё наконец работает."
    ).split()
    return {"words": [
        {
            "text": word, "start": index * 0.5, "end": index * 0.5 + 0.46,
            "confidence": confidence, "timing_source": "verified",
        }
        for index, word in enumerate(text)
    ]}


def _plain_intent(mapping: SourceOutputTimeMap) -> CreativeIntent:
    return _intent().model_copy(update={
        "source_output_mapping": mapping,
        "beats": (),
        "semantic_emphasis": (),
    })


def test_interview_contiguous_micro_cuts_repartition_to_real_cps_fit() -> None:
    mapping = SourceOutputTimeMap(segments=(
        EditMapSegment(
            map_id="micro-lead", source=SourceInterval.from_seconds(0, 0.25),
            output=OutputInterval(start_frame=0, end_frame=8),
        ),
        EditMapSegment(
            map_id="micro-001", source=SourceInterval.from_seconds(0.25, 0.93),
            output=OutputInterval(start_frame=8, end_frame=28),
        ),
        EditMapSegment(
            map_id="micro-002", source=SourceInterval.from_seconds(0.93, 1.43),
            output=OutputInterval(start_frame=28, end_frame=43),
        ),
        EditMapSegment(
            map_id="micro-003", source=SourceInterval.from_seconds(1.43, 1.87),
            output=OutputInterval(start_frame=43, end_frame=57),
        ),
        EditMapSegment(
            map_id="micro-004", source=SourceInterval.from_seconds(1.87, 3.13),
            output=OutputInterval(start_frame=57, end_frame=95),
        ),
        EditMapSegment(
            map_id="micro-005", source=SourceInterval.from_seconds(3.13, 4.88),
            output=OutputInterval(start_frame=95, end_frame=148),
        ),
    ))
    word_specs = (
        ("\u0433" * 8, 0.25, 0.70), ("\u0433" * 9, 0.70, 0.93),
        ("\u0433" * 2, 0.930, 1.055), ("\u0433" * 3, 1.055, 1.180),
        ("\u0433" * 3, 1.180, 1.305), ("\u0433" * 4, 1.305, 1.430),
        ("\u0433" * 6, 1.43, 1.65), ("\u0433" * 7, 1.65, 1.87),
        ("\u0433" * 7, 1.87, 2.29), ("\u0433" * 2, 2.29, 2.71),
        ("\u0433" * 9, 2.71, 3.13),
        ("\u0433", 3.13, 3.33), ("\u0433" * 7, 3.33, 3.63),
        ("\u0433" * 2, 3.63, 3.93), ("\u0433" * 10, 3.93, 4.23),
    )
    transcript = {"words": [
        {
            "text": text, "start": start, "end": end,
            "confidence": 0.99, "timing_source": "verified",
        }
        for text, start, end in word_specs
    ]}

    config = _config()
    config.subtitle_max_words_per_cue = 9
    plan = build_caption_plan(_plain_intent(mapping), transcript, config)

    assert plan.quality_report.status == "PASS"
    assert plan.quality_report.metrics.max_cps <= 20.0
    assert "CAPTION_READABILITY_COALESCED" not in plan.diagnostics
    assert "CAPTION_PRESENTATION_WINDOW_EXTENDED" in plan.diagnostics
    assert "CAPTION_CPS_HIGH" not in {finding.code for finding in plan.quality_report.findings}
    assert [len(cue.words) for cue in plan.cues] == [2, 9, 4]
    assert [(cue.output.start_frame, cue.output.end_frame) for cue in plan.cues] == [
        (2, 28), (28, 95), (95, 135),
    ]
    assert all(left.output.end_frame <= right.output.start_frame for left, right in zip(plan.cues, plan.cues[1:]))
    expected_word_outputs = [
        mapping.map_interval(SourceInterval.from_seconds(start, end))
        for _text, start, end in word_specs
    ]
    assert [word.output for cue in plan.cues for word in cue.words] == expected_word_outputs
    assert all(cue.fallback_reason is None for cue in plan.cues)


def test_infeasible_micro_cut_keeps_reading_speed_ceiling_blocker() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="micro-infeasible", source=SourceInterval.from_seconds(0, 0.5),
        output=OutputInterval(start_frame=0, end_frame=15),
    ),))
    transcript = {"words": [
        {
            "text": "\u0410" * 15, "start": 0, "end": 0.25,
            "confidence": 0.99, "timing_source": "verified",
        },
        {
            "text": "\u0411" * 15, "start": 0.25, "end": 0.5,
            "confidence": 0.99, "timing_source": "verified",
        },
    ]}

    plan = build_caption_plan(_plain_intent(mapping), transcript, _config())

    assert plan.quality_report.status == "BLOCKED"
    blockers = [
        finding for finding in plan.quality_report.findings
        if finding.code == "CAPTION_CPS_INFEASIBLE" and finding.severity == "blocker"
    ]
    assert blockers
    assert blockers[0].threshold == 20.0
    assert plan.feasibility_decision is not None
    assert plan.feasibility_decision.status == "INFEASIBLE"
    assert plan.feasibility_decision.reason_code == "CAPTION_CPS_INFEASIBLE"
    assert plan.feasibility_decision.speech_retiming_allowed is False
    assert plan.feasibility_decision.transcript_rewrite_allowed is False
    evidence = plan.feasibility_decision.evidence[0]
    assert evidence.character_count == 30
    assert evidence.available_frames == 15
    assert evidence.required_frames == 45
    assert evidence.measured_cps == 60.0


def test_real_interview_phrase_has_exact_temporal_infeasibility_evidence() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="interview-dialogue-001",
        source=SourceInterval.from_seconds(1200.67, 1202.58),
        output=OutputInterval(start_frame=0, end_frame=57),
    ),))
    specs = (
        ("Представьте,", 1200.92, 1201.32),
        ("что", 1201.40, 1201.50),
        ("вам", 1201.50, 1201.62),
        ("нужно", 1201.62, 1201.82),
        ("вырастить", 1201.82, 1202.14),
        ("покупатели,", 1202.14, 1202.58),
    )
    transcript = {"words": [
        {
            "text": text, "start": start, "end": end,
            "confidence": 0.99, "timing_source": "verified",
        }
        for text, start, end in specs
    ]}

    plan = build_caption_plan(_plain_intent(mapping), transcript, _config())

    assert plan.feasibility_decision is not None
    assert plan.feasibility_decision.status == "INFEASIBLE"
    evidence = next(
        item for item in plan.feasibility_decision.evidence
        if item.text == "нужно вырастить покупатели,"
    )
    assert evidence.character_count == 25
    assert evidence.available_frames == 30
    assert evidence.required_frames == 38
    assert evidence.measured_cps == 25.0
    assert evidence.hard_cps_ceiling == 20.0
    assert evidence.mapping_segment_ids == ("interview-dialogue-001",)
    assert any(
        item.code == "CAPTION_CPS_INFEASIBLE" and item.severity == "blocker"
        for item in plan.quality_report.findings
    )


def test_words_at_known_interview_boundary_keep_identity_order_timing_and_provenance() -> None:
    mapping = SourceOutputTimeMap(segments=(
        EditMapSegment(
            map_id="interview-before-1210-90",
            source=SourceInterval.from_seconds(1210.50, 1210.90),
            output=OutputInterval(start_frame=0, end_frame=12),
        ),
        EditMapSegment(
            map_id="interview-between-boundaries",
            source=SourceInterval.from_seconds(1210.90, 1211.50),
            output=OutputInterval(start_frame=12, end_frame=30),
        ),
        EditMapSegment(
            map_id="interview-after-1211-50",
            source=SourceInterval.from_seconds(1211.50, 1212.00),
            output=OutputInterval(start_frame=30, end_frame=45),
        ),
    ))
    transcript = {"words": [
        {
            "text": "человек", "start": 1210.74, "end": 1211.04,
            "confidence": 0.99, "timing_source": "verified",
        },
        {
            "text": "умеет", "start": 1211.38, "end": 1211.66,
            "confidence": 0.99, "timing_source": "verified",
        },
    ]}

    plan = build_caption_plan(_plain_intent(mapping), transcript, _config())
    words = [word for cue in plan.cues for word in cue.words]

    assert [word.word_id for word in words] == ["word-00001", "word-00002"]
    assert [word.text for word in words] == ["человек", "умеет"]
    assert words[0].source == SourceInterval.from_seconds(1210.74, 1211.04)
    assert words[0].mapping_segment_ids == (
        "interview-before-1210-90", "interview-between-boundaries",
    )
    assert words[0].output == OutputInterval(start_frame=7, end_frame=17)
    assert words[1].source == SourceInterval.from_seconds(1211.38, 1211.66)
    assert words[1].mapping_segment_ids == (
        "interview-between-boundaries", "interview-after-1211-50",
    )
    assert words[1].output == OutputInterval(start_frame=26, end_frame=35)
    assert plan.quality_report.status == "PASS"


def test_word_mapping_does_not_join_a_real_discontinuous_cut() -> None:
    mapping = SourceOutputTimeMap(segments=(
        EditMapSegment(
            map_id="cut-left", source=SourceInterval.from_seconds(10.0, 11.0),
            output=OutputInterval(start_frame=0, end_frame=30),
        ),
        EditMapSegment(
            map_id="cut-right", source=SourceInterval.from_seconds(20.0, 21.0),
            output=OutputInterval(start_frame=30, end_frame=60),
        ),
    ))
    transcript = {"words": [{
        "text": "человек", "start": 10.8, "end": 20.2,
        "confidence": 0.99, "timing_source": "verified",
    }]}

    plan = build_caption_plan(_plain_intent(mapping), transcript, _config())

    assert not plan.cues
    assert "UNMAPPED_WORD_DROPPED" in plan.diagnostics
    assert plan.feasibility_decision is not None
    assert plan.feasibility_decision.status == "NOT_APPLICABLE"


def test_frame_overlap_does_not_create_false_cps_blocker() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="overlap-quantized", source=SourceInterval.from_seconds(0, 5),
        output=OutputInterval(start_frame=0, end_frame=150),
    ),))
    specs = (
        ("АА,", 7, 20), ("А", 26, 36), ("АА", 35, 40), ("А", 39, 48), ("ААА", 47, 53),
        ("АААААА", 52, 67), ("ААА", 66, 76), ("АА", 75, 80), ("ААА", 79, 84),
        ("АА", 83, 87), ("АААААААА", 86, 96), ("АААААА.", 95, 102),
        ("АА,", 104, 108), ("ААА", 107, 111), ("АА", 110, 114), ("ААААА,", 113, 123),
    )
    transcript = {"words": [
        {
            "text": text, "start": start / 30, "end": end / 30,
            "confidence": 0.99, "timing_source": "verified",
        }
        for text, start, end in specs
    ]}
    config = _config()
    config.subtitle_max_words_per_cue = 9

    plan = build_caption_plan(_plain_intent(mapping), transcript, config)

    assert plan.quality_report.status != "BLOCKED"
    assert plan.quality_report.metrics.max_cps <= 20.0
    assert "CAPTION_PRESENTATION_WINDOW_EXTENDED" in plan.diagnostics
    assert not any(item.code == "CAPTION_CPS_HIGH" for item in plan.quality_report.findings)


def test_semantic_motion_only_uses_brain_events_and_keeps_non_events_static() -> None:
    plan = build_caption_plan(_intent(), _word_transcript(), _config())

    assert plan.schema_version == "7C.caption-plan.1"
    assert plan.backend_id == "libass"
    assert plan.font_manifest is not None and plan.font_manifest.file_sha256
    assert plan.typography is not None
    assert plan.typography.token_id == "caption-preset:accent_yellow:1.0.0"
    assert any(cue.beat_role == BeatRole.HOOK and cue.primitive_id == "slide" for cue in plan.cues)
    assert any(cue.beat_role == BeatRole.PAYOFF and cue.primitive_id == "scale" for cue in plan.cues)
    emphasized = [cue for cue in plan.cues if cue.emphasis is not None]
    assert len(emphasized) == 1
    assert emphasized[0].primitive_id == "karaoke"
    assert emphasized[0].emphasis is not None
    assert emphasized[0].emphasis.word_indexes
    assert len(emphasized[0].emphasis.word_indexes) <= 2
    assert any("три шага" in line.casefold() for line in emphasized[0].resolved_lines)
    assert all(cue.evidence_refs for cue in plan.cues if cue.primitive_id != "static")
    assert sum(cue.primitive_id == "karaoke" for cue in plan.cues) == 1
    assert plan.quality_report.metrics.semantic_emphasis_count == 1
    with pytest.raises(ValidationError, match="Instance is frozen"):
        plan.quality_report.metrics.cue_count = 99  # type: ignore[misc]


def test_payoff_presentation_keeps_independent_semantic_emphasis() -> None:
    intent = _intent()
    payoff = intent.beats[1].model_copy(update={
        "source": SourceInterval.from_seconds(2, 3),
        "output": OutputInterval(start_frame=60, end_frame=90),
    })
    intent = intent.model_copy(update={"beats": (intent.beats[0], payoff)})

    plan = build_caption_plan(intent, _word_transcript(), _config())

    cue = next(item for item in plan.cues if item.beat_role == BeatRole.PAYOFF)
    assert cue.primitive_id == "scale"
    assert cue.emphasis is not None
    assert cue.emphasis.treatment == "bounded_scale"
    assert {"evidence-claim", "evidence-payoff"} <= set(cue.evidence_refs)
    assert plan.quality_report.metrics.semantic_emphasis_count == 1


def test_native_7c_plan_cannot_omit_font_geometry_or_quality_contracts() -> None:
    with pytest.raises(ValidationError, match="deterministic font"):
        CaptionPlan(
            schema_version="7C.caption-plan.1", intent_id="intent-caption-001", backend_id="libass",
        )


def test_weak_phrase_timing_degrades_to_phrase_static_without_per_word_effects() -> None:
    transcript = {"segments": [{
        "start": 0, "end": 7.5,
        "text": "Это главный вопрос. Сделай ровно три шага и проверь результат. Потом всё наконец работает.",
    }]}
    plan = build_caption_plan(_intent(), transcript, _config())

    assert plan.cues
    assert all(cue.timing_mode in {"phrase", "static"} for cue in plan.cues)
    assert all(cue.primitive_id == "static" for cue in plan.cues)
    assert "WEAK_TIMING_DEGRADED_TO_PHRASE_STATIC" in plan.diagnostics
    assert "CAPTION_TIMING_WEAK" in {finding.code for finding in plan.quality_report.findings}
    assert plan.quality_report.status == "PASS_WITH_WARNINGS"


def test_transcript_word_probability_flows_to_plan_and_degrades_low_confidence_emphasis() -> None:
    transcript = _word_transcript()
    for word in transcript["words"]:
        word.pop("confidence")
        word["probability"] = 0.38

    plan = build_caption_plan(_intent(), transcript, _config())

    words = [word for cue in plan.cues for word in cue.words]
    assert words
    assert {word.confidence for word in words} == {0.38}
    assert all(cue.timing_confidence == 0.38 for cue in plan.cues)
    assert all(cue.timing_mode == "static" for cue in plan.cues)
    assert all(cue.primitive_id == "static" for cue in plan.cues)
    assert all(cue.emphasis is None for cue in plan.cues)


def test_weak_semantic_confidence_uses_static_safe_treatment() -> None:
    intent = _intent(Intensity.HIGH).model_copy(update={
        "beats": tuple(item.model_copy(update={"confidence": 0.55}) for item in _intent().beats),
        "semantic_emphasis": tuple(
            item.model_copy(update={"confidence": 0.55}) for item in _intent().semantic_emphasis
        ),
    })

    plan = build_caption_plan(intent, _word_transcript(), _config())

    assert all(cue.primitive_id == "static" for cue in plan.cues)
    assert all(cue.emphasis is None for cue in plan.cues)


def test_collision_resolver_avoids_lower_screen_region_and_holds_a_stable_lane() -> None:
    protected = CaptionProtectedRegion(
        region_id="screen-ui", output=OutputInterval(start_frame=0, end_frame=360),
        bounds=NormalizedRect(x=0, y=0.68, width=1, height=0.30),
        kind="screen", importance=1, confidence=1,
    )
    plan = build_caption_plan(_intent(), _word_transcript(), _config(), protected_regions=(protected,))

    assert all(cue.lane != "lower" for cue in plan.cues)
    assert len({cue.lane for cue in plan.cues}) == 1
    assert all(cue.collision is not None and cue.collision.overlap_ratio == 0 for cue in plan.cues)
    assert plan.quality_report.metrics.protected_overlap_count == 0


def test_unavoidable_collision_is_explicit_quality_blocker() -> None:
    protected = CaptionProtectedRegion(
        region_id="critical-full-frame", output=OutputInterval(start_frame=0, end_frame=360),
        bounds=NormalizedRect(x=0, y=0, width=1, height=1),
        kind="screen", importance=1, confidence=1,
    )
    plan = build_caption_plan(_intent(), _word_transcript(), _config(), protected_regions=(protected,))

    assert plan.quality_report.status == "BLOCKED"
    assert "CAPTION_PROTECTED_REGION_OVERLAP" in {finding.code for finding in plan.quality_report.findings}


def test_font_manifest_and_ass_preview_final_share_semantics_and_normalized_geometry(tmp_path: Path) -> None:
    manifest = resolve_caption_font_manifest("Arial")
    plan = build_caption_plan(_intent(Intensity.BALANCED), _word_transcript(), _config(), font_manifest=manifest)
    preview = write_caption_plan_ass(plan, tmp_path / "preview.ass", 540, 960)
    final = write_caption_plan_ass(plan, tmp_path / "final.ass", 1080, 1920)
    preview_text = preview.read_text(encoding="utf-8-sig")
    final_text = final.read_text(encoding="utf-8-sig")

    assert manifest.file_sha256 and manifest.file_name
    assert f"FontSHA256: {manifest.file_sha256}" in preview_text
    assert [line.split(",", 3)[:3] for line in preview_text.splitlines() if line.startswith("Dialogue:")] == [
        line.split(",", 3)[:3] for line in final_text.splitlines() if line.startswith("Dialogue:")
    ]
    assert [cue.resolved_lines for cue in plan.cues] == [cue.resolved_lines for cue in plan.model_copy().cues]
    assert all(cue.normalized_bounds is not None for cue in plan.cues)
    karaoke_cues = [cue for cue in plan.cues if cue.primitive_id == "karaoke"]
    assert karaoke_cues and all(cue.emphasis is not None for cue in karaoke_cues)
    assert "\\kf" in final_text


def test_missing_font_uses_approved_deterministic_fallback() -> None:
    config = _config()
    config.subtitle_font_family = "__definitely_missing_caption_font__"
    plan = build_caption_plan(_intent(), _word_transcript(), config)

    assert plan.font_manifest is not None and plan.font_manifest.fallback_used is True
    assert plan.font_manifest.resolved_family != "__definitely_missing_caption_font__"
    assert "CAPTION_FONT_FALLBACK" in {finding.code for finding in plan.quality_report.findings}
    assert all(cue.fallback_reason in {None, "missing_font"} or cue.fallback_reason == "readability" for cue in plan.cues)


def test_contrast_box_caption_token_maps_to_deterministic_ass_style(tmp_path: Path) -> None:
    plan = build_caption_plan(_intent(), _word_transcript(), _config())
    assert plan.typography is not None
    boxed = plan.model_copy(update={
        "typography": plan.typography.model_copy(update={
            "token_id": "caption-preset:contrast_box:1.0.0",
        }),
    })
    path = write_caption_plan_ass(boxed, tmp_path / "boxed.ass", 1080, 1920)
    style = next(
        line for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("Style: CaptionPlan,")
    ).split(",")
    assert style[6].startswith("&H52")
    assert style[15] == "3"
