from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.caption_planning import (
    CaptionProtectedRegion,
    _Layout,
    _MappedWord,
    _fit_single_layout,
    _mapped_words,
    _normalize_cue_output_overlaps,
    _simultaneous_caption_collisions,
    build_caption_plan,
    materialize_caption_font_directory,
    resolve_caption_font_manifest,
    write_caption_plan_ass,
)
from app.caption_presets import CAPTION_PRESET_DEFINITIONS
from app.config import AppConfig
from app.creative_contracts import (
    BeatRole,
    CaptionCuePlan,
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


def _preset_intent(preset_id: str, *, reduced_motion: bool = False) -> CreativeIntent:
    preset = CAPTION_PRESET_DEFINITIONS[preset_id]  # type: ignore[index]
    intent = _intent()
    return intent.model_copy(update={
        "policy": intent.policy.model_copy(update={
            "preset_id": preset_id,
            "preset_version": preset.preset_version,
            "caption_style_family": preset.style_family,
            "reduced_motion": reduced_motion,
        }),
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
    assert [len(cue.words) for cue in plan.cues] == [2, 3, 5, 5]
    assert [(cue.output.start_frame, cue.output.end_frame) for cue in plan.cues] == [
        (2, 28), (28, 40), (40, 82), (82, 140),
    ]
    assert all(left.output.end_frame <= right.output.start_frame for left, right in zip(plan.cues, plan.cues[1:]))
    assert "CAPTION_FRAME_QUANTIZATION_NORMALIZED" in plan.diagnostics
    words = [word for cue in plan.cues for word in cue.words]
    assert [word.source for word in words] == [
        SourceInterval.from_seconds(start, end) for _text, start, end in word_specs
    ]
    assert all(left.output.end_frame <= right.output.start_frame for left, right in zip(words, words[1:]))
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


def test_feasible_gameplay_phrase_promotes_the_fitted_layout_into_the_compiled_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the Fresh Gameplay overflow: use the proven layout."""

    import app.caption_planning as caption_planning

    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="gameplay-overflow-001",
        source=SourceInterval.from_seconds(0, 3),
        output=OutputInterval(start_frame=0, end_frame=90),
    ),))
    words = "\u0412\u043e\u0437\u044c\u043c\u0438 \u043c\u043e\u044e \u043c\u0430\u0448\u0438\u043d\u043a\u0443 \u043e\u043d\u0430 \u0437\u0434\u0435\u0441\u044c \u043d\u0430 \u043c\u0435\u0442\u043a\u0435".split()
    transcript = {"words": [
        {
            "text": word,
            "start": index * 3 / len(words),
            "end": (index + 1) * 3 / len(words) - 0.01,
            "confidence": 0.99,
            "timing_source": "verified",
        }
        for index, word in enumerate(words)
    ]}
    config = _config()
    config.subtitle_max_words_per_cue = len(words)
    original_fit_layout = caption_planning._fit_layout

    def stale_wide_layout(group, measurer, base_size, minimum_size, maximum_width, protected_phrases):
        if len(group) > 1:
            return [_Layout(group, (" ".join(word.text for word in group),), base_size, True)]
        return original_fit_layout(
            group, measurer, base_size, minimum_size, maximum_width, protected_phrases,
        )

    monkeypatch.setattr(caption_planning, "_fit_layout", stale_wide_layout)
    plan = build_caption_plan(_plain_intent(mapping), transcript, config)

    assert plan.feasibility_decision is not None
    assert plan.feasibility_decision.status == "FEASIBLE"
    assert "CAPTION_FEASIBILITY_LAYOUT_APPLIED" in plan.diagnostics
    assert any(len(cue.resolved_lines) == 2 for cue in plan.cues)
    base_size = round(plan.typography.font_size_ratio * config.output_height)
    assert all(
        round(cue.resolved_font_size_ratio * config.output_height) == base_size
        for cue in plan.cues
    )
    assert all(cue.fallback_reason is None for cue in plan.cues)
    assert "CAPTION_READABILITY_FALLBACK" not in {
        finding.code for finding in plan.quality_report.findings
    }
    assert "CAPTION_LINE_OVERFLOW" not in {
        finding.code for finding in plan.quality_report.findings
    }


def test_feasibility_layout_uses_minimum_font_only_when_larger_sizes_do_not_fit() -> None:
    class _LinearMeasurer:
        def width(self, text: str, pixel_size: int) -> float:
            return len(text) * pixel_size

    word = _MappedWord(
        word_id="minimum-size-word",
        text="x" * 9,
        output=OutputInterval(start_frame=0, end_frame=90),
        timing_source="verified",
        confidence=0.99,
        source=SourceInterval.from_seconds(0, 3),
        map_ids=("minimum-size-map",),
    )

    fitted = _fit_single_layout(
        (word,), _LinearMeasurer(), base_size=20, minimum_size=16,
        maximum_width=150, protected_phrases=(),
    )

    assert fitted is not None
    assert fitted.font_size == 16
    assert fitted.fallback is True


def test_physically_unbreakable_caption_remains_blocked_by_the_native_gate() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="unbreakable-overflow-001",
        source=SourceInterval.from_seconds(0, 3),
        output=OutputInterval(start_frame=0, end_frame=90),
    ),))
    transcript = {"words": [{
        "text": "\u0416" * 60,
        "start": 0.0,
        "end": 3.0,
        "confidence": 0.99,
        "timing_source": "verified",
    }]}

    plan = build_caption_plan(_plain_intent(mapping), transcript, _config())

    assert plan.quality_report.status == "BLOCKED"
    assert "CAPTION_LINE_OVERFLOW" in {
        finding.code for finding in plan.quality_report.findings
    }


def test_adjacent_verified_words_with_one_frame_rounding_overlap_still_fit() -> None:
    """Frame rounding must not suppress legal source-time caption breaks."""

    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="rounded-adjacent-words-001", source=SourceInterval.from_seconds(0, 9.8),
        output=OutputInterval(start_frame=0, end_frame=294),
    ),))
    transcript = {"words": [
        {
            "text": "да",
            "start": index * 0.35,
            "end": (index + 1) * 0.35,
            "confidence": 0.99,
            "timing_source": "verified",
        }
        for index in range(28)
    ]}

    plan = build_caption_plan(_plain_intent(mapping), transcript, _config())

    assert plan.feasibility_decision is not None
    assert plan.feasibility_decision.status == "FEASIBLE"
    assert len(plan.cues) > 1
    assert all(len(cue.words) <= _config().subtitle_max_words_per_cue for cue in plan.cues)
    assert "CAPTION_LINE_OVERFLOW" not in {
        finding.code for finding in plan.quality_report.findings
    }


def test_adjacent_verified_words_normalize_quantized_frame_overlap_before_cue_timing() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="rounded-word-timing-001", source=SourceInterval.from_seconds(0, 0.7),
        output=OutputInterval(start_frame=0, end_frame=21),
    ),))
    transcript = {"words": [
        {"text": "да", "start": 0.0, "end": 0.35, "confidence": 0.99, "timing_source": "verified"},
        {"text": "нет", "start": 0.35, "end": 0.7, "confidence": 0.99, "timing_source": "verified"},
    ]}
    config = _config()
    config.subtitle_min_words_per_cue = 1
    config.subtitle_max_words_per_cue = 1

    plan = build_caption_plan(_plain_intent(mapping), transcript, config)

    assert plan.quality_report.status == "PASS"
    assert "CAPTION_FRAME_QUANTIZATION_NORMALIZED" in plan.diagnostics
    assert all(
        left.output.end_frame <= right.output.start_frame
        for left, right in zip(plan.cues, plan.cues[1:])
    )
    assert not any(
        finding.code == "CAPTION_SIMULTANEOUS_OVERLAP" and finding.severity == "blocker"
        for finding in plan.quality_report.findings
    )


def test_true_source_overlap_is_not_normalized_as_frame_quantization() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="true-word-overlap-001", source=SourceInterval.from_seconds(0, 0.7),
        output=OutputInterval(start_frame=0, end_frame=21),
    ),))
    transcript = {"words": [
        {"text": "да", "start": 0.0, "end": 0.36, "confidence": 0.99, "timing_source": "verified"},
        {"text": "нет", "start": 0.35, "end": 0.7, "confidence": 0.99, "timing_source": "verified"},
    ]}

    words, diagnostics = _mapped_words(_plain_intent(mapping), transcript)

    assert words[0].output.end_frame > words[1].output.start_frame
    assert "CAPTION_FRAME_QUANTIZATION_NORMALIZED" not in diagnostics


def test_real_interview_quantized_adjacent_words_keep_real_cps_blocker() -> None:
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
    assert "CAPTION_FRAME_QUANTIZATION_NORMALIZED" in plan.diagnostics
    assert [word.text for cue in plan.cues for word in cue.words] == [item[0] for item in specs]
    assert all(
        left.output.end_frame <= right.output.start_frame
        for left, right in zip(plan.cues, plan.cues[1:])
    )
    evidence = plan.feasibility_decision.evidence[0]
    assert evidence.text == "Представьте, что вам нужно вырастить покупатели,"
    assert evidence.character_count == 43
    assert evidence.available_frames == 57
    assert evidence.required_frames == 65
    assert evidence.measured_cps == 22.631579
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
    assert all(
        left.output.end_frame <= right.output.start_frame
        for left, right in zip(plan.cues, plan.cues[1:])
    )


@pytest.mark.parametrize(("source", "raw_ranges", "proposed_ranges", "overlap_frames"), (
    ("podcast", ((365, 402), (411, 466)), ((365, 407), (402, 470)), 5),
    ("interview", ((258, 289), (297, 349)), ((258, 296), (290, 358)), 6),
    ("food", ((145, 180), (194, 276)), ((145, 189), (184, 282)), 5),
    ("gameplay", ((549, 563), (574, 594)), ((549, 573), (562, 596)), 11),
))
def test_real_media_presentation_overlap_is_safely_normalized(
    source: str,
    raw_ranges: tuple[tuple[int, int], tuple[int, int]],
    proposed_ranges: tuple[tuple[int, int], tuple[int, int]],
    overlap_frames: int,
) -> None:
    raw = [OutputInterval(start_frame=start, end_frame=end) for start, end in raw_ranges]
    resolved = [
        OutputInterval(start_frame=start, end_frame=end)
        for start, end in proposed_ranges
    ]

    normalized = _normalize_cue_output_overlaps(
        raw, resolved, run_start=0, run_end=2,
    )

    assert normalized == {1: overlap_frames}, source
    assert resolved[0].end_frame == resolved[1].start_frame
    assert all(output.contains(words) for output, words in zip(resolved, raw))


def test_simultaneous_caption_collision_uses_half_open_time_and_screen_area() -> None:
    bounds = NormalizedRect(x=0.1, y=0.7, width=0.8, height=0.15)
    left = CaptionCuePlan(
        cue_id="caption-left",
        output=OutputInterval(start_frame=0, end_frame=10),
        resolved_lines=("left",),
        lane="lower",
        typography_token_id="caption-token-test",
        normalized_bounds=bounds,
    )
    touching = left.model_copy(update={
        "cue_id": "caption-touching",
        "output": OutputInterval(start_frame=10, end_frame=20),
    })
    overlapping = touching.model_copy(update={
        "cue_id": "caption-overlapping",
        "output": OutputInterval(start_frame=8, end_frame=20),
    })
    separate_area = overlapping.model_copy(update={
        "cue_id": "caption-separate-area",
        "normalized_bounds": NormalizedRect(x=0.1, y=0.1, width=0.8, height=0.15),
    })

    assert _simultaneous_caption_collisions((left, touching)) == []
    assert _simultaneous_caption_collisions((left, overlapping)) == [(0, 1, 2)]
    assert _simultaneous_caption_collisions((left, separate_area)) == []


def test_fixable_presentation_overlap_is_normalized_with_warning() -> None:
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="presentation-padding",
        source=SourceInterval.from_seconds(0, 4),
        output=OutputInterval(start_frame=0, end_frame=120),
    ),))
    transcript = {"words": [
        {
            "text": text, "start": start / 30, "end": end / 30,
            "confidence": 0.99, "timing_source": "verified",
        }
        for text, start, end in (
            ("abcdefghij", 10, 20),
            ("klmnopqrst", 30, 40),
            ("uvwxyzabcd", 45, 55),
        )
    ]}
    config = _config()
    config.subtitle_min_words_per_cue = 1
    config.subtitle_max_words_per_cue = 1

    plan = build_caption_plan(_plain_intent(mapping), transcript, config)

    assert plan.quality_report.status == "PASS_WITH_WARNINGS"
    assert all(
        left.output.end_frame <= right.output.start_frame
        for left, right in zip(plan.cues, plan.cues[1:])
    )
    finding = next(
        item for item in plan.quality_report.findings
        if item.code == "CAPTION_SIMULTANEOUS_OVERLAP"
    )
    assert finding.severity == "warning"
    assert finding.measured_value == 4
    assert "CAPTION_PRESENTATION_WINDOW_NORMALIZED" in plan.diagnostics


def test_semantic_motion_only_uses_brain_events_and_keeps_non_events_static() -> None:
    plan = build_caption_plan(_intent(), _word_transcript(), _config())

    assert plan.schema_version == "7C.caption-plan.1"
    assert plan.backend_id == "libass"
    assert plan.font_manifest is not None and plan.font_manifest.file_sha256
    assert plan.typography is not None
    assert plan.typography.token_id == "caption-preset:accent_yellow:2.1.0"
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
    assert plan.font_manifest is not None
    assert plan.font_manifest.font_id == "font.oswald.bold"
    assert plan.font_manifest.file_sha256 != manifest.file_sha256
    assert f"FontSHA256: {plan.font_manifest.file_sha256}" in preview_text
    assert [line.split(",", 3)[:3] for line in preview_text.splitlines() if line.startswith("Dialogue:")] == [
        line.split(",", 3)[:3] for line in final_text.splitlines() if line.startswith("Dialogue:")
    ]
    assert [cue.resolved_lines for cue in plan.cues] == [cue.resolved_lines for cue in plan.model_copy().cues]
    assert all(cue.normalized_bounds is not None for cue in plan.cues)
    karaoke_cues = [cue for cue in plan.cues if cue.primitive_id == "karaoke"]
    assert karaoke_cues and all(cue.emphasis is not None for cue in karaoke_cues)
    assert "\\kf" in final_text


def test_host_font_setting_cannot_override_approved_bundled_preset_font() -> None:
    config = _config()
    config.subtitle_font_family = "__definitely_missing_caption_font__"
    plan = build_caption_plan(_intent(), _word_transcript(), config)

    assert plan.font_manifest is not None and plan.font_manifest.fallback_used is False
    assert plan.font_manifest.font_id == "font.oswald.bold"
    assert plan.font_manifest.deployment_status == "bundled"
    assert "CAPTION_FONT_FALLBACK" not in {finding.code for finding in plan.quality_report.findings}


def test_contrast_box_caption_token_maps_to_deterministic_ass_style(tmp_path: Path) -> None:
    plan = build_caption_plan(_intent(), _word_transcript(), _config())
    assert plan.typography is not None
    boxed = plan.model_copy(update={
        "typography": plan.typography.model_copy(update={
            "token_id": "caption-preset:contrast_box:2.0.0",
        }),
    })
    path = write_caption_plan_ass(boxed, tmp_path / "boxed.ass", 1080, 1920)
    style = next(
        line for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("Style: CaptionPlan,")
    ).split(",")
    assert style[6].startswith("&H47")
    assert style[15] == "3"


@pytest.mark.parametrize("preset_id", tuple(CAPTION_PRESET_DEFINITIONS))
def test_all_approved_caption_presets_compile_with_exact_bundled_font_identity(
    preset_id: str,
    tmp_path: Path,
) -> None:
    preset = CAPTION_PRESET_DEFINITIONS[preset_id]  # type: ignore[index]
    plan = build_caption_plan(_preset_intent(preset_id), _word_transcript(), _config())

    assert plan.typography is not None and plan.typography.token_id == preset.token_id
    assert plan.font_manifest is not None
    assert plan.font_manifest.font_id == preset.preferred_font_asset_id
    assert plan.font_manifest.deployment_status == "bundled"
    assert plan.font_manifest.fallback_used is False
    assert {item.font_id for item in plan.font_manifest.companion_faces} == (
        set(preset.font_asset_ids) - {preset.preferred_font_asset_id}
    )
    controlled = materialize_caption_font_directory(plan.font_manifest, tmp_path / preset_id)
    assert len(tuple(controlled.iterdir())) == len(preset.font_asset_ids)
    ass = write_caption_plan_ass(plan, tmp_path / f"{preset_id}.ass", 540, 960)
    assert f"FontSHA256: {plan.font_manifest.file_sha256}" in ass.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize(("preset_id", "render_family", "ass_bold"), (
    ("minimal_light", "Commissioner Light", "0"),
    ("contrast_box", "Rubik SemiBold", "0"),
))
def test_static_weight_family_names_are_exact_for_libass(
    preset_id: str,
    render_family: str,
    ass_bold: str,
    tmp_path: Path,
) -> None:
    plan = build_caption_plan(_preset_intent(preset_id), _word_transcript(), _config())
    assert plan.font_manifest is not None
    assert plan.font_manifest.resolved_family == render_family
    ass = write_caption_plan_ass(plan, tmp_path / f"{preset_id}.ass", 540, 960)
    style = next(
        line for line in ass.read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("Style: CaptionPlan,")
    ).split(",")
    assert style[1] == render_family
    assert style[7] == ass_bold


def test_active_karaoke_uses_semantic_emphasis_even_when_a_beat_shares_the_cue() -> None:
    plan = build_caption_plan(_preset_intent("karaoke_yellow"), _word_transcript(), _config())

    semantic = [cue for cue in plan.cues if cue.emphasis is not None]
    assert semantic
    assert all(cue.primitive_id == "karaoke" for cue in semantic)


def test_word_pop_is_one_aligned_word_at_a_time_with_bounded_evidence_pop(tmp_path: Path) -> None:
    plan = build_caption_plan(_preset_intent("word_pop"), _word_transcript(), _config())

    assert plan.typography is not None and plan.typography.highlight_color == "#C6FF00"
    assert plan.cues and all(len(cue.words) == 1 for cue in plan.cues)
    assert all(cue.display_mode == "single_spoken_word" for cue in plan.cues)
    assert all(cue.output.start_frame == cue.words[0].output.start_frame for cue in plan.cues)
    assert all(left.output.end_frame <= right.output.start_frame for left, right in zip(plan.cues, plan.cues[1:]))
    assert all(cue.primitive_id == "word_pop" for cue in plan.cues)
    semantic = [cue for cue in plan.cues if cue.emphasis is not None]
    assert len(semantic) == 1
    assert semantic[0].scale_keyframes == (84, 118, 100)
    assert semantic[0].evidence_refs
    assert all(cue.scale_keyframes == (88, 112, 100) for cue in plan.cues if cue.emphasis is None)
    ass = write_caption_plan_ass(plan, tmp_path / "word-pop.ass", 1080, 1920).read_text(encoding="utf-8-sig")
    assert "\\fscx84\\fscy84" in ass and "\\fscx118\\fscy118" in ass
    assert "\\fscx88\\fscy88" in ass and "\\fscx112\\fscy112" in ass
    assert "\\frz" not in ass and "\\blur" not in ass and "\\move" not in ass


def test_word_pop_reduced_motion_and_short_timing_use_static_single_word_fallback(tmp_path: Path) -> None:
    reduced = build_caption_plan(
        _preset_intent("word_pop", reduced_motion=True), _word_transcript(), _config(),
    )
    assert all(cue.primitive_id == "static" for cue in reduced.cues)
    semantic = [cue for cue in reduced.cues if cue.emphasis is not None]
    assert len(semantic) == 1 and semantic[0].scale_keyframes == (106, 106, 106)
    reduced_ass = write_caption_plan_ass(
        reduced, tmp_path / "word-pop-reduced.ass", 1080, 1920,
    ).read_text(encoding="utf-8-sig")
    assert "\\fscx106\\fscy106" in reduced_ass
    assert "\\t(" not in reduced_ass

    short = build_caption_plan(
        _preset_intent("word_pop"),
        {"words": [{
            "text": "Кратко", "start": 0.0, "end": 0.1,
            "confidence": 0.99, "timing_source": "aligned",
        }]},
        _config(),
    )
    assert len(short.cues) == 1
    assert short.cues[0].primitive_id == "static"
    assert short.cues[0].fallback_reason == "short_timing"
    assert short.cues[0].output.start_frame == short.cues[0].words[0].output.start_frame


def test_word_pop_weak_timing_degrades_to_phrase_level_static_caption() -> None:
    transcript = {"segments": [{
        "start": 0.0, "end": 3.0,
        "text": "Слабый тайминг остаётся статичной фразой",
    }]}
    plan = build_caption_plan(_preset_intent("word_pop"), transcript, _config())

    assert plan.cues
    assert all(cue.display_mode == "phrase" for cue in plan.cues)
    assert all(cue.primitive_id == "static" for cue in plan.cues)
    assert all(cue.timing_mode != "word" for cue in plan.cues)
    assert "WEAK_TIMING_DEGRADED_TO_PHRASE_STATIC" in plan.diagnostics


def test_word_pop_localizes_weak_timing_without_disabling_trusted_word_cadence() -> None:
    transcript = {"words": [
        {"text": "trusted-one", "start": 0.0, "end": 0.4, "confidence": 0.99, "timing_source": "aligned"},
        {"text": "weak", "start": 0.5, "end": 0.9, "confidence": 0.40, "timing_source": "phrase"},
        {"text": "timing", "start": 0.9, "end": 1.3, "confidence": 0.40, "timing_source": "phrase"},
        {"text": "trusted-two", "start": 1.4, "end": 1.9, "confidence": 0.99, "timing_source": "verified"},
    ]}
    plan = build_caption_plan(_preset_intent("word_pop"), transcript, _config())

    single_word = [cue for cue in plan.cues if cue.display_mode == "single_spoken_word"]
    phrase = [cue for cue in plan.cues if cue.display_mode == "phrase"]
    assert [cue.words[0].text for cue in single_word] == ["trusted-one", "trusted-two"]
    assert all(cue.primitive_id == "word_pop" for cue in single_word)
    assert phrase and all(cue.primitive_id == "static" for cue in phrase)
    assert " ".join(word.text for cue in phrase for word in cue.words) == "weak timing"
    assert all(cue.output.start_frame >= cue.words[0].output.start_frame for cue in plan.cues)


def test_editorial_bundles_regular_and_bold_faces_and_marks_semantic_weight(tmp_path: Path) -> None:
    plan = build_caption_plan(_preset_intent("editorial_narrow"), _word_transcript(), _config())
    assert plan.font_manifest is not None
    assert plan.font_manifest.font_id == "font.pt-sans-narrow.regular"
    assert [face.font_id for face in plan.font_manifest.companion_faces] == [
        "font.pt-sans-narrow.bold",
    ]
    controlled = materialize_caption_font_directory(plan.font_manifest, tmp_path / "editorial-fonts")
    assert len(tuple(controlled.iterdir())) == 2
    ass = write_caption_plan_ass(plan, tmp_path / "editorial.ass", 1080, 1920).read_text(encoding="utf-8-sig")
    assert "\\b1" in ass and "\\b0" in ass
