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


def test_semantic_motion_only_uses_brain_events_and_keeps_non_events_static() -> None:
    plan = build_caption_plan(_intent(), _word_transcript(), _config())

    assert plan.schema_version == "7C.caption-plan.1"
    assert plan.backend_id == "libass"
    assert plan.font_manifest is not None and plan.font_manifest.file_sha256
    assert any(cue.beat_role == BeatRole.HOOK and cue.primitive_id == "slide" for cue in plan.cues)
    assert any(cue.beat_role == BeatRole.PAYOFF and cue.primitive_id == "scale" for cue in plan.cues)
    emphasized = [cue for cue in plan.cues if cue.emphasis is not None]
    assert len(emphasized) == 1
    assert emphasized[0].primitive_id == "karaoke"
    assert emphasized[0].emphasis is not None
    assert emphasized[0].emphasis.word_indexes
    assert any("три шага" in line.casefold() for line in emphasized[0].resolved_lines)
    assert all(cue.evidence_refs for cue in plan.cues if cue.primitive_id != "static")
    assert sum(cue.primitive_id == "karaoke" for cue in plan.cues) == 1
    assert plan.quality_report.metrics.semantic_emphasis_count == 1
    with pytest.raises(ValidationError, match="Instance is frozen"):
        plan.quality_report.metrics.cue_count = 99  # type: ignore[misc]


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
