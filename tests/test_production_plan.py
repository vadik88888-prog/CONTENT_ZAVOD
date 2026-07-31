from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.cli import build_parser
from app.config import AppConfig
from app.content_transformation import run_content_transformation
from app.errors import ProductionPlanError
from app.models import Candidate
from app.pipeline import Pipeline, StageTracker
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan, validate_renderer_handoff
from app.production_plan import ProductionPlanEnvelopeContext, build_production_plan, production_summary
from app.reporting import make_report
from app.semantic_extraction import build_source_context
from app.utils import read_json, stable_text_hash, write_json


def _transformation_outcome() -> dict:
    config = AppConfig()
    config.transformation.ai_strategy = "local_only"
    text = "First, measure the source data. Then keep the original claim without changing the facts."
    candidate = Candidate("candidate-production-001", 4, 20, text, transcript_segment_ids=[0])
    transcript = {"language": "en", "segments": [{"start": 4, "end": 20, "text": text}]}
    features = {"segments": [{
        "id": 0, "start": 4, "end": 20, "sentence_start": True, "sentence_end": True,
        "speech_density": 0.6, "pause_before_seconds": 0.1, "pause_after_seconds": 0.2,
        "filler_word_ratio": 0.0, "repetition_score": 0.0,
    }]}
    context = build_source_context(
        {"id": "source-production", "path": "source.mp4"}, {}, candidate, transcript,
        features, {}, {"boundaries": []}, config.transformation,
    )
    result = run_content_transformation(context, config.transformation, None, force_local=True)
    assert result["final_script"]["full_text"]
    return result


def _voiceover_production():
    production = AppConfig().production
    production.audio_mode = "voiceover"
    return production


def _boundary_decision(*, semantic_completion: bool = True, payoff_preserved: bool = True) -> dict:
    source_range = {"start_seconds": 4.0, "end_seconds": 20.0}
    return {
        "schema_version": "5C.1",
        "decision_id": "boundary-candidate-production-001-safe",
        "candidate_id": "candidate-production-001",
        "rough_range": dict(source_range),
        "refined_range": dict(source_range),
        "allowed_source_range": dict(source_range),
        "start_reason": "Complete opening word and sentence boundary.",
        "end_reason": "Complete ending word and completed thought.",
        "word_integrity": True,
        "sentence_integrity": True,
        "semantic_completion": semantic_completion,
        "payoff_preserved": payoff_preserved,
        "continuation_risk": 0.1,
        "continuation_risk_threshold": 0.65,
        "pre_roll_seconds": 0.1,
        "post_roll_seconds": 0.2,
        "confidence": 0.9,
        "start_evidence": {"reason": "sentence_start", "speaker_change": False, "scene_boundary_distance": 0.2},
        "end_evidence": {"reason": "sentence_completion", "speaker_change": False, "scene_boundary_distance": 0.3},
        "pause_evidence": {"pre_roll_seconds": 0.1, "post_roll_seconds": 0.2},
        "required_evidence": [
            {
                "requirement_type": "hook", "required": True,
                "source_range": {"start_seconds": 4.0, "end_seconds": 12.0},
                "transcript_segment_id": 0, "reason": "Hook survives.", "evidence": {"text": "First claim"},
            },
            {
                "requirement_type": "completion", "required": True,
                "source_range": {"start_seconds": 12.0, "end_seconds": 20.0},
                "transcript_segment_id": 0, "reason": "Thought completes.", "evidence": {"text": "Complete ending"},
            },
            {
                "requirement_type": "payoff", "required": True,
                "source_range": {"start_seconds": 18.0, "end_seconds": 20.0},
                "transcript_segment_id": 0, "reason": "Payoff survives.", "evidence": {"text": "Payoff"},
            },
        ],
        "safe_start_points": [4.0, 12.0, 18.0],
        "safe_end_points": [12.0, 18.0, 20.0],
        "fallback_used": False,
        "fallback_reason": None,
    }


def _outcome_with_boundary() -> dict:
    outcome = _transformation_outcome()
    outcome["source_context"]["boundary_decision"] = _boundary_decision()
    return outcome


def _native_envelope_context() -> ProductionPlanEnvelopeContext:
    return ProductionPlanEnvelopeContext(
        project_id="project-production", run_id="run-production", analysis_id="analysis-production",
        analysis_fingerprint="b" * 64, source_sha256="c" * 64,
        transcript_sha256="d" * 64, preset_id="documentary", preset_version="4B.1",
        platform="universal", target_width=1080, target_height=1920, target_fps=30,
        created_at="2026-07-31T00:00:00Z",
    )


def _native_plan() -> ProductionPlan:
    return build_production_plan(
        _outcome_with_boundary(),
        AppConfig().production,
        envelope_context=_native_envelope_context(),
    )


def test_pydantic_production_models_reject_invalid_narration() -> None:
    with pytest.raises(ValidationError):
        NarrationSegment(
            segment_id="narration-001", order=1, estimated_duration_seconds=1,
            text="", narration_role="intro", source_sentence_id="s", fact_ids=[],
            source_segment_ids=[], word_count=0, words_per_second=2, voice_profile_id="v",
        )


def test_builder_creates_narration_dialogue_pauses_and_placeholders() -> None:
    plan = build_production_plan(_transformation_outcome(), _voiceover_production())
    assert isinstance(plan, ProductionPlan)
    assert plan.timeline.narration_count == len(plan.subtitle_track.cues)
    assert plan.timeline.dialogue_count == len(plan.dialogue_mappings)
    assert plan.timeline.pause_count == plan.timeline.narration_count - 1
    assert {item.layer_type for item in plan.audio_layers} == {"narration", "original_dialogue", "music", "effects"}
    assert all(item.status == "placeholder" for item in plan.audio_layers)
    assert plan.metadata.tts_generated is False
    assert plan.metadata.render_generated is False


def test_native_envelope_is_deterministic_and_binds_v2_identity_contract() -> None:
    first = _native_plan()
    second = _native_plan()

    assert first.schema_version == "5F.1"
    assert first.envelope and first.envelope.compatibility_mode == "native"
    assert first.envelope.identity.project_id == "project-production"
    assert first.envelope.boundary_decision_ref == first.boundary_decision.decision_id
    assert first.envelope.input_fingerprints.final_script_sha256 == first.metadata.final_script_hash
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.plan_fingerprint() == second.plan_fingerprint()


def test_native_renderer_handoff_rejects_wrong_identity_and_stale_transcript_without_rendering() -> None:
    plan = _native_plan()
    transcript = {"segments": []}
    # The envelope context deliberately uses a known transcript fingerprint.
    plan = plan.model_copy(update={
        "envelope": plan.envelope.model_copy(update={
            "input_fingerprints": plan.envelope.input_fingerprints.model_copy(update={
                "transcript_sha256": stable_text_hash(json.dumps(transcript, sort_keys=True)),
            }),
        }),
    })
    audio = SimpleNamespace(metadata=SimpleNamespace(
        plan_reference=plan.reference(), production_plan_id=plan.plan_id, source_id=plan.metadata.source_id,
    ))

    accepted = validate_renderer_handoff(
        plan, audio, source_id=plan.metadata.source_id, source_sha256="c" * 64, transcript=transcript,
        expected_project_id="project-production", expected_analysis_id="analysis-production",
        expected_plan_run_id="run-production", expected_candidate_id=plan.metadata.candidate_id,
        expected_preset_id="documentary", expected_preset_version="4B.1", expected_platform="universal",
        expected_target=(1080, 1920, 30.0),
    )
    assert accepted is None

    wrong_source = validate_renderer_handoff(
        plan, audio, source_id="wrong-source", source_sha256="c" * 64, transcript=transcript,
    )
    stale_transcript = validate_renderer_handoff(
        plan, audio, source_id=plan.metadata.source_id, source_sha256="c" * 64,
        transcript={"segments": [{"id": 0}]},
    )
    changed_boundary_plan = plan.model_copy(update={
        "boundary_decision": plan.boundary_decision.model_copy(update={"confidence": 0.8}),
    })
    changed_boundary_audio = SimpleNamespace(metadata=SimpleNamespace(
        plan_reference=changed_boundary_plan.reference(),
        production_plan_id=changed_boundary_plan.plan_id,
        source_id=changed_boundary_plan.metadata.source_id,
    ))
    stale_boundary = validate_renderer_handoff(
        changed_boundary_plan,
        changed_boundary_audio,
        source_id=changed_boundary_plan.metadata.source_id,
        source_sha256="c" * 64,
        transcript=transcript,
    )
    assert wrong_source and wrong_source.code == "IDENTITY_MISMATCH"
    assert stale_transcript and stale_transcript.code == "STALE_INPUTS"
    assert stale_boundary and stale_boundary.code == "STALE_INPUTS"


def test_native_envelope_blocks_candidate_or_boundary_tampering_and_unknown_legacy_versions() -> None:
    plan = _native_plan()
    wrong_candidate = plan.model_dump(mode="json")
    wrong_candidate["metadata"]["candidate_id"] = "candidate-other"
    wrong_boundary = plan.model_dump(mode="json")
    wrong_boundary["envelope"]["boundary_decision_ref"] = "boundary-other"
    unknown_legacy = build_production_plan(_transformation_outcome(), AppConfig().production).model_dump(mode="json")
    unknown_legacy.pop("envelope")
    unknown_legacy["metadata"]["plan_version"] = "2A.9"
    unknown_top_level = build_production_plan(_transformation_outcome(), AppConfig().production).model_dump(mode="json")
    unknown_top_level.pop("envelope")
    unknown_top_level["schema_version"] = "2A.9"

    with pytest.raises(ValidationError, match="IDENTITY_MISMATCH"):
        ProductionPlan.model_validate(wrong_candidate)
    with pytest.raises(ValidationError, match="IDENTITY_MISMATCH"):
        ProductionPlan.model_validate(wrong_boundary)
    with pytest.raises(ValidationError, match="UNSUPPORTED_LEGACY_PLAN_VERSION"):
        ProductionPlan.model_validate(unknown_legacy)
    with pytest.raises(ValidationError, match="UNSUPPORTED_LEGACY_PLAN_VERSION"):
        ProductionPlan.model_validate(unknown_top_level)


def test_production_plan_persists_boundary_decision_and_links_every_dialogue_source_range() -> None:
    plan = build_production_plan(_outcome_with_boundary(), AppConfig().production)

    assert plan.boundary_decision is not None
    assert plan.boundary_decision.schema_version == "5C.1"
    assert all(
        segment.boundary_decision_id == plan.boundary_decision.decision_id
        for segment in plan.dialogue_mappings
    )
    assert all(
        plan.boundary_decision.allowed_source_range.start_seconds <= segment.source_start_seconds
        and segment.source_end_seconds <= plan.boundary_decision.allowed_source_range.end_seconds
        for segment in plan.dialogue_mappings
    )

    voiceover_plan = build_production_plan(_outcome_with_boundary(), _voiceover_production())
    narration = [segment for segment in voiceover_plan.segments if isinstance(segment, NarrationSegment)]
    assert narration and all(segment.source_ranges for segment in narration)
    assert all(
        source.source_start_seconds >= voiceover_plan.boundary_decision.allowed_source_range.start_seconds
        and source.source_end_seconds <= voiceover_plan.boundary_decision.allowed_source_range.end_seconds
        and segment.boundary_decision_id == voiceover_plan.boundary_decision.decision_id
        for segment in narration for source in segment.source_ranges
    )


def test_production_plan_blocks_dialogue_word_cut_after_boundary_handoff() -> None:
    outcome = _outcome_with_boundary()
    outcome["semantic_representation"]["supporting_facts"][0]["evidence_start"] = 4.2

    with pytest.raises(ProductionPlanError, match="BOUNDARY_WORD_CUT"):
        build_production_plan(outcome, AppConfig().production)


def test_production_plan_blocks_dialogue_range_outside_selected_boundary() -> None:
    outcome = _outcome_with_boundary()
    decision = outcome["source_context"]["boundary_decision"]
    decision["refined_range"]["start_seconds"] = 5.0
    decision["allowed_source_range"]["start_seconds"] = 5.0
    decision["required_evidence"][0]["source_range"]["start_seconds"] = 5.0
    decision["safe_start_points"].append(5.0)

    with pytest.raises(ProductionPlanError, match="BOUNDARY_SOURCE_RANGE_OUTSIDE"):
        build_production_plan(outcome, AppConfig().production)


def test_production_plan_blocks_lost_payoff_after_boundary_handoff() -> None:
    outcome = _outcome_with_boundary()
    for fact in outcome["semantic_representation"]["supporting_facts"]:
        fact["evidence_end"] = 18.0

    with pytest.raises(ProductionPlanError, match="BOUNDARY_PAYOFF_LOST"):
        build_production_plan(outcome, AppConfig().production)


def test_production_plan_blocks_incomplete_boundary_decision() -> None:
    outcome = _outcome_with_boundary()
    outcome["source_context"]["boundary_decision"] = _boundary_decision(semantic_completion=False)

    with pytest.raises(ProductionPlanError, match="BOUNDARY_INCOMPLETE_THOUGHT"):
        build_production_plan(outcome, AppConfig().production)


def test_production_plan_blocks_question_without_answer_context() -> None:
    outcome = _outcome_with_boundary()
    outcome["source_context"]["boundary_decision"]["question_context"] = {
        "end_is_question": True,
        "answer_or_completion_included": False,
    }

    with pytest.raises(ProductionPlanError, match="BOUNDARY_QUESTION_CONTEXT_MISSING"):
        build_production_plan(outcome, AppConfig().production)


def test_legacy_production_plan_without_boundary_decision_remains_readable() -> None:
    plan = build_production_plan(_transformation_outcome(), AppConfig().production)
    legacy = deepcopy(plan.model_dump(mode="json"))
    legacy.pop("boundary_decision")
    for segment in legacy["dialogue_mappings"]:
        segment.pop("boundary_decision_id")
    for segment in legacy["segments"]:
        if segment["segment_type"] == "original_dialogue":
            segment.pop("boundary_decision_id")

    migrated = ProductionPlan.model_validate(legacy)

    assert migrated.boundary_decision is None
    assert all(segment.boundary_decision_id is None for segment in migrated.dialogue_mappings)


def test_dialogue_mapping_has_fact_transcript_timestamps_speaker_and_confidence() -> None:
    plan = build_production_plan(_transformation_outcome(), AppConfig().production)
    dialogue = plan.dialogue_mappings[0]
    assert dialogue.fact_id.startswith("fact-")
    assert dialogue.transcript_segment_id == 0
    assert dialogue.source_start_seconds == 4
    assert dialogue.source_end_seconds == 20
    assert dialogue.speaker == "original_speaker_unknown"
    assert 0 <= dialogue.confidence <= 1
    assert dialogue.is_placeholder and not dialogue.timeline_included


def test_timeline_is_deterministic_and_keeps_dialogue_linked_without_counting_it_twice() -> None:
    config = AppConfig()
    config.production.audio_mode = "voiceover"
    first = build_production_plan(_transformation_outcome(), config.production)
    second = build_production_plan(_transformation_outcome(), config.production)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    active = [entry for entry in first.timeline.entries if entry.included_in_master_timeline]
    assert active[0].estimated_start_seconds == 0
    assert all(left.estimated_end_seconds <= right.estimated_start_seconds for left, right in zip(active, active[1:]))
    assert first.timeline.estimated_duration_seconds == active[-1].estimated_end_seconds
    assert all(entry.linked_segment_ids for entry in first.timeline.entries if not entry.included_in_master_timeline)


def test_builder_marks_cta_and_summary_is_media_free() -> None:
    transformation = _transformation_outcome()
    transformation["final_script"]["sentences"][-1]["role"] = "cta"
    plan = build_production_plan(transformation, _voiceover_production())
    assert any(item.narration_role == "cta" for item in plan.segments if isinstance(item, NarrationSegment))
    summary = production_summary(plan)
    assert "TTS" in summary and "no media was generated" in summary


def test_production_plan_cache_and_artifacts(tmp_path: Path) -> None:
    config = AppConfig()
    pipeline = Pipeline(tmp_path, config)
    tracker = StageTracker(tmp_path / "state.json")
    transformation = {"items": [_transformation_outcome()]}
    first = pipeline._build_production_plans(tracker, transformation, tmp_path / "work", tmp_path / "output")
    second = pipeline._build_production_plans(tracker, transformation, tmp_path / "work", tmp_path / "output")
    assert first["status"] == "completed"
    assert second["cache"]["hit_count"] == 1
    for name in ("production-plan.json", "timeline.json", "production-summary.txt"):
        assert (tmp_path / "output" / name).is_file()
    assert json.loads((tmp_path / "output" / "production-plan.json").read_text(encoding="utf-8"))["timeline"]["timeline_version"] == "3A.0"


def test_production_report_and_cli_flags(tmp_path: Path) -> None:
    plan = build_production_plan(_transformation_outcome(), AppConfig().production).model_dump(mode="json")
    report_path = tmp_path / "report.json"
    make_report(
        report_path, {}, {}, AppConfig(), {}, 0, 0, [], [], [], {}, False, False,
        production_plan={"enabled": True, "status": "completed", "production_plan": plan, "segments": plan["segments"], "estimated_duration": plan["timeline"]["estimated_duration_seconds"], "dialogue_count": plan["timeline"]["dialogue_count"], "narration_count": plan["timeline"]["narration_count"], "pause_count": plan["timeline"]["pause_count"], "timeline_version": "3A.0"},
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["production_plan"]["timeline_version"] == "3A.0"
    arguments = build_parser().parse_args([
        "process", "--input", "source.mp4", "--production-plan-only", "--recompute-production-plan",
    ])
    assert arguments.production_plan_only and arguments.recompute_production_plan


def test_provider_disabled_local_final_script_still_builds_plan() -> None:
    outcome = _transformation_outcome()
    assert outcome["provider"] == "local"
    plan = build_production_plan(outcome, AppConfig().production)
    assert plan.status == "draft"


def test_production_plan_rejects_invalid_final_script_with_safe_boundary_diagnostics() -> None:
    outcome = _transformation_outcome()
    source_text = outcome["source_context"]["transcript_text"]
    outcome["final_script"]["candidate_id"] = "candidate-wrong"
    outcome["final_script_source"] = "ai"

    with pytest.raises(ProductionPlanError) as raised:
        build_production_plan(outcome, AppConfig().production)

    message = str(raised.value)
    assert "expected_candidate_id=candidate-production-001" in message
    assert "actual_candidate_id=candidate-wrong" in message
    assert "sentences_count=" in message
    assert "source=ai" in message
    assert source_text not in message


def test_dialogue_only_final_script_builds_plan_without_tts_narration() -> None:
    outcome = _transformation_outcome()
    outcome["final_script"]["production_ready_for_tts"] = False

    plan = build_production_plan(outcome, _voiceover_production())

    assert not any(isinstance(segment, NarrationSegment) for segment in plan.segments)
    assert plan.dialogue_mappings
    assert all(isinstance(segment, DialogueSegment) for segment in plan.segments)
    assert plan.timeline.narration_count == 0
    assert plan.timeline.dialogue_count == len(plan.dialogue_mappings)


def test_default_audio_mode_builds_source_dialogue_without_tts_narration() -> None:
    plan = build_production_plan(_transformation_outcome(), AppConfig().production)

    assert plan.audio_mode == "original"
    assert not plan.tts_eligible
    assert not any(isinstance(segment, NarrationSegment) for segment in plan.segments)
    assert plan.dialogue_mappings


def test_production_plan_only_does_not_overwrite_existing_render_cache(tmp_path: Path) -> None:
    pipeline = Pipeline(tmp_path, AppConfig(), production_plan_only=True)
    tracker = StageTracker(tmp_path / "state.json")
    render_path = tmp_path / "render.json"
    write_json(render_path, {"output_files": ["already-rendered.mp4"]})
    tracker.start("render", "stable-render-cache")
    tracker.finish("render")
    result = pipeline._skip_render_for_production_plan(tracker, render_path)
    assert result["output_files"] == []
    assert read_json(render_path, {})["output_files"] == ["already-rendered.mp4"]
    assert tracker.data["stages"]["render"]["status"] == "completed"
