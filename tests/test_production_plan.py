from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.cli import build_parser
from app.config import AppConfig
from app.content_transformation import run_content_transformation
from app.errors import ProductionPlanError
from app.models import Candidate
from app.pipeline import Pipeline, StageTracker
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan
from app.production_plan import build_production_plan, production_summary
from app.reporting import make_report
from app.semantic_extraction import build_source_context
from app.utils import read_json, write_json


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
