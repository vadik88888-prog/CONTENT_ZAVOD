from __future__ import annotations

import json
from pathlib import Path

from app.audio_service import AudioCompositionService
from app.config import AppConfig
from app.creative_contracts import (
    CompiledRenderPlan,
    EditMapSegment,
    OutputInterval,
    SourceInterval,
    SourceOutputTimeMap,
)
from app.creative_evidence import build_native_evidence_handoff
from app.pipeline import Pipeline, StageTracker, _hash
from app.production_models import BoundaryDecision, ProductionPlan
from app.production_plan import ProductionPlanEnvelopeContext
from app.sources import Source
from app.tts_providers import MockTTSProvider
from app.tts_service import TTSService
from app.utils import stable_file_hash
from tests.test_audio_composition import _audio_config, _plan
from tests.test_video_composition import _source_video


def _phase6_artifacts(candidate_id: str) -> tuple[dict, dict, dict]:
    candidate = {
        "id": candidate_id,
        "story_unit_id": "story-native-1",
        "story_unit_ids": ["story-native-1"],
        "multimodal_provenance": {
            "schema_version": "6C.1",
            "analysis_run_id": "analysis-phase6",
            "visual_evidence": [],
        },
        "vision_pass2_evidence": {"status": "skipped", "result": None},
    }
    timeline = {
        "schema_version": "6A.1",
        "source_id": "source-audio",
        "analysis_run_id": "analysis-phase6",
        "scenes": [{
            "scene_id": "scene-phase6-1",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "confidence": 0.94,
        }, {
            "scene_id": "scene-phase6-cutaway",
            "start_seconds": 2.1,
            "end_seconds": 2.8,
            "confidence": 0.93,
        }],
        "visual_event_map": [{
            "event_id": "visual-subject-1",
            "event_type": "subject_observation",
            "start_seconds": 1.35,
            "end_seconds": 1.35,
            "confidence": 0.96,
            "observation": {
                "faces": {"visible_count": 1, "active_speaker_confidence": 0.93},
                "active_subject": {
                    "target_type": "primary_face",
                    "normalized_bbox": {
                        "normalized_x": 0.38,
                        "normalized_y": 0.43,
                        "normalized_width": 0.28,
                        "normalized_height": 0.52,
                    },
                },
                "objects_persons": {"person_is_active_target": True},
                "motion_action": {"gesture_observed": True},
                "framing_relevance": {"scene_id": "scene-phase6-1"},
            },
        }, {
            "event_id": "visual-object-cutaway",
            "event_type": "subject_observation",
            "start_seconds": 2.35,
            "end_seconds": 2.35,
            "confidence": 0.94,
            "observation": {
                "active_subject": {
                    "target_type": "important_object",
                    "normalized_bbox": {
                        "normalized_x": 0.58,
                        "normalized_y": 0.50,
                        "normalized_width": 0.34,
                        "normalized_height": 0.46,
                    },
                },
                "motion_action": {"gesture_observed": True},
                "framing_relevance": {"scene_id": "scene-phase6-cutaway"},
            },
            "action": "demonstration",
        }],
    }
    stories = {"story_units": [{
        "story_unit_id": "story-native-1",
        "start": 1.0,
        "end": 2.8,
        "confidence": 0.93,
        "information_density": 0.86,
        "hook_seed": "Source dialogue",
        "setup": "A deterministic setup.",
        "development": "Evidence is explained.",
        "payoff": "remains audible",
        "ending": "The payoff is preserved.",
        "multimodal_evidence": {
            "schema_version": "6A.story-evidence.1",
            "analysis_run_id": "analysis-phase6",
        },
    }]}
    return candidate, timeline, stories


def _native_plan_for_media(config: AppConfig, source: Source, transcript: dict) -> ProductionPlan:
    legacy = _plan(narration=False, dialogue=True)
    boundary = BoundaryDecision.model_validate({
        "schema_version": "5C.1",
        "decision_id": "boundary-candidate-audio-native",
        "candidate_id": legacy.metadata.candidate_id,
        "rough_range": {"start_seconds": 1.0, "end_seconds": 2.0},
        "refined_range": {"start_seconds": 1.0, "end_seconds": 2.0},
        "allowed_source_range": {"start_seconds": 1.0, "end_seconds": 2.0},
        "start_reason": "Complete source sentence start.",
        "end_reason": "Complete source sentence and payoff.",
        "word_integrity": True,
        "sentence_integrity": True,
        "semantic_completion": True,
        "payoff_preserved": True,
        "continuation_risk": 0.0,
        "continuation_risk_threshold": 0.65,
        "pre_roll_seconds": 0.0,
        "post_roll_seconds": 0.0,
        "confidence": 0.95,
        "start_evidence": {"reason": "sentence_start"},
        "end_evidence": {"reason": "sentence_completion"},
        "pause_evidence": {},
        "required_evidence": [{
            "requirement_type": "completion",
            "required": True,
            "source_range": {"start_seconds": 1.0, "end_seconds": 2.0},
            "transcript_segment_id": 0,
            "reason": "The source sentence remains complete.",
            "evidence": {"text": "Source dialogue remains audible."},
        }],
        "safe_start_points": [1.0],
        "safe_end_points": [2.0],
        "fallback_used": False,
        "fallback_reason": None,
    })
    context = ProductionPlanEnvelopeContext(
        project_id="project-native-regression",
        run_id="run-native-regression",
        analysis_id="analysis-phase6",
        analysis_fingerprint="b" * 64,
        source_sha256=stable_file_hash(source.path),
        transcript_sha256=_hash(transcript),
        preset_id=config.product_flow.subtitle_preset,
        preset_version=config.product_flow.preset_version,
        platform=config.product_flow.platform,
        target_width=config.production_render.output_width,
        target_height=config.production_render.output_height,
        target_fps=config.production_render.output_fps,
        created_at="2026-08-10T00:00:00Z",
    )
    raw = legacy.model_dump(mode="json")
    raw["boundary_decision"] = boundary.model_dump(mode="json")
    for item in raw["dialogue_mappings"]:
        item["boundary_decision_id"] = boundary.decision_id
    for item in raw["segments"]:
        if item.get("segment_type") == "original_dialogue":
            item["boundary_decision_id"] = boundary.decision_id
    raw["envelope"] = context.build(
        candidate_id=legacy.metadata.candidate_id,
        source_id=legacy.metadata.source_id,
        final_script_hash=legacy.metadata.final_script_hash,
        boundary_decision=boundary,
    ).model_dump(mode="json")
    return ProductionPlan.model_validate(raw)


def test_phase6_artifacts_build_rich_native_handoff_without_analysis_calls() -> None:
    plan = _plan()
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 2.0),
        output=OutputInterval.from_seconds(0.0, 1.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)

    config = AppConfig()
    config.production_render.same_source_broll_allowed = True
    handoff = build_native_evidence_handoff(
        plan,
        mapping,
        config,
        candidate=candidate,
        multimodal_timeline=timeline,
        story_units=stories,
    )

    assert handoff.intent.confidence > 0
    assert handoff.intent.provenance[:3] == (
        "phase6:candidates.scored",
        "phase6:multimodal_timeline",
        "phase6:story_units",
    )
    assert handoff.intent.beats
    assert handoff.intent.semantic_emphasis
    assert handoff.intent.composition_targets
    assert handoff.intent.motion_events
    assert handoff.intent.source_broll
    assert handoff.execution_status == "native_rich"
    assert handoff.target_observations[0].evidence_ref in {
        item.evidence_ref for item in handoff.intent.evidence_manifest
    }
    assert handoff.source_scenes[0].story_unit_ids == ("story-native-1",)


def test_same_source_cutaway_without_explicit_usage_contract_falls_back_to_a_roll() -> None:
    plan = _plan()
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 2.0),
        output=OutputInterval.from_seconds(0.0, 1.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)

    handoff = build_native_evidence_handoff(
        plan, mapping, AppConfig(), candidate=candidate,
        multimodal_timeline=timeline, story_units=stories,
    )

    assert handoff.execution_status == "native_fallback"
    assert not handoff.intent.source_broll
    assert "SOURCE_BROLL_USAGE_NOT_AUTHORIZED" in handoff.reason_codes
    assert all(scene.rights_status == "uncertain" for scene in handoff.source_scenes)


def test_pipeline_production_runner_hands_phase6_evidence_to_native_mp4(
    tmp_path: Path, monkeypatch,
) -> None:
    config = _audio_config()
    config.production_render.enabled = True
    config.production_render.output_width = 180
    config.production_render.output_height = 320
    config.production_render.output_fps = 30
    config.production_render.video_bitrate = "500k"
    config.production_render.encoder = "cpu"
    config.production_render.same_source_broll_allowed = True
    config.validate()
    source = Source(
        "source-audio",
        _source_video(tmp_path / "source.mp4"),
        "source.mp4",
        "test",
    )
    transcript = {
        "source_id": source.id,
        "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": "Source dialogue."}],
        "words": [
            {"start": 1.0, "end": 1.45, "text": "Source"},
            {"start": 1.45, "end": 2.0, "text": "dialogue."},
        ],
    }
    plan = _native_plan_for_media(config, source, transcript)
    candidate_output = tmp_path / "candidate-output"
    tts = TTSService(tmp_path, config).generate(
        plan,
        tmp_path / "work",
        candidate_output,
        provider=MockTTSProvider(),
    )
    audio = AudioCompositionService(tmp_path, config).compose(
        plan,
        source,
        transcript,
        tts.model_dump(mode="json"),
        tmp_path / "work",
        candidate_output,
    )
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    production = {"items": [{
        "candidate_id": plan.metadata.candidate_id,
        "status": "completed",
        "requested_index": 1,
        "plan": plan.model_dump(mode="json"),
    }]}
    audio_report = {"items": [{
        "candidate_id": plan.metadata.candidate_id,
        "status": audio.status,
        "output_directory": str(candidate_output),
    }]}
    pipeline = Pipeline(tmp_path, config, run_id="run-native-regression")
    monkeypatch.setattr(
        "app.video_composition.build_reframe_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy reframe builder called")),
    )
    monkeypatch.setattr(
        "app.video_composition.build_subtitle_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy subtitle builder called")),
    )

    rendered = pipeline._run_production_render(
        StageTracker(tmp_path / "pipeline-state.json"),
        production,
        audio_report,
        source,
        transcript,
        tmp_path / "work",
        tmp_path / "run-output",
        creative_candidates=[candidate],
        multimodal_timeline=timeline,
        story_units=stories,
    )

    assert rendered.get("output_files"), rendered
    output = Path(rendered["output_files"][0])
    item_report = rendered["items"][0]["report"]
    compiled_path = candidate_output / "production-render" / "compiled-render-plan.json"
    compiled = CompiledRenderPlan.model_validate(json.loads(compiled_path.read_text(encoding="utf-8")))
    assert output.is_file() and output.suffix == ".mp4"
    assert compiled.compatibility_mode == "native"
    assert compiled.composition_plan.segments
    assert compiled.motion_plan.events
    assert compiled.source_broll_plan.segments
    assert compiled.input_fingerprints.creative_intent_sha256 != compiled.input_fingerprints.production_plan_sha256
    assert item_report["creative_qc_source"] == "compiled_render_plan"
    assert item_report["quality"]["source_of_truth"] == "compiled_render_plan"
    assert item_report["caption_plan"]["intent_id"] == compiled.intent_id
    assert item_report["composition_plan"]["intent_id"] == compiled.intent_id
    assert item_report["execution_status"] == "native_rich"
    assert (candidate_output / "production-render" / "creative-intent.json").is_file()
    assert (candidate_output / "production-render" / "creative-handoff.json").is_file()
    assert (candidate_output / "production-render" / "creative-execution.json").is_file()
    assert not (candidate_output / "production-render" / "reframe-plan.json").exists()
    assert not (candidate_output / "production-render" / "subtitle-project.json").exists()
