from __future__ import annotations

import json
from pathlib import Path

from app.audio_service import AudioCompositionService
from app.config import AppConfig
from app.content_transformation import run_content_transformation
from app.creative_contracts import (
    CompiledRenderPlan,
    EditMapSegment,
    LayoutFamily,
    OutputInterval,
    SourceInterval,
    SourceOutputTimeMap,
)
from app.creative_evidence import build_native_evidence_handoff
from app.creative_execution import compile_native_creative_plan
from app.models import Candidate
from app.pipeline import Pipeline, StageTracker, _hash
from app.production_models import BoundaryDecision, ProductionPlan
from app.production_plan import ProductionPlanEnvelopeContext, build_production_plan
from app.semantic_extraction import build_source_context
from app.sources import Source
from app.tts_providers import MockTTSProvider
from app.tts_service import TTSService
from app.utils import stable_file_hash
from app.video_composition import _reconcile_native_execution_status
from tests.test_audio_composition import _audio_config
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


def _current_native_plan(
    config: AppConfig | None = None,
    *,
    transcript: dict | None = None,
    boundary: BoundaryDecision | None = None,
    source_sha256: str = "c" * 64,
) -> ProductionPlan:
    config = config or AppConfig()
    transcript = transcript or {
        "source_id": "source-audio",
        "segments": [{
            "id": 0,
            "start": 1.0,
            "end": 2.0,
            "text": "Source dialogue remains audible.",
        }],
        "words": [],
    }
    source_id = str(transcript.get("source_id") or "source-audio")
    source_segments = [
        item for item in transcript.get("segments", [])
        if isinstance(item, dict)
    ]
    assert source_segments
    start = min(float(item["start"]) for item in source_segments)
    end = max(float(item["end"]) for item in source_segments)
    candidate_id = boundary.candidate_id if boundary is not None else "candidate-audio"
    boundary_payload = boundary.model_dump(mode="json") if boundary is not None else {
        "schema_version": "5C.1",
        "decision_id": "boundary-candidate-audio-native",
        "candidate_id": candidate_id,
        "rough_range": {"start_seconds": start, "end_seconds": end},
        "refined_range": {"start_seconds": start, "end_seconds": end},
        "allowed_source_range": {"start_seconds": start, "end_seconds": end},
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
            "source_range": {"start_seconds": start, "end_seconds": end},
            "transcript_segment_id": int(source_segments[0]["id"]),
            "reason": "The source sentence remains complete.",
            "evidence": {"text": str(source_segments[0].get("text") or "Source dialogue")},
        }],
        "safe_start_points": [start],
        "safe_end_points": [end],
        "fallback_used": False,
        "fallback_reason": None,
    }
    text = " ".join(str(item.get("text") or "").strip() for item in source_segments).strip()
    candidate = Candidate(
        candidate_id,
        start,
        end,
        text,
        transcript_segment_ids=[int(item["id"]) for item in source_segments],
        boundary_diagnostics={"eligible": True, "boundary_decision": boundary_payload},
    )
    features = {"segments": [{
        **item,
        "sentence_start": True,
        "sentence_end": True,
        "speech_density": 0.7,
        "pause_before_seconds": 0.1,
        "pause_after_seconds": 0.1,
        "filler_word_ratio": 0.0,
        "repetition_score": 0.0,
    } for item in source_segments]}
    source_context = build_source_context(
        {"id": source_id, "path": "source.mp4"},
        {},
        candidate,
        transcript,
        features,
        {},
        {"boundaries": []},
        config.transformation,
    )
    transformation = run_content_transformation(
        source_context,
        config.transformation,
        None,
        force_local=True,
    )
    context = ProductionPlanEnvelopeContext(
        project_id="project-native-regression",
        run_id="run-native-regression",
        analysis_id="analysis-phase6",
        analysis_fingerprint="b" * 64,
        source_sha256=source_sha256,
        transcript_sha256=_hash(transcript),
        preset_id=config.product_flow.subtitle_preset,
        preset_version=config.product_flow.preset_version,
        platform=config.product_flow.platform,
        target_width=config.production_render.output_width,
        target_height=config.production_render.output_height,
        target_fps=config.production_render.output_fps,
        created_at="2026-08-10T00:00:00Z",
    )
    return build_production_plan(
        transformation,
        config.production,
        envelope_context=context,
    )


def _native_plan_for_media(config: AppConfig, source: Source, transcript: dict) -> ProductionPlan:
    return _current_native_plan(
        config,
        transcript=transcript,
        source_sha256=stable_file_hash(source.path),
    )


def test_phase6_artifacts_build_rich_native_handoff_without_analysis_calls() -> None:
    plan = _current_native_plan()
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


def test_gameplay_facecam_observation_authorizes_split_layout_family() -> None:
    plan = _current_native_plan()
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 2.0),
        output=OutputInterval.from_seconds(0.0, 1.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    timeline["visual_event_map"] = []
    candidate["vision_pass2_evidence"] = {
        "status": "completed",
        "result": {"observations": [{
            "keyframe_id": "gameplay-facecam-1",
            "timestamp": 1.35,
            "scene_type": "GAMEPLAY",
            "primary_subject": "face",
            "normalized_center_x": 0.16,
            "normalized_center_y": 0.68,
            "visible_face_count": 1,
            "confidence": 0.94,
            "origin": "provider",
        }]},
    }

    handoff = build_native_evidence_handoff(
        plan,
        mapping,
        AppConfig(),
        candidate=candidate,
        multimodal_timeline=timeline,
        story_units=stories,
    )

    assert handoff.intent.composition_targets
    assert handoff.intent.composition_targets[0].allowed_layouts[0] == LayoutFamily.SPLIT


def test_center_only_multiface_observation_protects_wide_group_extent() -> None:
    plan = _current_native_plan()
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 2.0),
        output=OutputInterval.from_seconds(0.0, 1.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    timeline["visual_event_map"] = []
    candidate["vision_pass2_evidence"] = {
        "status": "completed",
        "result": {"observations": [{
            "keyframe_id": "podcast-group-1",
            "timestamp": 1.35,
            "scene_type": "PODCAST",
            "primary_subject": "group",
            "normalized_center_x": 0.5,
            "normalized_center_y": 0.55,
            "visible_face_count": 2,
            "confidence": 0.94,
            "origin": "provider",
        }]},
    }

    handoff = build_native_evidence_handoff(
        plan,
        mapping,
        AppConfig(),
        candidate=candidate,
        multimodal_timeline=timeline,
        story_units=stories,
    )

    assert handoff.target_observations[0].bounds.width == 0.82
    assert handoff.target_observations[0].bounds.height == 0.62


def test_discontiguous_story_edges_follow_authoritative_boundary_ranges() -> None:
    boundary = BoundaryDecision.model_validate({
        "schema_version": "5C.1",
        "decision_id": "boundary-discontiguous-native",
        "candidate_id": "candidate-audio",
        "rough_range": {"start_seconds": 1.0, "end_seconds": 10.0},
        "refined_range": {"start_seconds": 1.0, "end_seconds": 10.0},
        "allowed_source_range": {"start_seconds": 1.0, "end_seconds": 10.0},
        "start_reason": "Complete hook boundary.",
        "end_reason": "Complete payoff boundary.",
        "word_integrity": True,
        "sentence_integrity": True,
        "semantic_completion": True,
        "payoff_preserved": True,
        "continuation_risk": 0.0,
        "continuation_risk_threshold": 0.65,
        "pre_roll_seconds": 0.0,
        "post_roll_seconds": 0.0,
        "confidence": 0.95,
        "start_evidence": {},
        "end_evidence": {},
        "pause_evidence": {},
        "required_evidence": [{
            "requirement_type": "hook",
            "required": True,
            "source_range": {"start_seconds": 1.2, "end_seconds": 1.8},
            "transcript_segment_id": 0,
            "reason": "The opening question must remain present.",
            "evidence": {"text": "Source dialogue"},
        }, {
            "requirement_type": "completion",
            "required": True,
            "source_range": {"start_seconds": 9.1, "end_seconds": 9.7},
            "transcript_segment_id": 1,
            "reason": "The ending must remain complete.",
            "evidence": {"text": "remains audible"},
        }, {
            "requirement_type": "payoff",
            "required": False,
            "source_range": {"start_seconds": 9.1, "end_seconds": 9.7},
            "transcript_segment_id": 1,
            "reason": "The detected payoff is presented at the ending.",
            "evidence": {"text": "remains audible"},
        }],
        "safe_start_points": [1.0],
        "safe_end_points": [10.0],
        "fallback_used": False,
        "fallback_reason": None,
        "multimodal_context": {
            "audio_or_visual_payoff_times": [1.25],
            "confidence": 0.0,
        },
    })
    transcript = {
        "source_id": "source-audio",
        "segments": [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "Source dialogue opens clearly."},
            {"id": 1, "start": 9.0, "end": 10.0, "text": "The ending remains audible."},
        ],
        "words": [],
    }
    plan = _current_native_plan(transcript=transcript, boundary=boundary)
    mapping = SourceOutputTimeMap(segments=(
        EditMapSegment(
            map_id="opening-map",
            source=SourceInterval.from_seconds(1.0, 2.0),
            output=OutputInterval(start_frame=0, end_frame=30),
        ),
        EditMapSegment(
            map_id="ending-map",
            source=SourceInterval.from_seconds(9.0, 10.0),
            output=OutputInterval(start_frame=30, end_frame=60),
        ),
    ))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    stories["story_units"][0]["end"] = 10.0

    handoff = build_native_evidence_handoff(
        plan, mapping, AppConfig(), candidate=candidate,
        multimodal_timeline=timeline, story_units=stories,
    )

    hook = next(item for item in handoff.intent.beats if item.role.value == "hook")
    payoff = next(item for item in handoff.intent.beats if item.role.value == "payoff")
    hook_emphasis = next(
        item for item in handoff.intent.semantic_emphasis
        if item.semantic_class.value == "claim"
    )
    payoff_emphasis = next(
        item for item in handoff.intent.semantic_emphasis
        if item.semantic_class.value == "payoff"
    )

    assert hook.source == SourceInterval.from_seconds(1.2, 1.8)
    assert hook.output == OutputInterval(start_frame=6, end_frame=24)
    assert payoff.source == SourceInterval.from_seconds(9.1, 9.7)
    assert payoff.output == OutputInterval(start_frame=33, end_frame=51)
    assert hook_emphasis.output == hook.output
    assert payoff_emphasis.output == payoff.output
    assert payoff.output.start_frame >= mapping.segments[1].output.start_frame
    assert not any(
        item.decision_id.startswith("motion-boundary-payoff")
        for item in handoff.intent.motion_events
    )


def test_optional_broll_off_does_not_demote_other_native_creative_layers() -> None:
    plan = _current_native_plan()
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

    assert handoff.execution_status == "native_rich"
    assert not handoff.intent.source_broll
    assert "SOURCE_BROLL_USAGE_NOT_AUTHORIZED" in handoff.reason_codes
    assert "SAFE_A_ROLL_FALLBACK" not in handoff.reason_codes
    assert all(scene.rights_status == "uncertain" for scene in handoff.source_scenes)


def test_native_rich_status_is_demoted_when_compilation_drops_required_layers() -> None:
    plan = _current_native_plan()
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 2.0),
        output=OutputInterval.from_seconds(0.0, 1.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    config = AppConfig()
    handoff = build_native_evidence_handoff(
        plan, mapping, config, candidate=candidate,
        multimodal_timeline=timeline, story_units=stories,
    )
    assert handoff.execution_status == "native_rich"

    compiled = compile_native_creative_plan(
        handoff.intent,
        {"segments": [], "words": []},
        config,
        source_width=1920,
        source_height=1080,
        target_observations=handoff.target_observations,
        source_scenes=handoff.source_scenes,
    )
    status, codes, diagnostics = _reconcile_native_execution_status(
        handoff.execution_status,
        handoff.reason_codes,
        handoff.diagnostics,
        compiled,
    )

    assert status == "native_fallback"
    assert "NATIVE_REQUIRED_LAYER_FALLBACK" in codes
    assert "CAPTION_LAYER_NOT_EXECUTED" in codes
    assert "MOTION_LAYER_NOT_EXECUTED" in codes
    assert diagnostics


def test_compiled_hook_motion_can_present_hook_alongside_semantic_emphasis_without_broll() -> None:
    words = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau omega"
    ).split()
    transcript = {
        "source_id": "source-audio",
        "segments": [{
            "id": 0, "start": 1.0, "end": 11.0, "text": " ".join(words),
        }],
        "words": [
            {
                "start": 1.0 + index * 0.5,
                "end": 1.0 + (index + 1) * 0.5,
                "text": word,
            }
            for index, word in enumerate(words)
        ],
    }
    boundary = BoundaryDecision.model_validate({
        "schema_version": "5C.1",
        "decision_id": "boundary-candidate-audio-long-native",
        "candidate_id": "candidate-audio",
        "rough_range": {"start_seconds": 1.0, "end_seconds": 11.0},
        "refined_range": {"start_seconds": 1.0, "end_seconds": 11.0},
        "allowed_source_range": {"start_seconds": 1.0, "end_seconds": 11.0},
        "start_reason": "Complete source hook.",
        "end_reason": "Complete source payoff.",
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
        "required_evidence": [
            {
                "requirement_type": "hook",
                "required": True,
                "source_range": {"start_seconds": 1.0, "end_seconds": 2.5},
                "transcript_segment_id": 0,
                "reason": "The opening hook remains present.",
                "evidence": {"text": "Alpha beta gamma"},
            },
            {
                "requirement_type": "completion",
                "required": True,
                "source_range": {"start_seconds": 9.5, "end_seconds": 11.0},
                "transcript_segment_id": 0,
                "reason": "The ending remains complete.",
                "evidence": {"text": "sigma tau omega"},
            },
            {
                "requirement_type": "payoff",
                "required": True,
                "source_range": {"start_seconds": 10.0, "end_seconds": 11.0},
                "transcript_segment_id": 0,
                "reason": "The payoff remains present.",
                "evidence": {"text": "tau omega"},
            },
        ],
        "safe_start_points": [1.0],
        "safe_end_points": [11.0],
        "fallback_used": False,
        "fallback_reason": None,
    })
    plan = _current_native_plan(transcript=transcript, boundary=boundary)
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id="candidate-map",
        source=SourceInterval.from_seconds(1.0, 11.0),
        output=OutputInterval.from_seconds(0.0, 10.0),
    ),))
    candidate, timeline, stories = _phase6_artifacts(plan.metadata.candidate_id)
    stories["story_units"][0]["end"] = 11.0
    stories["story_units"][0]["duration"] = 10.0
    config = AppConfig()
    handoff = build_native_evidence_handoff(
        plan, mapping, config, candidate=candidate,
        multimodal_timeline=timeline, story_units=stories,
    )
    compiled = compile_native_creative_plan(
        handoff.intent,
        transcript,
        config,
        source_width=1920,
        source_height=1080,
        target_observations=handoff.target_observations,
        source_scenes=handoff.source_scenes,
    )

    assert handoff.execution_status == "native_rich"
    assert not compiled.source_broll_plan.segments
    assert any(cue.emphasis is not None for cue in compiled.caption_plan.cues)
    assert compiled.caption_plan.font_manifest is not None
    expected_font_assets = {
        compiled.caption_plan.font_manifest.font_id: compiled.caption_plan.font_manifest.file_sha256,
        **{
            face.font_id: face.file_sha256
            for face in compiled.caption_plan.font_manifest.companion_faces
        },
    }
    assert {
        item.asset_id: item.checksum for item in compiled.assets if item.asset_type == "font"
    } == expected_font_assets
    assert {
        event.purpose.value
        for event in compiled.motion_plan.events
        if event.primitive_id != "static"
    }.issuperset({"hook", "payoff"})
    assert any(
        segment.target.value != "stable_source"
        for segment in compiled.composition_plan.segments
    )

    status, codes, diagnostics = _reconcile_native_execution_status(
        handoff.execution_status,
        handoff.reason_codes,
        handoff.diagnostics,
        compiled,
    )

    assert status == "native_rich"
    assert "NATIVE_REQUIRED_LAYER_FALLBACK" not in codes
    assert "SOURCE_BROLL_USAGE_NOT_AUTHORIZED" in codes
    assert not diagnostics


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
    # This one-second fixture intentionally collapses hook, emphasis, and
    # payoff onto one cue. The compiler keeps the higher-priority payoff
    # motion, so the post-compile status must describe the missing hook layer
    # instead of inheriting the richer evidence-only label.
    assert item_report["execution_status"] == "native_fallback"
    assert "NATIVE_REQUIRED_LAYER_FALLBACK" in item_report["execution_reason_codes"]
    assert "HOOK_PRESENTATION_NOT_EXECUTED" in item_report["execution_reason_codes"]
    assert (candidate_output / "production-render" / "creative-intent.json").is_file()
    assert (candidate_output / "production-render" / "creative-handoff.json").is_file()
    assert (candidate_output / "production-render" / "creative-execution.json").is_file()
    assert not (candidate_output / "production-render" / "reframe-plan.json").exists()
    assert not (candidate_output / "production-render" / "subtitle-project.json").exists()
