from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from app.config import AppConfig
from app.content_understanding import build_global_content_map, build_video_content_profile, story_units_artifact
from app.multimodal_evidence import (
    MULTIMODAL_ANALYSIS_VERSION,
    build_multimodal_timeline,
    evidence_for_range,
    multimodal_analysis_run_id,
    validate_multimodal_timeline,
)
from app.pipeline import Pipeline, StageTracker
from app.transcript_features import analyse_transcript
from app.utils import read_json, write_json


def _inputs(*, visual_available: bool = True, scenes_available: bool = True) -> tuple[dict, dict, dict, dict]:
    transcript = {
        "source_id": "source-1",
        "language": "en",
        "duration": 30.0,
        "segments": [
            {
                "id": 10, "start": 1.0, "end": 10.0,
                "text": "Why does this matter? [laughter]", "speaker_id": "host", "confidence": 0.92,
                "words": [
                    {"start": 1.0, "end": 1.4, "text": "Why", "probability": 0.96},
                    {"start": 1.5, "end": 2.0, "text": "matter", "probability": 0.9},
                ],
            },
            {
                "id": 11, "start": 12.0, "end": 26.0,
                "text": "Because measured evidence leads to a clear conclusion.",
                "speaker_id": "guest", "confidence": 0.88,
            },
        ],
        "words": [],
    }
    audio = {
        "sample_rate": 16000,
        "duration_seconds": 30.0,
        "window_seconds": 0.5,
        "energy_frames": [
            {"time": 1.0, "audio_energy": 0.1, "normalized_loudness": 0.2},
            {"time": 8.0, "audio_energy": 0.7, "normalized_loudness": 0.91},
            {"time": 14.0, "audio_energy": 0.3, "normalized_loudness": 0.45},
            {"time": 22.0, "audio_energy": 0.8, "normalized_loudness": 1.0},
        ],
        "silence_intervals": [{"start": 10.0, "end": 12.0}],
    }
    scenes = {
        "enabled": scenes_available,
        "threshold": 0.3,
        "boundaries": [{"timestamp": 11.0, "scene_change_score": 0.84}] if scenes_available else [],
        "scene_boundary_count": 1 if scenes_available else 0,
    }
    visual = {
        "schema_version": "5D.0",
        "enabled": visual_available,
        "status": "completed" if visual_available else "skipped",
        "evidence_status": "valid" if visual_available else "evidence_unavailable",
        "reason": None if visual_available else "disabled",
        "subject_keyframes": [
            {
                "time_seconds": 3.0, "normalized_x": 0.25, "normalized_y": 0.4,
                "normalized_width": 0.3, "normalized_height": 0.5, "confidence": 0.94,
                "tracking_target": "primary_face", "visible_face_count": 1,
                "active_speaker_confidence": 0.86, "scene_id": "setup-a",
                "scene_type": "TALKING_HEAD", "framing_observation": "chest_up",
                "eye_line_y": 0.32, "gesture_active": False, "gesture_area_visible": True,
            },
            {
                "time_seconds": 5.0, "normalized_x": 0.52, "normalized_y": 0.42,
                "normalized_width": 0.29, "normalized_height": 0.49, "confidence": 0.9,
                "tracking_target": "primary_face", "visible_face_count": 1,
                "active_speaker_confidence": 0.8, "scene_id": "setup-a",
                "scene_type": "TALKING_HEAD", "framing_observation": "chest_up",
                "eye_line_y": 0.33, "gesture_active": True, "gesture_area_visible": True,
            },
            {
                "time_seconds": 16.0, "normalized_x": 0.62, "normalized_y": 0.42,
                "normalized_width": 0.28, "normalized_height": 0.48, "confidence": 0.87,
                "tracking_target": "important_object", "visible_face_count": 1,
                "active_speaker_confidence": 0.2, "scene_id": "setup-b",
                "scene_type": "PRODUCT_DEMO", "framing_observation": "object",
                "eye_line_y": 0.34, "gesture_active": True, "gesture_area_visible": True,
            },
        ] if visual_available else [],
        "sample_count": 3 if visual_available else 0,
    }
    return transcript, audio, scenes, visual


def _timeline(*, visual_available: bool = True, scenes_available: bool = True) -> dict:
    transcript, audio, scenes, visual = _inputs(
        visual_available=visual_available, scenes_available=scenes_available,
    )
    return build_multimodal_timeline(
        source_id="source-1", source_duration_seconds=30.0, transcript=transcript,
        audio_features=audio, scenes=scenes, visual_analysis=visual,
    )


def test_timeline_synchronizes_grounded_event_maps_and_sparse_keyframes() -> None:
    timeline = _timeline()

    validate_multimodal_timeline(timeline, expected_source_id="source-1")
    assert timeline["analysis_version"] == MULTIMODAL_ANALYSIS_VERSION
    assert timeline["time_base"]["origin"] == "source_media_start"
    assert {item["event_type"] for item in timeline["audio_event_map"]} >= {
        "speech", "silence", "pause", "energy", "emphasis", "speaker_change", "reaction_label",
    }
    assert {item["event_type"] for item in timeline["visual_event_map"]} == {
        "scene_change", "subject_observation",
    }
    subject = next(item for item in timeline["visual_event_map"] if item["event_type"] == "subject_observation")
    assert subject["observation"]["faces"]["visible_count"] == 1
    assert subject["observation"]["screen_text_product"]["text_evidence"] == "missing"
    assert subject["observation"]["screen_text_product"]["product_evidence"] == "missing"
    assert all(0 <= item["confidence"] <= 1 and item["provenance"] for name in (
        "transcript_events", "audio_event_map", "visual_event_map",
    ) for item in timeline[name])

    reasons = {reason for item in timeline["keyframes"] for reason in item["selection_reasons"]}
    assert {"scene_boundary", "measured_motion", "framing_relevance"} <= reasons
    assert len(timeline["keyframes"]) <= timeline["diagnostics"]["keyframes"]["limit"]
    assert len(timeline["keyframes"]) < 30 * 30  # sparse plan, never one item per source frame
    assert timeline["diagnostics"]["external_vision_api_calls"] == 0


def test_missing_visual_evidence_is_explicit_and_does_not_call_vision(monkeypatch) -> None:
    class ForbiddenOpenAI:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("Multimodal evidence construction must not call a Vision API.")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=ForbiddenOpenAI))
    timeline = _timeline(visual_available=False, scenes_available=False)

    assert timeline["visual_event_map"] == []
    assert timeline["diagnostics"]["evidence"]["visual"]["status"] == "missing"
    assert timeline["diagnostics"]["evidence"]["scenes"]["status"] == "missing"
    assert {item["modality"] for item in timeline["diagnostics"]["missing_evidence"]} >= {"visual", "scenes"}
    assert timeline["keyframes"]
    assert all(item["analysis_status"] == "not_analyzed" for item in timeline["keyframes"])
    assert timeline["diagnostics"]["external_vision_api_calls"] == 0


def test_story_units_receive_only_timeline_evidence_overlapping_their_source_range() -> None:
    transcript, audio, scenes, visual = _inputs()
    config = AppConfig()
    features = analyse_transcript(transcript, config.transcript_features)
    source = {"id": "source-1", "display_name": "source.mp4"}
    metadata = {"duration": 30.0}
    profile = build_video_content_profile(
        source, metadata, transcript, features, audio, scenes, visual, config,
    )
    timeline = _timeline()

    content_map = build_global_content_map(
        source, metadata, transcript, features, audio, scenes, visual, profile, config, timeline,
    )

    assert content_map["story_units"]
    for story in content_map["story_units"]:
        evidence = story["multimodal_evidence"]
        assert evidence["source_id"] == "source-1"
        assert evidence["analysis_run_id"] == timeline["analysis_run_id"]
        assert evidence["interval"] == {
            "start_seconds": story["start"], "end_seconds": story["end"],
        }
        for modality_refs in evidence["event_refs"].values():
            assert all(
                ref["start_seconds"] <= story["end"] and ref["end_seconds"] >= story["start"]
                for ref in modality_refs
            )
    artifact = story_units_artifact(content_map, transcript)
    assert artifact["story_units"][0]["multimodal_evidence"]["analysis_run_id"] == timeline["analysis_run_id"]


def test_source_cache_reuses_valid_timeline_and_recovers_corrupt_artifact(tmp_path: Path) -> None:
    transcript, audio, scenes, visual = _inputs()
    analysis_run_id = multimodal_analysis_run_id("source-1", transcript, audio, scenes, visual)
    artifact = tmp_path / "multimodal_timeline.json"
    cache = StageTracker(tmp_path / "cache-state.json")
    calls = {"count": 0}

    def build() -> dict:
        calls["count"] += 1
        value = build_multimodal_timeline(
            source_id="source-1", source_duration_seconds=30.0, transcript=transcript,
            audio_features=audio, scenes=scenes, visual_analysis=visual,
            analysis_run_id=analysis_run_id,
        )
        write_json(artifact, value)
        return value

    def validate(value: dict) -> dict:
        return validate_multimodal_timeline(
            value, expected_source_id="source-1", expected_analysis_run_id=analysis_run_id,
        )

    pipeline = Pipeline(tmp_path, AppConfig())
    fingerprint = {"source": "source-1", "analysis_run_id": analysis_run_id}
    first = pipeline._cached(
        StageTracker(tmp_path / "runs" / "one.json"), "multimodal_timeline",
        artifact, fingerprint, build, cache_tracker=cache, validator=validate,
    )
    second_run = StageTracker(tmp_path / "runs" / "two.json")
    second = pipeline._cached(
        second_run, "multimodal_timeline", artifact, fingerprint, build,
        cache_tracker=cache, validator=validate,
    )
    assert first == second
    assert calls["count"] == 1
    assert second_run.data["stages"]["multimodal_timeline"]["cache_hit"] is True

    artifact.write_text("{corrupt", encoding="utf-8")
    third_run = StageTracker(tmp_path / "runs" / "three.json")
    recovered = pipeline._cached(
        third_run, "multimodal_timeline", artifact, fingerprint, build,
        cache_tracker=cache, validator=validate,
    )
    assert calls["count"] == 2
    assert recovered["analysis_run_id"] == analysis_run_id
    assert read_json(artifact, {})["source_id"] == "source-1"
    assert third_run.data["stages"]["multimodal_timeline"]["cache_hit"] is False


def test_range_query_keeps_machine_readable_availability() -> None:
    timeline = _timeline(visual_available=False, scenes_available=False)
    evidence = evidence_for_range(timeline, 1.0, 10.0)

    assert evidence["event_refs"]["transcript"]
    assert evidence["event_refs"]["visual"] == []
    assert evidence["evidence_status"]["visual"]["status"] == "missing"
    assert evidence["missing_evidence"]
