from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
import pytest

from app.visual_analysis import (
    _SUBJECT_SCHEMA,
    VisualAnalysisSchemaError,
    _keyframe,
    _sample_times,
    _validate_subject_response,
    analyse_video_subjects,
)


def test_visual_analysis_stays_disabled_without_user_choice(tmp_path: Path) -> None:
    result = analyse_video_subjects(tmp_path / "missing.mp4", 120, AppConfig())
    assert result["enabled"] is False
    assert result["status"] == "skipped"
    assert result["evidence_status"] == "evidence_unavailable"
    assert result["fallback_provenance"]["stage"] == "configuration"
    assert result["subject_keyframes"] == []


def test_visual_analysis_uses_safe_fallback_when_provider_is_not_available(tmp_path: Path) -> None:
    config = AppConfig(optional_visual_features=True)
    config.ai.provider = "mock"
    result = analyse_video_subjects(tmp_path / "missing.mp4", 120, config)
    assert result["status"] == "fallback"
    assert result["evidence_status"] == "fallback"
    assert result["fallback_provenance"]["stage"] == "visual_provider"
    assert result["subject_keyframes"] == []


def test_visual_samples_and_keyframes_are_bounded() -> None:
    samples = _sample_times(600)
    assert 2 <= len(samples) <= 8
    assert samples == sorted(samples)
    assert _keyframe({"time_seconds": 1, "normalized_x": 0.5, "normalized_y": 0.4, "confidence": 0.9})
    assert _keyframe({"time_seconds": 1, "normalized_x": 2, "normalized_y": 0.4, "confidence": 0.9}) is None


def test_strict_visual_schema_requires_every_property_and_rejects_undeclared_fields() -> None:
    item_schema = _SUBJECT_SCHEMA["properties"]["subjects"]["items"]
    required = set(item_schema["required"])
    assert item_schema["additionalProperties"] is False
    assert required == set(item_schema["properties"])

    subject = {
        "time_seconds": 1.0,
        "normalized_x": 0.5,
        "normalized_y": 0.42,
        "normalized_width": 0.28,
        "normalized_height": 0.46,
        "confidence": 0.92,
        "tracking_target": "primary_face",
        "visible_face_count": 1,
        "active_speaker_confidence": 0.8,
        "scene_id": "setup-01",
        "scene_type": "TALKING_HEAD",
        "framing_observation": "chest_up",
        "eye_line_y": 0.35,
        "gesture_active": True,
        "gesture_area_visible": True,
    }
    keyframes = _validate_subject_response({"subjects": [subject]})
    assert keyframes[0]["scene_type"] == "TALKING_HEAD"
    assert keyframes[0]["gesture_area_visible"] is True

    with pytest.raises(VisualAnalysisSchemaError):
        _validate_subject_response({"subjects": [{key: value for key, value in subject.items() if key != "eye_line_y"}]})
    with pytest.raises(VisualAnalysisSchemaError):
        _validate_subject_response({"subjects": [{**subject, "unbounded": "no"}]})
    with pytest.raises(VisualAnalysisSchemaError):
        _validate_subject_response({"subjects": [{**subject, "confidence": "0.92"}]})
