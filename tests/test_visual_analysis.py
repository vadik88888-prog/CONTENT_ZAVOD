from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.visual_analysis import _keyframe, _sample_times, analyse_video_subjects


def test_visual_analysis_stays_disabled_without_user_choice(tmp_path: Path) -> None:
    result = analyse_video_subjects(tmp_path / "missing.mp4", 120, AppConfig())
    assert result == {"enabled": False, "status": "skipped", "reason": "disabled", "subject_keyframes": []}


def test_visual_analysis_uses_safe_fallback_when_provider_is_not_available(tmp_path: Path) -> None:
    config = AppConfig(optional_visual_features=True)
    config.ai.provider = "mock"
    result = analyse_video_subjects(tmp_path / "missing.mp4", 120, config)
    assert result["status"] == "fallback"
    assert result["subject_keyframes"] == []


def test_visual_samples_and_keyframes_are_bounded() -> None:
    samples = _sample_times(600)
    assert 2 <= len(samples) <= 8
    assert samples == sorted(samples)
    assert _keyframe({"time_seconds": 1, "normalized_x": 0.5, "normalized_y": 0.4, "confidence": 0.9})
    assert _keyframe({"time_seconds": 1, "normalized_x": 2, "normalized_y": 0.4, "confidence": 0.9}) is None
