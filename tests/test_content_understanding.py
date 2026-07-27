from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.content_understanding import (
    VIDEO_CONTENT_PROFILE_SCHEMA_VERSION,
    VideoContentProfile,
    build_video_content_profile,
)
from app.pipeline import Pipeline, StageTracker
from app.transcript_features import analyse_transcript
from app.utils import write_json


def _profile(text: str, *, speakers: list[str] | None = None, filename: str = "source.mp4") -> dict:
    speakers = speakers or []
    segments = []
    for index, sentence in enumerate(text.split("|")):
        segment = {
            "start": float(index * 12),
            "end": float(index * 12 + 10),
            "text": sentence.strip(),
            "words": [],
        }
        if index < len(speakers):
            segment["speaker_id"] = speakers[index]
        segments.append(segment)
    transcript = {"source_id": "source-1", "language": "ru", "duration": max(10.0, len(segments) * 12.0), "segments": segments}
    features = analyse_transcript(transcript, AppConfig().transcript_features)
    return build_video_content_profile(
        {"id": "source-1", "display_name": filename},
        {"duration": transcript["duration"]},
        transcript,
        features,
        {"windows": []},
        {"boundaries": []},
        {"samples": []},
        AppConfig(),
    )


def test_motivational_monologue_profile_is_grounded_in_transcript() -> None:
    data = _profile("Никогда не сдавайтесь: верьте в свой шанс и боритесь за победу.")

    assert data["schema_version"] == VIDEO_CONTENT_PROFILE_SCHEMA_VERSION
    assert data["detected_content_type"] == "motivational"
    assert data["dominant_format"] == "single_speaker_monologue"
    assert data["strategy_id"] == "motivational_monologue"
    assert data["estimated_publishable_clip_range"]["min"] >= 1
    VideoContentProfile.from_dict(data)


def test_dialogue_and_educational_profiles_choose_compatible_strategies() -> None:
    dialogue = _profile(
        "Вопрос: почему это произошло? | Ответ: потому что мы пропустили важный шаг.",
        speakers=["host", "guest"],
    )
    educational = _profile("Как работает этот метод? Сейчас объясню каждый шаг и приведу пример.")

    assert dialogue["dominant_format"] == "multi_speaker_dialogue"
    assert dialogue["strategy_id"] == "generic_dialogue"
    assert educational["detected_content_type"] == "educational"
    assert educational["strategy_id"] == "generic_educational"


def test_unknown_profile_uses_safe_fallback_and_filename_is_only_weak_signal() -> None:
    unknown = _profile("Ладно.")
    educational = _profile(
        "Как работает этот метод? Объясню каждый шаг и приведу пример.",
        filename="мотивация-название.mp4",
    )

    assert unknown["detected_content_type"] == "unknown"
    assert unknown["strategy_id"] == "generic_monologue"
    assert educational["detected_content_type"] == "educational"
    assert educational["evidence"]["filename_signal_used"] is False


def test_profile_cache_is_source_scoped_and_stable(tmp_path: Path) -> None:
    artifact = tmp_path / "video_content_profile.json"
    source_cache = StageTracker(tmp_path / "cache-state.json")
    first_run = StageTracker(tmp_path / "runs" / "one" / "state.json")
    second_run = StageTracker(tmp_path / "runs" / "two" / "state.json")
    calls = {"count": 0}

    def build() -> dict:
        calls["count"] += 1
        value = _profile("Никогда не сдавайтесь и верьте в победу.")
        write_json(artifact, value)
        return value

    pipeline = Pipeline(tmp_path, AppConfig())
    first = pipeline._cached(first_run, "video_content_profile", artifact, {"transcript": "same"}, build, cache_tracker=source_cache)
    second = pipeline._cached(second_run, "video_content_profile", artifact, {"transcript": "same"}, build, cache_tracker=source_cache)

    assert calls == {"count": 1}
    assert first == second
    assert second_run.data["stages"]["video_content_profile"]["status"] == "completed"


def test_profile_cache_key_changes_when_strategy_version_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "video_content_profile.json"
    source_cache = StageTracker(tmp_path / "cache-state.json")
    run = StageTracker(tmp_path / "runs" / "state.json")
    calls = {"count": 0}

    def build() -> dict:
        calls["count"] += 1
        value = _profile("Никогда не сдавайтесь и верьте в победу.")
        write_json(artifact, value)
        return value

    pipeline = Pipeline(tmp_path, AppConfig())
    pipeline._cached(run, "video_content_profile", artifact, {"strategy_version": "one"}, build, cache_tracker=source_cache)
    pipeline._cached(run, "video_content_profile", artifact, {"strategy_version": "two"}, build, cache_tracker=source_cache)

    assert calls == {"count": 2}
