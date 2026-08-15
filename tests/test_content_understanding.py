from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppConfig
from app.content_profile_taxonomy import content_profile_preset_ids, content_profile_preset_mapping
from app.content_understanding import (
    VIDEO_CONTENT_PROFILE_SCHEMA_VERSION,
    VideoContentProfile,
    build_global_content_map,
    build_video_content_profile,
    story_units_artifact,
    validate_global_content_map,
    validate_video_content_profile,
)
from app.pipeline import Pipeline, StageTracker
from app.transcript_features import analyse_transcript
from app.utils import write_json


def _profile(
    text: str, *, speakers: list[str] | None = None, filename: str = "source.mp4",
    config: AppConfig | None = None, audio_features: dict | None = None,
    visual_analysis: dict | None = None, scenes: dict | None = None,
) -> dict:
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
    active_config = config or AppConfig()
    features = analyse_transcript(transcript, active_config.transcript_features)
    return build_video_content_profile(
        {"id": "source-1", "display_name": filename},
        {"duration": transcript["duration"]},
        transcript,
        features,
        audio_features or {"energy_frames": []},
        scenes or {"boundaries": []},
        visual_analysis or {"status": "skipped", "evidence_status": "skipped", "subject_keyframes": [], "sample_count": 0},
        active_config,
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
    filename_only = _profile("Ладно.", filename="pubg-gameplay.mp4")

    assert unknown["detected_content_type"] == "unknown"
    assert unknown["strategy_id"] == "generic_monologue"
    assert educational["detected_content_type"] == "educational"
    assert educational["evidence"]["filename_signal_used"] is False
    assert filename_only["detected_profile"]["format"]["value"] == "gameplay"
    assert filename_only["effective_profile"]["format"] == "unknown"
    assert filename_only["effective_profile"]["resolution"]["format"] == "safe_fallback"


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("Это влог про путешествие: сначала мы приехали, потом пошли в горы.", ("scene_driven", "narrative", "lifestyle")),
        ("Покажу фитнес-тренировку workout шаг за шагом.", ("scene_driven", "demonstration", "health")),
        ("Это обзор и product review: комментирую плюсы и минусы.", ("mixed", "commentary", "general")),
    ),
)
def test_auto_profile_aliases_use_only_existing_transcript_evidence(
    text: str, expected: tuple[str, str, str],
) -> None:
    data = _profile(text)

    assert (
        data["detected_profile"]["format"]["value"],
        data["detected_profile"]["editorial_mode"]["value"],
        data["detected_profile"]["domain"]["value"],
    ) == expected
    for axis in ("format", "editorial_mode", "domain"):
        assert all(not item.startswith("filename:") for item in data["detected_profile"][axis]["evidence"])


def test_profile_v2_auto_detection_keeps_detected_and_effective_axes() -> None:
    data = _profile(
        "В этой катке PUBG разберём матч и покажем решающий момент gameplay.",
        filename="pubg-stream.mp4",
        scenes={"boundaries": [{"timestamp": float(index)} for index in range(10)]},
    )

    assert data["schema_version"] == "5A.2"
    assert data["detected_profile"]["format"]["value"] == "gameplay"
    assert data["detected_profile"]["domain"]["value"] == "gaming"
    assert data["effective_profile"]["format"] == "gameplay"
    assert data["effective_profile"]["resolution"]["format"] == "detected"
    assert data["detected_content_type"] == "gameplay"
    assert data["evidence"]["filename_signal_used"] is True


def test_manual_override_changes_effective_profile_without_replacing_detection() -> None:
    config = AppConfig()
    config.content_understanding.manual_override = {
        "format": "dialogue", "editorial_mode": "interview", "domain": "business",
        "traits": ["question_answer"],
    }
    data = _profile("Как работает этот урок? Объясню каждый шаг и приведу пример.", config=config)

    assert data["detected_profile"]["editorial_mode"]["value"] == "explanatory"
    assert data["effective_profile"] == {
        "format": "dialogue",
        "editorial_mode": "interview",
        "domain": "business",
        "traits": ["question_answer"],
        "resolution": {
            "format": "manual_override", "editorial_mode": "manual_override",
            "domain": "manual_override", "traits": "manual_override",
        },
    }
    assert data["manual_override"]["provenance"] == "user"
    assert data["manual_override"]["revision_id"]
    assert data["detected_content_type"] == "interview"


@pytest.mark.parametrize("preset_id", content_profile_preset_ids())
def test_each_manual_content_preset_creates_effective_profile_and_preserves_detection(preset_id: str) -> None:
    text = "Объясняю принцип на примере, потому что важно понять вывод."
    detected = _profile(text)["detected_profile"]
    expected = content_profile_preset_mapping(preset_id)
    config = AppConfig()
    config.content_understanding.manual_override = expected

    data = _profile(text, config=config)

    assert data["detected_profile"] == detected
    assert {axis: data["effective_profile"][axis] for axis in ("format", "editorial_mode", "domain", "traits")} == expected
    assert data["effective_profile"]["resolution"] == {
        "format": "manual_override",
        "editorial_mode": "manual_override",
        "domain": "manual_override",
        "traits": "manual_override",
    }


def test_profile_reads_current_audio_and_visual_evidence_contracts() -> None:
    data = _profile(
        "Спокойное объяснение с примером и понятным итогом для зрителя.",
        audio_features={"energy_frames": [{"normalized_loudness": 0.9}]},
        visual_analysis={
            "status": "completed", "evidence_status": "observed", "sample_count": 20,
            "subject_keyframes": [{"timestamp": 1.0}],
        },
    )

    assert data["emotional_curve_summary"] == "выраженные эмоциональные пики"
    assert data["visual_density"] > 0
    assert data["analysis_confidence"] >= 0.75


def test_profile_validator_migrates_legacy_shape_and_checks_source_identity() -> None:
    current = _profile("Как работает урок? Объясню метод и приведу пример.")
    legacy = {key: value for key, value in current.items() if key not in {"detected_profile", "effective_profile", "manual_override"}}
    legacy["schema_version"] = "5A.1"

    migrated = VideoContentProfile.from_dict(legacy)
    assert migrated.schema_version == "5A.2"
    assert migrated.effective_profile["resolution"]["format"] == "legacy_migration"
    validate_video_content_profile(current, expected_source_id="source-1")
    with pytest.raises(ValueError, match="source_id"):
        validate_video_content_profile(current, expected_source_id="other-source")


def test_disabled_detection_ignores_filename_but_keeps_manual_override() -> None:
    config = AppConfig()
    config.content_understanding.enabled = False
    config.content_understanding.manual_override = {"domain": "business"}

    data = _profile("PUBG gameplay match and decisive win.", filename="pubg-gameplay.mp4", config=config)

    assert data["evidence"]["detection_enabled"] is False
    assert data["detected_profile"]["format"]["value"] == "unknown"
    assert data["evidence"]["filename_signal_used"] is False
    assert data["effective_profile"]["domain"] == "business"
    assert data["effective_profile"]["resolution"]["domain"] == "manual_override"


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


def _content_map(segments: list[dict]) -> tuple[dict, dict]:
    transcript = {
        "source_id": "source-1", "language": "ru", "duration": float(segments[-1]["end"]), "segments": segments,
    }
    config = AppConfig()
    features = analyse_transcript(transcript, config.transcript_features)
    profile = build_video_content_profile(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": transcript["duration"]},
        transcript, features, {"windows": []}, {"boundaries": []}, {"samples": []}, config,
    )
    return build_global_content_map(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": transcript["duration"]},
        transcript, features, {"windows": []}, {"boundaries": []}, {"samples": []}, profile, config,
    ), transcript


def test_content_map_covers_transcript_in_order_and_keeps_question_with_answer() -> None:
    content_map, transcript = _content_map([
        {"id": 10, "start": 0.0, "end": 8.0, "text": "Почему люди сдаются слишком рано?"},
        {"id": 11, "start": 8.2, "end": 20.0, "text": "Потому что они не видят результат до последнего шага."},
        {"id": 12, "start": 22.0, "end": 33.0, "text": "Теперь другой урок: дисциплина важнее настроения."},
        {"id": 13, "start": 33.1, "end": 46.0, "text": "Поэтому побеждает тот, кто продолжает действовать каждый день."},
    ])

    chapters = content_map["chapters"]
    stories = content_map["story_units"]
    assert len(chapters) == 2
    assert [item["chapter_id"] for item in chapters] == ["chapter-001", "chapter-002"]
    assert [identifier for chapter in chapters for identifier in chapter["transcript_segment_ids"]] == [10, 11, 12, 13]
    assert any(unit["transcript_segment_ids"] == [10, 11] for unit in stories)
    assert all(unit["chapter_id"] in {chapter["chapter_id"] for chapter in chapters} for unit in stories)
    assert all(unit["start"] >= next(chapter for chapter in chapters if chapter["chapter_id"] == unit["chapter_id"])["start"] for unit in stories)
    validate_global_content_map(content_map, transcript)


def test_chapter_pause_does_not_make_subminimum_chapters() -> None:
    content_map, _transcript = _content_map([
        {"id": 1, "start": 0.0, "end": 4.0, "text": "First complete sentence."},
        {"id": 2, "start": 6.0, "end": 10.0, "text": "Second complete sentence."},
        {"id": 3, "start": 12.0, "end": 16.0, "text": "Third complete sentence."},
    ])

    assert len(content_map["chapters"]) == 1
    assert content_map["chapters"][0]["transcript_segment_ids"] == [1, 2, 3]
    assert content_map["story_units"][0]["duration"] >= AppConfig().content_understanding.min_story_unit_seconds


def test_story_units_have_grounded_signatures_and_detect_repeated_ideas() -> None:
    content_map, transcript = _content_map([
        {"start": 0.0, "end": 16.0, "text": "Дисциплина важнее настроения, потому что действие создаёт результат."},
        {"start": 18.0, "end": 34.0, "text": "Дисциплина важнее настроения, потому что действие создаёт результат."},
    ])
    signatures = [item["content_signature"] for item in content_map["story_units"]]

    assert len(signatures) == 2
    assert signatures[0]["transcript_fingerprint"] == signatures[1]["transcript_fingerprint"]
    assert signatures[0]["semantic_embedding_ref"] is None
    artifact = story_units_artifact(content_map, transcript)
    assert artifact["schema_version"] == "5A.1"
    assert len(artifact["story_units"]) == 2


def test_content_map_rejects_ungrounded_or_out_of_chapter_story_unit() -> None:
    content_map, transcript = _content_map([
        {"start": 0.0, "end": 16.0, "text": "Завершённая понятная мысль с естественной точкой."},
    ])
    content_map["story_units"][0]["transcript_segment_ids"] = [999]

    with pytest.raises(ValueError, match="StoryUnit"):
        validate_global_content_map(content_map, transcript)
