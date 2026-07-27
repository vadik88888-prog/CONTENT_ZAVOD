from __future__ import annotations

from dataclasses import replace

from app.config import AppConfig
from app.content_understanding import (
    GlobalContentMap,
    SemanticBoundaryEngine,
    build_global_content_map,
    build_video_content_profile,
)
from app.models import Candidate, ScoredCandidate
from app.selection import select_clips
from app.semantic_extraction import build_source_context
from app.transcript_features import analyse_transcript


def _unit(transcript: dict, features: dict, config: AppConfig) -> object:
    profile = build_video_content_profile(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": transcript["duration"]}, transcript,
        features, {"windows": []}, {"boundaries": []}, {"samples": []}, config,
    )
    content_map = build_global_content_map(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": transcript["duration"]}, transcript,
        features, {"windows": []}, {"boundaries": []}, {"samples": []}, profile, config,
    )
    return GlobalContentMap.from_dict(content_map, transcript).story_units[0]


def test_boundary_engine_keeps_first_and_last_words_and_natural_tail() -> None:
    config = AppConfig()
    transcript = {
        "source_id": "source-1", "language": "ru", "duration": 10.0,
        "segments": [{
            "start": 0.5, "end": 6.9, "text": "Сначала сделайте шаг, а потом подведите итог.",
            "words": [
                {"start": 0.6, "end": 1.0, "text": "Сначала"},
                {"start": 1.1, "end": 1.5, "text": "сделайте"},
                {"start": 5.9, "end": 6.5, "text": "итог"},
            ],
        }],
    }
    features = analyse_transcript(transcript, config.transcript_features)
    unit = _unit(transcript, features, config)

    resolution = SemanticBoundaryEngine(config.content_understanding).resolve(unit, transcript, features, {"boundaries": []})

    assert resolution.start <= 0.6
    assert resolution.end >= 6.5
    assert resolution.diagnostics["word_integrity"] is True
    assert resolution.diagnostics["sentence_integrity"] is True
    assert resolution.diagnostics["tail_padding_seconds"] >= config.content_understanding.min_tail_padding_seconds
    assert resolution.diagnostics["eligible"] is True


def test_boundary_engine_extends_unfinished_setup_to_sentence_completion() -> None:
    config = AppConfig()
    transcript = {
        "source_id": "source-1", "language": "ru", "duration": 9.0,
        "segments": [
            {"id": 0, "start": 0.2, "end": 2.5, "text": "Если вы хотите изменить результат"},
            {"id": 1, "start": 2.7, "end": 5.5, "text": "то начните действовать каждый день."},
        ],
    }
    features = analyse_transcript(transcript, config.transcript_features)
    original = _unit(transcript, features, config)
    unfinished = replace(
        original, end=2.5, duration=2.3, transcript_segment_ids=[0],
        development="Если вы хотите изменить результат", ending="Если вы хотите изменить результат",
    )

    resolution = SemanticBoundaryEngine(config.content_understanding).resolve(unfinished, transcript, features, {"boundaries": []})

    assert resolution.transcript_segment_ids == [0, 1]
    assert resolution.diagnostics["semantic_extension_reason"] == "extended_to_sentence_completion"
    assert resolution.diagnostics["eligible"] is True

    config.content_understanding.max_semantic_extension_seconds = 0.0
    forbidden = SemanticBoundaryEngine(config.content_understanding).resolve(unfinished, transcript, features, {"boundaries": []})
    assert forbidden.diagnostics["eligible"] is False
    assert forbidden.diagnostics["end_boundary"]["boundary_type"] == "forbidden_end"


def test_selection_rejects_forbidden_semantic_boundary() -> None:
    candidate = Candidate(
        "candidate-boundary", 0, 20, "Полный текст.",
        boundary_diagnostics={"eligible": False, "fallback_reason": "Конец не завершает предложение."},
    )
    scored = ScoredCandidate(candidate, "", "", "", 90, 90, 90, 90, 90, 0, None, True)

    selected = select_clips([scored], AppConfig(score_threshold=0))

    assert selected == []
    assert scored.selection_diagnostics["decision"] == "rejected_boundary"


def test_source_context_preserves_resolved_candidate_range() -> None:
    candidate = Candidate("candidate-range", 1.0, 2.5, "Первое предложение.", transcript_segment_ids=[0])
    transcript = {
        "source_id": "source-1", "language": "ru", "duration": 4.0,
        "segments": [{"start": 0.5, "end": 3.0, "text": "Первое предложение."}],
    }
    features = analyse_transcript(transcript, AppConfig().transcript_features)

    context = build_source_context(
        {"id": "source-1", "path": "source.mp4"}, {"duration": 4.0}, candidate, transcript,
        features, {"windows": []}, {"boundaries": []}, AppConfig().transformation,
    )

    assert context.primary_evidence[0].start == 1.0
    assert context.primary_evidence[0].end == 2.5
