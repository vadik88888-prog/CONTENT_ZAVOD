from __future__ import annotations

from app.config import AppConfig
from app.content_understanding import (
    GlobalContentMap,
    build_global_content_map,
    build_video_content_profile,
    recommend_clip_count,
    select_with_coverage,
)
from app.models import Candidate, ScoredCandidate
from app.transcript_features import analyse_transcript


def _content_map(texts: list[str]) -> dict:
    segments = [
        {"id": index, "start": float(index * 25), "end": float(index * 25 + 18), "text": text}
        for index, text in enumerate(texts)
    ]
    transcript = {"source_id": "source-1", "language": "ru", "duration": float(len(segments) * 25), "segments": segments}
    config = AppConfig(score_threshold=0)
    features = analyse_transcript(transcript, config.transcript_features)
    profile = build_video_content_profile(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": transcript["duration"]}, transcript,
        features, {"windows": []}, {"boundaries": []}, {"samples": []}, config,
    )
    return build_global_content_map(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": transcript["duration"]}, transcript,
        features, {"windows": []}, {"boundaries": []}, {"samples": []}, profile, config,
    )


def _scored(content_map: dict, scores: list[int]) -> list[ScoredCandidate]:
    stories = GlobalContentMap.from_dict(content_map).story_units
    result: list[ScoredCandidate] = []
    for story, score in zip(stories, scores, strict=True):
        candidate = Candidate(
            id=f"candidate-{story.story_unit_id}", start=story.start, end=story.end, text=story.development,
            transcript_segment_ids=story.transcript_segment_ids, chapter_id=story.chapter_id,
            story_unit_id=story.story_unit_id, core_idea=story.core_idea,
            content_signature=story.content_signature,
            boundary_diagnostics={"eligible": True, "overall_boundary_score": 0.9},
        )
        result.append(ScoredCandidate(candidate, story.title, story.hook_seed, story.core_idea, score, score, 90, 50, 90, 10, None, True))
    return result


def test_coverage_selection_prefers_different_strong_story_units_and_chapters() -> None:
    content_map = _content_map([
        "Дисциплина важнее настроения, потому что действие создаёт результат.",
        "Смелость появляется после первого решения, а не до него.",
        "Ответственность возвращает контроль над собственной жизнью.",
        "Повторяйте полезное действие каждый день и увидите итог.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 3
    selected, coverage = select_with_coverage(_scored(content_map, [99, 96, 95, 94]), config, content_map)

    assert len(selected) == 3
    assert len({item.candidate.story_unit_id for item in selected}) == 3
    assert len({item.candidate.chapter_id for item in selected}) == 3
    assert coverage["coverage_ratio_by_dimension"]["chapter"] == 0.75
    assert all(item.candidate.incremental_coverage_score > 0 for item in selected)
    assert len(coverage["selection_explanations"]) == 3


def test_semantic_duplicate_is_not_used_to_fill_requested_count() -> None:
    content_map = _content_map([
        "Дисциплина важнее настроения, потому что действие создаёт результат.",
        "Дисциплина важнее настроения, потому что действие создаёт результат.",
        "Смелость появляется после первого решения, а не до него.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 3
    scored = _scored(content_map, [99, 98, 95])
    selected, coverage = select_with_coverage(scored, config, content_map)

    assert len(selected) == 2
    assert len(coverage["duplicate_content_clusters"]) == 1
    assert any("Семантически повторяет" in (item.selection_reason or "") for item in scored if item not in selected)


def test_clip_count_recommendation_uses_distinct_story_units_not_duration() -> None:
    dense = _content_map([
        "Первый законченный аргумент и его вывод.",
        "Второй самостоятельный урок с понятным итогом.",
        "Третий пример доказывает отдельную мысль.",
        "Четвёртая история заканчивается полезным выводом.",
        "Пятый совет можно использовать сразу.",
    ])
    repeated = _content_map([
        "Одна и та же мысль повторяется для закрепления результата.",
        "Одна и та же мысль повторяется для закрепления результата.",
        "Одна и та же мысль повторяется для закрепления результата.",
        "Одна и та же мысль повторяется для закрепления результата.",
        "Одна и та же мысль повторяется для закрепления результата.",
    ])
    profile = build_video_content_profile(
        {"id": "source-1", "display_name": "source.mp4"}, {"duration": 125.0},
        {"source_id": "source-1", "language": "ru", "duration": 125.0, "segments": []},
        {"language": "ru", "segments": []}, {"windows": []}, {"boundaries": []}, {"samples": []}, AppConfig(),
    )

    dense_recommendation = recommend_clip_count(dense, profile, 3)
    repeated_recommendation = recommend_clip_count(repeated, profile, 3)

    assert dense_recommendation["estimated_publishable_clip_range"]["max"] > repeated_recommendation["estimated_publishable_clip_range"]["max"]
    assert dense_recommendation["estimated_story_count"] == 5
    assert "самостоятельных сильных" in dense_recommendation["explanation"]


def test_coverage_selection_is_deterministic() -> None:
    content_map = _content_map([
        "Первый законченный аргумент и его вывод.",
        "Второй самостоятельный урок с понятным итогом.",
        "Третий пример доказывает отдельную мысль.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 2

    first, _ = select_with_coverage(_scored(content_map, [90, 90, 90]), config, content_map)
    second, _ = select_with_coverage(_scored(content_map, [90, 90, 90]), config, content_map)

    assert [item.candidate.id for item in first] == [item.candidate.id for item in second]
