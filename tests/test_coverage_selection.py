from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.analysis_artifact import candidate_is_draftable
from app.candidate_quality import CANDIDATE_QUALITY_SCHEMA_VERSION, EligibilityDecision, EligibilityState
from app.content_understanding import (
    GlobalContentMap,
    build_global_content_map,
    build_video_content_profile,
    recommend_clip_count,
    select_with_coverage,
)
from app.models import Candidate, ScoredCandidate
from app.pipeline import Pipeline
from app.transcript_features import analyse_transcript
from app.utils import read_json


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
            eligibility_decision=EligibilityDecision(
                schema_version=CANDIDATE_QUALITY_SCHEMA_VERSION,
                config_version="test",
                state=EligibilityState.ASSESSED,
                eligible=True,
            ),
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
    decision = coverage["diversity_decision"]
    assert decision["schema_version"] == "5B.2"
    assert decision["result_reason_code"] == "INSUFFICIENT_UNIQUE_CANDIDATES"
    assert any(item["reason_code"] == "SEMANTIC_DUPLICATE" for item in decision["exclusions"])


def test_guaranteed_production_blocker_is_replaced_without_disabling_manual_choice() -> None:
    content_map = _content_map([
        "The first complete claim has a clear result and a useful conclusion.",
        "The second complete claim explains a different useful conclusion.",
        "The third complete claim provides another independent result.",
        "The fourth complete claim is a viable diverse alternative.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 3
    scored = _scored(content_map, [99, 98, 97, 96])
    blocked_id = scored[0].candidate.id
    scored[3].selected = False
    feasibility = {
        "allow_ranked_replacements": True,
        "candidates": [{
            "candidate_id": blocked_id,
            "status": "GUARANTEED_BLOCKED",
            "reason_code": "CAPTION_CPS_INFEASIBLE",
            "reason": "Guaranteed blocked by provider-free A-3 policy: CAPTION_CPS_INFEASIBLE.",
            "blockers": [{"gate": "A-3", "reason_code": "CAPTION_CPS_INFEASIBLE"}],
        }],
    }

    selected, coverage = select_with_coverage(
        scored,
        config,
        content_map,
        production_feasibility=feasibility,
    )

    assert blocked_id not in [item.candidate.id for item in selected]
    assert len(selected) == 3
    assert scored[3] in selected
    exclusion = next(
        item for item in coverage["diversity_decision"]["exclusions"]
        if item["candidate_id"] == blocked_id
    )
    assert exclusion["reason_code"] == "PRODUCTION_FEASIBILITY_BLOCKED"
    assert scored[0].selection_diagnostics["production_feasibility"]["reason_code"] == "CAPTION_CPS_INFEASIBLE"
    # Recommendation is filtered, but explicit manual selection still owns the
    # same eligibility decision and is allowed to reach the ordinary downstream gate.
    assert candidate_is_draftable(scored[0].to_dict()) is True


def test_mmr_selects_weaker_unique_candidate_instead_of_multiple_semantic_clones() -> None:
    content_map = _content_map([
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Clear ownership makes decisions faster. Therefore, name the owner before the meeting ends.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 3
    config.content_understanding.diversity_lambda = 0.72
    scored = _scored(content_map, [99, 98, 97, 91])

    selected, coverage = select_with_coverage(scored, config, content_map)

    assert scored[3] in selected
    assert len([item for item in selected if item in scored[:3]]) == 1
    decision = coverage["diversity_decision"]
    assert decision["lambda"] == 0.72
    assert len(decision["selections"]) == len(selected)
    assert all(item["reason_code"] == "SELECTED_MMR" for item in decision["selections"])
    assert decision["result_reason_code"] == "INSUFFICIENT_UNIQUE_CANDIDATES"


def test_similarity_matrix_excludes_ineligible_candidates() -> None:
    content_map = _content_map([
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Clear ownership makes decisions faster. Therefore, name the owner before the meeting ends.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 2
    scored = _scored(content_map, [99, 98, 95])
    ineligible = scored[1]
    assert ineligible.candidate.eligibility_decision is not None
    ineligible.candidate.eligibility_decision.eligible = False

    _selected, coverage = select_with_coverage(scored, config, content_map)

    decision = coverage["diversity_decision"]
    assert ineligible.candidate.id not in decision["eligible_candidate_ids"]
    assert all(
        ineligible.candidate.id not in {item["candidate_id"], item["other_candidate_id"]}
        for item in decision["similarities"]
    )
    exclusion = next(item for item in decision["exclusions"] if item["candidate_id"] == ineligible.candidate.id)
    assert exclusion["reason_code"] == "ELIGIBILITY_NOT_PASSED"


def test_final_selection_artifact_persists_versioned_diversity_decision(tmp_path: Path) -> None:
    content_map = _content_map([
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Teams make progress by acting before certainty arrives. Therefore, act before you feel ready.",
        "Clear ownership makes decisions faster. Therefore, name the owner before the meeting ends.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 3
    path = tmp_path / "final_selection.json"

    data = Pipeline(tmp_path, config, mock_ai=True)._final_selection(_scored(content_map, [99, 98, 91]), path, content_map)

    persisted = read_json(path, {})
    assert data["diversity_decision"]["schema_version"] == "5B.2"
    assert persisted["diversity_decision"] == data["diversity_decision"]
    assert persisted["coverage"]["diversity_decision"] == data["diversity_decision"]


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


def test_coverage_selection_rejects_subminimum_duration_candidates() -> None:
    content_map = _content_map([
        "Discipline creates progress when motivation disappears at the end of a hard day.",
        "Courage starts after a person makes the first difficult decision.",
        "Responsibility gives people control over the choices they make tomorrow.",
        "Small repeated actions create durable results over a long period of time.",
    ])
    config = AppConfig(score_threshold=0)
    config.ai_reranking.final_clip_count = 3
    scored = _scored(content_map, [99, 98, 97, 96])
    short = scored[0]
    short.candidate.end = short.candidate.start + config.min_clip_duration - 0.1

    selected, _coverage = select_with_coverage(scored, config, content_map)

    assert short not in selected
    assert len(selected) == 3
    assert "минимальных" in (short.selection_reason or "")


def test_coverage_selection_uses_strong_story_above_coverage_floor() -> None:
    content_map = _content_map([
        "Discipline creates progress when motivation disappears at the end of a hard day.",
        "Courage starts after a person makes the first difficult decision.",
        "Responsibility gives people control over the choices they make tomorrow.",
    ])
    config = AppConfig(score_threshold=60)
    config.ai_reranking.final_clip_count = 3

    selected, _coverage = select_with_coverage(_scored(content_map, [64, 62, 57]), config, content_map)

    assert len(selected) == 3
    assert any(item.score == 57 for item in selected)


def test_coverage_selection_uses_enabled_virality_floor_without_dropping_boundary_checks() -> None:
    content_map = _content_map([
        "Discipline creates progress when motivation disappears at the end of a hard day.",
        "Courage starts after a person makes the first difficult decision.",
        "Responsibility gives people control over the choices they make tomorrow.",
    ])
    config = AppConfig(score_threshold=60)
    config.ai_reranking.final_clip_count = 3
    config.virality.enabled = True
    config.virality.minimum_quality_score = 0.45
    scored = _scored(content_map, [50, 49, 48])
    for item in scored:
        item.virality = {"selection_eligible": True}

    selected, _coverage = select_with_coverage(scored, config, content_map)

    assert len(selected) == 3
