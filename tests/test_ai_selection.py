from app.ai import MockScorer
from app.config import AppConfig
from app.models import Candidate, ScoredCandidate
from app.selection import select_clips


def test_mock_scorer_is_deterministic_and_selection_removes_overlap() -> None:
    config = AppConfig(score_threshold=0, max_clips=2, overlap_threshold=0.5)
    candidates = [
        Candidate("first", 0, 35, "Это самостоятельная и законченная мысль для короткого видео."),
        Candidate("overlap", 4, 37, "Второй фрагмент пересекается с первым, но тоже понятен."),
        Candidate("third", 50, 85, "Третий фрагмент расположен отдельно и содержит завершённую мысль."),
    ]
    scored, usage = MockScorer(config).score(candidates, {"language": "ru"})

    selected = select_clips(scored, config)

    assert usage["provider"] == "mock"
    assert len(selected) == 2
    assert {item.candidate.id for item in selected} != {"first", "overlap"}
    assert any(item.selection_reason for item in scored)


def _scored(candidate_id: str, start: float, end: float, text: str, score: int) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=Candidate(candidate_id, start, end, text), title=candidate_id, hook="hook", summary="summary",
        score=score, hook_score=score, completeness_score=score, emotional_score=score,
        clarity_score=score, context_dependency_score=0, rejection_reason=None, selected=True,
    )


def test_selection_keeps_three_distinct_temporal_candidates() -> None:
    config = AppConfig(score_threshold=0, max_clips=3)
    config.ai_reranking.final_clip_count = 3
    selected = select_clips([
        _scored("one", 0, 20, "Первый самостоятельный тезис.", 96),
        _scored("two", 35, 55, "Второй самостоятельный тезис.", 95),
        _scored("three", 70, 90, "Третий самостоятельный тезис.", 94),
    ], config)

    assert [item.candidate.id for item in selected] == ["one", "two", "three"]


def test_selection_replaces_overlap_and_containment_with_next_alternative() -> None:
    config = AppConfig(score_threshold=0, max_clips=3, overlap_threshold=0.5)
    config.ai_reranking.final_clip_count = 3
    scored = [
        _scored("best", 100, 140, "Сильный отдельный тезис.", 100),
        _scored("contained", 108, 128, "Вложенная версия того же тезиса.", 99),
        _scored("overlap", 125, 150, "Пересекающаяся версия того же тезиса.", 98),
        _scored("alternative-one", 170, 195, "Первый альтернативный тезис.", 97),
        _scored("alternative-two", 220, 245, "Второй альтернативный тезис.", 96),
    ]

    selected = select_clips(scored, config)

    assert [item.candidate.id for item in selected] == ["best", "alternative-one", "alternative-two"]
    assert scored[1].selection_diagnostics["containment"] == 1.0


def test_selection_rejects_same_text_even_with_different_candidate_ids_and_timestamps() -> None:
    config = AppConfig(score_threshold=0, max_clips=2)
    config.ai_reranking.final_clip_count = 2
    selected = select_clips([
        _scored("first-copy", 0, 20, "Одна и та же законченная мысль для короткого видео", 100),
        _scored("second-copy", 80, 100, "Одна и та же законченная мысль для короткого видео", 99),
        _scored("alternative", 140, 160, "Новая самостоятельная мысль для другого ролика", 98),
    ], config)

    assert [item.candidate.id for item in selected] == ["first-copy", "alternative"]
