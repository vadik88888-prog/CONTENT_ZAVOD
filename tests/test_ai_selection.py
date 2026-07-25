from app.ai import MockScorer
from app.config import AppConfig
from app.models import Candidate
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
