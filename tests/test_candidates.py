from app.candidates import build_candidates
from app.config import AppConfig


def test_candidates_follow_segment_boundaries_and_add_padding() -> None:
    transcript = {
        "duration": 42,
        "segments": [
            {"start": 1, "end": 9, "text": "Первое важное предложение."},
            {"start": 9.2, "end": 18, "text": "Второе предложение завершает мысль."},
            {"start": 20, "end": 29, "text": "Новый самостоятельный фрагмент."},
            {"start": 29.2, "end": 39, "text": "Ещё одно законченное предложение."},
        ],
    }
    config = AppConfig(min_clip_duration=15, target_clip_duration=16, max_clip_duration=30)

    candidates = build_candidates(transcript, config)

    assert len(candidates) == 2
    assert candidates[0].start == 0.65
    assert candidates[0].end == 18.35
    assert all(15 <= candidate.duration <= 60 for candidate in candidates)


def test_short_tail_is_not_returned_as_clip() -> None:
    transcript = {
        "duration": 10,
        "segments": [{"start": 1, "end": 8, "text": "Слишком короткая часть."}],
    }
    assert build_candidates(transcript, AppConfig()) == []
