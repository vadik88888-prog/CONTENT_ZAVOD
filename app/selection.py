from __future__ import annotations

from app.config import AppConfig
from app.models import ScoredCandidate


def select_clips(scored: list[ScoredCandidate], config: AppConfig) -> list[ScoredCandidate]:
    accepted: list[ScoredCandidate] = []
    for item in sorted(scored, key=lambda value: value.score, reverse=True):
        if not item.selected:
            continue
        if item.score < config.score_threshold:
            item.selected = False
            item.selection_reason = "Оценка ниже порога."
            continue
        conflict = next((chosen for chosen in accepted if _overlap(item, chosen) >= config.overlap_threshold), None)
        if conflict:
            item.selected = False
            item.selection_reason = f"Сильно пересекается с более высоким кандидатом {conflict.candidate.id}."
            continue
        nearby = next(
            (
                chosen for chosen in accepted
                if abs(item.candidate.start - chosen.candidate.start) < config.min_selected_clip_distance_seconds
            ),
            None,
        )
        if nearby:
            item.selected = False
            item.selection_reason = (
                f"Слишком близок по времени к выбранному кандидату {nearby.candidate.id}; "
                "сохранено разнообразие клипов."
            )
            continue
        item.selection_reason = "Выбран: прошёл порог качества и не пересекается с лучшими моментами."
        accepted.append(item)
        if len(accepted) >= min(config.max_clips, config.ai_reranking.final_clip_count):
            break
    for item in scored:
        if item not in accepted and item.selected:
            item.selected = False
            item.selection_reason = "Не вошёл в лимит количества клипов."
    return accepted


def _overlap(first: ScoredCandidate, second: ScoredCandidate) -> float:
    left, right = first.candidate, second.candidate
    overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    shortest = min(left.duration, right.duration)
    return overlap / shortest if shortest else 0.0
