from __future__ import annotations

from app.config import AppConfig
from app.diversity import interval_metrics, is_temporal_duplicate, transcript_similarity
from app.models import ScoredCandidate


TRANSCRIPT_DUPLICATE_THRESHOLD = 0.90


def select_clips(
    scored: list[ScoredCandidate], config: AppConfig, content_map: dict | None = None,
) -> list[ScoredCandidate]:
    """Select the strongest candidates while preserving source-content diversity."""

    if content_map is not None:
        from app.content_understanding import select_with_coverage

        selected, _coverage = select_with_coverage(scored, config, content_map)
        return selected

    accepted: list[ScoredCandidate] = []
    for item in sorted(scored, key=lambda value: value.score, reverse=True):
        if not item.selected:
            continue
        boundary = item.candidate.boundary_diagnostics
        if boundary and not bool(boundary.get("eligible", False)):
            item.selected = False
            item.selection_reason = str(boundary.get("fallback_reason") or "Semantic boundary не прошла no-cut-off validation.")
            item.selection_diagnostics = {"decision": "rejected_boundary", "boundary": boundary}
            continue
        decision = item.candidate.eligibility_decision
        if decision is None or not decision.explicitly_eligible:
            item.selected = False
            state = decision.state.value if decision is not None else "legacy_unassessed"
            item.selection_reason = "Candidate не имеет явного eligibility PASS."
            item.selection_diagnostics = {
                "decision": "rejected_eligibility",
                "eligibility_state": state,
                "reason_codes": [code.value for code in decision.reason_codes] if decision else ["LEGACY_UNASSESSED"],
            }
            continue
        if item.score < config.score_threshold:
            item.selected = False
            item.selection_reason = "Оценка ниже порога."
            continue
        duplicate = _duplicate_against(item, accepted, config)
        if duplicate is not None:
            kind, chosen, details = duplicate
            item.selected = False
            item.selection_diagnostics = details
            if kind == "transcript":
                item.selection_reason = f"Почти повторяет текст кандидата {chosen.candidate.id}."
            else:
                item.selection_reason = f"Дублирует исходниковый диапазон кандидата {chosen.candidate.id}."
            continue
        item.selection_reason = "Выбран: качество и временное разнообразие подтверждены."
        item.selection_diagnostics = {"decision": "accepted"}
        accepted.append(item)
        if len(accepted) >= min(config.max_clips, config.ai_reranking.final_clip_count):
            break
    for item in scored:
        if item not in accepted and item.selected:
            item.selected = False
            item.selection_reason = "Не вошёл в лимит количества клипов."
            item.selection_diagnostics = {"decision": "limit"}
    return accepted


def _duplicate_against(
    item: ScoredCandidate, accepted: list[ScoredCandidate], config: AppConfig,
) -> tuple[str, ScoredCandidate, dict[str, float | str]] | None:
    for chosen in accepted:
        metrics = interval_metrics(
            item.candidate.start,
            item.candidate.end,
            chosen.candidate.start,
            chosen.candidate.end,
        )
        details: dict[str, float | str] = {
            "against_candidate_id": chosen.candidate.id,
            "overlap_seconds": round(metrics.overlap_seconds, 3),
            "iou": round(metrics.iou, 4),
            "containment": round(metrics.containment, 4),
            "midpoint_distance_seconds": round(metrics.midpoint_distance_seconds, 3),
        }
        if is_temporal_duplicate(
            metrics,
            overlap_threshold=config.overlap_threshold,
            minimum_distance_seconds=config.min_selected_clip_distance_seconds,
        ):
            return "temporal", chosen, details
        similarity = transcript_similarity(item.candidate.text, chosen.candidate.text)
        if similarity >= TRANSCRIPT_DUPLICATE_THRESHOLD:
            details["transcript_similarity"] = round(similarity, 4)
            return "transcript", chosen, details
    return None
