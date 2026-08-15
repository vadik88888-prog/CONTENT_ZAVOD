from __future__ import annotations

from app.config import AppConfig
from app.diversity import interval_metrics, is_temporal_duplicate, transcript_similarity
from app.models import ScoredCandidate
from app.editorial_profile_policy import editorial_decision_from_candidate, evaluate_editorial_candidate
from app.production_feasibility import production_feasibility_index


TRANSCRIPT_DUPLICATE_THRESHOLD = 0.90


def select_clips(
    scored: list[ScoredCandidate], config: AppConfig, content_map: dict | None = None,
    production_feasibility: dict | None = None, content_profile: dict | None = None,
) -> list[ScoredCandidate]:
    """Select the strongest candidates while preserving source-content diversity."""

    if content_map is not None:
        from app.content_understanding import select_with_coverage

        selected, _coverage = select_with_coverage(
            scored, config, content_map, production_feasibility=production_feasibility,
            content_profile=content_profile,
        )
        return selected

    accepted: list[ScoredCandidate] = []
    feasibility_by_id = production_feasibility_index(production_feasibility)
    allow_ranked_replacements = bool(
        isinstance(production_feasibility, dict)
        and production_feasibility.get("allow_ranked_replacements")
    )
    rankable: list[ScoredCandidate] = []
    for item in scored:
        if not item.selected and not allow_ranked_replacements:
            continue
        boundary = item.candidate.boundary_diagnostics
        if boundary and not bool(boundary.get("eligible", False)):
            item.selected = False
            item.selection_reason = str(boundary.get("fallback_reason") or "Semantic boundary не прошла no-cut-off validation.")
            item.selection_diagnostics = {"decision": "rejected_boundary", "boundary": boundary}
            continue
        decision = (
            evaluate_editorial_candidate(
                item.candidate,
                content_profile,
                score=float(item.score),
                production_feasibility=feasibility_by_id.get(item.candidate.id),
            )
            if content_profile is not None
            else editorial_decision_from_candidate(item.candidate)
            or evaluate_editorial_candidate(
                item.candidate,
                None,
                score=float(item.score),
                production_feasibility=feasibility_by_id.get(item.candidate.id),
            )
        )
        item.candidate.editorial_decision = decision
        if not decision.selectable:
            item.selected = False
            item.selection_reason = "Candidate не прошёл structural/technical policy check."
            item.selection_diagnostics = {
                "decision": "rejected_editorial_integrity",
                "surfacing_state": decision.surfacing_state.value,
                "reason_codes": list(decision.hard_blockers),
                "eligibility_state": (
                    item.candidate.eligibility_decision.state.value
                    if item.candidate.eligibility_decision is not None else "legacy_unassessed"
                ),
            }
            continue
        feasibility = feasibility_by_id.get(item.candidate.id)
        if feasibility is not None and feasibility.get("status") == "GUARANTEED_BLOCKED":
            item.selected = False
            item.selection_reason = str(
                feasibility.get("reason")
                or "Guaranteed blocked by provider-free production feasibility."
            )
            item.selection_diagnostics = {
                "decision": "rejected_production_feasibility",
                "reason_code": "PRODUCTION_FEASIBILITY_BLOCKED",
                "production_feasibility": feasibility,
            }
            continue
        if item.score < config.score_threshold:
            item.selected = False
            item.selection_diagnostics = {
                "decision": "rejected_score",
                "production_feasibility": feasibility,
            }
            item.selection_reason = "Оценка ниже порога."
            continue
        rankable.append(item)

    # Profile-aware rank is consumed only after every hard gate above has
    # passed. It never participates in eligibility, boundary, feasibility, or
    # threshold decisions.
    for item in sorted(
        rankable,
        key=lambda value: float(
            value.virality.get("ranking_sort_score", value.score / 100)
        ),
        reverse=True,
    ):
        feasibility = feasibility_by_id.get(item.candidate.id)
        duplicate = _duplicate_against(item, accepted, config)
        if duplicate is not None:
            kind, chosen, details = duplicate
            item.selected = False
            item.selection_diagnostics = {
                **details,
                "production_feasibility": feasibility,
            }
            if kind == "transcript":
                item.selection_reason = f"Почти повторяет текст кандидата {chosen.candidate.id}."
            else:
                item.selection_reason = f"Дублирует исходниковый диапазон кандидата {chosen.candidate.id}."
            continue
        item.selection_reason = "Выбран: качество и временное разнообразие подтверждены."
        item.selection_diagnostics = {
            "decision": "accepted",
            "production_feasibility": feasibility,
        }
        accepted.append(item)
        if len(accepted) >= min(config.max_clips, config.ai_reranking.final_clip_count):
            break
    for item in scored:
        if item not in accepted and item.selected:
            item.selected = False
            item.selection_reason = "Не вошёл в лимит количества клипов."
            item.selection_diagnostics = {
                "decision": "limit",
                "production_feasibility": feasibility_by_id.get(item.candidate.id),
            }
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
