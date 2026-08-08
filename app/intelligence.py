from __future__ import annotations

from typing import Any

from app.config import AppConfig
from app.candidate_quality import apply_ai_factor_assessments, set_ai_merge_provenance
from app.models import Candidate, ScoredCandidate


def shortlist(candidates: list[Candidate], size: int) -> list[Candidate]:
    return sorted(candidates, key=lambda item: item.local_quality_score, reverse=True)[:size]


def local_rank(candidates: list[Candidate]) -> list[ScoredCandidate]:
    return [_local_scored(candidate) for candidate in candidates]


def merge_ai_ranking(
    candidates: list[Candidate], ai_scored: list[ScoredCandidate], ai_ok: bool
) -> list[ScoredCandidate]:
    if not ai_ok:
        local_items = local_rank(candidates)
        for item in local_items:
            set_ai_merge_provenance(item.candidate, ai_score=None, merged_score=None, reason="ai_result_unavailable")
        return local_items
    ai_by_id = {item.candidate.id: item for item in ai_scored}
    ranked: list[ScoredCandidate] = []
    for candidate in candidates:
        local = _local_scored(candidate)
        semantic = ai_by_id.get(candidate.id)
        if semantic is None:
            set_ai_merge_provenance(candidate, ai_score=None, merged_score=None, reason="ai_result_missing_or_ungrounded")
            ranked.append(local)
            continue
        # Overall score/selected are compatibility fields. Code consumes only
        # factor assessments and remains the final-score/ranking owner.
        candidate.ai_score = round((
            semantic.hook_score + semantic.completeness_score + semantic.emotional_score
            + semantic.clarity_score + (100 - semantic.context_dependency_score)
        ) / 5, 3)
        apply_ai_factor_assessments(candidate, semantic)
        if candidate.candidate_score_v2 is not None:
            candidate.local_quality_score = round(candidate.candidate_score_v2.final_score, 3)
        semantic.candidate = candidate
        semantic.score = max(0, min(100, round(candidate.local_quality_score)))
        semantic.selected = True
        semantic.selection_reason = None
        set_ai_merge_provenance(candidate, ai_score=candidate.ai_score, merged_score=semantic.score, reason="ai_factor_assessment")
        ranked.append(semantic)
    return ranked


def intelligence_summary(
    transcript_features: dict[str, Any], audio_features: dict[str, Any], scene_boundaries: dict[str, Any],
    candidates: list[Candidate], shortlist_items: list[Candidate], ai_used: bool, ai_fallback: bool,
    selection_mode: str,
    candidates_generated: int | None = None,
) -> dict[str, Any]:
    return {
        "version": "1.6",
        "transcript_feature_count": len(transcript_features.get("segments", [])),
        "scene_boundary_count": len(scene_boundaries.get("boundaries", [])),
        "silence_interval_count": len(audio_features.get("silence_intervals", [])),
        "candidates_generated": candidates_generated if candidates_generated is not None else len(candidates),
        "candidates_after_deduplication": len(candidates),
        "shortlist_size": len(shortlist_items),
        "selection_mode": selection_mode,
        "ai_reranking_used": ai_used,
        "ai_fallback_used": ai_fallback,
    }


def _local_scored(candidate: Candidate) -> ScoredCandidate:
    local = candidate.local_scores
    score = int(round(candidate.local_quality_score))
    return ScoredCandidate(
        candidate=candidate,
        title=_title(candidate.text),
        hook=_hook(candidate.text),
        summary=candidate.text[:600],
        score=score,
        hook_score=int(round(float(local.get("hook", 0)))),
        completeness_score=int(round(float(local.get("completeness", 0)))),
        emotional_score=0,
        clarity_score=int(round(float(local.get("clarity", 0)))),
        context_dependency_score=int(round(100 - float(local.get("context_independence", 0)))),
        rejection_reason=None,
        selected=True,
    )


def _title(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()[:8]).rstrip(".,!?…") or "Фрагмент видео"


def _hook(text: str) -> str:
    return text.split(".", 1)[0].strip()[:300] or text[:300]
