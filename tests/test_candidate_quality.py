from __future__ import annotations

import pytest

from app.candidate_quality import (
    CANDIDATE_QUALITY_SCHEMA_VERSION,
    EligibilityReasonCode,
    assess_context_debt,
)
from app.config import AppConfig
from app.content_understanding import select_with_coverage
from app.intelligence import local_rank, merge_ai_ranking
from app.local_scoring import score_candidates
from app.models import Candidate, ScoredCandidate, candidate_from_dict, scored_from_dict
from app.selection import select_clips


def _candidate(identifier: str = "candidate-quality", text: str = "Why deliberate practice works. The result is reliable progress.") -> Candidate:
    return Candidate(
        identifier, 0.0, 30.0, text, reason="semantic candidate",
        story_unit_id="story-quality",
        semantic_evidence={
            "hook": "Why deliberate practice works",
            "payoff": "The result is reliable progress",
            "completeness_score": 0.9,
            "information_density": 0.7,
        },
        boundary_diagnostics={
            "eligible": True, "word_integrity": True, "sentence_integrity": True,
            "semantic_completion": 0.9, "payoff_preserved": True,
            "overall_boundary_score": 0.9,
        },
        feature_vector={
            "hook_phrase_score": 80.0, "completeness_score": 90.0,
            "context_dependency_score": 0.0, "speech_density": 0.7,
            "words_per_second": 2.5, "word_count": 18, "sentence_start": True,
            "sentence_end": True, "transcript_confidence": 0.95,
            "repetition_score": 0.0, "filler_word_ratio": 0.0,
        },
    )


def _score(candidate: Candidate) -> Candidate:
    config = AppConfig()
    score_candidates(
        [candidate], {"energy_frames": [], "silence_intervals": []}, {"boundaries": []}, config.scoring,
        min_duration_seconds=config.min_clip_duration, max_duration_seconds=config.max_clip_duration,
        visual_analysis={"status": "completed", "subject_keyframes": [{"timestamp": 1.0}]},
    )
    return candidate


def test_eligible_candidate_persists_versioned_decision_score_and_evidence() -> None:
    candidate = _score(_candidate())

    assert candidate.eligibility_decision is not None
    assert candidate.eligibility_decision.eligible is True
    assert candidate.candidate_score_v2 is not None
    assert candidate.candidate_score_v2.final_score == candidate.local_quality_score
    serialized = candidate.to_dict()
    assert serialized["reason"] == "semantic candidate"
    assert serialized["eligibility_decision"]["schema_version"] == CANDIDATE_QUALITY_SCHEMA_VERSION
    assert serialized["eligibility_decision"]["config_version"] == AppConfig().scoring.candidate_quality_config_version
    assert serialized["eligibility_decision"]["evidence_refs"]
    assert serialized["candidate_score_v2"]["component_scores"]
    assert serialized["candidate_score_v2"]["penalties"] == []
    assert candidate_from_dict(serialized).eligibility_decision is not None


def test_high_score_explicitly_ineligible_candidate_cannot_enter_coverage_selection() -> None:
    candidate = _score(_candidate("candidate-ineligible", "It fixes the issue. The result is reliable progress."))
    assert candidate.eligibility_decision is not None and candidate.eligibility_decision.eligible is False
    scored = ScoredCandidate(candidate, "title", "hook", "summary", 100, 100, 100, 100, 100, 0, None, True)
    content_map = {
        "schema_version": "5A.1", "source_id": "source-1", "source_duration_seconds": 30.0,
        "chapters": [], "story_units": [], "evidence": {},
    }
    config = AppConfig(score_threshold=0)

    selected, _coverage = select_with_coverage([scored], config, content_map)

    assert selected == []
    assert scored.selected is False
    assert scored.selection_diagnostics["decision"] == "rejected_coverage"
    assert "CONTEXT_DEBT_CRITICAL" in (scored.selection_reason or "")


def test_legacy_candidate_is_explicitly_unassessed_and_is_not_a_v2_pass() -> None:
    raw = {
        "id": "legacy-candidate", "start": 0.0, "end": 20.0, "text": "A legacy completed sentence.",
        "score": 98, "hook_score": 98, "completeness_score": 98, "emotional_score": 50,
        "clarity_score": 98, "context_dependency_score": 0, "rejection_reason": None, "selected": True,
    }
    scored = scored_from_dict(raw)

    serialized = scored.to_dict()
    assert serialized["eligibility_decision"]["state"] == "legacy_unassessed"
    assert serialized["eligibility_decision"]["eligible"] is None
    assert scored.candidate.eligibility_decision is not None
    assert scored.candidate.eligibility_decision.state.value == "legacy_unassessed"
    assert select_clips([scored], AppConfig(score_threshold=0)) == []
    assert scored.selection_diagnostics["eligibility_state"] == "legacy_unassessed"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It fixes the issue. The result is reliable progress.", EligibilityReasonCode.UNRESOLVED_PRONOUN),
        ("Yes, use a shorter feedback loop. The result is reliable progress.", EligibilityReasonCode.ANSWER_WITHOUT_QUESTION_CONTEXT),
        ("This method removes the delay. The result is reliable progress.", EligibilityReasonCode.UNDEFINED_TERM_OR_SETUP),
    ],
)
def test_context_debt_has_deterministic_reason_codes(text: str, expected: EligibilityReasonCode) -> None:
    candidate = _score(_candidate("candidate-context", text))
    assert candidate.eligibility_decision is not None
    assert expected in candidate.eligibility_decision.reason_codes
    assert EligibilityReasonCode.CONTEXT_DEBT_CRITICAL in candidate.eligibility_decision.reason_codes
    assert candidate.eligibility_decision.eligible is False


def test_missing_payoff_and_false_hook_are_blocked_with_evidence() -> None:
    candidate = _candidate("candidate-missing-payoff", "Why this changes everything")
    candidate.semantic_evidence["payoff"] = ""
    candidate.boundary_diagnostics.update({"semantic_completion": 0.1, "payoff_preserved": False})
    candidate.feature_vector["completeness_score"] = 30.0
    candidate = _score(candidate)

    assert candidate.eligibility_decision is not None
    assert EligibilityReasonCode.NO_PAYOFF in candidate.eligibility_decision.reason_codes
    assert EligibilityReasonCode.FALSE_HOOK_RISK in candidate.eligibility_decision.reason_codes
    assert candidate.eligibility_decision.eligible is False
    assert any(item.code == "payoff" for item in candidate.eligibility_decision.evidence_refs)


def test_missing_visual_evidence_is_explicit_recoverable_state_not_a_pass_signal() -> None:
    candidate = _candidate("candidate-visual")
    config = AppConfig()
    score_candidates(
        [candidate], {"energy_frames": [], "silence_intervals": []}, {"boundaries": []}, config.scoring,
        min_duration_seconds=config.min_clip_duration, max_duration_seconds=config.max_clip_duration,
        visual_analysis={"enabled": True, "status": "fallback", "reason": "visual_provider_unavailable", "subject_keyframes": []},
    )

    assert candidate.eligibility_decision is not None
    assert EligibilityReasonCode.VISUAL_EVIDENCE_UNAVAILABLE in candidate.eligibility_decision.reason_codes
    assert EligibilityReasonCode.VISUAL_EVIDENCE_UNAVAILABLE in candidate.eligibility_decision.recoverable_issues
    evidence = next(item for item in candidate.eligibility_decision.evidence_refs if item.code == "visual_viability")
    assert evidence.state.value == "unavailable"
    assert candidate.candidate_score_v2 is not None
    assert any(item.code == "VISUAL_EVIDENCE_UNAVAILABLE" for item in candidate.candidate_score_v2.penalties)


def test_ungrounded_ai_item_keeps_deterministic_v2_fallback_provenance() -> None:
    candidate = _score(_candidate())
    ungrounded = ScoredCandidate(
        Candidate("unknown-candidate", 0, 30, "Unrelated text."), "", "", "", 100, 100, 100, 100, 100, 0, None, True,
    )

    ranked = merge_ai_ranking([candidate], [ungrounded], ai_ok=True)

    assert ranked[0].score == round(candidate.local_quality_score)
    assert candidate.candidate_score_v2 is not None
    assert candidate.candidate_score_v2.provenance["fallback_used"] is True
    assert candidate.candidate_score_v2.provenance["fallback_reason"] == "ai_result_missing_or_ungrounded"


def test_context_debt_detector_uses_only_candidate_text_and_features() -> None:
    codes, evidence = assess_context_debt("As I said earlier, this changes everything.", {"context_dependency_score": 70})

    assert EligibilityReasonCode.REFERENCES_EARLIER_CONTENT in codes
    assert evidence
