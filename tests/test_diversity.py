from __future__ import annotations

from app.diversity import DIVERSITY_LEGACY_REASON_CODE, DiversityDecision


def test_legacy_diversity_artifact_is_explicitly_unassessed() -> None:
    decision = DiversityDecision.from_dict(None)

    assert decision.schema_version == "legacy"
    assert decision.result_reason_code == DIVERSITY_LEGACY_REASON_CODE


def test_diversity_decision_round_trips_all_recorded_evidence() -> None:
    raw = {
        "schema_version": "5B.2",
        "config_version": "test",
        "requested_count": 2,
        "lambda": 0.76,
        "eligible_candidate_ids": ["candidate-a", "candidate-b"],
        "selected_candidate_ids": ["candidate-a"],
        "selections": [{
            "candidate_id": "candidate-a", "reason_code": "SELECTED_MMR",
            "coverage_quality_score": 0.84, "max_similarity": 0.0,
            "against_candidate_id": None, "mmr_score": 0.6384,
        }],
        "exclusions": [{
            "candidate_id": "candidate-b", "reason_code": "SEMANTIC_DUPLICATE",
            "reason": "duplicate", "against_candidate_id": "candidate-a", "max_similarity": 0.91,
            "similarity": {
                "candidate_id": "candidate-a", "other_candidate_id": "candidate-b",
                "composite_similarity": 0.91, "components": {"semantic": 1.0},
                "available_components": ["semantic"],
            },
        }],
        "similarities": [{
            "candidate_id": "candidate-a", "other_candidate_id": "candidate-b",
            "composite_similarity": 0.91, "components": {"semantic": 1.0},
            "available_components": ["semantic"],
        }],
        "result_reason_code": "INSUFFICIENT_UNIQUE_CANDIDATES",
    }

    assert DiversityDecision.from_dict(raw).to_dict() == raw
