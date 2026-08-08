from __future__ import annotations

import pytest

from app.config import AppConfig
from app.intelligence import merge_ai_ranking
from app.local_scoring import score_candidates
from app.models import Candidate, ScoredCandidate, candidate_from_dict


EXPECTED_FACTORS = {
    "hook", "narrative_completeness", "payoff", "information_value",
    "emotional_intensity", "visual_interest", "audio_energy",
    "self_containedness", "context_debt", "vertical_viability", "novelty",
    "confidence",
}


def _candidate(identifier: str, *, hook: float, completeness: float, information: float) -> Candidate:
    return Candidate(
        identifier, 0.0, 30.0, "Why this matters. The result is clear.",
        story_unit_id=identifier, story_unit_ids=[identifier],
        semantic_evidence={
            "hook": "Why this matters", "payoff": "The result is clear",
            "completeness_score": completeness / 100, "information_density": information / 100,
        },
        boundary_diagnostics={
            "eligible": True, "word_integrity": True, "sentence_integrity": True,
            "semantic_completion": completeness / 100, "payoff_preserved": True,
            "overall_boundary_score": completeness / 100,
        },
        feature_vector={
            "hook_phrase_score": hook, "completeness_score": completeness,
            "context_dependency_score": 0, "speech_density": 0.7,
            "words_per_second": 2.5, "word_count": 20, "sentence_start": True,
            "sentence_end": True, "transcript_confidence": 0.92,
            "repetition_score": 0, "filler_word_ratio": 0,
        },
    )


def _score(candidates: list[Candidate], *, energetic: bool = False) -> None:
    config = AppConfig()
    audio = {
        "energy_frames": ([{"time": 10.0, "normalized_loudness": 0.85}] if energetic else []),
        "silence_intervals": [],
    }
    score_candidates(
        candidates, audio, {"boundaries": []}, config.scoring,
        min_duration_seconds=15, max_duration_seconds=60,
        visual_analysis={"status": "fallback", "subject_keyframes": []},
    )


def _attach_strong_pass2(candidate: Candidate) -> None:
    observation = {
        "timestamp": 29.0, "confidence": 0.95, "primary_subject": "face",
        "reaction": "surprise", "payoff_signal": "result", "composition_risk": "none",
        "visible_face_count": 1, "action": "gesture", "scene_type": "INTERVIEW_SINGLE",
    }
    candidate.multimodal_provenance = {
        "analysis_run_id": "analysis-mm", "visual_evidence": [observation],
        "generation": {
            "reasons": ["editorial_roles:action,reaction,payoff"],
            "anchors": {"payoff": 29.0},
        },
    }
    candidate.vision_pass2_evidence = {
        "status": "completed",
        "result": {
            "schema_version": "6B.pass2-result.1", "request": {"anchors": {"payoff": 29.0}},
            "verification": {
                "hook_visible": True, "action_visible": True, "reaction_visible": True,
                "payoff_visible": True, "continuity_risk": "low", "confidence": 0.95,
            },
            "observations": [observation],
        },
    }


def test_every_editorial_factor_is_explainable_and_round_trips() -> None:
    candidate = _candidate("factor-contract", hook=70, completeness=84, information=75)
    _attach_strong_pass2(candidate)
    _score([candidate], energetic=True)

    score = candidate.candidate_score_v2
    assert score is not None
    assert set(score.factors) == EXPECTED_FACTORS
    assert all(item.evidence_refs and 0 <= item.confidence <= 1 and item.provenance for item in score.factors.values())
    assert score.provenance["ai_owns_final_score"] is False
    assert score.diagnostics["positive_contributions"]
    restored = candidate_from_dict(candidate.to_dict())
    assert restored.candidate_score_v2 is not None
    assert set(restored.candidate_score_v2.factors) == EXPECTED_FACTORS


def test_pass2_reaction_and_payoff_provably_change_ranking() -> None:
    stronger_text = _candidate("stronger-text", hook=90, completeness=88, information=90)
    multimodal = _candidate("multimodal-payoff", hook=52, completeness=72, information=55)
    _score([stronger_text, multimodal])
    assert stronger_text.local_quality_score > multimodal.local_quality_score

    _attach_strong_pass2(multimodal)
    _score([multimodal], energetic=True)

    assert multimodal.local_quality_score > stronger_text.local_quality_score
    assert multimodal.candidate_score_v2 is not None
    assert multimodal.candidate_score_v2.factors["visual_interest"].score >= 90
    assert multimodal.candidate_score_v2.factors["payoff"].score == 92
    assert multimodal.composition_intent["reaction"]["value"] == "surprise"


def test_missing_vision_is_safe_and_lowers_confidence_only() -> None:
    candidate = _candidate("no-vision", hook=78, completeness=86, information=82)
    candidate.vision_pass2_evidence = {"status": "skipped", "reason": "provider unavailable", "result": None}
    _score([candidate], energetic=True)

    score = candidate.candidate_score_v2
    assert score is not None and score.final_score > 0
    assert score.factors["visual_interest"].evidence_refs[0].state.value == "unavailable"
    assert score.factors["confidence"].score < 90
    assert candidate.composition_intent["evidence_status"] == "unavailable"


def test_low_confidence_visual_evidence_cannot_boost_editorial_or_composition_intent() -> None:
    candidate = _candidate("low-confidence-visual", hook=70, completeness=84, information=75)
    _attach_strong_pass2(candidate)
    assert candidate.vision_pass2_evidence is not None
    result = candidate.vision_pass2_evidence["result"]
    result["verification"]["confidence"] = 0.25
    result["observations"][0]["confidence"] = 0.25
    candidate.multimodal_provenance["visual_evidence"][0]["confidence"] = 0.25

    _score([candidate])

    score = candidate.candidate_score_v2
    assert score is not None
    assert score.factors["visual_interest"].score == 32.5
    assert score.factors["vertical_viability"].score == 50
    assert score.provenance["pass2_status"] == "completed"
    assert candidate.composition_intent["evidence_status"] == "unavailable"


def test_raw_loudness_is_weaker_than_grounded_audio_editorial_event() -> None:
    ungrounded = _candidate("ungrounded-loudness", hook=70, completeness=84, information=75)
    grounded = _candidate("grounded-audio-event", hook=70, completeness=84, information=75)
    grounded.multimodal_provenance = {
        "audio_evidence": [{"event_type": "emphasis", "confidence": 0.9}],
        "generation": {"reasons": ["editorial_roles:hook"]},
    }

    _score([ungrounded, grounded], energetic=True)

    assert grounded.candidate_score_v2 is not None
    assert ungrounded.candidate_score_v2 is not None
    assert grounded.candidate_score_v2.factors["audio_energy"].score == 85
    assert ungrounded.candidate_score_v2.factors["audio_energy"].score == pytest.approx(29.75)
    assert grounded.local_quality_score > ungrounded.local_quality_score + 6


def test_ai_overall_score_and_selected_flag_do_not_own_final_ranking() -> None:
    candidates = [
        _candidate("ai-low-overall", hook=70, completeness=84, information=75),
        _candidate("ai-high-overall", hook=70, completeness=84, information=75),
    ]
    _score(candidates)
    assessments = [
        ScoredCandidate(
            candidate, "title", "hook", "summary", overall, 70, 80, 60, 85, 10,
            None, selected,
        )
        for candidate, overall, selected in zip(candidates, (0, 100), (False, True))
    ]

    ranked = merge_ai_ranking(candidates, assessments, ai_ok=True)

    assert ranked[0].score == ranked[1].score
    assert all(item.selected for item in ranked)
    assert all(item.candidate.candidate_score_v2 is not None for item in ranked)
    assert all(item.candidate.candidate_score_v2.provenance["final_score_owner"] == "code" for item in ranked)  # type: ignore[union-attr]
