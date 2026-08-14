from __future__ import annotations

from app.config import AppConfig
from app.candidate_quality import EligibilityDecision, EligibilityState
from app.content_understanding import StoryUnit
from app.models import Candidate, ScoredCandidate
from app.virality import (
    VIRALITY_COMPONENTS,
    ViralPotentialScore,
    aggregate_viral_potential,
    apply_virality_ranking,
    assess_candidate_eligibility,
    build_estimated_retention_profile,
    build_publishability_assessment,
    build_virality_assessments,
    build_virality_feature_profile,
    resolve_virality_strategy,
)


def _story(payoff: str, segment_ids: list[int]) -> StoryUnit:
    return StoryUnit(
        story_unit_id="story-1", chapter_id="chapter-1", start=0, end=30, duration=30,
        transcript_segment_ids=segment_ids, title="A clear story", core_idea="Act before certainty arrives",
        hook_seed="A bold claim", setup="A real risk", development="A useful action", payoff=payoff, ending=payoff,
        emotional_arc="rising", dominant_emotion="determination", speaker_context="single_speaker",
        required_previous_context="", required_next_context="", standalone_score=0.86, completeness_score=0.88,
        clarity_score=0.84, context_dependency_score=0.10, information_density=0.76, repetition_score=0.04,
        transformation_potential=0.78, publishability_precheck=True, content_signature={"topic_ids": ["action"]},
        confidence=0.9, evidence={"segments": segment_ids},
    )


def _candidate(text: str, *, boundary: bool = True, eligible: bool = True) -> Candidate:
    return Candidate(
        "candidate-1", 0, 30, text, transcript_segment_ids=[0, 1, 2], story_unit_id="story-1",
        core_idea="Act before certainty arrives", boundary_diagnostics={"eligible": boundary, "overall_boundary_score": 0.9 if boundary else 0.1},
        feature_vector={"words_per_second": 2.5, "speech_density": 0.74, "filler_word_ratio": 0.02, "repetition_score": 0.04},
        eligibility_decision=EligibilityDecision(
            schema_version="6D.1", config_version="test",
            state=EligibilityState.ASSESSED, eligible=eligible,
        ),
    )


def _transcript(parts: list[str]) -> dict:
    return {
        "segments": [
            {
                "id": index, "start": index * 10.0, "end": (index + 1) * 10.0, "text": text,
                "transcript_confidence": 0.93, "speech_density": 0.74, "filler_word_ratio": 0.02,
                "repetition_score": 0.04, "context_dependency_score": 0, "exclamation_count": text.count("!"),
            }
            for index, text in enumerate(parts)
        ]
    }


def _diagnostics(parts: list[str], *, payoff: str = "Therefore, begin now!", boundary: bool = True):
    candidate = _candidate(" ".join(parts), boundary=boundary)
    story = _story(payoff, [0, 1, 2])
    transcript = _transcript(parts)
    audio = {"energy_frames": [{"time": 0, "normalized_loudness": 0.3}, {"time": 10, "normalized_loudness": 0.5}, {"time": 20, "normalized_loudness": 0.82}], "silence_intervals": []}
    profile = build_virality_feature_profile(candidate, {"story_units": [story.to_dict()]}, transcript, audio, content_strategy="motivational_monologue")
    retention = build_estimated_retention_profile(candidate, profile, transcript, audio)
    publishability = build_publishability_assessment(candidate, profile, retention)
    eligibility = assess_candidate_eligibility(candidate, profile, retention, publishability)
    return candidate, story, transcript, audio, profile, retention, publishability, eligibility


def test_strategy_weights_are_complete_and_materially_different():
    candidate, _story_item, _transcript_data, _audio, profile, retention, publishability, eligibility = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ])
    config = AppConfig()
    motivational = aggregate_viral_potential(
        candidate, profile, retention, publishability, eligibility,
        config.virality.strategy_weights["motivational_monologue"], {"strategy_id": "motivational_monologue"},
    )
    educational = aggregate_viral_potential(
        candidate, profile, retention, publishability, eligibility,
        config.virality.strategy_weights["generic_educational"], {"strategy_id": "generic_educational"},
    )

    assert set(motivational.components) == set(VIRALITY_COMPONENTS)
    assert motivational.components["emotion"].strategy_weight > educational.components["emotion"].strategy_weight
    assert educational.components["usefulness"].strategy_weight > motivational.components["usefulness"].strategy_weight
    assert motivational.confidence.overall.score > 0
    assert ViralPotentialScore.from_dict(motivational.to_dict()).to_dict() == motivational.to_dict()


def test_structured_effective_profile_owns_strategy_and_traits_are_inert() -> None:
    profile = {
        "strategy_id": "motivational_monologue",
        "effective_profile": {
            "format": "dialogue",
            "editorial_mode": "interview",
            "domain": "business",
            "traits": ["high_pacing", "visual_led"],
        },
    }
    trait_only_change = {
        **profile,
        "effective_profile": {
            **profile["effective_profile"],
            "traits": ["scene_driven", "instructional"],
        },
    }

    assert resolve_virality_strategy(profile) == "generic_dialogue"
    assert resolve_virality_strategy(trait_only_change) == "generic_dialogue"
    assert resolve_virality_strategy({"strategy_id": "generic_educational"}) == "generic_educational"


def test_complete_story_beats_hook_without_payoff_and_confidence_stays_separate():
    complete = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ])
    hook_only = _diagnostics([
        "The only reason teams fail is fear.", "Everyone waits and waits.", "The meeting continues.",
    ], payoff="")
    config = AppConfig()
    complete_score = aggregate_viral_potential(*complete[0:1], complete[4], complete[5], complete[6], complete[7], config.virality.strategy_weights["motivational_monologue"])
    hook_score = aggregate_viral_potential(*hook_only[0:1], hook_only[4], hook_only[5], hook_only[6], hook_only[7], config.virality.strategy_weights["motivational_monologue"])

    assert complete_score.viral_potential_score > hook_score.viral_potential_score
    assert hook_score.penalties["missing_payoff"].contribution > 0
    assert complete_score.confidence.overall.score != complete_score.viral_potential_score


def test_ranking_respects_eligibility_floor_and_uses_stable_tie_break():
    candidate, story, transcript, audio, _profile, _retention, _publishability, _eligibility = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ], boundary=False)
    config = AppConfig()
    assessments = build_virality_assessments(
        [candidate], {"story_units": [story.to_dict()]}, transcript, audio, {},
        {"strategy_id": "motivational_monologue", "analysis_confidence": 0.9}, config.virality,
    )
    base = ScoredCandidate(candidate, "title", "hook", "summary", 99, 99, 99, 99, 99, 0, None, True)
    ranking = apply_virality_ranking([base], assessments, config.virality, {"strategy_id": "motivational_monologue", "analysis_confidence": 0.9})
    ranked = ranking["candidates"][0]

    assert ranked["score"] >= 0
    assert ranked["virality"]["eligibility"]["status"] == "rejected"
    assert ranked["virality"]["selection_eligible"] is False
    assert ranked["selected"] is False
    assert ranked["rejection_reason"] == "semantic_boundary_violation"


def test_profile_weights_apply_only_after_hard_gates_and_cannot_raise_reject() -> None:
    candidate, story, transcript, audio, _profile, _retention, _publishability, _eligibility = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ], boundary=False)
    config = AppConfig()

    def rank(effective_profile: dict) -> dict:
        content_profile = {
            "effective_profile": {**effective_profile, "traits": ["high_pacing", "visual_led"]},
            "analysis_confidence": 0.9,
        }
        assessments = build_virality_assessments(
            [candidate], {"story_units": [story.to_dict()]}, transcript, audio, {},
            content_profile, config.virality,
        )
        base = ScoredCandidate(candidate, "title", "hook", "summary", 99, 99, 99, 99, 99, 0, None, True)
        feasibility = {"candidates": [{"candidate_id": candidate.id, "status": "VIABLE"}]}
        return apply_virality_ranking(
            [base], assessments, config.virality, content_profile, feasibility,
        )["candidates"][0]

    motivational = rank({"format": "talking_head", "editorial_mode": "motivational", "domain": "general"})
    educational = rank({"format": "screen_demo", "editorial_mode": "explanatory", "domain": "education"})

    assert motivational["score"] == educational["score"]
    assert motivational["virality"]["ranking_sort_score"] == educational["virality"]["ranking_sort_score"]
    assert motivational["virality"]["profile_weighting_applied"] is False
    assert educational["virality"]["profile_weighting_applied"] is False
    assert motivational["virality"]["selection_eligible"] is False
    assert educational["virality"]["selection_eligible"] is False


def test_structured_profile_changes_only_downstream_rank_for_eligible_candidate() -> None:
    candidate, story, transcript, audio, _profile, _retention, _publishability, _eligibility = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ])
    config = AppConfig()

    def rank(effective_profile: dict) -> dict:
        content_profile = {
            "effective_profile": {**effective_profile, "traits": []},
            "analysis_confidence": 0.9,
        }
        assessments = build_virality_assessments(
            [candidate], {"story_units": [story.to_dict()]}, transcript, audio, {},
            content_profile, config.virality,
        )
        base = ScoredCandidate(candidate, "title", "hook", "summary", 99, 99, 99, 99, 99, 0, None, True)
        feasibility = {"candidates": [{"candidate_id": candidate.id, "status": "VIABLE"}]}
        return apply_virality_ranking(
            [base], assessments, config.virality, content_profile, feasibility,
        )["candidates"][0]

    motivational = rank({"format": "talking_head", "editorial_mode": "motivational", "domain": "general"})
    educational = rank({"format": "screen_demo", "editorial_mode": "explanatory", "domain": "education"})

    assert motivational["virality"]["profile_weighting_applied"] is True
    assert educational["virality"]["profile_weighting_applied"] is True
    assert motivational["virality"]["profile_strategy_id"] == "motivational_monologue"
    assert educational["virality"]["profile_strategy_id"] == "generic_educational"
    assert motivational["score"] == educational["score"]
    assert motivational["virality"]["ranking_sort_score"] != educational["virality"]["ranking_sort_score"]


def test_phase6_eligibility_reject_never_receives_profile_weighting() -> None:
    candidate, story, transcript, audio, _profile, _retention, _publishability, _eligibility = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ])
    candidate.eligibility_decision = EligibilityDecision(
        schema_version="6D.1", config_version="test",
        state=EligibilityState.ASSESSED, eligible=False,
    )
    config = AppConfig()
    feasibility = {"candidates": [{"candidate_id": candidate.id, "status": "VIABLE"}]}

    def rank(effective_profile: dict) -> dict:
        content_profile = {"effective_profile": {**effective_profile, "traits": []}}
        assessments = build_virality_assessments(
            [candidate], {"story_units": [story.to_dict()]}, transcript, audio, {},
            content_profile, config.virality,
        )
        base = ScoredCandidate(candidate, "title", "hook", "summary", 99, 99, 99, 99, 99, 0, None, True)
        return apply_virality_ranking(
            [base], assessments, config.virality, content_profile, feasibility,
        )["candidates"][0]

    motivational = rank({"format": "talking_head", "editorial_mode": "motivational", "domain": "general"})
    educational = rank({"format": "screen_demo", "editorial_mode": "explanatory", "domain": "education"})

    assert motivational["virality"]["profile_weighting_applied"] is False
    assert educational["virality"]["profile_weighting_applied"] is False
    assert motivational["virality"]["profile_weighting_gates"]["phase6_eligibility_passed"] is False
    assert motivational["virality"]["ranking_sort_score"] == educational["virality"]["ranking_sort_score"]


def test_guaranteed_blocked_feasibility_never_receives_profile_weighting() -> None:
    candidate, story, transcript, audio, _profile, _retention, _publishability, _eligibility = _diagnostics([
        "The only reason teams fail is fear.", "Choose one difficult action.", "Therefore, begin now!",
    ])
    config = AppConfig()
    feasibility = {"candidates": [{"candidate_id": candidate.id, "status": "GUARANTEED_BLOCKED"}]}

    def rank(effective_profile: dict) -> dict:
        content_profile = {"effective_profile": {**effective_profile, "traits": []}}
        assessments = build_virality_assessments(
            [candidate], {"story_units": [story.to_dict()]}, transcript, audio, {},
            content_profile, config.virality,
        )
        base = ScoredCandidate(candidate, "title", "hook", "summary", 99, 99, 99, 99, 99, 0, None, True)
        return apply_virality_ranking(
            [base], assessments, config.virality, content_profile, feasibility,
        )["candidates"][0]

    motivational = rank({"format": "talking_head", "editorial_mode": "motivational", "domain": "general"})
    educational = rank({"format": "screen_demo", "editorial_mode": "explanatory", "domain": "education"})

    assert motivational["virality"]["profile_weighting_applied"] is False
    assert educational["virality"]["profile_weighting_applied"] is False
    assert motivational["virality"]["profile_weighting_gates"]["feasibility_status"] == "GUARANTEED_BLOCKED"
    assert motivational["virality"]["ranking_sort_score"] == educational["virality"]["ranking_sort_score"]
