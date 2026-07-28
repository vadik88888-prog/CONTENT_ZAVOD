from __future__ import annotations

from app.content_understanding import StoryUnit
from app.models import Candidate
from app.virality import FEATURE_NAMES, VIRALITY_SCHEMA_VERSION, ViralityFeatureProfile, build_virality_feature_profile


def _story(*, payoff: str = "", standalone: float = 0.82, dependency: float = 0.12) -> StoryUnit:
    return StoryUnit(
        story_unit_id="story-1", chapter_id="chapter-1", start=0.0, end=24.0, duration=24.0,
        transcript_segment_ids=[0, 1, 2], title="A test story", core_idea="A clear useful conclusion",
        hook_seed="A meaningful opening", setup="A question creates tension", development="The idea develops",
        payoff=payoff, ending=payoff, emotional_arc="rising", dominant_emotion="determination",
        speaker_context="single_speaker", required_previous_context="", required_next_context="",
        standalone_score=standalone, completeness_score=0.86, clarity_score=0.83,
        context_dependency_score=dependency, information_density=0.72, repetition_score=0.08,
        transformation_potential=0.76, publishability_precheck=True,
        content_signature={"theme_tags": ["test"]}, confidence=0.88, evidence={"segments": [0, 1, 2]},
    )


def _candidate(text: str, *, ids: list[int] | None = None) -> Candidate:
    return Candidate(
        "candidate-1", 0.0, 24.0, text, transcript_segment_ids=ids or [0, 1, 2],
        story_unit_id="story-1", core_idea="A clear useful conclusion",
        feature_vector={
            "words_per_second": 2.5, "speech_density": 0.72, "repetition_score": 0.08,
            "filler_word_ratio": 0.03, "completeness_score": 86, "context_dependency_score": 12,
        },
        boundary_diagnostics={"eligible": True},
    )


def _transcript(*parts: str, confidence: float | None = 0.92) -> dict:
    length = 24.0 / max(1, len(parts))
    return {
        "segments": [
            {
                "id": index, "start": round(index * length, 3), "end": round((index + 1) * length, 3),
                "text": text, "transcript_confidence": confidence, "exclamation_count": text.count("!"),
            }
            for index, text in enumerate(parts)
        ]
    }


def _audio() -> dict:
    return {
        "energy_frames": [
            {"time": 0.0, "normalized_loudness": 0.28},
            {"time": 8.0, "normalized_loudness": 0.46},
            {"time": 16.0, "normalized_loudness": 0.88},
            {"time": 23.0, "normalized_loudness": 0.74},
        ],
        "silence_intervals": [],
    }


def _profile(text: str, *parts: str, story: StoryUnit | None = None, audio: dict | None = None):
    current_story = story or _story()
    return build_virality_feature_profile(
        _candidate(text), {"story_units": [current_story.to_dict()]}, _transcript(*parts),
        _audio() if audio is None else audio, content_strategy="motivational",
    )


def test_profile_has_complete_grounded_contract_and_round_trips_to_dict():
    profile = _profile(
        "The only reason teams fail is fear of the first step. Choose one action today. Therefore, begin now!",
        "The only reason teams fail is fear of the first step.", "Choose one action today.", "Therefore, begin now!",
        story=_story(payoff="Therefore, begin now!"),
    )

    assert profile.schema_version == VIRALITY_SCHEMA_VERSION
    assert set(profile.features) == set(FEATURE_NAMES)
    assert profile.hook_assessment.hook_type == "bold_claim"
    assert profile.hook_assessment.hook_strength.score >= 0.5
    assert profile.payoff_assessment.payoff_present is True
    serialized = profile.to_dict()
    restored = ViralityFeatureProfile.from_dict(serialized)
    assert restored.to_dict() == serialized
    assert serialized["features"]["hook_strength"]["evidence"][0]["segment_ids"] == [0]
    assert serialized["features"]["hook_strength"]["evidence"][0]["feature_name"] == "hook_strength"
    for feature in profile.features.values():
        for evidence in feature.evidence:
            assert set(evidence.segment_ids) <= {0, 1, 2}


def test_question_only_earns_strong_curiosity_when_answered_inside_candidate():
    answered = _profile(
        "Why do capable teams fail? Because they wait for certainty. The answer is to act before confidence arrives.",
        "Why do capable teams fail?", "Because they wait for certainty.", "The answer is to act before confidence arrives.",
    )
    unanswered = _profile(
        "Why do capable teams fail? Their next meeting begins tomorrow. Everyone knows the feeling.",
        "Why do capable teams fail?", "Their next meeting begins tomorrow.", "Everyone knows the feeling.",
        story=_story(payoff=""),
    )

    assert answered.hook_assessment.curiosity_opened is True
    assert answered.hook_assessment.curiosity_resolved is True
    assert answered.hook_assessment.resolution_timestamp is not None
    assert unanswered.hook_assessment.curiosity_resolved is False
    assert unanswered.hook_assessment.unresolved_curiosity_penalty.score > 0.6
    assert answered.features["curiosity_gap"].score > unanswered.features["curiosity_gap"].score


def test_greeting_and_contextual_openings_receive_slow_start_and_context_penalties():
    greeting = _profile(
        "Hello everyone, today we will discuss success. Start with one specific task. Therefore, act now.",
        "Hello everyone, today we will discuss success.", "Start with one specific task.", "Therefore, act now.",
    )
    contextual = _profile(
        "And that is why you cannot wait. Choose the difficult action. Therefore, start today.",
        "And that is why you cannot wait.", "Choose the difficult action.", "Therefore, start today.",
    )

    assert greeting.hook_assessment.slow_start_penalty.score > 0.6
    assert greeting.hook_assessment.hook_strength.score < 0.45
    assert contextual.hook_assessment.context_dependency.score > 0.7
    assert contextual.features["confusion_penalty"].score > 0.5


def test_emotional_arc_conflict_and_payoff_are_distinct_and_grounded():
    resolved = _profile(
        "Fear makes the risk feel impossible. We fight the problem with one choice. Therefore, we win!",
        "Fear makes the risk feel impossible.", "We fight the problem with one choice.", "Therefore, we win!",
    )
    aggressive_words_only = _profile(
        "I hate pointless noise. The room is loud today. We will meet again tomorrow.",
        "I hate pointless noise.", "The room is loud today.", "We will meet again tomorrow.",
        story=_story(payoff=""),
    )

    assert 0 <= resolved.emotional_arc.peak_timestamp <= 24
    assert resolved.emotional_arc.escalation_strength.score > 0.05
    assert resolved.conflict_assessment.conflict_resolution.score > 0.6
    assert resolved.payoff_assessment.payoff_strength.score > 0.5
    assert aggressive_words_only.conflict_assessment.conflict_strength.score < 0.35
    assert aggressive_words_only.payoff_assessment.payoff_present is False


def test_missing_audio_reduces_confidence_without_disabling_deterministic_profile():
    with_audio = _profile(
        "What if the risk is the point? Because action creates the answer. Therefore, start.",
        "What if the risk is the point?", "Because action creates the answer.", "Therefore, start.",
    )
    without_audio = _profile(
        "What if the risk is the point? Because action creates the answer. Therefore, start.",
        "What if the risk is the point?", "Because action creates the answer.", "Therefore, start.", audio={"energy_frames": [], "silence_intervals": []},
    )

    assert without_audio.analysis_mode == "deterministic"
    assert without_audio.features["analysis_confidence"].score < with_audio.features["analysis_confidence"].score
    assert without_audio.features["speech_energy"].confidence < with_audio.features["speech_energy"].confidence
