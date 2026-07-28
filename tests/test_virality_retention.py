from __future__ import annotations

from app.content_understanding import StoryUnit
from app.models import Candidate
from app.virality import (
    EligibilityAssessment,
    EstimatedRetentionProfile,
    PublishabilityAssessment,
    assess_candidate_eligibility,
    build_estimated_retention_profile,
    build_publishability_assessment,
    build_virality_feature_profile,
)


def _story(payoff: str = "Therefore, begin now!") -> StoryUnit:
    return StoryUnit(
        story_unit_id="story-1", chapter_id="chapter-1", start=0.0, end=30.0, duration=30.0,
        transcript_segment_ids=[0, 1, 2, 3], title="Clear story", core_idea="Act before certainty arrives",
        hook_seed="A bold claim", setup="A real problem", development="A useful action", payoff=payoff,
        ending=payoff, emotional_arc="rising", dominant_emotion="determination", speaker_context="single_speaker",
        required_previous_context="", required_next_context="", standalone_score=0.84, completeness_score=0.88,
        clarity_score=0.84, context_dependency_score=0.10, information_density=0.74, repetition_score=0.06,
        transformation_potential=0.78, publishability_precheck=True, content_signature={"theme_tags": ["test"]},
        confidence=0.9, evidence={"segments": [0, 1, 2, 3]},
    )


def _run(
    parts: list[dict], *, story_payoff: str = "Therefore, begin now!", boundary_eligible: bool = True,
    visual: dict | None = None,
):
    duration = 30.0
    text = " ".join(str(item["text"]) for item in parts)
    transcript_parts = []
    for index, item in enumerate(parts):
        start = float(item.get("start", index * duration / len(parts)))
        end = float(item.get("end", (index + 1) * duration / len(parts)))
        transcript_parts.append({
            "id": index, "start": start, "end": end, "text": str(item["text"]),
            "transcript_confidence": float(item.get("confidence", 0.92)),
            "speech_density": float(item.get("speech_density", 0.72)),
            "filler_word_ratio": float(item.get("filler", 0.0)),
            "repetition_score": float(item.get("repetition", 0.0)),
            "context_dependency_score": float(item.get("context", 0.0)),
            "exclamation_count": str(item["text"]).count("!"),
        })
    candidate = Candidate(
        "candidate-1", 0.0, duration, text, transcript_segment_ids=list(range(len(parts))), story_unit_id="story-1",
        core_idea="Act before certainty arrives", boundary_diagnostics={"eligible": boundary_eligible, "overall_boundary_score": 0.9 if boundary_eligible else 0.1},
        feature_vector={"words_per_second": 2.5, "speech_density": 0.72, "filler_word_ratio": 0.02, "repetition_score": 0.06},
    )
    story = _story(story_payoff)
    profile = build_virality_feature_profile(
        candidate, {"story_units": [story.to_dict()]}, {"segments": transcript_parts},
        {"energy_frames": [{"time": 0.0, "normalized_loudness": 0.3}, {"time": 10.0, "normalized_loudness": 0.5}, {"time": 22.0, "normalized_loudness": 0.85}], "silence_intervals": []},
        content_strategy="motivational_monologue",
    )
    retention = build_estimated_retention_profile(candidate, profile, {"segments": transcript_parts}, {"energy_frames": [{"time": 0.0, "normalized_loudness": 0.3}, {"time": 10.0, "normalized_loudness": 0.5}, {"time": 22.0, "normalized_loudness": 0.85}], "silence_intervals": []})
    publishability = build_publishability_assessment(candidate, profile, retention, visual)
    eligibility = assess_candidate_eligibility(candidate, profile, retention, publishability)
    return candidate, profile, retention, publishability, eligibility


def test_fast_strong_opening_improves_early_retention_without_claiming_view_percentage():
    strong = _run([
        {"text": "The only reason teams fail is fear of the first step."},
        {"text": "Choose one concrete action today."},
        {"text": "Therefore, begin now!"},
    ])[2]
    slow = _run([
        {"text": "Hello everyone, today we will discuss success."},
        {"text": "Choose one concrete action today."},
        {"text": "Therefore, begin now!"},
    ])[2]

    assert strong.early_retention.score > slow.early_retention.score
    assert strong.relative_level(strong.early_retention.score) in {"low", "medium", "high", "very_high"}
    assert "процент" in strong.opening_retention.explanation


def test_long_mid_filler_becomes_dead_zone_inside_candidate_without_mutating_source():
    candidate, _profile, retention, _publishability, _eligibility = _run([
        {"text": "What if fear is the real cost?", "start": 0, "end": 5},
        {"text": "um um um um um um um um um um", "start": 5, "end": 18, "filler": 0.86, "repetition": 0.92, "speech_density": 0.12},
        {"text": "Because action creates the answer. Therefore, begin now!", "start": 18, "end": 30},
    ])
    original_text = candidate.text

    assert retention.dead_zone_ranges
    assert all(candidate.start <= zone.start < zone.end <= candidate.end for zone in retention.dead_zone_ranges)
    assert all(zone.removable_in_future is True for zone in retention.dead_zone_ranges)
    assert candidate.text == original_text
    assert retention.estimated_drop_points


def test_late_payoff_lowers_mid_retention_and_strong_ending_supports_completion():
    early = _run([
        {"text": "What is the fastest way to move?", "start": 0, "end": 5},
        {"text": "Because action creates confidence. Therefore, begin now!", "start": 5, "end": 13},
        {"text": "Use that lesson in the next decision.", "start": 13, "end": 30},
    ])[2]
    late = _run([
        {"text": "What is the fastest way to move?", "start": 0, "end": 5},
        {"text": "Consider several ordinary examples and keep waiting.", "start": 5, "end": 24},
        {"text": "Because action creates confidence. Therefore, begin now!", "start": 24, "end": 30},
    ])[2]

    assert late.mid_retention.score < early.mid_retention.score
    assert late.completion_potential.score > 0.35
    restored = EstimatedRetentionProfile.from_dict(late.to_dict())
    assert restored.to_dict() == late.to_dict()


def test_publishability_and_eligibility_are_separate_and_deterministic():
    complete = _run([
        {"text": "The only reason teams fail is fear."},
        {"text": "Choose one difficult action."},
        {"text": "Therefore, begin now!"},
    ])
    weak_opening = _run([
        {"text": "Hello everyone, today we will discuss success."},
        {"text": "Choose one difficult action."},
        {"text": "Therefore, begin now!"},
    ])

    assert complete[3].level == "ready"
    assert complete[4].status == "publishable_now"
    assert weak_opening[4].status in {"needs_reconstruction", "publishable_with_minor_adjustment"}
    assert EligibilityAssessment.from_dict(complete[4].to_dict()).to_dict() == complete[4].to_dict()
    assert PublishabilityAssessment.from_dict(complete[3].to_dict()).to_dict() == complete[3].to_dict()


def test_critical_boundary_or_incomplete_story_cannot_be_publishable_now_but_visual_warning_is_soft():
    boundary_failure = _run([
        {"text": "The only reason teams fail is fear."},
        {"text": "Choose one difficult action."},
        {"text": "Therefore, begin now!"},
    ], boundary_eligible=False)
    incomplete = _run([
        {"text": "The first risk is fear."},
        {"text": "The team keeps waiting for certainty."},
        {"text": "The meeting ends in silence."},
    ], story_payoff="")
    warning_only = _run([
        {"text": "The only reason teams fail is fear."},
        {"text": "Choose one difficult action."},
        {"text": "Therefore, begin now!"},
    ], visual={"composition_quality_status": "passed_with_warning", "warnings": ["safe layout"]})

    assert boundary_failure[4].status == "rejected"
    assert incomplete[4].status == "rejected"
    assert warning_only[3].critical_failures == []
    assert warning_only[4].status != "rejected"
