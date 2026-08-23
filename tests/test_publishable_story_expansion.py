from __future__ import annotations

import pytest

from app.candidate_quality import (
    CANDIDATE_QUALITY_SCHEMA_VERSION,
    EligibilityDecision,
    EligibilityState,
)
from app.config import AppConfig
from app.content_understanding import (
    SemanticBoundaryEngine,
    build_global_content_map,
    build_semantic_candidate,
    build_video_content_profile,
    ensure_candidate_boundary_decision,
    expand_publishable_story_candidates,
)
from app.models import ScoredCandidate
from app.pipeline import Pipeline
from app.semantic_extraction import build_source_context
from app.transcript_features import analyse_transcript
from app.utils import read_json


def _story_fixture() -> tuple[dict, dict, dict, AppConfig]:
    config = AppConfig()
    transcript = {
        "source_id": "source-story-expansion",
        "language": "en",
        "duration": 270.0,
        "segments": [
            {"id": 54, "start": 220.56, "end": 232.92, "text": "I need to engage now. Let's do it."},
            {"id": 58, "start": 235.04, "end": 249.19, "text": "He is standing there. Where is he? Keep moving. I need to hide."},
            {"id": 62, "start": 250.49, "end": 264.42, "text": "What flew by? I do not think that is all. Look right."},
        ],
    }
    features = analyse_transcript(transcript, config.transcript_features)
    # Story Expansion feasibility reuses the persisted A-1 evidence.  Keep the
    # baseline fixture intelligible; dedicated cases below exercise warnings
    # and material low-confidence spans.
    for segment in features["segments"]:
        segment["transcript_confidence"] = 0.9
    source = {"id": transcript["source_id"], "display_name": "story-expansion.mp4"}
    metadata = {"duration": transcript["duration"]}
    profile = build_video_content_profile(
        source, metadata, transcript, features, {"energy_frames": []}, {"boundaries": []},
        {"status": "skipped", "subject_keyframes": []}, config,
    )
    content_map = build_global_content_map(
        source, metadata, transcript, features, {"energy_frames": []}, {"boundaries": []},
        {"status": "skipped", "subject_keyframes": []}, profile, config,
    )
    return content_map, transcript, features, config


def _selected_middle_candidate(content_map: dict, transcript: dict, features: dict, config: AppConfig) -> ScoredCandidate:
    middle = content_map["story_units"][1]
    from app.content_understanding import StoryUnit

    candidate = build_semantic_candidate(
        [StoryUnit.from_dict(middle)], transcript, features, {"boundaries": []},
        SemanticBoundaryEngine(config.content_understanding),
        candidate_id="candidate-chapter-010-story-001",
    )
    candidate.eligibility_decision = EligibilityDecision(
        schema_version=CANDIDATE_QUALITY_SCHEMA_VERSION,
        config_version="test",
        state=EligibilityState.ASSESSED,
        eligible=True,
    )
    return ScoredCandidate(
        candidate=candidate, title="Gameplay moment", hook="", summary="", score=90,
        hook_score=80, completeness_score=80, emotional_score=60, clarity_score=80,
        context_dependency_score=10, rejection_reason=None, selected=True,
    )


def test_post_selection_expansion_adds_grounded_setup_and_reaction_without_reranking() -> None:
    content_map, transcript, features, config = _story_fixture()
    selected = _selected_middle_candidate(content_map, transcript, features, config)

    assert (selected.candidate.start, selected.candidate.end) == pytest.approx((234.79, 249.84), abs=0.001)
    original_score = selected.score
    reports = expand_publishable_story_candidates(
        [selected], content_map, transcript, features, {"boundaries": []}, config,
    )

    candidate = selected.candidate
    assert (candidate.start, candidate.end) == pytest.approx((220.31, 265.07), abs=0.001)
    assert candidate.story_unit_ids == [
        content_map["story_units"][0]["story_unit_id"],
        content_map["story_units"][1]["story_unit_id"],
        content_map["story_units"][2]["story_unit_id"],
    ]
    assert selected.score == original_score
    expansion = reports[0]
    assert expansion["decision"] == "expanded"
    assert expansion["original_range"] == {"start_seconds": 234.79, "end_seconds": 249.84}
    assert expansion["expanded_range"] == {"start_seconds": 220.31, "end_seconds": 265.07}
    assert [item["contribution"] for item in expansion["added_story_evidence"]] == [
        "setup", "result_or_reaction",
    ]
    assert expansion["brain_reused"] is True
    assert expansion["vision_reused"] is True
    boundary = candidate.boundary_diagnostics["boundary_decision"]
    assert boundary["candidate_id"] == candidate.id
    assert boundary["rough_range"] == {"start_seconds": 234.79, "end_seconds": 249.84}
    assert boundary["refined_range"] == {"start_seconds": 220.31, "end_seconds": 265.07}
    validated = ensure_candidate_boundary_decision(candidate)
    assert validated is not None
    assert validated["candidate_id"] == candidate.id
    assert validated["refined_range"] == boundary["refined_range"]


def test_self_contained_short_candidate_is_not_padded_by_adjacent_story_units() -> None:
    content_map, transcript, features, config = _story_fixture()
    content_map["story_units"][1]["payoff"] = "The result worked."
    content_map["story_units"][2].update({
        "setup": "", "development": "A separate neutral topic.",
        "payoff": "", "ending": "A separate neutral topic.",
    })
    selected = _selected_middle_candidate(content_map, transcript, features, config)
    original_range = (selected.candidate.start, selected.candidate.end)

    reports = expand_publishable_story_candidates(
        [selected], content_map, transcript, features, {"boundaries": []}, config,
    )

    assert (selected.candidate.start, selected.candidate.end) == pytest.approx(original_range, abs=0.001)
    assert reports[0]["decision"] == "not_expanded"
    assert reports[0]["reason"] == "no_additional_grounded_story_arc"


@pytest.mark.parametrize(
    ("profile", "grounded_ending"),
    [
        ("food", "The tasting reaction is wow, it worked."),
        ("gameplay", "The player reacts: wow, the round is won."),
        ("podcast", "The guest reacts: that was the result."),
    ],
)
def test_natural_story_ending_extends_publishable_candidates_without_reranking(
    profile: str, grounded_ending: str,
) -> None:
    """Profiles are regression labels only: the policy itself is generic."""

    content_map, transcript, features, config = _story_fixture()
    content_map["story_units"][1].update({
        "payoff": "The main point is already complete.",
        "ending": "The main point is already complete.",
    })
    content_map["story_units"][2].update({
        "setup": "", "development": grounded_ending,
        "payoff": "", "ending": grounded_ending,
    })
    selected = _selected_middle_candidate(content_map, transcript, features, config)
    original_score = selected.score

    reports = expand_publishable_story_candidates(
        [selected], content_map, transcript, features, {"boundaries": []}, config,
    )

    candidate = selected.candidate
    expansion = reports[0]
    assert profile in {"food", "gameplay", "podcast"}
    assert selected.score == original_score
    assert candidate.end > 249.84
    assert candidate.duration <= config.max_clip_duration
    assert expansion["decision"] == "expanded"
    assert expansion["reason"] == "a1_feasible_grounded_natural_story_ending_added_without_reranking"
    natural = expansion["natural_ending"]
    assert natural["requires_extension"] is True
    assert natural["grounded"] is True
    assert natural["reason"] == "grounded_adjacent_continuation_resolved_by_existing_boundary_owner"
    assert natural["following_story_unit_id"] == content_map["story_units"][2]["story_unit_id"]
    assert "reaction" in natural["following_story_roles"]
    assert natural["decision"] == "extended"
    # A-1 was evaluated before the boundary is accepted; A-2 is then built
    # from the exact expanded boundary rather than a second, profile-specific
    # ending path.
    assert "a1_speech_clarity" in expansion
    assert ensure_candidate_boundary_decision(candidate) is not None
    context = build_source_context(
        {"id": transcript["source_id"], "path": "story-expansion.mp4"},
        {"duration": transcript["duration"]}, candidate, transcript, features,
        {"energy_frames": []}, {"boundaries": []}, config.transformation,
    )
    assert context.continuity_decision
    assert context.continuity_decision["candidate_id"] == candidate.id


def test_natural_story_ending_never_forces_food_past_maximum_duration() -> None:
    content_map, transcript, features, config = _story_fixture()
    content_map["story_units"][1].update({
        "payoff": "The tasting verdict is complete.",
        "ending": "The tasting verdict is complete.",
    })
    content_map["story_units"][2].update({
        "setup": "", "development": "Wow, the diner reacts to the tasting.",
        "payoff": "", "ending": "Wow, the diner reacts to the tasting.",
    })
    selected = _selected_middle_candidate(content_map, transcript, features, config)
    original_range = (selected.candidate.start, selected.candidate.end)
    config.max_clip_duration = selected.candidate.duration + 1.0

    reports = expand_publishable_story_candidates(
        [selected], content_map, transcript, features, {"boundaries": []}, config,
    )

    assert (selected.candidate.start, selected.candidate.end) == pytest.approx(original_range, abs=0.001)
    assert reports[0]["decision"] == "not_expanded"
    assert reports[0]["natural_ending"]["reason"] == "grounded_continuation_exceeds_maximum_duration"


def test_expansion_uses_a1_feasible_result_side_when_setup_is_materially_unclear() -> None:
    content_map, transcript, features, config = _story_fixture()
    for segment in features["segments"]:
        segment["transcript_confidence"] = 0.9
    features["segments"][0]["transcript_confidence"] = 0.323
    selected = _selected_middle_candidate(content_map, transcript, features, config)

    reports = expand_publishable_story_candidates(
        [selected], content_map, transcript, features, {"boundaries": []}, config,
    )

    candidate = selected.candidate
    assert (candidate.start, candidate.end) == pytest.approx((234.79, 265.07), abs=0.001)
    assert candidate.story_unit_ids == [
        content_map["story_units"][1]["story_unit_id"],
        content_map["story_units"][2]["story_unit_id"],
    ]
    expansion = reports[0]
    assert expansion["a1_speech_clarity"] is None
    rejected = next(option for option in expansion["rejected_adjacent_options"] if option["reason"] == "a1_speech_clarity_material")
    assert "LOW_CONFIDENCE_DURATION_MATERIAL" in rejected["a1_speech_clarity"]["materiality_reasons"]


def test_expansion_safely_falls_back_when_no_adjacent_arc_passes_a1() -> None:
    content_map, transcript, features, config = _story_fixture()
    features["segments"][0]["transcript_confidence"] = 0.323
    features["segments"][2]["transcript_confidence"] = 0.323
    selected = _selected_middle_candidate(content_map, transcript, features, config)

    reports = expand_publishable_story_candidates(
        [selected], content_map, transcript, features, {"boundaries": []}, config,
    )

    assert (selected.candidate.start, selected.candidate.end) == pytest.approx((234.79, 249.84), abs=0.001)
    assert reports[0]["decision"] == "not_expanded"
    assert reports[0]["reason"] == "no_a1_feasible_grounded_story_arc"
    assert reports[0]["rejected_adjacent_options"]


def test_final_selection_persists_post_selection_story_expansion(tmp_path) -> None:
    content_map, transcript, features, config = _story_fixture()
    config.score_threshold = 0
    config.ai_reranking.final_clip_count = 1
    selected = _selected_middle_candidate(content_map, transcript, features, config)
    destination = tmp_path / "final_selection.json"

    data = Pipeline(tmp_path, config, mock_ai=True)._final_selection(
        [selected], destination, content_map,
        transcript=transcript, transcript_features=features, scenes={"boundaries": []},
    )

    persisted = read_json(destination, {})
    expansion = data["publishable_story_expansion"]["candidates"][0]
    assert expansion["decision"] == "expanded"
    assert persisted["publishable_story_expansion"] == data["publishable_story_expansion"]
    candidate = data["candidates"][0]
    assert (candidate["start"], candidate["end"]) == pytest.approx((220.31, 265.07), abs=0.001)


def test_post_selection_expansion_persists_for_manual_draft_candidate_without_reranking(tmp_path) -> None:
    content_map, transcript, features, config = _story_fixture()
    config.score_threshold = 0
    config.ai_reranking.final_clip_count = 1
    draftable = _selected_middle_candidate(content_map, transcript, features, config)
    draftable.selected = False
    destination = tmp_path / "final_selection.json"

    data = Pipeline(tmp_path, config, mock_ai=True)._final_selection(
        [draftable], destination, content_map,
        transcript=transcript, transcript_features=features, scenes={"boundaries": []},
    )

    assert data["selected_ids"] == []
    persisted = read_json(destination, {})
    candidate = persisted["candidates"][0]
    assert (candidate["start"], candidate["end"]) == pytest.approx((220.31, 265.07), abs=0.001)
    boundary = candidate["boundary_diagnostics"]["boundary_decision"]
    assert boundary["rough_range"] == {"start_seconds": 234.79, "end_seconds": 249.84}
    assert boundary["refined_range"] == {"start_seconds": 220.31, "end_seconds": 265.07}
    assert persisted["selected_ids"] == []
