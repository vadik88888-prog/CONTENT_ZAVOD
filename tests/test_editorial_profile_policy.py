from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from app.analysis_artifact import (
    AnalysisArtifact,
    _moments_review_candidate,
    candidate_is_draftable,
    candidate_review_payload,
)
from app.content_profile_taxonomy import CONTENT_PROFILE_PRESETS
from app.config import AppConfig
from app.editorial_profile_policy import (
    EDITORIAL_PROFILE_POLICIES,
    EDITORIAL_PROFILE_POLICY_VERSION,
    EditorialSurfacingState,
    evaluate_editorial_candidate,
    resolve_editorial_profile,
)
from app.models import Candidate, ScoredCandidate
from app.pipeline import Pipeline, StageTracker


def _eligibility(*reason_codes: str, boundary: dict | None = None) -> dict:
    evidence = []
    if boundary is not None:
        evidence.append({
            "code": "semantic_boundary",
            "state": "available",
            "source": "boundary_diagnostics",
            "details": boundary,
        })
    return {
        "schema_version": "6D.1",
        "config_version": "test",
        "state": "assessed",
        "eligible": not reason_codes,
        "reason_codes": list(reason_codes),
        "recoverable_issues": [],
        "required_boundary_actions": [],
        "evidence_refs": evidence,
    }


def _candidate(*reason_codes: str, boundary: dict | None = None, score: float = 60) -> dict:
    return {
        "candidate_id": "candidate-1",
        "start_seconds": 1.0,
        "end_seconds": 31.0,
        "score": score,
        "confidence": 0.8,
        "eligibility_decision": _eligibility(*reason_codes, boundary=boundary),
    }


def _profile(profile_id: str) -> dict:
    preset = CONTENT_PROFILE_PRESETS[profile_id]
    effective = preset.profile()
    effective["profile_id"] = profile_id
    effective["resolution"] = {
        "format": "manual_override",
        "editorial_mode": "manual_override",
        "domain": "manual_override",
        "traits": "manual_override",
    }
    return {
        "content_profile_preset": profile_id,
        "requested_mode": "manual",
        "requested_profile_id": profile_id,
        "effective_profile_reason": "manual_profile_selected",
        "detector_version": "test-detector.1",
        "effective_profile": effective,
        "detected_profile": {"format": {"value": "unknown", "confidence": 0.2}},
        "manual_override": {**preset.profile(), "provenance": "user", "revision_id": "revision-1"},
    }


def test_registry_is_versioned_and_covers_auto_resolvable_15_profiles() -> None:
    assert EDITORIAL_PROFILE_POLICY_VERSION == "editorial-profile-policy.2"
    assert tuple(EDITORIAL_PROFILE_POLICIES) == tuple(CONTENT_PROFILE_PRESETS)
    assert len(EDITORIAL_PROFILE_POLICIES) == 15


@pytest.mark.parametrize("profile_id", CONTENT_PROFILE_PRESETS)
def test_all_profiles_keep_editorial_weakness_selectable(profile_id: str) -> None:
    decision = evaluate_editorial_candidate(
        _candidate("NO_PAYOFF", "FALSE_HOOK_RISK", "DURATION_OUT_OF_RANGE", score=45),
        _profile(profile_id),
    )

    assert decision.profile_id == profile_id
    assert decision.surfacing_state in {
        EditorialSurfacingState.RECOMMENDED,
        EditorialSurfacingState.AVAILABLE,
    }
    assert decision.selectable is True
    assert not decision.hard_blockers
    assert {"NO_PAYOFF", "FALSE_HOOK_RISK", "DURATION_OUT_OF_RANGE"} <= set(decision.soft_issues)


def test_profile_changes_ranking_semantics_without_changing_permission() -> None:
    candidate = _candidate("NO_PAYOFF", "FALSE_HOOK_RISK", score=45)

    movie = evaluate_editorial_candidate(candidate, _profile("movie_series"))
    tutorial = evaluate_editorial_candidate(candidate, _profile("tutorial_education"))

    assert movie.selectable is tutorial.selectable is True
    assert movie.editorial_score > tutorial.editorial_score
    assert movie.surfacing_state is EditorialSurfacingState.RECOMMENDED
    assert tutorial.surfacing_state is EditorialSurfacingState.AVAILABLE


def test_semantic_incomplete_without_truncation_evidence_is_available() -> None:
    decision = evaluate_editorial_candidate(
        _candidate(
            "SEMANTIC_INCOMPLETE",
            boundary={
                "eligible": True,
                "word_integrity": True,
                "sentence_integrity": True,
                "semantic_completion": 0.95,
            },
        ),
        _profile("movie_series"),
    )

    assert decision.selectable is True
    assert "SEMANTIC_INCOMPLETE" in decision.soft_issues
    assert not decision.hard_blockers


def test_sparse_multimodal_content_downgrades_recommended_to_selectable_available() -> None:
    candidate = _candidate(score=100)
    candidate["candidate_score_v2"] = {
        "diagnostics": {"sparse_content": {"applies": True, "blocked": False}},
    }

    decision = evaluate_editorial_candidate(candidate, _profile("gameplay"))

    assert decision.surfacing_state is EditorialSurfacingState.AVAILABLE
    assert decision.selectable is True
    assert "SPARSE_MULTIMODAL_CONTENT" in decision.soft_issues
    assert not decision.hard_blockers
    assert decision.profile_provenance["sparse_multimodal_content"] == {
        "applies": True, "effect": "recommended_to_available", "selectable": True,
    }


@pytest.mark.parametrize(
    "boundary",
    [
        {"eligible": False, "word_integrity": True, "sentence_integrity": True, "semantic_completion": 0.2},
        {"eligible": False, "word_integrity": True, "sentence_integrity": False, "semantic_completion": 0.9},
    ],
)
def test_evidence_backed_truncation_remains_blocked(boundary: dict) -> None:
    decision = evaluate_editorial_candidate(
        _candidate("SEMANTIC_INCOMPLETE", boundary=boundary),
        _profile("movie_series"),
    )

    assert decision.surfacing_state is EditorialSurfacingState.BLOCKED
    assert decision.selectable is False
    assert "SEMANTIC_INCOMPLETE" in decision.hard_blockers


def test_moments_projection_keeps_production_hard_blocker_out_of_draft() -> None:
    source_candidate = {
        "id": "candidate-blocked",
        "start": 1.0,
        "end": 31.0,
        "score": 60.0,
        "confidence": 0.8,
        "eligibility_decision": _eligibility(
            "SEMANTIC_INCOMPLETE",
            "AUDIO_UNINTELLIGIBLE",
            boundary={
                "eligible": False,
                "word_integrity": True,
                "sentence_integrity": True,
                "semantic_completion": 0.2,
            },
        ),
    }

    review_payload = candidate_review_payload(source_candidate, set(), _profile("movie_series"))

    assert review_payload["surfacing_state"] == "AVAILABLE"
    assert review_payload["selectable"] is True
    assert review_payload["production_editorial_decision"]["selectable"] is False
    assert set(review_payload["production_editorial_decision"]["hard_blockers"]) == {
        "SEMANTIC_INCOMPLETE", "AUDIO_UNINTELLIGIBLE",
    }
    assert candidate_is_draftable(review_payload) is False


def test_auto_preserves_detected_effective_and_manual_provenance() -> None:
    profile = {
        "detected_content_type": "gameplay",
        "content_type_confidence": 0.737,
        "detected_profile": {"format": {"value": "gameplay", "confidence": 0.7}},
        "effective_profile": {"format": "gameplay", "editorial_mode": "narrative", "domain": "lifestyle", "traits": []},
        "manual_override": {"provenance": "none", "revision_id": None},
    }

    resolved = resolve_editorial_profile(profile, source={"filename": "Сериал Холод — 1 серия.webm"})

    assert resolved.profile_id == "movie_series"
    assert resolved.resolution == "auto_source_metadata_hint"
    assert resolved.detected_profile == profile["detected_profile"]
    assert resolved.effective_profile == profile["effective_profile"]
    assert resolved.manual_override == profile["manual_override"]


def test_editorial_policy_uses_effective_contract_over_conflicting_legacy_and_filename_hints() -> None:
    profile = _profile("gameplay")
    profile.update({
        "content_profile_preset": "movie_series",
        "detected_content_type": "movie_or_series",
        "requested_mode": "auto",
        "requested_profile_id": None,
        "effective_profile_reason": "auto_detected_profile_accepted",
    })
    profile["detected_profile"] = {
        "profile_id": {"value": "movie_series", "confidence": 0.8, "evidence": ["legacy:test"]},
    }

    resolved = resolve_editorial_profile(profile, source={"filename": "movie-series.mp4"})

    assert resolved.profile_id == "gameplay"
    assert resolved.resolution == "effective_profile_contract"
    assert resolved.effective_profile == profile["effective_profile"]


def test_real_95_analysis_is_immutable_and_has_only_evidence_backed_integrity_blocks() -> None:
    path = Path(
        "output/Сериал_Холод_1_серия_2026_Wink_I_Любовь_Аксёнова-0QWbay4LlUU-3b9aac2f10b4/"
        "runs/62b19eaea3004d4e80016463044f18f0/analysis.json"
    )
    if not path.is_file():
        pytest.skip("local real-series Analysis 1.1 is unavailable")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = AnalysisArtifact.read_verified(path)
    decisions = [
        evaluate_editorial_candidate(
            candidate,
            artifact.content_profile,
            score=float(candidate.get("score") or 0),
            confidence=float(candidate.get("confidence") or 0),
            source=artifact.source,
        )
        for candidate in artifact.candidates
    ]
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = {
        state: sum(decision.surfacing_state is state for decision in decisions)
        for state in EditorialSurfacingState
    }

    assert len(decisions) == 95
    assert counts[EditorialSurfacingState.RECOMMENDED] > 0
    assert counts[EditorialSurfacingState.AVAILABLE] > 0
    assert counts[EditorialSurfacingState.BLOCKED] == 6
    assert sum(decision.selectable for decision in decisions) == 89
    assert before == after
    assert all(decision.profile_id == "movie_series" for decision in decisions)


def test_current_gameplay_moments_keeps_all_quality_and_feasibility_risks_selectable() -> None:
    fixture = json.loads(
        Path("tests/fixtures/gameplay_moments_statuses.json").read_text(encoding="utf-8")
    )
    candidates = fixture["candidates"]
    profile = _profile("gameplay")
    canonical = [
        evaluate_editorial_candidate(
            candidate,
            profile,
            score=float(candidate["score"]),
            confidence=float(candidate["confidence"]),
        )
        for candidate in candidates
    ]
    before_states = [
        (
            "BLOCKED"
            if candidate.get("production_feasibility", {}).get("status")
            == "GUARANTEED_BLOCKED"
            else decision.surfacing_state.value
        )
        for candidate, decision in zip(candidates, canonical, strict=True)
    ]

    projected = []
    for candidate, ranking_decision in zip(candidates, canonical, strict=True):
        item = _moments_review_candidate(candidate)
        brain_recommended = (
            item.get("selected_by_recommendation") is True
            or ranking_decision.surfacing_state is EditorialSurfacingState.RECOMMENDED
        )
        decision = evaluate_editorial_candidate(
            item,
            profile,
            score=float(item["score"]),
            confidence=float(item["confidence"]),
            recommended=brain_recommended,
            production_feasibility=item.get("production_feasibility"),
        )
        item.update(
            editorial_decision=decision.to_dict(),
            surfacing_state=decision.surfacing_state.value,
            selectable=decision.selectable,
        )
        projected.append(item)

    assert Counter(before_states) == Counter(fixture["before_moments_counts"])
    assert Counter(item["surfacing_state"] for item in projected) == Counter(
        fixture["expected_moments_counts"]
    )
    assert [
        item["candidate_id"] for item in projected
        if item["surfacing_state"] == "RECOMMENDED"
    ] == ["candidate-chapter-011-story-001"]
    assert candidate_is_draftable(projected[0]) is True
    assert candidate_is_draftable(projected[1]) is False

    boundary_risk = projected[1]
    assert "SENTENCE_BOUNDARY_UNRECOVERABLE" in boundary_risk["editorial_decision"]["soft_issues"]
    assert boundary_risk["editorial_decision"]["profile_provenance"]["moments_projection"] == {
        "policy_version": "moments-surfacing.1",
        "permission_effect": "ranking_and_warning_only",
        "risk_codes": ["SENTENCE_BOUNDARY_UNRECOVERABLE"],
    }
    assert canonical[1].surfacing_state is EditorialSurfacingState.BLOCKED
    assert canonical[1].selectable is False

    feasibility = projected[0]["production_feasibility"]
    assert feasibility["status"] == "ADVISORY"
    assert feasibility["diagnostic_status"] == "GUARANTEED_BLOCKED"
    assert feasibility["reason_code"] == "CAPTION_CPS_INFEASIBLE"
    assert feasibility["blockers"] == [
        {"gate": "A-3", "reason_code": "CAPTION_CPS_INFEASIBLE"}
    ]

    source_candidate = {
        "id": candidates[0]["candidate_id"],
        "start": candidates[0]["start_seconds"],
        "end": candidates[0]["end_seconds"],
        "score": candidates[0]["score"],
        "confidence": candidates[0]["confidence"],
        "feature_vector": {"transcript_confidence": candidates[0]["confidence"]},
        "eligibility_decision": candidates[0]["eligibility_decision"],
        "selection_diagnostics": {
            "production_feasibility": candidates[0]["production_feasibility"]
        },
    }
    review_payload = candidate_review_payload(source_candidate, set(), profile)
    assert review_payload["surfacing_state"] == "RECOMMENDED"
    assert review_payload["selectable"] is True
    assert review_payload["production_editorial_decision"]["selectable"] is True
    assert review_payload["production_editorial_decision"]["hard_blockers"] == []
    assert review_payload["selected_by_recommendation"] is True
    assert review_payload["production_feasibility"]["status"] == "ADVISORY"
    assert review_payload["production_feasibility"]["diagnostic_status"] == "GUARANTEED_BLOCKED"
    assert "Caption CPS exceeds the configured production limit." in review_payload["warnings"]


def test_movie_editorial_weakness_passes_draft_preflight_without_brain_or_vision_rerun(
    tmp_path: Path,
) -> None:
    candidate = Candidate(
        "candidate-movie-scene",
        0.0,
        30.0,
        "Персонажи спорят, затем разговор заканчивается на естественной паузе.",
        feature_vector={"transcript_confidence": 0.9, "completeness_score": 75.0},
        semantic_evidence={"hook": "Спор", "payoff": "", "completeness_score": 75.0},
        boundary_diagnostics={
            "schema_version": "5A.1",
            "eligible": True,
            "word_integrity": True,
            "sentence_integrity": True,
            "semantic_completion": 0.95,
            "payoff_preserved": True,
            "requested_range": {"start": 0.0, "end": 30.0},
            "resolved_range": {"start": 0.0, "end": 30.0},
            "start_boundary": {
                "timestamp": 0.2,
                "transcript_segment_id": 1,
                "reason": "complete first word",
                "silence_before": 0.2,
            },
            "end_boundary": {
                "timestamp": 29.8,
                "transcript_segment_id": 2,
                "reason": "natural dialogue pause",
                "silence_after": 0.2,
            },
            "head_padding_seconds": 0.2,
            "tail_padding_seconds": 0.2,
            "continuation_risk": 0.1,
            "overall_boundary_score": 0.9,
        },
    )
    scored = ScoredCandidate(
        candidate,
        "Диалог",
        "Спор",
        "Связный фрагмент сцены",
        48,
        55,
        75,
        40,
        90,
        0,
        None,
        True,
        virality={"eligibility": {"status": "rejected", "critical_failures": ["incomplete_story"]}},
    )
    pipeline = Pipeline(tmp_path, AppConfig())
    selected, failures = pipeline._preflight_selected_candidates(
        [scored],
        {"status": "completed", "subject_keyframes": [{"timestamp": 1.0}]},
        StageTracker(tmp_path / "state.json"),
        content_profile=_profile("movie_series"),
        source={"filename": "Сериал — серия 1.mp4"},
    )

    assert selected == [scored]
    assert failures == {}
    assert candidate.eligibility_decision is not None and candidate.eligibility_decision.eligible is False
    assert candidate.editorial_decision is not None and candidate.editorial_decision.selectable is True
    assert "NO_PAYOFF" in candidate.editorial_decision.soft_issues
