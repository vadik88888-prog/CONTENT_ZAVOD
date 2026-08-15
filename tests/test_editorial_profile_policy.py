from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.analysis_artifact import AnalysisArtifact
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
    effective["resolution"] = {
        "format": "manual_override",
        "editorial_mode": "manual_override",
        "domain": "manual_override",
        "traits": "manual_override",
    }
    return {
        "content_profile_preset": profile_id,
        "effective_profile": effective,
        "detected_profile": {"format": {"value": "unknown", "confidence": 0.2}},
        "manual_override": {**preset.profile(), "provenance": "user", "revision_id": "revision-1"},
    }


def test_registry_is_versioned_and_covers_auto_resolvable_15_profiles() -> None:
    assert EDITORIAL_PROFILE_POLICY_VERSION == "editorial-profile-policy.1"
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
