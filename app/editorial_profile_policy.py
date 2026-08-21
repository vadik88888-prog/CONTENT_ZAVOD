from __future__ import annotations

"""Deterministic, profile-aware editorial surfacing policy.

The policy interprets persisted candidate evidence; it never creates evidence,
changes source boundaries, or owns production safety.  Editorial weakness is a
ranking input.  Only evidence-backed structural or technical failures remove
selectability.
"""

from dataclasses import dataclass, field
from enum import StrEnum
import re
from types import MappingProxyType
from typing import Any, Mapping

from app.content_profile_taxonomy import CONTENT_PROFILE_PRESETS, UNKNOWN_PROFILE_ID


EDITORIAL_PROFILE_POLICY_VERSION = "editorial-profile-policy.1"
MOMENTS_SURFACING_POLICY_VERSION = "moments-surfacing.1"


class EditorialSurfacingState(StrEnum):
    RECOMMENDED = "RECOMMENDED"
    AVAILABLE = "AVAILABLE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class EditorialProfilePolicy:
    profile_id: str
    preferred_archetypes: tuple[str, ...]
    completion_semantics: str
    signal_priorities: tuple[str, ...]
    required_qualities: tuple[str, ...]
    optional_qualities: tuple[str, ...]
    soft_penalties: Mapping[str, float]
    hard_blocker_mapping: Mapping[str, str]
    preferred_duration_seconds: tuple[float, float]
    diversity_preferences: tuple[str, ...]
    recommended_score_threshold: float
    recommended_confidence_floor: float = 0.55


@dataclass(frozen=True, slots=True)
class ResolvedEditorialProfile:
    profile_id: str
    detected_profile: Mapping[str, Any]
    effective_profile: Mapping[str, Any]
    manual_override: Mapping[str, Any]
    resolution: str
    confidence: float
    requested_mode: str = "auto"
    requested_profile_id: str | None = None
    effective_profile_reason: str = "legacy_effective_profile_resolution"
    detector_version: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "detected_profile": dict(self.detected_profile),
            "effective_profile": dict(self.effective_profile),
            "manual_override": dict(self.manual_override),
            "resolution": self.resolution,
            "confidence": round(self.confidence, 6),
            "requested_mode": self.requested_mode,
            "requested_profile_id": self.requested_profile_id,
            "effective_profile_reason": self.effective_profile_reason,
            "detector_version": self.detector_version,
        }


@dataclass(frozen=True, slots=True)
class CandidateEditorialDecision:
    profile_id: str
    archetype: str
    editorial_score: float
    strengths: tuple[str, ...]
    soft_issues: tuple[str, ...]
    hard_blockers: tuple[str, ...]
    surfacing_state: EditorialSurfacingState
    selectable: bool
    primary_reason: str
    policy_version: str = EDITORIAL_PROFILE_POLICY_VERSION
    profile_provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "archetype": self.archetype,
            "editorial_score": round(self.editorial_score, 3),
            "strengths": list(self.strengths),
            "soft_issues": list(self.soft_issues),
            "hard_blockers": list(self.hard_blockers),
            "surfacing_state": self.surfacing_state.value,
            "selectable": self.selectable,
            "primary_reason": self.primary_reason,
            "policy_version": self.policy_version,
            "profile_provenance": dict(self.profile_provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateEditorialDecision":
        state = EditorialSurfacingState(str(value.get("surfacing_state") or "BLOCKED"))
        return cls(
            profile_id=str(value.get("profile_id") or "podcast"),
            archetype=str(value.get("archetype") or "complete_thought"),
            editorial_score=float(value.get("editorial_score") or 0),
            strengths=tuple(str(item) for item in value.get("strengths", []) if str(item)),
            soft_issues=tuple(str(item) for item in value.get("soft_issues", []) if str(item)),
            hard_blockers=tuple(str(item) for item in value.get("hard_blockers", []) if str(item)),
            surfacing_state=state,
            selectable=bool(value.get("selectable", state is not EditorialSurfacingState.BLOCKED)),
            primary_reason=str(value.get("primary_reason") or ""),
            policy_version=str(value.get("policy_version") or EDITORIAL_PROFILE_POLICY_VERSION),
            profile_provenance=dict(value.get("profile_provenance") or {}),
        )


_GLOBAL_TECHNICAL_BLOCKERS = frozenset({
    "SOURCE_INTERVAL_INVALID",
    "CANDIDATE_IDENTITY_INVALID",
    "WORD_BOUNDARY_UNRECOVERABLE",
    "SENTENCE_BOUNDARY_UNRECOVERABLE",
    "AUDIO_UNINTELLIGIBLE",
    "VERTICAL_COMPOSITION_IMPOSSIBLE",
})

_GLOBAL_SOFT_PENALTIES = MappingProxyType({
    "NO_PAYOFF": 4.0,
    "FALSE_HOOK_RISK": 3.0,
    "DURATION_OUT_OF_RANGE": 2.0,
    "SEMANTIC_INCOMPLETE": 5.0,
    "CONTEXT_DEBT_CRITICAL": 5.0,
    "UNRESOLVED_PRONOUN": 2.0,
    "UNNAMED_ENTITY": 2.0,
    "ANSWER_WITHOUT_QUESTION_CONTEXT": 3.0,
    "REFERENCES_EARLIER_CONTENT": 2.0,
    "UNDEFINED_TERM_OR_SETUP": 2.0,
    "BOUNDARY_EVIDENCE_UNAVAILABLE": 3.0,
    "SPEECH_CLARITY_EVIDENCE_UNAVAILABLE": 2.0,
    "VISUAL_EVIDENCE_UNAVAILABLE": 1.0,
})

_STRUCTURAL_RULES = MappingProxyType({
    "SEMANTIC_INCOMPLETE": "block_only_with_evidence_backed_truncation",
    "CONTEXT_DEBT_CRITICAL": "block_only_with_explicit_essential_context_loss",
})


def _policy(
    profile_id: str,
    archetypes: tuple[str, ...],
    completion: str,
    priorities: tuple[str, ...],
    required: tuple[str, ...],
    optional: tuple[str, ...],
    duration: tuple[float, float],
    diversity: tuple[str, ...],
    threshold: float,
    *,
    penalties: Mapping[str, float] | None = None,
) -> EditorialProfilePolicy:
    merged = dict(_GLOBAL_SOFT_PENALTIES)
    merged.update(dict(penalties or {}))
    return EditorialProfilePolicy(
        profile_id=profile_id,
        preferred_archetypes=archetypes,
        completion_semantics=completion,
        signal_priorities=priorities,
        required_qualities=required,
        optional_qualities=optional,
        soft_penalties=MappingProxyType(merged),
        hard_blocker_mapping=_STRUCTURAL_RULES,
        preferred_duration_seconds=duration,
        diversity_preferences=diversity,
        recommended_score_threshold=threshold,
    )


EDITORIAL_PROFILE_POLICIES = MappingProxyType({
    "podcast": _policy(
        "podcast", ("expert_insight", "personal_story", "strong_opinion", "funny_exchange"),
        "complete idea, answer, anecdote beat, or exchange", ("complete_thought", "specificity", "quotability", "usefulness"),
        ("meaning_preserved",), ("hook", "payoff", "high_emotion", "visual_activity"), (20, 90),
        ("insight", "story", "opinion", "humor", "personal", "practical"), 55,
    ),
    "interview": _policy(
        "interview", ("strong_answer", "revelation", "confession", "interpersonal_exchange"),
        "answer or exchange remains understandable", ("answer_present", "complete_thought", "authenticity", "tension"),
        ("meaning_preserved",), ("hook", "dramatic_payoff", "high_energy"), (18, 90),
        ("answer", "revelation", "story", "disagreement", "humor"), 54,
    ),
    "talking_head_expert": _policy(
        "talking_head_expert", ("expert_explainer", "myth_fact", "tactical_tip", "case_study"),
        "promised insight or takeaway is present", ("usefulness", "specificity", "authority", "complete_thought"),
        ("meaning_preserved",), ("hook", "visual_motion", "emotional_payoff"), (15, 75),
        ("explainer", "tip", "framework", "example", "opinion"), 57,
    ),
    "gameplay": _policy(
        "gameplay", ("action_peak", "near_miss", "failure_success", "reaction", "banter"),
        "event outcome when it is the point, or a logical conversational unit", ("action_peak", "result_present", "reaction_present", "surprise"),
        ("event_or_exchange_coherent",), ("spoken_hook", "classical_payoff", "speech_density"), (8, 75),
        ("clutch", "fail", "reaction", "banter", "challenge"), 48,
        penalties={"NO_PAYOFF": 1.0, "FALSE_HOOK_RISK": 1.0},
    ),
    "stream": _policy(
        "stream", ("spontaneous_reaction", "rant", "chat_moment", "story", "social_interaction"),
        "logical micro-episode with understandable trigger when required", ("authenticity", "reaction", "humor", "surprise"),
        ("micro_episode_coherent",), ("formal_hook", "formal_payoff", "polished_delivery"), (10, 80),
        ("reaction", "story", "rant", "community", "surprise"), 48,
        penalties={"NO_PAYOFF": 1.0, "FALSE_HOOK_RISK": 1.0},
    ),
    "vlog_lifestyle": _policy(
        "vlog_lifestyle", ("day_in_life", "relatable_observation", "mishap", "small_win", "visual_transformation"),
        "logical micro-scene; hook and payoff are optional", ("authenticity", "relatability", "visual_interest", "emotional_truth"),
        ("scene_coherent",), ("hook", "explicit_outcome", "high_virality"), (10, 75),
        ("routine", "personal", "mishap", "discovery", "transformation"), 48,
        penalties={"NO_PAYOFF": 0.5, "FALSE_HOOK_RISK": 1.0},
    ),
    "food": _policy(
        "food", ("taste_reaction", "cooking_reveal", "technique", "comparison", "verdict"),
        "complete useful step, taste/verdict beat, or satisfying visual sequence", ("visual_payoff", "reaction", "discovery", "usefulness"),
        ("food_event_coherent",), ("spoken_hook", "dialogue_density", "classical_story"), (8, 70),
        ("taste", "process", "discovery", "result", "verdict"), 49,
        penalties={"NO_PAYOFF": 1.0, "FALSE_HOOK_RISK": 0.5},
    ),
    "travel": _policy(
        "travel", ("visual_reveal", "hidden_gem", "local_interaction", "mishap", "useful_tip"),
        "destination or event micro-episode; visual discovery may stand alone", ("discovery", "visual_spectacle", "curiosity", "authenticity"),
        ("scene_coherent",), ("formal_story", "verbal_payoff", "speech"), (10, 80),
        ("view", "culture", "interaction", "mishap", "tip"), 48,
        penalties={"NO_PAYOFF": 0.5, "FALSE_HOOK_RISK": 0.5},
    ),
    "tutorial_education": _policy(
        "tutorial_education", ("expert_explainer", "step_by_step", "myth_fact", "problem_solution"),
        "actual useful answer or complete instructional step", ("usefulness", "complete_thought", "answer_present", "specificity"),
        ("instruction_not_misleading",), ("exciting_hook", "high_emotion", "short_duration"), (20, 120),
        ("explanation", "method", "correction", "example", "result"), 58,
        penalties={"NO_PAYOFF": 6.0, "SEMANTIC_INCOMPLETE": 8.0},
    ),
    "review": _policy(
        "review", ("verdict", "pro_con", "comparison", "test_result", "value_judgment"),
        "claim has enough evidence or reasoning to preserve the opinion", ("verdict", "specificity", "comparison", "evidence"),
        ("reviewer_meaning_preserved",), ("final_score", "hook", "binary_opinion"), (15, 90),
        ("verdict", "pro", "con", "test", "comparison"), 54,
    ),
    "reaction": _policy(
        "reaction", ("shock", "laughter", "disbelief", "emotional_response", "commentary"),
        "trigger is understandable or the reaction is self-explanatory", ("reaction_present", "surprise", "emotional_intensity", "authenticity"),
        ("reaction_meaning_preserved",), ("formal_hook", "long_explanation", "classical_payoff"), (5, 55),
        ("shock", "humor", "emotion", "opinion", "conflict"), 47,
        penalties={"NO_PAYOFF": 0.5, "FALSE_HOOK_RISK": 1.0},
    ),
    "story_entertainment": _policy(
        "story_entertainment", ("banter", "conflict", "funny_exchange", "twist", "emotional_beat"),
        "logical scene or dialogue unit; payoff only when explicitly promised", ("humor", "surprise", "conflict", "character"),
        ("scene_coherent",), ("hook", "formal_payoff", "full_backstory"), (8, 90),
        ("conflict", "humor", "emotion", "suspense", "character"), 47,
        penalties={"NO_PAYOFF": 0.5, "FALSE_HOOK_RISK": 0.5},
    ),
    "movie_series": _policy(
        "movie_series", ("logical_scene_unit", "dialogue_exchange", "action_beat", "emotional_beat", "memorable_quote"),
        "intentional coherent scene/dialogue/action fragment; no classical arc required", ("logical_scene_unit", "conflict", "emotion", "humor", "character"),
        ("intentional_boundary",), ("hook", "payoff", "standalone_story", "full_plot_context"), (8, 120),
        ("conflict", "humor", "emotion", "suspense", "character"), 46,
        penalties={"NO_PAYOFF": 0.0, "FALSE_HOOK_RISK": 0.0, "DURATION_OUT_OF_RANGE": 1.0, "SEMANTIC_INCOMPLETE": 2.0},
    ),
    "sports_fitness": _policy(
        "sports_fitness", ("sports_highlight", "attempt_result", "reaction", "technique_demo", "form_correction"),
        "decisive sports event or complete safe coaching step", ("action_peak", "result_present", "visual_clarity", "usefulness"),
        ("event_or_instruction_coherent",), ("hook", "dramatic_payoff", "short_instruction"), (8, 90),
        ("highlight", "attempt", "reaction", "technique", "transformation"), 50,
        penalties={"NO_PAYOFF": 1.0},
    ),
    "news_commentary": _policy(
        "news_commentary", ("expert_explainer", "hot_take", "counterargument", "fact_interpretation", "why_it_matters"),
        "claim and intended meaning retain necessary factual context", ("context_sufficient", "complete_thought", "specificity", "authority"),
        ("meaning_and_fairness_preserved",), ("viral_hook", "high_emotion", "short_context"), (20, 120),
        ("update", "explanation", "argument", "evidence", "prediction"), 60,
        penalties={"CONTEXT_DEBT_CRITICAL": 9.0, "SEMANTIC_INCOMPLETE": 9.0},
    ),
})


if tuple(EDITORIAL_PROFILE_POLICIES) != tuple(CONTENT_PROFILE_PRESETS):
    raise RuntimeError("Editorial policy registry must cover the canonical 15 content profiles in order.")

_CONSERVATIVE_AUTO_POLICY = _policy(
    UNKNOWN_PROFILE_ID,
    ("coherent_editorial_unit", "complete_thought", "logical_scene_unit"),
    "conservative mixed fallback; preserve plausible human choices",
    ("meaning_preserved", "context_sufficient", "coherence"),
    ("meaning_preserved",),
    ("hook", "payoff", "profile_fit", "high_emotion", "visual_activity"),
    (8, 120),
    ("topic", "speaker", "scene", "emotion"),
    101,
)


_LEGACY_CONTENT_TYPE_PROFILE = {
    "podcast": "podcast",
    "interview": "interview",
    "educational": "tutorial_education",
    "tutorial": "tutorial_education",
    "gameplay": "gameplay",
    "movie_or_series": "movie_series",
    "news_or_analysis": "news_commentary",
    "commentary": "talking_head_expert",
}


def resolve_editorial_profile(
    content_profile: Mapping[str, Any] | None,
    *,
    source: Mapping[str, Any] | None = None,
) -> ResolvedEditorialProfile:
    raw = dict(content_profile or {})
    nested = raw.get("content_profile")
    if isinstance(nested, Mapping):
        raw = {**dict(nested), **raw}
    detected = dict(raw.get("detected_profile") or {})
    effective = dict(raw.get("effective_profile") or {})
    manual = dict(raw.get("manual_override") or {})
    confidence = _bounded(float(raw.get("content_type_confidence") or 0.0), 0.0, 1.0)
    requested_mode = str(raw.get("requested_mode") or "auto")
    requested_profile_id = (
        str(raw["requested_profile_id"])
        if raw.get("requested_profile_id") is not None else None
    )
    effective_reason = str(raw.get("effective_profile_reason") or "legacy_effective_profile_resolution")
    detector_version = str(raw.get("detector_version") or "legacy")

    effective_id = str(effective.get("profile_id") or "")
    if effective_id in {*EDITORIAL_PROFILE_POLICIES, UNKNOWN_PROFILE_ID}:
        detected_id = detected.get("profile_id")
        detected_confidence = (
            float(detected_id.get("confidence", 0))
            if isinstance(detected_id, Mapping) else confidence
        )
        return ResolvedEditorialProfile(
            effective_id,
            detected,
            effective,
            manual,
            "effective_profile_contract",
            1.0 if requested_mode == "manual" else _bounded(detected_confidence, 0.0, 1.0),
            requested_mode,
            requested_profile_id,
            effective_reason,
            detector_version,
        )

    # Compatibility for persisted pre-contract artifacts only.  New artifacts
    # always resolve through ``effective_profile.profile_id`` above.
    explicit_id = str(raw.get("editorial_policy_profile_id") or raw.get("content_profile_preset") or "")
    if explicit_id in EDITORIAL_PROFILE_POLICIES:
        return ResolvedEditorialProfile(explicit_id, detected, effective, manual, "explicit_profile_id", 1.0)

    manual_active = str(manual.get("provenance") or "none") != "none" or any(
        manual.get(key) for key in ("format", "editorial_mode", "domain", "traits")
    )
    if manual_active:
        profile_id, affinity = _closest_profile(effective)
        return ResolvedEditorialProfile(profile_id, detected, effective, manual, "manual_override", max(confidence, affinity))

    source_tokens = set(re.findall(
        r"[A-Za-zА-Яа-яЁё0-9]+",
        " ".join(str((source or {}).get(key) or "") for key in ("filename", "name", "path", "original_url")).casefold(),
    ))
    if source_tokens.intersection({"сериал", "фильм", "series", "movie"}):
        return ResolvedEditorialProfile("movie_series", detected, effective, manual, "auto_source_metadata_hint", max(confidence, 0.75))

    projected = _LEGACY_CONTENT_TYPE_PROFILE.get(str(raw.get("detected_content_type") or ""))
    if projected:
        return ResolvedEditorialProfile(projected, detected, effective, manual, "auto_detected_content_type", confidence)
    profile_id, affinity = _closest_profile(effective)
    return ResolvedEditorialProfile(profile_id, detected, effective, manual, "auto_effective_profile", max(confidence, affinity))


def evaluate_editorial_candidate(
    candidate: Any,
    content_profile: Mapping[str, Any] | None,
    *,
    score: float | None = None,
    confidence: float | None = None,
    recommended: bool | None = None,
    production_feasibility: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> CandidateEditorialDecision:
    """Evaluate ranking quality and, when requested, project it for Moments.

    Omitting ``recommended`` keeps the canonical production-facing decision:
    evidence-backed integrity failures remain blockers.  Moments always passes
    an explicit Brain recommendation flag.  In that projection the same
    findings remain ranked risk signals, while only an invalid identity/source
    range can remove the generated candidate from user selection.
    """

    resolved = resolve_editorial_profile(content_profile, source=source)
    policy = EDITORIAL_PROFILE_POLICIES.get(resolved.profile_id, _CONSERVATIVE_AUTO_POLICY)
    eligibility = _eligibility_mapping(candidate)
    reason_codes = _unique(str(item) for item in eligibility.get("reason_codes", []) if str(item))
    hard_blockers: list[str] = []
    soft_issues: list[str] = []

    state = str(eligibility.get("state") or "legacy_unassessed")
    if state != "assessed":
        hard_blockers.append("EDITORIAL_EVIDENCE_UNASSESSED")
    for code in reason_codes:
        if code in _GLOBAL_TECHNICAL_BLOCKERS:
            hard_blockers.append(code)
        elif code == "SEMANTIC_INCOMPLETE" and _evidence_backed_truncation(candidate, eligibility):
            hard_blockers.append(code)
        elif code == "CONTEXT_DEBT_CRITICAL" and _evidence_backed_essential_context_loss(candidate, eligibility):
            hard_blockers.append(code)
        else:
            soft_issues.append(code)

    if not _identity_and_range_valid(candidate):
        hard_blockers.append("INVALID_SOURCE_MAPPING")
    hard_blockers = _unique(hard_blockers)
    soft_issues = _unique(code for code in soft_issues if code not in hard_blockers)
    assessment_blockers = list(hard_blockers)
    moments_projection = recommended is not None
    if moments_projection:
        hard_blockers = [code for code in hard_blockers if code == "INVALID_SOURCE_MAPPING"]
        soft_issues = _unique([
            *soft_issues,
            *(code for code in assessment_blockers if code not in hard_blockers),
        ])
    coherent = not _evidence_backed_truncation(candidate, eligibility)
    archetype = _candidate_archetype(candidate, policy)
    strengths = _candidate_strengths(candidate, policy, reason_codes, coherent)
    base_score = _candidate_score(candidate, score)
    penalty = sum(float(policy.soft_penalties.get(code, 1.0)) for code in soft_issues)
    coherence_boost = 8.0 if coherent and resolved.profile_id == "movie_series" else 4.0 if coherent else 0.0
    profile_fit_boost = min(6.0, len(strengths) * 1.5)
    editorial_score = _bounded(base_score + coherence_boost + profile_fit_boost - penalty, 0.0, 100.0)
    candidate_confidence = _candidate_confidence(candidate, confidence)

    if hard_blockers:
        surfacing = EditorialSurfacingState.BLOCKED
    elif (
        bool(recommended)
        if moments_projection
        else (
            editorial_score >= policy.recommended_score_threshold
            and candidate_confidence >= policy.recommended_confidence_floor
        )
    ):
        surfacing = EditorialSurfacingState.RECOMMENDED
    else:
        surfacing = EditorialSurfacingState.AVAILABLE
    selectable = surfacing is not EditorialSurfacingState.BLOCKED
    if hard_blockers:
        primary_reason = hard_blockers[0]
    elif surfacing is EditorialSurfacingState.RECOMMENDED:
        primary_reason = strengths[0] if strengths else "STRONG_PROFILE_FIT"
    elif soft_issues:
        primary_reason = soft_issues[0]
    else:
        primary_reason = "LEGITIMATE_HUMAN_CHOICE"
    profile_provenance = resolved.to_dict()
    if moments_projection:
        profile_provenance["moments_projection"] = {
            "policy_version": MOMENTS_SURFACING_POLICY_VERSION,
            "permission_effect": "ranking_and_warning_only",
            "risk_codes": assessment_blockers,
        }
    return CandidateEditorialDecision(
        profile_id=resolved.profile_id,
        archetype=archetype,
        editorial_score=editorial_score,
        strengths=tuple(strengths),
        soft_issues=tuple(soft_issues),
        hard_blockers=tuple(hard_blockers),
        surfacing_state=surfacing,
        selectable=selectable,
        primary_reason=primary_reason,
        profile_provenance=profile_provenance,
    )


def editorial_decision_from_candidate(candidate: Any) -> CandidateEditorialDecision | None:
    value = _value(candidate, "editorial_decision")
    if isinstance(value, CandidateEditorialDecision):
        return value
    if isinstance(value, Mapping):
        try:
            return CandidateEditorialDecision.from_dict(value)
        except (TypeError, ValueError):
            return None
    return None


def _closest_profile(effective: Mapping[str, Any]) -> tuple[str, float]:
    best_id = "podcast"
    best_score = -1.0
    traits = set(str(item) for item in effective.get("traits", []) if str(item))
    for profile_id, preset in CONTENT_PROFILE_PRESETS.items():
        score = 0.0
        score += 0.45 if str(effective.get("format") or "") == preset.format else 0.0
        score += 0.30 if str(effective.get("editorial_mode") or "") == preset.editorial_mode else 0.0
        score += 0.15 if str(effective.get("domain") or "") == preset.domain else 0.0
        score += 0.10 * len(traits.intersection(preset.traits)) / max(1, len(preset.traits))
        if score > best_score:
            best_id, best_score = profile_id, score
    return best_id, _bounded(best_score, 0.0, 1.0)


def _eligibility_mapping(candidate: Any) -> dict[str, Any]:
    raw = _value(candidate, "eligibility_decision")
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    return dict(raw) if isinstance(raw, Mapping) else {}


def _identity_and_range_valid(candidate: Any) -> bool:
    candidate_id = str(_value(candidate, "candidate_id") or _value(candidate, "id") or "").strip()
    start = _number(_value(candidate, "start_seconds"), _value(candidate, "start"))
    end = _number(_value(candidate, "end_seconds"), _value(candidate, "end"))
    return bool(candidate_id and start is not None and end is not None and start >= 0 and end > start)


def _evidence_backed_truncation(candidate: Any, eligibility: Mapping[str, Any]) -> bool:
    boundary = _boundary_evidence(candidate, eligibility)
    if boundary:
        if boundary.get("word_integrity") is False or boundary.get("sentence_integrity") is False:
            return True
        if boundary.get("eligible") is False:
            return True
        semantic = _number(boundary.get("semantic_completion"))
        if semantic is not None and semantic < 0.5:
            return True
    return "semantic_boundary_violation" in _cached_failures(eligibility)


def _evidence_backed_essential_context_loss(candidate: Any, eligibility: Mapping[str, Any]) -> bool:
    if "critical_context_dependency" in _cached_failures(eligibility):
        return True
    boundary = _boundary_evidence(candidate, eligibility)
    independence = _number(boundary.get("context_independence")) if boundary else None
    return bool(boundary.get("eligible") is False and independence is not None and independence < 0.25) if boundary else False


def _boundary_evidence(candidate: Any, eligibility: Mapping[str, Any]) -> dict[str, Any]:
    boundary = _value(candidate, "boundary_diagnostics")
    if isinstance(boundary, Mapping):
        return dict(boundary)
    for raw in eligibility.get("evidence_refs", []):
        if isinstance(raw, Mapping) and raw.get("code") == "semantic_boundary" and isinstance(raw.get("details"), Mapping):
            return dict(raw["details"])
    review_boundary = _value(candidate, "boundary_evidence")
    return dict(review_boundary) if isinstance(review_boundary, Mapping) else {}


def _cached_failures(eligibility: Mapping[str, Any]) -> set[str]:
    failures: set[str] = set()
    for raw in eligibility.get("evidence_refs", []):
        if not isinstance(raw, Mapping) or raw.get("code") != "cached_hard_eligibility":
            continue
        details = raw.get("details")
        if isinstance(details, Mapping):
            failures.update(str(item) for item in details.get("critical_failures", []) if str(item))
    return failures


def _candidate_archetype(candidate: Any, policy: EditorialProfilePolicy) -> str:
    explicit = _value(candidate, "archetype") or _value(candidate, "candidate_kind")
    if explicit and str(explicit) not in {"transcript", "unknown"}:
        return str(explicit)
    semantic = _value(candidate, "semantic_evidence")
    if isinstance(semantic, Mapping):
        explicit = semantic.get("archetype") or semantic.get("type")
        if explicit:
            return str(explicit)
    return policy.preferred_archetypes[0]


def _candidate_strengths(
    candidate: Any,
    policy: EditorialProfilePolicy,
    reasons: list[str],
    coherent: bool,
) -> list[str]:
    strengths: list[str] = []
    if coherent:
        strengths.append("logical_scene_unit" if policy.profile_id in {"movie_series", "story_entertainment"} else "coherent_editorial_unit")
    if "NO_PAYOFF" not in reasons:
        strengths.append("payoff_or_natural_completion")
    if "CONTEXT_DEBT_CRITICAL" not in reasons:
        strengths.append("context_sufficient")
    if _candidate_confidence(candidate, None) >= 0.75:
        strengths.append("evidence_confidence")
    return _unique(strengths)


def _candidate_score(candidate: Any, explicit: float | None) -> float:
    if explicit is not None:
        return _bounded(float(explicit), 0.0, 100.0)
    for name in ("score", "viral_score", "local_quality_score"):
        value = _number(_value(candidate, name))
        if value is not None:
            return _bounded(value * 100 if 0 <= value <= 1 else value, 0.0, 100.0)
    return 35.0


def _candidate_confidence(candidate: Any, explicit: float | None) -> float:
    if explicit is not None:
        return _bounded(float(explicit), 0.0, 1.0)
    value = _number(_value(candidate, "confidence"))
    if value is None:
        features = _value(candidate, "feature_vector")
        value = _number(features.get("transcript_confidence")) if isinstance(features, Mapping) else None
    return _bounded(value if value is not None else 0.6, 0.0, 1.0)


def _value(candidate: Any, name: str) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name)
    return getattr(candidate, name, None)


def _number(*values: Any) -> float | None:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if value and value not in result:
            result.append(value)
    return result
