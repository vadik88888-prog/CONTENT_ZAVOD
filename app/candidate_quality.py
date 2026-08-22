from __future__ import annotations

"""Versioned, deterministic candidate-quality contracts used by the existing scorer.

This module deliberately does not introduce another ranking pipeline.  It turns
the inputs already assembled for ``local_scoring`` into durable eligibility and
score evidence so that the existing local/AI merge has one authoritative record.
"""

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any


CANDIDATE_QUALITY_SCHEMA_VERSION = "6D.2"
SPEECH_CLARITY_CONFIDENCE_THRESHOLD = 0.45
SPEECH_CLARITY_MATERIAL_DURATION_SECONDS = 1.0
SPEECH_CLARITY_MATERIAL_COVERAGE_RATIO = 0.15

FACTOR_WEIGHTS: dict[str, float] = {
    "hook": 0.12,
    "narrative_completeness": 0.12,
    "payoff": 0.12,
    "information_value": 0.10,
    "emotional_intensity": 0.07,
    "visual_interest": 0.09,
    "audio_energy": 0.06,
    "self_containedness": 0.12,
    "vertical_viability": 0.07,
    "novelty": 0.08,
    "confidence": 0.05,
}
CONTEXT_DEBT_WEIGHT = 0.10
MIN_EDITORIAL_MULTIMODAL_CONFIDENCE = 0.65


class EvidenceState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LEGACY = "legacy"


class EligibilityState(StrEnum):
    ASSESSED = "assessed"
    LEGACY_UNASSESSED = "legacy_unassessed"


class EligibilityReasonCode(StrEnum):
    SOURCE_INTERVAL_INVALID = "SOURCE_INTERVAL_INVALID"
    CANDIDATE_IDENTITY_INVALID = "CANDIDATE_IDENTITY_INVALID"
    WORD_BOUNDARY_UNRECOVERABLE = "WORD_BOUNDARY_UNRECOVERABLE"
    SENTENCE_BOUNDARY_UNRECOVERABLE = "SENTENCE_BOUNDARY_UNRECOVERABLE"
    BOUNDARY_EVIDENCE_UNAVAILABLE = "BOUNDARY_EVIDENCE_UNAVAILABLE"
    SEMANTIC_INCOMPLETE = "SEMANTIC_INCOMPLETE"
    CONTEXT_DEBT_CRITICAL = "CONTEXT_DEBT_CRITICAL"
    UNRESOLVED_PRONOUN = "UNRESOLVED_PRONOUN"
    UNNAMED_ENTITY = "UNNAMED_ENTITY"
    ANSWER_WITHOUT_QUESTION_CONTEXT = "ANSWER_WITHOUT_QUESTION_CONTEXT"
    REFERENCES_EARLIER_CONTENT = "REFERENCES_EARLIER_CONTENT"
    UNDEFINED_TERM_OR_SETUP = "UNDEFINED_TERM_OR_SETUP"
    NO_PAYOFF = "NO_PAYOFF"
    FALSE_HOOK_RISK = "FALSE_HOOK_RISK"
    AUDIO_UNINTELLIGIBLE = "AUDIO_UNINTELLIGIBLE"
    SPEECH_CLARITY_RISK = "SPEECH_CLARITY_RISK"
    SPEECH_CLARITY_EVIDENCE_UNAVAILABLE = "SPEECH_CLARITY_EVIDENCE_UNAVAILABLE"
    VERTICAL_COMPOSITION_IMPOSSIBLE = "VERTICAL_COMPOSITION_IMPOSSIBLE"
    VISUAL_EVIDENCE_UNAVAILABLE = "VISUAL_EVIDENCE_UNAVAILABLE"
    DURATION_OUT_OF_RANGE = "DURATION_OUT_OF_RANGE"
    LEGACY_UNASSESSED = "LEGACY_UNASSESSED"


_CACHED_HARD_FAILURE_REASON_CODES: dict[str, tuple[EligibilityReasonCode, ...]] = {
    "incomplete_story": (
        # A sentence may end cleanly while the story still has no resolution.
        # Preserve both facts carried by the cached publishability assessment.
        EligibilityReasonCode.SEMANTIC_INCOMPLETE,
        EligibilityReasonCode.NO_PAYOFF,
    ),
    "critical_context_dependency": (EligibilityReasonCode.CONTEXT_DEBT_CRITICAL,),
    "semantic_boundary_violation": (EligibilityReasonCode.SEMANTIC_INCOMPLETE,),
}


def cached_hard_eligibility_reason_codes(
    cached_eligibility: dict[str, Any] | None,
) -> tuple[list[EligibilityReasonCode], list[str]]:
    """Map an explicit cached hard verdict into the typed eligibility vocabulary.

    The raw failures are returned as well so callers can preserve exact evidence.
    A hard verdict requires both parts of the old contract: ``status=rejected``
    and at least one ``critical_failures`` item.
    """

    cached = cached_eligibility if isinstance(cached_eligibility, dict) else {}
    critical_failures = [
        str(item) for item in cached.get("critical_failures", [])
        if str(item).strip()
    ]
    if str(cached.get("status") or "") != "rejected" or not critical_failures:
        return [], []
    mapped: list[EligibilityReasonCode] = []
    for failure in critical_failures:
        for code in _CACHED_HARD_FAILURE_REASON_CODES.get(failure, ()):
            if code not in mapped:
                mapped.append(code)
    return mapped, critical_failures


@dataclass(slots=True)
class EvidenceReference:
    code: str
    state: EvidenceState
    source: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "state": self.state.value,
            "source": self.source,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceReference":
        raw_state = str(data.get("state") or EvidenceState.UNAVAILABLE.value)
        try:
            state = EvidenceState(raw_state)
        except ValueError:
            state = EvidenceState.UNAVAILABLE
        return cls(
            code=str(data.get("code") or "UNKNOWN_EVIDENCE"),
            state=state,
            source=str(data.get("source") or "unknown"),
            details=dict(data.get("details") or {}),
        )


@dataclass(slots=True)
class EligibilityDecision:
    schema_version: str
    config_version: str
    state: EligibilityState
    eligible: bool | None
    reason_codes: list[EligibilityReasonCode] = field(default_factory=list)
    recoverable_issues: list[EligibilityReasonCode] = field(default_factory=list)
    required_boundary_actions: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceReference] = field(default_factory=list)

    @property
    def explicitly_eligible(self) -> bool:
        return self.state is EligibilityState.ASSESSED and self.eligible is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "state": self.state.value,
            "eligible": self.eligible,
            "reason_codes": [code.value for code in self.reason_codes],
            "recoverable_issues": [code.value for code in self.recoverable_issues],
            "required_boundary_actions": self.required_boundary_actions,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EligibilityDecision":
        def codes(value: Any) -> list[EligibilityReasonCode]:
            parsed: list[EligibilityReasonCode] = []
            for raw in value if isinstance(value, list) else []:
                try:
                    parsed.append(EligibilityReasonCode(str(raw)))
                except ValueError:
                    continue
            return parsed

        raw_state = str(data.get("state") or EligibilityState.ASSESSED.value)
        try:
            state = EligibilityState(raw_state)
        except ValueError:
            state = EligibilityState.LEGACY_UNASSESSED
        raw_evidence = data.get("evidence_refs")
        return cls(
            schema_version=str(data.get("schema_version") or CANDIDATE_QUALITY_SCHEMA_VERSION),
            config_version=str(data.get("config_version") or "unknown"),
            state=state,
            eligible=data.get("eligible") if isinstance(data.get("eligible"), bool) else None,
            reason_codes=codes(data.get("reason_codes")),
            recoverable_issues=codes(data.get("recoverable_issues")),
            required_boundary_actions=[str(item) for item in data.get("required_boundary_actions", []) if str(item)],
            evidence_refs=[EvidenceReference.from_dict(item) for item in raw_evidence if isinstance(item, dict)] if isinstance(raw_evidence, list) else [],
        )


@dataclass(slots=True)
class ScorePenalty:
    code: str
    amount: float
    evidence_refs: list[EvidenceReference] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "amount": round(self.amount, 3),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScorePenalty":
        evidence = data.get("evidence_refs")
        return cls(
            code=str(data.get("code") or "UNKNOWN_PENALTY"),
            amount=float(data.get("amount") or 0),
            evidence_refs=[EvidenceReference.from_dict(item) for item in evidence if isinstance(item, dict)] if isinstance(evidence, list) else [],
        )


@dataclass(slots=True)
class FactorAssessment:
    """One evidence-bearing editorial factor; never a final selection decision."""

    score: float
    evidence_refs: list[EvidenceReference]
    confidence: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(_bounded(self.score), 3),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "confidence": round(max(0.0, min(1.0, self.confidence)), 6),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactorAssessment":
        evidence = data.get("evidence_refs")
        return cls(
            score=_bounded(float(data.get("score") or 0)),
            evidence_refs=[EvidenceReference.from_dict(item) for item in evidence if isinstance(item, dict)] if isinstance(evidence, list) else [],
            confidence=max(0.0, min(1.0, float(data.get("confidence") or 0))),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(slots=True)
class CandidateScoreV2:
    schema_version: str
    config_version: str
    components: dict[str, float]
    penalties: list[ScorePenalty]
    raw_score: float
    final_score: float
    evidence_refs: list[EvidenceReference]
    provenance: dict[str, Any]
    factors: dict[str, FactorAssessment] = field(default_factory=dict)
    factor_weights: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    state: EligibilityState = EligibilityState.ASSESSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "state": self.state.value,
            "component_scores": {key: round(value, 3) for key, value in self.components.items()},
            "factors": {key: value.to_dict() for key, value in self.factors.items()},
            "factor_weights": {key: round(value, 6) for key, value in self.factor_weights.items()},
            "penalties": [item.to_dict() for item in self.penalties],
            "raw_score": round(self.raw_score, 3),
            "final_score": round(self.final_score, 3),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "provenance": self.provenance,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateScoreV2":
        raw_state = str(data.get("state") or EligibilityState.ASSESSED.value)
        try:
            state = EligibilityState(raw_state)
        except ValueError:
            state = EligibilityState.LEGACY_UNASSESSED
        penalties = data.get("penalties")
        evidence = data.get("evidence_refs")
        components = {str(key): float(value) for key, value in dict(data.get("component_scores") or {}).items()}
        raw_factors = data.get("factors")
        factors = {
            str(key): FactorAssessment.from_dict(value)
            for key, value in raw_factors.items()
            if isinstance(value, dict)
        } if isinstance(raw_factors, dict) else {}
        if not factors:
            factors = {
                key: FactorAssessment(
                    score=value,
                    evidence_refs=[EvidenceReference(key, EvidenceState.LEGACY, "candidate_score_v2")],
                    confidence=0.25,
                    provenance={"mode": "legacy_component_adapter"},
                )
                for key, value in components.items()
            }
        return cls(
            schema_version=str(data.get("schema_version") or CANDIDATE_QUALITY_SCHEMA_VERSION),
            config_version=str(data.get("config_version") or "unknown"),
            components=components,
            penalties=[ScorePenalty.from_dict(item) for item in penalties if isinstance(item, dict)] if isinstance(penalties, list) else [],
            raw_score=float(data.get("raw_score") or 0),
            final_score=float(data.get("final_score") or 0),
            evidence_refs=[EvidenceReference.from_dict(item) for item in evidence if isinstance(item, dict)] if isinstance(evidence, list) else [],
            provenance=dict(data.get("provenance") or {}),
            factors=factors,
            factor_weights={str(key): float(value) for key, value in dict(data.get("factor_weights") or {}).items()},
            diagnostics=dict(data.get("diagnostics") or {}),
            state=state,
        )


def legacy_eligibility_decision() -> EligibilityDecision:
    return EligibilityDecision(
        schema_version="legacy",
        config_version="legacy",
        state=EligibilityState.LEGACY_UNASSESSED,
        eligible=None,
        reason_codes=[EligibilityReasonCode.LEGACY_UNASSESSED],
        evidence_refs=[EvidenceReference("candidate_quality", EvidenceState.LEGACY, "serialized_candidate")],
    )


def legacy_score_v2() -> CandidateScoreV2:
    return CandidateScoreV2(
        schema_version="legacy",
        config_version="legacy",
        components={},
        penalties=[],
        raw_score=0.0,
        final_score=0.0,
        evidence_refs=[EvidenceReference("candidate_score_v2", EvidenceState.LEGACY, "serialized_candidate")],
        provenance={"mode": "legacy_unassessed"},
        state=EligibilityState.LEGACY_UNASSESSED,
    )


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
_UNRESOLVED_OPENINGS = {
    "it", "this", "that", "these", "those", "he", "she", "they", "him", "her", "them",
    "это", "этот", "эта", "эти", "тот", "та", "те", "он", "она", "они", "его", "её", "их",
}
_UNNAMED_ENTITY_OPENINGS = {"he", "she", "they", "him", "her", "them", "он", "она", "они", "его", "её", "их"}
_ANSWER_OPENINGS = {"yes", "no", "because", "therefore", "да", "нет", "потому", "поэтому"}
_EARLIER_REFERENCES = (
    "as i said", "as we said", "earlier", "previously", "before this", "above",
    "как я говорил", "как мы говорили", "ранее", "до этого", "выше",
)
_PAYOFF_MARKERS = (
    "therefore", "the result", "that means", "the answer", "so ", "in the end", "finally",
    "итог", "результат", "значит", "вывод", "поэтому", "в итоге", "наконец",
)


def assess_context_debt(text: str, features: dict[str, Any]) -> tuple[list[EligibilityReasonCode], list[EvidenceReference]]:
    normalized = " ".join(text.split())
    words = [item.casefold() for item in _WORD_RE.findall(normalized)]
    opening = words[0] if words else ""
    codes: list[EligibilityReasonCode] = []
    evidence: list[EvidenceReference] = []

    def add(code: EligibilityReasonCode, details: dict[str, Any]) -> None:
        if code not in codes:
            codes.append(code)
            evidence.append(EvidenceReference(code.value, EvidenceState.AVAILABLE, "candidate_transcript", details))

    if opening in _UNRESOLVED_OPENINGS:
        add(EligibilityReasonCode.UNRESOLVED_PRONOUN, {"opening_token": opening})
    if opening in _UNNAMED_ENTITY_OPENINGS:
        add(EligibilityReasonCode.UNNAMED_ENTITY, {"opening_token": opening})
    if opening in _ANSWER_OPENINGS:
        add(EligibilityReasonCode.ANSWER_WITHOUT_QUESTION_CONTEXT, {"opening_token": opening})
    lowered = normalized.casefold()
    if any(marker in lowered for marker in _EARLIER_REFERENCES):
        add(EligibilityReasonCode.REFERENCES_EARLIER_CONTENT, {"matched_reference": True})
    if opening in {"this", "that", "these", "those", "это", "этот", "эта", "эти", "тот", "та", "те"}:
        add(EligibilityReasonCode.UNDEFINED_TERM_OR_SETUP, {"opening_token": opening})
    value = features.get("context_dependency_score")
    if value is not None and float(value) >= 65:
        evidence.append(EvidenceReference("context_dependency_score", EvidenceState.AVAILABLE, "transcript_features", {"value": round(float(value), 3)}))
    return codes, evidence


def build_eligibility_decision(
    candidate: Any,
    features: dict[str, Any],
    *,
    config_version: str,
    min_duration_seconds: float | None,
    max_duration_seconds: float | None,
    visual_analysis: dict[str, Any] | None,
) -> EligibilityDecision:
    reasons: list[EligibilityReasonCode] = []
    recoverable: list[EligibilityReasonCode] = []
    actions: list[str] = []
    evidence: list[EvidenceReference] = []

    def reason(code: EligibilityReasonCode) -> None:
        if code not in reasons:
            reasons.append(code)

    valid_range = candidate.start >= 0 and candidate.end > candidate.start
    evidence.append(EvidenceReference(
        "source_range", EvidenceState.AVAILABLE, "candidate", {
            "candidate_id": candidate.id, "source_start": round(candidate.start, 3), "source_end": round(candidate.end, 3), "valid": valid_range,
        },
    ))
    if not valid_range:
        reason(EligibilityReasonCode.SOURCE_INTERVAL_INVALID)
    if not str(candidate.id).strip():
        reason(EligibilityReasonCode.CANDIDATE_IDENTITY_INVALID)
    if min_duration_seconds is not None and candidate.duration < min_duration_seconds:
        reason(EligibilityReasonCode.DURATION_OUT_OF_RANGE)
    if max_duration_seconds is not None and candidate.duration > max_duration_seconds:
        reason(EligibilityReasonCode.DURATION_OUT_OF_RANGE)

    boundary = candidate.boundary_diagnostics or {}
    if boundary:
        evidence.append(EvidenceReference("semantic_boundary", EvidenceState.AVAILABLE, "boundary_diagnostics", {
            "eligible": boundary.get("eligible"), "word_integrity": boundary.get("word_integrity"),
            "sentence_integrity": boundary.get("sentence_integrity"), "semantic_completion": boundary.get("semantic_completion"),
            "payoff_preserved": boundary.get("payoff_preserved"),
        }))
        if boundary.get("word_integrity") is False:
            reason(EligibilityReasonCode.WORD_BOUNDARY_UNRECOVERABLE)
            actions.append("refine_to_word_boundary")
        if boundary.get("sentence_integrity") is False:
            reason(EligibilityReasonCode.SENTENCE_BOUNDARY_UNRECOVERABLE)
            actions.append("refine_to_sentence_boundary")
        if boundary.get("eligible") is False or float(boundary.get("semantic_completion", 1.0)) < 0.5:
            reason(EligibilityReasonCode.SEMANTIC_INCOMPLETE)
            actions.append("restore_semantic_completion")
    else:
        reason(EligibilityReasonCode.BOUNDARY_EVIDENCE_UNAVAILABLE)
        recoverable.append(EligibilityReasonCode.BOUNDARY_EVIDENCE_UNAVAILABLE)
        evidence.append(EvidenceReference("semantic_boundary", EvidenceState.UNAVAILABLE, "boundary_diagnostics"))

    semantic = candidate.semantic_evidence or {}
    semantic_completeness = semantic.get("completeness_score", features.get("completeness_score"))
    if semantic_completeness is not None and float(semantic_completeness) < 50 and float(semantic_completeness) <= 1:
        semantic_completeness = float(semantic_completeness) * 100
    if semantic_completeness is not None and float(semantic_completeness) < 50:
        reason(EligibilityReasonCode.SEMANTIC_INCOMPLETE)
    evidence.append(EvidenceReference(
        "semantic_story_unit", EvidenceState.AVAILABLE if semantic else EvidenceState.UNAVAILABLE,
        "story_unit" if semantic else "candidate_features",
        {"story_unit_id": candidate.story_unit_id, "completeness_score": semantic_completeness},
    ))

    context_codes, context_evidence = assess_context_debt(candidate.text, features)
    evidence.extend(context_evidence)
    if context_codes:
        reasons.extend(code for code in context_codes if code not in reasons)
    if context_codes or float(features.get("context_dependency_score", 0)) >= 65:
        reason(EligibilityReasonCode.CONTEXT_DEBT_CRITICAL)
        actions.append("include_required_context")

    hook_strength, payoff_strength, hook_evidence, payoff_evidence, false_hook = assess_hook_and_payoff(candidate, features, boundary)
    evidence.extend([hook_evidence, payoff_evidence])
    if not payoff_strength:
        reason(EligibilityReasonCode.NO_PAYOFF)
        actions.append("extend_to_payoff")
    if false_hook:
        reason(EligibilityReasonCode.FALSE_HOOK_RISK)

    confidence = features.get("transcript_confidence")
    if confidence is None or float(confidence) <= 0:
        reason(EligibilityReasonCode.SPEECH_CLARITY_EVIDENCE_UNAVAILABLE)
        recoverable.append(EligibilityReasonCode.SPEECH_CLARITY_EVIDENCE_UNAVAILABLE)
        evidence.append(EvidenceReference("speech_clarity", EvidenceState.UNAVAILABLE, "transcript_features"))
    else:
        clarity = float(confidence)
        clarity_materiality = _speech_clarity_materiality(candidate, features, clarity)
        clarity_details = {"transcript_confidence": round(clarity, 3)}
        if clarity_materiality is not None:
            clarity_details.update(clarity_materiality)
        evidence.append(EvidenceReference("speech_clarity", EvidenceState.AVAILABLE, "transcript_features", clarity_details))
        if clarity < SPEECH_CLARITY_CONFIDENCE_THRESHOLD:
            if clarity_materiality is None or clarity_materiality["material"]:
                reason(EligibilityReasonCode.AUDIO_UNINTELLIGIBLE)
            else:
                reason(EligibilityReasonCode.SPEECH_CLARITY_RISK)
                recoverable.append(EligibilityReasonCode.SPEECH_CLARITY_RISK)

    visual_status = str((visual_analysis or {}).get("status") or "unavailable")
    visual_keyframes = (visual_analysis or {}).get("subject_keyframes", [])
    pass2 = _pass2_verification(candidate)
    candidate_visual = (candidate.multimodal_provenance or {}).get("visual_evidence", [])
    if (visual_status == "completed" and visual_keyframes) or candidate_visual or pass2 is not None:
        evidence.append(EvidenceReference("visual_viability", EvidenceState.AVAILABLE, "multimodal_visual_evidence", {
            "status": visual_status, "keyframe_count": len(visual_keyframes),
            "candidate_visual_evidence_count": len(candidate_visual), "pass2_available": pass2 is not None,
        }))
    else:
        reason(EligibilityReasonCode.VISUAL_EVIDENCE_UNAVAILABLE)
        recoverable.append(EligibilityReasonCode.VISUAL_EVIDENCE_UNAVAILABLE)
        evidence.append(EvidenceReference("visual_viability", EvidenceState.UNAVAILABLE, "visual_analysis", {"status": visual_status, "reason": (visual_analysis or {}).get("reason")}))

    blockers = {
        EligibilityReasonCode.SOURCE_INTERVAL_INVALID,
        EligibilityReasonCode.CANDIDATE_IDENTITY_INVALID,
        EligibilityReasonCode.WORD_BOUNDARY_UNRECOVERABLE,
        EligibilityReasonCode.SENTENCE_BOUNDARY_UNRECOVERABLE,
        EligibilityReasonCode.SEMANTIC_INCOMPLETE,
        EligibilityReasonCode.CONTEXT_DEBT_CRITICAL,
        EligibilityReasonCode.NO_PAYOFF,
        EligibilityReasonCode.FALSE_HOOK_RISK,
        EligibilityReasonCode.AUDIO_UNINTELLIGIBLE,
        EligibilityReasonCode.DURATION_OUT_OF_RANGE,
    }
    eligible = not any(code in blockers for code in reasons)
    return EligibilityDecision(
        schema_version=CANDIDATE_QUALITY_SCHEMA_VERSION,
        config_version=config_version,
        state=EligibilityState.ASSESSED,
        eligible=eligible,
        reason_codes=reasons,
        recoverable_issues=list(dict.fromkeys(recoverable)),
        required_boundary_actions=list(dict.fromkeys(actions)),
        evidence_refs=evidence,
    )


def _speech_clarity_materiality(
    candidate: Any,
    features: dict[str, Any],
    aggregate_confidence: float,
) -> dict[str, Any] | None:
    """Classify low ASR confidence by exact saved speech coverage.

    The aggregate remains an evidence signal, but only exact low-confidence
    speech that is sustained or occupies a material share of the candidate can
    prove that its dialogue is unintelligible.  Missing exact evidence keeps
    the existing strict outcome rather than silently upgrading legacy data.
    """

    raw_segments = features.get("speech_clarity_segments")
    if not isinstance(raw_segments, list):
        return None
    low_ranges: list[tuple[float, float]] = []
    low_segments: list[dict[str, Any]] = []
    malformed_low_segment = False
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        try:
            segment_confidence = float(raw.get("transcript_confidence"))
        except (TypeError, ValueError):
            continue
        if segment_confidence >= SPEECH_CLARITY_CONFIDENCE_THRESHOLD:
            continue
        try:
            start = max(float(candidate.start), float(raw["start"]))
            end = min(float(candidate.end), float(raw["end"]))
        except (KeyError, TypeError, ValueError):
            malformed_low_segment = True
            continue
        if end < start:
            malformed_low_segment = True
            continue
        low_ranges.append((start, end))
        low_segments.append({
            "segment_id": raw.get("id"),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "confidence": round(segment_confidence, 6),
        })
    if not low_segments:
        return None
    low_duration = _merged_duration(low_ranges)
    candidate_duration = max(0.0, float(candidate.end) - float(candidate.start))
    coverage_ratio = low_duration / candidate_duration if candidate_duration > 0 else None
    reasons: list[str] = []
    if malformed_low_segment or coverage_ratio is None:
        reasons.append("SPEECH_COVERAGE_UNPROVABLE")
    if low_duration >= SPEECH_CLARITY_MATERIAL_DURATION_SECONDS:
        reasons.append("LOW_CONFIDENCE_DURATION_MATERIAL")
    if coverage_ratio is not None and coverage_ratio >= SPEECH_CLARITY_MATERIAL_COVERAGE_RATIO:
        reasons.append("LOW_CONFIDENCE_COVERAGE_MATERIAL")
    return {
        "decision": "blocker" if reasons else "warning",
        "material": bool(reasons),
        "low_confidence_segments": low_segments,
        "low_confidence_duration_seconds": round(low_duration, 3),
        "candidate_duration_seconds": round(candidate_duration, 3),
        "low_confidence_coverage_ratio": round(coverage_ratio, 6) if coverage_ratio is not None else None,
        "duration_threshold_seconds": SPEECH_CLARITY_MATERIAL_DURATION_SECONDS,
        "coverage_threshold_ratio": SPEECH_CLARITY_MATERIAL_COVERAGE_RATIO,
        "materiality_reasons": reasons,
        "aggregate_confidence": round(aggregate_confidence, 6),
    }


def _merged_duration(ranges: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def resolve_eligibility_decision(
    candidate: Any,
    features: dict[str, Any],
    *,
    config_version: str,
    min_duration_seconds: float | None,
    max_duration_seconds: float | None,
    visual_analysis: dict[str, Any] | None,
    cached_eligibility: dict[str, Any] | None,
) -> EligibilityDecision:
    """Resolve current evidence without upgrading an explicit cached rejection."""

    decision = build_eligibility_decision(
        candidate,
        features,
        config_version=config_version,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        visual_analysis=visual_analysis,
    )
    cached_codes, critical_failures = cached_hard_eligibility_reason_codes(cached_eligibility)
    if not critical_failures:
        return decision

    actions_by_code = {
        EligibilityReasonCode.SEMANTIC_INCOMPLETE: "restore_semantic_completion",
        EligibilityReasonCode.NO_PAYOFF: "extend_to_payoff",
        EligibilityReasonCode.CONTEXT_DEBT_CRITICAL: "include_required_context",
    }
    reasons = list(decision.reason_codes)
    actions = list(decision.required_boundary_actions)
    for code in cached_codes:
        if code not in reasons:
            reasons.append(code)
        action = actions_by_code.get(code)
        if action and action not in actions:
            actions.append(action)
    return EligibilityDecision(
        schema_version=decision.schema_version,
        config_version=decision.config_version,
        state=EligibilityState.ASSESSED,
        eligible=False,
        reason_codes=reasons,
        recoverable_issues=list(decision.recoverable_issues),
        required_boundary_actions=actions,
        evidence_refs=[
            *decision.evidence_refs,
            EvidenceReference(
                "cached_hard_eligibility",
                EvidenceState.AVAILABLE,
                "cached_candidate.virality.eligibility",
                {
                    "status": "rejected",
                    "critical_failures": critical_failures,
                    "mapped_reason_codes": [code.value for code in cached_codes],
                },
            ),
        ],
    )


def assess_hook_and_payoff(
    candidate: Any, features: dict[str, Any], boundary: dict[str, Any] | None = None,
) -> tuple[float, bool, EvidenceReference, EvidenceReference, bool]:
    semantic = candidate.semantic_evidence or {}
    hook_text = str(semantic.get("hook") or "")
    payoff_text = str(semantic.get("payoff") or "")
    hook_score = max(0.0, min(100.0, float(features.get("hook_phrase_score", 0))))
    if hook_text:
        hook_score = max(hook_score, 55.0)
    ending = candidate.text[-300:].casefold()
    boundary = boundary or {}
    natural_end = bool(boundary.get("payoff_preserved", False)) or bool(boundary.get("sentence_integrity", False))
    pass2 = _pass2_verification(candidate)
    visible_payoff = bool(pass2 and pass2.get("payoff_visible") is True)
    generation = (candidate.multimodal_provenance or {}).get("generation", {})
    generation_reasons = generation.get("reasons", []) if isinstance(generation, dict) else []
    linked_payoff = any("payoff" in str(item) for item in generation_reasons)
    payoff_present = (
        bool(payoff_text)
        or visible_payoff
        or linked_payoff
        or any(marker in ending for marker in _PAYOFF_MARKERS)
    )
    hook_evidence = EvidenceReference("hook", EvidenceState.AVAILABLE, "story_unit" if hook_text else "transcript_features", {
        "hook_strength": round(hook_score, 3), "hook_text_available": bool(hook_text),
    })
    payoff_evidence = EvidenceReference("payoff", EvidenceState.AVAILABLE if payoff_present else EvidenceState.UNAVAILABLE, "vision_pass2" if visible_payoff else ("multimodal_candidate_provenance" if linked_payoff else ("story_unit" if payoff_text else "semantic_boundary")), {
        "payoff_present": payoff_present, "story_payoff_available": bool(payoff_text), "natural_end": natural_end,
        "visual_payoff_verified": visible_payoff, "linked_multimodal_payoff": linked_payoff,
    })
    return hook_score, payoff_present, hook_evidence, payoff_evidence, hook_score >= 55 and not payoff_present


def build_score_v2(
    candidate: Any,
    scores: dict[str, float],
    features: dict[str, Any],
    decision: EligibilityDecision,
    *,
    config_version: str,
    min_duration_seconds: float | None,
    max_duration_seconds: float | None,
    visual_analysis: dict[str, Any] | None,
) -> CandidateScoreV2:
    boundary = candidate.boundary_diagnostics or {}
    hook, payoff_present, hook_evidence, payoff_evidence, _false_hook = assess_hook_and_payoff(candidate, features, boundary)
    semantic = candidate.semantic_evidence or {}
    completeness = float(scores.get("completeness", 0))
    context_independence = float(scores.get("context_independence", 0))
    boundary_score = float(scores.get("boundary_quality", 0))
    transcript_confidence = max(0.0, min(1.0, float(features.get("transcript_confidence", 0.65) or 0.65)))
    information = float(semantic.get("information_density", 0))
    if information <= 1:
        information *= 100
    audio_score = _bounded(float(features.get("audio_energy", 0)) * 100)
    audio_evidence = (candidate.multimodal_provenance or {}).get("audio_evidence", [])
    audio_editorial_grounded = bool(audio_evidence) or float(features.get("exclamation_count", 0)) > 0
    # Goal 6E benchmark: raw loudness alone repeatedly lifted candidates without
    # an editorial event.  Preserve it as weak relevance evidence, while a
    # grounded emphasis/reaction event keeps the full audio contribution.
    audio_editorial_score = audio_score if audio_editorial_grounded else audio_score * 0.35
    emotional = _bounded(audio_editorial_score * 0.72 + float(features.get("exclamation_count", 0)) * 10)
    novelty = _bounded((100 - float(features.get("repetition_score", 0)) * 100) * 0.55 + hook * 0.45)
    text_ref = EvidenceReference(
        "candidate_text", EvidenceState.AVAILABLE, "candidate_transcript",
        {"candidate_id": candidate.id, "story_unit_ids": list(candidate.story_unit_ids)},
    )
    audio_ref = EvidenceReference(
        "audio_energy", EvidenceState.AVAILABLE, "multimodal_timeline_audio",
        {"score": round(audio_score, 3), "event_count": len((candidate.multimodal_provenance or {}).get("audio_evidence", []))},
    )
    visual_ref, visual_score, vertical_score, visual_confidence = _visual_factor_evidence(candidate, scores, visual_analysis)
    pass2 = _pass2_verification(candidate)
    pass2_ref = _pass2_evidence_ref(candidate, pass2)
    if pass2 is not None:
        pass2_confidence = float(pass2.get("confidence", 0))
        if pass2.get("hook_visible") is True:
            hook = max(hook, 80 * pass2_confidence)
        if pass2.get("payoff_visible") is True:
            payoff_present = True
        if pass2.get("reaction_visible") is True:
            emotional = max(emotional, 88 * pass2_confidence)
        visible_roles = sum(pass2.get(name) is True for name in ("hook_visible", "action_visible", "reaction_visible", "payoff_visible"))
        visual_score = max(visual_score, visible_roles / 4 * 100 * pass2_confidence)
        continuity = str(pass2.get("continuity_risk") or "unknown")
        if continuity == "low":
            completeness = max(completeness, 85 * pass2_confidence)
        elif continuity == "high":
            completeness = min(completeness, 55)

    context_codes = sum(code in decision.reason_codes for code in (
        EligibilityReasonCode.UNRESOLVED_PRONOUN, EligibilityReasonCode.UNNAMED_ENTITY,
        EligibilityReasonCode.ANSWER_WITHOUT_QUESTION_CONTEXT, EligibilityReasonCode.REFERENCES_EARLIER_CONTENT,
        EligibilityReasonCode.UNDEFINED_TERM_OR_SETUP,
    ))
    context_debt = _bounded(100 - context_independence + context_codes * 12)
    multimodal_confidence = _multimodal_confidence(
        transcript_confidence, audio_ref, visual_ref, visual_confidence, pass2,
    )
    provenance = {
        "mode": "deterministic_evidence_fusion",
        "formula_version": CANDIDATE_QUALITY_SCHEMA_VERSION,
        "ai_owns_final_score": False,
    }

    def factor(score: float, refs: list[EvidenceReference], confidence: float, *modalities: str) -> FactorAssessment:
        return FactorAssessment(
            score=_bounded(score), evidence_refs=refs,
            confidence=max(0.0, min(1.0, confidence)),
            provenance={**provenance, "modalities": list(modalities)},
        )

    visual_refs = [visual_ref, pass2_ref] if pass2_ref.state is EvidenceState.AVAILABLE else [visual_ref]
    narrative_refs = [text_ref, payoff_evidence, pass2_ref] if pass2_ref.state is EvidenceState.AVAILABLE else [text_ref, payoff_evidence]
    factors = {
        "hook": factor(hook, [hook_evidence, *( [pass2_ref] if pass2_ref.state is EvidenceState.AVAILABLE else [])], max(transcript_confidence, visual_confidence), "text", "visual"),
        "narrative_completeness": factor((completeness + boundary_score) / 2, narrative_refs, max(0.45, min(transcript_confidence, float(boundary.get("overall_boundary_score", 0.65)))), "text", "visual"),
        "payoff": factor(92 if payoff_present and pass2 and pass2.get("payoff_visible") else (82 if payoff_present else 0), [payoff_evidence, *( [pass2_ref] if pass2_ref.state is EvidenceState.AVAILABLE else [])], max(transcript_confidence, visual_confidence), "text", "visual", "audio"),
        "information_value": factor(information or min(100.0, float(features.get("word_count", 0)) * 3), [text_ref], transcript_confidence, "text"),
        "emotional_intensity": factor(emotional, [text_ref, audio_ref, *( [pass2_ref] if pass2_ref.state is EvidenceState.AVAILABLE else [])], max(transcript_confidence, visual_confidence, 0.7), "text", "audio", "visual"),
        "visual_interest": factor(visual_score, visual_refs, visual_confidence, "visual"),
        "audio_energy": factor(audio_editorial_score, [audio_ref], 0.8 if audio_editorial_grounded else 0.45, "audio"),
        "self_containedness": factor((completeness + context_independence + boundary_score) / 3, [text_ref, *[item for item in decision.evidence_refs if item.code in {"semantic_boundary", "semantic_story_unit", "context_dependency_score"}]], transcript_confidence, "text"),
        # For context_debt a higher score is deliberately worse and is subtracted below.
        "context_debt": factor(context_debt, [text_ref, *[item for item in decision.evidence_refs if item.code in {code.value for code in decision.reason_codes}]], transcript_confidence, "text"),
        "vertical_viability": factor(vertical_score, visual_refs, visual_confidence, "visual"),
        "novelty": factor(novelty, [text_ref], transcript_confidence, "text"),
        "confidence": factor(multimodal_confidence * 100, [text_ref, audio_ref, visual_ref, pass2_ref], 1.0, "text", "audio", "visual"),
    }
    components = {name: assessment.score for name, assessment in factors.items()}
    contributions = {name: factors[name].score * weight for name, weight in FACTOR_WEIGHTS.items()}
    context_contribution = factors["context_debt"].score * CONTEXT_DEBT_WEIGHT
    raw = sum(contributions.values()) - context_contribution
    penalties: list[ScorePenalty] = []

    def penalty(code: str, amount: float, refs: list[EvidenceReference]) -> None:
        if amount > 0:
            penalties.append(ScorePenalty(code, amount, refs))

    repetition = float(features.get("repetition_score", 0)) * 15
    filler = float(features.get("filler_word_ratio", 0)) * 18
    penalty("INTERNAL_REPETITION", repetition, [EvidenceReference("repetition_score", EvidenceState.AVAILABLE, "transcript_features", {"value": round(float(features.get("repetition_score", 0)), 3)})])
    penalty("FILLER_WORDS", filler, [EvidenceReference("filler_word_ratio", EvidenceState.AVAILABLE, "transcript_features", {"value": round(float(features.get("filler_word_ratio", 0)), 3)})])
    penalty("CONTEXT_DEBT", min(15.0, context_codes * 5.0), [item for item in decision.evidence_refs if item.code in {code.value for code in decision.reason_codes}])
    if visual_ref.state is EvidenceState.UNAVAILABLE:
        # Absence is uncertainty, not negative visual evidence.  The small
        # audit penalty plus the confidence factor keeps text/audio usable.
        penalty("VISUAL_EVIDENCE_UNAVAILABLE", 2.0, [visual_ref])
    if float(scores.get("clarity", 0)) < 45:
        penalty("LOW_SPEECH_CLARITY", 15.0, [item for item in decision.evidence_refs if item.code == "speech_clarity"])
    final = round(_bounded(raw - sum(item.amount for item in penalties)), 3)
    return CandidateScoreV2(
        schema_version=CANDIDATE_QUALITY_SCHEMA_VERSION,
        config_version=config_version,
        components=components,
        penalties=penalties,
        raw_score=raw,
        final_score=final,
        evidence_refs=[hook_evidence, payoff_evidence, *decision.evidence_refs],
        provenance={**provenance, "fallback_used": True, "ai_merge": "not_attempted", "pass2_status": str((candidate.vision_pass2_evidence or {}).get("status") or "not_available")},
        factors=factors,
        factor_weights={**FACTOR_WEIGHTS, "context_debt": -CONTEXT_DEBT_WEIGHT},
        diagnostics={
            "positive_contributions": {key: round(value, 3) for key, value in contributions.items()},
            "context_debt_deduction": round(context_contribution, 3),
            "penalty_total": round(sum(item.amount for item in penalties), 3),
            "modalities_available": _modalities_available(candidate, visual_ref),
        },
    )


def apply_ai_factor_assessments(candidate: Any, assessment: Any) -> None:
    """Blend grounded AI factor assessments; code still calculates the final score."""

    score = candidate.candidate_score_v2
    if score is None or not score.factors:
        return
    mappings = {
        "hook": float(assessment.hook_score),
        "narrative_completeness": float(assessment.completeness_score),
        "emotional_intensity": float(assessment.emotional_score),
        "self_containedness": _bounded((float(assessment.clarity_score) + 100 - float(assessment.context_dependency_score)) / 2),
        "context_debt": float(assessment.context_dependency_score),
    }
    for name, ai_value in mappings.items():
        local = score.factors[name]
        ai_ref = EvidenceReference(
            f"ai_factor:{name}", EvidenceState.AVAILABLE, "ai_candidate_assessment",
            {"candidate_id": candidate.id, "score": round(ai_value, 3)},
        )
        # AI is an assessor, not an authority: evidence-grounded local factors
        # retain 80% ownership and all thresholds/penalties stay in code.
        local.score = _bounded(local.score * 0.8 + ai_value * 0.2)
        local.confidence = max(local.confidence, 0.65)
        local.evidence_refs.append(ai_ref)
        local.provenance = {**local.provenance, "ai_assessment_weight": 0.2, "ai_final_selection_ignored": True}
        score.components[name] = local.score
    _recalculate_score(score)


def set_ai_merge_provenance(candidate: Any, *, ai_score: float | None, merged_score: float | None, reason: str) -> None:
    score = candidate.candidate_score_v2
    if score is None:
        return
    score.provenance = {
        "mode": "code_owned_factor_fusion" if ai_score is not None else "deterministic_evidence_fusion",
        "fallback_used": ai_score is None,
        "fallback_reason": reason if ai_score is None else None,
        "local_score": round(score.final_score, 3),
        "ai_score": round(ai_score, 3) if ai_score is not None else None,
        "ai_overall_score_ignored": ai_score is not None,
        "ai_selected_ignored": ai_score is not None,
        "factor_assessment_weight": 0.2 if ai_score is not None else 0.0,
        "final_score_owner": "code",
    }


def _recalculate_score(score: CandidateScoreV2) -> None:
    contributions = {name: score.factors[name].score * weight for name, weight in FACTOR_WEIGHTS.items()}
    context = score.factors["context_debt"].score * CONTEXT_DEBT_WEIGHT
    score.raw_score = round(sum(contributions.values()) - context, 6)
    score.final_score = round(_bounded(score.raw_score - sum(item.amount for item in score.penalties)), 3)
    score.components = {name: item.score for name, item in score.factors.items()}
    score.diagnostics.update({
        "positive_contributions": {key: round(value, 3) for key, value in contributions.items()},
        "context_debt_deduction": round(context, 3),
        "penalty_total": round(sum(item.amount for item in score.penalties), 3),
    })


def _pass2_verification(candidate: Any) -> dict[str, Any] | None:
    wrapper = candidate.vision_pass2_evidence or {}
    result = wrapper.get("result") if isinstance(wrapper, dict) else None
    verification = result.get("verification") if isinstance(result, dict) else None
    if wrapper.get("status") not in {"completed", "partial"} or not isinstance(verification, dict):
        return None
    if float(verification.get("confidence", 0) or 0) < MIN_EDITORIAL_MULTIMODAL_CONFIDENCE:
        return None
    return verification


def _pass2_evidence_ref(candidate: Any, verification: dict[str, Any] | None) -> EvidenceReference:
    wrapper = candidate.vision_pass2_evidence or {}
    result = wrapper.get("result") if isinstance(wrapper, dict) else None
    if verification is None:
        return EvidenceReference(
            "vision_pass2", EvidenceState.UNAVAILABLE, "vision_pass2",
            {"status": str(wrapper.get("status") or "not_available"), "reason": wrapper.get("reason")},
        )
    return EvidenceReference(
        "vision_pass2", EvidenceState.AVAILABLE, "vision_pass2",
        {
            "candidate_id": candidate.id,
            "verification": dict(verification),
            "observation_count": len(result.get("observations", [])) if isinstance(result, dict) else 0,
            "schema_version": result.get("schema_version") if isinstance(result, dict) else None,
        },
    )


def _visual_factor_evidence(
    candidate: Any, scores: dict[str, float], visual_analysis: dict[str, Any] | None,
) -> tuple[EvidenceReference, float, float, float]:
    provenance = candidate.multimodal_provenance or {}
    raw_local_visual = provenance.get("visual_evidence", []) if isinstance(provenance, dict) else []
    pass2 = candidate.vision_pass2_evidence or {}
    result = pass2.get("result") if isinstance(pass2, dict) else None
    raw_observations = result.get("observations", []) if isinstance(result, dict) else []
    local_visual = [
        item for item in raw_local_visual
        if isinstance(item, dict) and float(item.get("confidence", (item.get("observation") or {}).get("confidence", 0)) or 0) >= MIN_EDITORIAL_MULTIMODAL_CONFIDENCE
    ]
    observations = [
        item for item in raw_observations
        if isinstance(item, dict) and float(item.get("confidence", 0) or 0) >= MIN_EDITORIAL_MULTIMODAL_CONFIDENCE
    ]
    global_available = bool((visual_analysis or {}).get("status") == "completed" and (visual_analysis or {}).get("subject_keyframes"))
    available = bool(local_visual or observations or global_available)
    confidences = [
        float(item.get("confidence", 0))
        for item in [*local_visual, *observations]
        if isinstance(item, dict) and item.get("confidence") is not None
    ]
    confidence = max(confidences, default=0.65 if global_available else 0.2)
    roles = (provenance.get("generation", {}) or {}).get("reasons", []) if available and isinstance(provenance, dict) else []
    role_bonus = 0.0
    joined = " ".join(str(item) for item in roles)
    for role in ("action", "reaction", "payoff"):
        if role in joined:
            role_bonus += 18
    base = float(scores.get("scene_structure", 0)) if available else 50.0
    interest = _bounded(base * 0.65 + role_bonus)
    risks = [str(item.get("composition_risk") or "unknown") for item in observations if isinstance(item, dict)]
    if not available:
        vertical = 50.0
    elif any(item in {"target_missing", "crowded", "text_overlap", "face_edge"} for item in risks):
        vertical = 38.0
    else:
        vertical = 82.0
    return (
        EvidenceReference(
            "visual_editorial_evidence", EvidenceState.AVAILABLE if available else EvidenceState.UNAVAILABLE,
            "multimodal_candidate_provenance",
            {"local_evidence_count": len(local_visual), "pass2_observation_count": len(observations), "global_visual_available": global_available, "composition_risks": risks},
        ),
        interest,
        vertical,
        confidence,
    )


def _multimodal_confidence(
    transcript_confidence: float, audio_ref: EvidenceReference, visual_ref: EvidenceReference,
    visual_confidence: float, pass2: dict[str, Any] | None,
) -> float:
    audio_confidence = 0.75 if audio_ref.state is EvidenceState.AVAILABLE else 0.25
    visual_value = visual_confidence if visual_ref.state is EvidenceState.AVAILABLE else 0.2
    pass2_value = float(pass2.get("confidence", 0)) if pass2 is not None else 0.2
    return max(0.0, min(1.0, transcript_confidence * 0.48 + audio_confidence * 0.22 + visual_value * 0.20 + pass2_value * 0.10))


def _modalities_available(candidate: Any, visual_ref: EvidenceReference) -> list[str]:
    result = ["text", "audio"]
    if visual_ref.state is EvidenceState.AVAILABLE:
        result.append("visual")
    return result


def build_composition_intent(candidate: Any) -> dict[str, Any]:
    """Translate only observed multimodal evidence into the existing composition hand-off."""

    provenance = candidate.multimodal_provenance or {}
    raw_visual = [item for item in provenance.get("visual_evidence", []) if isinstance(item, dict)] if isinstance(provenance, dict) else []
    visual = []
    for item in raw_visual:
        nested = item.get("observation")
        if isinstance(nested, dict):
            visual.append({
                **nested,
                "timestamp": nested.get("timestamp", item.get("start_seconds", 0)),
                "confidence": nested.get("confidence", item.get("confidence", 0)),
            })
        else:
            visual.append(item)
    wrapper = candidate.vision_pass2_evidence or {}
    result = wrapper.get("result") if isinstance(wrapper, dict) else None
    observations = [item for item in result.get("observations", []) if isinstance(item, dict)] if isinstance(result, dict) else []
    combined = [
        item for item in [*visual, *observations]
        if float(item.get("confidence", 0) or 0) >= MIN_EDITORIAL_MULTIMODAL_CONFIDENCE
    ]
    source = "vision_pass2" if observations else "multimodal_candidate_provenance"
    status = "available" if combined else "unavailable"
    confidence = max((float(item.get("confidence", 0)) for item in combined), default=0.0)
    subjects = {str(item.get("primary_subject") or "none") for item in combined}
    scenes = {str(item.get("scene_type") or "UNKNOWN") for item in combined}
    face_count = max((int(item.get("visible_face_count", 0)) for item in combined), default=0)
    reactions = [str(item.get("reaction") or "none") for item in combined]
    risks = [str(item.get("composition_risk") or "unknown") for item in combined]
    refs = [
        {
            "source": source,
            "candidate_id": candidate.id,
            "timestamps": sorted({float(item.get("timestamp", 0)) for item in observations}),
            "observation_count": len(combined),
        }
    ]

    def intent(value: Any, reason: str) -> dict[str, Any]:
        return {
            "value": value,
            "confidence": round(confidence, 6),
            "evidence_refs": refs,
            "provenance": {"mode": "observed_evidence_only", "source": source, "reason": reason},
        }

    risky = any(item in {"face_edge", "target_missing", "crowded", "text_overlap"} for item in risks)
    return {
        "schema_version": "6D.composition-intent.1",
        "evidence_status": status,
        "active_speaker": intent(
            any(item.get("action") == "speaking" and item.get("primary_subject") in {"face", "person"} for item in combined),
            "speaking person/face was explicitly observed",
        ),
        "important_subject_or_object": intent(
            next((item for item in ("object", "person", "face", "group") if item in subjects), None),
            "primary_subject classification",
        ),
        "screen_or_product": intent(
            "screen" if "screen" in subjects or "PRESENTATION_SCREEN" in scenes else ("product" if scenes & {"PRODUCT_DEMO", "HANDS_ON_DEMO"} else None),
            "screen/product scene classification",
        ),
        "reaction": intent(next((item for item in reactions if item not in {"none", "unknown"}), None), "visible reaction classification"),
        "multiple_subjects": intent(face_count > 1 or "group" in subjects, "visible face count/group classification"),
        "vertical_viability": intent("risky" if risky else ("viable" if combined else "unknown"), "explicit composition risks"),
    }


def boundary_multimodal_context(candidate: Any) -> dict[str, Any]:
    """Persist payoff timing so later boundary consumers do not infer it from text end."""

    provenance = candidate.multimodal_provenance or {}
    generation = provenance.get("generation", {}) if isinstance(provenance, dict) else {}
    anchors = generation.get("anchors", {}) if isinstance(generation, dict) else {}
    pass2 = candidate.vision_pass2_evidence or {}
    result = pass2.get("result") if isinstance(pass2, dict) else None
    request = result.get("request", {}) if isinstance(result, dict) else {}
    pass2_anchors = request.get("anchors", {}) if isinstance(request, dict) else {}
    verification = result.get("verification", {}) if isinstance(result, dict) else {}
    reasons = generation.get("reasons", []) if isinstance(generation, dict) else []
    linked_payoff = any(any(role in str(item) for role in ("payoff", "reaction")) for item in reasons)
    payoff_times = []
    for value in (anchors.get("payoff"), pass2_anchors.get("payoff")):
        if value is None:
            continue
        try:
            payoff_times.append(float(value))
        except (TypeError, ValueError):
            continue
    if linked_payoff:
        for section in (provenance.get("audio_evidence", []), provenance.get("visual_evidence", [])):
            for item in section if isinstance(section, list) else []:
                if not isinstance(item, dict):
                    continue
                for key in ("end_seconds", "timestamp", "time_seconds"):
                    value = item.get(key)
                    if value is None:
                        continue
                    try:
                        payoff_times.append(float(value))
                    except (TypeError, ValueError):
                        continue
    preserve_until = max([candidate.end, *payoff_times])
    return {
        "schema_version": "6D.boundary-context.1",
        "preserve_until_seconds": round(preserve_until, 3),
        "visual_payoff_verified": verification.get("payoff_visible") is True,
        "visual_reaction_verified": verification.get("reaction_visible") is True,
        "multimodal_payoff_grounded": linked_payoff or verification.get("payoff_visible") is True or verification.get("reaction_visible") is True,
        "audio_or_visual_payoff_times": sorted(round(item, 3) for item in payoff_times),
        "confidence": round(float(verification.get("confidence", 0)) if verification else 0.0, 6),
        "evidence_refs": [
            {"source": "multimodal_candidate_provenance", "analysis_run_id": provenance.get("analysis_run_id")},
            {"source": "vision_pass2", "status": pass2.get("status")},
        ],
        "provenance": {"mode": "observed_evidence_only", "candidate_id": candidate.id},
    }


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, value))
