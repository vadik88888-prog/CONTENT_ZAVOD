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


CANDIDATE_QUALITY_SCHEMA_VERSION = "5B.1"


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
    SPEECH_CLARITY_EVIDENCE_UNAVAILABLE = "SPEECH_CLARITY_EVIDENCE_UNAVAILABLE"
    VERTICAL_COMPOSITION_IMPOSSIBLE = "VERTICAL_COMPOSITION_IMPOSSIBLE"
    VISUAL_EVIDENCE_UNAVAILABLE = "VISUAL_EVIDENCE_UNAVAILABLE"
    DURATION_OUT_OF_RANGE = "DURATION_OUT_OF_RANGE"
    LEGACY_UNASSESSED = "LEGACY_UNASSESSED"


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
class CandidateScoreV2:
    schema_version: str
    config_version: str
    components: dict[str, float]
    penalties: list[ScorePenalty]
    raw_score: float
    final_score: float
    evidence_refs: list[EvidenceReference]
    provenance: dict[str, Any]
    state: EligibilityState = EligibilityState.ASSESSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "state": self.state.value,
            "component_scores": {key: round(value, 3) for key, value in self.components.items()},
            "penalties": [item.to_dict() for item in self.penalties],
            "raw_score": round(self.raw_score, 3),
            "final_score": round(self.final_score, 3),
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "provenance": self.provenance,
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
        return cls(
            schema_version=str(data.get("schema_version") or CANDIDATE_QUALITY_SCHEMA_VERSION),
            config_version=str(data.get("config_version") or "unknown"),
            components={str(key): float(value) for key, value in dict(data.get("component_scores") or {}).items()},
            penalties=[ScorePenalty.from_dict(item) for item in penalties if isinstance(item, dict)] if isinstance(penalties, list) else [],
            raw_score=float(data.get("raw_score") or 0),
            final_score=float(data.get("final_score") or 0),
            evidence_refs=[EvidenceReference.from_dict(item) for item in evidence if isinstance(item, dict)] if isinstance(evidence, list) else [],
            provenance=dict(data.get("provenance") or {}),
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
    boundary_completion = boundary.get("semantic_completion")
    if boundary_completion is not None:
        boundary_completion = float(boundary_completion)
        if boundary_completion <= 1:
            boundary_completion *= 100
        # A resolved semantic boundary is the final source-range decision.  A
        # lower rough StoryUnit precheck must not overturn that later evidence.
        semantic_completeness = max(float(semantic_completeness or 0), boundary_completion)
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
        evidence.append(EvidenceReference("speech_clarity", EvidenceState.AVAILABLE, "transcript_features", {"transcript_confidence": round(clarity, 3)}))
        if clarity < 0.45:
            reason(EligibilityReasonCode.AUDIO_UNINTELLIGIBLE)

    visual_status = str((visual_analysis or {}).get("status") or "unavailable")
    visual_keyframes = (visual_analysis or {}).get("subject_keyframes", [])
    if visual_status == "completed" and visual_keyframes:
        evidence.append(EvidenceReference("visual_viability", EvidenceState.AVAILABLE, "visual_analysis", {"status": visual_status, "keyframe_count": len(visual_keyframes)}))
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
    payoff_present = bool(payoff_text) or any(marker in ending for marker in _PAYOFF_MARKERS) or (
        natural_end and float(boundary.get("semantic_completion", features.get("completeness_score", 0))) >= 0.5
    )
    hook_evidence = EvidenceReference("hook", EvidenceState.AVAILABLE, "story_unit" if hook_text else "transcript_features", {
        "hook_strength": round(hook_score, 3), "hook_text_available": bool(hook_text),
    })
    payoff_evidence = EvidenceReference("payoff", EvidenceState.AVAILABLE if payoff_present else EvidenceState.UNAVAILABLE, "story_unit" if payoff_text else "semantic_boundary", {
        "payoff_present": payoff_present, "story_payoff_available": bool(payoff_text), "natural_end": natural_end,
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
    visual_available = bool((visual_analysis or {}).get("status") == "completed" and (visual_analysis or {}).get("subject_keyframes"))
    duration_fit = 100.0
    if min_duration_seconds is not None and max_duration_seconds is not None:
        target = (min_duration_seconds + max_duration_seconds) / 2
        duration_fit = _bounded(100 - abs(candidate.duration - target) / max(1.0, (max_duration_seconds - min_duration_seconds) / 2) * 45)
    information = float(semantic.get("information_density", 0))
    if information <= 1:
        information *= 100
    emotional = _bounded(float(features.get("audio_energy", 0)) * 100 + float(features.get("exclamation_count", 0)) * 8)
    novelty = _bounded((100 - float(features.get("repetition_score", 0)) * 100) * 0.55 + hook * 0.45)
    components = {
        "self_containment": _bounded((completeness + context_independence + boundary_score) / 3),
        "hook_strength": _bounded(hook),
        "payoff_strength": 82.0 if payoff_present else 0.0,
        "narrative_arc": _bounded((completeness + boundary_score) / 2),
        "informational_value": _bounded(information or min(100.0, float(features.get("word_count", 0)) * 3)),
        "emotional_intensity": emotional,
        "novelty_or_conflict": novelty,
        "speech_clarity": _bounded(float(scores.get("clarity", 0))),
        "visual_viability": _bounded(float(scores.get("scene_structure", 0)) if visual_available else 0.0),
        "pacing_density": _bounded((float(scores.get("pacing", 0)) + float(scores.get("speech_density", 0))) / 2),
        "platform_fit": duration_fit,
    }
    weights = {
        "self_containment": 0.16, "hook_strength": 0.14, "payoff_strength": 0.12,
        "narrative_arc": 0.10, "informational_value": 0.10, "emotional_intensity": 0.08,
        "novelty_or_conflict": 0.08, "speech_clarity": 0.08, "visual_viability": 0.06,
        "pacing_density": 0.04, "platform_fit": 0.04,
    }
    raw = sum(components[name] * weights[name] for name in weights)
    penalties: list[ScorePenalty] = []

    def penalty(code: str, amount: float, refs: list[EvidenceReference]) -> None:
        if amount > 0:
            penalties.append(ScorePenalty(code, amount, refs))

    repetition = float(features.get("repetition_score", 0)) * 15
    filler = float(features.get("filler_word_ratio", 0)) * 18
    context_count = sum(code in decision.reason_codes for code in (
        EligibilityReasonCode.UNRESOLVED_PRONOUN, EligibilityReasonCode.UNNAMED_ENTITY,
        EligibilityReasonCode.ANSWER_WITHOUT_QUESTION_CONTEXT, EligibilityReasonCode.REFERENCES_EARLIER_CONTENT,
        EligibilityReasonCode.UNDEFINED_TERM_OR_SETUP,
    ))
    penalty("INTERNAL_REPETITION", repetition, [EvidenceReference("repetition_score", EvidenceState.AVAILABLE, "transcript_features", {"value": round(float(features.get("repetition_score", 0)), 3)})])
    penalty("FILLER_WORDS", filler, [EvidenceReference("filler_word_ratio", EvidenceState.AVAILABLE, "transcript_features", {"value": round(float(features.get("filler_word_ratio", 0)), 3)})])
    penalty("CONTEXT_DEBT", min(25.0, context_count * 10.0), [item for item in decision.evidence_refs if item.code in {code.value for code in decision.reason_codes}])
    if not visual_available:
        penalty("VISUAL_EVIDENCE_UNAVAILABLE", 8.0, [item for item in decision.evidence_refs if item.code == "visual_viability"])
    if float(scores.get("clarity", 0)) < 45:
        penalty("LOW_SPEECH_CLARITY", 15.0, [item for item in decision.evidence_refs if item.code == "speech_clarity"])
    final = _bounded(raw - sum(item.amount for item in penalties))
    return CandidateScoreV2(
        schema_version=CANDIDATE_QUALITY_SCHEMA_VERSION,
        config_version=config_version,
        components=components,
        penalties=penalties,
        raw_score=raw,
        final_score=final,
        evidence_refs=[hook_evidence, payoff_evidence, *decision.evidence_refs],
        provenance={"mode": "deterministic_local", "fallback_used": True, "ai_merge": "not_attempted"},
    )


def set_ai_merge_provenance(candidate: Any, *, ai_score: float | None, merged_score: float | None, reason: str) -> None:
    score = candidate.candidate_score_v2
    if score is None:
        return
    score.provenance = {
        "mode": "local_ai_merge" if ai_score is not None else "deterministic_local",
        "fallback_used": ai_score is None,
        "fallback_reason": reason if ai_score is None else None,
        "local_score": round(score.final_score, 3),
        "ai_score": round(ai_score, 3) if ai_score is not None else None,
        "merge_weights": {"local": 0.55, "ai": 0.45} if ai_score is not None else {"local": 1.0},
    }
    if merged_score is not None:
        score.final_score = _bounded(merged_score)


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, value))
