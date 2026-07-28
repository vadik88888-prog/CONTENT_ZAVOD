"""Goal 5B: grounded, comparative content-strength analysis.

The module deliberately describes *potential* inside one source.  It does not
predict platform reach, change text, or make render decisions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from app.audio_features import window_audio_features
from app.content_understanding import StoryUnit
from app.models import Candidate


VIRALITY_SCHEMA_VERSION = "5B.2"
FEATURE_NAMES = (
    "hook_strength", "curiosity_gap", "emotional_intensity", "emotional_progression",
    "conflict_tension", "surprise_novelty", "specificity", "clarity", "relatability",
    "usefulness", "controversy_potential", "quotability", "narrative_momentum",
    "payoff_strength", "ending_satisfaction", "standalone_strength", "context_independence",
    "speech_energy", "pacing_quality", "information_density", "repetition_penalty",
    "filler_penalty", "confusion_penalty", "slow_start_penalty", "weak_ending_penalty",
    "platform_fit", "publishability", "retention_potential", "analysis_confidence",
)

HOOK_TYPES = frozenset({
    "bold_claim", "question", "conflict", "surprise", "confession", "warning", "promise",
    "problem", "consequence", "emotional_statement", "visual_action", "quote", "informational",
    "weak_contextual", "none",
})
CONFLICT_TYPES = frozenset({
    "person_vs_person", "person_vs_self", "person_vs_system", "belief_vs_reality",
    "expectation_vs_result", "problem_vs_solution", "risk_vs_reward", "none",
})
PAYOFF_TYPES = frozenset({
    "answer", "insight", "conclusion", "reveal", "consequence", "emotional_release", "solution",
    "punchline", "transformation", "warning", "none",
})
EMOTIONS = frozenset({
    "excitement", "anger", "fear", "tension", "sadness", "inspiration", "hope", "surprise",
    "humor", "empathy", "determination", "frustration", "relief", "neutral", "mixed",
})
RETENTION_LEVELS = frozenset({"low", "medium", "high", "very_high"})
PUBLISHABILITY_LEVELS = frozenset({"ready", "limited", "blocked"})
ELIGIBILITY_STATUSES = frozenset({
    "publishable_now", "publishable_with_minor_adjustment", "needs_reconstruction", "weak", "rejected",
})
VIRALITY_COMPONENTS = (
    "hook", "curiosity", "emotion", "conflict", "specificity", "novelty", "usefulness", "quotability",
    "momentum", "payoff", "retention", "publishability",
)
VIRALITY_PENALTIES = (
    "slow_start", "context_dependency", "unresolved_curiosity", "missing_payoff", "repetition", "filler",
    "confusion", "dead_zone", "weak_ending", "boundary_risk", "semantic_duplication",
)
PENALTY_WEIGHTS = {
    "slow_start": 0.06, "context_dependency": 0.08, "unresolved_curiosity": 0.06, "missing_payoff": 0.10,
    "repetition": 0.04, "filler": 0.05, "confusion": 0.06, "dead_zone": 0.10,
    "weak_ending": 0.07, "boundary_risk": 0.14, "semantic_duplication": 0.06,
}
RETENTION_ZONE_DEFINITIONS = (
    ("opening", 0.00, 0.10),
    ("early", 0.10, 0.25),
    ("mid", 0.25, 0.50),
    ("late", 0.50, 0.75),
    ("pre_payoff", 0.75, 0.90),
    ("ending", 0.90, 1.00),
)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_GREETING_PREFIXES = (
    "привет", "добрый", "здравствуйте", "всем привет", "hello", "hi ", "welcome",
)
_TECHNICAL_PREFIXES = (
    "сегодня мы", "в этом видео", "сейчас я расскажу", "как я уже", "today we", "in this video",
)
_CONTEXTUAL_PREFIXES = (
    "и ", "а ", "но ", "поэтому ", "это ", "он ", "она ", "они ", "and ", "but ", "so ",
    "this ", "that ", "they ", "it ",
)
_QUESTION_WORDS = ("почему", "как", "что будет", "зачем", "кто", "why", "how", "what happens", "what if")
_ANSWER_MARKERS = (
    "потому", "ответ", "вот почему", "значит", "поэтому", "вывод", "итог", "суть", "потому что",
    "because", "the answer", "that is why", "therefore", "the point", "means",
)
_PAYOFF_MARKERS = (
    "вывод", "итог", "поэтому", "значит", "вот почему", "разница", "победа", "решение", "ответ",
    "therefore", "the point", "that is why", "solution", "answer", "conclusion",
)
_SOFT_PAYOFF_MARKERS = (
    "это и есть", "всегда", "никогда", "побед", "умр", "решит", "нужн", "долж", "вырв", "сраж", "борь",
    "this is", "always", "never", "win", "lose", "must", "choose", "act", "therefore",
)
_CONFLICT_MARKERS = (
    "но", "против", "или", "риск", "ошибк", "поражен", "страх", "потер", "бор", "ад", "цен",
    "but", "versus", "risk", "fail", "fear", "lose", "fight", "mistake", "or ",
)
_ACTION_WORDS = (
    "сдел", "начн", "выбер", "действ", "работ", "уч", "проверь", "реш", "build", "start", "choose",
    "do ", "make ", "learn", "check", "act", "work",
)
_ABSTRACT_WORDS = frozenset({
    "жизнь", "успех", "смысл", "сила", "время", "всё", "каждый", "всегда", "никогда",
    "life", "success", "meaning", "power", "everything", "always", "never",
})
_EMOTION_LEXICON: dict[str, tuple[str, ...]] = {
    "excitement": ("побед", "вперёд", "давай", "вместе", "win", "go", "let's"),
    "anger": ("злю", "ненав", "ярост", "anger", "hate"),
    "fear": ("страх", "бою", "ужас", "fear", "afraid"),
    "tension": ("риск", "бой", "последн", "схват", "на грани", "risk", "fight", "last"),
    "sadness": ("потер", "больно", "груст", "один", "loss", "hurt", "alone"),
    "inspiration": ("можем", "верь", "шанс", "смож", "can", "believe", "chance"),
    "hope": ("надеж", "будет", "выйдем", "hope", "will"),
    "surprise": ("неожидан", "вдруг", "никто", "surprise", "suddenly"),
    "humor": ("смеш", "шут", "funny", "joke"),
    "empathy": ("понима", "каждый", "тебя", "understand", "you"),
    "determination": ("долж", "готов", "не сдам", "will", "ready", "must"),
    "frustration": ("не могу", "устал", "слом", "can't", "tired", "broken"),
    "relief": ("наконец", "получилось", "спас", "finally", "relief"),
}


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 6)


@dataclass(slots=True)
class FeatureEvidence:
    """Grounded input for one score; evidence never points outside a candidate."""

    source: str
    raw_value: float | str | bool | None
    normalized_value: float
    confidence: float
    segment_ids: list[int] = field(default_factory=list)
    excerpt: str = ""
    feature_name: str = ""

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.source:
            raise ValueError("FeatureEvidence requires a source.")
        if not 0 <= self.normalized_value <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("FeatureEvidence scores must be bounded.")
        if allowed_segment_ids is not None and not set(self.segment_ids).issubset(allowed_segment_ids):
            raise ValueError("Virality evidence must stay inside candidate transcript segments.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureEvidence":
        result = cls(
            source=str(data.get("source") or "unknown"), raw_value=data.get("raw_value"),
            normalized_value=_bounded(float(data.get("normalized_value", 0))),
            confidence=_bounded(float(data.get("confidence", 0))),
            segment_ids=[int(item) for item in data.get("segment_ids", [])],
            excerpt=str(data.get("excerpt") or "")[:800],
            feature_name=str(data.get("feature_name") or ""),
        )
        result.validate()
        return result


@dataclass(slots=True)
class FeatureScore:
    """Quality and confidence intentionally remain separate values."""

    score: float
    confidence: float
    evidence: list[FeatureEvidence]
    explanation: str

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not 0 <= self.score <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("FeatureScore must be bounded from zero to one.")
        if not self.explanation:
            raise ValueError("FeatureScore requires an explanation.")
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self, feature_name: str | None = None) -> dict[str, Any]:
        self.validate()
        return {
            "score": round(self.score, 6), "confidence": round(self.confidence, 6),
            "evidence": [
                {
                    **item.to_dict(),
                    "feature_name": feature_name or item.feature_name,
                }
                for item in self.evidence
            ],
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureScore":
        result = cls(
            score=_bounded(float(data.get("score", 0))), confidence=_bounded(float(data.get("confidence", 0))),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
            explanation=str(data.get("explanation") or "Нет объяснения."),
        )
        result.validate()
        return result


@dataclass(slots=True)
class HookAssessment:
    hook_type: str
    hook_strength: FeatureScore
    immediate_clarity: FeatureScore
    curiosity_gap: FeatureScore
    specificity: FeatureScore
    stakes: FeatureScore
    emotional_charge: FeatureScore
    first_sentence_quality: FeatureScore
    context_dependency: FeatureScore
    slow_start_penalty: FeatureScore
    time_to_value_seconds: float
    curiosity_opened: bool
    curiosity_resolved: bool
    resolution_timestamp: float | None
    curiosity_resolution_quality: FeatureScore
    unresolved_curiosity_penalty: FeatureScore
    weak_opening_reason: str | None
    evidence: list[FeatureEvidence]

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if self.hook_type not in HOOK_TYPES or self.time_to_value_seconds < 0:
            raise ValueError("HookAssessment contains an invalid hook type or time-to-value.")
        if self.curiosity_resolved and not self.curiosity_opened:
            raise ValueError("Curiosity cannot resolve when it was never opened.")
        for score in (
            self.hook_strength, self.immediate_clarity, self.curiosity_gap, self.specificity, self.stakes,
            self.emotional_charge, self.first_sentence_quality, self.context_dependency, self.slow_start_penalty,
            self.curiosity_resolution_quality, self.unresolved_curiosity_penalty,
        ):
            score.validate(allowed_segment_ids)
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        for name, value in tuple(result.items()):
            if isinstance(value, dict) and {"score", "confidence", "evidence", "explanation"}.issubset(value):
                result[name] = getattr(self, name).to_dict()
        result["evidence"] = [item.to_dict() for item in self.evidence]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookAssessment":
        result = cls(
            hook_type=str(data.get("hook_type") or "none"),
            hook_strength=FeatureScore.from_dict(dict(data.get("hook_strength") or {})),
            immediate_clarity=FeatureScore.from_dict(dict(data.get("immediate_clarity") or {})),
            curiosity_gap=FeatureScore.from_dict(dict(data.get("curiosity_gap") or {})),
            specificity=FeatureScore.from_dict(dict(data.get("specificity") or {})),
            stakes=FeatureScore.from_dict(dict(data.get("stakes") or {})),
            emotional_charge=FeatureScore.from_dict(dict(data.get("emotional_charge") or {})),
            first_sentence_quality=FeatureScore.from_dict(dict(data.get("first_sentence_quality") or {})),
            context_dependency=FeatureScore.from_dict(dict(data.get("context_dependency") or {})),
            slow_start_penalty=FeatureScore.from_dict(dict(data.get("slow_start_penalty") or {})),
            time_to_value_seconds=max(0.0, float(data.get("time_to_value_seconds", 0))),
            curiosity_opened=bool(data.get("curiosity_opened", False)),
            curiosity_resolved=bool(data.get("curiosity_resolved", False)),
            resolution_timestamp=(float(data["resolution_timestamp"]) if data.get("resolution_timestamp") is not None else None),
            curiosity_resolution_quality=FeatureScore.from_dict(dict(data.get("curiosity_resolution_quality") or {})),
            unresolved_curiosity_penalty=FeatureScore.from_dict(dict(data.get("unresolved_curiosity_penalty") or {})),
            weak_opening_reason=(str(data["weak_opening_reason"]) if data.get("weak_opening_reason") else None),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class EmotionalArc:
    opening_emotion: str
    dominant_emotion: str
    peak_emotion: str
    ending_emotion: str
    emotional_start_level: FeatureScore
    emotional_peak_level: FeatureScore
    emotional_end_level: FeatureScore
    peak_timestamp: float
    escalation_strength: FeatureScore
    emotional_change: FeatureScore
    emotional_consistency: FeatureScore
    emotional_payoff: FeatureScore
    flatness_penalty: FeatureScore
    evidence: list[FeatureEvidence]

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if any(item not in EMOTIONS for item in (self.opening_emotion, self.dominant_emotion, self.peak_emotion, self.ending_emotion)):
            raise ValueError("EmotionalArc contains an unsupported emotion.")
        if self.peak_timestamp < 0:
            raise ValueError("EmotionalArc peak timestamp cannot be negative.")
        for score in (
            self.emotional_start_level, self.emotional_peak_level, self.emotional_end_level,
            self.escalation_strength, self.emotional_change, self.emotional_consistency,
            self.emotional_payoff, self.flatness_penalty,
        ):
            score.validate(allowed_segment_ids)
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionalArc":
        result = cls(
            opening_emotion=str(data.get("opening_emotion") or "neutral"),
            dominant_emotion=str(data.get("dominant_emotion") or "neutral"),
            peak_emotion=str(data.get("peak_emotion") or "neutral"),
            ending_emotion=str(data.get("ending_emotion") or "neutral"),
            emotional_start_level=FeatureScore.from_dict(dict(data.get("emotional_start_level") or {})),
            emotional_peak_level=FeatureScore.from_dict(dict(data.get("emotional_peak_level") or {})),
            emotional_end_level=FeatureScore.from_dict(dict(data.get("emotional_end_level") or {})),
            peak_timestamp=max(0.0, float(data.get("peak_timestamp", 0))),
            escalation_strength=FeatureScore.from_dict(dict(data.get("escalation_strength") or {})),
            emotional_change=FeatureScore.from_dict(dict(data.get("emotional_change") or {})),
            emotional_consistency=FeatureScore.from_dict(dict(data.get("emotional_consistency") or {})),
            emotional_payoff=FeatureScore.from_dict(dict(data.get("emotional_payoff") or {})),
            flatness_penalty=FeatureScore.from_dict(dict(data.get("flatness_penalty") or {})),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class ConflictAssessment:
    conflict_type: str
    conflict_presence: FeatureScore
    conflict_strength: FeatureScore
    stakes_clarity: FeatureScore
    stakes_magnitude: FeatureScore
    conflict_resolution: FeatureScore
    conflict_payoff: FeatureScore
    evidence: list[FeatureEvidence]

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if self.conflict_type not in CONFLICT_TYPES:
            raise ValueError("Unsupported conflict type.")
        for score in (
            self.conflict_presence, self.conflict_strength, self.stakes_clarity,
            self.stakes_magnitude, self.conflict_resolution, self.conflict_payoff,
        ):
            score.validate(allowed_segment_ids)
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictAssessment":
        result = cls(
            conflict_type=str(data.get("conflict_type") or "none"),
            conflict_presence=FeatureScore.from_dict(dict(data.get("conflict_presence") or {})),
            conflict_strength=FeatureScore.from_dict(dict(data.get("conflict_strength") or {})),
            stakes_clarity=FeatureScore.from_dict(dict(data.get("stakes_clarity") or {})),
            stakes_magnitude=FeatureScore.from_dict(dict(data.get("stakes_magnitude") or {})),
            conflict_resolution=FeatureScore.from_dict(dict(data.get("conflict_resolution") or {})),
            conflict_payoff=FeatureScore.from_dict(dict(data.get("conflict_payoff") or {})),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class QuoteCandidate:
    text: str
    segment_ids: list[int]
    start: float
    end: float
    memorability: FeatureScore
    clarity: FeatureScore
    emotional_strength: FeatureScore
    standalone_quality: FeatureScore
    cliche_risk: FeatureScore

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.text or self.end < self.start:
            raise ValueError("QuoteCandidate must contain a valid range and text.")
        if allowed_segment_ids is not None and not set(self.segment_ids).issubset(allowed_segment_ids):
            raise ValueError("QuoteCandidate must stay inside the candidate.")
        for score in (self.memorability, self.clarity, self.emotional_strength, self.standalone_quality, self.cliche_risk):
            score.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuoteCandidate":
        result = cls(
            text=str(data.get("text") or ""), segment_ids=[int(item) for item in data.get("segment_ids", [])],
            start=float(data.get("start", 0)), end=float(data.get("end", 0)),
            memorability=FeatureScore.from_dict(dict(data.get("memorability") or {})),
            clarity=FeatureScore.from_dict(dict(data.get("clarity") or {})),
            emotional_strength=FeatureScore.from_dict(dict(data.get("emotional_strength") or {})),
            standalone_quality=FeatureScore.from_dict(dict(data.get("standalone_quality") or {})),
            cliche_risk=FeatureScore.from_dict(dict(data.get("cliche_risk") or {})),
        )
        result.validate()
        return result


@dataclass(slots=True)
class PayoffAssessment:
    payoff_present: bool
    payoff_type: str
    payoff_strength: FeatureScore
    payoff_timestamp: float | None
    setup_payoff_alignment: FeatureScore
    emotional_resolution: FeatureScore
    informational_resolution: FeatureScore
    surprise_resolution: FeatureScore
    ending_satisfaction: FeatureScore
    payoff_missing_reason: str | None
    evidence: list[FeatureEvidence]

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if self.payoff_type not in PAYOFF_TYPES:
            raise ValueError("Unsupported payoff type.")
        if self.payoff_present != (self.payoff_type != "none"):
            raise ValueError("Payoff type must agree with payoff presence.")
        for score in (
            self.payoff_strength, self.setup_payoff_alignment, self.emotional_resolution,
            self.informational_resolution, self.surprise_resolution, self.ending_satisfaction,
        ):
            score.validate(allowed_segment_ids)
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PayoffAssessment":
        payoff_type = str(data.get("payoff_type") or "none")
        result = cls(
            payoff_present=bool(data.get("payoff_present", payoff_type != "none")), payoff_type=payoff_type,
            payoff_strength=FeatureScore.from_dict(dict(data.get("payoff_strength") or {})),
            payoff_timestamp=(float(data["payoff_timestamp"]) if data.get("payoff_timestamp") is not None else None),
            setup_payoff_alignment=FeatureScore.from_dict(dict(data.get("setup_payoff_alignment") or {})),
            emotional_resolution=FeatureScore.from_dict(dict(data.get("emotional_resolution") or {})),
            informational_resolution=FeatureScore.from_dict(dict(data.get("informational_resolution") or {})),
            surprise_resolution=FeatureScore.from_dict(dict(data.get("surprise_resolution") or {})),
            ending_satisfaction=FeatureScore.from_dict(dict(data.get("ending_satisfaction") or {})),
            payoff_missing_reason=(str(data["payoff_missing_reason"]) if data.get("payoff_missing_reason") else None),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class ViralityFeatureProfile:
    schema_version: str
    candidate_id: str
    story_unit_id: str
    content_strategy: str
    features: dict[str, FeatureScore]
    hook_assessment: HookAssessment
    emotional_arc: EmotionalArc
    conflict_assessment: ConflictAssessment
    payoff_assessment: PayoffAssessment
    quote_candidates: list[QuoteCandidate]
    analysis_confidence: FeatureScore
    analysis_mode: str = "deterministic"
    warnings: list[str] = field(default_factory=list)

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if self.schema_version != VIRALITY_SCHEMA_VERSION or not self.candidate_id or not self.story_unit_id:
            raise ValueError("ViralityFeatureProfile identity or schema is invalid.")
        if set(self.features) != set(FEATURE_NAMES):
            raise ValueError("ViralityFeatureProfile must expose every required feature.")
        for item in self.features.values():
            item.validate(allowed_segment_ids)
        self.hook_assessment.validate(allowed_segment_ids)
        self.emotional_arc.validate(allowed_segment_ids)
        self.conflict_assessment.validate(allowed_segment_ids)
        self.payoff_assessment.validate(allowed_segment_ids)
        for quote in self.quote_candidates:
            quote.validate(allowed_segment_ids)
        self.analysis_confidence.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "story_unit_id": self.story_unit_id,
            "content_strategy": self.content_strategy,
            "features": {name: value.to_dict(name) for name, value in self.features.items()},
            "hook_assessment": self.hook_assessment.to_dict(),
            "emotional_arc": self.emotional_arc.to_dict(),
            "conflict_assessment": self.conflict_assessment.to_dict(),
            "payoff_assessment": self.payoff_assessment.to_dict(),
            "quote_candidates": [item.to_dict() for item in self.quote_candidates],
            "analysis_confidence": self.analysis_confidence.to_dict(),
            "analysis_mode": self.analysis_mode,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViralityFeatureProfile":
        raw_features = data.get("features", {})
        features = {
            name: FeatureScore.from_dict(dict(raw_features.get(name) or {}))
            for name in FEATURE_NAMES
        } if isinstance(raw_features, dict) else {}
        result = cls(
            schema_version=str(data.get("schema_version") or ""),
            candidate_id=str(data.get("candidate_id") or ""),
            story_unit_id=str(data.get("story_unit_id") or ""),
            content_strategy=str(data.get("content_strategy") or "generic_fallback"),
            features=features,
            hook_assessment=HookAssessment.from_dict(dict(data.get("hook_assessment") or {})),
            emotional_arc=EmotionalArc.from_dict(dict(data.get("emotional_arc") or {})),
            conflict_assessment=ConflictAssessment.from_dict(dict(data.get("conflict_assessment") or {})),
            payoff_assessment=PayoffAssessment.from_dict(dict(data.get("payoff_assessment") or {})),
            quote_candidates=[QuoteCandidate.from_dict(item) for item in data.get("quote_candidates", []) if isinstance(item, dict)],
            analysis_confidence=FeatureScore.from_dict(dict(data.get("analysis_confidence") or {})),
            analysis_mode=str(data.get("analysis_mode") or "deterministic"),
            warnings=[str(item) for item in data.get("warnings", [])],
        )
        result.validate()
        return result


@dataclass(slots=True)
class RetentionZone:
    """One normalized candidate section; values are internal comparative indices."""

    name: str
    start: float
    end: float
    attention_strength: FeatureScore
    information_gain: FeatureScore
    emotional_energy: FeatureScore
    curiosity_state: str
    narrative_momentum: FeatureScore
    repetition: FeatureScore
    confusion: FeatureScore
    drop_risk: FeatureScore
    evidence: list[FeatureEvidence]

    def validate(self, candidate_start: float | None = None, candidate_end: float | None = None, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.name or self.end < self.start or self.curiosity_state not in {"none", "open", "resolved"}:
            raise ValueError("RetentionZone identity or range is invalid.")
        if candidate_start is not None and self.start < candidate_start - 0.001:
            raise ValueError("RetentionZone starts outside its candidate.")
        if candidate_end is not None and self.end > candidate_end + 0.001:
            raise ValueError("RetentionZone ends outside its candidate.")
        for score in (
            self.attention_strength, self.information_gain, self.emotional_energy,
            self.narrative_momentum, self.repetition, self.confusion, self.drop_risk,
        ):
            score.validate(allowed_segment_ids)
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetentionZone":
        result = cls(
            name=str(data.get("name") or "unknown"), start=float(data.get("start", 0)), end=float(data.get("end", 0)),
            attention_strength=FeatureScore.from_dict(dict(data.get("attention_strength") or {})),
            information_gain=FeatureScore.from_dict(dict(data.get("information_gain") or {})),
            emotional_energy=FeatureScore.from_dict(dict(data.get("emotional_energy") or {})),
            curiosity_state=str(data.get("curiosity_state") or "none"),
            narrative_momentum=FeatureScore.from_dict(dict(data.get("narrative_momentum") or {})),
            repetition=FeatureScore.from_dict(dict(data.get("repetition") or {})),
            confusion=FeatureScore.from_dict(dict(data.get("confusion") or {})),
            drop_risk=FeatureScore.from_dict(dict(data.get("drop_risk") or {})),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class DeadZone:
    """A diagnostic only. Goal 5B never edits, removes, or moves this range."""

    start: float
    end: float
    duration: float
    reason: str
    severity: FeatureScore
    removable_in_future: bool
    evidence: list[FeatureEvidence]

    def validate(self, candidate_start: float | None = None, candidate_end: float | None = None, allowed_segment_ids: set[int] | None = None) -> None:
        if self.end < self.start or self.duration < 0 or abs(self.duration - (self.end - self.start)) > 0.01:
            raise ValueError("DeadZone range or duration is invalid.")
        if not self.reason:
            raise ValueError("DeadZone requires a diagnostic reason.")
        if candidate_start is not None and self.start < candidate_start - 0.001:
            raise ValueError("DeadZone starts outside its candidate.")
        if candidate_end is not None and self.end > candidate_end + 0.001:
            raise ValueError("DeadZone ends outside its candidate.")
        self.severity.validate(allowed_segment_ids)
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeadZone":
        result = cls(
            start=float(data.get("start", 0)), end=float(data.get("end", 0)), duration=float(data.get("duration", 0)),
            reason=str(data.get("reason") or "unknown"),
            severity=FeatureScore.from_dict(dict(data.get("severity") or {})),
            removable_in_future=bool(data.get("removable_in_future", False)),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class EstimatedRetentionProfile:
    """Relative candidate-local retention signal, never a platform-view forecast."""

    candidate_id: str
    candidate_start: float
    candidate_end: float
    zones: list[RetentionZone]
    opening_retention: FeatureScore
    early_retention: FeatureScore
    mid_retention: FeatureScore
    late_retention: FeatureScore
    completion_potential: FeatureScore
    estimated_drop_points: list[float]
    strongest_moment_timestamp: float
    weakest_moment_timestamp: float
    dead_zone_ranges: list[DeadZone]
    retention_confidence: FeatureScore
    analysis_mode: str = "deterministic"
    warnings: list[str] = field(default_factory=list)

    def relative_level(self, score: float) -> str:
        if score >= 0.8:
            return "very_high"
        if score >= 0.62:
            return "high"
        if score >= 0.42:
            return "medium"
        return "low"

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.candidate_id or self.candidate_end <= self.candidate_start:
            raise ValueError("EstimatedRetentionProfile has an invalid candidate range.")
        if len(self.zones) != len(RETENTION_ZONE_DEFINITIONS):
            raise ValueError("EstimatedRetentionProfile must retain every normalized zone.")
        for zone in self.zones:
            zone.validate(self.candidate_start, self.candidate_end, allowed_segment_ids)
        for point in self.estimated_drop_points:
            if not self.candidate_start <= point <= self.candidate_end:
                raise ValueError("Retention drop point must stay inside candidate.")
        for timestamp in (self.strongest_moment_timestamp, self.weakest_moment_timestamp):
            if not self.candidate_start <= timestamp <= self.candidate_end:
                raise ValueError("Retention moment timestamp must stay inside candidate.")
        for zone in self.dead_zone_ranges:
            zone.validate(self.candidate_start, self.candidate_end, allowed_segment_ids)
        for score in (
            self.opening_retention, self.early_retention, self.mid_retention,
            self.late_retention, self.completion_potential, self.retention_confidence,
        ):
            score.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "candidate_start": round(self.candidate_start, 3),
            "candidate_end": round(self.candidate_end, 3),
            "zones": [item.to_dict() for item in self.zones],
            "opening_retention": self.opening_retention.to_dict(),
            "early_retention": self.early_retention.to_dict(),
            "mid_retention": self.mid_retention.to_dict(),
            "late_retention": self.late_retention.to_dict(),
            "completion_potential": self.completion_potential.to_dict(),
            "estimated_drop_points": [round(item, 3) for item in self.estimated_drop_points],
            "strongest_moment_timestamp": round(self.strongest_moment_timestamp, 3),
            "weakest_moment_timestamp": round(self.weakest_moment_timestamp, 3),
            "dead_zone_ranges": [item.to_dict() for item in self.dead_zone_ranges],
            "retention_confidence": self.retention_confidence.to_dict(),
            "analysis_mode": self.analysis_mode,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EstimatedRetentionProfile":
        result = cls(
            candidate_id=str(data.get("candidate_id") or ""), candidate_start=float(data.get("candidate_start", 0)),
            candidate_end=float(data.get("candidate_end", 0)),
            zones=[RetentionZone.from_dict(item) for item in data.get("zones", []) if isinstance(item, dict)],
            opening_retention=FeatureScore.from_dict(dict(data.get("opening_retention") or {})),
            early_retention=FeatureScore.from_dict(dict(data.get("early_retention") or {})),
            mid_retention=FeatureScore.from_dict(dict(data.get("mid_retention") or {})),
            late_retention=FeatureScore.from_dict(dict(data.get("late_retention") or {})),
            completion_potential=FeatureScore.from_dict(dict(data.get("completion_potential") or {})),
            estimated_drop_points=[float(item) for item in data.get("estimated_drop_points", [])],
            strongest_moment_timestamp=float(data.get("strongest_moment_timestamp", 0)),
            weakest_moment_timestamp=float(data.get("weakest_moment_timestamp", 0)),
            dead_zone_ranges=[DeadZone.from_dict(item) for item in data.get("dead_zone_ranges", []) if isinstance(item, dict)],
            retention_confidence=FeatureScore.from_dict(dict(data.get("retention_confidence") or {})),
            analysis_mode=str(data.get("analysis_mode") or "deterministic"),
            warnings=[str(item) for item in data.get("warnings", [])],
        )
        result.validate()
        return result


@dataclass(slots=True)
class PublishabilityAssessment:
    """Can the unedited candidate be published as an understandable standalone short?"""

    candidate_id: str
    level: str
    publishability_score: FeatureScore
    opening_clarity: FeatureScore
    standalone_strength: FeatureScore
    story_completeness: FeatureScore
    boundary_safety: FeatureScore
    context_independence: FeatureScore
    duration_fit: FeatureScore
    source_signal_quality: FeatureScore
    payoff_presence: FeatureScore
    filler_control: FeatureScore
    visual_usability: FeatureScore
    subtitle_compatibility: FeatureScore
    reasons: list[str]
    critical_failures: list[str]
    warnings: list[str] = field(default_factory=list)

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.candidate_id or self.level not in PUBLISHABILITY_LEVELS:
            raise ValueError("PublishabilityAssessment identity or level is invalid.")
        for score in (
            self.publishability_score, self.opening_clarity, self.standalone_strength,
            self.story_completeness, self.boundary_safety, self.context_independence,
            self.duration_fit, self.source_signal_quality, self.payoff_presence,
            self.filler_control, self.visual_usability, self.subtitle_compatibility,
        ):
            score.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublishabilityAssessment":
        result = cls(
            candidate_id=str(data.get("candidate_id") or ""), level=str(data.get("level") or "blocked"),
            publishability_score=FeatureScore.from_dict(dict(data.get("publishability_score") or {})),
            opening_clarity=FeatureScore.from_dict(dict(data.get("opening_clarity") or {})),
            standalone_strength=FeatureScore.from_dict(dict(data.get("standalone_strength") or {})),
            story_completeness=FeatureScore.from_dict(dict(data.get("story_completeness") or {})),
            boundary_safety=FeatureScore.from_dict(dict(data.get("boundary_safety") or {})),
            context_independence=FeatureScore.from_dict(dict(data.get("context_independence") or {})),
            duration_fit=FeatureScore.from_dict(dict(data.get("duration_fit") or {})),
            source_signal_quality=FeatureScore.from_dict(dict(data.get("source_signal_quality") or {})),
            payoff_presence=FeatureScore.from_dict(dict(data.get("payoff_presence") or {})),
            filler_control=FeatureScore.from_dict(dict(data.get("filler_control") or {})),
            visual_usability=FeatureScore.from_dict(dict(data.get("visual_usability") or {})),
            subtitle_compatibility=FeatureScore.from_dict(dict(data.get("subtitle_compatibility") or {})),
            reasons=[str(item) for item in data.get("reasons", [])],
            critical_failures=[str(item) for item in data.get("critical_failures", [])],
            warnings=[str(item) for item in data.get("warnings", [])],
        )
        result.validate()
        return result


@dataclass(slots=True)
class EligibilityAssessment:
    candidate_id: str
    status: str
    reasons: list[str]
    critical_failures: list[str]
    reconstruction_opportunities: list[str]
    confidence: FeatureScore

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.candidate_id or self.status not in ELIGIBILITY_STATUSES:
            raise ValueError("EligibilityAssessment identity or status is invalid.")
        self.confidence.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EligibilityAssessment":
        result = cls(
            candidate_id=str(data.get("candidate_id") or ""), status=str(data.get("status") or "weak"),
            reasons=[str(item) for item in data.get("reasons", [])],
            critical_failures=[str(item) for item in data.get("critical_failures", [])],
            reconstruction_opportunities=[str(item) for item in data.get("reconstruction_opportunities", [])],
            confidence=FeatureScore.from_dict(dict(data.get("confidence") or {})),
        )
        result.validate()
        return result


@dataclass(slots=True)
class ScoreContribution:
    name: str
    raw_score: float
    normalized_score: float
    confidence_adjusted_score: float
    confidence: float
    strategy_weight: float
    contribution: float
    explanation: str
    evidence: list[FeatureEvidence]

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.name or not self.explanation:
            raise ValueError("ScoreContribution requires a name and explanation.")
        for value in (self.raw_score, self.normalized_score, self.confidence_adjusted_score, self.confidence, self.strategy_weight, self.contribution):
            if not 0 <= value <= 1:
                raise ValueError("ScoreContribution values must be bounded.")
        for item in self.evidence:
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _nested_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreContribution":
        result = cls(
            name=str(data.get("name") or "unknown"), raw_score=_bounded(float(data.get("raw_score", 0))),
            normalized_score=_bounded(float(data.get("normalized_score", 0))),
            confidence_adjusted_score=_bounded(float(data.get("confidence_adjusted_score", 0))),
            confidence=_bounded(float(data.get("confidence", 0))), strategy_weight=_bounded(float(data.get("strategy_weight", 0))),
            contribution=_bounded(float(data.get("contribution", 0))), explanation=str(data.get("explanation") or "No explanation."),
            evidence=[FeatureEvidence.from_dict(item) for item in data.get("evidence", []) if isinstance(item, dict)],
        )
        result.validate()
        return result


@dataclass(slots=True)
class ViralityConfidence:
    overall: FeatureScore
    factors: dict[str, FeatureScore]
    warnings: list[str] = field(default_factory=list)

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.factors:
            raise ValueError("ViralityConfidence requires explainable factors.")
        self.overall.validate(allowed_segment_ids)
        for item in self.factors.values():
            item.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "overall": self.overall.to_dict(),
            "factors": {name: score.to_dict(name) for name, score in self.factors.items()},
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViralityConfidence":
        raw_factors = data.get("factors", {})
        result = cls(
            overall=FeatureScore.from_dict(dict(data.get("overall") or {})),
            factors={name: FeatureScore.from_dict(dict(value)) for name, value in raw_factors.items() if isinstance(value, dict)} if isinstance(raw_factors, dict) else {},
            warnings=[str(item) for item in data.get("warnings", [])],
        )
        result.validate()
        return result


@dataclass(slots=True)
class ViralPotentialScore:
    candidate_id: str
    strategy_id: str
    components: dict[str, ScoreContribution]
    penalties: dict[str, ScoreContribution]
    positive_score: float
    penalty_score: float
    viral_potential_score: float
    retention_potential_score: float
    publishability_score: float
    level: str
    confidence: ViralityConfidence
    strongest_factors: list[str]
    weakest_factors: list[str]
    ranking_explanation: str
    eligibility_status: str

    def validate(self, allowed_segment_ids: set[int] | None = None) -> None:
        if not self.candidate_id or self.strategy_id not in {
            "motivational_monologue", "generic_monologue", "generic_dialogue", "generic_educational",
            "generic_scene_driven", "generic_fallback",
        }:
            raise ValueError("ViralPotentialScore identity or strategy is invalid.")
        if set(self.components) != set(VIRALITY_COMPONENTS) or set(self.penalties) != set(VIRALITY_PENALTIES):
            raise ValueError("ViralPotentialScore must expose every component and penalty.")
        if self.level not in {"weak", "moderate", "strong", "excellent"} or self.eligibility_status not in ELIGIBILITY_STATUSES:
            raise ValueError("ViralPotentialScore level or eligibility is invalid.")
        for value in (self.positive_score, self.penalty_score, self.viral_potential_score, self.retention_potential_score, self.publishability_score):
            if not 0 <= value <= 1:
                raise ValueError("ViralPotentialScore values must be bounded.")
        for item in [*self.components.values(), *self.penalties.values()]:
            item.validate(allowed_segment_ids)
        self.confidence.validate(allowed_segment_ids)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "candidate_id": self.candidate_id, "strategy_id": self.strategy_id,
            "components": {name: value.to_dict() for name, value in self.components.items()},
            "penalties": {name: value.to_dict() for name, value in self.penalties.items()},
            "positive_score": round(self.positive_score, 6), "penalty_score": round(self.penalty_score, 6),
            "viral_potential_score": round(self.viral_potential_score, 6),
            "retention_potential_score": round(self.retention_potential_score, 6),
            "publishability_score": round(self.publishability_score, 6), "level": self.level,
            "confidence": self.confidence.to_dict(), "strongest_factors": self.strongest_factors,
            "weakest_factors": self.weakest_factors, "ranking_explanation": self.ranking_explanation,
            "eligibility_status": self.eligibility_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViralPotentialScore":
        raw_components, raw_penalties = data.get("components", {}), data.get("penalties", {})
        result = cls(
            candidate_id=str(data.get("candidate_id") or ""), strategy_id=str(data.get("strategy_id") or "generic_fallback"),
            components={name: ScoreContribution.from_dict(dict(raw_components.get(name) or {})) for name in VIRALITY_COMPONENTS} if isinstance(raw_components, dict) else {},
            penalties={name: ScoreContribution.from_dict(dict(raw_penalties.get(name) or {})) for name in VIRALITY_PENALTIES} if isinstance(raw_penalties, dict) else {},
            positive_score=_bounded(float(data.get("positive_score", 0))), penalty_score=_bounded(float(data.get("penalty_score", 0))),
            viral_potential_score=_bounded(float(data.get("viral_potential_score", 0))),
            retention_potential_score=_bounded(float(data.get("retention_potential_score", 0))),
            publishability_score=_bounded(float(data.get("publishability_score", 0))), level=str(data.get("level") or "weak"),
            confidence=ViralityConfidence.from_dict(dict(data.get("confidence") or {})),
            strongest_factors=[str(item) for item in data.get("strongest_factors", [])],
            weakest_factors=[str(item) for item in data.get("weakest_factors", [])],
            ranking_explanation=str(data.get("ranking_explanation") or "No explanation."),
            eligibility_status=str(data.get("eligibility_status") or "weak"),
        )
        result.validate()
        return result


def _nested_dict(value: Any) -> dict[str, Any]:
    """Serialize nested FeatureScore models without leaking dataclass internals."""

    data = asdict(value)
    for name in value.__dataclass_fields__:
        original = getattr(value, name)
        if isinstance(original, FeatureScore):
            data[name] = original.to_dict()
        elif isinstance(original, list) and original and isinstance(original[0], FeatureEvidence):
            data[name] = [item.to_dict() for item in original]
    return data


def _words(text: str) -> list[str]:
    return [item.casefold() for item in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_RE.split(text.strip()) if item.strip()] or ([text.strip()] if text.strip() else [])


def _average(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    lower = text.casefold()
    return any(pattern in lower for pattern in patterns)


def _feature(
    score: float, confidence: float, explanation: str, *, source: str, raw: float | str | bool | None,
    segment_ids: list[int], excerpt: str,
) -> FeatureScore:
    return FeatureScore(
        score=_bounded(score), confidence=_bounded(confidence), explanation=explanation,
        evidence=[FeatureEvidence(source, raw, _bounded(score), _bounded(confidence), list(segment_ids), excerpt[:800])],
    )


def _candidate_segments(candidate: Candidate, transcript_features: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = set(candidate.transcript_segment_ids)
    result = [
        item for item in transcript_features.get("segments", [])
        if isinstance(item, dict)
        and float(item.get("end", -1)) > candidate.start
        and float(item.get("start", math.inf)) < candidate.end
        and (not allowed or int(item.get("id", -1)) in allowed)
    ]
    return sorted(result, key=lambda item: (float(item.get("start", 0)), int(item.get("id", 0))))


def _story_for(candidate: Candidate, content_map: dict[str, Any]) -> StoryUnit | None:
    story_id = str(candidate.story_unit_id or "")
    for raw in content_map.get("story_units", []) if isinstance(content_map, dict) else []:
        if isinstance(raw, dict) and str(raw.get("story_unit_id") or "") == story_id:
            return StoryUnit.from_dict(raw)
    return None


def _emotion(text: str, punctuation_bonus: bool = True) -> tuple[str, float]:
    lower = text.casefold()
    scores = {name: sum(term in lower for term in terms) for name, terms in _EMOTION_LEXICON.items()}
    name, hits = max(scores.items(), key=lambda item: item[1])
    if hits <= 0:
        return "neutral", 0.16 + (0.10 if punctuation_bonus and "!" in text else 0.0)
    return name, _bounded(0.28 + hits * 0.18 + (0.12 if punctuation_bonus and "!" in text else 0.0))


def _hook_type(opening: str, contextual: bool, curiosity: bool, conflict: bool, emotion: float) -> str:
    lower = opening.casefold()
    if not opening:
        return "none"
    if any(lower.startswith(item) for item in _GREETING_PREFIXES + _TECHNICAL_PREFIXES) or contextual:
        return "weak_contextual"
    if curiosity:
        return "question"
    if _contains_any(lower, ("никогда", "самая", "всё", "единствен", "never", "the most", "only")):
        return "bold_claim"
    if _contains_any(lower, ("если", "опас", "потер", "warning", "risk")):
        return "warning"
    if conflict:
        return "conflict"
    if emotion >= 0.52:
        return "emotional_statement"
    if re.search(r"\d", opening):
        return "informational"
    return "quote" if len(_words(opening)) <= 18 else "informational"


def _score_hook(candidate: Candidate, segments: list[dict[str, Any]], story: StoryUnit | None) -> HookAssessment:
    opening = _sentences(candidate.text)[0] if _sentences(candidate.text) else ""
    segment_ids = [int(item.get("id", -1)) for item in segments if int(item.get("id", -1)) >= 0]
    first_segment = segments[0] if segments else {}
    contextual = any(opening.casefold().startswith(item) for item in _CONTEXTUAL_PREFIXES)
    greeting = any(opening.casefold().startswith(item) for item in _GREETING_PREFIXES + _TECHNICAL_PREFIXES)
    curiosity_opened = "?" in opening or _contains_any(opening, _QUESTION_WORDS) or _contains_any(opening, ("но затем", "but then", "что если", "what if"))
    conflict = _contains_any(opening, _CONFLICT_MARKERS)
    emotion_name, emotion = _emotion(opening)
    meaningful = bool(_words(opening)) and not greeting
    clarity = 0.82 if meaningful and not contextual else 0.36 if contextual else 0.20
    specificity = _specificity(candidate.text, segment_ids, opening)[0]
    stakes = 0.18 + (0.36 if conflict else 0) + (0.18 if _contains_any(opening, ("послед", "цена", "риск", "risk", "cost", "last")) else 0)
    hook_type = _hook_type(opening, contextual, curiosity_opened, conflict, emotion)
    strength = 0.26 + (0.24 if hook_type in {"bold_claim", "conflict", "warning"} else 0)
    strength += 0.20 if curiosity_opened else 0
    strength += emotion * 0.18 + specificity * 0.12 + stakes * 0.12
    if greeting:
        strength -= 0.34
    if contextual:
        strength -= 0.24
    time_to_value = 0.0 if meaningful else min(candidate.duration, 5.0)
    slow_start = 0.72 if greeting else 0.55 if contextual else 0.12 if meaningful else 0.40
    later = candidate.text[len(opening):].casefold()
    answer_present = curiosity_opened and (_contains_any(later, _ANSWER_MARKERS) or bool(story and story.payoff.strip()))
    resolution_time = None
    if answer_present:
        for segment in segments[1:]:
            if _contains_any(str(segment.get("text") or ""), _ANSWER_MARKERS):
                resolution_time = max(candidate.start, float(segment.get("start", candidate.start)))
                break
        resolution_time = resolution_time if resolution_time is not None else candidate.end
    resolution_quality = 0.78 if answer_present else 0.0
    unresolved = 0.72 if curiosity_opened and not answer_present else 0.0
    confidence = _transcript_confidence(segments)
    evidence = FeatureEvidence("candidate_opening", opening, _bounded(strength), confidence, segment_ids[:1], opening)
    weak_reason = "Начало требует предыдущего контекста." if contextual else "Приветствие или техническая подводка задерживает ценность." if greeting else None
    return HookAssessment(
        hook_type=hook_type,
        hook_strength=_feature(strength, confidence, "Сила первой законченной фразы и первых секунд.", source="candidate_opening", raw=opening, segment_ids=segment_ids[:1], excerpt=opening),
        immediate_clarity=_feature(clarity, confidence, "Насколько быстро понятна проблема или тезис.", source="candidate_opening", raw=opening, segment_ids=segment_ids[:1], excerpt=opening),
        curiosity_gap=_feature(0.72 if curiosity_opened and answer_present else 0.45 if curiosity_opened else 0.0, confidence, "Интрига учитывается только при подтверждённом раскрытии.", source="candidate_opening", raw=curiosity_opened, segment_ids=segment_ids, excerpt=opening),
        specificity=_feature(specificity, confidence, "Конкретность первой фразы.", source="candidate_opening", raw=opening, segment_ids=segment_ids[:1], excerpt=opening),
        stakes=_feature(stakes, confidence, "Ставка определяется ясным риском, следствием или конфликтом.", source="candidate_opening", raw=conflict, segment_ids=segment_ids[:1], excerpt=opening),
        emotional_charge=_feature(emotion, confidence, f"Эмоциональный тон открытия: {emotion_name}.", source="candidate_opening", raw=emotion_name, segment_ids=segment_ids[:1], excerpt=opening),
        first_sentence_quality=_feature(_bounded((strength + clarity + specificity) / 3), confidence, "Сочетание силы, ясности и конкретности первой фразы.", source="candidate_opening", raw=opening, segment_ids=segment_ids[:1], excerpt=opening),
        context_dependency=_feature(0.78 if contextual else 0.18, confidence, "Штраф за неясный референт или связку без контекста.", source="candidate_opening", raw=contextual, segment_ids=segment_ids[:1], excerpt=opening),
        slow_start_penalty=_feature(slow_start, confidence, "Штраф за приветствие, техническую подводку или медленное начало.", source="candidate_opening", raw=greeting, segment_ids=segment_ids[:1], excerpt=opening),
        time_to_value_seconds=round(time_to_value, 3), curiosity_opened=curiosity_opened,
        curiosity_resolved=answer_present, resolution_timestamp=resolution_time,
        curiosity_resolution_quality=_feature(resolution_quality, confidence, "Качество раскрытия вопроса внутри candidate.", source="candidate_body", raw=answer_present, segment_ids=segment_ids, excerpt=candidate.text[-260:]),
        unresolved_curiosity_penalty=_feature(unresolved, confidence, "Штраф, если поднятый вопрос не получает payoff.", source="candidate_body", raw=not answer_present, segment_ids=segment_ids, excerpt=candidate.text[-260:]),
        weak_opening_reason=weak_reason, evidence=[evidence],
    )


def _specificity(text: str, segment_ids: list[int], excerpt: str) -> tuple[float, int, int, float, float]:
    tokens = _words(text)
    numbers = len(re.findall(r"\b\d+(?:[,.]\d+)?%?\b", text))
    actionable = sum(any(pattern in token for pattern in _ACTION_WORDS) for token in tokens)
    abstract = sum(token in _ABSTRACT_WORDS for token in tokens) / max(1, len(tokens))
    cliche = min(1.0, sum(token in {"успех", "жизнь", "success", "life"} for token in tokens) / max(1, len(tokens)) * 3)
    score = _bounded(0.28 + min(0.30, numbers * 0.12) + min(0.28, actionable * 0.055) - abstract * 0.24 - cliche * 0.12)
    return score, numbers, actionable, abstract, cliche


def _transcript_confidence(segments: list[dict[str, Any]]) -> float:
    values = [float(item["transcript_confidence"]) for item in segments if item.get("transcript_confidence") is not None]
    return _bounded(_average(values, 0.66) if values else 0.62)


def _score_emotional_arc(candidate: Candidate, segments: list[dict[str, Any]], audio_features: dict[str, Any]) -> EmotionalArc:
    segment_ids = [int(item.get("id", -1)) for item in segments if int(item.get("id", -1)) >= 0]
    audio = window_audio_features(candidate.start, candidate.end, audio_features)
    rows: list[tuple[str, float, float, int]] = []
    for index, segment in enumerate(segments):
        label, value = _emotion(str(segment.get("text") or ""))
        value = _bounded(value + float(segment.get("exclamation_count", 0)) * 0.06)
        rows.append((label, value, float(segment.get("start", candidate.start)), int(segment.get("id", -1))))
    if not rows:
        rows = [("neutral", 0.16, candidate.start, -1)]
    opening, start_level, _start_time, _ = rows[0]
    peak, peak_level, peak_time, peak_id = max(rows, key=lambda item: item[1])
    ending, end_level, _end_time, _ = rows[-1]
    dominant = max(EMOTIONS, key=lambda label: sum(1 for item in rows if item[0] == label))
    escalation = _bounded(max(0.0, peak_level - start_level) + float(audio.get("audio_energy_change", 0)) * 0.22)
    change = _bounded(abs(end_level - start_level) + escalation * 0.45)
    consistency = _bounded(1.0 - min(0.8, _average(abs(item[1] - start_level) for item in rows) * 1.2))
    emotional_payoff = _bounded(end_level * 0.55 + escalation * 0.45)
    flatness = _bounded(0.72 - change * 0.78 - float(audio.get("audio_energy_change", 0)) * 0.20)
    confidence = _bounded(_transcript_confidence(segments) * 0.72 + (0.22 if audio_features.get("energy_frames") else 0.0))
    peak_ids = [peak_id] if peak_id >= 0 else []
    evidence = [FeatureEvidence("transcript_emotion", peak, peak_level, confidence, peak_ids, next((str(item.get("text") or "") for item in segments if int(item.get("id", -1)) == peak_id), candidate.text[:240]))]
    return EmotionalArc(
        opening_emotion=opening, dominant_emotion=dominant, peak_emotion=peak, ending_emotion=ending,
        emotional_start_level=_feature(start_level, confidence, "Эмоциональная интенсивность в начале.", source="transcript_emotion", raw=opening, segment_ids=segment_ids[:1], excerpt=candidate.text[:240]),
        emotional_peak_level=_feature(peak_level, confidence, "Максимальная семантическая и речевая интенсивность внутри candidate.", source="transcript_emotion", raw=peak, segment_ids=peak_ids, excerpt=evidence[0].excerpt),
        emotional_end_level=_feature(end_level, confidence, "Эмоциональная интенсивность на естественном завершении.", source="transcript_emotion", raw=ending, segment_ids=segment_ids[-1:], excerpt=candidate.text[-240:]),
        peak_timestamp=round(min(candidate.end, max(candidate.start, peak_time)), 3),
        escalation_strength=_feature(escalation, confidence, "Рост эмоции и доступное изменение энергии голоса.", source="audio_transcript", raw=audio.get("audio_energy_change", 0), segment_ids=segment_ids, excerpt=evidence[0].excerpt),
        emotional_change=_feature(change, confidence, "Изменение эмоционального состояния от начала к концу.", source="transcript_emotion", raw=change, segment_ids=segment_ids, excerpt=candidate.text[:240]),
        emotional_consistency=_feature(consistency, confidence, "Последовательность эмоциональной линии без случайных скачков.", source="transcript_emotion", raw=consistency, segment_ids=segment_ids, excerpt=candidate.text[:240]),
        emotional_payoff=_feature(emotional_payoff, confidence, "Эмоциональное разрешение к естественному концу.", source="transcript_emotion", raw=end_level, segment_ids=segment_ids[-1:], excerpt=candidate.text[-240:]),
        flatness_penalty=_feature(flatness, confidence, "Штраф за плоскую эмоциональную и аудио-динамику.", source="audio_transcript", raw=change, segment_ids=segment_ids, excerpt=candidate.text[:240]),
        evidence=evidence,
    )


def _score_conflict(candidate: Candidate, segments: list[dict[str, Any]], payoff: PayoffAssessment | None = None) -> ConflictAssessment:
    text = candidate.text.casefold()
    segment_ids = [int(item.get("id", -1)) for item in segments if int(item.get("id", -1)) >= 0]
    hits = sum(marker in text for marker in _CONFLICT_MARKERS)
    problem = _contains_any(text, ("проблем", "ошиб", "риск", "страх", "потер", "problem", "risk", "fear", "loss"))
    resolution = _contains_any(text, _PAYOFF_MARKERS)
    presence = _bounded(0.12 + hits * 0.12 + (0.20 if problem else 0))
    conflict_type = "problem_vs_solution" if problem and resolution else "risk_vs_reward" if _contains_any(text, ("риск", "цена", "risk", "reward")) else "belief_vs_reality" if _contains_any(text, ("но", "однако", "but", "however")) else "none"
    strength = _bounded(presence * 0.78 + (0.16 if resolution else 0))
    stakes = _bounded(0.10 + (0.34 if _contains_any(text, ("цена", "послед", "побед", "поражен", "cost", "consequence", "win", "lose")) else 0) + (0.18 if problem else 0))
    confidence = _transcript_confidence(segments)
    excerpt = candidate.text[:360]
    return ConflictAssessment(
        conflict_type=conflict_type,
        conflict_presence=_feature(presence, confidence, "Наличие понятной проблемы, противопоставления или риска.", source="candidate_transcript", raw=hits, segment_ids=segment_ids, excerpt=excerpt),
        conflict_strength=_feature(strength, confidence, "Сила конфликта не повышается только из-за агрессивных слов.", source="candidate_transcript", raw=hits, segment_ids=segment_ids, excerpt=excerpt),
        stakes_clarity=_feature(stakes, confidence, "Понятность последствий для зрителя.", source="candidate_transcript", raw=problem, segment_ids=segment_ids, excerpt=excerpt),
        stakes_magnitude=_feature(_bounded(stakes * 0.82 + presence * 0.18), confidence, "Относительная величина обозначенной ставки.", source="candidate_transcript", raw=stakes, segment_ids=segment_ids, excerpt=excerpt),
        conflict_resolution=_feature(0.72 if resolution else 0.0, confidence, "Разрешён ли конфликт внутри candidate.", source="candidate_ending", raw=resolution, segment_ids=segment_ids[-1:], excerpt=candidate.text[-280:]),
        conflict_payoff=_feature(0.68 if resolution and presence >= 0.3 else 0.0, confidence, "Payoff конфликта засчитывается только при присутствии и завершении.", source="candidate_ending", raw=resolution, segment_ids=segment_ids[-1:], excerpt=candidate.text[-280:]),
        evidence=[FeatureEvidence("candidate_transcript", hits, presence, confidence, segment_ids, excerpt)],
    )


def _score_payoff(candidate: Candidate, story: StoryUnit | None, segments: list[dict[str, Any]], hook: HookAssessment | None = None) -> PayoffAssessment:
    text = candidate.text.casefold()
    ending = candidate.text[-300:]
    segment_ids = [int(item.get("id", -1)) for item in segments if int(item.get("id", -1)) >= 0]
    marker = _contains_any(ending, _PAYOFF_MARKERS)
    story_payoff = bool(story and story.payoff.strip())
    question = bool(hook and hook.curiosity_opened)
    resolved = bool(hook and hook.curiosity_resolved)
    ending_natural = bool(candidate.boundary_diagnostics.get("eligible", False)) if candidate.boundary_diagnostics else bool(candidate.text.rstrip().endswith((".", "!", "?", "…")))
    soft_story_payoff = bool(
        story and story.publishability_precheck and story.completeness_score >= 0.70
        and ending_natural and _contains_any(ending, _SOFT_PAYOFF_MARKERS)
    )
    present = marker or story_payoff or resolved or soft_story_payoff
    payoff_type = "answer" if resolved else "conclusion" if marker else "insight" if story_payoff or soft_story_payoff else "none"
    strength = _bounded(
        (0.46 if present else 0.0) + (0.20 if marker else 0) + (0.17 if resolved else 0)
        + (0.08 if soft_story_payoff else 0) + (0.10 if ending_natural else 0)
    )
    alignment = _bounded(0.82 if question and resolved else 0.62 if present else 0.0)
    confidence = _transcript_confidence(segments)
    payoff_segment: dict[str, Any] | None = None
    if resolved and hook and hook.resolution_timestamp is not None:
        payoff_segment = next(
            (item for item in segments if float(item.get("start", candidate.end)) <= hook.resolution_timestamp <= float(item.get("end", candidate.end))),
            None,
        )
    if payoff_segment is None and marker:
        payoff_segment = next(
            (item for item in segments if _contains_any(str(item.get("text") or ""), _PAYOFF_MARKERS)),
            None,
        )
    if payoff_segment is None and story_payoff:
        payoff_segment = next(
            (item for item in segments if str(story.payoff).casefold() in str(item.get("text") or "").casefold()),
            None,
        )
    timestamp = (
        max(candidate.start, float(payoff_segment.get("start", candidate.end)))
        if payoff_segment is not None else candidate.end if present else None
    )
    return PayoffAssessment(
        payoff_present=present, payoff_type=payoff_type,
        payoff_strength=_feature(strength, confidence, "Сила завершения, которое отвечает на setup или даёт самостоятельный вывод.", source="candidate_ending", raw=marker or story_payoff, segment_ids=segment_ids[-1:], excerpt=ending),
        payoff_timestamp=round(timestamp, 3) if timestamp is not None else None,
        setup_payoff_alignment=_feature(alignment, confidence, "Связь между поднятым вопросом/проблемой и концовкой.", source="candidate_transcript", raw=resolved, segment_ids=segment_ids, excerpt=ending),
        emotional_resolution=_feature(0.72 if "!" in ending and present else 0.42 if present else 0.0, confidence, "Эмоциональное разрешение на естественной границе.", source="candidate_ending", raw=ending, segment_ids=segment_ids[-1:], excerpt=ending),
        informational_resolution=_feature(0.78 if payoff_type in {"answer", "insight", "conclusion"} else 0.0, confidence, "Информационное разрешение вопроса или тезиса.", source="candidate_ending", raw=payoff_type, segment_ids=segment_ids[-1:], excerpt=ending),
        surprise_resolution=_feature(0.55 if resolved and _contains_any(ending, ("но", "однако", "but", "however")) else 0.0, confidence, "Неожиданный payoff засчитывается только внутри candidate.", source="candidate_ending", raw=resolved, segment_ids=segment_ids[-1:], excerpt=ending),
        ending_satisfaction=_feature(_bounded(strength * 0.72 + (0.22 if ending_natural else 0)), confidence, "Естественная граница и содержательное завершение мысли.", source="semantic_boundary", raw=ending_natural, segment_ids=segment_ids[-1:], excerpt=ending),
        payoff_missing_reason=None if present else "Начало не получает самостоятельного ответа или вывода внутри candidate.",
        evidence=[FeatureEvidence("candidate_ending", payoff_type, strength, confidence, segment_ids[-1:], ending)],
    )


def _quote_candidates(candidate: Candidate, segments: list[dict[str, Any]], emotional: EmotionalArc, story: StoryUnit | None) -> list[QuoteCandidate]:
    sentences = _sentences(candidate.text)
    if not sentences:
        return []
    segment_ids = [int(item.get("id", -1)) for item in segments if int(item.get("id", -1)) >= 0]
    selected = sorted(sentences, key=lambda sentence: ("!" in sentence, "?" in sentence, len(_words(sentence))), reverse=True)[:2]
    result: list[QuoteCandidate] = []
    for index, sentence in enumerate(selected):
        words = _words(sentence)
        clarity = _bounded(0.76 - max(0, len(words) - 22) * 0.018)
        emotional_strength = _bounded(emotional.emotional_peak_level.score if "!" in sentence else emotional.emotional_end_level.score)
        cliche = _bounded(sum(token in _ABSTRACT_WORDS for token in words) / max(1, len(words)) * 2.2)
        memorability = _bounded(0.34 + clarity * 0.28 + emotional_strength * 0.22 + (0.16 if "!" in sentence or "?" in sentence else 0) - cliche * 0.14)
        start = candidate.start + (candidate.duration * index / max(1, len(selected)))
        end = min(candidate.end, start + candidate.duration / max(1, len(selected)))
        confidence = _transcript_confidence(segments)
        result.append(QuoteCandidate(
            text=sentence[:500], segment_ids=segment_ids, start=round(start, 3), end=round(end, 3),
            memorability=_feature(memorability, confidence, "Запоминаемость короткой самостоятельной формулировки.", source="candidate_sentence", raw=sentence, segment_ids=segment_ids, excerpt=sentence),
            clarity=_feature(clarity, confidence, "Ясность потенциальной цитаты вне полного видео.", source="candidate_sentence", raw=len(words), segment_ids=segment_ids, excerpt=sentence),
            emotional_strength=_feature(emotional_strength, confidence, "Эмоциональный резонанс цитаты.", source="candidate_sentence", raw=sentence, segment_ids=segment_ids, excerpt=sentence),
            standalone_quality=_feature(story.standalone_score if story else 0.5, confidence, "Самостоятельность цитаты наследует только grounded StoryUnit evidence.", source="story_unit", raw=story.standalone_score if story else None, segment_ids=segment_ids, excerpt=sentence),
            cliche_risk=_feature(cliche, confidence, "Риск слишком абстрактной или клишированной формулировки.", source="candidate_sentence", raw=cliche, segment_ids=segment_ids, excerpt=sentence),
        ))
    return result


def _source_relative_novelty(candidate: Candidate, story: StoryUnit | None, content_map: dict[str, Any]) -> tuple[float, float]:
    words = set(_words(candidate.core_idea or candidate.text))
    similarities: list[float] = []
    for raw in content_map.get("story_units", []) if isinstance(content_map, dict) else []:
        if not isinstance(raw, dict) or str(raw.get("story_unit_id") or "") == str(candidate.story_unit_id or ""):
            continue
        other = set(_words(str(raw.get("core_idea") or raw.get("development") or "")))
        if words or other:
            similarities.append(len(words & other) / max(1, len(words | other)))
    repetition = max(similarities, default=0.0)
    surprise = 0.35 if _contains_any(candidate.text, ("но", "однако", "вместо", "не", "but", "however", "instead")) else 0.10
    return _bounded(1 - repetition), _bounded(surprise)


def build_virality_feature_profile(
    candidate: Candidate, content_map: dict[str, Any], transcript_features: dict[str, Any],
    audio_features: dict[str, Any], visual_analysis: dict[str, Any] | None = None,
    content_strategy: str = "generic_fallback",
) -> ViralityFeatureProfile:
    """Build all Goal 5B feature components from local, candidate-scoped evidence."""

    if candidate.duration <= 0:
        raise ValueError("Virality analysis requires a positive candidate duration.")
    story = _story_for(candidate, content_map)
    segments = _candidate_segments(candidate, transcript_features)
    ids = [int(item.get("id", -1)) for item in segments if int(item.get("id", -1)) >= 0]
    allowed_ids = set(candidate.transcript_segment_ids or ids)
    hook = _score_hook(candidate, segments, story)
    payoff = _score_payoff(candidate, story, segments, hook)
    emotional = _score_emotional_arc(candidate, segments, audio_features)
    conflict = _score_conflict(candidate, segments, payoff)
    quote_candidates = _quote_candidates(candidate, segments, emotional, story)
    specificity, number_count, actionable_count, abstract_ratio, cliche = _specificity(candidate.text, ids, candidate.text[:300])
    novelty, surprise = _source_relative_novelty(candidate, story, content_map)
    audio = window_audio_features(candidate.start, candidate.end, audio_features)
    transcript_confidence = _transcript_confidence(segments)
    story_dependency = story.context_dependency_score if story else 0.0
    candidate_dependency = float(candidate.feature_vector.get("context_dependency_score", 25)) / 100
    context_dependency = _bounded(max(story_dependency, candidate_dependency, hook.context_dependency.score))
    standalone = _bounded(story.standalone_score if story else float(candidate.feature_vector.get("completeness_score", 0)) / 100)
    clarity = _bounded((hook.immediate_clarity.score * 0.55) + (1 - float(candidate.feature_vector.get("filler_word_ratio", 0))) * 0.25 + (1 - context_dependency) * 0.20)
    usefulness = _bounded(0.18 + min(0.42, actionable_count * 0.10) + (0.18 if _contains_any(candidate.text, ("нужно", "сделай", "выберите", "should", "do ", "choose")) else 0) + specificity * 0.18)
    relatability = _bounded(0.18 + (0.30 if _contains_any(candidate.text, ("ты", "мы", "каждый", "you", "we", "people")) else 0) + emotional.emotional_change.score * 0.17)
    quote_score = max((item.memorability.score for item in quote_candidates), default=0.0)
    momentum = _bounded(0.34 + emotional.escalation_strength.score * 0.24 + payoff.setup_payoff_alignment.score * 0.28 + min(0.14, float(audio.get("audio_energy_change", 0)) * 0.45) - float(candidate.feature_vector.get("filler_word_ratio", 0)) * 0.22)
    repetition = _bounded(float(candidate.feature_vector.get("repetition_score", 0)))
    filler = _bounded(float(candidate.feature_vector.get("filler_word_ratio", 0)))
    confusion = _bounded(context_dependency * 0.72 + (0.18 if hook.weak_opening_reason else 0))
    weak_ending = _bounded(1 - payoff.ending_satisfaction.score)
    platform_fit = _bounded(0.80 if 15 <= candidate.duration <= 60 else 0.54 if 10 <= candidate.duration <= 75 else 0.22)
    retention = _bounded(hook.hook_strength.score * 0.24 + hook.curiosity_gap.score * 0.10 + emotional.escalation_strength.score * 0.16 + momentum * 0.22 + payoff.payoff_strength.score * 0.28 - (hook.slow_start_penalty.score + filler + repetition) * 0.10)
    publishability = _bounded(standalone * 0.26 + clarity * 0.19 + payoff.ending_satisfaction.score * 0.23 + platform_fit * 0.15 + (1 - confusion) * 0.17)
    visual_available = bool((visual_analysis or {}).get("subject_keyframes") or (visual_analysis or {}).get("samples"))
    confidence = _bounded(transcript_confidence * 0.64 + (0.20 if audio_features.get("energy_frames") else 0.0) + (0.10 if visual_available else 0.0) + (0.06 if story else 0.0))
    excerpt = candidate.text[:320]
    features = {
        "hook_strength": hook.hook_strength,
        "curiosity_gap": hook.curiosity_gap,
        "emotional_intensity": emotional.emotional_peak_level,
        "emotional_progression": emotional.escalation_strength,
        "conflict_tension": conflict.conflict_strength,
        "surprise_novelty": _feature((novelty + surprise) / 2, transcript_confidence, "Новизна сравнивается только с другими StoryUnit этого source.", source="content_map", raw=novelty, segment_ids=ids, excerpt=excerpt),
        "specificity": _feature(specificity, transcript_confidence, "Конкретные детали, действия и ограничения абстрактности.", source="candidate_transcript", raw=number_count + actionable_count, segment_ids=ids, excerpt=excerpt),
        "clarity": _feature(clarity, transcript_confidence, "Ясность не смешивается с уверенностью модели.", source="candidate_transcript", raw=clarity, segment_ids=ids, excerpt=excerpt),
        "relatability": _feature(relatability, transcript_confidence, "Потенциальная узнаваемость ситуации внутри текста.", source="candidate_transcript", raw=relatability, segment_ids=ids, excerpt=excerpt),
        "usefulness": _feature(usefulness, transcript_confidence, "Практическая или информационная ценность candidate.", source="candidate_transcript", raw=actionable_count, segment_ids=ids, excerpt=excerpt),
        "controversy_potential": _feature(_bounded(conflict.conflict_presence.score * 0.34), transcript_confidence, "Обсуждаемость не приравнивается к качеству или shareability.", source="candidate_transcript", raw=conflict.conflict_presence.score, segment_ids=ids, excerpt=excerpt),
        "quotability": _feature(quote_score, transcript_confidence, "Сильная короткая цитата с понятным смыслом вне полного видео.", source="candidate_sentence", raw=quote_score, segment_ids=ids, excerpt=quote_candidates[0].text if quote_candidates else excerpt),
        "narrative_momentum": _feature(momentum, _bounded(transcript_confidence * 0.8 + 0.15), "Развитие мысли, новая информация и приближение к payoff.", source="audio_transcript", raw=momentum, segment_ids=ids, excerpt=excerpt),
        "payoff_strength": payoff.payoff_strength,
        "ending_satisfaction": payoff.ending_satisfaction,
        "standalone_strength": _feature(standalone, _bounded(transcript_confidence * 0.85), "Самостоятельность берётся из grounded StoryUnit.", source="story_unit", raw=standalone, segment_ids=ids, excerpt=excerpt),
        "context_independence": _feature(1 - context_dependency, transcript_confidence, "Независимость от не показанного предыдущего контекста.", source="candidate_opening", raw=context_dependency, segment_ids=ids[:1], excerpt=excerpt),
        "speech_energy": _feature(float(audio.get("audio_energy", 0)), _bounded(0.72 if audio_features.get("energy_frames") else 0.35), "Средняя нормализованная аудио-энергия без предположения, что громкость равна качеству.", source="audio_features", raw=audio.get("audio_energy", 0), segment_ids=[], excerpt=""),
        "pacing_quality": _feature(_bounded(1 - min(1.0, abs(float(candidate.feature_vector.get("words_per_second", 2.5)) - 2.5) / 2.5)), transcript_confidence, "Соответствие темпа комфортной речи; это не оценка смысла.", source="transcript_features", raw=candidate.feature_vector.get("words_per_second", 0), segment_ids=ids, excerpt=excerpt),
        "information_density": _feature(_bounded(float(candidate.feature_vector.get("speech_density", story.information_density if story else 0))), transcript_confidence, "Плотность содержательной речи в candidate.", source="transcript_features", raw=candidate.feature_vector.get("speech_density", 0), segment_ids=ids, excerpt=excerpt),
        "repetition_penalty": _feature(repetition, transcript_confidence, "Штраф за повторяющиеся слова и близкие тезисы source.", source="transcript_features", raw=repetition, segment_ids=ids, excerpt=excerpt),
        "filler_penalty": _feature(filler, transcript_confidence, "Штраф за filler, не изменение исходного текста.", source="transcript_features", raw=filler, segment_ids=ids, excerpt=excerpt),
        "confusion_penalty": _feature(confusion, transcript_confidence, "Штраф за зависимость от отсутствующего контекста.", source="candidate_opening", raw=context_dependency, segment_ids=ids[:1], excerpt=excerpt),
        "slow_start_penalty": hook.slow_start_penalty,
        "weak_ending_penalty": _feature(weak_ending, transcript_confidence, "Штраф за слабый или отсутствующий payoff на естественной границе.", source="candidate_ending", raw=weak_ending, segment_ids=ids[-1:], excerpt=candidate.text[-260:]),
        "platform_fit": _feature(platform_fit, 0.9, "Пригодность длительности для вертикального short; это не прогноз платформы.", source="candidate_range", raw=candidate.duration, segment_ids=[], excerpt=""),
        "publishability": _feature(publishability, confidence, "Предварительная пригодность к публикации без смыслового редактирования.", source="candidate_contract", raw=publishability, segment_ids=ids, excerpt=excerpt),
        "retention_potential": _feature(retention, confidence, "Относительный internal retention potential, не процент реального удержания.", source="candidate_curve", raw=retention, segment_ids=ids, excerpt=excerpt),
        "analysis_confidence": _feature(confidence, confidence, "Надёжность transcript/audio/visual evidence отображается отдельно от качества.", source="analysis_inputs", raw=visual_available, segment_ids=ids, excerpt=""),
    }
    result = ViralityFeatureProfile(
        schema_version=VIRALITY_SCHEMA_VERSION, candidate_id=candidate.id,
        story_unit_id=str(candidate.story_unit_id or "unknown"), content_strategy=content_strategy,
        features=features, hook_assessment=hook, emotional_arc=emotional,
        conflict_assessment=conflict, payoff_assessment=payoff, quote_candidates=quote_candidates,
        analysis_confidence=features["analysis_confidence"], warnings=[] if visual_available else ["Недостаточно визуальных данных: использована transcript/audio оценка."],
    )
    result.validate(allowed_ids)
    return result


def _candidate_zone_segments(start: float, end: float, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in segments
        if float(item.get("end", start)) > start and float(item.get("start", end)) < end
    ]


def _filler_ratio_for_zone(segments: list[dict[str, Any]], text: str) -> float:
    explicit = _average(float(item.get("filler_word_ratio", 0)) for item in segments)
    words = _words(text)
    lexical = sum(word in {"um", "uh", "erm", "like", "well", "ну", "ээ", "это"} for word in words) / max(1, len(words))
    return _bounded(max(explicit, lexical * 2.5))


def _repetition_ratio_for_zone(segments: list[dict[str, Any]], text: str) -> float:
    explicit = _average(float(item.get("repetition_score", 0)) for item in segments)
    words = _words(text)
    if not words:
        return _bounded(explicit)
    repeated = sum(count - 1 for count in Counter(words).values() if count > 1)
    lexical = repeated / len(words)
    return _bounded(max(explicit, lexical))


def _retention_zone(
    candidate: Candidate, profile: ViralityFeatureProfile, name: str, start_ratio: float, end_ratio: float,
    segments: list[dict[str, Any]], prior_words: set[str], audio_features: dict[str, Any],
) -> RetentionZone:
    start = candidate.start + candidate.duration * start_ratio
    end = candidate.start + candidate.duration * end_ratio
    rows = _candidate_zone_segments(start, end, segments)
    ids = [int(item.get("id", -1)) for item in rows if int(item.get("id", -1)) >= 0]
    text = " ".join(str(item.get("text") or "") for item in rows).strip()
    words = _words(text)
    audio = window_audio_features(start, end, audio_features)
    new_words = set(words) - prior_words
    density = _average(float(item.get("speech_density", 0)) for item in rows)
    information_gain = _bounded(0.14 + density * 0.32 + min(0.32, len(new_words) / max(1, len(words)) * 0.36)) if words else 0.12
    filler = _filler_ratio_for_zone(rows, text)
    repetition = _repetition_ratio_for_zone(rows, text)
    emotion_label, emotion = _emotion(text)
    if rows:
        emotion = _bounded(emotion + _average(float(item.get("exclamation_count", 0)) for item in rows) * 0.06)
    contextual = any(str(item.get("text") or "").strip().casefold().startswith(_CONTEXTUAL_PREFIXES) for item in rows)
    confusion = _bounded(
        (0.46 if contextual else 0.0)
        + _average(float(item.get("context_dependency_score", 0)) for item in rows) / 100 * 0.34
        + (0.18 if not words else 0.0)
    )
    payoff_time = profile.payoff_assessment.payoff_timestamp
    hook_open = profile.hook_assessment.curiosity_opened
    if hook_open and payoff_time is not None and end >= payoff_time:
        curiosity_state = "resolved"
    elif hook_open and start < (payoff_time if payoff_time is not None else candidate.end):
        curiosity_state = "open"
    else:
        curiosity_state = "none"
    zone_hook = profile.hook_assessment.hook_strength.score if name == "opening" else 0.0
    payoff_bonus = profile.payoff_assessment.payoff_strength.score if payoff_time is not None and start <= payoff_time <= end else 0.0
    momentum = _bounded(
        profile.features["narrative_momentum"].score * 0.42 + information_gain * 0.28
        + min(0.16, float(audio.get("audio_energy_change", 0)) * 0.45)
        + (0.12 if curiosity_state in {"open", "resolved"} else 0.0) + payoff_bonus * 0.14
        - repetition * 0.18 - filler * 0.16 - confusion * 0.18
    )
    attention = _bounded(
        0.14 + zone_hook * 0.30 + information_gain * 0.23 + emotion * 0.18 + momentum * 0.23
        + payoff_bonus * 0.12 - repetition * 0.20 - filler * 0.22 - confusion * 0.22
    )
    drop_risk = _bounded(
        (1 - attention) * 0.40 + repetition * 0.23 + filler * 0.24 + confusion * 0.20
        + (0.12 if not words else 0.0) - payoff_bonus * 0.08
    )
    confidence = _bounded(_transcript_confidence(rows) * 0.72 + (0.20 if audio_features.get("energy_frames") else 0.0))
    excerpt = text[:480]
    common = {
        "source": "retention_zone", "segment_ids": ids, "excerpt": excerpt,
    }
    evidence = [FeatureEvidence("retention_zone", name, attention, confidence, ids, excerpt)]
    prior_words.update(words)
    return RetentionZone(
        name=name, start=round(start, 3), end=round(end, 3),
        attention_strength=_feature(attention, confidence, "Внутренний сравнительный сигнал внимания для временной зоны, не прогноз реальных просмотров.", raw=attention, **common),
        information_gain=_feature(information_gain, confidence, "Новая информация относительно уже пройденной части candidate.", raw=len(new_words), **common),
        emotional_energy=_feature(emotion, confidence, f"Эмоциональная энергия зоны: {emotion_label}.", raw=emotion_label, **common),
        curiosity_state=curiosity_state,
        narrative_momentum=_feature(momentum, confidence, "Темп развития мысли в этой зоне с учётом новой информации и payoff.", raw=momentum, **common),
        repetition=_feature(repetition, confidence, "Повтор внутри временной зоны повышает риск потери внимания.", raw=repetition, **common),
        confusion=_feature(confusion, confidence, "Контекстная неясность в зоне отделена от качества истории в целом.", raw=confusion, **common),
        drop_risk=_feature(drop_risk, confidence, "Сравнительный риск потери внимания; не оценка реального процента удержания.", raw=drop_risk, **common),
        evidence=evidence,
    )


def _dead_zones_from_retention(
    candidate: Candidate, zones: list[RetentionZone], minimum_seconds: float,
) -> list[DeadZone]:
    result: list[DeadZone] = []
    for zone in zones:
        severity = _bounded(
            zone.repetition.score * 0.27 + zone.confusion.score * 0.23
            + zone.drop_risk.score * 0.34 + (1 - zone.information_gain.score) * 0.16
        )
        if zone.end - zone.start < minimum_seconds or severity < 0.56:
            continue
        reasons: list[str] = []
        if zone.repetition.score >= 0.25:
            reasons.append("repetition")
        if zone.confusion.score >= 0.35:
            reasons.append("context_or_transition")
        if zone.information_gain.score <= 0.35:
            reasons.append("low_information_gain")
        if zone.drop_risk.score >= 0.60:
            reasons.append("drop_risk")
        reason = ", ".join(reasons) or "flat_development"
        ids = [item for evidence in zone.evidence for item in evidence.segment_ids]
        confidence = _average((zone.drop_risk.confidence, zone.information_gain.confidence), 0.5)
        result.append(DeadZone(
            start=zone.start, end=zone.end, duration=round(zone.end - zone.start, 3), reason=reason,
            severity=_feature(severity, confidence, "Диагностика слабой зоны для будущей реконструкции; этот этап не меняет исходный candidate.", source="retention_zone", raw=reason, segment_ids=ids, excerpt=zone.evidence[0].excerpt if zone.evidence else ""),
            removable_in_future=True,
            evidence=[FeatureEvidence("retention_zone", reason, severity, confidence, ids, zone.evidence[0].excerpt if zone.evidence else "")],
        ))
    return result


def _retention_summary_feature(
    score: float, confidence: float, explanation: str, candidate: Candidate, source: str,
) -> FeatureScore:
    return _feature(
        score, confidence, explanation, source=source, raw=score,
        segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[:360],
    )


def build_estimated_retention_profile(
    candidate: Candidate, feature_profile: ViralityFeatureProfile, transcript_features: dict[str, Any],
    audio_features: dict[str, Any], *, dead_zone_minimum_seconds: float = 1.4,
) -> EstimatedRetentionProfile:
    """Estimate internal retention signals without inventing platform analytics."""

    if candidate.id != feature_profile.candidate_id or candidate.duration <= 0:
        raise ValueError("Retention profile must use the matching positive-duration candidate.")
    if not 0.2 <= dead_zone_minimum_seconds <= 20:
        raise ValueError("dead_zone_minimum_seconds must stay in the safe configured range.")
    segments = _candidate_segments(candidate, transcript_features)
    prior_words: set[str] = set()
    zones = [
        _retention_zone(candidate, feature_profile, name, start_ratio, end_ratio, segments, prior_words, audio_features)
        for name, start_ratio, end_ratio in RETENTION_ZONE_DEFINITIONS
    ]
    dead_zones = _dead_zones_from_retention(candidate, zones, dead_zone_minimum_seconds)
    by_name = {item.name: item for item in zones}
    opening = by_name["opening"].attention_strength.score
    early = _average((by_name["opening"].attention_strength.score, by_name["early"].attention_strength.score))
    mid = _average((by_name["mid"].attention_strength.score, by_name["late"].attention_strength.score))
    late = _average((by_name["pre_payoff"].attention_strength.score, by_name["ending"].attention_strength.score))
    payoff_position = (
        (feature_profile.payoff_assessment.payoff_timestamp - candidate.start) / candidate.duration
        if feature_profile.payoff_assessment.payoff_timestamp is not None else None
    )
    delayed_payoff = payoff_position is not None and payoff_position > 0.78
    if delayed_payoff:
        mid = _bounded(mid - 0.10)
    dead_ratio = sum(zone.duration for zone in dead_zones) / candidate.duration
    completion = _bounded(
        late * 0.28 + feature_profile.payoff_assessment.payoff_strength.score * 0.34
        + feature_profile.features["narrative_momentum"].score * 0.17
        + feature_profile.payoff_assessment.ending_satisfaction.score * 0.17
        + (0.08 if payoff_position is not None and 0.55 <= payoff_position <= 0.92 else 0)
        - dead_ratio * 0.30 - (0.12 if delayed_payoff else 0)
    )
    if candidate.duration < 8 and feature_profile.features["narrative_momentum"].score < 0.60:
        completion = min(completion, 0.58)
    zone_strengths = {
        item.name: _bounded(item.attention_strength.score * 0.48 + item.narrative_momentum.score * 0.32 + item.emotional_energy.score * 0.20)
        for item in zones
    }
    strongest = max(zones, key=lambda item: (zone_strengths[item.name], -item.start))
    weakest = min(zones, key=lambda item: (zone_strengths[item.name], item.start))
    drop_points = sorted({
        round((item.start + item.end) / 2, 3) for item in zones if item.drop_risk.score >= 0.56
    } | {round((item.start + item.end) / 2, 3) for item in dead_zones})
    confidence = _bounded(
        feature_profile.analysis_confidence.score * 0.70
        + (0.18 if audio_features.get("energy_frames") else 0.0)
        + (0.12 if segments else 0.0)
    )
    result = EstimatedRetentionProfile(
        candidate_id=candidate.id, candidate_start=round(candidate.start, 3), candidate_end=round(candidate.end, 3), zones=zones,
        opening_retention=_retention_summary_feature(opening, confidence, "Сравнительный opening retention index, не процент реальных просмотров.", candidate, "retention_opening"),
        early_retention=_retention_summary_feature(early, confidence, "Сравнительный early retention index по первым двум зонам.", candidate, "retention_early"),
        mid_retention=_retention_summary_feature(mid, confidence, "Сравнительный mid retention index; поздний payoff учитывает риск провисания середины.", candidate, "retention_mid"),
        late_retention=_retention_summary_feature(late, confidence, "Сравнительный late retention index перед естественным завершением.", candidate, "retention_late"),
        completion_potential=_retention_summary_feature(completion, confidence, "Сравнительный потенциал досмотра по payoff, завершению и слабым зонам; не прогноз платформы.", candidate, "retention_completion"),
        estimated_drop_points=drop_points,
        strongest_moment_timestamp=round((strongest.start + strongest.end) / 2, 3),
        weakest_moment_timestamp=round((weakest.start + weakest.end) / 2, 3),
        dead_zone_ranges=dead_zones,
        retention_confidence=_retention_summary_feature(confidence, confidence, "Надёжность retention-диагностики отображается отдельно от её качества.", candidate, "retention_confidence"),
        warnings=["Retention values are comparative internal indices, not projected viewer percentages."],
    )
    result.validate(set(candidate.transcript_segment_ids))
    return result


def build_publishability_assessment(
    candidate: Candidate, feature_profile: ViralityFeatureProfile, retention_profile: EstimatedRetentionProfile,
    visual_diagnostics: dict[str, Any] | None = None,
) -> PublishabilityAssessment:
    """Assess unedited publishability separately from content-strength ranking."""

    if candidate.id != feature_profile.candidate_id or candidate.id != retention_profile.candidate_id:
        raise ValueError("Publishability assessment requires matching candidate diagnostics.")
    visual = visual_diagnostics or {}
    ids = list(candidate.transcript_segment_ids)
    confidence = _bounded(_average((feature_profile.analysis_confidence.score, retention_profile.retention_confidence.score), 0.5))
    diagnostics = candidate.boundary_diagnostics or {}
    inferred_boundary = candidate.text.rstrip().endswith((".", "!", "?", "…"))
    boundary_ok = bool(diagnostics.get("eligible", inferred_boundary))
    boundary_value = _bounded(float(diagnostics.get("overall_boundary_score", 0.82 if boundary_ok else 0.08)))
    visual_status = str(visual.get("composition_quality_status") or visual.get("status") or "unknown")
    visual_warning = visual_status in {"passed_with_warning", "warning", "safe_fallback"} or bool(visual.get("warnings"))
    visual_failed = visual_status in {"failed", "failed_repairable", "unsafe"}
    visual_score = 0.28 if visual_failed else 0.62 if visual_warning else 0.74
    subtitle_ready = visual.get("subtitle_compatible", visual.get("subtitle_ready", True))
    subtitle_score = 0.35 if subtitle_ready is False else 0.74
    duration_score = 0.82 if 12 <= candidate.duration <= 75 else 0.60 if 8 <= candidate.duration <= 90 else 0.26
    opening = feature_profile.hook_assessment.immediate_clarity.score
    standalone = feature_profile.features["standalone_strength"].score
    completeness = _bounded(
        feature_profile.payoff_assessment.ending_satisfaction.score * 0.58
        + feature_profile.features["clarity"].score * 0.22
        + (0.20 if boundary_ok else 0.0)
    )
    context = feature_profile.features["context_independence"].score
    signal = feature_profile.analysis_confidence.score
    payoff = feature_profile.payoff_assessment.payoff_strength.score
    filler_control = _bounded(1 - feature_profile.features["filler_penalty"].score)
    score = _bounded(
        opening * 0.15 + standalone * 0.15 + completeness * 0.13 + boundary_value * 0.16
        + context * 0.12 + duration_score * 0.08 + signal * 0.06 + payoff * 0.08
        + filler_control * 0.03 + visual_score * 0.02 + subtitle_score * 0.02
    )
    critical: list[str] = []
    if not boundary_ok:
        critical.append("semantic_boundary_violation")
    if context < 0.20:
        critical.append("critical_context_dependency")
    if visual_failed:
        critical.append("visual_usability_failure")
    if signal < 0.18:
        critical.append("insufficient_source_signal")
    if not feature_profile.payoff_assessment.payoff_present and feature_profile.payoff_assessment.ending_satisfaction.score < 0.35:
        critical.append("incomplete_story")
    level = "blocked" if critical or score < 0.42 else "ready" if score >= 0.70 else "limited"
    reasons = [
        "clear_opening" if opening >= 0.58 else "opening_needs_context",
        "standalone_story" if standalone >= 0.58 else "standalone_is_limited",
        "payoff_present" if payoff >= 0.50 else "payoff_is_weak_or_missing",
        "boundary_safe" if boundary_ok else "boundary_is_not_safe",
    ]
    warnings = []
    if visual_warning:
        warnings.append("visual_composition_warning")
    if subtitle_ready is False:
        warnings.append("subtitle_compatibility_warning")
    result = PublishabilityAssessment(
        candidate_id=candidate.id, level=level,
        publishability_score=_retention_summary_feature(score, confidence, "Пригодность к публикации без смыслового редактирования оценивается отдельно от viral potential.", candidate, "publishability"),
        opening_clarity=_retention_summary_feature(opening, confidence, "Понятность первой законченной мысли.", candidate, "publishability_opening"),
        standalone_strength=_retention_summary_feature(standalone, confidence, "Самостоятельность StoryUnit в исходном контексте.", candidate, "publishability_standalone"),
        story_completeness=_retention_summary_feature(completeness, confidence, "Законченность мысли и естественное завершение.", candidate, "publishability_completeness"),
        boundary_safety=_retention_summary_feature(boundary_value, confidence, "Безопасность semantic boundary без переписывания source.", candidate, "publishability_boundary"),
        context_independence=_retention_summary_feature(context, confidence, "Понятность без отсутствующего предыдущего контекста.", candidate, "publishability_context"),
        duration_fit=_retention_summary_feature(duration_score, 0.92, "Пригодность длительности для vertical short.", candidate, "publishability_duration"),
        source_signal_quality=_retention_summary_feature(signal, confidence, "Качество доступных transcript/audio/visual observations.", candidate, "publishability_source_signal"),
        payoff_presence=_retention_summary_feature(payoff, confidence, "Наличие самостоятельного payoff внутри candidate.", candidate, "publishability_payoff"),
        filler_control=_retention_summary_feature(filler_control, confidence, "Отсутствие чрезмерного filler в исходном candidate.", candidate, "publishability_filler"),
        visual_usability=_retention_summary_feature(visual_score, 0.65 if visual_status == "unknown" else 0.82, "Визуальная пригодность использует только доступные diagnostics и не дублирует quality validator.", candidate, "publishability_visual"),
        subtitle_compatibility=_retention_summary_feature(subtitle_score, 0.65 if visual_status == "unknown" else 0.82, "Совместимость с субтитрами — отдельный мягкий сигнал.", candidate, "publishability_subtitles"),
        reasons=reasons, critical_failures=critical, warnings=warnings,
    )
    result.validate(set(ids))
    return result


def assess_candidate_eligibility(
    candidate: Candidate, feature_profile: ViralityFeatureProfile, retention_profile: EstimatedRetentionProfile,
    publishability: PublishabilityAssessment,
) -> EligibilityAssessment:
    """Return deterministic Goal 5B routing without reconstructing the source."""

    if len({candidate.id, feature_profile.candidate_id, retention_profile.candidate_id, publishability.candidate_id}) != 1:
        raise ValueError("Eligibility assessment requires matching candidate diagnostics.")
    critical = list(publishability.critical_failures)
    serious_dead = [zone for zone in retention_profile.dead_zone_ranges if zone.severity.score >= 0.72]
    minor_dead = [zone for zone in retention_profile.dead_zone_ranges if zone.severity.score >= 0.56]
    payoff_position = (
        (feature_profile.payoff_assessment.payoff_timestamp - candidate.start) / candidate.duration
        if feature_profile.payoff_assessment.payoff_timestamp is not None else None
    )
    strong_core = _average((
        feature_profile.features["standalone_strength"].score,
        feature_profile.features["clarity"].score,
        feature_profile.features["usefulness"].score,
        feature_profile.features["narrative_momentum"].score,
        retention_profile.completion_potential.score,
    ))
    opportunities: list[str] = []
    if feature_profile.hook_assessment.hook_strength.score < 0.46:
        opportunities.append("weak_original_opening")
    if payoff_position is not None and payoff_position > 0.78:
        opportunities.append("late_payoff")
    if serious_dead:
        opportunities.append("critical_dead_zone")
    elif minor_dead:
        opportunities.append("local_dead_zone")
    if critical:
        status = "rejected"
    elif strong_core < 0.40:
        status = "weak"
    elif (
        publishability.level == "ready" and feature_profile.payoff_assessment.payoff_present
        and feature_profile.hook_assessment.immediate_clarity.score >= 0.55 and not serious_dead
    ):
        status = "publishable_now"
    elif strong_core >= 0.58 and (opportunities or publishability.level == "limited"):
        status = "needs_reconstruction"
    elif publishability.publishability_score.score >= 0.52:
        status = "publishable_with_minor_adjustment"
    else:
        status = "weak"
    reasons = list(publishability.reasons)
    reasons.append(f"retention_{retention_profile.relative_level(retention_profile.completion_potential.score)}")
    if status == "needs_reconstruction":
        reasons.append("strong_idea_requires_future_story_reconstruction")
    if status == "rejected":
        reasons.append("critical_publishability_failure")
    confidence = _bounded(_average((feature_profile.analysis_confidence.score, retention_profile.retention_confidence.score, publishability.publishability_score.confidence)))
    result = EligibilityAssessment(
        candidate_id=candidate.id, status=status, reasons=reasons, critical_failures=critical,
        reconstruction_opportunities=opportunities,
        confidence=_retention_summary_feature(confidence, confidence, "Уверенность в eligibility основана на доступных source diagnostics, а не на viral score.", candidate, "eligibility_confidence"),
    )
    result.validate(set(candidate.transcript_segment_ids))
    return result


def resolve_virality_strategy(content_profile: dict[str, Any] | None) -> str:
    strategy = str((content_profile or {}).get("strategy_id") or "generic_fallback")
    return strategy if strategy in {
        "motivational_monologue", "generic_monologue", "generic_dialogue", "generic_educational",
        "generic_scene_driven", "generic_fallback",
    } else "generic_fallback"


def build_virality_assessments(
    candidates: list[Candidate], content_map: dict[str, Any], transcript_features: dict[str, Any],
    audio_features: dict[str, Any], visual_analysis: dict[str, Any] | None,
    content_profile: dict[str, Any] | None, settings: Any,
) -> dict[str, Any]:
    """Build source-scoped deterministic diagnostics without changing source or render data."""

    strategy_id = resolve_virality_strategy(content_profile)
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        feature = build_virality_feature_profile(
            candidate, content_map, transcript_features, audio_features, visual_analysis, strategy_id,
        )
        retention = build_estimated_retention_profile(
            candidate, feature, transcript_features, audio_features,
            dead_zone_minimum_seconds=float(settings.dead_zone_minimum_seconds),
        )
        publishability = build_publishability_assessment(candidate, feature, retention, visual_analysis)
        eligibility = assess_candidate_eligibility(candidate, feature, retention, publishability)
        items.append({
            "candidate_id": candidate.id, "feature_profile": feature.to_dict(), "retention_profile": retention.to_dict(),
            "publishability": publishability.to_dict(), "eligibility": eligibility.to_dict(),
        })
    semantic_mode = str(getattr(settings, "semantic_ai_mode", "auto"))
    return {
        "schema_version": VIRALITY_SCHEMA_VERSION, "strategy_id": strategy_id,
        "analysis_mode": "deterministic", "candidates": items,
        "semantic_ai": {
            "requested_mode": semantic_mode, "used": False,
            "fallback_used": semantic_mode != "off",
            "reason": "Grounded deterministic fallback is active; no semantic AI score was required for this source.",
        },
        "cost": {
            "estimated_ai_cost": 0.0, "actual_ai_cost": 0.0, "cache_savings": 0.0,
            "tokens_per_candidate": 0, "batch_count": 0, "fallback_usage": len(candidates) if semantic_mode != "off" else 0,
        },
    }


def _contribution(name: str, signal: FeatureScore, weight: float, explanation: str) -> ScoreContribution:
    normalized = _bounded(signal.score)
    confidence_adjusted = _bounded(normalized * 0.88 + signal.confidence * 0.12)
    return ScoreContribution(
        name=name, raw_score=signal.score, normalized_score=normalized,
        confidence_adjusted_score=confidence_adjusted, confidence=signal.confidence,
        strategy_weight=_bounded(weight), contribution=_bounded(normalized * weight),
        explanation=explanation, evidence=list(signal.evidence),
    )


def _confidence_model(
    candidate: Candidate, profile: ViralityFeatureProfile, retention: EstimatedRetentionProfile,
    publishability: PublishabilityAssessment, content_profile: dict[str, Any] | None,
) -> ViralityConfidence:
    ids = list(candidate.transcript_segment_ids)
    source_confidence = _bounded(float((content_profile or {}).get("analysis_confidence", 0.6)))
    visual_available = not any("визуаль" in warning.casefold() or "visual" in warning.casefold() for warning in profile.warnings)
    evidence_coverage = _bounded(sum(bool(item.evidence) for item in profile.features.values()) / max(1, len(profile.features)))
    completeness = publishability.story_completeness.score
    consistency = _bounded(1 - max(
        abs(profile.features["retention_potential"].score - retention.completion_potential.score),
        abs(profile.features["publishability"].score - publishability.publishability_score.score),
    ))
    factors = {
        "transcript": _retention_summary_feature(profile.analysis_confidence.confidence, profile.analysis_confidence.confidence, "Надёжность transcript-derived evidence.", candidate, "confidence_transcript"),
        "semantic_evidence_coverage": _retention_summary_feature(evidence_coverage, profile.analysis_confidence.confidence, "Покрытие score components локальными evidence.", candidate, "confidence_evidence"),
        "audio": _retention_summary_feature(profile.features["speech_energy"].confidence, profile.features["speech_energy"].confidence, "Доступность audio features.", candidate, "confidence_audio"),
        "visual": _retention_summary_feature(0.82 if visual_available else 0.42, 0.82 if visual_available else 0.42, "Доступность visual observations отображается отдельно от качества истории.", candidate, "confidence_visual"),
        "content_type": _retention_summary_feature(source_confidence, source_confidence, "Уверенность в source content strategy.", candidate, "confidence_content_type"),
        "score_consistency": _retention_summary_feature(consistency, profile.analysis_confidence.confidence, "Согласованность independent deterministic signals.", candidate, "confidence_consistency"),
        "candidate_completeness": _retention_summary_feature(completeness, publishability.story_completeness.confidence, "Завершённость candidate для надёжности оценки.", candidate, "confidence_completeness"),
        "ai_local_agreement": _retention_summary_feature(0.72, 0.72, "Semantic AI не применялся: сохранён нейтральный deterministic agreement.", candidate, "confidence_ai_local"),
    }
    overall = _bounded(_average(item.score for item in factors.values()))
    warnings = [] if visual_available else ["Visual observations are unavailable; content/audio evidence remains usable."]
    result = ViralityConfidence(
        overall=_retention_summary_feature(overall, overall, "Confidence is reported separately and only limits ranking tie-breaks.", candidate, "virality_confidence"),
        factors=factors, warnings=warnings,
    )
    result.validate(set(ids))
    return result


def _potential_level(score: float) -> str:
    if score >= 0.78:
        return "excellent"
    if score >= 0.62:
        return "strong"
    if score >= 0.43:
        return "moderate"
    return "weak"


def aggregate_viral_potential(
    candidate: Candidate, feature_profile: ViralityFeatureProfile, retention_profile: EstimatedRetentionProfile,
    publishability: PublishabilityAssessment, eligibility: EligibilityAssessment,
    strategy_weights: dict[str, float], content_profile: dict[str, Any] | None = None,
    *, dead_zone_penalty_weight: float = 0.10,
) -> ViralPotentialScore:
    """Code-owned score aggregation; confidence is visible but never replaces quality."""

    if set(strategy_weights) != set(VIRALITY_COMPONENTS):
        raise ValueError("Strategy weights must cover each ViralPotentialScore component.")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in strategy_weights.values()):
        raise ValueError("Strategy weights must be finite non-negative values.")
    if abs(sum(float(value) for value in strategy_weights.values()) - 1.0) > 0.001:
        raise ValueError("Strategy weights must sum to one.")
    if not 0 <= dead_zone_penalty_weight <= 1:
        raise ValueError("dead_zone_penalty_weight must be bounded.")
    components_signals = {
        "hook": feature_profile.features["hook_strength"],
        "curiosity": feature_profile.features["curiosity_gap"],
        "emotion": _feature(
            _average((feature_profile.features["emotional_intensity"].score, feature_profile.features["emotional_progression"].score)),
            _average((feature_profile.features["emotional_intensity"].confidence, feature_profile.features["emotional_progression"].confidence)),
            "Combined emotional intensity and progression.", source="feature_profile", raw="emotion", segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[:320],
        ),
        "conflict": _feature(
            _average((feature_profile.features["conflict_tension"].score, feature_profile.hook_assessment.stakes.score)),
            _average((feature_profile.features["conflict_tension"].confidence, feature_profile.hook_assessment.stakes.confidence)),
            "Conflict strength and visible stakes.", source="feature_profile", raw="conflict", segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[:320],
        ),
        "specificity": feature_profile.features["specificity"],
        "novelty": feature_profile.features["surprise_novelty"],
        "usefulness": feature_profile.features["usefulness"],
        "quotability": feature_profile.features["quotability"],
        "momentum": feature_profile.features["narrative_momentum"],
        "payoff": feature_profile.features["payoff_strength"],
        "retention": _feature(
            _average((retention_profile.completion_potential.score, retention_profile.early_retention.score, retention_profile.late_retention.score)),
            retention_profile.retention_confidence.score, "Relative retention and completion potential.", source="retention_profile", raw="retention", segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[:320],
        ),
        "publishability": publishability.publishability_score,
    }
    components = {
        name: _contribution(name, signal, float(strategy_weights[name]), f"Strategy-weighted {name} contribution.")
        for name, signal in components_signals.items()
    }
    boundary = candidate.boundary_diagnostics or {}
    boundary_risk = _bounded(1 - float(boundary.get("overall_boundary_score", 0.82 if boundary.get("eligible", True) else 0.08)))
    dead_ratio = _bounded(sum(zone.duration * zone.severity.score for zone in retention_profile.dead_zone_ranges) / max(candidate.duration, 0.01))
    raw_penalties = {
        "slow_start": feature_profile.features["slow_start_penalty"],
        "context_dependency": _feature(1 - feature_profile.features["context_independence"].score, feature_profile.features["context_independence"].confidence, "Context dependence penalty.", source="feature_profile", raw="context", segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[:320]),
        "unresolved_curiosity": feature_profile.hook_assessment.unresolved_curiosity_penalty,
        "missing_payoff": _feature(1 - feature_profile.features["payoff_strength"].score, feature_profile.features["payoff_strength"].confidence, "Missing payoff penalty.", source="feature_profile", raw="payoff", segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[-320:]),
        "repetition": feature_profile.features["repetition_penalty"],
        "filler": feature_profile.features["filler_penalty"],
        "confusion": feature_profile.features["confusion_penalty"],
        "dead_zone": _feature(dead_ratio, retention_profile.retention_confidence.score, "Dead-zone duration and severity are diagnostic only; no source is edited.", source="retention_profile", raw=dead_ratio, segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[:320]),
        "weak_ending": feature_profile.features["weak_ending_penalty"],
        "boundary_risk": _feature(boundary_risk, 0.9, "Semantic boundary risk has priority over high hook scores.", source="semantic_boundary", raw=boundary_risk, segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.text[-320:]),
        "semantic_duplication": _feature(_bounded(float(candidate.feature_vector.get("semantic_duplicate_score", 0))), feature_profile.analysis_confidence.score, "Source-relative semantic duplication penalty.", source="content_signature", raw=candidate.feature_vector.get("semantic_duplicate_score", 0), segment_ids=list(candidate.transcript_segment_ids), excerpt=candidate.core_idea[:320]),
    }
    penalty_weights = dict(PENALTY_WEIGHTS)
    penalty_weights["dead_zone"] = dead_zone_penalty_weight
    penalties = {
        name: _contribution(name, signal, penalty_weights[name], f"Bounded {name} penalty.")
        for name, signal in raw_penalties.items()
    }
    positive = _bounded(sum(item.contribution for item in components.values()))
    # Components already express missing payoff, unclear context and flatness.
    # Penalties are a bounded corrective signal rather than a second full score.
    penalty = _bounded(sum(item.contribution for item in penalties.values()) * 0.25)
    viral_score = _bounded(positive - penalty)
    confidence = _confidence_model(candidate, feature_profile, retention_profile, publishability, content_profile)
    strongest = [item.name for item in sorted(components.values(), key=lambda item: (-item.contribution, item.name))[:3] if item.contribution > 0]
    weakest = [item.name for item in sorted(penalties.values(), key=lambda item: (-item.contribution, item.name))[:3] if item.contribution > 0]
    reason = (
        f"Comparative potential is driven by {', '.join(strongest) or 'available content signals'}"
        f"; main constraints: {', '.join(weakest) or 'none detected'}.")
    result = ViralPotentialScore(
        candidate_id=candidate.id, strategy_id=feature_profile.content_strategy, components=components, penalties=penalties,
        positive_score=positive, penalty_score=penalty, viral_potential_score=viral_score,
        retention_potential_score=retention_profile.completion_potential.score,
        publishability_score=publishability.publishability_score.score, level=_potential_level(viral_score), confidence=confidence,
        strongest_factors=strongest, weakest_factors=weakest, ranking_explanation=reason, eligibility_status=eligibility.status,
    )
    result.validate(set(candidate.transcript_segment_ids))
    return result


def apply_virality_ranking(
    scored: list[Any], assessment_data: dict[str, Any], settings: Any,
    content_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach code-owned scoring to existing candidates before Goal 5A coverage selection."""

    strategy = resolve_virality_strategy(content_profile)
    all_weights = getattr(settings, "strategy_weights", {})
    weights = dict(all_weights.get(strategy) or getattr(settings, "weights", {}))
    raw_items = assessment_data.get("candidates", []) if isinstance(assessment_data, dict) else []
    by_id = {str(item.get("candidate_id") or ""): item for item in raw_items if isinstance(item, dict)}
    ranked: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    for item in scored:
        raw = by_id.get(item.candidate.id)
        if raw is None:
            item.selected = False
            item.rejection_reason = "Missing virality assessment."
            item.virality = {"status": "missing_assessment"}
            ranked.append(item)
            continue
        feature = ViralityFeatureProfile.from_dict(dict(raw.get("feature_profile") or {}))
        retention = EstimatedRetentionProfile.from_dict(dict(raw.get("retention_profile") or {}))
        publishability = PublishabilityAssessment.from_dict(dict(raw.get("publishability") or {}))
        eligibility = EligibilityAssessment.from_dict(dict(raw.get("eligibility") or {}))
        potential = aggregate_viral_potential(
            item.candidate, feature, retention, publishability, eligibility, weights, content_profile,
            dead_zone_penalty_weight=float(getattr(settings, "dead_zone_penalty_weight", 0.10)),
        )
        passes_floor = potential.viral_potential_score >= float(getattr(settings, "minimum_quality_score", 0.52))
        publishable = publishability.publishability_score.score >= float(getattr(settings, "minimum_publishability_score", 0.55))
        allowed = eligibility.status in {"publishable_now", "publishable_with_minor_adjustment"} and passes_floor and publishable
        item.score = int(round(potential.viral_potential_score * 100))
        item.hook_score = int(round(feature.hook_assessment.hook_strength.score * 100))
        item.completeness_score = int(round(publishability.story_completeness.score * 100))
        item.emotional_score = int(round(feature.features["emotional_progression"].score * 100))
        item.clarity_score = int(round(feature.features["clarity"].score * 100))
        item.context_dependency_score = int(round((1 - feature.features["context_independence"].score) * 100))
        item.selected = allowed
        item.rejection_reason = None if allowed else {
            "rejected": "; ".join(eligibility.critical_failures) or "critical_publishability_failure",
            "needs_reconstruction": "requires_future_story_reconstruction",
            "weak": "weak_content_value",
        }.get(eligibility.status, "below_virality_or_publishability_floor")
        item.virality = {
            "feature_profile": feature.to_dict(), "retention_profile": retention.to_dict(),
            "publishability": publishability.to_dict(), "eligibility": eligibility.to_dict(),
            "viral_potential": potential.to_dict(),
            "selection_eligible": allowed,
            "ranking_sort_score": potential.viral_potential_score,
        }
        diagnostics.append({
            "candidate_id": item.candidate.id, "viral_potential_score": potential.viral_potential_score,
            "retention_potential_score": potential.retention_potential_score,
            "publishability_score": potential.publishability_score, "eligibility": eligibility.status,
            "passes_quality_floor": passes_floor, "passes_publishability_floor": publishable,
        })
        ranked.append(item)
    confidence_weight = _bounded(float(getattr(settings, "uncertainty_tiebreak_weight", 0.08)))
    ranked.sort(key=lambda value: (
        -float(value.virality.get("ranking_sort_score", value.score / 100)),
        -float(value.virality.get("viral_potential", {}).get("confidence", {}).get("overall", {}).get("score", 0)) * confidence_weight,
        value.candidate.id,
    ))
    for index, item in enumerate(ranked, 1):
        if item.virality:
            item.virality["overall_rank"] = index
    return {
        "schema_version": VIRALITY_SCHEMA_VERSION, "strategy_id": strategy,
        "candidates": [item.to_dict() for item in ranked], "ranking": diagnostics,
        "minimum_quality_score": float(getattr(settings, "minimum_quality_score", 0.52)),
        "minimum_publishability_score": float(getattr(settings, "minimum_publishability_score", 0.55)),
    }
