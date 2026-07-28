"""Goal 5B: grounded, comparative content-strength analysis.

The module deliberately describes *potential* inside one source.  It does not
predict platform reach, change text, or make render decisions.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from app.audio_features import window_audio_features
from app.content_understanding import StoryUnit
from app.models import Candidate


VIRALITY_SCHEMA_VERSION = "5B.1"
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

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "score": round(self.score, 6), "confidence": round(self.confidence, 6),
            "evidence": [item.to_dict() for item in self.evidence], "explanation": self.explanation,
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
            "features": {name: value.to_dict() for name, value in self.features.items()},
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
    present = marker or story_payoff or resolved
    payoff_type = "answer" if resolved else "conclusion" if marker else "insight" if story_payoff else "none"
    ending_natural = bool(candidate.boundary_diagnostics.get("eligible", False)) if candidate.boundary_diagnostics else bool(candidate.text.rstrip().endswith((".", "!", "?", "…")))
    strength = _bounded((0.46 if present else 0.0) + (0.20 if marker else 0) + (0.17 if resolved else 0) + (0.10 if ending_natural else 0))
    alignment = _bounded(0.82 if question and resolved else 0.62 if present else 0.0)
    confidence = _transcript_confidence(segments)
    timestamp = float(segments[-1].get("end", candidate.end)) if present and segments else (candidate.end if present else None)
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
