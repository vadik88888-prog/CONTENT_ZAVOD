from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.errors import (
    GroundingValidationError,
    NarrativePlanningError,
    ScriptGenerationError,
    SemanticExtractionError,
)


FINAL_SCRIPT_CONTRACT_VERSION = "3.0"


class ContentType(str, Enum):
    EDUCATIONAL = "educational"
    STORY = "story"
    OPINION = "opinion"
    INTERVIEW_ANSWER = "interview_answer"
    TUTORIAL = "tutorial"
    NEWS_COMMENTARY = "news_commentary"
    MOTIVATIONAL = "motivational"
    ENTERTAINMENT = "entertainment"
    LIST = "list"
    WARNING = "warning"
    CASE_STUDY = "case_study"
    UNKNOWN = "unknown"


class FactSourceScope(str, Enum):
    PRIMARY_CANDIDATE = "primary_candidate"
    SUPPORTING_CONTEXT = "supporting_context"


class FactualityType(str, Enum):
    EXPLICIT = "explicit"
    STRONGLY_IMPLIED = "strongly_implied"
    OPINION = "opinion"
    UNCERTAIN = "uncertain"


class TransformationMode(str, Enum):
    FAITHFUL_COMPRESSION = "faithful_compression"
    HOOK_FIRST = "hook_first"
    EDUCATIONAL = "educational"
    STORY = "story"
    LISTICLE = "listicle"
    PROVOCATIVE = "provocative"
    CALM_EXPERT = "calm_expert"
    DIRECT_RESPONSE = "direct_response"
    AUTO = "auto"


class SentenceRole(str, Enum):
    HOOK = "hook"
    SETUP = "setup"
    CONTEXT = "context"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    TRANSITION = "transition"
    PAYOFF = "payoff"
    ENDING = "ending"
    CTA = "cta"


class FallbackReason(str, Enum):
    AI_DISABLED = "ai_disabled"
    PROVIDER_FAILURE = "provider_failure"
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    GROUNDING_FAILED = "grounding_failed"
    QUALITY_FAILED = "quality_failed"
    REPAIR_FAILED = "repair_failed"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    EMPTY_RESULT = "empty_result"


@dataclass(slots=True)
class EvidenceSegment:
    segment_id: int
    start: float
    end: float
    text: str
    scope: FactSourceScope

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "scope": self.scope.value,
        }


@dataclass(slots=True)
class SourceContext:
    candidate_id: str
    source_id: str
    source_path: str
    start_time: float
    end_time: float
    duration: float
    language: str
    transcript_text: str
    primary_evidence: list[EvidenceSegment]
    supporting_context: list[EvidenceSegment]
    local_quality_score: float
    ai_rerank_score: float | None
    candidate_explanations: list[str]
    sentence_boundaries: list[dict[str, Any]] = field(default_factory=list)
    pause_features: dict[str, Any] = field(default_factory=dict)
    speech_density: float = 0.0
    filler_information: dict[str, Any] = field(default_factory=dict)
    repetition_information: dict[str, Any] = field(default_factory=dict)
    scene_boundaries: list[dict[str, Any]] = field(default_factory=list)
    audio_energy_summary: dict[str, Any] = field(default_factory=dict)
    candidate_features: dict[str, Any] = field(default_factory=dict)
    # Empty is the explicit compatibility state for transformation artifacts
    # written before Goal 5C. New candidates carry the immutable decision here.
    boundary_decision: dict[str, Any] = field(default_factory=dict)
    multimodal_context: dict[str, Any] = field(default_factory=dict)
    composition_intent: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"

    def primary_text(self) -> str:
        return " ".join(item.text.strip() for item in self.primary_evidence if item.text.strip()).strip()

    def evidence_by_id(self) -> dict[int, EvidenceSegment]:
        return {item.segment_id: item for item in [*self.primary_evidence, *self.supporting_context]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "source": {"id": self.source_id, "path": self.source_path},
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "language": self.language,
            "transcript_text": self.transcript_text,
            "primary_evidence": [item.to_dict() for item in self.primary_evidence],
            "supporting_context": [item.to_dict() for item in self.supporting_context],
            "local_quality_score": round(self.local_quality_score, 3),
            "ai_rerank_score": self.ai_rerank_score,
            "candidate_explanations": self.candidate_explanations,
            "sentence_boundaries": self.sentence_boundaries,
            "pause_features": self.pause_features,
            "speech_density": self.speech_density,
            "filler_information": self.filler_information,
            "repetition_information": self.repetition_information,
            "scene_boundaries": self.scene_boundaries,
            "audio_energy_summary": self.audio_energy_summary,
            "candidate_features": self.candidate_features,
            "boundary_decision": self.boundary_decision,
            "multimodal_context": self.multimodal_context,
            "composition_intent": self.composition_intent,
        }


@dataclass(slots=True)
class SemanticFact:
    fact_id: str
    statement: str
    evidence_segment_ids: list[int]
    evidence_quote: str
    evidence_start: float
    evidence_end: float
    confidence: float
    source_scope: FactSourceScope
    factuality_type: FactualityType

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "statement": self.statement,
            "evidence_segment_ids": self.evidence_segment_ids,
            "evidence_quote": self.evidence_quote,
            "evidence_start": round(self.evidence_start, 3),
            "evidence_end": round(self.evidence_end, 3),
            "confidence": round(self.confidence, 3),
            "source_scope": self.source_scope.value,
            "factuality_type": self.factuality_type.value,
        }


@dataclass(slots=True)
class SemanticRepresentation:
    candidate_id: str
    language: str
    content_type: ContentType
    main_idea: str
    core_claim: str
    supporting_facts: list[SemanticFact]
    numbers_and_metrics: list[str] = field(default_factory=list)
    named_entities: list[str] = field(default_factory=list)
    opinions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    causal_links: list[str] = field(default_factory=list)
    chronology: list[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    target_viewer_takeaway: str = ""
    context_dependencies: list[str] = field(default_factory=list)
    removable_details: list[str] = field(default_factory=list)
    risky_claims: list[str] = field(default_factory=list)
    source_evidence_map: dict[str, list[int]] = field(default_factory=dict)
    confidence: float = 0.0
    schema_version: str = "2.0"

    def validate(self, context: SourceContext) -> None:
        if self.candidate_id != context.candidate_id:
            raise SemanticExtractionError("SemanticRepresentation содержит другой candidate_id.")
        if not self.language:
            raise SemanticExtractionError("SemanticRepresentation не содержит язык.")
        evidence = context.evidence_by_id()
        seen: set[str] = set()
        for fact in self.supporting_facts:
            if not fact.fact_id or fact.fact_id in seen:
                raise SemanticExtractionError("fact_id должен быть непустым и уникальным.")
            seen.add(fact.fact_id)
            if not fact.statement.strip() or not fact.evidence_quote.strip() or not fact.evidence_segment_ids:
                raise SemanticExtractionError(f"Факт {fact.fact_id} не связан с evidence.")
            if not 0 <= fact.confidence <= 1:
                raise SemanticExtractionError(f"Некорректная confidence факта {fact.fact_id}.")
            if any(identifier not in evidence for identifier in fact.evidence_segment_ids):
                raise SemanticExtractionError(f"Факт {fact.fact_id} ссылается на неизвестный transcript segment.")
            scopes = {evidence[identifier].scope for identifier in fact.evidence_segment_ids}
            if fact.source_scope == FactSourceScope.PRIMARY_CANDIDATE and scopes != {FactSourceScope.PRIMARY_CANDIDATE}:
                raise SemanticExtractionError(f"Факт {fact.fact_id} неверно помечен как primary evidence.")
            if fact.fact_id not in self.source_evidence_map:
                raise SemanticExtractionError(f"Для факта {fact.fact_id} нет source_evidence_map.")
            if set(self.source_evidence_map[fact.fact_id]) != set(fact.evidence_segment_ids):
                raise SemanticExtractionError(f"source_evidence_map не совпадает для факта {fact.fact_id}.")

    def fact_map(self) -> dict[str, SemanticFact]:
        return {fact.fact_id: fact for fact in self.supporting_facts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "language": self.language,
            "content_type": self.content_type.value,
            "main_idea": self.main_idea,
            "core_claim": self.core_claim,
            "supporting_facts": [fact.to_dict() for fact in self.supporting_facts],
            "numbers_and_metrics": self.numbers_and_metrics,
            "named_entities": self.named_entities,
            "opinions": self.opinions,
            "assumptions": self.assumptions,
            "examples": self.examples,
            "causal_links": self.causal_links,
            "chronology": self.chronology,
            "emotional_tone": self.emotional_tone,
            "target_viewer_takeaway": self.target_viewer_takeaway,
            "context_dependencies": self.context_dependencies,
            "removable_details": self.removable_details,
            "risky_claims": self.risky_claims,
            "source_evidence_map": self.source_evidence_map,
            "confidence": round(self.confidence, 3),
        }


@dataclass(slots=True)
class NarrativePlan:
    candidate_id: str
    transformation_mode: TransformationMode
    target_duration_seconds: float
    target_word_count: int
    hook: str
    setup: str
    key_points: list[str]
    payoff: str
    ending: str
    optional_cta: str | None
    omitted_content: list[str]
    reordered_content: list[str]
    required_fact_ids: list[str]
    tone: str
    pacing: str
    rationale: str
    schema_version: str = "2.0"

    def validate(self, semantic: SemanticRepresentation, allow_cta: bool) -> None:
        if self.candidate_id != semantic.candidate_id:
            raise NarrativePlanningError("NarrativePlan содержит другой candidate_id.")
        facts = semantic.fact_map()
        if not self.required_fact_ids:
            raise NarrativePlanningError("NarrativePlan не содержит required_fact_ids.")
        unknown = set(self.required_fact_ids) - set(facts)
        if unknown:
            raise NarrativePlanningError(f"NarrativePlan ссылается на неизвестные факты: {', '.join(sorted(unknown))}.")
        if any(facts[identifier].source_scope != FactSourceScope.PRIMARY_CANDIDATE for identifier in self.required_fact_ids):
            raise NarrativePlanningError("NarrativePlan не может использовать supporting_context как основной материал.")
        if self.optional_cta and not allow_cta:
            raise NarrativePlanningError("CTA отключён конфигурацией.")
        if self.target_duration_seconds <= 0 or self.target_word_count <= 0:
            raise NarrativePlanningError("У NarrativePlan должны быть положительные target duration и word count.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "transformation_mode": self.transformation_mode.value,
            "target_duration_seconds": round(self.target_duration_seconds, 3),
            "target_word_count": self.target_word_count,
            "hook": self.hook,
            "setup": self.setup,
            "key_points": self.key_points,
            "payoff": self.payoff,
            "ending": self.ending,
            "optional_cta": self.optional_cta,
            "omitted_content": self.omitted_content,
            "reordered_content": self.reordered_content,
            "required_fact_ids": self.required_fact_ids,
            "tone": self.tone,
            "pacing": self.pacing,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ScriptSentence:
    sentence_id: str
    text: str
    role: SentenceRole
    supported_by_fact_ids: list[str]
    source_segment_ids: list[int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "role": self.role.value,
            "supported_by_fact_ids": self.supported_by_fact_ids,
            "source_segment_ids": self.source_segment_ids,
            "confidence": round(self.confidence, 3),
        }


@dataclass(slots=True)
class ScriptDraft:
    candidate_id: str
    language: str
    title: str
    hook: str
    body: str
    ending: str
    full_text: str
    sentences: list[ScriptSentence]
    estimated_duration_seconds: float
    word_count: int
    used_fact_ids: list[str]
    transformation_notes: list[str]
    source_coverage: float
    novelty_risk: float
    status: str = "draft"
    schema_version: str = "2.0"

    def validate_shape(self, semantic: SemanticRepresentation) -> None:
        if self.candidate_id != semantic.candidate_id:
            raise ScriptGenerationError("ScriptDraft содержит другой candidate_id.")
        if not self.sentences or not self.full_text.strip() or not self.used_fact_ids:
            raise ScriptGenerationError("ScriptDraft не содержит предложений, текста или used_fact_ids.")
        facts = semantic.fact_map()
        if set(self.used_fact_ids) - set(facts):
            raise ScriptGenerationError("ScriptDraft ссылается на неизвестный fact_id.")
        if any(not item.text.strip() or not item.supported_by_fact_ids or not item.source_segment_ids for item in self.sentences):
            raise ScriptGenerationError("Каждое предложение ScriptDraft должно иметь текст, fact ids и segment ids.")
        if any(set(item.supported_by_fact_ids) - set(facts) for item in self.sentences):
            raise ScriptGenerationError("Предложение ScriptDraft ссылается на неизвестный fact_id.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "language": self.language,
            "title": self.title,
            "hook": self.hook,
            "body": self.body,
            "ending": self.ending,
            "full_text": self.full_text,
            "sentences": [item.to_dict() for item in self.sentences],
            "estimated_duration_seconds": round(self.estimated_duration_seconds, 3),
            "word_count": self.word_count,
            "used_fact_ids": self.used_fact_ids,
            "transformation_notes": self.transformation_notes,
            "source_coverage": round(self.source_coverage, 3),
            "novelty_risk": round(self.novelty_risk, 3),
            "status": self.status,
        }


@dataclass(slots=True)
class FinalScript:
    candidate_id: str
    language: str
    title: str
    hook: str
    body: str
    ending: str
    full_text: str
    sentences: list[ScriptSentence]
    estimated_duration_seconds: float
    word_count: int
    used_fact_ids: list[str]
    transformation_notes: list[str]
    source_coverage: float
    novelty_risk: float
    status: str
    production_ready_for_tts: bool
    fallback_reason: FallbackReason | None = None
    schema_version: str = "2.0"

    @classmethod
    def from_draft(
        cls, draft: ScriptDraft, status: str, production_ready_for_tts: bool,
        fallback_reason: FallbackReason | None = None,
    ) -> "FinalScript":
        return cls(
            candidate_id=draft.candidate_id, language=draft.language, title=draft.title,
            hook=draft.hook, body=draft.body, ending=draft.ending, full_text=draft.full_text,
            sentences=draft.sentences, estimated_duration_seconds=draft.estimated_duration_seconds,
            word_count=draft.word_count, used_fact_ids=draft.used_fact_ids,
            transformation_notes=draft.transformation_notes,
            source_coverage=draft.source_coverage, novelty_risk=draft.novelty_risk,
            status=status, production_ready_for_tts=production_ready_for_tts,
            fallback_reason=fallback_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "language": self.language,
            "title": self.title,
            "hook": self.hook,
            "body": self.body,
            "ending": self.ending,
            "full_text": self.full_text,
            "sentences": [item.to_dict() for item in self.sentences],
            "estimated_duration_seconds": round(self.estimated_duration_seconds, 3),
            "word_count": self.word_count,
            "used_fact_ids": self.used_fact_ids,
            "transformation_notes": self.transformation_notes,
            "source_coverage": round(self.source_coverage, 3),
            "novelty_risk": round(self.novelty_risk, 3),
            "status": self.status,
            "production_ready_for_tts": self.production_ready_for_tts,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
        }


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    score: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 3),
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


@dataclass(slots=True)
class ScriptQualityScore:
    final_score: float
    subscores: dict[str, float]
    penalties: dict[str, float]
    explanations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 3),
            "subscores": {key: round(value, 3) for key, value in self.subscores.items()},
            "penalties": {key: round(value, 3) for key, value in self.penalties.items()},
            "explanations": self.explanations,
        }


def fact_from_dict(data: dict[str, Any]) -> SemanticFact:
    return SemanticFact(
        fact_id=str(data.get("fact_id", "")), statement=str(data.get("statement", "")),
        evidence_segment_ids=[int(item) for item in data.get("evidence_segment_ids", [])],
        evidence_quote=str(data.get("evidence_quote", "")),
        evidence_start=float(data.get("evidence_start", 0)), evidence_end=float(data.get("evidence_end", 0)),
        confidence=float(data.get("confidence", 0)),
        source_scope=FactSourceScope(str(data.get("source_scope", FactSourceScope.PRIMARY_CANDIDATE.value))),
        factuality_type=FactualityType(str(data.get("factuality_type", FactualityType.UNCERTAIN.value))),
    )


def semantic_from_dict(data: dict[str, Any]) -> SemanticRepresentation:
    raw_source_map = data.get("source_evidence_map", {})
    if isinstance(raw_source_map, list):
        source_evidence_map = {
            str(item.get("semantic_element", "")): [int(value) for value in item.get("segment_ids", [])]
            for item in raw_source_map if isinstance(item, dict)
        }
    else:
        source_evidence_map = {
            str(key): [int(item) for item in value]
            for key, value in dict(raw_source_map).items()
        }
    return SemanticRepresentation(
        candidate_id=str(data.get("candidate_id", "")), language=str(data.get("language", "")),
        content_type=ContentType(str(data.get("content_type", ContentType.UNKNOWN.value))),
        main_idea=str(data.get("main_idea", "")), core_claim=str(data.get("core_claim", "")),
        supporting_facts=[fact_from_dict(item) for item in data.get("supporting_facts", []) if isinstance(item, dict)],
        numbers_and_metrics=[str(item) for item in data.get("numbers_and_metrics", [])],
        named_entities=[str(item) for item in data.get("named_entities", [])],
        opinions=[str(item) for item in data.get("opinions", [])], assumptions=[str(item) for item in data.get("assumptions", [])],
        examples=[str(item) for item in data.get("examples", [])], causal_links=[str(item) for item in data.get("causal_links", [])],
        chronology=[str(item) for item in data.get("chronology", [])], emotional_tone=str(data.get("emotional_tone", "neutral")),
        target_viewer_takeaway=str(data.get("target_viewer_takeaway", "")),
        context_dependencies=[str(item) for item in data.get("context_dependencies", [])],
        removable_details=[str(item) for item in data.get("removable_details", [])],
        risky_claims=[str(item) for item in data.get("risky_claims", [])],
        source_evidence_map=source_evidence_map,
        confidence=float(data.get("confidence", 0)), schema_version=str(data.get("schema_version", "2.0")),
    )


def plan_from_dict(data: dict[str, Any]) -> NarrativePlan:
    return NarrativePlan(
        candidate_id=str(data.get("candidate_id", "")),
        transformation_mode=TransformationMode(str(data.get("transformation_mode", TransformationMode.AUTO.value))),
        target_duration_seconds=float(data.get("target_duration_seconds", 0)), target_word_count=int(data.get("target_word_count", 0)),
        hook=str(data.get("hook", "")), setup=str(data.get("setup", "")), key_points=[str(item) for item in data.get("key_points", [])],
        payoff=str(data.get("payoff", "")), ending=str(data.get("ending", "")),
        optional_cta=str(data["optional_cta"]) if data.get("optional_cta") is not None else None,
        omitted_content=[str(item) for item in data.get("omitted_content", [])], reordered_content=[str(item) for item in data.get("reordered_content", [])],
        required_fact_ids=[str(item) for item in data.get("required_fact_ids", [])], tone=str(data.get("tone", "")),
        pacing=str(data.get("pacing", "")), rationale=str(data.get("rationale", "")), schema_version=str(data.get("schema_version", "2.0")),
    )


def sentence_from_dict(data: dict[str, Any]) -> ScriptSentence:
    return ScriptSentence(
        sentence_id=str(data.get("sentence_id", "")), text=str(data.get("text", "")),
        role=SentenceRole(str(data.get("role", SentenceRole.CLAIM.value))),
        supported_by_fact_ids=[str(item) for item in data.get("supported_by_fact_ids", [])],
        source_segment_ids=[int(item) for item in data.get("source_segment_ids", [])], confidence=float(data.get("confidence", 0)),
    )


def draft_from_dict(data: dict[str, Any]) -> ScriptDraft:
    return ScriptDraft(
        candidate_id=str(data.get("candidate_id", "")), language=str(data.get("language", "")), title=str(data.get("title", "")),
        hook=str(data.get("hook", "")), body=str(data.get("body", "")), ending=str(data.get("ending", "")), full_text=str(data.get("full_text", "")),
        sentences=[sentence_from_dict(item) for item in data.get("sentences", []) if isinstance(item, dict)],
        estimated_duration_seconds=float(data.get("estimated_duration_seconds", 0)), word_count=int(data.get("word_count", 0)),
        used_fact_ids=[str(item) for item in data.get("used_fact_ids", [])], transformation_notes=[str(item) for item in data.get("transformation_notes", [])],
        source_coverage=float(data.get("source_coverage", 0)), novelty_risk=float(data.get("novelty_risk", 0)),
        status=str(data.get("status", "draft")), schema_version=str(data.get("schema_version", "2.0")),
    )


def source_context_from_dict(data: dict[str, Any]) -> SourceContext:
    """Rehydrate a persisted context only for contract validation."""

    source = data.get("source", {}) if isinstance(data.get("source"), dict) else {}
    return SourceContext(
        candidate_id=str(data.get("candidate_id", "")),
        source_id=str(source.get("id", "")),
        source_path=str(source.get("path", "")),
        start_time=float(data.get("start_time", 0)),
        end_time=float(data.get("end_time", 0)),
        duration=float(data.get("duration", 0)),
        language=str(data.get("language", "")),
        transcript_text=str(data.get("transcript_text", "")),
        primary_evidence=[_evidence_from_dict(item) for item in data.get("primary_evidence", []) if isinstance(item, dict)],
        supporting_context=[_evidence_from_dict(item) for item in data.get("supporting_context", []) if isinstance(item, dict)],
        local_quality_score=float(data.get("local_quality_score", 0)),
        ai_rerank_score=data.get("ai_rerank_score"),
        candidate_explanations=[str(item) for item in data.get("candidate_explanations", [])],
        sentence_boundaries=[dict(item) for item in data.get("sentence_boundaries", []) if isinstance(item, dict)],
        pause_features=dict(data.get("pause_features", {})),
        speech_density=float(data.get("speech_density", 0)),
        filler_information=dict(data.get("filler_information", {})),
        repetition_information=dict(data.get("repetition_information", {})),
        scene_boundaries=[dict(item) for item in data.get("scene_boundaries", []) if isinstance(item, dict)],
        audio_energy_summary=dict(data.get("audio_energy_summary", {})),
        candidate_features=dict(data.get("candidate_features", {})),
        boundary_decision=dict(data.get("boundary_decision", {})),
        schema_version=str(data.get("schema_version", "2.0")),
    )


def final_from_dict(data: dict[str, Any]) -> FinalScript:
    fallback_reason = data.get("fallback_reason")
    try:
        parsed_reason = FallbackReason(str(fallback_reason)) if fallback_reason else None
    except ValueError:
        parsed_reason = None
    return FinalScript(
        candidate_id=str(data.get("candidate_id", "")),
        language=str(data.get("language", "")),
        title=str(data.get("title", "")),
        hook=str(data.get("hook", "")),
        body=str(data.get("body", "")),
        ending=str(data.get("ending", "")),
        full_text=str(data.get("full_text", "")),
        sentences=[sentence_from_dict(item) for item in data.get("sentences", []) if isinstance(item, dict)],
        estimated_duration_seconds=float(data.get("estimated_duration_seconds", 0)),
        word_count=int(data.get("word_count", 0)),
        used_fact_ids=[str(item) for item in data.get("used_fact_ids", [])],
        transformation_notes=[str(item) for item in data.get("transformation_notes", [])],
        source_coverage=float(data.get("source_coverage", 0)),
        novelty_risk=float(data.get("novelty_risk", 0)),
        status=str(data.get("status", "")),
        production_ready_for_tts=data.get("production_ready_for_tts", False),
        fallback_reason=parsed_reason,
        schema_version=str(data.get("schema_version", "2.0")),
    )


def validate_final_script(
    final: FinalScript | dict[str, Any],
    context: SourceContext | dict[str, Any],
    semantic: SemanticRepresentation | dict[str, Any],
    expected_candidate_id: str | None = None,
) -> ValidationResult:
    """Validate the exact boundary accepted by ProductionPlan.

    This deliberately checks only identifiers, source references and timeline
    invariants.  It never includes transcript text in diagnostics.
    """

    errors: list[str] = []
    checks: dict[str, Any] = {"contract_version": FINAL_SCRIPT_CONTRACT_VERSION}
    try:
        final_value = final_from_dict(final) if isinstance(final, dict) else final
        context_value = source_context_from_dict(context) if isinstance(context, dict) else context
        semantic_value = semantic_from_dict(semantic) if isinstance(semantic, dict) else semantic
    except (TypeError, ValueError, KeyError) as error:
        return ValidationResult(False, 0.0, [f"FinalScript cannot be parsed: {error}"], [], checks)

    expected = str(expected_candidate_id or context_value.candidate_id or "")
    actual = final_value.candidate_id
    checks.update({"expected_candidate_id": expected, "actual_candidate_id": actual})
    if not expected:
        errors.append("Expected candidate_id is missing.")
    if not actual:
        errors.append("FinalScript candidate_id is missing.")
    elif actual != expected:
        errors.append("FinalScript candidate_id does not match the current candidate.")
    if final_value.status not in {"completed", "fallback"}:
        errors.append("FinalScript status is not production-eligible.")
    if not isinstance(final_value.production_ready_for_tts, bool):
        errors.append("FinalScript production_ready_for_tts must be boolean.")
    if not final_value.full_text.strip():
        errors.append("FinalScript full_text is empty.")
    if final_value.word_count <= 0:
        errors.append("FinalScript word_count must be positive.")
    if final_value.estimated_duration_seconds <= 0:
        errors.append("FinalScript estimated_duration_seconds must be positive.")
    if not isinstance(final_value.sentences, list) or not final_value.sentences:
        errors.append("FinalScript sentences must contain at least one item.")

    primary = {item.segment_id: item for item in context_value.primary_evidence}
    facts = semantic_value.fact_map()
    try:
        semantic_value.validate(context_value)
        checks["semantic_source_evidence"] = True
    except Exception:
        checks["semantic_source_evidence"] = False
        errors.append("FinalScript semantic source_evidence_map is invalid.")

    for fact in semantic_value.supporting_facts:
        if (
            fact.evidence_end < fact.evidence_start
            or fact.evidence_start < context_value.start_time - 0.01
            or fact.evidence_end > context_value.end_time + 0.01
        ):
            errors.append("FinalScript fact source timing is outside the current candidate.")
            break

    sentence_ids: set[str] = set()
    used_fact_ids: set[str] = set()
    for sentence in final_value.sentences:
        if not sentence.sentence_id or sentence.sentence_id in sentence_ids:
            errors.append("FinalScript sentence_id must be present and unique.")
        sentence_ids.add(sentence.sentence_id)
        if not sentence.text.strip():
            errors.append(f"FinalScript sentence {sentence.sentence_id or '<unknown>'} is empty.")
        if not sentence.supported_by_fact_ids:
            errors.append(f"FinalScript sentence {sentence.sentence_id or '<unknown>'} has no fact references.")
        if not sentence.source_segment_ids:
            errors.append(f"FinalScript sentence {sentence.sentence_id or '<unknown>'} has no source references.")
        unknown_facts = set(sentence.supported_by_fact_ids) - set(facts)
        if unknown_facts:
            errors.append(f"FinalScript sentence {sentence.sentence_id or '<unknown>'} has unknown fact references.")
        invalid_sources = set(sentence.source_segment_ids) - set(primary)
        if invalid_sources:
            errors.append(f"FinalScript sentence {sentence.sentence_id or '<unknown>'} has source references outside the candidate.")
        referenced_fact_sources = {
            source_id
            for fact_id in sentence.supported_by_fact_ids
            if fact_id in facts
            for source_id in facts[fact_id].evidence_segment_ids
        }
        if sentence.source_segment_ids and not set(sentence.source_segment_ids).issubset(referenced_fact_sources):
            errors.append(f"FinalScript sentence {sentence.sentence_id or '<unknown>'} source references do not match its facts.")
        used_fact_ids.update(sentence.supported_by_fact_ids)

    for evidence in primary.values():
        if evidence.end < evidence.start or evidence.start < context_value.start_time - 0.01 or evidence.end > context_value.end_time + 0.01:
            errors.append("FinalScript source timing is outside the current candidate.")
            break
    if set(final_value.used_fact_ids) != used_fact_ids:
        errors.append("FinalScript used_fact_ids do not match sentence fact references.")
    reconstructed = " ".join(item.text.strip() for item in final_value.sentences if item.text.strip())
    if reconstructed and final_value.full_text.strip() != reconstructed:
        errors.append("FinalScript full_text does not match its sentences.")
    checks.update({
        "sentences_count": len(final_value.sentences),
        "source_reference_count": sum(len(item.source_segment_ids) for item in final_value.sentences),
        "source_timing_valid": not any("timing" in item for item in errors),
    })
    return ValidationResult(not errors, 1.0 if not errors else 0.0, errors, [], checks)


def _evidence_from_dict(data: dict[str, Any]) -> EvidenceSegment:
    return EvidenceSegment(
        segment_id=int(data.get("segment_id", 0)),
        start=float(data.get("start", 0)),
        end=float(data.get("end", 0)),
        text=str(data.get("text", "")),
        scope=FactSourceScope(str(data.get("scope", FactSourceScope.PRIMARY_CANDIDATE.value))),
    )
