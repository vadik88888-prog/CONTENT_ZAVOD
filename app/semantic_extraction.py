from __future__ import annotations

import re
from typing import Any

from app.audio_features import window_audio_features
from app.config import TransformationConfig
from app.models import Candidate
from app.transformation_models import (
    ContentType,
    EvidenceSegment,
    FactSourceScope,
    FactualityType,
    SemanticFact,
    SemanticRepresentation,
    SourceContext,
)


SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d[\d\s,.]*%?", re.UNICODE)
ENTITY_RE = re.compile(r"\b(?:[A-ZА-ЯЁ][a-zа-яё]{2,}|[A-Z]{2,})\b", re.UNICODE)
OPINION_RE = re.compile(r"\b(я считаю|по-моему|мне кажется|думаю|считаю|i think|in my opinion|i believe)\b", re.I)
UNCERTAIN_RE = re.compile(r"\b(может|могут|возможно|вероятно|may|might|could|perhaps)\b", re.I)


def split_sentences(text: str) -> list[str]:
    """A deliberately conservative RU/EN segmentation used by local fallbacks."""

    return [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]


def build_source_context(
    source: dict[str, Any], metadata: dict[str, Any], candidate: Candidate,
    transcript: dict[str, Any], transcript_features: dict[str, Any],
    audio_features: dict[str, Any], scenes: dict[str, Any], config: TransformationConfig,
) -> SourceContext:
    language = str(transcript.get("language") or transcript_features.get("language") or "unknown")
    feature_by_id = {
        int(item.get("id", index)): item
        for index, item in enumerate(transcript_features.get("segments", []))
        if isinstance(item, dict)
    }
    candidate_ids = set(candidate.transcript_segment_ids)
    primary: list[EvidenceSegment] = []
    supporting: list[EvidenceSegment] = []
    before_start = candidate.start - config.context_before_seconds
    after_end = candidate.end + config.context_after_seconds
    for index, raw in enumerate(transcript.get("segments", [])):
        if not isinstance(raw, dict):
            continue
        start = float(raw.get("start", 0))
        end = float(raw.get("end", start))
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        identifier = int(raw.get("id", index))
        belongs_to_candidate = identifier in candidate_ids or (end > candidate.start and start < candidate.end)
        if belongs_to_candidate:
            # Candidate boundaries are resolved before transformation.  Keep the
            # exact source range in every evidence object so ProductionPlan,
            # dialogue extraction and subtitle timing cannot silently expand a
            # safe semantic boundary back to the full Whisper segment.
            primary_start = max(start, candidate.start)
            primary_end = min(end, candidate.end)
            if primary_start < primary_end:
                primary.append(EvidenceSegment(identifier, primary_start, primary_end, text, FactSourceScope.PRIMARY_CANDIDATE))
        elif end > before_start and start < after_end:
            supporting.append(EvidenceSegment(identifier, start, end, text, FactSourceScope.SUPPORTING_CONTEXT))
    if not primary and candidate.text.strip():
        # The source candidate is still primary evidence when a legacy transcript
        # has no per-segment overlap information.
        primary.append(EvidenceSegment(0, candidate.start, candidate.end, candidate.text.strip(), FactSourceScope.PRIMARY_CANDIDATE))
    selected_features = [feature_by_id[item.segment_id] for item in primary if item.segment_id in feature_by_id]
    scene_boundaries = [
        dict(item) for item in scenes.get("boundaries", [])
        if candidate.start <= float(item.get("timestamp", -1)) <= candidate.end
    ]
    audio_summary = window_audio_features(candidate.start, candidate.end, audio_features)
    return SourceContext(
        candidate_id=candidate.id,
        source_id=str(source.get("id", "")),
        source_path=str(source.get("path", "")),
        start_time=candidate.start,
        end_time=candidate.end,
        duration=candidate.duration,
        language=language,
        transcript_text=candidate.text.strip() or " ".join(item.text for item in primary),
        primary_evidence=primary,
        supporting_context=supporting,
        local_quality_score=candidate.local_quality_score,
        ai_rerank_score=candidate.ai_score,
        candidate_explanations=list(candidate.explanations),
        sentence_boundaries=[
            {
                "segment_id": item.segment_id,
                "sentence_start": bool(feature_by_id.get(item.segment_id, {}).get("sentence_start", False)),
                "sentence_end": bool(feature_by_id.get(item.segment_id, {}).get("sentence_end", False)),
            }
            for item in primary
        ],
        pause_features={
            "before_seconds": float(selected_features[0].get("pause_before_seconds", 0)) if selected_features else 0.0,
            "after_seconds": float(selected_features[-1].get("pause_after_seconds", 0)) if selected_features else 0.0,
        },
        speech_density=_average(selected_features, "speech_density"),
        filler_information={"ratio": _average(selected_features, "filler_word_ratio")},
        repetition_information={"score": _average(selected_features, "repetition_score")},
        scene_boundaries=scene_boundaries,
        audio_energy_summary=audio_summary,
        candidate_features=dict(candidate.feature_vector),
        boundary_decision=(
            dict(candidate.boundary_diagnostics.get("boundary_decision", {}))
            if isinstance(candidate.boundary_diagnostics, dict)
            else {}
        ),
    )


def extract_semantic_representation(context: SourceContext) -> SemanticRepresentation:
    """Produce a safe local semantic representation; every fact quotes evidence."""

    facts: list[SemanticFact] = []
    opinions: list[str] = []
    numbers: list[str] = []
    entities: list[str] = []
    for evidence in context.primary_evidence:
        for sentence in split_sentences(evidence.text):
            fact_id = f"fact-{len(facts) + 1:03d}"
            factuality = (
                FactualityType.OPINION if OPINION_RE.search(sentence)
                else FactualityType.STRONGLY_IMPLIED if UNCERTAIN_RE.search(sentence)
                else FactualityType.EXPLICIT
            )
            if factuality == FactualityType.OPINION:
                opinions.append(sentence)
            numbers.extend(_unique(NUMBER_RE.findall(sentence)))
            entities.extend(_unique(ENTITY_RE.findall(sentence)))
            facts.append(SemanticFact(
                fact_id=fact_id,
                statement=sentence,
                evidence_segment_ids=[evidence.segment_id],
                evidence_quote=evidence.text,
                evidence_start=evidence.start,
                evidence_end=evidence.end,
                confidence=1.0,
                source_scope=FactSourceScope.PRIMARY_CANDIDATE,
                factuality_type=factuality,
            ))
    main_idea = facts[0].statement if facts else ""
    semantic = SemanticRepresentation(
        candidate_id=context.candidate_id,
        language=context.language,
        content_type=_content_type(context.primary_text()),
        main_idea=main_idea,
        core_claim=main_idea,
        supporting_facts=facts,
        numbers_and_metrics=_unique(numbers),
        named_entities=_unique(entities),
        opinions=opinions,
        assumptions=[],
        examples=[],
        causal_links=[item.statement for item in facts if re.search(r"\b(поэтому|потому что|из-за|because|therefore|so)\b", item.statement, re.I)],
        chronology=[item.statement for item in facts if re.search(r"\b(сначала|потом|затем|после|first|then|after)\b", item.statement, re.I)],
        emotional_tone="emphatic" if any("!" in item.statement or "?" in item.statement for item in facts) else "neutral",
        target_viewer_takeaway=main_idea,
        context_dependencies=["Фрагмент начинается с зависимого союза."] if context.transcript_text.lower().startswith(("и ", "а ", "но ", "and ", "but ")) else [],
        removable_details=[],
        risky_claims=[item.statement for item in facts if UNCERTAIN_RE.search(item.statement)],
        source_evidence_map={item.fact_id: item.evidence_segment_ids for item in facts},
        confidence=1.0 if facts else 0.0,
    )
    semantic.validate(context)
    return semantic


def _average(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else 0.0


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


def _content_type(text: str) -> ContentType:
    lowered = text.lower()
    if OPINION_RE.search(text):
        return ContentType.OPINION
    if any(token in lowered for token in ("сначала", "потом", "затем", "first", "then")):
        return ContentType.STORY
    if any(token in lowered for token in ("как ", "почему", "how ", "why ")):
        return ContentType.EDUCATIONAL
    if "?" in text:
        return ContentType.INTERVIEW_ANSWER
    return ContentType.UNKNOWN
