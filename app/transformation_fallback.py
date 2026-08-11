from __future__ import annotations

import re

from app.config import TransformationConfig
from app.errors import TransformationFallbackError
from app.script_generation import estimate_duration_seconds, word_count
from app.transformation_models import (
    FallbackReason,
    ScriptDraft,
    ScriptSentence,
    SemanticFact,
    SemanticRepresentation,
    SentenceRole,
    SourceContext,
)


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
LEADING_FILLER_RE = re.compile(r"^\s*(?:ну|ээ|эм|короче|как бы|типа|uh|um)\s*[,—-]?\s*", re.I)


def build_local_fallback(
    context: SourceContext, semantic: SemanticRepresentation, config: TransformationConfig,
    reason: FallbackReason,
) -> ScriptDraft:
    """Build a conservative artifact from original primary sentences only.

    The target duration is a soft preference here. A fallback must not turn an
    approved complete candidate into an unfinished prefix merely to hit a word
    budget; all non-duplicate source facts stay in source order. Facts carrying
    required boundary evidence are retained even when their wording is close to
    an earlier fact.
    """

    required_boundary_facts = _required_boundary_fact_ids(context, semantic)
    selected: list[tuple[SemanticFact, str]] = []
    selected_tokens: list[set[str]] = []
    for fact in semantic.supporting_facts:
        text = LEADING_FILLER_RE.sub("", fact.statement.strip())
        if not text:
            continue
        tokens = _tokens(text)
        if not tokens:
            continue
        if (
            fact.fact_id not in required_boundary_facts
            and any(_near_duplicate(tokens, prior) for prior in selected_tokens)
        ):
            continue
        selected.append((fact, text))
        selected_tokens.append(tokens)
    if not selected:
        raise TransformationFallbackError("Local fallback не нашёл безопасного предложения в selected transcript.")
    sentences: list[ScriptSentence] = []
    for index, (fact, text) in enumerate(selected, start=1):
        role = SentenceRole.HOOK if index == 1 else SentenceRole.ENDING if index == len(selected) else SentenceRole.CLAIM
        sentences.append(ScriptSentence(
            sentence_id=f"fallback-{index:03d}", text=text, role=role,
            supported_by_fact_ids=[fact.fact_id], source_segment_ids=list(fact.evidence_segment_ids),
            confidence=fact.confidence,
        ))
    full_text = " ".join(item.text for item in sentences)
    hook = sentences[0].text
    return ScriptDraft(
        candidate_id=context.candidate_id,
        language=context.language,
        title=" ".join(WORD_RE.findall(hook)[:8]) or "Фрагмент",
        hook=hook,
        body=" ".join(item.text for item in sentences[1:-1]),
        ending=sentences[-1].text if len(sentences) > 1 else "",
        full_text=full_text,
        sentences=sentences,
        estimated_duration_seconds=estimate_duration_seconds(full_text, config.target_words_per_second),
        word_count=word_count(full_text),
        used_fact_ids=[item.supported_by_fact_ids[0] for item in sentences],
        transformation_notes=[f"Conservative local fallback: {reason.value}.", "Сохранён исходный порядок; новые факты не добавлялись."],
        source_coverage=len(sentences) / max(1, len(semantic.supporting_facts)),
        novelty_risk=0.0,
        status="fallback",
    )


def _tokens(text: str) -> set[str]:
    return {item.casefold() for item in WORD_RE.findall(text)}


def _near_duplicate(current: set[str], prior: set[str]) -> bool:
    if not current or not prior:
        return False
    return len(current & prior) / len(current | prior) >= 0.75


def _required_boundary_fact_ids(
    context: SourceContext, semantic: SemanticRepresentation,
) -> set[str]:
    """Resolve required hook/completion/payoff ranges to their source facts."""

    raw_requirements = context.boundary_decision.get("required_evidence", [])
    if not isinstance(raw_requirements, list):
        return set()
    required_ranges: list[tuple[float, float, int | None]] = []
    for item in raw_requirements:
        if not isinstance(item, dict) or not item.get("required"):
            continue
        raw_range = item.get("source_range")
        if not isinstance(raw_range, dict):
            continue
        try:
            start = float(raw_range["start_seconds"])
            end = float(raw_range["end_seconds"])
            segment_id = (
                int(item["transcript_segment_id"])
                if item.get("transcript_segment_id") is not None else None
            )
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            required_ranges.append((start, end, segment_id))

    result: set[str] = set()
    for fact in semantic.supporting_facts:
        for start, end, segment_id in required_ranges:
            segment_match = segment_id is not None and segment_id in fact.evidence_segment_ids
            range_match = fact.evidence_start < end and start < fact.evidence_end
            if segment_match or range_match:
                result.add(fact.fact_id)
                break
    return result
