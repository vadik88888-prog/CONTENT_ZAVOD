from __future__ import annotations

import re

from app.config import TransformationConfig
from app.errors import TransformationFallbackError
from app.script_generation import estimate_duration_seconds, word_count
from app.transformation_models import (
    FallbackReason,
    ScriptDraft,
    ScriptSentence,
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
    """Build a conservative artifact from original primary sentences only."""

    target_words = max(1, round(config.target_duration_seconds * config.target_words_per_second))
    selected: list[tuple[object, str]] = []
    selected_tokens: list[set[str]] = []
    total_words = 0
    for fact in semantic.supporting_facts:
        text = LEADING_FILLER_RE.sub("", fact.statement.strip())
        if not text:
            continue
        tokens = _tokens(text)
        if not tokens or any(_near_duplicate(tokens, prior) for prior in selected_tokens):
            continue
        count = word_count(text)
        if selected and total_words + count > target_words:
            break
        selected.append((fact, text))
        selected_tokens.append(tokens)
        total_words += count
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
