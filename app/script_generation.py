from __future__ import annotations

import re

from app.transformation_models import (
    NarrativePlan,
    ScriptDraft,
    ScriptSentence,
    SemanticRepresentation,
    SentenceRole,
)


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
LEADING_FILLER_RE = re.compile(r"^\s*(?:ну|ээ|эм|короче|как бы|типа|uh|um)\s*[,—-]?\s*", re.I)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def estimate_duration_seconds(text: str, words_per_second: float) -> float:
    return round(word_count(text) / max(0.1, words_per_second), 3)


def generate_script_draft(
    semantic: SemanticRepresentation, plan: NarrativePlan, words_per_second: float,
) -> ScriptDraft:
    """Deterministic, nearly verbatim generation used by local and mock modes."""

    facts = semantic.fact_map()
    selected = []
    used_words = 0
    for fact_id in plan.required_fact_ids:
        fact = facts[fact_id]
        text = _normalise_source_sentence(fact.statement)
        if not text:
            continue
        count = word_count(text)
        if selected and used_words + count > plan.target_word_count:
            break
        selected.append((fact, text))
        used_words += count
    if not selected and plan.required_fact_ids:
        fact = facts[plan.required_fact_ids[0]]
        selected = [(fact, _normalise_source_sentence(fact.statement))]
    sentences: list[ScriptSentence] = []
    for index, (fact, text) in enumerate(selected, start=1):
        if index == 1:
            role = SentenceRole.HOOK
        elif index == len(selected):
            role = SentenceRole.ENDING
        else:
            role = SentenceRole.CLAIM
        sentences.append(ScriptSentence(
            sentence_id=f"sentence-{index:03d}", text=text, role=role,
            supported_by_fact_ids=[fact.fact_id], source_segment_ids=fact.evidence_segment_ids,
            confidence=fact.confidence,
        ))
    full_text = " ".join(item.text for item in sentences).strip()
    hook = sentences[0].text if sentences else ""
    ending = sentences[-1].text if len(sentences) > 1 else ""
    body = " ".join(item.text for item in sentences[1:-1]).strip()
    title = " ".join(WORD_RE.findall(hook)[:8]).strip() or "Фрагмент"
    draft = ScriptDraft(
        candidate_id=semantic.candidate_id,
        language=semantic.language,
        title=title,
        hook=hook,
        body=body,
        ending=ending,
        full_text=full_text,
        sentences=sentences,
        estimated_duration_seconds=estimate_duration_seconds(full_text, words_per_second),
        word_count=word_count(full_text),
        used_fact_ids=[item.supported_by_fact_ids[0] for item in sentences],
        transformation_notes=["Локальная faithful compression: исходный порядок и подтверждённые предложения."],
        source_coverage=(len(sentences) / max(1, len(semantic.supporting_facts))),
        novelty_risk=0.0,
    )
    draft.validate_shape(semantic)
    return draft


def recompute_script_metrics(draft: ScriptDraft, words_per_second: float) -> ScriptDraft:
    """AI-provided duration/count are never trusted; reconstruct text and metrics here."""

    full_text = " ".join(item.text.strip() for item in draft.sentences if item.text.strip())
    draft.full_text = full_text
    draft.word_count = word_count(full_text)
    draft.estimated_duration_seconds = estimate_duration_seconds(full_text, words_per_second)
    draft.hook = draft.sentences[0].text if draft.sentences else ""
    draft.ending = draft.sentences[-1].text if len(draft.sentences) > 1 else ""
    draft.body = " ".join(item.text for item in draft.sentences[1:-1]).strip()
    return draft


def _normalise_source_sentence(text: str) -> str:
    return LEADING_FILLER_RE.sub("", text.strip())
