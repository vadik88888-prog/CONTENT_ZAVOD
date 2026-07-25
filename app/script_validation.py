from __future__ import annotations

import re
from collections import Counter

from app.config import TransformationConfig
from app.script_generation import word_count
from app.transformation_models import (
    FactSourceScope,
    FactualityType,
    ScriptDraft,
    ScriptQualityScore,
    SemanticRepresentation,
    SourceContext,
    ValidationResult,
)


NUMBER_RE = re.compile(r"(?<![\w/])[+-]?\d[\d\s,.]*%?(?!\w)", re.UNICODE)
CURRENCY_RE = re.compile(r"(?:[$€£₽]\s*\d[\d\s,.]*|\d[\d\s,.]*\s*(?:руб(?:лей|ля)?|rur|usd|eur|dollars?))", re.I)
DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{4})\b")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
ENTITY_RE = re.compile(r"\b(?:[A-ZА-ЯЁ][a-zа-яё]{2,}|[A-Z]{2,})\b", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
ABSOLUTE_WORDS = {
    "всегда", "никогда", "гарантированно", "лучший", "лучшая", "лучшее",
    "единственный", "доказано", "полностью", "точно", "always", "never",
    "guaranteed", "best", "only", "proven", "completely", "exactly",
}
COMPARISON_WORDS = {"лучше", "лучший", "самый", "больше", "меньше", "best", "better", "more", "less", "only"}
NEGATION_WORDS = {"не", "нет", "никогда", "без", "not", "no", "never", "without", "cannot", "can't"}
MODAL_WORDS = {"может", "могут", "мог", "возможно", "вероятно", "may", "might", "could", "can", "perhaps"}
OPINION_WORDS = {"считаю", "думаю", "кажется", "по-моему", "мнение", "think", "believe", "opinion"}
QUANTITY_WORDS = {"несколько", "много", "мало", "некоторые", "сотни", "тысячи", "some", "several", "many", "few", "hundreds", "thousands"}
FILLERS = {"ну", "ээ", "эм", "короче", "типа", "uh", "um", "like"}
CLICKBAIT = {"шок", "секрет", "никто", "взорвёт", "shocking", "secret", "nobody"}


def validate_script_grounding(
    draft: ScriptDraft, semantic: SemanticRepresentation, context: SourceContext,
    allow_cta: bool,
) -> ValidationResult:
    """Check provenance and high-risk surface forms without relying on an LLM."""

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}
    try:
        semantic.validate(context)
        draft.validate_shape(semantic)
    except Exception as error:
        return ValidationResult(False, 0.0, [str(error)], [], {"structure": False})
    facts = semantic.fact_map()
    primary_text = context.primary_text()
    source_surface = _surface_set(primary_text)
    sentence_ids: set[str] = set()
    used_fact_ids: set[str] = set()
    for sentence in draft.sentences:
        if not sentence.sentence_id or sentence.sentence_id in sentence_ids:
            errors.append("Каждое ScriptSentence должно иметь уникальный sentence_id.")
        sentence_ids.add(sentence.sentence_id)
        if not sentence.supported_by_fact_ids or not sentence.source_segment_ids:
            errors.append(f"Предложение {sentence.sentence_id} не имеет evidence references.")
            continue
        unknown = set(sentence.supported_by_fact_ids) - set(facts)
        if unknown:
            errors.append(f"Предложение {sentence.sentence_id} ссылается на неизвестный fact_id.")
            continue
        used_fact_ids.update(sentence.supported_by_fact_ids)
        # A fact can quote a full Whisper segment with several sentences. Modality
        # and negation must be compared with the claimed fact, not neighbouring text.
        sentence_fact_evidence = " ".join(facts[item].statement for item in sentence.supported_by_fact_ids)
        valid_segment_ids = {identifier for item in sentence.supported_by_fact_ids for identifier in facts[item].evidence_segment_ids}
        if not set(sentence.source_segment_ids).issubset(valid_segment_ids):
            errors.append(f"Предложение {sentence.sentence_id} содержит segment id вне evidence выбранного факта.")
        if any(facts[item].source_scope != FactSourceScope.PRIMARY_CANDIDATE for item in sentence.supported_by_fact_ids):
            errors.append(f"Предложение {sentence.sentence_id} использует supporting_context как основной факт.")
        _validate_sentence_surface(sentence.text, sentence_fact_evidence, errors, sentence.sentence_id)
        if any(facts[item].factuality_type == FactualityType.OPINION for item in sentence.supported_by_fact_ids):
            if not _tokens(sentence.text).intersection(OPINION_WORDS):
                errors.append(f"Предложение {sentence.sentence_id} превращает opinion в факт.")
    if not set(draft.used_fact_ids).issubset(set(facts)):
        errors.append("used_fact_ids содержит неизвестный факт.")
    if set(draft.used_fact_ids) != used_fact_ids:
        errors.append("used_fact_ids не совпадает с фактически использованными предложениями.")
    if any(sentence.role.value == "cta" for sentence in draft.sentences) and not allow_cta:
        errors.append("CTA запрещён конфигурацией.")
    script_surface = _surface_set(f"{draft.title} {draft.full_text}")
    for category in ("numbers", "currency", "dates", "entities", "urls"):
        unsupported = script_surface[category] - source_surface[category]
        if unsupported:
            errors.append(f"Неподтверждённые {category}: {', '.join(sorted(unsupported))}.")
    script_words = _tokens(draft.full_text)
    source_words = _tokens(primary_text)
    unsupported_absolutes = (script_words & ABSOLUTE_WORDS) - source_words
    if unsupported_absolutes:
        errors.append(f"Абсолютные утверждения отсутствуют в source evidence: {', '.join(sorted(unsupported_absolutes))}.")
    unsupported_comparisons = (script_words & COMPARISON_WORDS) - source_words
    if unsupported_comparisons:
        errors.append(f"Новые сравнительные утверждения: {', '.join(sorted(unsupported_comparisons))}.")
    unsupported_quantities = (script_words & QUANTITY_WORDS) - source_words
    if unsupported_quantities:
        errors.append(f"Новые количественные утверждения: {', '.join(sorted(unsupported_quantities))}.")
    checks["fact_references"] = not any("fact" in item or "evidence" in item for item in errors)
    checks["surface_terms"] = not any(item.startswith("Неподтверждённые") for item in errors)
    checks["negation_modal_opinion"] = not any(
        "отрицание" in item or "модальность" in item or "opinion" in item for item in errors
    )
    score = 1.0 if not errors else max(0.0, 1.0 - len(errors) / max(4, len(draft.sentences) * 3))
    return ValidationResult(not errors, score, errors, warnings, checks)


def score_script_quality(
    draft: ScriptDraft, semantic: SemanticRepresentation, grounding: ValidationResult,
    config: TransformationConfig,
) -> ScriptQualityScore:
    words = _token_list(draft.full_text)
    word_set = set(words)
    unique_ratio = len(word_set) / max(1, len(words))
    sentence_count = max(1, len(draft.sentences))
    average_sentence_words = len(words) / sentence_count
    source_coverage = min(1.0, draft.source_coverage)
    target_words = max(1, round(config.target_duration_seconds * config.target_words_per_second))
    ratio_to_target = len(words) / target_words
    filler_ratio = sum(token in FILLERS for token in words) / max(1, len(words))
    repetition = max(0.0, 1.0 - unique_ratio)
    hook_words = _token_list(draft.hook)
    hook_strength = min(1.0, 0.45 + (0.2 if "?" in draft.hook or "!" in draft.hook else 0) + (0.2 if any(char.isdigit() for char in draft.hook) else 0) + min(0.15, len(hook_words) / 50))
    clarity = 1.0 if 4 <= average_sentence_words <= 28 else max(0.2, 1.0 - abs(average_sentence_words - 16) / 32)
    brevity = max(0.0, 1.0 - abs(ratio_to_target - 1.0))
    information_density = min(1.0, unique_ratio + (0.1 if len(words) >= 8 else 0.0))
    context_independence = 0.55 if draft.hook.lower().startswith(("и ", "а ", "но ", "это ", "and ", "but ", "so ")) else 0.9
    naturalness = max(0.0, 1.0 - filler_ratio * 4 - repetition * 0.45)
    duration = draft.estimated_duration_seconds
    pacing = 1.0 if config.min_duration_seconds <= duration <= config.max_duration_seconds else max(0.2, 1.0 - min(abs(duration - config.target_duration_seconds) / max(1.0, config.target_duration_seconds), 0.8))
    ending_strength = 0.8 if draft.ending.strip().endswith((".", "!", "?", "…")) else 0.45
    subscores = {
        "hook_strength": hook_strength,
        "clarity": clarity,
        "completeness": source_coverage,
        "brevity": brevity,
        "information_density": information_density,
        "context_independence": context_independence,
        "naturalness": naturalness,
        "pacing": pacing,
        "ending_strength": ending_strength,
        "factual_grounding": grounding.score,
        "source_coverage": source_coverage,
    }
    penalties = {
        "repetition_penalty": repetition * 0.15,
        "filler_penalty": filler_ratio * 0.2,
        "clickbait_penalty": (len(word_set & CLICKBAIT) / max(1, len(words))) * 2,
        "unsupported_claim_penalty": (1.0 - grounding.score) * 0.35,
    }
    final_score = max(0.0, min(1.0, sum(config.weights[key] * subscores[key] for key in config.weights) - sum(penalties.values())))
    explanations: list[str] = []
    if hook_strength >= 0.7:
        explanations.append("Сильное начало на подтверждённом исходном предложении.")
    if information_density >= 0.75:
        explanations.append("Высокая информационная плотность.")
    if context_independence < 0.7:
        explanations.append("Начало зависит от предыдущего контекста.")
    if ending_strength < 0.7:
        explanations.append("Слабое завершение.")
    if repetition > 0.3:
        explanations.append("Слишком много повторов.")
    if not grounding.passed:
        explanations.append("Есть неподтверждённый или искажённый материал.")
    return ScriptQualityScore(final_score, subscores, penalties, explanations)


def validate_script_quality(
    draft: ScriptDraft, semantic: SemanticRepresentation, grounding: ValidationResult,
    config: TransformationConfig,
) -> tuple[ValidationResult, ScriptQualityScore]:
    quality = score_script_quality(draft, semantic, grounding, config)
    errors: list[str] = []
    if not grounding.passed:
        errors.append("Quality validation не может принять сценарий с failed grounding.")
    if quality.final_score < config.minimum_quality_score:
        errors.append(f"Script quality {quality.final_score:.3f} ниже порога {config.minimum_quality_score:.3f}.")
    return ValidationResult(not errors, quality.final_score, errors, quality.explanations, {"subscores": quality.subscores, "penalties": quality.penalties}), quality


def _validate_sentence_surface(text: str, evidence: str, errors: list[str], sentence_id: str) -> None:
    script_terms = _tokens(text)
    source_terms = _tokens(evidence)
    if source_terms & NEGATION_WORDS and not script_terms & NEGATION_WORDS:
        errors.append(f"Предложение {sentence_id} потеряло отрицание из source evidence.")
    if source_terms & MODAL_WORDS and not script_terms & MODAL_WORDS:
        errors.append(f"Предложение {sentence_id} усилило модальность source evidence.")
    if script_terms & QUANTITY_WORDS - source_terms:
        errors.append(f"Предложение {sentence_id} меняет количество из source evidence.")


def _surface_set(text: str) -> dict[str, set[str]]:
    return {
        "numbers": {_normalise_number(item) for item in NUMBER_RE.findall(text)},
        "currency": {_normalise_number(item) for item in CURRENCY_RE.findall(text)},
        "dates": {_normalise_number(item) for item in DATE_RE.findall(text)},
        "entities": {item.casefold() for item in ENTITY_RE.findall(text)},
        "urls": {item.rstrip(".,;:!?").casefold() for item in URL_RE.findall(text)},
    }


def _normalise_number(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", ".").casefold()


def _tokens(text: str) -> set[str]:
    return set(_token_list(text))


def _token_list(text: str) -> list[str]:
    return [item.casefold() for item in WORD_RE.findall(text)]
