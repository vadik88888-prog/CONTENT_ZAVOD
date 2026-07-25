from __future__ import annotations

from app.config import TransformationConfig
from app.transformation_models import (
    FactSourceScope,
    NarrativePlan,
    SemanticRepresentation,
    TransformationMode,
)


def build_narrative_plan(
    semantic: SemanticRepresentation, config: TransformationConfig
) -> NarrativePlan:
    """Make a plan from approved primary facts only, without adding propositions."""

    facts = [
        item for item in semantic.supporting_facts
        if item.source_scope == FactSourceScope.PRIMARY_CANDIDATE
    ]
    required = [item.fact_id for item in facts]
    if not facts:
        # validate() turns this into a clear typed error for the orchestrator/fallback.
        required = []
    target_words = max(1, round(config.target_duration_seconds * config.target_words_per_second))
    hook = facts[0].statement if facts else ""
    ending = facts[-1].statement if facts else ""
    plan = NarrativePlan(
        candidate_id=semantic.candidate_id,
        transformation_mode=TransformationMode(config.mode),
        target_duration_seconds=config.target_duration_seconds,
        target_word_count=target_words,
        hook=hook,
        setup=facts[1].statement if len(facts) > 1 else "",
        key_points=[item.fact_id for item in facts[1:-1]],
        payoff=ending,
        ending=ending,
        optional_cta=None,
        omitted_content=[],
        reordered_content=[],
        required_fact_ids=required,
        tone="clear and faithful" if semantic.language == "en" else "ясный и достоверный",
        pacing="compact",
        rationale="План использует только подтверждённые primary_candidate facts в исходном порядке.",
    )
    plan.validate(semantic, config.allow_cta)
    return plan
