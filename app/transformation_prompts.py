from __future__ import annotations

import json
from typing import Any

from app.transformation_models import SourceContext


SEMANTIC_PROMPT_VERSION = "2.0.0"
NARRATIVE_PROMPT_VERSION = "2.0.0"
SCRIPT_PROMPT_VERSION = "2.0.0"
VALIDATION_PROMPT_VERSION = "2.0.0"
REPAIR_PROMPT_VERSION = "2.0.0"
TRANSFORMATION_SCHEMA_VERSION = "2.0"
GROUNDING_RULES_VERSION = "2.0"

PROMPT_VERSIONS = {
    "semantic": SEMANTIC_PROMPT_VERSION,
    "narrative": NARRATIVE_PROMPT_VERSION,
    "script": SCRIPT_PROMPT_VERSION,
    "validation": VALIDATION_PROMPT_VERSION,
    "repair": REPAIR_PROMPT_VERSION,
    "schema": TRANSFORMATION_SCHEMA_VERSION,
    "grounding_rules": GROUNDING_RULES_VERSION,
}


_STRING = {"type": "string", "maxLength": 1200}
_SHORT_STRING = {"type": "string", "maxLength": 300}
_STRING_ARRAY = {"type": "array", "items": _SHORT_STRING, "maxItems": 30}
_ID_ARRAY = {"type": "array", "items": {"type": "string", "maxLength": 80}, "minItems": 1, "maxItems": 30}
_SEGMENT_ARRAY = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "maxItems": 30}

_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fact_id": {"type": "string", "pattern": "^fact-[0-9]{3}$"},
        "statement": _STRING,
        "evidence_segment_ids": _SEGMENT_ARRAY,
        "evidence_quote": _STRING,
        "evidence_start": {"type": "number", "minimum": 0},
        "evidence_end": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "source_scope": {"type": "string", "enum": ["primary_candidate", "supporting_context"]},
        "factuality_type": {"type": "string", "enum": ["explicit", "strongly_implied", "opinion", "uncertain"]},
    },
    "required": [
        "fact_id", "statement", "evidence_segment_ids", "evidence_quote", "evidence_start",
        "evidence_end", "confidence", "source_scope", "factuality_type",
    ],
}

_SOURCE_MAP_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_element": {"type": "string", "maxLength": 80},
        "segment_ids": _SEGMENT_ARRAY,
    },
    "required": ["semantic_element", "segment_ids"],
}

_SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_id": {"type": "string", "maxLength": 100},
        "language": {"type": "string", "enum": ["ru", "en", "unknown"]},
        "content_type": {"type": "string", "enum": ["educational", "story", "opinion", "interview_answer", "tutorial", "news_commentary", "motivational", "entertainment", "list", "warning", "case_study", "unknown"]},
        "main_idea": _STRING,
        "core_claim": _STRING,
        "supporting_facts": {"type": "array", "items": _FACT_SCHEMA, "minItems": 1, "maxItems": 30},
        "numbers_and_metrics": _STRING_ARRAY,
        "named_entities": _STRING_ARRAY,
        "opinions": _STRING_ARRAY,
        "assumptions": _STRING_ARRAY,
        "examples": _STRING_ARRAY,
        "causal_links": _STRING_ARRAY,
        "chronology": _STRING_ARRAY,
        "emotional_tone": _SHORT_STRING,
        "target_viewer_takeaway": _STRING,
        "context_dependencies": _STRING_ARRAY,
        "removable_details": _STRING_ARRAY,
        "risky_claims": _STRING_ARRAY,
        "source_evidence_map": {"type": "array", "items": _SOURCE_MAP_ENTRY_SCHEMA, "minItems": 1, "maxItems": 30},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "candidate_id", "language", "content_type", "main_idea", "core_claim", "supporting_facts",
        "numbers_and_metrics", "named_entities", "opinions", "assumptions", "examples", "causal_links",
        "chronology", "emotional_tone", "target_viewer_takeaway", "context_dependencies", "removable_details",
        "risky_claims", "source_evidence_map", "confidence",
    ],
}

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_id": {"type": "string", "maxLength": 100},
        "transformation_mode": {"type": "string", "enum": ["faithful_compression", "hook_first", "educational", "story", "listicle", "provocative", "calm_expert", "direct_response", "auto"]},
        "target_duration_seconds": {"type": "number", "minimum": 1, "maximum": 180},
        "target_word_count": {"type": "integer", "minimum": 1, "maximum": 900},
        "hook": _STRING,
        "setup": _STRING,
        "key_points": _STRING_ARRAY,
        "payoff": _STRING,
        "ending": _STRING,
        "optional_cta": {"type": ["string", "null"], "maxLength": 300},
        "omitted_content": _STRING_ARRAY,
        "reordered_content": _STRING_ARRAY,
        "required_fact_ids": _ID_ARRAY,
        "tone": _SHORT_STRING,
        "pacing": _SHORT_STRING,
        "rationale": _STRING,
    },
    "required": [
        "candidate_id", "transformation_mode", "target_duration_seconds", "target_word_count", "hook", "setup",
        "key_points", "payoff", "ending", "optional_cta", "omitted_content", "reordered_content",
        "required_fact_ids", "tone", "pacing", "rationale",
    ],
}

_SENTENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentence_id": {"type": "string", "maxLength": 80},
        "text": _STRING,
        "role": {"type": "string", "enum": ["hook", "setup", "context", "claim", "evidence", "transition", "payoff", "ending", "cta"]},
        "supported_by_fact_ids": _ID_ARRAY,
        "source_segment_ids": _SEGMENT_ARRAY,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["sentence_id", "text", "role", "supported_by_fact_ids", "source_segment_ids", "confidence"],
}

_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_id": {"type": "string", "maxLength": 100},
        "language": {"type": "string", "enum": ["ru", "en", "unknown"]},
        "title": _SHORT_STRING,
        "hook": _STRING,
        "body": _STRING,
        "ending": _STRING,
        "full_text": {"type": "string", "maxLength": 4000},
        "sentences": {"type": "array", "items": _SENTENCE_SCHEMA, "minItems": 1, "maxItems": 30},
        "estimated_duration_seconds": {"type": "number", "minimum": 0, "maximum": 180},
        "word_count": {"type": "integer", "minimum": 0, "maximum": 900},
        "used_fact_ids": _ID_ARRAY,
        "transformation_notes": _STRING_ARRAY,
        "source_coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "novelty_risk": {"type": "number", "minimum": 0, "maximum": 1},
        "status": {"type": "string", "maxLength": 80},
    },
    "required": [
        "candidate_id", "language", "title", "hook", "body", "ending", "full_text", "sentences",
        "estimated_duration_seconds", "word_count", "used_fact_ids", "transformation_notes", "source_coverage",
        "novelty_risk", "status",
    ],
}

# Reused by bounded repair calls; it is intentionally a separate strict output,
# because semantic facts and the plan are immutable repair input rather than output.
OPENAI_SCRIPT_DRAFT_SCHEMA: dict[str, Any] = _DRAFT_SCHEMA

OPENAI_TRANSFORMATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_representation": _SEMANTIC_SCHEMA,
        "narrative_plan": _PLAN_SCHEMA,
        "script_draft": _DRAFT_SCHEMA,
    },
    "required": ["semantic_representation", "narrative_plan", "script_draft"],
}


def compact_instructions() -> str:
    return (
        "Transform a selected short-video transcript into a grounded script. Use only primary_evidence as factual material. "
        "supporting_context is for comprehension only and must not become a script fact. Do not use web knowledge or invent numbers, "
        "names, dates, brands, results, causal links, chronology, CTA, or certainty. Preserve negation, modality, subject, time, "
        "and quantity. Every fact and every sentence must reference supplied evidence ids. If evidence is insufficient, return fewer "
        "facts and a conservative script. Return only the strict structured output."
    )


def compact_payload(context: SourceContext, settings: dict[str, Any]) -> str:
    return json.dumps({
        "prompt_versions": PROMPT_VERSIONS,
        "source_context": context.to_dict(),
        "settings": settings,
        "rules": {
            "supporting_context_not_primary": True,
            "translation_disabled": not bool(settings.get("allow_translation")),
            "cta_allowed": bool(settings.get("allow_cta")),
            "recompute_duration_and_word_count_in_python": True,
        },
    }, ensure_ascii=False)


def repair_instructions() -> str:
    return (
        "Repair only the listed deterministic validation errors. Do not add facts, numbers, entities, causal claims, CTA, or new "
        "wording unsupported by the approved evidence. Return strict structured output."
    )


def repair_payload(
    context: SourceContext, semantic: dict[str, Any], plan: dict[str, Any], draft: dict[str, Any],
    validation_errors: list[str], settings: dict[str, Any],
) -> str:
    return json.dumps({
        "prompt_versions": PROMPT_VERSIONS,
        "source_context": {
            "candidate_id": context.candidate_id,
            "language": context.language,
            "primary_evidence": [item.to_dict() for item in context.primary_evidence],
        },
        "approved_semantic_representation": semantic,
        "narrative_plan": plan,
        "draft_to_repair": draft,
        "validation_errors": validation_errors,
        "target_duration_seconds": settings.get("target_duration_seconds"),
        "prohibited_changes": [
            "new facts", "new numbers", "new named entities", "new dates", "new currencies",
            "new causal claims", "new CTA", "changed negation", "strengthened modality",
        ],
    }, ensure_ascii=False)
