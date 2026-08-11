from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from app.ai import sanitize_api_error
from app.config import TransformationConfig
from app.errors import (
    NarrativePlanningError,
    ScriptGenerationError,
    SemanticExtractionError,
    TransformationFallbackError,
    TransformationProviderError,
)
from app.narrative_planning import build_narrative_plan
from app.script_generation import generate_script_draft, recompute_script_metrics
from app.script_validation import validate_script_grounding, validate_script_quality
from app.semantic_extraction import extract_semantic_representation
from app.transformation_fallback import build_local_fallback
from app.transformation_models import (
    FallbackReason,
    FINAL_SCRIPT_CONTRACT_VERSION,
    FinalScript,
    ScriptDraft,
    SourceContext,
    ValidationResult,
    draft_from_dict,
    plan_from_dict,
    semantic_from_dict,
    validate_final_script,
)
from app.transformation_prompts import PROMPT_VERSIONS


TRANSFORMATION_ENGINE_VERSION = "2.2.0"


def run_content_transformation(
    context: SourceContext, config: TransformationConfig, provider: Any | None,
    force_local: bool = False,
) -> dict[str, Any]:
    """Run the safe control plane. A provider result is never final without Python checks."""

    started = time.perf_counter()
    timings: dict[str, float] = {}
    usage: dict[str, Any] = _local_usage("local")
    repair_attempts: list[dict[str, Any]] = []
    fallback_reason: FallbackReason | None = None
    semantic = None
    plan = None
    draft = None
    grounding = None
    quality_validation = None
    quality = None
    normalization_warnings: list[str] = []
    strategy = "local_only" if force_local else config.ai_strategy
    requested_language = context.language if config.output_language == "auto" else config.output_language
    if requested_language != context.language:
        message = "Translation stage is configured but not implemented in Goal 2; source-language script was not fabricated."
        return _failed_result(context, strategy, usage, timings, message, FallbackReason.EMPTY_RESULT, started)
    try:
        step = time.perf_counter()
        if strategy == "local_only":
            semantic = extract_semantic_representation(context)
            timings["semantic_extraction"] = _elapsed(step)
            step = time.perf_counter()
            plan = build_narrative_plan(semantic, config)
            timings["narrative_planning"] = _elapsed(step)
            step = time.perf_counter()
            draft = generate_script_draft(semantic, plan, config.target_words_per_second)
            timings["script_generation"] = _elapsed(step)
            fallback_reason = FallbackReason.AI_DISABLED
        else:
            if provider is None:
                raise TransformationProviderError("AI provider для transformation не создан.")
            step = time.perf_counter()
            raw, usage = provider.transform_compact(context)
            timings["provider_compact"] = _elapsed(step)
            raw, response_warnings = _normalize_provider_response(raw, context)
            normalization_warnings.extend(response_warnings)
            step = time.perf_counter()
            semantic = semantic_from_dict(dict(raw["semantic_representation"]))
            semantic.validate(context)
            timings["semantic_extraction"] = _elapsed(step)
            step = time.perf_counter()
            plan = plan_from_dict(dict(raw["narrative_plan"]))
            plan.validate(semantic, config.allow_cta)
            timings["narrative_planning"] = _elapsed(step)
            step = time.perf_counter()
            draft = draft_from_dict(dict(raw["script_draft"]))
            draft = recompute_script_metrics(draft, config.target_words_per_second)
            draft.validate_shape(semantic)
            timings["script_generation"] = _elapsed(step)
    except TransformationProviderError as error:
        fallback_reason = _provider_reason(str(error))
        usage = _failed_usage(provider, error)
    except (KeyError, TypeError, ValueError, SemanticExtractionError, NarrativePlanningError, ScriptGenerationError) as error:
        fallback_reason = FallbackReason.INVALID_STRUCTURED_OUTPUT
        usage = _failed_usage(provider, error)

    if fallback_reason is not None:
        # A partially parsed AI payload must never become fallback input.  The
        # deterministic path starts only from current candidate source data.
        semantic = None
        plan = None
        draft = None

    if semantic is None:
        # Even when an external provider failed, source-grounded local semantics are
        # created afresh and are never treated as an AI response cache hit.
        step = time.perf_counter()
        try:
            semantic = extract_semantic_representation(context)
            timings["semantic_extraction"] = timings.get("semantic_extraction", 0) + _elapsed(step)
            step = time.perf_counter()
            plan = build_narrative_plan(semantic, config)
            timings["narrative_planning"] = timings.get("narrative_planning", 0) + _elapsed(step)
        except Exception as error:
            return _failed_result(context, strategy, usage, timings, str(error), fallback_reason or FallbackReason.EMPTY_RESULT, started)

    if draft is not None:
        step = time.perf_counter()
        grounding = validate_script_grounding(draft, semantic, context, config.allow_cta)
        timings["grounding_validation"] = _elapsed(step)
        step = time.perf_counter()
        quality_validation, quality = validate_script_quality(draft, semantic, grounding, config)
        timings["quality_validation"] = _elapsed(step)

    if draft is None or grounding is None or quality_validation is None or not grounding.passed or not quality_validation.passed:
        failure_reason = fallback_reason or (
            FallbackReason.GROUNDING_FAILED if grounding is not None and not grounding.passed
            else FallbackReason.QUALITY_FAILED if quality_validation is not None and not quality_validation.passed
            else FallbackReason.EMPTY_RESULT
        )
        if not config.fallback_enabled and draft is None:
            return _failed_result(context, strategy, usage, timings, "AI transformation failed and fallback_enabled=false.", failure_reason, started, semantic, plan, repair_attempts)
        repaired, repair_usage = (
            _repair_with_approved_facts(
                context, semantic, plan, draft, config, failure_reason, repair_attempts,
                provider if strategy != "local_only" else None,
                [*(grounding.errors if grounding else []), *(quality_validation.errors if quality_validation else [])],
            ) if draft is not None else (None, None)
        )
        if repair_usage:
            usage = _merge_usage(usage, repair_usage)
        if repaired is not None:
            draft = repaired
            step = time.perf_counter()
            grounding = validate_script_grounding(draft, semantic, context, config.allow_cta)
            timings["grounding_validation"] = timings.get("grounding_validation", 0) + _elapsed(step)
            step = time.perf_counter()
            quality_validation, quality = validate_script_quality(draft, semantic, grounding, config)
            timings["quality_validation"] = timings.get("quality_validation", 0) + _elapsed(step)
        if draft is None or grounding is None or quality_validation is None or not grounding.passed or not quality_validation.passed:
            fallback_reason = FallbackReason.REPAIR_FAILED if repair_attempts else failure_reason
            if not config.fallback_enabled:
                return _failed_result(
                    context, strategy, usage, timings,
                    "Transformation validation failed and fallback_enabled=false.", fallback_reason,
                    started, semantic, plan, repair_attempts,
                )
            step = time.perf_counter()
            try:
                # A provider-owned semantic representation is only valid for the
                # provider draft that was checked against it. If that draft and
                # its bounded repair both fail, rebuild the final fallback from
                # the current SourceContext. Otherwise a structurally valid but
                # paraphrased provider semantic can leak into the supposedly
                # local fallback and invent terms or lose the approved ending.
                semantic = extract_semantic_representation(context)
                plan = build_narrative_plan(semantic, config)
                draft = build_local_fallback(context, semantic, config, fallback_reason)
                grounding = validate_script_grounding(draft, semantic, context, config.allow_cta)
                quality_validation, quality = validate_script_quality(draft, semantic, grounding, config)
                timings["fallback"] = _elapsed(step)
                if not grounding.passed:
                    raise TransformationFallbackError("Local fallback не прошёл deterministic grounding.")
            except Exception as error:
                return _failed_result(context, strategy, usage, timings, str(error), fallback_reason, started, semantic, plan, repair_attempts)
            final = FinalScript.from_draft(draft, "fallback", True, fallback_reason)
            final_validation = validate_final_script(final, context, semantic, context.candidate_id)
            if not final_validation.passed:
                return _failed_result(
                    context, strategy, usage, timings, _contract_error(final_validation), fallback_reason,
                    started, semantic, plan, repair_attempts, normalization_warnings,
                )
            return _result(
                context, strategy, usage, semantic, plan, draft, grounding, quality_validation, quality,
                repair_attempts, final, final_validation, fallback_reason, normalization_warnings, timings, started,
            )

    final_status = "fallback" if strategy == "local_only" else "completed"
    final = FinalScript.from_draft(draft, final_status, True, fallback_reason if strategy == "local_only" else None)
    final_validation = validate_final_script(final, context, semantic, context.candidate_id)
    if not final_validation.passed:
        return _failed_result(
            context, strategy, usage, timings, _contract_error(final_validation),
            fallback_reason or FallbackReason.INVALID_STRUCTURED_OUTPUT,
            started, semantic, plan, repair_attempts, normalization_warnings,
        )
    return _result(
        context, strategy, usage, semantic, plan, draft, grounding, quality_validation, quality,
        repair_attempts, final, final_validation, fallback_reason if strategy == "local_only" else None,
        normalization_warnings, timings, started,
    )


def _repair_with_approved_facts(
    context: SourceContext, semantic: Any, plan: Any, draft: ScriptDraft | None,
    config: TransformationConfig, reason: FallbackReason, attempts: list[dict[str, Any]],
    provider: Any | None, validation_errors: list[str],
) -> tuple[ScriptDraft | None, dict[str, Any] | None]:
    if config.max_repair_attempts < 1 or plan is None:
        return None, None
    if config.mock_mode == "repair_failure":
        attempts.append({"attempt": 1, "method": "deterministic_approved_facts", "status": "failed", "reason": "mock repair_failure"})
        return None, None
    for attempt in range(1, config.max_repair_attempts + 1):
        if provider is not None and hasattr(provider, "repair_script"):
            try:
                raw, repair_usage = provider.repair_script(
                    context, semantic.to_dict(), plan.to_dict(), draft.to_dict() if draft else {}, validation_errors,
                )
                repaired = recompute_script_metrics(draft_from_dict(raw), config.target_words_per_second)
                repaired.validate_shape(semantic)
                repaired.status = "repaired"
                repaired.transformation_notes.append(f"Structured repair attempt {attempt}: exact validation errors supplied to provider.")
                attempts.append({
                    "attempt": attempt,
                    "method": "structured_provider_repair",
                    "status": "completed",
                    "validation_errors": validation_errors,
                    "prohibited_changes": ["new facts", "new numbers", "new entities", "new CTA", "changed modality"],
                })
                return repaired, repair_usage
            except Exception as error:
                attempts.append({
                    "attempt": attempt, "method": "structured_provider_repair", "status": "failed",
                    "error": sanitize_api_error(error), "validation_errors": validation_errors,
                })
        try:
            repaired = generate_script_draft(semantic, plan, config.target_words_per_second)
            repaired.status = "repaired"
            repaired.transformation_notes.append(f"Deterministic repair attempt {attempt}: rebuilt only from approved facts after {reason.value}.")
            attempts.append({
                "attempt": attempt,
                "method": "deterministic_approved_facts",
                "status": "completed",
                "input_draft_present": draft is not None,
                "prohibited_changes": ["new facts", "new numbers", "new entities", "new CTA", "changed modality"],
            })
            return repaired, None
        except Exception as error:
            attempts.append({"attempt": attempt, "method": "deterministic_approved_facts", "status": "failed", "error": str(error)})
    return None, None


def _result(
    context: SourceContext, strategy: str, usage: dict[str, Any], semantic: Any, plan: Any,
    draft: ScriptDraft, grounding: Any, quality_validation: Any, quality: Any,
    repair_attempts: list[dict[str, Any]], final: FinalScript,
    final_validation: Any, fallback_reason: FallbackReason | None,
    normalization_warnings: list[str], timings: dict[str, float], started: float,
) -> dict[str, Any]:
    timings["total_transformation"] = _elapsed(started)
    return {
        "engine_version": TRANSFORMATION_ENGINE_VERSION,
        "enabled": True,
        "status": final.status,
        "candidate_id": context.candidate_id,
        "strategy": strategy,
        "provider": str(usage.get("provider", "local")),
        "model": usage.get("model"),
        "prompt_versions": PROMPT_VERSIONS,
        "source_context": context.to_dict(),
        "semantic_representation": semantic.to_dict(),
        "narrative_plan": plan.to_dict(),
        "draft_script": draft.to_dict(),
        "validation": {
            "grounding": grounding.to_dict(),
            "quality": quality_validation.to_dict(),
            "final_script": final_validation.to_dict(),
            "errors": [*grounding.errors, *quality_validation.errors, *final_validation.errors],
            "warnings": [*grounding.warnings, *quality_validation.warnings, *normalization_warnings],
        },
        "repair_attempts": repair_attempts,
        "final_script": final.to_dict(),
        "final_script_contract_version": FINAL_SCRIPT_CONTRACT_VERSION,
        "final_script_source": "fallback" if final.status == "fallback" else "local" if strategy == "local_only" else "ai",
        "normalization": {"warnings": normalization_warnings},
        "fallback": {"used": final.status == "fallback", "reason": fallback_reason.value if fallback_reason else None},
        "scores": {"grounding": grounding.score, "quality": quality.final_score, "quality_details": quality.to_dict()},
        "ai_usage": usage,
        "timings": timings,
        "cacheable": final_validation.passed and not (
            final.status == "fallback"
            and fallback_reason in {FallbackReason.PROVIDER_FAILURE, FallbackReason.TIMEOUT, FallbackReason.RATE_LIMIT}
        ),
        "production_note": "Transformed script is a future TTS artifact; existing MP4 audio and subtitles remain original.",
    }


def _failed_result(
    context: SourceContext, strategy: str, usage: dict[str, Any], timings: dict[str, float],
    error: str, reason: FallbackReason, started: float, semantic: Any | None = None,
    plan: Any | None = None, repair_attempts: list[dict[str, Any]] | None = None,
    normalization_warnings: list[str] | None = None,
) -> dict[str, Any]:
    timings["total_transformation"] = _elapsed(started)
    return {
        "engine_version": TRANSFORMATION_ENGINE_VERSION,
        "enabled": True,
        "status": "failed",
        "candidate_id": context.candidate_id,
        "strategy": strategy,
        "provider": str(usage.get("provider", "local")),
        "model": usage.get("model"),
        "prompt_versions": PROMPT_VERSIONS,
        "source_context": context.to_dict(),
        "semantic_representation": semantic.to_dict() if semantic else {},
        "narrative_plan": plan.to_dict() if plan else {},
        "draft_script": {},
        "validation": {
            "grounding": {}, "quality": {},
            "final_script": {
                "passed": False,
                "errors": [error],
                "checks": {"contract_version": FINAL_SCRIPT_CONTRACT_VERSION},
            },
            "errors": [error],
            "warnings": normalization_warnings or [],
        },
        "repair_attempts": repair_attempts or [],
        "final_script": {"production_ready_for_tts": False},
        "final_script_contract_version": FINAL_SCRIPT_CONTRACT_VERSION,
        "final_script_source": "none",
        "normalization": {"warnings": normalization_warnings or []},
        "fallback": {"used": False, "reason": reason.value},
        "scores": {},
        "ai_usage": usage,
        "timings": timings,
        "cacheable": False,
        "production_note": "Transformation failed safely; original render remains unaffected.",
    }


def _provider_reason(message: str) -> FallbackReason:
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return FallbackReason.TIMEOUT
    if "429" in lowered or "rate limit" in lowered:
        return FallbackReason.RATE_LIMIT
    return FallbackReason.PROVIDER_FAILURE


def validate_transformation_outcome(
    outcome: dict[str, Any], context: SourceContext,
) -> ValidationResult:
    """Revalidate a persisted result before it is treated as a completed cache hit."""

    try:
        return validate_final_script(
            outcome.get("final_script", {}),
            context,
            outcome.get("semantic_representation", {}),
            context.candidate_id,
        )
    except Exception as error:
        return ValidationResult(False, 0.0, [f"Cached FinalScript cannot be validated: {error}"], [])


def _normalize_provider_response(raw: Any, context: SourceContext) -> tuple[dict[str, Any], list[str]]:
    """Bind untrusted AI content to the current candidate before domain parsing."""

    if not isinstance(raw, dict):
        raise ValueError("AI transformation response must be an object.")
    normalized = deepcopy(raw)
    warnings: list[str] = []
    expected = context.candidate_id
    for name in ("semantic_representation", "narrative_plan", "script_draft"):
        value = normalized.get(name)
        if not isinstance(value, dict):
            continue
        proposed = str(value.get("candidate_id", "")).strip()
        if proposed != expected:
            warnings.append(f"AI {name}.candidate_id was ignored; current candidate identity was retained.")
        value["candidate_id"] = expected
    semantic = normalized.get("semantic_representation")
    if isinstance(semantic, dict):
        _restore_semantic_evidence_map(semantic, context, warnings)
    return normalized, warnings


def _restore_semantic_evidence_map(
    semantic: dict[str, Any], context: SourceContext, warnings: list[str],
) -> None:
    """Recover only direct, real candidate segment references; never invent evidence."""

    evidence_by_id = context.evidence_by_id()
    known = set(evidence_by_id)
    raw_map = semantic.get("source_evidence_map")
    direct_map = raw_map if isinstance(raw_map, dict) else {}
    normalized_map: dict[str, list[int]] = {}
    for fact in semantic.get("supporting_facts", []):
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("fact_id", "")).strip()
        try:
            source_ids = [int(item) for item in fact.get("evidence_segment_ids", [])]
        except (TypeError, ValueError):
            source_ids = []
        if not fact_id or not source_ids or not set(source_ids).issubset(known):
            continue
        current = direct_map.get(fact_id)
        try:
            current_ids = [int(item) for item in current] if isinstance(current, list) else []
        except (TypeError, ValueError):
            current_ids = []
        if set(current_ids) != set(source_ids):
            warnings.append(f"AI source_evidence_map was restored from real candidate segments for {fact_id}.")
        source_confidence = min(evidence_by_id[item].confidence for item in source_ids)
        try:
            proposed_confidence = float(fact.get("confidence", 0.0))
        except (TypeError, ValueError):
            proposed_confidence = 0.0
        grounded_confidence = max(0.0, min(1.0, min(proposed_confidence, source_confidence)))
        if grounded_confidence != proposed_confidence:
            warnings.append(
                f"AI confidence was capped by transcript evidence for {fact_id}."
            )
        fact["confidence"] = grounded_confidence
        normalized_map[fact_id] = source_ids
    semantic["source_evidence_map"] = normalized_map


def _contract_error(validation: Any) -> str:
    errors = list(getattr(validation, "errors", []))
    return "FinalScript contract validation failed: " + "; ".join(errors[:5])


def _failed_usage(provider: Any | None, error: BaseException) -> dict[str, Any]:
    ai_config = getattr(getattr(provider, "config", None), "ai", None)
    return {
        "provider": getattr(provider, "name", "unavailable"),
        "model": getattr(ai_config, "model", None),
        "input_tokens": 0,
        "output_tokens": 0,
        "retries": 0,
        "api_errors": [sanitize_api_error(error)],
    }


def _local_usage(provider: str) -> dict[str, Any]:
    return {"provider": provider, "model": None, "input_tokens": 0, "output_tokens": 0, "retries": 0, "api_errors": []}


def _merge_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Aggregate compact + bounded repair usage without losing a safe request id."""

    merged = dict(first)
    merged["provider"] = second.get("provider", first.get("provider"))
    merged["model"] = second.get("model", first.get("model"))
    for name in ("input_tokens", "output_tokens", "retries"):
        merged[name] = int(first.get(name, 0) or 0) + int(second.get(name, 0) or 0)
    merged["api_errors"] = [*first.get("api_errors", []), *second.get("api_errors", [])]
    request_ids = [value for value in (first.get("request_id"), second.get("request_id")) if value]
    if request_ids:
        merged["request_id"] = request_ids[-1]
        merged["request_ids"] = request_ids
    if second.get("response_status") is not None:
        merged["response_status"] = second["response_status"]
    return merged


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 4)
