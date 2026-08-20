from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


AI_COST_TELEMETRY_VERSION = "ai-cost.1"
OPENAI_TERRA_PRICING_VERSION = "openai-gpt-5.6-terra-2026-08-21"
OPENAI_TERRA_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-terra"
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPricing:
    provider: str
    model: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    cache_write_input_multiplier: float
    output_usd_per_million: float
    long_context_threshold_tokens: int
    long_context_input_multiplier: float
    long_context_output_multiplier: float
    version: str
    source_url: str


_MODEL_PRICING: dict[tuple[str, str], ModelPricing] = {
    ("openai", "gpt-5.6-terra"): ModelPricing(
        provider="openai",
        model="gpt-5.6-terra",
        input_usd_per_million=2.0,
        cached_input_usd_per_million=0.2,
        cache_write_input_multiplier=1.25,
        output_usd_per_million=12.0,
        long_context_threshold_tokens=272_000,
        long_context_input_multiplier=2.0,
        long_context_output_multiplier=1.5,
        version=OPENAI_TERRA_PRICING_VERSION,
        source_url=OPENAI_TERRA_PRICING_SOURCE,
    ),
}


def collect_vision_usage(
    pass1: dict[str, Any] | None,
    pass2: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Collect billed Vision usage without walking duplicated candidate evidence."""

    records: list[dict[str, Any]] = []
    _append_vision_usage(records, pass1, "vision.pass1")
    candidates = pass2.get("candidates", []) if isinstance(pass2, dict) else []
    if not isinstance(candidates, list):
        return records
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        evidence = candidate.get("vision_pass2_evidence")
        result = evidence.get("result") if isinstance(evidence, dict) else None
        candidate_id = str(candidate.get("id") or candidate.get("candidate_id") or "unknown")
        _append_vision_usage(records, result, f"vision.pass2.{candidate_id}")
    return records


def calculate_ai_cost_telemetry(
    semantic_usage: dict[str, Any] | None,
    vision_usage: Iterable[dict[str, Any]] | None,
    *,
    source_duration_seconds: Any,
    default_provider: str,
    default_model: str,
) -> dict[str, Any]:
    """Price actual provider usage; never participate in provider admission."""

    semantic_records = [dict(semantic_usage or {})]
    if semantic_records[0].get("usage_source") is None:
        semantic_records[0]["usage_source"] = "semantic.provider_response_usage"
    semantic = calculate_component_cost(
        "semantic",
        semantic_records,
        default_provider=default_provider,
        default_model=default_model,
    )
    vision = calculate_component_cost(
        "vision",
        list(vision_usage or []),
        default_provider=default_provider,
        default_model=default_model,
    )
    component_totals = [semantic["cost_usd"]["total"], vision["cost_usd"]["total"]]
    total_cost = (
        round(sum(float(value) for value in component_totals), 8)
        if all(isinstance(value, (int, float)) for value in component_totals)
        else None
    )
    duration = _nonnegative_float(source_duration_seconds)
    source_hours = round(duration / 3600.0, 8) if duration is not None and duration > 0 else None
    cost_per_source_hour = (
        round(float(total_cost) / (duration / 3600.0), 8)
        if isinstance(total_cost, (int, float)) and duration is not None and duration > 0
        else None
    )
    return {
        "schema_version": AI_COST_TELEMETRY_VERSION,
        "currency": "USD",
        "semantic": semantic,
        "vision": vision,
        "total_cost_usd": total_cost,
        "source_duration_seconds": duration,
        "source_hours": source_hours,
        "cost_per_source_hour_usd": cost_per_source_hour,
        "provenance": {
            "owner": "app.ai_cost.calculate_ai_cost_telemetry",
            "cost_basis": "provider_reported_token_usage",
            "calculation": "tokens * applied_rate_usd_per_1m / 1000000",
            "admission_behavior": "independent; config AI token prices remain admission-only",
        },
    }


def calculate_component_cost(
    component: str,
    usage_records: Iterable[dict[str, Any]],
    *,
    default_provider: str,
    default_model: str,
) -> dict[str, Any]:
    records = [
        _price_usage_record(
            record,
            default_provider=default_provider,
            default_model=default_model,
        )
        for record in usage_records
        if isinstance(record, dict)
    ]
    if not records:
        records = [_price_usage_record(
            {"usage_source": f"{component}.no_provider_calls"},
            default_provider=default_provider,
            default_model=default_model,
        )]
    providers = sorted({str(item["provider"]) for item in records})
    models = sorted({str(item["model"]) for item in records})
    statuses = {str(item["status"]) for item in records}
    if statuses <= {"priced", "no_usage"}:
        status = "priced" if "priced" in statuses else "no_usage"
    elif len(statuses) == 1:
        status = next(iter(statuses))
    else:
        status = "partially_priced"
    totals = [item["cost_usd"]["total"] for item in records]
    costs_available = all(isinstance(value, (int, float)) for value in totals)
    token_keys = ("input", "cached_input", "cache_write_input", "uncached_input", "output")
    cost_keys = ("uncached_input", "cached_input", "cache_write_input", "output")
    rates: list[dict[str, Any]] = []
    for item in records:
        applied = item.get("applied_rates_usd_per_1m_tokens")
        if isinstance(applied, dict) and applied not in rates:
            rates.append(applied)
    api_errors: list[str] = []
    for item in records:
        for error in item.get("api_errors", []):
            value = str(error)
            if value and value not in api_errors:
                api_errors.append(value)
    cost_parts = {
        key: (
            round(sum(float(item["cost_usd"][key]) for item in records), 8)
            if costs_available else None
        )
        for key in cost_keys
    }
    cost_parts["total"] = (
        round(sum(float(value) for value in totals), 8) if costs_available else None
    )
    pricing_versions = sorted({
        str(item["provenance"]["pricing_version"])
        for item in records
        if item["provenance"].get("pricing_version")
    })
    pricing_sources = sorted({
        str(item["provenance"]["pricing_source_url"])
        for item in records
        if item["provenance"].get("pricing_source_url")
    })
    return {
        "status": status,
        "provider": providers[0] if len(providers) == 1 else "mixed",
        "model": models[0] if len(models) == 1 else "mixed",
        "request_count": sum(int(item["request_count"]) for item in records),
        "long_context_request_count": sum(int(item["long_context_request_count"]) for item in records),
        "tokens": {
            key: sum(int(item["tokens"][key]) for item in records)
            for key in token_keys
        },
        "applied_rates_usd_per_1m_tokens": rates,
        "cost_usd": cost_parts,
        "api_errors": api_errors,
        "pricing_segments": records,
        "provenance": {
            "pricing_owner": "app.ai_cost.calculate_component_cost",
            "pricing_versions": pricing_versions,
            "pricing_source_urls": pricing_sources,
            "usage_sources": [str(item["usage_source"]) for item in records],
        },
    }


def _price_usage_record(
    usage: dict[str, Any],
    *,
    default_provider: str,
    default_model: str,
) -> dict[str, Any]:
    provider = str(usage.get("provider") or default_provider or "not-called")
    model = str(usage.get("model") or default_model or "unknown")
    input_tokens = _nonnegative_int(usage.get("input_tokens"))
    reported_cached_tokens = _nonnegative_int(usage.get("cached_input_tokens"))
    cached_input_tokens = min(reported_cached_tokens, input_tokens)
    reported_cache_write_tokens = _nonnegative_int(usage.get("cache_write_input_tokens"))
    cache_write_input_tokens = min(
        reported_cache_write_tokens,
        input_tokens - cached_input_tokens,
    )
    uncached_input_tokens = input_tokens - cached_input_tokens - cache_write_input_tokens
    output_tokens = _nonnegative_int(usage.get("output_tokens"))
    total_tokens = input_tokens + output_tokens
    request_count = _nonnegative_int(usage.get("request_count") or usage.get("calls"))
    if request_count == 0 and total_tokens > 0:
        request_count = 1
    raw_errors = usage.get("api_errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = [raw_errors]
    api_errors = [str(value) for value in raw_errors if str(value)]
    adjustments: list[str] = []
    if reported_cached_tokens > input_tokens:
        adjustments.append("cached_input_tokens_clamped_to_input_tokens")
    if reported_cache_write_tokens > input_tokens - cached_input_tokens:
        adjustments.append("cache_write_input_tokens_clamped_to_remaining_input_tokens")
    if "cached_input_tokens" not in usage or "cache_write_input_tokens" not in usage:
        adjustments.append("input_token_details_not_reported; missing categories assumed_zero")
    pricing = _MODEL_PRICING.get((provider, model))
    rates: dict[str, Any] | None = None
    cost = {
        "uncached_input": None,
        "cached_input": None,
        "cache_write_input": None,
        "output": None,
        "total": None,
    }
    long_context_request_count = 0
    status = "no_usage" if total_tokens == 0 else "pricing_unavailable"
    if total_tokens == 0:
        cost = {
            "uncached_input": 0.0,
            "cached_input": 0.0,
            "cache_write_input": 0.0,
            "output": 0.0,
            "total": 0.0,
        }
    elif pricing is not None:
        is_long_context = input_tokens > pricing.long_context_threshold_tokens
        if is_long_context and request_count > 1:
            status = "request_granularity_required"
        else:
            input_multiplier = pricing.long_context_input_multiplier if is_long_context else 1.0
            output_multiplier = pricing.long_context_output_multiplier if is_long_context else 1.0
            rates = {
                "context_tier": "long" if is_long_context else "standard",
                "uncached_input": pricing.input_usd_per_million * input_multiplier,
                "cached_input": pricing.cached_input_usd_per_million * input_multiplier,
                "cache_write_input": (
                    pricing.input_usd_per_million
                    * pricing.cache_write_input_multiplier
                    * input_multiplier
                ),
                "output": pricing.output_usd_per_million * output_multiplier,
            }
            uncached_cost = uncached_input_tokens * float(rates["uncached_input"]) / TOKENS_PER_MILLION
            cached_cost = cached_input_tokens * float(rates["cached_input"]) / TOKENS_PER_MILLION
            cache_write_cost = (
                cache_write_input_tokens * float(rates["cache_write_input"]) / TOKENS_PER_MILLION
            )
            output_cost = output_tokens * float(rates["output"]) / TOKENS_PER_MILLION
            cost = {
                "uncached_input": round(uncached_cost, 8),
                "cached_input": round(cached_cost, 8),
                "cache_write_input": round(cache_write_cost, 8),
                "output": round(output_cost, 8),
                "total": round(uncached_cost + cached_cost + cache_write_cost + output_cost, 8),
            }
            status = "priced"
            long_context_request_count = 1 if is_long_context else 0
    return {
        "status": status,
        "usage_source": str(usage.get("usage_source") or "provider_response_usage"),
        "provider": provider,
        "model": model,
        "request_count": request_count,
        "long_context_request_count": long_context_request_count,
        "tokens": {
            "input": input_tokens,
            "cached_input": cached_input_tokens,
            "cache_write_input": cache_write_input_tokens,
            "uncached_input": uncached_input_tokens,
            "output": output_tokens,
        },
        "applied_rates_usd_per_1m_tokens": rates,
        "cost_usd": cost,
        "api_errors": api_errors,
        "provenance": {
            "pricing_version": pricing.version if pricing is not None else None,
            "pricing_source_url": pricing.source_url if pricing is not None else None,
            "long_context_rule": (
                f"input_tokens > {pricing.long_context_threshold_tokens}: "
                f"input x{pricing.long_context_input_multiplier:g}, "
                f"output x{pricing.long_context_output_multiplier:g}"
                if pricing is not None else None
            ),
            "cache_write_rule": (
                f"cache_write_input_tokens x{pricing.cache_write_input_multiplier:g} "
                "of applied uncached input rate"
                if pricing is not None else None
            ),
            "usage_adjustments": adjustments,
        },
    }


def _append_vision_usage(
    records: list[dict[str, Any]],
    artifact: Any,
    usage_source: str,
) -> None:
    if not isinstance(artifact, dict):
        return
    diagnostics = artifact.get("diagnostics")
    usage = diagnostics.get("usage") if isinstance(diagnostics, dict) else None
    if not isinstance(usage, dict):
        return
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        observations = artifact.get("observations", [])
        first = observations[0] if isinstance(observations, list) and observations else None
        provenance = first.get("provenance") if isinstance(first, dict) else None
    record = dict(usage)
    record["usage_source"] = usage_source
    if isinstance(provenance, dict):
        record["provider"] = provenance.get("provider")
        record["model"] = provenance.get("model")
    failure_reason = diagnostics.get("failure_reason") if isinstance(diagnostics, dict) else None
    if isinstance(failure_reason, str) and failure_reason.startswith("provider_failure:"):
        record["api_errors"] = [failure_reason.removeprefix("provider_failure:")]
    records.append(record)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None
