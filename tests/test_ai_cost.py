from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai_cost import calculate_ai_cost_telemetry, collect_vision_usage
from app.config import AIConfig, AppConfig
from app.reporting import make_report


TERRA_USAGE = {"provider": "openai", "model": "gpt-5.6-terra"}


def _telemetry(input_tokens: int, output_tokens: int, **usage: int) -> dict:
    return calculate_ai_cost_telemetry(
        {**TERRA_USAGE, "input_tokens": input_tokens, "output_tokens": output_tokens, **usage},
        [],
        source_duration_seconds=1081.761,
        default_provider="openai",
        default_model="gpt-5.6-terra",
    )


def test_compact_terra_usage_uses_standard_rates() -> None:
    telemetry = _telemetry(21_770, 2_419)
    semantic = telemetry["semantic"]

    assert semantic["tokens"] == {
        "input": 21_770,
        "cached_input": 0,
        "cache_write_input": 0,
        "uncached_input": 21_770,
        "output": 2_419,
    }
    assert semantic["applied_rates_usd_per_1m_tokens"] == [{
        "context_tier": "standard",
        "uncached_input": 2.0,
        "cached_input": 0.2,
        "cache_write_input": 2.5,
        "output": 12.0,
    }]
    assert semantic["cost_usd"] == {
        "uncached_input": 0.04354,
        "cached_input": 0.0,
        "cache_write_input": 0.0,
        "output": 0.029028,
        "total": 0.072568,
    }


def test_old_terra_baseline_applies_long_context_rates_to_full_request() -> None:
    telemetry = _telemetry(325_508, 2_274)
    semantic = telemetry["semantic"]

    assert semantic["long_context_request_count"] == 1
    assert semantic["applied_rates_usd_per_1m_tokens"] == [{
        "context_tier": "long",
        "uncached_input": 4.0,
        "cached_input": 0.4,
        "cache_write_input": 5.0,
        "output": 18.0,
    }]
    assert semantic["cost_usd"] == {
        "uncached_input": 1.302032,
        "cached_input": 0.0,
        "cache_write_input": 0.0,
        "output": 0.040932,
        "total": 1.342964,
    }
    assert semantic["pricing_segments"][0]["provenance"]["usage_adjustments"] == [
        "input_token_details_not_reported; missing categories assumed_zero"
    ]


def test_cached_input_tokens_are_priced_and_reported_separately() -> None:
    semantic = _telemetry(100_000, 1_000, cached_input_tokens=40_000)["semantic"]

    assert semantic["tokens"]["cached_input"] == 40_000
    assert semantic["tokens"]["uncached_input"] == 60_000
    assert semantic["cost_usd"] == {
        "uncached_input": 0.12,
        "cached_input": 0.008,
        "cache_write_input": 0.0,
        "output": 0.012,
        "total": 0.14,
    }


def test_cache_write_tokens_use_the_reported_terra_write_rate() -> None:
    semantic = _telemetry(
        100_000,
        1_000,
        cached_input_tokens=20_000,
        cache_write_input_tokens=30_000,
    )["semantic"]

    assert semantic["tokens"]["uncached_input"] == 50_000
    assert semantic["tokens"]["cached_input"] == 20_000
    assert semantic["tokens"]["cache_write_input"] == 30_000
    assert semantic["cost_usd"] == {
        "uncached_input": 0.1,
        "cached_input": 0.004,
        "cache_write_input": 0.075,
        "output": 0.012,
        "total": 0.191,
    }


def test_report_totals_semantic_and_existing_baseline_vision_usage(tmp_path: Path) -> None:
    config = AppConfig(ai=AIConfig(
        provider="openai",
        model="gpt-5.6-terra",
        input_token_price=99.0,
        output_token_price=99.0,
    ))
    pass1 = {
        "provenance": {"provider": "openai", "model": "gpt-5.6-terra"},
        "diagnostics": {"usage": {
            "frames": 18,
            "calls": 6,
            "input_tokens": 5_814,
            "output_tokens": 2_313,
        }},
    }
    pass2 = {"candidates": [{
        "id": "candidate-chapter-003-story-001",
        "vision_pass2_evidence": {"result": {
            "observations": [{"provenance": {
                "provider": "openai",
                "model": "gpt-5.6-terra",
            }}],
            "diagnostics": {"usage": {
                "frames": 7,
                "calls": 3,
                "input_tokens": 2_539,
                "output_tokens": 1_056,
            }},
        }},
    }]}
    report_path = tmp_path / "report.json"

    make_report(
        report_path,
        {},
        {"duration": 1081.761},
        config,
        {},
        0,
        14,
        [],
        [],
        [],
        {**TERRA_USAGE, "input_tokens": 21_770, "output_tokens": 2_419},
        False,
        False,
        vision_ai_usage=collect_vision_usage(pass1, pass2),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    cost = report["ai_cost"]
    assert cost["vision"]["request_count"] == 9
    assert cost["vision"]["tokens"] == {
        "input": 8_353,
        "cached_input": 0,
        "cache_write_input": 0,
        "uncached_input": 8_353,
        "output": 3_369,
    }
    assert cost["vision"]["cost_usd"]["total"] == 0.057134
    assert cost["semantic"]["cost_usd"]["total"] == 0.072568
    assert cost["total_cost_usd"] == 0.129702
    assert cost["cost_per_source_hour_usd"] == pytest.approx(0.43163619)
    assert report["ai"]["estimated_cost"] == 0.129702
    assert report["ai"]["semantic_cost_usd"] == 0.072568
    assert report["ai"]["vision_cost_usd"] == 0.057134
    assert cost["provenance"]["admission_behavior"].startswith("independent")


def test_multiple_calls_over_long_context_threshold_require_request_granularity() -> None:
    telemetry = calculate_ai_cost_telemetry(
        {**TERRA_USAGE, "input_tokens": 1, "output_tokens": 1},
        [{**TERRA_USAGE, "calls": 2, "input_tokens": 300_000, "output_tokens": 1_000}],
        source_duration_seconds=60,
        default_provider="openai",
        default_model="gpt-5.6-terra",
    )

    assert telemetry["vision"]["status"] == "request_granularity_required"
    assert telemetry["vision"]["cost_usd"]["total"] is None
    assert telemetry["total_cost_usd"] is None
