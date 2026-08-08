"""Explicitly opt-in, bounded real-provider smoke for Goal 6B.

Run only with ``RUN_VISION_REAL_SMOKE=1`` and a configured provider key.  The
test sends at most one low-detail frame in one request and never processes a
long source.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import pytest

from app.ai import get_vision_provider
from app.config import AppConfig
from app.multimodal_evidence import build_multimodal_timeline
from app.vision_intelligence import VisionGateway


def _smoke_timeline() -> dict[str, Any]:
    return build_multimodal_timeline(
        source_id="vision-real-smoke",
        source_duration_seconds=60.0,
        transcript={
            "source_id": "vision-real-smoke", "duration": 60.0, "language": "en",
            "segments": [
                {"id": 1, "start": 2.0, "end": 12.0, "text": "A visible opening beat.", "confidence": 0.9},
                {"id": 2, "start": 28.0, "end": 40.0, "text": "A visible result beat.", "confidence": 0.9},
            ],
            "words": [],
        },
        audio_features={"window_seconds": 0.5, "energy_frames": [], "silence_intervals": []},
        scenes={
            "enabled": True,
            "boundaries": [{"timestamp": 25.0, "scene_change_score": 0.8}],
        },
        visual_analysis={
            "schema_version": "5D.0", "enabled": False, "status": "fallback",
            "evidence_status": "fallback", "reason": "vision_gateway",
            "subject_keyframes": [], "sample_count": 0,
        },
    )


def test_real_provider_one_frame_hard_budget_smoke() -> None:
    if os.getenv("RUN_VISION_REAL_SMOKE") != "1":
        pytest.skip("Set RUN_VISION_REAL_SMOKE=1 to opt in to the paid smoke.")
    provider_name = os.getenv("VISION_SMOKE_PROVIDER", "openai")
    key_name = "OPENAI_API_KEY" if provider_name == "openai" else "GEMINI_API_KEY"
    if not os.getenv(key_name):
        pytest.skip(f"{key_name} is unavailable; no paid request was made.")
    source = Path("input/smoke-test.mp4")
    if not source.is_file():
        pytest.skip("The short local smoke source is unavailable.")

    config = AppConfig(optional_visual_features=True)
    config.ai.provider = provider_name
    config.product_flow.processing_mode = "standard"
    config.vision.pass1_batch_size = 2
    config.vision.prompt_version = "6B.real-smoke.3"
    config.vision.standard_max_frames = 1
    config.vision.standard_max_calls = 1
    config.vision.standard_max_tokens = 3200
    config.vision.standard_max_estimated_cost = 0.005
    config.vision.max_output_tokens_per_call = 2000
    config.validate()
    artifact = VisionGateway(
        config=config,
        cache_directory=Path("work/vision-real-smoke-cache"),
        provider=get_vision_provider(config),
    ).analyze_pass1(source=source, timeline=_smoke_timeline(), content_type="unknown")

    usage = artifact["diagnostics"]["usage"]
    print("VISION_SMOKE_USAGE=" + json.dumps({
        "status": artifact["status"],
        "cache_hits": artifact["diagnostics"]["cache_hits"],
        "cache_misses": artifact["diagnostics"]["cache_misses"],
        "frames_sent": artifact["diagnostics"]["frames_sent"],
        "usage": usage,
    }, sort_keys=True))
    if artifact["status"] not in {"completed", "partial"}:
        pytest.fail(json.dumps(artifact["diagnostics"], ensure_ascii=False, sort_keys=True))
    assert usage["calls"] <= 1
    assert usage["frames"] <= 1
    assert usage["reserved_total_tokens"] <= 3200
    assert usage["hard_budget_consumed_estimated_cost"] <= 0.005
