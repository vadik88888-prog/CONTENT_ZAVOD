from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import AppConfig
from app.multimodal_evidence import build_multimodal_timeline
from app.vision_intelligence import (
    CostController,
    VisionBudget,
    VisionContractError,
    VisionGateway,
    VisionProviderCallError,
    build_pass2_request,
    dynamic_frame_budget,
    validate_provider_response,
)


def _timeline() -> dict[str, Any]:
    segments = [
        {
            "id": index,
            "start": float(index * 10 + 1),
            "end": float(index * 10 + 8),
            "text": f"Grounded statement {index} has a complete visible beat.",
            "confidence": 0.9,
        }
        for index in range(8)
    ]
    transcript = {
        "source_id": "source-vision",
        "duration": 90.0,
        "language": "en",
        "segments": segments,
        "words": [],
    }
    audio = {
        "window_seconds": 0.5,
        "energy_frames": [
            {"time": float(index * 10 + 2), "normalized_loudness": 0.85, "audio_energy": 0.5}
            for index in range(8)
        ],
        "silence_intervals": [],
    }
    scenes = {
        "enabled": True,
        "boundaries": [
            {"timestamp": float(value), "scene_change_score": 0.8}
            for value in (15, 30, 45, 60, 75)
        ],
    }
    visual = {
        "schema_version": "5D.0",
        "enabled": False,
        "status": "fallback",
        "evidence_status": "fallback",
        "reason": "delegated_to_budgeted_vision_gateway",
        "subject_keyframes": [],
        "sample_count": 0,
    }
    return build_multimodal_timeline(
        source_id="source-vision",
        source_duration_seconds=90.0,
        transcript=transcript,
        audio_features=audio,
        scenes=scenes,
        visual_analysis=visual,
    )


def _observation(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "keyframe_id": frame["keyframe_id"],
        "timestamp": frame["timestamp"],
        "scene_type": "TALKING_HEAD",
        "primary_subject": "face",
        "normalized_center_x": 0.5,
        "normalized_center_y": 0.4,
        "visible_face_count": 1,
        "action": "speaking",
        "reaction": "none",
        "payoff_signal": "result",
        "on_screen_text": "",
        "composition_risk": "none",
        "confidence": 0.9,
        "missing_evidence": ["text"],
    }


class _Provider:
    name = "fake-vision"

    def __init__(self, *, fail: bool = False, malformed: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail
        self.malformed = malformed

    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: str,
        pass_kind: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append([str(item["keyframe_id"]) for item in frames])
        if self.fail:
            raise RuntimeError("provider is unavailable")
        observations = [_observation(frame) for frame in frames]
        if self.malformed:
            observations[0]["unknown"] = True
        return {"observations": observations}, {
            "input_tokens": 180 + 80 * len(frames),
            "output_tokens": 90 * len(frames),
            "request_id": "vision-request-1",
        }


class _UsageFailureProvider(_Provider):
    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: str,
        pass_kind: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append([str(item["keyframe_id"]) for item in frames])
        raise VisionProviderCallError(
            "provider returned empty JSON",
            {"input_tokens": 444, "output_tokens": 222, "request_id": "failed-response"},
        )


def _config(mode: str = "standard") -> AppConfig:
    config = AppConfig(optional_visual_features=True)
    config.product_flow.processing_mode = mode
    config.vision.standard_max_frames = 8
    config.vision.standard_max_calls = 4
    config.vision.standard_max_tokens = 10000
    config.vision.standard_max_estimated_cost = 1.0
    config.vision.maximum_max_frames = 8
    config.vision.maximum_max_calls = 4
    config.vision.maximum_max_tokens = 10000
    config.vision.maximum_max_estimated_cost = 1.0
    config.validate()
    return config


def _loader(_source: Path, timestamp: float, _width: int) -> bytes:
    return f"jpeg:{timestamp:.3f}".encode()


def test_fast_mode_is_a_hard_zero_call_budget(tmp_path: Path) -> None:
    provider = _Provider()
    artifact = VisionGateway(
        config=_config("fast"), cache_directory=tmp_path / "cache",
        provider=provider, frame_loader=_loader,
    ).analyze_pass1(source=tmp_path / "source.mp4", timeline=_timeline(), content_type="gameplay")

    assert provider.calls == []
    assert artifact["status"] == "skipped"
    assert artifact["diagnostics"]["failure_reason"] == "fast_mode_zero_calls"
    assert artifact["diagnostics"]["usage"]["calls"] == 0


def test_pass1_schema_binding_and_frame_cache_prevent_repeat_charges(tmp_path: Path) -> None:
    timeline = _timeline()
    provider = _Provider()
    gateway = VisionGateway(
        config=_config(), cache_directory=tmp_path / "cache",
        provider=provider, frame_loader=_loader,
    )

    first = gateway.analyze_pass1(source=tmp_path / "source.mp4", timeline=timeline, content_type="interview")
    call_count = len(provider.calls)
    second = gateway.analyze_pass1(source=tmp_path / "source.mp4", timeline=timeline, content_type="interview")

    assert first["status"] == "completed"
    assert first["observations"]
    assert all(item["keyframe_id"] in {frame["keyframe_id"] for frame in timeline["keyframes"]} for item in first["observations"])
    assert all(item["provenance"]["frame_hash"] and item["provenance"]["cache_key"] for item in first["observations"])
    assert len(provider.calls) == call_count
    assert first["diagnostics"]["keyframes_found"]
    assert first["diagnostics"]["selected_keyframes"]
    assert first["diagnostics"]["projected_uncached_usage"]["within_hard_budget"] is True
    assert first["diagnostics"]["analysis_stop_reason"]
    assert second["diagnostics"]["cache_hits"] == len(second["observations"])
    assert second["diagnostics"]["frames_sent"] == 0
    assert second["diagnostics"]["usage"]["estimated_cost"] == 0
    assert second["diagnostics"]["analysis_stop_reason"] == "cache_satisfied_selected_frames"


def test_cost_controller_stops_before_a_call_that_exceeds_any_hard_limit(tmp_path: Path) -> None:
    config = _config()
    config.vision.standard_max_estimated_cost = 0.000001
    provider = _Provider()
    artifact = VisionGateway(
        config=config, cache_directory=tmp_path / "cache",
        provider=provider, frame_loader=_loader,
    ).analyze_pass1(source=tmp_path / "source.mp4", timeline=_timeline(), content_type="gameplay")

    assert provider.calls == []
    assert artifact["status"] == "fallback"
    assert artifact["diagnostics"]["failure_reason"] == "cost_budget_exhausted"
    assert artifact["diagnostics"]["usage"]["calls"] == 0
    assert artifact["diagnostics"]["usage"]["estimated_cost"] == 0

    controller = CostController(VisionBudget("standard", 2, 1, 1000, 1.0, 2), config)
    assert controller.reserve(1, "low") is None
    assert controller.stop_reason == "token_budget_exhausted"


def test_provider_or_schema_failure_falls_back_without_breaking_pipeline(tmp_path: Path) -> None:
    for provider in (_Provider(fail=True), _Provider(malformed=True)):
        artifact = VisionGateway(
            config=_config(), cache_directory=tmp_path / provider.__class__.__name__,
            provider=provider, frame_loader=_loader,
        ).analyze_pass1(source=tmp_path / "source.mp4", timeline=_timeline(), content_type="podcast")

        assert artifact["status"] == "fallback"
        assert artifact["diagnostics"]["failure_reason"].startswith("provider_failure:")
        assert all(item["origin"] == "local_fallback" for item in artifact["observations"])

    billed_failure = VisionGateway(
        config=_config(), cache_directory=tmp_path / "billed-failure",
        provider=_UsageFailureProvider(), frame_loader=_loader,
    ).analyze_pass1(source=tmp_path / "source.mp4", timeline=_timeline(), content_type="podcast")
    assert billed_failure["diagnostics"]["usage"]["input_tokens"] == 444
    assert billed_failure["diagnostics"]["usage"]["output_tokens"] == 222


def test_strict_response_rejects_unknown_fields_and_wrong_frame_identity() -> None:
    frame = {"keyframe_id": "keyframe-1", "timestamp": 1.25}
    valid = _observation(frame)
    assert validate_provider_response({"observations": [valid]}, [frame])[0]["timestamp"] == 1.25
    with pytest.raises(VisionContractError):
        validate_provider_response({"observations": [{**valid, "extra": True}]}, [frame])
    with pytest.raises(VisionContractError):
        validate_provider_response({"observations": [{**valid, "keyframe_id": "other"}]}, [frame])


def test_pass2_contract_selects_three_to_seven_frames_and_is_callable(tmp_path: Path) -> None:
    timeline = _timeline()
    request = build_pass2_request(
        candidate_id="candidate-1",
        window_start=0.0,
        window_end=80.0,
        anchors={"hook": 2.0, "action": 31.0, "reaction": 45.0, "payoff": 72.0},
        timeline=timeline,
        max_frames=7,
    )
    provider = _Provider()
    result = VisionGateway(
        config=_config("maximum"), cache_directory=tmp_path / "cache",
        provider=provider, frame_loader=_loader,
    ).analyze_pass2(source=tmp_path / "source.mp4", timeline=timeline, request=request)

    assert 3 <= len(request["frames"]) <= 7
    assert result["schema_version"] == "6B.pass2-result.1"
    assert result["candidate_id"] == "candidate-1"
    assert set(result["verification"]) == {
        "hook_visible", "action_visible", "reaction_visible", "payoff_visible", "continuity_risk", "confidence",
    }
    assert provider.calls


def test_dynamic_budget_uses_mode_duration_density_motion_and_content() -> None:
    config = _config().vision
    fast = dynamic_frame_budget(
        duration_seconds=3600, scene_density=1, motion=1, content_type="gameplay",
        processing_mode="fast", config=config,
    )
    quiet = dynamic_frame_budget(
        duration_seconds=120, scene_density=0, motion=0, content_type="podcast",
        processing_mode="standard", config=config,
    )
    dense = dynamic_frame_budget(
        duration_seconds=120, scene_density=1, motion=1, content_type="gameplay",
        processing_mode="maximum", config=config,
    )

    assert fast.dynamic_frame_limit == fast.max_calls == 0
    assert 0 < quiet.dynamic_frame_limit < dense.dynamic_frame_limit <= dense.max_frames


def test_long_source_ceiling_scales_calls_and_tokens_but_not_hard_dollars() -> None:
    config = AppConfig().vision
    short = dynamic_frame_budget(
        duration_seconds=300, scene_density=1, motion=1, content_type="gameplay",
        processing_mode="maximum", config=config,
    )
    long_rich = dynamic_frame_budget(
        duration_seconds=1800, scene_density=1, motion=1, content_type="gameplay",
        processing_mode="maximum", config=config,
    )
    long_podcast = dynamic_frame_budget(
        duration_seconds=7200, scene_density=0.4, motion=0, content_type="podcast",
        processing_mode="standard", config=config,
    )

    assert short.dynamic_frame_limit <= config.maximum_max_frames
    assert long_rich.configured_frame_limit == config.maximum_max_frames
    assert long_rich.dynamic_frame_limit > config.maximum_max_frames
    assert long_rich.max_calls > config.maximum_max_calls
    assert long_rich.max_tokens > config.maximum_max_tokens
    assert long_rich.max_estimated_cost == config.maximum_max_estimated_cost
    assert long_rich.limit_reason == "duration_content_ceiling_reached"
    assert long_podcast.dynamic_frame_limit > config.standard_max_frames
    assert long_podcast.max_estimated_cost == config.standard_max_estimated_cost

    app_config = AppConfig()
    controller = CostController(long_rich, app_config)
    remaining = long_rich.dynamic_frame_limit
    while remaining:
        batch = min(app_config.vision.pass1_batch_size, remaining)
        assert controller.reserve(batch, "low") is not None
        remaining -= batch
    assert controller.estimated_cost <= long_rich.max_estimated_cost
    assert controller.calls <= long_rich.max_calls
    assert controller.reserved_input_tokens + controller.reserved_output_tokens <= long_rich.max_tokens


def test_explicit_zero_budget_still_disables_long_sources() -> None:
    config = AppConfig().vision
    config.standard_max_calls = 0
    budget = dynamic_frame_budget(
        duration_seconds=7200, scene_density=1, motion=1, content_type="gameplay",
        processing_mode="standard", config=config,
    )

    assert budget.disabled
    assert budget.dynamic_frame_limit == 0
    assert budget.limit_reason == "configured_budget_disabled"
