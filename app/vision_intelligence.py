"""Budgeted external vision observations for sparse 6A.1 timeline keyframes.

The local multimodal timeline remains the source of frame selection.  This
module adds a companion artifact; it does not mutate candidate ranking or the
renderer.  Every paid request is admitted by :class:`CostController` and every
provider result is schema-validated before it can be cached or persisted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from app.config import AppConfig, VisionConfig
from app.multimodal_evidence import validate_multimodal_timeline
from app.utils import read_json, stable_text_hash, write_json


VISION_OBSERVATION_SCHEMA_VERSION = "6B.1"
VISION_ARTIFACT_SCHEMA_VERSION = "6B.pass-result.1"
VISION_PASS2_REQUEST_SCHEMA_VERSION = "6B.pass2-request.1"
VISION_PASS2_RESULT_SCHEMA_VERSION = "6B.pass2-result.1"

SCENE_TYPES = frozenset({
    "TALKING_HEAD", "INTERVIEW_SINGLE", "INTERVIEW_MULTI", "PODCAST",
    "PRODUCT_DEMO", "HANDS_ON_DEMO", "PRESENTATION_SCREEN", "GAMEPLAY",
    "CINEMATIC_SCENE", "FULL_BODY_ACTION", "UNKNOWN",
})
PRIMARY_SUBJECTS = frozenset({
    "face", "person", "group", "object", "screen", "scene", "none",
})
ACTIONS = frozenset({"none", "speaking", "gesture", "movement", "demonstration", "interaction", "unknown"})
REACTIONS = frozenset({"none", "positive", "negative", "surprise", "laughter", "attention_shift", "unknown"})
PAYOFF_SIGNALS = frozenset({"none", "setup", "reveal", "result", "resolution", "unknown"})
COMPOSITION_RISKS = frozenset({"none", "face_edge", "target_missing", "crowded", "text_overlap", "motion_blur", "unknown"})
MISSING_EVIDENCE = frozenset({"subject", "action", "reaction", "payoff", "text", "composition", "frame_unavailable"})

_PROVIDER_OBSERVATION_FIELDS = frozenset({
    "keyframe_id", "timestamp", "scene_type", "primary_subject",
    "normalized_center_x", "normalized_center_y", "visible_face_count",
    "action", "reaction", "payoff_signal", "on_screen_text",
    "composition_risk", "confidence", "missing_evidence",
})

VISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "keyframe_id": {"type": "string", "minLength": 1},
                    "timestamp": {"type": "number", "minimum": 0},
                    "scene_type": {"type": "string", "enum": sorted(SCENE_TYPES)},
                    "primary_subject": {"type": "string", "enum": sorted(PRIMARY_SUBJECTS)},
                    "normalized_center_x": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "normalized_center_y": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "visible_face_count": {"type": "integer", "minimum": 0, "maximum": 32},
                    "action": {"type": "string", "enum": sorted(ACTIONS)},
                    "reaction": {"type": "string", "enum": sorted(REACTIONS)},
                    "payoff_signal": {"type": "string", "enum": sorted(PAYOFF_SIGNALS)},
                    "on_screen_text": {"type": "string", "maxLength": 240},
                    "composition_risk": {"type": "string", "enum": sorted(COMPOSITION_RISKS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "missing_evidence": {
                        "type": "array", "items": {"type": "string", "enum": sorted(MISSING_EVIDENCE)},
                    },
                },
                "required": sorted(_PROVIDER_OBSERVATION_FIELDS),
            },
        },
    },
    "required": ["observations"],
}


class VisionContractError(ValueError):
    """A request, response, cache entry, or persisted artifact is invalid."""


class VisionProviderCallError(RuntimeError):
    """A provider responded, but its payload could not be consumed safely."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


class VisionProvider(Protocol):
    name: str

    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: Literal["low", "high"],
        pass_kind: Literal["pass1", "pass2"],
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform exactly one provider request; retry admission belongs to the gateway."""


FrameLoader = Callable[[Path, float, int], bytes | None]


@dataclass(frozen=True, slots=True)
class VisionBudget:
    mode: str
    max_frames: int
    max_calls: int
    max_tokens: int
    max_estimated_cost: float
    dynamic_frame_limit: int
    configured_frame_limit: int = 0
    limit_reason: str = "configured_frame_limit"

    @property
    def disabled(self) -> bool:
        return (
            self.mode == "fast" or self.max_frames <= 0 or self.max_calls <= 0
            or self.max_tokens <= 0 or self.max_estimated_cost <= 0
            or self.dynamic_frame_limit <= 0
        )


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    frames: int
    input_tokens: int
    output_tokens: int
    estimated_cost: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostController:
    """Single admission point for all 6B provider calls.

    A reservation is charged before I/O.  Failed or ambiguous provider calls do
    not release it because they may still be billable.  This prevents retries or
    alternate visual callers from stepping around the configured hard ceiling.
    """

    def __init__(self, budget: VisionBudget, config: AppConfig) -> None:
        self.budget = budget
        self.config = config
        self.calls = 0
        self.frames = 0
        self.reserved_input_tokens = 0
        self.reserved_output_tokens = 0
        self.actual_input_tokens = 0
        self.actual_cached_input_tokens = 0
        self.actual_cache_write_input_tokens = 0
        self.actual_output_tokens = 0
        self.estimated_cost = 0.0
        self.stop_reason: str | None = None

    def reserve(self, frame_count: int, detail: Literal["low", "high"]) -> BudgetReservation | None:
        vision = self.config.vision
        per_frame = (
            vision.low_detail_input_tokens_per_frame
            if detail == "low" else vision.high_detail_input_tokens_per_frame
        )
        input_tokens = vision.prompt_input_tokens + frame_count * per_frame
        remaining_tokens = self.budget.max_tokens - (
            self.reserved_input_tokens + self.reserved_output_tokens
        )
        output_tokens = min(vision.max_output_tokens_per_call, remaining_tokens - input_tokens)
        if frame_count <= 0 or output_tokens <= 0:
            self.stop_reason = "token_budget_exhausted"
            return None
        price_in = self.config.ai.input_token_price
        price_out = self.config.ai.output_token_price
        if price_in is None or price_out is None:
            self.stop_reason = "pricing_unavailable"
            return None
        cost = input_tokens * float(price_in) + output_tokens * float(price_out)
        projected_frames = self.frames + frame_count
        projected_calls = self.calls + 1
        projected_tokens = self.reserved_input_tokens + self.reserved_output_tokens + input_tokens + output_tokens
        projected_cost = self.estimated_cost + cost
        if projected_frames > min(self.budget.max_frames, self.budget.dynamic_frame_limit):
            self.stop_reason = "frame_budget_exhausted"
            return None
        if projected_calls > self.budget.max_calls:
            self.stop_reason = "call_budget_exhausted"
            return None
        if projected_tokens > self.budget.max_tokens:
            self.stop_reason = "token_budget_exhausted"
            return None
        if projected_cost > self.budget.max_estimated_cost + 1e-12:
            self.stop_reason = "cost_budget_exhausted"
            return None
        reservation = BudgetReservation(frame_count, input_tokens, output_tokens, cost)
        self.frames = projected_frames
        self.calls = projected_calls
        self.reserved_input_tokens += input_tokens
        self.reserved_output_tokens += output_tokens
        self.estimated_cost = projected_cost
        return reservation

    def record_usage(self, usage: dict[str, Any]) -> None:
        self.actual_input_tokens += _safe_int(usage.get("input_tokens"))
        self.actual_cached_input_tokens += _safe_int(usage.get("cached_input_tokens"))
        self.actual_cache_write_input_tokens += _safe_int(usage.get("cache_write_input_tokens"))
        self.actual_output_tokens += _safe_int(usage.get("output_tokens"))

    def diagnostics(self) -> dict[str, Any]:
        actual_cost = 0.0
        if self.config.ai.input_token_price is not None and self.config.ai.output_token_price is not None:
            actual_cost = (
                self.actual_input_tokens * float(self.config.ai.input_token_price)
                + self.actual_output_tokens * float(self.config.ai.output_token_price)
            )
        return {
            "frames": self.frames,
            "calls": self.calls,
            "reserved_input_tokens": self.reserved_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "reserved_total_tokens": self.reserved_input_tokens + self.reserved_output_tokens,
            "input_tokens": self.actual_input_tokens,
            "cached_input_tokens": self.actual_cached_input_tokens,
            "cache_write_input_tokens": self.actual_cache_write_input_tokens,
            "output_tokens": self.actual_output_tokens,
            "total_tokens": self.actual_input_tokens + self.actual_output_tokens,
            "estimated_cost": round(actual_cost, 8),
            "hard_budget_consumed_estimated_cost": round(self.estimated_cost, 8),
            "stop_reason": self.stop_reason,
        }


def _reservation_projection(
    frame_count: int,
    *,
    batch_size: int,
    prompt_tokens: int,
    input_tokens_per_frame: int,
    output_tokens_per_call: int,
    input_token_price: float | None,
    output_token_price: float | None,
) -> dict[str, Any]:
    """Project the same conservative reservations used before provider I/O."""

    frames = max(0, int(frame_count))
    size = max(1, int(batch_size))
    calls = int(math.ceil(frames / size)) if frames else 0
    input_tokens = 0
    for offset in range(0, frames, size):
        input_tokens += int(prompt_tokens) + min(size, frames - offset) * int(input_tokens_per_frame)
    output_tokens = calls * int(output_tokens_per_call)
    estimated_cost: float | None = None
    if input_token_price is not None and output_token_price is not None:
        estimated_cost = round(
            input_tokens * float(input_token_price) + output_tokens * float(output_token_price), 8,
        )
    return {
        "frames": frames,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": estimated_cost,
    }


def dynamic_frame_budget(
    *,
    duration_seconds: float,
    scene_density: float,
    motion: float,
    content_type: str,
    processing_mode: str,
    config: VisionConfig,
) -> VisionBudget:
    """Resolve a sparse frame allowance from source evidence and product mode."""

    if processing_mode == "fast":
        return VisionBudget("fast", 0, 0, 0, 0.0, 0, 0, "fast_mode_zero_calls")
    if processing_mode == "maximum":
        hard = (
            config.maximum_max_frames, config.maximum_max_calls,
            config.maximum_max_tokens, config.maximum_max_estimated_cost,
        )
        frames_per_minute = 4.5
    else:
        hard = (
            config.standard_max_frames, config.standard_max_calls,
            config.standard_max_tokens, config.standard_max_estimated_cost,
        )
        frames_per_minute = 2.5
        processing_mode = "standard"
    if any(float(value) <= 0 for value in hard):
        return VisionBudget(
            processing_mode,
            int(hard[0]),
            int(hard[1]),
            int(hard[2]),
            float(hard[3]),
            0,
            int(hard[0]),
            "configured_budget_disabled",
        )
    minutes = max(0.25, max(0.0, duration_seconds) / 60.0)
    scene_factor = 0.75 + 0.75 * _clamp(scene_density)
    motion_factor = 0.75 + 0.75 * _clamp(motion)
    content_factor = {
        "podcast": 0.72,
        "interview": 0.82,
        "lecture": 0.82,
        "screen_demo": 1.15,
        "product_demo": 1.25,
        "gameplay": 1.35,
        "cinematic": 1.35,
    }.get(content_type.casefold(), 1.0)
    desired = max(2, int(math.ceil(minutes * frames_per_minute * scene_factor * motion_factor * content_factor)))

    # 12/32 are useful short-source defaults, not universal ceilings.  Long
    # sources need more independent samples, especially formats where visual
    # state changes carry editorial meaning.  The multiplier is deliberately
    # bounded at 2x so the plan remains predictable before any provider I/O.
    duration_multiplier = min(2.0, max(1.0, math.sqrt(minutes / 10.0)))
    visual_bonus = {
        "screen_demo": 0.10,
        "product_demo": 0.15,
        "gameplay": 0.25,
        "cinematic": 0.25,
    }.get(content_type.casefold(), 0.0)
    long_source_weight = _clamp((minutes - 10.0) / 20.0)
    ceiling_multiplier = min(2.0, duration_multiplier + visual_bonus * long_source_weight)
    configured_frames = int(hard[0])
    content_ceiling = max(configured_frames, int(math.ceil(configured_frames * ceiling_multiplier)))
    evidence_limited = min(content_ceiling, desired)

    # Calls and token reservations scale only as far as the selected frame
    # plan requires.  The configured dollar ceiling never scales: it remains
    # the final admission guard in CostController and bounds worst-case spend.
    dynamic_limit = evidence_limited
    projected = _reservation_projection(
        dynamic_limit,
        batch_size=config.pass1_batch_size,
        prompt_tokens=config.prompt_input_tokens,
        input_tokens_per_frame=config.low_detail_input_tokens_per_frame,
        output_tokens_per_call=config.max_output_tokens_per_call,
        input_token_price=None,
        output_token_price=None,
    )
    max_calls = max(int(hard[1]), int(projected["calls"]))
    max_tokens = max(int(hard[2]), int(projected["total_tokens"]))
    if desired < content_ceiling:
        limit_reason = "evidence_demand_satisfied"
    elif content_ceiling > configured_frames:
        limit_reason = "duration_content_ceiling_reached"
    else:
        limit_reason = "configured_frame_limit_reached"
    return VisionBudget(
        processing_mode,
        content_ceiling,
        max_calls,
        max_tokens,
        float(hard[3]),
        dynamic_limit,
        configured_frames,
        limit_reason,
    )


def build_pass2_request(
    *,
    candidate_id: str,
    window_start: float,
    window_end: float,
    anchors: dict[str, float | None],
    timeline: dict[str, Any],
    max_frames: int = 7,
) -> dict[str, Any]:
    """Build a deterministic 3–7-frame hook/action/reaction/payoff window."""

    validate_multimodal_timeline(timeline)
    if not candidate_id or window_start < 0 or not window_start < window_end:
        raise VisionContractError("PASS 2 candidate window is invalid.")
    if not 3 <= max_frames <= 7:
        raise VisionContractError("PASS 2 max_frames must be between 3 and 7.")
    required_anchor_names = {"hook", "action", "reaction", "payoff"}
    if set(anchors) != required_anchor_names:
        raise VisionContractError("PASS 2 anchors must contain hook/action/reaction/payoff.")
    keyframes = [
        item for item in timeline["keyframes"]
        if window_start <= float(item["time_seconds"]) <= window_end
    ]
    anchor_times = [float(value) for value in anchors.values() if value is not None]
    scored = sorted(
        keyframes,
        key=lambda item: (
            min((abs(float(item["time_seconds"]) - value) for value in anchor_times), default=0.0),
            -float(item.get("relevance_score", 0.0)),
            float(item["time_seconds"]),
        ),
    )
    chosen = scored[:max_frames]
    if len(chosen) < 3:
        raise VisionContractError("PASS 2 requires at least three timeline keyframes in the candidate window.")
    chosen.sort(key=lambda item: float(item["time_seconds"]))
    request = {
        "schema_version": VISION_PASS2_REQUEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "analysis_run_id": timeline["analysis_run_id"],
        "window": {"start_seconds": round(window_start, 3), "end_seconds": round(window_end, 3)},
        "anchors": {name: (round(float(value), 3) if value is not None else None) for name, value in anchors.items()},
        "frames": [
            {"keyframe_id": item["keyframe_id"], "timestamp": item["time_seconds"]}
            for item in chosen
        ],
    }
    return validate_pass2_request(request, timeline)


def build_candidate_bounded_pass2_timeline(
    timeline: dict[str, Any],
    *,
    candidate_id: str,
    window_start: float,
    window_end: float,
    anchors: dict[str, float | None],
    max_frames: int = 7,
) -> dict[str, Any]:
    """Return a runtime-only timeline with enough in-range frames for PASS 2.

    The source analysis timeline is intentionally never modified. Long sparse
    sources may have no globally selected keyframe inside a later Draft
    candidate even though PASS 2 can safely inspect that bounded range. This
    adapter adds deterministic candidate-owned frame references to a deep copy
    and keeps the original analysis identity for lineage validation.
    """

    validate_multimodal_timeline(timeline)
    if not candidate_id or window_start < 0 or not window_start < window_end:
        raise VisionContractError("Candidate-bounded PASS 2 window is invalid.")
    if not 3 <= max_frames <= 7:
        raise VisionContractError("Candidate-bounded PASS 2 max_frames must be between 3 and 7.")
    if set(anchors) != {"hook", "action", "reaction", "payoff"}:
        raise VisionContractError("Candidate-bounded PASS 2 anchors are invalid.")
    duration = float(timeline["source_duration_seconds"])
    if window_end > duration + 0.001:
        raise VisionContractError("Candidate-bounded PASS 2 window exceeds the source duration.")

    in_range = [
        item for item in timeline["keyframes"]
        if window_start <= float(item["time_seconds"]) <= window_end
    ]
    if len(in_range) >= 3:
        return timeline

    runtime = json.loads(json.dumps(timeline))
    span = window_end - window_start
    proposed = [
        float(value) for value in anchors.values()
        if value is not None and window_start <= float(value) <= window_end
    ]
    proposed.extend(
        window_start + span * fraction
        for fraction in (0.08, 0.24, 0.42, 0.60, 0.78, 0.92)
    )
    existing_times = {round(float(item["time_seconds"]), 3) for item in in_range}
    added: list[dict[str, Any]] = []
    for raw_timestamp in proposed:
        timestamp = round(min(window_end, max(window_start, raw_timestamp)), 3)
        if timestamp in existing_times:
            continue
        existing_times.add(timestamp)
        identity = stable_text_hash(f"{candidate_id}|{timestamp:.3f}")[:16]
        added.append({
            "keyframe_id": f"candidate-keyframe-{identity}",
            "time_seconds": timestamp,
            "selection_reasons": ["draft_candidate_composition_gap"],
            "relevance_score": 1.0,
            "confidence": 1.0,
            "analysis_status": "candidate_bounded_pass2_planned",
            "future_vision_api_eligible": True,
            "evidence_refs": [],
            "provenance": [{
                "artifact": "candidate-composition-pass2.json",
                "locator": f"candidate/{candidate_id}/source-range",
                "method": "deterministic_candidate_bounded_frame_plan",
            }],
        })
        if len(in_range) + len(added) >= max_frames:
            break
    if len(in_range) + len(added) < 3:
        raise VisionContractError("Candidate-bounded PASS 2 could not plan three distinct frames.")

    runtime["keyframes"] = sorted(
        [*runtime["keyframes"], *added],
        key=lambda item: (float(item["time_seconds"]), str(item["keyframe_id"])),
    )
    diagnostics = runtime["diagnostics"]["keyframes"]
    diagnostics["count"] = len(runtime["keyframes"])
    diagnostics["limit"] = max(int(diagnostics.get("limit") or 0), len(runtime["keyframes"]))
    diagnostics["analyzed_count"] = sum(
        item.get("analysis_status") == "existing_visual_evidence"
        for item in runtime["keyframes"]
    )
    diagnostics["future_vision_api_eligible_count"] = sum(
        bool(item.get("future_vision_api_eligible")) for item in runtime["keyframes"]
    )
    diagnostics["candidate_bounded_pass2_added_count"] = len(added)
    return validate_multimodal_timeline(
        runtime,
        expected_source_id=str(timeline["source_id"]),
        expected_analysis_run_id=str(timeline["analysis_run_id"]),
    )


def validate_pass2_request(data: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    validate_multimodal_timeline(timeline)
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "candidate_id", "analysis_run_id", "window", "anchors", "frames",
    }:
        raise VisionContractError("PASS 2 request has an invalid top-level schema.")
    if data["schema_version"] != VISION_PASS2_REQUEST_SCHEMA_VERSION:
        raise VisionContractError("Unsupported PASS 2 request schema.")
    if data["analysis_run_id"] != timeline["analysis_run_id"] or not str(data["candidate_id"]):
        raise VisionContractError("PASS 2 request identity mismatch.")
    window = data["window"]
    anchors = data["anchors"]
    frames = data["frames"]
    if not isinstance(window, dict) or set(window) != {"start_seconds", "end_seconds"}:
        raise VisionContractError("PASS 2 window is invalid.")
    start, end = _number(window.get("start_seconds")), _number(window.get("end_seconds"))
    if start is None or end is None or start < 0 or not start < end:
        raise VisionContractError("PASS 2 window bounds are invalid.")
    if not isinstance(anchors, dict) or set(anchors) != {"hook", "action", "reaction", "payoff"}:
        raise VisionContractError("PASS 2 anchors are invalid.")
    for value in anchors.values():
        parsed = _number(value) if value is not None else None
        if value is not None and (parsed is None or not start <= parsed <= end):
            raise VisionContractError("PASS 2 anchor timestamp is outside the candidate window.")
    if not isinstance(frames, list) or not 3 <= len(frames) <= 7:
        raise VisionContractError("PASS 2 must contain three to seven frames.")
    timeline_frames = {item["keyframe_id"]: float(item["time_seconds"]) for item in timeline["keyframes"]}
    seen: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"keyframe_id", "timestamp"}:
            raise VisionContractError("PASS 2 frame reference is invalid.")
        keyframe_id = str(frame["keyframe_id"])
        timestamp = _number(frame["timestamp"])
        if (
            keyframe_id in seen or keyframe_id not in timeline_frames or timestamp is None
            or abs(timestamp - timeline_frames[keyframe_id]) > 0.001 or not start <= timestamp <= end
        ):
            raise VisionContractError("PASS 2 frame does not match the timeline window.")
        seen.add(keyframe_id)
    return data


class VisionGateway:
    """Extract, cache, budget, call, validate, and diagnose all 6B vision work."""

    def __init__(
        self,
        *,
        config: AppConfig,
        cache_directory: Path,
        provider: VisionProvider | None,
        frame_loader: FrameLoader | None = None,
    ) -> None:
        self.config = config
        self.cache_directory = cache_directory
        self.provider = provider
        self.frame_loader = frame_loader or extract_jpeg_frame

    def analyze_pass1(
        self,
        *,
        source: Path,
        timeline: dict[str, Any],
        content_type: str,
    ) -> dict[str, Any]:
        validate_multimodal_timeline(timeline)
        budget = _budget_for_timeline(timeline, content_type, self.config)
        selected = _select_pass1_frames(timeline, budget.dynamic_frame_limit)
        return self._analyze(
            source=source, timeline=timeline, frame_refs=selected, budget=budget,
            detail="low", pass_kind="pass1", prompt_version=self.config.vision.prompt_version,
            candidate_id=None,
        )

    def analyze_pass2(
        self,
        *,
        source: Path,
        timeline: dict[str, Any],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        request = validate_pass2_request(request, timeline)
        base = _budget_for_timeline(timeline, "unknown", self.config)
        frame_limit = min(
            max(base.dynamic_frame_limit, self.config.vision.pass2_min_frames),
            self.config.vision.pass2_max_frames,
            base.max_frames,
        )
        budget = VisionBudget(
            base.mode,
            base.max_frames,
            base.max_calls,
            base.max_tokens,
            base.max_estimated_cost,
            frame_limit,
            base.configured_frame_limit,
            base.limit_reason,
        )
        artifact = self._analyze(
            source=source, timeline=timeline, frame_refs=list(request["frames"]), budget=budget,
            detail="high", pass_kind="pass2", prompt_version=self.config.vision.pass2_prompt_version,
            candidate_id=str(request["candidate_id"]),
        )
        observations = artifact["observations"]
        result = {
            "schema_version": VISION_PASS2_RESULT_SCHEMA_VERSION,
            "candidate_id": request["candidate_id"],
            "analysis_run_id": timeline["analysis_run_id"],
            "request": request,
            "status": artifact["status"],
            "verification": {
                "hook_visible": _anchor_visible(request, observations, "hook", {"setup", "reveal"}),
                "action_visible": any(item["action"] not in {"none", "unknown"} for item in observations),
                "reaction_visible": any(item["reaction"] not in {"none", "unknown"} for item in observations),
                "payoff_visible": _anchor_visible(request, observations, "payoff", {"reveal", "result", "resolution"}),
                "continuity_risk": _continuity_risk(observations),
                "confidence": round(_average_confidence(observations), 6),
            },
            "observations": observations,
            "diagnostics": artifact["diagnostics"],
        }
        return validate_pass2_result(result, timeline=timeline, request=request)

    def _analyze(
        self,
        *,
        source: Path,
        timeline: dict[str, Any],
        frame_refs: list[dict[str, Any]],
        budget: VisionBudget,
        detail: Literal["low", "high"],
        pass_kind: Literal["pass1", "pass2"],
        prompt_version: str,
        candidate_id: str | None,
    ) -> dict[str, Any]:
        controller = CostController(budget, self.config)
        model = self.config.ai.model
        diagnostics: dict[str, Any] = {
            "pass": pass_kind,
            "processing_mode": self.config.product_flow.processing_mode,
            "budget": {
                "dynamic_frame_limit": budget.dynamic_frame_limit,
                "max_frames": budget.max_frames,
                "configured_frame_limit": budget.configured_frame_limit,
                "max_calls": budget.max_calls,
                "max_tokens": budget.max_tokens,
                "max_estimated_cost": budget.max_estimated_cost,
                "limit_reason": budget.limit_reason,
            },
            "keyframes_found": [
                {"keyframe_id": str(item["keyframe_id"]), "timestamp": float(item["time_seconds"])}
                for item in timeline["keyframes"]
            ],
            "selected_keyframes": [
                {
                    "keyframe_id": str(item["keyframe_id"]),
                    "timestamp": float(item.get("timestamp", item.get("time_seconds", 0.0))),
                }
                for item in frame_refs
            ],
            "frames_requested": len(frame_refs),
            "frames_extracted": 0,
            "frames_sent": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "missing_evidence": [],
            "failure_reason": None,
            "analysis_stop_reason": _selection_stop_reason(timeline, frame_refs, budget, pass_kind),
        }
        batch_size = self.config.vision.pass1_batch_size if pass_kind == "pass1" else min(3, self.config.vision.pass2_max_frames)
        projected_usage = _reservation_projection(
            len(frame_refs),
            batch_size=batch_size,
            prompt_tokens=self.config.vision.prompt_input_tokens,
            input_tokens_per_frame=(
                self.config.vision.low_detail_input_tokens_per_frame
                if detail == "low" else self.config.vision.high_detail_input_tokens_per_frame
            ),
            output_tokens_per_call=self.config.vision.max_output_tokens_per_call,
            input_token_price=self.config.ai.input_token_price,
            output_token_price=self.config.ai.output_token_price,
        )
        projected_cost = projected_usage["estimated_cost"]
        projected_usage["within_hard_budget"] = (
            projected_cost is not None
            and float(projected_cost) <= budget.max_estimated_cost + 1e-12
            and int(projected_usage["calls"]) <= budget.max_calls
            and int(projected_usage["total_tokens"]) <= budget.max_tokens
            and int(projected_usage["frames"]) <= min(budget.max_frames, budget.dynamic_frame_limit)
        )
        diagnostics["projected_uncached_usage"] = projected_usage
        if not self.config.vision.enabled or not self.config.optional_visual_features:
            return _empty_artifact(timeline, pass_kind, candidate_id, "skipped", "vision_not_opted_in", diagnostics, controller)
        if budget.disabled:
            reason = "fast_mode_zero_calls" if budget.mode == "fast" else "vision_budget_disabled"
            return _empty_artifact(timeline, pass_kind, candidate_id, "skipped", reason, diagnostics, controller)
        if self.provider is None:
            return self._fallback_artifact(
                timeline, frame_refs, pass_kind, candidate_id, "provider_unavailable", diagnostics, controller,
            )
        prepared: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        for ref in frame_refs[:budget.dynamic_frame_limit]:
            keyframe_id = str(ref["keyframe_id"])
            timestamp = float(ref.get("timestamp", ref.get("time_seconds", 0.0)))
            image = self.frame_loader(source, timestamp, self.config.vision.frame_width)
            if not image:
                diagnostics["missing_evidence"].append({"keyframe_id": keyframe_id, "reason": "frame_extraction_failed"})
                observations.append(_local_fallback_observation(timeline, keyframe_id, timestamp, "frame_unavailable"))
                continue
            diagnostics["frames_extracted"] += 1
            frame_hash = hashlib.sha256(image).hexdigest()
            cache_key = _vision_cache_key(frame_hash, model, detail, prompt_version, self.config.vision.schema_version)
            cached = self._cache_read(cache_key, keyframe_id, timestamp)
            if cached is not None:
                diagnostics["cache_hits"] += 1
                observations.append(_persisted_observation(
                    cached, origin="cache", provider=str(cached["_cache_provider"]), model=model,
                    detail=detail, prompt_version=prompt_version, frame_hash=frame_hash,
                    cache_key=cache_key, request_id=None,
                ))
                continue
            diagnostics["cache_misses"] += 1
            prepared.append({
                "keyframe_id": keyframe_id,
                "timestamp": timestamp,
                "image_base64": base64.b64encode(image).decode("ascii"),
                "frame_hash": frame_hash,
                "cache_key": cache_key,
            })

        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset:offset + batch_size]
            reservation = controller.reserve(len(batch), detail)
            if reservation is None:
                diagnostics["failure_reason"] = controller.stop_reason
                diagnostics["analysis_stop_reason"] = controller.stop_reason
                for frame in batch:
                    observations.append(_local_fallback_observation(
                        timeline, str(frame["keyframe_id"]), float(frame["timestamp"]), controller.stop_reason or "budget_exhausted",
                    ))
                for frame in prepared[offset + len(batch):]:
                    observations.append(_local_fallback_observation(
                        timeline, str(frame["keyframe_id"]), float(frame["timestamp"]), controller.stop_reason or "budget_exhausted",
                    ))
                break
            diagnostics["frames_sent"] += len(batch)
            try:
                payload, usage = self.provider.analyze_vision(
                    batch, detail=detail, pass_kind=pass_kind,
                    max_output_tokens=reservation.output_tokens,
                )
                controller.record_usage(usage)
                validated = validate_provider_response(payload, batch)
                request_id = str(usage.get("request_id") or "") or None
                for core, frame in zip(validated, batch, strict=True):
                    if self.config.vision.cache_enabled:
                        self._cache_write(str(frame["cache_key"]), core, self.provider.name)
                    observations.append(_persisted_observation(
                        core, origin="provider", provider=self.provider.name, model=model,
                        detail=detail, prompt_version=prompt_version,
                        frame_hash=str(frame["frame_hash"]), cache_key=str(frame["cache_key"]),
                        request_id=request_id,
                    ))
            except Exception as error:
                if isinstance(error, VisionProviderCallError):
                    controller.record_usage(error.usage)
                from app.ai import sanitize_api_error

                secret = getattr(self.provider, "api_key", None)
                diagnostics["failure_reason"] = f"provider_failure:{sanitize_api_error(error, secret)}"
                diagnostics["analysis_stop_reason"] = "provider_failure"
                for frame in batch:
                    observations.append(_local_fallback_observation(
                        timeline, str(frame["keyframe_id"]), float(frame["timestamp"]), "provider_failure",
                    ))
                for frame in prepared[offset + len(batch):]:
                    observations.append(_local_fallback_observation(
                        timeline, str(frame["keyframe_id"]), float(frame["timestamp"]), "provider_failure",
                    ))
                break

        observations.sort(key=lambda item: (float(item["timestamp"]), str(item["keyframe_id"])))
        recorded_missing = {
            (str(item.get("keyframe_id")), str(item.get("reason")))
            for item in diagnostics["missing_evidence"] if isinstance(item, dict)
        }
        for observation in observations:
            for code in observation["missing_evidence"]:
                pair = (str(observation["keyframe_id"]), str(code))
                if pair not in recorded_missing:
                    diagnostics["missing_evidence"].append({"keyframe_id": pair[0], "reason": pair[1]})
                    recorded_missing.add(pair)
        status = "completed"
        if any(item["origin"] == "local_fallback" for item in observations):
            status = "partial" if any(item["origin"] != "local_fallback" for item in observations) else "fallback"
        diagnostics["usage"] = controller.diagnostics()
        if diagnostics["cache_hits"] == len(frame_refs) and frame_refs:
            diagnostics["analysis_stop_reason"] = "cache_satisfied_selected_frames"
        diagnostics["cache_savings_estimated_cost"] = round(
            _estimated_cache_savings(diagnostics["cache_hits"], detail, self.config), 8,
        )
        artifact = {
            "schema_version": VISION_ARTIFACT_SCHEMA_VERSION,
            "pass": pass_kind,
            "status": status,
            "source_id": timeline["source_id"],
            "analysis_run_id": timeline["analysis_run_id"],
            "candidate_id": candidate_id,
            "observations": observations,
            "diagnostics": diagnostics,
            "provenance": {
                "provider": self.provider.name,
                "model": model,
                "detail": detail,
                "prompt_version": prompt_version,
                "schema_version": self.config.vision.schema_version,
                "timeline_schema_version": timeline["schema_version"],
            },
        }
        return validate_vision_artifact(artifact, timeline)

    def _fallback_artifact(
        self,
        timeline: dict[str, Any],
        frame_refs: list[dict[str, Any]],
        pass_kind: Literal["pass1", "pass2"],
        candidate_id: str | None,
        reason: str,
        diagnostics: dict[str, Any],
        controller: CostController,
    ) -> dict[str, Any]:
        diagnostics["failure_reason"] = reason
        diagnostics["analysis_stop_reason"] = reason
        observations = [
            _local_fallback_observation(
                timeline, str(item["keyframe_id"]),
                float(item.get("timestamp", item.get("time_seconds", 0.0))), reason,
            )
            for item in frame_refs
        ]
        diagnostics["usage"] = controller.diagnostics()
        diagnostics["cache_savings_estimated_cost"] = 0.0
        artifact = {
            "schema_version": VISION_ARTIFACT_SCHEMA_VERSION,
            "pass": pass_kind,
            "status": "fallback",
            "source_id": timeline["source_id"],
            "analysis_run_id": timeline["analysis_run_id"],
            "candidate_id": candidate_id,
            "observations": observations,
            "diagnostics": diagnostics,
            "provenance": {
                "provider": "local",
                "model": self.config.ai.model,
                "detail": "low" if pass_kind == "pass1" else "high",
                "prompt_version": self.config.vision.prompt_version if pass_kind == "pass1" else self.config.vision.pass2_prompt_version,
                "schema_version": self.config.vision.schema_version,
                "timeline_schema_version": timeline["schema_version"],
            },
        }
        return validate_vision_artifact(artifact, timeline)

    def _cache_read(self, key: str, keyframe_id: str, timestamp: float) -> dict[str, Any] | None:
        if not self.config.vision.cache_enabled:
            return None
        data = read_json(self.cache_directory / f"{key}.json", None)
        if not isinstance(data, dict) or set(data) != {"cache_key", "provider", "observation"} or data.get("cache_key") != key:
            return None
        try:
            core = validate_provider_observation(data["observation"], keyframe_id, timestamp)
        except VisionContractError:
            return None
        return {**core, "_cache_provider": str(data["provider"])}

    def _cache_write(self, key: str, core: dict[str, Any], provider: str) -> None:
        write_json(self.cache_directory / f"{key}.json", {
            "cache_key": key, "provider": provider, "observation": core,
        })


def validate_provider_response(value: Any, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"observations"} or not isinstance(value["observations"], list):
        raise VisionContractError("Vision response must contain only an observations array.")
    if len(value["observations"]) != len(frames):
        raise VisionContractError("Vision response must contain exactly one observation per frame.")
    by_id: dict[str, dict[str, Any]] = {}
    expected = {str(item["keyframe_id"]): float(item["timestamp"]) for item in frames}
    for item in value["observations"]:
        keyframe_id = str(item.get("keyframe_id") or "") if isinstance(item, dict) else ""
        if keyframe_id not in expected or keyframe_id in by_id:
            raise VisionContractError("Vision response contains an unknown or duplicate keyframe_id.")
        by_id[keyframe_id] = validate_provider_observation(item, keyframe_id, expected[keyframe_id])
    return [by_id[str(frame["keyframe_id"])] for frame in frames]


def validate_provider_observation(value: Any, keyframe_id: str, timestamp: float) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PROVIDER_OBSERVATION_FIELDS:
        raise VisionContractError("Vision observation does not satisfy the strict schema.")
    if value["keyframe_id"] != keyframe_id:
        raise VisionContractError("Vision observation keyframe identity mismatch.")
    observed_time = _number(value["timestamp"])
    if observed_time is None or abs(observed_time - timestamp) > 0.05:
        raise VisionContractError("Vision observation timestamp mismatch.")
    if value["scene_type"] not in SCENE_TYPES or value["primary_subject"] not in PRIMARY_SUBJECTS:
        raise VisionContractError("Vision observation uses an invalid subject classification.")
    if value["action"] not in ACTIONS or value["reaction"] not in REACTIONS or value["payoff_signal"] not in PAYOFF_SIGNALS:
        raise VisionContractError("Vision observation uses an invalid event classification.")
    if value["composition_risk"] not in COMPOSITION_RISKS:
        raise VisionContractError("Vision observation uses an invalid composition risk.")
    for coordinate in (value["normalized_center_x"], value["normalized_center_y"]):
        if coordinate is not None and (_number(coordinate) is None or not 0 <= float(coordinate) <= 1):
            raise VisionContractError("Vision observation has an invalid normalized coordinate.")
    faces = value["visible_face_count"]
    confidence = _number(value["confidence"])
    missing = value["missing_evidence"]
    if isinstance(faces, bool) or not isinstance(faces, int) or not 0 <= faces <= 32:
        raise VisionContractError("Vision observation has an invalid face count.")
    if confidence is None or not 0 <= confidence <= 1:
        raise VisionContractError("Vision observation has invalid confidence.")
    if not isinstance(value["on_screen_text"], str) or len(value["on_screen_text"]) > 240:
        raise VisionContractError("Vision observation has invalid on-screen text.")
    if not isinstance(missing, list) or len(set(missing)) != len(missing) or any(item not in MISSING_EVIDENCE for item in missing):
        raise VisionContractError("Vision observation has invalid missing-evidence codes.")
    return {
        **value,
        "timestamp": round(float(observed_time), 3),
        "normalized_center_x": None if value["normalized_center_x"] is None else float(value["normalized_center_x"]),
        "normalized_center_y": None if value["normalized_center_y"] is None else float(value["normalized_center_y"]),
        "confidence": round(float(confidence), 6),
        "missing_evidence": sorted(missing),
    }


def validate_vision_artifact(data: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version", "pass", "status", "source_id", "analysis_run_id",
        "candidate_id", "observations", "diagnostics", "provenance",
    }
    if not isinstance(data, dict) or set(data) != fields or data["schema_version"] != VISION_ARTIFACT_SCHEMA_VERSION:
        raise VisionContractError("Vision artifact has an invalid top-level schema.")
    if data["source_id"] != timeline["source_id"] or data["analysis_run_id"] != timeline["analysis_run_id"]:
        raise VisionContractError("Vision artifact identity mismatch.")
    if data["pass"] not in {"pass1", "pass2"} or data["status"] not in {"completed", "partial", "fallback", "skipped"}:
        raise VisionContractError("Vision artifact has an invalid pass or status.")
    if not isinstance(data["observations"], list) or not isinstance(data["diagnostics"], dict) or not isinstance(data["provenance"], dict):
        raise VisionContractError("Vision artifact sections are invalid.")
    timeline_frames = {item["keyframe_id"]: float(item["time_seconds"]) for item in timeline["keyframes"]}
    seen: set[str] = set()
    for item in data["observations"]:
        if not isinstance(item, dict) or item.get("keyframe_id") not in timeline_frames or item["keyframe_id"] in seen:
            raise VisionContractError("Persisted observation is not tied to a unique timeline keyframe.")
        if abs(float(item["timestamp"]) - timeline_frames[item["keyframe_id"]]) > 0.05:
            raise VisionContractError("Persisted observation timestamp mismatch.")
        _validate_persisted_observation(item)
        seen.add(item["keyframe_id"])
    return data


def validate_pass2_result(
    data: dict[str, Any],
    *,
    timeline: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "candidate_id", "analysis_run_id", "request", "status",
        "verification", "observations", "diagnostics",
    } or data["schema_version"] != VISION_PASS2_RESULT_SCHEMA_VERSION:
        raise VisionContractError("PASS 2 result has an invalid schema.")
    verification = data["verification"]
    if not isinstance(verification, dict) or set(verification) != {
        "hook_visible", "action_visible", "reaction_visible", "payoff_visible", "continuity_risk", "confidence",
    }:
        raise VisionContractError("PASS 2 verification is invalid.")
    if any(not isinstance(verification[name], bool) for name in ("hook_visible", "action_visible", "reaction_visible", "payoff_visible")):
        raise VisionContractError("PASS 2 visibility decisions must be boolean.")
    if verification["continuity_risk"] not in {"low", "medium", "high", "unknown"}:
        raise VisionContractError("PASS 2 continuity risk is invalid.")
    if not isinstance(data["observations"], list) or not isinstance(data["diagnostics"], dict):
        raise VisionContractError("PASS 2 observations or diagnostics are invalid.")
    if timeline is not None:
        validate_multimodal_timeline(timeline)
        expected_request = validate_pass2_request(request or data["request"], timeline)
        if (
            data["request"] != expected_request
            or data["candidate_id"] != expected_request["candidate_id"]
            or data["analysis_run_id"] != timeline["analysis_run_id"]
        ):
            raise VisionContractError("PASS 2 result lineage does not match its request.")
        expected_frames = {
            str(item["keyframe_id"]): float(item["timestamp"])
            for item in expected_request["frames"]
        }
        seen: set[str] = set()
        for observation in data["observations"]:
            if not isinstance(observation, dict):
                raise VisionContractError("PASS 2 observation is invalid.")
            keyframe_id = str(observation.get("keyframe_id") or "")
            timestamp = _number(observation.get("timestamp"))
            if (
                keyframe_id not in expected_frames
                or keyframe_id in seen
                or timestamp is None
                or abs(timestamp - expected_frames[keyframe_id]) > 0.05
            ):
                raise VisionContractError("PASS 2 observation does not match a requested frame.")
            _validate_persisted_observation(observation)
            seen.add(keyframe_id)
    return data


def vision_prompt(pass_kind: str) -> str:
    purpose = (
        "Perform a broad low-detail editorial observation of each independent frame."
        if pass_kind == "pass1" else
        "Perform a deep candidate-window check for visible hook, action, reaction, and payoff evidence."
    )
    composite_guidance = (
        " For a GAMEPLAY frame with a visible facecam overlay, use scene_type GAMEPLAY, "
        "primary_subject face, and coordinates at the facecam subject; do not report the "
        "gameplay canvas as the primary subject in that composite frame."
        if pass_kind == "pass2" else ""
    )
    return (
        f"{purpose} Return exactly one observation for every supplied keyframe_id and timestamp. "
        "Use only visible evidence; never identify people or infer hidden facts. Use null coordinates and explicit "
        "missing_evidence when uncertain. Keep on_screen_text brief and verbatim only when clearly legible."
        f"{composite_guidance}"
    )


def extract_jpeg_frame(source: Path, timestamp: float, width: int) -> bytes | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not source.is_file():
        return None
    try:
        result = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                "-i", str(source), "-frames:v", "1", "-vf", f"scale={width}:-2",
                "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ],
            capture_output=True, timeout=45, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bytes(result.stdout) or None


def _budget_for_timeline(timeline: dict[str, Any], content_type: str, config: AppConfig) -> VisionBudget:
    duration = float(timeline["source_duration_seconds"])
    minutes = max(duration / 60.0, 1.0)
    scene_density = _clamp(len(timeline["scenes"]) / minutes / 12.0)
    motion_frames = sum("measured_motion" in item.get("selection_reasons", []) for item in timeline["keyframes"])
    motion = _clamp(motion_frames / max(1, len(timeline["keyframes"])) * 3.0) if timeline["keyframes"] else 0.0
    return dynamic_frame_budget(
        duration_seconds=duration,
        scene_density=scene_density,
        motion=motion,
        content_type=content_type,
        processing_mode=config.product_flow.processing_mode,
        config=config.vision,
    )


def _select_pass1_frames(timeline: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    ranked = sorted(
        timeline["keyframes"],
        key=lambda item: (-float(item.get("relevance_score", 0.0)), float(item["time_seconds"])),
    )
    selected: list[dict[str, Any]] = []
    spacing = max(0.75, float(timeline["source_duration_seconds"]) / max(limit * 4, 1))
    for item in ranked:
        if any(abs(float(item["time_seconds"]) - float(other["timestamp"])) < spacing for other in selected):
            continue
        selected.append({"keyframe_id": item["keyframe_id"], "timestamp": item["time_seconds"]})
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: float(item["timestamp"]))
    return selected


def _selection_stop_reason(
    timeline: dict[str, Any],
    selected: list[dict[str, Any]],
    budget: VisionBudget,
    pass_kind: str,
) -> str:
    if budget.mode == "fast":
        return "fast_mode_zero_calls"
    if pass_kind == "pass2":
        return "candidate_window_request_satisfied" if selected else "candidate_window_has_no_keyframes"
    found = len(timeline["keyframes"])
    if found == 0:
        return "no_eligible_keyframes"
    if len(selected) >= found:
        return "all_eligible_keyframes_selected"
    if len(selected) < min(found, budget.dynamic_frame_limit):
        return "minimum_temporal_spacing_reached"
    return budget.limit_reason


def _empty_artifact(
    timeline: dict[str, Any],
    pass_kind: str,
    candidate_id: str | None,
    status: str,
    reason: str,
    diagnostics: dict[str, Any],
    controller: CostController,
) -> dict[str, Any]:
    diagnostics["failure_reason"] = reason
    diagnostics["analysis_stop_reason"] = reason
    diagnostics["usage"] = controller.diagnostics()
    diagnostics["cache_savings_estimated_cost"] = 0.0
    return {
        "schema_version": VISION_ARTIFACT_SCHEMA_VERSION,
        "pass": pass_kind,
        "status": status,
        "source_id": timeline["source_id"],
        "analysis_run_id": timeline["analysis_run_id"],
        "candidate_id": candidate_id,
        "observations": [],
        "diagnostics": diagnostics,
        "provenance": {
            "provider": "not-called",
            "model": "not-called",
            "detail": "low" if pass_kind == "pass1" else "high",
            "prompt_version": "not-called",
            "schema_version": VISION_OBSERVATION_SCHEMA_VERSION,
            "timeline_schema_version": timeline["schema_version"],
        },
    }


def _persisted_observation(
    core: dict[str, Any],
    *,
    origin: str,
    provider: str,
    model: str,
    detail: str,
    prompt_version: str,
    frame_hash: str,
    cache_key: str,
    request_id: str | None,
) -> dict[str, Any]:
    clean = {key: value for key, value in core.items() if key in _PROVIDER_OBSERVATION_FIELDS}
    return {
        "observation_schema_version": VISION_OBSERVATION_SCHEMA_VERSION,
        **clean,
        "origin": origin,
        "provenance": {
            "provider": provider,
            "model": model,
            "detail": detail,
            "prompt_version": prompt_version,
            "schema_version": VISION_OBSERVATION_SCHEMA_VERSION,
            "frame_hash": frame_hash,
            "cache_key": cache_key,
            "request_id": request_id,
        },
    }


def _validate_persisted_observation(item: dict[str, Any]) -> None:
    expected = {*_PROVIDER_OBSERVATION_FIELDS, "observation_schema_version", "origin", "provenance"}
    if set(item) != expected or item.get("observation_schema_version") != VISION_OBSERVATION_SCHEMA_VERSION:
        raise VisionContractError("Persisted vision observation has an invalid strict schema.")
    origin = item.get("origin")
    if origin not in {"provider", "cache", "local_fallback"} or not isinstance(item.get("provenance"), dict):
        raise VisionContractError("Persisted vision observation provenance is invalid.")
    validate_provider_observation(
        {key: item[key] for key in _PROVIDER_OBSERVATION_FIELDS},
        str(item["keyframe_id"]), float(item["timestamp"]),
    )
    provenance = item["provenance"]
    common = {
        "provider", "model", "detail", "prompt_version", "schema_version",
        "frame_hash", "cache_key", "request_id",
    }
    expected_provenance = common | ({"reason", "evidence_refs"} if origin == "local_fallback" else set())
    if set(provenance) != expected_provenance or any(
        not isinstance(provenance.get(name), str) for name in (
            "provider", "model", "detail", "prompt_version", "schema_version", "frame_hash", "cache_key",
        )
    ):
        raise VisionContractError("Persisted vision observation has invalid provenance fields.")


def _local_fallback_observation(
    timeline: dict[str, Any], keyframe_id: str, timestamp: float, reason: str,
) -> dict[str, Any]:
    keyframe = next((item for item in timeline["keyframes"] if item["keyframe_id"] == keyframe_id), None)
    confidence = float(keyframe.get("confidence", 0.0)) if keyframe else 0.0
    refs = list(keyframe.get("evidence_refs", [])) if keyframe else []
    core: dict[str, Any] = {
        "keyframe_id": keyframe_id,
        "timestamp": round(timestamp, 3),
        "scene_type": "UNKNOWN",
        "primary_subject": "scene" if refs else "none",
        "normalized_center_x": None,
        "normalized_center_y": None,
        "visible_face_count": 0,
        "action": "unknown",
        "reaction": "unknown",
        "payoff_signal": "unknown",
        "on_screen_text": "",
        "composition_risk": "unknown",
        "confidence": round(min(0.5, confidence), 6),
        "missing_evidence": ["action", "composition", "payoff", "reaction", "subject", "text"],
    }
    if reason == "frame_unavailable":
        core["missing_evidence"] = sorted({*core["missing_evidence"], "frame_unavailable"})
    return {
        "observation_schema_version": VISION_OBSERVATION_SCHEMA_VERSION,
        **core,
        "origin": "local_fallback",
        "provenance": {
            "provider": "local",
            "model": "none",
            "detail": "none",
            "prompt_version": "local-evidence-fallback",
            "schema_version": VISION_OBSERVATION_SCHEMA_VERSION,
            "frame_hash": "",
            "cache_key": "",
            "request_id": None,
            "reason": reason,
            "evidence_refs": refs,
        },
    }


def _vision_cache_key(
    frame_hash: str, model: str, detail: str, prompt_version: str, schema_version: str,
) -> str:
    return stable_text_hash(json.dumps({
        "frame_hash": frame_hash,
        "model": model,
        "detail": detail,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
    }, sort_keys=True, separators=(",", ":")))


def _estimated_cache_savings(count: int, detail: str, config: AppConfig) -> float:
    if config.ai.input_token_price is None or config.ai.output_token_price is None:
        return 0.0
    per_frame = (
        config.vision.low_detail_input_tokens_per_frame if detail == "low"
        else config.vision.high_detail_input_tokens_per_frame
    )
    return count * (
        per_frame * float(config.ai.input_token_price)
        + (config.vision.max_output_tokens_per_call / max(config.vision.pass1_batch_size, 1))
        * float(config.ai.output_token_price)
    )


def _anchor_visible(request: dict[str, Any], observations: list[dict[str, Any]], anchor: str, signals: set[str]) -> bool:
    timestamp = request["anchors"].get(anchor)
    if timestamp is None or not observations:
        return False
    nearest = min(observations, key=lambda item: abs(float(item["timestamp"]) - float(timestamp)))
    return nearest["payoff_signal"] in signals and nearest["confidence"] >= 0.45


def _continuity_risk(observations: list[dict[str, Any]]) -> str:
    usable = [item for item in observations if item["origin"] != "local_fallback"]
    if len(usable) < 3:
        return "unknown"
    scene_changes = sum(first["scene_type"] != second["scene_type"] for first, second in zip(usable, usable[1:]))
    missing = sum(bool(item["missing_evidence"]) for item in usable)
    if scene_changes >= 2 or missing >= len(usable) - 1:
        return "high"
    if scene_changes or missing:
        return "medium"
    return "low"


def _average_confidence(observations: list[dict[str, Any]]) -> float:
    if not observations:
        return 0.0
    return sum(float(item["confidence"]) for item in observations) / len(observations)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
