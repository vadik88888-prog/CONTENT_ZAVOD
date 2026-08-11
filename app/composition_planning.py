from __future__ import annotations

"""Phase 7D evidence-driven, stable vertical composition planning.

The planner consumes the bounded decisions in :class:`CreativeIntent` and
trusted target observations.  Observations are evidence, never crop commands:
all geometry, switching, smoothing and fallbacks are resolved deterministically
here before a renderer sees the plan.
"""

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Literal, Sequence

from pydantic import Field

from app.creative_contracts import (
    AttentionTarget,
    COMPOSITION_PLAN_SCHEMA_VERSION,
    CompositionCropKeyframe,
    CompositionGeometryContract,
    CompositionGeometryRegion,
    CompositionPlan,
    CompositionPunchIn,
    CompositionQualityFinding,
    CompositionQualityMetrics,
    CompositionQualityProvenance,
    CompositionQualityReport,
    CompositionSegmentPlan,
    CreativeIntent,
    FrozenContract,
    LayoutFamily,
    MotionDomain,
    NormalizedRect,
    OutputInterval,
    ResolvedCompositionTarget,
    ResolvedMotionEvent,
)


COMPOSITION_PLANNER_VERSION = "7D.composition-planner.3"

CompositionFallback = Literal["none", "wider_crop", "stable_source", "fit_background"]
MovementReason = Literal[
    "none", "target_acquired", "target_switch", "editorial_punch_in",
    "punch_out", "scene_reset", "safe_fallback",
]
CompositionQualityStatus = Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED"]


class TargetObservation(FrozenContract):
    """A trusted visual observation in output time and normalized source space."""

    observation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
    time_base: Literal["output_frame_30fps"] = "output_frame_30fps"
    frame: int = Field(ge=0)
    target: AttentionTarget
    target_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
    bounds: NormalizedRect
    confidence: float = Field(ge=0, le=1)
    evidence_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
    occlusion_ratio: float = Field(default=0, ge=0, le=1)
    scene_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
    protected: bool = True

    @property
    def effective_confidence(self) -> float:
        return self.confidence * (1 - self.occlusion_ratio)


@dataclass(frozen=True, slots=True)
class CompositionPlannerConfig:
    enter_confidence: float = 0.68
    exit_confidence: float = 0.48
    switch_advantage: float = 0.12
    minimum_hold_frames: int = 45
    switch_cooldown_frames: int = 30
    maximum_switches_per_minute: float = 18.0
    target_margin_ratio: float = 0.16
    minimum_fill_crop_width: float = 0.18
    transition_frames: int = 18
    maximum_velocity_per_frame: float = 0.014
    maximum_acceleration_per_frame_sq: float = 0.0028
    punch_in_low: float = 1.025
    punch_in_balanced: float = 1.045
    punch_in_high: float = 1.065

    def __post_init__(self) -> None:
        if not 0 <= self.exit_confidence < self.enter_confidence <= 1:
            raise ValueError("composition confidence thresholds must satisfy exit < enter")
        if not 0 <= self.switch_advantage <= 1:
            raise ValueError("composition switch advantage must be normalized")
        if self.minimum_hold_frames < 1 or self.switch_cooldown_frames < 0:
            raise ValueError("composition hold and cooldown must be non-negative")
        if self.transition_frames < 1:
            raise ValueError("composition transition must contain at least one frame")
        if self.maximum_velocity_per_frame <= 0 or self.maximum_acceleration_per_frame_sq <= 0:
            raise ValueError("composition motion limits must be positive")


@dataclass(frozen=True, slots=True)
class _TargetState:
    decision: ResolvedCompositionTarget
    target_ref: str
    bounds: NormalizedRect
    confidence: float
    observations: tuple[TargetObservation, ...]

    @property
    def key(self) -> tuple[AttentionTarget, str]:
        return self.decision.target, self.target_ref


@dataclass(frozen=True, slots=True)
class _AtomicState:
    output: OutputInterval
    state: _TargetState | None
    layout: LayoutFamily
    desired_crop: NormalizedRect
    target: AttentionTarget
    target_ref: str | None
    confidence: float
    evidence_refs: tuple[str, ...]
    fallback: CompositionFallback
    reason: MovementReason
    punch_event: ResolvedMotionEvent | None


@dataclass(frozen=True, slots=True)
class _MotionState:
    values: tuple[float, float, float, float]
    velocity: tuple[float, float, float, float]


class CompositionPlanner:
    """Compile a calm, bounded crop timeline from evidence-backed intent."""

    def __init__(self, config: CompositionPlannerConfig | None = None) -> None:
        self.config = config or CompositionPlannerConfig()

    def plan(
        self,
        intent: CreativeIntent,
        observations: Iterable[TargetObservation],
        *,
        source_width: int,
        source_height: int,
    ) -> CompositionPlan:
        if source_width <= 0 or source_height <= 0:
            raise ValueError("composition source dimensions must be positive")
        ordered = tuple(sorted(observations, key=lambda item: (item.frame, item.observation_id)))
        diagnostics: list[str] = []
        trusted = _trusted_observations(intent, ordered, diagnostics)
        cuts = _scene_cut_frames(intent, trusted)
        intervals = _atomic_intervals(intent, cuts)
        if not intervals:
            report = _quality_report((), (), 0, 0, intent, self.config)
            return CompositionPlan(
                schema_version=COMPOSITION_PLAN_SCHEMA_VERSION,
                intent_id=intent.intent_id,
                quality_report=report,
                diagnostics=("NO_OUTPUT_INTERVALS",),
            )

        raw_states = [
            _candidate_for_interval(intent, interval, trusted)
            for interval in intervals
        ]
        atomic, suppressed = self._resolve_state_machine(
            intent, intervals, raw_states, cuts, source_width, source_height, diagnostics,
        )
        segments, track_diagnostics = self._smooth_and_freeze(
            atomic, trusted, source_width, source_height,
        )
        diagnostics.extend(track_diagnostics)
        report = _quality_report(segments, tuple(diagnostics), suppressed, len(intervals), intent, self.config)
        return CompositionPlan(
            schema_version=COMPOSITION_PLAN_SCHEMA_VERSION,
            intent_id=intent.intent_id,
            segments=segments,
            quality_report=report,
            diagnostics=tuple(dict.fromkeys(diagnostics)),
        )

    def _resolve_state_machine(
        self,
        intent: CreativeIntent,
        intervals: Sequence[OutputInterval],
        candidates: Sequence[_TargetState | None],
        cuts: frozenset[int],
        source_width: int,
        source_height: int,
        diagnostics: list[str],
    ) -> tuple[tuple[_AtomicState, ...], int]:
        current: _TargetState | None = None
        held_since = -100_000
        last_switch = -100_000
        suppressed = 0
        previous_punch = False
        result: list[_AtomicState] = []
        work: list[tuple[OutputInterval, _TargetState | None]] = list(zip(intervals, candidates))
        index = 0
        while index < len(work):
            interval, candidate = work[index]
            frame = interval.start_frame
            cut = frame in cuts

            # A target may disappear before the minimum hold has elapsed.
            # Split the otherwise-atomic interval at the exact hold boundary
            # so a stable crop cannot be kept beyond the configured limit.
            # Scene/edit-map cuts are hard resets and never inherit this hold.
            hold_end = held_since + self.config.minimum_hold_frames
            needs_hold_boundary = (
                current is not None
                and not cut
                and candidate is None
                and frame < hold_end < interval.end_frame
            )
            if needs_hold_boundary:
                work[index:index + 1] = [
                    (
                        OutputInterval(start_frame=frame, end_frame=hold_end),
                        candidate,
                    ),
                    (
                        OutputInterval(start_frame=hold_end, end_frame=interval.end_frame),
                        candidate,
                    ),
                ]
                interval, candidate = work[index]

            reason: MovementReason = "none"
            fallback: CompositionFallback = "none"
            selected: _TargetState | None = None

            if cut:
                current = None
                held_since = frame
                last_switch = frame
                reason = "scene_reset"
            if candidate is not None and candidate.confidence >= self.config.enter_confidence:
                if current is None:
                    selected = candidate
                    current = candidate
                    held_since = frame
                    if not cut:
                        reason = "target_acquired"
                elif candidate.key == current.key:
                    selected = candidate
                    current = candidate
                else:
                    hold_ok = frame - held_since >= self.config.minimum_hold_frames
                    cooldown_ok = frame - last_switch >= self.config.switch_cooldown_frames
                    held_confidence = (
                        current.confidence if current.decision.output.contains(interval) else 0.0
                    )
                    advantage_ok = (
                        held_confidence < self.config.exit_confidence
                        or candidate.confidence >= held_confidence + self.config.switch_advantage
                    )
                    if hold_ok and cooldown_ok and advantage_ok:
                        selected = candidate
                        current = candidate
                        held_since = frame
                        last_switch = frame
                        reason = "target_switch"
                    else:
                        suppressed += 1
                        fallback = "stable_source"
                        reason = "safe_fallback"
                        diagnostics.append(f"SWITCH_SUPPRESSED_HYSTERESIS:{frame}")
            elif current is not None and candidate is not None and candidate.key == current.key:
                if candidate.confidence >= self.config.exit_confidence:
                    selected = candidate
                    current = candidate
                else:
                    current = None
            elif (
                current is not None
                and not cut
                and frame - held_since < self.config.minimum_hold_frames
            ):
                # Preserve visual continuity after sparse evidence ends, but do
                # not extend the semantic target claim.  The emitted segment
                # below is an evidence-free stable_source fallback whose crop
                # is inherited from the last acquired state.
                fallback = "stable_source"
                reason = "safe_fallback"
            else:
                current = None

            punch_event = _punch_event(intent, interval, selected)
            if punch_event is not None:
                reason = "editorial_punch_in"
            elif previous_punch and selected is not None:
                reason = "punch_out"
            previous_punch = punch_event is not None

            if selected is None:
                if fallback == "stable_source" and current is not None:
                    crop, _crop_fallback = _target_crop(
                        current.bounds, source_width, source_height, self.config,
                    )
                    layout = _layout_for(current, "none")
                    diagnostics.append(f"STABLE_CROP_HELD:{frame}")
                else:
                    crop, layout, fallback = _calm_fallback_crop(source_width, source_height)
                    diagnostics.append(f"LOW_CONFIDENCE_SAFE_FRAME:{frame}")
                result.append(_AtomicState(
                    output=interval, state=None, layout=layout, desired_crop=crop,
                    target=AttentionTarget.STABLE_SOURCE, target_ref=None, confidence=0,
                    evidence_refs=(), fallback=fallback,
                    reason="scene_reset" if cut else "safe_fallback",
                    punch_event=None,
                ))
                index += 1
                continue

            crop, crop_fallback = _target_crop(
                selected.bounds, source_width, source_height, self.config,
            )
            layout = _layout_for(selected, crop_fallback)
            if crop_fallback != "none":
                fallback = crop_fallback
                diagnostics.append(f"WIDER_CROP_FOR_CONTAINMENT:{frame}")
            if punch_event is not None:
                scale = _punch_scale(punch_event, self.config)
                crop = _scale_crop(crop, scale, selected.bounds)
            result.append(_AtomicState(
                output=interval, state=selected, layout=layout, desired_crop=crop,
                target=selected.decision.target, target_ref=selected.target_ref,
                confidence=selected.confidence, evidence_refs=selected.decision.evidence_refs,
                fallback=fallback, reason=reason, punch_event=punch_event,
            ))
            index += 1
        return tuple(result), suppressed

    def _smooth_and_freeze(
        self,
        atomic: Sequence[_AtomicState],
        observations: Sequence[TargetObservation],
        source_width: int,
        source_height: int,
    ) -> tuple[tuple[CompositionSegmentPlan, ...], tuple[str, ...]]:
        result: list[CompositionSegmentPlan] = []
        diagnostics: list[str] = []
        motion: _MotionState | None = None
        previous_end = -1
        for index, item in enumerate(atomic, start=1):
            reset = previous_end != item.output.start_frame or item.reason == "scene_reset"
            incoming_motion = motion
            keyframes, motion = _crop_track(item, incoming_motion, reset, self.config)
            crop = keyframes[-1].crop
            geometry, protected, containment = _geometry_for(
                item, crop, observations, source_width, source_height,
            )
            fallback = item.fallback
            layout = item.layout
            if containment < 0.98 and item.state is not None:
                crop = NormalizedRect(x=0, y=0, width=1, height=1)
                layout = LayoutFamily.FIT_BACKGROUND
                fallback = "fit_background"
                keyframes, motion = _crop_track(
                    _replace_atomic_crop(item, crop), incoming_motion, True, self.config,
                )
                crop = keyframes[-1].crop
                geometry, protected, containment = _geometry_for(
                    item, crop, observations, source_width, source_height,
                )
                diagnostics.append(f"FIT_BACKGROUND_FOR_CLIPPING:{item.output.start_frame}")
            punch = None
            if item.punch_event is not None:
                punch = CompositionPunchIn(
                    event_id=item.punch_event.decision_id,
                    output=item.output,
                    scale=_punch_scale(item.punch_event, self.config),
                    evidence_refs=item.punch_event.evidence_refs,
                )
            result.append(CompositionSegmentPlan(
                segment_id=f"composition-{index:03d}",
                output=item.output,
                layout=layout,
                target=item.target,
                target_ref=item.target_ref,
                crop=crop,
                target_confidence=item.confidence,
                target_bounds=item.state.bounds if item.state is not None else None,
                protected_regions=protected,
                geometry=geometry.model_copy(update={"source_crop": crop}),
                crop_keyframes=keyframes,
                movement_reason=item.reason,
                punch_in=punch,
                easing_id="ease_in_out" if len(keyframes) > 1 else "none",
                evidence_refs=item.evidence_refs,
                fallback=fallback,
            ))
            previous_end = item.output.end_frame
        return tuple(result), tuple(diagnostics)


def build_composition_plan(
    intent: CreativeIntent,
    observations: Iterable[TargetObservation],
    *,
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig | None = None,
) -> CompositionPlan:
    return CompositionPlanner(config).plan(
        intent, observations, source_width=source_width, source_height=source_height,
    )


def _trusted_observations(
    intent: CreativeIntent,
    observations: Sequence[TargetObservation],
    diagnostics: list[str],
) -> tuple[TargetObservation, ...]:
    evidence_refs = {item.evidence_ref for item in intent.evidence_manifest}
    output_ranges = tuple(item.output for item in intent.source_output_mapping.segments)
    trusted: list[TargetObservation] = []
    for item in observations:
        if item.evidence_ref not in evidence_refs:
            diagnostics.append(f"UNTRUSTED_OBSERVATION_DROPPED:{item.observation_id}")
            continue
        if not any(interval.start_frame <= item.frame < interval.end_frame for interval in output_ranges):
            diagnostics.append(f"OUT_OF_TIMELINE_OBSERVATION_DROPPED:{item.observation_id}")
            continue
        trusted.append(item)
    return tuple(trusted)


def _scene_cut_frames(
    intent: CreativeIntent,
    observations: Sequence[TargetObservation],
) -> frozenset[int]:
    cuts = {item.output.start_frame for item in intent.source_output_mapping.segments}
    previous: TargetObservation | None = None
    for item in observations:
        if previous is not None and item.scene_id and previous.scene_id and item.scene_id != previous.scene_id:
            cuts.add(item.frame)
        previous = item
    return frozenset(cuts)


def _atomic_intervals(intent: CreativeIntent, cuts: frozenset[int]) -> tuple[OutputInterval, ...]:
    result: list[OutputInterval] = []
    composition_events = tuple(
        item for item in intent.motion_events if item.domain == MotionDomain.COMPOSITION
    )
    for mapping in intent.source_output_mapping.segments:
        boundaries = {mapping.output.start_frame, mapping.output.end_frame}
        for target in intent.composition_targets:
            if _overlaps(mapping.output, target.output):
                boundaries.update((
                    max(mapping.output.start_frame, target.output.start_frame),
                    min(mapping.output.end_frame, target.output.end_frame),
                ))
        for event in composition_events:
            if _overlaps(mapping.output, event.output):
                boundaries.update((
                    max(mapping.output.start_frame, event.output.start_frame),
                    min(mapping.output.end_frame, event.output.end_frame),
                ))
        boundaries.update(frame for frame in cuts if mapping.output.start_frame < frame < mapping.output.end_frame)
        ordered = sorted(boundaries)
        result.extend(
            OutputInterval(start_frame=left, end_frame=right)
            for left, right in zip(ordered, ordered[1:]) if right > left
        )
    return tuple(result)


def _candidate_for_interval(
    intent: CreativeIntent,
    interval: OutputInterval,
    observations: Sequence[TargetObservation],
) -> _TargetState | None:
    candidates: list[_TargetState] = []
    for decision in intent.composition_targets:
        if not decision.output.contains(interval):
            continue
        local_scene_ids = {
            item.scene_id for item in observations
            if interval.start_frame <= item.frame < interval.end_frame and item.scene_id is not None
        }
        matches = [
            item for item in observations
            if decision.output.start_frame <= item.frame < decision.output.end_frame
            and item.target == decision.target
            and item.evidence_ref in decision.evidence_refs
            and (decision.target_ref is None or item.target_ref == decision.target_ref)
            and (not local_scene_ids or item.scene_id in local_scene_ids)
        ]
        if not matches:
            continue
        by_ref: dict[str, list[TargetObservation]] = {}
        for item in matches:
            by_ref.setdefault(item.target_ref, []).append(item)
        groups = list(by_ref.items())
        if decision.target == AttentionTarget.GROUP:
            groups = [(decision.target_ref or "group", matches)]
        for target_ref, group in groups:
            effective = median(item.effective_confidence for item in group)
            confidence = min(decision.confidence, effective)
            candidates.append(_TargetState(
                decision=decision,
                target_ref=target_ref,
                bounds=_union_rect(tuple(item.bounds for item in group)),
                confidence=confidence,
                observations=tuple(group),
            ))
    if not candidates:
        return None
    semantic_rank = {
        AttentionTarget.SCREEN: 7,
        AttentionTarget.PRODUCT: 7,
        AttentionTarget.OBJECT: 6,
        AttentionTarget.SPEAKER: 5,
        AttentionTarget.REACTION: 4,
        AttentionTarget.SUBJECT: 3,
        AttentionTarget.GROUP: 2,
        AttentionTarget.STABLE_SOURCE: 1,
    }
    return max(
        candidates,
        key=lambda item: (
            item.decision.priority,
            semantic_rank[item.decision.target],
            item.confidence,
            item.target_ref,
        ),
    )


def _target_crop(
    bounds: NormalizedRect,
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig,
) -> tuple[NormalizedRect, Literal["none", "wider_crop"]]:
    output_aspect = 9 / 16
    source_aspect = source_width / source_height
    normalized_ratio = output_aspect / source_aspect
    usable = 1 - 2 * config.target_margin_ratio
    required_height = max(0.52, bounds.height / usable, bounds.width / max(normalized_ratio * usable, 1e-9))
    if required_height <= 1:
        height = min(1.0, required_height)
        width = max(config.minimum_fill_crop_width, normalized_ratio * height)
        return _centered_rect(bounds, width, height), "none"
    width = min(1.0, max(bounds.width / usable, normalized_ratio, config.minimum_fill_crop_width))
    height = min(1.0, max(bounds.height / usable, width / max(normalized_ratio, 1e-9) * 0.72))
    return _centered_rect(bounds, width, height), "wider_crop"


def _calm_fallback_crop(
    source_width: int,
    source_height: int,
) -> tuple[NormalizedRect, LayoutFamily, Literal["stable_source", "fit_background"]]:
    source_aspect = source_width / source_height
    output_aspect = 9 / 16
    if source_aspect <= output_aspect * 1.12:
        return NormalizedRect(x=0, y=0, width=1, height=1), LayoutFamily.SINGLE_SUBJECT, "stable_source"
    return NormalizedRect(x=0, y=0, width=1, height=1), LayoutFamily.FIT_BACKGROUND, "fit_background"


def _layout_for(
    state: _TargetState,
    fallback: Literal["none", "wider_crop"],
) -> LayoutFamily:
    allowed = state.decision.allowed_layouts
    preferred: tuple[LayoutFamily, ...]
    if state.decision.target == AttentionTarget.SPEAKER:
        preferred = (LayoutFamily.STABLE_SPEAKER, LayoutFamily.SINGLE_SUBJECT)
    elif state.decision.target in {AttentionTarget.SCREEN, AttentionTarget.PRODUCT}:
        preferred = (LayoutFamily.SCREEN_PRODUCT, LayoutFamily.SCREEN_PRIORITY, LayoutFamily.WIDE_GROUP)
    elif state.decision.target == AttentionTarget.GROUP:
        width = state.bounds.width
        height = state.bounds.height
        preferred = (
            (LayoutFamily.SPLIT, LayoutFamily.WIDE_GROUP, LayoutFamily.STACKED)
            if width >= height
            else (LayoutFamily.STACKED, LayoutFamily.WIDE_GROUP, LayoutFamily.SPLIT)
        )
    else:
        preferred = (LayoutFamily.SINGLE_SUBJECT, LayoutFamily.STABLE_SPEAKER)
    if fallback == "wider_crop":
        preferred = (LayoutFamily.WIDE_GROUP, *preferred)
    return next((item for item in preferred if item in allowed), allowed[0])


def _punch_event(
    intent: CreativeIntent,
    interval: OutputInterval,
    state: _TargetState | None,
) -> ResolvedMotionEvent | None:
    if state is None:
        return None
    return next((
        event for event in intent.motion_events
        if event.domain == MotionDomain.COMPOSITION
        and event.output.contains(interval)
        and bool(set(event.evidence_refs).intersection(state.decision.evidence_refs))
    ), None)


def _punch_scale(event: ResolvedMotionEvent, config: CompositionPlannerConfig) -> float:
    return {
        "low": config.punch_in_low,
        "balanced": config.punch_in_balanced,
        "high": config.punch_in_high,
    }[event.intensity.value]


def _scale_crop(crop: NormalizedRect, scale: float, target: NormalizedRect) -> NormalizedRect:
    width = crop.width / scale
    height = crop.height / scale
    return _centered_rect(target, width, height)


def _crop_track(
    item: _AtomicState,
    previous: _MotionState | None,
    reset: bool,
    config: CompositionPlannerConfig,
) -> tuple[tuple[CompositionCropKeyframe, ...], _MotionState]:
    desired = _rect_values(item.desired_crop)
    if previous is None or reset:
        state = _MotionState(values=desired, velocity=(0, 0, 0, 0))
        return (CompositionCropKeyframe(
            frame=item.output.start_frame, crop=item.desired_crop,
            reason="scene_reset" if reset else "static",
        ),), state

    values = previous.values
    velocity = previous.velocity
    frames: list[CompositionCropKeyframe] = []
    previous_speed = max(abs(value) for value in velocity)
    # Continue sampling until the bounded controller settles or the semantic
    # segment ends.  A fixed short transition could strand a far-away target
    # outside the crop even though a longer, slower pan is safe.
    max_steps = item.output.end_frame - item.output.start_frame
    reason = _keyframe_reason(item.reason)
    for offset in range(max_steps):
        next_values: list[float] = []
        next_velocity: list[float] = []
        for value, speed, target in zip(values, velocity, desired):
            requested = (target - value) * 0.20 - speed * 0.72
            acceleration = _clamp(
                requested,
                -config.maximum_acceleration_per_frame_sq,
                config.maximum_acceleration_per_frame_sq,
            )
            new_speed = _clamp(
                speed + acceleration,
                -config.maximum_velocity_per_frame,
                config.maximum_velocity_per_frame,
            )
            new_value = value + new_speed
            if abs(target - new_value) < 0.00005 and abs(new_speed) < 0.0001:
                new_value = target
                new_speed = 0
            next_values.append(new_value)
            next_velocity.append(new_speed)
        values = tuple(next_values)  # type: ignore[assignment]
        velocity = tuple(next_velocity)  # type: ignore[assignment]
        rect = _rect_from_values(values)
        speed = max(abs(value) for value in velocity)
        acceleration = abs(speed - previous_speed)
        frames.append(CompositionCropKeyframe(
            frame=item.output.start_frame + offset,
            crop=rect,
            velocity_per_frame=round(speed, 9),
            acceleration_per_frame_sq=round(acceleration, 9),
            reason=reason,
        ))
        previous_speed = speed
        if values == desired and not any(velocity):
            break
    if not frames:
        frames.append(CompositionCropKeyframe(
            frame=item.output.start_frame, crop=_rect_from_values(values), reason="static",
        ))
    return tuple(frames), _MotionState(values=values, velocity=velocity)


def _geometry_for(
    item: _AtomicState,
    crop: NormalizedRect,
    observations: Sequence[TargetObservation],
    source_width: int,
    source_height: int,
) -> tuple[CompositionGeometryContract, tuple[NormalizedRect, ...], float]:
    content = _output_content_bounds(crop, source_width, source_height)
    relevant = [
        observation for observation in observations
        if item.output.start_frame <= observation.frame < item.output.end_frame
        and observation.protected
        and observation.effective_confidence >= 0.48
    ]
    if item.state is not None:
        relevant.extend(item.state.observations)
    by_target: dict[tuple[AttentionTarget, str], list[TargetObservation]] = {}
    for observation in relevant:
        bucket = by_target.setdefault((observation.target, observation.target_ref), [])
        if observation.observation_id not in {item.observation_id for item in bucket}:
            bucket.append(observation)
    regions: list[CompositionGeometryRegion] = []
    for (target, target_ref), values in sorted(by_target.items(), key=lambda item: (item[0][0].value, item[0][1])):
        source_bounds = _union_rect(tuple(item.bounds for item in values))
        projected = _project_rect(source_bounds, crop, content)
        if projected is None:
            continue
        confidence = median(item.effective_confidence for item in values)
        regions.append(CompositionGeometryRegion(
            region_id=f"region-{target.value}-{target_ref}",
            kind=_region_kind(target),
            bounds=projected,
            target_ref=target_ref,
            confidence=round(confidence, 7),
            importance=1 if item.state is not None and target == item.target and target_ref == item.target_ref else 0.75,
        ))
    target_regions = tuple(
        region for region in regions
        if item.target_ref is not None and region.target_ref == item.target_ref and region.kind == _region_kind(item.target)
    )
    containment = 1.0
    if item.state is not None:
        containment = _containment(item.state.bounds, crop)
    geometry = CompositionGeometryContract(
        source_crop=crop,
        output_content_bounds=content,
        target_regions=target_regions,
        protected_regions=tuple(regions),
    )
    return geometry, tuple(region.bounds for region in regions), containment


def _quality_report(
    segments: Sequence[CompositionSegmentPlan],
    diagnostics: Sequence[str],
    suppressed_switches: int,
    interval_count: int,
    intent: CreativeIntent,
    config: CompositionPlannerConfig,
) -> CompositionQualityReport:
    findings: list[CompositionQualityFinding] = []
    switch_count = sum(item.movement_reason == "target_switch" for item in segments)
    layout_switch_count = sum(
        left.layout != right.layout for left, right in zip(segments, segments[1:])
    )
    fallback_segments = [item for item in segments if item.fallback != "none"]
    punch_count = sum(item.punch_in is not None for item in segments)
    maximum_velocity = max((key.velocity_per_frame for item in segments for key in item.crop_keyframes), default=0)
    maximum_acceleration = max((key.acceleration_per_frame_sq for item in segments for key in item.crop_keyframes), default=0)
    containments = [
        _containment(item.target_bounds, item.crop)
        for item in segments if item.target_bounds is not None
    ]
    clipped = [value for value in containments if value < 0.98]
    unsafe = [
        item for item in segments
        if item.layout not in {LayoutFamily.FIT_BACKGROUND, LayoutFamily.LEGACY_PASSTHROUGH}
        and item.crop.width < config.minimum_fill_crop_width - 1e-7
    ]
    duration_frames = sum(item.output.end_frame - item.output.start_frame for item in segments)
    switches_per_minute = switch_count * 1800 / duration_frames if duration_frames else 0
    layout_switches_per_minute = (
        layout_switch_count * 1800 / duration_frames if duration_frames else 0
    )
    jitter = _jitter_events(segments)
    if clipped:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_TARGET_CLIPPED", severity="blocker",
            measured_value=round(min(clipped), 7), threshold=0.98,
            message="A selected semantic target is not safely contained by its final crop.",
        ))
    if unsafe:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_UNSAFE_CROP", severity="blocker",
            segment_id=unsafe[0].segment_id, measured_value=unsafe[0].crop.width,
            threshold=config.minimum_fill_crop_width,
            message="A fill crop exceeds the configured safe zoom limit.",
        ))
    if maximum_velocity > config.maximum_velocity_per_frame + 1e-7:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_VELOCITY_LIMIT", severity="blocker",
            measured_value=maximum_velocity, threshold=config.maximum_velocity_per_frame,
            message="Crop velocity exceeds the deterministic composition limit.",
        ))
    if maximum_acceleration > config.maximum_acceleration_per_frame_sq + 1e-7:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_ACCELERATION_LIMIT", severity="blocker",
            measured_value=maximum_acceleration, threshold=config.maximum_acceleration_per_frame_sq,
            message="Crop acceleration exceeds the deterministic composition limit.",
        ))
    if jitter:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_JITTER", severity="blocker", measured_value=jitter,
            threshold=0, message="The crop track contains unintended direction reversals.",
        ))
    if switches_per_minute > config.maximum_switches_per_minute:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_SWITCH_RATE_HIGH", severity="warning",
            measured_value=round(switches_per_minute, 4), threshold=config.maximum_switches_per_minute,
            message="Evidence-backed target switches are more frequent than the calm-framing budget.",
        ))
    if layout_switches_per_minute > config.maximum_switches_per_minute:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_LAYOUT_SWITCH_RATE_HIGH", severity="warning",
            measured_value=round(layout_switches_per_minute, 4),
            threshold=config.maximum_switches_per_minute,
            message="Visible layout/framing changes are more frequent than the calm-framing budget.",
        ))
    if suppressed_switches:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_MINIMUM_HOLD_VIOLATION", severity="warning",
            measured_value=suppressed_switches, threshold=0,
            message="Unstable target proposals were suppressed by hold, cooldown or hysteresis.",
        ))
    low_confidence_count = sum(item.target == AttentionTarget.STABLE_SOURCE for item in segments)
    if low_confidence_count:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_LOW_CONFIDENCE", severity="warning",
            measured_value=low_confidence_count, threshold=0,
            message="Low-confidence evidence uses a calm stable or fit/background frame.",
        ))
    if fallback_segments:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_SAFE_FALLBACK", severity="warning",
            measured_value=len(fallback_segments), threshold=0,
            message="Composition applied the declared wider, stable or fit/background fallback chain.",
        ))
    status: CompositionQualityStatus = (
        "BLOCKED" if any(item.severity == "blocker" for item in findings)
        else "PASS_WITH_WARNINGS" if findings
        else "PASS"
    )
    return CompositionQualityReport(
        status=status,
        findings=tuple(findings),
        metrics=CompositionQualityMetrics(
            segment_count=len(segments),
            target_switch_count=switch_count,
            layout_switch_count=layout_switch_count,
            suppressed_switch_count=suppressed_switches,
            fallback_count=len(fallback_segments),
            punch_in_count=punch_count,
            clipped_target_count=len(clipped),
            unsafe_crop_count=len(unsafe),
            jitter_event_count=jitter,
            minimum_hold_violation_count=0,
            max_velocity_per_frame=round(maximum_velocity, 9),
            max_acceleration_per_frame_sq=round(maximum_acceleration, 9),
            switches_per_minute=round(switches_per_minute, 6),
            layout_switches_per_minute=round(layout_switches_per_minute, 6),
            minimum_target_containment=round(min(containments), 7) if containments else 1,
        ),
        provenance=CompositionQualityProvenance(
            producer="composition_planner",
            planner_version=COMPOSITION_PLANNER_VERSION,
            intent_id=intent.intent_id,
        ),
    )


def _replace_atomic_crop(item: _AtomicState, crop: NormalizedRect) -> _AtomicState:
    return _AtomicState(
        output=item.output, state=item.state, layout=LayoutFamily.FIT_BACKGROUND,
        desired_crop=crop, target=item.target, target_ref=item.target_ref,
        confidence=item.confidence, evidence_refs=item.evidence_refs,
        fallback="fit_background", reason="safe_fallback", punch_event=None,
    )


def _centered_rect(target: NormalizedRect, width: float, height: float) -> NormalizedRect:
    width = _clamp(width, 0.000001, 1)
    height = _clamp(height, 0.000001, 1)
    center_x = target.x + target.width / 2
    center_y = target.y + target.height / 2
    x = _clamp(center_x - width / 2, 0, 1 - width)
    y = _clamp(center_y - height / 2, 0, 1 - height)
    return NormalizedRect(
        x=round(x, 8), y=round(y, 8), width=round(width, 8), height=round(height, 8),
    )


def _union_rect(rectangles: Sequence[NormalizedRect]) -> NormalizedRect:
    left = min(item.x for item in rectangles)
    top = min(item.y for item in rectangles)
    right = max(item.x + item.width for item in rectangles)
    bottom = max(item.y + item.height for item in rectangles)
    return NormalizedRect(
        x=round(left, 8), y=round(top, 8),
        width=round(right - left, 8), height=round(bottom - top, 8),
    )


def _output_content_bounds(
    crop: NormalizedRect,
    source_width: int,
    source_height: int,
) -> NormalizedRect:
    source_crop_aspect = crop.width * source_width / (crop.height * source_height)
    output_aspect = 9 / 16
    if source_crop_aspect > output_aspect:
        height = output_aspect / source_crop_aspect
        return NormalizedRect(x=0, y=round((1 - height) / 2, 8), width=1, height=round(height, 8))
    width = source_crop_aspect / output_aspect
    return NormalizedRect(x=round((1 - width) / 2, 8), y=0, width=round(width, 8), height=1)


def _project_rect(
    bounds: NormalizedRect,
    crop: NormalizedRect,
    content: NormalizedRect,
) -> NormalizedRect | None:
    left = max(bounds.x, crop.x)
    top = max(bounds.y, crop.y)
    right = min(bounds.x + bounds.width, crop.x + crop.width)
    bottom = min(bounds.y + bounds.height, crop.y + crop.height)
    if right <= left or bottom <= top:
        return None
    x = content.x + ((left - crop.x) / crop.width) * content.width
    y = content.y + ((top - crop.y) / crop.height) * content.height
    width = ((right - left) / crop.width) * content.width
    height = ((bottom - top) / crop.height) * content.height
    return NormalizedRect(
        x=round(_clamp(x, 0, 1), 8), y=round(_clamp(y, 0, 1), 8),
        width=round(min(width, 1 - x), 8), height=round(min(height, 1 - y), 8),
    )


def _containment(bounds: NormalizedRect, crop: NormalizedRect) -> float:
    left = max(bounds.x, crop.x)
    top = max(bounds.y, crop.y)
    right = min(bounds.x + bounds.width, crop.x + crop.width)
    bottom = min(bounds.y + bounds.height, crop.y + crop.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / (bounds.width * bounds.height)


def _rect_values(rect: NormalizedRect) -> tuple[float, float, float, float]:
    return rect.x + rect.width / 2, rect.y + rect.height / 2, rect.width, rect.height


def _rect_from_values(values: tuple[float, float, float, float]) -> NormalizedRect:
    center_x, center_y, width, height = values
    width = _clamp(width, 0.000001, 1)
    height = _clamp(height, 0.000001, 1)
    x = _clamp(center_x - width / 2, 0, 1 - width)
    y = _clamp(center_y - height / 2, 0, 1 - height)
    return NormalizedRect(
        x=round(x, 8), y=round(y, 8), width=round(width, 8), height=round(height, 8),
    )


def _region_kind(target: AttentionTarget) -> Literal[
    "face", "subject", "object", "product", "screen", "reaction", "group", "overlay"
]:
    return {
        AttentionTarget.SPEAKER: "face",
        AttentionTarget.SUBJECT: "subject",
        AttentionTarget.OBJECT: "object",
        AttentionTarget.PRODUCT: "product",
        AttentionTarget.SCREEN: "screen",
        AttentionTarget.REACTION: "reaction",
        AttentionTarget.GROUP: "group",
        AttentionTarget.STABLE_SOURCE: "overlay",
    }[target]  # type: ignore[return-value]


def _keyframe_reason(reason: str) -> Literal[
    "static", "target_acquired", "target_switch", "editorial_punch_in",
    "punch_out", "scene_reset", "safe_fallback"
]:
    return reason if reason != "none" else "static"  # type: ignore[return-value]


def _jitter_events(segments: Sequence[CompositionSegmentPlan]) -> int:
    # Direction changes are only suspicious inside one fixed editorial state;
    # target-switch and punch boundaries legitimately reverse a move.
    count = 0
    for segment in segments:
        centers = [
            (item.crop.x + item.crop.width / 2, item.crop.y + item.crop.height / 2)
            for item in segment.crop_keyframes
        ]
        previous: tuple[int, int] | None = None
        for left, right in zip(centers, centers[1:]):
            direction = (_sign(right[0] - left[0]), _sign(right[1] - left[1]))
            if previous is not None and any(a and b and a != b for a, b in zip(previous, direction)):
                count += 1
            previous = direction
    return count


def _sign(value: float) -> int:
    return 1 if value > 1e-7 else -1 if value < -1e-7 else 0


def _overlaps(left: OutputInterval, right: OutputInterval) -> bool:
    return left.start_frame < right.end_frame and right.start_frame < left.end_frame


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
