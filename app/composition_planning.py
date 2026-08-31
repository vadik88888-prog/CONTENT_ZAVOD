from __future__ import annotations

"""Phase 7D evidence-driven, stable vertical composition planning.

The planner consumes the bounded decisions in :class:`CreativeIntent` and
trusted target observations.  Observations are evidence, never crop commands:
all geometry, switching, smoothing and fallbacks are resolved deterministically
here before a renderer sees the plan.
"""

from dataclasses import dataclass, replace
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


COMPOSITION_PLANNER_VERSION = "7K.1.confirmed-target-handoff.1"
SPLIT_FACE_CAM_RATIO = 0.35

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
    minimum_layout_dwell_frames: int = 60
    minimum_edge_fragment_frames: int = 18
    local_burst_window_frames: int = 90
    maximum_local_layout_switches: int = 1
    target_margin_ratio: float = 0.16
    # Vision subject bounds often include shoulders or the edge of a shared
    # shot. On a 2:1 source a true 9:16 reframe cannot contain that entire
    # coarse box even when the face/semantic centre is safely preserved.
    minimum_target_containment: float = 0.82
    # A single sparse observation must not jerk a 9:16 crop across a face.
    # Require a second nearby observation before moving an already selected
    # target this far within the same source scene.
    same_target_reframe_distance: float = 0.09
    same_target_reframe_confirmation_window_frames: int = 120
    minimum_fill_crop_width: float = 0.18
    transition_frames: int = 18
    maximum_velocity_per_frame: float = 0.014
    maximum_acceleration_per_frame_sq: float = 0.0028
    # All temporal behaviour of the virtual camera lives here.  The values
    # describe editorial evidence, never detector-smoothing knobs.
    subject_switch_confirmation_frames: int = 90
    subject_occlusion_hold_frames: int = 90
    camera_move_frames: int = 18
    camera_dead_zone_ratio: float = 0.035
    maximum_pan_episode_distance: float = 0.28
    core_width_ratio: float = 0.58
    core_height_ratio: float = 0.62
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
        if self.minimum_layout_dwell_frames < 1 or self.minimum_edge_fragment_frames < 1:
            raise ValueError("composition layout dwell limits must be positive")
        if self.local_burst_window_frames < self.minimum_layout_dwell_frames:
            raise ValueError("composition burst window must cover minimum layout dwell")
        if self.maximum_local_layout_switches < 0:
            raise ValueError("composition local switch limit must be non-negative")
        if not 0 < self.minimum_target_containment <= 1:
            raise ValueError("composition target containment must be normalized and positive")
        if not 0 < self.same_target_reframe_distance <= 1:
            raise ValueError("composition same-target reframe distance must be normalized and positive")
        if self.same_target_reframe_confirmation_window_frames < 1:
            raise ValueError("composition same-target reframe confirmation window must be positive")
        if self.transition_frames < 1:
            raise ValueError("composition transition must contain at least one frame")
        if self.maximum_velocity_per_frame <= 0 or self.maximum_acceleration_per_frame_sq <= 0:
            raise ValueError("composition motion limits must be positive")
        if self.subject_switch_confirmation_frames < 1 or self.subject_occlusion_hold_frames < 1:
            raise ValueError("composition subject evidence holds must be positive")
        if self.camera_move_frames < 2:
            raise ValueError("composition camera moves must have a bounded duration")
        if not 0 <= self.camera_dead_zone_ratio < 0.5:
            raise ValueError("composition camera dead zone must be normalized")
        if not 0 < self.maximum_pan_episode_distance <= 1:
            raise ValueError("composition pan episode distance must be normalized")
        if not 0 < self.core_width_ratio <= 1 or not 0 < self.core_height_ratio <= 1:
            raise ValueError("composition semantic core ratios must be normalized")


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
    hold_reframe: bool = False
    camera_mode: Literal["STATIONARY", "PAN_ONLY", "TRACKING", "GROUP", "BLOCKED"] = "STATIONARY"
    camera_phase: Literal["HOLD", "MOVE"] = "HOLD"
    subject_lock_state: Literal[
        "ACQUIRE", "LOCKED", "TEMPORARILY_OCCLUDED", "SWITCH_PENDING", "SWITCH_CONFIRMED", "EVIDENCE_GAP",
    ] = "EVIDENCE_GAP"
    movement_explanation: str | None = None
    move_from: NormalizedRect | None = None


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
        atomic, family_diagnostics = _lock_scene_layout_families(
            atomic, cuts, source_width, source_height, self.config,
        )
        diagnostics.extend(family_diagnostics)
        atomic, calibration_diagnostics = _calibrate_layout_timeline(atomic, self.config)
        diagnostics.extend(calibration_diagnostics)
        atomic, camera_diagnostics = _plan_shot_virtual_camera(
            atomic, cuts, source_width, source_height, self.config,
        )
        diagnostics.extend(camera_diagnostics)
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
        last_confirmed: _TargetState | None = None
        pending_reframe: _TargetState | None = None
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
            hold_reframe = False

            if cut:
                current = None
                last_confirmed = None
                pending_reframe = None
                held_since = frame
                last_switch = frame
                reason = "scene_reset"
            if candidate is not None and candidate.confidence >= self.config.enter_confidence:
                if current is None:
                    if (
                        last_confirmed is not None
                        and candidate.key == last_confirmed.key
                        and _same_target_reframe_requires_confirmation(
                            candidate, last_confirmed, self.config,
                        )
                    ):
                        if (
                            pending_reframe is not None
                            and pending_reframe.key == candidate.key
                            and not _same_target_reframe_requires_confirmation(
                                candidate, pending_reframe, self.config,
                            )
                            and _reframe_confirmation_is_timely(
                                candidate, pending_reframe, self.config,
                            )
                        ):
                            selected = candidate
                            current = candidate
                            last_confirmed = candidate
                            pending_reframe = None
                            held_since = frame
                            diagnostics.append(f"TARGET_REFRAME_CONFIRMED:{frame}")
                            if not cut:
                                reason = "target_acquired"
                        else:
                            current = last_confirmed
                            pending_reframe = candidate
                            held_since = frame
                            fallback = "stable_source"
                            reason = "safe_fallback"
                            hold_reframe = True
                            diagnostics.append(f"TARGET_REFRAME_HELD:{frame}")
                    else:
                        selected = candidate
                        current = candidate
                        last_confirmed = candidate
                        pending_reframe = None
                        held_since = frame
                        if not cut:
                            reason = "target_acquired"
                elif candidate.key == current.key:
                    if _same_target_reframe_requires_confirmation(
                        candidate, current, self.config,
                    ):
                        if (
                            pending_reframe is not None
                            and pending_reframe.key == candidate.key
                            and not _same_target_reframe_requires_confirmation(
                                candidate, pending_reframe, self.config,
                            )
                            and _reframe_confirmation_is_timely(
                                candidate, pending_reframe, self.config,
                            )
                        ):
                            selected = candidate
                            current = candidate
                            last_confirmed = candidate
                            pending_reframe = None
                            diagnostics.append(f"TARGET_REFRAME_CONFIRMED:{frame}")
                        else:
                            pending_reframe = candidate
                            fallback = "stable_source"
                            reason = "safe_fallback"
                            hold_reframe = True
                            diagnostics.append(f"TARGET_REFRAME_HELD:{frame}")
                    else:
                        selected = candidate
                        current = candidate
                        last_confirmed = candidate
                        pending_reframe = None
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
                    if _is_explicit_target_handoff(current, candidate, interval):
                        # Do not let a minimum hold preserve a target whose
                        # own decision has ended. The replacement is already
                        # confidence-gated above and is therefore an explicit,
                        # evidence-backed handoff rather than a speculative
                        # camera ping-pong proposal.
                        selected = candidate
                        current = candidate
                        last_confirmed = candidate
                        pending_reframe = None
                        held_since = frame
                        last_switch = frame
                        reason = "target_switch"
                        diagnostics.append(f"TARGET_SWITCH_TIMELY:{frame}")
                    elif hold_ok and cooldown_ok and advantage_ok:
                        selected = candidate
                        current = candidate
                        last_confirmed = candidate
                        pending_reframe = None
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
                    last_confirmed = candidate
                    pending_reframe = None
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
                hold_reframe = pending_reframe is not None
            elif pending_reframe is not None and last_confirmed is not None:
                # Sparse target evidence cannot turn an unconfirmed move into
                # an eager pan. Keep the last safe crop until the next
                # corroborating observation or an actual scene reset.
                current = last_confirmed
                fallback = "stable_source"
                reason = "safe_fallback"
                hold_reframe = True
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
                    hold_reframe=hold_reframe,
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
            core_bounds = _must_keep_core(item.state.bounds, self.config) if item.state is not None else None
            track_requires_recovery = (
                _track_requires_safety_recovery(keyframes, core_bounds, self.config)
                if item.state else False
            )
            if track_requires_recovery and item.state is not None:
                # A bounded pan can finish on the new target while leaving it
                # clipped at the beginning of a target handoff. Checking the
                # final crop (or merely the raw crop rectangle) misses that
                # failure: a face can technically be visible while sitting on
                # the frame edge for several rendered keyframes. Resolve the
                # declared safe hierarchy as static, evidence-backed crops;
                # never escape to a candidate-wide blurred presentation.
                resolved = False
                for fallback_kind, fallback_crop in _track_safety_fallbacks(
                    item, observations, result, source_width, source_height, self.config,
                ):
                    candidate_keyframes, candidate_motion = _crop_track(
                        _reset_atomic_crop(item, fallback_crop), incoming_motion, True, self.config,
                    )
                    if (
                        _track_minimum_safe_containment(
                            candidate_keyframes, core_bounds, self.config,
                        )
                        < self.config.minimum_target_containment
                    ):
                        continue
                    keyframes, motion = candidate_keyframes, candidate_motion
                    crop = keyframes[-1].crop
                    geometry, protected, containment = _geometry_for(
                        item, crop, observations, source_width, source_height,
                    )
                    fallback = "stable_source" if fallback_kind == "last_safe" else "wider_crop"
                    diagnostics.append(f"CROP_FALLBACK_{fallback_kind.upper()}:{item.output.start_frame}")
                    if fallback_kind == "target":
                        # Retain the existing diagnostic name for persisted
                        # audit tooling while making its safe-area meaning
                        # explicit above.
                        diagnostics.append(f"CROP_RESET_FOR_TARGET_TRACK_SAFETY:{item.output.start_frame}")
                    resolved = True
                    break
                if not resolved:
                    # Preserve the target crop and let the existing assessed
                    # composition quality report expose the real impossible
                    # geometry. A fit/blur frame would hide the failure and
                    # make the candidate look falsely usable.
                    diagnostics.append(f"TARGET_TRACK_SAFE_AREA_UNRESOLVED:{item.output.start_frame}")
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
                target_bounds=core_bounds,
                protected_regions=protected,
                geometry=geometry.model_copy(update={"source_crop": crop}),
                crop_keyframes=keyframes,
                movement_reason=item.reason,
                punch_in=punch,
                easing_id="ease_in_out" if len(keyframes) > 1 else "none",
                evidence_refs=item.evidence_refs,
                fallback=fallback,
                camera_mode=item.camera_mode,
                camera_phase=item.camera_phase,
                subject_lock_state=item.subject_lock_state,
                movement_explanation=item.movement_explanation,
            ))
            previous_end = item.output.end_frame
        return tuple(result), tuple(diagnostics)


def _calibrate_layout_timeline(
    atomic: Sequence[_AtomicState], config: CompositionPlannerConfig,
) -> tuple[tuple[_AtomicState, ...], tuple[str, ...]]:
    """Remove short visible layout fragments only when the replacement is safe.

    Target/evidence identity stays attached to its original interval.  This
    pass changes presentation geometry only, and only when the adopted crop
    contains every protected observation of that interval.
    """

    if len(atomic) < 2:
        return tuple(atomic), ()
    calibrated = list(atomic)
    diagnostics: list[str] = []

    # A -> B -> A is perceived as a flash even when B is evidence-backed.  A
    # full-frame A crop can safely present B without discarding B's evidence.
    index = 1
    while index < len(calibrated) - 1:
        run_start = index
        run_layout = calibrated[index].layout
        run_end = index + 1
        while run_end < len(calibrated) and calibrated[run_end].layout == run_layout:
            run_end += 1
        if run_end >= len(calibrated):
            break
        left, right = calibrated[run_start - 1], calibrated[run_end]
        duration = calibrated[run_end - 1].output.end_frame - calibrated[run_start].output.start_frame
        if (
            left.output.end_frame == calibrated[run_start].output.start_frame
            and calibrated[run_end - 1].output.end_frame == right.output.start_frame
            and left.layout == right.layout != run_layout
            and duration < config.minimum_layout_dwell_frames
        ):
            donor = max((left, right), key=lambda item: _crop_area(item.desired_crop))
            replacements = [
                _adopt_layout_if_safe(item, donor) for item in calibrated[run_start:run_end]
            ]
            if all(item is not None for item in replacements):
                calibrated[run_start:run_end] = [
                    item for item in replacements if item is not None
                ]
                diagnostics.append(
                    f"SHORT_LAYOUT_ISLAND_REMOVED:{calibrated[run_start].output.start_frame}"
                )
        index = run_end

    # Tiny leading/trailing fragments are handled separately because they do
    # not form an A -> B -> A triple.
    edge_pairs = ((0, 1), (len(calibrated) - 1, len(calibrated) - 2))
    for edge_index, neighbor_index in edge_pairs:
        edge = calibrated[edge_index]
        neighbor = calibrated[neighbor_index]
        duration = edge.output.end_frame - edge.output.start_frame
        if edge.layout == neighbor.layout or duration >= config.minimum_edge_fragment_frames:
            continue
        replacement = _adopt_layout_if_safe(edge, neighbor)
        if replacement is not None:
            calibrated[edge_index] = replacement
            diagnostics.append(f"EDGE_LAYOUT_FRAGMENT_REMOVED:{edge.output.start_frame}")

    # Detect bursts in local time rather than relying only on a per-minute
    # average.  Collapse a burst only when one of its existing crops safely
    # contains every protected target across the window.
    start = 0
    while start < len(calibrated) - 1:
        end = start + 1
        while (
            end < len(calibrated)
            and calibrated[end].output.end_frame - calibrated[start].output.start_frame
            <= config.local_burst_window_frames
        ):
            end += 1
        window = calibrated[start:end]
        switches = sum(left.layout != right.layout for left, right in zip(window, window[1:]))
        if switches > config.maximum_local_layout_switches:
            donors = sorted(window, key=lambda item: _crop_area(item.desired_crop), reverse=True)
            donor = next(
                (
                    candidate for candidate in donors
                    if all(_adopt_layout_if_safe(item, candidate) is not None for item in window)
                ),
                None,
            )
            if donor is not None:
                calibrated[start:end] = [
                    _adopt_layout_if_safe(item, donor) or item for item in window
                ]
                diagnostics.append(
                    f"LOCAL_LAYOUT_BURST_CALMED:{window[0].output.start_frame}-{window[-1].output.end_frame}"
                )
                start = end
                continue
        start += 1
    return tuple(calibrated), tuple(diagnostics)


def _lock_scene_layout_families(
    atomic: Sequence[_AtomicState],
    cuts: frozenset[int],
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig,
) -> tuple[tuple[_AtomicState, ...], tuple[str, ...]]:
    """Choose presentation family once per source scene, then only track.

    Sparse observations remain evidence for target position, not permission to
    oscillate between unrelated layouts.  Gaps inherit the scene family and a
    calm nearest target crop without inheriting that target's semantic claim.
    """

    if not atomic:
        return (), ()
    groups: list[list[_AtomicState]] = []
    current: list[_AtomicState] = []
    for item in atomic:
        boundary = bool(current) and (
            item.output.start_frame in cuts
            or current[-1].output.end_frame != item.output.start_frame
        )
        if boundary:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)

    groups = _merge_continuous_scene_families(groups)

    result: list[_AtomicState] = []
    diagnostics: list[str] = []
    scene_groups: list[list[_AtomicState]] = []
    for scene in groups:
        sustained = (
            [scene]
            if _continuity_family(scene) in {"conversation", "facecam_split"}
            else _split_sustained_family_runs(scene, config)
        )
        scene_groups.extend(sustained)
        for group in sustained[1:]:
            diagnostics.append(
                f"SUSTAINED_TARGET_FAMILY_CHANGE:{group[0].output.start_frame}"
            )
    for scene in scene_groups:
        evidence = [item for item in scene if item.state is not None]
        if not evidence:
            result.extend(scene)
            continue
        family = _scene_layout_family(evidence)
        family_evidence = _family_anchor_evidence(evidence, family)
        if family == LayoutFamily.SPLIT:
            scene_crops = [
                _split_facecam_crop(
                    item.state, source_width, source_height, config,
                )
                for item in family_evidence
                if item.layout == LayoutFamily.SPLIT and item.state is not None
            ]
        else:
            # Pick the scene family from the protected target geometry, not
            # from an observation-local editorial punch-in.  A punch-in may
            # narrow the atomic crop enough to turn the whole scene into a
            # false fit-background fallback; scene-locked composition permits
            # tracking inside the chosen family, not observation-local zooms.
            scene_crops = [
                _target_crop(
                    item.state.bounds, source_width, source_height, config,
                )[0]
                for item in family_evidence if item.state is not None
            ]
        if not scene_crops:
            result.extend(scene)
            continue

        width = max(item.width for item in scene_crops)
        height = max(item.height for item in scene_crops)
        if family != LayoutFamily.SPLIT:
            width, height = _aspect_locked_size(
                width, height, source_width, source_height,
            )
        family_items: list[_AtomicState] = []
        suppressed_punch = any(item.punch_event is not None for item in scene)
        for index, item in enumerate(scene):
            anchor = (
                item.state
                if item.state is not None and _state_matches_family(item, family)
                else _nearest_scene_state(
                    scene, index,
                    layout=LayoutFamily.SPLIT if family == LayoutFamily.SPLIT else None,
                    targets=(
                        frozenset({
                            AttentionTarget.SPEAKER,
                            AttentionTarget.SUBJECT,
                            AttentionTarget.GROUP,
                        })
                        if family == LayoutFamily.STABLE_SPEAKER else None
                    ),
                )
            )
            assert anchor is not None
            if family == LayoutFamily.STABLE_SPEAKER and anchor.decision.target == AttentionTarget.GROUP:
                speaker_reference = _nearest_scene_state(
                    scene,
                    index,
                    targets=frozenset({AttentionTarget.SPEAKER, AttentionTarget.SUBJECT}),
                )
                if speaker_reference is not None:
                    anchor = _speaker_anchor_within_group(anchor, speaker_reference)
            # The state machine may deliberately hold a prior actual crop
            # while it waits for a second observation to corroborate a large
            # same-target move.  Family locking must not look ahead to a
            # later target and undo that safety decision.
            crop = (
                item.desired_crop
                if item.hold_reframe
                else _split_facecam_crop(
                    anchor, source_width, source_height, config,
                )
                if family == LayoutFamily.SPLIT else
                _centered_rect(anchor.bounds, width, height)
            )
            retained_state = (
                item.state
                if item.state is not None and _state_matches_family(item, family)
                else None
            )
            family_items.append(replace(
                item,
                state=retained_state,
                layout=family,
                desired_crop=crop,
                fallback=(
                    item.fallback
                    if retained_state is not None and item.fallback != "fit_background"
                    else "stable_source" if retained_state is None else "none"
                ),
                target=(item.target if retained_state is not None else AttentionTarget.STABLE_SOURCE),
                target_ref=(item.target_ref if retained_state is not None else None),
                confidence=(item.confidence if retained_state is not None else 0),
                evidence_refs=(item.evidence_refs if retained_state is not None else ()),
                punch_event=None,
                reason=(
                    "target_acquired"
                    if retained_state is not None and item.reason == "editorial_punch_in"
                    else "none"
                    if retained_state is not None and item.reason == "punch_out"
                    else item.reason
                    if retained_state is not None
                    else "scene_reset" if item.reason == "scene_reset" else "safe_fallback"
                ),
            ))

        if any(
            item.state is not None
            and _containment(item.state.bounds, item.desired_crop) < config.minimum_target_containment
            for item in family_items
        ):
            # Keep the scene/target-aware crop rather than escaping to a
            # candidate-wide fit/blur frame.  The quality report below owns
            # an impossible protected-target combination as a blocker.
            diagnostics.append(
                f"SCENE_TARGET_CONTAINMENT_UNRESOLVED:{scene[0].output.start_frame}"
            )
        else:
            diagnostics.append(
                f"SCENE_LAYOUT_FAMILY_LOCKED:{scene[0].output.start_frame}:{family.value}"
            )
            if suppressed_punch:
                diagnostics.append(
                    f"SCENE_PUNCH_IN_SUPPRESSED:{scene[0].output.start_frame}"
                )
        result.extend(family_items)
    return tuple(result), tuple(diagnostics)


def _split_sustained_family_runs(
    scene: Sequence[_AtomicState], config: CompositionPlannerConfig,
) -> list[list[_AtomicState]]:
    runs: list[tuple[int, int, LayoutFamily | None, int]] = []
    start = 0
    while start < len(scene):
        family = scene[start].layout if scene[start].state is not None else None
        end = start + 1
        while end < len(scene):
            candidate = scene[end].layout if scene[end].state is not None else None
            if candidate != family:
                break
            end += 1
        duration = scene[end - 1].output.end_frame - scene[start].output.start_frame
        runs.append((start, end, family, duration))
        start = end
    sustained = [
        run for run in runs
        if run[2] is not None and run[3] >= config.minimum_layout_dwell_frames
    ]
    distinct = [
        run for index, run in enumerate(sustained)
        if index == 0 or run[2] != sustained[index - 1][2]
    ]
    if len(distinct) < 2:
        return [list(scene)]
    boundaries = [run[0] for run in distinct[1:]]
    points = [0, *boundaries, len(scene)]
    return [list(scene[left:right]) for left, right in zip(points, points[1:]) if right > left]


def _scene_layout_family(evidence: Sequence[_AtomicState]) -> LayoutFamily:
    if any(item.layout == LayoutFamily.SPLIT for item in evidence):
        return LayoutFamily.SPLIT
    if any(item.target == AttentionTarget.SCREEN for item in evidence):
        return LayoutFamily.SCREEN_PRIORITY
    if any(item.target == AttentionTarget.PRODUCT for item in evidence):
        return LayoutFamily.SCREEN_PRODUCT
    if any(item.target == AttentionTarget.SPEAKER for item in evidence):
        return LayoutFamily.STABLE_SPEAKER
    if any(item.target == AttentionTarget.SUBJECT for item in evidence):
        return LayoutFamily.SINGLE_SUBJECT
    if any(item.target == AttentionTarget.GROUP for item in evidence):
        return LayoutFamily.WIDE_GROUP
    return max(
        evidence,
        key=lambda item: (item.confidence, _crop_area(item.desired_crop)),
    ).layout


def _nearest_scene_state(
    scene: Sequence[_AtomicState],
    index: int,
    *,
    layout: LayoutFamily | None = None,
    targets: frozenset[AttentionTarget] | None = None,
) -> _TargetState | None:
    candidates = [
        (abs(position - index), position, item.state)
        for position, item in enumerate(scene)
        if item.state is not None
        and (layout is None or item.layout == layout)
        and (targets is None or item.target in targets)
    ]
    return min(candidates, default=(0, 0, None))[2]


def _aspect_locked_size(
    width: float, height: float, source_width: int, source_height: int,
) -> tuple[float, float]:
    normalized_ratio = (9 / 16) / (source_width / source_height)
    required_height = max(height, width / max(normalized_ratio, 1e-9))
    return normalized_ratio * required_height, required_height


def _split_facecam_crop(
    state: _TargetState,
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig,
) -> NormalizedRect:
    bounds = state.bounds
    bounds_aspect = (
        bounds.width * source_width
        / max(bounds.height * source_height, 1e-9)
    )
    if (
        state.decision.target in {AttentionTarget.SPEAKER, AttentionTarget.SUBJECT}
        and abs(bounds_aspect - 16 / 9) <= 0.03
    ):
        # Candidate-scoped handoff geometry identifies the webcam panel, not
        # merely the face inside it. Preserve that exact source region; the
        # renderer performs the final aspect-preserving pane fill.
        return bounds
    source_aspect = source_width / source_height
    pane_aspect = (9 / 16) / SPLIT_FACE_CAM_RATIO
    normalized_ratio = pane_aspect / source_aspect
    usable = 1 - config.target_margin_ratio
    height = max(
        0.22,
        bounds.height / usable,
        bounds.width / max(normalized_ratio * usable, 1e-9),
    )
    height = min(1.0, height, 1 / max(normalized_ratio, 1e-9))
    return _centered_rect(bounds, normalized_ratio * height, height)


def _continuity_family(scene: Sequence[_AtomicState]) -> str | None:
    evidence = [item for item in scene if item.state is not None]
    if not evidence:
        return None
    if all(item.layout == LayoutFamily.SPLIT for item in evidence):
        return "facecam_split"
    if all(
        item.target in {
            AttentionTarget.SPEAKER,
            AttentionTarget.SUBJECT,
            AttentionTarget.GROUP,
            AttentionTarget.REACTION,
        }
        for item in evidence
    ):
        return "conversation"
    return None


def _merge_continuous_scene_families(
    groups: Sequence[Sequence[_AtomicState]],
) -> list[list[_AtomicState]]:
    """Carry a presentation family across ordinary camera cuts and gaps."""

    merged: list[list[_AtomicState]] = []
    for raw in groups:
        scene = list(raw)
        family = _continuity_family(scene)
        if not merged:
            merged.append(scene)
            continue
        previous_family = _continuity_family(merged[-1])
        if (
            family == previous_family
            and family in {"conversation", "facecam_split"}
        ) or (
            family is None
            and previous_family in {"conversation", "facecam_split"}
        ) or (
            previous_family is None
            and family in {"conversation", "facecam_split"}
        ):
            merged[-1].extend(scene)
        else:
            merged.append(scene)
    return merged


def _family_anchor_evidence(
    evidence: Sequence[_AtomicState], family: LayoutFamily,
) -> list[_AtomicState]:
    if family == LayoutFamily.STABLE_SPEAKER:
        anchors = [
            item for item in evidence
            if item.target in {AttentionTarget.SPEAKER, AttentionTarget.SUBJECT}
        ]
        return anchors or list(evidence)
    if family == LayoutFamily.SINGLE_SUBJECT:
        anchors = [item for item in evidence if item.target == AttentionTarget.SUBJECT]
        return anchors or list(evidence)
    if family == LayoutFamily.SPLIT:
        anchors = [item for item in evidence if item.layout == LayoutFamily.SPLIT]
        return anchors or list(evidence)
    return list(evidence)


def _state_matches_family(item: _AtomicState, family: LayoutFamily) -> bool:
    if item.state is None:
        return False
    if family == LayoutFamily.STABLE_SPEAKER:
        return item.target in {AttentionTarget.SPEAKER, AttentionTarget.SUBJECT}
    if family == LayoutFamily.SINGLE_SUBJECT:
        return item.target == AttentionTarget.SUBJECT
    if family == LayoutFamily.SPLIT:
        return item.layout == LayoutFamily.SPLIT
    return True


def _speaker_anchor_within_group(
    group: _TargetState,
    speaker_reference: _TargetState,
) -> _TargetState:
    """Project the proven speaker side into a wide conversational camera shot."""

    group_center_x = group.bounds.x + group.bounds.width / 2
    group_center_y = group.bounds.y + group.bounds.height / 2
    speaker_center_x = speaker_reference.bounds.x + speaker_reference.bounds.width / 2
    side = -1 if speaker_center_x < group_center_x else 1
    center_x = group_center_x + side * group.bounds.width * 0.30
    width = speaker_reference.bounds.width
    height = speaker_reference.bounds.height
    x = _clamp(center_x - width / 2, 0, 1 - width)
    y = _clamp(group_center_y - height / 2, 0, 1 - height)
    return replace(
        speaker_reference,
        bounds=NormalizedRect(
            x=round(x, 8), y=round(y, 8),
            width=round(width, 8), height=round(height, 8),
        ),
    )


def _adopt_layout_if_safe(item: _AtomicState, donor: _AtomicState) -> _AtomicState | None:
    if item.punch_event is not None:
        return None
    crop = donor.desired_crop
    if (
        item.state is not None
        and item.desired_crop != crop
        and any(observation.protected for observation in item.state.observations)
    ):
        # Containment alone is not permission to discard an evidence-backed
        # close framing decision.  Calibration may remove a layout flash when
        # geometry is already equivalent, but protected evidence owns its crop.
        return None
    if item.state is not None and any(
        _containment(observation.bounds, crop) < 0.98
        for observation in item.state.observations if observation.protected
    ):
        return None
    fallback: CompositionFallback = (
        "fit_background" if donor.layout == LayoutFamily.FIT_BACKGROUND else "stable_source"
    )
    return replace(
        item,
        layout=donor.layout,
        desired_crop=crop,
        fallback=fallback if item.fallback == "none" else item.fallback,
        reason="safe_fallback" if item.reason == "none" else item.reason,
    )


def _crop_area(crop: NormalizedRect) -> float:
    return crop.width * crop.height


def _plan_shot_virtual_camera(
    atomic: Sequence[_AtomicState],
    cuts: frozenset[int],
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig,
) -> tuple[tuple[_AtomicState, ...], tuple[str, ...]]:
    """Resolve evidence samples into a shot-local, static-first camera plan.

    This is intentionally after target selection and before renderer keyframes:
    target observations retain their role as evidence, while crop decisions are
    made with lookahead over a whole source shot.  The greedy interval
    intersection is a deterministic minimum partition for ordered feasible
    ranges: a hold is extended until the next constraint makes it impossible.
    """

    if not atomic:
        return (), ()
    shots: list[list[_AtomicState]] = [[]]
    for item in atomic:
        if shots[-1] and (
            item.output.start_frame in cuts
            or _starts_confirmed_target_handoff(shots[-1][-1], item)
        ):
            shots.append([])
        shots[-1].append(item)
    planned: list[_AtomicState] = []
    diagnostics: list[str] = []
    for shot in shots:
        resolved, shot_diagnostics = _plan_one_source_shot(
            tuple(shot), source_width, source_height, config,
        )
        planned.extend(resolved)
        diagnostics.extend(shot_diagnostics)
    return tuple(planned), tuple(diagnostics)


def _starts_confirmed_target_handoff(
    previous: _AtomicState,
    current: _AtomicState,
) -> bool:
    """Keep an admitted subject handoff out of the previous camera hold.

    The state machine has already applied confidence, editorial and hysteresis
    rules before it marks ``target_switch``.  Treating that event as ordinary
    in-shot evidence used to replace the new state with the old lock, making
    the composed crop look safe only because it was measured against the wrong
    person.  A handoff is a controlled cut/reset, never a pan through a new
    face; quick unconfirmed proposals still never reach this point.
    """

    return (
        current.reason == "target_switch"
        and previous.state is not None
        and current.state is not None
        and previous.state.key != current.state.key
    )


def _plan_one_source_shot(
    shot: Sequence[_AtomicState],
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig,
) -> tuple[tuple[_AtomicState, ...], tuple[str, ...]]:
    # Split/facecam and screen/product layouts own a different fixed geometry
    # contract.  They are already shot-local and must not be coerced through
    # the human virtual-camera policy.
    target_kinds = {item.target for item in shot if item.state is not None}
    if (
        any(item.layout == LayoutFamily.SPLIT for item in shot)
        or len(target_kinds) > 1
        or any(kind in {AttentionTarget.SCREEN, AttentionTarget.PRODUCT, AttentionTarget.OBJECT} for kind in target_kinds)
    ):
        return tuple(replace(item, hold_reframe=True) for item in shot), ()
    evidence = [item for item in shot if item.state is not None and item.confidence >= config.exit_confidence]
    if not evidence:
        return tuple(replace(
            item,
            subject_lock_state="EVIDENCE_GAP",
            camera_mode="GROUP" if item.layout == LayoutFamily.WIDE_GROUP else "STATIONARY",
            camera_phase="HOLD",
            hold_reframe=True,
        ) for item in shot), (f"SHOT_EVIDENCE_GAP:{shot[0].output.start_frame}",)

    # First sufficiently supported identity wins this shot.  Confirmed target
    # handoffs were split before this point, while short or unconfirmed
    # proposals remain inside the original shot and therefore cannot trigger a
    # camera chase.
    lock = evidence[0].state
    assert lock is not None
    locked_ref = lock.target_ref
    distinct_refs = {item.state.target_ref for item in evidence if item.state is not None}
    group_mode = len(distinct_refs) > 1
    lock_states = [item for item in evidence if item.state is not None and item.state.target_ref == locked_ref]
    fallback_state = lock_states[0].state
    assert fallback_state is not None

    template = _vertical_fill_crop(source_width, source_height)
    constraints: list[tuple[float, float, float, float] | None] = []
    normalized: list[_AtomicState] = []
    for item in shot:
        state = item.state
        if state is None:
            normalized.append(replace(
                item, state=fallback_state, target=fallback_state.decision.target,
                target_ref=locked_ref, confidence=fallback_state.confidence,
                subject_lock_state="TEMPORARILY_OCCLUDED", hold_reframe=True,
            ))
            constraints.append(None)
            continue
        if state.target_ref != locked_ref:
            # Fast dialogue gets one stable group framing, not a pan ping-pong.
            normalized.append(replace(
                item, state=fallback_state, target=fallback_state.decision.target,
                target_ref=locked_ref, confidence=fallback_state.confidence,
                layout=LayoutFamily.WIDE_GROUP if group_mode else item.layout,
                subject_lock_state="SWITCH_PENDING", hold_reframe=True,
            ))
            constraints.append(None)
            continue
        core = _must_keep_core(state.bounds, config)
        feasible = _feasible_crop_center_range(core, template, config)
        normalized.append(replace(
            item,
            subject_lock_state=(
                "SWITCH_CONFIRMED" if item.reason == "target_switch" else "LOCKED"
            ),
            hold_reframe=True,
        ))
        constraints.append(feasible)

    if not any(item is not None for item in constraints):
        return tuple(normalized), (f"SHOT_EVIDENCE_GAP:{shot[0].output.start_frame}",)
    if any(item is not None and item[0] > item[1] or item is not None and item[2] > item[3] for item in constraints):
        blocked = tuple(replace(
            item, camera_mode="BLOCKED", camera_phase="HOLD",
            movement_explanation="must_keep_core cannot fit inside the approved safe crop",
        ) for item in normalized)
        return blocked, (f"SHOT_CORE_GEOMETRY_BLOCKED:{shot[0].output.start_frame}",)

    # Partition into the fewest stationary holds by maintaining the running
    # intersection.  Missing evidence represents temporary occlusion and does
    # not manufacture a new crop constraint.
    holds: list[tuple[int, int, tuple[float, float, float, float]]] = []
    start = 0
    current: tuple[float, float, float, float] | None = None
    for index, feasible in enumerate(constraints):
        if feasible is None:
            continue
        candidate = feasible if current is None else _intersect_center_ranges(current, feasible)
        if candidate[0] > candidate[1] or candidate[2] > candidate[3]:
            assert current is not None
            holds.append((start, index, current))
            start = index
            current = feasible
        else:
            current = candidate
    if current is not None:
        holds.append((start, len(normalized), current))

    crops: list[NormalizedRect | None] = [None] * len(normalized)
    for start, end, feasible in holds:
        anchor_centers = [
            _rect_values(normalized[index].desired_crop)[:2]
            for index in range(start, end)
            if constraints[index] is not None
        ]
        desired_x = median(value[0] for value in anchor_centers)
        desired_y = median(value[1] for value in anchor_centers)
        center_x = _clamp(desired_x, feasible[0], feasible[1])
        # Talking-head default is horizontal only.  If its static Y range is
        # empty it was handled above; otherwise the shot keeps one Y anchor.
        center_y = _clamp(desired_y, feasible[2], feasible[3])
        crop = _crop_at_center(template, center_x, center_y)
        for index in range(start, end):
            crops[index] = crop

    result: list[_AtomicState] = []
    previous_crop: NormalizedRect | None = None
    for index, (item, crop) in enumerate(zip(normalized, crops)):
        assert crop is not None
        mode: Literal["STATIONARY", "PAN_ONLY", "TRACKING", "GROUP", "BLOCKED"] = (
            "GROUP" if group_mode else "STATIONARY"
        )
        move_from: NormalizedRect | None = None
        if previous_crop is not None:
            distance_x = abs((crop.x + crop.width / 2) - (previous_crop.x + previous_crop.width / 2))
            distance_y = abs((crop.y + crop.height / 2) - (previous_crop.y + previous_crop.height / 2))
            if distance_x <= config.camera_dead_zone_ratio and distance_y <= config.camera_dead_zone_ratio:
                crop = previous_crop
            elif distance_y <= 1e-7 and distance_x <= config.maximum_pan_episode_distance:
                if item.output.end_frame - item.output.start_frame > config.camera_move_frames:
                    mode = "PAN_ONLY"
                    move_from = previous_crop
                else:
                    item = replace(
                        item, camera_mode="BLOCKED", camera_phase="HOLD",
                        movement_explanation="edit-map interval is too short for a comfortable pan",
                    )
                    result.append(replace(item, desired_crop=previous_crop, hold_reframe=True))
                    previous_crop = previous_crop
                    continue
            elif distance_y > 1e-7 or distance_x > config.maximum_pan_episode_distance:
                item = replace(
                    item, camera_mode="BLOCKED", camera_phase="HOLD",
                    movement_explanation="no safe static or bounded pan-only solution for must_keep_core",
                )
                result.append(replace(item, desired_crop=crop, hold_reframe=True))
                previous_crop = crop
                continue
        phase: Literal["HOLD", "MOVE"] = "MOVE" if move_from is not None else "HOLD"
        lock_state = item.subject_lock_state
        result.append(replace(
            item, desired_crop=crop, hold_reframe=move_from is None,
            camera_mode=mode, camera_phase=phase, move_from=move_from,
            movement_explanation=("must_keep_core left the inner comfort zone" if move_from is not None else None),
            subject_lock_state=lock_state if lock_state != "EVIDENCE_GAP" else "ACQUIRE",
        ))
        previous_crop = crop
    diagnostics = [f"SHOT_SUBJECT_LOCK:{shot[0].output.start_frame}:{locked_ref}"]
    diagnostics.append(
        f"SHOT_STATIC_FIRST_{'PARTITIONED' if len(holds) > 1 else 'STATIONARY'}:{shot[0].output.start_frame}"
    )
    if group_mode:
        diagnostics.append(f"SHOT_GROUP_LOCK_NO_PING_PONG:{shot[0].output.start_frame}")
    return tuple(result), tuple(diagnostics)


def _must_keep_core(bounds: NormalizedRect, config: CompositionPlannerConfig) -> NormalizedRect:
    """Derive a conservative face/head/shoulder core from verified person evidence.

    Until face landmarks are present in the shared observation contract, this
    is deliberately smaller than the full person box.  The full upper body is
    retained as a soft geometry region and cannot force camera chatter.
    """

    width = bounds.width * config.core_width_ratio
    height = bounds.height * config.core_height_ratio
    return NormalizedRect(
        x=round(bounds.x + (bounds.width - width) / 2, 8),
        y=round(bounds.y + bounds.height * 0.03, 8),
        width=round(width, 8), height=round(height, 8),
    )


def _feasible_crop_center_range(
    core: NormalizedRect,
    crop: NormalizedRect,
    config: CompositionPlannerConfig,
) -> tuple[float, float, float, float]:
    safe_half_x = crop.width * (0.5 - config.target_margin_ratio)
    safe_half_y = crop.height * (0.5 - config.target_margin_ratio)
    return (
        max(crop.width / 2, core.x + core.width - safe_half_x),
        min(1 - crop.width / 2, core.x + safe_half_x),
        max(crop.height / 2, core.y + core.height - safe_half_y),
        min(1 - crop.height / 2, core.y + safe_half_y),
    )


def _intersect_center_ranges(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return max(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), min(left[3], right[3])


def _crop_at_center(template: NormalizedRect, center_x: float, center_y: float) -> NormalizedRect:
    return NormalizedRect(
        x=round(_clamp(center_x - template.width / 2, 0, 1 - template.width), 8),
        y=round(_clamp(center_y - template.height / 2, 0, 1 - template.height), 8),
        width=template.width, height=template.height,
    )


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
    if required_height <= 1 and normalized_ratio <= 1:
        height = min(1.0, required_height)
        width = normalized_ratio * height
        return _centered_rect(bounds, width, height), "none"
    height = min(1.0, 1 / max(normalized_ratio, 1e-9))
    width = min(1.0, normalized_ratio * height)
    return _centered_rect(bounds, width, height), "wider_crop"


def _same_target_reframe_requires_confirmation(
    candidate: _TargetState,
    reference: _TargetState,
    config: CompositionPlannerConfig,
) -> bool:
    """Return true when a same-target observation would visibly jump crop."""

    return _target_state_distance(candidate, reference) > config.same_target_reframe_distance


def _is_explicit_target_handoff(
    current: _TargetState,
    candidate: _TargetState,
    interval: OutputInterval,
) -> bool:
    """Whether current target evidence has been explicitly superseded.

    The state machine's hold/cooldown protects two competing candidates inside
    the *same* target decision. Once that decision no longer covers the active
    output interval, preserving it would turn a continuity guard into a stale
    crop. A replacement remains subject to the normal enter-confidence
    threshold before this helper is reached.
    """

    return (
        current.decision.decision_id != candidate.decision.decision_id
        and not current.decision.output.contains(interval)
        and candidate.decision.output.contains(interval)
    )


def _target_state_distance(left: _TargetState, right: _TargetState) -> float:
    left_x = left.bounds.x + left.bounds.width / 2
    left_y = left.bounds.y + left.bounds.height / 2
    right_x = right.bounds.x + right.bounds.width / 2
    right_y = right.bounds.y + right.bounds.height / 2
    return max(abs(left_x - right_x), abs(left_y - right_y))


def _reframe_confirmation_is_timely(
    candidate: _TargetState,
    pending: _TargetState,
    config: CompositionPlannerConfig,
) -> bool:
    return (
        max(item.frame for item in candidate.observations)
        - max(item.frame for item in pending.observations)
        <= config.same_target_reframe_confirmation_window_frames
    )


def _calm_fallback_crop(
    source_width: int,
    source_height: int,
) -> tuple[NormalizedRect, LayoutFamily, Literal["stable_source"]]:
    """Return a calm *actual* 9:16 crop when evidence is temporarily sparse."""

    return _vertical_fill_crop(source_width, source_height), LayoutFamily.WIDE_GROUP, "stable_source"


def _vertical_fill_crop(source_width: int, source_height: int) -> NormalizedRect:
    """A centred crop with the source-space aspect ratio of the 9:16 canvas."""

    output_aspect = 9 / 16
    source_aspect = source_width / source_height
    if source_aspect >= output_aspect:
        return NormalizedRect(
            x=round((1 - output_aspect / source_aspect) / 2, 8),
            y=0,
            width=round(output_aspect / source_aspect, 8),
            height=1,
        )
    return NormalizedRect(
        x=0,
        y=round((1 - source_aspect / output_aspect) / 2, 8),
        width=1,
        height=round(source_aspect / output_aspect, 8),
    )


def _track_safety_fallbacks(
    item: _AtomicState,
    observations: Sequence[TargetObservation],
    completed: Sequence[CompositionSegmentPlan],
    source_width: int,
    source_height: int,
    config: CompositionPlannerConfig,
) -> tuple[tuple[str, NormalizedRect], ...]:
    """Return the allowed short-form recovery crops in strict priority order.

    Each option remains an actual 9:16 source crop.  ``last_safe`` is only
    admitted when it already contains the newly selected target in its safe
    area, so it cannot resurrect the old on-screen character after a cut.
    """

    if item.state is None:
        return ()
    options: list[tuple[str, NormalizedRect]] = [("target", item.desired_crop)]
    fill = _vertical_fill_crop(source_width, source_height)
    human_targets = {
        AttentionTarget.SPEAKER,
        AttentionTarget.SUBJECT,
        AttentionTarget.REACTION,
        AttentionTarget.GROUP,
    }
    if item.target in human_targets:
        options.append((
            "wider_person",
            _centered_rect(item.state.bounds, fill.width, fill.height),
        ))
        group_bounds = _visible_human_group_bounds(item, observations)
        if group_bounds is not None:
            options.append((
                "group",
                _centered_rect(group_bounds, fill.width, fill.height),
            ))
    last_safe = next((
        segment.crop for segment in reversed(completed)
        if segment.target_ref == item.target_ref
        and _safe_area_containment(item.state.bounds, segment.crop, config) > 0
    ), None)
    if last_safe is not None:
        options.append(("last_safe", last_safe))
    unique: list[tuple[str, NormalizedRect]] = []
    for option in options:
        if option[1] not in {crop for _, crop in unique}:
            unique.append(option)
    return tuple(unique)


def _visible_human_group_bounds(
    item: _AtomicState,
    observations: Sequence[TargetObservation],
) -> NormalizedRect | None:
    human_targets = {
        AttentionTarget.SPEAKER,
        AttentionTarget.SUBJECT,
        AttentionTarget.REACTION,
        AttentionTarget.GROUP,
    }
    members = [
        observation.bounds for observation in observations
        if item.output.start_frame <= observation.frame < item.output.end_frame
        and observation.protected
        and observation.effective_confidence >= 0.48
        and observation.target in human_targets
    ]
    return _union_rect(tuple(members)) if members else None


def _layout_for(
    state: _TargetState,
    fallback: Literal["none", "wider_crop"],
) -> LayoutFamily:
    allowed = state.decision.allowed_layouts
    preferred: tuple[LayoutFamily, ...]
    if LayoutFamily.SPLIT in allowed:
        preferred = (LayoutFamily.SPLIT, *allowed)
    elif state.decision.target == AttentionTarget.SPEAKER:
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
    if item.move_from is not None:
        # A virtual-camera move is one bounded, planned episode.  The five
        # eased samples are renderer interpolation anchors, not detector
        # samples, so the path cannot resume a continuous tracker chase.
        start = _rect_values(item.move_from)
        available = item.output.end_frame - item.output.start_frame - 1
        if available < 1:
            return (CompositionCropKeyframe(
                frame=item.output.start_frame, crop=item.desired_crop,
                reason="safe_fallback", camera_phase="HOLD",
            ),), _MotionState(values=desired, velocity=(0, 0, 0, 0))
        duration = min(available, max(config.camera_move_frames, 2))
        if duration < config.camera_move_frames:
            # The current edit-map partition leaves no legal interval for a
            # comfortable pan.  Retain a static crop and let quality expose
            # the geometry instead of emitting an invisible cross-cut chase.
            return (CompositionCropKeyframe(
                frame=item.output.start_frame, crop=item.move_from,
                reason="safe_fallback", camera_phase="HOLD",
            ),), _MotionState(values=start, velocity=(0, 0, 0, 0))
        frames: list[CompositionCropKeyframe] = []
        previous = start
        previous_velocity = 0.0
        for ratio in (0.0, 0.15625, 0.5, 0.84375, 1.0):
            frame = item.output.start_frame + round(duration * ratio)
            values = tuple(left + (right - left) * ratio for left, right in zip(start, desired))
            rect = _rect_from_values(values)  # type: ignore[arg-type]
            span = max(1, frame - (frames[-1].frame if frames else item.output.start_frame))
            velocity = max(abs(left - right) for left, right in zip(values, previous)) / span
            acceleration = abs(velocity - previous_velocity) / span
            frames.append(CompositionCropKeyframe(
                frame=frame, crop=rect, velocity_per_frame=round(velocity, 9),
                acceleration_per_frame_sq=round(acceleration, 9),
                reason=_keyframe_reason(item.reason), camera_phase="MOVE",
            ))
            previous = values
            previous_velocity = velocity
        unique = tuple(item for index, item in enumerate(frames) if index == 0 or item.frame != frames[index - 1].frame)
        return unique, _MotionState(values=desired, velocity=(0, 0, 0, 0))
    if item.hold_reframe:
        return (CompositionCropKeyframe(
            frame=item.output.start_frame, crop=item.desired_crop,
            reason="scene_reset" if reset else "static", camera_phase="HOLD",
        ),), _MotionState(values=desired, velocity=(0, 0, 0, 0))
    if previous is None or reset:
        state = _MotionState(values=desired, velocity=(0, 0, 0, 0))
        return (CompositionCropKeyframe(
            frame=item.output.start_frame, crop=item.desired_crop,
            reason="scene_reset" if reset else "static", camera_phase="HOLD",
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
            reason=reason, camera_phase="MOVE",
        ))
        previous_speed = speed
        if values == desired and not any(velocity):
            break
    if not frames:
        frames.append(CompositionCropKeyframe(
            frame=item.output.start_frame, crop=_rect_from_values(values), reason="static", camera_phase="HOLD",
        ))
    return tuple(frames), _MotionState(values=values, velocity=velocity)


def _geometry_for(
    item: _AtomicState,
    crop: NormalizedRect,
    observations: Sequence[TargetObservation],
    source_width: int,
    source_height: int,
) -> tuple[CompositionGeometryContract, tuple[NormalizedRect, ...], float]:
    content = (
        NormalizedRect(x=0, y=0, width=1, height=SPLIT_FACE_CAM_RATIO)
        if item.layout == LayoutFamily.SPLIT else
        _output_content_bounds(crop, source_width, source_height)
    )
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
    blocked_shots = [item for item in diagnostics if item.startswith("SHOT_CORE_GEOMETRY_BLOCKED:")]
    if blocked_shots:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_TARGET_CLIPPED", severity="blocker",
            measured_value="must_keep_core_uncontainable", threshold=config.minimum_target_containment,
            message="The selected must_keep_core cannot fit inside the approved safe 9:16 crop.",
        ))
    switch_count = sum(item.movement_reason == "target_switch" for item in segments)
    layout_switch_count = sum(
        left.layout != right.layout for left, right in zip(segments, segments[1:])
    )
    fallback_segments = [item for item in segments if item.fallback != "none"]
    punch_count = sum(item.punch_in is not None for item in segments)
    maximum_velocity = max((key.velocity_per_frame for item in segments for key in item.crop_keyframes), default=0)
    maximum_acceleration = max((key.acceleration_per_frame_sq for item in segments for key in item.crop_keyframes), default=0)
    # ``segment.crop`` is the final keyframe.  A pan can therefore finish on
    # target while having clipped it earlier in the same emitted segment.
    # Assess every frozen keyframe and retain the exact segment/frame that
    # makes the quality gate fail; the latter is essential for a safe replay
    # to distinguish impossible source geometry from a stale composition.
    containment_samples = [
        (item.segment_id, keyframe.frame, _containment(item.target_bounds, keyframe.crop))
        for item in segments
        if item.target_bounds is not None
        for keyframe in item.crop_keyframes
    ]
    containments = [value for _, _, value in containment_samples]
    clipped = [sample for sample in containment_samples if sample[2] < config.minimum_target_containment]
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
        clipped_segment_id, clipped_frame, clipped_value = min(clipped, key=lambda item: item[2])
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_TARGET_CLIPPED", severity="blocker",
            segment_id=clipped_segment_id,
            measured_value=round(clipped_value, 7), threshold=config.minimum_target_containment,
            message=(
                "A selected semantic target is not safely contained by its crop "
                f"keyframe at output frame {clipped_frame}."
            ),
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
            message="Low-confidence evidence uses a calm, actual 9:16 stable crop.",
        ))
    if fallback_segments:
        findings.append(CompositionQualityFinding(
            code="COMPOSITION_SAFE_FALLBACK", severity="warning",
            measured_value=len(fallback_segments), threshold=0,
            message="Composition applied the declared wider or stable-crop fallback chain.",
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
            clipped_target_count=len({segment_id for segment_id, _, _ in clipped}),
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


def _reset_atomic_crop(item: _AtomicState, crop: NormalizedRect) -> _AtomicState:
    return _AtomicState(
        output=item.output, state=item.state, layout=item.layout,
        desired_crop=crop, target=item.target, target_ref=item.target_ref,
        confidence=item.confidence, evidence_refs=item.evidence_refs,
        fallback="wider_crop" if item.fallback == "none" else item.fallback,
        reason="safe_fallback", punch_event=None,
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


def _safe_crop_area(
    crop: NormalizedRect,
    config: CompositionPlannerConfig,
) -> NormalizedRect:
    """Return the subject-safe inset of an actual source crop."""

    inset_x = min(crop.width * config.target_margin_ratio, crop.width / 2 - 0.0000005)
    inset_y = min(crop.height * config.target_margin_ratio, crop.height / 2 - 0.0000005)
    return NormalizedRect(
        x=round(crop.x + inset_x, 8),
        y=round(crop.y + inset_y, 8),
        width=round(crop.width - 2 * inset_x, 8),
        height=round(crop.height - 2 * inset_y, 8),
    )


def _safe_area_containment(
    bounds: NormalizedRect,
    crop: NormalizedRect,
    config: CompositionPlannerConfig,
) -> float:
    return _containment(bounds, _safe_crop_area(crop, config))


def _track_minimum_safe_containment(
    keyframes: Sequence[CompositionCropKeyframe],
    bounds: NormalizedRect,
    config: CompositionPlannerConfig,
) -> float:
    """Return the weakest safe-area containment across every crop keyframe."""

    return min((
        _safe_area_containment(bounds, item.crop, config) for item in keyframes
    ), default=1.0)


def _track_requires_safety_recovery(
    keyframes: Sequence[CompositionCropKeyframe],
    bounds: NormalizedRect,
    config: CompositionPlannerConfig,
) -> bool:
    """Protect every moving keyframe while retaining feasible static crops.

    A static evidence-resolved crop is already the best geometry available at
    a source edge or for an unusually wide observation. A moving crop has no
    such exemption: it must keep the protected target inside the safe inset at
    every emitted keyframe, in addition to the existing containment floor.
    """

    raw = min((_containment(bounds, item.crop) for item in keyframes), default=1.0)
    if raw < config.minimum_target_containment:
        return True
    return len(keyframes) > 1 and (
        _track_minimum_safe_containment(keyframes, bounds, config)
        < config.minimum_target_containment
    )


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
