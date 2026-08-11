from __future__ import annotations

"""Phase 7F deterministic editorial motion compiler.

Motion is downstream of the evidence-backed CreativeIntent and the assessed
7C/7D/7E domain plans.  It never invents periodic effects: every emitted event
is tied to one resolved editorial motion decision and a concrete domain target.
"""

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Literal, Sequence

from pydantic import Field, model_validator

from app.creative_contracts import (
    CaptionCuePlan,
    CaptionPlan,
    CompositionPlan,
    CreativeIntent,
    FrozenContract,
    Intensity,
    MOTION_PLAN_SCHEMA_VERSION,
    MotionAnimationBudget,
    MotionDomain,
    MotionEventPlan,
    MotionPlan,
    MotionPurpose,
    MotionQualityFinding,
    MotionQualityMetrics,
    MotionQualityProvenance,
    MotionQualityReport,
    OutputInterval,
    ResolvedMotionEvent,
    SourceBRollPlan,
)


MOTION_PLANNER_VERSION = "7J.1.motion-calibration.1"
CAPABILITY_REGISTRY_VERSION: Literal["7B.capability-registry.1"] = "7B.capability-registry.1"

MotionPrimitive = Literal[
    "static", "fade", "scale", "slide", "dissolve", "crop_translate", "punch_in",
]


class MotionCapability(FrozenContract):
    """A bounded renderer primitive admitted by the frozen 7B decision."""

    primitive_id: MotionPrimitive
    backend_id: Literal["none", "libass", "ffmpeg"]
    domains: tuple[MotionDomain, ...] = Field(min_length=1)
    fallback_primitive_id: Literal["static", "fade"]
    deterministic: Literal[True] = True


class MotionCapabilityRegistry(FrozenContract):
    schema_version: Literal["7B.capability-registry.1"] = CAPABILITY_REGISTRY_VERSION
    selected_caption_backend: Literal["libass"] = "libass"
    tier_2_status: Literal["benchmark_only_unqualified"] = "benchmark_only_unqualified"
    entries: tuple[MotionCapability, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_domain_primitives(self) -> "MotionCapabilityRegistry":
        pairs = [
            (entry.primitive_id, domain)
            for entry in self.entries
            for domain in entry.domains
        ]
        if len(pairs) != len(set(pairs)):
            raise ValueError("motion capability registry contains duplicate domain primitives")
        if not any(entry.primitive_id == "static" for entry in self.entries):
            raise ValueError("motion capability registry requires a static fallback")
        return self

    def resolve(self, primitive: MotionPrimitive, domain: MotionDomain) -> MotionCapability | None:
        return next(
            (
                entry for entry in self.entries
                if entry.primitive_id == primitive and domain in entry.domains
            ),
            None,
        )


PHASE_7B_MOTION_CAPABILITIES = MotionCapabilityRegistry(entries=(
    MotionCapability(
        primitive_id="static", backend_id="none",
        domains=(MotionDomain.CAPTION, MotionDomain.COMPOSITION, MotionDomain.TRANSITION),
        fallback_primitive_id="static",
    ),
    MotionCapability(
        primitive_id="fade", backend_id="libass", domains=(MotionDomain.CAPTION,),
        fallback_primitive_id="static",
    ),
    MotionCapability(
        primitive_id="scale", backend_id="libass", domains=(MotionDomain.CAPTION,),
        fallback_primitive_id="fade",
    ),
    MotionCapability(
        primitive_id="slide", backend_id="libass", domains=(MotionDomain.CAPTION,),
        fallback_primitive_id="fade",
    ),
    MotionCapability(
        primitive_id="dissolve", backend_id="ffmpeg", domains=(MotionDomain.TRANSITION,),
        fallback_primitive_id="static",
    ),
    MotionCapability(
        primitive_id="crop_translate", backend_id="ffmpeg", domains=(MotionDomain.COMPOSITION,),
        fallback_primitive_id="static",
    ),
    MotionCapability(
        primitive_id="punch_in", backend_id="ffmpeg", domains=(MotionDomain.COMPOSITION,),
        fallback_primitive_id="static",
    ),
))


@dataclass(frozen=True, slots=True)
class _IntensityPolicy:
    cooldown_frames: int
    max_concurrent_layers: int
    base_point_limit: int
    points_per_minute: int
    animated_frame_ratio: float
    minimum_animated_frames: int
    maximum_event_frames: int


_INTENSITY_POLICIES: dict[Intensity, _IntensityPolicy] = {
    Intensity.LOW: _IntensityPolicy(60, 1, 3, 4, 0.08, 12, 10),
    Intensity.BALANCED: _IntensityPolicy(36, 2, 6, 8, 0.14, 24, 14),
    Intensity.HIGH: _IntensityPolicy(24, 2, 9, 12, 0.20, 36, 18),
}


@dataclass(frozen=True, slots=True)
class MotionPlannerConfig:
    dense_caption_cps: float = 17.0
    minimum_animated_event_frames: int = 3

    def __post_init__(self) -> None:
        if self.dense_caption_cps <= 0:
            raise ValueError("dense caption threshold must be positive")
        if self.minimum_animated_event_frames < 3:
            raise ValueError("motion events shorter than three frames must not animate")


@dataclass(frozen=True, slots=True)
class _Candidate:
    request: ResolvedMotionEvent
    primitive: MotionPrimitive
    requested_primitive: MotionPrimitive
    output: OutputInterval
    duration_frames: int
    target_ids: tuple[str, ...]
    scale_from: float = 1
    scale_to: float = 1
    translate_x_ratio: float = 0
    translate_y_ratio: float = 0
    opacity_from: float = 1
    opacity_to: float = 1
    reduced_motion: bool = False
    fallback_reason: Literal["reduced_motion", "unsupported_primitive", "short_event"] | None = None


@dataclass(frozen=True, slots=True)
class _Suppression:
    code: Literal[
        "MOTION_COOLDOWN_SUPPRESSED", "MOTION_CONCURRENCY_SUPPRESSED",
        "MOTION_BUDGET_SUPPRESSED", "MOTION_READABILITY_SUPPRESSED",
        "MOTION_DOMAIN_TARGET_MISSING",
    ]
    event_id: str
    measured_value: float | int | str | bool
    threshold: float | int | str | bool
    message: str


class MotionPlanner:
    """Compile editorial events into a calm, bounded cross-domain track."""

    def __init__(
        self,
        config: MotionPlannerConfig | None = None,
        registry: MotionCapabilityRegistry | None = None,
    ) -> None:
        self.config = config or MotionPlannerConfig()
        self.registry = registry or PHASE_7B_MOTION_CAPABILITIES

    def plan(
        self,
        intent: CreativeIntent,
        caption_plan: CaptionPlan,
        composition_plan: CompositionPlan,
        source_broll_plan: SourceBRollPlan,
    ) -> MotionPlan:
        _validate_inputs(intent, caption_plan, composition_plan, source_broll_plan)
        intensity = intent.policy.intensity
        policy = _INTENSITY_POLICIES[intensity]
        timeline_frames = _timeline_frames(intent)
        point_limit = policy.base_point_limit + ceil(
            timeline_frames / (60 * 30) * policy.points_per_minute
        )
        animated_frame_limit = max(
            policy.minimum_animated_frames,
            round(timeline_frames * policy.animated_frame_ratio),
        )

        candidates: list[_Candidate] = []
        suppressions: list[_Suppression] = []
        fallback_findings: list[MotionQualityFinding] = []
        if intent.policy.reduced_motion:
            fallback_findings.append(MotionQualityFinding(
                code="MOTION_REDUCED_MOTION_FALLBACK",
                severity="warning",
                measured_value="enabled",
                threshold="static or short caption fade",
                message="Reduced-motion policy retained editorial meaning with calm fallbacks.",
            ))
        for request in sorted(
            intent.motion_events,
            key=lambda item: (item.output.start_frame, item.output.end_frame, item.decision_id),
        ):
            candidate = self._candidate(
                intent, request, caption_plan, composition_plan, source_broll_plan, policy,
            )
            if candidate is None:
                suppressions.append(_Suppression(
                    "MOTION_DOMAIN_TARGET_MISSING", request.decision_id,
                    request.domain.value, "matching assessed domain plan target",
                    "The editorial event has no concrete domain target; calm state was retained.",
                ))
                continue
            candidate, finding = self._apply_registry(candidate)
            if (
                candidate.primitive != "static"
                and candidate.duration_frames < self.config.minimum_animated_event_frames
            ):
                requested = candidate.primitive
                candidate = _fallback_candidate(candidate, "static", "short_event")
                fallback_findings.append(MotionQualityFinding(
                    code="MOTION_SHORT_EVENT_FALLBACK",
                    severity="warning",
                    event_id=request.decision_id,
                    measured_value=f"{requested}:{request.output.end_frame - request.output.start_frame}_frames",
                    threshold=f">={self.config.minimum_animated_event_frames}_frames",
                    message="An event too short for stable animation retained its semantic timing as static.",
                ))
            candidates.append(candidate)
            if finding is not None:
                fallback_findings.append(finding)

        accepted: list[_Candidate] = []
        points_used = 0
        animated_frames_used = 0
        ranked = sorted(
            candidates,
            key=lambda item: (
                -_purpose_priority(item.request.purpose),
                item.output.start_frame,
                item.request.decision_id,
            ),
        )
        for candidate in ranked:
            animated = candidate.primitive != "static"
            if animated and candidate.request.domain != MotionDomain.CAPTION and _dense_reading_overlap(
                candidate.output, caption_plan.cues, self.config.dense_caption_cps,
            ):
                suppressions.append(_Suppression(
                    "MOTION_READABILITY_SUPPRESSED", candidate.request.decision_id,
                    "dense_caption_overlap", f"caption_cps <= {self.config.dense_caption_cps}",
                    "Non-caption animation was suppressed while dense text must be read.",
                ))
                continue
            if animated and _violates_domain_cooldown(candidate, accepted, policy.cooldown_frames):
                suppressions.append(_Suppression(
                    "MOTION_COOLDOWN_SUPPRESSED", candidate.request.decision_id,
                    _nearest_domain_gap(candidate, accepted), policy.cooldown_frames,
                    "A same-domain animation inside cooldown was suppressed.",
                ))
                continue
            if animated and _maximum_concurrency((*accepted, candidate)) > policy.max_concurrent_layers:
                suppressions.append(_Suppression(
                    "MOTION_CONCURRENCY_SUPPRESSED", candidate.request.decision_id,
                    _maximum_concurrency((*accepted, candidate)), policy.max_concurrent_layers,
                    "The lower-priority conflicting animation exceeded the layer limit.",
                ))
                continue
            cost = _budget_points(candidate.primitive)
            frames = candidate.duration_frames if animated else 0
            if animated and (
                points_used + cost > point_limit
                or animated_frames_used + frames > animated_frame_limit
            ):
                suppressions.append(_Suppression(
                    "MOTION_BUDGET_SUPPRESSED", candidate.request.decision_id,
                    f"points={points_used + cost},frames={animated_frames_used + frames}",
                    f"points<={point_limit},frames<={animated_frame_limit}",
                    "The lower-priority animation exceeded the global animation budget.",
                ))
                continue
            accepted.append(candidate)
            points_used += cost
            animated_frames_used += frames

        accepted.sort(key=lambda item: (item.output.start_frame, item.request.decision_id))
        events = tuple(self._event(candidate) for candidate in accepted)
        findings = (
            *fallback_findings,
            *(_suppression_finding(item) for item in suppressions),
        )
        maximum_concurrency = _maximum_concurrency(accepted)
        report = MotionQualityReport(
            status="PASS_WITH_WARNINGS" if findings else "PASS",
            findings=tuple(findings),
            metrics=MotionQualityMetrics(
                requested_event_count=len(intent.motion_events),
                emitted_event_count=len(events),
                animated_event_count=sum(item.primitive_id != "static" for item in events),
                suppressed_event_count=len(suppressions),
                cooldown_suppression_count=sum(
                    item.code == "MOTION_COOLDOWN_SUPPRESSED" for item in suppressions
                ),
                concurrency_suppression_count=sum(
                    item.code == "MOTION_CONCURRENCY_SUPPRESSED" for item in suppressions
                ),
                budget_suppression_count=sum(
                    item.code == "MOTION_BUDGET_SUPPRESSED" for item in suppressions
                ),
                readability_suppression_count=sum(
                    item.code == "MOTION_READABILITY_SUPPRESSED" for item in suppressions
                ),
                fallback_count=len(fallback_findings),
                max_concurrent_layers=maximum_concurrency,
                animation_points_used=points_used,
                animated_frames_used=animated_frames_used,
                events_per_minute=round(len(events) * 1800 / max(1, timeline_frames), 4),
            ),
            provenance=MotionQualityProvenance(
                producer="motion_planner",
                planner_version=MOTION_PLANNER_VERSION,
                capability_registry_version=self.registry.schema_version,
                intent_id=intent.intent_id,
                caption_plan_sha256=caption_plan.canonical_hash(),
                composition_plan_sha256=composition_plan.canonical_hash(),
                source_broll_plan_sha256=source_broll_plan.canonical_hash(),
            ),
        )
        budget = MotionAnimationBudget(
            intensity=intensity,
            point_limit=point_limit,
            points_used=points_used,
            animated_frame_limit=animated_frame_limit,
            animated_frames_used=animated_frames_used,
            cooldown_frames=policy.cooldown_frames,
            max_concurrent_layers=policy.max_concurrent_layers,
        )
        return MotionPlan(
            schema_version=MOTION_PLAN_SCHEMA_VERSION,
            intent_id=intent.intent_id,
            events=events,
            intensity=intensity,
            reduced_motion=intent.policy.reduced_motion,
            capability_registry_version=self.registry.schema_version,
            animation_budget=budget,
            caption_plan_sha256=caption_plan.canonical_hash(),
            composition_plan_sha256=composition_plan.canonical_hash(),
            source_broll_plan_sha256=source_broll_plan.canonical_hash(),
            quality_report=report,
            diagnostics=tuple(
                [f"{item.code}:{item.event_id}" for item in suppressions]
                + [f"{item.code}:{item.event_id or 'plan'}" for item in fallback_findings]
            ),
        )

    def _candidate(
        self,
        intent: CreativeIntent,
        request: ResolvedMotionEvent,
        captions: CaptionPlan,
        composition: CompositionPlan,
        broll: SourceBRollPlan,
        policy: _IntensityPolicy,
    ) -> _Candidate | None:
        request = request.model_copy(update={
            "intensity": _effective_intensity(request.intensity, intent.policy.intensity),
        })
        if request.domain == MotionDomain.CAPTION:
            cues = tuple(cue for cue in captions.cues if cue.output.overlaps(request.output))
            if not cues:
                return None
            primitive = _caption_primitive(request, intent.policy.intensity, cues)
            output = _event_window(request.output, policy.maximum_event_frames)
            return _caption_candidate(request, primitive, output, cues, intent.policy.reduced_motion)

        if request.domain == MotionDomain.COMPOSITION:
            punch_segments = tuple(
                segment for segment in composition.segments
                if segment.punch_in is not None
                and segment.punch_in.event_id == request.decision_id
                and segment.output.overlaps(request.output)
            )
            if punch_segments:
                punch = punch_segments[0].punch_in
                assert punch is not None
                candidate = _Candidate(
                    request=request,
                    primitive="punch_in",
                    requested_primitive="punch_in",
                    output=punch.output,
                    duration_frames=min(
                        policy.maximum_event_frames,
                        punch.output.end_frame - punch.output.start_frame,
                    ),
                    target_ids=tuple(item.segment_id for item in punch_segments),
                    scale_from=1,
                    scale_to=round(punch.scale, 4),
                )
            else:
                segments = tuple(
                    segment for segment in composition.segments
                    if segment.output.overlaps(request.output)
                    and segment.movement_reason in {"target_acquired", "target_switch"}
                )
                if not segments:
                    return None
                output = _event_window(request.output, policy.maximum_event_frames)
                candidate = _Candidate(
                    request=request,
                    primitive="crop_translate",
                    requested_primitive="crop_translate",
                    output=output,
                    duration_frames=output.end_frame - output.start_frame,
                    target_ids=tuple(item.segment_id for item in segments),
                )
            return _reduced_candidate(candidate) if intent.policy.reduced_motion else candidate

        broll_segments = tuple(
            segment for segment in broll.segments if segment.destination.overlaps(request.output)
        )
        if broll_segments:
            use_dissolve = (
                any(item.transition == "short_dissolve" for item in broll_segments)
                and intent.policy.intensity != Intensity.LOW
            )
            transition_primitive: MotionPrimitive = "dissolve" if use_dissolve else "static"
            output = _event_window(request.output, min(8, policy.maximum_event_frames))
            candidate = _Candidate(
                request=request,
                primitive=transition_primitive,
                requested_primitive=transition_primitive,
                output=output,
                duration_frames=(output.end_frame - output.start_frame) if use_dissolve else 0,
                target_ids=tuple(item.segment_id for item in broll_segments),
            )
            return _reduced_candidate(candidate) if intent.policy.reduced_motion else candidate

        composition_segments = tuple(
            segment for segment in composition.segments
            if segment.output.overlaps(request.output)
            and segment.movement_reason in {"target_acquired", "target_switch"}
        )
        if not composition_segments:
            return None
        output = _event_window(request.output, policy.maximum_event_frames)
        candidate = _Candidate(
            request=request,
            primitive="dissolve",
            requested_primitive="dissolve",
            output=output,
            duration_frames=output.end_frame - output.start_frame,
            target_ids=tuple(item.segment_id for item in composition_segments),
        )
        return _reduced_candidate(candidate) if intent.policy.reduced_motion else candidate

    def _apply_registry(
        self, candidate: _Candidate,
    ) -> tuple[_Candidate, MotionQualityFinding | None]:
        capability = self.registry.resolve(candidate.primitive, candidate.request.domain)
        if capability is not None:
            return candidate, None
        fallback: Literal["static", "fade"] = (
            "fade" if candidate.request.domain == MotionDomain.CAPTION
            and self.registry.resolve("fade", MotionDomain.CAPTION) is not None
            else "static"
        )
        fallback_capability = self.registry.resolve(fallback, candidate.request.domain)
        if fallback_capability is None:
            fallback = "static"
            fallback_capability = self.registry.resolve("static", candidate.request.domain)
        if fallback_capability is None:
            raise ValueError("MOTION_REGISTRY_STATIC_FALLBACK_MISSING")
        replacement = _fallback_candidate(candidate, fallback, "unsupported_primitive")
        return replacement, MotionQualityFinding(
            code="MOTION_PRIMITIVE_FALLBACK",
            severity="warning",
            event_id=candidate.request.decision_id,
            measured_value=candidate.primitive,
            threshold=f"registered {candidate.request.domain.value} primitive",
            message=f"Unsupported primitive fell back to {fallback}.",
        )

    def _event(self, candidate: _Candidate) -> MotionEventPlan:
        capability = self.registry.resolve(candidate.primitive, candidate.request.domain)
        if capability is None:
            raise ValueError("MOTION_EVENT_PRIMITIVE_NOT_REGISTERED")
        return MotionEventPlan(
            event_id=candidate.request.decision_id,
            output=candidate.output,
            purpose=candidate.request.purpose,
            domain=candidate.request.domain,
            primitive_id=candidate.primitive,
            requested_primitive_id=candidate.requested_primitive,
            backend_id=capability.backend_id,
            easing_id="none" if candidate.primitive == "static" else "ease_in_out",
            intensity=candidate.request.intensity,
            evidence_refs=candidate.request.evidence_refs,
            fallback_primitive_id=capability.fallback_primitive_id,
            duration_frames=candidate.duration_frames,
            scale_from=candidate.scale_from,
            scale_to=candidate.scale_to,
            translate_x_ratio=candidate.translate_x_ratio,
            translate_y_ratio=candidate.translate_y_ratio,
            opacity_from=candidate.opacity_from,
            opacity_to=candidate.opacity_to,
            target_plan_ids=candidate.target_ids,
            budget_points=_budget_points(candidate.primitive),
            reduced_motion_fallback=candidate.reduced_motion,
            fallback_reason=candidate.fallback_reason,
        )


def build_motion_plan(
    intent: CreativeIntent,
    caption_plan: CaptionPlan,
    composition_plan: CompositionPlan,
    source_broll_plan: SourceBRollPlan,
    *,
    config: MotionPlannerConfig | None = None,
    registry: MotionCapabilityRegistry | None = None,
) -> MotionPlan:
    return MotionPlanner(config, registry).plan(
        intent, caption_plan, composition_plan, source_broll_plan,
    )


def _validate_inputs(
    intent: CreativeIntent,
    captions: CaptionPlan,
    composition: CompositionPlan,
    broll: SourceBRollPlan,
) -> None:
    if any(plan.intent_id != intent.intent_id for plan in (captions, composition, broll)):
        raise ValueError("MOTION_DOMAIN_PLAN_INTENT_MISMATCH")
    if broll.schema_version == "7E.source-broll-plan.1":
        if broll.composition_plan_sha256 != composition.canonical_hash():
            raise ValueError("MOTION_SOURCE_BROLL_COMPOSITION_STALE")


def _timeline_frames(intent: CreativeIntent) -> int:
    return max((item.output.end_frame for item in intent.source_output_mapping.segments), default=1)


def _purpose_priority(purpose: MotionPurpose) -> int:
    return {
        MotionPurpose.PAYOFF: 5,
        MotionPurpose.HOOK: 4,
        MotionPurpose.EVIDENCE_REVEAL: 3,
        MotionPurpose.REACTION: 2,
        MotionPurpose.CLAIM_CHANGE: 1,
    }[purpose]


def _effective_intensity(requested: Intensity, policy: Intensity) -> Intensity:
    order = (Intensity.LOW, Intensity.BALANCED, Intensity.HIGH)
    return order[min(order.index(requested), order.index(policy))]


def _caption_primitive(
    request: ResolvedMotionEvent,
    intensity: Intensity,
    cues: Sequence[CaptionCuePlan],
) -> MotionPrimitive:
    if any(cue.fallback_reason in {"readability", "collision", "missing_font"} for cue in cues):
        return "static"
    if intensity == Intensity.LOW:
        return "fade"
    if request.purpose == MotionPurpose.HOOK:
        primitive: MotionPrimitive = "slide" if intensity == Intensity.HIGH else "fade"
    elif request.purpose in {MotionPurpose.EVIDENCE_REVEAL, MotionPurpose.PAYOFF, MotionPurpose.REACTION}:
        primitive = "scale"
    else:
        primitive = "fade"
    if primitive in {"scale", "slide"} and any(cue.timing_mode != "word" for cue in cues):
        return "fade"
    return primitive


def _caption_candidate(
    request: ResolvedMotionEvent,
    primitive: MotionPrimitive,
    output: OutputInterval,
    cues: Sequence[CaptionCuePlan],
    reduced_motion: bool,
) -> _Candidate:
    intensity = request.intensity
    scale = {Intensity.LOW: 1.02, Intensity.BALANCED: 1.04, Intensity.HIGH: 1.06}[intensity]
    slide = {Intensity.LOW: 0.012, Intensity.BALANCED: 0.02, Intensity.HIGH: 0.03}[intensity]
    candidate = _Candidate(
        request=request,
        primitive=primitive,
        requested_primitive=primitive,
        output=output,
        duration_frames=0 if primitive == "static" else output.end_frame - output.start_frame,
        target_ids=tuple(cue.cue_id for cue in cues),
        scale_from=1 if primitive != "scale" else round(1 / scale, 4),
        scale_to=1,
        translate_y_ratio=slide if primitive == "slide" else 0,
        opacity_from=0 if primitive == "fade" else 1,
        opacity_to=1,
    )
    return _reduced_candidate(candidate) if reduced_motion else candidate


def _reduced_candidate(candidate: _Candidate) -> _Candidate:
    if candidate.primitive == "static":
        return candidate
    fallback: Literal["static", "fade"] = (
        "fade" if candidate.request.domain == MotionDomain.CAPTION
        and candidate.primitive != "static" else "static"
    )
    return _fallback_candidate(candidate, fallback, "reduced_motion")


def _fallback_candidate(
    candidate: _Candidate,
    primitive: Literal["static", "fade"],
    reason: Literal["reduced_motion", "unsupported_primitive", "short_event"],
) -> _Candidate:
    return _Candidate(
        request=candidate.request,
        primitive=primitive,
        requested_primitive=candidate.requested_primitive,
        output=candidate.output,
        duration_frames=candidate.duration_frames if primitive == "fade" else 0,
        target_ids=candidate.target_ids,
        opacity_from=0 if primitive == "fade" else 1,
        opacity_to=1,
        reduced_motion=reason == "reduced_motion",
        fallback_reason=reason,
    )


def _event_window(output: OutputInterval, maximum_frames: int) -> OutputInterval:
    return OutputInterval(
        start_frame=output.start_frame,
        end_frame=min(output.end_frame, output.start_frame + maximum_frames),
    )


def _budget_points(primitive: MotionPrimitive) -> int:
    return {
        "static": 0,
        "fade": 1,
        "scale": 2,
        "slide": 2,
        "dissolve": 2,
        "crop_translate": 2,
        "punch_in": 2,
    }[primitive]


def _dense_reading_overlap(
    output: OutputInterval,
    cues: Sequence[CaptionCuePlan],
    threshold: float,
) -> bool:
    for cue in cues:
        if not cue.output.overlaps(output):
            continue
        seconds = (cue.output.end_frame - cue.output.start_frame) / 30
        characters = sum(len(line.replace(" ", "")) for line in cue.resolved_lines)
        if characters / max(seconds, 1 / 30) > threshold:
            return True
    return False


def _violates_domain_cooldown(
    candidate: _Candidate,
    accepted: Sequence[_Candidate],
    cooldown: int,
) -> bool:
    return any(
        item.primitive != "static"
        and item.request.domain == candidate.request.domain
        and _interval_gap(item.output, candidate.output) < cooldown
        for item in accepted
    )


def _nearest_domain_gap(candidate: _Candidate, accepted: Sequence[_Candidate]) -> int:
    gaps = [
        _interval_gap(item.output, candidate.output)
        for item in accepted
        if item.primitive != "static" and item.request.domain == candidate.request.domain
    ]
    return min(gaps, default=0)


def _interval_gap(left: OutputInterval, right: OutputInterval) -> int:
    if left.overlaps(right):
        return 0
    if left.end_frame <= right.start_frame:
        return right.start_frame - left.end_frame
    return left.start_frame - right.end_frame


def _maximum_concurrency(candidates: Iterable[_Candidate]) -> int:
    edges: list[tuple[int, int]] = []
    for item in candidates:
        if item.primitive == "static":
            continue
        edges.append((item.output.start_frame, 1))
        edges.append((item.output.end_frame, -1))
    current = maximum = 0
    for _frame, delta in sorted(edges, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _suppression_finding(item: _Suppression) -> MotionQualityFinding:
    return MotionQualityFinding(
        code=item.code,
        severity="warning",
        event_id=item.event_id,
        measured_value=item.measured_value,
        threshold=item.threshold,
        message=item.message,
    )
