from __future__ import annotations

"""Deterministic Phase 6 evidence hand-off for the native creative renderer.

This module is deliberately an adapter, not another analysis pass.  It consumes
only artifacts already persisted by the Phase 6 pipeline and turns their
bounded editorial/visual observations into the Phase 7 native contracts.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from app.candidate_quality import MIN_EDITORIAL_MULTIMODAL_CONFIDENCE
from app.composition_planning import TargetObservation
from app.config import AppConfig
from app.creative_contracts import (
    AttentionTarget,
    BeatRole,
    CreativeIntent,
    EvidenceItem,
    Intensity,
    LayoutFamily,
    MotionDomain,
    MotionPurpose,
    NormalizedRect,
    OutputInterval,
    ResolvedBeat,
    ResolvedCompositionTarget,
    ResolvedEmphasis,
    ResolvedMotionEvent,
    ResolvedSourceBRoll,
    SemanticClass,
    SourceBRollSemanticKind,
    SourceInterval,
    SourceOutputTimeMap,
    canonical_hash,
)
from app.creative_execution import default_native_creative_intent
from app.production_models import BoundaryDecision, ProductionPlan
from app.source_broll_planning import SourceSceneEvidence


NATIVE_EVIDENCE_HANDOFF_VERSION = "7G.3"
MAX_NATIVE_EMPHASIS_WORDS = 2


@dataclass(frozen=True, slots=True)
class NativeEvidenceHandoff:
    intent: CreativeIntent
    target_observations: tuple[TargetObservation, ...] = ()
    source_scenes: tuple[SourceSceneEvidence, ...] = ()
    execution_status: Literal["native_rich", "native_fallback"] = "native_fallback"
    reason_codes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def _short_emphasis_text(value: str) -> str:
    return " ".join(value.split()[:MAX_NATIVE_EMPHASIS_WORDS])[:240]


def build_native_evidence_handoff(
    plan: ProductionPlan,
    mapping: SourceOutputTimeMap,
    config: AppConfig,
    *,
    candidate: Mapping[str, Any] | None,
    multimodal_timeline: Mapping[str, Any] | None,
    story_units: Mapping[str, Any] | None,
) -> NativeEvidenceHandoff:
    """Adapt persisted Phase 6 artifacts without invoking Brain or Vision."""

    fallback = default_native_creative_intent(plan, mapping, config)
    if not isinstance(candidate, Mapping):
        return NativeEvidenceHandoff(
            intent=fallback,
            reason_codes=("PHASE6_CANDIDATE_MISSING", "SAFE_A_ROLL_FALLBACK"),
            diagnostics=("No persisted candidate record was supplied to the native hand-off.",),
        )

    timeline = multimodal_timeline if isinstance(multimodal_timeline, Mapping) else {}
    stories_artifact = story_units if isinstance(story_units, Mapping) else {}
    candidate_id = str(candidate.get("id") or plan.metadata.candidate_id)
    story_ids = _candidate_story_ids(candidate)
    stories = [
        item for item in stories_artifact.get("story_units", [])
        if isinstance(item, Mapping) and str(item.get("story_unit_id") or "") in story_ids
    ]

    manifest: list[EvidenceItem] = []
    beats: list[ResolvedBeat] = []
    emphasis: list[ResolvedEmphasis] = []
    motion: list[ResolvedMotionEvent] = []
    observations: list[TargetObservation] = []
    composition: list[ResolvedCompositionTarget] = []
    source_broll: list[ResolvedSourceBRoll] = []
    story_evidence_refs: dict[str, str] = {}

    for index, story in enumerate(stories, start=1):
        story_source = _source_interval(_float(story.get("start")), _float(story.get("end")))
        source = _mapped_intersection(mapping, _float(story.get("start")), _float(story.get("end")))
        if source is None or story_source is None:
            continue
        output = mapping.map_interval(source)
        if output is None:
            continue
        confidence = _confidence(story.get("confidence"), default=0.72)
        evidence_ref = _id("story", candidate_id, str(story.get("story_unit_id") or index))
        manifest.append(EvidenceItem(
            evidence_ref=evidence_ref,
            evidence_kind="story_unit",
            source=story_source,
            confidence=confidence,
            artifact_fingerprint=canonical_hash(story),
            provenance="phase6:story_units",
        ))
        story_id = str(story.get("story_unit_id") or f"story-{index}")
        story_evidence_refs[story_id] = evidence_ref
        claim = ResolvedBeat(
            decision_id=_id("beat-claim", candidate_id, story_id),
            source=source,
            output=output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            role=BeatRole.CLAIM,
            importance=max(0.55, _confidence(story.get("information_density"), default=0.7)),
        )
        beats.append(claim)
        _append_edge_decisions(
            candidate_id, story_id, story, mapping, story_source, plan.boundary_decision,
            confidence, evidence_ref,
            beats, emphasis, motion,
        )

    _append_audio_decisions(
        plan, mapping, timeline, candidate_id, manifest, emphasis, motion,
    )

    visual_rows = _visual_rows(candidate, timeline)
    composition_intent = _composition_intent(candidate, plan)
    seen_observations: set[tuple[int, str, str]] = set()
    for index, row in enumerate(visual_rows, start=1):
        timestamp = _timestamp(row)
        source = _source_interval_at(mapping, timestamp)
        bounds = _target_bounds(row)
        target = _attention_target(row, composition_intent)
        if source is None or bounds is None or target is None:
            continue
        output = mapping.map_interval(source)
        if output is None:
            continue
        confidence = _confidence(row.get("confidence"), default=0.0)
        if confidence < MIN_EDITORIAL_MULTIMODAL_CONFIDENCE:
            continue
        scene_id = _scene_at(timeline, timestamp)
        target_ref = _id("target", target.value, scene_id or candidate_id)
        key = (output.start_frame, target.value, target_ref)
        if key in seen_observations:
            continue
        seen_observations.add(key)
        evidence_ref = _id("visual", candidate_id, str(row.get("event_id") or row.get("keyframe_id") or index))
        manifest.append(EvidenceItem(
            evidence_ref=evidence_ref,
            evidence_kind="visual",
            source=source,
            confidence=confidence,
            artifact_fingerprint=canonical_hash(row),
            provenance=_visual_provenance(row),
        ))
        frame = (output.start_frame + output.end_frame - 1) // 2
        observations.append(TargetObservation(
            observation_id=_id("observation", candidate_id, str(index)),
            frame=frame,
            target=target,
            target_ref=target_ref,
            bounds=bounds,
            confidence=confidence,
            evidence_ref=evidence_ref,
            scene_id=scene_id,
        ))
        composition.append(ResolvedCompositionTarget(
            decision_id=_id("composition", candidate_id, str(index)),
            source=source,
            output=output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            target=target,
            target_ref=target_ref,
            priority=_target_priority(target),
            allowed_layouts=_target_layouts(target),
        ))

    scenes, scene_manifest = _source_scenes(
        plan, timeline, stories, beats, visual_rows, candidate_id,
        same_source_usage_allowed=config.production_render.same_source_broll_allowed,
    )
    manifest.extend(scene_manifest)
    if config.production_render.same_source_broll_allowed:
        source_broll.extend(_source_broll_decisions(
            mapping, beats, scenes, story_evidence_refs, candidate_id,
        ))
    manifest = _unique_manifest(manifest)
    if not manifest:
        return NativeEvidenceHandoff(
            intent=fallback,
            reason_codes=("PHASE6_EVIDENCE_UNAVAILABLE", "SAFE_A_ROLL_FALLBACK"),
            diagnostics=("Persisted Phase 6 artifacts contained no mappable trusted evidence.",),
        )

    evidence_fingerprint = canonical_hash([item.model_dump(mode="json") for item in manifest])
    proposal_hash = canonical_hash({
        "version": NATIVE_EVIDENCE_HANDOFF_VERSION,
        "creative_identity": list(fallback.provenance),
        "candidate_id": candidate_id,
        "evidence_fingerprint": evidence_fingerprint,
        "beats": [item.model_dump(mode="json") for item in beats],
        "emphasis": [item.model_dump(mode="json") for item in emphasis],
        "composition": [item.model_dump(mode="json") for item in composition],
        "motion": [item.model_dump(mode="json") for item in motion],
        "source_broll": [item.model_dump(mode="json") for item in source_broll],
    })
    accepted_confidence = [
        item.confidence for item in (*beats, *emphasis, *composition, *motion, *source_broll)
    ]
    policy = fallback.policy.model_copy(update={
        "source_broll_enabled": bool(
            config.production_render.same_source_broll_allowed and source_broll
        ),
    })
    intent = fallback.model_copy(update={
        "intent_id": f"intent-phase6-{proposal_hash[:17]}",
        "evidence_fingerprint": evidence_fingerprint,
        "evidence_manifest": tuple(manifest),
        "proposal_hash": proposal_hash,
        "confidence": (
            round(sum(accepted_confidence) / len(accepted_confidence), 7)
            if accepted_confidence else 0.0
        ),
        "provenance": (
            "phase6:candidates.scored",
            "phase6:multimodal_timeline",
            "phase6:story_units",
            NATIVE_EVIDENCE_HANDOFF_VERSION,
            *fallback.provenance,
        ),
        "policy": policy,
        "beats": tuple(sorted(beats, key=lambda item: (item.output.start_frame, item.decision_id))),
        "semantic_emphasis": tuple(sorted(emphasis, key=lambda item: (item.output.start_frame, item.decision_id))),
        "composition_targets": tuple(sorted(composition, key=lambda item: (item.output.start_frame, item.decision_id))),
        "motion_events": tuple(sorted(motion, key=lambda item: (item.output.start_frame, item.decision_id))),
        "source_broll": tuple(sorted(source_broll, key=lambda item: (item.output.start_frame, item.decision_id))),
    })
    reason_codes: list[str] = []
    diagnostics: list[str] = []
    if not composition:
        reason_codes.append("COMPOSITION_EVIDENCE_UNAVAILABLE")
    if not observations:
        reason_codes.append("STABLE_COMPOSITION_FALLBACK")
    if not config.production_render.same_source_broll_allowed:
        reason_codes.append("SOURCE_BROLL_USAGE_NOT_AUTHORIZED")
    elif not source_broll:
        reason_codes.append("SOURCE_BROLL_RELEVANCE_UNPROVEN")
    if not motion:
        reason_codes.append("STATIC_MOTION_FALLBACK")
    # Same-source B-roll is an optional, default-off layer. Rich native
    # execution is defined by the required evidence-backed creative domains;
    # withholding an optional cutaway must not demote captions, emphasis,
    # hook/payoff presentation, reframing, and motion to a fallback result.
    beat_roles = {item.role for item in beats}
    motion_purposes = {item.purpose for item in motion}
    rich = bool(
        {BeatRole.HOOK, BeatRole.PAYOFF}.issubset(beat_roles)
        and emphasis
        and composition
        and observations
        and {MotionPurpose.HOOK, MotionPurpose.PAYOFF}.issubset(motion_purposes)
    )
    if not rich:
        reason_codes.append("SAFE_A_ROLL_FALLBACK")
        diagnostics.append("Native execution retained only evidence-backed domains and used safe fallbacks elsewhere.")
    return NativeEvidenceHandoff(
        intent=intent,
        target_observations=tuple(sorted(observations, key=lambda item: (item.frame, item.observation_id))),
        source_scenes=scenes,
        execution_status="native_rich" if rich else "native_fallback",
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        diagnostics=tuple(diagnostics),
    )


def _append_edge_decisions(
    candidate_id: str,
    story_id: str,
    story: Mapping[str, Any],
    mapping: SourceOutputTimeMap,
    story_source: SourceInterval,
    boundary: BoundaryDecision | None,
    confidence: float,
    evidence_ref: str,
    beats: list[ResolvedBeat],
    emphasis: list[ResolvedEmphasis],
    motion: list[ResolvedMotionEvent],
) -> None:
    decisions = (
        ("hook", BeatRole.HOOK, MotionPurpose.HOOK, SemanticClass.CLAIM, str(story.get("hook_seed") or "")),
        ("payoff", BeatRole.PAYOFF, MotionPurpose.PAYOFF, SemanticClass.PAYOFF, str(story.get("payoff") or story.get("ending") or "")),
    )
    for name, role, purpose, semantic, text in decisions:
        if not text.strip():
            continue
        resolved = _edge_decision_interval(mapping, story_source, boundary, role)
        if resolved is None:
            continue
        resolved_source, resolved_output = resolved
        beats.append(ResolvedBeat(
            decision_id=_id(f"beat-{name}", candidate_id, story_id),
            source=resolved_source,
            output=resolved_output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            role=role,
            importance=0.9 if role == BeatRole.HOOK else 0.95,
        ))
        emphasis.append(ResolvedEmphasis(
            decision_id=_id(f"emphasis-{name}", candidate_id, story_id),
            source=resolved_source,
            output=resolved_output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            text_span=_short_emphasis_text(text),
            semantic_class=semantic,
            importance=0.9,
        ))
        motion.append(ResolvedMotionEvent(
            decision_id=_id(f"motion-{name}", candidate_id, story_id),
            source=resolved_source,
            output=resolved_output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            purpose=purpose,
            domain=MotionDomain.CAPTION,
            intensity=Intensity.BALANCED,
        ))


def _edge_decision_interval(
    mapping: SourceOutputTimeMap,
    story_source: SourceInterval,
    boundary: BoundaryDecision | None,
    role: BeatRole,
) -> tuple[SourceInterval, OutputInterval] | None:
    """Resolve hook/payoff against persisted boundary evidence when available.

    A selected story may be represented by several discontiguous edit-map
    segments.  Mapping both editorial edges through the first intersection
    places the payoff on the hook frames.  The Phase 5 boundary decision is
    already the authoritative source range, so use its hook/payoff ranges and
    fall back to the first/last mapped story edge only for legacy plans.
    """

    requirement_kind = "hook" if role == BeatRole.HOOK else "payoff"
    requirements = []
    if boundary is not None:
        requirements = [
            item for item in boundary.required_evidence
            if item.requirement_type == requirement_kind
        ]
        if role == BeatRole.PAYOFF and not requirements:
            requirements = [
                item for item in boundary.required_evidence
                if item.requirement_type == "completion"
            ]
    if requirements:
        ordered = sorted(
            requirements,
            key=lambda item: (item.source_range.start_seconds, item.source_range.end_seconds),
        )
        requirement = ordered[0] if role == BeatRole.HOOK else ordered[-1]
        required_source = _source_interval(
            requirement.source_range.start_seconds,
            requirement.source_range.end_seconds,
        )
        resolved = _mapped_parts(
            mapping,
            _intersection(story_source, required_source) if required_source is not None else None,
        )
        if resolved:
            return resolved[0] if role == BeatRole.HOOK else resolved[-1]

    parts = _mapped_parts(mapping, story_source)
    if not parts:
        return None
    source, output = parts[0] if role == BeatRole.HOOK else parts[-1]
    duration = output.end_frame - output.start_frame
    edge = max(1, min(45, duration // 3))
    if role == BeatRole.HOOK:
        sliced = OutputInterval(
            start_frame=output.start_frame,
            end_frame=min(output.end_frame, output.start_frame + edge),
        )
    else:
        sliced = OutputInterval(
            start_frame=max(output.start_frame, output.end_frame - edge),
            end_frame=output.end_frame,
        )
    return _source_for_output_slice(source, output, sliced), sliced


def _mapped_parts(
    mapping: SourceOutputTimeMap,
    source: SourceInterval | None,
) -> list[tuple[SourceInterval, OutputInterval]]:
    if source is None:
        return []
    result: list[tuple[SourceInterval, OutputInterval]] = []
    for segment in mapping.segments:
        part = _intersection(source, segment.source)
        if part is None:
            continue
        output = mapping.map_interval(part)
        if output is not None:
            result.append((part, output))
    return sorted(result, key=lambda item: (item[1].start_frame, item[1].end_frame))


def _intersection(left: SourceInterval, right: SourceInterval) -> SourceInterval | None:
    start = max(left.start_tick, right.start_tick)
    end = min(left.end_tick, right.end_tick)
    if end <= start:
        return None
    return SourceInterval(start_tick=start, end_tick=end)


def _source_scenes(
    plan: ProductionPlan,
    timeline: Mapping[str, Any],
    stories: list[Mapping[str, Any]],
    beats: Iterable[ResolvedBeat],
    visual_rows: list[Mapping[str, Any]],
    candidate_id: str,
    *,
    same_source_usage_allowed: bool,
) -> tuple[tuple[SourceSceneEvidence, ...], list[EvidenceItem]]:
    if not stories:
        return (), []
    result: list[SourceSceneEvidence] = []
    manifest: list[EvidenceItem] = []
    beat_list = tuple(beats)
    for index, raw in enumerate(timeline.get("scenes", []), start=1):
        if not isinstance(raw, Mapping):
            continue
        source = _source_interval(_float(raw.get("start_seconds")), _float(raw.get("end_seconds")))
        if source is None:
            continue
        source_start = source.start_tick / 1_000_000
        source_end = source.end_tick / 1_000_000
        rows = [row for row in visual_rows if source_start <= _timestamp(row) < source_end]
        geometry_row = max(
            (row for row in rows if _target_bounds(row) is not None and _attention_target(row) is not None),
            key=lambda item: _confidence(item.get("confidence"), default=0.0),
            default=None,
        )
        kinds = tuple(dict.fromkeys(
            kind for row in rows for kind in _semantic_kinds(row)
        )) or (SourceBRollSemanticKind.CONTEXT,)
        linked_story_ids = tuple(dict.fromkeys(
            str(story.get("story_unit_id") or "")
            for story in stories
            if str(story.get("story_unit_id") or "")
            and (
                story_source := _source_interval(
                    _float(story.get("start")), _float(story.get("end")),
                )
            ) is not None
            and story_source.overlaps(source)
        ))
        if not linked_story_ids:
            continue
        roles = tuple(dict.fromkeys(
            beat.role for beat in beat_list
            if any(
                beat.evidence_refs == (_id("story", candidate_id, story_id),)
                for story_id in linked_story_ids
            )
        )) or (BeatRole.CLAIM,)
        evidence_ref = _id("scene", candidate_id, str(raw.get("scene_id") or index))
        confidence = _confidence(raw.get("confidence"), default=0.65)
        manifest.append(EvidenceItem(
            evidence_ref=evidence_ref,
            evidence_kind="scene",
            source=source,
            confidence=confidence,
            artifact_fingerprint=canonical_hash(raw),
            provenance="phase6:multimodal_timeline.scenes",
        ))
        result.append(SourceSceneEvidence(
            scene_id=_id("scene", str(raw.get("scene_id") or index)),
            source_id=plan.reference().identity.source_id,
            source=source,
            semantic_kinds=kinds,
            story_unit_ids=linked_story_ids,
            beat_roles=roles,
            evidence_refs=(evidence_ref,),
            source_crop=_target_bounds(geometry_row) if geometry_row is not None else None,
            source_target=_attention_target(geometry_row) if geometry_row is not None else None,
            confidence=confidence,
            identity_status="verified",
            attribution_status="verified",
            chronology_status="safe",
            causality_status="not_claimed",
            rights_status="verified" if same_source_usage_allowed else "uncertain",
            payoff_signal=_scene_payoff(rows),
            provenance=("phase6:multimodal_timeline", NATIVE_EVIDENCE_HANDOFF_VERSION),
        ))
    return tuple(result), manifest


def _append_audio_decisions(
    plan: ProductionPlan,
    mapping: SourceOutputTimeMap,
    timeline: Mapping[str, Any],
    candidate_id: str,
    manifest: list[EvidenceItem],
    emphasis: list[ResolvedEmphasis],
    motion: list[ResolvedMotionEvent],
) -> None:
    """Carry persisted emphasis/reaction evidence without another audio pass."""

    for index, raw in enumerate(timeline.get("audio_event_map", []), start=1):
        if not isinstance(raw, Mapping) or raw.get("event_type") not in {"emphasis", "reaction_label"}:
            continue
        source = _mapped_intersection(
            mapping, _float(raw.get("start_seconds")), _float(raw.get("end_seconds")),
        )
        if source is None:
            continue
        output = mapping.map_interval(source)
        if output is None:
            continue
        confidence = _confidence(raw.get("confidence"), default=0.0)
        if confidence < MIN_EDITORIAL_MULTIMODAL_CONFIDENCE:
            continue
        event_type = str(raw.get("event_type"))
        evidence_ref = _id("audio", candidate_id, str(raw.get("event_id") or index))
        manifest.append(EvidenceItem(
            evidence_ref=evidence_ref,
            evidence_kind="audio",
            source=source,
            confidence=confidence,
            artifact_fingerprint=canonical_hash(raw),
            provenance="phase6:multimodal_timeline.audio_event_map",
        ))
        raw_observation = raw.get("observation")
        observation: Mapping[str, Any] = raw_observation if isinstance(raw_observation, Mapping) else {}
        text = str(observation.get("label") or "audio emphasis")
        emphasis.append(ResolvedEmphasis(
            decision_id=_id("emphasis-audio", candidate_id, str(index)),
            source=source,
            output=output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            text_span=_short_emphasis_text(text),
            semantic_class=SemanticClass.ACTION if event_type == "reaction_label" else SemanticClass.CLAIM,
            importance=0.82 if event_type == "reaction_label" else 0.72,
        ))
        motion.append(ResolvedMotionEvent(
            decision_id=_id("motion-audio", candidate_id, str(index)),
            source=source,
            output=output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            purpose=MotionPurpose.REACTION if event_type == "reaction_label" else MotionPurpose.CLAIM_CHANGE,
            domain=MotionDomain.CAPTION,
            intensity=Intensity.BALANCED,
        ))

    boundary = plan.boundary_decision
    context = boundary.multimodal_context if boundary is not None else {}
    if not isinstance(context, Mapping):
        return
    for index, value in enumerate(context.get("audio_or_visual_payoff_times", []), start=1):
        timestamp = _float(value)
        source = _source_interval_at(mapping, timestamp) if timestamp is not None else None
        if source is None:
            continue
        output = mapping.map_interval(source)
        if output is None:
            continue
        confidence = _confidence(context.get("confidence"), default=0.0)
        if confidence < MIN_EDITORIAL_MULTIMODAL_CONFIDENCE:
            continue
        evidence_ref = _id("boundary", candidate_id, str(index))
        manifest.append(EvidenceItem(
            evidence_ref=evidence_ref,
            evidence_kind="boundary",
            source=source,
            confidence=confidence,
            artifact_fingerprint=canonical_hash(context),
            provenance="phase6:BoundaryDecision.multimodal_context",
        ))
        motion.append(ResolvedMotionEvent(
            decision_id=_id("motion-boundary-payoff", candidate_id, str(index)),
            source=source,
            output=output,
            confidence=confidence,
            evidence_refs=(evidence_ref,),
            purpose=MotionPurpose.PAYOFF,
            domain=MotionDomain.CAPTION,
            intensity=Intensity.BALANCED,
        ))


def _source_broll_decisions(
    mapping: SourceOutputTimeMap,
    beats: list[ResolvedBeat],
    scenes: tuple[SourceSceneEvidence, ...],
    story_evidence_refs: Mapping[str, str],
    candidate_id: str,
) -> tuple[ResolvedSourceBRoll, ...]:
    """Resolve deterministic same-source cutaways from already-proven scenes."""

    results: list[ResolvedSourceBRoll] = []
    used_ranges: list[SourceInterval] = []
    for scene in sorted(scenes, key=lambda item: (-item.confidence, item.source.start_tick, item.scene_id)):
        if scene.confidence < 0.72 or scene.rights_status != "verified":
            continue
        semantic = next(
            (kind for kind in scene.semantic_kinds if kind != SourceBRollSemanticKind.CONTEXT),
            None,
        )
        if semantic is None:
            continue
        story_id = next(
            (item for item in scene.story_unit_ids if item in story_evidence_refs), None,
        )
        if story_id is None:
            continue
        story_ref = story_evidence_refs[story_id]
        linked = [beat for beat in beats if story_ref in beat.evidence_refs]
        if scene.payoff_signal in {"reveal", "result", "resolution"}:
            linked = [beat for beat in linked if beat.role == BeatRole.PAYOFF]
        else:
            linked = [beat for beat in linked if beat.role != BeatRole.PAYOFF] or linked
        beat = max(linked, key=lambda item: (item.importance, item.confidence), default=None)
        if beat is None or scene.source.overlaps(beat.source):
            continue
        destination_frames = min(
            beat.output.end_frame - beat.output.start_frame,
            max(12, min(45, round((scene.source.end_tick - scene.source.start_tick) * 30 / 1_000_000))),
        )
        if destination_frames <= 0:
            continue
        destination = OutputInterval(
            start_frame=beat.output.start_frame,
            end_frame=beat.output.start_frame + destination_frames,
        )
        destination_source = _source_for_output_slice(beat.source, beat.output, destination)
        cutaway_duration = max(1, round(destination_frames * 1_000_000 / 30))
        cutaway = SourceInterval(
            start_tick=scene.source.start_tick,
            end_tick=min(scene.source.end_tick, scene.source.start_tick + cutaway_duration),
        )
        if any(item.overlaps(cutaway) for item in used_ranges):
            continue
        used_ranges.append(cutaway)
        results.append(ResolvedSourceBRoll(
            decision_id=_id("source-broll", candidate_id, str(len(results) + 1)),
            source=destination_source,
            output=destination,
            confidence=min(scene.confidence, beat.confidence),
            evidence_refs=(story_ref,),
            source_cutaway=cutaway,
            source_cutaway_evidence_refs=scene.evidence_refs,
            story_unit_id=story_id,
            story_unit_evidence_ref=story_ref,
            semantic_kind=semantic,
            retain_source_audio=False,
        ))
    return tuple(results)


def _visual_rows(candidate: Mapping[str, Any], timeline: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for event in timeline.get("visual_event_map", []):
        if isinstance(event, Mapping) and event.get("event_type") == "subject_observation":
            rows.append(event)
    provenance = candidate.get("multimodal_provenance")
    if isinstance(provenance, Mapping):
        rows.extend(item for item in provenance.get("visual_evidence", []) if isinstance(item, Mapping))
    pass2 = candidate.get("vision_pass2_evidence")
    result = pass2.get("result") if isinstance(pass2, Mapping) else None
    if isinstance(result, Mapping):
        rows.extend(item for item in result.get("observations", []) if isinstance(item, Mapping))
    return rows


def _candidate_story_ids(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    values = candidate.get("story_unit_ids")
    result = [str(item) for item in values if str(item)] if isinstance(values, list) else []
    if candidate.get("story_unit_id"):
        result.append(str(candidate["story_unit_id"]))
    return tuple(dict.fromkeys(result))


def _composition_intent(
    candidate: Mapping[str, Any], plan: ProductionPlan,
) -> Mapping[str, Any]:
    value = candidate.get("composition_intent")
    if isinstance(value, Mapping) and value.get("evidence_status") == "available":
        return value
    return plan.composition_intent if isinstance(plan.composition_intent, Mapping) else {}


def _source_interval(start: float | None, end: float | None) -> SourceInterval | None:
    if start is None or end is None or end <= start:
        return None
    return SourceInterval(start_tick=round(start * 1_000_000), end_tick=round(end * 1_000_000))


def _mapped_intersection(mapping: SourceOutputTimeMap, start: float | None, end: float | None) -> SourceInterval | None:
    if start is None or end is None or end <= start:
        return None
    start_tick = round(start * 1_000_000)
    end_tick = round(end * 1_000_000)
    for segment in mapping.segments:
        left = max(start_tick, segment.source.start_tick)
        right = min(end_tick, segment.source.end_tick)
        if right > left:
            return SourceInterval(start_tick=left, end_tick=right)
    return None


def _source_interval_at(mapping: SourceOutputTimeMap, timestamp: float) -> SourceInterval | None:
    tick = round(timestamp * 1_000_000)
    for segment in mapping.segments:
        if segment.source.start_tick <= tick < segment.source.end_tick:
            radius = min(50_000, max(1, (segment.source.end_tick - segment.source.start_tick) // 60))
            start = max(segment.source.start_tick, tick - radius)
            end = min(segment.source.end_tick, tick + radius)
            if end <= start:
                end = min(segment.source.end_tick, start + 1)
            return SourceInterval(start_tick=start, end_tick=end)
    return None


def _source_for_output_slice(
    source: SourceInterval, output: OutputInterval, sliced: OutputInterval,
) -> SourceInterval:
    source_duration = source.end_tick - source.start_tick
    output_duration = output.end_frame - output.start_frame
    start = source.start_tick + (sliced.start_frame - output.start_frame) * source_duration // output_duration
    end = source.start_tick + (
        (sliced.end_frame - output.start_frame) * source_duration + output_duration - 1
    ) // output_duration
    return SourceInterval(start_tick=start, end_tick=max(start + 1, min(source.end_tick, end)))


def _target_bounds(row: Mapping[str, Any]) -> NormalizedRect | None:
    observation = row.get("observation") if isinstance(row.get("observation"), Mapping) else row
    active = observation.get("active_subject") if isinstance(observation, Mapping) else None
    bbox = active.get("normalized_bbox") if isinstance(active, Mapping) else None
    if isinstance(bbox, Mapping):
        center_x = _float(bbox.get("normalized_x"))
        center_y = _float(bbox.get("normalized_y"))
        width = _float(bbox.get("normalized_width"))
        height = _float(bbox.get("normalized_height"))
    else:
        center_x = _float(row.get("normalized_center_x"))
        center_y = _float(row.get("normalized_center_y"))
        width = _float(row.get("normalized_width"))
        height = _float(row.get("normalized_height"))
    if center_x is None or center_y is None:
        return None
    width = width if width is not None and width > 0 else 0.32
    height = height if height is not None and height > 0 else 0.52
    width, height = min(1.0, width), min(1.0, height)
    x = min(1.0 - width, max(0.0, center_x - width / 2))
    y = min(1.0 - height, max(0.0, center_y - height / 2))
    return NormalizedRect(x=x, y=y, width=width, height=height)


def _attention_target(
    row: Mapping[str, Any], composition_intent: Mapping[str, Any] | None = None,
) -> AttentionTarget | None:
    observation = row.get("observation") if isinstance(row.get("observation"), Mapping) else row
    active = observation.get("active_subject") if isinstance(observation, Mapping) else None
    raw = str(active.get("target_type") if isinstance(active, Mapping) else row.get("primary_subject") or "")
    faces = observation.get("faces") if isinstance(observation, Mapping) else None
    count = faces.get("visible_count") if isinstance(faces, Mapping) else row.get("visible_face_count")
    if isinstance(count, int) and count > 1:
        return AttentionTarget.GROUP
    resolved = {
        "primary_face": AttentionTarget.SPEAKER,
        "face": AttentionTarget.SPEAKER,
        "primary_person": AttentionTarget.SUBJECT,
        "person": AttentionTarget.SUBJECT,
        "subject_group": AttentionTarget.GROUP,
        "group": AttentionTarget.GROUP,
        "important_object": AttentionTarget.OBJECT,
        "object": AttentionTarget.OBJECT,
        "product": AttentionTarget.PRODUCT,
        "screen_region": AttentionTarget.SCREEN,
        "screen": AttentionTarget.SCREEN,
    }.get(raw)
    if resolved is not None:
        return resolved
    intent = composition_intent or {}
    explicit = {
        "screen": AttentionTarget.SCREEN,
        "product": AttentionTarget.PRODUCT,
        "object": AttentionTarget.OBJECT,
        "person": AttentionTarget.SUBJECT,
        "face": AttentionTarget.SPEAKER,
        "group": AttentionTarget.GROUP,
    }.get(str(_intent_value(intent, "screen_or_product") or _intent_value(intent, "important_subject_or_object") or ""))
    if explicit is not None:
        return explicit
    if _intent_value(intent, "multiple_subjects") is True:
        return AttentionTarget.GROUP
    if _intent_value(intent, "reaction") not in {None, "", "none", "unknown"}:
        return AttentionTarget.REACTION
    if _intent_value(intent, "active_speaker") is True:
        return AttentionTarget.SPEAKER
    return None


def _intent_value(intent: Mapping[str, Any], name: str) -> Any:
    raw = intent.get(name)
    return raw.get("value") if isinstance(raw, Mapping) else raw


def _semantic_kinds(row: Mapping[str, Any]) -> tuple[SourceBRollSemanticKind, ...]:
    target = _attention_target(row)
    result: list[SourceBRollSemanticKind] = []
    if target == AttentionTarget.OBJECT:
        result.append(SourceBRollSemanticKind.OBJECT)
    elif target == AttentionTarget.PRODUCT:
        result.append(SourceBRollSemanticKind.PRODUCT)
    elif target == AttentionTarget.SCREEN:
        result.append(SourceBRollSemanticKind.SCREEN)
    reaction = str(row.get("reaction") or "none")
    if reaction not in {"", "none", "unknown"}:
        result.append(SourceBRollSemanticKind.REACTION)
    action = str(row.get("action") or "none")
    if action not in {"", "none", "speaking", "unknown"}:
        result.append(SourceBRollSemanticKind.ACTION)
    return tuple(dict.fromkeys(result))


def _target_priority(target: AttentionTarget) -> int:
    return {
        AttentionTarget.SCREEN: 96,
        AttentionTarget.PRODUCT: 94,
        AttentionTarget.OBJECT: 90,
        AttentionTarget.SPEAKER: 86,
        AttentionTarget.REACTION: 84,
        AttentionTarget.GROUP: 80,
        AttentionTarget.SUBJECT: 78,
    }.get(target, 60)


def _target_layouts(target: AttentionTarget) -> tuple[LayoutFamily, ...]:
    if target == AttentionTarget.SCREEN:
        return (LayoutFamily.SCREEN_PRIORITY, LayoutFamily.FIT_BACKGROUND)
    if target == AttentionTarget.PRODUCT:
        return (LayoutFamily.SCREEN_PRODUCT, LayoutFamily.SINGLE_SUBJECT)
    if target == AttentionTarget.GROUP:
        return (LayoutFamily.WIDE_GROUP, LayoutFamily.FIT_BACKGROUND)
    if target == AttentionTarget.SPEAKER:
        return (LayoutFamily.STABLE_SPEAKER, LayoutFamily.SINGLE_SUBJECT)
    return (LayoutFamily.SINGLE_SUBJECT, LayoutFamily.FIT_BACKGROUND)


def _scene_payoff(rows: Iterable[Mapping[str, Any]]) -> str:
    values = {str(row.get("payoff_signal") or "unknown") for row in rows}
    for value in ("resolution", "result", "reveal", "setup", "none"):
        if value in values:
            return value
    return "unknown"


def _scene_at(timeline: Mapping[str, Any], timestamp: float) -> str | None:
    for raw in timeline.get("scenes", []):
        if not isinstance(raw, Mapping):
            continue
        start, end = _float(raw.get("start_seconds")), _float(raw.get("end_seconds"))
        if start is not None and end is not None and start <= timestamp < end:
            return _id("scene", str(raw.get("scene_id") or "unknown"))
    return None


def _timestamp(row: Mapping[str, Any]) -> float:
    for value in (row.get("timestamp"), row.get("time_seconds"), row.get("start_seconds")):
        result = _float(value)
        if result is not None:
            return result
    return 0.0


def _visual_provenance(row: Mapping[str, Any]) -> str:
    if row.get("event_id"):
        return "phase6:multimodal_timeline.visual_event_map"
    return "phase6:vision_observation"


def _unique_manifest(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    result: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        if item.evidence_ref not in seen:
            result.append(item)
            seen.add(item.evidence_ref)
    return result


def _id(*parts: str) -> str:
    readable = "-".join(
        "".join(
            character
            if (character.isascii() and character.isalnum()) or character in "._:-"
            else "-"
            for character in part
        )
        for part in parts if part
    ).strip("-._:")
    if readable and len(readable) <= 140:
        return readable
    return f"phase6-{canonical_hash(parts)[:24]}"


def _confidence(value: Any, *, default: float) -> float:
    result = _float(value)
    return max(0.0, min(1.0, result if result is not None else default))


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
