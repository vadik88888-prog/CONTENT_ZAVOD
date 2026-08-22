from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.caption_planning import (
    CAPTION_FEASIBILITY_VERSION,
    build_caption_plan,
    resolve_caption_font_manifest,
)
from app.config import AppConfig
from app.content_transformation import TRANSFORMATION_ENGINE_VERSION, run_content_transformation
from app.content_understanding import ensure_candidate_boundary_decision
from app.creative_contracts import (
    EditMapSegment,
    OutputInterval,
    SourceInterval,
    SourceOutputTimeMap,
    canonical_hash,
    seconds_to_output_frame,
)
from app.creative_evidence import NATIVE_EVIDENCE_HANDOFF_VERSION, build_native_evidence_handoff
from app.models import ScoredCandidate, scored_from_dict
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan
from app.production_plan import PRODUCTION_PLAN_VERSION, ProductionPlanEnvelopeContext, build_production_plan
from app.quality_report import exact_dialogue_semantic_finding
from app.semantic_extraction import build_source_context


PRODUCTION_FEASIBILITY_SCHEMA_VERSION = "7J.3.production-feasibility.2"
PRODUCTION_FEASIBILITY_POLICY_VERSION = "7J.3.provider-free-a1-a3.2"
A1_POLICY_VERSION = "7J.2A-1.material-speech-clarity.2"


def assess_production_feasibility(
    scored: Iterable[ScoredCandidate],
    *,
    source: dict[str, Any],
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    multimodal_timeline: Mapping[str, Any] | None,
    story_units: Mapping[str, Any] | None,
    config: AppConfig,
    envelope_context: ProductionPlanEnvelopeContext,
) -> dict[str, Any]:
    """Assess only deterministic blockers already enforced by A-1 and A-3.

    The preselection assessment deliberately uses the local transformation path
    and never creates or receives a provider.  An assessment failure is UNKNOWN,
    not a blocker: only the existing policies may prove a candidate impossible.
    """

    font_manifest = resolve_caption_font_manifest(
        config.production_render.subtitle_font_family,
    )
    items = [
        _assess_candidate(
            item,
            source=source,
            metadata=metadata,
            transcript=transcript,
            transcript_features=transcript_features,
            audio_features=audio_features,
            scenes=scenes,
            multimodal_timeline=multimodal_timeline,
            story_units=story_units,
            config=config,
            envelope_context=envelope_context,
            font_manifest=font_manifest,
        )
        for item in scored
    ]
    return {
        "schema_version": PRODUCTION_FEASIBILITY_SCHEMA_VERSION,
        "policy_version": PRODUCTION_FEASIBILITY_POLICY_VERSION,
        "producer": "app.production_feasibility.assess_production_feasibility",
        "assessment_scope": "supplied_base-selection-eligible_ranking_pool",
        "provider_mode": "local_only",
        "provider_calls": {"brain": 0, "vision": 0, "transformation": 0},
        "policy_provenance": {
            "a1": {
                "policy": "app.quality_report.exact_dialogue_semantic_finding",
                "version": A1_POLICY_VERSION,
            },
            "a3": {
                "policy": "app.caption_planning.build_caption_plan",
                "version": CAPTION_FEASIBILITY_VERSION,
                "evidence_handoff_version": NATIVE_EVIDENCE_HANDOFF_VERSION,
            },
            "transformation_engine_version": TRANSFORMATION_ENGINE_VERSION,
            "production_plan_version": PRODUCTION_PLAN_VERSION,
        },
        "candidates": items,
        "summary": {
            "candidate_count": len(items),
            "viable_count": sum(item["status"] == "VIABLE" for item in items),
            "guaranteed_blocked_count": sum(item["status"] == "GUARANTEED_BLOCKED" for item in items),
            "unknown_count": sum(item["status"] == "UNKNOWN" for item in items),
            "guaranteed_blocked_candidate_ids": [
                item["candidate_id"] for item in items if item["status"] == "GUARANTEED_BLOCKED"
            ],
        },
    }


def resolve_recommendation_production_feasibility(
    scored: list[ScoredCandidate],
    *,
    content_map: dict[str, Any] | None,
    source: dict[str, Any],
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    multimodal_timeline: Mapping[str, Any] | None,
    story_units: Mapping[str, Any] | None,
    config: AppConfig,
    envelope_context: ProductionPlanEnvelopeContext,
) -> dict[str, Any]:
    """Assess the provisional MMR winners until recommendation IDs stabilize."""

    from app.content_understanding import select_with_coverage
    from app.selection import select_clips

    data: dict[str, Any] | None = None
    iterations: list[dict[str, Any]] = []
    scored_by_id = {item.candidate.id: item for item in scored}
    while True:
        feasibility_by_id = production_feasibility_index(data)
        selection_input = [scored_from_dict(item.to_dict()) for item in scored]
        if content_map:
            provisional, _coverage = select_with_coverage(
                selection_input,
                config,
                content_map,
                production_feasibility=data,
            )
        else:
            provisional = select_clips(
                selection_input,
                config,
                production_feasibility=data,
            )
        provisional_ids = [item.candidate.id for item in provisional]
        pending_ids = [
            candidate_id for candidate_id in provisional_ids
            if candidate_id not in feasibility_by_id
        ]
        iterations.append({
            "provisional_candidate_ids": provisional_ids,
            "new_assessment_candidate_ids": pending_ids,
        })
        if not pending_ids:
            break
        additional = assess_production_feasibility(
            [scored_by_id[candidate_id] for candidate_id in pending_ids],
            source=source,
            metadata=metadata,
            transcript=transcript,
            transcript_features=transcript_features,
            audio_features=audio_features,
            scenes=scenes,
            multimodal_timeline=multimodal_timeline,
            story_units=story_units,
            config=config,
            envelope_context=envelope_context,
        )
        data = merge_production_feasibility_artifacts(data, additional)
        data["allow_ranked_replacements"] = bool(
            data["summary"]["guaranteed_blocked_count"]
        )
    if data is None:
        data = assess_production_feasibility(
            [],
            source=source,
            metadata=metadata,
            transcript=transcript,
            transcript_features=transcript_features,
            audio_features=audio_features,
            scenes=scenes,
            multimodal_timeline=multimodal_timeline,
            story_units=story_units,
            config=config,
            envelope_context=envelope_context,
        )
    data.setdefault("allow_ranked_replacements", False)
    data["selection_iterations"] = iterations
    return data


def validate_production_feasibility_artifact(value: dict[str, Any]) -> None:
    if value.get("schema_version") != PRODUCTION_FEASIBILITY_SCHEMA_VERSION:
        raise ValueError("Unsupported production feasibility schema.")
    if value.get("policy_version") != PRODUCTION_FEASIBILITY_POLICY_VERSION:
        raise ValueError("Unsupported production feasibility policy.")
    provider_calls = value.get("provider_calls")
    if provider_calls != {"brain": 0, "vision": 0, "transformation": 0}:
        raise ValueError("Production feasibility must remain provider-free.")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Production feasibility candidates are invalid.")
    identifiers: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("Production feasibility candidate is invalid.")
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id or item.get("status") not in {"VIABLE", "GUARANTEED_BLOCKED", "UNKNOWN"}:
            raise ValueError("Production feasibility candidate identity/status is invalid.")
        if item.get("status") == "GUARANTEED_BLOCKED" and not item.get("blockers"):
            raise ValueError("Guaranteed blocked feasibility requires blocker provenance.")
        identifiers.append(candidate_id)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Production feasibility candidate IDs must be unique.")


def production_feasibility_index(value: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    candidates = value.get("candidates") if isinstance(value, Mapping) else None
    if not isinstance(candidates, list):
        return {}
    return {
        str(item.get("candidate_id")): dict(item)
        for item in candidates
        if isinstance(item, dict) and str(item.get("candidate_id") or "")
    }


def merge_production_feasibility_artifacts(
    current: dict[str, Any] | None,
    additional: dict[str, Any],
) -> dict[str, Any]:
    validate_production_feasibility_artifact(additional)
    if current is None:
        return additional
    validate_production_feasibility_artifact(current)
    merged = dict(current)
    by_id = {
        str(item["candidate_id"]): dict(item)
        for item in current["candidates"]
    }
    order = list(by_id)
    for item in additional["candidates"]:
        candidate_id = str(item["candidate_id"])
        if candidate_id not in by_id:
            order.append(candidate_id)
        by_id[candidate_id] = dict(item)
    items = [by_id[candidate_id] for candidate_id in order]
    merged["candidates"] = items
    merged["summary"] = {
        "candidate_count": len(items),
        "viable_count": sum(item["status"] == "VIABLE" for item in items),
        "guaranteed_blocked_count": sum(item["status"] == "GUARANTEED_BLOCKED" for item in items),
        "unknown_count": sum(item["status"] == "UNKNOWN" for item in items),
        "guaranteed_blocked_candidate_ids": [
            item["candidate_id"] for item in items if item["status"] == "GUARANTEED_BLOCKED"
        ],
    }
    return merged


def _assess_candidate(
    scored: ScoredCandidate,
    *,
    source: dict[str, Any],
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    multimodal_timeline: Mapping[str, Any] | None,
    story_units: Mapping[str, Any] | None,
    config: AppConfig,
    envelope_context: ProductionPlanEnvelopeContext,
    font_manifest: Any,
) -> dict[str, Any]:
    candidate = scored.candidate
    base: dict[str, Any] = {
        "candidate_id": candidate.id,
        "status": "UNKNOWN",
        "reason_code": "PRODUCTION_FEASIBILITY_UNKNOWN",
        "blockers": [],
        "gate_results": [],
        "provenance": {
            "provider_mode": "local_only",
            "candidate_fingerprint": canonical_hash(candidate.to_dict()),
            "source_id": source.get("id"),
        },
    }
    try:
        if ensure_candidate_boundary_decision(candidate) is None:
            base["reason"] = "Provider-free production feasibility requires validated boundary evidence."
            base["gate_results"] = [{
                "gate": "provider_free_plan",
                "status": "UNKNOWN",
                "reason_code": "BOUNDARY_DECISION_REQUIRED",
                "reason": base["reason"],
            }]
            return base
        context = build_source_context(
            source,
            metadata,
            candidate,
            transcript,
            transcript_features,
            audio_features,
            scenes,
            config.transformation,
        )
        transformation = run_content_transformation(
            context,
            config.transformation,
            provider=None,
            force_local=True,
        )
        if transformation.get("status") not in {"completed", "fallback"}:
            base["reason"] = _transformation_error(transformation)
            base["gate_results"] = [{
                "gate": "provider_free_plan",
                "status": "UNKNOWN",
                "reason_code": "LOCAL_TRANSFORMATION_UNAVAILABLE",
                "reason": base["reason"],
            }]
            return base
        plan = build_production_plan(
            transformation,
            config.production,
            envelope_context=envelope_context,
        )
        a1 = _a1_gate_result(plan.model_dump(mode="json"))
        mapping = _source_output_map_from_plan(plan)
        handoff = build_native_evidence_handoff(
            plan,
            mapping,
            config,
            candidate=scored.to_dict(),
            multimodal_timeline=multimodal_timeline,
            story_units=story_units,
        )
        caption_plan = build_caption_plan(
            handoff.intent,
            transcript,
            config.production_render,
            font_manifest=font_manifest,
        )
        decision = caption_plan.feasibility_decision
        a3_blocked = decision is not None and decision.status == "INFEASIBLE"
        a3 = {
            "gate": "A-3",
            "status": "BLOCKED" if a3_blocked else "PASS",
            "reason_code": (
                decision.reason_code if decision is not None else "NO_CAPTION_FEASIBILITY_DECISION"
            ),
            "policy": "app.caption_planning.build_caption_plan",
            "policy_version": CAPTION_FEASIBILITY_VERSION,
            "decision": decision.model_dump(mode="json") if decision is not None else None,
            "intent_id": handoff.intent.intent_id,
            "evidence_handoff_version": NATIVE_EVIDENCE_HANDOFF_VERSION,
        }
        blockers = [item for item in (a1, a3) if item["status"] == "BLOCKED"]
        base.update({
            "status": "GUARANTEED_BLOCKED" if blockers else "VIABLE",
            "reason_code": (
                str(blockers[0]["reason_code"]) if blockers else "PROVIDER_FREE_PRODUCTION_VIABLE"
            ),
            "reason": (
                _blocker_reason(blockers[0])
                if blockers else "Provider-free A-1/A-3 production feasibility passed."
            ),
            "blockers": blockers,
            "gate_results": [a1, a3],
            "production_plan_id": plan.plan_id,
            "production_plan_fingerprint": plan.reference().plan_fingerprint,
        })
        return base
    except Exception as error:
        base["reason"] = f"Provider-free production feasibility could not be proven: {error}"
        base["gate_results"] = [{
            "gate": "provider_free_plan",
            "status": "UNKNOWN",
            "reason_code": "PRODUCTION_FEASIBILITY_ASSESSMENT_ERROR",
            "reason": str(error),
        }]
        return base


def _a1_gate_result(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the canonical A-1 result without promoting a warning to a block."""

    finding = exact_dialogue_semantic_finding(dict(plan))
    blocker = finding if finding is not None and finding["severity"] == "blocker" else None
    return {
        "gate": "A-1",
        "status": "BLOCKED" if blocker is not None else "PASS",
        "reason_code": (
            blocker["code"] if blocker is not None
            else "A1_DIALOGUE_CONFIDENCE_RISK" if finding is not None
            else "A1_DIALOGUE_CONFIDENCE_PASS"
        ),
        "policy": "app.quality_report.exact_dialogue_semantic_finding",
        "policy_version": A1_POLICY_VERSION,
        "evidence": finding,
    }


def _source_output_map_from_plan(plan: ProductionPlan) -> SourceOutputTimeMap:
    segments: list[EditMapSegment] = []
    output_cursor = 0
    timeline_cursor = 0.0
    for segment in plan.segments:
        duration = max(0.0, float(segment.estimated_duration_seconds))
        output_start = max(output_cursor, seconds_to_output_frame(timeline_cursor))
        timeline_cursor = round(timeline_cursor + duration, 3)
        output_end = max(output_start + 1, seconds_to_output_frame(timeline_cursor, end=True))
        output_cursor = output_end
        source_range = _segment_source_range(segment)
        if source_range is None:
            continue
        start, end = source_range
        if end <= start:
            continue
        segments.append(EditMapSegment(
            map_id=f"production-feasibility-{segment.segment_id}",
            source=SourceInterval.from_seconds(start, end),
            output=OutputInterval(start_frame=output_start, end_frame=output_end),
        ))
    if not segments:
        raise ValueError("PROVIDER_FREE_PLAN_HAS_NO_SOURCE_MAPPING")
    continuity = plan.continuity_decision
    return SourceOutputTimeMap(
        segments=tuple(segments),
        continuity_decision_id=continuity.decision_id if continuity is not None else None,
        continuity_decision_version=continuity.schema_version if continuity is not None else None,
        continuity_decision_sha256=continuity.fingerprint() if continuity is not None else None,
    )


def _segment_source_range(segment: Any) -> tuple[float, float] | None:
    if isinstance(segment, DialogueSegment):
        return segment.source_start_seconds, segment.source_end_seconds
    if isinstance(segment, NarrationSegment) and segment.source_ranges:
        source = segment.source_ranges[0]
        return source.source_start_seconds, source.source_end_seconds
    return None


def _transformation_error(value: Mapping[str, Any]) -> str:
    validation = value.get("validation")
    errors = validation.get("errors") if isinstance(validation, Mapping) else None
    if isinstance(errors, list) and errors:
        return "; ".join(str(item) for item in errors)
    return str(value.get("error") or "Local transformation did not produce a valid FinalScript.")


def _blocker_reason(blocker: Mapping[str, Any]) -> str:
    gate = str(blocker.get("gate") or "production")
    code = str(blocker.get("reason_code") or "BLOCKED")
    return f"Guaranteed blocked by provider-free {gate} policy: {code}."
