"""A-2 provenance decisions for non-dialogue continuity inside a boundary.

This module deliberately derives only from persisted candidate evidence.  It
does not consult ContentProfile labels and it never uses source-category
special cases: weak evidence becomes an explicit ``uncertain`` decision.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.production_models import (
    BOUNDARY_EPSILON_SECONDS,
    BoundaryDecision,
    BoundaryRange,
    ContinuityDecision,
    ContinuityOmittedSpan,
    ContinuityRequiredSpan,
)
from app.utils import stable_text_hash


def build_continuity_decision(
    *,
    candidate_id: str,
    boundary_decision: dict[str, Any] | BoundaryDecision | None,
    primary_evidence: Iterable[dict[str, Any]],
    multimodal_context: dict[str, Any] | None = None,
) -> ContinuityDecision | None:
    """Create the candidate-owned A-2 decision from persisted evidence only.

    Internal gaps between transcript-backed primary spans are the exact
    non-dialogue spans production would otherwise silently remove.  In the
    absence of valid evidence for either preserving or compacting them, they
    remain typed ``unexplained`` omissions under an ``uncertain`` decision.
    """

    if boundary_decision in (None, {}):
        return None
    boundary = (
        boundary_decision
        if isinstance(boundary_decision, BoundaryDecision)
        else BoundaryDecision.model_validate(boundary_decision)
    )
    if boundary.candidate_id != candidate_id:
        raise ValueError("CONTINUITY_CANDIDATE_MISMATCH")
    approved = boundary.refined_range
    multimodal = dict(multimodal_context or {})
    coverage = _primary_coverage(primary_evidence, approved)
    gaps = _internal_gaps(coverage)
    required = _required_spans(multimodal, approved)
    required = _with_grounded_payoff_tail(required, multimodal, coverage, approved)
    declared_omissions = _declared_omissions(multimodal, approved)
    omitted = _omissions_for_gaps(gaps, required, declared_omissions)
    has_unexplained_omission = any(item.rationale_type == "unexplained" for item in omitted)
    mode = (
        "preserve_required_spans" if required
        else "uncertain" if has_unexplained_omission
        else "compact_dialogue"
    )
    boundary_sha256 = stable_text_hash(boundary.model_dump_json())
    identity = {
        "schema_version": "A-2.continuity.1",
        "candidate_id": candidate_id,
        "boundary_decision_id": boundary.decision_id,
        "boundary_decision_sha256": boundary_sha256,
        "approved_source_range": approved.model_dump(mode="json"),
        "mode": mode,
        "required_spans": [item.model_dump(mode="json") for item in required],
        "omitted_spans": [item.model_dump(mode="json") for item in omitted],
    }
    return ContinuityDecision(
        decision_id=f"continuity-{candidate_id}-{stable_text_hash(str(identity))[:16]}",
        candidate_id=candidate_id,
        boundary_decision_id=boundary.decision_id,
        boundary_decision_sha256=boundary_sha256,
        approved_source_range=approved,
        mode=mode,
        required_spans=required,
        omitted_spans=omitted,
        evidence={
            "producer": "A-2.continuity-decision",
            "primary_coverage_span_count": len(coverage),
            "internal_gap_count": len(gaps),
            "internal_gap_duration_seconds": round(
                sum(item[0].end_seconds - item[0].start_seconds for item in gaps), 3,
            ),
            "multimodal_evidence_present": bool(multimodal),
        },
    )


def _primary_coverage(
    primary_evidence: Iterable[dict[str, Any]], approved: BoundaryRange,
) -> list[tuple[float, float, int | None]]:
    intervals: list[tuple[float, float, int | None]] = []
    for raw in primary_evidence:
        try:
            start = max(approved.start_seconds, float(raw.get("start", 0)))
            end = min(approved.end_seconds, float(raw.get("end", start)))
        except (AttributeError, TypeError, ValueError):
            continue
        if end <= start + BOUNDARY_EPSILON_SECONDS:
            continue
        identifier = raw.get("segment_id")
        try:
            segment_id = int(identifier) if identifier is not None else None
        except (TypeError, ValueError):
            segment_id = None
        intervals.append((start, end, segment_id))
    intervals.sort(key=lambda item: (item[0], item[1], item[2] if item[2] is not None else -1))
    merged: list[tuple[float, float, int | None]] = []
    for start, end, identifier in intervals:
        if merged and start <= merged[-1][1] + BOUNDARY_EPSILON_SECONDS:
            previous_start, previous_end, previous_id = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end), previous_id)
        else:
            merged.append((start, end, identifier))
    return merged


def _internal_gaps(coverage: list[tuple[float, float, int | None]]) -> list[tuple[BoundaryRange, int | None, int | None]]:
    gaps: list[tuple[BoundaryRange, int | None, int | None]] = []
    for left, right in zip(coverage, coverage[1:]):
        if right[0] > left[1] + BOUNDARY_EPSILON_SECONDS:
            gaps.append((
                BoundaryRange(start_seconds=left[1], end_seconds=right[0]), left[2], right[2],
            ))
    return gaps


def _required_spans(
    multimodal: dict[str, Any], approved: BoundaryRange,
) -> list[ContinuityRequiredSpan]:
    raw_spans = multimodal.get("continuity_required_spans")
    if not isinstance(raw_spans, list):
        raw_spans = multimodal.get("required_spans", [])
    required: list[ContinuityRequiredSpan] = []
    for raw in raw_spans if isinstance(raw_spans, list) else []:
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        if "rationale" not in payload and payload.get("reason"):
            payload["rationale"] = payload["reason"]
        if "source_range" not in payload:
            payload["source_range"] = {
                "start_seconds": payload.get("start_seconds"),
                "end_seconds": payload.get("end_seconds"),
            }
        required.append(ContinuityRequiredSpan.model_validate(payload))
    _validate_ranges(required, approved)
    return _unique_required(required)


def _with_grounded_payoff_tail(
    required: list[ContinuityRequiredSpan],
    multimodal: dict[str, Any],
    coverage: list[tuple[float, float, int | None]],
    approved: BoundaryRange,
) -> list[ContinuityRequiredSpan]:
    """Promote the already-grounded 6D payoff tail, never a guessed one."""

    if not multimodal.get("multimodal_payoff_grounded") or not coverage:
        return required
    try:
        preserve_until = min(float(multimodal.get("preserve_until_seconds")), approved.end_seconds)
    except (TypeError, ValueError):
        return required
    tail_start = coverage[-1][1]
    if preserve_until <= tail_start + BOUNDARY_EPSILON_SECONDS:
        return required
    tail = ContinuityRequiredSpan(
        requirement_type="payoff",
        source_range=BoundaryRange(start_seconds=tail_start, end_seconds=preserve_until),
        rationale="Observed multimodal payoff extends beyond the final dialogue evidence.",
        evidence={
            "source": "multimodal_context",
            "multimodal_payoff_grounded": True,
            "evidence_refs": list(multimodal.get("evidence_refs", [])),
        },
    )
    return _unique_required([*required, tail])


def _declared_omissions(
    multimodal: dict[str, Any], approved: BoundaryRange,
) -> list[ContinuityOmittedSpan]:
    raw_spans = multimodal.get("continuity_omissions", [])
    omissions: list[ContinuityOmittedSpan] = []
    for raw in raw_spans if isinstance(raw_spans, list) else []:
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        if "rationale" not in payload and payload.get("reason"):
            payload["rationale"] = payload["reason"]
        if "source_range" not in payload:
            payload["source_range"] = {
                "start_seconds": payload.get("start_seconds"),
                "end_seconds": payload.get("end_seconds"),
            }
        omissions.append(ContinuityOmittedSpan.model_validate(payload))
    _validate_ranges(omissions, approved)
    return omissions


def _omissions_for_gaps(
    gaps: list[tuple[BoundaryRange, int | None, int | None]],
    required: list[ContinuityRequiredSpan],
    declared: list[ContinuityOmittedSpan],
) -> list[ContinuityOmittedSpan]:
    required_ranges = [item.source_range for item in required]
    omissions: list[ContinuityOmittedSpan] = []
    for gap, left_id, right_id in gaps:
        for remaining in _subtract(gap, required_ranges):
            explicit = next((
                item for item in declared
                if _same_range(item.source_range, remaining)
            ), None)
            if explicit is not None:
                omissions.append(explicit)
                continue
            omissions.append(ContinuityOmittedSpan(
                source_range=remaining,
                rationale_type="unexplained",
                rationale="No persisted visual, semantic, reaction, payoff, or compaction evidence explains this non-dialogue gap.",
                evidence={
                    "source": "transcript_primary_coverage",
                    "left_transcript_segment_id": left_id,
                    "right_transcript_segment_id": right_id,
                },
            ))
    return omissions


def _subtract(source: BoundaryRange, exclusions: list[BoundaryRange]) -> list[BoundaryRange]:
    points: list[tuple[float, float]] = [(source.start_seconds, source.end_seconds)]
    for exclusion in exclusions:
        next_points: list[tuple[float, float]] = []
        for start, end in points:
            if exclusion.end_seconds <= start + BOUNDARY_EPSILON_SECONDS or exclusion.start_seconds >= end - BOUNDARY_EPSILON_SECONDS:
                next_points.append((start, end))
                continue
            if exclusion.start_seconds > start + BOUNDARY_EPSILON_SECONDS:
                next_points.append((start, min(end, exclusion.start_seconds)))
            if exclusion.end_seconds < end - BOUNDARY_EPSILON_SECONDS:
                next_points.append((max(start, exclusion.end_seconds), end))
        points = next_points
    return [BoundaryRange(start_seconds=start, end_seconds=end) for start, end in points]


def _validate_ranges(spans: Iterable[Any], approved: BoundaryRange) -> None:
    for span in spans:
        source_range = span.source_range
        if (
            source_range.start_seconds < approved.start_seconds - BOUNDARY_EPSILON_SECONDS
            or source_range.end_seconds > approved.end_seconds + BOUNDARY_EPSILON_SECONDS
        ):
            raise ValueError("CONTINUITY_SOURCE_RANGE_OUTSIDE_BOUNDARY")


def _unique_required(spans: list[ContinuityRequiredSpan]) -> list[ContinuityRequiredSpan]:
    unique: list[ContinuityRequiredSpan] = []
    seen: set[str] = set()
    for span in spans:
        key = stable_text_hash(span.model_dump_json())
        if key not in seen:
            unique.append(span)
            seen.add(key)
    return unique


def _same_range(left: BoundaryRange, right: BoundaryRange) -> bool:
    return (
        abs(left.start_seconds - right.start_seconds) <= BOUNDARY_EPSILON_SECONDS
        and abs(left.end_seconds - right.end_seconds) <= BOUNDARY_EPSILON_SECONDS
    )
