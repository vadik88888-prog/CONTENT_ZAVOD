from __future__ import annotations

"""Canonical deterministic materiality policy for exact ASR speech evidence."""

import math
from typing import Any, Iterable, Mapping


SPEECH_CLARITY_POLICY_VERSION = "friend-beta-speech-clarity-materiality.1"
SPEECH_CLARITY_CONFIDENCE_THRESHOLD = 0.5
SPEECH_CLARITY_MATERIAL_DURATION_SECONDS = 1.0
SPEECH_CLARITY_MATERIAL_COVERAGE_RATIO = 0.15


def assess_speech_clarity_materiality(
    exact_mappings: Iterable[Mapping[str, Any]],
    *,
    coverage_ranges: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Classify every low-confidence exact mapping without averaging confidence.

    A short isolated mapping is a warning only when both its exact geometry and
    the candidate/published dialogue coverage are reliable.  Missing geometry
    for low-confidence speech is intentionally a strict blocker: materiality
    cannot be disproven from an ungrounded mapping.
    """

    low_mappings: list[dict[str, Any]] = []
    invalid_confidence = False
    invalid_geometry = False
    for raw in exact_mappings:
        confidence = _finite_number(raw.get("confidence"))
        if confidence is None:
            # An exact mapping with no reliable confidence cannot prove speech
            # clarity.  Preserve it as a strict low-confidence evidence item.
            confidence = 0.0
            invalid_confidence = True
        if confidence >= SPEECH_CLARITY_CONFIDENCE_THRESHOLD:
            continue
        start = _finite_number(raw.get("start_seconds"))
        end = _finite_number(raw.get("end_seconds"))
        if start is None or end is None or end < start:
            invalid_geometry = True
        low_mappings.append({
            "segment_id": raw.get("segment_id"),
            "fact_id": raw.get("fact_id"),
            "transcript_segment_id": raw.get("transcript_segment_id"),
            "confidence": round(confidence, 6),
            "source_start_seconds": round(start, 6) if start is not None else None,
            "source_end_seconds": round(end, 6) if end is not None else None,
        })
    if not low_mappings:
        return None

    coverage, coverage_invalid = _normalized_ranges(coverage_ranges)
    low_ranges: list[tuple[float, float]] = []
    for item in low_mappings:
        start = item["source_start_seconds"]
        end = item["source_end_seconds"]
        if start is None or end is None:
            continue
        intersections = _intersections((float(start), float(end)), coverage)
        if not intersections:
            invalid_geometry = True
        low_ranges.extend(intersections)
    for item in low_mappings:
        start = item["source_start_seconds"]
        end = item["source_end_seconds"]
        item["duration_seconds"] = (
            round(float(end) - float(start), 6)
            if start is not None and end is not None
            else None
        )

    low_duration = _merged_duration(low_ranges)
    coverage_duration = _merged_duration(coverage)
    coverage_ratio = low_duration / coverage_duration if coverage_duration > 0 else None
    reasons: list[str] = []
    if invalid_confidence:
        reasons.append("EXACT_SPEECH_CONFIDENCE_UNAVAILABLE")
    if invalid_geometry or coverage_invalid or coverage_ratio is None:
        reasons.append("EXACT_SPEECH_GEOMETRY_UNAVAILABLE")
    if low_duration >= SPEECH_CLARITY_MATERIAL_DURATION_SECONDS:
        reasons.append("LOW_CONFIDENCE_DURATION_MATERIAL")
    if coverage_ratio is not None and coverage_ratio >= SPEECH_CLARITY_MATERIAL_COVERAGE_RATIO:
        reasons.append("LOW_CONFIDENCE_COVERAGE_MATERIAL")
    severity = "blocker" if reasons else "warning"
    return {
        "policy_version": SPEECH_CLARITY_POLICY_VERSION,
        "decision": severity,
        "severity": severity,
        "material": severity == "blocker",
        "low_confidence_mappings": low_mappings,
        "low_confidence_duration_seconds": round(low_duration, 6),
        "coverage_duration_seconds": round(coverage_duration, 6),
        "low_confidence_coverage_ratio": round(coverage_ratio, 6) if coverage_ratio is not None else None,
        "confidence_threshold": SPEECH_CLARITY_CONFIDENCE_THRESHOLD,
        "duration_threshold_seconds": SPEECH_CLARITY_MATERIAL_DURATION_SECONDS,
        "coverage_threshold_ratio": SPEECH_CLARITY_MATERIAL_COVERAGE_RATIO,
        "materiality_reasons": reasons,
    }


def _normalized_ranges(ranges: Iterable[Mapping[str, Any]]) -> tuple[list[tuple[float, float]], bool]:
    normalized: list[tuple[float, float]] = []
    invalid = False
    for item in ranges:
        start = _finite_number(item.get("start_seconds"))
        end = _finite_number(item.get("end_seconds"))
        if start is None or end is None or end < start:
            invalid = True
            continue
        normalized.append((start, end))
    if not normalized:
        invalid = True
    return _merged_ranges(normalized), invalid


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _intersections(target: tuple[float, float], coverage: list[tuple[float, float]]) -> list[tuple[float, float]]:
    start, end = target
    return [(max(start, left), min(end, right)) for left, right in coverage if min(end, right) >= max(start, left)]


def _merged_ranges(ranges: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _merged_duration(ranges: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merged_ranges(ranges))
