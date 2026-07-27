from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class IntervalMetrics:
    overlap_seconds: float
    iou: float
    containment: float
    midpoint_distance_seconds: float


def interval_metrics(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> IntervalMetrics:
    first_duration = max(0.0, first_end - first_start)
    second_duration = max(0.0, second_end - second_start)
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = first_duration + second_duration - overlap
    shortest = min(first_duration, second_duration)
    first_midpoint = first_start + (first_duration / 2.0)
    second_midpoint = second_start + (second_duration / 2.0)
    return IntervalMetrics(
        overlap_seconds=overlap,
        iou=(overlap / union) if union else 0.0,
        containment=(overlap / shortest) if shortest else 0.0,
        midpoint_distance_seconds=abs(first_midpoint - second_midpoint),
    )


def transcript_similarity(first: str, second: str) -> float:
    """Return a deterministic, inexpensive token Jaccard similarity."""
    first_tokens = set(_normalised_tokens(first))
    second_tokens = set(_normalised_tokens(second))
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def is_temporal_duplicate(
    metrics: IntervalMetrics,
    *,
    overlap_threshold: float,
    minimum_distance_seconds: float,
) -> bool:
    """Treat substantially overlapping or effectively adjacent clips as duplicates."""
    return (
        metrics.containment >= overlap_threshold
        or metrics.iou >= overlap_threshold
        or (
            metrics.overlap_seconds > 0.0
            and metrics.midpoint_distance_seconds < minimum_distance_seconds
        )
    )


def _normalised_tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(value)]
