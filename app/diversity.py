from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


DIVERSITY_DECISION_SCHEMA_VERSION = "5B.2"
DIVERSITY_LEGACY_REASON_CODE = "LEGACY_DIVERSITY_UNASSESSED"


_TOKEN_PATTERN = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class IntervalMetrics:
    overlap_seconds: float
    iou: float
    containment: float
    midpoint_distance_seconds: float


@dataclass(frozen=True, slots=True)
class DiversitySimilarity:
    """Deterministic evidence for one eligible candidate pair."""

    candidate_id: str
    other_candidate_id: str
    composite_similarity: float
    components: dict[str, float] = field(default_factory=dict)
    available_components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "other_candidate_id": self.other_candidate_id,
            "composite_similarity": round(self.composite_similarity, 6),
            "components": {
                name: round(value, 6)
                for name, value in sorted(self.components.items())
            },
            "available_components": list(self.available_components),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiversitySimilarity":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            other_candidate_id=str(data.get("other_candidate_id") or ""),
            composite_similarity=float(data.get("composite_similarity", 0)),
            components={str(name): float(value) for name, value in dict(data.get("components") or {}).items()},
            available_components=[str(item) for item in data.get("available_components", [])],
        )


@dataclass(frozen=True, slots=True)
class DiversityExclusion:
    """A machine-readable reason why a candidate did not enter the result."""

    candidate_id: str
    reason_code: str
    reason: str
    against_candidate_id: str | None = None
    max_similarity: float | None = None
    similarity: DiversitySimilarity | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "against_candidate_id": self.against_candidate_id,
            "max_similarity": round(self.max_similarity, 6) if self.max_similarity is not None else None,
        }
        if self.similarity is not None:
            data["similarity"] = self.similarity.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiversityExclusion":
        raw_similarity = data.get("similarity")
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            reason_code=str(data.get("reason_code") or "UNKNOWN"),
            reason=str(data.get("reason") or ""),
            against_candidate_id=(str(data["against_candidate_id"]) if data.get("against_candidate_id") else None),
            max_similarity=(float(data["max_similarity"]) if data.get("max_similarity") is not None else None),
            similarity=DiversitySimilarity.from_dict(raw_similarity) if isinstance(raw_similarity, dict) else None,
        )


@dataclass(frozen=True, slots=True)
class DiversitySelection:
    """Recorded MMR rationale for an accepted candidate."""

    candidate_id: str
    coverage_quality_score: float
    max_similarity: float
    against_candidate_id: str | None
    mmr_score: float
    similarity: DiversitySimilarity | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "reason_code": "SELECTED_MMR",
            "coverage_quality_score": round(self.coverage_quality_score, 6),
            "max_similarity": round(self.max_similarity, 6),
            "against_candidate_id": self.against_candidate_id,
            "mmr_score": round(self.mmr_score, 6),
        }
        if self.similarity is not None:
            data["similarity"] = self.similarity.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiversitySelection":
        raw_similarity = data.get("similarity")
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            coverage_quality_score=float(data.get("coverage_quality_score", 0)),
            max_similarity=float(data.get("max_similarity", 0)),
            against_candidate_id=(str(data["against_candidate_id"]) if data.get("against_candidate_id") else None),
            mmr_score=float(data.get("mmr_score", 0)),
            similarity=DiversitySimilarity.from_dict(raw_similarity) if isinstance(raw_similarity, dict) else None,
        )


@dataclass(frozen=True, slots=True)
class DiversityDecision:
    """Versioned, reproducible selection decision persisted with final selection."""

    schema_version: str
    config_version: str
    requested_count: int
    lambda_value: float | None
    eligible_candidate_ids: list[str] = field(default_factory=list)
    selected_candidate_ids: list[str] = field(default_factory=list)
    selections: list[DiversitySelection] = field(default_factory=list)
    exclusions: list[DiversityExclusion] = field(default_factory=list)
    similarities: list[DiversitySimilarity] = field(default_factory=list)
    result_reason_code: str = "REQUEST_SATISFIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.config_version,
            "requested_count": self.requested_count,
            "lambda": round(self.lambda_value, 6) if self.lambda_value is not None else None,
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selections": [item.to_dict() for item in self.selections],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "similarities": [item.to_dict() for item in self.similarities],
            "result_reason_code": self.result_reason_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DiversityDecision":
        """Read pre-5B-2 artifacts explicitly instead of treating them as decisions."""

        if not isinstance(data, dict) or not data.get("schema_version"):
            return legacy_diversity_decision()
        return cls(
            schema_version=str(data.get("schema_version")),
            config_version=str(data.get("config_version") or "unknown"),
            requested_count=int(data.get("requested_count", 0)),
            lambda_value=(float(data["lambda"]) if data.get("lambda") is not None else None),
            eligible_candidate_ids=[str(item) for item in data.get("eligible_candidate_ids", [])],
            selected_candidate_ids=[str(item) for item in data.get("selected_candidate_ids", [])],
            selections=[
                DiversitySelection.from_dict(item)
                for item in data.get("selections", []) if isinstance(item, dict)
            ],
            exclusions=[
                DiversityExclusion.from_dict(item)
                for item in data.get("exclusions", []) if isinstance(item, dict)
            ],
            similarities=[
                DiversitySimilarity.from_dict(item)
                for item in data.get("similarities", []) if isinstance(item, dict)
            ],
            result_reason_code=str(data.get("result_reason_code") or "UNKNOWN"),
        )


def legacy_diversity_decision() -> DiversityDecision:
    """Explicit compatibility state for artifacts created before Goal 5B-2."""

    return DiversityDecision(
        schema_version="legacy",
        config_version="legacy",
        requested_count=0,
        lambda_value=None,
        result_reason_code=DIVERSITY_LEGACY_REASON_CODE,
    )


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
