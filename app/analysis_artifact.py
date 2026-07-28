from __future__ import annotations

"""Versioned hand-off contract between source analysis and selected rendering.

The artifact deliberately keeps large source-derived data in the existing work
cache.  It contains the immutable identifiers, a compact review payload and
references needed to load the full scored candidates again for rendering.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils import read_json, utc_now, write_json


ANALYSIS_ARTIFACT_SCHEMA_VERSION = "1.0"


class AnalysisArtifactError(ValueError):
    """Raised when an analysis hand-off cannot be trusted."""


@dataclass(slots=True)
class AnalysisArtifact:
    analysis_id: str
    project_id: str | None
    created_at: str
    source: dict[str, Any]
    source_fingerprint: str
    analysis_fingerprint: str
    work_directory: str
    candidate_data_ref: str
    references: dict[str, str]
    candidates: list[dict[str, Any]]
    recommendation: dict[str, Any]
    summary: dict[str, Any]
    status: str = "analysis_ready"
    schema_version: str = ANALYSIS_ARTIFACT_SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.schema_version != ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            raise AnalysisArtifactError("Unsupported analysis artifact schema.")
        if self.status != "analysis_ready":
            raise AnalysisArtifactError("Analysis artifact is not ready for rendering.")
        required = (self.analysis_id, self.source_fingerprint, self.analysis_fingerprint, self.work_directory, self.candidate_data_ref)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise AnalysisArtifactError("Analysis artifact is missing required identifiers.")
        if not isinstance(self.source, dict) or not str(self.source.get("id") or "").strip():
            raise AnalysisArtifactError("Analysis artifact does not identify its source.")
        if not isinstance(self.candidates, list):
            raise AnalysisArtifactError("Analysis artifact candidates are invalid.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "AnalysisArtifact":
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            raise AnalysisArtifactError("Analysis artifact file is missing or corrupted.")
        artifact = cls(
            analysis_id=str(raw.get("analysis_id") or ""),
            project_id=str(raw["project_id"]) if raw.get("project_id") else None,
            created_at=str(raw.get("created_at") or ""),
            source=dict(raw.get("source") or {}),
            source_fingerprint=str(raw.get("source_fingerprint") or ""),
            analysis_fingerprint=str(raw.get("analysis_fingerprint") or ""),
            work_directory=str(raw.get("work_directory") or ""),
            candidate_data_ref=str(raw.get("candidate_data_ref") or ""),
            references={str(key): str(value) for key, value in dict(raw.get("references") or {}).items()},
            candidates=[dict(item) for item in raw.get("candidates", []) if isinstance(item, dict)],
            recommendation=dict(raw.get("recommendation") or {}),
            summary=dict(raw.get("summary") or {}),
            status=str(raw.get("status") or ""),
            schema_version=str(raw.get("schema_version") or ""),
            warnings=[str(item) for item in raw.get("warnings", [])],
        )
        artifact.validate()
        return artifact


def candidate_review_payload(candidate: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    """Expose only review-relevant fields; full evidence remains in cache."""

    candidate_id = str(candidate.get("id") or "")
    viral = candidate.get("virality") if isinstance(candidate.get("virality"), dict) else {}
    potential = viral.get("viral_potential") if isinstance(viral.get("viral_potential"), dict) else {}
    eligibility = viral.get("eligibility") if isinstance(viral.get("eligibility"), dict) else {}
    return {
        "candidate_id": candidate_id,
        "start": candidate.get("start"),
        "end": candidate.get("end"),
        "duration": candidate.get("duration"),
        "text": candidate.get("text", ""),
        "reason": candidate.get("reason", ""),
        "score": candidate.get("score"),
        "state": "analyzed",
        "selected_by_recommendation": candidate_id in selected_ids,
        "recommendation_status": "recommended" if candidate_id in selected_ids else "analyzed",
        "virality_level": potential.get("level"),
        "publishability_status": eligibility.get("status"),
        "warnings": list(candidate.get("warnings") or []),
    }


def new_analysis_artifact(**kwargs: Any) -> AnalysisArtifact:
    """Small constructor boundary that keeps the timestamp creation consistent."""

    return AnalysisArtifact(created_at=utc_now(), **kwargs)
