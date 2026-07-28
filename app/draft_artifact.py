from __future__ import annotations

"""Persistent review contract for assembled draft candidates."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils import read_json, utc_now, write_json


DRAFT_ARTIFACT_SCHEMA_VERSION = "1.0"
CANDIDATE_DRAFT_STATES = frozenset({
    "analyzed", "draft_planning", "draft_ready", "draft_failed",
    "selected", "production_rendering", "rendered",
})


class DraftArtifactError(ValueError):
    pass


@dataclass(slots=True)
class DraftArtifact:
    draft_id: str
    analysis_id: str
    analysis_fingerprint: str
    analysis_artifact_path: str
    project_id: str | None
    source_fingerprint: str
    created_at: str
    candidates: list[dict[str, Any]]
    status: str = "draft_ready"
    schema_version: str = DRAFT_ARTIFACT_SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.schema_version != DRAFT_ARTIFACT_SCHEMA_VERSION:
            raise DraftArtifactError("Unsupported draft artifact schema.")
        if self.status not in {"draft_ready", "draft_partial"}:
            raise DraftArtifactError("Draft artifact is not ready for review.")
        if not all(isinstance(value, str) and value.strip() for value in (
            self.draft_id, self.analysis_id, self.analysis_fingerprint,
            self.analysis_artifact_path, self.source_fingerprint,
        )):
            raise DraftArtifactError("Draft artifact is missing required identifiers.")
        for candidate in self.candidates:
            if not isinstance(candidate, dict) or not str(candidate.get("candidate_id") or "").strip():
                raise DraftArtifactError("Draft candidate is malformed.")
            if str(candidate.get("state") or "") not in CANDIDATE_DRAFT_STATES:
                raise DraftArtifactError("Draft candidate has an unsupported state.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "DraftArtifact":
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            raise DraftArtifactError("Draft artifact file is missing or corrupted.")
        artifact = cls(
            draft_id=str(raw.get("draft_id") or ""),
            analysis_id=str(raw.get("analysis_id") or ""),
            analysis_fingerprint=str(raw.get("analysis_fingerprint") or ""),
            analysis_artifact_path=str(raw.get("analysis_artifact_path") or ""),
            project_id=str(raw["project_id"]) if raw.get("project_id") else None,
            source_fingerprint=str(raw.get("source_fingerprint") or ""),
            created_at=str(raw.get("created_at") or ""),
            candidates=[dict(item) for item in raw.get("candidates", []) if isinstance(item, dict)],
            status=str(raw.get("status") or ""),
            schema_version=str(raw.get("schema_version") or ""),
            warnings=[str(item) for item in raw.get("warnings", [])],
        )
        artifact.validate()
        return artifact


def new_draft_artifact(**kwargs: Any) -> DraftArtifact:
    return DraftArtifact(created_at=utc_now(), **kwargs)
