from __future__ import annotations

"""Persistent review contract for assembled draft candidates."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils import read_json, utc_now, write_json


DRAFT_ARTIFACT_SCHEMA_VERSION = "1.1"
LEGACY_DRAFT_ARTIFACT_SCHEMA_VERSION = "1.0"
LEGACY_DRAFT_WARNING = (
    "LEGACY_DRAFT_ARTIFACT_1_0: analysis lineage predates the immutable run/checksum contract."
)
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
    # An in-progress draft artifact is owned by one isolated engine run.  The
    # field is optional to keep already completed v1.0 artifacts readable.
    run_id: str = ""
    analysis_run_id: str = ""
    analysis_artifact_sha256: str = ""

    def validate(self) -> None:
        if self.schema_version not in {
            DRAFT_ARTIFACT_SCHEMA_VERSION, LEGACY_DRAFT_ARTIFACT_SCHEMA_VERSION,
        }:
            raise DraftArtifactError("Unsupported draft artifact schema.")
        if self.status not in {"draft_ready", "draft_partial"}:
            raise DraftArtifactError("Draft artifact is not ready for review.")
        if not all(isinstance(value, str) and value.strip() for value in (
            self.draft_id, self.analysis_id, self.analysis_fingerprint,
            self.analysis_artifact_path, self.source_fingerprint,
        )):
            raise DraftArtifactError("Draft artifact is missing required identifiers.")
        if not isinstance(self.run_id, str):
            raise DraftArtifactError("Draft artifact run identifier is invalid.")
        if self.schema_version == DRAFT_ARTIFACT_SCHEMA_VERSION:
            if not self.analysis_run_id.strip():
                raise DraftArtifactError("Draft artifact is missing its analysis run identity.")
            checksum = self.analysis_artifact_sha256
            if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
                raise DraftArtifactError("Draft artifact analysis checksum is invalid.")
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
            run_id=str(raw.get("run_id") or ""),
            analysis_run_id=str(raw.get("analysis_run_id") or ""),
            analysis_artifact_sha256=str(raw.get("analysis_artifact_sha256") or ""),
        )
        if artifact.schema_version == LEGACY_DRAFT_ARTIFACT_SCHEMA_VERSION:
            _append_warning(artifact.warnings, LEGACY_DRAFT_WARNING)
        artifact.validate()
        return artifact


def new_draft_artifact(**kwargs: Any) -> DraftArtifact:
    if not kwargs.get("analysis_run_id") or not kwargs.get("analysis_artifact_sha256"):
        kwargs.setdefault("schema_version", LEGACY_DRAFT_ARTIFACT_SCHEMA_VERSION)
    return DraftArtifact(created_at=utc_now(), **kwargs)


def _append_warning(warnings: list[str], value: str) -> None:
    if value not in warnings:
        warnings.append(value)
