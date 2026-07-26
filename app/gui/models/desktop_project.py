from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from app.utils import utc_now


class ProjectStatus:
    DRAFT = "draft"
    READY = "ready"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    ALL: ClassVar[frozenset[str]] = frozenset({
        DRAFT, READY, QUEUED, PROCESSING, COMPLETED, COMPLETED_WITH_WARNINGS,
        FAILED, CANCELLED, INTERRUPTED,
    })


@dataclass(slots=True)
class ProjectOptions:
    """Only settings backed by current engine switches are exposed by the GUI."""

    subtitles_enabled: bool = True
    subtitle_style: str = "documentary"
    encoder: str = "auto"
    use_cache: bool = True
    recompute_all: bool = False

    def validate(self) -> None:
        if self.subtitle_style not in {"minimal", "documentary", "dynamic", "clean"}:
            raise ValueError("Unsupported subtitle style.")
        if self.encoder not in {"auto", "cpu", "nvenc"}:
            raise ValueError("Unsupported encoder.")
        if not all(isinstance(item, bool) for item in (
            self.subtitles_enabled, self.use_cache, self.recompute_all,
        )):
            raise ValueError("Project options must contain booleans.")


@dataclass(slots=True)
class DesktopProject:
    """Durable user project.  The source is referenced, never copied."""

    project_id: str
    name: str
    created_at: str
    updated_at: str
    source_path: str
    project_directory: str
    status: str = ProjectStatus.DRAFT
    settings: ProjectOptions = field(default_factory=ProjectOptions)
    latest_run_id: str | None = None
    thumbnail_path: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if not self.project_id or not self.name.strip():
            raise ValueError("Project id and name are required.")
        if self.status not in ProjectStatus.ALL:
            raise ValueError("Unsupported project status.")
        if self.schema_version != 1:
            raise ValueError("Unsupported project schema version.")
        self.settings.validate()

    @property
    def source(self) -> Path:
        return Path(self.source_path)

    @property
    def directory(self) -> Path:
        return Path(self.project_directory)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DesktopProject":
        settings = value.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("Project settings are corrupted.")
        project = cls(
            project_id=str(value["project_id"]),
            name=str(value["name"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            source_path=str(value["source_path"]),
            project_directory=str(value["project_directory"]),
            status=str(value.get("status", ProjectStatus.DRAFT)),
            settings=ProjectOptions(**settings),
            latest_run_id=str(value["latest_run_id"]) if value.get("latest_run_id") else None,
            thumbnail_path=str(value["thumbnail_path"]) if value.get("thumbnail_path") else None,
            source_metadata=dict(value.get("source_metadata") or {}),
            schema_version=int(value.get("schema_version", 1)),
        )
        project.validate()
        return project
