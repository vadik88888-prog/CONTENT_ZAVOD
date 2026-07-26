from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


class RunStatus:
    PREPARING = "preparing"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    ACTIVE: ClassVar[frozenset[str]] = frozenset({PREPARING, RUNNING, CANCELLING})
    ALL: ClassVar[frozenset[str]] = frozenset({
        PREPARING, RUNNING, CANCELLING, COMPLETED, COMPLETED_WITH_WARNINGS,
        FAILED, CANCELLED, INTERRUPTED,
    })


class RunKind:
    FULL = "full"
    RENDER_REVISION = "render_revision"
    ALL: ClassVar[frozenset[str]] = frozenset({FULL, RENDER_REVISION})


@dataclass(slots=True)
class ProjectRun:
    run_id: str
    project_id: str
    started_at: str
    finished_at: str | None
    status: str
    settings_snapshot: dict[str, Any]
    source_snapshot: dict[str, Any]
    pipeline_version: str
    artifact_paths: list[str] = field(default_factory=list)
    report_path: str | None = None
    log_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    error_summary: str | None = None
    technical_details: str | None = None
    cost_estimate: float | None = None
    actual_cost: float | None = None
    run_kind: str = RunKind.FULL
    parent_run_id: str | None = None
    changed_settings: dict[str, Any] = field(default_factory=dict)
    invalidated_stages: list[str] = field(default_factory=list)
    schema_version: int = 2

    def validate(self) -> None:
        if not self.run_id or not self.project_id or self.status not in RunStatus.ALL:
            raise ValueError("Run record is invalid.")
        if self.actual_cost is not None and not isinstance(self.actual_cost, (int, float)):
            raise ValueError("Actual cost must be numeric or null.")
        if self.run_kind not in RunKind.ALL:
            raise ValueError("Unsupported run kind.")
        if self.run_kind == RunKind.RENDER_REVISION and not self.parent_run_id:
            raise ValueError("Render revision needs its parent run.")
        if self.schema_version != 2:
            raise ValueError("Unsupported run schema version.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectRun":
        run = cls(
            run_id=str(value["run_id"]),
            project_id=str(value["project_id"]),
            started_at=str(value["started_at"]),
            finished_at=str(value["finished_at"]) if value.get("finished_at") else None,
            status=str(value["status"]),
            settings_snapshot=dict(value.get("settings_snapshot") or {}),
            source_snapshot=dict(value.get("source_snapshot") or {}),
            pipeline_version=str(value.get("pipeline_version", "unknown")),
            artifact_paths=[str(item) for item in value.get("artifact_paths", [])],
            report_path=str(value["report_path"]) if value.get("report_path") else None,
            log_path=str(value["log_path"]) if value.get("log_path") else None,
            warnings=[str(item) for item in value.get("warnings", [])],
            error_summary=str(value["error_summary"]) if value.get("error_summary") else None,
            technical_details=str(value["technical_details"]) if value.get("technical_details") else None,
            cost_estimate=float(value["cost_estimate"]) if value.get("cost_estimate") is not None else None,
            actual_cost=float(value["actual_cost"]) if value.get("actual_cost") is not None else None,
            run_kind=str(value.get("run_kind", RunKind.FULL)),
            parent_run_id=str(value["parent_run_id"]) if value.get("parent_run_id") else None,
            changed_settings=dict(value.get("changed_settings") or {}),
            invalidated_stages=[str(item) for item in value.get("invalidated_stages", [])],
            # Schema v1 records are ordinary full runs with no revision metadata.
            schema_version=2,
        )
        run.validate()
        return run
