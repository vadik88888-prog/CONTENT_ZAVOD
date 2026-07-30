from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from app.gui.models import DesktopProject, ProjectRun, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore, PersistenceError
from app.utils import read_json, utc_now, write_json


class RunHistoryStore:
    """Append-only run records and snapshots, independent from engine report.json."""

    def __init__(self, project_store: DesktopProjectStore) -> None:
        self.project_store = project_store

    def runs_directory(self, project_id: str) -> Path:
        return self.project_store.project_directory(project_id) / "runs"

    def run_directory(self, project_id: str, run_id: str) -> Path:
        if not run_id or any(part in run_id for part in ("/", "\\", "..")):
            raise PersistenceError("Некорректный идентификатор запуска.")
        return self.runs_directory(project_id) / run_id

    def create(
        self, project: DesktopProject, settings_snapshot: dict, source_snapshot: dict, pipeline_version: str,
        *, run_kind: str = RunKind.FULL, parent_run_id: str | None = None,
        changed_settings: dict | None = None, invalidated_stages: list[str] | None = None,
    ) -> ProjectRun:
        run_id = uuid.uuid4().hex
        directory = self.run_directory(project.project_id, run_id)
        directory.mkdir(parents=True, exist_ok=False)
        run = ProjectRun(
            run_id=run_id,
            project_id=project.project_id,
            started_at=utc_now(),
            finished_at=None,
            status=RunStatus.PREPARING,
            settings_snapshot=dict(settings_snapshot),
            source_snapshot=dict(source_snapshot),
            pipeline_version=pipeline_version,
            run_kind=run_kind,
            parent_run_id=parent_run_id,
            changed_settings=dict(changed_settings or {}),
            invalidated_stages=list(invalidated_stages or []),
            log_path=str(directory / "pipeline.log"),
        )
        self.save(run)
        return run

    def save(self, run: ProjectRun) -> None:
        write_json(self.run_directory(run.project_id, run.run_id) / "run.json", run.to_dict())

    def load(self, project_id: str, run_id: str) -> ProjectRun:
        try:
            raw = read_json(self.run_directory(project_id, run_id) / "run.json")
            if not isinstance(raw, dict):
                raise ValueError("Run JSON root is not an object.")
            run = ProjectRun.from_dict(raw)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PersistenceError("Не удалось открыть запись запуска.") from error
        if run.project_id != project_id or run.run_id != run_id:
            raise PersistenceError("Запись запуска не соответствует проекту.")
        return run

    def list(self, project_id: str) -> list[ProjectRun]:
        directory = self.runs_directory(project_id)
        if not directory.exists():
            return []
        runs: list[ProjectRun] = []
        for item in directory.iterdir():
            if not item.is_dir() or item.is_symlink():
                continue
            try:
                runs.append(self.load(project_id, item.name))
            except PersistenceError:
                continue
        return sorted(runs, key=lambda item: item.started_at, reverse=True)

    def mark_interrupted(self, project: DesktopProject) -> bool:
        changed = False
        for run in self.list(project.project_id):
            if run.status in RunStatus.ACTIVE:
                run.status = RunStatus.INTERRUPTED
                run.finished_at = utc_now()
                run.error_summary = "Предыдущий запуск был прерван при закрытии приложения."
                self.save(run)
                changed = True
        if changed and project.status == "processing":
            project.status = "interrupted"
            self.project_store.save(project)
        return changed

    def snapshot_report_and_outputs(self, run: ProjectRun, report_path: Path, output_files: list[Path]) -> ProjectRun:
        """Keep immutable history without copying the source video.

        Reports are copied; generated final files use a hard link on one volume and
        fall back to a byte copy only when Windows cannot link across volumes.
        """

        directory = self.run_directory(run.project_id, run.run_id)
        artifacts = directory / "artifacts"
        artifacts.mkdir(exist_ok=True)
        stored: list[str] = []
        if report_path.is_file():
            report_copy = directory / "report.json"
            shutil.copy2(report_path, report_copy)
            run.report_path = str(report_copy)
            stored.append(str(report_copy))
        for index, source in enumerate(output_files, start=1):
            if not source.is_file():
                continue
            target = artifacts / source.name
            # Every fast draft preview is deliberately named
            # ``draft-preview.mp4`` inside its candidate-owned directory.  A
            # run-history snapshot must retain all of them rather than letting
            # the last candidate overwrite the earlier previews.
            if target.exists() and source != target:
                target = artifacts / f"{index:02d}-{source.name}"
            if target.exists():
                target.unlink()
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
            stored.append(str(target))
        run.artifact_paths = stored
        self.save(run)
        return run
