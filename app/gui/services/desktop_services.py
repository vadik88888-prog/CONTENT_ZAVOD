from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.gui.models import DesktopProject, DesktopSettings, ProjectRun, ProjectStatus, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError
from app.gui.services.error_mapping import redact_secrets
from app.gui.services.pipeline_facade import PipelineCompletion, PipelineFacade, PreparedPipelineRun
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.utils import utc_now


@dataclass(slots=True)
class DesktopServices:
    """Application-service boundary shared by view-models and Qt widgets."""

    engine_root: Path
    settings_store: SettingsStore
    settings: DesktopSettings
    projects: DesktopProjectStore
    runs: RunHistoryStore
    pipeline: PipelineFacade
    system: SystemService

    @classmethod
    def create(cls, engine_root: Path) -> "DesktopServices":
        settings_store = SettingsStore()
        settings = settings_store.load()
        projects = DesktopProjectStore(Path(settings.data_directory))
        runs = RunHistoryStore(projects)
        services = cls(
            engine_root=engine_root.resolve(),
            settings_store=settings_store,
            settings=settings,
            projects=projects,
            runs=runs,
            pipeline=PipelineFacade(engine_root),
            system=SystemService(engine_root),
        )
        services.recover_interrupted_runs()
        return services

    def save_settings(self) -> None:
        self.settings_store.save(self.settings)

    def reconfigure_data_directory(self, directory: Path) -> None:
        self.settings.data_directory = str(directory.expanduser().resolve())
        self.settings.validate()
        self.projects = DesktopProjectStore(Path(self.settings.data_directory))
        self.runs = RunHistoryStore(self.projects)
        self.save_settings()
        self.recover_interrupted_runs()

    def list_projects(self) -> list[DesktopProject]:
        return self.projects.list()

    def create_project(self, source_path: str | Path) -> DesktopProject:
        metadata = self.pipeline.inspect_source(source_path)
        return self.projects.create(source_path, source_metadata=metadata)

    def save_project(self, project: DesktopProject) -> None:
        project.settings.validate()
        self.projects.save(project)

    def delete_project(self, project_id: str) -> None:
        self.projects.delete(project_id)

    def runs_for(self, project: DesktopProject) -> list[ProjectRun]:
        return self.runs.list(project.project_id)

    def processing_estimate(self, project: DesktopProject):
        """A serialisable preflight estimate for the project screen."""

        return self.pipeline.plan_processing(project, self.settings)[2]

    def prepare_run(self, project: DesktopProject) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status == ProjectStatus.PROCESSING:
            raise RuntimeError("Этот проект уже обрабатывается.")
        source = Path(project.source_path)
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        run = self.runs.create(
            project,
            settings_snapshot={
                "project_options": {
                    "processing_mode": project.settings.processing_mode,
                    "deep_analysis": project.settings.deep_analysis,
                    "platform": project.settings.platform,
                    "clip_count": project.settings.clip_count,
                    "subtitles_enabled": project.settings.subtitles_enabled,
                    "subtitle_style": project.settings.subtitle_style,
                    "encoder": project.settings.encoder,
                    "use_cache": project.settings.use_cache,
                    "recompute_all": project.settings.recompute_all,
                },
                "product_flow": {
                    "user_intent": intent.to_dict(),
                    "resolved_config": resolved.to_dict(),
                    "estimate": estimate.to_dict(),
                },
                "local_test_mode": self.settings.local_test_mode,
            },
            source_snapshot={"path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            pipeline_version="0.1.0",
        )
        run.cost_estimate = estimate.estimated_ai_cost_max
        self.runs.save(run)
        try:
            prepared = self.pipeline.prepare(project, run, self.settings)
        except Exception:
            run.status = RunStatus.FAILED
            run.finished_at = utc_now()
            run.error_summary = "Не удалось подготовить запуск."
            self.runs.save(run)
            raise
        # This must happen before Qt starts the child process.  A silent CLI
        # must never leave a run without an immediately inspectable log.
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.PROCESSING
        project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def record_launch_context(self, run: ProjectRun, prepared: PreparedPipelineRun) -> None:
        """Persist the desktop launch contract without leaking environment secrets."""

        flags = [argument for argument in prepared.arguments if argument.startswith("--")]
        self.append_log(run, "Desktop pipeline launch prepared.")
        self.append_log(run, f"command: {prepared.command_line()}")
        self.append_log(run, f"cwd: {prepared.working_directory}")
        self.append_log(run, f"source: {prepared.source_path or '<unknown>'}")
        self.append_log(run, f"runtime config: {prepared.runtime_config_path}")
        self.append_log(run, f"output directory: {prepared.output_directory}")
        self.append_log(run, f"flags: {' '.join(flags) if flags else '<none>'}")
        runtime_flags = "; ".join(
            f"{key}={value}" for key, value in prepared.runtime_flags.items()
        )
        self.append_log(run, f"runtime flags: {runtime_flags or '<unknown>'}")
        self.append_log(
            run,
            "environment: PYTHONUNBUFFERED=1; PYTHONIOENCODING=utf-8; "
            "inherited environment values are intentionally not logged.",
        )

    def append_log(self, run: ProjectRun, line: str) -> None:
        if not run.log_path:
            return
        path = Path(run.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        limit = 2 * 1024 * 1024
        if path.exists() and path.stat().st_size >= limit:
            rotated = path.with_suffix(".1.log")
            if rotated.exists():
                rotated.unlink()
            path.replace(rotated)
        with path.open("a", encoding="utf-8") as file:
            file.write(f"{utc_now()} {redact_secrets(line).rstrip()}\n")

    def finish_success(self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun) -> ProjectRun:
        completion = self.pipeline.completion(prepared)
        if completion.error_summary:
            return self.finish_failure(project, run, completion.error_summary, completion.technical_details)
        run.status = RunStatus.COMPLETED_WITH_WARNINGS if completion.warnings else RunStatus.COMPLETED
        run.finished_at = utc_now()
        run.warnings = completion.warnings
        run.cost_estimate = completion.cost_estimate
        run.actual_cost = None  # Local estimates are intentionally never treated as billed cost.
        self.runs.snapshot_report_and_outputs(run, completion.report_path, completion.output_files)
        self.runs.save(run)
        project.status = ProjectStatus.COMPLETED_WITH_WARNINGS if completion.warnings else ProjectStatus.COMPLETED
        project.latest_run_id = run.run_id
        self.projects.save(project)
        return run

    def finish_failure(
        self, project: DesktopProject, run: ProjectRun, message: str, technical_details: str | None = None,
    ) -> ProjectRun:
        run.status = RunStatus.FAILED
        run.finished_at = utc_now()
        run.error_summary = redact_secrets(message)
        run.technical_details = redact_secrets(technical_details or message)
        self.append_log(run, run.technical_details)
        self.runs.save(run)
        project.status = ProjectStatus.FAILED
        self.projects.save(project)
        return run

    def finish_cancelled(self, project: DesktopProject, run: ProjectRun) -> ProjectRun:
        run.status = RunStatus.CANCELLED
        run.finished_at = utc_now()
        run.error_summary = "Создание ролика отменено пользователем."
        self.runs.save(run)
        project.status = ProjectStatus.CANCELLED
        self.projects.save(project)
        return run

    def recover_interrupted_runs(self) -> int:
        return sum(self.runs.mark_interrupted(project) for project in self.projects.list())
