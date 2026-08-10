from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from app.candidate_review import validate_boundary_override
from app.gui.models import DesktopProject, DesktopSettings, ProjectRun, ProjectStatus, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError, validate_video_path
from app.gui.services.error_mapping import redact_secrets
from app.gui.services.pipeline_facade import (
    STATE_PERSISTENCE_WARNING,
    PipelineCompletion,
    PipelineFacade,
    PreparedPipelineRun,
)
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.product_flow import calibrate_processing_estimate
from app.source_download import cleanup_partial_downloads, validate_public_video_url
from app.utils import read_json, stable_text_hash, utc_now


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
        services.recover_ready_analysis_runs()
        services.recover_interrupted_downloads()
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
        self.recover_ready_analysis_runs()

    def list_projects(self) -> list[DesktopProject]:
        return self.projects.list()

    def create_project(self, source_path: str | Path) -> DesktopProject:
        source = self._resolve_valid_source(source_path)
        metadata = self.pipeline.inspect_source(source)
        project = self.projects.create(source, source_metadata=metadata)
        self._refresh_setup_state(project, "Источник готов. Настройте обработку и запустите анализ, когда будете готовы.")
        self.projects.save(project)
        return project

    def create_url_project(self, url: str, metadata: dict) -> DesktopProject:
        project = self.projects.create_url(validate_public_video_url(url), metadata)
        self._refresh_setup_state(project, "Ссылка проверена. Настройте обработку; загрузка начнётся только после вашего запуска.")
        self.projects.save(project)
        return project

    def mark_url_download_started(self, project: DesktopProject) -> None:
        if project.source_spec.kind != "url":
            raise InputValidationError("Этот проект использует локальный файл.")
        project.source_spec.download_state = "downloading"
        project.source_spec.error_message = None
        self.projects.save(project)

    def complete_url_download(self, project: DesktopProject, path: str | Path) -> DesktopProject:
        source = self._resolve_valid_source(path)
        source_directory = (Path(project.project_directory) / "sources").resolve()
        if not source.is_relative_to(source_directory):
            raise InputValidationError("Загруженный файл должен находиться в папке проекта.")
        metadata = self.pipeline.inspect_source(source)
        project.source_path = str(source)
        project.source_metadata = metadata
        project.source_spec.downloaded_path = str(source)
        project.source_spec.metadata = metadata
        project.source_spec.download_state = "downloaded"
        project.source_spec.error_message = None
        project.status = ProjectStatus.SOURCE_READY
        self._refresh_setup_state(project, "Видео загружено. Настройки и оценка обновлены по фактическому файлу.")
        self.projects.save(project)
        return project

    @staticmethod
    def _resolve_valid_source(path: str | Path) -> Path:
        """Use one validation contract before either source enters the project flow."""

        return validate_video_path(path)

    def fail_url_download(self, project: DesktopProject, message: str, *, cancelled: bool = False) -> None:
        if project.source_spec.kind != "url":
            return
        project.source_spec.download_state = "cancelled" if cancelled else "failed"
        project.source_spec.error_message = redact_secrets(message)
        self.projects.save(project)

    def recover_interrupted_downloads(self) -> int:
        """Make an abandoned link download safely repeatable after restart.

        A QProcess cannot survive the desktop client, so a persisted
        ``downloading`` value must never be presented as a live transfer on
        the next launch. Only project-local partial markers are removed.
        """

        recovered = 0
        for project in self.projects.list():
            source = project.source_spec
            if source.kind != "url" or source.download_state != "downloading":
                continue
            cleanup_partial_downloads(project.directory / "sources")
            source.download_state = "cancelled"
            source.error_message = "Загрузка была прервана при закрытии приложения. Её можно начать снова."
            self.projects.save(project)
            recovered += 1
        return recovered

    def save_project(self, project: DesktopProject) -> None:
        project.settings.validate()
        self.projects.save(project)

    def update_project_options(self, project: DesktopProject, **values: object) -> DesktopProject:
        """Persist a setup change and tell the person what the next run will reuse.

        The method never launches or invalidates pipeline artifacts.  Existing
        candidates and drafts remain inspectable; only the setup explanation
        describes whether a later analysis is needed for a changed option.
        """

        changed: list[str] = []
        for name, value in values.items():
            if not hasattr(project.settings, name):
                continue
            if getattr(project.settings, name) != value:
                setattr(project.settings, name, value)
                changed.append(name)
        project.settings.validate()
        if not changed:
            return project
        has_analysis = bool(project.analysis_artifact_path)
        # Clip count is a post-analysis Top-N choice over the saved ranking;
        # changing it must never re-run Brain/Vision.
        analysis_options = {"processing_mode", "deep_analysis"}
        needs_analysis = has_analysis and bool(analysis_options.intersection(changed))
        preview_options = {
            "platform", "subtitles_enabled", "subtitle_style",
            "composition_strategy", "same_source_broll_allowed", "encoder",
        }
        stale_preview_ids = [
            candidate_id
            for candidate_id in project.review_selected_candidate_ids
            if candidate_id in project.candidate_draft_artifacts
            and project.candidate_states.get(candidate_id) in {"draft_ready", "selected"}
        ] if preview_options.intersection(changed) else []
        for candidate_id in stale_preview_ids:
            # Keep the last valid artifact addressable while the 7G cache
            # creates a dependency-bounded revision. Approval is intentionally
            # cleared until the replacement passes parity/QC.
            self._set_candidate_lifecycle(
                project, candidate_id, draft="pending", approval="pending", export="pending",
            )
            project.selected_candidate_ids = [
                item for item in project.selected_candidate_ids if item != candidate_id
            ]
        if needs_analysis:
            summary = (
                "Настройки сохранены для следующего анализа. Текущие найденные моменты и черновики не изменены; "
                "чтобы применить этот режим, выполните новый анализ видео."
            )
            reused = ["текущие моменты и черновики остаются доступными для просмотра"]
        elif stale_preview_ids:
            summary = (
                "Настройки сохранены. Предпросмотр нужно обновить только для выбранных черновиков; "
                "предыдущая готовая версия останется доступной до завершения обновления."
            )
            reused = ["сохранённый анализ", "найденные моменты", "готовые части предыдущего предпросмотра"]
        elif has_analysis:
            summary = (
                "Настройки сохранены. Сохранённый анализ и найденные моменты можно использовать повторно; "
                "новый анализ не нужен."
            )
            reused = ["сохранённый анализ", "найденные моменты"]
        else:
            summary = "Настройки сохранены и будут применены к первому анализу."
            reused = []
        self._refresh_setup_state(project, summary, needs_new_analysis=needs_analysis, reused_stages=reused)
        self.projects.save(project)
        return project

    def _refresh_setup_state(
        self,
        project: DesktopProject,
        summary: str | None = None,
        *,
        needs_new_analysis: bool | None = None,
        reused_stages: list[str] | None = None,
    ) -> None:
        """Store a safe, serialisable estimate for the next time the project opens."""

        try:
            estimate = self.processing_estimate(project)
        except Exception:
            # A first-run configuration can be incomplete.  Keep the source and
            # choices durable and let the screen show its honest fallback state.
            return
        project.setup_state.last_estimate = estimate.to_dict()
        project.setup_state.estimated_at = utc_now()
        if summary is not None:
            project.setup_state.change_summary = summary
        if needs_new_analysis is not None:
            project.setup_state.needs_new_analysis = needs_new_analysis
        if reused_stages is not None:
            project.setup_state.reused_stages = list(reused_stages)

    @staticmethod
    def _ensure_candidate_lifecycle(project: DesktopProject, candidate_id: str) -> None:
        """Populate independent lifecycle axes for legacy combined states."""

        state = project.candidate_states.get(candidate_id, "analyzed")
        project.candidate_draft_statuses.setdefault(
            candidate_id,
            "running" if state == "draft_planning" else
            "failed" if state == "draft_failed" else
            "ready" if state in {"draft_ready", "selected", "production_rendering", "rendered"} else "pending",
        )
        project.candidate_approval_states.setdefault(
            candidate_id,
            "approved" if state in {"selected", "production_rendering", "rendered"} else "pending",
        )
        project.candidate_export_statuses.setdefault(
            candidate_id,
            "running" if state == "production_rendering" else
            "ready" if state == "rendered" else "pending",
        )

    @classmethod
    def _set_candidate_lifecycle(
        cls,
        project: DesktopProject,
        candidate_id: str,
        *,
        draft: str | None = None,
        approval: str | None = None,
        export: str | None = None,
    ) -> None:
        """Update an item atomically and keep the old screen projection honest."""

        cls._ensure_candidate_lifecycle(project, candidate_id)
        if draft is not None:
            project.candidate_draft_statuses[candidate_id] = draft
        if approval is not None:
            project.candidate_approval_states[candidate_id] = approval
        if export is not None:
            project.candidate_export_statuses[candidate_id] = export
        draft_state = project.candidate_draft_statuses[candidate_id]
        approval_state = project.candidate_approval_states[candidate_id]
        export_state = project.candidate_export_statuses[candidate_id]
        if draft_state == "failed":
            project.candidate_states[candidate_id] = "draft_failed"
        elif export_state == "ready":
            project.candidate_states[candidate_id] = "rendered"
        elif export_state == "running":
            project.candidate_states[candidate_id] = "production_rendering"
        elif draft_state == "running":
            project.candidate_states[candidate_id] = "draft_planning"
        elif draft_state == "ready" and approval_state == "approved":
            project.candidate_states[candidate_id] = "selected"
        elif draft_state == "ready":
            project.candidate_states[candidate_id] = "draft_ready"
        else:
            project.candidate_states[candidate_id] = "analyzed"

    def set_active_preview_candidate(
        self, project: DesktopProject, candidate_id: str | None,
    ) -> DesktopProject:
        """Persist a deliberate review-card selection for stable reopen/navigation."""

        value = str(candidate_id).strip() if candidate_id else None
        if value is not None and value not in project.candidate_states:
            raise InputValidationError("Выбранный фрагмент отсутствует в сохранённом анализе.")
        project.active_preview_candidate_id = value
        self.projects.save(project)
        return project

    def set_review_selection(self, project: DesktopProject, candidate_ids: list[str]) -> DesktopProject:
        """Persist the user's candidate choice before any draft or render starts."""

        unique = list(dict.fromkeys(str(item) for item in candidate_ids if str(item)))
        unknown = [item for item in unique if item not in project.candidate_states]
        if unknown:
            raise InputValidationError("Один из выбранных моментов отсутствует в сохранённом анализе.")
        project.review_selected_candidate_ids = unique
        for candidate_id in unique:
            self._ensure_candidate_lifecycle(project, candidate_id)
        removed = set(project.selected_candidate_ids) - set(unique)
        for candidate_id in removed:
            self._set_candidate_lifecycle(project, candidate_id, approval="rejected", export="pending")
        project.selected_candidate_ids = [item for item in project.selected_candidate_ids if item in unique]
        if project.active_preview_candidate_id and project.active_preview_candidate_id not in project.candidate_states:
            project.active_preview_candidate_id = None
        if project.status not in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            project.status = ProjectStatus.REVIEWING_CANDIDATES if project.analysis_artifact_path else project.status
        self.projects.save(project)
        return project

    def delete_project(self, project_id: str) -> None:
        self.projects.delete(project_id)

    def runs_for(self, project: DesktopProject) -> list[ProjectRun]:
        return self.runs.list(project.project_id)

    def processing_estimate(self, project: DesktopProject):
        """A serialisable preflight estimate for the project screen."""

        return self.setup_preflight(project)[1]

    def setup_preflight(self, project: DesktopProject):
        """Resolve the existing pipeline's setup choices for a view-model."""

        _intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        return resolved, calibrate_processing_estimate(estimate, self.runs_for(project))

    def refresh_setup_estimate(self, project: DesktopProject) -> DesktopProject:
        """Refresh the durable preflight when a project is opened again."""

        previous = dict(project.setup_state.last_estimate)
        previous_at = project.setup_state.estimated_at
        self._refresh_setup_state(project)
        if project.setup_state.last_estimate != previous or (project.setup_state.last_estimate and previous_at is None):
            self.projects.save(project)
        return project

    def prepare_run(self, project: DesktopProject) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status == ProjectStatus.PROCESSING:
            raise RuntimeError("Этот проект уже обрабатывается.")
        if not project.settings.recompute_all and project.analysis_artifact_path and Path(project.analysis_artifact_path).is_file():
            project.status = ProjectStatus.ANALYSIS_READY
            self.projects.save(project)
            raise InputValidationError(
                "Сохранённый анализ уже готов. Откройте «Моменты»; повторный анализ не запускался."
            )
        source = project.source
        if not source.is_file():
            if project.source_spec.kind == "url":
                raise InputValidationError("Сначала загрузите видео по ссылке.")
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        estimate = calibrate_processing_estimate(estimate, self.runs_for(project))
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
                    "audio_mode": project.settings.audio_mode,
                    "composition_strategy": project.settings.composition_strategy,
                    "same_source_broll_allowed": project.settings.same_source_broll_allowed,
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
            source_snapshot={
                "kind": project.source_spec.kind,
                "path": str(source),
                "name": source.name,
                "size_bytes": source.stat().st_size,
            },
            pipeline_version="0.1.0",
        )
        run.cost_estimate = estimate.estimated_ai_cost_max
        self.runs.save(run)
        try:
            prepared = self._with_observability_paths(self.pipeline.prepare(project, run, self.settings), run)
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

    def prepare_analysis(self, project: DesktopProject) -> tuple[ProjectRun, PreparedPipelineRun]:
        if not project.settings.recompute_all and project.analysis_artifact_path and Path(project.analysis_artifact_path).is_file():
            project.status = ProjectStatus.ANALYSIS_READY
            self.projects.save(project)
            raise InputValidationError(
                "Сохранённый анализ уже готов. Откройте «Моменты»; повторный анализ не запускался."
            )
        if project.status in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            raise RuntimeError("Этот проект уже обрабатывается.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        run = self.runs.create(
            project,
            settings_snapshot={
                "project_options": asdict(project.settings),
                "product_flow": {"user_intent": intent.to_dict(), "resolved_config": resolved.to_dict(), "estimate": estimate.to_dict()},
                "local_test_mode": self.settings.local_test_mode,
            },
            source_snapshot={"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            pipeline_version="0.1.0", run_kind=RunKind.ANALYSIS,
        )
        run.cost_estimate = estimate.estimated_ai_cost_max
        self.runs.save(run)
        try:
            prepared = self._with_observability_paths(self.pipeline.prepare_analysis(project, run, self.settings), run)
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить анализ."
            self.runs.save(run)
            raise
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.ANALYZING; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def prepare_draft(self, project: DesktopProject, candidate_ids: list[str]) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            raise RuntimeError("Этот проект уже обрабатывается.")
        if not project.analysis_artifact_path:
            raise InputValidationError("Сначала выполните анализ видео.")
        requested_ids = list(dict.fromkeys(str(item) for item in candidate_ids if str(item)))
        if not requested_ids:
            raise InputValidationError("Выберите хотя бы один момент для подготовки черновика.")
        outside_selection = [item for item in requested_ids if item not in project.review_selected_candidate_ids]
        if outside_selection:
            raise InputValidationError("Сначала выберите моменты для подготовки черновика.")
        candidate_ids = []
        stale_ready_ids: list[str] = []
        for candidate_id in requested_ids:
            self._ensure_candidate_lifecycle(project, candidate_id)
            # A completed immutable preview is already the retry boundary.  A
            # batch retry must not recreate it merely because a neighbour failed.
            if project.candidate_draft_statuses.get(candidate_id) == "ready":
                artifact_path = Path(project.candidate_draft_artifacts.get(candidate_id, ""))
                if artifact_path.is_file():
                    continue
                # A persisted "ready" marker is only trustworthy while its
                # candidate-owned artifact is available.  Do not let a
                # deleted preview become an un-retryable phantom: invalidate
                # just this item and rebuild it in the current request.
                self._set_candidate_lifecycle(
                    project, candidate_id, draft="failed", approval="pending", export="pending",
                )
                project.candidate_draft_artifacts.pop(candidate_id, None)
                project.selected_candidate_ids = [
                    item for item in project.selected_candidate_ids if item != candidate_id
                ]
                project.candidate_errors[candidate_id] = (
                    "Сохранённый предпросмотр больше недоступен; "
                    "будет создан заново."
                )
                stale_ready_ids.append(candidate_id)
            candidate_ids.append(candidate_id)
        if not candidate_ids:
            raise InputValidationError("Для выбранных моментов уже есть готовые черновики.")
        if stale_ready_ids:
            # Persist the invalidation before process preparation.  If that
            # preparation itself fails, a restart still offers exactly the
            # missing candidates for retry instead of claiming they are ready.
            self.projects.save(project)
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        run = self.runs.create(
            project, {
                "analysis_id": project.analysis_id,
                "candidate_ids": list(candidate_ids),
                "previous_draft_artifacts": {
                    candidate_id: project.candidate_draft_artifacts[candidate_id]
                    for candidate_id in candidate_ids
                    if candidate_id in project.candidate_draft_artifacts
                    and Path(project.candidate_draft_artifacts[candidate_id]).is_file()
                },
                "local_test_mode": self.settings.local_test_mode,
            },
            {"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            "0.1.0", run_kind=RunKind.DRAFT,
        )
        try:
            prepared = self._with_observability_paths(
                self.pipeline.prepare_draft(project, run, self.settings, candidate_ids), run,
            )
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить черновик."
            self.runs.save(run)
            raise
        for candidate_id in candidate_ids:
            self._set_candidate_lifecycle(
                project, candidate_id, draft="running", approval="pending", export="pending",
            )
            project.candidate_errors.pop(candidate_id, None)
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.PROCESSING; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def select_draft_candidates(self, project: DesktopProject, candidate_ids: list[str]) -> DesktopProject:
        if not project.candidate_draft_artifacts:
            raise InputValidationError("Сначала подготовьте предпросмотр черновика.")
        if not candidate_ids:
            raise InputValidationError("Выберите хотя бы один готовый черновик.")
        eligible: list[str] = []
        for candidate_id in dict.fromkeys(str(item) for item in candidate_ids if str(item)):
            self._ensure_candidate_lifecycle(project, candidate_id)
            if project.candidate_states.get(candidate_id) not in {"draft_ready", "selected"}:
                project.candidate_errors[candidate_id] = "Этап подтверждения: черновик ещё не готов к экспорту."
                continue
            if not Path(project.candidate_draft_artifacts.get(candidate_id, "")).is_file():
                self._set_candidate_lifecycle(
                    project, candidate_id, draft="failed", approval="pending", export="pending",
                )
                project.candidate_errors[candidate_id] = "Сохранённый предпросмотр больше недоступен."
                continue
            eligible.append(candidate_id)
        if not eligible:
            self.projects.save(project)
            raise InputValidationError("Нет готовых черновиков, которые можно отправить в production.")
        for candidate_id, state in list(project.candidate_states.items()):
            if state == "selected" and candidate_id not in eligible:
                self._set_candidate_lifecycle(project, candidate_id, approval="rejected", export="pending")
        for candidate_id in eligible:
            self._set_candidate_lifecycle(project, candidate_id, draft="ready", approval="approved", export="pending")
            project.candidate_errors.pop(candidate_id, None)
        project.selected_candidate_ids = list(eligible)
        project.review_selected_candidate_ids = list(eligible)
        project.status = ProjectStatus.REVIEWING_CANDIDATES
        self.projects.save(project)
        return project

    def set_draft_approval(
        self, project: DesktopProject, candidate_id: str, approved: bool,
    ) -> DesktopProject:
        """Persist one explicit production decision without changing draft selection.

        ``review_selected_candidate_ids`` answers "which moments are being
        prepared or reviewed".  ``selected_candidate_ids`` remains the much
        narrower, deliberate hand-off to the expensive production render.
        Keeping those choices separate prevents an accidental render when a
        user is merely looking through ready drafts.
        """

        if candidate_id not in project.candidate_states:
            raise InputValidationError("Черновик отсутствует в сохранённом анализе.")
        if approved:
            if project.candidate_states.get(candidate_id) not in {"draft_ready", "selected"}:
                raise InputValidationError("Подтвердить можно только готовый черновик.")
            if not Path(project.candidate_draft_artifacts.get(candidate_id, "")).is_file():
                self._set_candidate_lifecycle(
                    project, candidate_id, draft="failed", approval="pending", export="pending",
                )
                project.candidate_errors[candidate_id] = "Сохранённый предпросмотр больше недоступен."
                project.selected_candidate_ids = [
                    item for item in project.selected_candidate_ids if item != candidate_id
                ]
                self.projects.save(project)
                return project
            if candidate_id not in project.selected_candidate_ids:
                project.selected_candidate_ids.append(candidate_id)
            if candidate_id not in project.review_selected_candidate_ids:
                project.review_selected_candidate_ids.append(candidate_id)
            self._set_candidate_lifecycle(project, candidate_id, draft="ready", approval="approved", export="pending")
            project.candidate_errors.pop(candidate_id, None)
        else:
            project.selected_candidate_ids = [
                item for item in project.selected_candidate_ids if item != candidate_id
            ]
            self._set_candidate_lifecycle(project, candidate_id, approval="rejected", export="pending")
        project.status = ProjectStatus.REVIEWING_CANDIDATES
        self.projects.save(project)
        return project

    def adjust_candidate_boundary(
        self, project: DesktopProject, candidate_id: str, boundary: str, delta_seconds: float,
    ) -> tuple[DesktopProject, dict]:
        """Persist one review edit after cached-only boundary revalidation."""

        if boundary not in {"start", "end"} or delta_seconds not in {-1.0, -0.5, 0.5, 1.0}:
            raise InputValidationError("Доступны только шаги границы ±0.5 или ±1.0 секунды.")
        artifact_path = Path(str(project.analysis_artifact_path or ""))
        artifact = read_json(artifact_path, {}) if artifact_path.is_file() else {}
        candidates = artifact.get("candidates", []) if isinstance(artifact, dict) else []
        candidate = next((item for item in candidates if isinstance(item, dict) and str(item.get("candidate_id")) == candidate_id), None)
        if not candidate:
            raise InputValidationError("Кандидат не найден в сохранённом анализе.")
        current = dict(project.candidate_boundary_overrides.get(candidate_id) or {})
        start_value = current.get("start", candidate.get("start", 0))
        end_value = current.get("end", candidate.get("end", 0))
        if start_value is None or end_value is None:
            raise InputValidationError("Не удалось прочитать сохранённые границы момента.")
        try:
            start = float(start_value)
            end = float(end_value)
        except (TypeError, ValueError) as error:
            raise InputValidationError("Не удалось прочитать сохранённые границы момента.") from error
        if boundary == "start":
            start += delta_seconds
        else:
            end += delta_seconds
        references = artifact.get("references", {}) if isinstance(artifact, dict) else {}
        transcript_features = read_json(Path(str(references.get("transcript_features") or "")), {}) if references.get("transcript_features") else {}
        scenes = read_json(Path(str(references.get("scene_boundaries") or "")), {}) if references.get("scene_boundaries") else {}
        validation = validate_boundary_override(
            start, end,
            source_duration=float(project.source_metadata.get("duration")) if project.source_metadata.get("duration") is not None else None,
            minimum_duration=15.0, maximum_duration=60.0,
            transcript_features=transcript_features if isinstance(transcript_features, dict) else {},
            scenes=scenes if isinstance(scenes, dict) else {},
        )
        if not validation["valid"]:
            raise InputValidationError(" ".join(validation["errors"]))
        project.candidate_boundary_overrides[candidate_id] = {
            "start": validation["start"], "end": validation["end"],
            "warnings": validation["warnings"], "revalidation": validation["revalidation"],
            "candidate_boundary_fingerprint": stable_text_hash(
                f"{project.analysis_fingerprint or ''}:{candidate_id}:{validation['start']:.3f}:{validation['end']:.3f}"
            ),
        }
        self._set_candidate_lifecycle(
            project, candidate_id, draft="pending", approval="pending", export="pending",
        )
        project.candidate_errors.pop(candidate_id, None)
        project.selected_candidate_ids = [item for item in project.selected_candidate_ids if item != candidate_id]
        self.projects.save(project)
        return project, validation

    def prepare_selected_render(
        self, project: DesktopProject, candidate_ids: list[str] | None = None,
    ) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            raise RuntimeError("Этот проект уже обрабатывается.")
        approved_ids = list(dict.fromkeys(project.selected_candidate_ids))
        if candidate_ids is None:
            candidate_ids = approved_ids
        else:
            candidate_ids = list(dict.fromkeys(str(item) for item in candidate_ids if str(item)))
            outside_approval = [item for item in candidate_ids if item not in approved_ids]
            if outside_approval:
                raise InputValidationError("Экспортировать можно только подтверждённые черновики.")
        if not candidate_ids:
            raise InputValidationError("Сначала подтвердите черновики для создания готовых роликов.")
        inspection = self.pipeline.inspect_approved_drafts(project, candidate_ids)
        for candidate_id, message in inspection.errors.items():
            self._set_candidate_lifecycle(
                project, candidate_id, draft="failed", approval="pending", export="pending",
            )
            project.candidate_draft_artifacts.pop(candidate_id, None)
            project.candidate_errors[candidate_id] = message
        candidate_ids = inspection.candidate_ids
        # A one-item retry must not discard neighbouring approved/failed
        # exports.  Keep the project-wide approved set intact, only removing
        # items whose immutable draft inspection proved invalid above.
        if inspection.errors:
            invalid_ids = set(inspection.errors)
            project.selected_candidate_ids = [
                item for item in project.selected_candidate_ids if item not in invalid_ids
            ]
        # Artifact validation can itself rule out a subset.  Persist that
        # decision before any later source/config preparation raises, so a
        # restart keeps the per-item retry state rather than reviving a stale
        # approved choice.
        if inspection.errors:
            self.projects.save(project)
        if not candidate_ids:
            project.status = ProjectStatus.REVIEWING_CANDIDATES
            self.projects.save(project)
            raise InputValidationError("Нет готовых подтверждённых черновиков. Повторите только отмеченные элементы.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        run = self.runs.create(
            project, {"draft_id": project.draft_id, "candidate_ids": candidate_ids, "local_test_mode": self.settings.local_test_mode},
            {"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            "0.1.0", run_kind=RunKind.SELECTED_RENDER,
        )
        try:
            prepared = self._with_observability_paths(
                self.pipeline.prepare_selected_render(project, run, self.settings, candidate_ids), run,
            )
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить создание готовых роликов."
            self.runs.save(run)
            raise
        for candidate_id in candidate_ids:
            self._set_candidate_lifecycle(project, candidate_id, draft="ready", approval="approved", export="running")
            project.candidate_errors.pop(candidate_id, None)
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.RENDERING_SELECTED; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def prepare_render_revision(self, project: DesktopProject, parent_run: ProjectRun) -> tuple[ProjectRun, PreparedPipelineRun]:
        """Create an immutable export revision from existing production/audio artifacts."""

        if project.status == ProjectStatus.PROCESSING:
            raise RuntimeError("Этот проект уже обрабатывается.")
        if parent_run.status not in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}:
            raise InputValidationError("Повторный экспорт доступен только после успешного запуска.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        previous = parent_run.settings_snapshot.get("project_options", {})
        previous_audio_mode = str(previous.get("audio_mode", "original"))
        if previous_audio_mode != project.settings.audio_mode:
            raise InputValidationError("Для изменения аудиорежима запустите полное создание ролика.")
        current = {
            "subtitle_style": project.settings.subtitle_style,
            "subtitles_enabled": project.settings.subtitles_enabled,
            "platform": project.settings.platform,
            "audio_mode": project.settings.audio_mode,
            "composition_strategy": project.settings.composition_strategy,
            "same_source_broll_allowed": project.settings.same_source_broll_allowed,
            "encoder": project.settings.encoder,
        }
        changed = {name: value for name, value in current.items() if previous.get(name) != value}
        run = self.runs.create(
            project,
            settings_snapshot={
                "project_options": {**current, "processing_mode": project.settings.processing_mode, "deep_analysis": project.settings.deep_analysis, "clip_count": project.settings.clip_count, "use_cache": project.settings.use_cache, "recompute_all": False},
                "product_flow": {"user_intent": intent.to_dict(), "resolved_config": resolved.to_dict(), "estimate": estimate.to_dict()},
                "local_test_mode": self.settings.local_test_mode,
            },
            source_snapshot={"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            pipeline_version="0.1.0", run_kind=RunKind.RENDER_REVISION, parent_run_id=parent_run.run_id,
            changed_settings=changed, invalidated_stages=["production_render"],
        )
        run.cost_estimate = 0.0
        self.runs.save(run)
        try:
            prepared = self._with_observability_paths(
                self.pipeline.prepare_render_revision(project, run, self.settings, parent_run), run,
            )
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить повторный экспорт."
            self.runs.save(run)
            raise
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.PROCESSING; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    @staticmethod
    def _with_observability_paths(prepared: PreparedPipelineRun, run: ProjectRun) -> PreparedPipelineRun:
        return replace(
            prepared,
            # A pending run has only its engine metadata lookup path.  The
            # runner switches to the real heartbeat path once the child engine
            # publishes it; guessing it from a source slug is forbidden.
            heartbeat_path=None if prepared.artifact_metadata_path else prepared.state_path.with_name("heartbeat.json"),
            log_path=Path(run.log_path),
        )

    def record_launch_context(self, run: ProjectRun, prepared: PreparedPipelineRun) -> None:
        """Persist the desktop launch contract without leaking environment secrets."""

        run.settings_snapshot["execution"] = {
            "runtime_config_path": str(prepared.runtime_config_path),
            "run_id": prepared.run_id,
            "project_id": prepared.project_id or run.project_id,
            "artifact_metadata_path": str(prepared.artifact_metadata_path) if prepared.artifact_metadata_path else None,
            "allow_legacy_artifact_scan": prepared.allow_legacy_artifact_scan,
            "source_path": str(prepared.source_path) if prepared.source_path else None,
            "runtime_flags": dict(prepared.runtime_flags),
        }
        # Hand-built PreparedPipelineRun values remain useful in tests and for
        # older external callers.  Those values already are their engine
        # contract, so preserve them as a backward-compatible fallback.
        if prepared.artifact_metadata_path is None:
            run.settings_snapshot["execution"].update({
                "state_path": str(prepared.state_path),
                "report_path": str(prepared.report_path),
                "output_directory": str(prepared.output_directory),
                "manifest_path": str(prepared.manifest_path) if prepared.manifest_path else None,
            })
        self.runs.save(run)
        flags = [argument for argument in prepared.arguments if argument.startswith("--")]
        self.append_log(run, "Desktop pipeline launch prepared.")
        self.append_log(run, f"command: {prepared.command_line()}")
        self.append_log(run, f"cwd: {prepared.working_directory}")
        self.append_log(run, f"source: {prepared.source_path or '<unknown>'}")
        self.append_log(run, f"runtime config: {prepared.runtime_config_path}")
        self.append_log(run, "output directory: awaiting engine artifact metadata")
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
            reported = self.recover_reported_failure(project, run, prepared)
            if reported is not None:
                return reported
            return self.finish_failure(project, run, completion.error_summary, completion.technical_details)
        return self._finish_completion(project, run, completion)

    def recover_failed_process(
        self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun,
    ) -> ProjectRun | None:
        """Keep verified outputs when only final service-state persistence failed."""

        completion = self.pipeline.recovery_completion(prepared, run.started_at)
        if completion is None:
            return None
        return self._finish_completion(project, run, completion, state_persistence_degraded=True)

    @staticmethod
    def _candidate_report_message(item: dict, fallback: str) -> str:
        """Keep actionable item/stage context without exposing process diagnostics."""

        stage = str(item.get("stage") or "").strip()
        message = str(
            item.get("message") or item.get("error") or item.get("reason") or fallback
        ).strip()
        message = redact_secrets(message or fallback)
        return f"Этап {stage}: {message}" if stage else message

    @classmethod
    def _apply_draft_report(
        cls, project: DesktopProject, report: dict, expected_candidate_ids: list[str],
        *, fallback: str, previous_artifacts: dict[str, str] | None = None,
    ) -> None:
        run_info = report.get("run", {}) if isinstance(report, dict) else {}
        if not isinstance(run_info, dict):
            run_info = {}
        artifact_path = str(run_info.get("draft_artifact_path") or "").strip()
        if artifact_path:
            project.draft_artifact_path = artifact_path
        draft_id = str(run_info.get("draft_id") or "").strip()
        if draft_id:
            project.draft_id = draft_id
        flow = report.get("candidate_flow", {}) if isinstance(report, dict) else {}
        candidates = flow.get("draft_candidates", []) if isinstance(flow, dict) else []
        allowed = set(expected_candidate_ids) or set(project.candidate_states)
        reported_ids: set[str] = set()
        for item in candidates:
            if not isinstance(item, dict) or not item.get("candidate_id"):
                continue
            candidate_id = str(item["candidate_id"])
            if candidate_id not in allowed:
                continue
            reported_ids.add(candidate_id)
            state = str(item.get("state") or "")
            if state == "draft_ready" and artifact_path and Path(artifact_path).is_file():
                cls._set_candidate_lifecycle(
                    project, candidate_id, draft="ready", approval="pending", export="pending",
                )
                project.candidate_draft_artifacts[candidate_id] = artifact_path
                project.candidate_errors.pop(candidate_id, None)
            else:
                previous = str((previous_artifacts or {}).get(candidate_id) or "")
                if previous and Path(previous).is_file():
                    cls._set_candidate_lifecycle(
                        project, candidate_id, draft="ready", approval="pending", export="pending",
                    )
                    project.candidate_draft_artifacts[candidate_id] = previous
                    project.candidate_errors[candidate_id] = (
                        "Не удалось обновить предпросмотр. Предыдущая готовая версия сохранена; "
                        "можно повторить обновление."
                    )
                    continue
                cls._set_candidate_lifecycle(
                    project, candidate_id, draft="failed", approval="pending", export="pending",
                )
                project.candidate_draft_artifacts.pop(candidate_id, None)
                project.candidate_errors[candidate_id] = cls._candidate_report_message(item, fallback)
        for candidate_id in expected_candidate_ids:
            if candidate_id in reported_ids:
                continue
            if project.candidate_draft_statuses.get(candidate_id) == "running" or project.candidate_states.get(candidate_id) == "draft_planning":
                previous = str((previous_artifacts or {}).get(candidate_id) or "")
                if previous and Path(previous).is_file():
                    cls._set_candidate_lifecycle(
                        project, candidate_id, draft="ready", approval="pending", export="pending",
                    )
                    project.candidate_draft_artifacts[candidate_id] = previous
                    project.candidate_errors[candidate_id] = (
                        "Не удалось обновить предпросмотр. Предыдущая готовая версия сохранена."
                    )
                    continue
                cls._set_candidate_lifecycle(
                    project, candidate_id, draft="failed", approval="pending", export="pending",
                )
                project.candidate_draft_artifacts.pop(candidate_id, None)
                project.candidate_errors[candidate_id] = fallback

    @classmethod
    def _apply_selected_render_report(
        cls, project: DesktopProject, report: dict, expected_candidate_ids: list[str],
        *, fallback: str,
    ) -> tuple[set[str], set[str]]:
        """Apply individual final-export outcomes and return completed/failed IDs."""

        production = report.get("production_render", {}) if isinstance(report, dict) else {}
        production_items = production.get("items", []) if isinstance(production, dict) else []
        allowed = set(expected_candidate_ids) or set(project.selected_candidate_ids)
        completed_ids = {
            str(item.get("candidate_id")) for item in production_items
            if isinstance(item, dict) and str(item.get("candidate_id") or "") in allowed
            and item.get("status") in {"completed", "warning"}
        }
        flow = report.get("candidate_flow", {}) if isinstance(report, dict) else {}
        flow_items = flow.get("items", []) if isinstance(flow, dict) else []
        failures: dict[str, dict] = {}
        for item in [*production_items, *flow_items]:
            if not isinstance(item, dict) or not item.get("candidate_id"):
                continue
            candidate_id = str(item["candidate_id"])
            if candidate_id not in allowed:
                continue
            if item.get("outcome") == "failed" or item.get("status") == "failed":
                failures[candidate_id] = item

        failed_ids: set[str] = set()
        for candidate_id in expected_candidate_ids:
            if candidate_id in completed_ids:
                cls._set_candidate_lifecycle(
                    project, candidate_id, draft="ready", approval="approved", export="ready",
                )
                project.candidate_errors.pop(candidate_id, None)
                project.selected_candidate_ids = [
                    item_id for item_id in project.selected_candidate_ids if item_id != candidate_id
                ]
                continue
            failed_ids.add(candidate_id)
            item = failures.get(candidate_id, {})
            stage = str(item.get("stage") or "")
            # A malformed/stale plan cannot be fixed by re-running the costly
            # export.  Return only that candidate to Draft Preview generation.
            if stage.startswith("approved_draft_plan") or item.get("reason") == "production_plan_failed":
                cls._set_candidate_lifecycle(
                    project, candidate_id, draft="failed", approval="pending", export="pending",
                )
                project.candidate_draft_artifacts.pop(candidate_id, None)
                project.selected_candidate_ids = [
                    item_id for item_id in project.selected_candidate_ids if item_id != candidate_id
                ]
            else:
                cls._set_candidate_lifecycle(
                    project, candidate_id, draft="ready", approval="approved", export="failed",
                )
                if candidate_id not in project.selected_candidate_ids:
                    project.selected_candidate_ids.append(candidate_id)
            project.candidate_errors[candidate_id] = cls._candidate_report_message(item, fallback)
        return completed_ids, failed_ids

    def recover_reported_failure(
        self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun,
    ) -> ProjectRun | None:
        """Persist a terminal engine report even when QProcess exits with code 2."""

        reported = self.pipeline.reported_failure(prepared, run.started_at)
        if reported is None:
            return None
        # A process can finish after its engine report is written but between
        # the two durable writes below.  A terminal run snapshot must never
        # prevent the project projection from being reconciled on restart.
        # We only skip the *duplicate snapshot*, not the item-level recovery.
        already_snapshotted = self._reported_failure_is_already_snapshotted(
            run, reported.report, reported.terminal,
        )
        terminal = reported.terminal
        code = str(terminal.get("error_code") or "PIPELINE_FAILED")
        message = redact_secrets(str(terminal.get("message") or "Обработка завершилась с ошибкой."))
        stage = str(terminal.get("stage") or "").strip()
        if not already_snapshotted:
            # ``snapshot_report_and_outputs`` writes the run itself.  Keep the
            # saved status active until the project has its matching durable
            # candidate states, so a crash remains recoverable on the next
            # launch instead of stranding a project in "running".
            run.status = RunStatus.RUNNING
            self._snapshot_engine_paths(run)
            self.runs.snapshot_report_and_outputs(run, reported.prepared.report_path, [])
            run.status = RunStatus.FAILED
            run.finished_at = utc_now()
            run.error_summary = message
            run.technical_details = redact_secrets(
                f"terminal error_code={code}" + (f"; stage={stage}" if stage else "") + f"; message={message}"
            )
            self.append_log(run, run.technical_details)
        expected = [
            str(candidate_id) for candidate_id in run.settings_snapshot.get("candidate_ids", [])
            if str(candidate_id)
        ] or list(project.selected_candidate_ids)
        if run.run_kind == RunKind.DRAFT:
            self._apply_draft_report(
                project, reported.report, expected, fallback=message,
                previous_artifacts={
                    str(key): str(value)
                    for key, value in dict(run.settings_snapshot.get("previous_draft_artifacts") or {}).items()
                },
            )
            project.status = ProjectStatus.REVIEWING_CANDIDATES if project.analysis_artifact_path else ProjectStatus.SOURCE_READY
        elif run.run_kind == RunKind.SELECTED_RENDER:
            completed_ids, _failed_ids = self._apply_selected_render_report(
                project, reported.report, expected, fallback=message,
            )
            project.status = ProjectStatus.PARTIALLY_RENDERED if completed_ids else ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.ANALYSIS:
            project.status = ProjectStatus.SOURCE_READY
        else:
            project.status = ProjectStatus.FAILED
        project.latest_run_id = run.run_id
        # Project state is the user-facing source of truth.  Save it first;
        # an interrupted second write can then be safely replayed from the
        # active/snapshotted run above.
        self.projects.save(project)
        if not already_snapshotted:
            self.runs.save(run)
        # A valid terminal engine report is handled even when its run snapshot
        # already existed.  Callers must not fall through to a generic process
        # error and replace its per-item recovery state.
        return run

    @staticmethod
    def _reported_failure_is_already_snapshotted(
        run: ProjectRun, report: dict, terminal: dict,
    ) -> bool:
        """Avoid rewriting a completed recovery on every application start."""

        if run.status != RunStatus.FAILED or not run.finished_at or not run.report_path:
            return False
        try:
            stored = read_json(Path(run.report_path), {})
        except (OSError, ValueError):
            return False
        stored_terminal = stored.get("terminal") if isinstance(stored, dict) else None
        if not isinstance(stored_terminal, dict):
            return False
        return (
            stored_terminal.get("status") == "failed"
            and str(stored_terminal.get("error_code") or "") == str(terminal.get("error_code") or "")
            and stored.get("run") == report.get("run")
        )

    def _finish_completion(
        self,
        project: DesktopProject,
        run: ProjectRun,
        completion: PipelineCompletion,
        *,
        state_persistence_degraded: bool = False,
    ) -> ProjectRun:
        warnings = list(completion.warnings)
        quality_driven = completion.quality_status is not None and not completion.legacy_technical_completion
        if state_persistence_degraded and STATE_PERSISTENCE_WARNING not in warnings:
            warnings.append(STATE_PERSISTENCE_WARNING)
        report = read_json(completion.report_path, {})
        completion_status = (
            RunStatus.COMPLETED_WITH_WARNINGS
            if quality_driven and completion.quality_status == "PASS_WITH_WARNINGS"
            else RunStatus.COMPLETED_WITH_WARNINGS if not quality_driven and warnings else RunStatus.COMPLETED
        )
        # Snapshot while the record remains active.  See
        # ``recover_reported_failure`` for why the final terminal status is
        # written only after its matching project projection is durable.
        run.status = RunStatus.RUNNING
        self._snapshot_engine_paths(run)
        self.runs.snapshot_report_and_outputs(
            run, completion.report_path, completion.output_files, completion.quality_report_paths,
        )
        run.finished_at = utc_now()
        run.warnings = warnings
        run.cost_estimate = completion.cost_estimate
        run.actual_cost = None  # Local estimates are intentionally never treated as billed cost.
        run_info = report.get("run", {}) if isinstance(report, dict) else {}
        if run.run_kind == RunKind.ANALYSIS:
            run.status = RunStatus.ANALYSIS_READY
            project.analysis_artifact_path = str(run_info.get("analysis_artifact_path") or "") or None
            project.analysis_id = str(run_info.get("analysis_id") or "") or None
            project.analysis_fingerprint = str(run_info.get("analysis_fingerprint") or "") or None
            candidates = report.get("clip_intelligence", {}).get("candidates", []) if isinstance(report.get("clip_intelligence"), dict) else []
            project.candidate_states = {
                str(item.get("id")): "analyzed" for item in candidates
                if isinstance(item, dict) and item.get("id")
            }
            project.candidate_draft_statuses = {
                candidate_id: "pending" for candidate_id in project.candidate_states
            }
            project.candidate_approval_states = {
                candidate_id: "pending" for candidate_id in project.candidate_states
            }
            project.candidate_export_statuses = {
                candidate_id: "pending" for candidate_id in project.candidate_states
            }
            project.selected_candidate_ids = []
            project.review_selected_candidate_ids = []
            project.candidate_draft_artifacts = {}
            project.candidate_errors = {}
            project.draft_artifact_path = None
            project.draft_id = None
            project.active_preview_candidate_id = None
            project.status = ProjectStatus.ANALYSIS_READY
        elif run.run_kind == RunKind.DRAFT:
            run.status = RunStatus.DRAFT_READY
            expected = [
                str(candidate_id) for candidate_id in run.settings_snapshot.get("candidate_ids", [])
                if str(candidate_id)
            ] or list(project.review_selected_candidate_ids)
            self._apply_draft_report(
                project, report, expected, fallback="Не удалось подготовить черновик.",
                previous_artifacts={
                    str(key): str(value)
                    for key, value in dict(run.settings_snapshot.get("previous_draft_artifacts") or {}).items()
                },
            )
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.SELECTED_RENDER:
            expected = [
                str(candidate_id) for candidate_id in run.settings_snapshot.get("candidate_ids", [])
                if str(candidate_id)
            ] or list(project.selected_candidate_ids)
            completed_ids, failed_ids = self._apply_selected_render_report(
                project, report, expected, fallback="Не удалось создать ролик.",
            )
            # The current run may finish a one-item retry while other
            # independently approved exports are still failed/pending.  Those
            # neighbours stay in the durable hand-off queue and must not be
            # cleared just because this one report completed successfully.
            project.selected_candidate_ids = [
                candidate_id for candidate_id in project.selected_candidate_ids
                if project.candidate_export_statuses.get(candidate_id) != "ready"
            ]
            has_remaining_exports = bool(project.selected_candidate_ids)
            if failed_ids:
                run.status = RunStatus.PARTIALLY_RENDERED
            elif quality_driven and completion.quality_status == "PASS_WITH_WARNINGS":
                run.status = RunStatus.COMPLETED_WITH_WARNINGS
            elif quality_driven:
                run.status = RunStatus.COMPLETED
            elif warnings:
                run.status = RunStatus.COMPLETED_WITH_WARNINGS
            else:
                run.status = RunStatus.COMPLETED
            if failed_ids or has_remaining_exports:
                project.status = ProjectStatus.PARTIALLY_RENDERED
            elif run.status == RunStatus.COMPLETED_WITH_WARNINGS:
                project.status = ProjectStatus.COMPLETED_WITH_WARNINGS
            else:
                project.status = ProjectStatus.COMPLETED
        else:
            run.status = completion_status
            project.status = (
                ProjectStatus.COMPLETED_WITH_WARNINGS
                if quality_driven and completion.quality_status == "PASS_WITH_WARNINGS"
                else ProjectStatus.COMPLETED_WITH_WARNINGS if not quality_driven and warnings else ProjectStatus.COMPLETED
            )
        project.latest_run_id = run.run_id
        # Run-kind-specific terminal states (analysis_ready/draft_ready) are
        # assigned after snapshots are copied.  Project first keeps restart
        # recovery deterministic if the process stops between these writes.
        self.projects.save(project)
        self.runs.save(run)
        return run

    def _snapshot_engine_paths(self, run: ProjectRun) -> None:
        """Persist only paths returned by engine metadata for restart recovery."""

        prepared = self.pipeline.prepared_from_execution(run)
        resolved = self.pipeline.resolve_engine_paths(prepared)
        if resolved is None or resolved is prepared and prepared and prepared.artifact_metadata_path:
            return
        execution = run.settings_snapshot.setdefault("execution", {})
        if not isinstance(execution, dict):
            return
        execution["engine_paths"] = {
            "state_path": str(resolved.state_path.resolve()),
            "report_path": str(resolved.report_path.resolve()),
            "output_directory": str(resolved.output_directory.resolve()),
            "manifest_path": str(resolved.manifest_path.resolve()) if resolved.manifest_path else None,
            "heartbeat_path": str(resolved.heartbeat_path.resolve()) if resolved.heartbeat_path else None,
        }
        execution["artifact_metadata_path"] = (
            str(resolved.artifact_metadata_path.resolve()) if resolved.artifact_metadata_path else execution.get("artifact_metadata_path")
        )

    def finish_failure(
        self, project: DesktopProject, run: ProjectRun, message: str, technical_details: str | None = None,
    ) -> ProjectRun:
        run.status = RunStatus.FAILED
        run.finished_at = utc_now()
        run.error_summary = redact_secrets(message)
        run.technical_details = redact_secrets(technical_details or message)
        self.append_log(run, run.technical_details)
        if run.run_kind == RunKind.SELECTED_RENDER:
            for candidate_id in project.selected_candidate_ids:
                if project.candidate_states.get(candidate_id) == "production_rendering":
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="ready", approval="approved", export="failed",
                    )
                    project.candidate_errors[candidate_id] = run.error_summary
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.DRAFT:
            previous_artifacts = {
                str(key): str(value)
                for key, value in dict(run.settings_snapshot.get("previous_draft_artifacts") or {}).items()
            }
            for candidate_id, state in list(project.candidate_states.items()):
                if state == "draft_planning":
                    previous = previous_artifacts.get(candidate_id)
                    if previous and Path(previous).is_file():
                        self._set_candidate_lifecycle(
                            project, candidate_id, draft="ready", approval="pending", export="pending",
                        )
                        project.candidate_draft_artifacts[candidate_id] = previous
                        project.candidate_errors[candidate_id] = (
                            "Не удалось обновить предпросмотр. Предыдущая готовая версия сохранена."
                        )
                        continue
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="failed", approval="pending", export="pending",
                    )
                    project.candidate_errors[candidate_id] = run.error_summary
            project.status = ProjectStatus.REVIEWING_CANDIDATES if project.analysis_artifact_path else ProjectStatus.SOURCE_READY
        elif run.run_kind == RunKind.ANALYSIS:
            project.status = ProjectStatus.SOURCE_READY
        else:
            project.status = ProjectStatus.FAILED
        self.projects.save(project)
        self.runs.save(run)
        return run

    def finish_cancelled(self, project: DesktopProject, run: ProjectRun) -> ProjectRun:
        run.status = RunStatus.CANCELLED
        run.finished_at = utc_now()
        run.error_summary = "Создание ролика отменено пользователем."
        if run.run_kind == RunKind.SELECTED_RENDER:
            for candidate_id in project.selected_candidate_ids:
                if project.candidate_states.get(candidate_id) == "production_rendering":
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="ready", approval="approved", export="pending",
                    )
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.DRAFT:
            previous_artifacts = {
                str(key): str(value)
                for key, value in dict(run.settings_snapshot.get("previous_draft_artifacts") or {}).items()
            }
            for candidate_id, state in list(project.candidate_states.items()):
                if state == "draft_planning":
                    previous = previous_artifacts.get(candidate_id)
                    if previous and Path(previous).is_file():
                        self._set_candidate_lifecycle(
                            project, candidate_id, draft="ready", approval="pending", export="pending",
                        )
                        project.candidate_draft_artifacts[candidate_id] = previous
                        continue
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="pending", approval="pending", export="pending",
                    )
            project.status = ProjectStatus.REVIEWING_CANDIDATES if project.analysis_artifact_path else ProjectStatus.SOURCE_READY
        elif run.run_kind == RunKind.ANALYSIS:
            project.status = ProjectStatus.SOURCE_READY
        else:
            project.status = ProjectStatus.CANCELLED
        self.projects.save(project)
        self.runs.save(run)
        return run

    def recover_interrupted_runs(self) -> int:
        recovered_or_interrupted = 0
        for project in self.projects.list():
            # Older v3 project files recorded only a combined candidate state.
            # A partial draft artifact could therefore retain a failed third
            # item while the project reopened as if it had never been chosen.
            # Reconcile only the project-owned latest artifact, never an
            # output-directory scan, before considering active processes.
            changed = self._reconcile_legacy_draft_artifact(project)
            for run in self.runs.list(project.project_id):
                is_active = run.status in RunStatus.ACTIVE
                is_draft_reconciliation = (
                    run.status == RunStatus.INTERRUPTED
                    and run.run_kind == RunKind.DRAFT
                    and project.latest_run_id == run.run_id
                )
                is_terminal_report_reconciliation = (
                    run.status in {RunStatus.FAILED, RunStatus.INTERRUPTED}
                    and run.run_kind in {RunKind.DRAFT, RunKind.SELECTED_RENDER}
                    and project.latest_run_id == run.run_id
                )
                if not is_active and not is_draft_reconciliation and not is_terminal_report_reconciliation:
                    continue
                prepared = self.pipeline.prepared_from_execution(run)
                if is_active and prepared and self.recover_failed_process(project, run, prepared):
                    recovered_or_interrupted += 1
                    changed = True
                    continue
                was_terminal_failure = run.status == RunStatus.FAILED
                if prepared and self.recover_reported_failure(project, run, prepared):
                    # An already-terminal run may be replaying only the second
                    # (project) write after a crash.  It is useful recovery,
                    # but not a newly interrupted process to count again.
                    if not was_terminal_failure:
                        recovered_or_interrupted += 1
                        changed = True
                    continue
                # A previously persisted terminal item failure is already in
                # review state.  Never reinterpret it as an abandoned draft
                # and overwrite its per-item error on the next startup.
                if is_terminal_report_reconciliation and run.status == RunStatus.FAILED:
                    continue
                if run.run_kind == RunKind.DRAFT:
                    reconciled = self._recover_interrupted_draft(project, run, prepared)
                    if reconciled or is_active:
                        recovered_or_interrupted += int(is_active)
                        changed = True
                    continue
                if run.run_kind == RunKind.SELECTED_RENDER:
                    reconciled = self._recover_interrupted_selected_render(project, run)
                    if reconciled or is_active:
                        recovered_or_interrupted += int(is_active)
                        changed = True
                    continue
                if is_active:
                    run.status = RunStatus.INTERRUPTED
                    run.finished_at = utc_now()
                    run.error_summary = "Предыдущий запуск был прерван при закрытии приложения."
                    project.status = ProjectStatus.INTERRUPTED
                    self.projects.save(project)
                    self.runs.save(run)
                    recovered_or_interrupted += 1
                    changed = True
            if changed:
                latest = next(iter(self.runs.list(project.project_id)), None)
                if latest and latest.status == RunStatus.COMPLETED_WITH_WARNINGS:
                    project.status = ProjectStatus.COMPLETED_WITH_WARNINGS
                elif latest and latest.status == RunStatus.COMPLETED:
                    project.status = ProjectStatus.COMPLETED
                elif (
                    latest
                    and latest.status == RunStatus.INTERRUPTED
                    and project.status not in {
                        ProjectStatus.REVIEWING_CANDIDATES,
                        ProjectStatus.PARTIALLY_RENDERED,
                    }
                ):
                    project.status = ProjectStatus.INTERRUPTED
                self.projects.save(project)
        return recovered_or_interrupted

    def _reconcile_legacy_draft_artifact(self, project: DesktopProject) -> bool:
        """Recover item failure metadata from one legacy project draft artifact.

        This is intentionally a narrow, one-way migration.  It only promotes
        a still-``analyzed`` candidate from the project's own matching draft
        artifact to ``draft_failed``.  Once persisted, a person can skip that
        candidate and later restarts will respect the choice rather than add
        it back.  Newer lifecycle state and unrelated artifact folders are
        never touched.
        """

        artifact_path = Path(str(project.draft_artifact_path or ""))
        if not artifact_path.is_file():
            return False
        try:
            artifact = read_json(artifact_path, {})
        except (OSError, ValueError):
            return False
        if not isinstance(artifact, dict):
            return False
        if str(artifact.get("project_id") or "") != project.project_id:
            return False
        artifact_analysis_id = str(artifact.get("analysis_id") or "")
        artifact_fingerprint = str(artifact.get("analysis_fingerprint") or "")
        if project.analysis_id and artifact_analysis_id and artifact_analysis_id != project.analysis_id:
            return False
        if project.analysis_fingerprint and artifact_fingerprint and artifact_fingerprint != project.analysis_fingerprint:
            return False
        candidates = artifact.get("candidates")
        if not isinstance(candidates, list):
            return False

        changed = False
        for record in candidates:
            if not isinstance(record, dict) or str(record.get("state") or "") != "draft_failed":
                continue
            candidate_id = str(record.get("candidate_id") or "")
            if not candidate_id or project.candidate_states.get(candidate_id) != "analyzed":
                continue
            # A candidate can have an independent draft artifact from a later
            # retry; that is already authoritative even if the aggregate
            # legacy artifact still contains its original failure.
            if project.candidate_draft_artifacts.get(candidate_id):
                continue
            self._set_candidate_lifecycle(
                project, candidate_id, draft="failed", approval="pending", export="pending",
            )
            if candidate_id not in project.review_selected_candidate_ids:
                project.review_selected_candidate_ids.append(candidate_id)
            project.candidate_errors[candidate_id] = (
                "Этап подготовки черновика: план ролика не прошёл проверку границ. "
                "Повторите только этот черновик или продолжите с готовыми."
            )
            changed = True
        if changed and project.analysis_artifact_path:
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        return changed

    def _recover_interrupted_selected_render(
        self, project: DesktopProject, run: ProjectRun,
    ) -> bool:
        """Return an abandoned final export to its retained draft boundary.

        This path is deliberately narrower than ``recover_failed_process``:
        that method runs first and may recover a verified engine report/output.
        When no such evidence exists we never scan arbitrary media files; each
        candidate remains approved with its immutable Draft Preview and can be
        retried independently from the review workspace.
        """

        expected = [
            str(candidate_id) for candidate_id in run.settings_snapshot.get("candidate_ids", [])
            if str(candidate_id)
        ] or list(project.selected_candidate_ids)
        changed = False
        interruption_message = (
            "Экспорт этого ролика был прерван. Черновик сохранён: можно повторить только экспорт."
        )
        for candidate_id in expected:
            self._ensure_candidate_lifecycle(project, candidate_id)
            state = project.candidate_states.get(candidate_id)
            export_state = project.candidate_export_statuses.get(candidate_id)
            if state != "production_rendering" and export_state != "running":
                continue
            self._set_candidate_lifecycle(
                project, candidate_id, draft="ready", approval="approved", export="failed",
            )
            if candidate_id not in project.selected_candidate_ids:
                project.selected_candidate_ids.append(candidate_id)
            if candidate_id not in project.review_selected_candidate_ids:
                project.review_selected_candidate_ids.append(candidate_id)
            if project.candidate_errors.get(candidate_id) != interruption_message:
                project.candidate_errors[candidate_id] = interruption_message
            changed = True

        if project.status == ProjectStatus.RENDERING_SELECTED:
            project.status = ProjectStatus.REVIEWING_CANDIDATES
            changed = True
        if run.status != RunStatus.INTERRUPTED:
            run.status = RunStatus.INTERRUPTED
            changed = True
        if run.finished_at is None:
            run.finished_at = utc_now()
            changed = True
        if run.error_summary != interruption_message:
            run.error_summary = interruption_message
            changed = True
        if changed:
            # Save the user-visible retry state first.  If the second write is
            # interrupted, this same routine is idempotent on the next start.
            self.projects.save(project)
            self.runs.save(run)
        return changed

    def _recover_interrupted_draft(
        self,
        project: DesktopProject,
        run: ProjectRun,
        prepared: PreparedPipelineRun | None,
    ) -> bool:
        """Reconcile an abandoned draft run without discovering arbitrary MP4s."""

        expected = [
            str(candidate_id) for candidate_id in run.settings_snapshot.get("candidate_ids", [])
            if str(candidate_id)
        ]
        progress = self.pipeline.recover_draft_progress(prepared, expected) if prepared else None
        ready_ids = set(progress.ready_candidate_ids) if progress else set()
        invalid_ids = set(progress.invalid_candidate_ids) if progress else set()
        changed = False

        if progress:
            artifact_path = str(progress.artifact_path.resolve())
            if project.draft_artifact_path != artifact_path:
                project.draft_artifact_path = artifact_path
                changed = True
            if project.draft_id != progress.artifact.draft_id:
                project.draft_id = progress.artifact.draft_id
                changed = True
            records = {
                str(item.get("candidate_id") or ""): item
                for item in progress.artifact.candidates if isinstance(item, dict)
            }
            for candidate_id in ready_ids:
                if project.candidate_states.get(candidate_id) != "draft_ready":
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="ready", approval="pending", export="pending",
                    )
                    changed = True
                if project.candidate_draft_artifacts.get(candidate_id) != artifact_path:
                    project.candidate_draft_artifacts[candidate_id] = artifact_path
                    changed = True
                if candidate_id in project.candidate_errors:
                    project.candidate_errors.pop(candidate_id, None)
                    changed = True
            for candidate_id in invalid_ids:
                if project.candidate_draft_artifacts.pop(candidate_id, None) is not None:
                    changed = True
                if project.candidate_states.get(candidate_id) != "draft_failed":
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="failed", approval="pending", export="pending",
                    )
                    changed = True
                message = "Неполный файл черновика не был принят после перезапуска."
                if project.candidate_errors.get(candidate_id) != message:
                    project.candidate_errors[candidate_id] = message
                    changed = True
            # ``draft_failed`` never counts as ready after an abrupt close: a
            # later click starts only these missing candidates in a new draft run.
            for candidate_id in expected:
                if candidate_id in ready_ids or candidate_id in invalid_ids:
                    continue
                record = records.get(candidate_id, {})
                if record.get("state") == "draft_failed" and record.get("error"):
                    message = str(record["error"])
                    if project.candidate_errors.get(candidate_id) != message:
                        project.candidate_errors[candidate_id] = message
                        changed = True
                if project.candidate_states.get(candidate_id) != "draft_failed":
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="failed", approval="pending", export="pending",
                    )
                    changed = True
                if project.candidate_draft_artifacts.pop(candidate_id, None) is not None:
                    changed = True
        else:
            # Legacy interrupted draft runs have no candidate-owned progress
            # contract.  Their files are deliberately ignored rather than being
            # guessed from a directory listing.
            for candidate_id in expected:
                if project.candidate_states.get(candidate_id) == "draft_planning":
                    self._set_candidate_lifecycle(
                        project, candidate_id, draft="pending", approval="pending", export="pending",
                    )
                    changed = True

        interruption_message = "Предыдущий запуск черновиков был прерван при закрытии приложения."
        if ready_ids:
            interruption_message = (
                f"Предыдущий запуск черновиков был прерван; готово: {len(ready_ids)} из {len(expected)}. "
                "Можно продолжить только недостающие черновики."
            )
        if run.status != RunStatus.INTERRUPTED:
            run.status = RunStatus.INTERRUPTED
            changed = True
        if run.finished_at is None:
            run.finished_at = utc_now()
            changed = True
        if run.error_summary != interruption_message:
            run.error_summary = interruption_message
            changed = True
        if project.status != ProjectStatus.INTERRUPTED:
            project.status = ProjectStatus.INTERRUPTED
            changed = True
        if changed:
            self._snapshot_engine_paths(run)
            self.projects.save(project)
            self.runs.save(run)
        return changed

    def recover_ready_analysis_runs(self) -> int:
        """Repair legacy desktop history from a completed engine analysis.

        The recovery only reads finished engine artifacts by run/project ID and
        never starts another analysis process.
        """

        restored = 0
        for project in self.projects.list():
            if project.analysis_artifact_path and Path(project.analysis_artifact_path).is_file():
                continue
            for run in self.runs.list(project.project_id):
                if run.run_kind != RunKind.ANALYSIS:
                    continue
                prepared = self.pipeline.prepared_from_execution(run)
                if prepared is None:
                    continue
                completion = self.pipeline.completion(prepared)
                if completion.error_summary:
                    continue
                report = read_json(completion.report_path, {})
                terminal = report.get("terminal", {}) if isinstance(report, dict) else {}
                if not isinstance(terminal, dict) or terminal.get("status") != "analysis_ready":
                    continue
                self._finish_completion(project, run, completion)
                restored += 1
                break
        return restored
