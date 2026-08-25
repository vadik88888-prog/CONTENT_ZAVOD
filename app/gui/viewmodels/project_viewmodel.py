from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from app.gui.models import DesktopProject, ProcessingPhase, ProcessingSnapshot, ProjectRun, RunKind, RunStatus
from app.gui.services.background_task import BackgroundTask
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.error_mapping import UserFacingError, map_error
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner
from app.gui.services.url_source_service import URLSourceService


@dataclass(frozen=True, slots=True)
class _FinalizationResult:
    phase: ProcessingPhase
    message: str
    run: ProjectRun
    error_message: str | None = None


class ProjectViewModel(QObject):
    project_changed = Signal(object)
    runs_changed = Signal(list)
    processing_changed = Signal(object)
    error_occurred = Signal(object)
    log_received = Signal(str)
    run_finished = Signal(object)
    project_persisted = Signal(str)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self.project: DesktopProject | None = None
        self.run: ProjectRun | None = None
        self.prepared: PreparedPipelineRun | None = None
        self.snapshot = ProcessingSnapshot()
        self._job_project: DesktopProject | None = None
        self._job_snapshot = ProcessingSnapshot()
        self._launching = False
        self._started_at: float | None = None
        self._after_download = "process"
        self._source_probe_task: BackgroundTask | None = None
        self._finalization_task: BackgroundTask | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._emit_elapsed)
        self.runner = QtPipelineRunner(self)
        self.runner.run_started.connect(self._run_started)
        self.runner.stage_changed.connect(self._stage_changed)
        self.runner.stage_running_longer_than_usual.connect(self._stage_running_longer_than_usual)
        self.runner.activity_changed.connect(self._activity_changed)
        self.runner.log_received.connect(self._log_received)
        self.runner.run_completed.connect(self._completed)
        self.runner.run_failed.connect(self._failed)
        self.runner.run_cancelled.connect(self._cancelled)
        self.source_downloader = URLSourceService(self)
        self.source_downloader.download_progress.connect(self._download_progress)
        self.source_downloader.download_completed.connect(self._download_completed)
        self.source_downloader.failed.connect(self._download_failed)
        self.source_downloader.cancelled.connect(self._download_cancelled)

    @property
    def active(self) -> bool:
        return self._launching or self.runner.active or self.source_downloader.busy

    @property
    def active_project_id(self) -> str | None:
        return self._job_project.project_id if self.active and self._job_project else None

    @property
    def active_project_name(self) -> str | None:
        return self._job_project.name if self.active and self._job_project else None

    @property
    def owns_active_job(self) -> bool:
        return bool(self.project and self.active_project_id == self.project.project_id)

    @property
    def blocked_by_other_project(self) -> bool:
        return bool(self.active and self.project and self.active_project_id != self.project.project_id)

    def open(self, project: DesktopProject) -> None:
        # Cards are navigation handles, not authoritative project snapshots.
        # Always reopen by stable identity so Projects and Project cannot drift.
        project = self.services.projects.load(project.project_id)
        try:
            self.project = self.services.refresh_setup_estimate(project)
        except Exception:
            # An unavailable optional provider tariff must not prevent a person
            # from opening their existing candidates and completed videos.
            self.project = project
        self.snapshot = self._job_snapshot if self.owns_active_job else ProcessingSnapshot()
        self.project_changed.emit(self.project)
        self.runs_changed.emit(self.services.runs_for(self.project))
        self.processing_changed.emit(self.snapshot)

    def save_options(self, **values: object) -> None:
        if not self.project or self.owns_active_job:
            return
        try:
            self.project = self.services.update_project_options(self.project, **values)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return
        self.project_changed.emit(self.project)

    def setup_preflight(self):
        """Return the current setup recommendation without starting a run."""

        if not self.project:
            raise RuntimeError("Проект не открыт.")
        return self.services.setup_preflight(self.project)

    def start(self) -> None:
        if not self.project or not self._can_start_heavy_job():
            return
        self._launching = True
        self._after_download = "process"
        if not self.project.source_spec.is_ready:
            self._start_source_download()
            return
        try:
            self._start_prepared_job("full")
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            if self.project:
                self.project_changed.emit(self.project)

    def start_analysis(self) -> None:
        """Start analysis only; delivery remains unavailable until draft review."""

        if not self.project or not self._can_start_heavy_job():
            return
        self._launching = True
        self._after_download = "analysis"
        if not self.project.source_spec.is_ready:
            self._start_source_download()
            return
        try:
            self._start_prepared_job("analysis")
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            if self.project:
                self.project_changed.emit(self.project)

    def start_download(self) -> None:
        """Download a public source as its own explicit user step."""

        if not self.project or not self._can_start_heavy_job():
            return
        if self.project.source_spec.kind != "url" or self.project.source_spec.is_ready:
            return
        self._launching = True
        self._after_download = "none"
        self._start_source_download()

    def build_drafts(self, candidate_ids: list[str]) -> None:
        if not self.project or not self._can_start_heavy_job():
            return
        self._launching = True
        try:
            self.run, self.prepared = self.services.prepare_draft(self.project, candidate_ids)
            self._bind_job(self.project, ProcessingSnapshot(
                ProcessingPhase.PREPARING, message="Собираем быстрые черновики",
            ))
            self._emit_owner_project()
            self._emit_owner_runs()
            self._emit_owner_processing()
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            self.project_changed.emit(self.project)

    def select_drafts(self, candidate_ids: list[str]) -> None:
        if not self.project or self.owns_active_job:
            return
        try:
            self.project = self.services.select_draft_candidates(self.project, candidate_ids)
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def set_draft_approval(self, candidate_id: str, approved: bool) -> None:
        """Record an explicit keep/reject decision for one ready draft."""

        if not self.project or self.owns_active_job:
            return
        try:
            self.project = self.services.set_draft_approval(self.project, candidate_id, approved)
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def set_review_selection(self, candidate_ids: list[str]) -> None:
        if not self.project or self.owns_active_job:
            return
        try:
            self.project = self.services.set_review_selection(self.project, candidate_ids)
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def set_active_preview_candidate(self, candidate_id: str | None) -> None:
        """Persist the card whose source/draft preview is currently in focus."""

        if not self.project or self.owns_active_job:
            return
        try:
            self.project = self.services.set_active_preview_candidate(self.project, candidate_id)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return
        self.project_changed.emit(self.project)

    def adjust_candidate_boundary(self, candidate_id: str, boundary: str, delta_seconds: float) -> None:
        if not self.project or self.owns_active_job:
            return
        try:
            self.project, _validation = self.services.adjust_candidate_boundary(
                self.project, candidate_id, boundary, delta_seconds,
            )
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def revise_draft(self, candidate_id: str, **values: object) -> None:
        """Persist one candidate-scoped pending visual revision."""

        if not self.project or not self._can_start_heavy_job():
            return
        try:
            self.project = self.services.update_candidate_creative_override(
                self.project, candidate_id, **values,
            )
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return

    def revise_draft_boundary(
        self, candidate_id: str, boundary: str, delta_seconds: float,
    ) -> None:
        """Revalidate and persist one pending candidate boundary revision."""

        if not self.project or not self._can_start_heavy_job():
            return
        try:
            self.project, _validation = self.services.adjust_candidate_boundary(
                self.project, candidate_id, boundary, delta_seconds,
            )
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return

    def select_final_output(self, result_id: str) -> None:
        """Persist the exact canonical result currently open in the viewer."""

        if not self.project or self.owns_active_job or not result_id.strip():
            return
        try:
            self.project.last_final_result_id = result_id
            self.services.projects.save(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return
        self.project_changed.emit(self.project)

    def render_selected(self, candidate_ids: list[str] | None = None) -> None:
        if not self.project:
            self.error_occurred.emit(UserFacingError(
                "Проект не открыт",
                "Не удалось запустить финальный экспорт: проект не открыт.",
                "Откройте проект, подтвердите черновики и повторите запуск.",
                "render_selected called without an open project",
                "project_not_open",
            ))
            return
        if not self._can_start_heavy_job(error_code="render_already_active"):
            return
        self._launching = True
        try:
            self.run, self.prepared = self.services.prepare_selected_render(self.project, candidate_ids)
            self._bind_job(self.project, ProcessingSnapshot(
                ProcessingPhase.PREPARING, message="Создаём итоговые ролики",
            ))
            self._emit_owner_project()
            self._emit_owner_runs()
            self._emit_owner_processing()
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            self.project_changed.emit(self.project)

    def rerender(self, parent_run: ProjectRun) -> None:
        if not self.project or not self._can_start_heavy_job():
            return
        self._launching = True
        try:
            self.run, self.prepared = self.services.prepare_render_revision(self.project, parent_run)
            self._bind_job(self.project, ProcessingSnapshot(
                ProcessingPhase.PREPARING, message="Повторно создаём итоговые ролики",
            ))
            self._emit_owner_project()
            self._emit_owner_runs()
            self._emit_owner_processing()
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            self.project_changed.emit(self.project)

    def _can_start_heavy_job(self, *, error_code: str = "heavy_job_already_active") -> bool:
        if not self.active:
            return True
        owner = self.active_project_name or "другом проекте"
        self.error_occurred.emit(UserFacingError(
            "Обработка уже идёт",
            f"Сейчас выполняется тяжёлая работа в проекте «{owner}». Второй запуск временно недоступен.",
            "Можно просматривать и настраивать другие проекты. Дождитесь завершения текущей работы или остановите её в проекте-владельце.",
            f"heavy job already active for project_id={self.active_project_id or 'unknown'}",
            error_code,
        ))
        return False

    def _start_prepared_job(self, mode: str, project: DesktopProject | None = None) -> None:
        owner = project or self.project
        if owner is None:
            raise RuntimeError("Проект не открыт.")
        if mode == "analysis":
            self.run, self.prepared = self.services.prepare_analysis(owner)
            message = "Анализируем видео"
        else:
            self.run, self.prepared = self.services.prepare_run(owner)
            message = "Подготавливаем запуск"
        self._bind_job(owner, ProcessingSnapshot(ProcessingPhase.PREPARING, message=message))
        self._emit_owner_project()
        self._emit_owner_runs()
        self._emit_owner_processing()
        self.runner.start(self.prepared)

    def _bind_job(self, project: DesktopProject, snapshot: ProcessingSnapshot) -> None:
        self._job_project = project
        self._job_snapshot = snapshot
        if self.project and self.project.project_id == project.project_id:
            self.project = project
            self.snapshot = snapshot

    def _ensure_job_context(self) -> bool:
        """Adapt direct callback tests/integrations to the explicit owner model."""

        if self._job_project is None and self.project is not None:
            self._job_project = self.project
            self._job_snapshot = self.snapshot
        return self._job_project is not None

    def _owner_is_open(self) -> bool:
        return bool(
            self.project and self._job_project
            and self.project.project_id == self._job_project.project_id
        )

    def _emit_owner_project(self) -> None:
        if not self._job_project:
            return
        self.project_persisted.emit(self._job_project.project_id)
        if self._owner_is_open():
            self.project = self._job_project
            self.project_changed.emit(self.project)

    def _emit_owner_runs(self) -> None:
        if self._job_project and self._owner_is_open():
            self.runs_changed.emit(self.services.runs_for(self._job_project))

    def _emit_owner_processing(self) -> None:
        if self._owner_is_open():
            self.snapshot = self._job_snapshot
            self.processing_changed.emit(self.snapshot)

    def _release_job(self) -> None:
        owner_was_open = self._owner_is_open()
        self._job_project = None
        self._started_at = None
        self._launching = False
        self.prepared = None
        if not owner_was_open and self.project:
            # The viewed project was disabled only by the global one-job lock.
            # Re-emit its own snapshot as soon as the owner reaches a terminal state.
            self.processing_changed.emit(self.snapshot)

    def cancel(self) -> None:
        if self.source_downloader.busy:
            self._job_snapshot.phase = ProcessingPhase.CANCELLING
            self._job_snapshot.message = "Останавливаем загрузку"
            self._emit_owner_processing()
            self.source_downloader.cancel()
            return
        if self._source_probe_task is not None:
            self._source_probe_task.cancel_delivery()
            self._source_probe_task = None
            self._download_cancelled()
            return
        # Completion validation and run-history snapshot are an atomic commit.
        # Once the renderer has exited, preserve any verified artifacts instead
        # of cancelling between their snapshot and terminal project state.
        if self._finalization_task is not None:
            return
        if not self.active or not self.run:
            return
        self.run.status = RunStatus.CANCELLING
        self.services.runs.save(self.run)
        self._job_snapshot.phase = ProcessingPhase.CANCELLING
        self._job_snapshot.message = "Останавливаем обработку"
        self._emit_owner_processing()
        self.runner.cancel()

    def continue_waiting(self) -> None:
        if not self.runner.active:
            return
        self.runner.continue_waiting()
        self._job_snapshot.long_stage_warning = None
        self._emit_owner_processing()

    def _start_source_download(self) -> None:
        if not self.project or self.project.source_spec.kind != "url" or not self.project.source_spec.original_url:
            return
        try:
            self.services.mark_url_download_started(self.project)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            return
        self._started_at = time.monotonic()
        self._elapsed_timer.start()
        snapshot = ProcessingSnapshot(
            ProcessingPhase.PREPARING, stage="download", message="Загружаем видео",
            last_activity_reason="yt-dlp launch requested",
        )
        self._bind_job(self.project, snapshot)
        self._emit_owner_project()
        self._emit_owner_processing()
        self.source_downloader.download(
            self.project.source_spec.original_url,
            self.project.directory / "sources",
        )

    def _download_progress(self, progress) -> None:
        if not self._ensure_job_context():
            return
        self._job_snapshot.phase = ProcessingPhase.RUNNING
        self._job_snapshot.stage = "download"
        self._job_snapshot.message = "Загружаем видео"
        self._job_snapshot.progress_fraction = progress.fraction
        self._job_snapshot.transfer_speed = progress.speed
        self._job_snapshot.transfer_downloaded = progress.downloaded
        self._job_snapshot.transfer_total = progress.total
        self._job_snapshot.eta_seconds = progress.eta_seconds
        self._job_snapshot.last_activity_reason = "yt-dlp progress updated"
        self._emit_owner_processing()

    def _download_completed(self, path: str) -> None:
        self._ensure_job_context()
        owner = self._job_project
        if not owner or self._source_probe_task is not None:
            return
        self._job_snapshot.phase = ProcessingPhase.RUNNING
        self._job_snapshot.stage = "download"
        self._job_snapshot.message = "Проверяем загруженное видео"
        self._emit_owner_processing()
        task = BackgroundTask(lambda: self.services.validate_source(path))
        task.result_ready.connect(self._download_source_validated)
        task.error_raised.connect(self._download_source_validation_failed)
        self._source_probe_task = task
        task.start()

    def _download_source_validated(self, source: object) -> None:
        self._source_probe_task = None
        owner = self._job_project
        if owner is None:
            return
        try:
            self._job_project = self.services.complete_validated_url_download(owner, source)  # type: ignore[arg-type]
        except Exception as error:
            self._download_failed(str(error))
            return
        self._elapsed_timer.stop()
        self._job_snapshot = ProcessingSnapshot(message="Видео загружено")
        self._emit_owner_project()
        self._emit_owner_processing()
        next_action = self._after_download
        self._after_download = "none"
        try:
            if next_action == "analysis":
                self._launching = True
                self._start_prepared_job("analysis", self._job_project)
            elif next_action == "process":
                self._launching = True
                self._start_prepared_job("full", self._job_project)
            else:
                self._release_job()
        except Exception as error:
            self._job_snapshot = ProcessingSnapshot(
                ProcessingPhase.FAILED,
                message="Видео загружено, но обработку не удалось запустить",
            )
            self._emit_owner_processing()
            self.error_occurred.emit(map_error(error))
            self._release_job()

    def _download_source_validation_failed(self, error: object) -> None:
        self._source_probe_task = None
        self._download_failed(str(error))

    def _download_failed(self, message: str) -> None:
        self._ensure_job_context()
        if not self._job_project:
            return
        self._elapsed_timer.stop()
        self._launching = False
        self.services.fail_url_download(
            self._job_project,
            message,
            diagnostics=self.source_downloader.last_failure,
        )
        self._job_snapshot = ProcessingSnapshot(ProcessingPhase.FAILED, message="Не удалось загрузить видео")
        self._emit_owner_project()
        self._emit_owner_processing()
        self.error_occurred.emit(map_error(message))
        self._release_job()

    def _download_cancelled(self) -> None:
        self._ensure_job_context()
        if not self._job_project:
            return
        self._elapsed_timer.stop()
        self._launching = False
        self.services.fail_url_download(self._job_project, "Загрузка видео отменена.", cancelled=True)
        self._job_snapshot = ProcessingSnapshot(ProcessingPhase.CANCELLED, message="Загрузка отменена")
        self._emit_owner_project()
        self._emit_owner_processing()
        self._release_job()

    def _run_started(self) -> None:
        if not self._ensure_job_context():
            return
        self._started_at = time.monotonic()
        self._job_snapshot = ProcessingSnapshot(
            ProcessingPhase.PREPARING,
            message="Подготавливаем видео",
            last_activity_at=self.runner.last_activity_at,
            last_activity_reason="QProcess entered Running",
        )
        self._elapsed_timer.start()
        if self.run:
            self.run.status = RunStatus.RUNNING
            self.services.runs.save(self.run)
        self._emit_owner_runs()
        self._emit_owner_processing()

    def _stage_changed(self, stage: str, label: str) -> None:
        if not self._ensure_job_context():
            return
        self._job_snapshot.phase = ProcessingPhase.RUNNING
        self._job_snapshot.stage = stage
        self._job_snapshot.message = label
        self._job_snapshot.long_stage_warning = None
        self._emit_owner_processing()

    def _stage_running_longer_than_usual(self, stage: str, _timeout_ms: int) -> None:
        if not self._ensure_job_context():
            return
        self._job_snapshot.phase = ProcessingPhase.RUNNING
        if stage != "processing":
            self._job_snapshot.stage = stage
            self._job_snapshot.message = self._job_snapshot.stage_label
        self._job_snapshot.long_stage_warning = "Этап выполняется дольше обычного"
        self._emit_owner_processing()

    def _activity_changed(self, timestamp: str, reason: str) -> None:
        if not self._ensure_job_context():
            return
        self._job_snapshot.last_activity_at = timestamp
        self._job_snapshot.last_activity_reason = reason
        self._emit_owner_processing()

    def _emit_elapsed(self) -> None:
        if self._started_at is not None:
            self._job_snapshot.elapsed_seconds = time.monotonic() - self._started_at
            self._emit_owner_processing()

    def _log_received(self, line: str) -> None:
        if self.run:
            self.services.append_log(self.run, line)
        self.log_received.emit(line)

    def _completed(self, _exit_code: int) -> None:
        self._ensure_job_context()
        if not self._job_project or not self.run or not self.prepared:
            return
        project, run, prepared = self._job_project, self.run, self.prepared
        self._begin_finalization(
            lambda: self._finalize_success(project, run, prepared),
            "Проверяем и сохраняем результат",
        )

    def _finalize_success(
        self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun,
    ) -> _FinalizationResult:
        try:
            run = self.services.finish_success(project, run, prepared)
        except Exception as error:
            return self._finalize_failure(project, run, prepared, str(error), str(error))
        if run.status == RunStatus.FAILED:
            return _FinalizationResult(
                ProcessingPhase.FAILED,
                "Не удалось создать итоговый видеофайл",
                run,
                run.error_summary or "Не удалось создать итоговый видеофайл.",
            )
        phase = ProcessingPhase.COMPLETED_WITH_WARNINGS if run.warnings else ProcessingPhase.COMPLETED
        messages = {
            RunKind.ANALYSIS: "Анализ готов: выберите кандидаты для черновика",
            RunKind.DRAFT: "Черновики готовы: проверьте будущую сборку",
            RunKind.SELECTED_RENDER: "Итоговые ролики готовы",
        }
        message = messages.get(run.run_kind, "Ролик готов")
        if run.warnings:
            message += " с предупреждениями"
        return _FinalizationResult(phase, message, run)

    def _failed(self, message: str) -> None:
        self._ensure_job_context()
        if not self._job_project or not self.run or self._finalization_task is not None:
            return
        project, run, prepared = self._job_project, self.run, self.prepared
        details = self.runner.failure_details or message
        self._begin_finalization(
            lambda: self._finalize_failure(project, run, prepared, message, details),
            "Проверяем состояние после остановки",
        )

    def _finalize_failure(
        self,
        project: DesktopProject,
        run: ProjectRun,
        prepared: PreparedPipelineRun | None,
        message: str,
        details: str,
    ) -> _FinalizationResult:
        try:
            recovered = self.services.recover_failed_process(project, run, prepared) if prepared else None
            if recovered:
                return _FinalizationResult(
                    ProcessingPhase.COMPLETED_WITH_WARNINGS,
                    "Ролики созданы, но не удалось сохранить служебное состояние",
                    recovered,
                )
            reported = self.services.recover_reported_failure(project, run, prepared) if prepared else None
            if reported:
                return _FinalizationResult(
                    ProcessingPhase.FAILED,
                    reported.error_summary or "Не удалось завершить выбранный этап",
                    reported,
                    reported.error_summary or message,
                )
        except Exception as recovery_error:
            details = f"{details}; recovery failed: {recovery_error}"
        run = self.services.finish_failure(project, run, message, details)
        final_message = (
            "Обработка остановилась и не отвечает."
            if message == "Обработка остановилась и не отвечает."
            else "Не удалось создать ролик"
        )
        return _FinalizationResult(ProcessingPhase.FAILED, final_message, run, message)

    def _cancelled(self) -> None:
        self._ensure_job_context()
        if not self._job_project or not self.run or self._finalization_task is not None:
            return
        project, run, prepared = self._job_project, self.run, self.prepared
        self._begin_finalization(
            lambda: self._finalize_cancelled(project, run, prepared),
            "Проверяем сохранённые результаты",
        )

    def _finalize_cancelled(
        self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun | None,
    ) -> _FinalizationResult:
        if prepared:
            # A cancellation can race the final persistence step after one or
            # more candidate MP4s were already completed.  Preserve verified
            # canonical outputs instead of replacing that partial success with
            # an opaque cancelled run.
            recovered = self.services.recover_failed_process(project, run, prepared)
            if recovered:
                return _FinalizationResult(
                    ProcessingPhase.COMPLETED_WITH_WARNINGS,
                    "Часть роликов готова; незавершённые можно запустить снова",
                    recovered,
                )
        run = self.services.finish_cancelled(project, run)
        return _FinalizationResult(ProcessingPhase.CANCELLED, "Создание отменено", run)

    def _begin_finalization(
        self, operation: Callable[[], _FinalizationResult], message: str,
    ) -> None:
        if self._finalization_task is not None:
            return
        self._job_snapshot.phase = ProcessingPhase.RUNNING
        self._job_snapshot.stage = "terminal"
        self._job_snapshot.message = message
        self._job_snapshot.long_stage_warning = None
        self._emit_owner_processing()
        task = BackgroundTask(operation)
        task.result_ready.connect(self._finalization_ready)
        task.error_raised.connect(self._finalization_failed)
        self._finalization_task = task
        task.start()

    def _finalization_ready(self, value: object) -> None:
        self._finalization_task = None
        if not isinstance(value, _FinalizationResult):
            self._finalization_failed(RuntimeError("Invalid finalization result."))
            return
        self._finish(value.phase, value.message, value.run)
        if value.error_message:
            self.error_occurred.emit(map_error(value.error_message))

    def _finalization_failed(self, error: object) -> None:
        self._finalization_task = None
        self._elapsed_timer.stop()
        self._job_snapshot.phase = ProcessingPhase.FAILED
        self._job_snapshot.stage = None
        self._job_snapshot.message = "Не удалось завершить обработку"
        self._job_snapshot.elapsed_seconds = 0.0
        self._emit_owner_processing()
        self.error_occurred.emit(map_error(error if isinstance(error, Exception) else str(error)))
        self._release_job()

    def _finish(self, phase: ProcessingPhase, message: str, run: ProjectRun) -> None:
        self._elapsed_timer.stop()
        self._job_snapshot.phase = phase
        self._job_snapshot.stage = None
        self._job_snapshot.message = message
        self._job_snapshot.elapsed_seconds = 0.0
        self._emit_owner_processing()
        self._emit_owner_project()
        self._emit_owner_runs()
        self.run_finished.emit(run)
        self._release_job()
