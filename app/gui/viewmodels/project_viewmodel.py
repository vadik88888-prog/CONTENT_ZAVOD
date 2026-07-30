from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from app.gui.models import DesktopProject, ProcessingPhase, ProcessingSnapshot, ProjectRun, RunKind, RunStatus
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.error_mapping import map_error
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner
from app.gui.services.url_source_service import URLSourceService


class ProjectViewModel(QObject):
    project_changed = Signal(object)
    runs_changed = Signal(list)
    processing_changed = Signal(object)
    error_occurred = Signal(object)
    log_received = Signal(str)
    run_finished = Signal(object)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self.project: DesktopProject | None = None
        self.run: ProjectRun | None = None
        self.prepared: PreparedPipelineRun | None = None
        self.snapshot = ProcessingSnapshot()
        self._launching = False
        self._started_at: float | None = None
        self._after_download = "process"
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

    def open(self, project: DesktopProject) -> None:
        try:
            self.project = self.services.refresh_setup_estimate(project)
        except Exception:
            # An unavailable optional provider tariff must not prevent a person
            # from opening their existing candidates and completed videos.
            self.project = project
        self.snapshot = ProcessingSnapshot()
        self.project_changed.emit(self.project)
        self.runs_changed.emit(self.services.runs_for(self.project))
        self.processing_changed.emit(self.snapshot)

    def save_options(self, **values: object) -> None:
        if not self.project or self.active:
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
        if not self.project or self.active:
            return
        self._launching = True
        self._after_download = "process"
        if not self.project.source_spec.is_ready:
            self._start_source_download()
            return
        try:
            self.run, self.prepared = self.services.prepare_run(self.project)
            self.snapshot = ProcessingSnapshot(ProcessingPhase.PREPARING, message="Подготавливаем запуск")
            self.project_changed.emit(self.project)
            self.runs_changed.emit(self.services.runs_for(self.project))
            self.processing_changed.emit(self.snapshot)
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            if self.project:
                self.project_changed.emit(self.project)

    def start_analysis(self) -> None:
        """Start analysis only; delivery remains unavailable until draft review."""

        if not self.project or self.active:
            return
        self._launching = True
        self._after_download = "analysis"
        if not self.project.source_spec.is_ready:
            self._start_source_download()
            return
        try:
            self.run, self.prepared = self.services.prepare_analysis(self.project)
            self.snapshot = ProcessingSnapshot(ProcessingPhase.PREPARING, message="Анализируем видео")
            self.project_changed.emit(self.project)
            self.runs_changed.emit(self.services.runs_for(self.project))
            self.processing_changed.emit(self.snapshot)
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            if self.project:
                self.project_changed.emit(self.project)

    def start_download(self) -> None:
        """Download a public source as its own explicit user step."""

        if not self.project or self.active:
            return
        if self.project.source_spec.kind != "url" or self.project.source_spec.is_ready:
            return
        self._launching = True
        self._after_download = "none"
        self._start_source_download()

    def build_drafts(self, candidate_ids: list[str]) -> None:
        if not self.project or self.active:
            return
        self._launching = True
        try:
            self.run, self.prepared = self.services.prepare_draft(self.project, candidate_ids)
            self.snapshot = ProcessingSnapshot(ProcessingPhase.PREPARING, message="Собираем быстрые черновики")
            self.project_changed.emit(self.project)
            self.runs_changed.emit(self.services.runs_for(self.project))
            self.processing_changed.emit(self.snapshot)
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            self.project_changed.emit(self.project)

    def select_drafts(self, candidate_ids: list[str]) -> None:
        if not self.project or self.active:
            return
        try:
            self.project = self.services.select_draft_candidates(self.project, candidate_ids)
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def set_draft_approval(self, candidate_id: str, approved: bool) -> None:
        """Record an explicit keep/reject decision for one ready draft."""

        if not self.project or self.active:
            return
        try:
            self.project = self.services.set_draft_approval(self.project, candidate_id, approved)
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def set_review_selection(self, candidate_ids: list[str]) -> None:
        if not self.project or self.active:
            return
        try:
            self.project = self.services.set_review_selection(self.project, candidate_ids)
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def adjust_candidate_boundary(self, candidate_id: str, boundary: str, delta_seconds: float) -> None:
        if not self.project or self.active:
            return
        try:
            self.project, _validation = self.services.adjust_candidate_boundary(
                self.project, candidate_id, boundary, delta_seconds,
            )
            self.project_changed.emit(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))

    def render_selected(self) -> None:
        if not self.project or self.active:
            return
        self._launching = True
        try:
            self.run, self.prepared = self.services.prepare_selected_render(self.project)
            self.snapshot = ProcessingSnapshot(ProcessingPhase.PREPARING, message="Создаём итоговые ролики")
            self.project_changed.emit(self.project)
            self.runs_changed.emit(self.services.runs_for(self.project))
            self.processing_changed.emit(self.snapshot)
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            self.project_changed.emit(self.project)

    def rerender(self, parent_run: ProjectRun) -> None:
        if not self.project or self.active:
            return
        self._launching = True
        try:
            self.run, self.prepared = self.services.prepare_render_revision(self.project, parent_run)
            self.project_changed.emit(self.project)
            self.runs_changed.emit(self.services.runs_for(self.project))
            self.runner.start(self.prepared)
        except Exception as error:
            self._launching = False
            self.error_occurred.emit(map_error(error))
            self.project_changed.emit(self.project)

    def cancel(self) -> None:
        if self.source_downloader.busy:
            self.snapshot.phase = ProcessingPhase.CANCELLING
            self.snapshot.message = "Останавливаем загрузку"
            self.processing_changed.emit(self.snapshot)
            self.source_downloader.cancel()
            return
        if not self.active or not self.run:
            return
        self.run.status = RunStatus.CANCELLING
        self.services.runs.save(self.run)
        self.snapshot.phase = ProcessingPhase.CANCELLING
        self.snapshot.message = "Останавливаем обработку"
        self.processing_changed.emit(self.snapshot)
        self.runner.cancel()

    def continue_waiting(self) -> None:
        if not self.runner.active:
            return
        self.runner.continue_waiting()
        self.snapshot.long_stage_warning = None
        self.processing_changed.emit(self.snapshot)

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
        self.snapshot = ProcessingSnapshot(
            ProcessingPhase.PREPARING, stage="download", message="Загружаем видео",
            last_activity_reason="yt-dlp launch requested",
        )
        self.project_changed.emit(self.project)
        self.processing_changed.emit(self.snapshot)
        self.source_downloader.download(
            self.project.source_spec.original_url,
            self.project.directory / "sources",
        )

    def _download_progress(self, progress) -> None:
        self.snapshot.phase = ProcessingPhase.RUNNING
        self.snapshot.stage = "download"
        self.snapshot.message = "Загружаем видео"
        self.snapshot.progress_fraction = progress.fraction
        self.snapshot.transfer_speed = progress.speed
        self.snapshot.transfer_downloaded = progress.downloaded
        self.snapshot.transfer_total = progress.total
        self.snapshot.eta_seconds = progress.eta_seconds
        self.snapshot.last_activity_reason = "yt-dlp progress updated"
        self.processing_changed.emit(self.snapshot)

    def _download_completed(self, path: str) -> None:
        if not self.project:
            return
        try:
            self.project = self.services.complete_url_download(self.project, path)
        except Exception as error:
            self._download_failed(str(error))
            return
        self._elapsed_timer.stop()
        self._launching = False
        self.snapshot = ProcessingSnapshot(message="Видео загружено")
        self.project_changed.emit(self.project)
        self.processing_changed.emit(self.snapshot)
        next_action = self._after_download
        self._after_download = "none"
        if next_action == "analysis":
            self.start_analysis()
        elif next_action == "process":
            self.start()

    def _download_failed(self, message: str) -> None:
        if not self.project:
            return
        self._elapsed_timer.stop()
        self._launching = False
        self.services.fail_url_download(self.project, message)
        self.snapshot = ProcessingSnapshot(ProcessingPhase.FAILED, message="Не удалось загрузить видео")
        self.project_changed.emit(self.project)
        self.processing_changed.emit(self.snapshot)
        self.error_occurred.emit(map_error(message))

    def _download_cancelled(self) -> None:
        if not self.project:
            return
        self._elapsed_timer.stop()
        self._launching = False
        self.services.fail_url_download(self.project, "Загрузка видео отменена.", cancelled=True)
        self.snapshot = ProcessingSnapshot(ProcessingPhase.CANCELLED, message="Загрузка отменена")
        self.project_changed.emit(self.project)
        self.processing_changed.emit(self.snapshot)

    def _run_started(self) -> None:
        self._started_at = time.monotonic()
        self.snapshot = ProcessingSnapshot(
            ProcessingPhase.PREPARING,
            message="Подготавливаем видео",
            last_activity_at=self.runner.last_activity_at,
            last_activity_reason="QProcess entered Running",
        )
        self._elapsed_timer.start()
        if self.run:
            self.run.status = RunStatus.RUNNING
            self.services.runs.save(self.run)
        if self.project:
            self.runs_changed.emit(self.services.runs_for(self.project))
        self.processing_changed.emit(self.snapshot)

    def _stage_changed(self, stage: str, label: str) -> None:
        self.snapshot.phase = ProcessingPhase.RUNNING
        self.snapshot.stage = stage
        self.snapshot.message = label
        self.snapshot.long_stage_warning = None
        self.processing_changed.emit(self.snapshot)

    def _stage_running_longer_than_usual(self, stage: str, _timeout_ms: int) -> None:
        self.snapshot.phase = ProcessingPhase.RUNNING
        if stage != "processing":
            self.snapshot.stage = stage
            self.snapshot.message = self.snapshot.stage_label
        self.snapshot.long_stage_warning = "Этап выполняется дольше обычного"
        self.processing_changed.emit(self.snapshot)

    def _activity_changed(self, timestamp: str, reason: str) -> None:
        self.snapshot.last_activity_at = timestamp
        self.snapshot.last_activity_reason = reason
        self.processing_changed.emit(self.snapshot)

    def _emit_elapsed(self) -> None:
        if self._started_at is not None:
            self.snapshot.elapsed_seconds = time.monotonic() - self._started_at
            self.processing_changed.emit(self.snapshot)

    def _log_received(self, line: str) -> None:
        if self.run:
            self.services.append_log(self.run, line)
        self.log_received.emit(line)

    def _completed(self, _exit_code: int) -> None:
        if not self.project or not self.run or not self.prepared:
            return
        try:
            run = self.services.finish_success(self.project, self.run, self.prepared)
        except Exception as error:
            self._failed(str(error))
            return
        if run.status == RunStatus.FAILED:
            self._finish(ProcessingPhase.FAILED, "Не удалось создать итоговый видеофайл", run)
            self.error_occurred.emit(map_error(run.error_summary or "Не удалось создать итоговый видеофайл."))
            return
        phase = ProcessingPhase.COMPLETED_WITH_WARNINGS if run.warnings else ProcessingPhase.COMPLETED
        messages = {
            RunKind.ANALYSIS: "Анализ готов: выберите кандидаты для черновика",
            RunKind.DRAFT: "Черновики готовы: проверьте будущую сборку",
            RunKind.SELECTED_RENDER: "Итоговые ролики готовы",
        }
        message = messages.get(run.run_kind, "Ролик готов")
        if run.warnings:
            message += " с предупреждениями"
        self._finish(phase, message, run)

    def _failed(self, message: str) -> None:
        if not self.project or not self.run:
            return
        if self.prepared:
            recovered = self.services.recover_failed_process(self.project, self.run, self.prepared)
            if recovered:
                self._finish(
                    ProcessingPhase.COMPLETED_WITH_WARNINGS,
                    "Ролики созданы, но не удалось сохранить служебное состояние",
                    recovered,
                )
                return
        run = self.services.finish_failure(self.project, self.run, message, self.runner.failure_details or message)
        final_message = (
            "Обработка остановилась и не отвечает."
            if message == "Обработка остановилась и не отвечает."
            else "Не удалось создать ролик"
        )
        self._finish(ProcessingPhase.FAILED, final_message, run)
        self.error_occurred.emit(map_error(message))

    def _cancelled(self) -> None:
        if not self.project or not self.run:
            return
        if self.prepared:
            # A cancellation can race the final persistence step after one or
            # more candidate MP4s were already completed.  Preserve verified
            # canonical outputs instead of replacing that partial success with
            # an opaque cancelled run.
            recovered = self.services.recover_failed_process(self.project, self.run, self.prepared)
            if recovered:
                self._finish(
                    ProcessingPhase.COMPLETED_WITH_WARNINGS,
                    "Часть роликов готова; незавершённые можно запустить снова",
                    recovered,
                )
                return
        run = self.services.finish_cancelled(self.project, self.run)
        self._finish(ProcessingPhase.CANCELLED, "Создание отменено", run)

    def _finish(self, phase: ProcessingPhase, message: str, run: ProjectRun) -> None:
        self._elapsed_timer.stop()
        self.snapshot.phase = phase
        self.snapshot.stage = None
        self.snapshot.message = message
        self.snapshot.elapsed_seconds = 0.0
        self.processing_changed.emit(self.snapshot)
        self.project_changed.emit(self.project)
        self.runs_changed.emit(self.services.runs_for(self.project))
        self.run_finished.emit(run)
        self._started_at = None
        self._launching = False
        self.prepared = None
