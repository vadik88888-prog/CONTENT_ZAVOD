from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from app.gui.models import DesktopProject, ProcessingPhase, ProcessingSnapshot, ProjectRun, RunStatus
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.error_mapping import map_error
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner


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
        self._started_at: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._elapsed_timer.timeout.connect(self._emit_elapsed)
        self.runner = QtPipelineRunner(self)
        self.runner.run_started.connect(self._run_started)
        self.runner.stage_changed.connect(self._stage_changed)
        self.runner.log_received.connect(self._log_received)
        self.runner.run_completed.connect(self._completed)
        self.runner.run_failed.connect(self._failed)
        self.runner.run_cancelled.connect(self._cancelled)

    @property
    def active(self) -> bool:
        return self.runner.active

    def open(self, project: DesktopProject) -> None:
        self.project = project
        self.snapshot = ProcessingSnapshot()
        self.project_changed.emit(project)
        self.runs_changed.emit(self.services.runs_for(project))
        self.processing_changed.emit(self.snapshot)

    def save_options(self, **values: object) -> None:
        if not self.project or self.active:
            return
        for name, value in values.items():
            if hasattr(self.project.settings, name):
                setattr(self.project.settings, name, value)
        try:
            self.services.save_project(self.project)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return
        self.project_changed.emit(self.project)

    def start(self) -> None:
        if not self.project or self.active:
            return
        try:
            self.run, self.prepared = self.services.prepare_run(self.project)
            self.project_changed.emit(self.project)
            self.runs_changed.emit(self.services.runs_for(self.project))
            self.runner.start(self.prepared)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            if self.project:
                self.project_changed.emit(self.project)

    def cancel(self) -> None:
        if not self.active or not self.run:
            return
        self.run.status = RunStatus.CANCELLING
        self.services.runs.save(self.run)
        self.snapshot.phase = ProcessingPhase.CANCELLING
        self.snapshot.message = "Останавливаем обработку"
        self.processing_changed.emit(self.snapshot)
        self.runner.cancel()

    def _run_started(self) -> None:
        self._started_at = time.monotonic()
        self.snapshot = ProcessingSnapshot(ProcessingPhase.PREPARING, message="Подготавливаем видео")
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
        self._finish(phase, "Ролик готов" if not run.warnings else "Ролик готов с предупреждениями", run)

    def _failed(self, message: str) -> None:
        if not self.project or not self.run:
            return
        run = self.services.finish_failure(self.project, self.run, message)
        self._finish(ProcessingPhase.FAILED, "Не удалось создать ролик", run)
        self.error_occurred.emit(map_error(message))

    def _cancelled(self) -> None:
        if not self.project or not self.run:
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
        self.prepared = None
