from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from app.gui.models.processing_state import STAGE_LABELS
from app.gui.services.error_mapping import redact_secrets
from app.gui.services.pipeline_facade import PreparedPipelineRun


class QtPipelineRunner(QObject):
    """One cancellable QProcess and honest state-file based stage notifications."""

    run_started = Signal()
    stage_changed = Signal(str, str)
    progress_changed = Signal(str)
    log_received = Signal(str)
    warning_received = Signal(str)
    run_completed = Signal(int)
    run_failed = Signal(str)
    run_cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_stage)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)
        self._prepared: PreparedPipelineRun | None = None
        self._last_stage: str | None = None
        self._cancel_requested = False
        self._terminal_emitted = False

    @property
    def active(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def start(self, prepared: PreparedPipelineRun) -> None:
        if self.active:
            raise RuntimeError("Обработка уже выполняется.")
        self._prepared = prepared
        self._last_stage = None
        self._cancel_requested = False
        self._terminal_emitted = False
        self._process.setWorkingDirectory(str(prepared.working_directory))
        self._process.start(prepared.program, prepared.arguments)
        self._timer.start()
        self.run_started.emit()

    def cancel(self) -> None:
        if not self.active or self._cancel_requested:
            return
        self._cancel_requested = True
        self.log_received.emit("Запрошена безопасная остановка обработки.")
        self._process.terminate()
        QTimer.singleShot(5000, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        if self.active:
            self._process.kill()

    def _read_stdout(self) -> None:
        self._emit_log(bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace"))

    def _read_stderr(self) -> None:
        self._emit_log(bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace"))

    def _emit_log(self, value: str) -> None:
        for line in value.splitlines():
            if line.strip():
                safe = redact_secrets(line)
                self.log_received.emit(safe)
                if "предупреждение" in safe.lower():
                    self.warning_received.emit(safe)

    def _poll_stage(self) -> None:
        prepared = self._prepared
        if prepared is None or not prepared.state_path.is_file():
            return
        try:
            raw = json.loads(prepared.state_path.read_text(encoding="utf-8"))
            stages = raw.get("stages", {}) if isinstance(raw, dict) else {}
            active = [name for name, value in stages.items() if isinstance(value, dict) and value.get("status") == "running"]
            stage = active[-1] if active else None
        except (OSError, ValueError, TypeError):
            return
        if stage and stage != self._last_stage:
            self._last_stage = stage
            label = STAGE_LABELS.get(stage.split(":", 1)[0], "Обрабатываем видео")
            self.stage_changed.emit(stage, label)
            self.progress_changed.emit(label)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._finish_failure("Не удалось запустить локальный процесс обработки.")

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self._cancel_requested:
            self._finish_cancelled()
        elif exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self._finish_completed(exit_code)
        else:
            self._finish_failure(f"Процесс обработки завершился с кодом {exit_code}.")

    def _finish_completed(self, code: int) -> None:
        if self._terminal_emitted:
            return
        self._stop_timer()
        self._terminal_emitted = True
        self.run_completed.emit(code)

    def _finish_failure(self, message: str) -> None:
        if self._terminal_emitted:
            return
        self._stop_timer()
        self._terminal_emitted = True
        self.run_failed.emit(message)

    def _finish_cancelled(self) -> None:
        if self._terminal_emitted:
            return
        self._stop_timer()
        self._terminal_emitted = True
        self.run_cancelled.emit()

    def _stop_timer(self) -> None:
        self._timer.stop()
