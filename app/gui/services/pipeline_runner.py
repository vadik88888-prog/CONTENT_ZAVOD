from __future__ import annotations

import json
import subprocess
import sys
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from app.gui.models.processing_state import STAGE_LABELS
from app.gui.services.error_mapping import redact_secrets
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.utils import utc_now


class QtPipelineRunner(QObject):
    """Run the existing CLI with observable, bounded QProcess lifecycle handling."""

    DEFAULT_STARTUP_TIMEOUT_MS = 15_000
    # Some real stages (Whisper or a network request) legitimately run for minutes
    # without producing stdout.  This is deliberately much longer than a normal
    # stage transition, but it still makes an abandoned child process finite.
    DEFAULT_STALL_TIMEOUT_MS = 15 * 60 * 1000
    DEFAULT_KILL_TIMEOUT_MS = 5_000

    run_started = Signal()
    stage_changed = Signal(str, str)
    progress_changed = Signal(str)
    activity_changed = Signal(str, str)
    log_received = Signal(str)
    warning_received = Signal(str)
    run_completed = Signal(int)
    run_failed = Signal(str)
    run_cancelled = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        startup_timeout_ms: int = DEFAULT_STARTUP_TIMEOUT_MS,
        stall_timeout_ms: int = DEFAULT_STALL_TIMEOUT_MS,
        kill_timeout_ms: int = DEFAULT_KILL_TIMEOUT_MS,
    ) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.setInputChannelMode(QProcess.InputChannelMode.ManagedInputChannel)

        self._stage_timer = QTimer(self)
        self._stage_timer.setInterval(500)
        self._stage_timer.timeout.connect(self._poll_stage)
        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(min(1_000, max(20, stall_timeout_ms // 4)))
        self._watchdog_timer.timeout.connect(self._check_stall)
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.setInterval(startup_timeout_ms)
        self._startup_timer.timeout.connect(self._startup_timed_out)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.setInterval(kill_timeout_ms)
        self._kill_timer.timeout.connect(self._kill_process_tree)

        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.stateChanged.connect(self._state_changed)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)

        self._prepared: PreparedPipelineRun | None = None
        self._stall_timeout_ms = stall_timeout_ms
        self._last_stage: str | None = None
        self._state_fingerprint: tuple[int, int] | None = None
        self._launch_wall_time = 0.0
        self._last_activity_monotonic = 0.0
        self._last_activity_at: str | None = None
        self._last_activity_reason = ""
        self._failure_details = ""
        self._cancel_requested = False
        self._watchdog_failed = False
        self._running_seen = False
        self._terminal_emitted = False
        self._process_id = 0
        self._job_handle: object | None = None

    @property
    def active(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def last_activity_at(self) -> str | None:
        return self._last_activity_at

    @property
    def failure_details(self) -> str:
        return self._failure_details

    def start(self, prepared: PreparedPipelineRun) -> None:
        if self.active:
            raise RuntimeError("Обработка уже выполняется.")
        self._prepared = prepared
        self._last_stage = None
        self._state_fingerprint = None
        self._launch_wall_time = time.time()
        self._last_activity_monotonic = time.monotonic()
        self._last_activity_at = utc_now()
        self._last_activity_reason = "launch requested"
        self._failure_details = ""
        self._cancel_requested = False
        self._watchdog_failed = False
        self._running_seen = False
        self._terminal_emitted = False
        self._process_id = 0
        self._release_process_job()

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(prepared.working_directory))
        self._record("QProcess launch requested.")
        self._record(f"QProcess program: {prepared.program}")
        self._record(f"QProcess arguments: {prepared.arguments!r}")
        self._record(f"QProcess command: {prepared.command_line()}")
        self._record(f"QProcess cwd: {prepared.working_directory}")
        self._record("QProcess environment overrides: PYTHONUNBUFFERED=1; PYTHONIOENCODING=utf-8")
        self._record(
            "QProcess environment: "
            f"inherited_variable_count={len(environment.keys())}; "
            f"OPENAI_API_KEY: {'present' if environment.contains('OPENAI_API_KEY') else 'absent'}; "
            f"GEMINI_API_KEY: {'present' if environment.contains('GEMINI_API_KEY') else 'absent'}; "
            "values are intentionally not logged."
        )
        self._record("QProcess channels: separate stdout/stderr; stdin closes after startup.")

        self._stage_timer.start()
        self._startup_timer.start()
        self._process.start(prepared.program, prepared.arguments)

    def cancel(self) -> None:
        if not self.active or self._cancel_requested:
            return
        self._cancel_requested = True
        self._record("Cancellation requested; terminating QProcess and its child tree if needed.")
        self._process.terminate()
        self._kill_timer.start()

    def _state_changed(self, state: QProcess.ProcessState) -> None:
        self._record(f"QProcess state: {state.name}")
        if state == QProcess.ProcessState.Running:
            self._startup_timer.stop()
            self._running_seen = True
            self._process_id = int(self._process.processId())
            self._process.closeWriteChannel()
            self._record("QProcess stdin closed.")
            self._bind_process_tree()
            self._note_activity("QProcess entered Running")
            self._watchdog_timer.start()
            self.run_started.emit()

    def _read_stdout(self) -> None:
        self._emit_output("stdout", bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace"))

    def _read_stderr(self) -> None:
        self._emit_output("stderr", bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace"))

    def _emit_output(self, channel: str, value: str) -> None:
        if not value:
            return
        self._note_activity(f"{channel} output")
        lines = value.splitlines() or [value]
        for line in lines:
            if not line.strip():
                continue
            safe = redact_secrets(line)
            self.log_received.emit(f"{channel}: {safe}")
            if "предупреждение" in safe.lower():
                self.warning_received.emit(safe)

    def _poll_stage(self) -> None:
        prepared = self._prepared
        if prepared is None or not prepared.state_path.is_file():
            return
        try:
            stat = prepared.state_path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            # Ignore a state file from a previous run until the current engine has
            # touched it.  Reused work directories must not resurrect stale stages.
            if stat.st_mtime < self._launch_wall_time - 2:
                return
            if fingerprint != self._state_fingerprint:
                self._state_fingerprint = fingerprint
                self._note_activity("state file updated")
            # Keep the handle lifetime to a single polling read.  Windows does
            # not allow an atomic replace while some readers retain the file.
            with prepared.state_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
            stages = raw.get("stages", {}) if isinstance(raw, dict) else {}
            active_stages = [
                (str(name), value)
                for name, value in stages.items()
                if isinstance(value, dict) and value.get("status") == "running"
            ]
            stage = (
                max(active_stages, key=lambda item: str(item[1].get("started_at", "")))[0]
                if active_stages
                else None
            )
        except (OSError, ValueError, TypeError):
            return
        if stage and stage != self._last_stage:
            self._last_stage = stage
            label = STAGE_LABELS.get(stage.split(":", 1)[0], "Обрабатываем видео")
            self._note_activity(f"stage changed to {stage}")
            self._record(f"Pipeline stage: {stage} ({label})")
            self.stage_changed.emit(stage, label)
            self.progress_changed.emit(label)

    def _startup_timed_out(self) -> None:
        if self._terminal_emitted or self._running_seen:
            return
        details = self._diagnostic_details("startup watchdog timeout")
        self._record("Startup watchdog timed out before QProcess reached Running.")
        self._request_stop()
        self._finish_failure("Не удалось запустить локальный процесс обработки.", details)

    def _check_stall(self) -> None:
        if self._terminal_emitted or not self._running_seen or not self.active:
            return
        elapsed_ms = (time.monotonic() - self._last_activity_monotonic) * 1000
        if elapsed_ms < self._stall_timeout_ms:
            return
        details = self._diagnostic_details(f"stall watchdog timeout after {elapsed_ms / 1000:.1f}s")
        self._watchdog_failed = True
        self._record(f"Stall watchdog triggered after {elapsed_ms / 1000:.1f}s without activity.")
        self._request_stop()
        self._finish_failure("Обработка остановилась и не отвечает.", details)

    def _request_stop(self) -> None:
        if self.active:
            self._process.terminate()
            self._kill_timer.start()

    def _kill_process_tree(self) -> None:
        if self._terminate_process_job():
            if self.active:
                self._process.kill()
            return
        pid = self._process_id or int(self._process.processId())
        if sys.platform == "win32" and pid:
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                    **UTF8_REPLACE_TEXT,
                )
                self._record(f"taskkill /T completed with exit code {result.returncode} for PID {pid}.")
            except (OSError, subprocess.SubprocessError) as error:
                self._record(f"taskkill /T could not complete for PID {pid}: {redact_secrets(error)}")
        if self.active:
            self._record("QProcess.kill invoked after termination timeout.")
            self._process.kill()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        detail = redact_secrets(self._process.errorString())
        self._record(f"QProcess error: {error.name}; {detail}")
        if error == QProcess.ProcessError.FailedToStart:
            self._finish_failure(
                "Не удалось запустить локальный процесс обработки.",
                self._diagnostic_details(f"QProcess FailedToStart: {detail}"),
            )

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._read_stdout()
        self._read_stderr()
        self._kill_timer.stop()
        self._record(f"QProcess finished: exit_code={exit_code}; exit_status={exit_status.name}")
        self._note_activity("QProcess finished")
        if self._terminal_emitted:
            if self._watchdog_failed:
                self._kill_process_tree()
            else:
                self._release_process_job()
            return
        if self._cancel_requested:
            self._kill_process_tree()
            self._finish_cancelled()
        elif exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self._release_process_job()
            self._finish_completed(exit_code)
        else:
            self._release_process_job()
            self._finish_failure(
                f"Процесс обработки завершился с кодом {exit_code}.",
                self._diagnostic_details(f"QProcess exit: code={exit_code}; status={exit_status.name}"),
            )

    def _note_activity(self, reason: str) -> None:
        self._last_activity_monotonic = time.monotonic()
        self._last_activity_at = utc_now()
        self._last_activity_reason = reason
        self.activity_changed.emit(self._last_activity_at, reason)

    def _record(self, value: str) -> None:
        self.log_received.emit(redact_secrets(value))

    def _diagnostic_details(self, reason: str) -> str:
        prepared = self._prepared
        command = prepared.command_line() if prepared else "<unknown>"
        cwd = prepared.working_directory if prepared else "<unknown>"
        return redact_secrets(
            f"{reason}; last_activity_at={self._last_activity_at or '<none>'}; "
            f"last_activity_reason={self._last_activity_reason or '<none>'}; "
            f"process_state={self._process.state().name}; command={command}; cwd={cwd}"
        )

    def _bind_process_tree(self) -> None:
        """Put the CLI and its future ffmpeg children into one Windows job object."""

        if sys.platform != "win32" or not self._process_id:
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            job = kernel32.CreateJobObjectW(None, None)
            process = kernel32.OpenProcess(0x0100 | 0x0001, False, self._process_id)
            if not job or not process:
                error = ctypes.get_last_error()
                if process:
                    kernel32.CloseHandle(process)
                if job:
                    kernel32.CloseHandle(job)
                self._record(f"Windows job object unavailable for PID {self._process_id}; error={error}.")
                return
            assigned = kernel32.AssignProcessToJobObject(job, process)
            kernel32.CloseHandle(process)
            if not assigned:
                error = ctypes.get_last_error()
                kernel32.CloseHandle(job)
                self._record(f"Windows job object assignment failed for PID {self._process_id}; error={error}.")
                return
            self._job_handle = job
            self._record(f"Windows job object attached to PID {self._process_id} for child-tree cleanup.")
        except (AttributeError, OSError) as error:
            self._record(f"Windows job object setup failed: {redact_secrets(error)}")

    def _terminate_process_job(self) -> bool:
        if sys.platform != "win32" or self._job_handle is None:
            return False
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            terminated = bool(kernel32.TerminateJobObject(self._job_handle, 1))
            error = ctypes.get_last_error() if not terminated else 0
            self._record(f"Windows job tree termination requested; success={terminated}; error={error}.")
            return terminated
        except (AttributeError, OSError) as error:
            self._record(f"Windows job tree termination failed: {redact_secrets(error)}")
            return False
        finally:
            self._release_process_job()

    def _release_process_job(self) -> None:
        if self._job_handle is None:
            return
        try:
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
                kernel32.CloseHandle.restype = wintypes.BOOL
                kernel32.CloseHandle(self._job_handle)
        finally:
            self._job_handle = None

    def _finish_completed(self, code: int) -> None:
        if self._terminal_emitted:
            return
        self._stop_timers()
        self._terminal_emitted = True
        self.run_completed.emit(code)

    def _finish_failure(self, message: str, details: str | None = None) -> None:
        if self._terminal_emitted:
            return
        self._failure_details = redact_secrets(details or self._diagnostic_details(message))
        self._stop_timers()
        self._terminal_emitted = True
        self.run_failed.emit(message)

    def _finish_cancelled(self) -> None:
        if self._terminal_emitted:
            return
        self._stop_timers()
        self._terminal_emitted = True
        self.run_cancelled.emit()

    def _stop_timers(self) -> None:
        self._stage_timer.stop()
        self._watchdog_timer.stop()
        self._startup_timer.stop()
