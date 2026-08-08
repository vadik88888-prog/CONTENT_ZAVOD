from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from app.gui.models.processing_state import STAGE_LABELS
from app.gui.services.error_mapping import redact_secrets
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.run_artifacts import find_run_artifact_metadata
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.utils import utc_now


class QtPipelineRunner(QObject):
    """Run the existing CLI with observable, bounded QProcess lifecycle handling."""

    DEFAULT_STARTUP_TIMEOUT_MS = 15_000
    # This is a *warning* threshold, not a process-kill deadline. A real
    # Whisper/ffmpeg/remote-provider stage can be silent for a long time.
    DEFAULT_STALL_TIMEOUT_MS = 15 * 60 * 1000
    DEFAULT_LONG_STAGE_TIMEOUT_MS = 90 * 60 * 1000
    DEFAULT_KILL_TIMEOUT_MS = 5_000
    LONG_SOURCE_DURATION_SECONDS = 90 * 60
    LONG_RUNNING_STAGES = frozenset({
        "transcription", "audio_features", "scene_detection", "visual_analysis",
        "draft", "production_render", "render", "tts", "audio",
    })
    # A focused desktop job must not revive an actively-running stage from a
    # prior analysis. Completed analysis entries intentionally remain in the
    # engine state file, so this filters only the current running stage.
    ANALYSIS_STAGE_PREFIXES = frozenset({
        "source", "media", "metadata", "transcription", "transcript_features",
        "audio_features", "scene_detection", "visual_analysis", "content_profile",
        "content_map", "semantic_boundaries", "story_units", "coverage_map",
        "clip_count_recommendation", "candidate_generation", "ai_reranking",
        "virality_profiles", "virality_ranking", "analysis_artifact", "report", "terminal",
        # Current engine names, plus the legacy aliases above retained for
        # existing persisted state files.
        "candidates_v2", "local_scoring", "shortlist", "ai_ranking", "final_selection",
        "video_content_profile", "global_content_map",
    })
    DRAFT_STAGE_PREFIXES = frozenset({
        "analysis_handoff", "content_transformation", "production_plan", "draft_preview",
        "draft_artifact",
        "transformation_source_context", "transformation_semantic_representation",
        "transformation_narrative_plan", "transformation_script_draft",
        "transformation_script_validation", "transformation_final_script", "transformation_result",
    })
    FINAL_STAGE_PREFIXES = frozenset({
        "approved_draft_plan", "approved_draft_handoff", "tts_generation", "tts",
        "audio_composition", "audio", "production_render", "render",
    })
    DELIVERY_STAGE_PREFIXES = DRAFT_STAGE_PREFIXES | FINAL_STAGE_PREFIXES
    STAGE_LABEL_OVERRIDES = {
        "video_content_profile": "Анализируем содержание",
        "global_content_map": "Анализируем содержание",
        "story_units": "Анализируем содержание",
        "semantic_boundaries": "Ищем сильные моменты",
        "candidates_v2": "Ищем сильные моменты",
        "local_scoring": "Ищем сильные моменты",
        "shortlist": "Ищем сильные моменты",
        "ai_ranking": "Ищем сильные моменты",
        "final_selection": "Ищем сильные моменты",
        "analysis_artifact": "Сохраняем анализ",
        "analysis_handoff": "Используем готовый анализ",
        "content_transformation": "Готовим сценарий",
        "transformation_result": "Готовим сценарий",
        "draft_preview": "Собираем черновик",
        "draft_artifact": "Сохраняем черновики",
        "approved_draft_plan": "Проверяем утверждённый черновик",
        "approved_draft_handoff": "Готовим утверждённые черновики",
        "tts_generation": "Готовим озвучку",
        "audio_composition": "Собираем звук",
        "production_render": "Экспортируем итоговый ролик",
        "report": "Проверяем результат",
        "terminal": "Завершаем этап",
    }

    run_started = Signal()
    stage_changed = Signal(str, str)
    progress_changed = Signal(str)
    activity_changed = Signal(str, str)
    log_received = Signal(str)
    warning_received = Signal(str)
    stage_running_longer_than_usual = Signal(str, int)
    run_completed = Signal(int)
    run_failed = Signal(str)
    run_cancelled = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        startup_timeout_ms: int = DEFAULT_STARTUP_TIMEOUT_MS,
        stall_timeout_ms: int = DEFAULT_STALL_TIMEOUT_MS,
        long_stage_timeout_ms: int = DEFAULT_LONG_STAGE_TIMEOUT_MS,
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
        self._long_stage_timeout_ms = max(stall_timeout_ms, long_stage_timeout_ms)
        self._last_stage: str | None = None
        self._last_ignored_engine_stage: str | None = None
        self._state_fingerprint: tuple[int, int] | None = None
        self._activity_fingerprints: dict[str, object] = {}
        self._launch_wall_time = 0.0
        self._stage_started_monotonic = 0.0
        self._next_stage_warning_monotonic = 0.0
        self._warned_stage: str | None = None
        self._last_child_activity: tuple[int, int] | None = None
        self._last_activity_monotonic = 0.0
        self._last_activity_at: str | None = None
        self._last_activity_reason = ""
        self._failure_details = ""
        self._cancel_requested = False
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
        self._last_ignored_engine_stage = None
        self._state_fingerprint = None
        self._activity_fingerprints = {}
        self._launch_wall_time = time.time()
        self._last_activity_monotonic = time.monotonic()
        self._stage_started_monotonic = self._last_activity_monotonic
        self._next_stage_warning_monotonic = self._last_activity_monotonic + self._current_timeout_seconds()
        self._warned_stage = None
        self._last_child_activity = None
        self._last_activity_at = utc_now()
        self._last_activity_reason = "launch requested"
        self._failure_details = ""
        self._cancel_requested = False
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

    def continue_waiting(self) -> None:
        """Acknowledge a slow-stage warning without disturbing the child process."""

        if not self.active:
            return
        self._warned_stage = None
        self._next_stage_warning_monotonic = time.monotonic() + self._current_timeout_seconds()
        self._record("User chose to continue waiting for the current pipeline stage.")

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
            self._stage_started_monotonic = time.monotonic()
            self._next_stage_warning_monotonic = (
                self._stage_started_monotonic + self._current_timeout_seconds()
            )
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
        self._resolve_engine_paths()
        self._poll_activity_sources()
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
        except (OSError, ValueError, TypeError):
            return
        presentations = [
            (name, value, self._present_stage(prepared, name))
            for name, value in active_stages
        ]
        eligible = [item for item in presentations if item[2] is not None]
        if not eligible:
            if active_stages:
                stale_stage = max(active_stages, key=lambda item: str(item[1].get("started_at", "")))[0]
                if stale_stage != self._last_ignored_engine_stage:
                    self._last_ignored_engine_stage = stale_stage
                    self._record(
                        f"Ignoring stale pipeline stage for {self._job_mode(prepared)} job: {stale_stage}."
                    )
            else:
                self._last_ignored_engine_stage = None
            return

        engine_stage, _value, presentation = max(
            eligible, key=lambda item: str(item[1].get("started_at", "")),
        )
        assert presentation is not None
        stage, label = presentation
        self._last_ignored_engine_stage = None
        if stage != self._last_stage:
            self._last_stage = stage
            self._stage_started_monotonic = time.monotonic()
            self._warned_stage = None
            self._next_stage_warning_monotonic = (
                self._stage_started_monotonic + self._current_timeout_seconds()
            )
            self._note_activity(f"stage changed to {engine_stage}")
            self._record(f"Pipeline stage: {engine_stage} -> {stage} ({label})")
            self.stage_changed.emit(stage, label)
            self.progress_changed.emit(label)

    @classmethod
    def _job_mode(cls, prepared: PreparedPipelineRun) -> str:
        """Classify the focused desktop job without depending on source titles."""

        flags = prepared.runtime_flags
        mode = str(flags.get("mode") or "").strip().lower()
        if mode == "analysis":
            return "analysis"
        if mode == "draft":
            return "draft"
        if mode == "selected_render" or str(flags.get("render_only") or "").lower() == "true":
            return "final"
        # Older reconstructed records can lack runtime flags. Infer only from
        # an explicit CLI subcommand, never from a source path or filename.
        arguments = {str(argument).strip().lower() for argument in prepared.arguments}
        if "draft" in arguments:
            return "draft"
        if "analyze" in arguments:
            return "analysis"
        if "render" in arguments or "--production-render-only" in arguments:
            return "final"
        return "full"

    @classmethod
    def _present_stage(cls, prepared: PreparedPipelineRun, engine_stage: str) -> tuple[str, str] | None:
        """Translate an engine stage into the current job's visible stage.

        A draft is represented as draft generation and a selected render as
        final export. This leaves the existing generic progress component in
        place while preventing analysis rows from appearing during export.
        """

        parent, separator, suffix = engine_stage.partition(":")
        mode = cls._job_mode(prepared)
        if mode == "analysis":
            if parent not in cls.ANALYSIS_STAGE_PREFIXES:
                return None
            return engine_stage, cls._stage_label(parent)
        if mode == "draft":
            if parent not in cls.DRAFT_STAGE_PREFIXES:
                return None
            return cls._scoped_stage("draft", parent, separator, suffix), cls._stage_label(parent)
        if mode == "final":
            if parent not in cls.FINAL_STAGE_PREFIXES:
                return None
            return cls._scoped_stage("production_render", parent, separator, suffix), cls._stage_label(parent)

        # A legacy full process really does include its own analysis. Keep it
        # visible, then scope only the delivery portion as final export.
        if parent in cls.DELIVERY_STAGE_PREFIXES:
            return cls._scoped_stage("production_render", parent, separator, suffix), cls._stage_label(parent)
        return engine_stage, cls._stage_label(parent)

    @staticmethod
    def _scoped_stage(scope: str, parent: str, separator: str, suffix: str) -> str:
        detail = f"{parent}{separator}{suffix}" if separator else parent
        return f"{scope}:{detail}"

    @classmethod
    def _stage_label(cls, parent: str) -> str:
        return cls.STAGE_LABEL_OVERRIDES.get(
            parent,
            STAGE_LABELS.get(parent, "Обрабатываем видео"),
        )

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
        self._poll_activity_sources()
        # Running is the authoritative liveness signal. Never turn a silent
        # but still-live QProcess into a failed run or terminate it here.
        if self._process.state() != QProcess.ProcessState.Running:
            return
        now = time.monotonic()
        if now < self._next_stage_warning_monotonic or self._warned_stage == self._current_stage():
            return
        stage = self._current_stage()
        timeout_ms = int(self._current_timeout_seconds() * 1000)
        self._warned_stage = stage
        self._record(
            f"Stage {stage} is running longer than usual after "
            f"{now - self._stage_started_monotonic:.1f}s; QProcess is still Running."
        )
        self.stage_running_longer_than_usual.emit(stage, timeout_ms)

    def _current_stage(self) -> str:
        return self._last_stage or "processing"

    def _current_timeout_seconds(self) -> float:
        parent_stage = self._current_stage().split(":", 1)[0]
        source_duration = self._prepared.source_duration_seconds if self._prepared else None
        timeout_ms = (
            self._long_stage_timeout_ms
            if parent_stage in self.LONG_RUNNING_STAGES
            or (source_duration is not None and source_duration > self.LONG_SOURCE_DURATION_SECONDS)
            else self._stall_timeout_ms
        )
        return timeout_ms / 1000

    def _poll_activity_sources(self) -> None:
        """Treat durable engine evidence as liveness, even without progress text."""

        self._resolve_engine_paths()
        prepared = self._prepared
        if prepared is None:
            return
        heartbeat = prepared.heartbeat_path or prepared.state_path.with_name("heartbeat.json")
        self._observe_file("heartbeat file", heartbeat)
        if prepared.log_path is not None:
            self._observe_file("pipeline log", prepared.log_path)
        self._observe_artifact_directory("work artifact", prepared.state_path.parent)
        self._observe_artifact_directory("output artifact", prepared.output_directory)
        self._observe_child_process_activity()

    def _resolve_engine_paths(self) -> None:
        """Adopt paths emitted by the child engine as soon as they are available."""

        prepared = self._prepared
        if prepared is None or not prepared.run_id:
            return
        metadata = find_run_artifact_metadata(
            prepared.working_directory,
            run_id=prepared.run_id,
            project_id=prepared.project_id,
            preferred_path=prepared.artifact_metadata_path,
            allow_legacy_scan=prepared.allow_legacy_artifact_scan,
        )
        if metadata is None:
            return
        paths = metadata.get("paths", {})
        if not isinstance(paths, dict):
            return
        try:
            state_path = Path(str(paths["state_path"]))
            report_path = Path(str(paths["report_path"]))
            output_directory = Path(str(paths["output_directory"]))
        except (KeyError, TypeError, ValueError):
            return
        heartbeat = paths.get("heartbeat_path")
        manifest = paths.get("manifest_path")
        self._prepared = replace(
            prepared,
            state_path=state_path,
            report_path=report_path,
            output_directory=output_directory,
            heartbeat_path=Path(str(heartbeat)) if heartbeat else None,
            manifest_path=Path(str(manifest)) if manifest else None,
            artifact_metadata_path=Path(str(metadata["metadata_path"])) if metadata.get("metadata_path") else prepared.artifact_metadata_path,
            project_id=str(metadata.get("project_id") or prepared.project_id or "") or None,
        )

    def _observe_file(self, name: str, path: Path) -> None:
        try:
            stat = path.stat()
            fingerprint: object = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            fingerprint = None
        self._observe_fingerprint(name, fingerprint)

    def _observe_artifact_directory(self, name: str, directory: Path) -> None:
        self._observe_fingerprint(name, _artifact_fingerprint(directory))

    def _observe_fingerprint(self, name: str, fingerprint: object) -> None:
        unseen = object()
        previous = self._activity_fingerprints.get(name, unseen)
        self._activity_fingerprints[name] = fingerprint
        if fingerprint is not None and previous != fingerprint:
            self._note_activity(f"{name} updated")

    def _observe_child_process_activity(self) -> None:
        token = self._child_process_activity_token()
        if token is None:
            return
        if self._last_child_activity is not None and token != self._last_child_activity:
            self._note_activity("child process CPU activity")
        self._last_child_activity = token

    def _child_process_activity_token(self) -> tuple[int, int] | None:
        """Return aggregate CPU work for the Windows job (CLI plus children)."""

        if sys.platform != "win32" or self._job_handle is None:
            return None
        try:
            import ctypes

            class _JobAccounting(ctypes.Structure):
                _fields_ = [
                    ("TotalUserTime", ctypes.c_longlong),
                    ("TotalKernelTime", ctypes.c_longlong),
                    ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                    ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                    ("TotalPageFaultCount", ctypes.c_uint32),
                    ("TotalProcesses", ctypes.c_uint32),
                    ("ActiveProcesses", ctypes.c_uint32),
                    ("TotalTerminatedProcesses", ctypes.c_uint32),
                ]

            accounting = _JobAccounting()
            returned = ctypes.c_uint32()
            queried = ctypes.WinDLL("kernel32", use_last_error=True).QueryInformationJobObject(
                self._job_handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), ctypes.byref(returned),
            )
            if not queried:
                return None
            return accounting.TotalUserTime + accounting.TotalKernelTime, accounting.ActiveProcesses
        except (AttributeError, OSError):
            return None

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


def _artifact_fingerprint(directory: Path) -> tuple[int, int, int] | None:
    """A cheap, content-free signal that an engine artifact is being written."""

    try:
        if not directory.is_dir():
            return None
        latest_mtime = 0
        total_size = 0
        count = 0
        for root, _directories, filenames in os.walk(directory):
            for filename in filenames:
                if filename in {"state.json", "heartbeat.json"} or filename.endswith(".write.lock"):
                    continue
                path = Path(root) / filename
                try:
                    stat = path.stat()
                except OSError:
                    continue
                latest_mtime = max(latest_mtime, stat.st_mtime_ns)
                total_size += stat.st_size
                count += 1
        return latest_mtime, total_size, count
    except OSError:
        return None
