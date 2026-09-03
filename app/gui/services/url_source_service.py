"""Non-blocking public URL inspection and download for the Qt desktop client."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from app.source_download import (
    build_ytdlp_download_arguments,
    build_ytdlp_inspect_arguments,
    classify_ytdlp_failure,
    cleanup_partial_downloads,
    detect_ytdlp_capabilities,
    normalize_ytdlp_diagnostics,
    parse_download_progress,
    parse_url_metadata,
    should_retry_with_po_token_fallback,
    sanitize_public_url_for_diagnostics,
    validate_public_video_url,
    YtDlpCapabilities,
)
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.utils import utc_now
from app.gui.services.windows_process_job import (
    attach_windows_process_job,
    close_windows_process_job,
    terminate_windows_process_job,
)


@dataclass(frozen=True, slots=True)
class URLDownloadFailure:
    """Safe terminal yt-dlp evidence retained for project-local diagnostics."""

    url: str
    exit_code: int | None
    reason: str
    last_diagnostics: str
    occurred_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class URLSourceService(QObject):
    """Runs yt-dlp through QProcess so link operations never block the GUI thread."""

    metadata_ready = Signal(dict)
    download_progress = Signal(object)
    download_completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.stateChanged.connect(self._state_changed)
        self._mode: str | None = None
        self._url: str | None = None
        self._target_directory: Path | None = None
        self._stdout_chunks: list[str] = []
        self._stderr_chunks: list[str] = []
        self._stdout_lines: list[str] = []
        self._stdout_pending = ""
        self._capabilities: YtDlpCapabilities | None = None
        self._using_po_token_fallback = False
        self.last_diagnostics = ""
        self.last_failure: URLDownloadFailure | None = None
        self._cancel_requested = False
        self._reported_process_error = False
        self._process_id = 0
        self._job_handle: object | None = None
        self._cancel_kill_timer = QTimer(self)
        self._cancel_kill_timer.setSingleShot(True)
        self._cancel_kill_timer.timeout.connect(self._kill_if_cancelling)

    @property
    def busy(self) -> bool:
        return self._mode is not None

    def inspect(self, url: str) -> None:
        safe_url = self._begin(url, "metadata")
        if not safe_url:
            return
        assert self._capabilities is not None
        self._start(build_ytdlp_inspect_arguments(self._capabilities, safe_url))

    def download(self, url: str, target_directory: Path) -> None:
        safe_url = self._begin(url, "download")
        if not safe_url:
            return
        directory = target_directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self._target_directory = directory
        assert self._capabilities is not None
        self._start(build_ytdlp_download_arguments(self._capabilities, safe_url, directory))

    def cancel(self) -> None:
        if not self.busy:
            return
        self._cancel_requested = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            self._cancel_kill_timer.start(5_000)

    def _begin(self, url: str, mode: str) -> str | None:
        if self.busy:
            self.failed.emit("Получение видео уже выполняется.")
            return None
        try:
            safe_url = validate_public_video_url(url)
        except Exception as error:
            self.failed.emit(str(error))
            return None
        self.last_failure = None
        capabilities = detect_ytdlp_capabilities()
        if not capabilities.executable:
            self._record_failure(safe_url, None, "unknown", "yt-dlp executable is unavailable")
            self.failed.emit("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")
            return None
        self._mode = mode
        self._url = safe_url
        self._target_directory = None
        self._stdout_chunks = []
        self._stderr_chunks = []
        self._stdout_lines = []
        self._stdout_pending = ""
        self._capabilities = capabilities
        self.last_diagnostics = ""
        self._cancel_requested = False
        self._reported_process_error = False
        self._process_id = 0
        self._release_process_job()
        self.process.setProgram(capabilities.executable)
        self._using_po_token_fallback = False
        self.busy_changed.emit(True)
        return safe_url

    def _start(self, arguments: list[str], *, use_po_token_fallback: bool = False) -> None:
        assert self._capabilities is not None
        # The Qt and non-Qt adapters receive exactly the same child-only
        # environment. BGutil's Deno cache is enabled only for the bounded
        # fallback attempt, never for an ordinary public YouTube request.
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in self._capabilities.process_environment(
            use_po_token_fallback=use_po_token_fallback,
        ).items():
            environment.insert(key, value)
        self.process.setProcessEnvironment(environment)
        self._using_po_token_fallback = use_po_token_fallback
        self.process.setArguments(arguments)
        self.process.start()

    def _read_stdout(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._stdout_chunks.append(text)
        pending = self._stdout_pending + text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            self._consume_stdout_line(line.rstrip("\r"))
        self._stdout_pending = pending

    def _read_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self._stderr_chunks.append(text)

    def _consume_stdout_line(self, line: str) -> None:
        value = line.strip()
        if value:
            self._stdout_lines.append(value)
        if self._mode == "download":
            progress = parse_download_progress(value)
            if progress:
                self.download_progress.emit(progress)

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if self._cancel_requested:
            return
        self._reported_process_error = True
        if _error == QProcess.ProcessError.FailedToStart:
            mode, url, directory = self._mode, self._url, self._target_directory
            self.last_diagnostics = normalize_ytdlp_diagnostics(self.process.errorString())
            if mode == "download" and url:
                self._record_failure(url, None, "unknown", self.last_diagnostics)
            if directory:
                cleanup_partial_downloads(directory)
            self._release_process_job()
            self._mode = self._url = None
            self.busy_changed.emit(False)
            self.failed.emit("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")

    def _state_changed(self, state: QProcess.ProcessState) -> None:
        if state == QProcess.ProcessState.Running:
            self._process_id = int(self.process.processId())
            self._bind_process_tree()
        elif state == QProcess.ProcessState.NotRunning:
            self._process_id = 0

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_stdout()
        self._read_stderr()
        if self._stdout_pending:
            self._consume_stdout_line(self._stdout_pending)
            self._stdout_pending = ""
        stdout = "".join(self._stdout_chunks)
        self.last_diagnostics = normalize_ytdlp_diagnostics("".join(self._stderr_chunks))
        # If yt-dlp exited before one of its helpers, closing this final GUI
        # Job handle stops that helper instead of letting it keep writing to a
        # source directory after the operation has reached a terminal state.
        self._release_process_job()
        mode, url, directory = self._mode, self._url, self._target_directory
        if mode is None:
            return
        self._cancel_kill_timer.stop()
        if self._cancel_requested:
            self._mode = self._url = None
            self.busy_changed.emit(False)
            if directory:
                cleanup_partial_downloads(directory)
            self.cancelled.emit()
            return
        if exit_code != 0 or self._reported_process_error:
            if directory:
                cleanup_partial_downloads(directory)
            failure = classify_ytdlp_failure(self.last_diagnostics or stdout)
            if (
                self._capabilities is not None
                and not self._using_po_token_fallback
                and should_retry_with_po_token_fallback(self._capabilities, failure)
            ):
                self._retry_with_po_token_fallback(mode, url, directory)
                return
            self._mode = self._url = None
            self.busy_changed.emit(False)
            if mode == "download" and url:
                self._record_failure(url, exit_code, failure.reason.value, self.last_diagnostics)
            self.failed.emit(str(failure))
            return
        self._mode = self._url = None
        self.busy_changed.emit(False)
        try:
            if mode == "metadata" and url:
                self.metadata_ready.emit(parse_url_metadata(url, stdout).to_dict())
                return
            if mode == "download" and directory:
                path = next((Path(line).resolve() for line in reversed(self._stdout_lines) if _project_child(Path(line), directory) and Path(line).is_file()), None)
                if path is None:
                    # Direct-media extractors can complete a file without
                    # emitting the requested after_move line. The command has
                    # already exited successfully, so recover only the newest
                    # complete file inside this project's source folder.
                    completed = [
                        item.resolve() for item in directory.iterdir()
                        if item.is_file()
                        and not item.name.endswith((".part", ".ytdl"))
                        and _project_child(item, directory)
                    ]
                    path = max(completed, key=lambda item: item.stat().st_mtime_ns, default=None)
                if path is None:
                    raise ValueError("Загрузка завершилась, но итоговый видеофайл не найден.")
                self.download_completed.emit(str(path))
                return
            raise ValueError("Неподдерживаемое состояние загрузки.")
        except Exception as error:
            self.failed.emit(str(error))

    def _retry_with_po_token_fallback(
        self,
        mode: str,
        url: str | None,
        directory: Path | None,
    ) -> None:
        if self._capabilities is None or url is None:
            return
        self._mode = mode
        self._url = url
        self._target_directory = directory
        self._stdout_chunks = []
        self._stderr_chunks = []
        self._stdout_lines = []
        self._stdout_pending = ""
        self.last_diagnostics = ""
        self._reported_process_error = False
        if mode == "metadata":
            arguments = build_ytdlp_inspect_arguments(
                self._capabilities,
                url,
                use_po_token_fallback=True,
            )
        else:
            assert directory is not None
            arguments = build_ytdlp_download_arguments(
                self._capabilities,
                url,
                directory,
                use_po_token_fallback=True,
            )
        self._start(arguments, use_po_token_fallback=True)

    def _record_failure(
        self, url: str, exit_code: int | None, reason: str, diagnostics: str,
    ) -> None:
        self.last_failure = URLDownloadFailure(
            url=sanitize_public_url_for_diagnostics(url),
            exit_code=exit_code,
            reason=reason,
            last_diagnostics=normalize_ytdlp_diagnostics(diagnostics),
            occurred_at=utc_now(),
        )

    def _kill_if_cancelling(self) -> None:
        if not self._cancel_requested or self.process.state() == QProcess.ProcessState.NotRunning:
            return
        # yt-dlp can launch a media helper. On Windows stop its exact process
        # tree so cancellation never leaves a background download writing into
        # this project while the UI claims it has stopped.
        if self._terminate_process_job():
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
            return
        if sys.platform == "win32" and self._process_id:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self._process_id), "/T", "/F"],
                    capture_output=True, timeout=10, check=False, **UTF8_REPLACE_TEXT,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _bind_process_tree(self) -> None:
        if sys.platform != "win32" or not self._process_id:
            return
        job, _problem = attach_windows_process_job(self._process_id)
        if job is not None:
            self._job_handle = job

    def _terminate_process_job(self) -> bool:
        if self._job_handle is None:
            return False
        try:
            terminated, _error = terminate_windows_process_job(self._job_handle)
            return terminated
        finally:
            self._release_process_job()

    def _release_process_job(self) -> None:
        if self._job_handle is None:
            return
        job, self._job_handle = self._job_handle, None
        close_windows_process_job(job)


def _project_child(path: Path, directory: Path) -> bool:
    try:
        return path.expanduser().resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
