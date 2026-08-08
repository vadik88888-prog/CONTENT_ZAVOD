"""Non-blocking public URL inspection and download for the Qt desktop client."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from app.source_download import (
    cleanup_partial_downloads,
    describe_public_url_failure,
    find_ytdlp_executable,
    parse_download_progress,
    parse_url_metadata,
    validate_public_video_url,
    YTDLP_DOWNLOAD_PROGRESS_TEMPLATE,
)
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.gui.services.windows_process_job import (
    attach_windows_process_job,
    close_windows_process_job,
    terminate_windows_process_job,
)


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
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.stateChanged.connect(self._state_changed)
        self._mode: str | None = None
        self._url: str | None = None
        self._target_directory: Path | None = None
        self._output: list[str] = []
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
        self._start([
            "--no-playlist", "--skip-download", "--no-warnings", "--dump-single-json", safe_url,
        ])

    def download(self, url: str, target_directory: Path) -> None:
        safe_url = self._begin(url, "download")
        if not safe_url:
            return
        directory = target_directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self._target_directory = directory
        output_template = str(directory / "%(title).120B-%(id)s.%(ext)s")
        self._start([
            "--no-playlist", "--newline", "--no-colors", "--no-warnings", "--no-overwrites", "--progress",
            "--progress-template",
            YTDLP_DOWNLOAD_PROGRESS_TEMPLATE,
            "--print", "after_move:filepath", "-o", output_template, safe_url,
        ])

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
        executable = find_ytdlp_executable()
        if not executable:
            self.failed.emit("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")
            return None
        self._mode = mode
        self._url = safe_url
        self._target_directory = None
        self._output = []
        self._cancel_requested = False
        self._reported_process_error = False
        self._process_id = 0
        self._release_process_job()
        self.process.setProgram(executable)
        self.busy_changed.emit(True)
        return safe_url

    def _start(self, arguments: list[str]) -> None:
        self.process.setArguments(arguments)
        self.process.start()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            value = line.strip()
            if value:
                self._output.append(value)
            if self._mode == "download":
                progress = parse_download_progress(value)
                if progress:
                    self.download_progress.emit(progress)

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if self._cancel_requested:
            return
        self._reported_process_error = True
        if _error == QProcess.ProcessError.FailedToStart:
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
        self._read_output()
        # If yt-dlp exited before one of its helpers, closing this final GUI
        # Job handle stops that helper instead of letting it keep writing to a
        # source directory after the operation has reached a terminal state.
        self._release_process_job()
        mode, url, directory = self._mode, self._url, self._target_directory
        if mode is None:
            return
        self._mode = self._url = None
        self._cancel_kill_timer.stop()
        self.busy_changed.emit(False)
        if self._cancel_requested:
            if directory:
                cleanup_partial_downloads(directory)
            self.cancelled.emit()
            return
        if exit_code != 0 or self._reported_process_error:
            if directory:
                cleanup_partial_downloads(directory)
            self.failed.emit(describe_public_url_failure("\n".join(self._output)))
            return
        try:
            if mode == "metadata" and url:
                self.metadata_ready.emit(parse_url_metadata(url, "\n".join(self._output)).to_dict())
                return
            if mode == "download" and directory:
                path = next((Path(line).resolve() for line in reversed(self._output) if _project_child(Path(line), directory) and Path(line).is_file()), None)
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
