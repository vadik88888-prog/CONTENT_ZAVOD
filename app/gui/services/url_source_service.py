"""Non-blocking public URL inspection and download for the Qt desktop client."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from app.source_download import (
    cleanup_partial_downloads,
    parse_download_progress,
    parse_url_metadata,
    validate_public_video_url,
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
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._mode: str | None = None
        self._url: str | None = None
        self._target_directory: Path | None = None
        self._output: list[str] = []
        self._cancel_requested = False
        self._reported_process_error = False

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
            "--no-playlist", "--newline", "--no-warnings", "--no-overwrites",
            "--progress-template", "download:%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
            "--print", "after_move:filepath", "-o", output_template, safe_url,
        ])

    def cancel(self) -> None:
        if not self.busy:
            return
        self._cancel_requested = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _begin(self, url: str, mode: str) -> str | None:
        if self.busy:
            self.failed.emit("Получение видео уже выполняется.")
            return None
        try:
            safe_url = validate_public_video_url(url)
        except Exception as error:
            self.failed.emit(str(error))
            return None
        executable = shutil.which("yt-dlp")
        if not executable:
            self.failed.emit("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")
            return None
        self._mode = mode
        self._url = safe_url
        self._target_directory = None
        self._output = []
        self._cancel_requested = False
        self._reported_process_error = False
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
            self._mode = self._url = None
            self.busy_changed.emit(False)
            self.failed.emit("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        mode, url, directory = self._mode, self._url, self._target_directory
        if mode is None:
            return
        self._mode = self._url = None
        self.busy_changed.emit(False)
        if self._cancel_requested:
            if directory:
                cleanup_partial_downloads(directory)
            self.cancelled.emit()
            return
        if exit_code != 0 or self._reported_process_error:
            if directory:
                cleanup_partial_downloads(directory)
            self.failed.emit("Не удалось получить видео по этой ссылке.")
            return
        try:
            if mode == "metadata" and url:
                self.metadata_ready.emit(parse_url_metadata(url, "\n".join(self._output)).to_dict())
                return
            if mode == "download" and directory:
                path = next((Path(line).resolve() for line in reversed(self._output) if _project_child(Path(line), directory) and Path(line).is_file()), None)
                if path is None:
                    raise ValueError("Загрузка завершилась, но итоговый видеофайл не найден.")
                self.download_completed.emit(str(path))
                return
            raise ValueError("Неподдерживаемое состояние загрузки.")
        except Exception as error:
            self.failed.emit(str(error))


def _project_child(path: Path, directory: Path) -> bool:
    try:
        return path.expanduser().resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
