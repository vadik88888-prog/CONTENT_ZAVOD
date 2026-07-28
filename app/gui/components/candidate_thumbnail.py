from __future__ import annotations

"""Small, lazy source-frame thumbnails for the native candidate workspace.

They are deliberately separate from draft and production rendering.  A single
FFmpeg process creates one lightweight JPEG at a time so opening a long review
list never launches a batch of heavy encodes or blocks the UI thread.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import shutil

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from app.utils import safe_name, stable_text_hash


@dataclass(frozen=True, slots=True)
class CandidateThumbnailRequest:
    candidate_id: str
    source_path: Path
    timestamp_seconds: float
    destination: Path


def thumbnail_path(
    cache_directory: Path, analysis_id: str, candidate_id: str, source_path: Path, timestamp_seconds: float,
) -> Path:
    """Return a cache path tied to both the source revision and analysis."""

    try:
        stat = source_path.stat()
        source_revision = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        source_revision = str(source_path)
    digest = stable_text_hash(f"{analysis_id}:{candidate_id}:{timestamp_seconds:.3f}:{source_revision}")[:16]
    return cache_directory / safe_name(analysis_id, "analysis") / f"{safe_name(candidate_id, 'candidate')}-{digest}.jpg"


class CandidateThumbnailLoader(QObject):
    """Queue frame extraction and expose ready files to candidate cards."""

    thumbnail_ready = Signal(str, str)
    thumbnail_unavailable = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: deque[CandidateThumbnailRequest] = deque()
        self._requested: set[str] = set()
        self._active: CandidateThumbnailRequest | None = None
        self._process = QProcess(self)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)

    def request(
        self, *, cache_directory: Path, analysis_id: str, candidate_id: str,
        source_path: Path, timestamp_seconds: float,
    ) -> Path:
        destination = thumbnail_path(cache_directory, analysis_id, candidate_id, source_path, timestamp_seconds)
        if destination.is_file() and destination.stat().st_size > 0:
            QTimer.singleShot(0, lambda cid=candidate_id, path=destination: self.thumbnail_ready.emit(cid, str(path)))
            return destination
        if candidate_id not in self._requested:
            self._requested.add(candidate_id)
            self._queue.append(CandidateThumbnailRequest(
                candidate_id=candidate_id,
                source_path=source_path,
                timestamp_seconds=max(0.0, timestamp_seconds),
                destination=destination,
            ))
            self._start_next()
        return destination

    def _start_next(self) -> None:
        if self._active is not None or not self._queue:
            return
        self._active = self._queue.popleft()
        executable = shutil.which("ffmpeg")
        if not executable:
            self._complete_active(False)
            return
        request = self._active
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        self._process.setProgram(executable)
        self._process.setArguments([
            "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{request.timestamp_seconds:.3f}", "-i", str(request.source_path),
            "-frames:v", "1", "-vf", "scale=240:-2", "-q:v", "5", str(request.destination),
        ])
        self._process.start()

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        request = self._active
        self._complete_active(bool(request and exit_code == 0 and request.destination.is_file() and request.destination.stat().st_size > 0))

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        request = self._active
        if request is None:
            return
        # Let a same-process ``finished`` signal win first; otherwise it could
        # arrive after the next queue item starts and complete that new item.
        QTimer.singleShot(0, lambda: self._complete_failed_request(request))

    def _complete_failed_request(self, request: CandidateThumbnailRequest) -> None:
        if self._active is request and self._process.state() == QProcess.ProcessState.NotRunning:
            self._complete_active(False)

    def _complete_active(self, success: bool) -> None:
        request = self._active
        if request is None:
            return
        self._active = None
        if success:
            self.thumbnail_ready.emit(request.candidate_id, str(request.destination))
        else:
            self.thumbnail_unavailable.emit(request.candidate_id)
        self._start_next()
