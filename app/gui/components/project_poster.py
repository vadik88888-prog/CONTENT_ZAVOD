from __future__ import annotations

"""Durable high-quality posters for project-level desktop surfaces.

Project posters are intentionally separate from the small candidate thumbnail
queue. They prefer YouTube's real artwork, fall back to a source frame, and
write one identity-bound image into the owning project's cache.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid
from typing import Any, Mapping
from urllib.parse import urlparse

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from app.gui.models import DesktopProject
from app.utils import safe_name, stable_text_hash


PROJECT_POSTER_PROFILE_ID = "project-poster-v2-1280"
PROJECT_POSTER_MAX_EDGE = 1280
PROJECT_POSTER_FILTER = (
    f"scale={PROJECT_POSTER_MAX_EDGE}:{PROJECT_POSTER_MAX_EDGE}:"
    "force_original_aspect_ratio=decrease:force_divisible_by=2"
)


@dataclass(frozen=True, slots=True)
class ProjectPosterRequest:
    project_id: str
    source_path: Path | None
    thumbnail_url: str | None
    timestamp_seconds: float
    destination: Path
    temporary_path: Path


def youtube_thumbnail_url(metadata: Mapping[str, Any]) -> str | None:
    """Return the inspected YouTube artwork URL when it is safe to fetch."""

    extractor = str(metadata.get("extractor") or "").casefold()
    if "youtube" not in extractor:
        return None
    value = str(metadata.get("thumbnail_url") or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    if parsed.scheme != "https" or not hostname:
        return None
    if not (
        hostname == "ytimg.com"
        or hostname.endswith(".ytimg.com")
        or hostname == "img.youtube.com"
    ):
        return None
    return parsed.geturl()


def project_youtube_thumbnail_url(project: DesktopProject) -> str | None:
    """Read URL metadata without letting local probe fields erase artwork."""

    metadata = dict(project.source_spec.metadata)
    metadata.update(project.source_metadata)
    return youtube_thumbnail_url(metadata)


def project_poster_has_input(project: DesktopProject) -> bool:
    return bool(
        project_youtube_thumbnail_url(project)
        or (project.source_spec.is_ready and project.source.is_file())
    )


def project_poster_path(project: DesktopProject, *, timestamp_seconds: float = 1.0) -> Path:
    """Return a project/source-revision-bound cache path for the HQ profile."""

    thumbnail_url = project_youtube_thumbnail_url(project)
    if thumbnail_url and project.source_spec.original_url:
        source_revision = f"url:{project.source_spec.original_url}"
    elif project.source_spec.is_ready:
        source = project.source
        try:
            stat = source.stat()
            source_revision = f"file:{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            source_revision = f"file:{source}"
    else:
        source_revision = f"url:{project.source_spec.original_url or ''}"
    identity = ":".join((
        PROJECT_POSTER_PROFILE_ID,
        project.project_id,
        source_revision,
        thumbnail_url or "generated-source-frame",
        f"{max(0.0, timestamp_seconds):.3f}",
    ))
    digest = stable_text_hash(identity)[:16]
    return (
        project.directory
        / "thumbnails"
        / PROJECT_POSTER_PROFILE_ID
        / f"{safe_name(project.project_id, 'project')}-{digest}.jpg"
    )


def project_poster_ffmpeg_arguments(
    request: ProjectPosterRequest, *, use_thumbnail_url: bool,
) -> list[str]:
    """Compile fixed, non-shell FFmpeg arguments for the project profile."""

    arguments = ["-y", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    if use_thumbnail_url:
        if not request.thumbnail_url:
            raise ValueError("YouTube thumbnail URL is unavailable.")
        arguments.extend([
            "-protocol_whitelist", "http,https,tcp,tls",
            "-i", request.thumbnail_url,
        ])
    else:
        if request.source_path is None:
            raise ValueError("Project source is unavailable.")
        arguments.extend([
            "-ss", f"{request.timestamp_seconds:.3f}",
            "-i", str(request.source_path),
        ])
    arguments.extend([
        "-frames:v", "1",
        "-vf", PROJECT_POSTER_FILTER,
        "-q:v", "2",
        str(request.temporary_path),
    ])
    return arguments


class ProjectPosterLoader(QObject):
    """Serial, asynchronous loader for durable project-level posters."""

    poster_ready = Signal(str, str)
    poster_unavailable = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: deque[ProjectPosterRequest] = deque()
        self._requested: set[Path] = set()
        self._active: ProjectPosterRequest | None = None
        self._active_uses_thumbnail = False
        self._active_cancelled = False
        self._process = QProcess(self)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._timed_out)

    def replace_pending(self) -> None:
        """Drop poster work owned by a stale screen projection."""

        self._queue.clear()
        if self._active is None:
            self._requested.clear()
            return
        self._requested = {self._request_key(self._active.destination)}
        self._active_cancelled = True
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def request(self, project: DesktopProject, *, timestamp_seconds: float = 1.0) -> Path:
        destination = project_poster_path(project, timestamp_seconds=timestamp_seconds)
        if destination.is_file() and destination.stat().st_size > 0:
            QTimer.singleShot(
                0,
                lambda pid=project.project_id, path=destination: self.poster_ready.emit(pid, str(path)),
            )
            return destination

        thumbnail_url = project_youtube_thumbnail_url(project)
        source_path = (
            project.source.resolve()
            if project.source_spec.is_ready and project.source.is_file()
            else None
        )
        if thumbnail_url is None and source_path is None:
            QTimer.singleShot(
                0,
                lambda pid=project.project_id, path=destination: self.poster_unavailable.emit(pid, str(path)),
            )
            return destination

        request_key = self._request_key(destination)
        if request_key not in self._requested:
            self._requested.add(request_key)
            self._queue.append(ProjectPosterRequest(
                project_id=project.project_id,
                source_path=source_path,
                thumbnail_url=thumbnail_url,
                timestamp_seconds=max(0.0, timestamp_seconds),
                destination=destination,
                temporary_path=_temporary_path(destination, uuid.uuid4().hex),
            ))
            self._start_next()
        return destination

    @staticmethod
    def _request_key(destination: Path) -> Path:
        return destination.resolve(strict=False)

    def _start_next(self) -> None:
        if self._active is not None or not self._queue:
            return
        self._active = self._queue.popleft()
        self._active_uses_thumbnail = bool(self._active.thumbnail_url)
        self._active_cancelled = False
        self._start_active_attempt()

    def _start_active_attempt(self) -> None:
        request = self._active
        if request is None:
            return
        executable = shutil.which("ffmpeg")
        if not executable:
            QTimer.singleShot(0, lambda: self._complete_active(False))
            return
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.temporary_path.unlink(missing_ok=True)
        self._process.setProgram(executable)
        self._process.setArguments(project_poster_ffmpeg_arguments(
            request, use_thumbnail_url=self._active_uses_thumbnail,
        ))
        self._timeout.start(30_000)
        self._process.start()

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._finish_attempt(exit_code == 0)

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        request = self._active
        if request is not None:
            QTimer.singleShot(0, lambda req=request: self._complete_process_error(req))

    def _complete_process_error(self, request: ProjectPosterRequest) -> None:
        if self._active is request and self._process.state() == QProcess.ProcessState.NotRunning:
            self._finish_attempt(False)

    def _timed_out(self) -> None:
        if self._active is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def _finish_attempt(self, process_succeeded: bool) -> None:
        request = self._active
        if request is None:
            return
        self._timeout.stop()
        temporary = request.temporary_path
        temporary_ready = bool(
            process_succeeded and temporary.is_file() and temporary.stat().st_size > 0
        )
        if temporary_ready:
            try:
                temporary.replace(request.destination)
            except OSError:
                pass
        destination_ready = bool(
            request.destination.is_file() and request.destination.stat().st_size > 0
        )
        if destination_ready:
            temporary.unlink(missing_ok=True)
        if (
            not destination_ready
            and not self._active_cancelled
            and self._active_uses_thumbnail
            and request.source_path is not None
        ):
            temporary.unlink(missing_ok=True)
            self._active_uses_thumbnail = False
            self._start_active_attempt()
            return
        self._complete_active(destination_ready)

    def _complete_active(self, success: bool) -> None:
        request = self._active
        if request is None:
            return
        self._timeout.stop()
        self._active = None
        self._requested.discard(self._request_key(request.destination))
        if not success:
            request.temporary_path.unlink(missing_ok=True)
        if success:
            self.poster_ready.emit(request.project_id, str(request.destination))
        else:
            self.poster_unavailable.emit(request.project_id, str(request.destination))
        self._start_next()


def _temporary_path(destination: Path, token: str) -> Path:
    return destination.with_name(f".{destination.stem}.{token}.tmp{destination.suffix}")
