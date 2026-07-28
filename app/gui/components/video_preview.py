from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

from PySide6.QtCore import QProcess, QSize, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from app.media import probe_video
from app.utils import safe_name, stable_text_hash


PREVIEW_PROXY_FORMAT_VERSION = "h264-30fps-v2"


@dataclass(frozen=True, slots=True)
class _ProxyRequest:
    token: int
    source_path: Path
    start_seconds: float
    end_seconds: float
    destination: Path


class _BoundedVideoWidget(QVideoWidget):
    """A video surface whose native stream size cannot widen the page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._preview_size = QSize(840, 420)

    def set_preview_size(self, width: int, height: int) -> None:
        self._preview_size = QSize(width, height)
        self.updateGeometry()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        return self._preview_size

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        return QSize(0, 0)


def preview_proxy_path(
    cache_directory: Path, source_path: Path, start_seconds: float, end_seconds: float,
) -> Path:
    """Return an immutable cache path for one source interval preview."""

    try:
        stat = source_path.stat()
        revision = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        revision = str(source_path)
    digest = stable_text_hash(
        f"{PREVIEW_PROXY_FORMAT_VERSION}:{revision}:{start_seconds:.3f}:{end_seconds:.3f}"
    )[:20]
    return cache_directory / f"{safe_name(source_path.stem, 'source')}-{digest}.mp4"


class VideoPreview(QFrame):
    """Bounded candidate preview with a compatible local-proxy fallback."""

    preview_error = Signal(str)
    preview_ready = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preview")
        self.setMinimumSize(0, 0)
        self._path: Path | None = None
        self._source_path: Path | None = None
        self._source_range_seconds: tuple[float, float] | None = None
        self._active_candidate_title: str | None = None
        self._range_start_ms: int | None = None
        self._range_end_ms: int | None = None
        self._range_autoplay = False
        self._range_media_ready = False
        self._using_proxy = False
        self._presentation = "source"
        self._selection_token = 0
        self._proxy_cache_directory = Path(tempfile.gettempdir()) / "content-factory-preview-proxies"
        self._support_cache: dict[str, bool] = {}
        self._active_proxy: _ProxyRequest | None = None
        self._pending_proxy: _ProxyRequest | None = None
        self._proxy_process = QProcess(self)
        self._proxy_process.finished.connect(self._proxy_finished)
        self._proxy_process.errorOccurred.connect(self._proxy_process_error)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        # QVideoWidget changes its native size hint after a stream is loaded.
        # Without a bound, a 1080/1440p source can make the whole review page
        # several screens wide even though the visible player is small.
        self.video = _BoundedVideoWidget(self)
        self.player.setVideoOutput(self.video)
        self.player.errorOccurred.connect(self._media_error)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.positionChanged.connect(self._position_changed)

        layout = QVBoxLayout(self)
        # Do not let a loaded stream's native frame size become a hard
        # minimum for the whole page.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(12, 12, 12, 12)
        self.placeholder = QLabel("Выберите видео, чтобы увидеть предпросмотр")
        self.placeholder.setObjectName("muted")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.media_stage = QWidget(self)
        self.media_stage_layout = QHBoxLayout(self.media_stage)
        self.media_stage_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.media_stage_layout.setContentsMargins(0, 0, 0, 0)
        self.media_stage_layout.addStretch()
        self.media_stage_layout.addWidget(self.video)
        self.media_stage_layout.addWidget(self.placeholder)
        self.media_stage_layout.addStretch()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.active_candidate = QLabel()
        self.active_candidate.setObjectName("active-candidate")
        self.active_candidate.setWordWrap(True)
        self.active_candidate.setStyleSheet("font-weight: 600;")
        self.active_candidate.hide()
        layout.addWidget(self.active_candidate)
        layout.addWidget(self.media_stage, 0, Qt.AlignmentFlag.AlignHCenter)
        self.preview_status = QLabel()
        self.preview_status.setObjectName("muted")
        self.preview_status.setWordWrap(True)
        self.preview_status.hide()
        layout.addWidget(self.preview_status)
        buttons = QHBoxLayout()
        self.play_button = QPushButton("Воспроизвести")
        self.open_button = QPushButton("Открыть в проигрывателе")
        self.play_button.clicked.connect(self._play)
        self.open_button.clicked.connect(self.open_externally)
        buttons.addWidget(self.play_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self._set_available(False)
        self._set_presentation("source")

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def source_range_seconds(self) -> tuple[float, float] | None:
        """The original candidate range, even when the player uses a proxy."""

        return self._source_range_seconds

    @property
    def using_proxy(self) -> bool:
        return self._using_proxy

    @property
    def active_media_path(self) -> Path | None:
        return self._path

    @property
    def active_candidate_title(self) -> str | None:
        return self._active_candidate_title

    @property
    def presentation(self) -> str:
        """The visual framing of the current media: source or vertical."""

        return self._presentation

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        if self._presentation == "vertical":
            return QSize(500, 600)
        return QSize(864, 520)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        # QVideoWidget's native frame size can change after media loads.  The
        # outer player must still be allowed to fit a normal desktop window.
        return QSize(0, 0)

    def show_source(self, path: str | Path | None) -> None:
        """Show the original landscape source in a normal wide player."""

        self.set_file(path, presentation="source", title="Исходное видео")

    def show_draft(self, path: str | Path | None, candidate_title: str | None = None) -> None:
        """Show a draft in a phone-sized 9:16 player."""

        suffix = f" · {candidate_title}" if candidate_title else ""
        self.set_file(path, presentation="vertical", title=f"Черновик{suffix}")

    def show_final(self, path: str | Path | None, candidate_title: str | None = None) -> None:
        """Show a completed short in a phone-sized 9:16 player."""

        suffix = f" · {candidate_title}" if candidate_title else ""
        self.set_file(path, presentation="vertical", title=f"Готовый ролик{suffix}")

    def set_file(
        self, path: str | Path | None, *, presentation: str = "auto", title: str | None = None,
    ) -> None:
        self._selection_token += 1
        self._cancel_proxy()
        self._source_range_seconds = None
        self._active_candidate_title = title
        self._range_start_ms = None
        self._range_end_ms = None
        self._range_autoplay = False
        self._range_media_ready = False
        self._using_proxy = False
        candidate = Path(path) if path else None
        self._set_presentation(self._file_presentation(candidate, presentation))
        if title:
            self.active_candidate.setText(title)
            self.active_candidate.show()
        else:
            self.active_candidate.clear()
            self.active_candidate.hide()
        self._source_path = candidate if self.usable_media_path(candidate) else None
        self._path = self._source_path
        self._clear_status()
        if self._path:
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(str(self._path)))
            self.placeholder.hide()
            self.video.show()
        else:
            self._clear_media()
        self._set_available(self._path is not None)

    def set_range(
        self,
        path: str | Path,
        start_seconds: float,
        end_seconds: float,
        *,
        autoplay: bool = True,
        cache_directory: Path | None = None,
        candidate_title: str | None = None,
    ) -> None:
        """Bind the player to one candidate's source interval.

        AV1/WebM candidate intervals are played from a small H.264/AAC proxy.
        This keeps seeking reliable on Windows without starting a production
        render or converting the original source.
        """

        start = max(0.0, float(start_seconds))
        end = max(start, float(end_seconds))
        self._selection_token += 1
        self._cancel_proxy()
        self._source_range_seconds = (start, end)
        self._source_path = Path(path)
        self._set_presentation("source")
        self._active_candidate_title = candidate_title or "Выбранный кандидат"
        self.active_candidate.setText(
            f"Кандидат: {self._active_candidate_title}\nФрагмент: {start:.1f}–{end:.1f} с"
        )
        self.active_candidate.show()
        self._range_autoplay = autoplay
        self._range_media_ready = False
        self._clear_status()
        if not self.usable_media_path(self._source_path):
            self._show_error("Не удалось открыть исходный файл для предпросмотра.")
            self._clear_media()
            return
        cache = cache_directory or (Path(tempfile.gettempdir()) / "content-factory-preview-proxies")
        self._proxy_cache_directory = cache
        if not self._qt_can_decode_source(self._source_path):
            self._request_proxy(cache, "Готовим совместимый H.264/AAC предпросмотр для этого формата.")
            return
        self._activate_direct_source()

    def _activate_direct_source(self) -> None:
        assert self._source_path is not None and self._source_range_seconds is not None
        start, end = self._source_range_seconds
        self._path = self._source_path
        self._using_proxy = False
        self._range_start_ms = int(round(start * 1000))
        self._range_end_ms = int(round(end * 1000))
        self._range_media_ready = False
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(self._path)))
        self.placeholder.hide()
        self.video.show()
        self._set_available(True)
        self._show_status("Загружаем исходный фрагмент…")

    def _file_presentation(self, candidate: Path | None, presentation: str) -> str:
        if presentation in {"source", "vertical"}:
            return presentation
        if candidate and self.usable_media_path(candidate):
            try:
                metadata = probe_video(candidate)
                if float(metadata.get("height") or 0) > float(metadata.get("width") or 0):
                    return "vertical"
            except Exception:
                pass
        return "source"

    def _set_presentation(self, presentation: str) -> None:
        """Keep short-form outputs recognisable as phone-video previews."""

        self._presentation = "vertical" if presentation == "vertical" else "source"
        vertical = self._presentation == "vertical"
        if vertical:
            width, height = 270, 480
            self.setMaximumWidth(500)
            self.video.set_preview_size(width, height)
            self.video.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.placeholder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.video.setFixedSize(width, height)
            self.placeholder.setFixedSize(width, height)
            self.media_stage.setFixedSize(width, height)
            self.media_stage.setObjectName("phoneStage")
        else:
            self.setMaximumWidth(864)
            self.video.set_preview_size(840, 420)
            for widget in (self.video, self.placeholder):
                widget.setMinimumSize(0, 260)
                widget.setMaximumSize(840, 420)
                widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.media_stage.setMinimumSize(0, 260)
            self.media_stage.setMaximumSize(840, 420)
            self.media_stage.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.media_stage.setObjectName("")
        self.media_stage.style().unpolish(self.media_stage)
        self.media_stage.style().polish(self.media_stage)

    def _qt_can_decode_source(self, source_path: Path) -> bool:
        """Return whether a candidate interval is safe for direct Qt playback."""

        try:
            stat = source_path.stat()
            key = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            key = str(source_path)
        cached = self._support_cache.get(key)
        if cached is not None:
            return cached
        supported = True
        try:
            metadata = probe_video(source_path)
            if str(metadata.get("video_codec") or "").lower() == "av1" and source_path.suffix.lower() == ".webm":
                # Media Foundation can report AV1/WebM as supported yet show a
                # black frame after a later range seek. Candidate previews
                # need predictable seeking, so use the existing short proxy.
                supported = False
        except Exception:
            # Probe failure is not a reason to reject a potentially playable
            # file; the QMediaPlayer error path still has the proxy fallback.
            supported = True
        self._support_cache[key] = supported
        return supported

    def _request_proxy(self, cache_directory: Path, reason: str) -> None:
        assert self._source_path is not None and self._source_range_seconds is not None
        start, end = self._source_range_seconds
        request = _ProxyRequest(
            token=self._selection_token,
            source_path=self._source_path,
            start_seconds=start,
            end_seconds=end,
            destination=preview_proxy_path(cache_directory, self._source_path, start, end),
        )
        self.player.stop()
        self._path = None
        self._using_proxy = True
        self._set_available(False)
        self.placeholder.setText("Подготавливаем совместимый предпросмотр…")
        self.placeholder.show()
        self.video.hide()
        self._show_status(reason)
        if self.usable_media_path(request.destination):
            self._cancel_proxy()
            self._activate_proxy(request)
            return
        if self._active_proxy is not None:
            self._pending_proxy = request
            if self._proxy_process.state() != QProcess.ProcessState.NotRunning:
                self._proxy_process.kill()
            return
        self._start_proxy(request)

    def _start_proxy(self, request: _ProxyRequest) -> None:
        executable = shutil.which("ffmpeg")
        if not executable:
            self._show_error("Не найден FFmpeg: совместимый предпросмотр создать нельзя.")
            return
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        self._active_proxy = request
        duration = max(0.05, request.end_seconds - request.start_seconds)
        self._proxy_process.setProgram(executable)
        self._proxy_process.setArguments([
            "-y", "-hide_banner", "-loglevel", "error",
            # Input seek avoids decoding the full source before a late PUBG
            # candidate.  Transcoding keeps FFmpeg's accurate-seek discard.
            "-ss", f"{request.start_seconds:.3f}", "-i", str(request.source_path), "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", "scale=-2:480,fps=30,setsar=1", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(request.destination),
        ])
        self._proxy_process.start()

    def _proxy_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        request = self._active_proxy
        if request is None:
            return
        success = exit_code == 0 and self.usable_media_path(request.destination)
        self._complete_proxy(request, success, self._ffmpeg_error_text())

    def _proxy_process_error(self, _error: QProcess.ProcessError) -> None:
        request = self._active_proxy
        if request is not None:
            QTimer.singleShot(0, lambda req=request: self._complete_proxy_error(req))

    def _complete_proxy_error(self, request: _ProxyRequest) -> None:
        if self._active_proxy is request and self._proxy_process.state() == QProcess.ProcessState.NotRunning:
            self._complete_proxy(request, False, self._ffmpeg_error_text() or "Не удалось запустить FFmpeg.")

    def _complete_proxy(self, request: _ProxyRequest, success: bool, details: str) -> None:
        if self._active_proxy is not request:
            return
        self._active_proxy = None
        pending = self._pending_proxy
        self._pending_proxy = None
        if pending is not None:
            self._start_proxy(pending)
            return
        if request.token != self._selection_token:
            return
        if success:
            self._activate_proxy(request)
            return
        self._show_error(
            "Предпросмотр недоступен: не удалось создать совместимую H.264/AAC копию. "
            + (details[-300:] if details else "")
        )

    def _activate_proxy(self, request: _ProxyRequest) -> None:
        if request.token != self._selection_token:
            return
        self._path = request.destination
        self._using_proxy = True
        self._range_start_ms = 0
        self._range_end_ms = int(round((request.end_seconds - request.start_seconds) * 1000))
        self._range_media_ready = False
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(self._path)))
        self.placeholder.hide()
        self.video.show()
        self._set_available(True)
        self._show_status(
            f"Совместимый preview готов: {request.start_seconds:.1f}–{request.end_seconds:.1f} с исходного видео."
        )
        self.preview_ready.emit(str(request.destination))

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if self._path is None:
            return
        if (
            status == QMediaPlayer.MediaStatus.EndOfMedia
            and self._range_media_ready
            and self._range_end_ms is not None
        ):
            # Windows Media Foundation can reset position to zero after the
            # terminal frame.  Keep the stopped candidate visibly at its end.
            self._stop_at_range_end()
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            if self._source_path and self._source_range_seconds and not self._using_proxy:
                self._request_proxy(self._proxy_cache_directory, "Qt Multimedia не смог открыть исходный формат; создаём совместимый preview.")
            else:
                self._show_error("Qt Multimedia не смог открыть подготовленный предпросмотр.")
            return
        if self._range_start_ms is None:
            return
        if status in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia}:
            if self._range_media_ready:
                return
            # Media Foundation updates QVideoWidget's native size after the
            # source is loaded.  Reapply the bounded presentation afterwards
            # so a 1440p landscape source cannot widen the review page.
            self._set_presentation(self._presentation)
            self._range_media_ready = True
            if self._range_start_ms > 0:
                self.player.setPosition(self._range_start_ms)
            if self._range_autoplay:
                self._range_autoplay = False
                self._start_range_playback()

    def _position_changed(self, position: int) -> None:
        if not self._range_media_ready or self._range_end_ms is None:
            return
        if self._using_proxy:
            duration = self.player.duration()
            if duration > 0 and position > duration:
                # A queued position notification may still belong to the
                # previous source. It cannot be valid for this short proxy.
                return
        if position >= self._range_end_ms:
            self._stop_at_range_end()

    def _stop_at_range_end(self) -> None:
        if self._range_end_ms is None:
            return
        self.player.pause()
        # Seeking to an exact file duration is normalized to zero by Windows
        # Media Foundation.  The final millisecond is the same visible end
        # frame while remaining a stable, paused position.
        last_frame = max(self._range_start_ms or 0, self._range_end_ms - 1)
        QTimer.singleShot(0, lambda position=last_frame: self.player.setPosition(position))
        self._show_status("Просмотр завершён на конце выбранного фрагмента.")

    def open_externally(self) -> None:
        target = self._source_path or self._path
        if target and target.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _play(self) -> None:
        if not self._path:
            if self._active_proxy is not None:
                self._show_status("Предпросмотр ещё подготавливается.")
            return
        if self._range_start_ms is None:
            self.player.play()
            return
        self._start_range_playback()

    def _start_range_playback(self) -> None:
        if self._range_start_ms is None:
            return
        if self._range_start_ms > 0:
            self.player.setPosition(self._range_start_ms)
        QTimer.singleShot(0, self.player.play)
        self._show_status("Воспроизведение выбранного фрагмента…")

    def _media_error(self, *_: object) -> None:
        if self._source_path and self._source_range_seconds and not self._using_proxy:
            self._request_proxy(self._proxy_cache_directory, "Qt Multimedia не поддержал исходный формат; создаём совместимый preview.")
            return
        details = self.player.errorString().strip()
        self._show_error("Встроенное воспроизведение недоступно." + (f" {details}" if details else ""))

    def _cancel_proxy(self) -> None:
        self._pending_proxy = None
        if self._active_proxy is not None and self._proxy_process.state() != QProcess.ProcessState.NotRunning:
            self._proxy_process.kill()

    def _clear_media(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self._path = None
        self._range_media_ready = False
        self.video.hide()
        self.placeholder.show()
        self._set_available(False)

    def _show_status(self, message: str) -> None:
        self.preview_status.setStyleSheet("")
        self.preview_status.setText(message)
        self.preview_status.show()

    def _clear_status(self) -> None:
        self.preview_status.clear()
        self.preview_status.hide()

    def _show_error(self, message: str) -> None:
        self.player.stop()
        self._path = None
        self._range_media_ready = False
        self.placeholder.setText(message)
        self.placeholder.show()
        self.video.hide()
        self.preview_status.setStyleSheet("color: #d66;")
        self.preview_status.setText(message)
        self.preview_status.show()
        self._set_available(False)
        self.preview_error.emit(message)

    def _ffmpeg_error_text(self) -> str:
        return bytes(self._proxy_process.readAllStandardError()).decode("utf-8", errors="replace").strip()

    def _set_available(self, value: bool) -> None:
        self.play_button.setEnabled(value)
        self.open_button.setEnabled(bool(self._source_path and self._source_path.is_file()))

    @staticmethod
    def usable_media_path(path: Path | None) -> bool:
        return bool(path and path.is_file() and path.stat().st_size > 0)
