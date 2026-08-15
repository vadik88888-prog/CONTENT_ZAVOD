from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import tempfile

from PySide6.QtCore import QEvent, QProcess, QSignalBlocker, QSize, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLayout, QLabel, QPushButton, QSlider,
    QSizePolicy, QVBoxLayout, QWidget,
)

from app.gui.responsive import make_label_shrinkable, set_responsive_text
from app.utils import safe_name, stable_text_hash


# v3 changes proxy writes to an atomic temporary path.  Keeping the cache key
# versioned ensures that a non-empty partial v2 file is never treated as a
# usable preview after an interrupted FFmpeg process.
PREVIEW_PROXY_FORMAT_VERSION = "h264-30fps-v3"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ProxyRequest:
    token: int
    source_path: Path
    start_seconds: float
    end_seconds: float
    destination: Path
    temporary: Path


@dataclass(frozen=True, slots=True)
class _PosterRequest:
    token: int
    source_path: Path
    destination: Path
    timestamp_seconds: float = 0.05


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


def preview_proxy_temporary_path(destination: Path) -> Path:
    """Return the same-directory FFmpeg temporary output for one proxy.

    FFmpeg infers the muxer from the final suffix, therefore ``.part`` must
    precede ``.mp4`` rather than replace it.  The output is promoted only when
    FFmpeg exits successfully, so a killed preview process cannot poison the
    durable cache with a non-empty truncated MP4.
    """

    return destination.with_name(f"{destination.stem}.part{destination.suffix}")


def preview_poster_path(
    cache_directory: Path, source_path: Path, timestamp_seconds: float = 0.05,
) -> Path:
    """Return a source-revision-and-time-bound poster image path."""

    try:
        stat = source_path.stat()
        revision = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        revision = str(source_path)
    digest = stable_text_hash(
        f"preview-poster-v2:{revision}:{max(0.0, timestamp_seconds):.3f}"
    )[:20]
    return cache_directory / f"{safe_name(source_path.stem, 'video')}-{digest}.jpg"


class VideoPreview(QFrame):
    """Bounded candidate preview with a compatible local-proxy fallback."""

    MEDIA_LOAD_TIMEOUT_MS = 15_000
    PROXY_RENDER_TIMEOUT_MS = 120_000

    preview_error = Signal(str)
    geometry_requirement_changed = Signal()
    preview_ready = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preview")
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._path: Path | None = None
        self._source_path: Path | None = None
        self._source_range_seconds: tuple[float, float] | None = None
        # Source codec comes from the one-time project ffprobe metadata.  It
        # lets the review click avoid waking Qt Multimedia's D3D11 decoder for
        # AV1 sources which cannot be decoded in hardware on this Windows host.
        self._source_codec: str | None = None
        self._force_compatible_proxy: bool | None = None
        self._active_candidate_title: str | None = None
        self._range_start_ms: int | None = None
        self._range_end_ms: int | None = None
        self._range_autoplay = False
        self._range_media_ready = False
        # QMediaPlayer callbacks only contain a source URL.  Two candidate
        # ranges from the same source therefore need an additional binding
        # token and a confirmed seek position before they may update the UI.
        self._range_ready_token: int | None = None
        self._range_seek_pending_token: int | None = None
        self._range_seek_target_ms: int | None = None
        # A freshly loaded source naturally starts at frame zero.  Retargeting
        # an already-loaded source does not, so only the latter needs an
        # explicit zero seek when the next candidate begins at 0:00.
        self._range_requires_seek = False
        self._media_ready = False
        self._using_proxy = False
        self._presentation = "source"
        self._vertical_frame_size = (270, 480)
        self._selection_token = 0
        self._expected_source = QUrl()
        self._media_loading = False
        self._media_load_token: int | None = None
        self._last_audible_volume = 100
        self._proxy_cache_directory = Path(tempfile.gettempdir()) / "content-factory-preview-proxies"
        self._active_proxy: _ProxyRequest | None = None
        self._pending_proxy: _ProxyRequest | None = None
        self._poster_cache_directory = Path(tempfile.gettempdir()) / "content-factory-preview-posters"
        self._active_poster: _PosterRequest | None = None
        self._pending_poster: _PosterRequest | None = None
        self._proxy_process = QProcess(self)
        self._proxy_process.finished.connect(self._proxy_finished)
        self._proxy_process.errorOccurred.connect(self._proxy_process_error)
        self._proxy_timeout_timer = QTimer(self)
        self._proxy_timeout_timer.setSingleShot(True)
        self._proxy_timeout_timer.timeout.connect(self._proxy_timed_out)
        self._proxy_timed_out_request: _ProxyRequest | None = None
        self._poster_process = QProcess(self)
        self._poster_process.finished.connect(self._poster_finished)
        self._poster_process.errorOccurred.connect(self._poster_process_error)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self._frame_sink_output = False
        self._frame_sink = QVideoSink(self)
        self._frame_sink.videoFrameChanged.connect(self._software_video_frame_changed)
        # QVideoWidget changes its native size hint after a stream is loaded.
        # Without a bound, a 1080/1440p source can make the whole review page
        # several screens wide even though the visible player is small.
        self.video = _BoundedVideoWidget(self)
        self.player.setVideoOutput(self.video)
        self.player.errorOccurred.connect(self._media_error)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.video.installEventFilter(self)
        self.video.videoSink().videoSizeChanged.connect(self._video_size_changed)
        self._media_load_timer = QTimer(self)
        self._media_load_timer.setSingleShot(True)
        self._media_load_timer.timeout.connect(self._media_load_timed_out)

        layout = QVBoxLayout(self)
        self._root_layout = layout
        # Do not let a loaded stream's native frame size become a hard
        # minimum for the whole page.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(12, 12, 12, 12)
        self.placeholder = QLabel("Выберите видео, чтобы увидеть предпросмотр")
        self.placeholder.setObjectName("muted")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setContentsMargins(12, 8, 12, 8)
        self.placeholder.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.media_stage = QWidget(self)
        self.media_stage_layout = QHBoxLayout(self.media_stage)
        self.media_stage_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.media_stage_layout.setContentsMargins(0, 0, 0, 0)
        self.media_stage_layout.addStretch()
        self.media_stage_layout.addWidget(self.video)
        self.media_stage_layout.addStretch()
        # Keep the QVideoWidget visible for its entire lifetime. On Windows,
        # hiding its native surface for a poster can let the Qt media backend
        # continue audio playback while it stops presenting video frames.
        # Poster and loading text are lightweight overlays, not layout peers.
        self.poster = QLabel(self.media_stage)
        self.poster.setObjectName("videoPoster")
        self.poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.poster.setWordWrap(True)
        self.poster.hide()
        self.placeholder.setParent(self.media_stage)
        self.placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.poster.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.media_stage.installEventFilter(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.active_candidate = QLabel()
        self.active_candidate.setObjectName("active-candidate")
        make_label_shrinkable(self.active_candidate)
        self.active_candidate.setStyleSheet("font-weight: 600;")
        self.active_candidate.hide()
        layout.addWidget(self.active_candidate)
        layout.addWidget(self.media_stage, 0, Qt.AlignmentFlag.AlignHCenter)
        self.preview_status = QLabel()
        self.preview_status.setObjectName("muted")
        make_label_shrinkable(self.preview_status)
        self.preview_status.hide()
        layout.addWidget(self.preview_status)
        self.controls_host = QWidget()
        self.controls_host.setObjectName("previewControls")
        self.controls_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._controls_layout = QGridLayout(self.controls_host)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self._controls_layout.setHorizontalSpacing(8)
        self._controls_layout.setVerticalSpacing(6)
        self._compact_controls: bool | None = None
        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Воспроизвести")
        self.play_button.setProperty("transportControl", True)
        self.play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self._seek_preview)
        self.seek_slider.sliderReleased.connect(self._seek_released)
        self.volume_button = QPushButton("🔊")
        self.volume_button.setToolTip("Выключить звук")
        self.volume_button.setProperty("transportControl", True)
        self.volume_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setMinimumWidth(48)
        self.volume_slider.setMaximumWidth(90)
        self.fullscreen_button = QPushButton("⛶")
        self.fullscreen_button.setToolTip("На весь экран")
        self.fullscreen_button.setProperty("transportControl", True)
        self.fullscreen_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.open_button = QPushButton("Открыть в проигрывателе")
        self.open_button.setToolTip("Открыть в системном проигрывателе")
        self.open_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.play_button.clicked.connect(self._toggle_playback)
        self.volume_button.clicked.connect(self._toggle_mute)
        self.volume_slider.valueChanged.connect(self._set_volume)
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        self.video.fullScreenChanged.connect(self._fullscreen_changed)
        self.open_button.clicked.connect(self.open_externally)
        layout.addWidget(self.controls_host)
        self._apply_controls_layout(force=True)
        self.audio.volumeChanged.connect(self._audio_volume_changed)
        self.audio.mutedChanged.connect(self._audio_muted_changed)
        self.audio.setVolume(1.0)
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

    def suspend(self) -> None:
        """Release hidden native media work until an exact binding is shown.

        Windows keeps ``QVideoWidget`` as a native surface.  Leaving a source
        attached while its Results workspace is hidden can keep decoding and
        can paint stale native frames over another route during window grabs.
        The durable candidate/result id remains owned by ProjectScreen; the
        next ``show_*`` call restores that exact path rather than guessing.
        """

        if (
            self._path is None
            and self._proxy_process.state() == QProcess.ProcessState.NotRunning
            and self._poster_process.state() == QProcess.ProcessState.NotRunning
        ):
            return
        self._selection_token += 1
        self._cancel_proxy()
        self._cancel_poster()
        self._active_proxy = None
        self._active_poster = None
        self._clear_media()
        self._source_path = None
        self._source_range_seconds = None
        self._active_candidate_title = None

    def set_vertical_frame_size(self, width: int, height: int) -> None:
        """Set a compact 9:16 stage without changing the media contract."""

        if width <= 0 or height <= 0:
            raise ValueError("Vertical preview dimensions must be positive.")
        self._vertical_frame_size = (width, height)
        if self._presentation == "vertical":
            self._set_presentation("vertical")

    def set_frame_sink_output(self, enabled: bool) -> None:
        """Present decoded frames in the QWidget tree instead of a native surface.

        Windows native video surfaces are intentionally used for full source,
        Draft and Final playback.  A compact Settings sample also needs to be
        capturable by real-window QA, so it consumes the *same decoded MP4*
        through QVideoSink and paints those frames in the existing poster
        layer.  This is media playback, not a synthetic Qt animation.
        """

        self._frame_sink_output = bool(enabled)
        self.player.setVideoOutput(self._frame_sink if enabled else self.video)
        if not enabled:
            self.poster.hide()

    def _software_video_frame_changed(self, frame: QVideoFrame) -> None:
        if not self._frame_sink_output or not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image).scaled(
            self.poster.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.poster.setText("")
        self.poster.setPixmap(pixmap)
        self.placeholder.hide()
        self._sync_stage_overlays()
        self.poster.show()
        self.poster.raise_()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_controls_layout()
        self._refresh_layout_geometry()

    def _apply_controls_layout(self, *, force: bool = False) -> None:
        """Reflow transport controls before their themed hints can collide."""

        # The frame/layout consume 26 px before controls receive their row.
        # With the production Windows font the wide row needs 761 px, so a
        # 760 px preview still clips the full external-player CTA.  Keep a
        # small DPI/font-metric safety margin and switch to one row at 800.
        compact = self._presentation == "vertical" or self.width() < 800
        if not force and compact == self._compact_controls:
            return
        self._compact_controls = compact
        controls = (
            self.play_button, self.time_label, self.seek_slider, self.volume_button,
            self.volume_slider, self.fullscreen_button, self.open_button,
        )
        for control in controls:
            self._controls_layout.removeWidget(control)
        for column in range(7):
            self._controls_layout.setColumnStretch(column, 0)
        if compact:
            self._controls_layout.addWidget(self.play_button, 0, 0)
            self._controls_layout.addWidget(self.time_label, 0, 1)
            self._controls_layout.addWidget(self.seek_slider, 0, 2)
            self._controls_layout.setColumnStretch(2, 1)
            self._controls_layout.addWidget(self.volume_button, 1, 0)
            self._controls_layout.addWidget(self.volume_slider, 1, 1)
            self._controls_layout.addWidget(self.fullscreen_button, 1, 2)
            self._controls_layout.addWidget(self.open_button, 1, 3)
            self.open_button.setText("↗")
        else:
            self._controls_layout.addWidget(self.play_button, 0, 0)
            self._controls_layout.addWidget(self.time_label, 0, 1)
            self._controls_layout.addWidget(self.seek_slider, 0, 2)
            self._controls_layout.setColumnStretch(2, 1)
            self._controls_layout.addWidget(self.volume_button, 0, 3)
            self._controls_layout.addWidget(self.volume_slider, 0, 4)
            self._controls_layout.addWidget(self.fullscreen_button, 0, 5)
            self._controls_layout.addWidget(self.open_button, 0, 6)
            self.open_button.setText("Открыть в проигрывателе")
        self._controls_layout.invalidate()
        QTimer.singleShot(0, self._refresh_layout_geometry)

    def _refresh_layout_geometry(self) -> None:
        """Reserve current height-for-width so labels never paint over video."""

        layout = self.layout()
        if layout is None:
            return
        layout.invalidate()
        width = max(1, self.contentsRect().width())
        required_height = layout.totalHeightForWidth(width)
        if required_height < 0:
            required_height = layout.totalSizeHint().height()
        required_height = max(0, required_height)
        if self.minimumHeight() != required_height:
            self.setMinimumHeight(required_height)
            self.geometry_requirement_changed.emit()
        self.updateGeometry()

    def show_bound_poster(self, path: str | Path) -> bool:
        """Show a real identity-bound thumbnail while media becomes ready.

        The caller owns the candidate/result binding.  This method only
        paints the supplied image and never discovers media by name or index.
        """

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        self.poster.setText("")
        self.poster.setPixmap(pixmap.scaled(
            self.poster.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self.placeholder.hide()
        self._sync_stage_overlays()
        self.poster.show()
        self.poster.raise_()
        return True

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        if self._presentation == "vertical":
            width, height = self._vertical_frame_size
            return QSize(max(360, width + 230), max(height + 120, self.minimumHeight()))
        return QSize(864, max(520, self.minimumHeight()))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        # QVideoWidget's native frame size can change after media loads.  The
        # outer player must still be allowed to fit a normal desktop window.
        return QSize(0, self.minimumHeight())

    def show_source(
        self, path: str | Path | None, *, source_codec: str | None = None,
        poster_cache_directory: Path | None = None,
    ) -> None:
        """Show the original landscape source in a normal wide player."""

        self.set_file(
            path, presentation="source", title="Исходное видео", source_codec=source_codec,
            poster_cache_directory=poster_cache_directory,
        )

    def show_draft(
        self, path: str | Path | None, candidate_title: str | None = None, *,
        poster_cache_directory: Path | None = None,
    ) -> None:
        """Show a draft in a phone-sized 9:16 player."""

        suffix = f" · {candidate_title}" if candidate_title else ""
        self.set_file(
            path, presentation="vertical", title=f"Черновик{suffix}",
            poster_cache_directory=poster_cache_directory,
        )

    def show_final(
        self, path: str | Path | None, candidate_title: str | None = None, *,
        poster_cache_directory: Path | None = None,
    ) -> None:
        """Show a completed short in a phone-sized 9:16 player."""

        suffix = f" · {candidate_title}" if candidate_title else ""
        self.set_file(
            path, presentation="vertical", title=f"Готовый ролик{suffix}",
            poster_cache_directory=poster_cache_directory,
        )

    def set_file(
        self,
        path: str | Path | None,
        *,
        presentation: str = "auto",
        title: str | None = None,
        source_codec: str | None = None,
        poster_cache_directory: Path | None = None,
    ) -> None:
        self._selection_token += 1
        self._cancel_proxy()
        self._cancel_poster()
        self._stop_media_load_watchdog()
        self._source_range_seconds = None
        self._source_codec = self._normalise_source_codec(source_codec)
        if poster_cache_directory is not None:
            self._poster_cache_directory = poster_cache_directory
        self._force_compatible_proxy = None
        self._active_candidate_title = title
        self._range_start_ms = None
        self._range_end_ms = None
        self._range_autoplay = False
        self._range_media_ready = False
        self._range_ready_token = None
        self._range_seek_pending_token = None
        self._range_seek_target_ms = None
        self._range_requires_seek = False
        self._media_ready = False
        self._using_proxy = False
        candidate = Path(path) if path else None
        self._set_presentation(self._file_presentation(candidate, presentation))
        if title:
            set_responsive_text(self.active_candidate, title)
            self.active_candidate.show()
        else:
            self.active_candidate.clear()
            self.active_candidate.setToolTip("")
            self.active_candidate.hide()
        self._source_path = candidate if self.usable_media_path(candidate) else None
        self._path = self._source_path
        self._clear_status()
        if self._path:
            self._expected_source = QUrl.fromLocalFile(str(self._path))
            self._stop_current_playback()
            self._reset_timeline()
            self._show_placeholder("Готовим первый кадр…")
            self._request_poster(self._path)
            if self._requires_compatible_proxy():
                # Never ask the Windows Qt/FFmpeg backend to initialise a
                # full-size AV1 decoder when the one-time source probe has
                # already classified it as a compatibility-only source.  A
                # selected range will make a tiny H.264 proxy instead.
                self._release_player_source()
                self._show_placeholder(
                    "Для AV1-исходника выберите момент: его короткий preview будет подготовлен отдельно."
                )
                self._show_status(
                    "Исходник AV1 не открываем во встроенном плеере, чтобы не запускать неподдерживаемый декодер."
                )
                self._set_available(False)
            else:
                self._queue_source_load()
        else:
            self._clear_media()
        if self._path is None:
            self._set_available(False)

    def set_range(
        self,
        path: str | Path,
        start_seconds: float,
        end_seconds: float,
        *,
        autoplay: bool = True,
        cache_directory: Path | None = None,
        candidate_title: str | None = None,
        source_codec: str | None = None,
        force_compatible_proxy: bool | None = None,
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
        self._cancel_poster()
        self._stop_media_load_watchdog()
        self._source_range_seconds = (start, end)
        self._source_path = Path(path)
        self._source_codec = self._normalise_source_codec(source_codec)
        self._force_compatible_proxy = force_compatible_proxy
        self._set_presentation("source")
        self._active_candidate_title = candidate_title or "Выбранный кандидат"
        set_responsive_text(
            self.active_candidate,
            f"Кандидат: {self._active_candidate_title}\nФрагмент: {start:.1f}–{end:.1f} с"
        )
        self.active_candidate.show()
        self._range_autoplay = autoplay
        self._range_media_ready = False
        self._range_ready_token = None
        self._range_seek_pending_token = None
        self._range_seek_target_ms = None
        self._media_ready = False
        self._clear_status()
        if not self.usable_media_path(self._source_path):
            self._show_error("Не удалось открыть исходный файл для предпросмотра.")
            self._clear_media()
            return
        cache = cache_directory or (Path(tempfile.gettempdir()) / "content-factory-preview-proxies")
        self._proxy_cache_directory = cache
        # Posters share the same source-revision identity as proxies, but are
        # durable project artifacts rather than temporary OS-cache entries.
        self._poster_cache_directory = cache.parent / "preview-posters"
        # Codec metadata was obtained once while the project source was
        # registered.  Do not run ffprobe from this click path.  In
        # particular, AV1 must bypass QMediaPlayer entirely on a host that
        # cannot initialise its D3D11 AV1 decoder.
        if self._requires_compatible_proxy():
            self._request_proxy(
                self._proxy_cache_directory,
                "Для этого исходника сразу готовим совместимый preview.",
            )
            # The exact candidate frame is useful immediately while the
            # short playable proxy is encoded in the background.  It is a
            # real source frame, never a placeholder or guessed file.
            self._request_poster(self._source_path, timestamp_seconds=start)
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
        self._range_ready_token = None
        self._range_seek_pending_token = None
        self._range_seek_target_ms = None
        self._range_requires_seek = False
        self._media_ready = False
        self._expected_source = QUrl.fromLocalFile(str(self._path))
        self._stop_current_playback()
        self.poster.hide()
        self.video.show()
        self._show_placeholder("Загружаем исходный фрагмент…")
        self._show_status("Загружаем исходный фрагмент…")
        # Re-selecting a different interval from the same source must not
        # depend on QMediaPlayer emitting a second LoadedMedia notification.
        # Windows often treats ``setSource(the_same_url)`` as a no-op.
        if self._player_has_loaded_source(self._expected_source):
            self._range_requires_seek = True
            self._media_loading = False
            self._stop_media_load_watchdog()
            self._set_available(True)
            self._activate_range_ready(self._selection_token)
            return
        self._queue_source_load()

    def _queue_source_load(self) -> None:
        """Schedule the inexpensive Qt source handoff after the card repaint.

        QMediaPlayer must stay on the GUI thread because it owns a widget
        output, but the Windows backend performs the actual open/decode
        asynchronously.  Deferring the handoff lets the clicked card, loading
        state and controls repaint first, and coalesces rapid selections.
        """

        if self._path is None:
            return
        token = self._selection_token
        self._media_loading = True
        self._media_load_token = token
        self._media_load_timer.start(self.MEDIA_LOAD_TIMEOUT_MS)
        self._set_available(False)
        QTimer.singleShot(0, lambda value=token: self._load_selected_source(value))

    def _load_selected_source(self, token: int) -> None:
        if token != self._selection_token or self._path is None:
            return
        self._ensure_video_output()
        source = QUrl.fromLocalFile(str(self._path))
        if source != self._expected_source:
            return
        if self._range_start_ms is not None and self._player_has_loaded_source(source):
            self._range_requires_seek = True
            self._media_loading = False
            self._stop_media_load_watchdog(token)
            self._set_available(True)
            self._activate_range_ready(token)
            return
        logger.info("media source load requested token=%s source=%s", token, self._path)
        # Do not set an empty source between selections. With the Windows Qt
        # multimedia backend that forces renderer teardown and was the source
        # of multi-second UI stalls on rapid card switches.
        self.player.setSource(source)

    def _player_has_loaded_source(self, source: QUrl) -> bool:
        """Whether the live player can be retargeted without another load."""

        source_getter = getattr(self.player, "source", None)
        status_getter = getattr(self.player, "mediaStatus", None)
        if not callable(source_getter) or not callable(status_getter) or source_getter() != source:
            return False
        return status_getter() in {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            # EndOfMedia retains the same decoded source.  Reuse it for a
            # new range selection instead of asking the Windows backend to
            # set the same URL again (which it may silently ignore).
            QMediaPlayer.MediaStatus.EndOfMedia,
        }

    def _activate_range_ready(self, token: int) -> None:
        if token != self._selection_token or self._range_start_ms is None:
            return
        if self._range_ready_token == token and self._range_media_ready:
            return
        # The Qt backend updates QVideoWidget's native size after the source
        # is loaded. Reapply the bounded presentation afterwards so a 1440p
        # landscape source cannot widen the review page.
        self._set_presentation(self._presentation)
        self._range_media_ready = True
        self._range_ready_token = token
        if self._range_autoplay:
            self._range_autoplay = False
            self._start_range_playback()
        elif self._range_start_ms > 0 or self._range_requires_seek:
            self._seek_range_start(token)

    def _seek_range_start(self, token: int) -> None:
        if token != self._selection_token or self._range_start_ms is None:
            return
        target = self._range_start_ms
        self._range_seek_pending_token = token
        self._range_seek_target_ms = target
        self.player.setPosition(target)

    def _stop_current_playback(self) -> None:
        state_getter = getattr(self.player, "playbackState", None)
        stop = getattr(self.player, "stop", None)
        if callable(stop) and (not callable(state_getter) or state_getter() != QMediaPlayer.PlaybackState.StoppedState):
            logger.info("media playback stopped before source switch")
            stop()

    @staticmethod
    def _normalise_source_codec(value: str | None) -> str | None:
        codec = value.strip().casefold() if isinstance(value, str) else ""
        return codec or None

    def _requires_compatible_proxy(self) -> bool:
        """Whether this source must not enter the Qt multimedia decoder.

        ``force_compatible_proxy`` is intentionally an explicit integration
        hook for a capability probe.  The safe default for an AV1 source is a
        short local proxy: codec support in the external FFmpeg binary does
        not guarantee matching D3D11 support in Qt's bundled backend.
        """

        if self._force_compatible_proxy is not None:
            return bool(self._force_compatible_proxy)
        return self._source_codec in {"av1", "av01"}

    def _stop_media_load_watchdog(self, token: int | None = None) -> None:
        if token is not None and token != self._media_load_token:
            return
        self._media_load_timer.stop()
        self._media_load_token = None

    def _media_load_timed_out(self) -> None:
        """End a source handoff that produced neither ready nor error events."""

        token = self._media_load_token
        if (
            token is None
            or token != self._selection_token
            or not self._media_loading
        ):
            return
        self._media_load_token = None
        self._media_loading = False
        logger.warning("media source load timed out token=%s source=%s", token, self._path)
        if self._source_path and self._source_range_seconds and not self._using_proxy:
            self._request_proxy(
                self._proxy_cache_directory,
                "Исходный фрагмент не открылся вовремя; готовим совместимый preview.",
            )
            return
        self._show_error(
            "Не удалось подготовить preview вовремя. Выберите момент ещё раз или откройте журнал проекта."
        )

    def _release_player_source(self) -> None:
        """Stop and detach the backend only for an error/fallback teardown.

        Normal candidate switches intentionally keep their source attached to
        avoid a Windows renderer stall.  An invalid source or a compatibility
        fallback is different: retaining it can leave its decoder alive and
        continue emitting stale callbacks after the UI has moved on.
        """

        self._expected_source = QUrl()
        self._stop_current_playback()
        clear_source = getattr(self.player, "setSource", None)
        if callable(clear_source):
            try:
                clear_source(QUrl())
            except RuntimeError:
                # A closing Qt object can reject the final handoff; it is
                # already being torn down and must not create a second error.
                pass

    def _ensure_video_output(self) -> None:
        """Recover a detached output without rebinding a healthy live sink."""

        output_getter = getattr(self.player, "videoOutput", None)
        output = output_getter() if callable(output_getter) else self.video
        expected_output = self._frame_sink if self._frame_sink_output else self.video
        if output is not expected_output:
            logger.warning("media video output was detached; restoring persistent output")
            self.player.setVideoOutput(expected_output)

    def _request_poster(
        self, source_path: Path, *, timestamp_seconds: float = 0.05,
    ) -> None:
        request = _PosterRequest(
            token=self._selection_token,
            source_path=source_path,
            destination=preview_poster_path(
                self._poster_cache_directory, source_path, timestamp_seconds,
            ),
            timestamp_seconds=max(0.0, timestamp_seconds),
        )
        if self.usable_media_path(request.destination):
            self._show_poster(request)
            return
        if self._active_poster is not None:
            self._pending_poster = request
            if self._poster_process.state() != QProcess.ProcessState.NotRunning:
                self._poster_process.kill()
            return
        self._start_poster(request)

    def _start_poster(self, request: _PosterRequest) -> None:
        executable = shutil.which("ffmpeg")
        if not executable:
            self._show_placeholder("Первый кадр будет доступен после запуска видео.")
            return
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        self._active_poster = request
        self._poster_process.setProgram(executable)
        self._poster_process.setArguments([
            "-y", "-hide_banner", "-loglevel", "error", "-threads", "1",
            "-ss", f"{request.timestamp_seconds:.3f}",
            "-i", str(request.source_path), "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3",
            str(request.destination),
        ])
        self._poster_process.start()

    def _poster_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        request = self._active_poster
        if request is None:
            return
        self._active_poster = None
        pending = self._pending_poster
        self._pending_poster = None
        if pending is not None:
            self._start_poster(pending)
            return
        if request.token != self._selection_token:
            return
        if exit_code == 0 and self.usable_media_path(request.destination):
            self._show_poster(request)
            return
        self._show_placeholder("Первый кадр недоступен. Нажмите «Воспроизвести».")

    def _poster_process_error(self, _error: QProcess.ProcessError) -> None:
        request = self._active_poster
        if request is not None:
            QTimer.singleShot(0, lambda req=request: self._complete_poster_error(req))

    def _complete_poster_error(self, request: _PosterRequest) -> None:
        if self._active_poster is request and self._poster_process.state() == QProcess.ProcessState.NotRunning:
            self._poster_finished(1, QProcess.ExitStatus.CrashExit)

    def _show_poster(self, request: _PosterRequest) -> None:
        if request.token != self._selection_token or self._source_path != request.source_path:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            return
        pixmap = QPixmap(str(request.destination))
        if pixmap.isNull():
            self._show_placeholder("Первый кадр недоступен. Нажмите «Воспроизвести».")
            return
        self.poster.setText("")
        self.poster.setPixmap(pixmap.scaled(
            self.poster.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self.placeholder.hide()
        self._sync_stage_overlays()
        self.poster.show()
        self.poster.raise_()

    def _show_placeholder(self, message: str) -> None:
        self.poster.hide()
        self.video.show()
        set_responsive_text(self.placeholder, message)
        self._sync_stage_overlays()
        self.placeholder.show()
        self.placeholder.raise_()

    def _show_video(self) -> None:
        if not self._frame_sink_output:
            self.poster.hide()
        self.placeholder.hide()
        self.video.show()

    def _file_presentation(self, candidate: Path | None, presentation: str) -> str:
        if presentation in {"source", "vertical"}:
            return presentation
        # This is invoked from the UI thread.  Final and draft callers pass an
        # explicit presentation, and auto mode deliberately avoids a blocking
        # ffprobe just to infer an optional visual frame.
        return "source"

    def _set_presentation(self, presentation: str) -> None:
        """Keep short-form outputs recognisable as phone-video previews."""

        self._presentation = "vertical" if presentation == "vertical" else "source"
        vertical = self._presentation == "vertical"
        if vertical:
            width, height = self._vertical_frame_size
            self.setMaximumWidth(500)
            self.video.set_preview_size(width, height)
            self.video.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.poster.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.placeholder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.video.setFixedSize(width, height)
            self.poster.setFixedSize(width, height)
            self.placeholder.setFixedSize(width, height)
            self.media_stage.setFixedSize(width, height)
            self.media_stage.setObjectName("phoneStage")
        else:
            self.setMaximumWidth(864)
            self.video.set_preview_size(840, 420)
            for widget in (self.video, self.poster, self.placeholder):
                widget.setMinimumSize(0, 260)
                widget.setMaximumSize(840, 420)
                widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.media_stage.setMinimumSize(0, 260)
            self.media_stage.setMaximumSize(840, 420)
            self.media_stage.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.media_stage.setObjectName("")
        self.media_stage.style().unpolish(self.media_stage)
        self.media_stage.style().polish(self.media_stage)
        self._apply_controls_layout()

        # Layout geometry is applied asynchronously by Qt after an output
        # changes its native stream size.
        QTimer.singleShot(0, self._sync_stage_overlays)
        QTimer.singleShot(0, self._refresh_layout_geometry)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched in {self.video, self.media_stage} and event.type() in {
            QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.LayoutRequest,
        }:
            QTimer.singleShot(0, self._sync_stage_overlays)
        if (
            watched is self.video
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
            and self.video.isFullScreen()
        ):
            self.video.setFullScreen(False)
            return True
        return super().eventFilter(watched, event)

    def _sync_stage_overlays(self) -> None:
        if self.video.isFullScreen():
            return
        geometry = self.video.geometry()
        if geometry.width() <= 0 or geometry.height() <= 0:
            geometry = self.media_stage.rect()
        self.poster.setGeometry(geometry)
        self.placeholder.setGeometry(geometry)

    def _request_proxy(self, cache_directory: Path, reason: str) -> None:
        assert self._source_path is not None and self._source_range_seconds is not None
        start, end = self._source_range_seconds
        destination = preview_proxy_path(cache_directory, self._source_path, start, end)
        request = _ProxyRequest(
            token=self._selection_token,
            source_path=self._source_path,
            start_seconds=start,
            end_seconds=end,
            destination=destination,
            temporary=preview_proxy_temporary_path(destination),
        )
        self._stop_media_load_watchdog()
        self._release_player_source()
        self._path = None
        self._media_loading = True
        self._using_proxy = True
        self._set_available(False)
        self.placeholder.setText("Подготавливаем совместимый предпросмотр…")
        self.placeholder.show()
        self.video.show()
        self._sync_stage_overlays()
        self.placeholder.raise_()
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
            self._show_error("Не хватает компонента обработки видео для совместимого предпросмотра.")
            return
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            request.temporary.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("could not remove stale preview temporary %s: %s", request.temporary, error)
        self._active_proxy = request
        self._proxy_timed_out_request = None
        duration = max(0.05, request.end_seconds - request.start_seconds)
        self._proxy_process.setProgram(executable)
        self._proxy_process.setArguments([
            "-y", "-hide_banner", "-loglevel", "error",
            # Input seek avoids decoding the full source before a late PUBG
            # candidate.  Transcoding keeps FFmpeg's accurate-seek discard.
            # This is UI-only work, so do not let a 4K AV1 source consume all
            # host cores while the production pipeline remains available.
            "-threads", "2", "-ss", f"{request.start_seconds:.3f}", "-i", str(request.source_path),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", "scale=-2:480,fps=30,setsar=1", "-c:v", "libx264", "-threads", "2",
            "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(request.temporary),
        ])
        self._proxy_process.start()
        self._proxy_timeout_timer.start(self.PROXY_RENDER_TIMEOUT_MS)

    def _proxy_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        request = self._active_proxy
        if request is None:
            return
        self._proxy_timeout_timer.stop()
        details = self._ffmpeg_error_text()
        success = False
        if exit_code == 0 and self.usable_media_path(request.temporary):
            try:
                request.temporary.replace(request.destination)
                success = self.usable_media_path(request.destination)
            except OSError as error:
                details = f"Could not finalise compatible preview: {error}"
        if self._proxy_timed_out_request is request:
            success = False
            details = details or "Compatible preview exceeded its time limit and was stopped."
        self._complete_proxy(request, success, details)

    def _proxy_process_error(self, _error: QProcess.ProcessError) -> None:
        request = self._active_proxy
        if request is not None:
            QTimer.singleShot(0, lambda req=request: self._complete_proxy_error(req))

    def _complete_proxy_error(self, request: _ProxyRequest) -> None:
        if self._active_proxy is request and self._proxy_process.state() == QProcess.ProcessState.NotRunning:
            self._complete_proxy(request, False, self._ffmpeg_error_text() or "Не удалось подготовить совместимый предпросмотр.")

    def _complete_proxy(self, request: _ProxyRequest, success: bool, details: str) -> None:
        if self._active_proxy is not request:
            return
        self._proxy_timeout_timer.stop()
        if self._proxy_timed_out_request is request:
            self._proxy_timed_out_request = None
        try:
            request.temporary.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("could not clean preview temporary %s: %s", request.temporary, error)
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
        if details:
            # FFmpeg's stderr can contain codec internals and local paths.  It
            # is useful in the application log, but is not recovery guidance
            # for the person reviewing a moment.
            logger.error("compatible preview proxy failed: %s", details[-1200:])
        self._show_error(
            "Предпросмотр недоступен: не удалось создать совместимую локальную копию. "
            "Проверьте исходное видео или откройте журнал проекта."
        )

    def _proxy_timed_out(self) -> None:
        request = self._active_proxy
        if request is None:
            return
        self._proxy_timed_out_request = request
        logger.error("compatible preview timed out source=%s", request.source_path)
        self._show_status("Создание совместимого preview заняло слишком много времени; останавливаем его.")
        if self._proxy_process.state() != QProcess.ProcessState.NotRunning:
            self._proxy_process.kill()
            return
        self._complete_proxy(
            request, False, "Compatible preview exceeded its time limit and was stopped.",
        )

    def _activate_proxy(self, request: _ProxyRequest) -> None:
        if request.token != self._selection_token:
            return
        self._path = request.destination
        self._using_proxy = True
        self._range_start_ms = 0
        self._range_end_ms = int(round((request.end_seconds - request.start_seconds) * 1000))
        self._range_media_ready = False
        self._range_ready_token = None
        self._range_seek_pending_token = None
        self._range_seek_target_ms = None
        self._range_requires_seek = False
        self._media_ready = False
        self._expected_source = QUrl.fromLocalFile(str(self._path))
        self._stop_current_playback()
        self.poster.hide()
        self.video.show()
        self._show_placeholder("Загружаем совместимый предпросмотр…")
        self._queue_source_load()
        self._show_status(
            f"Совместимый preview готов: {request.start_seconds:.1f}–{request.end_seconds:.1f} с исходного видео."
        )
        self.preview_ready.emit(str(request.destination))

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if not self._is_current_player_source():
            return
        logger.info("media status=%s source=%s", status.name, self._path)
        if (
            status == QMediaPlayer.MediaStatus.EndOfMedia
            and self._range_media_ready
            and self._range_end_ms is not None
        ):
            # The Windows Qt backend can reset position to zero after the
            # terminal frame.  Keep the stopped candidate visibly at its end.
            self._stop_at_range_end()
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._media_loading = False
            self._stop_media_load_watchdog()
            if self._source_path and self._source_range_seconds and not self._using_proxy:
                self._request_proxy(self._proxy_cache_directory, "Qt Multimedia не смог открыть исходный формат; создаём совместимый preview.")
            else:
                self._show_error("Системный проигрыватель не смог открыть подготовленный предпросмотр.")
            return
        if status in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia}:
            self._media_loading = False
            self._stop_media_load_watchdog()
            self._set_available(True)
            if self._range_start_ms is None:
                # Decode/present frame zero while the poster is still merely an
                # overlay.  The QVideoWidget has never been hidden, so Play can
                # transition directly to moving frames rather than recreating a
                # native video surface.
                if not self._media_ready:
                    self._media_ready = True
                    self.player.setPosition(0)
                self._clear_status()
                return
            if self._range_media_ready:
                return
            self._activate_range_ready(self._selection_token)

    def _position_changed(self, position: int) -> None:
        if not self._is_current_player_source():
            return
        if self._range_start_ms is not None:
            if not self._range_media_ready or self._range_ready_token != self._selection_token:
                return
            if self._range_seek_pending_token == self._selection_token:
                target = self._range_seek_target_ms
                # A notification queued by the previous interval carries the
                # same URL.  Ignore it until the player acknowledges the
                # current interval's explicit seek.
                if target is not None and abs(position - target) > 500:
                    return
                self._range_seek_pending_token = None
                self._range_seek_target_ms = None
        self._update_timeline(position)
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

    def _duration_changed(self, _duration: int) -> None:
        if not self._is_current_player_source():
            return
        if (
            self._range_start_ms is not None
            and (
                not self._range_media_ready
                or self._range_ready_token != self._selection_token
                or self._range_seek_pending_token == self._selection_token
            )
        ):
            return
        self._update_timeline(self.player.position())

    def _update_timeline(self, position: int) -> None:
        if self._range_start_ms is not None and self._range_end_ms is not None:
            duration = max(0, self._range_end_ms - self._range_start_ms)
            shown_position = position if self._using_proxy else position - self._range_start_ms
        else:
            duration = max(0, self.player.duration())
            shown_position = position
        shown_position = max(0, min(shown_position, duration if duration else shown_position))
        self.time_label.setText(f"{self._format_time(shown_position)} / {self._format_time(duration)}")
        enabled = bool(self._path and duration > 0)
        self.seek_slider.setEnabled(enabled)
        if enabled and not self.seek_slider.isSliderDown():
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(round(shown_position * 1000 / duration))
            self.seek_slider.blockSignals(False)

    def _seek_preview(self, value: int) -> None:
        duration = self._timeline_duration()
        if duration > 0:
            self.time_label.setText(f"{self._format_time(round(duration * value / 1000))} / {self._format_time(duration)}")

    def _seek_released(self) -> None:
        if not self._path or self._media_loading:
            return
        duration = self._timeline_duration()
        if duration <= 0:
            return
        relative = round(duration * self.seek_slider.value() / 1000)
        if self._range_start_ms is not None and not self._using_proxy:
            target = self._range_start_ms + relative
        else:
            target = relative
        self._show_video()
        if self._range_start_ms is not None:
            self._range_seek_pending_token = self._selection_token
            self._range_seek_target_ms = target
        self.player.setPosition(target)

    def _timeline_duration(self) -> int:
        if self._range_start_ms is not None and self._range_end_ms is not None:
            return max(0, self._range_end_ms - self._range_start_ms)
        return max(0, self.player.duration())

    @staticmethod
    def _format_time(milliseconds: int) -> str:
        seconds = max(0, milliseconds // 1000)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _stop_at_range_end(self) -> None:
        if self._range_end_ms is None:
            return
        self.player.pause()
        # Seeking to an exact file duration is normalized to zero by Windows
        # Windows Qt backend. The final millisecond is the same visible end
        # frame while remaining a stable, paused position.
        last_frame = max(self._range_start_ms or 0, self._range_end_ms - 1)
        token = self._selection_token
        QTimer.singleShot(0, lambda position=last_frame, value=token: self._seek_if_current(value, position))
        self._show_status("Просмотр завершён на конце выбранного фрагмента.")

    def _seek_if_current(self, token: int, position: int) -> None:
        if token == self._selection_token and self._is_current_player_source():
            self.player.setPosition(position)

    def open_externally(self) -> None:
        target = self._source_path or self._path
        if target and target.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return
        self._play()

    def _play(self) -> None:
        if not self._path or self._media_loading:
            if self._active_proxy is not None:
                self._show_status("Предпросмотр ещё подготавливается.")
            elif self._media_loading:
                self._show_status("Видео ещё загружается.")
            return
        self._ensure_video_output()
        self._show_video()
        if self._range_start_ms is None:
            self.player.play()
            return
        self._start_range_playback()

    def _start_range_playback(self) -> None:
        if self._range_start_ms is None:
            return
        token = self._selection_token
        self._seek_range_start(token)
        QTimer.singleShot(0, lambda value=token: self._play_if_current(value))
        self._show_status("Воспроизведение выбранного фрагмента…")

    def _play_if_current(self, token: int) -> None:
        if token == self._selection_token and self._is_current_player_source():
            self.player.play()

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = not self._media_loading and state == QMediaPlayer.PlaybackState.PlayingState
        logger.info("media playback state=%s source=%s", state.name, self._path)
        self.play_button.setText("❚❚" if playing else "▶")
        self.play_button.setToolTip("Пауза" if playing else "Воспроизвести")

    def _toggle_mute(self) -> None:
        if self.audio.isMuted():
            if self.volume_slider.value() == 0:
                self._set_slider_volume(self._last_audible_volume)
            self.audio.setMuted(False)
        elif self.volume_slider.value() == 0:
            # A zero slider is silent but not necessarily QAudioOutput-muted.
            # The speaker control restores the last audible level as users
            # expect from an unmute action.
            self._set_slider_volume(self._last_audible_volume)
            self.audio.setMuted(False)
        else:
            self.audio.setMuted(True)
        self._update_volume_button()

    def _set_volume(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        if value > 0:
            self._last_audible_volume = value
        self.audio.setVolume(value / 100.0)
        if value > 0 and self.audio.isMuted():
            self.audio.setMuted(False)
        self._update_volume_button()

    def _audio_volume_changed(self, value: float) -> None:
        percent = max(0, min(100, round(float(value) * 100)))
        if percent > 0:
            self._last_audible_volume = percent
        if self.volume_slider.value() != percent:
            self._set_slider_volume(percent)
        self._update_volume_button()

    def _audio_muted_changed(self, _muted: bool) -> None:
        self._update_volume_button()

    def _set_slider_volume(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        blocker = QSignalBlocker(self.volume_slider)
        self.volume_slider.setValue(value)
        del blocker
        self._set_volume(value)

    def _update_volume_button(self) -> None:
        muted = self.audio.isMuted() or self.volume_slider.value() == 0
        self.volume_button.setText("🔇" if muted else "🔊")
        self.volume_button.setToolTip("Включить звук" if muted else "Выключить звук")

    def _toggle_fullscreen(self) -> None:
        if not self._path or self._media_loading:
            return
        self._ensure_video_output()
        self._show_video()
        self.video.setFullScreen(not self.video.isFullScreen())

    def _fullscreen_changed(self, enabled: bool) -> None:
        logger.info("media fullscreen=%s source=%s", enabled, self._path)
        self.fullscreen_button.setText("⤢" if enabled else "⛶")
        self.fullscreen_button.setToolTip("Выйти из полного экрана" if enabled else "На весь экран")
        if not enabled:
            QTimer.singleShot(0, self._sync_stage_overlays)

    def _video_size_changed(self) -> None:
        size = self.video.videoSink().videoSize()
        logger.info("media video size=%sx%s source=%s", size.width(), size.height(), self._path)
        if not self.video.isFullScreen():
            self._set_presentation(self._presentation)

    def _is_current_player_source(self) -> bool:
        if self._path is None:
            return False
        source_getter = getattr(self.player, "source", None)
        if not self._expected_source.isValid():
            # Keeps small unit fakes that only exercise state handling simple,
            # while rejecting real backend notifications after a source clear.
            return not callable(source_getter)
        return not callable(source_getter) or source_getter() == self._expected_source

    def _media_error(self, *_: object) -> None:
        if not self._is_current_player_source():
            return
        self._media_loading = False
        self._stop_media_load_watchdog()
        logger.error("media error source=%s details=%s", self._path, self.player.errorString().strip())
        if self._source_path and self._source_range_seconds and not self._using_proxy:
            self._request_proxy(self._proxy_cache_directory, "Qt Multimedia не поддержал исходный формат; создаём совместимый preview.")
            return
        self._show_error(
            "Встроенное воспроизведение недоступно. "
            "Попробуйте открыть видео во внешнем проигрывателе или откройте журнал проекта."
        )

    def _cancel_proxy(self) -> None:
        self._pending_proxy = None
        self._proxy_timeout_timer.stop()
        self._proxy_timed_out_request = None
        if self._active_proxy is not None and self._proxy_process.state() != QProcess.ProcessState.NotRunning:
            self._proxy_process.kill()

    def _cancel_poster(self) -> None:
        self._pending_poster = None
        if self._active_poster is not None and self._poster_process.state() != QProcess.ProcessState.NotRunning:
            self._poster_process.kill()

    def _clear_media(self) -> None:
        self._stop_media_load_watchdog()
        self._release_player_source()
        self._path = None
        self._media_loading = False
        self._range_media_ready = False
        self._range_ready_token = None
        self._range_seek_pending_token = None
        self._range_seek_target_ms = None
        self._media_ready = False
        self.poster.hide()
        self.video.show()
        self.placeholder.show()
        self._sync_stage_overlays()
        self.placeholder.raise_()
        self._reset_timeline()
        self._set_available(False)

    def _show_status(self, message: str) -> None:
        self.preview_status.setStyleSheet("")
        set_responsive_text(self.preview_status, message)
        self.preview_status.show()
        self._refresh_layout_geometry()

    def _clear_status(self) -> None:
        self.preview_status.clear()
        self.preview_status.setToolTip("")
        self.preview_status.hide()
        self._refresh_layout_geometry()

    def _show_error(self, message: str) -> None:
        self._stop_media_load_watchdog()
        self._release_player_source()
        self._path = None
        self._media_loading = False
        self._range_media_ready = False
        self._range_ready_token = None
        self._range_seek_pending_token = None
        self._range_seek_target_ms = None
        self._media_ready = False
        self.poster.hide()
        set_responsive_text(self.placeholder, message)
        self.placeholder.show()
        self.video.show()
        self._sync_stage_overlays()
        self.placeholder.raise_()
        self.preview_status.setStyleSheet("color: #d66;")
        set_responsive_text(self.preview_status, message)
        self.preview_status.show()
        self._refresh_layout_geometry()
        self._set_available(False)
        self.preview_error.emit(message)

    def _reset_timeline(self) -> None:
        self.time_label.setText("00:00 / 00:00")
        blocker = QSignalBlocker(self.seek_slider)
        self.seek_slider.setValue(0)
        del blocker

    def _ffmpeg_error_text(self) -> str:
        return bytes(self._proxy_process.readAllStandardError()).decode("utf-8", errors="replace").strip()

    def _set_available(self, value: bool) -> None:
        self.play_button.setEnabled(value)
        self.volume_button.setEnabled(value)
        self.volume_slider.setEnabled(value)
        self.fullscreen_button.setEnabled(value)
        self.seek_slider.setEnabled(False)
        self.open_button.setEnabled(bool(self._source_path and self._source_path.is_file()))

    @staticmethod
    def usable_media_path(path: Path | None) -> bool:
        return bool(path and path.is_file() and path.stat().st_size > 0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.video.isFullScreen():
            self.video.setFullScreen(False)
        self._stop_media_load_watchdog()
        self._release_player_source()
        self._cancel_proxy()
        self._cancel_poster()
        super().closeEvent(event)
