from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class VideoPreview(QFrame):
    """Qt Multimedia preview with a system-player escape hatch on every machine."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("preview")
        self._path: Path | None = None
        self._range_start_ms: int | None = None
        self._range_end_ms: int | None = None
        self._range_autoplay = False
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self)
        self.video.setMinimumHeight(220)
        self.player.setVideoOutput(self.video)
        self.player.errorOccurred.connect(self._media_error)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.positionChanged.connect(self._position_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.placeholder = QLabel("Выберите видео, чтобы увидеть предпросмотр")
        self.placeholder.setObjectName("muted")
        self.placeholder.setMinimumHeight(220)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video)
        layout.addWidget(self.placeholder)
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

    def set_file(self, path: str | Path | None) -> None:
        self._range_start_ms = None
        self._range_end_ms = None
        self._range_autoplay = False
        candidate = Path(path) if path else None
        self._path = candidate if self.usable_media_path(candidate) else None
        if self._path:
            self.player.setSource(QUrl.fromLocalFile(str(self._path)))
            self.placeholder.hide()
            self.video.show()
        else:
            self.player.stop()
            self.player.setSource(QUrl())
            self.placeholder.show()
            self.video.hide()
        self._set_available(self._path is not None)

    def set_range(self, path: str | Path, start_seconds: float, end_seconds: float, *, autoplay: bool = True) -> None:
        """Preview an original source interval without creating a render artifact."""

        start = max(0.0, float(start_seconds))
        end = max(start, float(end_seconds))
        self._range_start_ms = int(round(start * 1000))
        self._range_end_ms = int(round(end * 1000))
        self._range_autoplay = autoplay
        candidate = Path(path)
        self._path = candidate if self.usable_media_path(candidate) else None
        if not self._path:
            self.set_file(None)
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(self._path)))
        self.placeholder.hide()
        self.video.show()
        self._set_available(True)

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if self._path is None or self._range_start_ms is None:
            return
        if status in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia}:
            self.player.setPosition(self._range_start_ms)
            if self._range_autoplay:
                self.player.play()
                self._range_autoplay = False

    def _position_changed(self, position: int) -> None:
        if self._range_end_ms is not None and position >= self._range_end_ms:
            self.player.pause()
            self.player.setPosition(self._range_end_ms)

    def open_externally(self) -> None:
        if self._path and self._path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))

    def _play(self) -> None:
        if self._path:
            self.player.play()

    def _media_error(self, *_) -> None:
        # Some Windows builds do not include an applicable codec. The system button remains usable.
        self.placeholder.setText("Встроенное воспроизведение недоступно. Откройте файл системным проигрывателем.")
        self.placeholder.show()

    def _set_available(self, value: bool) -> None:
        self.play_button.setEnabled(value)
        self.open_button.setEnabled(value)

    @staticmethod
    def usable_media_path(path: Path | None) -> bool:
        return bool(path and path.is_file() and path.stat().st_size > 0)


from PySide6.QtCore import Qt
