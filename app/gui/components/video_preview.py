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
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self)
        self.video.setMinimumHeight(220)
        self.player.setVideoOutput(self.video)
        self.player.errorOccurred.connect(self._media_error)
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
        candidate = Path(path) if path else None
        self._path = candidate if candidate and candidate.is_file() else None
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


from PySide6.QtCore import Qt
