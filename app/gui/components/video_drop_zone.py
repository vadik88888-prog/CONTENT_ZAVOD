from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy


class VideoDropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(4)
        icon = QLabel("⇧")
        icon.setObjectName("dropZoneIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Перетащите видео сюда")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        detail = QLabel("или выберите файл кнопкой выше")
        detail.setObjectName("muted")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        formats = QLabel("Поддерживаются MP4, MOV, MKV, AVI и WebM")
        formats.setObjectName("muted")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Do not make the full one-line copy a minimum page width. On a
        # scaled laptop it must wrap inside the source card, not be clipped by
        # a deliberately hidden horizontal scrollbar.
        for label in (title, detail, formats):
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addWidget(formats)

    def dragEnterEvent(self, event):  # type: ignore[override]
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()
