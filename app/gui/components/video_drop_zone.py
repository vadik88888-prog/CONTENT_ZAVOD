from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel


class VideoDropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel("Перетащите одно видео сюда")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        detail = QLabel("или выберите файл кнопкой выше")
        detail.setObjectName("muted")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(detail)

    def dragEnterEvent(self, event):  # type: ignore[override]
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):  # type: ignore[override]
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()


from PySide6.QtCore import Qt
