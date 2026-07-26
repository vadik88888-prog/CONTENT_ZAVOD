from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QProgressBar, QVBoxLayout


class ProcessingProgress(QFrame):
    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        top = QHBoxLayout()
        self.stage = QLabel("Готово к созданию ролика")
        self.stage.setStyleSheet("font-weight: 600;")
        self.elapsed = QLabel("")
        self.elapsed.setObjectName("muted")
        top.addWidget(self.stage)
        top.addStretch()
        top.addWidget(self.elapsed)
        layout.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.recent = QLabel("")
        self.recent.setObjectName("muted")
        self.recent.setWordWrap(True)
        self._history: list[str] = []
        layout.addWidget(self.recent)
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignLeft)

    def set_running(self, stage: str, elapsed: str, progress_fraction: float | None = None) -> None:
        self.stage.setText(stage)
        self.elapsed.setText(elapsed)
        if not self._history or self._history[-1] != stage:
            self._history.append(stage)
            self._history = self._history[-3:]
            self.recent.setText(" · ".join(self._history[:-1]))
        self.progress.show()
        if progress_fraction is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, round(progress_fraction * 100))))
        self.cancel_button.show()

    def set_finished(self, message: str) -> None:
        self.stage.setText(message)
        self.elapsed.clear()
        if message not in self._history:
            self._history.append(message)
        self.recent.setText(" · ".join(self._history[-3:-1]))
        self.progress.hide()
        self.cancel_button.hide()


from PySide6.QtCore import Qt
