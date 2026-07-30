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
        self.detail = QLabel("")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignLeft)

    def set_running(
        self,
        stage: str,
        elapsed: str,
        progress_fraction: float | None = None,
        detail: str = "",
        *,
        cancelling: bool = False,
    ) -> None:
        self.stage.setText(stage)
        self.elapsed.setText(elapsed)
        self.detail.setText(detail)
        self.progress.show()
        if progress_fraction is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, round(progress_fraction * 100))))
        self.cancel_button.setText("Останавливаем…" if cancelling else "Отменить")
        self.cancel_button.setDisabled(cancelling)
        self.cancel_button.show()

    def set_finished(self, message: str) -> None:
        self.stage.setText(message)
        self.elapsed.clear()
        self.detail.clear()
        self.progress.hide()
        self.cancel_button.hide()
        self.cancel_button.setDisabled(False)


from PySide6.QtCore import Qt
