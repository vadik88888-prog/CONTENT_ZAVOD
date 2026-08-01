from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QProgressBar, QVBoxLayout


class ProcessingProgress(QFrame):
    """Human-facing progress for a persisted local processing run.

    The screen receives the actual stage and timing from the existing run
    snapshot.  It deliberately does not estimate a percentage when the engine
    has not supplied one.
    """

    cancel_requested = Signal()
    continue_waiting_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)

        self.eyebrow = QLabel("ОБРАБОТКА")
        self.eyebrow.setObjectName("muted")
        self.eyebrow.setStyleSheet("color: #FF8A2A; font-size: 11px; font-weight: 700;")
        layout.addWidget(self.eyebrow)

        top = QHBoxLayout()
        top.setSpacing(12)
        stage_copy = QVBoxLayout()
        stage_copy.setSpacing(3)
        self.stage_label = QLabel("Текущий этап")
        self.stage_label.setObjectName("muted")
        self.stage = QLabel("Готово к созданию ролика")
        self.stage.setWordWrap(True)
        self.stage.setStyleSheet("font-size: 20px; font-weight: 700;")
        stage_copy.addWidget(self.stage_label)
        stage_copy.addWidget(self.stage)
        top.addLayout(stage_copy, 1)

        elapsed_copy = QVBoxLayout()
        elapsed_copy.setSpacing(3)
        self.elapsed_label = QLabel("В РАБОТЕ")
        self.elapsed_label.setObjectName("muted")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.elapsed = QLabel("")
        self.elapsed.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.elapsed.setStyleSheet("font-size: 16px; font-weight: 600;")
        elapsed_copy.addWidget(self.elapsed_label)
        elapsed_copy.addWidget(self.elapsed)
        top.addLayout(elapsed_copy)
        layout.addLayout(top)

        progress_top = QHBoxLayout()
        self.progress_caption = QLabel("Общий прогресс")
        self.progress_caption.setStyleSheet("font-weight: 600;")
        self.progress_value = QLabel("")
        self.progress_value.setStyleSheet("color: #FF8A2A; font-size: 18px; font-weight: 700;")
        progress_top.addWidget(self.progress_caption)
        progress_top.addStretch()
        progress_top.addWidget(self.progress_value)
        layout.addLayout(progress_top)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.progress_note = QLabel("")
        self.progress_note.setObjectName("muted")
        self.progress_note.setWordWrap(True)
        self.progress_note.hide()
        layout.addWidget(self.progress_note)

        self.detail = QLabel("")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        self.detail.hide()
        layout.addWidget(self.detail)

        self.warning = QLabel("")
        self.warning.setObjectName("warning")
        self.warning.setWordWrap(True)
        self.warning.hide()
        layout.addWidget(self.warning)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.continue_waiting_button = QPushButton("Продолжить ждать")
        self.continue_waiting_button.clicked.connect(self.continue_waiting_requested)
        self.continue_waiting_button.hide()
        action_row.addWidget(self.continue_waiting_button)

        self.cancel_button = QPushButton("Остановить обработку")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.cancel_button.hide()
        action_row.addWidget(self.cancel_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.cancel_hint = QLabel("Готовые результаты останутся в проекте.")
        self.cancel_hint.setObjectName("muted")
        self.cancel_hint.setWordWrap(True)
        self.cancel_hint.hide()
        layout.addWidget(self.cancel_hint)

    def set_running(
        self,
        stage: str,
        elapsed: str,
        progress_fraction: float | None = None,
        detail: str = "",
        *,
        cancelling: bool = False,
        long_stage_warning: str | None = None,
    ) -> None:
        actual_stage = stage.strip() or "Подготавливаем обработку"
        self.stage.setText(actual_stage)
        self.stage.setToolTip(actual_stage)
        self.elapsed.setText(elapsed)
        self.elapsed.setVisible(bool(elapsed))
        self.elapsed_label.setVisible(bool(elapsed))
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.warning.setText(long_stage_warning or "")
        self.warning.setVisible(bool(long_stage_warning))
        self.continue_waiting_button.setVisible(bool(long_stage_warning) and not cancelling)
        self.progress.show()
        self.progress_caption.show()

        if progress_fraction is None:
            self.progress.setRange(0, 0)
            self.progress_value.setText("Считаем ход работы")
            self.progress_note.setText("Точный процент появится, когда его сообщит текущий этап.")
            self.progress_note.show()
        else:
            value = max(0, min(100, round(progress_fraction * 100)))
            self.progress.setRange(0, 100)
            self.progress.setValue(value)
            self.progress_value.setText(f"{value}%")
            self.progress_note.clear()
            self.progress_note.hide()

        self.cancel_button.setText("Останавливаем…" if cancelling else "Остановить обработку")
        self.cancel_button.setDisabled(cancelling)
        self.cancel_button.show()
        self.cancel_hint.setText(
            "Останавливаем безопасно: уже готовые результаты сохранятся."
            if cancelling else "Можно перейти к другим проектам — ход работы сохранится автоматически."
        )
        self.cancel_hint.show()

    def set_finished(self, message: str) -> None:
        self.stage.setText(message)
        self.stage.setToolTip(message)
        self.elapsed.clear()
        self.elapsed.hide()
        self.elapsed_label.hide()
        self.detail.clear()
        self.detail.hide()
        self.warning.clear()
        self.warning.hide()
        self.continue_waiting_button.hide()
        self.progress.hide()
        self.progress_caption.hide()
        self.progress_value.clear()
        self.progress_note.clear()
        self.progress_note.hide()
        self.cancel_button.hide()
        self.cancel_button.setDisabled(False)
        self.cancel_hint.hide()
