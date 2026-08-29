from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
)

from app.gui.responsive import make_label_shrinkable, set_responsive_text


class ProcessingProgress(QFrame):
    """Human-facing progress for a persisted local processing run.

    The screen receives the actual stage and timing from the existing run
    snapshot.  It deliberately does not estimate a percentage when the engine
    has not supplied one.
    """

    cancel_requested = Signal()
    continue_waiting_requested = Signal()
    retry_requested = Signal()
    # Includes the optional long-stage warning, two action buttons and wrapped
    # safety hint at the shell's 760 px logical-width regression viewport.
    # Ordinary progress fits beside the source summary and stage list on a
    # normal desktop. Hostile wrapped warnings still grow this dynamically
    # from the real layout height in ``_refresh_geometry``.
    # Keep the running surface compact enough that the real stage list remains
    # visible in the initial desktop viewport.  Wrapped recovery/warning copy
    # still grows the card from its measured layout height below.
    _BASE_MINIMUM_HEIGHT = 208
    # A persisted terminal error can contain an entire subprocess diagnostic.
    # Keep the recovery action near the top of the page while preserving the
    # complete message in the tooltip and project log.
    _FINISHED_MESSAGE_MAX_CHARS = 240

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.eyebrow = QLabel("ОБРАБОТКА")
        self.eyebrow.setObjectName("eyebrow")
        layout.addWidget(self.eyebrow)

        top = QHBoxLayout()
        top.setSpacing(12)
        stage_copy = QVBoxLayout()
        stage_copy.setSpacing(3)
        self.stage_label = QLabel("Текущий этап")
        self.stage_label.setObjectName("muted")
        self.stage = QLabel("Готово к созданию ролика")
        make_label_shrinkable(self.stage)
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
        self.progress_value.setObjectName("progressValue")
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
        make_label_shrinkable(self.progress_note)
        self.progress_note.hide()
        layout.addWidget(self.progress_note)

        self.detail = QLabel("")
        self.detail.setObjectName("muted")
        make_label_shrinkable(self.detail)
        self.detail.hide()
        layout.addWidget(self.detail)

        self.warning = QLabel("")
        self.warning.setObjectName("warning")
        make_label_shrinkable(self.warning)
        self.warning.hide()
        layout.addWidget(self.warning)

        # Start stacked so a hidden page never publishes a wide button-row
        # minimum before ProjectScreen gives it its real client width.
        action_row = QVBoxLayout()
        self._action_layout = action_row
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

        self.retry_button = QPushButton("Повторить")
        self.retry_button.setObjectName("primary")
        self.retry_button.clicked.connect(self.retry_requested)
        self.retry_button.hide()
        action_row.addWidget(self.retry_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.cancel_hint = QLabel("Готовые результаты останутся в проекте.")
        self.cancel_hint.setObjectName("muted")
        make_label_shrinkable(self.cancel_hint)
        self.cancel_hint.hide()
        layout.addWidget(self.cancel_hint)
        # Reserve the complete ordinary running/recovery surface before this
        # widget enters ProjectScreen's hidden stage stack.  Otherwise Qt can
        # cache the compact, all-optional-rows-hidden hint and position the
        # following cards through rows revealed on the first update.
        self.setMinimumHeight(self._BASE_MINIMUM_HEIGHT)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_geometry()

    def _refresh_geometry(self) -> None:
        """Keep wrapped rows inside the card and release stale narrow minima."""

        layout = self.layout()
        if layout is None:
            return
        # Visibility changes do not always invalidate a hidden page's cached
        # size hint before ProjectScreen moves it into the active workspace.
        # Recompute it synchronously so labels and actions cannot paint over
        # one another while the outer project page remains scrollable.
        self.setMinimumHeight(0)
        self.setMaximumHeight(16_777_215)
        for label in (
            self.stage,
            self.progress_note,
            self.detail,
            self.warning,
            self.cancel_hint,
        ):
            label.setMinimumHeight(0)
        self._action_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if self.width() < 720
            else QBoxLayout.Direction.LeftToRight
        )
        layout.invalidate()
        layout.activate()
        required_height = layout.totalHeightForWidth(max(1, self.width()))
        if required_height < 0:
            required_height = layout.totalSizeHint().height()
        required_height = max(
            self._BASE_MINIMUM_HEIGHT,
            required_height,
            layout.totalSizeHint().height(),
            layout.totalMinimumSize().height(),
        )
        # This surface has no useful stretch region. A fixed content height
        # prevents Qt from positioning the following stage list from a stale
        # pre-warning size hint while a wrapped warning becomes visible.
        self.setFixedHeight(required_height)
        self.updateGeometry()

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
        set_responsive_text(self.stage, actual_stage)
        self.elapsed.setText(elapsed)
        self.elapsed.setVisible(bool(elapsed))
        self.elapsed_label.setVisible(bool(elapsed))
        set_responsive_text(self.detail, detail)
        self.detail.setVisible(bool(detail))
        set_responsive_text(self.warning, long_stage_warning or "")
        self.warning.setVisible(bool(long_stage_warning))
        self.continue_waiting_button.setVisible(bool(long_stage_warning) and not cancelling)
        self.progress.show()
        self.progress_caption.show()

        if progress_fraction is None:
            self.progress.setRange(0, 0)
            self.progress_value.setText("Считаем ход работы")
            set_responsive_text(
                self.progress_note,
                "Точный процент появится, когда его сообщит текущий этап.",
            )
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
        self.retry_button.hide()
        set_responsive_text(
            self.cancel_hint,
            "Останавливаем безопасно: уже готовые результаты сохранятся."
            if cancelling else "Можно перейти к другим проектам — ход работы сохранится автоматически."
        )
        self.cancel_hint.show()
        self._refresh_geometry()

    def set_finished(self, message: str, retry_label: str | None = None) -> None:
        full_message = str(message)
        visible_message = self._bounded_finished_message(full_message)
        set_responsive_text(self.stage, visible_message)
        self.stage.setToolTip(full_message)
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
        self.retry_button.setText(retry_label or "Повторить")
        self.retry_button.setVisible(bool(retry_label))
        self.cancel_hint.hide()
        self._refresh_geometry()

    @classmethod
    def _bounded_finished_message(cls, message: str) -> str:
        if len(message) <= cls._FINISHED_MESSAGE_MAX_CHARS:
            return message
        prefix = message[:cls._FINISHED_MESSAGE_MAX_CHARS - 1].rstrip()
        # Prefer a nearby word boundary, but still cap hostile unbroken tokens.
        boundary = prefix.rfind(" ")
        if boundary >= cls._FINISHED_MESSAGE_MAX_CHARS // 2:
            prefix = prefix[:boundary].rstrip()
        return prefix + "…"
