from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QBoxLayout, QDialog, QLabel, QPushButton, QPlainTextEdit, QVBoxLayout

from app.doctor import format_report
from app.gui.responsive import make_label_shrinkable
from app.gui.viewmodels import SettingsViewModel


class OnboardingDialog(QDialog):
    def __init__(self, viewmodel: SettingsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.setWindowTitle("Добро пожаловать в Content Factory")
        self.setMinimumSize(420, 300)
        self.resize(600, 360)
        layout = QVBoxLayout(self)
        title = QLabel("Добро пожаловать")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)
        message = QLabel("Проекты и история запусков будут сохранены локально. Ключ API не нужен, чтобы открыть приложение и проверить систему.")
        make_label_shrinkable(message); layout.addWidget(message)
        self.checks = QPlainTextEdit(); self.checks.setReadOnly(True); self.checks.setMinimumHeight(120)
        self.checks.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.checks.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.checks)
        actions = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._actions_layout = actions
        self.check_button = QPushButton("Проверить систему")
        self.check_button.clicked.connect(self.viewmodel.diagnostics)
        self.continue_button = QPushButton("Перейти к проектам")
        self.continue_button.setObjectName("primary")
        self.continue_button.clicked.connect(self._finish)
        actions.addWidget(self.check_button); actions.addStretch(); actions.addWidget(self.continue_button)
        layout.addLayout(actions)
        self.viewmodel.diagnostics_ready.connect(lambda checks: self.checks.setPlainText(format_report(checks)))
        self._apply_responsive_layout()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        self._actions_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if self.width() < 620
            else QBoxLayout.Direction.LeftToRight
        )

    def _finish(self) -> None:
        self.viewmodel.settings.onboarding_completed = True
        self.viewmodel.save()
        self.accept()
