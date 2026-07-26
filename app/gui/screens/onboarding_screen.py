from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit, QVBoxLayout

from app.doctor import format_report
from app.gui.viewmodels import SettingsViewModel


class OnboardingDialog(QDialog):
    def __init__(self, viewmodel: SettingsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.viewmodel = viewmodel
        self.setWindowTitle("Добро пожаловать в Content Factory")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        title = QLabel("Добро пожаловать")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)
        message = QLabel("Проекты и история запусков будут сохранены локально. Ключ API не нужен, чтобы открыть приложение и проверить систему.")
        message.setWordWrap(True); layout.addWidget(message)
        self.checks = QPlainTextEdit(); self.checks.setReadOnly(True); self.checks.setMinimumHeight(160)
        layout.addWidget(self.checks)
        actions = QHBoxLayout()
        check = QPushButton("Проверить систему")
        check.clicked.connect(self.viewmodel.diagnostics)
        continue_button = QPushButton("Перейти к проектам")
        continue_button.setObjectName("primary")
        continue_button.clicked.connect(self._finish)
        actions.addWidget(check); actions.addStretch(); actions.addWidget(continue_button)
        layout.addLayout(actions)
        self.viewmodel.diagnostics_ready.connect(lambda checks: self.checks.setPlainText(format_report(checks)))

    def _finish(self) -> None:
        self.viewmodel.settings.onboarding_completed = True
        self.viewmodel.save()
        self.accept()
