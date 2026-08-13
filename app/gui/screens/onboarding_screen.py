from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
)

from app.doctor import DoctorReadiness, format_report, summarize_checks
from app.gui.responsive import make_label_shrinkable
from app.gui.viewmodels import SettingsViewModel


class OnboardingDialog(QDialog):
    def __init__(self, viewmodel: SettingsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.viewmodel = viewmodel
        self._readiness: DoctorReadiness | None = None
        self.setWindowTitle("Добро пожаловать в Content Factory")
        self.setMinimumSize(420, 360)
        self.resize(620, 460)
        layout = QVBoxLayout(self)
        title = QLabel("Добро пожаловать")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)
        message = QLabel("Проекты и история запусков будут сохранены локально. Для реальной AI-обработки нужен подтверждённый ключ; локальный тестовый режим доступен без него.")
        make_label_shrinkable(message); layout.addWidget(message)

        self.api_label = QLabel()
        self.api_label.setObjectName("muted")
        make_label_shrinkable(self.api_label)
        self.local_test = QCheckBox("Использовать локальный тестовый режим без API")
        self.local_test.setChecked(self.viewmodel.settings.local_test_mode)
        self.local_test.setToolTip("Результаты mock-режима предназначены только для проверки интерфейса и процесса.")
        self.local_test.toggled.connect(self._set_local_test_mode)
        self.api_setup_button = QPushButton("Настроить API-ключ")
        self.api_setup_button.setCheckable(True)
        self.api_setup_button.toggled.connect(self._set_api_setup_visible)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("API-ключ — значение останется скрытым")
        self.api_key.setClearButtonEnabled(True)
        self.save_key_button = QPushButton("Сохранить ключ локально")
        self.save_key_button.clicked.connect(self._save_api_key)
        self.api_result = QLabel()
        self.api_result.setObjectName("muted")
        make_label_shrinkable(self.api_result)
        layout.addWidget(self.api_label)
        layout.addWidget(self.local_test)
        layout.addWidget(self.api_setup_button)
        layout.addWidget(self.api_key)
        layout.addWidget(self.save_key_button)
        layout.addWidget(self.api_result)

        self.checks = QPlainTextEdit(); self.checks.setReadOnly(True); self.checks.setMinimumHeight(80)
        self.checks.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.checks.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.checks)
        actions = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._actions_layout = actions
        self.check_button = QPushButton("Проверить систему")
        self.check_button.clicked.connect(self.viewmodel.diagnostics)
        self.continue_button = QPushButton("Перейти к проектам")
        self.continue_button.setObjectName("primary")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self._finish)
        actions.addWidget(self.check_button); actions.addStretch(); actions.addWidget(self.continue_button)
        layout.addLayout(actions)
        self.viewmodel.diagnostics_started.connect(self._diagnostics_started)
        self.viewmodel.diagnostics_ready.connect(self._diagnostics_ready)
        self._render_provider()
        self._apply_responsive_layout()
        QTimer.singleShot(0, self.viewmodel.diagnostics)

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
        if self._readiness in {None, DoctorReadiness.SETUP_REQUIRED}:
            self.api_result.setText("Сначала исправьте BLOCKING-проблемы и повторите проверку.")
            return
        self.viewmodel.settings.onboarding_completed = True
        self.viewmodel.save()
        self.accept()

    def _diagnostics_started(self) -> None:
        self.check_button.setEnabled(False)
        self.check_button.setText("Проверяем…")
        self.continue_button.setEnabled(False)
        self.checks.setPlainText("Проверяем обязательные компоненты. Окно можно перемещать — приложение не зависло.")

    def _diagnostics_ready(self, checks) -> None:
        summary = summarize_checks(checks)
        self._readiness = summary.readiness
        self.checks.setPlainText(format_report(checks))
        self.check_button.setText("Проверить снова")
        self.check_button.setEnabled(True)
        self.continue_button.setEnabled(summary.readiness != DoctorReadiness.SETUP_REQUIRED)
        if summary.readiness == DoctorReadiness.SETUP_REQUIRED:
            self.api_result.setText("Требуется настройка: выполните действия из BLOCKING-пунктов.")
        elif summary.readiness == DoctorReadiness.LIMITED:
            self.api_result.setText("Можно продолжить с ограничениями; WARNING не блокирует работу.")
        else:
            self.api_result.setText("Система готова к работе.")

    def _render_provider(self) -> None:
        provider = self.viewmodel.ai_provider()
        configurable = provider in {"openai", "gemini"}
        name = provider.title() if configurable and provider is not None else "локальный тестовый режим"
        if configurable:
            self.api_label.setText(
                f"AI: {name}. Для реальной обработки требуется подтверждённый ключ; значение останется скрытым."
            )
        else:
            self.api_label.setText(
                f"AI: {name}. Ключ не требуется; результаты не являются production-ready."
            )
        self.api_setup_button.setVisible(configurable)
        self._set_api_setup_visible(configurable and self.api_setup_button.isChecked())

    def _set_local_test_mode(self, enabled: bool) -> None:
        self.viewmodel.settings.local_test_mode = enabled
        self._readiness = None
        self._render_provider()
        self.viewmodel.diagnostics()

    def _save_api_key(self) -> None:
        result = self.viewmodel.save_api_key(self.api_key.text())
        self.api_key.clear()
        self.api_result.setText(result.message)

    def _set_api_setup_visible(self, visible: bool) -> None:
        configurable = self.viewmodel.ai_provider() in {"openai", "gemini"}
        show = configurable and visible
        self.api_key.setVisible(show)
        self.save_key_button.setVisible(show)
        self.api_setup_button.setText("Скрыть настройку ключа" if show else "Настроить API-ключ")
