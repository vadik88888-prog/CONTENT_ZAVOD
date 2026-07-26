from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from app.config import load_config
from app.doctor import format_report
from app.gui.services.secure_secrets import key_configured
from app.gui.viewmodels import SettingsViewModel


class SettingsScreen(QWidget):
    def __init__(self, viewmodel: SettingsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        root = QVBoxLayout(self); root.setContentsMargins(34, 30, 34, 30)
        title = QLabel("Настройки"); title.setObjectName("title"); root.addWidget(title)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        host = QWidget(); layout = QVBoxLayout(host); layout.setSpacing(14)
        general = self._section("Основное")
        self.data_directory = QLineEdit()
        self.data_directory.setReadOnly(True)
        choose_data = QPushButton("Выбрать папку данных")
        choose_data.clicked.connect(self._choose_data_directory)
        open_data = QPushButton("Открыть папку данных")
        open_data.clicked.connect(self._open_data)
        general.layout().addWidget(QLabel("Папка данных приложения")); general.layout().addWidget(self.data_directory)
        row = QHBoxLayout(); row.addWidget(choose_data); row.addWidget(open_data); row.addStretch(); general.layout().addLayout(row)
        general.layout().addWidget(QLabel("Тема: тёмная"))
        layout.addWidget(general)
        process = self._section("Обработка")
        self.config_path = QLineEdit()
        self.config_path.editingFinished.connect(self._save)
        choose_config = QPushButton("Выбрать config.yaml")
        choose_config.clicked.connect(self._choose_config)
        process.layout().addWidget(QLabel("Конфигурация движка")); process.layout().addWidget(self.config_path); process.layout().addWidget(choose_config)
        process.layout().addWidget(QLabel("Использование GPU"))
        self.device = QComboBox()
        self.device.addItem("Авто", "auto")
        self.device.addItem("Вкл.", "cuda")
        self.device.addItem("Выкл.", "cpu")
        self.device.currentIndexChanged.connect(self._save)
        process.layout().addWidget(self.device)
        self.local_test = QCheckBox("Локальный тестовый режим без внешних API")
        self.local_test.setToolTip("Использует существующие mock-провайдеры AI и озвучки только для этого приложения.")
        self.local_test.toggled.connect(self._save)
        process.layout().addWidget(self.local_test)
        cache_info = QLabel(f"Кэш: {self.viewmodel.services.engine_root / 'work'}")
        cache_info.setObjectName("muted")
        cache_info.setWordWrap(True)
        process.layout().addWidget(cache_info)
        layout.addWidget(process)
        self.ai_section = self._section("AI и озвучка")
        self.ai_info = QLabel(); self.ai_info.setWordWrap(True); self.ai_info.setObjectName("subtitle")
        self.ai_section.layout().addWidget(self.ai_info)
        note = QLabel("Ключи не отображаются и не сохраняются в настройках или истории запусков.")
        note.setObjectName("muted"); note.setWordWrap(True); self.ai_section.layout().addWidget(note)
        layout.addWidget(self.ai_section)
        diagnostics = self._section("Диагностика")
        check = QPushButton("Проверить систему"); check.clicked.connect(self.viewmodel.diagnostics)
        self.diagnostics = QPlainTextEdit(); self.diagnostics.setReadOnly(True); self.diagnostics.setMinimumHeight(160)
        diagnostics.layout().addWidget(check); diagnostics.layout().addWidget(self.diagnostics)
        layout.addWidget(diagnostics); layout.addStretch()
        scroll.setWidget(host); root.addWidget(scroll, 1)
        self.viewmodel.settings_changed.connect(self._render)
        self.viewmodel.diagnostics_ready.connect(lambda checks: self.diagnostics.setPlainText(format_report(checks)))
        self._render(self.viewmodel.settings)

    def _render(self, settings) -> None:
        self.data_directory.setText(settings.data_directory)
        self.config_path.setText(settings.config_path or "")
        index = self.device.findData(settings.device_preference)
        self.device.blockSignals(True); self.device.setCurrentIndex(max(index, 0)); self.device.blockSignals(False)
        self.local_test.blockSignals(True); self.local_test.setChecked(settings.local_test_mode); self.local_test.blockSignals(False)
        config_path = Path(settings.config_path) if settings.config_path else self.viewmodel.services.engine_root / "config.example.yaml"
        try:
            config = load_config(config_path)
            key = "настроен" if key_configured(config.ai.provider, self.viewmodel.services.engine_root) else "не настроен"
            self.ai_info.setText(f"AI: {config.ai.provider} · {config.ai.model}\nОзвучка: {config.tts.provider} · {config.tts.model}\nСтатус ключа: {key}")
        except Exception:
            self.ai_info.setText("Выберите корректный файл конфигурации, чтобы увидеть используемые параметры.")

    def _save(self) -> None:
        self.viewmodel.settings.config_path = self.config_path.text().strip() or None
        self.viewmodel.settings.device_preference = str(self.device.currentData())
        self.viewmodel.save()

    def _choose_data_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Папка данных", self.data_directory.text())
        if directory:
            self.viewmodel.set_data_directory(directory)

    def _choose_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Конфигурация", self.config_path.text(), "YAML (*.yaml *.yml)")
        if path:
            self.config_path.setText(path); self._save()

    def _open_data(self) -> None:
        path = Path(self.viewmodel.settings.data_directory)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _section(title: str) -> QFrame:
        frame = QFrame(); frame.setObjectName("card")
        layout = QVBoxLayout(frame); layout.setContentsMargins(18, 16, 18, 16)
        heading = QLabel(title); heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(heading)
        return frame
