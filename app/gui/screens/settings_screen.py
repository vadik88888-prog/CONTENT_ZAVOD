from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config import load_config
from app.doctor import format_report
from app.gui.services.secure_secrets import key_configured
from app.gui.viewmodels import SettingsViewModel


class SettingsScreen(QWidget):
    """Creator-facing settings with operational controls kept secondary."""

    def __init__(self, viewmodel: SettingsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 22)
        root.setSpacing(0)
        header = QVBoxLayout()
        header.setSpacing(3)
        title = QLabel("Настройки")
        title.setObjectName("title")
        subtitle = QLabel("Управляйте хранением проектов и локальной работой приложения.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)
        root.addSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 2, 6)
        layout.setSpacing(12)

        overview = self._section("Работа на этом компьютере")
        self.system_detail = QLabel()
        self.system_detail.setObjectName("subtitle")
        self.system_detail.setWordWrap(True)
        private_note = QLabel("Проекты, исходные видео и готовые ролики не отправляются в облако из этого приложения.")
        private_note.setObjectName("muted")
        private_note.setWordWrap(True)
        overview.layout().addWidget(self.system_detail)
        overview.layout().addWidget(private_note)
        layout.addWidget(overview)

        general = self._section("Хранение проектов")
        location_hint = QLabel("Выберите папку, где Content Factory хранит проекты, историю запусков и результаты.")
        location_hint.setObjectName("muted")
        location_hint.setWordWrap(True)
        self.data_directory = QLineEdit()
        self.data_directory.setReadOnly(True)
        choose_data = QPushButton("Изменить папку")
        choose_data.clicked.connect(self._choose_data_directory)
        open_data = QPushButton("Открыть папку")
        open_data.clicked.connect(self._open_data)
        data_actions = QHBoxLayout()
        data_actions.addWidget(choose_data)
        data_actions.addWidget(open_data)
        data_actions.addStretch()
        general.layout().addWidget(location_hint)
        general.layout().addWidget(self.data_directory)
        general.layout().addLayout(data_actions)
        layout.addWidget(general)

        self.advanced_toggle = QPushButton("Расширенные настройки")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip("Параметры подключения, производительности и локального тестового режима")
        self.advanced_toggle.toggled.connect(self._set_advanced_visible)
        layout.addWidget(self.advanced_toggle)

        self.advanced_content = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_content)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(12)

        process = self._section("Подключение и производительность")
        process_hint = QLabel("Эти параметры нужны только при настройке движка или локальной технической проверке.")
        process_hint.setObjectName("muted")
        process_hint.setWordWrap(True)
        self.config_path = QLineEdit()
        self.config_path.setPlaceholderText("Использовать стандартную конфигурацию")
        self.config_path.editingFinished.connect(self._save)
        choose_config = QPushButton("Выбрать config.yaml")
        choose_config.clicked.connect(self._choose_config)
        self.device = QComboBox()
        self.device.addItem("Автоматически", "auto")
        self.device.addItem("Предпочитать GPU", "cuda")
        self.device.addItem("Использовать CPU", "cpu")
        self.device.currentIndexChanged.connect(self._save)
        self.local_test = QCheckBox("Локальный тестовый режим без внешних API")
        self.local_test.setToolTip("Использует существующие mock-провайдеры AI и озвучки только для этого приложения.")
        self.local_test.toggled.connect(self._save)
        cache_info = QLabel(f"Рабочий кэш: {self.viewmodel.services.engine_root / 'work'}")
        cache_info.setObjectName("muted")
        cache_info.setWordWrap(True)
        process.layout().addWidget(process_hint)
        process.layout().addWidget(QLabel("Конфигурация движка"))
        process.layout().addWidget(self.config_path)
        process.layout().addWidget(choose_config)
        process.layout().addWidget(QLabel("Предпочтительное устройство"))
        process.layout().addWidget(self.device)
        process.layout().addWidget(self.local_test)
        process.layout().addWidget(cache_info)
        advanced_layout.addWidget(process)

        self.ai_section = self._section("Подключённые сервисы")
        self.ai_info = QLabel()
        self.ai_info.setWordWrap(True)
        self.ai_info.setObjectName("subtitle")
        note = QLabel("Ключи не отображаются и не сохраняются в настройках или истории запусков.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        self.ai_section.layout().addWidget(self.ai_info)
        self.ai_section.layout().addWidget(note)
        advanced_layout.addWidget(self.ai_section)
        self.advanced_content.hide()
        layout.addWidget(self.advanced_content)

        self.diagnostics_toggle = QPushButton("Диагностика и поддержка")
        self.diagnostics_toggle.setCheckable(True)
        self.diagnostics_toggle.setToolTip("Проверить FFmpeg, устройство и доступность настроенных сервисов")
        self.diagnostics_toggle.toggled.connect(self._set_diagnostics_visible)
        layout.addWidget(self.diagnostics_toggle)

        self.diagnostics_content = self._section("Проверка системы")
        diagnostics_hint = QLabel("Если что-то не запускается, выполните проверку и используйте результат для поддержки.")
        diagnostics_hint.setObjectName("muted")
        diagnostics_hint.setWordWrap(True)
        self.check_button = QPushButton("Проверить систему")
        self.check_button.clicked.connect(self.viewmodel.diagnostics)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMinimumHeight(150)
        self.diagnostics_content.layout().addWidget(diagnostics_hint)
        self.diagnostics_content.layout().addWidget(self.check_button)
        self.diagnostics_content.layout().addWidget(self.diagnostics)
        self.diagnostics_content.hide()
        layout.addWidget(self.diagnostics_content)
        layout.addStretch()

        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        self.viewmodel.settings_changed.connect(self._render)
        self.viewmodel.diagnostics_ready.connect(lambda checks: self.diagnostics.setPlainText(format_report(checks)))
        self._render(self.viewmodel.settings)

    def _set_advanced_visible(self, visible: bool) -> None:
        self.advanced_content.setVisible(visible)
        self.advanced_toggle.setText("Скрыть расширенные настройки" if visible else "Расширенные настройки")

    def _set_diagnostics_visible(self, visible: bool) -> None:
        self.diagnostics_content.setVisible(visible)
        self.diagnostics_toggle.setText("Скрыть диагностику" if visible else "Диагностика и поддержка")

    def _render(self, settings) -> None:
        self.data_directory.setText(settings.data_directory)
        self.config_path.setText(settings.config_path or "")
        index = self.device.findData(settings.device_preference)
        self.device.blockSignals(True)
        self.device.setCurrentIndex(max(index, 0))
        self.device.blockSignals(False)
        self.local_test.blockSignals(True)
        self.local_test.setChecked(settings.local_test_mode)
        self.local_test.blockSignals(False)
        self.system_detail.setText(
            "Локальный тестовый режим включён." if settings.local_test_mode
            else "Локальная обработка готова к работе."
        )
        config_path = Path(settings.config_path) if settings.config_path else self.viewmodel.services.engine_root / "config.example.yaml"
        try:
            config = load_config(config_path)
            key = "настроен" if key_configured(config.ai.provider, self.viewmodel.services.engine_root) else "не настроен"
            self.ai_info.setText(
                f"AI: {config.ai.provider} · {config.ai.model}\n"
                f"Озвучка: {config.tts.provider} · {config.tts.model}\n"
                f"Статус ключа: {key}"
            )
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
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Конфигурация",
            self.config_path.text(),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self.config_path.setText(path)
            self._save()

    def _open_data(self) -> None:
        path = Path(self.viewmodel.settings.data_directory)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _section(title: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(17, 15, 17, 15)
        layout.setSpacing(9)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(heading)
        return frame
