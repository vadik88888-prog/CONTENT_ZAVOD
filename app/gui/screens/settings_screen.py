from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.config import load_config
from app.doctor import format_report
from app.gui.responsive import make_label_shrinkable, set_responsive_text
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
        self.content_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 2, 6)
        layout.setSpacing(12)

        overview = self._section("Работа на этом компьютере")
        self.system_detail = QLabel()
        self.system_detail.setObjectName("subtitle")
        make_label_shrinkable(self.system_detail)
        private_note = QLabel("Проекты, исходные видео и готовые ролики не отправляются в облако из этого приложения.")
        private_note.setObjectName("muted")
        make_label_shrinkable(private_note)
        overview.layout().addWidget(self.system_detail)
        overview.layout().addWidget(private_note)
        layout.addWidget(overview)

        general = self._section("Хранение проектов")
        location_hint = QLabel("Выберите папку, где Content Factory хранит проекты, историю запусков и результаты.")
        location_hint.setObjectName("muted")
        make_label_shrinkable(location_hint)
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
        self.advanced_content.setMinimumWidth(0)
        advanced_layout = QVBoxLayout(self.advanced_content)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(12)

        process = self._section("Подключение и производительность")
        process_hint = QLabel("Эти параметры нужны только при настройке движка или локальной технической проверке.")
        process_hint.setObjectName("muted")
        make_label_shrinkable(process_hint)
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
        self.local_test = QCheckBox("Локальный тестовый режим")
        self.local_test.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.local_test.setToolTip("Работать без внешних API, используя локальные тестовые провайдеры.")
        self.local_test.toggled.connect(self._save)
        local_test_hint = QLabel("Без внешних API; предназначено только для локальной технической проверки.")
        local_test_hint.setObjectName("muted")
        make_label_shrinkable(local_test_hint)
        cache_info = QLabel()
        cache_info.setObjectName("muted")
        make_label_shrinkable(cache_info)
        set_responsive_text(cache_info, f"Рабочий кэш: {self.viewmodel.services.engine_root / 'work'}")
        process.layout().addWidget(process_hint)
        process.layout().addWidget(QLabel("Конфигурация движка"))
        process.layout().addWidget(self.config_path)
        process.layout().addWidget(choose_config)
        process.layout().addWidget(QLabel("Предпочтительное устройство"))
        process.layout().addWidget(self.device)
        process.layout().addWidget(self.local_test)
        process.layout().addWidget(local_test_hint)
        process.layout().addWidget(cache_info)
        advanced_layout.addWidget(process)

        self.ai_section = self._section("Подключённые сервисы")
        self.ai_info = QLabel()
        make_label_shrinkable(self.ai_info)
        self.ai_info.setObjectName("subtitle")
        note = QLabel("Ключи не отображаются и не сохраняются в настройках или истории запусков.")
        note.setObjectName("muted")
        make_label_shrinkable(note)
        self.ai_section.layout().addWidget(self.ai_info)
        self.ai_section.layout().addWidget(note)
        advanced_layout.addWidget(self.ai_section)
        self.advanced_content.hide()
        layout.addWidget(self.advanced_content)

        self.diagnostics_toggle = QPushButton("Диагностика и поддержка")
        self.diagnostics_toggle.setCheckable(True)
        self.diagnostics_toggle.setToolTip("Проверить компоненты обработки видео и доступность настроенных сервисов")
        self.diagnostics_toggle.toggled.connect(self._set_diagnostics_visible)
        layout.addWidget(self.diagnostics_toggle)

        self.diagnostics_content = self._section("Проверка системы")
        diagnostics_hint = QLabel("Если что-то не запускается, выполните проверку и используйте результат для поддержки.")
        diagnostics_hint.setObjectName("muted")
        make_label_shrinkable(diagnostics_hint)
        self.check_button = QPushButton("Проверить систему")
        self.check_button.clicked.connect(self.viewmodel.diagnostics)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMinimumHeight(150)
        self.diagnostics.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.diagnostics.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
        self.data_directory.setToolTip(settings.data_directory)
        self.config_path.setText(settings.config_path or "")
        self.config_path.setToolTip(settings.config_path or "Используется стандартная конфигурация")
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
            set_responsive_text(
                self.ai_info,
                f"AI: {config.ai.provider} · {config.ai.model}\n"
                f"Озвучка: {config.tts.provider} · {config.tts.model}\n"
                f"Статус ключа: {key}",
            )
        except Exception:
            set_responsive_text(
                self.ai_info,
                "Выберите корректный файл конфигурации, чтобы увидеть используемые параметры.",
            )

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
        make_label_shrinkable(heading)
        layout.addWidget(heading)
        return frame
