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

from app import __version__
from app.doctor import format_report, summarize_checks
from app.gui.responsive import make_label_shrinkable, set_responsive_text
from app.gui.viewmodels import SettingsViewModel
from app.secure_secrets import key_configured


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
        subtitle = QLabel("Основные настройки, проверка системы и помощь — в одном месте.")
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

        self.api_section = self._section("API-ключ")
        self.api_status = QLabel()
        self.api_status.setObjectName("subtitle")
        make_label_shrinkable(self.api_status)
        key_note = QLabel("Значение ключа остаётся скрытым и не сохраняется в настройках или истории запусков.")
        key_note.setObjectName("muted")
        make_label_shrinkable(key_note)
        self.key_setup_toggle = QPushButton("Настроить API-ключ")
        self.key_setup_toggle.setCheckable(True)
        self.key_setup_toggle.toggled.connect(self._set_api_setup_visible)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Новый API-ключ — значение останется скрытым")
        self.api_key.setClearButtonEnabled(True)
        self.save_key_button = QPushButton("Сохранить ключ локально")
        self.save_key_button.clicked.connect(self._save_api_key)
        self.api_key_result = QLabel()
        self.api_key_result.setObjectName("muted")
        make_label_shrinkable(self.api_key_result)
        self.api_section.layout().addWidget(self.api_status)
        self.api_section.layout().addWidget(key_note)
        self.api_section.layout().addWidget(self.key_setup_toggle)
        self.api_section.layout().addWidget(self.api_key)
        self.api_section.layout().addWidget(self.save_key_button)
        self.api_section.layout().addWidget(self.api_key_result)
        layout.addWidget(self.api_section)

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

        diagnostics = self._section("Проверка системы")
        diagnostics_hint = QLabel("Проверьте, что всё готово к обработке видео. Подробности доступны в расширенных настройках.")
        diagnostics_hint.setObjectName("muted")
        make_label_shrinkable(diagnostics_hint)
        self.check_button = QPushButton("Проверить систему")
        self.check_button.clicked.connect(self.viewmodel.diagnostics)
        self.system_detail = QLabel("Проверка ещё не запускалась.")
        self.system_detail.setObjectName("subtitle")
        make_label_shrinkable(self.system_detail)
        diagnostics.layout().addWidget(diagnostics_hint)
        diagnostics.layout().addWidget(self.check_button)
        diagnostics.layout().addWidget(self.system_detail)
        layout.addWidget(diagnostics)

        feedback = self._section("Данные для улучшения")
        feedback_hint = QLabel(
            "Создаёт один ZIP с событиями выбора и коротким summary. "
            "Ключи, исходные видео и полный transcript в него не попадают."
        )
        feedback_hint.setObjectName("muted")
        feedback_hint.setWordWrap(True)
        self.feedback_export_button = QPushButton("Экспортировать данные для улучшения")
        self.feedback_export_button.clicked.connect(self._export_feedback)
        self.feedback_export_status = QLabel()
        self.feedback_export_status.setObjectName("muted")
        self.feedback_export_status.setWordWrap(True)
        feedback.layout().addWidget(feedback_hint)
        feedback.layout().addWidget(self.feedback_export_button)
        feedback.layout().addWidget(self.feedback_export_status)
        layout.addWidget(feedback)

        version = QLabel(f"Content Factory {__version__}")
        version.setObjectName("muted")
        layout.addWidget(version)

        self.advanced_toggle = QPushButton("Расширенные настройки")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip("Параметры движка, производительности, тестового режима и подробной диагностики")
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

        raw_diagnostics = self._section("Подробности проверки")
        raw_hint = QLabel("Технический отчёт для самостоятельной диагностики или передачи в поддержку.")
        raw_hint.setObjectName("muted")
        make_label_shrinkable(raw_hint)
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMinimumHeight(150)
        self.diagnostics.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.diagnostics.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        raw_diagnostics.layout().addWidget(raw_hint)
        raw_diagnostics.layout().addWidget(self.diagnostics)
        advanced_layout.addWidget(raw_diagnostics)
        self.advanced_content.hide()
        layout.addWidget(self.advanced_content)
        layout.addStretch()

        scroll.setWidget(host)
        root.addWidget(scroll, 1)

        self.viewmodel.settings_changed.connect(self._render)
        self.viewmodel.diagnostics_started.connect(self._diagnostics_started)
        self.viewmodel.diagnostics_ready.connect(self._diagnostics_ready)
        self._render(self.viewmodel.settings)

    def _set_advanced_visible(self, visible: bool) -> None:
        self.advanced_content.setVisible(visible)
        self.advanced_toggle.setText("Скрыть расширенные настройки" if visible else "Расширенные настройки")

    def _set_api_setup_visible(self, visible: bool) -> None:
        configurable = self.viewmodel.ai_provider() in {"openai", "gemini"}
        show = configurable and visible
        self.api_key.setVisible(show)
        self.save_key_button.setVisible(show)
        self.key_setup_toggle.setText("Скрыть настройку ключа" if show else "Настроить API-ключ")

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
        provider = self.viewmodel.ai_provider()
        if provider is None:
            self.api_status.setText("Статус ключа недоступен. Проверьте расширенные настройки.")
        elif provider == "mock":
            self.api_status.setText("Статус ключа: не требуется в локальном тестовом режиме.")
        elif key_configured(provider, self.viewmodel.services.system.data_root):
            self.api_status.setText("Статус ключа: настроен.")
        else:
            self.api_status.setText("Статус ключа: не настроен.")
        configurable = provider in {"openai", "gemini"}
        self.key_setup_toggle.setVisible(configurable)
        if not configurable:
            self.key_setup_toggle.setChecked(False)
        self._set_api_setup_visible(self.key_setup_toggle.isChecked())

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

    def _save_api_key(self) -> None:
        result = self.viewmodel.save_api_key(self.api_key.text())
        self.api_key.clear()
        self.api_key_result.setText(result.message)

    def _export_feedback(self) -> None:
        self.feedback_export_button.setEnabled(False)
        try:
            result = self.viewmodel.export_feedback()
        except OSError:
            self.feedback_export_status.setText("Не удалось создать ZIP. Проверьте доступ к папке данных и повторите.")
        else:
            self.feedback_export_status.setText(
                f"Создан {result.path.name}: {result.event_count} событий из {result.project_count} проектов. "
                "Его можно отправить для улучшения Content Factory."
            )
            self.feedback_export_status.setToolTip(str(result.path))
        finally:
            self.feedback_export_button.setEnabled(True)

    def _diagnostics_started(self) -> None:
        self.check_button.setEnabled(False)
        self.check_button.setText("Проверяем…")
        self.system_detail.setText("Проверка выполняется в фоне. Интерфейс остаётся доступным.")
        self.diagnostics.setPlainText("Диагностика выполняется в фоне; интерфейс остаётся доступным.")

    def _diagnostics_ready(self, checks) -> None:
        summary = summarize_checks(checks)
        self.check_button.setEnabled(True)
        self.check_button.setText("Проверить снова")
        self.system_detail.setText(summary.detail)
        self.diagnostics.setPlainText(format_report(checks))

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
