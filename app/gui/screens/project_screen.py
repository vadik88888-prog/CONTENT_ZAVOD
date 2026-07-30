from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from app.gui.components import CandidateThumbnailLoader, ProcessingProgress, VideoPreview
from app.gui.models import DesktopProject, ProcessingSnapshot, ProjectRun
from app.gui.viewmodels import ProjectViewModel
from app.utils import format_seconds, read_json


_STATUS = {
    "new": "Источник выбран", "source_ready": "Готов к настройке", "analyzing": "Ищем моменты",
    "analysis_ready": "Моменты готовы", "reviewing_candidates": "Выбор моментов",
    "rendering_selected": "Создаём готовые ролики", "partially_rendered": "Готово частично",
    "draft": "Черновик", "ready": "Готов", "queued": "Ожидает", "processing": "Создаём ролик",
    "completed": "Готово", "completed_with_warnings": "Готово с предупреждениями",
    "failed": "Ошибка", "cancelled": "Отменено", "interrupted": "Прервано",
}

_FLOW_STEPS = (
    ("source", "Источник"),
    ("download", "Загрузка"),
    ("settings", "Настройка"),
    ("processing", "Обработка"),
    ("candidates", "Моменты"),
    ("drafts", "Черновики"),
    ("finished", "Готовые ролики"),
)
_FLOW_STEP_INDEX = {name: index for index, (name, _label) in enumerate(_FLOW_STEPS, start=1)}
_FLOW_STEP_LABELS = dict(_FLOW_STEPS)


class ProjectScreen(QWidget):
    back_requested = Signal()

    def __init__(self, viewmodel: ProjectViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        self.project: DesktopProject | None = None
        self.runs: list[ProjectRun] = []
        self._active_candidate_id: str | None = None
        self._candidate_thumbnail_labels: dict[str, list[QLabel]] = {}
        self._candidate_cards: dict[str, QFrame] = {}
        self._flow_step = "settings"
        self._thumbnail_loader = CandidateThumbnailLoader(self)
        self._thumbnail_loader.thumbnail_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.thumbnail_unavailable.connect(self._thumbnail_unavailable)
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 26, 34, 30)
        header = QHBoxLayout()
        back = QPushButton("← Проекты")
        back.clicked.connect(self.back_requested)
        self.title = QLabel("Проект")
        self.title.setObjectName("title")
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.settings_toggle = QPushButton("Дополнительно")
        self.settings_toggle.setCheckable(True)
        self.folder = QPushButton("Открыть папку")
        self.folder.clicked.connect(self._open_project_folder)
        header.addWidget(back)
        header.addWidget(self.title)
        header.addStretch()
        header.addWidget(self.status)
        header.addWidget(self.settings_toggle)
        header.addWidget(self.folder)
        root.addLayout(header)
        self.autosave = QLabel("Изменения сохраняются автоматически")
        self.autosave.setObjectName("muted")
        root.addWidget(self.autosave)
        self.flow_card = QFrame()
        self.flow_card.setObjectName("card")
        flow_layout = QVBoxLayout(self.flow_card)
        flow_layout.setContentsMargins(14, 10, 14, 10)
        self.flow_position = QLabel("Шаг 3 из 7")
        self.flow_position.setObjectName("muted")
        self.flow_title = QLabel("Настройка")
        self.flow_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.flow_hint = QLabel()
        self.flow_hint.setObjectName("muted")
        self.flow_hint.setWordWrap(True)
        self.flow_route = QLabel("Источник  →  Загрузка  →  Настройка  →  Обработка  →  Моменты  →  Черновики  →  Готовые ролики")
        self.flow_route.setObjectName("muted")
        self.flow_route.setWordWrap(True)
        flow_layout.addWidget(self.flow_position)
        flow_layout.addWidget(self.flow_title)
        flow_layout.addWidget(self.flow_hint)
        flow_layout.addWidget(self.flow_route)
        root.addWidget(self.flow_card)
        body = QHBoxLayout()
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_host = QWidget()
        left = QVBoxLayout(self.content_host)
        left.setContentsMargins(0, 0, 0, 0)
        self.preview = VideoPreview()
        self.preview.preview_ready.connect(self._focus_preview_player)
        left.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)
        self.download_card = self._card("Загрузка")
        self.download_source = QLabel()
        self.download_source.setObjectName("muted")
        self.download_source.setWordWrap(True)
        self.download_button = QPushButton("Скачать видео")
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self.viewmodel.start_download)
        self.download_card.layout().addWidget(self.download_source)
        self.download_card.layout().addWidget(self.download_button)
        left.addWidget(self.download_card)
        self.setup_card = self._card("Настройка")
        self.setup_card.setObjectName("setup-card")
        setup_layout = self.setup_card.layout()
        self.setup_source = QLabel()
        self.setup_source.setObjectName("muted")
        self.setup_source.setWordWrap(True)
        setup_layout.addWidget(self.setup_source)
        setup_layout.addWidget(QLabel("Как искать моменты"))
        self.setup_processing_mode = QComboBox()
        self.setup_processing_mode.addItem("Быстрее — для разговорных видео", "fast")
        self.setup_processing_mode.addItem("Сбалансировано", "standard")
        self.setup_processing_mode.addItem("Тщательнее — для динамичных видео", "maximum")
        self.setup_processing_mode.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("processing_mode", str(self.setup_processing_mode.currentData()))
        )
        setup_layout.addWidget(self.setup_processing_mode)
        self.setup_mode_help = QLabel()
        self.setup_mode_help.setObjectName("muted")
        self.setup_mode_help.setWordWrap(True)
        setup_layout.addWidget(self.setup_mode_help)
        setup_layout.addWidget(QLabel("Учитывать события в кадре"))
        self.setup_deep_analysis = QComboBox()
        self.setup_deep_analysis.addItem("Авто — выбрать по содержанию", "auto")
        self.setup_deep_analysis.addItem("Включить", "on")
        self.setup_deep_analysis.addItem("Выключить", "off")
        self.setup_deep_analysis.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("deep_analysis", str(self.setup_deep_analysis.currentData()))
        )
        setup_layout.addWidget(self.setup_deep_analysis)
        self.setup_deep_help = QLabel()
        self.setup_deep_help.setObjectName("muted")
        self.setup_deep_help.setWordWrap(True)
        setup_layout.addWidget(self.setup_deep_help)
        setup_layout.addWidget(QLabel("Где будет показываться ролик"))
        self.setup_platform = QComboBox()
        self.setup_platform.addItem("TikTok", "tiktok")
        self.setup_platform.addItem("Instagram Reels", "reels")
        self.setup_platform.addItem("YouTube Shorts", "shorts")
        self.setup_platform.addItem("Любая вертикальная лента", "universal")
        self.setup_platform.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("platform", str(self.setup_platform.currentData()))
        )
        setup_layout.addWidget(self.setup_platform)
        self.setup_platform_help = QLabel("Размер и поля ролика будут подготовлены для выбранного места.")
        self.setup_platform_help.setObjectName("muted")
        self.setup_platform_help.setWordWrap(True)
        setup_layout.addWidget(self.setup_platform_help)
        self.setup_estimate = QLabel()
        self.setup_estimate.setObjectName("muted")
        self.setup_estimate.setWordWrap(True)
        setup_layout.addWidget(self.setup_estimate)
        self.setup_change = QLabel()
        self.setup_change.setObjectName("muted")
        self.setup_change.setWordWrap(True)
        setup_layout.addWidget(self.setup_change)
        self.setup_advanced_toggle = QPushButton("Дополнительные настройки")
        self.setup_advanced_toggle.setCheckable(True)
        setup_layout.addWidget(self.setup_advanced_toggle)
        self.setup_start_button = QPushButton("Начать поиск моментов")
        self.setup_start_button.setObjectName("primary")
        self.setup_start_button.clicked.connect(self._primary_action)
        setup_layout.addWidget(self.setup_start_button)
        left.addWidget(self.setup_card)
        self.next_step = self._card("Следующий шаг")
        self.next_step_text = QLabel()
        self.next_step_text.setObjectName("muted")
        self.next_step_text.setWordWrap(True)
        self.next_step.layout().addWidget(self.next_step_text)
        left.addWidget(self.next_step)
        self.metadata = self._card("Сведения о видео")
        left.addWidget(self.metadata)
        self.estimate = self._card("Предварительная оценка")
        left.addWidget(self.estimate)
        self.content_summary = self._card("Что найдено в видео")
        self._replace_card_text(self.content_summary, ["Рекомендация появится после завершения анализа."])
        left.addWidget(self.content_summary)
        self.candidate_detail = self._card("Просмотр момента")
        self._replace_card_text(self.candidate_detail, ["Выберите момент в списке, чтобы просмотреть исходный фрагмент."])
        left.addWidget(self.candidate_detail)
        self.candidate_review = self._card("Моменты")
        # The primary action scrolls directly to this workspace.  Make the
        # destination focusable as well, so keyboard focus follows the review
        # action instead of remaining on the header button.
        self.candidate_review.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.candidate_review_layout = self.candidate_review.layout()
        self._candidate_selection_buttons: dict[str, QPushButton] = {}
        self._candidate_filter = "all"
        self._candidate_sort = "recommendation"
        self.workflow_hint = QLabel()
        self.workflow_hint.setObjectName("muted")
        self.workflow_hint.setWordWrap(True)
        self.draft_button = QPushButton("Создать черновики")
        self.draft_button.setObjectName("primary")
        self.draft_button.clicked.connect(self._draft_action)
        self.production_button = QPushButton("Создать готовые ролики")
        self.production_button.setObjectName("primary")
        self.production_button.clicked.connect(self._confirm_production_render)
        left.addWidget(self.candidate_review)
        self.progress = ProcessingProgress()
        self.progress.cancel_requested.connect(self.viewmodel.cancel)
        left.addWidget(self.progress)
        self.history_title = QLabel("История запусков")
        self.history_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        left.addWidget(self.history_title)
        self.history = QScrollArea()
        self.history.setWidgetResizable(True)
        self.history_host = QWidget()
        self.history_layout = QVBoxLayout(self.history_host)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.addStretch()
        self.history.setWidget(self.history_host)
        self.history.setMinimumHeight(180)
        left.addWidget(self.history, 1)
        self.content_scroll.setWidget(self.content_host)
        body.addWidget(self.content_scroll, 3)
        panel = QFrame()
        self.settings_panel = panel
        panel.setObjectName("card")
        panel.setMaximumWidth(300)
        settings = QVBoxLayout(panel)
        settings.setContentsMargins(18, 18, 18, 18)
        heading = QLabel("Как подготовить ролики")
        heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        settings.addWidget(heading)
        settings.addWidget(QLabel("Режим обработки"))
        self.processing_mode = QComboBox()
        self.processing_mode.addItem("Быстро", "fast")
        self.processing_mode.addItem("Стандарт", "standard")
        self.processing_mode.addItem("Максимальное качество", "maximum")
        self.processing_mode.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(processing_mode=str(self.processing_mode.currentData()))
        )
        settings.addWidget(self.processing_mode)
        settings.addWidget(QLabel("Глубокий анализ видео"))
        self.deep_analysis = QComboBox()
        self.deep_analysis.addItem("Автоматически", "auto")
        self.deep_analysis.addItem("Включён", "on")
        self.deep_analysis.addItem("Выключен", "off")
        self.deep_analysis.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(deep_analysis=str(self.deep_analysis.currentData()))
        )
        settings.addWidget(self.deep_analysis)
        settings.addWidget(QLabel("Площадка"))
        self.platform = QComboBox()
        self.platform.addItem("TikTok", "tiktok")
        self.platform.addItem("Instagram Reels", "reels")
        self.platform.addItem("YouTube Shorts", "shorts")
        self.platform.addItem("Универсальный вертикальный", "universal")
        self.platform.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(platform=str(self.platform.currentData()))
        )
        settings.addWidget(self.platform)
        settings.addWidget(QLabel("Количество роликов"))
        self.clip_count = QComboBox()
        for label, value in (("Авто", "auto"), ("1 ролик", "1"), ("3 ролика", "3"), ("5 роликов", "5")):
            self.clip_count.addItem(label, value)
        self.clip_count.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(clip_count=str(self.clip_count.currentData()))
        )
        settings.addWidget(self.clip_count)
        settings.addWidget(QLabel("Аудио"))
        self.audio_mode = QComboBox()
        self.audio_mode.addItem("Исходная речь", "original")
        self.audio_mode.addItem("Исходная речь, улучшить звук", "original_enhanced")
        self.audio_mode.addItem("Озвучка", "voiceover")
        self.audio_mode.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(audio_mode=str(self.audio_mode.currentData()))
        )
        settings.addWidget(self.audio_mode)
        settings.addWidget(QLabel("Композиция кадра"))
        self.composition_strategy = QComboBox()
        self.composition_strategy.addItem("Авто: сохранить важное", "safe_auto")
        self.composition_strategy.addItem("По центру", "center_crop")
        self.composition_strategy.addItem("С размытым фоном", "fit_blur_background")
        self.composition_strategy.addItem("С однотонным фоном", "fit_solid_background")
        self.composition_strategy.addItem("Верхняя часть кадра", "top_crop")
        self.composition_strategy.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(composition_strategy=str(self.composition_strategy.currentData()))
        )
        settings.addWidget(self.composition_strategy)
        settings.addWidget(QLabel("Субтитры"))
        self.subtitles = QCheckBox("Показывать субтитры")
        self.subtitles.toggled.connect(lambda value: self.viewmodel.save_options(subtitles_enabled=value))
        settings.addWidget(self.subtitles)
        settings.addWidget(QLabel("Стиль субтитров"))
        self.subtitle_style = QComboBox()
        self.subtitle_style.addItem("Документальный", "documentary")
        self.subtitle_style.addItem("Чистый", "clean")
        self.subtitle_style.addItem("Минималистичный", "minimal")
        self.subtitle_style.addItem("Динамичный", "dynamic")
        self.subtitle_style.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(subtitle_style=str(self.subtitle_style.currentData()))
        )
        settings.addWidget(self.subtitle_style)
        self.cache = QCheckBox("Использовать готовый анализ, если он есть")
        self.cache.toggled.connect(lambda value: self.viewmodel.save_options(use_cache=value))
        settings.addWidget(self.cache)
        settings.addStretch()
        self.run_button = QPushButton("Начать поиск моментов")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._primary_action)
        settings.addWidget(self.run_button)
        self.run_button.hide()
        panel.hide()
        self.settings_toggle.toggled.connect(self._set_advanced_visible)
        self.setup_advanced_toggle.toggled.connect(self._set_advanced_visible)
        body.addWidget(panel)
        root.addLayout(body, 1)
        self.viewmodel.project_changed.connect(self._project_changed)
        self.viewmodel.runs_changed.connect(self._runs_changed)
        self.viewmodel.processing_changed.connect(self._processing_changed)
        self.viewmodel.error_occurred.connect(self._error)

    def open(self, project: DesktopProject) -> None:
        self.viewmodel.open(project)

    def _project_changed(self, project: DesktopProject) -> None:
        is_new_project = self.project is None or self.project.project_id != project.project_id
        self.project = project
        self.title.setText(project.name)
        self.status.setText(_STATUS.get(project.status, "Неизвестно"))
        self.run_button.setText("Начать поиск моментов")
        if project.source_spec.is_ready and (is_new_project or self.preview.active_media_path is None):
            self.preview.show_source(str(project.source))
        source = project.source_metadata
        duration = format_seconds(source.get("duration")) if source else "н/д"
        resolution = f"{source.get('width', '—')} × {source.get('height', '—')}" if source else "н/д"
        fps = source.get("fps", "н/д") if source else "н/д"
        size = self._format_file_size(source.get("size_bytes") or source.get("estimated_size_bytes")) if source else "н/д"
        source_name = project.source.name if project.source_spec.is_ready else str(source.get("title") or "Видео по ссылке")
        source_kind = "Ссылка на видео" if project.source_spec.kind == "url" else "Файл"
        self._replace_card_text(self.metadata, [
            f"{source_kind}: {source_name}", f"Длительность: {duration}",
            f"Разрешение: {resolution}", f"Размер: {size}", f"FPS: {fps}",
        ])
        self._update_estimate(project)
        self._set_combo_data(self.processing_mode, project.settings.processing_mode)
        self._set_combo_data(self.setup_processing_mode, project.settings.processing_mode)
        self._set_combo_data(self.deep_analysis, project.settings.deep_analysis)
        self._set_combo_data(self.setup_deep_analysis, project.settings.deep_analysis)
        self._set_combo_data(self.platform, project.settings.platform)
        self._set_combo_data(self.setup_platform, project.settings.platform)
        self._set_combo_data(self.clip_count, str(project.settings.clip_count))
        self._set_combo_data(self.audio_mode, project.settings.audio_mode)
        self._set_combo_data(self.composition_strategy, project.settings.composition_strategy)
        self.subtitles.blockSignals(True); self.subtitles.setChecked(project.settings.subtitles_enabled); self.subtitles.blockSignals(False)
        self._set_combo_data(self.subtitle_style, project.settings.subtitle_style)
        self.cache.blockSignals(True); self.cache.setChecked(project.settings.use_cache); self.cache.blockSignals(False)
        self._update_download_card(project)
        self._update_setup_card(project)
        self._update_candidate_review(project)
        self._update_next_step(project)
        self._refresh_active_candidate_detail(project)
        self._apply_flow_visibility(project)

    def _set_advanced_visible(self, visible: bool) -> None:
        if self._flow_step != "settings":
            visible = False
        self.settings_panel.setVisible(visible)
        for toggle in (self.settings_toggle, self.setup_advanced_toggle):
            toggle.blockSignals(True)
            toggle.setChecked(visible)
            toggle.blockSignals(False)

    def _save_setup_option(self, name: str, value: str) -> None:
        self.viewmodel.save_options(**{name: value})

    def _update_download_card(self, project: DesktopProject) -> None:
        source = project.source_metadata
        name = str(source.get("title") or "Видео по ссылке")
        size = self._format_file_size(source.get("estimated_size_bytes") or source.get("size_bytes"))
        state = project.source_spec.download_state
        if state == "cancelled":
            message = "Загрузка была остановлена. Можно начать её снова — исходный файл не будет затронут."
            button = "Скачать видео ещё раз"
        elif state == "failed":
            message = project.source_spec.error_message or "Видео не удалось скачать. Проверьте ссылку и попробуйте ещё раз."
            button = "Попробовать скачать снова"
        elif state == "downloading":
            message = "Видео скачивается на этот компьютер. Статус, скорость и оставшееся время показаны ниже."
            button = "Скачивание идёт"
        else:
            message = "Видео проверено. Сначала скачайте его на этот компьютер, затем выберите настройки обработки."
            button = "Скачать видео"
        details = f"{name}"
        if size != "н/д":
            details += f" · ожидаемый объём: {size}"
        self.download_source.setText(f"{details}\n{message}")
        self.download_button.setText(button)
        self.download_button.setDisabled(state == "downloading" or self.viewmodel.active)

    def _derive_flow_step(self, project: DesktopProject) -> str:
        snapshot = self.viewmodel.snapshot
        if snapshot.phase in {"preparing", "running", "cancelling"}:
            return "download" if snapshot.stage == "download" else "processing"
        if not project.source_spec.is_ready:
            return "download"
        states = project.candidate_states.values()
        if project.selected_candidate_ids or any(state in {"draft_ready", "selected"} for state in states):
            return "drafts"
        if any(state == "rendered" for state in states):
            return "finished"
        if project.analysis_artifact_path:
            return "candidates"
        return "settings"

    def _flow_hint_for(self, step: str, project: DesktopProject) -> str:
        hints = {
            "download": "Скачайте видео отдельно. Когда файл будет готов, откроется настройка.",
            "settings": "Выберите основные параметры. Затем начнётся поиск подходящих моментов.",
            "processing": "Мы подготовим видео и перейдём к следующему готовому результату автоматически.",
            "candidates": "Посмотрите найденные моменты и выберите до трёх для черновиков.",
            "drafts": "Посмотрите черновики и подтвердите только те, из которых нужно сделать готовые ролики.",
            "finished": "Готовые ролики можно посмотреть здесь или открыть в папке проекта.",
        }
        return hints.get(step, "Выберите источник видео.")

    def _apply_flow_visibility(self, project: DesktopProject) -> None:
        step = self._derive_flow_step(project)
        self._flow_step = step
        active = self.viewmodel.snapshot.phase in {"preparing", "running", "cancelling"}
        self.flow_position.setText(f"Шаг {_FLOW_STEP_INDEX[step]} из {len(_FLOW_STEPS)}")
        self.flow_title.setText(_FLOW_STEP_LABELS[step])
        self.flow_hint.setText(self._flow_hint_for(step, project))
        self.download_card.setVisible(step == "download" and not active)
        self.setup_card.setVisible(step == "settings" and not active)
        review_visible = step in {"candidates", "drafts", "finished"} and not active
        self.candidate_review.setVisible(review_visible)
        self.preview.setVisible(review_visible and project.source_spec.is_ready)
        self.candidate_detail.setVisible(review_visible and self._active_candidate_id is not None)
        self.progress.setVisible(active)
        self.next_step.hide()
        self.metadata.hide()
        self.estimate.hide()
        self.content_summary.hide()
        self.history.hide()
        self.history_title.hide()
        self.settings_toggle.setVisible(step == "settings" and not active)
        self.autosave.setVisible(step == "settings" and not active)
        if step != "settings" or active:
            self._set_advanced_visible(False)

    def _update_setup_card(self, project: DesktopProject) -> None:
        source = project.source_metadata
        duration = format_seconds(source.get("duration")) if source.get("duration") is not None else "пока неизвестна"
        size = self._format_file_size(source.get("size_bytes") or source.get("estimated_size_bytes"))
        source_name = project.source.name if project.source_spec.is_ready else str(source.get("title") or "Видео по ссылке")
        source_kind = "Файл" if project.source_spec.kind == "local_file" else "Видео по ссылке"
        source_state = "готов" if project.source_spec.is_ready else "ещё не загружен"
        self.setup_source.setText(
            f"{source_kind}: {source_name}\nДлительность: {duration} · Размер: {size}\nИсточник {source_state}."
        )
        self.setup_mode_help.setText({
            "fast": "Быстрый вариант для интервью, лекций и других разговорных видео.",
            "standard": "Подходящий вариант по умолчанию: хороший баланс времени и качества.",
            "maximum": "Тщательнее учитывает контекст и события в кадре. Это займёт больше времени.",
        }[project.settings.processing_mode])
        try:
            resolved, estimate = self.viewmodel.setup_preflight()
            deep_state = "будет использован" if resolved.deep_analysis.resolved else "не потребуется"
            self.setup_deep_help.setText(f"{resolved.deep_analysis.reason} Дополнительный разбор {deep_state}.")
            self.setup_platform_help.setText(
                f"{resolved.platform.label}: ролик будет вертикальным, до {int(resolved.platform.maximum_duration_seconds)} секунд, "
                "с субтитрами и полями для интерфейса."
            )
            self.setup_estimate.setText(self._setup_estimate_text(estimate))
        except Exception:
            saved = project.setup_state.last_estimate
            self.setup_deep_help.setText("Рекомендация появится после проверки настроек.")
            self.setup_estimate.setText(self._saved_estimate_text(saved))
        self.setup_change.setText(
            project.setup_state.change_summary or "Настройки сохраняются в этом проекте."
        )
        preparing = not bool(project.analysis_artifact_path)
        heading = self.setup_card.layout().itemAt(0).widget()
        if isinstance(heading, QLabel):
            heading.setText("Настройка" if preparing else "Настройки следующего поиска")
        self.setup_start_button.setVisible(preparing)
        self.run_button.hide()

    @staticmethod
    def _format_cost_range(minimum: object, maximum: object) -> str:
        try:
            low, high = float(minimum), float(maximum)
        except (TypeError, ValueError):
            return "неизвестна до проверки тарифов"
        if high < 0.01:
            return "меньше $0.01"
        if low < 0.01:
            return f"до ${high:.2f}"
        return f"примерно ${low:.2f}–${high:.2f}"

    def _setup_estimate_text(self, estimate) -> str:
        minutes = (
            f"около {max(1, round(estimate.estimated_seconds_min / 60))}–"
            f"{max(1, round(estimate.estimated_seconds_max / 60))} мин"
        )
        drivers = ", ".join(estimate.cost_drivers[:3]) or "длительность и выбранные настройки"
        return f"Ориентировочное время: {minutes}. Оно зависит от: {drivers}."

    def _saved_estimate_text(self, saved: dict) -> str:
        if not isinstance(saved, dict):
            return "Оценка появится после проверки настроек."
        try:
            minutes = (
                f"около {max(1, round(float(saved['estimated_seconds_min']) / 60))}–"
                f"{max(1, round(float(saved['estimated_seconds_max']) / 60))} мин"
            )
        except (KeyError, TypeError, ValueError):
            return "Оценка появится после проверки настроек."
        return f"Последняя сохранённая оценка: {minutes}."

    def _primary_action(self) -> None:
        if not self.project:
            return
        self.viewmodel.start_analysis()

    def _update_candidate_review(self, project: DesktopProject) -> None:
        layout = self.candidate_review_layout
        heading = layout.itemAt(0).widget()
        if isinstance(heading, QLabel):
            step = self._derive_flow_step(project)
            heading.setText({
                "candidates": "Выберите моменты",
                "drafts": "Проверьте черновики",
                "finished": "Готовые ролики",
            }.get(step, "Моменты"))
        persistent = {self.workflow_hint, self.draft_button, self.production_button}
        while layout.count() > 1:
            item = layout.takeAt(1)
            widget = item.widget()
            if widget and widget not in persistent:
                widget.setParent(None)
                widget.deleteLater()
        self._candidate_selection_buttons = {}
        self._candidate_thumbnail_labels = {}
        self._candidate_cards = {}
        analysis_path = Path(project.analysis_artifact_path) if project.analysis_artifact_path else None
        analysis = read_json(analysis_path, {}) if analysis_path and analysis_path.is_file() else {}
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        previews: dict[str, dict] = {}
        for candidate_id, artifact_path in project.candidate_draft_artifacts.items():
            path = Path(artifact_path)
            draft = read_json(path, {}) if path.is_file() else {}
            if not isinstance(draft, dict):
                continue
            candidate = next(
                (item for item in draft.get("candidates", [])
                 if isinstance(item, dict) and str(item.get("candidate_id") or "") == candidate_id),
                None,
            )
            if isinstance(candidate, dict):
                previews[candidate_id] = candidate
        if not candidates:
            self.workflow_hint.setText("После поиска здесь появятся моменты, из которых можно выбрать до трёх черновиков.")
            self.workflow_hint.show()
            self.draft_button.hide()
            self.production_button.hide()
            layout.addWidget(self.workflow_hint)
            return
        draftable_ids = [
            candidate_id for candidate_id in project.review_selected_candidate_ids
            if project.candidate_states.get(candidate_id) not in {"draft_ready", "selected", "rendered", "production_rendering"}
        ]
        recommended_count = sum(
            bool(item.get("recommended", item.get("selected_by_recommendation")))
            for item in candidates if isinstance(item, dict)
        )
        rendered_count = sum(state == "rendered" for state in project.candidate_states.values())
        ready_count = sum(state in {"draft_ready", "selected"} for state in project.candidate_states.values())
        processing_count = sum(state in {"draft_planning", "production_rendering"} for state in project.candidate_states.values())
        selection_toolbar = QFrame()
        toolbar_layout = QHBoxLayout(selection_toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        summary = QLabel(
            f"Найдено моментов: {len(candidates)} · рекомендуем: {recommended_count} · "
            f"выбрано: {len(project.review_selected_candidate_ids)}/3"
        )
        summary.setWordWrap(True)
        toolbar_layout.addWidget(summary, 1)
        recommended_button = QPushButton("Выбрать рекомендованные")
        recommended_button.clicked.connect(self._select_recommended)
        clear_button = QPushButton("Снять выбор")
        clear_button.clicked.connect(self._clear_review_selection)
        toolbar_layout.addWidget(recommended_button)
        toolbar_layout.addWidget(clear_button)
        layout.addWidget(selection_toolbar)
        self._configure_workflow_action(project, draftable_ids, ready_count, rendered_count, processing_count)
        layout.addWidget(self.workflow_hint)
        # The action buttons are intentionally retained while the dynamic
        # candidate cards are rebuilt. A retained widget can temporarily be
        # detached from the layout, in which case ``isVisible()`` is false
        # even after ``show()``. ``isHidden()`` preserves the intended action
        # state and ensures a clickable button is put back into the layout.
        if not self.draft_button.isHidden():
            layout.addWidget(self.draft_button)
        if not self.production_button.isHidden():
            layout.addWidget(self.production_button)
        final_outputs = self._final_outputs_by_candidate()
        visible_candidates = self._filtered_candidates(candidates, project)
        for item in visible_candidates:
            if not isinstance(item, dict) or not item.get("candidate_id"):
                continue
            candidate_id = str(item["candidate_id"])
            state = project.candidate_states.get(candidate_id, str(item.get("recommendation_status") or "analyzed"))
            override = project.candidate_boundary_overrides.get(candidate_id, {})
            original_start = item.get("start_seconds", item.get("start", 0))
            original_end = item.get("end_seconds", item.get("end", original_start))
            start_value = override.get("start", original_start) if isinstance(override, dict) else original_start
            end_value = override.get("end", original_end) if isinstance(override, dict) else original_end
            start, end = format_seconds(start_value), format_seconds(end_value)
            potential = {"high": "Высокий потенциал", "medium": "Средний потенциал", "low": "Низкий потенциал"}.get(
                str(item.get("potential") or "low"), "Предварительная оценка",
            )
            confidence = float(item.get("confidence") or 0)
            status_label = {
                "analyzed": "можно посмотреть и добавить к черновикам", "draft_planning": "готовим черновик",
                "draft_ready": "черновик готов к проверке", "draft_failed": "черновик не готов",
                "selected": "подтверждён", "production_rendering": "создаём готовый ролик", "rendered": "готово",
            }.get(state, "можно посмотреть")
            if state == "draft_ready" and candidate_id not in project.review_selected_candidate_ids:
                status_label = "черновик не выбран для готового ролика"
            frame = QFrame(); frame.setObjectName("card")
            self._candidate_cards[candidate_id] = frame
            row = QHBoxLayout(frame); row.setContentsMargins(10, 8, 10, 8)
            thumbnail = QLabel("Кадр\nзагружается")
            thumbnail.setObjectName("muted")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setFixedSize(112, 64)
            thumbnail.setStyleSheet("border: 1px solid #303640; border-radius: 4px;")
            row.addWidget(thumbnail)
            self._candidate_thumbnail_labels.setdefault(candidate_id, []).append(thumbnail)
            try:
                start_seconds = float(start_value)
                end_seconds = float(end_value)
            except (TypeError, ValueError):
                start_seconds, end_seconds = 0.0, 0.0
            preview_contract = item.get("preview") if isinstance(item.get("preview"), dict) else {}
            thumbnail_contract = preview_contract.get("thumbnail") if isinstance(preview_contract.get("thumbnail"), dict) else {}
            thumbnail_time = thumbnail_contract.get("timestamp_seconds", start_seconds + max(0.0, min(1.0, (end_seconds - start_seconds) / 2)))
            try:
                thumbnail_seconds = float(thumbnail_time)
            except (TypeError, ValueError):
                thumbnail_seconds = start_seconds
            if project.source.is_file():
                self._thumbnail_loader.request(
                    cache_directory=project.directory / "candidate-thumbnails",
                    analysis_id=project.analysis_id or "analysis",
                    candidate_id=candidate_id,
                    source_path=project.source,
                    timestamp_seconds=thumbnail_seconds,
                )
            information = QVBoxLayout()
            title = QLabel(str(item.get("title") or item.get("core_idea") or "Момент из видео"))
            title.setStyleSheet("font-weight: 600;")
            title.setWordWrap(True)
            information.addWidget(title)
            recommended = " · рекомендуем" if item.get("recommended", item.get("selected_by_recommendation")) else ""
            details = QLabel(f"{start}–{end} · {potential} · уверенность {confidence * 100:.0f}%{recommended}")
            details.setObjectName("muted")
            details.setWordWrap(True)
            information.addWidget(details)
            reasons = [str(value) for value in item.get("reasons", []) if str(value)]
            if reasons:
                reason = QLabel("Почему: " + reasons[0])
                reason.setObjectName("muted")
                reason.setWordWrap(True)
                information.addWidget(reason)
            row.addLayout(information, 1)
            actions = QVBoxLayout()
            status = QLabel(status_label)
            status.setObjectName("muted")
            status.setMaximumWidth(258)
            candidate_error = project.candidate_errors.get(candidate_id)
            if candidate_error:
                status.setText("Черновик не создан. Попробуйте ещё раз или выберите другой момент.")
                status.setToolTip(candidate_error)
            status.setWordWrap(True)
            actions.addWidget(status)
            source_preview = QPushButton("Просмотреть")
            source_preview.setObjectName(f"preview-candidate-{candidate_id}")
            source_preview.clicked.connect(lambda _checked=False, value=dict(item): self._preview_candidate(value))
            actions.addWidget(source_preview)
            preview = previews.get(candidate_id, {}).get("preview", {}) if isinstance(previews.get(candidate_id), dict) else {}
            preview_file = Path(str(preview.get("output_file") or "")) if isinstance(preview, dict) else None
            if preview_file and preview_file.is_file():
                button = QPushButton("Смотреть черновик")
                button.clicked.connect(
                    lambda _checked=False, path=preview_file, title=str(item.get("title") or item.get("core_idea") or "момент"):
                    self._show_draft_preview(path, title)
                )
                actions.addWidget(button)
            if state == "draft_ready":
                if candidate_id in project.review_selected_candidate_ids:
                    approve = QPushButton("Подтвердить")
                    approve.clicked.connect(
                        lambda _checked=False, value=candidate_id: self._set_draft_approval(value, True)
                    )
                    reject = QPushButton("Отклонить")
                    reject.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                    actions.addWidget(approve)
                    actions.addWidget(reject)
                else:
                    restore = QPushButton("Вернуть к проверке")
                    restore.clicked.connect(lambda _checked=False, value=candidate_id: self._restore_draft(value))
                    actions.addWidget(restore)
            elif state == "selected":
                reject = QPushButton("Отклонить")
                reject.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                actions.addWidget(reject)
            elif state == "rendered":
                final_file = final_outputs.get(candidate_id)
                if final_file:
                    watch_final = QPushButton("Смотреть готовый ролик")
                    watch_final.clicked.connect(
                        lambda _checked=False, path=final_file, title=str(item.get("title") or item.get("core_idea") or "момент"):
                        self._show_final_preview(path, title)
                    )
                    actions.addWidget(watch_final)
                    open_final = QPushButton("Открыть готовый ролик")
                    open_final.clicked.connect(lambda _checked=False, path=final_file: self._open_file(path))
                    actions.addWidget(open_final)
            elif state not in {"draft_planning", "production_rendering"}:
                selected_for_draft = candidate_id in project.review_selected_candidate_ids
                select = QPushButton("Убрать из черновиков" if selected_for_draft else "Добавить к черновикам")
                select.setObjectName(f"select-candidate-{candidate_id}")
                select.clicked.connect(lambda _checked=False, value=candidate_id: self._toggle_candidate_selection(value))
                self._candidate_selection_buttons[candidate_id] = select
                actions.addWidget(select)
            row.addLayout(actions)
            layout.addWidget(frame)
        self._mark_active_candidate()

    def _configure_workflow_action(
        self,
        project: DesktopProject,
        draftable_ids: list[str],
        ready_count: int,
        rendered_count: int,
        processing_count: int,
    ) -> None:
        """Expose exactly the next safe pipeline action for the current state."""

        self.draft_button.hide()
        self.production_button.hide()
        self.draft_button.setDisabled(True)
        self.production_button.setDisabled(True)
        if project.status in {"analyzing", "processing", "rendering_selected"} or processing_count:
            self.workflow_hint.setText("Сейчас идёт работа. Прогресс и оставшееся время показаны на отдельном экране.")
            return
        if draftable_ids:
            count = len(draftable_ids)
            selected_count = len(project.review_selected_candidate_ids)
            selection_summary = (
                f"Выбрано {count} из 3."
                if selected_count == count
                else f"Выбрано {selected_count} из 3. Для {count} из них ещё нужен черновик."
            )
            self.workflow_hint.setText(
                f"{selection_summary} Следующий шаг — создать черновики, чтобы посмотреть ролики перед финальной сборкой."
            )
            self.draft_button.setText(f"Создать черновики ({count})")
            self.draft_button.setEnabled(True)
            self.draft_button.show()
            return
        if project.selected_candidate_ids:
            count = len(project.selected_candidate_ids)
            self.workflow_hint.setText(
                f"Подтверждено: {count}. Черновики проверены — теперь можно создать готовые ролики."
            )
            self.production_button.setText(f"Создать готовые ролики ({count})")
            self.production_button.setEnabled(True)
            self.production_button.show()
            return
        if ready_count:
            self.workflow_hint.setText(
                "Черновики готовы. Посмотрите каждый как вертикальный ролик, затем подтвердите нужные или отклоните их."
            )
            return
        if rendered_count:
            self.workflow_hint.setText("Готовые ролики можно посмотреть в карточках или открыть в папке проекта.")
            return
        self.workflow_hint.setText("Посмотрите моменты и добавьте к черновикам от одного до трёх лучших.")

    def _final_outputs_by_candidate(self) -> dict[str, Path]:
        """Find candidate-owned final MP4s listed by saved production reports."""

        outputs: dict[str, Path] = {}
        for run in self.runs:
            if not run.report_path:
                continue
            report = read_json(Path(run.report_path), {})
            if not isinstance(report, dict):
                continue
            items = report.get("primary_results", [])
            if not isinstance(items, list):
                production = report.get("production_render", {})
                items = production.get("items", []) if isinstance(production, dict) else []
            for item in items:
                if not isinstance(item, dict) or str(item.get("status") or "") not in {"completed", "warning"}:
                    continue
                candidate_id = str(item.get("candidate_id") or "")
                output_file = Path(str(item.get("output_file") or ""))
                if candidate_id and output_file.is_file():
                    outputs.setdefault(candidate_id, output_file)
        return outputs

    def _draft_action(self) -> None:
        if not self.project:
            return
        candidate_ids = list(self.project.review_selected_candidate_ids)
        if not candidate_ids:
            return
        needs_draft = [
            candidate_id for candidate_id in candidate_ids
            if self.project.candidate_states.get(candidate_id) not in {"draft_ready", "selected", "rendered", "production_rendering"}
        ]
        if needs_draft:
            self.viewmodel.build_drafts(needs_draft)

    def _toggle_candidate_selection(self, candidate_id: str) -> None:
        if not self.project:
            return
        selected = list(self.project.review_selected_candidate_ids)
        if candidate_id in selected:
            selected.remove(candidate_id)
        else:
            if len(selected) >= 3:
                QMessageBox.information(
                    self, "Выбрано три момента", "Для одного прохода можно добавить к черновикам не больше трёх моментов.",
                )
                return
            selected.append(candidate_id)
        self.viewmodel.set_review_selection(selected)

    def _set_draft_approval(self, candidate_id: str, approved: bool) -> None:
        self.viewmodel.set_draft_approval(candidate_id, approved)

    def _reject_draft(self, candidate_id: str) -> None:
        if not self.project:
            return
        if candidate_id in self.project.selected_candidate_ids:
            self.viewmodel.set_draft_approval(candidate_id, False)
        selected = [item for item in self.project.review_selected_candidate_ids if item != candidate_id]
        self.viewmodel.set_review_selection(selected)

    def _restore_draft(self, candidate_id: str) -> None:
        if not self.project or candidate_id in self.project.review_selected_candidate_ids:
            return
        if len(self.project.review_selected_candidate_ids) >= 3:
            QMessageBox.information(
                self, "Выбрано три момента", "Сначала уберите один момент из текущего выбора, затем верните этот черновик к проверке.",
            )
            return
        self.viewmodel.set_review_selection([*self.project.review_selected_candidate_ids, candidate_id])

    def _select_recommended(self) -> None:
        if not self.project:
            return
        path = Path(self.project.analysis_artifact_path) if self.project.analysis_artifact_path else None
        analysis = read_json(path, {}) if path and path.is_file() else {}
        candidate_ids = [
            str(item.get("candidate_id")) for item in analysis.get("candidates", [])
            if isinstance(item, dict) and item.get("recommended", item.get("selected_by_recommendation")) and item.get("candidate_id")
        ] if isinstance(analysis, dict) else []
        self.viewmodel.set_review_selection(candidate_ids[:3])

    def _clear_review_selection(self) -> None:
        self.viewmodel.set_review_selection([])

    def _change_candidate_filter(self, value: str) -> None:
        self._candidate_filter = value
        if self.project:
            self._update_candidate_review(self.project)

    def _change_candidate_sort(self, value: str) -> None:
        self._candidate_sort = value
        if self.project:
            self._update_candidate_review(self.project)

    def _filtered_candidates(self, candidates: list[object], project: DesktopProject) -> list[dict]:
        values = [dict(item) for item in candidates if isinstance(item, dict) and item.get("candidate_id")]
        if self._candidate_filter == "recommended":
            values = [item for item in values if item.get("recommended", item.get("selected_by_recommendation"))]
        elif self._candidate_filter in {"high", "medium"}:
            values = [item for item in values if item.get("potential") == self._candidate_filter]
        elif self._candidate_filter == "unselected":
            values = [item for item in values if item.get("candidate_id") not in project.review_selected_candidate_ids]
        potential_rank = {"high": 2, "medium": 1, "low": 0}
        if self._candidate_sort == "time":
            values.sort(key=lambda item: float(item.get("start_seconds", item.get("start", 0)) or 0))
        elif self._candidate_sort == "potential":
            values.sort(key=lambda item: (
                potential_rank.get(str(item.get("potential")), 0),
                float(item.get("confidence") or 0), float(item.get("score") or 0),
            ), reverse=True)
        else:
            values.sort(key=lambda item: (
                bool(item.get("recommended", item.get("selected_by_recommendation"))),
                potential_rank.get(str(item.get("potential")), 0),
                float(item.get("score") or 0),
            ), reverse=True)
        return values

    def _preview_candidate(self, candidate: dict) -> None:
        if not self.project:
            return
        try:
            start, end = self._candidate_range(candidate)
        except (TypeError, ValueError):
            return
        self.preview.set_range(
            self.project.source, start, end,
            cache_directory=self.project.directory / "preview-proxies",
            candidate_title=str(candidate.get("title") or candidate.get("core_idea") or "Выбранный кандидат"),
        )
        self._active_candidate_id = str(candidate.get("candidate_id") or "") or None
        self._mark_active_candidate()
        self._focus_preview_player()
        self._show_candidate_detail(candidate, start, end)

    def _show_draft_preview(self, path: Path, title: str) -> None:
        self.preview.show_draft(str(path), title)
        self._focus_preview_player()

    def _show_final_preview(self, path: Path, title: str | None = None) -> None:
        self.preview.show_final(str(path), title)
        self._focus_preview_player()

    def _mark_active_candidate(self) -> None:
        for candidate_id, card in self._candidate_cards.items():
            active = candidate_id == self._active_candidate_id
            card.setProperty("activeCandidate", active)
            card.setStyleSheet("border: 2px solid #4f9cff;" if active else "")

    def _focus_preview_player(self, *_: object) -> None:
        self.content_scroll.ensureWidgetVisible(self.preview, 0, 16)
        self.preview.play_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def _candidate_range(self, candidate: dict) -> tuple[float, float]:
        candidate_id = str(candidate.get("candidate_id") or "")
        override = self.project.candidate_boundary_overrides.get(candidate_id, {}) if self.project else {}
        start = float(override.get("start", candidate.get("start_seconds", candidate.get("start", 0))))
        end = float(override.get("end", candidate.get("end_seconds", candidate.get("end", start))))
        return start, max(start, end)

    def _show_candidate_detail(self, candidate: dict, start: float, end: float) -> None:
        if not self.project:
            return
        potential = {"high": "Высокий", "medium": "Средний", "low": "Низкий"}.get(str(candidate.get("potential")), "Предварительный")
        reasons = [str(item) for item in candidate.get("reasons", []) if str(item)]
        risks = [str(item) for item in candidate.get("risks", []) if str(item)]
        lines = [
            str(candidate.get("title") or "Момент"),
            f"{format_seconds(start)}–{format_seconds(end)} · {format_seconds(end - start)}",
            f"Потенциал: {potential} · Уверенность: {float(candidate.get('confidence') or 0) * 100:.0f}%",
            f"Идея: {candidate.get('core_idea') or candidate.get('summary') or '—'}",
            f"Начало: {candidate.get('hook_summary') or '—'}",
            f"Финал: {candidate.get('payoff_summary') or '—'}",
            "Оценки: " + " · ".join((
                f"удержание {self._score_text(candidate.get('retention_score'))}",
                f"публикация {self._score_text(candidate.get('publishability_score'))}",
                f"потенциал {self._score_text(candidate.get('viral_score'))}",
            )),
        ]
        excerpt = str(candidate.get("transcript_excerpt") or candidate.get("text") or "").strip()
        if excerpt:
            lines.append("Текст: " + excerpt)
        if reasons:
            lines.append("Почему рекомендуем: " + " ".join(reasons[:2]))
        if risks:
            lines.append("Риск: " + " ".join(risks[:2]))
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id and self.project.candidate_errors.get(candidate_id):
            lines.append("Причина ошибки: " + self.project.candidate_errors[candidate_id])
        self._replace_card_text(self.candidate_detail, lines)
        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 4, 0, 0)
        for index, (text, boundary, delta) in enumerate((
            ("Начало −1 с", "start", -1.0), ("Начало −0.5 с", "start", -0.5),
            ("Начало +0.5 с", "start", 0.5), ("Начало +1 с", "start", 1.0),
            ("Конец −1 с", "end", -1.0), ("Конец −0.5 с", "end", -0.5),
            ("Конец +0.5 с", "end", 0.5), ("Конец +1 с", "end", 1.0),
        )):
            button = QPushButton(text)
            button.setToolTip("Проверит только сохранённые границы речи и сцены; повторный анализ не нужен.")
            button.clicked.connect(
                lambda _checked=False, cid=candidate_id, name=boundary, value=delta: self._adjust_candidate_boundary(cid, name, value)
            )
            grid.addWidget(button, index // 4, index % 4)
        self.candidate_detail.layout().addWidget(controls)

    @staticmethod
    def _score_text(value: object) -> str:
        try:
            return f"{float(value):.0f}/100"
        except (TypeError, ValueError):
            return "—"

    def _adjust_candidate_boundary(self, candidate_id: str, boundary: str, delta_seconds: float) -> None:
        if candidate_id:
            self.viewmodel.adjust_candidate_boundary(candidate_id, boundary, delta_seconds)

    def _refresh_active_candidate_detail(self, project: DesktopProject) -> None:
        if not self._active_candidate_id or not project.analysis_artifact_path:
            return
        # A project update (for example, confirming a draft) must not replace
        # the draft/final the person is currently watching with the source
        # candidate again.  Refresh ranges only while the source-range player
        # itself is active, e.g. after a boundary adjustment.
        if self.preview.source_range_seconds is None:
            self._mark_active_candidate()
            return
        analysis = read_json(Path(project.analysis_artifact_path), {})
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        candidate = next(
            (item for item in candidates if isinstance(item, dict) and item.get("candidate_id") == self._active_candidate_id),
            None,
        )
        if isinstance(candidate, dict):
            try:
                start, end = self._candidate_range(candidate)
            except (TypeError, ValueError):
                return
            self.preview.set_range(
                project.source, start, end,
                cache_directory=project.directory / "preview-proxies",
                candidate_title=str(candidate.get("title") or candidate.get("core_idea") or "Выбранный кандидат"),
            )
            self._mark_active_candidate()
            self._show_candidate_detail(candidate, start, end)

    def _thumbnail_ready(self, candidate_id: str, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        for label in self._candidate_thumbnail_labels.get(candidate_id, []):
            try:
                label.setText("")
                label.setPixmap(pixmap.scaled(
                    label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
                ))
            except RuntimeError:
                continue

    def _thumbnail_unavailable(self, candidate_id: str) -> None:
        for label in self._candidate_thumbnail_labels.get(candidate_id, []):
            try:
                label.setText("Кадр\nнедоступен")
            except RuntimeError:
                continue

    def _confirm_production_render(self) -> None:
        if not self.project or not self.project.selected_candidate_ids:
            return
        answer = QMessageBox.question(
            self, "Создать итоговые ролики",
            "Создать готовые вертикальные ролики только для подтверждённых черновиков?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.viewmodel.render_selected()

    def _runs_changed(self, runs: list[ProjectRun]) -> None:
        self.runs = runs
        self._update_content_summary(runs)
        while self.history_layout.count() > 1:
            item = self.history_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for run in runs:
            frame = QFrame(); frame.setObjectName("card")
            layout = QHBoxLayout(frame); layout.setContentsMargins(12, 10, 12, 10)
            kind = {
                "analysis": "Анализ видео", "draft": "Подготовка черновика",
                "selected_render": "Создание роликов", "render_revision": "Создать заново",
                "full": "Полный запуск",
            }.get(run.run_kind, "Запуск")
            text = QLabel(f"{run.started_at[:16].replace('T', ' ')} · {kind} · {run.status.replace('_', ' ')}")
            text.setWordWrap(True)
            layout.addWidget(text, 1)
            results = [Path(item) for item in run.artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()]
            for index, result in enumerate(results, start=1):
                button = QPushButton("Открыть ролик" if len(results) == 1 else f"Ролик {index}")
                button.clicked.connect(lambda _, path=result: self._open_file(path))
                layout.addWidget(button)
            if run.status in {"completed", "completed_with_warnings"}:
                rerender = QPushButton("Создать заново")
                rerender.setToolTip("Повторно экспортировать с текущими стилем, платформой и качеством без нового AI-анализа.")
                rerender.clicked.connect(lambda _, parent_run=run: self.viewmodel.rerender(parent_run))
                layout.addWidget(rerender)
            folder = QPushButton("Папка")
            folder.clicked.connect(lambda _, path=Path(run.log_path).parent if run.log_path else None: self._open_folder(path))
            layout.addWidget(folder)
            self.history_layout.insertWidget(self.history_layout.count() - 1, frame)
        final_run = next(
            (run for run in runs if run.run_kind in {"selected_render", "render_revision"}), None,
        )
        if final_run:
            result = next(
                (Path(item) for item in final_run.artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()), None,
            )
            if result:
                self._show_final_preview(result)
        if self.project:
            self._update_candidate_review(self.project)
            self._update_next_step(self.project)

    def _update_content_summary(self, runs: list[ProjectRun]) -> None:
        for run in runs:
            if not run.report_path:
                continue
            report = read_json(Path(run.report_path), {})
            understanding = report.get("content_understanding", {}) if isinstance(report, dict) else {}
            if not isinstance(understanding, dict) or not understanding.get("enabled"):
                continue
            profile = understanding.get("profile", {})
            content_map = understanding.get("content_map", {})
            recommendation = understanding.get("clip_count_recommendation", {})
            coverage = understanding.get("coverage_map", understanding.get("coverage", {}))
            if not all(isinstance(item, dict) for item in (profile, content_map, recommendation, coverage)):
                continue
            clip_range = recommendation.get("estimated_publishable_clip_range", {})
            lower = clip_range.get("min", "—")
            upper = clip_range.get("max", "—")
            selected_chapters = coverage.get("selected_chapters", [])
            coverage_status = (
                "Подборка охватывает разные части видео."
                if isinstance(selected_chapters, list) and len(selected_chapters) > 1
                else "Подборка покрывает найденные самостоятельные фрагменты."
            )
            lines = [
                f"Тип: {profile.get('detected_content_type', 'не определён')}",
                f"Смысловых частей: {len(content_map.get('chapters', []))}",
                f"Самостоятельных историй: {recommendation.get('estimated_story_count', understanding.get('story_unit_count', 0))}",
                f"Рекомендуем создать: {lower}–{upper} ролика(ов)",
                coverage_status,
            ]
            lines.extend(self._virality_summary_lines(report))
            self._replace_card_text(self.content_summary, lines)
            return
        self._replace_card_text(self.content_summary, ["Рекомендация появится после завершения анализа."])

    def _update_next_step(self, project: DesktopProject) -> None:
        """Keep the page-level answer to “what do I do now?” short and stable."""

        states = project.candidate_states
        if project.status in {"analyzing", "processing", "rendering_selected"}:
            self.next_step_text.setText("Сейчас идёт работа. Дождитесь завершения или остановите её.")
        elif any(state == "rendered" for state in states.values()):
            self.next_step_text.setText("Ролики готовы. Откройте нужный ролик в карточке и проверьте его.")
        elif project.selected_candidate_ids:
            self.next_step_text.setText("Черновики подтверждены. Когда будете готовы, создайте готовые ролики.")
        elif any(state in {"draft_ready", "selected"} for state in states.values()):
            self.next_step_text.setText("Есть готовые вертикальные черновики. Посмотрите их, затем подтвердите нужные или отклоните остальные.")
        elif project.review_selected_candidate_ids:
            self.next_step_text.setText("Моменты выбраны. Следующая безопасная операция — создать черновики; финальный render пока не начнётся.")
        elif project.analysis_artifact_path:
            self.next_step_text.setText("Моменты готовы. Посмотрите их и добавьте к черновикам от одного до трёх лучших.")
        else:
            self.next_step_text.setText("Выберите настройки и начните поиск подходящих моментов.")

    @staticmethod
    def _virality_summary_lines(report: dict) -> list[str]:
        """Keep Goal 5B reasons short and human-readable; never expose its formula."""

        virality = report.get("virality", {}) if isinstance(report, dict) else {}
        intelligence = report.get("clip_intelligence", {}) if isinstance(report, dict) else {}
        candidates = intelligence.get("candidates", []) if isinstance(intelligence, dict) else []
        if not isinstance(virality, dict) or not virality.get("enabled") or not isinstance(candidates, list):
            return []
        chosen = next(
            (item for item in candidates if isinstance(item, dict) and item.get("selected") and isinstance(item.get("virality"), dict)),
            next((item for item in candidates if isinstance(item, dict) and isinstance(item.get("virality"), dict)), None),
        )
        if not isinstance(chosen, dict):
            return []
        details = chosen.get("virality", {})
        potential = details.get("viral_potential", {}) if isinstance(details, dict) else {}
        publishability = details.get("publishability", {}) if isinstance(details, dict) else {}
        retention = details.get("retention_profile", {}) if isinstance(details, dict) else {}
        if not all(isinstance(item, dict) for item in (potential, publishability, retention)):
            return []
        level = {
            "weak": "Низкий", "moderate": "Средний", "strong": "Высокий", "excellent": "Очень высокий",
        }.get(str(potential.get("level")), "Предварительный")
        lines = [f"Потенциал: {level}"]
        factors = potential.get("strongest_factors", [])
        if isinstance(factors, list):
            labels = {
                "hook": "Сильное начало", "curiosity": "Интрига раскрывается", "emotion": "Эмоциональная развязка",
                "payoff": "Самостоятельный вывод", "retention": "Хороший шанс удержания",
                "publishability": "Готов к публикации", "quotability": "Запоминающаяся фраза",
                "usefulness": "Практическая ценность", "momentum": "Мысль развивается",
            }
            for factor in factors:
                label = labels.get(str(factor))
                if label and label not in lines:
                    lines.append(label)
                if len(lines) >= 4:
                    break
        eligibility = details.get("eligibility", {}) if isinstance(details, dict) else {}
        status = eligibility.get("status") if isinstance(eligibility, dict) else ""
        if status == "publishable_now" and "Готов к публикации" not in lines:
            lines.append("Готов к публикации")
        elif status in {"needs_reconstruction", "publishable_with_minor_adjustment"}:
            lines.append("Лучше доработать перед публикацией")
        confidence = potential.get("confidence", {}) if isinstance(potential, dict) else {}
        if isinstance(confidence, dict) and confidence.get("warnings"):
            lines.append("Предварительная оценка: недостаточно визуальных данных")
        return lines[:5]

    def _processing_changed(self, snapshot: ProcessingSnapshot) -> None:
        active = snapshot.phase in {"preparing", "running", "cancelling"}
        if active:
            detail = self._processing_detail(snapshot)
            self.progress.set_running(
                snapshot.stage_label,
                f"Прошло {format_seconds(snapshot.elapsed_seconds)}",
                snapshot.progress_fraction,
                detail,
                cancelling=snapshot.phase == "cancelling",
            )
        else:
            self.progress.set_finished(snapshot.message)
        self.run_button.setDisabled(active)
        self.setup_start_button.setDisabled(active)
        self.draft_button.setDisabled(active or not (self.project and self.project.review_selected_candidate_ids))
        selected_drafts_exist = bool(self.project and self.project.selected_candidate_ids) and all(
            Path(self.project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
            for candidate_id in self.project.selected_candidate_ids
        ) if self.project else False
        self.production_button.setDisabled(active or not selected_drafts_exist)
        for widget in (
            self.processing_mode, self.deep_analysis, self.platform, self.clip_count,
            self.audio_mode, self.composition_strategy, self.subtitles, self.subtitle_style, self.cache,
            self.setup_processing_mode, self.setup_deep_analysis, self.setup_platform,
        ):
            widget.setDisabled(active)
        if self.project:
            self._update_download_card(self.project)
            self._apply_flow_visibility(self.project)

    def _processing_detail(self, snapshot: ProcessingSnapshot) -> str:
        if snapshot.stage == "download":
            details: list[str] = []
            if snapshot.transfer_downloaded and snapshot.transfer_total:
                details.append(f"Загружено: {snapshot.transfer_downloaded} из {snapshot.transfer_total}")
            elif snapshot.transfer_downloaded:
                details.append(f"Загружено: {snapshot.transfer_downloaded}")
            elif self.project:
                expected = self._format_file_size(self.project.source_metadata.get("estimated_size_bytes"))
                if expected != "н/д":
                    details.append(f"Ожидаемый объём: {expected}")
            if snapshot.transfer_speed:
                details.append(f"Скорость: {snapshot.transfer_speed}")
            if snapshot.eta_seconds is not None:
                details.append(f"Осталось: {format_seconds(snapshot.eta_seconds)}")
            if snapshot.phase == "cancelling":
                details.append("Останавливаем загрузку и удаляем неполный файл")
            elif not details:
                details.append("Подключаемся к видео и ждём первые данные")
            return " · ".join(details)
        next_result = {
            "analysis": "Дальше покажем найденные моменты.",
            "draft": "Дальше покажем готовые черновики.",
            "selected_render": "Дальше покажем готовые ролики.",
        }.get(self.viewmodel.run.run_kind if self.viewmodel.run else "", "Следующий результат откроется автоматически.")
        if snapshot.phase == "cancelling":
            return "Останавливаем работу. Уже готовые результаты останутся в проекте."
        return f"Сейчас: {snapshot.stage_label}. {next_result} Это может занять несколько минут."

    def _open_project_folder(self) -> None:
        if self.project: self._open_folder(Path(self.project.project_directory))

    @staticmethod
    def _open_folder(path: Path | None) -> None:
        if path and path.is_dir(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _open_file(path: Path) -> None:
        if path.is_file(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _card(title: str) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        layout = QVBoxLayout(card); layout.setContentsMargins(14, 12, 14, 12)
        label = QLabel(title); label.setStyleSheet("font-weight: 600;")
        layout.addWidget(label)
        return card

    @staticmethod
    def _replace_card_text(card: QFrame, values: list[str]) -> None:
        layout = card.layout()
        while layout.count() > 1:
            item = layout.takeAt(1)
            if item.widget(): item.widget().deleteLater()
        for value in values:
            label = QLabel(value); label.setObjectName("muted")
            # Candidate excerpts and error details can be long.  They are
            # explanatory copy, never a reason to widen the whole review
            # page or expose a horizontal scrollbar.
            label.setWordWrap(True)
            layout.addWidget(label)

    def _update_estimate(self, project: DesktopProject) -> None:
        try:
            estimate = self.viewmodel.services.processing_estimate(project)
            minutes = (
                f"около {max(1, round(estimate.estimated_seconds_min / 60))}–"
                f"{max(1, round(estimate.estimated_seconds_max / 60))} мин"
            )
            cost = self._format_cost_range(estimate.estimated_ai_cost_min, estimate.estimated_ai_cost_max)
            analysis = "будет использован" if estimate.deep_analysis_resolved else "не потребуется"
            self._replace_card_text(self.estimate, [
                f"Время: {minutes}",
                f"Результат: примерно {estimate.estimated_clips_min}–{estimate.estimated_clips_max} ролика(ов)",
                f"Глубокий анализ: {analysis}",
                f"Ориентир по стоимости: {cost}",
                estimate.cost_note,
            ])
        except Exception:
            self._replace_card_text(self.estimate, ["Оценка появится после проверки настроек."])

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        combo.blockSignals(True)
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @staticmethod
    def _format_file_size(value: object) -> str:
        try:
            size = float(value)
        except (TypeError, ValueError):
            return "н/д"
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if size < 1024 or unit == "ГБ":
                return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
            size /= 1024
        return "н/д"

    def _error(self, error) -> None:
        QMessageBox.warning(self, error.title, error.user_message)
