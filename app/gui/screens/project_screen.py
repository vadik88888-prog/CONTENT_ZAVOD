from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import (
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from app.analysis_artifact import candidate_is_draftable
from app.clip_results import ClipResult, primary_clip_results, unique_primary_results
from app.gui.components import CandidateThumbnailLoader, FinalOutput, FinalResultsWorkspace, ProcessingProgress, VideoPreview
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

# The persisted workflow keeps more detail than the product navigation.  In
# particular, URL download is a Source substate and Moments/Drafts/Finals are
# Results substates.  Retaining the internal names below keeps recovery and the
# existing ViewModel contract intact while the chrome stays intentionally calm.
_GLOBAL_FLOW_STEPS = (
    ("source", "Источник"),
    ("settings", "Настройка"),
    ("processing", "Обработка"),
    ("results", "Результаты"),
)
_GLOBAL_STEP_FOR_FLOW = {
    "download": "source",
    "settings": "settings",
    "processing": "processing",
    "candidates": "results",
    "drafts": "results",
    "finished": "results",
}


class _ElidedLabel(QLabel):
    """One-line card text that preserves the complete value in its tooltip."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        """Keep the source value when callers refresh a project title."""

        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._refresh_elision()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_elision()

    def _refresh_elision(self) -> None:
        width = self.contentsRect().width()
        if width <= 0:
            super().setText(self._full_text)
            return
        super().setText(self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideRight,
            width,
        ))


class ProjectScreen(QWidget):
    back_requested = Signal()

    def __init__(self, viewmodel: ProjectViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        self.project: DesktopProject | None = None
        self.runs: list[ProjectRun] = []
        self._active_candidate_id: str | None = None
        # The player is shared by Moments and Drafts.  Keep its review binding
        # separate from the user choices persisted in the project: a project
        # refresh must not replay an unchanged fragment, while a boundary edit
        # must replace an obsolete draft/source preview with the exact new
        # source range.
        self._active_candidate_range: tuple[str, float, float] | None = None
        self._active_preview_kind = "source"
        self._persisting_active_preview = False
        self._review_candidates_by_id: dict[str, dict] = {}
        self._review_visible_candidate_ids: list[str] = []
        self._draft_preview_paths: dict[str, Path] = {}
        self._candidate_thumbnail_labels: dict[str, list[QLabel]] = {}
        self._candidate_thumbnail_paths: dict[str, Path] = {}
        self._candidate_cards: dict[str, QFrame] = {}
        self._flow_step = "settings"
        self._results_subflow_override: str | None = None
        # Keep the long analysis list responsive.  Additional cards are added
        # only on explicit request; thumbnails stay on their existing async
        # loader.
        self._candidate_visible_limit = 12
        self._thumbnail_loader = CandidateThumbnailLoader(self)
        self._thumbnail_loader.thumbnail_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.thumbnail_unavailable_with_path.connect(self._thumbnail_unavailable)
        root = QVBoxLayout(self)
        self._root_layout = root
        root.setContentsMargins(34, 26, 34, 30)
        header = QHBoxLayout()
        self._header_layout = header
        back = QPushButton("← Проекты")
        back.clicked.connect(self.back_requested)
        self.title = _ElidedLabel("Проект")
        self.title.setObjectName("title")
        self.status = _ElidedLabel("")
        self.status.setObjectName("status")
        self.settings_toggle = QPushButton("Дополнительно")
        self.settings_toggle.setCheckable(True)
        self.folder = QPushButton("Открыть папку")
        self.folder.clicked.connect(self._open_project_folder)
        header.addWidget(back)
        header.addWidget(self.title, 1)
        header.addWidget(self.status)
        header.addWidget(self.settings_toggle)
        header.addWidget(self.folder)
        root.addLayout(header)
        self.autosave = QLabel("Изменения сохраняются автоматически")
        self.autosave.setObjectName("muted")
        root.addWidget(self.autosave)
        self.flow_card = QFrame()
        self.flow_card.setObjectName("workflowStepper")
        flow_layout = QVBoxLayout(self.flow_card)
        flow_layout.setContentsMargins(14, 10, 14, 10)
        self._global_step_labels: dict[str, QLabel] = {}
        self._global_step_dividers: list[QLabel] = []
        stepper_row = QHBoxLayout()
        self._stepper_layout = stepper_row
        stepper_row.setSpacing(0)
        for index, (step, label) in enumerate(_GLOBAL_FLOW_STEPS):
            item = QLabel(f"{index + 1}  {label}")
            item.setObjectName("workflowStep")
            item.setProperty("stepState", "pending")
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._global_step_labels[step] = item
            stepper_row.addWidget(item, 1)
            if index < len(_GLOBAL_FLOW_STEPS) - 1:
                divider = QLabel("›")
                divider.setObjectName("workflowDivider")
                divider.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._global_step_dividers.append(divider)
                stepper_row.addWidget(divider)
        flow_layout.addLayout(stepper_row)
        self.flow_position = QLabel("Источник")
        self.flow_position.setObjectName("muted")
        self.flow_title = QLabel("Настройка обработки")
        self.flow_title.setObjectName("flowScreenTitle")
        self.flow_hint = QLabel()
        self.flow_hint.setObjectName("muted")
        self.flow_hint.setWordWrap(True)
        self.flow_route = QLabel("")
        self.flow_route.setObjectName("muted")
        self.flow_route.setWordWrap(True)
        flow_layout.addWidget(self.flow_position)
        flow_layout.addWidget(self.flow_title)
        flow_layout.addWidget(self.flow_hint)
        root.addWidget(self.flow_card)
        body = QHBoxLayout()
        self._body_layout = body
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
        setup_layout.addWidget(QLabel("Сколько черновиков подготовить"))
        self.setup_clip_count = QComboBox()
        for label, value in (("Авто", "auto"), ("1 лучший", "1"), ("3 лучших", "3"), ("5 лучших", "5")):
            self.setup_clip_count.addItem(label, value)
        self.setup_clip_count.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("clip_count", str(self.setup_clip_count.currentData()))
        )
        setup_layout.addWidget(self.setup_clip_count)
        self.setup_count_help = QLabel("Количество можно изменить позже — анализ не будет запущен повторно из-за выбора списка.")
        self.setup_count_help.setObjectName("muted")
        self.setup_count_help.setWordWrap(True)
        setup_layout.addWidget(self.setup_count_help)
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
        self.view_all_button = QPushButton("Посмотреть все моменты")
        self.view_all_button.setObjectName("secondaryAction")
        self.view_all_button.clicked.connect(self._view_all_candidates)
        self.production_button = QPushButton("Создать готовые ролики")
        self.production_button.setObjectName("primary")
        # QPushButton.clicked carries a ``bool`` checked argument.  Keep it
        # out of the candidate-id API: the delivery set is always derived
        # from the durable approved-draft state below.
        self.production_button.clicked.connect(lambda _checked=False: self._confirm_production_render())
        left.addWidget(self.candidate_review)
        self.final_results = FinalResultsWorkspace()
        self.final_results.output_selected.connect(self._final_output_selected)
        self.final_results.create_more_requested.connect(self._create_more_outputs)
        self.final_results.rerender_requested.connect(self._rerender_final_output)
        self.final_results.projects_requested.connect(self.back_requested)
        left.addWidget(self.final_results)
        self.progress = ProcessingProgress()
        self.progress.cancel_requested.connect(self.viewmodel.cancel)
        self.progress.continue_waiting_requested.connect(self.viewmodel.continue_waiting)
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
        self.same_source_broll = QCheckBox("Использовать дополнительные кадры из этого видео")
        self.same_source_broll.setToolTip(
            "Разрешает использовать только подходящие фрагменты из загруженного видео. "
            "Без вашего выбора дополнительные кадры не добавляются."
        )
        self.same_source_broll.toggled.connect(
            lambda value: self.viewmodel.save_options(same_source_broll_allowed=value)
        )
        settings.addWidget(self.same_source_broll)
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
        self._compose_stage_workspaces(left, body)
        root.addLayout(body, 1)
        self._install_sticky_actions(root)
        self.viewmodel.project_changed.connect(self._project_changed)
        self.viewmodel.runs_changed.connect(self._runs_changed)
        self.viewmodel.processing_changed.connect(self._processing_changed)
        self.viewmodel.error_occurred.connect(self._error)
        self._compact_stage_layout: bool | None = None
        self._compact_action_layout: bool | None = None
        self._apply_stage_responsive_layout(force=True)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "_body_layout"):
            self._apply_stage_responsive_layout()

    def _apply_stage_responsive_layout(self, *, force: bool = False) -> None:
        """Stack dense stage panels before a scaled desktop can overflow.

        Windows at 150% scaling can leave this screen with roughly 700 logical
        pixels after the shell sidebar.  The focused layouts remain the same
        widgets and media instances; only their box direction and margins
        change so the outer scroll area stays vertically, not horizontally,
        scrollable.
        """

        # The review workspace contains a readable candidate card, a bounded
        # player and an inspector.  Stacking it at laptop widths is preferable
        # to squeezing the card actions into hidden horizontal overflow.  A
        # full-HD desktop keeps the three-column composition from the approved
        # visual reference.
        compact = self.width() < 1600
        compact_actions = self.width() < 900
        cards_need_reflow = (
            self._compact_action_layout is not None
            and compact_actions != self._compact_action_layout
        )
        if (
            not force
            and compact == self._compact_stage_layout
            and compact_actions == self._compact_action_layout
        ):
            return
        self._compact_stage_layout = compact
        self._compact_action_layout = compact_actions
        direction = (
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        action_direction = (
            QBoxLayout.Direction.TopToBottom
            if compact_actions
            else QBoxLayout.Direction.LeftToRight
        )
        spacing = 12 if compact else 18
        self._root_layout.setContentsMargins(
            18 if compact else 34,
            14 if compact else 26,
            18 if compact else 34,
            18 if compact else 30,
        )
        self._body_layout.setDirection(direction)
        self._body_layout.setSpacing(spacing)
        self._setup_workspace_layout.setDirection(direction)
        self._setup_workspace_layout.setSpacing(spacing)
        self._processing_workspace_layout.setDirection(direction)
        self._processing_workspace_layout.setSpacing(spacing)
        self._review_body_layout.setDirection(direction)
        self._review_body_layout.setSpacing(12 if compact else 14)
        self._header_layout.setDirection(action_direction)
        self._header_layout.setSpacing(6 if compact_actions else 8)
        self._stepper_layout.setDirection(action_direction)
        self._stepper_layout.setSpacing(4 if compact_actions else 0)
        for divider in self._global_step_dividers:
            divider.setVisible(not compact_actions)
        self._setup_action_layout.setDirection(action_direction)
        self._processing_actions_layout.setDirection(action_direction)
        self._review_header_layout.setDirection(action_direction)
        self._review_action_layout.setDirection(action_direction)
        self._review_action_layout.setSpacing(8 if compact_actions else 6)
        self.content_host.setMinimumWidth(0)
        if compact:
            for panel in (self.review_list_panel, self.review_preview_panel, self.review_inspector_panel):
                panel.setMinimumWidth(0)
                panel.setMaximumWidth(16_777_215)
        else:
            # These are content minima rather than fixed columns; the preview
            # still receives the extra room through its stretch factor.
            # The candidate card's layout needs a few pixels beyond its raw
            # size hint for the scroll-frame border; otherwise a hidden two
            # pixel horizontal scrollbar remains even at full HD.
            self.review_list_panel.setMinimumWidth(704)
            self.review_list_panel.setMaximumWidth(720)
            self.review_preview_panel.setMinimumWidth(460)
            self.review_preview_panel.setMaximumWidth(16_777_215)
            self.review_inspector_panel.setMinimumWidth(300)
            self.review_inspector_panel.setMaximumWidth(390)
        self.settings_panel.setMaximumWidth(16_777_215 if compact else 300)
        self.setup_summary.setMaximumWidth(16_777_215)
        self.processing_summary.setMaximumWidth(16_777_215)
        self.updateGeometry()
        # Candidate cards own an action rail.  Rebuild them only when the
        # breakpoint changes so that the rail can become a full-width block
        # instead of being horizontally squeezed or clipped.
        if cards_need_reflow and self.project:
            self._update_candidate_review(self.project)

    def _compose_stage_workspaces(self, legacy_layout: QVBoxLayout, body: QHBoxLayout) -> None:
        """Arrange the durable widgets into one focused workspace per stage.

        The old screen appended every stage to one long scroll area.  All of
        those widgets already own useful state (most importantly the source
        ``VideoPreview``), so this method deliberately *moves* them instead of
        recreating controls or introducing a second navigation/state layer.
        """

        while legacy_layout.count():
            item = legacy_layout.takeAt(0)
            if widget := item.widget():
                widget.setParent(None)
        body.removeWidget(self.settings_panel)
        self.settings_panel.setMaximumWidth(16_777_215)
        self.settings_panel.setMinimumWidth(0)

        legacy_layout.setContentsMargins(0, 0, 0, 0)
        legacy_layout.setSpacing(18)
        self._stage_widgets: dict[str, QWidget] = {}

        # Source / URL download is intentionally a Source substate.  It never
        # appears as a fifth global navigation step.
        self.source_workspace = QWidget()
        self.source_workspace.setObjectName("sourceWorkspace")
        source_layout = QVBoxLayout(self.source_workspace)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(16)
        source_heading = QLabel("Источник видео")
        source_heading.setObjectName("screenTitle")
        source_copy = QLabel("Подготовим локальную копию, а затем откроем настройки обработки.")
        source_copy.setObjectName("subtitle")
        source_copy.setWordWrap(True)
        source_layout.addWidget(source_heading)
        source_layout.addWidget(source_copy)
        source_layout.addWidget(self.download_card)
        source_layout.addStretch()
        self._stage_widgets["source"] = self.source_workspace

        # Settings combines the source summary, recommendation, compact
        # choices and a short right rail.  Advanced controls stay collapsed.
        self.setup_workspace = QWidget()
        self.setup_workspace.setObjectName("setupWorkspace")
        setup_workspace_layout = QHBoxLayout(self.setup_workspace)
        self._setup_workspace_layout = setup_workspace_layout
        setup_workspace_layout.setContentsMargins(0, 0, 0, 0)
        setup_workspace_layout.setSpacing(18)
        setup_main = QWidget()
        setup_main_layout = QVBoxLayout(setup_main)
        setup_main_layout.setContentsMargins(0, 0, 0, 0)
        setup_main_layout.setSpacing(14)
        self.setup_source_summary = self._card("Ваше видео")
        self.setup_source_summary_text = QLabel()
        self.setup_source_summary_text.setObjectName("muted")
        self.setup_source_summary_text.setWordWrap(True)
        self.setup_source_summary.layout().addWidget(self.setup_source_summary_text)
        setup_main_layout.addWidget(self.setup_source_summary)
        self.recommendation_banner = QFrame()
        self.recommendation_banner.setObjectName("recommendationBanner")
        recommendation_layout = QHBoxLayout(self.recommendation_banner)
        recommendation_layout.setContentsMargins(16, 14, 16, 14)
        recommendation_icon = QLabel("✦")
        recommendation_icon.setObjectName("recommendationIcon")
        self.recommendation_text = QLabel()
        self.recommendation_text.setWordWrap(True)
        recommendation_layout.addWidget(recommendation_icon)
        recommendation_layout.addWidget(self.recommendation_text, 1)
        setup_main_layout.addWidget(self.recommendation_banner)

        # The CTA belongs to one persistent action bar rather than to the
        # middle of an option card; this preserves the one-primary-action rule.
        self.setup_card.layout().removeWidget(self.setup_start_button)
        self.setup_start_button.setParent(None)
        setup_main_layout.addWidget(self.setup_card)
        setup_main_layout.addWidget(self.settings_panel)
        self.setup_action_bar = QFrame()
        self.setup_action_bar.setObjectName("stickyActionBar")
        setup_action_layout = QHBoxLayout(self.setup_action_bar)
        self._setup_action_layout = setup_action_layout
        setup_action_layout.setContentsMargins(14, 10, 14, 10)
        self.setup_back_button = QPushButton("← К источнику")
        self.setup_back_button.clicked.connect(self.back_requested)
        setup_action_layout.addWidget(self.setup_back_button)
        setup_action_layout.addStretch()
        setup_action_layout.addWidget(self.setup_start_button)
        setup_main_layout.addWidget(self.setup_action_bar)
        setup_main_layout.addStretch()
        setup_workspace_layout.addWidget(setup_main, 3)

        self.setup_summary = self._card("Краткая сводка")
        self.setup_summary.setObjectName("setupSummary")
        self.setup_summary_text = QLabel()
        self.setup_summary_text.setObjectName("muted")
        self.setup_summary_text.setWordWrap(True)
        self.setup_summary.layout().addWidget(self.setup_summary_text)
        setup_workspace_layout.addWidget(self.setup_summary, 1)
        self._stage_widgets["settings"] = self.setup_workspace

        # Processing gets a dedicated, honest progress surface instead of
        # sharing vertical space with review cards.  The stage labels are UI
        # context only; no percentage is invented for indeterminate work.
        self.processing_workspace = QWidget()
        self.processing_workspace.setObjectName("processingWorkspace")
        processing_layout = QHBoxLayout(self.processing_workspace)
        self._processing_workspace_layout = processing_layout
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.setSpacing(18)
        processing_main = QWidget()
        processing_main_layout = QVBoxLayout(processing_main)
        processing_main_layout.setContentsMargins(0, 0, 0, 0)
        processing_main_layout.setSpacing(14)
        self.processing_source_summary = self._card("Обрабатываемое видео")
        self.processing_source_text = QLabel()
        self.processing_source_text.setObjectName("muted")
        self.processing_source_text.setWordWrap(True)
        self.processing_source_summary.layout().addWidget(self.processing_source_text)
        processing_main_layout.addWidget(self.processing_source_summary)
        processing_main_layout.addWidget(self.progress)
        self.processing_stages = QFrame()
        self.processing_stages.setObjectName("processingStages")
        stages_layout = QVBoxLayout(self.processing_stages)
        stages_layout.setContentsMargins(16, 14, 16, 14)
        stages_heading = QLabel("Ход работы")
        stages_heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        stages_layout.addWidget(stages_heading)
        self._processing_stages_layout = stages_layout
        self.processing_stage_labels: dict[str, QLabel] = {}
        for stage, label in (
            ("prepare", "Подготавливаем видео"),
            ("transcribe", "Разбираем речь и структуру"),
            ("analyze", "Ищем сильные моменты"),
            ("render", "Собираем ролики"),
        ):
            row = QLabel(f"○  {label}")
            row.setObjectName("processingStage")
            row.setProperty("stageState", "pending")
            stages_layout.addWidget(row)
            self.processing_stage_labels[stage] = row
        self._processing_stage_rows: tuple[str, ...] = ("prepare", "transcribe", "analyze", "render")
        processing_main_layout.addWidget(self.processing_stages)
        self.processing_actions = QFrame()
        self.processing_actions.setObjectName("secondaryActionBar")
        processing_actions_layout = QHBoxLayout(self.processing_actions)
        self._processing_actions_layout = processing_actions_layout
        processing_actions_layout.setContentsMargins(12, 10, 12, 10)
        processing_projects = QPushButton("К проектам")
        processing_projects.clicked.connect(self.back_requested)
        processing_folder = QPushButton("Открыть папку проекта")
        processing_folder.clicked.connect(self._open_project_folder)
        processing_actions_layout.addWidget(processing_projects)
        processing_actions_layout.addWidget(processing_folder)
        processing_actions_layout.addStretch()
        processing_main_layout.addWidget(self.processing_actions)
        processing_main_layout.addStretch()
        processing_layout.addWidget(processing_main, 3)
        self.processing_summary = self._card("Текущий запуск")
        self.processing_summary.setObjectName("processingSummary")
        self.processing_summary_text = QLabel()
        self.processing_summary_text.setObjectName("muted")
        self.processing_summary_text.setWordWrap(True)
        self.processing_summary.layout().addWidget(self.processing_summary_text)
        self.processing_next = QLabel("Результаты сохраняются автоматически. После остановки останутся только готовые артефакты.")
        self.processing_next.setObjectName("muted")
        self.processing_next.setWordWrap(True)
        self.processing_summary.layout().addWidget(self.processing_next)
        processing_layout.addWidget(self.processing_summary, 1)
        self._stage_widgets["processing"] = self.processing_workspace

        # Moments and Drafts use the same durable source player.  The list,
        # preview and inspector move together visually while the player itself
        # remains alive for the whole ProjectScreen lifetime.
        self.review_workspace = QWidget()
        self.review_workspace.setObjectName("reviewWorkspace")
        review_layout = QVBoxLayout(self.review_workspace)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(12)
        review_header = QHBoxLayout()
        self._review_header_layout = review_header
        self.results_subflow = QLabel("Моменты")
        self.results_subflow.setObjectName("screenTitle")
        self.results_subflow_hint = QLabel()
        self.results_subflow_hint.setObjectName("subtitle")
        self.results_subflow_hint.setWordWrap(True)
        review_header.addWidget(self.results_subflow)
        review_header.addWidget(self.results_subflow_hint, 1)
        review_layout.addLayout(review_header)
        self.review_metrics = QFrame()
        self.review_metrics.setObjectName("resultsMetrics")
        metrics_layout = QHBoxLayout(self.review_metrics)
        metrics_layout.setContentsMargins(14, 10, 14, 10)
        self.review_metrics_text = QLabel()
        self.review_metrics_text.setWordWrap(True)
        metrics_layout.addWidget(self.review_metrics_text)
        review_layout.addWidget(self.review_metrics)
        review_body = QHBoxLayout()
        self._review_body_layout = review_body
        review_body.setSpacing(14)
        list_panel = QFrame()
        self.review_list_panel = list_panel
        list_panel.setObjectName("reviewListPanel")
        list_panel_layout = QVBoxLayout(list_panel)
        list_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.review_list_scroll = QScrollArea()
        self.review_list_scroll.setWidgetResizable(True)
        self.review_list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.review_list_scroll.setWidget(self.candidate_review)
        list_panel_layout.addWidget(self.review_list_scroll)
        review_body.addWidget(list_panel, 2)
        preview_panel = QFrame()
        self.review_preview_panel = preview_panel
        preview_panel.setObjectName("reviewPreviewPanel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        review_body.addWidget(preview_panel, 3)
        inspector_panel = QFrame()
        self.review_inspector_panel = inspector_panel
        inspector_panel.setObjectName("reviewInspectorPanel")
        inspector_layout = QVBoxLayout(inspector_panel)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        self.review_inspector_scroll = QScrollArea()
        self.review_inspector_scroll.setWidgetResizable(True)
        self.review_inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.review_inspector_scroll.setWidget(self.candidate_detail)
        inspector_layout.addWidget(self.review_inspector_scroll)
        review_body.addWidget(inspector_panel, 2)
        review_layout.addLayout(review_body, 1)
        self.review_action_bar = QFrame()
        self.review_action_bar.setObjectName("stickyActionBar")
        review_action_layout = QHBoxLayout(self.review_action_bar)
        self._review_action_layout = review_action_layout
        review_action_layout.setContentsMargins(14, 10, 14, 10)
        self.review_back_button = QPushButton("← Назад к обработке")
        self.review_back_button.clicked.connect(self.back_requested)
        review_action_layout.addWidget(self.review_back_button)
        review_action_layout.addWidget(self.workflow_hint, 1)
        review_action_layout.addWidget(self.view_all_button)
        review_action_layout.addWidget(self.draft_button)
        review_action_layout.addWidget(self.production_button)
        review_layout.addWidget(self.review_action_bar)
        self._stage_widgets["review"] = self.review_workspace

        self.final_workspace = QWidget()
        self.final_workspace.setObjectName("finalWorkspace")
        final_layout = QVBoxLayout(self.final_workspace)
        final_layout.setContentsMargins(0, 0, 0, 0)
        final_layout.addWidget(self.final_results, 1)
        self._stage_widgets["final"] = self.final_workspace

        # Keep project history and estimates available to experienced users,
        # but never compete with a single next action in the main flow.
        self.secondary_details = QFrame()
        self.secondary_details.setObjectName("secondaryDetails")
        secondary_layout = QVBoxLayout(self.secondary_details)
        secondary_layout.setContentsMargins(0, 10, 0, 0)
        secondary_layout.addWidget(self.next_step)
        secondary_layout.addWidget(self.metadata)
        secondary_layout.addWidget(self.estimate)
        secondary_layout.addWidget(self.content_summary)
        secondary_layout.addWidget(self.history_title)
        secondary_layout.addWidget(self.history)
        self.settings_panel.layout().addWidget(self.secondary_details)

        for widget in self._stage_widgets.values():
            legacy_layout.addWidget(widget)
            widget.hide()

    def _install_sticky_actions(self, root: QVBoxLayout) -> None:
        """Keep stage CTAs reachable when a compact desktop body scrolls."""

        self.stage_actions = QWidget()
        self.stage_actions.setObjectName("stageActions")
        actions_layout = QVBoxLayout(self.stage_actions)
        actions_layout.setContentsMargins(0, 10, 0, 0)
        actions_layout.setSpacing(0)
        # Adding a widget to its new layout moves it from the stage body; no
        # button is duplicated and therefore no second primary CTA appears.
        actions_layout.addWidget(self.setup_action_bar)
        actions_layout.addWidget(self.review_action_bar)
        self.setup_action_bar.hide()
        self.review_action_bar.hide()
        self.stage_actions.hide()
        root.addWidget(self.stage_actions)

    def open(self, project: DesktopProject) -> None:
        # Explicit navigation back to a project starts from its reconciled
        # persisted state; the temporary "create more" route is session-only.
        self._results_subflow_override = None
        self.viewmodel.open(project)

    def _project_changed(self, project: DesktopProject) -> None:
        if self._persisting_active_preview and self.project and self.project.project_id == project.project_id:
            # The only mutation was the durable preview id.  Rebuilding every
            # card during its click steals focus from the shared player and can
            # make a selected fragment appear to restart.
            self.project = project
            return
        previous_step = self._flow_step
        is_new_project = self.project is None or self.project.project_id != project.project_id
        self.project = project
        if is_new_project:
            self._results_subflow_override = None
            self._candidate_visible_limit = 12
            self._candidate_filter = "all"
            self._candidate_sort = "recommendation"
            self._active_candidate_id = getattr(project, "active_preview_candidate_id", None)
            self._active_candidate_range = None
            self._active_preview_kind = "source"
        self.title.setText(project.name)
        self.status.setText(_STATUS.get(project.status, "Неизвестно"))
        self.run_button.setText("Начать поиск моментов")
        if project.source_spec.is_ready and (is_new_project or self.preview.active_media_path is None):
            self.preview.show_source(
                str(project.source),
                source_codec=str(project.source_metadata.get("video_codec") or ""),
            )
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
        self._set_combo_data(self.setup_clip_count, str(project.settings.clip_count))
        self._set_combo_data(self.audio_mode, project.settings.audio_mode)
        self._set_combo_data(self.composition_strategy, project.settings.composition_strategy)
        self.same_source_broll.blockSignals(True)
        self.same_source_broll.setChecked(project.settings.same_source_broll_allowed)
        self.same_source_broll.blockSignals(False)
        self.subtitles.blockSignals(True); self.subtitles.setChecked(project.settings.subtitles_enabled); self.subtitles.blockSignals(False)
        self._set_combo_data(self.subtitle_style, project.settings.subtitle_style)
        self.cache.blockSignals(True); self.cache.setChecked(project.settings.use_cache); self.cache.blockSignals(False)
        self._update_download_card(project)
        self._update_setup_card(project)
        self._update_stage_context(project)
        self._update_candidate_review(project)
        self._update_final_results(project)
        self._update_next_step(project)
        self._reconcile_active_candidate_preview(project, previous_step=previous_step)
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

    def _update_stage_context(self, project: DesktopProject) -> None:
        """Fill stage summaries from durable project data without exposing IDs.

        These summaries deliberately use only source metadata and persisted
        project choices.  They neither inspect artifact folders nor start
        media probes on the GUI thread.
        """

        source = project.source_metadata
        source_name = project.source.name if project.source_spec.is_ready else str(source.get("title") or "Видео по ссылке")
        duration = format_seconds(source.get("duration")) if source.get("duration") is not None else "длительность уточняется"
        resolution = ""
        if source.get("width") and source.get("height"):
            resolution = f" · {source['width']}×{source['height']}"
        source_kind = "Локальный файл" if project.source_spec.kind == "local_file" else "Публичная ссылка"
        summary = f"{source_name}\n{source_kind} · {duration}{resolution}"
        self.setup_source_summary_text.setText(summary)
        self.processing_source_text.setText(summary)

        mode = {
            "fast": "Быстрый",
            "standard": "Стандартный",
            "maximum": "Максимальное качество",
        }.get(project.settings.processing_mode, "Стандартный")
        scope = {
            "auto": "Автоматически",
            "on": "С учётом событий в кадре",
            "off": "По речи и контексту",
        }.get(project.settings.deep_analysis, "Автоматически")
        platform = {
            "tiktok": "TikTok",
            "reels": "Instagram Reels",
            "shorts": "YouTube Shorts",
            "universal": "Вертикальный 9:16",
        }.get(project.settings.platform, "Вертикальный 9:16")
        count = self._selection_limit(project)
        self.setup_summary_text.setText(
            f"Режим: {mode}\nЧто анализируем: {scope}\nФормат: {platform}\n"
            f"Рекомендуемых черновиков: {count}\n"
            f"Дополнительные кадры: {'разрешены' if project.settings.same_source_broll_allowed else 'не использовать'}"
        )
        recommendation = {
            "fast": "Рекомендуем быстрый режим для разговорного материала. Он поможет быстрее перейти к просмотру моментов.",
            "standard": "Рекомендуем стандартный режим: он сохраняет хороший баланс между глубиной поиска и временем ожидания.",
            "maximum": "Максимальный режим тщательно проверит контекст и события в кадре. Он займёт больше времени.",
        }.get(project.settings.processing_mode, "Настройки сохраняются в проекте автоматически.")
        self.recommendation_text.setText(recommendation)
        latest = self._latest_run(project)
        launch = "Ожидает запуска"
        if latest:
            kind = {
                "analysis": "Поиск моментов",
                "draft": "Подготовка черновиков",
                "selected_render": "Создание готовых роликов",
                "render_revision": "Повторная сборка роликов",
                "full": "Обработка видео",
            }.get(latest.run_kind, "Обработка")
            launch = f"{kind} · {self._run_status_text(latest.status)}"
        self.processing_summary_text.setText(
            f"{launch}\nРежим: {mode}\nРезультаты сохраняются в проекте по мере готовности."
        )

    @staticmethod
    def _run_status_text(status: str) -> str:
        return {
            "preparing": "подготавливаем",
            "running": "выполняется",
            "cancelling": "останавливаем",
            "analysis_ready": "моменты готовы",
            "draft_ready": "черновики готовы",
            "completed": "готово",
            "completed_with_warnings": "готово с предупреждениями",
            "partially_rendered": "готово частично",
            "failed": "требует внимания",
            "cancelled": "остановлено",
            "interrupted": "прервано",
        }.get(status, "ожидает")

    def _selection_limit(self, project: DesktopProject) -> int:
        """Return only the persisted selection allowance for this project.

        Once analysis exists, every eligible candidate can be selected.  The
        requested clip count remains the Top-N recommendation, not a gate.
        """

        try:
            requested = project.settings.processing_intent().requested_clip_count
        except (TypeError, ValueError):
            requested = None
        if project.candidate_states:
            return len(project.candidate_states)
        return requested if requested is not None and requested > 0 else 5

    def _latest_run(self, project: DesktopProject) -> ProjectRun | None:
        runs = self._runs_for_project(project)
        if project.latest_run_id:
            matched = next((run for run in runs if run.run_id == project.latest_run_id), None)
            if matched is not None:
                return matched
        return max(runs, key=lambda run: (run.started_at, run.run_id), default=None)

    def _derive_flow_step(self, project: DesktopProject) -> str:
        snapshot = self.viewmodel.snapshot
        if snapshot.phase in {"preparing", "running", "cancelling"}:
            return "download" if snapshot.stage == "download" else "processing"
        if not project.source_spec.is_ready:
            return "download"
        if self._results_subflow_override == "candidates" and project.analysis_artifact_path:
            return "candidates"
        # Recovery is derived from the latest persisted run rather than from a
        # folder scan.  A stale process has already been converted to
        # ``interrupted`` by the service layer, so reopening it should return
        # to the stage where a person can safely inspect or retry the work.
        latest = self._latest_run(project)
        if latest and latest.status in {"failed", "interrupted", "cancelled", "partially_rendered"}:
            if latest.run_kind in {"draft"}:
                return "drafts"
            if latest.run_kind in {"selected_render", "render_revision"}:
                return "drafts" if project.candidate_draft_artifacts else "processing"
            if latest.run_kind in {"analysis", "full"} and not project.analysis_artifact_path:
                return "processing"
        # A completed retry must not hide other ready/failed drafts or
        # approved exports that remain in the project queue.  Those items are
        # reviewed before the final-result viewer, even if an earlier or
        # neighbouring output is already available.
        review_ids = set(project.review_selected_candidate_ids) | set(project.selected_candidate_ids)
        if any(
            project.candidate_states.get(candidate_id) in {"draft_ready", "draft_failed", "selected", "production_rendering"}
            for candidate_id in review_ids
        ):
            return "drafts"
        # A successful delivery remains available, but it must not mask a
        # later failed, interrupted, or partial render batch. That batch is
        # recoverable from Drafts and retains all existing artifacts.
        if self._final_output_records(project):
            return "finished"
        states = project.candidate_states.values()
        if project.candidate_draft_artifacts or project.selected_candidate_ids or any(
            state in {"draft_planning", "draft_ready", "draft_failed", "selected", "production_rendering"}
            for state in states
        ):
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
            "candidates": "Создайте рекомендованные варианты сразу или откройте весь список моментов.",
            "drafts": "Посмотрите черновики и подтвердите только те, из которых нужно сделать готовые ролики.",
            "finished": "Готовые ролики можно посмотреть здесь или открыть в папке проекта.",
        }
        return hints.get(step, "Выберите источник видео.")

    def _apply_flow_visibility(self, project: DesktopProject) -> None:
        step = self._derive_flow_step(project)
        self._flow_step = step
        active = self.viewmodel.snapshot.phase in {"preparing", "running", "cancelling"}
        global_step = _GLOBAL_STEP_FOR_FLOW[step]
        global_index = next(index for index, (name, _label) in enumerate(_GLOBAL_FLOW_STEPS, start=1) if name == global_step)
        self.flow_position.setText(f"Этап {global_index} из {len(_GLOBAL_FLOW_STEPS)}")
        screen_titles = {
            "download": "Источник видео",
            "settings": "Настройка обработки",
            "processing": "Идёт обработка",
            "candidates": "Найденные моменты",
            "drafts": "Черновики",
            "finished": "Готовые ролики",
        }
        self.flow_title.setText(screen_titles[step])
        self.flow_hint.setText(self._flow_hint_for(step, project))
        for index, (name, label) in enumerate(_GLOBAL_FLOW_STEPS, start=1):
            state = "current" if name == global_step else ("done" if index < global_index else "pending")
            widget = self._global_step_labels[name]
            widget.setProperty("stepState", state)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

        # Any live work uses the processing visual treatment.  The underlying
        # result substate remains in persisted project/run metadata and is
        # restored once the process reaches a terminal state.
        show_processing = active or step == "processing"
        show_source = step == "download" and not show_processing
        show_setup = step == "settings" and not show_processing
        # A legacy project can retain a rendered candidate state while its
        # canonical delivery registry is unavailable.  Keep its recoverable
        # review workspace visible instead of showing an empty final screen.
        show_review = (step in {"candidates", "drafts"} or (step == "finished" and not self._final_output_records(project))) and not show_processing
        show_final = step == "finished" and bool(self._final_output_records(project)) and not show_processing
        for name, widget in self._stage_widgets.items():
            widget.setVisible(
                (name == "source" and show_source)
                or (name == "settings" and show_setup)
                or (name == "processing" and show_processing)
                or (name == "review" and show_review)
                or (name == "final" and show_final)
            )
        self.download_card.setVisible(show_source)
        self.setup_card.setVisible(show_setup)
        self.candidate_review.setVisible(show_review)
        self.preview.setVisible(show_review and project.source_spec.is_ready)
        self.candidate_detail.setVisible(show_review)
        self.final_results.setVisible(show_final)
        self.progress.setVisible(show_processing)
        self.next_step.setVisible(False)
        self.metadata.setVisible(False)
        self.estimate.setVisible(False)
        self.content_summary.setVisible(False)
        self.history.setVisible(False)
        self.history_title.setVisible(False)
        self.secondary_details.setVisible(False)
        self.setup_action_bar.setVisible(show_setup and not self.setup_start_button.isHidden())
        self.review_action_bar.setVisible(show_review)
        self.stage_actions.setVisible(show_setup or show_review)
        self.results_subflow.setText({
            "candidates": "Найденные моменты",
            "drafts": "Черновики",
        }.get(step, "Результаты"))
        self.results_subflow_hint.setText({
            "candidates": "Просмотрите сильные фрагменты и выберите те, которые хотите превратить в черновики.",
            "drafts": "Проверьте созданные варианты и подтвердите только готовые к финальной сборке.",
        }.get(step, ""))
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
        self.setup_action_bar.setVisible(preparing)
        self.run_button.hide()

    @staticmethod
    def _format_cost_range(minimum: object, maximum: object) -> str:
        try:
            low, high = float(cast(Any, minimum)), float(cast(Any, maximum))
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
        workflow_step = self._derive_flow_step(project)
        self._review_candidates_by_id = {}
        self._review_visible_candidate_ids = []
        self._draft_preview_paths = {}
        heading = layout.itemAt(0).widget()
        if isinstance(heading, QLabel):
            heading.setText({
                "candidates": "Выберите моменты",
                "drafts": "Проверьте черновики",
                "finished": "Готовые ролики",
            }.get(workflow_step, "Моменты"))
        if workflow_step == "finished" and self._final_output_records(project):
            return
        while layout.count() > 1:
            item = layout.takeAt(1)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self._candidate_selection_buttons = {}
        self._candidate_thumbnail_labels = {}
        self._candidate_thumbnail_paths = {}
        self._candidate_cards = {}
        analysis_path = Path(project.analysis_artifact_path) if project.analysis_artifact_path else None
        analysis = read_json(analysis_path, {}) if analysis_path and analysis_path.is_file() else {}
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        if workflow_step == "candidates":
            # Phase 6 owns eligibility. Moments only projects its persisted
            # decision and never re-runs or reconstructs Brain/Vision evidence.
            candidates = [item for item in candidates if candidate_is_draftable(item)]
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
        if workflow_step == "drafts":
            # Draft review is deliberately not a second copy of the full
            # moments catalogue.  Keep pending/ready/approved draft items and
            # their exact persisted candidate bindings only.
            candidates = [
                item for item in candidates
                if isinstance(item, dict) and (
                    str(item.get("candidate_id") or "") in project.review_selected_candidate_ids
                    or str(item.get("candidate_id") or "") in project.candidate_draft_artifacts
                    or project.candidate_states.get(str(item.get("candidate_id") or ""))
                    in {"draft_planning", "draft_ready", "draft_failed", "selected", "production_rendering"}
                )
            ]
        self._review_candidates_by_id = {
            str(item["candidate_id"]): dict(item)
            for item in candidates
            if isinstance(item, dict) and item.get("candidate_id")
        }
        for candidate_id, record in previews.items():
            preview = record.get("preview", {}) if isinstance(record, dict) else {}
            preview_file = Path(str(preview.get("output_file") or "")) if isinstance(preview, dict) else None
            if preview_file and preview_file.is_file():
                self._draft_preview_paths[candidate_id] = preview_file
        if not candidates:
            self.workflow_hint.setText(
                "После поиска здесь появятся подходящие моменты и готовая рекомендация."
            )
            self.workflow_hint.show()
            self.view_all_button.hide()
            self.draft_button.hide()
            self.production_button.hide()
            return
        draftable_ids = [
            candidate_id for candidate_id in project.review_selected_candidate_ids
            if candidate_id in self._review_candidates_by_id
            and self._candidate_needs_draft(project, candidate_id)
        ]
        selected_moment_count = sum(
            candidate_id in self._review_candidates_by_id
            for candidate_id in project.review_selected_candidate_ids
        )
        recommended_count = sum(
            bool(item.get("recommended", item.get("selected_by_recommendation")))
            for item in candidates if isinstance(item, dict)
        )
        rendered_count = sum(state == "rendered" for state in project.candidate_states.values())
        ready_count = sum(state in {"draft_ready", "selected"} for state in project.candidate_states.values())
        processing_count = sum(state in {"draft_planning", "production_rendering"} for state in project.candidate_states.values())
        if workflow_step == "drafts":
            self.review_metrics_text.setText(
                f"Создано черновиков: {ready_count + rendered_count} · выбрано для финала: {len(project.selected_candidate_ids)}"
            )
        else:
            self.review_metrics_text.setText(
                f"Найдено {len(candidates)} · рекомендуем {recommended_count}"
            )
        if workflow_step == "candidates":
            # Moment selection belongs to the source-moment phase.  Drafts
            # have their own decisions (watch/approve/reject/retry), so do not
            # leak recommendation filters and draft-selection controls there.
            compact_actions = bool(self._compact_action_layout)
            action_direction = (
                QBoxLayout.Direction.TopToBottom
                if compact_actions
                else QBoxLayout.Direction.LeftToRight
            )
            selection_toolbar = QFrame()
            toolbar_layout = QHBoxLayout(selection_toolbar)
            # Three explicit bulk choices must remain readable in the bounded
            # Moments list column, including the full-HD three-panel layout.
            toolbar_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            summary = QLabel(
                f"Найдено {len(candidates)} · рекомендуем {recommended_count} · "
                f"выбрано {selected_moment_count}"
            )
            summary.setWordWrap(True)
            toolbar_layout.addWidget(summary, 0 if compact_actions else 1)
            recommended_button = QPushButton("Только рекомендованные")
            recommended_button.clicked.connect(self._select_recommended)
            select_all_button = QPushButton("Выбрать все")
            select_all_button.clicked.connect(self._select_all_candidates)
            clear_button = QPushButton("Снять все")
            clear_button.clicked.connect(self._clear_review_selection)
            toolbar_layout.addWidget(recommended_button)
            toolbar_layout.addWidget(select_all_button)
            toolbar_layout.addWidget(clear_button)
            layout.addWidget(selection_toolbar)
            filters = QFrame()
            filters.setObjectName("reviewFilters")
            filters_layout = QHBoxLayout(filters)
            filters_layout.setDirection(action_direction)
            filters_layout.setContentsMargins(0, 0, 0, 0)
            filters_layout.setSpacing(6)
            filter_combo = QComboBox()
            filter_combo.addItem("Рекомендованные", "recommended")
            filter_combo.addItem("Все моменты", "all")
            filter_combo.addItem("Не выбраны", "unselected")
            filter_combo.addItem("Высокий потенциал", "high")
            filter_combo.addItem("Средний потенциал", "medium")
            self._set_combo_data(filter_combo, self._candidate_filter)
            filter_combo.currentIndexChanged.connect(lambda _index: self._change_candidate_filter(str(filter_combo.currentData())))
            sort_combo = QComboBox()
            sort_combo.addItem("Сначала сильные", "recommendation")
            sort_combo.addItem("По времени", "time")
            sort_combo.addItem("По потенциалу", "potential")
            self._set_combo_data(sort_combo, self._candidate_sort)
            sort_combo.currentIndexChanged.connect(lambda _index: self._change_candidate_sort(str(sort_combo.currentData())))
            filters_layout.addWidget(filter_combo, 0 if compact_actions else 1)
            filters_layout.addWidget(sort_combo, 0 if compact_actions else 1)
            layout.addWidget(filters)
        self._configure_workflow_action(project, draftable_ids, ready_count, rendered_count, processing_count)
        final_outputs = self._final_outputs_by_candidate()
        if workflow_step == "drafts":
            review_order = {candidate_id: index for index, candidate_id in enumerate(project.review_selected_candidate_ids)}
            filtered_candidates = sorted(
                (dict(item) for item in candidates if isinstance(item, dict)),
                key=lambda item: (
                    str(item.get("candidate_id") or "") not in review_order,
                    review_order.get(str(item.get("candidate_id") or ""), len(review_order)),
                    float(item.get("start_seconds", item.get("start", 0)) or 0),
                ),
            )
            visible_candidates = filtered_candidates
        else:
            filtered_candidates = self._filtered_candidates(candidates, project)
            visible_candidates = filtered_candidates[:self._candidate_visible_limit]
        self._review_visible_candidate_ids = [
            str(item.get("candidate_id") or "") for item in visible_candidates
            if isinstance(item, dict) and item.get("candidate_id")
        ]
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
            draft_status = project.candidate_draft_statuses.get(candidate_id)
            export_status = project.candidate_export_statuses.get(candidate_id)
            status_label = {
                "analyzed": "можно посмотреть и добавить к черновикам", "draft_planning": "готовим черновик",
                "draft_ready": "черновик готов к проверке", "draft_failed": "черновик не готов",
                "selected": "подтверждён", "production_rendering": "создаём готовый ролик", "rendered": "готово",
            }.get(state, "можно посмотреть")
            if state == "draft_ready" and candidate_id not in project.review_selected_candidate_ids:
                status_label = "черновик не выбран для готового ролика"
            if draft_status == "failed" or state == "draft_failed":
                status_label = "Черновик не создан. Его можно повторить отдельно."
            elif export_status == "failed":
                status_label = "Готовый ролик не создан. Черновик сохранён и остаётся подтверждённым."
            elif (
                workflow_step == "drafts"
                and candidate_id in project.candidate_draft_artifacts
                and draft_status in {"pending", "running"}
            ):
                status_label = "Обновляем предпросмотр. Предыдущая готовая версия остаётся доступной."
            compact_actions = bool(self._compact_action_layout)
            frame = QFrame(); frame.setObjectName("card")
            frame.setMinimumWidth(0)
            frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self._candidate_cards[candidate_id] = frame
            row = QVBoxLayout(frame) if compact_actions else QHBoxLayout(frame)
            row.setContentsMargins(10, 8, 10, 8)
            row.setSpacing(8)
            thumbnail = QLabel("Кадр\nзагружается")
            thumbnail.setObjectName("muted")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setFixedSize(112, 64)
            thumbnail.setStyleSheet("border: 1px solid #303640; border-radius: 4px;")
            self._candidate_thumbnail_labels.setdefault(candidate_id, []).append(thumbnail)
            try:
                start_seconds = float(start_value)
                end_seconds = float(end_value)
            except (TypeError, ValueError):
                start_seconds, end_seconds = 0.0, 0.0
            preview_file = self._draft_preview_paths.get(candidate_id)
            raw_preview_contract = item.get("preview")
            preview_contract = raw_preview_contract if isinstance(raw_preview_contract, dict) else {}
            raw_thumbnail_contract = preview_contract.get("thumbnail")
            thumbnail_contract = raw_thumbnail_contract if isinstance(raw_thumbnail_contract, dict) else {}
            thumbnail_time = thumbnail_contract.get("timestamp_seconds", start_seconds + max(0.0, min(1.0, (end_seconds - start_seconds) / 2)))
            try:
                thumbnail_seconds = float(thumbnail_time)
            except (TypeError, ValueError):
                thumbnail_seconds = start_seconds
            thumbnail_source = preview_file if workflow_step == "drafts" and preview_file else project.source
            if thumbnail_source.is_file():
                if preview_file and workflow_step == "drafts":
                    thumbnail_seconds = 0.05
                thumbnail_path = self._thumbnail_loader.request(
                    cache_directory=project.directory / ("draft-thumbnails" if preview_file and workflow_step == "drafts" else "candidate-thumbnails"),
                    analysis_id=(project.analysis_id or "analysis") + ("-draft" if preview_file and workflow_step == "drafts" else ""),
                    candidate_id=candidate_id,
                    source_path=thumbnail_source,
                    timestamp_seconds=thumbnail_seconds,
                )
                self._candidate_thumbnail_paths[candidate_id] = thumbnail_path
            information = QVBoxLayout()
            title = _ElidedLabel(str(item.get("title") or item.get("core_idea") or "Момент из видео"))
            title.setObjectName("candidateTitle")
            title.setStyleSheet("font-weight: 600;")
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
            if compact_actions:
                summary_row = QHBoxLayout()
                summary_row.setContentsMargins(0, 0, 0, 0)
                summary_row.setSpacing(8)
                summary_row.addWidget(thumbnail)
                summary_row.addLayout(information, 1)
                row.addLayout(summary_row)
            else:
                row.addWidget(thumbnail)
                row.addLayout(information, 1)
            actions_host = QWidget()
            actions_host.setMinimumWidth(0)
            actions_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            if not compact_actions:
                # The longest normal action label needs roughly 266 logical
                # pixels with the current Windows font.  Keep the rail wide
                # enough for its text at Full HD; narrow layouts stack it
                # below the card summary instead.
                actions_host.setMaximumWidth(320)
            actions = QVBoxLayout(actions_host)
            actions.setContentsMargins(0, 0, 0, 0)
            actions.setSpacing(6)
            status = QLabel(status_label)
            status.setObjectName("muted")
            status.setMaximumWidth(16_777_215 if compact_actions else 320)
            status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            candidate_error = project.candidate_errors.get(candidate_id)
            if candidate_error:
                # The service stores an item/stage message for the log.  Card
                # copy stays short and non-technical, with the lifecycle state
                # above explaining exactly which retry is safe.
                status.setText(status_label)
                status.setToolTip("Подробности сохранены в журнале проекта.")
            status.setWordWrap(True)
            actions.addWidget(status)
            source_preview = QPushButton("Исходный фрагмент" if workflow_step == "drafts" else "Просмотреть")
            source_preview.setObjectName(f"preview-candidate-{candidate_id}")
            source_preview.clicked.connect(lambda _checked=False, value=dict(item): self._preview_candidate(value))
            actions.addWidget(source_preview)
            if preview_file and preview_file.is_file():
                button = QPushButton("Смотреть черновик")
                button.clicked.connect(
                    lambda _checked=False, path=preview_file, title=str(item.get("title") or item.get("core_idea") or "момент"), value=candidate_id:
                    self._show_draft_preview(path, title, value)
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
                if export_status == "failed":
                    retry_export = QPushButton("Повторить экспорт")
                    retry_export.setToolTip("Повторно создаст готовый ролик из сохранённого черновика; анализ и предпросмотр не повторяются.")
                    retry_export.clicked.connect(lambda _checked=False, value=candidate_id: self._retry_final_export(value))
                    actions.addWidget(retry_export)
                reject = QPushButton("Отклонить")
                reject.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                actions.addWidget(reject)
            elif state == "draft_failed" and workflow_step == "drafts":
                if candidate_id in project.review_selected_candidate_ids:
                    retry = QPushButton("Повторить черновик")
                    retry.setToolTip("Повторно создаст только этот черновик; найденные моменты не будут анализироваться заново.")
                    retry.clicked.connect(lambda _checked=False, value=candidate_id: self._retry_draft(value))
                    actions.addWidget(retry)
                    skip = QPushButton("Продолжить без этого")
                    skip.setToolTip("Уберёт этот неготовый черновик из текущего набора, не затрагивая готовые ролики.")
                    skip.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                    actions.addWidget(skip)
                else:
                    restore = QPushButton("Вернуть в набор")
                    restore.setToolTip("Вернёт этот черновик в текущий набор, после чего его можно повторить отдельно.")
                    restore.clicked.connect(lambda _checked=False, value=candidate_id: self._restore_draft(value))
                    actions.addWidget(restore)
            elif state == "rendered":
                final_file = final_outputs.get(candidate_id)
                if final_file:
                    watch_final = QPushButton("Смотреть готовый ролик")
                    watch_final.clicked.connect(
                        lambda _checked=False, path=final_file, title=str(item.get("title") or item.get("core_idea") or "момент"), value=candidate_id:
                        self._show_final_preview(path, title, value)
                    )
                    actions.addWidget(watch_final)
                    open_final = QPushButton("Открыть готовый ролик")
                    open_final.clicked.connect(lambda _checked=False, path=final_file: self._open_file(path))
                    actions.addWidget(open_final)
            elif workflow_step == "candidates" and state not in {"draft_planning", "production_rendering"}:
                selected_for_draft = candidate_id in project.review_selected_candidate_ids
                select = QPushButton("Убрать из черновиков" if selected_for_draft else "Добавить к черновикам")
                select.setObjectName(f"select-candidate-{candidate_id}")
                select.clicked.connect(lambda _checked=False, value=candidate_id: self._toggle_candidate_selection(value))
                self._candidate_selection_buttons[candidate_id] = select
                actions.addWidget(select)
            if candidate_error:
                open_log = QPushButton("Открыть журнал")
                open_log.setObjectName(f"candidate-log-{candidate_id}")
                open_log.clicked.connect(self._open_latest_run_log_folder)
                actions.addWidget(open_log)
            for button in actions_host.findChildren(QPushButton):
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(actions_host)
            layout.addWidget(frame)
        if len(filtered_candidates) > len(visible_candidates):
            show_more = QPushButton(f"Показать ещё {min(12, len(filtered_candidates) - len(visible_candidates))} моментов")
            show_more.setObjectName("secondaryAction")
            show_more.clicked.connect(self._show_more_candidates)
            layout.addWidget(show_more)
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
        self.view_all_button.hide()
        self.production_button.hide()
        self.draft_button.setDisabled(True)
        self.production_button.setDisabled(True)
        if project.status in {"analyzing", "processing", "rendering_selected"} or processing_count:
            self.workflow_hint.setText("Сейчас идёт работа. Прогресс и оставшееся время показаны на отдельном экране.")
            return
        if self._derive_flow_step(project) == "candidates":
            recommended_ids = self._recommended_candidate_ids()
            selected_ids = [
                candidate_id for candidate_id in project.review_selected_candidate_ids
                if candidate_id in self._review_candidates_by_id
            ]
            if not selected_ids and recommended_ids:
                count = len(recommended_ids)
                self.workflow_hint.setText(
                    f"Найдено {len(self._review_candidates_by_id)} · рекомендуем {count}. "
                    "Можно сразу подготовить лучшие варианты или изменить выбор в списке."
                )
                self.view_all_button.setText(f"Посмотреть все {len(self._review_candidates_by_id)}")
                self.view_all_button.show()
                self.draft_button.setText(f"Создать {count} рекомендованных")
                self.draft_button.setEnabled(True)
                self.draft_button.show()
                return
        if draftable_ids:
            count = len(draftable_ids)
            selected_count = sum(
                candidate_id in self._review_candidates_by_id
                for candidate_id in project.review_selected_candidate_ids
            )
            selection_summary = (
                f"Выбрано {count}."
                if selected_count == count
                else f"Выбрано {selected_count}. Для {count} из них ещё нужен черновик."
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
        self.workflow_hint.setText(
            "Посмотрите моменты и выберите любое число подходящих фрагментов."
        )

    def _runs_for_project(self, project: DesktopProject) -> list[ProjectRun]:
        if self.runs and all(run.project_id == project.project_id for run in self.runs):
            return self.runs
        return self.viewmodel.services.runs_for(project)

    def _final_output_records(self, project: DesktopProject) -> list[ClipResult]:
        """Read only the canonical result registry, never folders or list positions."""

        collected: list[ClipResult] = []
        for run in self._runs_for_project(project):
            if not run.report_path:
                continue
            report = read_json(Path(run.report_path), {})
            if not isinstance(report, dict):
                continue
            raw_registry = report.get("primary_results")
            if isinstance(raw_registry, list):
                registry = [item for raw in raw_registry if (item := ClipResult.from_dict(raw)) is not None]
            else:
                registry = primary_clip_results(report.get("production_render"))
            for result in registry:
                path = Path(result.output_file)
                if path.is_absolute() and VideoPreview.usable_media_path(path):
                    collected.append(result)
        return unique_primary_results(collected)

    def _final_outputs_by_candidate(self) -> dict[str, Path]:
        """Compatibility map for the candidate workspace, based on canonical records."""

        if not self.project:
            return {}
        return {
            result.candidate_id: Path(result.output_file)
            for result in self._final_output_records(self.project)
        }

    def _update_final_results(self, project: DesktopProject) -> None:
        records = self._final_output_records(project)
        if not records:
            return
        candidates = self._final_candidate_metadata(project)
        outputs: list[FinalOutput] = []
        for result in records:
            path = Path(result.output_file)
            # Do not call ffprobe from a screen refresh.  The canonical result
            # registry already validated this artifact; any optional media
            # dimensions are rendered as unknown rather than blocking the UI.
            media: dict[str, object] = {}
            result_id = result.clip_result_id or result.candidate_id
            outputs.append(FinalOutput(
                result_id=result_id,
                candidate_id=result.candidate_id,
                path=path,
                title=str(candidates.get(result.candidate_id, {}).get("title") or f"Ролик из момента {result.candidate_id}"),
                duration_seconds=self._float_or_none(media.get("duration")),
                width=self._int_or_none(media.get("width")),
                height=self._int_or_none(media.get("height")),
                source_start_seconds=(
                    result.source_start_seconds
                    if result.source_start_seconds is not None
                    else self._float_or_none(candidates.get(result.candidate_id, {}).get("start"))
                ),
                source_end_seconds=(
                    result.source_end_seconds
                    if result.source_end_seconds is not None
                    else self._float_or_none(candidates.get(result.candidate_id, {}).get("end"))
                ),
                status=result.status,
                run_id=self._run_id_for_result(project, result),
            ))
        self.final_results.set_results(
            outputs,
            selected_id=project.last_final_result_id,
            project_directory=project.directory,
            warnings=self._final_warnings(project),
        )

    def _final_candidate_metadata(self, project: DesktopProject) -> dict[str, dict[str, object]]:
        """Read titles and source ranges from already-persisted candidate metadata."""

        path = Path(project.analysis_artifact_path) if project.analysis_artifact_path else None
        analysis = read_json(path, {}) if path and path.is_file() else {}
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        metadata = {
            str(item.get("candidate_id")): {
                "title": str(item.get("title") or item.get("core_idea") or "Готовый ролик"),
                "start": item.get("start_seconds", item.get("start")),
                "end": item.get("end_seconds", item.get("end")),
            }
            for item in candidates if isinstance(item, dict) and item.get("candidate_id")
        }
        for run in self._runs_for_project(project):
            if not run.report_path:
                continue
            report = read_json(Path(run.report_path), {})
            intelligence = report.get("clip_intelligence", {}) if isinstance(report, dict) else {}
            candidates = intelligence.get("candidates", []) if isinstance(intelligence, dict) else []
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "")
                if not candidate_id or candidate_id in metadata:
                    continue
                excerpt = str(candidate.get("title") or candidate.get("core_idea") or candidate.get("text") or "").strip()
                metadata[candidate_id] = {
                    "title": (excerpt[:96].rstrip() + "…") if len(excerpt) > 96 else excerpt,
                    "start": candidate.get("start_seconds", candidate.get("start")),
                    "end": candidate.get("end_seconds", candidate.get("end")),
                }
        return metadata

    def _run_id_for_result(self, project: DesktopProject, result: ClipResult) -> str:
        """Resolve a legacy result's owner through canonical report metadata."""

        if result.run_id:
            return result.run_id
        target = str(Path(result.output_file)).replace("\\", "/").casefold()
        for run in self._runs_for_project(project):
            if not run.report_path:
                continue
            report = read_json(Path(run.report_path), {})
            if not isinstance(report, dict):
                continue
            raw_registry = report.get("primary_results")
            registry = (
                [item for raw in raw_registry if (item := ClipResult.from_dict(raw)) is not None]
                if isinstance(raw_registry, list)
                else primary_clip_results(report.get("production_render"))
            )
            if any(str(Path(item.output_file)).replace("\\", "/").casefold() == target for item in registry):
                return run.run_id
        return ""

    def _final_warnings(self, project: DesktopProject) -> list[str]:
        # QualityReport is the delivery source of truth.  Show its persisted
        # user-facing findings first; raw runner warnings remain a legacy
        # fallback for projects created before the quality gate existed.
        warnings: list[str] = []
        quality_found = False
        for result in self._final_output_records(project):
            quality_path = Path(result.quality_report_path) if result.quality_report_path else None
            raw_quality = read_json(quality_path, {}) if quality_path and quality_path.is_file() else {}
            if not isinstance(raw_quality, dict):
                continue
            quality_found = True
            status = str(raw_quality.get("status") or result.quality_status or "")
            if status == "BLOCKED":
                warnings.append("Этот ролик не прошёл проверку качества и не должен считаться готовым к публикации.")
            for finding in raw_quality.get("findings", []):
                if not isinstance(finding, dict):
                    continue
                message = str(finding.get("user_message") or "").strip()
                if message:
                    warnings.append(message)
            if status == "PASS_WITH_WARNINGS" and not raw_quality.get("findings"):
                warnings.append("Ролик создан с предупреждениями проверки качества. Откройте его и проверьте перед публикацией.")
        if quality_found:
            return list(dict.fromkeys(warnings))
        for run in self._runs_for_project(project):
            warnings.extend(run.warnings)
            if run.report_path:
                report = read_json(Path(run.report_path), {})
                if isinstance(report, dict):
                    values = report.get("warnings", [])
                    if isinstance(values, list):
                        warnings.extend(str(value) for value in values if str(value).strip())
                    production = report.get("production_render", {})
                    if isinstance(production, dict):
                        values = production.get("warnings", [])
                        if isinstance(values, list):
                            warnings.extend(str(value) for value in values if str(value).strip())
        return self._summarize_final_warnings(list(dict.fromkeys(warnings)))

    @staticmethod
    def _summarize_final_warnings(warnings: list[str]) -> list[str]:
        """Keep legacy warnings concrete without exposing implementation logs."""

        summarized: list[str] = []
        evidence_counts: dict[str, int] = {}
        cpu_fallback = False
        for warning in warnings:
            text = str(warning).strip()
            if not text:
                continue
            if text.startswith("Transformation ") and "source_evidence_map was restored" in text:
                evidence_counts["evidence"] = evidence_counts.get("evidence", 0) + 1
                continue
            if text.startswith("NVENC render failed; CPU fallback used"):
                cpu_fallback = True
                continue
            if text == "AI transformation failed -> local fallback used.":
                summarized.append("AI-преобразование недоступно; использован локальный fallback сценария.")
                continue
            summarized.append(text[:280] + ("…" if len(text) > 280 else ""))
        for _candidate_id, count in evidence_counts.items():
            summarized.append(
                f"Связь фактов с исходными фрагментами восстановлена автоматически ({count})."
            )
        if cpu_fallback:
            summarized.append(
                "Аппаратное ускорение было недоступно, поэтому ролик безопасно собран обычным способом."
            )
        return list(dict.fromkeys(summarized))

    def _final_output_selected(self, result_id: str) -> None:
        self.viewmodel.select_final_output(result_id)

    def _create_more_outputs(self) -> None:
        if not self.project:
            return
        if not self.project.analysis_artifact_path:
            QMessageBox.information(
                self,
                "Создать ещё ролики",
                "Для этого проекта не сохранился список моментов. Создать дополнительные ролики без нового анализа нельзя.",
            )
            return
        self._results_subflow_override = "candidates"
        self.preview.show_source(
            self.project.source,
            source_codec=str(self.project.source_metadata.get("video_codec") or ""),
        )
        self._update_candidate_review(self.project)
        self._apply_flow_visibility(self.project)

    def _rerender_final_output(self, run_id: str) -> None:
        """Re-export from the exact completed run without asking for analysis."""

        if not self.project or not run_id:
            return
        run = next((item for item in self._runs_for_project(self.project) if item.run_id == run_id), None)
        if run is None:
            QMessageBox.information(
                self,
                "Повторная сборка недоступна",
                "Для выбранного ролика не удалось найти сохранённый запуск. Остальные результаты останутся без изменений.",
            )
            return
        self.viewmodel.rerender(run)

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        try:
            return float(cast(Any, value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        try:
            return int(cast(Any, value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _draft_action(self) -> None:
        if not self.project:
            return
        candidate_ids = [
            candidate_id for candidate_id in self.project.review_selected_candidate_ids
            if candidate_id in self._review_candidates_by_id
        ]
        if candidate_ids != self.project.review_selected_candidate_ids:
            # A saved legacy/stale choice must not bypass the current Moments
            # boundary when the user starts a new draft batch.
            self.viewmodel.set_review_selection(candidate_ids)
            if not self.project:
                return
        if not candidate_ids:
            candidate_ids = self._recommended_candidate_ids()
            if not candidate_ids:
                return
            self.viewmodel.set_review_selection(candidate_ids)
            if not self.project:
                return
        needs_draft = [
            candidate_id for candidate_id in candidate_ids
            if self._candidate_needs_draft(self.project, candidate_id)
        ]
        if needs_draft:
            self.viewmodel.build_drafts(needs_draft)

    @staticmethod
    def _candidate_needs_draft(project: DesktopProject, candidate_id: str) -> bool:
        """Return whether one review item lacks a usable immutable draft."""

        state = project.candidate_states.get(candidate_id)
        if state in {"rendered", "production_rendering", "draft_planning"}:
            return False
        if state in {"draft_ready", "selected"}:
            return not Path(project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
        return True

    def _retry_draft(self, candidate_id: str) -> None:
        """Retry one failed draft without broadening the existing selection."""

        if not self.project or candidate_id not in self.project.review_selected_candidate_ids:
            return
        if self.project.candidate_states.get(candidate_id) == "draft_failed":
            self.viewmodel.build_drafts([candidate_id])

    def _retry_final_export(self, candidate_id: str) -> None:
        """Retry production only for the already-confirmed draft set."""

        if not self.project or candidate_id not in self.project.selected_candidate_ids:
            return
        if self.project.candidate_export_statuses.get(candidate_id) == "failed":
            self._confirm_production_render([candidate_id])

    def _open_latest_run_log_folder(self, *_: object) -> None:
        if not self.project:
            return
        latest = self._latest_run(self.project)
        self._open_folder(Path(latest.log_path).parent if latest and latest.log_path else None)

    def _toggle_candidate_selection(self, candidate_id: str) -> None:
        if not self.project or candidate_id not in self._review_candidates_by_id:
            return
        selected = list(self.project.review_selected_candidate_ids)
        if candidate_id in selected:
            selected.remove(candidate_id)
        else:
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
        self.viewmodel.set_review_selection([*self.project.review_selected_candidate_ids, candidate_id])

    def _recommended_candidate_ids(self) -> list[str]:
        return [
            candidate_id
            for candidate_id, item in self._review_candidates_by_id.items()
            if item.get("recommended", item.get("selected_by_recommendation"))
        ]

    def _select_recommended(self) -> None:
        if not self.project:
            return
        self.viewmodel.set_review_selection(self._recommended_candidate_ids())

    def _select_all_candidates(self) -> None:
        if self.project:
            self.viewmodel.set_review_selection(list(self._review_candidates_by_id))

    def _view_all_candidates(self) -> None:
        if not self.project:
            return
        self._candidate_filter = "all"
        self._candidate_visible_limit = max(12, len(self._review_candidates_by_id))
        self._update_candidate_review(self.project)
        self.content_scroll.ensureWidgetVisible(self.candidate_review, 0, 16)
        self.candidate_review.setFocus(Qt.FocusReason.OtherFocusReason)

    def _clear_review_selection(self) -> None:
        self.viewmodel.set_review_selection([])

    def _change_candidate_filter(self, value: str) -> None:
        self._candidate_filter = value
        self._candidate_visible_limit = 12
        if self.project:
            self._update_candidate_review(self.project)

    def _change_candidate_sort(self, value: str) -> None:
        self._candidate_sort = value
        if self.project:
            self._update_candidate_review(self.project)

    def _show_more_candidates(self) -> None:
        self._candidate_visible_limit += 12
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
        self._bind_source_candidate(candidate, start, end, autoplay=True, force=True)
        self._focus_preview_player()
        self._show_candidate_detail(candidate, start, end)

    def _show_draft_preview(self, path: Path, title: str, candidate_id: str | None = None) -> None:
        candidate = self._review_candidates_by_id.get(candidate_id or "")
        if isinstance(candidate, dict):
            try:
                start, end = self._candidate_range(candidate)
            except (TypeError, ValueError):
                start, end = 0.0, 0.0
            self._bind_draft_candidate(candidate, path, start, end, title=title, force=True)
        else:
            self._active_preview_kind = "draft"
            self.preview.show_draft(str(path), title)
        self._focus_preview_player()

    def _show_final_preview(self, path: Path, title: str | None = None, candidate_id: str | None = None) -> None:
        if candidate_id:
            candidate = self._review_candidates_by_id.get(candidate_id)
            if isinstance(candidate, dict):
                try:
                    start, end = self._candidate_range(candidate)
                except (TypeError, ValueError):
                    start, end = 0.0, 0.0
                self._set_active_candidate_binding(candidate_id, start, end, kind="final")
            else:
                self._active_candidate_id = candidate_id
                self._active_preview_kind = "final"
        self.preview.show_final(str(path), title)
        if candidate_id:
            self._persist_active_preview_candidate(candidate_id)
        self._focus_preview_player()

    def _set_active_candidate_binding(
        self, candidate_id: str, start: float, end: float, *, kind: str,
    ) -> None:
        self._active_candidate_id = candidate_id
        self._active_candidate_range = (candidate_id, start, end)
        self._active_preview_kind = kind
        self._mark_active_candidate()

    def _persist_active_preview_candidate(self, candidate_id: str | None) -> None:
        """Save a deliberate preview choice without coupling it to approvals.

        The setter is intentionally optional while older in-memory view models
        are still supported by tests and migration paths.
        """

        if not self.project or getattr(self.project, "active_preview_candidate_id", None) == candidate_id:
            return
        if candidate_id is not None and candidate_id not in self.project.candidate_states:
            return
        setter = getattr(self.viewmodel, "set_active_preview_candidate", None)
        if callable(setter):
            self._persisting_active_preview = True
            try:
                setter(candidate_id)
            finally:
                self._persisting_active_preview = False

    def _bind_source_candidate(
        self, candidate: dict, start: float, end: float, *, autoplay: bool, force: bool = False,
    ) -> None:
        if not self.project:
            return
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return
        binding = (candidate_id, start, end)
        reload_needed = (
            force
            or self._active_preview_kind != "source-range"
            or self._active_candidate_range != binding
            or self.preview.source_range_seconds != (start, end)
        )
        self._set_active_candidate_binding(candidate_id, start, end, kind="source-range")
        if reload_needed:
            self.preview.set_range(
                self.project.source, start, end,
                autoplay=autoplay,
                cache_directory=self.project.directory / "preview-proxies",
                candidate_title=str(candidate.get("title") or candidate.get("core_idea") or "Выбранный кандидат"),
                source_codec=str(self.project.source_metadata.get("video_codec") or ""),
            )
        self._persist_active_preview_candidate(candidate_id)

    def _bind_draft_candidate(
        self, candidate: dict, path: Path, start: float, end: float, *, title: str | None = None, force: bool = False,
    ) -> None:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return
        binding = (candidate_id, start, end)
        reload_needed = (
            force
            or self._active_preview_kind != "draft"
            or self._active_candidate_range != binding
            or self.preview.active_media_path != path
        )
        self._set_active_candidate_binding(candidate_id, start, end, kind="draft")
        if reload_needed:
            self.preview.show_draft(str(path), title or str(candidate.get("title") or candidate.get("core_idea") or "момент"))
        self._persist_active_preview_candidate(candidate_id)

    def _mark_active_candidate(self) -> None:
        for candidate_id, card in self._candidate_cards.items():
            active = candidate_id == self._active_candidate_id
            card.setProperty("activeCandidate", active)
            # Selection for preview is orange-outlined in the shared theme;
            # it remains independent from the candidate's checkbox/selection
            # state used to build drafts.
            card.style().unpolish(card)
            card.style().polish(card)

    def _focus_preview_player(self, *_: object) -> None:
        self.content_scroll.ensureWidgetVisible(self.preview, 0, 16)
        self.preview.play_button.setFocus(Qt.FocusReason.OtherFocusReason)
        # A persisted preview selection rebuilds the card list synchronously.
        # Queue one last focus hand-off after the originating card click has
        # completed, otherwise Qt restores focus to that now-stale button.
        QTimer.singleShot(0, self._restore_preview_focus)

    def _restore_preview_focus(self) -> None:
        if self.preview.isVisible():
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
            if self.project.candidate_export_statuses.get(candidate_id) == "failed":
                lines.append("Готовый ролик для этого момента не создан. Сохранённый черновик остаётся подтверждённым: повторите только экспорт или снимите подтверждение.")
            else:
                lines.append("Черновик для этого момента не создан. Его можно повторить отдельно; подробности сохранены в журнале проекта.")
        self._replace_card_text(self.candidate_detail, lines)
        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        columns = 2
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
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(button, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        self.candidate_detail.layout().addWidget(controls)

    @staticmethod
    def _score_text(value: object) -> str:
        try:
            return f"{float(cast(Any, value)):.0f}/100"
        except (TypeError, ValueError):
            return "—"

    def _adjust_candidate_boundary(self, candidate_id: str, boundary: str, delta_seconds: float) -> None:
        if candidate_id:
            self.viewmodel.adjust_candidate_boundary(candidate_id, boundary, delta_seconds)

    def _refresh_active_candidate_detail(self, project: DesktopProject) -> None:
        self._reconcile_active_candidate_preview(project, previous_step=self._flow_step)

    def _reconcile_active_candidate_preview(self, project: DesktopProject, *, previous_step: str) -> None:
        """Keep the visible player bound to a current, durable candidate.

        Project changes happen for selection, approval and run-state updates.
        They must not restart an unchanged source interval.  Conversely, an
        edited boundary makes an existing draft preview stale, so the player
        deliberately returns to that exact source range until a new draft is
        rendered.
        """

        workflow_step = self._derive_flow_step(project)
        if workflow_step not in {"candidates", "drafts"}:
            return
        persisted_id = getattr(project, "active_preview_candidate_id", None)
        candidate_id = self._active_candidate_id or persisted_id
        candidate = self._review_candidates_by_id.get(candidate_id or "")
        if not isinstance(candidate, dict) and workflow_step == "drafts":
            ordered_ids = [
                *project.review_selected_candidate_ids,
                *self._review_visible_candidate_ids,
                *self._review_candidates_by_id.keys(),
            ]
            for possible_id in dict.fromkeys(ordered_ids):
                possible = self._review_candidates_by_id.get(possible_id)
                if isinstance(possible, dict):
                    candidate = possible
                    candidate_id = possible_id
                    break
        if not isinstance(candidate, dict):
            had_active_candidate = bool(self._active_candidate_id or persisted_id)
            self._active_candidate_id = None
            self._active_candidate_range = None
            self._active_preview_kind = "source"
            self._mark_active_candidate()
            if had_active_candidate:
                self._persist_active_preview_candidate(None)
            if project.source_spec.is_ready and self.preview.source_range_seconds is not None:
                self.preview.show_source(
                    project.source,
                    source_codec=str(project.source_metadata.get("video_codec") or ""),
                )
            return
        try:
            start, end = self._candidate_range(candidate)
        except (TypeError, ValueError):
            return
        candidate_id = str(candidate.get("candidate_id") or candidate_id or "")
        if not candidate_id:
            return
        binding = (candidate_id, start, end)
        draft_path = self._draft_preview_paths.get(candidate_id)
        previous_binding = self._active_candidate_range
        boundary_changed = (
            previous_binding is not None
            and previous_binding[0] == candidate_id
            and previous_binding != binding
        )
        if previous_binding != binding:
            if workflow_step == "drafts" and draft_path and not boundary_changed:
                self._bind_draft_candidate(candidate, draft_path, start, end)
            else:
                self._bind_source_candidate(candidate, start, end, autoplay=False)
            self._show_candidate_detail(candidate, start, end)
            return
        if workflow_step == "drafts":
            if self._active_preview_kind == "draft" and not draft_path:
                self._bind_source_candidate(candidate, start, end, autoplay=False, force=True)
            elif previous_step != "drafts" and draft_path:
                self._bind_draft_candidate(candidate, draft_path, start, end)
        elif previous_step != "candidates" and self._active_preview_kind != "source-range":
            self._bind_source_candidate(candidate, start, end, autoplay=False)
        self._mark_active_candidate()
        self._show_candidate_detail(candidate, start, end)

    def _thumbnail_ready(self, candidate_id: str, path: str) -> None:
        expected = self._candidate_thumbnail_paths.get(candidate_id)
        if expected is None or expected.resolve(strict=False) != Path(path).resolve(strict=False):
            return
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

    def _thumbnail_unavailable(self, candidate_id: str, path: str) -> None:
        expected = self._candidate_thumbnail_paths.get(candidate_id)
        if expected is None or expected.resolve(strict=False) != Path(path).resolve(strict=False):
            return
        for label in self._candidate_thumbnail_labels.get(candidate_id, []):
            try:
                label.setText("Кадр\nнедоступен")
            except RuntimeError:
                continue

    def _confirm_production_render(self, candidate_ids: list[str] | None = None) -> None:
        """Ask for delivery using only the currently approved project state.

        ``candidate_ids`` is a narrow retry allow-list, never a source of
        truth.  In particular, a Qt ``clicked(bool)`` value cannot become an
        iterable here or start a delivery for a stale candidate.
        """

        if not self.project:
            QMessageBox.information(
                self,
                "Проект не открыт",
                "Сначала откройте проект с подтверждёнными черновиками.",
            )
            return
        approved_ids = list(dict.fromkeys(
            str(candidate_id) for candidate_id in self.project.selected_candidate_ids if str(candidate_id)
        ))
        retry_ids: set[str] | None = None
        if isinstance(candidate_ids, (list, tuple, set, frozenset)):
            retry_ids = {str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)}
        selected_ids = [candidate_id for candidate_id in approved_ids if retry_ids is None or candidate_id in retry_ids]
        if not selected_ids:
            if retry_ids:
                message = "Этот черновик больше не подтверждён для финального экспорта. Подтвердите его снова в списке черновиков."
            else:
                message = "Сначала подтвердите хотя бы один готовый черновик. После этого здесь появится запуск финального экспорта."
            QMessageBox.information(self, "Нет подтверждённых черновиков", message)
            return
        singular_retry = len(selected_ids) == 1 and retry_ids is not None
        prompt = (
            "Повторить экспорт только этого подтверждённого черновика?"
            if singular_retry
            else "Создать готовые вертикальные ролики только для подтверждённых черновиков?"
        )
        answer = QMessageBox.question(
            self, "Создать итоговые ролики",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.viewmodel.render_selected(selected_ids)

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
        if self.project:
            self._update_final_results(self.project)
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
            if (
                not isinstance(profile, dict)
                or not isinstance(content_map, dict)
                or not isinstance(recommendation, dict)
                or not isinstance(coverage, dict)
            ):
                continue
            clip_range = recommendation.get("estimated_publishable_clip_range", {})
            if not isinstance(clip_range, dict):
                clip_range = {}
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
            self.next_step_text.setText(
                f"Моменты готовы. Посмотрите их и добавьте к черновикам от одного до {self._selection_limit(project)} лучших."
            )
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
        self._update_processing_stages(snapshot)
        if active:
            detail = self._processing_detail(snapshot)
            self.progress.set_running(
                snapshot.stage_label,
                f"Прошло {format_seconds(snapshot.elapsed_seconds)}",
                snapshot.progress_fraction,
                detail,
                cancelling=snapshot.phase == "cancelling",
                long_stage_warning=snapshot.long_stage_warning,
            )
        else:
            message = snapshot.message
            if self.project:
                latest = self._latest_run(self.project)
                if latest and latest.status in {"failed", "interrupted", "cancelled"}:
                    message = self._recovery_message(latest)
            self.progress.set_finished(message)
        self.run_button.setDisabled(active)
        self.setup_start_button.setDisabled(active)
        has_draft_choice = bool(
            self.project
            and (self.project.review_selected_candidate_ids or self._recommended_candidate_ids())
        )
        self.draft_button.setDisabled(active or not has_draft_choice)
        self.view_all_button.setDisabled(active)
        selected_drafts_exist = bool(self.project and self.project.selected_candidate_ids) and all(
            Path(self.project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
            for candidate_id in self.project.selected_candidate_ids
        ) if self.project else False
        self.production_button.setDisabled(active or not selected_drafts_exist)
        for widget in (
            self.processing_mode, self.deep_analysis, self.platform, self.clip_count,
            self.audio_mode, self.composition_strategy, self.same_source_broll,
            self.subtitles, self.subtitle_style, self.cache,
            self.setup_processing_mode, self.setup_deep_analysis, self.setup_platform, self.setup_clip_count,
        ):
            widget.setDisabled(active)
        if self.project:
            self._update_download_card(self.project)
            self._update_stage_context(self.project)
            self._apply_flow_visibility(self.project)

    def _update_processing_stages(self, snapshot: ProcessingSnapshot) -> None:
        """Show stages for the current job, never a stale analysis template."""

        rows = self._processing_stage_plan(snapshot)
        names = tuple(name for name, _label in rows)
        if names != self._processing_stage_rows:
            while self._processing_stages_layout.count() > 1:
                item = self._processing_stages_layout.takeAt(1)
                if widget := item.widget():
                    widget.deleteLater()
            self.processing_stage_labels = {}
            for name, label in rows:
                row = QLabel(f"○  {label}")
                row.setObjectName("processingStage")
                row.setProperty("stageState", "pending")
                self._processing_stages_layout.addWidget(row)
                self.processing_stage_labels[name] = row
            self._processing_stage_rows = names

        raw = str(snapshot.stage or "").lower()
        target = self._processing_stage_target(raw, self._processing_run_kind(), len(rows))
        for index, (name, label) in enumerate(rows):
            state = "active" if index == target else ("done" if index < target else "pending")
            if snapshot.phase not in {"preparing", "running", "cancelling"} and state == "active":
                state = "pending"
            widget = self.processing_stage_labels[name]
            marker = {"done": "✓", "active": "◉", "pending": "○"}[state]
            widget.setText(f"{marker}  {label}")
            widget.setProperty("stageState", state)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _processing_run_kind(self) -> str:
        run = self.viewmodel.run
        if run is not None:
            return str(run.run_kind or "")
        if self.project:
            latest = self._latest_run(self.project)
            if latest:
                return str(latest.run_kind or "")
        return ""

    def _processing_stage_plan(self, snapshot: ProcessingSnapshot) -> tuple[tuple[str, str], ...]:
        if str(snapshot.stage or "").lower() == "download":
            return (
                ("download", "Загружаем исходное видео"),
                ("verify", "Проверяем локальный файл"),
            )
        kind = self._processing_run_kind()
        if kind == "draft":
            return (
                ("prepare", "Подготавливаем выбранные моменты"),
                ("draft", "Создаём черновики"),
                ("verify", "Проверяем черновики"),
            )
        if kind in {"selected_render", "render_revision"}:
            return (
                ("prepare", "Подготавливаем подтверждённые черновики"),
                ("render", "Создаём готовые ролики"),
                ("verify", "Проверяем готовые файлы"),
            )
        if kind == "full":
            return (
                ("prepare", "Подготавливаем видео"),
                ("transcribe", "Разбираем речь и структуру"),
                ("analyze", "Ищем сильные моменты"),
                ("render", "Собираем ролики"),
            )
        return (
            ("prepare", "Подготавливаем видео"),
            ("transcribe", "Разбираем речь и структуру"),
            ("analyze", "Ищем сильные моменты"),
        )

    @staticmethod
    def _processing_stage_target(raw: str, run_kind: str, row_count: int) -> int:
        if row_count <= 1:
            return 0
        if any(token in raw for token in ("verify", "valid", "finaliz", "manifest")):
            return row_count - 1
        if run_kind == "draft":
            return 1 if any(token in raw for token in ("draft", "script", "subtitle", "compose", "render")) else 0
        if run_kind in {"selected_render", "render_revision"}:
            return 1 if any(token in raw for token in ("render", "production", "export", "encode", "subtitle", "compose")) else 0
        if any(token in raw for token in ("transcrib", "speech", "audio")):
            return min(1, row_count - 1)
        if any(token in raw for token in ("analy", "candidate", "intelligence", "select")):
            return min(2, row_count - 1)
        if any(token in raw for token in ("render", "production", "export", "subtitle", "compose")):
            return row_count - 1
        return 0

    @staticmethod
    def _recovery_message(run: ProjectRun) -> str:
        if run.status == "interrupted":
            return "Работа была прервана. Готовые результаты сохранены; незавершённое можно запустить снова."
        if run.status == "cancelled":
            return "Работа остановлена. Готовые результаты сохранены; незавершённое можно запустить снова."
        return run.error_summary or "Не удалось завершить этот этап. Проверьте настройки и повторите только нужную операцию."

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
            size = float(cast(Any, value))
        except (TypeError, ValueError):
            return "н/д"
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if size < 1024 or unit == "ГБ":
                return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
            size /= 1024
        return "н/д"

    def _error(self, error) -> None:
        # Error dialogs are an action surface, not a dump of subprocess
        # diagnostics.  The redacted technical detail stays in the persisted
        # run log, where it can be inspected without exposing it in cards or
        # ordinary recovery dialogs.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(error.title)
        box.setText(error.user_message)
        box.setInformativeText(error.suggested_action)
        log_button = None
        if self.project:
            latest = self._latest_run(self.project)
            if latest and latest.log_path:
                log_button = box.addButton("Открыть папку журнала", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if log_button is not None and box.clickedButton() is log_button and self.project:
            latest = self._latest_run(self.project)
            self._open_folder(Path(latest.log_path).parent if latest and latest.log_path else None)
