from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import (
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QBoxLayout, QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from app.analysis_artifact import candidate_is_draftable
from app.caption_presets import CAPTION_PRESET_DEFINITIONS
from app.clip_results import ClipResult, unique_primary_results
from app.content_profile_taxonomy import (
    AUTO_PROFILE_INPUT,
    CONTENT_PROFILE_PRESETS,
    ProfileAxisId,
    user_overridable_values,
)
from app.editorial_profile_policy import evaluate_editorial_candidate
from app.font_assets import FONT_ASSET_DEFINITIONS
from app.gui.components import (
    CandidateThumbnailLoader, CaptionPresetPicker, CaptionPresetPickerDialog,
    FinalOutput,
    FinalResultsWorkspace,
    ProcessingProgress,
    ProjectPosterLoader,
    VideoPreview,
)
from app.gui.components.project_poster import project_poster_has_input, project_poster_path
from app.gui.models import DesktopProject, ProcessingSnapshot, ProjectPresentation, ProjectRun, RunKind
from app.gui.responsive import break_long_tokens, make_label_shrinkable, set_responsive_text
from app.gui.viewmodels import ProjectViewModel
from app.settings_preview_assets import settings_preview_path
from app.utils import format_seconds, read_json


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


def _populate_profile_override(combo: QComboBox, axis_id: ProfileAxisId) -> None:
    """Populate a user control from the canonical profile registry."""

    combo.addItem("Авто", AUTO_PROFILE_INPUT)
    for value in user_overridable_values(axis_id):
        combo.addItem(value.label, value.id)


def _populate_content_profile_preset(combo: QComboBox) -> None:
    """Populate the single user-facing category shortcut in contract order."""

    combo.addItem("Авто", AUTO_PROFILE_INPUT)
    for preset in CONTENT_PROFILE_PRESETS.values():
        combo.addItem(preset.label, preset.id)


_CREATIVE_STYLE_CHOICES = (
    ("Clean", "clean"),
    ("Dynamic", "dynamic"),
    ("Educational", "documentary"),
    ("Minimal Premium", "minimal"),
)
_GLOBAL_FLOW_COMPACT_LABELS = {
    "source": "1  Источник",
    "settings": "2  Настр.",
    "processing": "3  Обработка",
    "results": "4  Результаты",
}


@dataclass(frozen=True)
class _DraftPreviewProjection:
    """One candidate-owned projection of the visible Preview state."""

    stale: bool
    badge_text: str
    badge_state: str
    inspector_text: str
    inspector_state: str


def _populate_creative_styles(combo: QComboBox) -> None:
    for label, value in _CREATIVE_STYLE_CHOICES:
        combo.addItem(label, value)


def _populate_caption_presets(combo: QComboBox) -> None:
    for preset in CAPTION_PRESET_DEFINITIONS.values():
        font = FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id]
        combo.addItem(preset.label, preset.preset_id)
        combo.setItemData(
            combo.count() - 1,
            f"{preset.label} · встроенный шрифт {font.family} ({font.file_name})",
            Qt.ItemDataRole.ToolTipRole,
        )


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

    # The approved desktop composition uses three compact review columns at
    # ordinary scaled-laptop widths.  Below this point the same widgets stack
    # into the one outer scroll owner; no horizontal scroll is introduced.
    _WIDE_STAGE_MINIMUM_WIDTH = 1040
    # The full review rail (Back + hint + two dynamic actions) needs about
    # 1,000 logical pixels after page margins with the shipped Windows font.
    # Required scaled-laptop clients around 911–937px must retain short labels.
    _COMPACT_ACTION_BREAKPOINT = 1100
    # Four abbreviated steps remain readable inside the shell at scaled
    # desktop widths.  Below this genuinely narrow client only the current
    # step stays visible so the row never creates horizontal overflow.
    _COMPLETE_STEPPER_MINIMUM_WIDTH = 620

    _SAME_SOURCE_BROLL_TEXT = "Использовать дополнительные кадры из этого видео"
    _SAME_SOURCE_BROLL_COMPACT_TEXT = "Использовать дополнительные кадры\nиз этого видео"
    _CACHE_TEXT = "Использовать готовый анализ, если он есть"
    _CACHE_COMPACT_TEXT = "Использовать готовый анализ,\nесли он есть"
    _FLOW_HINT_MAX_CHARS = 240

    def __init__(self, viewmodel: ProjectViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        self.project: DesktopProject | None = None
        self.runs: list[ProjectRun] = []
        self._pending_project: DesktopProject | None = None
        self._pending_runs: list[ProjectRun] | None = None
        self._persisted_refresh_pending = False
        self._processing_structure_key: tuple[object, ...] | None = None
        self._analysis_cache_key: tuple[object, ...] | None = None
        self._analysis_cache: dict[str, Any] = {}
        self._analysis_reference_stats: tuple[tuple[object, ...], ...] = ()
        self._analysis_load_error: str | None = None
        self._active_candidate_id: str | None = None
        # The player is shared by Moments and Drafts.  Keep its review binding
        # separate from the user choices persisted in the project: a project
        # refresh must not replay an unchanged fragment, while a boundary edit
        # must replace an obsolete draft/source preview with the exact new
        # source range.
        self._active_candidate_range: tuple[str, float, float] | None = None
        self._active_preview_kind = "source"
        self._persisting_active_preview = False
        self._persisting_review_selection = False
        self._all_candidates_by_id: dict[str, dict] = {}
        self._draftable_candidates_by_id: dict[str, dict] = {}
        self._review_candidates_by_id: dict[str, dict] = {}
        self._review_visible_candidate_ids: list[str] = []
        self._draft_preview_paths: dict[str, Path] = {}
        self._candidate_thumbnail_labels: dict[str, list[QLabel]] = {}
        self._candidate_thumbnail_paths: dict[str, Path] = {}
        self._candidate_cards: dict[str, QFrame] = {}
        self._project_thumbnail_labels: list[QLabel] = []
        self._project_thumbnail_path: Path | None = None
        self._settings_demo_identity: tuple[str, str] | None = None
        # Drafts owns one vertical scroll surface at every size.  The two
        # inner scroll areas remain available to Moments, but are bypassed in
        # Drafts so wheel/trackpad input always has one unambiguous owner.
        self._drafts_single_scroll_layout: bool | None = None
        self._drafts_geometry_refresh_pending = False
        self._flow_step = "settings"
        self._results_subflow_override: str | None = None
        # Keep the long analysis list responsive.  Additional cards are added
        # only on explicit request; thumbnails stay on their existing async
        # loader.
        self._candidate_visible_limit = 12
        self._thumbnail_loader = CandidateThumbnailLoader(self)
        self._thumbnail_loader.thumbnail_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.thumbnail_unavailable_with_path.connect(self._thumbnail_unavailable)
        self._project_thumbnail_loader = ProjectPosterLoader(self)
        self._project_thumbnail_loader.poster_ready.connect(self._project_thumbnail_ready)
        self._project_thumbnail_loader.poster_unavailable.connect(self._project_thumbnail_unavailable)
        root = QVBoxLayout(self)
        self._root_layout = root
        root.setContentsMargins(24, 18, 24, 24)
        header = QHBoxLayout()
        self._header_layout = header
        self.back_button = QPushButton("← Проекты")
        self.back_button.setToolTip("Вернуться к списку проектов")
        self.back_button.clicked.connect(self.back_requested)
        self.title = _ElidedLabel("Проект")
        self.title.setObjectName("title")
        self.status = _ElidedLabel("")
        self.status.setObjectName("status")
        # Unlike the stretching project title, this header badge has zero
        # layout stretch. Retain a preferred width so it cannot silently
        # collapse to zero when the medium/wide chrome makes it visible.
        self.status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.status.setMaximumWidth(260)
        self.settings_toggle = QPushButton("Дополнительно")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setToolTip("Показать дополнительные настройки")
        self.folder = QPushButton("Открыть папку")
        self.folder.setToolTip("Открыть папку проекта")
        self.folder.clicked.connect(self._open_project_folder)
        header.addWidget(self.back_button)
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
        flow_layout.setContentsMargins(0, 2, 0, 0)
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
        make_label_shrinkable(self.flow_hint)
        self.flow_route = QLabel("")
        self.flow_route.setObjectName("muted")
        make_label_shrinkable(self.flow_route)
        flow_layout.addWidget(self.flow_position)
        flow_layout.addWidget(self.flow_title)
        flow_layout.addWidget(self.flow_hint)
        # Route workspaces own their approved screen titles and supporting
        # copy.  Keep these labels as compatibility/status owners without
        # repeating the same information below the global stepper.
        self.flow_position.hide()
        self.flow_title.hide()
        self.flow_hint.hide()
        root.addWidget(self.flow_card)
        body = QHBoxLayout()
        self._body_layout = body
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content_host = QWidget()
        left = QVBoxLayout(self.content_host)
        left.setContentsMargins(0, 0, 0, 0)
        self.preview = VideoPreview()
        self.preview.preview_ready.connect(self._focus_preview_player)
        self.preview.geometry_requirement_changed.connect(self._queue_drafts_workspace_geometry)
        left.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter)
        self.download_card = self._card("Загрузка")
        self.download_source = QLabel()
        self.download_source.setObjectName("muted")
        make_label_shrinkable(self.download_source)
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
        make_label_shrinkable(self.setup_source)
        setup_layout.addWidget(self.setup_source)
        self.setup_source.hide()

        editorial_heading = QLabel("Что искать?")
        editorial_heading.setObjectName("setupSectionTitle")
        setup_layout.addWidget(editorial_heading)
        editorial_copy = QLabel("Опишите нужные темы или оставьте поле пустым — ранжирование будет автоматическим.")
        editorial_copy.setObjectName("muted")
        editorial_copy.setWordWrap(True)
        setup_layout.addWidget(editorial_copy)
        editorial_copy.hide()
        self.setup_editorial_intent = QLineEdit()
        self.setup_editorial_intent.setMaxLength(500)
        self.setup_editorial_intent.setPlaceholderText("Например: практические советы, ошибки и сильные выводы")
        self.setup_editorial_intent.setToolTip(
            "Уточняет ранжирование подходящих моментов. Не меняет безопасные границы фрагментов."
        )
        self.setup_editorial_intent.editingFinished.connect(
            lambda: self.viewmodel.save_options(editorial_intent=self.setup_editorial_intent.text().strip())
        )
        setup_layout.addWidget(self.setup_editorial_intent)

        setup_choices = QWidget()
        setup_choices.setObjectName("setupChoices")
        setup_choices_layout = QGridLayout(setup_choices)
        setup_choices_layout.setContentsMargins(0, 4, 0, 0)
        setup_choices_layout.setHorizontalSpacing(12)
        setup_choices_layout.setVerticalSpacing(7)

        self.setup_content_profile = QComboBox()
        _populate_content_profile_preset(self.setup_content_profile)
        self.setup_content_profile.setToolTip(
            "Авто или одна из 15 существующих категорий. Выбор влияет на рекомендации, а не скрывает нормальные моменты."
        )
        self.setup_content_profile.currentIndexChanged.connect(
            lambda _index: self._save_setup_option(
                "content_profile_preset", str(self.setup_content_profile.currentData())
            )
        )
        self.setup_content_profile.hide()

        self.setup_creative_style = QComboBox()
        _populate_creative_styles(self.setup_creative_style)
        self.setup_creative_style.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(
                subtitle_style=str(self.setup_creative_style.currentData()),
                preset_selection_mode="explicit",
            )
        )
        self.setup_creative_style.hide()

        self.setup_caption_preset = QComboBox()
        _populate_caption_presets(self.setup_caption_preset)
        self.setup_caption_preset.setToolTip(
            "Семь production presets используют точные встроенные шрифты и одну identity в Preview и Final."
        )
        self.setup_caption_preset.currentIndexChanged.connect(
            lambda _index: self._save_setup_option(
                "caption_preset_id", str(self.setup_caption_preset.currentData())
            )
        )
        self.setup_caption_preset.hide()

        self.setup_clip_count = QComboBox()
        for label, value in (("Авто", "auto"), ("1 ролик", "1"), ("3 ролика", "3"), ("5 роликов", "5")):
            self.setup_clip_count.addItem(label, value)
        self.setup_clip_count.setToolTip(
            "Это начальная рекомендация. После анализа можно выбрать любое число доступных моментов."
        )
        self.setup_clip_count.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("clip_count", str(self.setup_clip_count.currentData()))
        )
        self.setup_clip_count.hide()

        self._setup_choice_buttons: dict[str, dict[object, QPushButton]] = {}
        self._setup_choice_groups: list[QButtonGroup] = []

        profile_section, profile_layout = self._setup_choice_section(
            "Тип контента", "Авто или любой из 15 существующих профилей."
        )
        profile_values = [("Авто", AUTO_PROFILE_INPUT), *(
            (preset.label, preset.id) for preset in CONTENT_PROFILE_PRESETS.values()
        )]
        profile_buttons = self._add_setup_choice_buttons(
            profile_layout, "profile", profile_values[:5], self.setup_content_profile, columns=5,
        )
        self.setup_profile_more = QWidget()
        more_profile_layout = QGridLayout(self.setup_profile_more)
        more_profile_layout.setContentsMargins(0, 4, 0, 0)
        more_profile_layout.setSpacing(7)
        profile_buttons.update(self._add_setup_choice_buttons(
            more_profile_layout, "profile", profile_values[5:], self.setup_content_profile,
            columns=4, reuse_group=True,
        ))
        profile_layout.addWidget(self.setup_profile_more)
        self.setup_profile_more.hide()
        self.setup_profile_more_toggle = QPushButton("Ещё 11 профилей")
        self.setup_profile_more_toggle.setObjectName("choiceMore")
        self.setup_profile_more_toggle.setCheckable(True)
        self.setup_profile_more_toggle.toggled.connect(self._set_all_profiles_visible)
        profile_layout.addWidget(self.setup_profile_more_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self._setup_choice_buttons["profile"] = profile_buttons
        setup_choices_layout.addWidget(profile_section, 0, 0, 1, 2)

        style_section, style_layout = self._setup_choice_section(
            "Стиль оформления", "Четыре текущие creative families."
        )
        self._setup_choice_buttons["style"] = self._add_setup_choice_buttons(
            style_layout, "style", list(_CREATIVE_STYLE_CHOICES), self.setup_creative_style, columns=4,
        )
        setup_choices_layout.addWidget(style_section, 1, 0, 1, 2)

        caption_section, caption_layout = self._setup_choice_section(
            "Субтитры", "Семь production presets — с теми же bundled fonts, что в Preview и Final."
        )
        self.setup_caption_picker = CaptionPresetPicker(columns=4)
        self.setup_caption_picker.setObjectName("setupCaptionPresetPicker")
        self.setup_caption_picker.setToolTip(
            "Каждая карточка использует точный bundled font и цвет выбранного production preset."
        )
        self.setup_caption_picker.preset_selected.connect(
            lambda preset_id: self._choose_setup_value(self.setup_caption_preset, preset_id)
        )
        caption_layout.addWidget(self.setup_caption_picker)
        setup_choices_layout.addWidget(caption_section, 2, 0, 1, 2)

        count_section, count_layout = self._setup_choice_section(
            "Количество роликов", "Это рекомендация; после анализа можно выбрать любые доступные моменты."
        )
        count_values = [
            (self.setup_clip_count.itemText(index), self.setup_clip_count.itemData(index))
            for index in range(self.setup_clip_count.count())
        ]
        self._setup_choice_buttons["count"] = self._add_setup_choice_buttons(
            count_layout, "count", count_values, self.setup_clip_count, columns=4,
        )
        setup_choices_layout.addWidget(count_section, 3, 0, 1, 2)
        setup_choices_layout.setColumnStretch(0, 1)
        setup_choices_layout.setColumnStretch(1, 1)
        setup_layout.addWidget(setup_choices)

        # These compatibility controls retain their public attributes for
        # persisted tests.  Their visible owners live in the collapsed
        # Advanced panel below.
        self.setup_processing_mode = QComboBox()
        self.setup_processing_mode.addItem("Быстрее — для разговорных видео", "fast")
        self.setup_processing_mode.addItem("Сбалансировано", "standard")
        self.setup_processing_mode.addItem("Тщательнее — для динамичных видео", "maximum")
        self.setup_processing_mode.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("processing_mode", str(self.setup_processing_mode.currentData()))
        )
        self.setup_mode_help = QLabel()
        self.setup_mode_help.setObjectName("muted")
        make_label_shrinkable(self.setup_mode_help)
        self.setup_deep_analysis = QComboBox()
        self.setup_deep_analysis.addItem("Авто — выбрать по содержанию", "auto")
        self.setup_deep_analysis.addItem("Включить", "on")
        self.setup_deep_analysis.addItem("Выключить", "off")
        self.setup_deep_analysis.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("deep_analysis", str(self.setup_deep_analysis.currentData()))
        )
        self.setup_deep_help = QLabel()
        self.setup_deep_help.setObjectName("muted")
        make_label_shrinkable(self.setup_deep_help)
        self.setup_platform = QComboBox()
        self.setup_platform.addItem("TikTok", "tiktok")
        self.setup_platform.addItem("Instagram Reels", "reels")
        self.setup_platform.addItem("YouTube Shorts", "shorts")
        self.setup_platform.addItem("Любая вертикальная лента", "universal")
        self.setup_platform.currentIndexChanged.connect(
            lambda _index: self._save_setup_option("platform", str(self.setup_platform.currentData()))
        )
        self.setup_platform_help = QLabel("Размер и поля ролика будут подготовлены для выбранного места.")
        self.setup_platform_help.setObjectName("muted")
        make_label_shrinkable(self.setup_platform_help)
        self.setup_count_help = QLabel("Количество можно изменить позже — анализ не будет запущен повторно из-за выбора списка.")
        self.setup_count_help.setObjectName("muted")
        make_label_shrinkable(self.setup_count_help)
        setup_layout.addWidget(self.setup_count_help)
        self.setup_count_help.hide()
        self.setup_estimate = QLabel()
        self.setup_estimate.setObjectName("muted")
        make_label_shrinkable(self.setup_estimate)
        setup_layout.addWidget(self.setup_estimate)
        self.setup_estimate.hide()
        self.setup_change = QLabel()
        self.setup_change.setObjectName("muted")
        make_label_shrinkable(self.setup_change)
        setup_layout.addWidget(self.setup_change)
        self.setup_change.hide()
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
        make_label_shrinkable(self.next_step_text)
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
        self.candidate_detail.setMinimumWidth(0)
        self.candidate_detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._replace_card_text(self.candidate_detail, ["Выберите момент в списке, чтобы просмотреть исходный фрагмент."])
        left.addWidget(self.candidate_detail)
        self.candidate_review = self._card("Моменты")
        # The primary action scrolls directly to this workspace.  Make the
        # destination focusable as well, so keyboard focus follows the review
        # action instead of remaining on the header button.
        self.candidate_review.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.candidate_review_layout = self.candidate_review.layout()
        self.candidate_review_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._candidate_selection_buttons: dict[str, QPushButton] = {}
        self._candidate_filter = "all"
        self._candidate_sort = "recommendation"
        self.workflow_hint = QLabel()
        self.workflow_hint.setObjectName("muted")
        make_label_shrinkable(self.workflow_hint)
        self.draft_button = QPushButton("Создать черновики")
        self.draft_button.setObjectName("primary")
        self.draft_button.setProperty("responsiveFullText", "Создать черновики")
        self.draft_button.setProperty("responsiveCompactText", "Создать")
        self.draft_button.clicked.connect(self._draft_action)
        self.view_all_button = QPushButton("Посмотреть все моменты")
        self.view_all_button.setObjectName("secondaryAction")
        self.view_all_button.setProperty("responsiveFullText", "Посмотреть все моменты")
        self.view_all_button.setProperty("responsiveCompactText", "Все моменты")
        self.view_all_button.clicked.connect(self._view_all_candidates)
        self.production_button = QPushButton("Создать готовые ролики")
        self.production_button.setObjectName("primary")
        self.production_button.setProperty("responsiveFullText", "Создать готовые ролики")
        self.production_button.setProperty("responsiveCompactText", "Создать ролики")
        # QPushButton.clicked carries a ``bool`` checked argument.  Keep it
        # out of the candidate-id API: the delivery set is always derived
        # from the durable approved-draft state below.
        self.production_button.clicked.connect(lambda _checked=False: self._confirm_production_render())
        left.addWidget(self.candidate_review)
        self.final_results = FinalResultsWorkspace()
        self.final_results.output_selected.connect(self._final_output_selected)
        self.final_results.create_more_requested.connect(self._create_more_outputs)
        self.final_results.drafts_requested.connect(self._back_to_drafts)
        self.final_results.rerender_requested.connect(self._rerender_final_output)
        self.final_results.projects_requested.connect(self.back_requested)
        left.addWidget(self.final_results)
        self.progress = ProcessingProgress()
        self.progress.cancel_requested.connect(self.viewmodel.cancel)
        self.progress.continue_waiting_requested.connect(self.viewmodel.continue_waiting)
        self.progress.retry_requested.connect(self._retry_processing)
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
        heading = QLabel("Дополнительные настройки")
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
        self.content_profile_preset = self.setup_content_profile
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
        self.clip_count = self.setup_clip_count
        self.audio_mode = QComboBox()
        self.audio_mode.addItem("Исходная речь", "original")
        self.audio_mode.addItem("Исходная речь, улучшить звук", "original_enhanced")
        self.audio_mode.addItem("Озвучка", "voiceover")
        self.audio_mode.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(audio_mode=str(self.audio_mode.currentData()))
        )
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
        self.same_source_broll = QCheckBox(self._SAME_SOURCE_BROLL_TEXT)
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
        self.reduced_motion = QCheckBox("Уменьшить движение текста и акцентов")
        self.reduced_motion.setToolTip(
            "Использует безопасные static/fade варианты текущего production caption preset."
        )
        self.reduced_motion.toggled.connect(
            lambda value: self.viewmodel.save_options(reduced_motion=value)
        )
        settings.addWidget(self.reduced_motion)
        self.subtitle_style = self.setup_creative_style
        self.caption_preset = self.setup_caption_preset
        self.cache = QCheckBox(self._CACHE_TEXT)
        self.cache.setToolTip(self._CACHE_TEXT)
        self.cache.toggled.connect(lambda value: self.viewmodel.save_options(use_cache=value))
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
        self.viewmodel.project_changed.connect(self._queue_project_changed)
        self.viewmodel.runs_changed.connect(self._queue_runs_changed)
        self.viewmodel.processing_changed.connect(self._processing_changed)
        self.viewmodel.error_occurred.connect(self._error)
        self._compact_stage_layout: bool | None = None
        self._compact_action_layout: bool | None = None
        self._short_stage_layout: bool | None = None
        self._apply_stage_responsive_layout(force=True)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "_body_layout"):
            self._apply_stage_responsive_layout()
        if hasattr(self, "candidate_detail"):
            QTimer.singleShot(0, self._refresh_candidate_detail_geometry)
        self._queue_drafts_workspace_geometry()

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
        compact = self.width() < self._WIDE_STAGE_MINIMUM_WIDTH
        compact_actions = self.width() < self._COMPACT_ACTION_BREAKPOINT
        window_height = self.window().height() if self.window() else 0
        short_stage = 0 < window_height <= 840
        cards_need_reflow = (
            self._compact_action_layout is not None
            and compact_actions != self._compact_action_layout
        )
        if (
            not force
            and compact == self._compact_stage_layout
            and compact_actions == self._compact_action_layout
            and short_stage == self._short_stage_layout
        ):
            # Height-for-width still changes inside one responsive profile.
            # In particular, the full review hint can need several rows just
            # above 1,100 px, then collapse again as the same medium layout is
            # widened.  Returning without releasing that cached minimum kept
            # the sticky bar tall until the next breakpoint was crossed.
            show_complete_stepper = self.width() >= self._COMPLETE_STEPPER_MINIMUM_WIDTH
            for divider in self._global_step_dividers:
                divider.setVisible(not compact_actions or show_complete_stepper)
            self._apply_compact_chrome()
            self._reflow_candidate_boundary_controls()
            self._refresh_stage_action_geometry()
            QTimer.singleShot(0, self._refresh_stage_action_geometry)
            self._queue_drafts_workspace_geometry()
            return
        self._compact_stage_layout = compact
        self._compact_action_layout = compact_actions
        self._short_stage_layout = short_stage
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
            14 if compact else 24,
            10 if compact else 18,
            14 if compact else 24,
            14 if compact else 24,
        )
        self._body_layout.setDirection(direction)
        self._body_layout.setSpacing(spacing)
        self._setup_workspace_layout.setDirection(direction)
        self._setup_workspace_layout.setSpacing(spacing)
        self._processing_workspace_layout.setDirection(direction)
        self._processing_workspace_layout.setSpacing(spacing)
        self._review_body_layout.setDirection(direction)
        self._review_body_layout.setSpacing(12 if compact else 14)
        # Compact chrome stays one row.  Stacking the header and all four
        # workflow steps consumed most of a 720×420 shell before the stage
        # body and its CTA were laid out.
        self._header_layout.setDirection(QBoxLayout.Direction.LeftToRight)
        self._header_layout.setSpacing(6 if compact_actions else 8)
        self._stepper_layout.setDirection(QBoxLayout.Direction.LeftToRight)
        self._stepper_layout.setSpacing(0)
        show_complete_stepper = self.width() >= self._COMPLETE_STEPPER_MINIMUM_WIDTH
        for divider in self._global_step_dividers:
            divider.setVisible(not compact_actions or show_complete_stepper)
        self._setup_action_layout.setDirection(QBoxLayout.Direction.LeftToRight)
        self._processing_actions_layout.setDirection(QBoxLayout.Direction.LeftToRight)
        self._review_header_layout.setDirection(action_direction)
        self._review_action_layout.setDirection(QBoxLayout.Direction.LeftToRight)
        self._review_action_layout.setSpacing(8 if compact_actions else 6)
        self.content_host.setMinimumWidth(0)
        if compact:
            for panel in (self.review_list_panel, self.review_preview_panel, self.review_inspector_panel):
                panel.setMinimumWidth(0)
                panel.setMaximumWidth(16_777_215)
        else:
            dense_columns = self.width() < 1320
            self.review_list_panel.setMinimumWidth(290 if dense_columns else 330)
            self.review_list_panel.setMaximumWidth(340 if dense_columns else 400)
            self.review_preview_panel.setMinimumWidth(340 if dense_columns else 410)
            self.review_preview_panel.setMaximumWidth(16_777_215)
            self.review_inspector_panel.setMinimumWidth(250 if dense_columns else 280)
            self.review_inspector_panel.setMaximumWidth(310 if dense_columns else 350)
        self.settings_panel.setMaximumWidth(16_777_215 if compact else 300)
        self.setup_summary.setMinimumWidth(0)
        self.setup_summary.setMaximumWidth(16_777_215)
        if not compact:
            self.setup_summary.setMinimumWidth(250)
            self.setup_summary.setMaximumWidth(330)
        self.processing_summary.setMaximumWidth(16_777_215)
        # Give Creative Preview visual priority at ordinary desktop widths,
        # while keeping the complete player and its controls above the sticky
        # CTA on short Windows viewports.  Tall desktop windows retain the
        # larger approved-reference stage.
        if short_stage:
            self.preview.set_source_frame_height_bounds(200, 310)
            self.preview.set_vertical_frame_size(188, 334)
        else:
            self.preview.set_source_frame_height_bounds(260, 420)
            self.preview.set_vertical_frame_size(
                252 if compact else 304,
                448 if compact else 540,
            )
        self._apply_compact_chrome()
        self._reflow_candidate_boundary_controls()
        self._review_body_layout.invalidate()
        self._review_body_layout.activate()
        self._refresh_stage_action_geometry()
        self.updateGeometry()
        QTimer.singleShot(0, self._refresh_stage_action_geometry)
        self._queue_drafts_workspace_geometry()
        # Candidate cards own an action rail.  Rebuild them only when the
        # breakpoint changes so that the rail can become a full-width block
        # instead of being horizontally squeezed or clipped.
        if cards_need_reflow and self.project:
            self._update_candidate_review(self.project)
            self._apply_compact_chrome()
            self._refresh_stage_action_geometry()

    def _apply_compact_chrome(self, global_step: str | None = None) -> None:
        """Keep the current route and actions readable without tall chrome."""

        compact = bool(self._compact_action_layout)
        ultra_compact = self.width() < self._COMPLETE_STEPPER_MINIMUM_WIDTH
        current_step = global_step or _GLOBAL_STEP_FOR_FLOW.get(self._flow_step, "settings")
        self.back_button.setText("←" if compact else "← Проекты")
        self.folder.setText("Папка" if compact else "Открыть папку")
        self.settings_toggle.setText("Ещё" if compact else "Дополнительно")
        self.setup_back_button.setText("←" if compact else "← К источнику")
        self.review_back_button.setText("←" if compact else "← Назад к обработке")
        self.setup_back_button.setToolTip("Вернуться к источнику")
        self.review_back_button.setToolTip("Вернуться к обработке")
        for button in (self.view_all_button, self.draft_button, self.production_button):
            full_text = str(button.property("responsiveFullText") or button.text())
            compact_text = str(button.property("responsiveCompactText") or full_text)
            button.setText(compact_text if compact else full_text)
            button.setToolTip(full_text)
        self.status.setVisible(not compact)
        for index, (name, full_label) in enumerate(_GLOBAL_FLOW_STEPS, start=1):
            label = self._global_step_labels[name]
            label.setText(
                _GLOBAL_FLOW_COMPACT_LABELS[name]
                if compact and not ultra_compact
                else f"{index}  {full_label}"
            )
            label.setVisible(not ultra_compact or name == current_step)
        self.same_source_broll.setText(
            self._SAME_SOURCE_BROLL_COMPACT_TEXT if compact else self._SAME_SOURCE_BROLL_TEXT
        )
        self.cache.setText(self._CACHE_COMPACT_TEXT if compact else self._CACHE_TEXT)
        # The same next-step explanation already lives in the flow card.  In
        # the compact sticky bar it stole a row from the body and could paint
        # through the vertically stacked CTA buttons.
        self.workflow_hint.setVisible(not compact and not self.review_action_bar.isHidden())

    def _refresh_stage_action_geometry(self) -> None:
        """Reserve the visible sticky action bar's real post-reflow height."""

        if not hasattr(self, "stage_actions"):
            return
        root_margins = self._root_layout.contentsMargins()
        available_width = max(
            1,
            self.width() - root_margins.left() - root_margins.right(),
        )
        for bar in (self.setup_action_bar, self.review_action_bar, self.final_results.action_bar):
            bar.setMinimumHeight(0)
            bar_layout = bar.layout()
            if bar_layout is None or bar.isHidden():
                continue
            bar_layout.invalidate()
            bar_layout.activate()
            required = bar_layout.totalHeightForWidth(available_width)
            if required < 0:
                required = bar_layout.totalSizeHint().height()
            bar.setMinimumHeight(max(required, bar_layout.totalMinimumSize().height()))
        actions_layout = self.stage_actions.layout()
        if actions_layout is None:
            return
        self.stage_actions.setMinimumHeight(0)
        actions_layout.invalidate()
        actions_layout.activate()
        required = actions_layout.totalHeightForWidth(available_width)
        if required < 0:
            required = actions_layout.totalSizeHint().height()
        if not self.stage_actions.isHidden():
            self.stage_actions.setMinimumHeight(max(required, actions_layout.totalMinimumSize().height()))
        self.stage_actions.updateGeometry()

    def _queue_drafts_workspace_geometry(self, *_: object) -> None:
        """Coalesce Drafts geometry changes after Qt finishes one layout pass."""

        if self._drafts_geometry_refresh_pending:
            return
        self._drafts_geometry_refresh_pending = True
        QTimer.singleShot(0, self._refresh_drafts_workspace_geometry)

    def _refresh_drafts_workspace_geometry(self) -> None:
        """Give stacked review workspaces one scroll owner and preserve content.

        The review body's 2:3:2 stretch factors can give the list and inspector
        tiny nested viewports and make a fixed phone preview taller than its
        panel.  Drafts always use the existing outer project scroll.  Moments
        keep independent catalogue/inspector panes in the wide three-column
        workspace, but switch to that same single owner when the panels stack.
        """

        self._drafts_geometry_refresh_pending = False
        if not hasattr(self, "review_workspace"):
            return

        flow_step = self._derive_flow_step(self.project) if self.project is not None else ""
        is_drafts = flow_step == "drafts"
        is_stacked_moments = flow_step == "candidates" and bool(self._compact_stage_layout)
        preview_height = self._draft_preview_natural_height()
        single_scroll = is_drafts or is_stacked_moments
        mode_changed = single_scroll != self._drafts_single_scroll_layout
        self._set_drafts_single_scroll_layout(single_scroll)
        previous_panel_constraints = tuple(
            (panel.minimumHeight(), panel.maximumHeight())
            for panel in (
                self.review_list_panel,
                self.review_preview_panel,
                self.review_inspector_panel,
            )
        )
        cards_changed = False

        if single_scroll:
            cards_changed = self._refresh_draft_candidate_card_heights()
            list_height = self._set_widget_layout_natural_height(
                self.candidate_review
            )
            inspector_height = self._set_widget_layout_natural_height(
                self.candidate_detail
            )
            # The wide Draft workspace deliberately equalises the three panel
            # shells to the dominant phone preview.  Keep the inspector itself
            # at its measured height and aligned to the top of that shell so
            # the first editable option never drifts downward.
            self.candidate_detail.setMaximumHeight(inspector_height)
            # Two pixels account for the themed panel frame.  Pin the current
            # natural extent in both directions: a Preferred child otherwise
            # retains its former narrow-column height after wide→narrow and
            # leaves a large blank gap above the boundary controls.
            panel_heights = (
                list_height + 2,
                preview_height + 2,
                inspector_height + 2,
            )
            if not self._compact_stage_layout:
                common_height = max(panel_heights)
                panel_heights = (common_height,) * 3
            for panel, panel_height in zip(
                (
                    self.review_list_panel,
                    self.review_preview_panel,
                    self.review_inspector_panel,
                ),
                panel_heights,
                strict=True,
            ):
                panel.setMinimumHeight(panel_height)
                panel.setMaximumHeight(panel_height)
            self.review_preview_panel.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
        else:
            self.candidate_detail.setMaximumHeight(16_777_215)
            for panel in (
                self.review_list_panel,
                self.review_preview_panel,
                self.review_inspector_panel,
            ):
                panel.setMinimumHeight(0)
                panel.setMaximumHeight(16_777_215)

        self._review_list_panel_layout.invalidate()
        self._review_preview_panel_layout.invalidate()
        self._review_inspector_panel_layout.invalidate()
        self._review_body_layout.invalidate()
        self._review_body_layout.activate()
        self.review_list_panel.updateGeometry()
        self.review_preview_panel.updateGeometry()
        self.review_inspector_panel.updateGeometry()
        self.review_workspace.updateGeometry()
        self.content_host.updateGeometry()
        current_panel_constraints = tuple(
            (panel.minimumHeight(), panel.maximumHeight())
            for panel in (
                self.review_list_panel,
                self.review_preview_panel,
                self.review_inspector_panel,
            )
        )
        if (
            mode_changed
            or cards_changed
            or current_panel_constraints != previous_panel_constraints
        ):
            # Reparenting and new cards receive their final widths in later
            # event turns. Repeat only while a natural extent is still
            # changing, then stop once the geometry is stable.
            self._queue_drafts_workspace_geometry()

    def _set_drafts_single_scroll_layout(self, enabled: bool) -> None:
        if enabled == self._drafts_single_scroll_layout:
            return
        pairs = (
            (
                self.review_list_panel,
                self._review_list_panel_layout,
                self.review_list_scroll,
                self.candidate_review,
            ),
            (
                self.review_inspector_panel,
                self._review_inspector_panel_layout,
                self.review_inspector_scroll,
                self.candidate_detail,
            ),
        )
        for panel, panel_layout, scroll, content in pairs:
            if enabled:
                if scroll.widget() is content:
                    scroll.takeWidget()
                scroll.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                )
                scroll.verticalScrollBar().setRange(0, 0)
                scroll.hide()
                content.setParent(panel)
                panel_layout.addWidget(
                    content, 0, Qt.AlignmentFlag.AlignTop,
                )
                # Once the inner viewport no longer owns width negotiation,
                # ignore its historic wide-column size hint.  Wrapped labels
                # and button rails then recompute height-for-width against the
                # stacked panel instead of widening the outer workspace.
                content.setMinimumWidth(0)
                content.setSizePolicy(
                    QSizePolicy.Policy.Ignored,
                    QSizePolicy.Policy.Preferred,
                )
                content.show()
                panel.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
            else:
                if scroll.widget() is not content:
                    panel_layout.removeWidget(content)
                    content.setParent(None)
                    scroll.setWidget(content)
                content.setSizePolicy(
                    QSizePolicy.Policy.Ignored,
                    QSizePolicy.Policy.Preferred,
                )
                scroll.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded
                )
                scroll.show()
                panel.setSizePolicy(
                    QSizePolicy.Policy.Preferred,
                    QSizePolicy.Policy.Preferred,
                )
        self._drafts_single_scroll_layout = enabled

    @staticmethod
    def _set_widget_layout_natural_height(widget: QWidget) -> int:
        layout = widget.layout()
        if layout is None:
            return max(0, widget.minimumHeight())
        widget.setMinimumHeight(0)
        layout.invalidate()
        layout.activate()
        required_height = layout.totalHeightForWidth(max(1, widget.width()))
        if required_height < 0:
            required_height = layout.totalSizeHint().height()
        natural_height = max(required_height, layout.totalMinimumSize().height())
        widget.setMinimumHeight(natural_height)
        widget.updateGeometry()
        return natural_height

    def _refresh_draft_candidate_card_heights(self) -> bool:
        changed = False
        for card in self._candidate_cards.values():
            previous_height = card.minimumHeight()
            natural_height = self._set_widget_layout_natural_height(card)
            changed = changed or natural_height != previous_height
        return changed

    def _draft_preview_natural_height(self) -> int:
        preview_layout = self.preview.layout()
        required_height = self.preview.sizeHint().height()
        if preview_layout is not None:
            layout_height = preview_layout.totalHeightForWidth(
                max(1, self.preview.width())
            )
            if layout_height < 0:
                layout_height = preview_layout.totalSizeHint().height()
            required_height = max(required_height, layout_height)
        return max(1, required_height)

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
        setup_heading = QLabel("Настройка обработки")
        setup_heading.setObjectName("screenTitle")
        setup_copy = QLabel("Выберите, что искать и как должны выглядеть будущие ролики.")
        setup_copy.setObjectName("subtitle")
        setup_copy.setWordWrap(True)
        setup_main_layout.addWidget(setup_heading)
        setup_main_layout.addWidget(setup_copy)
        setup_copy.hide()
        self.setup_source_summary = QFrame()
        self.setup_source_summary.setObjectName("setupSourceSummary")
        setup_source_layout = QHBoxLayout(self.setup_source_summary)
        setup_source_layout.setContentsMargins(12, 10, 12, 10)
        setup_source_layout.setSpacing(12)
        self.setup_source_summary_text = QLabel()
        self.setup_source_summary_text.setObjectName("muted")
        make_label_shrinkable(self.setup_source_summary_text)
        self.setup_source_poster = QLabel("Готовим кадр…")
        self.setup_source_poster.setObjectName("projectPoster")
        self.setup_source_poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setup_source_poster.setFixedSize(154, 76)
        self.setup_source_poster.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed,
        )
        self._project_thumbnail_labels.append(self.setup_source_poster)
        source_copy_layout = QVBoxLayout()
        source_copy_layout.setContentsMargins(0, 0, 0, 0)
        source_copy_layout.setSpacing(4)
        setup_source_heading = QLabel("Ваше видео")
        setup_source_heading.setObjectName("setupSectionTitle")
        source_copy_layout.addWidget(setup_source_heading)
        source_copy_layout.addWidget(self.setup_source_summary_text)
        setup_source_layout.addWidget(self.setup_source_poster)
        setup_source_layout.addLayout(source_copy_layout, 1)
        self.recommendation_banner = QFrame()
        self.recommendation_banner.setObjectName("recommendationBanner")
        recommendation_layout = QHBoxLayout(self.recommendation_banner)
        recommendation_layout.setContentsMargins(16, 14, 16, 14)
        recommendation_icon = QLabel("✦")
        recommendation_icon.setObjectName("recommendationIcon")
        self.recommendation_text = QLabel()
        make_label_shrinkable(self.recommendation_text)
        recommendation_layout.addWidget(recommendation_icon)
        recommendation_layout.addWidget(self.recommendation_text, 1)
        setup_main_layout.addWidget(self.setup_source_summary)
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

        self.setup_summary = self._card("Демо оформления")
        self.setup_summary.setObjectName("setupSummary")
        self.setup_summary_text = QLabel()
        self.setup_summary_text.setObjectName("muted")
        make_label_shrinkable(self.setup_summary_text)
        self.setup_demo_preview = VideoPreview()
        self.setup_demo_preview.setObjectName("settingsProductionPreview")
        self.setup_demo_preview.set_vertical_frame_size(270, 480)
        self.setup_demo_preview.set_frame_sink_output(True)
        self.setup_demo_preview.controls_host.hide()
        self.setup_demo_preview.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.setup_demo_preview.player.mediaStatusChanged.connect(
            self._settings_demo_media_status_changed
        )
        self.setup_summary.layout().addWidget(
            self.setup_demo_preview, 0, Qt.AlignmentFlag.AlignHCenter,
        )
        self.setup_demo_detail = QLabel()
        self.setup_demo_detail.setObjectName("setupDemoDetail")
        self.setup_demo_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setup_demo_detail.setWordWrap(True)
        self.setup_summary.layout().addWidget(self.setup_demo_detail)
        example_note = QLabel("Канонический production sample · без обработки вашего видео")
        example_note.setObjectName("muted")
        example_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        example_note.setWordWrap(True)
        self.setup_summary.layout().addWidget(example_note)
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
        self._processing_main = processing_main
        processing_main_layout = QVBoxLayout(processing_main)
        self._processing_main_layout = processing_main_layout
        processing_main_layout.setContentsMargins(0, 0, 0, 0)
        processing_main_layout.setSpacing(14)
        processing_heading = QLabel("Обработка видео")
        processing_heading.setObjectName("screenTitle")
        processing_copy = QLabel("Готовые этапы и артефакты сохраняются по мере выполнения.")
        processing_copy.setObjectName("subtitle")
        processing_copy.setWordWrap(True)
        processing_main_layout.addWidget(processing_heading)
        processing_main_layout.addWidget(processing_copy)
        self.processing_source_summary = self._card("Обрабатываемое видео")
        self.processing_source_text = QLabel()
        self.processing_source_text.setObjectName("muted")
        make_label_shrinkable(self.processing_source_text)
        self.processing_source_poster = QLabel("Готовим кадр…")
        self.processing_source_poster.setObjectName("projectPoster")
        self.processing_source_poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processing_source_poster.setFixedHeight(112)
        self.processing_source_poster.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed,
        )
        self._project_thumbnail_labels.append(self.processing_source_poster)
        self.processing_source_summary.layout().addWidget(self.processing_source_poster)
        self.processing_source_summary.layout().addWidget(self.processing_source_text)
        processing_main_layout.addWidget(self.progress)
        self._processing_progress_layout_height = self.progress.minimumHeight()
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
            ("transcribe", "Понимаем речь и структуру"),
            ("understand", "Учитываем содержание и события"),
            ("candidates", "Находим и оцениваем моменты"),
            ("save", "Сохраняем результаты"),
        ):
            row = QLabel(f"○  {label}")
            row.setObjectName("processingStage")
            row.setProperty("stageState", "pending")
            stages_layout.addWidget(row)
            self.processing_stage_labels[stage] = row
        self._processing_stage_rows: tuple[str, ...] = (
            "prepare", "transcribe", "understand", "candidates", "save",
        )
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
        self.processing_summary.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.processing_summary.layout().addWidget(self.processing_source_summary)
        self.processing_summary_text = QLabel()
        self.processing_summary_text.setObjectName("muted")
        make_label_shrinkable(self.processing_summary_text)
        self.processing_summary.layout().addWidget(self.processing_summary_text)
        self.processing_next = QLabel("Результаты сохраняются автоматически. После остановки останутся только готовые артефакты.")
        self.processing_next.setObjectName("muted")
        make_label_shrinkable(self.processing_next)
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
        make_label_shrinkable(self.results_subflow_hint)
        review_header.addWidget(self.results_subflow)
        review_header.addWidget(self.results_subflow_hint, 1)
        review_layout.addLayout(review_header)
        self.review_metrics = QFrame()
        self.review_metrics.setObjectName("resultsMetrics")
        metrics_layout = QHBoxLayout(self.review_metrics)
        metrics_layout.setContentsMargins(14, 10, 14, 10)
        self.review_metrics_text = QLabel()
        make_label_shrinkable(self.review_metrics_text)
        metrics_layout.addWidget(self.review_metrics_text)
        review_layout.addWidget(self.review_metrics)
        review_body = QHBoxLayout()
        self._review_body_layout = review_body
        review_body.setSpacing(14)
        list_panel = QFrame()
        self.review_list_panel = list_panel
        list_panel.setObjectName("reviewListPanel")
        list_panel_layout = QVBoxLayout(list_panel)
        self._review_list_panel_layout = list_panel_layout
        list_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.review_list_scroll = QScrollArea()
        self.review_list_scroll.setWidgetResizable(True)
        self.review_list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.review_list_scroll.setWidget(self.candidate_review)
        list_panel_layout.addWidget(self.review_list_scroll)
        review_body.addWidget(list_panel, 3)
        preview_panel = QFrame()
        self.review_preview_panel = preview_panel
        preview_panel.setObjectName("reviewPreviewPanel")
        preview_layout = QVBoxLayout(preview_panel)
        self._review_preview_panel_layout = preview_layout
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        review_body.addWidget(preview_panel, 5)
        inspector_panel = QFrame()
        self.review_inspector_panel = inspector_panel
        inspector_panel.setObjectName("reviewInspectorPanel")
        inspector_layout = QVBoxLayout(inspector_panel)
        self._review_inspector_panel_layout = inspector_layout
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        self.review_inspector_scroll = QScrollArea()
        self.review_inspector_scroll.setWidgetResizable(True)
        self.review_inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.review_inspector_scroll.setWidget(self.candidate_detail)
        inspector_layout.addWidget(self.review_inspector_scroll)
        review_body.addWidget(inspector_panel, 3)
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
        self.stage_actions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setup_action_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.review_action_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.final_results.action_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        actions_layout = QVBoxLayout(self.stage_actions)
        actions_layout.setContentsMargins(0, 10, 0, 0)
        actions_layout.setSpacing(0)
        # Adding a widget to its new layout moves it from the stage body; no
        # button is duplicated and therefore no second primary CTA appears.
        actions_layout.addWidget(self.setup_action_bar)
        actions_layout.addWidget(self.review_action_bar)
        actions_layout.addWidget(self.final_results.action_bar)
        self.setup_action_bar.hide()
        self.review_action_bar.hide()
        self.final_results.action_bar.hide()
        self.stage_actions.hide()
        root.addWidget(self.stage_actions)

    def open(self, project: DesktopProject) -> None:
        # Explicit navigation back to a project starts from its reconciled
        # persisted state; the temporary "create more" route is session-only.
        self._results_subflow_override = None
        # A deliberate project-open transaction verifies Analysis once.  Card
        # batches and selection-only refreshes reuse the verified projection
        # until the persisted artifact identity or file stat changes.
        self._analysis_cache_key = None
        self._analysis_cache = {}
        self._analysis_reference_stats = ()
        self._analysis_load_error = None
        self.viewmodel.open(project)
        # Navigation is a user-visible state transition and remains
        # synchronous.  The two persisted signals emitted by ``open`` are
        # still rendered by one coalesced flush instead of two rebuilds.
        if self._persisted_refresh_pending:
            self._flush_persisted_refresh()

    def _queue_project_changed(self, project: DesktopProject) -> None:
        if self._persisting_active_preview and self.project and self.project.project_id == project.project_id:
            self.project = project
            return
        if self._persisting_review_selection and self.project and self.project.project_id == project.project_id:
            self.project = project
            self._refresh_moment_selection_ui()
            return
        self._pending_project = project
        self._queue_persisted_refresh()

    def _queue_runs_changed(self, runs: list[ProjectRun]) -> None:
        self._pending_runs = list(runs)
        self._queue_persisted_refresh()

    def _queue_persisted_refresh(self) -> None:
        if self._persisted_refresh_pending:
            return
        self._persisted_refresh_pending = True
        QTimer.singleShot(0, self._flush_persisted_refresh)

    def _flush_persisted_refresh(self) -> None:
        self._persisted_refresh_pending = False
        project = self._pending_project
        runs = self._pending_runs
        self._pending_project = None
        self._pending_runs = None
        if runs is not None:
            self.runs = runs
        if project is not None:
            self._project_changed(project)
        if runs is not None:
            self._runs_changed(runs, refresh_project=project is None)
        if project is not None:
            # ``open()`` emits processing after project/runs in the same call
            # stack.  Reconcile once after the coalesced persisted refresh so
            # terminal recovery controls see the newly opened project.
            self._processing_changed(self.viewmodel.snapshot)

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
            self._processing_structure_key = None
        self.title.setText(project.name.replace("_", " "))
        self.title.setToolTip(project.name)
        presentation = self.viewmodel.services.presentation(
            project,
            snapshot=self.viewmodel.snapshot,
            runs=self._runs_for_project(project),
        )
        self.status.setText(presentation.status_label)
        self.run_button.setText("Начать поиск моментов")
        preview_step = self._derive_flow_step(project)
        if (
            project.source_spec.is_ready
            and preview_step in {"candidates", "drafts"}
            and (is_new_project or self.preview.active_media_path is None)
        ):
            self.preview.show_source(
                str(project.source),
                source_codec=str(project.source_metadata.get("video_codec") or ""),
                poster_cache_directory=project.directory / "preview-posters",
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
        self.setup_editorial_intent.blockSignals(True)
        self.setup_editorial_intent.setText(project.settings.editorial_intent)
        self.setup_editorial_intent.blockSignals(False)
        self._set_combo_data(self.content_profile_preset, project.settings.content_profile_preset)
        self._set_combo_data(self.audio_mode, project.settings.audio_mode)
        self._set_combo_data(self.composition_strategy, project.settings.composition_strategy)
        self.same_source_broll.blockSignals(True)
        self.same_source_broll.setChecked(project.settings.same_source_broll_allowed)
        self.same_source_broll.blockSignals(False)
        self.subtitles.blockSignals(True); self.subtitles.setChecked(project.settings.subtitles_enabled); self.subtitles.blockSignals(False)
        self._set_combo_data(self.subtitle_style, project.settings.subtitle_style)
        self._set_combo_data(self.caption_preset, project.settings.caption_preset_id)
        self._sync_setup_choice_buttons(project)
        self.reduced_motion.blockSignals(True)
        self.reduced_motion.setChecked(project.settings.reduced_motion)
        self.reduced_motion.blockSignals(False)
        self.cache.blockSignals(True); self.cache.setChecked(project.settings.use_cache); self.cache.blockSignals(False)
        self._update_download_card(project)
        self._update_setup_card(project)
        self._update_stage_context(project)
        self._ensure_project_thumbnail(project)
        if preview_step in {"candidates", "drafts"}:
            self._preload_moments_proxy(project)
        self._update_candidate_review(project)
        self._update_final_results(project)
        self._update_next_step(project)
        self._reconcile_active_candidate_preview(project, previous_step=previous_step)
        self._apply_flow_visibility(project, presentation=presentation)
        if previous_step != self._flow_step:
            # Each route is a screen, not a continuation of the previous
            # page's scroll position.  Reset after Qt has shown the newly
            # selected workspace so its reference composition starts at the
            # global stepper while sticky actions remain reachable below.
            QTimer.singleShot(0, lambda: self.content_scroll.verticalScrollBar().setValue(0))

    def _preload_moments_proxy(self, project: DesktopProject) -> None:
        """Begin the existing compatible source proxy before a Moment is clicked.

        The proxy is a local review cache only.  It shares the source revision
        identity and FFmpeg lifecycle with ``VideoPreview.set_range`` and never
        changes analysis, candidate selection, boundaries or render artifacts.
        """

        if not project.source_spec.is_ready:
            return
        self.preview.preload_compatible_proxy(
            project.source,
            cache_directory=project.directory / "preview-proxies",
            source_codec=str(project.source_metadata.get("video_codec") or ""),
        )

    def _set_advanced_visible(self, visible: bool) -> None:
        if self._flow_step != "settings":
            visible = False
        self.settings_panel.setVisible(visible)
        for toggle in (self.settings_toggle, self.setup_advanced_toggle):
            toggle.blockSignals(True)
            toggle.setChecked(visible)
            toggle.blockSignals(False)

    @staticmethod
    def _setup_choice_section(title: str, description: str) -> tuple[QFrame, QVBoxLayout]:
        section = QFrame()
        section.setObjectName("setupChoiceSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        heading = QLabel(title)
        heading.setObjectName("setupSectionTitle")
        layout.addWidget(heading)
        copy = QLabel(description)
        copy.setObjectName("muted")
        copy.setWordWrap(True)
        layout.addWidget(copy)
        section.setToolTip(description)
        copy.hide()
        return section, layout

    def _add_setup_choice_buttons(
        self,
        parent_layout: QVBoxLayout | QGridLayout,
        kind: str,
        choices: list[tuple[str, object]],
        owner: QComboBox,
        *,
        columns: int,
        reuse_group: bool = False,
    ) -> dict[object, QPushButton]:
        """Expose an existing combo owner as an approved visual choice grid."""

        group = next(
            (item for item in self._setup_choice_groups if item.property("choiceKind") == kind),
            None,
        ) if reuse_group else None
        if group is None:
            group = QButtonGroup(self)
            group.setExclusive(True)
            group.setProperty("choiceKind", kind)
            self._setup_choice_groups.append(group)
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(7)
        buttons: dict[object, QPushButton] = {}
        for index, (label, value) in enumerate(choices):
            button = QPushButton(label)
            button.setObjectName("setupChoice")
            button.setProperty("choiceKind", kind)
            button.setProperty("choiceValue", str(value))
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            group.addButton(button)
            button.clicked.connect(
                lambda checked=False, combo=owner, choice=value: self._choose_setup_value(combo, choice)
                if checked else None
            )
            grid.addWidget(button, index // columns, index % columns)
            buttons[value] = button
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        if isinstance(parent_layout, QGridLayout):
            parent_layout.addLayout(
                grid, parent_layout.rowCount(), 0, 1, max(1, parent_layout.columnCount()),
            )
        else:
            parent_layout.addLayout(grid)
        return buttons

    @staticmethod
    def _choose_setup_value(owner: QComboBox, value: object) -> None:
        index = owner.findData(value)
        if index >= 0 and index != owner.currentIndex():
            owner.setCurrentIndex(index)

    def _set_all_profiles_visible(self, visible: bool) -> None:
        self.setup_profile_more.setVisible(visible)
        self.setup_profile_more_toggle.setText(
            "Скрыть дополнительные" if visible else "Ещё 11 профилей"
        )

    def _sync_setup_choice_buttons(self, project: DesktopProject) -> None:
        values = {
            "profile": project.settings.content_profile_preset,
            "style": project.settings.subtitle_style,
            "count": str(project.settings.clip_count),
        }
        for kind, selected_value in values.items():
            for value, button in self._setup_choice_buttons.get(kind, {}).items():
                selected = value == selected_value
                button.blockSignals(True)
                button.setChecked(selected)
                button.setProperty("selected", selected)
                button.style().unpolish(button)
                button.style().polish(button)
                button.blockSignals(False)
        self.setup_caption_picker.set_selected(project.settings.caption_preset_id)
        hidden_profiles = list(CONTENT_PROFILE_PRESETS)[4:]
        if project.settings.content_profile_preset in hidden_profiles:
            self.setup_profile_more_toggle.setChecked(True)

    def _update_caption_style_demo(self, preset_id: str, style_id: str) -> None:
        preset = CAPTION_PRESET_DEFINITIONS.get(cast(Any, preset_id))
        if preset is None:
            return
        identity = (style_id, preset.preset_id)
        self._settings_demo_identity = identity
        if not self.setup_workspace.isVisible():
            return
        path = settings_preview_path(style_id, preset.preset_id)
        if self.setup_demo_preview.active_media_path == path:
            # The source can finish loading while Settings is hidden during
            # route projection. Start that exact media when Settings becomes
            # visible; Qt will not emit a second LoadedMedia notification.
            self.setup_demo_preview._play()
            return
        style_label = dict((value, label) for label, value in _CREATIVE_STYLE_CHOICES).get(
            style_id, style_id,
        )
        self.setup_demo_detail.setText(f"{style_label} · {preset.label}")
        if path is None:
            self.setup_demo_preview.set_file(
                None, presentation="vertical",
                title=f"{style_label} · {preset.label}: sample недоступен",
            )
            return
        self.setup_demo_preview.set_file(
            path,
            presentation="vertical",
            title=f"{style_label} · {preset.label}",
        )

    def _settings_demo_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if (
            status in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia}
            and self.setup_workspace.isVisible()
            and self._settings_demo_identity is not None
        ):
            self.setup_demo_preview._play()

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
        set_responsive_text(self.download_source, f"{details}\n{message}")
        self.download_button.setText(button)
        blocked = self.viewmodel.active and not self.viewmodel.owns_active_job
        self.download_button.setDisabled(state == "downloading" or self.viewmodel.active)
        self.download_button.setToolTip(
            self._other_project_job_hint() if blocked else ""
        )

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
        set_responsive_text(self.setup_source_summary_text, summary)
        set_responsive_text(self.processing_source_text, summary)

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
        profile_index = self.setup_content_profile.findData(project.settings.content_profile_preset)
        profile_label = self.setup_content_profile.itemText(profile_index) if profile_index >= 0 else "Авто"
        style_id = project.settings.subtitle_style
        style_label = dict((value, label) for label, value in _CREATIVE_STYLE_CHOICES).get(
            style_id, "Educational",
        )
        caption = CAPTION_PRESET_DEFINITIONS.get(project.settings.caption_preset_id)
        if caption is not None:
            self._update_caption_style_demo(caption.preset_id, style_id)
        set_responsive_text(
            self.setup_summary_text,
            f"Тип контента: {profile_label}\nСтиль оформления: {style_label}\n"
            f"Субтитры: {caption.label if caption else 'Editorial'}\n"
            f"Режим: {mode}\nЧто анализируем: {scope}\nФормат: {platform}\n"
            f"Рекомендуемых черновиков: {count}\n"
            f"Дополнительные кадры: {'разрешены' if project.settings.same_source_broll_allowed else 'не использовать'}"
        )
        recommendation = {
            "fast": "Рекомендуем быстрый режим для разговорного материала. Он поможет быстрее перейти к просмотру моментов.",
            "standard": "Рекомендуем стандартный режим: он сохраняет хороший баланс между глубиной поиска и временем ожидания.",
            "maximum": "Максимальный режим тщательно проверит контекст и события в кадре. Он займёт больше времени.",
        }.get(project.settings.processing_mode, "Настройки сохраняются в проекте автоматически.")
        set_responsive_text(self.recommendation_text, recommendation)
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
        set_responsive_text(
            self.processing_summary_text,
            f"{launch}\n{style_label} · {caption.label if caption else 'Editorial'} · {platform}\n"
            f"Режим: {mode} · цель: {count} роликов\n"
            "Результаты сохраняются в проекте по мере готовности."
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

    def _derive_flow_step(
        self,
        project: DesktopProject,
        *,
        presentation: ProjectPresentation | None = None,
    ) -> str:
        if presentation is None:
            presentation = self.viewmodel.services.presentation(
                project,
                snapshot=self.viewmodel.snapshot,
                runs=self._runs_for_project(project),
            )
        # ``create more`` and ``back to drafts`` are session-only navigation
        # affordances from an otherwise completed Results workspace.  They
        # must never win over a persisted Draft lifecycle transition: doing so
        # left a newly ready Preview on Moments and exposed its draft CTA.
        # The durable projection therefore remains canonical as soon as any
        # candidate has Draft work to review/retry/approve.
        if (
            presentation.flow_step == "finished"
            and self._results_subflow_override in {"candidates", "drafts"}
            and project.analysis_artifact_path
        ):
            return self._results_subflow_override
        return presentation.flow_step

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

    def _apply_flow_visibility(
        self,
        project: DesktopProject,
        *,
        presentation: ProjectPresentation | None = None,
    ) -> None:
        if presentation is None:
            presentation = self.viewmodel.services.presentation(
                project,
                snapshot=self.viewmodel.snapshot,
                runs=self._runs_for_project(project),
            )
        step = self._derive_flow_step(project, presentation=presentation)
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
        title = screen_titles[step]
        hint = self._flow_hint_for(step, project)
        latest = presentation.latest_run
        if step == "processing" and not presentation.active and latest and latest.status in {
            "failed", "interrupted", "cancelled", "partially_rendered",
        }:
            title = presentation.status_label
            hint = self._recovery_message(latest)
        self.flow_title.setText(title)
        self._set_flow_hint(hint)
        if self.viewmodel.blocked_by_other_project:
            self._set_flow_hint(self._other_project_job_hint())
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
        for name, stage_widget in self._stage_widgets.items():
            stage_widget.setVisible(
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
        if not show_review:
            self.preview.suspend()
        if not show_final:
            self.final_results.preview.suspend()
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
        self.final_results.action_bar.setVisible(show_final)
        self.stage_actions.setVisible(show_setup or show_review or show_final)
        self.results_subflow.setText({
            "candidates": "Найденные моменты",
            "drafts": "Черновики",
        }.get(step, "Результаты"))
        set_responsive_text(self.results_subflow_hint, {
            "candidates": "Просмотрите сильные фрагменты и выберите те, которые хотите превратить в черновики.",
            "drafts": "Проверьте созданные варианты и подтвердите только готовые к финальной сборке.",
        }.get(step, ""))
        self.settings_toggle.setVisible(step == "settings" and not active)
        self.autosave.setVisible(
            step == "settings" and not active and not bool(self._compact_action_layout)
        )
        if show_setup and self._settings_demo_identity is not None:
            style_id, preset_id = self._settings_demo_identity
            self._update_caption_style_demo(preset_id, style_id)
        elif not show_setup:
            self.setup_demo_preview.suspend()
        if step != "settings" or active:
            self._set_advanced_visible(False)
        self._apply_compact_chrome(global_step)
        self._refresh_stage_action_geometry()
        QTimer.singleShot(0, self._refresh_stage_action_geometry)
        self._queue_drafts_workspace_geometry()

    def _set_flow_hint(self, value: object) -> None:
        """Keep persisted diagnostics from turning global chrome into a page."""

        full_text = str(value)
        visible_text = full_text
        if len(visible_text) > self._FLOW_HINT_MAX_CHARS:
            visible_text = visible_text[: self._FLOW_HINT_MAX_CHARS - 1].rstrip() + "…"
        set_responsive_text(self.flow_hint, visible_text, tooltip=False)
        self.flow_hint.setToolTip(full_text)

    def _update_setup_card(self, project: DesktopProject) -> None:
        source = project.source_metadata
        duration = format_seconds(source.get("duration")) if source.get("duration") is not None else "пока неизвестна"
        size = self._format_file_size(source.get("size_bytes") or source.get("estimated_size_bytes"))
        source_name = project.source.name if project.source_spec.is_ready else str(source.get("title") or "Видео по ссылке")
        source_kind = "Файл" if project.source_spec.kind == "local_file" else "Видео по ссылке"
        source_state = "готов" if project.source_spec.is_ready else "ещё не загружен"
        set_responsive_text(
            self.setup_source,
            f"{source_kind}: {source_name}\nДлительность: {duration} · Размер: {size}\nИсточник {source_state}."
        )
        set_responsive_text(self.setup_mode_help, {
            "fast": "Быстрый вариант для интервью, лекций и других разговорных видео.",
            "standard": "Подходящий вариант по умолчанию: хороший баланс времени и качества.",
            "maximum": "Тщательнее учитывает контекст и события в кадре. Это займёт больше времени.",
        }[project.settings.processing_mode])
        if project.analysis_artifact_path:
            saved = project.setup_state.last_estimate
            set_responsive_text(self.setup_deep_help, "Используется сохранённый проверенный анализ.")
            set_responsive_text(self.setup_estimate, self._saved_estimate_text(saved))
        else:
            try:
                resolved, estimate = self.viewmodel.setup_preflight()
                deep_state = "будет использован" if resolved.deep_analysis.resolved else "не потребуется"
                set_responsive_text(
                    self.setup_deep_help,
                    f"{resolved.deep_analysis.reason} Дополнительный разбор {deep_state}.",
                )
                set_responsive_text(
                    self.setup_platform_help,
                    f"{resolved.platform.label}: ролик будет вертикальным, до {int(resolved.platform.maximum_duration_seconds)} секунд, "
                    "с субтитрами и полями для интерфейса."
                )
                set_responsive_text(self.setup_estimate, self._setup_estimate_text(estimate))
            except Exception:
                saved = project.setup_state.last_estimate
                set_responsive_text(self.setup_deep_help, "Рекомендация появится после проверки настроек.")
                set_responsive_text(self.setup_estimate, self._saved_estimate_text(saved))
        set_responsive_text(
            self.setup_change,
            project.setup_state.change_summary or "Настройки сохраняются в этом проекте."
        )
        preparing = not bool(project.analysis_artifact_path)
        heading = self.setup_card.layout().itemAt(0).widget()
        if isinstance(heading, QLabel):
            heading.setText("Основные настройки" if preparing else "Настройки следующего поиска")
            heading.hide()
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

    def _analysis_artifact(self, project: DesktopProject) -> dict[str, Any]:
        if not project.analysis_artifact_path:
            self._analysis_load_error = None
            return {}
        try:
            resolved = Path(project.analysis_artifact_path).resolve()
            stat = resolved.stat()
            key = (
                resolved, stat.st_size, stat.st_mtime_ns,
                project.analysis_id, project.analysis_fingerprint,
            )
            if (
                key == self._analysis_cache_key
                and ProjectScreen._analysis_reference_stat_signature(self._analysis_cache)
                == getattr(self, "_analysis_reference_stats", ())
            ):
                return self._analysis_cache
            artifact = self.viewmodel.services.pipeline.load_verified_analysis(project, required=True)
        except (OSError, ValueError) as error:
            self._analysis_load_error = str(error)
            self._analysis_cache_key = None
            self._analysis_cache = {}
            return {}
        assert artifact is not None
        self._analysis_load_error = None
        self._analysis_cache_key = key
        self._analysis_cache = artifact.to_dict()
        self._analysis_reference_stats = ProjectScreen._analysis_reference_stat_signature(
            self._analysis_cache
        )
        return self._analysis_cache

    @staticmethod
    def _analysis_reference_stat_signature(
        analysis: dict[str, Any],
    ) -> tuple[tuple[object, ...], ...]:
        """Cheaply detect reference changes between full integrity checks."""

        references = analysis.get("references") if isinstance(analysis, dict) else None
        if not isinstance(references, dict):
            return ()
        signature: list[tuple[object, ...]] = []
        for name, raw_path in sorted(references.items()):
            try:
                path = Path(str(raw_path)).resolve()
                stat = path.stat()
                signature.append((str(name), path, stat.st_size, stat.st_mtime_ns))
            except (OSError, ValueError):
                signature.append((str(name), str(raw_path), "missing"))
        return tuple(signature)

    @staticmethod
    def _draft_preview_projection(
        project: DesktopProject | None,
        candidate_id: str,
        *,
        has_preview: bool,
    ) -> _DraftPreviewProjection:
        """Project current/stale copy from one exact persisted candidate.

        A pending or running revision deliberately keeps its previous immutable
        Preview visible.  A failed targeted rebuild does the same after the
        service restores the candidate to ``ready``; that recovery boundary is
        recorded in the candidate-owned error message rather than the combined
        lifecycle state.  Both cases must therefore remain visibly stale.
        """

        if project is None:
            return _DraftPreviewProjection(
                stale=False,
                badge_text="●  Актуален" if has_preview else "",
                badge_state="ready" if has_preview else "warning",
                inspector_text=(
                    "Creative Preview готов" if has_preview else "Ожидает Creative Preview"
                ),
                inspector_state="finalReady" if has_preview else "warning",
            )
        draft_status = str(project.candidate_draft_statuses.get(candidate_id) or "")
        candidate_state = str(project.candidate_states.get(candidate_id) or "")
        candidate_error = str(project.candidate_errors.get(candidate_id) or "")
        preserved_previous = (
            has_preview
            and "предыдущая готовая версия сохранена" in candidate_error.casefold()
        )
        if preserved_previous:
            return _DraftPreviewProjection(
                stale=True,
                badge_text="●  Preview устарел",
                badge_state="warning",
                inspector_text="Обновление не удалось · показана предыдущая версия",
                inspector_state="warning",
            )
        if has_preview and draft_status in {"pending", "running"}:
            detail = (
                "Preview обновляется · показана предыдущая версия"
                if draft_status == "running"
                else "Изменения ожидают · показана предыдущая версия"
            )
            return _DraftPreviewProjection(
                stale=True,
                badge_text="●  Preview устарел",
                badge_state="warning",
                inspector_text=detail,
                inspector_state="warning",
            )
        if has_preview:
            return _DraftPreviewProjection(
                stale=False,
                badge_text="●  Актуален",
                badge_state="ready",
                inspector_text="Creative Preview готов",
                inspector_state="finalReady",
            )
        if candidate_state == "draft_failed" or draft_status == "failed":
            detail = "Нужен повтор черновика"
        elif candidate_state == "draft_planning" or draft_status == "running":
            detail = "Creative Preview обновляется"
        else:
            detail = "Ожидает Creative Preview"
        return _DraftPreviewProjection(
            stale=False,
            badge_text="",
            badge_state="warning",
            inspector_text=detail,
            inspector_state="warning",
        )

    def _update_candidate_review(self, project: DesktopProject) -> None:
        layout = self.candidate_review_layout
        self._thumbnail_loader.replace_pending()
        workflow_step = self._derive_flow_step(project)
        self._all_candidates_by_id = {}
        self._draftable_candidates_by_id = {}
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
            self._queue_drafts_workspace_geometry()
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
        analysis = self._analysis_artifact(project)
        raw_candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        profile = analysis.get("content_profile", {}) if isinstance(analysis, dict) else {}
        source = analysis.get("source", {}) if isinstance(analysis, dict) else {}
        all_candidates = []
        for raw in raw_candidates:
            if not isinstance(raw, dict) or not raw.get("candidate_id"):
                continue
            item = dict(raw)
            decision = evaluate_editorial_candidate(
                item,
                profile if isinstance(profile, dict) else {},
                score=float(item.get("score") or 0),
                confidence=float(item.get("confidence") or 0),
                recommended=bool(
                    item.get("selected_by_recommendation", item.get("recommended", False))
                ),
                production_feasibility=(
                    item.get("production_feasibility")
                    if isinstance(item.get("production_feasibility"), dict) else None
                ),
                source=source if isinstance(source, dict) else {},
            )
            production_decision = item.get("production_editorial_decision")
            if not isinstance(production_decision, dict):
                production_decision = evaluate_editorial_candidate(
                    item,
                    profile if isinstance(profile, dict) else {},
                    score=float(item.get("score") or 0),
                    confidence=float(item.get("confidence") or 0),
                    production_feasibility=(
                        item.get("production_feasibility")
                        if isinstance(item.get("production_feasibility"), dict) else None
                    ),
                    source=source if isinstance(source, dict) else {},
                ).to_dict()
            item["production_editorial_decision"] = production_decision
            item["editorial_decision"] = decision.to_dict()
            feasibility = item.get("production_feasibility")
            production_blocked = (
                isinstance(feasibility, dict)
                and feasibility.get("status") == "GUARANTEED_BLOCKED"
            )
            # The final visible state combines editorial surfacing with the
            # already-persisted production gate.  A candidate must never wear
            # a RECOMMENDED badge while its action is correctly disabled.
            visible_state = (
                "BLOCKED" if production_blocked else decision.surfacing_state.value
            )
            item["surfacing_state"] = visible_state
            item["selectable"] = decision.selectable and not production_blocked
            item["recommended"] = visible_state == "RECOMMENDED"
            item["recommendation_status"] = visible_state.lower()
            all_candidates.append(item)
        # Moments interprets persisted evidence through deterministic policy;
        # it never re-runs or reconstructs Brain/Vision evidence.
        draftable_candidates = [item for item in all_candidates if candidate_is_draftable(item)]
        self._all_candidates_by_id = {
            str(item["candidate_id"]): item for item in all_candidates
        }
        self._draftable_candidates_by_id = {
            str(item["candidate_id"]): item for item in draftable_candidates
        }
        candidates = all_candidates
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
            self._set_workflow_hint(
                self._analysis_load_error
                or "После поиска здесь появятся подходящие моменты и готовая рекомендация."
            )
            if self._analysis_load_error:
                integrity_notice = QLabel(
                    "Сохранённый анализ повреждён или изменён. Данные моментов не были открыты. "
                    "Повторите анализ видео."
                )
                integrity_notice.setObjectName("analysisIntegrityError")
                integrity_notice.setWordWrap(True)
                layout.addWidget(integrity_notice)
            self.workflow_hint.show()
            self.view_all_button.hide()
            self.draft_button.hide()
            self.production_button.hide()
            self._queue_drafts_workspace_geometry()
            return
        draftable_ids = [
            candidate_id for candidate_id in project.review_selected_candidate_ids
            if candidate_id in self._review_candidates_by_id
            and candidate_id in self._draftable_candidates_by_id
            and self._candidate_needs_draft(project, candidate_id)
        ]
        selected_moment_count = sum(
            candidate_id in self._draftable_candidates_by_id
            for candidate_id in project.review_selected_candidate_ids
        )
        recommended_count = sum(
            bool(item.get("recommended", item.get("selected_by_recommendation")))
            for item in draftable_candidates
        )
        available_count = sum(
            item.get("surfacing_state") == "AVAILABLE" and candidate_is_draftable(item)
            for item in all_candidates
        )
        unavailable_count = len(all_candidates) - len(draftable_candidates)
        rendered_count = sum(state == "rendered" for state in project.candidate_states.values())
        ready_count = sum(state in {"draft_ready", "selected"} for state in project.candidate_states.values())
        attention_count = sum(
            project.candidate_states.get(candidate_id) == "draft_failed"
            or project.candidate_draft_statuses.get(candidate_id) == "failed"
            or project.candidate_export_statuses.get(candidate_id) == "failed"
            for candidate_id in project.candidate_states
        )
        processing_count = sum(state in {"draft_planning", "production_rendering"} for state in project.candidate_states.values())
        if workflow_step == "drafts":
            set_responsive_text(
                self.review_metrics_text,
                f"Черновиков готовы: {ready_count + rendered_count} · "
                f"требуют внимания: {attention_count} · "
                f"выбрано для финала: {len(project.selected_candidate_ids)}"
            )
        else:
            set_responsive_text(
                self.review_metrics_text,
                f"Найдено {len(all_candidates)} · выбрано {selected_moment_count} · "
                f"рекомендуем {recommended_count} · доступны {available_count} · "
                f"недоступны {unavailable_count}"
            )
        if workflow_step == "candidates":
            # Moment selection belongs to the source-moment phase.  Drafts
            # have their own decisions (watch/approve/reject/retry), so do not
            # leak recommendation filters and draft-selection controls there.
            compact_actions = bool(self._compact_action_layout)
            action_direction = QBoxLayout.Direction.LeftToRight
            selection_toolbar = QFrame()
            selection_toolbar.setObjectName("momentSelectionToolbar")
            toolbar_layout = QHBoxLayout(selection_toolbar)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            toolbar_layout.setSpacing(6)
            has_draftable_candidates = bool(self._draftable_candidates_by_id)
            recommended_button = QPushButton("Лучшие")
            recommended_button.setToolTip("Выбрать только рекомендованные моменты")
            recommended_button.setObjectName("selectRecommendedCandidates")
            recommended_button.setEnabled(has_draftable_candidates)
            recommended_button.clicked.connect(self._select_recommended)
            select_all_button = QPushButton("Все")
            select_all_button.setToolTip("Выбрать все доступные моменты")
            select_all_button.setObjectName("selectAllCandidates")
            select_all_button.setEnabled(has_draftable_candidates)
            select_all_button.clicked.connect(self._select_all_candidates)
            clear_button = QPushButton("Снять")
            clear_button.setToolTip("Снять выбор со всех моментов")
            clear_button.setObjectName("clearCandidateSelection")
            clear_button.setEnabled(bool(selected_moment_count))
            clear_button.clicked.connect(self._clear_review_selection)
            toolbar_layout.addWidget(recommended_button)
            toolbar_layout.addWidget(select_all_button)
            toolbar_layout.addWidget(clear_button)
            layout.addWidget(selection_toolbar)
            if candidates and not has_draftable_candidates:
                quality_notice = QLabel(
                    f"Найдено {len(candidates)} моментов, но ни один пока не прошёл проверку качества"
                )
                quality_notice.setObjectName("candidateQualityNotice")
                quality_notice.setWordWrap(True)
                layout.addWidget(quality_notice)
            filters = QFrame()
            filters.setObjectName("reviewFilters")
            filters_layout = QHBoxLayout(filters)
            filters_layout.setDirection(action_direction)
            filters_layout.setContentsMargins(0, 0, 0, 0)
            filters_layout.setSpacing(6)
            filter_combo = QComboBox()
            filter_combo.addItem("Рекомендуем" if compact_actions else "Рекомендованные", "recommended")
            filter_combo.addItem("Все" if compact_actions else "Все моменты", "all")
            filter_combo.addItem("Не выбраны", "unselected")
            filter_combo.addItem("Высокий" if compact_actions else "Высокий потенциал", "high")
            filter_combo.addItem("Средний" if compact_actions else "Средний потенциал", "medium")
            self._set_combo_data(filter_combo, self._candidate_filter)
            filter_combo.currentIndexChanged.connect(lambda _index: self._change_candidate_filter(str(filter_combo.currentData())))
            sort_combo = QComboBox()
            sort_combo.addItem("Сильные" if compact_actions else "Сначала сильные", "recommendation")
            sort_combo.addItem("По времени", "time")
            sort_combo.addItem("По потенциалу", "potential")
            self._set_combo_data(sort_combo, self._candidate_sort)
            sort_combo.currentIndexChanged.connect(lambda _index: self._change_candidate_sort(str(sort_combo.currentData())))
            filters_layout.addWidget(filter_combo, 1)
            filters_layout.addWidget(sort_combo, 1)
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
            candidate_draftable = candidate_id in self._draftable_candidates_by_id
            state = project.candidate_states.get(candidate_id, str(item.get("recommendation_status") or "analyzed"))
            override = project.candidate_boundary_overrides.get(candidate_id, {})
            original_start = item.get("start_seconds", item.get("start", 0))
            original_end = item.get("end_seconds", item.get("end", original_start))
            start_value = override.get("start", original_start) if isinstance(override, dict) else original_start
            end_value = override.get("end", original_end) if isinstance(override, dict) else original_end
            start, end = format_seconds(start_value), format_seconds(end_value)
            draft_status = project.candidate_draft_statuses.get(candidate_id)
            export_status = project.candidate_export_statuses.get(candidate_id)
            preview_projection = self._draft_preview_projection(
                project,
                candidate_id,
                has_preview=candidate_id in self._draft_preview_paths,
            )
            status_label = {
                "analyzed": "Можно добавить к черновикам", "draft_planning": "Готовим черновик",
                "draft_ready": "Черновик готов к проверке", "draft_failed": "Черновик не готов",
                "selected": "Подтверждён", "production_rendering": "Создаём готовый ролик", "rendered": "Готово",
            }.get(state, "можно посмотреть")
            if state == "draft_ready" and candidate_id not in project.review_selected_candidate_ids:
                status_label = "черновик не выбран для готового ролика"
            if draft_status == "failed" or state == "draft_failed":
                requirement = self.viewmodel.services.draft_retry_requirement(project, candidate_id)
                status_label = requirement or "Черновик не создан. Его можно повторить отдельно."
            elif export_status == "failed":
                status_label = "Готовый ролик не создан. Черновик сохранён и остаётся подтверждённым."
            elif workflow_step == "drafts" and preview_projection.stale:
                status_label = preview_projection.inspector_text
            if workflow_step == "candidates" and not candidate_draftable:
                status_label = "Не прошёл проверку качества · только просмотр"
            frame = QFrame(); frame.setObjectName("card")
            frame.setProperty("candidateBlocked", workflow_step == "candidates" and not candidate_draftable)
            frame.setProperty(
                "candidateSelected",
                candidate_id in (
                    project.selected_candidate_ids
                    if workflow_step == "drafts"
                    else project.review_selected_candidate_ids
                ),
            )
            frame.setMinimumWidth(0)
            frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self._candidate_cards[candidate_id] = frame
            row = QVBoxLayout(frame)
            row.setContentsMargins(10, 8, 10, 8)
            row.setSpacing(8)
            thumbnail = QLabel("Кадр\nзагружается")
            thumbnail.setObjectName("candidateThumbnail")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if workflow_step == "drafts":
                thumbnail.setFixedSize(
                    58 if self._compact_action_layout else 68,
                    96 if self._compact_action_layout else 112,
                )
            else:
                thumbnail.setFixedSize(
                    88 if self._compact_action_layout else 104,
                    52 if self._compact_action_layout else 60,
                )
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
            editorial_state = str(item.get("surfacing_state") or "AVAILABLE")
            state_badge = QLabel({
                "RECOMMENDED": "Рекомендуем",
                "AVAILABLE": "Доступен",
                "BLOCKED": "Недоступен",
            }.get(editorial_state, "Доступен"))
            state_badge.setObjectName("momentState")
            state_badge.setProperty("surfacingState", editorial_state)
            information.addWidget(state_badge, 0, Qt.AlignmentFlag.AlignLeft)
            details = QLabel(f"{start}–{end} · {format_seconds(max(0.0, end_seconds - start_seconds))}")
            details.setObjectName("muted")
            make_label_shrinkable(details)
            information.addWidget(details)
            if workflow_step == "candidates" and not candidate_draftable:
                blocked_reason = QLabel(
                    self._candidate_block_reason(item)
                )
                blocked_reason.setObjectName("candidateBlockedReason")
                blocked_reason.setWordWrap(True)
                information.addWidget(blocked_reason)
            summary_row = QHBoxLayout()
            summary_row.setContentsMargins(0, 0, 0, 0)
            summary_row.setSpacing(8)
            summary_row.addWidget(thumbnail)
            summary_row.addLayout(information, 1)
            row.addLayout(summary_row)
            actions_host = QWidget()
            actions_host.setMinimumWidth(0)
            actions_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            actions = QGridLayout(actions_host)
            actions.setContentsMargins(0, 0, 0, 0)
            actions.setSpacing(6)
            actions.setColumnStretch(0, 1)
            actions.setColumnStretch(1, 1)
            action_index = 0

            def add_candidate_action(button: QPushButton, *, span: int = 1) -> None:
                nonlocal action_index
                button.setProperty("candidateAction", True)
                row_index = action_index // 2
                column_index = action_index % 2
                if span == 2 and column_index:
                    action_index += 1
                    row_index = action_index // 2
                    column_index = 0
                actions.addWidget(button, row_index, column_index, 1, span)
                action_index += span
            status = QLabel(status_label)
            status.setObjectName("muted")
            status.setMaximumWidth(16_777_215)
            status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            candidate_error = project.candidate_errors.get(candidate_id)
            if candidate_error:
                # The service stores an item/stage message for the log.  Card
                # copy stays short and non-technical, with the lifecycle state
                # above explaining exactly which retry is safe.
                status.setText(status_label)
                status.setToolTip("Подробности сохранены в журнале проекта.")
            make_label_shrinkable(status)
            row.addWidget(status)
            source_preview = QPushButton("Источник" if workflow_step == "drafts" else "Смотреть")
            source_preview.setObjectName(f"preview-candidate-{candidate_id}")
            source_preview.setToolTip("Посмотреть исходный фрагмент")
            source_preview.clicked.connect(lambda _checked=False, value=dict(item): self._preview_candidate(value))
            add_candidate_action(source_preview)
            if preview_file and preview_file.is_file():
                button = QPushButton("Creative Preview")
                button.setObjectName(f"draft-preview-candidate-{candidate_id}")
                button.setToolTip("Посмотреть Creative Preview этого черновика")
                button.clicked.connect(
                    lambda _checked=False, path=preview_file, title=str(item.get("title") or item.get("core_idea") or "момент"), value=candidate_id:
                    self._show_draft_preview(path, title, value)
                )
                add_candidate_action(button)
            if state == "draft_ready":
                if candidate_id in project.review_selected_candidate_ids:
                    approve = QPushButton("Подтвердить")
                    approve.setObjectName(f"approve-candidate-{candidate_id}")
                    approve.clicked.connect(
                        lambda _checked=False, value=candidate_id: self._set_draft_approval(value, True)
                    )
                    reject = QPushButton("Отклонить")
                    reject.setObjectName(f"reject-candidate-{candidate_id}")
                    reject.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                    add_candidate_action(approve)
                    add_candidate_action(reject)
                else:
                    restore = QPushButton("Вернуть")
                    restore.setObjectName(f"restore-candidate-{candidate_id}")
                    restore.setToolTip("Вернуть черновик к проверке")
                    restore.clicked.connect(lambda _checked=False, value=candidate_id: self._restore_draft(value))
                    add_candidate_action(restore)
            elif state == "selected":
                if export_status == "failed":
                    retry_export = QPushButton("Повторить экспорт")
                    retry_export.setObjectName(f"retry-final-candidate-{candidate_id}")
                    retry_export.setToolTip("Повторно создаст готовый ролик из сохранённого черновика; анализ и предпросмотр не повторяются.")
                    retry_export.setDisabled(self.viewmodel.blocked_by_other_project)
                    retry_export.clicked.connect(lambda _checked=False, value=candidate_id: self._retry_final_export(value))
                    add_candidate_action(retry_export)
                reject = QPushButton("Отклонить")
                reject.setObjectName(f"reject-candidate-{candidate_id}")
                reject.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                add_candidate_action(reject)
            elif state == "draft_failed" and workflow_step == "drafts":
                retry_requirement = self.viewmodel.services.draft_retry_requirement(project, candidate_id)
                if candidate_id in self._draftable_candidates_by_id:
                    retry = QPushButton(
                        "Исправьте границы" if retry_requirement else "Повторить черновик"
                    )
                    retry.setObjectName(f"retry-candidate-{candidate_id}")
                    retry.setToolTip(
                        retry_requirement
                        or "Повторно создаст только этот черновик; найденные моменты не будут анализироваться заново."
                    )
                    retry.setDisabled(self.viewmodel.blocked_by_other_project or bool(retry_requirement))
                    retry.clicked.connect(lambda _checked=False, value=candidate_id: self._retry_draft(value))
                    add_candidate_action(retry)
                if candidate_id in project.review_selected_candidate_ids:
                    skip = QPushButton("Пропустить")
                    skip.setObjectName(f"skip-candidate-{candidate_id}")
                    skip.setToolTip("Уберёт этот неготовый черновик из текущего набора, не затрагивая готовые ролики.")
                    skip.clicked.connect(lambda _checked=False, value=candidate_id: self._reject_draft(value))
                    add_candidate_action(skip)
                else:
                    if retry_requirement:
                        repair = QPushButton("Исправить границы")
                        repair.setObjectName(f"fix-boundary-candidate-{candidate_id}")
                        repair.setToolTip(retry_requirement)
                        repair.clicked.connect(
                            lambda _checked=False, value=dict(item): self._preview_candidate(value)
                        )
                        add_candidate_action(repair)
                    else:
                        restore = QPushButton("Вернуть в набор")
                        restore.setObjectName(f"restore-candidate-{candidate_id}")
                        restore.setToolTip("Вернёт этот черновик в текущий набор, после чего его можно повторить отдельно.")
                        restore.clicked.connect(lambda _checked=False, value=candidate_id: self._restore_draft(value))
                        add_candidate_action(restore)
            elif state == "rendered":
                final_file = final_outputs.get(candidate_id)
                if final_file:
                    watch_final = QPushButton("Смотреть готовый ролик")
                    watch_final.setObjectName(f"watch-final-candidate-{candidate_id}")
                    watch_final.clicked.connect(
                        lambda _checked=False, path=final_file, title=str(item.get("title") or item.get("core_idea") or "момент"), value=candidate_id:
                        self._show_final_preview(path, title, value)
                    )
                    add_candidate_action(watch_final)
                    open_final = QPushButton("Открыть готовый ролик")
                    open_final.setObjectName(f"open-final-candidate-{candidate_id}")
                    open_final.clicked.connect(lambda _checked=False, path=final_file: self._open_file(path))
                    add_candidate_action(open_final)
            elif workflow_step == "candidates" and state not in {"draft_planning", "production_rendering"}:
                if candidate_draftable:
                    selected_for_draft = candidate_id in project.review_selected_candidate_ids
                    select = QPushButton("Убрать" if selected_for_draft else "Добавить")
                    select.setObjectName(f"select-candidate-{candidate_id}")
                    select.setToolTip(
                        "Убрать из черновиков" if selected_for_draft else "Добавить к черновикам"
                    )
                    select.clicked.connect(lambda _checked=False, value=candidate_id: self._toggle_candidate_selection(value))
                    self._candidate_selection_buttons[candidate_id] = select
                    add_candidate_action(select)
                else:
                    blocked = QPushButton("Черновик недоступен")
                    blocked.setObjectName(f"blocked-candidate-{candidate_id}")
                    blocked.setEnabled(False)
                    add_candidate_action(blocked, span=2)
            if candidate_error:
                open_log = QPushButton("Открыть журнал")
                open_log.setObjectName(f"candidate-log-{candidate_id}")
                open_log.clicked.connect(self._open_latest_run_log_folder)
                add_candidate_action(open_log)
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
        self._queue_drafts_workspace_geometry()

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
        if self.viewmodel.blocked_by_other_project:
            self._set_workflow_hint(self._other_project_job_hint())
            return
        if project.status in {"analyzing", "processing", "rendering_selected"} or processing_count:
            self._set_workflow_hint("Сейчас идёт работа. Прогресс и оставшееся время показаны на отдельном экране.")
            return
        workflow_step = self._derive_flow_step(project)
        if workflow_step == "candidates":
            recommended_ids = self._recommended_candidate_ids()
            selected_ids = [
                candidate_id for candidate_id in project.review_selected_candidate_ids
                if candidate_id in self._draftable_candidates_by_id
            ]
            if self._all_candidates_by_id and not self._draftable_candidates_by_id:
                count = len(self._all_candidates_by_id)
                self._set_workflow_hint(
                    f"Найдено {count} моментов, но ни один пока не прошёл проверку качества"
                )
                self._set_review_action_text(
                    self.view_all_button,
                    f"Посмотреть все {count}",
                    f"Все ({count})",
                )
                self.view_all_button.show()
                return
            if not selected_ids and recommended_ids:
                count = len(recommended_ids)
                self._set_workflow_hint(
                    f"Найдено {len(self._review_candidates_by_id)} · рекомендуем {count}. "
                    "Можно сразу подготовить лучшие варианты или изменить выбор в списке."
                )
                self._set_review_action_text(
                    self.view_all_button,
                    f"Посмотреть все {len(self._review_candidates_by_id)}",
                    f"Все ({len(self._review_candidates_by_id)})",
                )
                self.view_all_button.show()
                self._set_review_action_text(
                    self.draft_button,
                    f"Создать {count} рекомендованных",
                    f"Создать лучшие ({count})",
                )
                self.draft_button.setEnabled(True)
                self.draft_button.show()
                return
        # An explicit approval is the durable hand-off to Final.  It wins over
        # failed/pending siblings: those candidates retain their own card-level
        # retry/skip actions and must not hold successful drafts hostage.
        if workflow_step == "drafts" and project.selected_candidate_ids:
            count = len(project.selected_candidate_ids)
            self._set_workflow_hint(
                f"Подтверждено: {count}. Черновики проверены — теперь можно создать готовые ролики."
            )
            self._set_review_action_text(
                self.production_button,
                f"Создать готовые ролики ({count})",
                f"Создать ролики ({count})",
            )
            self.production_button.setEnabled(True)
            self.production_button.show()
            return
        if draftable_ids:
            count = len(draftable_ids)
            is_drafts = workflow_step == "drafts"
            changed_count = sum(
                candidate_id in project.candidate_draft_artifacts
                for candidate_id in draftable_ids
            )
            if is_drafts and changed_count:
                self._set_workflow_hint(
                    "Изменения сохранены как ожидающие. Предыдущие Preview остаются доступны, "
                    "пока новые версии не будут готовы."
                )
                full = (
                    "Пересоздать черновик" if changed_count == 1
                    else f"Пересоздать изменённые ({changed_count})"
                )
                compact = "Пересоздать" if changed_count == 1 else f"Пересоздать ({changed_count})"
                self._set_review_action_text(self.draft_button, full, compact)
                self.draft_button.setEnabled(True)
                self.draft_button.show()
                return
            selected_count = sum(
                candidate_id in self._review_candidates_by_id
                for candidate_id in project.review_selected_candidate_ids
            )
            selection_summary = (
                f"Выбрано {count}."
                if selected_count == count
                else f"Выбрано {selected_count}. Для {count} из них ещё нужен черновик."
            )
            self._set_workflow_hint(
                f"{selection_summary} Следующий шаг — создать черновики, чтобы посмотреть ролики перед финальной сборкой."
            )
            self._set_review_action_text(
                self.draft_button,
                f"Создать черновики ({count})",
                f"Создать ({count})",
            )
            self.draft_button.setEnabled(True)
            self.draft_button.show()
            return
        if project.selected_candidate_ids:
            count = len(project.selected_candidate_ids)
            self._set_workflow_hint(
                f"Подтверждено: {count}. Черновики проверены — теперь можно создать готовые ролики."
            )
            self._set_review_action_text(
                self.production_button,
                f"Создать готовые ролики ({count})",
                f"Создать ролики ({count})",
            )
            self.production_button.setEnabled(True)
            self.production_button.show()
            return
        if ready_count:
            self._set_workflow_hint(
                "Черновики готовы. Посмотрите каждый как вертикальный ролик, затем подтвердите нужные или отклоните их."
            )
            return
        if rendered_count:
            self._set_workflow_hint("Готовые ролики можно посмотреть в карточках или открыть в папке проекта.")
            return
        self._set_workflow_hint(
            "Посмотрите моменты и выберите любое число подходящих фрагментов."
        )

    def _set_workflow_hint(self, value: object) -> None:
        set_responsive_text(self.workflow_hint, value)
        if hasattr(self, "stage_actions"):
            QTimer.singleShot(0, self._refresh_stage_action_geometry)

    def _set_review_action_text(
        self,
        button: QPushButton,
        full_text: str,
        compact_text: str,
    ) -> None:
        button.setProperty("responsiveFullText", full_text)
        button.setProperty("responsiveCompactText", compact_text)
        button.setToolTip(full_text)
        button.setText(compact_text if self._compact_action_layout else full_text)
        if hasattr(self, "stage_actions"):
            QTimer.singleShot(0, self._refresh_stage_action_geometry)

    def _runs_for_project(self, project: DesktopProject) -> list[ProjectRun]:
        if self.runs and all(run.project_id == project.project_id for run in self.runs):
            return self.runs
        return self.viewmodel.services.runs_for(project)

    def _final_output_records(self, project: DesktopProject) -> list[ClipResult]:
        """Project the canonical manifest registry without parsing full reports."""

        collected: list[ClipResult] = []
        for run in self._runs_for_project(project):
            if run.run_kind not in {RunKind.FULL, RunKind.SELECTED_RENDER, RunKind.RENDER_REVISION}:
                continue
            for result in self.viewmodel.services.run_projection(run).primary_results:
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
            # QualityReport is already bound to this exact persisted artifact.
            # Its technical facts fill the UI without filename guessing or a
            # synchronous ffprobe during screen refresh.
            media = self._quality_media_for_result(project, result)
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
                status=(
                    "completed" if result.quality_status in {"", "PASS"}
                    else "warning"
                ),
                run_id=self._run_id_for_result(project, result),
            ))
        self.final_results.set_results(
            outputs,
            selected_id=project.last_final_result_id,
            project_directory=project.directory,
            warnings=self._final_warnings(project),
        )

    def _quality_media_for_result(
        self, project: DesktopProject, result: ClipResult,
    ) -> dict[str, object]:
        """Return technical metadata only from this result's bound report."""

        if not result.quality_report_path:
            return {}
        quality_path = Path(result.quality_report_path)
        raw = read_json(quality_path, {}) if quality_path.is_file() else {}
        if not isinstance(raw, dict):
            return {}
        if str(raw.get("candidate_id") or "") != result.candidate_id:
            return {}
        if result.artifact_id and str(raw.get("artifact_id") or "") != result.artifact_id:
            return {}
        if str(raw.get("project_id") or project.project_id) != project.project_id:
            return {}
        report_path = Path(str(raw.get("artifact_path") or ""))
        try:
            if report_path.resolve() != Path(result.output_file).resolve():
                return {}
        except (OSError, ValueError):
            return {}
        metrics = raw.get("metrics")
        technical = metrics.get("technical") if isinstance(metrics, dict) else None
        if not isinstance(technical, dict):
            return {}
        media: dict[str, object] = {"duration": technical.get("duration")}
        resolution = str(technical.get("resolution") or "").lower().replace("×", "x")
        width_text, separator, height_text = resolution.partition("x")
        if separator:
            try:
                media["width"] = int(width_text.strip())
                media["height"] = int(height_text.strip())
            except ValueError:
                pass
        return media

    def _final_candidate_metadata(self, project: DesktopProject) -> dict[str, dict[str, object]]:
        """Read titles and source ranges from already-persisted candidate metadata."""

        analysis = self._analysis_artifact(project)
        candidates = analysis.get("candidates", [])
        metadata = {
            str(item.get("candidate_id")): {
                "title": str(item.get("title") or item.get("core_idea") or "Готовый ролик"),
                "start": item.get("start_seconds", item.get("start")),
                "end": item.get("end_seconds", item.get("end")),
            }
            for item in candidates if isinstance(item, dict) and item.get("candidate_id")
        }
        if not metadata:
            for run in self._runs_for_project(project):
                for candidate_id, values in self.viewmodel.services.run_projection(run).candidate_metadata.items():
                    metadata.setdefault(candidate_id, dict(values))
        return metadata

    def _run_id_for_result(self, project: DesktopProject, result: ClipResult) -> str:
        """Resolve a legacy result's owner through canonical report metadata."""

        if result.run_id:
            return result.run_id
        target = str(Path(result.output_file)).replace("\\", "/").casefold()
        for run in self._runs_for_project(project):
            registry = self.viewmodel.services.run_projection(run).primary_results
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
                warnings.append(self._quality_finding_message(finding))
            if status == "PASS_WITH_WARNINGS" and not raw_quality.get("findings"):
                warnings.append("Ролик создан с предупреждениями проверки качества. Откройте его и проверьте перед публикацией.")
        if quality_found:
            return list(dict.fromkeys(warnings))
        for run in self._runs_for_project(project):
            warnings.extend(run.warnings)
            if run.status == "completed_with_warnings" and not run.warnings:
                warnings.extend(self.viewmodel.services.run_projection(run).warnings)
        return self._summarize_final_warnings(list(dict.fromkeys(warnings)))

    @staticmethod
    def _quality_finding_message(finding: dict[str, object]) -> str:
        """Translate report codes without leaking raw/internal report prose."""

        code = str(finding.get("code") or "")
        messages = {
            "QUALITY_CONFIDENCE_LOW": "Некоторые оценки качества предварительные — проверьте ролик перед публикацией.",
            "NATIVE_CREATIVE_FALLBACK": "Часть творческого оформления использует безопасный упрощённый вариант.",
            "CAPTION_READABILITY_FALLBACK": "Субтитры упрощены, чтобы сохранить читаемость.",
            "COMPOSITION_LOW_CONFIDENCE": "Кадрирование выбрано осторожно из-за неопределённости в кадре.",
            "COMPOSITION_SAFE_FALLBACK": "Использовано безопасное кадрирование, сохраняющее важную область видео.",
            "CAPTION_METRICS_FALLBACK": "Для субтитров использованы консервативные параметры читаемости.",
            "AUDIO_UNINTELLIGIBLE": "В одном из фрагментов речь может быть недостаточно разборчивой.",
            "BOUNDARY_WORD_CUT": "Граница фрагмента может обрывать слово.",
        }
        if code in messages:
            return messages[code]
        if str(finding.get("severity") or "") == "blocker":
            return "Проверка качества нашла проблему, которую нужно исправить перед публикацией."
        return "Проверка качества рекомендует посмотреть этот ролик перед публикацией."

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
            if not any("А" <= character <= "я" or character == "ё" for character in text):
                summarized.append(
                    "Во время сборки использован безопасный запасной вариант. Проверьте ролик перед публикацией."
                )
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
            poster_cache_directory=self.project.directory / "preview-posters",
        )
        self._update_candidate_review(self.project)
        self._apply_flow_visibility(self.project)

    def _back_to_drafts(self) -> None:
        if not self.project or not self.project.candidate_draft_artifacts:
            return
        self._results_subflow_override = "drafts"
        self._update_candidate_review(self.project)
        self._reconcile_active_candidate_preview(self.project, previous_step="finished")
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
            if candidate_id in self._draftable_candidates_by_id
        ]
        if candidate_ids != self.project.review_selected_candidate_ids:
            # A saved legacy/stale choice must not bypass the current Moments
            # boundary when the user starts a new draft batch.
            self.viewmodel.set_review_selection(candidate_ids)
            if not self.project:
                return
        if not candidate_ids and self._derive_flow_step(self.project) == "candidates":
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
        if state in {"rendered", "production_rendering", "draft_planning", "draft_failed"}:
            return False
        if state in {"draft_ready", "selected"}:
            return not Path(project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
        return True

    def _retry_draft(self, candidate_id: str) -> None:
        """Retry one failed draft without broadening the existing selection."""

        if not self.project or candidate_id not in self._draftable_candidates_by_id:
            return
        if self.project.candidate_states.get(candidate_id) == "draft_failed":
            if self.viewmodel.services.draft_retry_requirement(self.project, candidate_id):
                return
            if candidate_id not in self.project.review_selected_candidate_ids:
                self.viewmodel.set_review_selection([
                    *self.project.review_selected_candidate_ids, candidate_id,
                ])
            if not self.project:
                return
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
        if not self.project or candidate_id not in self._draftable_candidates_by_id:
            return
        selected = list(self.project.review_selected_candidate_ids)
        if candidate_id in selected:
            selected.remove(candidate_id)
        else:
            selected.append(candidate_id)
        self._set_review_selection_without_rebuild(selected)

    def _set_draft_approval(self, candidate_id: str, approved: bool) -> None:
        self.viewmodel.set_draft_approval(candidate_id, approved)

    def _reject_draft(self, candidate_id: str) -> None:
        # One persisted candidate identity owns this action.  Do not compose
        # two UI-side mutations from a potentially refreshed card list: that
        # was prone to routing a skip through a sibling's pending selection.
        self.viewmodel.exclude_draft_candidate(candidate_id)

    def _restore_draft(self, candidate_id: str) -> None:
        if not self.project or candidate_id in self.project.review_selected_candidate_ids:
            return
        self.viewmodel.set_review_selection([*self.project.review_selected_candidate_ids, candidate_id])

    def _recommended_candidate_ids(self) -> list[str]:
        return [
            candidate_id
            for candidate_id, item in self._draftable_candidates_by_id.items()
            if item.get("recommended", item.get("selected_by_recommendation"))
            and self.project
            and self.project.candidate_draft_statuses.get(candidate_id) != "failed"
        ]

    def _select_recommended(self) -> None:
        if not self.project:
            return
        self._set_review_selection_without_rebuild(self._recommended_candidate_ids())

    def _select_all_candidates(self) -> None:
        if self.project:
            self._set_review_selection_without_rebuild(list(self._draftable_candidates_by_id))

    def _view_all_candidates(self) -> None:
        if not self.project:
            return
        self._candidate_filter = "all"
        # Initial paint stays bounded.  This explicit user action may reveal
        # the complete catalogue, but it still reuses the verified Analysis
        # projection and asynchronous thumbnail loader.
        self._candidate_visible_limit = max(12, len(self._all_candidates_by_id))
        self._update_candidate_review(self.project)
        self.content_scroll.ensureWidgetVisible(self.candidate_review, 0, 16)
        self.candidate_review.setFocus(Qt.FocusReason.OtherFocusReason)

    def _clear_review_selection(self) -> None:
        self._set_review_selection_without_rebuild([])

    def _set_review_selection_without_rebuild(self, candidate_ids: list[str]) -> None:
        if not self.project:
            return
        self._persisting_review_selection = True
        try:
            self.viewmodel.set_review_selection(candidate_ids)
        finally:
            self._persisting_review_selection = False

    def _refresh_moment_selection_ui(self) -> None:
        """Update checkbox/CTA state without rebuilding the candidate cards."""

        if not self.project:
            return
        if self._candidate_filter == "unselected":
            self._update_candidate_review(self.project)
            return
        selected = set(self.project.review_selected_candidate_ids)
        for candidate_id, button in self._candidate_selection_buttons.items():
            button.setText(
                "Убрать" if candidate_id in selected else "Добавить"
            )
            button.setToolTip(
                "Убрать из черновиков" if candidate_id in selected else "Добавить к черновикам"
            )
        recommended_count = sum(
            bool(item.get("recommended", item.get("selected_by_recommendation")))
            for item in self._draftable_candidates_by_id.values()
        )
        available_count = sum(
            item.get("surfacing_state") == "AVAILABLE"
            for item in self._draftable_candidates_by_id.values()
        )
        set_responsive_text(
            self.review_metrics_text,
            f"Найдено {len(self._all_candidates_by_id)} · выбрано {len(selected)} · "
            f"рекомендуем {recommended_count} · доступны {available_count} · "
            f"недоступны {len(self._all_candidates_by_id) - len(self._draftable_candidates_by_id)}",
        )
        draftable_ids = [
            candidate_id for candidate_id in self.project.review_selected_candidate_ids
            if candidate_id in self._draftable_candidates_by_id
            and self._candidate_needs_draft(self.project, candidate_id)
        ]
        ready_count = sum(
            state in {"draft_ready", "selected"}
            for state in self.project.candidate_states.values()
        )
        rendered_count = sum(
            state == "rendered" for state in self.project.candidate_states.values()
        )
        processing_count = sum(
            state in {"draft_planning", "production_rendering"}
            for state in self.project.candidate_states.values()
        )
        self._configure_workflow_action(
            self.project, draftable_ids, ready_count, rendered_count, processing_count,
        )

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

    def _filtered_candidates(self, candidates: list[dict[str, Any]], project: DesktopProject) -> list[dict]:
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

    @staticmethod
    def _candidate_block_reason(candidate: dict) -> str:
        """Translate the first structural/technical policy blocker into concise UI copy."""

        decision = candidate.get("production_editorial_decision")
        if not isinstance(decision, dict):
            decision = candidate.get("editorial_decision")
        if not isinstance(decision, dict):
            return "Для этого момента нет актуальной проверки качества."
        labels = {
            "SOURCE_INTERVAL_INVALID": "Некорректный диапазон исходного видео.",
            "CANDIDATE_IDENTITY_INVALID": "Не удалось подтвердить идентичность фрагмента.",
            "WORD_BOUNDARY_UNRECOVERABLE": "Начало или конец обрывает слово.",
            "SENTENCE_BOUNDARY_UNRECOVERABLE": "Начало или конец обрывает фразу.",
            "BOUNDARY_EVIDENCE_UNAVAILABLE": "Недостаточно данных, чтобы надёжно определить границы.",
            "SEMANTIC_INCOMPLETE": "Мысль во фрагменте не завершена.",
            "CONTEXT_DEBT_CRITICAL": "Фрагмент непонятен без важного контекста.",
            "UNRESOLVED_PRONOUN": "Непонятно, о ком или о чём идёт речь.",
            "UNNAMED_ENTITY": "Важный участник или объект не назван.",
            "ANSWER_WITHOUT_QUESTION_CONTEXT": "Ответ показан без необходимого вопроса.",
            "REFERENCES_EARLIER_CONTENT": "Фрагмент ссылается на более ранний контекст.",
            "UNDEFINED_TERM_OR_SETUP": "Не хватает объяснения или завязки.",
            "NO_PAYOFF": "Во фрагменте нет завершения или развязки.",
            "FALSE_HOOK_RISK": "Начало обещает то, чего фрагмент не раскрывает.",
            "AUDIO_UNINTELLIGIBLE": "Речь недостаточно разборчива.",
            "SPEECH_CLARITY_EVIDENCE_UNAVAILABLE": "Недостаточно данных, чтобы подтвердить разборчивость речи.",
            "VERTICAL_COMPOSITION_IMPOSSIBLE": "Сцену нельзя безопасно адаптировать под вертикальный кадр.",
            "VISUAL_EVIDENCE_UNAVAILABLE": "Недостаточно визуальных данных для проверки.",
            "DURATION_OUT_OF_RANGE": "Длительность фрагмента вне допустимого диапазона.",
            "LEGACY_UNASSESSED": "Для этого момента нет актуальной проверки качества.",
            "EDITORIAL_EVIDENCE_UNASSESSED": "Для этого момента нет актуальной проверки качества.",
            "INVALID_SOURCE_MAPPING": "Не удалось подтвердить диапазон исходного видео.",
            "PRODUCTION_FEASIBILITY_BLOCKED": "Из этого фрагмента нельзя безопасно создать черновик.",
        }
        reason_codes = decision.get("hard_blockers")
        if isinstance(reason_codes, list):
            for raw_code in reason_codes:
                label = labels.get(str(raw_code))
                if label:
                    return label
        return "Момент пока не прошёл проверку качества."

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
            self.preview.show_draft(
                str(path), title,
                poster_cache_directory=(
                    self.project.directory / "preview-posters" if self.project else None
                ),
            )
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
        self.preview.show_final(
            str(path), title,
            poster_cache_directory=(
                self.project.directory / "preview-posters" if self.project else None
            ),
        )
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
            poster_path = self._candidate_thumbnail_paths.get(candidate_id)
            if poster_path is not None and poster_path.is_file():
                self.preview.show_bound_poster(poster_path)
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
            self.preview.show_draft(
                str(path),
                title or str(candidate.get("title") or candidate.get("core_idea") or "момент"),
                poster_cache_directory=(
                    self.project.directory / "preview-posters" if self.project else None
                ),
            )
        self._update_draft_preview_badge(candidate_id)
        self._persist_active_preview_candidate(candidate_id)

    def _update_draft_preview_badge(self, candidate_id: str) -> None:
        """Apply the shared candidate projection to the visible phone player."""

        projection = self._draft_preview_projection(
            self.project,
            candidate_id,
            has_preview=True,
        )
        self.preview.set_context_badge(
            projection.badge_text,
            state=projection.badge_state,
            object_name="creativePreviewStatus",
        )

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
        workflow_step = self._derive_flow_step(self.project)
        editorial_state = str(candidate.get("surfacing_state") or "AVAILABLE")
        state_copy = {
            "RECOMMENDED": "Рекомендуем: самостоятельный сильный фрагмент.",
            "AVAILABLE": "Доступен: можно выбрать и сделать черновик.",
            "BLOCKED": "Только просмотр: черновик недоступен.",
        }.get(editorial_state, "Можно просмотреть исходный фрагмент.")
        lines = [
            str(candidate.get("title") or "Момент"),
            f"{format_seconds(start)}–{format_seconds(end)} · {format_seconds(end - start)}",
            state_copy,
        ]
        candidate_id = str(candidate.get("candidate_id") or "")
        preview_projection = self._draft_preview_projection(
            self.project,
            candidate_id,
            has_preview=candidate_id in self._draft_preview_paths,
        )
        if (
            workflow_step == "drafts"
            and candidate_id
            and candidate_id == self._active_candidate_id
            and self._active_preview_kind == "draft"
        ):
            self._update_draft_preview_badge(candidate_id)
        candidate_draftable = (
            self._flow_step != "candidates"
            or candidate_id in self._draftable_candidates_by_id
        )
        if not candidate_draftable:
            lines.append(
                "Черновик недоступен: " + self._candidate_block_reason(candidate)
            )
        if candidate_id and self.project.candidate_errors.get(candidate_id):
            if self.project.candidate_export_statuses.get(candidate_id) == "failed":
                lines.append("Готовый ролик для этого момента не создан. Сохранённый черновик остаётся подтверждённым: повторите только экспорт или снимите подтверждение.")
            elif requirement := self.viewmodel.services.draft_retry_requirement(self.project, candidate_id):
                lines.append(requirement)
            elif preview_projection.stale:
                lines.append(
                    f"{preview_projection.inspector_text}. "
                    "Подробности сохранены в журнале проекта."
                )
            else:
                lines.append("Черновик для этого момента не создан. Его можно повторить отдельно; подробности сохранены в журнале проекта.")
        self._replace_card_text(self.candidate_detail, lines)
        if not candidate_draftable:
            self._refresh_candidate_detail_geometry()
            return
        heading = self.candidate_detail.layout().itemAt(0).widget()
        if isinstance(heading, QLabel):
            heading.setText("Оформление черновика" if workflow_step == "drafts" else "О моменте")
        if workflow_step == "drafts" and candidate_id:
            self._append_draft_inspector(candidate_id, start, end)
        boundary_heading = QLabel("Границы фрагмента")
        boundary_heading.setObjectName("inspectorSectionTitle")
        self.candidate_detail.layout().addWidget(boundary_heading)
        boundary_copy = QLabel("Подвиньте начало или конец. Повторный анализ не запускается.")
        boundary_copy.setObjectName("muted")
        boundary_copy.setWordWrap(True)
        self.candidate_detail.layout().addWidget(boundary_copy)
        controls = QWidget()
        controls.setObjectName("candidateBoundaryControls")
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        columns = 2 if self._compact_stage_layout else 4
        for index, (full_text, compact_text, boundary, delta) in enumerate((
            ("Начало −1 с", "Н −1", "start", -1.0),
            ("Начало −0.5 с", "Н −½", "start", -0.5),
            ("Начало +0.5 с", "Н +½", "start", 0.5),
            ("Начало +1 с", "Н +1", "start", 1.0),
            ("Конец −1 с", "К −1", "end", -1.0),
            ("Конец −0.5 с", "К −½", "end", -0.5),
            ("Конец +0.5 с", "К +½", "end", 0.5),
            ("Конец +1 с", "К +1", "end", 1.0),
        )):
            button = QPushButton(full_text if columns == 2 else compact_text)
            button.setProperty("boundaryOrder", index)
            button.setProperty("boundaryFullText", full_text)
            button.setProperty("boundaryCompactText", compact_text)
            button.setToolTip(
                f"{full_text}. Проверит только сохранённые границы речи и сцены; повторный анализ не нужен."
            )
            button.clicked.connect(
                lambda _checked=False, cid=candidate_id, name=boundary, value=delta: self._adjust_candidate_boundary(cid, name, value)
            )
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(button, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        self.candidate_detail.layout().addWidget(controls)
        # The wide inspector is viewport-sized.  Let one terminal spacer own
        # any surplus so Qt keeps the explanatory copy and boundary controls
        # together at the top instead of spreading labels through the pane.
        self.candidate_detail.layout().addStretch()
        self._refresh_candidate_detail_geometry()
        self.review_inspector_scroll.verticalScrollBar().setValue(0)
        QTimer.singleShot(
            0, lambda: self.review_inspector_scroll.verticalScrollBar().setValue(0),
        )

    def _refresh_candidate_detail_geometry(self) -> None:
        """Keep inspector copy above its controls and scroll when necessary."""

        layout = self.candidate_detail.layout()
        if layout is None:
            return
        self.candidate_detail.setMinimumHeight(0)
        layout.invalidate()
        layout.activate()
        required_height = layout.totalHeightForWidth(max(1, self.candidate_detail.width()))
        if required_height < 0:
            required_height = layout.totalSizeHint().height()
        self.candidate_detail.setMinimumHeight(
            max(required_height, layout.totalMinimumSize().height())
        )
        self.candidate_detail.updateGeometry()
        self._queue_drafts_workspace_geometry()

    def _reflow_candidate_boundary_controls(self) -> None:
        """Keep the same inspector controls compact across review breakpoints."""

        if not hasattr(self, "candidate_detail"):
            return
        controls = self.candidate_detail.findChild(QWidget, "candidateBoundaryControls")
        if controls is None or not isinstance(controls.layout(), QGridLayout):
            return
        grid = controls.layout()
        buttons = sorted(
            controls.findChildren(QPushButton),
            key=lambda button: int(button.property("boundaryOrder") or 0),
        )
        if not buttons:
            return
        columns = 2 if self._compact_stage_layout else 4
        for button in buttons:
            grid.removeWidget(button)
        for index, button in enumerate(buttons):
            button.setText(str(
                button.property("boundaryFullText")
                if columns == 2
                else button.property("boundaryCompactText")
            ))
            grid.addWidget(button, index // columns, index % columns)
        for column in range(4):
            grid.setColumnStretch(column, 1 if column < columns else 0)
        controls.updateGeometry()
        self._refresh_candidate_detail_geometry()

    @staticmethod
    def _score_text(value: object) -> str:
        try:
            return f"{float(cast(Any, value)):.0f}/100"
        except (TypeError, ValueError):
            return "—"

    def _adjust_candidate_boundary(self, candidate_id: str, boundary: str, delta_seconds: float) -> None:
        if candidate_id:
            if self.project and self._derive_flow_step(self.project) == "drafts":
                self.viewmodel.revise_draft_boundary(candidate_id, boundary, delta_seconds)
            else:
                self.viewmodel.adjust_candidate_boundary(candidate_id, boundary, delta_seconds)

    def _append_draft_inspector(self, candidate_id: str, start: float, end: float) -> None:
        """Show compact per-draft values backed by candidate-owned overrides."""

        if not self.project:
            return
        override = self.project.candidate_creative_overrides.get(candidate_id, {})
        style_id = str(override.get("creative_style", self.project.settings.subtitle_style))
        caption_id = str(override.get("caption_preset_id", self.project.settings.caption_preset_id))
        crop_id = str(override.get("composition_strategy", self.project.settings.composition_strategy))
        broll = bool(override.get(
            "same_source_broll_allowed", self.project.settings.same_source_broll_allowed,
        ))
        style_labels = {value: label for label, value in _CREATIVE_STYLE_CHOICES}
        crop_labels = {
            "safe_auto": "Авто — сохранить важное",
            "center_crop": "По центру",
            "fit_blur_background": "С размытым фоном",
            "fit_solid_background": "С однотонным фоном",
            "top_crop": "Верхняя часть кадра",
        }
        caption = CAPTION_PRESET_DEFINITIONS.get(caption_id)
        panel = QFrame()
        panel.setObjectName("draftInspector")
        grid = QGridLayout(panel)
        grid.setContentsMargins(0, 8, 0, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(7)
        panel_heading = QLabel("Вид этого черновика")
        panel_heading.setObjectName("inspectorSectionTitle")
        grid.addWidget(panel_heading, 0, 0, 1, 2)
        candidate_note = QLabel(
            "Изменения относятся только к этому черновику и не запускают анализ."
        )
        candidate_note.setObjectName("muted")
        candidate_note.setWordWrap(True)
        grid.addWidget(candidate_note, 1, 0, 1, 2)
        rows = (
            ("Стиль", style_labels.get(style_id, style_id), "creative_style"),
            ("Субтитры", caption.label if caption else caption_id, "caption_preset_id"),
            ("Кадрирование", crop_labels.get(crop_id, crop_id), "composition_strategy"),
        )
        for row, (label, value, option) in enumerate(rows):
            name = QLabel(label)
            name.setObjectName("muted")
            name.setProperty("draftInspectorField", True)
            name.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            current = QLabel(value)
            current.setObjectName("draftInspectorValue")
            current.setWordWrap(True)
            current.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            change = QPushButton("Изменить")
            change.setObjectName(f"draftChange-{option}-{candidate_id}")
            change.setProperty("secondaryAction", True)
            change.setMinimumWidth(88)
            change.setToolTip(
                "Сохранит pending-изменение только для этого черновика; render не начнётся автоматически."
            )
            change.clicked.connect(
                lambda _checked=False, cid=candidate_id, key=option: self._edit_draft_option(cid, key)
            )
            label_row = 2 + row * 2
            grid.addWidget(name, label_row, 0)
            grid.addWidget(change, label_row, 1)
            grid.addWidget(current, label_row + 1, 0, 1, 2)
        broll_row = 2 + len(rows) * 2
        broll_toggle = QCheckBox("Дополнительные кадры из этого видео")
        broll_toggle.setObjectName("draftExtraShots")
        broll_toggle.setChecked(broll)
        broll_toggle.setToolTip(
            "По умолчанию выключено. Меняет только этот черновик и не запускает анализ заново."
        )
        broll_toggle.toggled.connect(
            lambda value, cid=candidate_id: self._set_draft_broll(cid, value)
        )
        grid.addWidget(broll_toggle, broll_row, 0, 1, 2)
        fragment_row = broll_row + 1
        grid.addWidget(QLabel("Фрагмент"), fragment_row, 0)
        grid.addWidget(QLabel(f"{format_seconds(start)}–{format_seconds(end)}"), fragment_row, 1)
        preview_projection = self._draft_preview_projection(
            self.project,
            candidate_id,
            has_preview=candidate_id in self._draft_preview_paths,
        )
        grid.addWidget(QLabel("Качество"), fragment_row + 1, 0)
        quality_value = QLabel(preview_projection.inspector_text)
        quality_value.setWordWrap(True)
        quality_value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        quality_value.setObjectName(preview_projection.inspector_state)
        quality_value.setProperty(
            "draftPreviewFreshness", "stale" if preview_projection.stale else "current",
        )
        grid.addWidget(quality_value, fragment_row + 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnMinimumWidth(1, 88)
        self.candidate_detail.layout().addWidget(panel)

    def _set_draft_broll(self, candidate_id: str, enabled: bool) -> None:
        if not self.project or self.viewmodel.active:
            return
        override = self.project.candidate_creative_overrides.get(candidate_id, {})
        current = bool(override.get(
            "same_source_broll_allowed", self.project.settings.same_source_broll_allowed,
        ))
        if current != enabled:
            self.viewmodel.revise_draft(
                candidate_id, same_source_broll_allowed=enabled,
            )

    def _edit_draft_option(self, candidate_id: str, option: str) -> None:
        if not self.project or self.viewmodel.active:
            return
        override = self.project.candidate_creative_overrides.get(candidate_id, {})
        choices: list[tuple[str, object]]
        title: str
        if option == "creative_style":
            title = "Стиль оформления"
            choices = [(label, value) for label, value in _CREATIVE_STYLE_CHOICES]
            current = override.get(option, self.project.settings.subtitle_style)
        elif option == "caption_preset_id":
            current = override.get(option, self.project.settings.caption_preset_id)
            picker = CaptionPresetPickerDialog(str(current), self)
            if picker.exec() != picker.DialogCode.Accepted:
                return
            value = picker.selected_preset_id
            if value and value != current:
                self.viewmodel.revise_draft(candidate_id, caption_preset_id=value)
            return
        elif option == "composition_strategy":
            title = "Кадрирование"
            choices = [
                ("Авто — сохранить важное", "safe_auto"),
                ("По центру", "center_crop"),
                ("С размытым фоном", "fit_blur_background"),
                ("С однотонным фоном", "fit_solid_background"),
                ("Верхняя часть кадра", "top_crop"),
            ]
            current = override.get(option, self.project.settings.composition_strategy)
        elif option == "same_source_broll_allowed":
            title = "Дополнительные кадры из этого видео"
            choices = [("Не использовать", False), ("Использовать", True)]
            current = override.get(option, self.project.settings.same_source_broll_allowed)
        else:
            return
        labels = [label for label, _value in choices]
        current_index = next(
            (index for index, (_label, value) in enumerate(choices) if value == current), 0,
        )
        selected, accepted = QInputDialog.getItem(
            self, title, "Выберите значение для этого черновика:",
            labels, current_index, False,
        )
        if not accepted:
            return
        value = next(value for label, value in choices if label == selected)
        if value == current:
            return
        self.viewmodel.revise_draft(candidate_id, **{option: value})

    def _refresh_active_candidate_detail(self, project: DesktopProject) -> None:
        self._reconcile_active_candidate_preview(project, previous_step=self._flow_step)

    def _reconcile_active_candidate_preview(self, project: DesktopProject, *, previous_step: str) -> None:
        """Keep the visible player bound to a current, durable candidate.

        Project changes happen for selection, approval and run-state updates.
        They must not restart an unchanged source interval. Pending overrides
        keep the previous immutable Preview visible until a replacement has
        been fully rendered and atomically published.
        """

        workflow_step = self._derive_flow_step(project)
        if workflow_step not in {"candidates", "drafts"}:
            return
        persisted_id = getattr(project, "active_preview_candidate_id", None)
        candidate_id = self._active_candidate_id or persisted_id
        candidate = self._review_candidates_by_id.get(candidate_id or "")
        if not isinstance(candidate, dict):
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
                    poster_cache_directory=project.directory / "preview-posters",
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
        if previous_binding != binding:
            if workflow_step == "drafts" and draft_path:
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
        if (
            candidate_id == self._active_candidate_id
            and self._active_preview_kind == "source-range"
        ):
            self.preview.show_bound_poster(path)

    def _ensure_project_thumbnail(self, project: DesktopProject) -> None:
        """Bind Source/Processing to one durable source-revision poster."""

        expected = project_poster_path(project).resolve(strict=False)
        persisted = Path(project.thumbnail_path).resolve(strict=False) if project.thumbnail_path else None
        if persisted == expected and expected.is_file():
            self._project_thumbnail_path = expected
            self._paint_project_thumbnail(self._project_thumbnail_path)
            return
        if not project_poster_has_input(project):
            self._project_thumbnail_path = None
            for label in self._project_thumbnail_labels:
                label.setText("Кадр появится после загрузки")
                label.setPixmap(QPixmap())
            return
        destination = self._project_thumbnail_loader.request(project)
        self._project_thumbnail_path = destination.resolve(strict=False)

    def _project_thumbnail_ready(self, project_id: str, path: str) -> None:
        if not self.project or self.project.project_id != project_id:
            return
        actual = Path(path).resolve(strict=False)
        if self._project_thumbnail_path != actual:
            return
        self._paint_project_thumbnail(actual)
        if self.project.thumbnail_path != str(actual):
            try:
                self.project = self.viewmodel.services.update_project_thumbnail(
                    self.project, actual,
                )
            except Exception:
                pass

    def _project_thumbnail_unavailable(self, project_id: str, path: str) -> None:
        if (
            not self.project or self.project.project_id != project_id
            or self._project_thumbnail_path != Path(path).resolve(strict=False)
        ):
            return
        for label in self._project_thumbnail_labels:
            label.setPixmap(QPixmap())
            label.setText("Кадр недоступен · видео можно открыть")

    def _paint_project_thumbnail(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        for label in self._project_thumbnail_labels:
            label.setText("")
            label.setPixmap(pixmap.scaled(
                max(1, label.width()), max(1, label.height()),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            ))

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

    def _runs_changed(self, runs: list[ProjectRun], *, refresh_project: bool = True) -> None:
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
        if self.project and refresh_project:
            self._update_final_results(self.project)
            self._update_candidate_review(self.project)
            self._update_next_step(self.project)

    def _update_content_summary(self, runs: list[ProjectRun]) -> None:
        reports: list[dict[str, Any]] = []
        if self.project:
            analysis_report = self._analysis_content_summary_report(self._analysis_artifact(self.project))
            if analysis_report is not None:
                reports.append(analysis_report)
        if not reports:
            for run in runs:
                projection = self.viewmodel.services.run_projection(run)
                if projection.content_summary_report is not None:
                    reports.append(projection.content_summary_report)
        for report in reports:
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

    @staticmethod
    def _analysis_content_summary_report(analysis: dict[str, Any]) -> dict[str, Any] | None:
        if not analysis:
            return None
        profile = analysis.get("content_profile", {})
        recommendation_root = analysis.get("recommendation", {})
        summary = analysis.get("summary", {})
        if not isinstance(profile, dict) or not isinstance(recommendation_root, dict):
            return None
        recommendation = recommendation_root.get("clip_count", {})
        coverage = recommendation_root.get("coverage", {})
        if not isinstance(recommendation, dict) or not isinstance(coverage, dict):
            return None
        available_chapters = coverage.get("available_chapters", [])
        selected_chapters = coverage.get("selected_chapters", [])
        understanding = {
            "enabled": True,
            "profile": {"detected_content_type": profile.get("detected_content_type", "не определён")},
            "content_map": {
                "chapters": [None] * len(available_chapters) if isinstance(available_chapters, list) else [],
            },
            "clip_count_recommendation": {
                "estimated_publishable_clip_range": recommendation.get("estimated_publishable_clip_range", {}),
                "estimated_story_count": recommendation.get(
                    "estimated_story_count",
                    summary.get("candidate_count", 0) if isinstance(summary, dict) else 0,
                ),
            },
            "coverage_map": {
                "selected_chapters": list(selected_chapters) if isinstance(selected_chapters, list) else [],
            },
        }
        report: dict[str, Any] = {"content_understanding": understanding}
        candidates = analysis.get("candidates", [])
        chosen = None
        if isinstance(candidates, list):
            chosen = next(
                (item for item in candidates if isinstance(item, dict) and item.get("selected_by_recommendation")),
                next((item for item in candidates if isinstance(item, dict) and item.get("recommended")), None),
            )
        if isinstance(chosen, dict):
            level = {
                "low": "weak", "medium": "moderate", "high": "strong", "very_high": "excellent",
            }.get(str(chosen.get("virality_level") or chosen.get("potential") or "").lower(), "moderate")
            report["virality"] = {"enabled": True}
            report["clip_intelligence"] = {"candidates": [{
                "selected": True,
                "virality": {
                    "viral_potential": {
                        "level": level,
                        "strongest_factors": [],
                        "confidence": {"warnings": chosen.get("warnings", [])},
                    },
                    "publishability": {},
                    "retention_profile": {},
                    "eligibility": {"status": chosen.get("publishability_status", "")},
                },
            }]}
        return report

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
        elif any(
            candidate_id in self._draftable_candidates_by_id
            for candidate_id in project.review_selected_candidate_ids
        ):
            self.next_step_text.setText("Моменты выбраны. Следующая безопасная операция — создать черновики; финальный render пока не начнётся.")
        elif self._all_candidates_by_id and not self._draftable_candidates_by_id:
            self.next_step_text.setText(
                f"Найдено {len(self._all_candidates_by_id)} моментов, но ни один пока не прошёл проверку качества"
            )
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
        blocked = self.viewmodel.blocked_by_other_project
        if active and self.project:
            latest = self._latest_run(self.project)
            run_kind = self.viewmodel.run.run_kind if self.viewmodel.run else (latest.run_kind if latest else RunKind.FULL)
            if run_kind in {RunKind.ANALYSIS, RunKind.FULL}:
                self._preload_moments_proxy(self.project)
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
            retry_label = None
            if self.project:
                latest = self._latest_run(self.project)
                if latest and latest.status in {"failed", "interrupted", "cancelled"}:
                    message = self._recovery_message(latest)
                    if (
                        latest.run_kind in {"analysis", "full"}
                        and not self.project.analysis_artifact_path
                    ):
                        retry_label = "Повторить поиск моментов"
            self.progress.set_finished(message, retry_label)
        structure_key = (
            snapshot.phase,
            snapshot.stage,
            blocked,
            self.project.project_id if self.project else None,
        )
        # Telemetry can reveal a wrapped long-stage warning without changing
        # the structural stage key. Publish that height before an early return.
        self._refresh_processing_geometry()
        if structure_key == self._processing_structure_key:
            # Elapsed/activity/progress telemetry owns only the progress
            # surface.  Persisted project/run projection is unchanged.
            return
        self._processing_structure_key = structure_key
        self._update_processing_stages(snapshot)
        self._refresh_processing_geometry()
        self.run_button.setDisabled(active or blocked)
        self.setup_start_button.setDisabled(active or blocked)
        has_draft_choice = bool(
            self.project
            and (
                any(
                    candidate_id in self._draftable_candidates_by_id
                    for candidate_id in self.project.review_selected_candidate_ids
                )
                or self._recommended_candidate_ids()
            )
        )
        self.draft_button.setDisabled(active or blocked or not has_draft_choice)
        self.view_all_button.setDisabled(active)
        selected_drafts_exist = bool(self.project and self.project.selected_candidate_ids) and all(
            Path(self.project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
            for candidate_id in self.project.selected_candidate_ids
        ) if self.project else False
        self.production_button.setDisabled(active or blocked or not selected_drafts_exist)
        for widget in (
            self.processing_mode, self.deep_analysis, self.platform, self.clip_count,
            self.audio_mode, self.composition_strategy, self.same_source_broll,
            self.subtitles, self.subtitle_style, self.cache,
            self.setup_editorial_intent, self.content_profile_preset,
            self.setup_processing_mode, self.setup_deep_analysis, self.setup_platform, self.setup_clip_count,
        ):
            widget.setDisabled(active)
        for buttons in self._setup_choice_buttons.values():
            for button in buttons.values():
                button.setDisabled(active)
        heavy_hint = self._other_project_job_hint() if blocked else ""
        for button in (self.run_button, self.setup_start_button):
            button.setToolTip(heavy_hint)
        for button in (self.draft_button, self.production_button):
            button.setToolTip(
                heavy_hint or str(button.property("responsiveFullText") or button.text())
            )
        if self.project:
            if active:
                latest = self._latest_run(self.project)
                run_kind = self.viewmodel.run.run_kind if self.viewmodel.run else (latest.run_kind if latest else RunKind.FULL)
                status_label = (
                    "Загружаем видео" if snapshot.stage == "download" else {
                        RunKind.ANALYSIS: "Ищем моменты",
                        RunKind.DRAFT: "Создаём черновики",
                        RunKind.SELECTED_RENDER: "Создаём ролики",
                        RunKind.RENDER_REVISION: "Создаём ролики",
                    }.get(run_kind, "Идёт обработка")
                )
                presentation = ProjectPresentation(
                    "download" if snapshot.stage == "download" else "processing",
                    status_label,
                    latest,
                    True,
                )
            else:
                presentation = self.viewmodel.services.presentation(
                    self.project,
                    snapshot=snapshot,
                    runs=self._runs_for_project(self.project),
                )
            self.status.setText(presentation.status_label)
            self._update_download_card(self.project)
            self._update_stage_context(self.project)
            self._apply_flow_visibility(self.project, presentation=presentation)

    def _other_project_job_hint(self) -> str:
        owner = self.viewmodel.active_project_name or "другом проекте"
        return (
            f"Сейчас идёт обработка в проекте «{owner}». Этот проект можно просматривать и настраивать, "
            "но второй тяжёлый запуск станет доступен после завершения текущего."
        )

    def _retry_processing(self) -> None:
        if not self.project:
            return
        latest = self._latest_run(self.project)
        if latest and latest.run_kind in {"analysis", "full"}:
            self.viewmodel.start_analysis()

    def _update_processing_stages(self, snapshot: ProcessingSnapshot) -> None:
        """Show stages for the current job, never a stale analysis template."""

        rows = self._processing_stage_plan(snapshot)
        names = tuple(name for name, _label in rows)
        if names != self._processing_stage_rows:
            while self._processing_stages_layout.count() > 1:
                item = self._processing_stages_layout.takeAt(1)
                if widget := item.widget():
                    widget.hide()
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

    def _refresh_processing_geometry(self) -> None:
        """Publish dynamic progress height before placing the stage rows."""

        progress_height = self.progress.minimumHeight()
        if progress_height != self._processing_progress_layout_height:
            # Recreate only the affected QLayoutItem. On Windows/Qt a hidden
            # page can otherwise retain the height cached before an optional
            # warning row appeared, letting the stage list paint underneath.
            self._processing_main_layout.removeWidget(self.progress)
            self._processing_main_layout.insertWidget(2, self.progress)
            self._processing_progress_layout_height = progress_height
        self._processing_main.setMinimumHeight(0)
        self._processing_main_layout.invalidate()
        self._processing_main_layout.activate()
        required = self._processing_main_layout.totalHeightForWidth(
            max(1, self._processing_main.width())
        )
        if required < 0:
            required = self._processing_main_layout.totalMinimumSize().height()
        self._processing_main.setMinimumHeight(max(
            required,
            self._processing_main_layout.totalMinimumSize().height(),
        ))
        self.processing_workspace.updateGeometry()
        QTimer.singleShot(0, self._activate_processing_layout)

    def _activate_processing_layout(self) -> None:
        self._processing_main_layout.invalidate()
        self._processing_main_layout.setGeometry(self._processing_main.rect())
        self._processing_main_layout.activate()

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
            ("transcribe", "Понимаем речь и структуру"),
            ("understand", "Учитываем содержание и события"),
            ("candidates", "Находим и оцениваем моменты"),
            ("save", "Сохраняем результаты"),
        )

    @staticmethod
    def _processing_stage_target(raw: str, run_kind: str, row_count: int) -> int:
        if row_count <= 1:
            return 0
        if any(token in raw for token in ("verify", "valid", "finaliz", "manifest", "save", "report")):
            return row_count - 1
        if run_kind == "draft":
            return 1 if any(token in raw for token in ("draft", "script", "subtitle", "compose", "render")) else 0
        if run_kind in {"selected_render", "render_revision"}:
            return 1 if any(token in raw for token in ("render", "production", "export", "encode", "subtitle", "compose")) else 0
        if any(token in raw for token in ("transcrib", "speech", "audio")):
            return min(1, row_count - 1)
        if any(token in raw for token in ("candidate", "ranking", "select", "shortlist", "scoring")):
            return min(3, row_count - 1)
        if any(token in raw for token in ("analy", "intelligence", "vision", "scene", "content", "semantic")):
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
            if widget := item.widget():
                # Detached dynamic inspector rows otherwise remain paintable
                # until the next event-loop deletion and can appear beneath a
                # newly inserted boundary-control grid in captured frames.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        for value in values:
            label = QLabel(); label.setObjectName("muted")
            # Candidate excerpts and error details can be long.  They are
            # explanatory copy, never a reason to widen the whole review
            # page or expose a horizontal scrollbar.
            make_label_shrinkable(label)
            set_responsive_text(label, value)
            layout.addWidget(label)

    def _update_estimate(self, project: DesktopProject) -> None:
        if project.analysis_artifact_path:
            self._replace_card_text(
                self.estimate,
                [self._saved_estimate_text(project.setup_state.last_estimate)],
            )
            return
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
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(break_long_tokens(error.user_message))
        box.setInformativeText(break_long_tokens(error.suggested_action))
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
