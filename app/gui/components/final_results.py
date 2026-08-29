from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from app.gui.components.candidate_thumbnail import CandidateThumbnailLoader
from app.gui.components.video_preview import VideoPreview
from app.gui.responsive import make_label_shrinkable, set_responsive_text


@dataclass(frozen=True, slots=True)
class FinalOutput:
    """One validated, metadata-bound production output shown to a person."""

    result_id: str
    candidate_id: str
    path: Path
    title: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    status: str = "completed"
    run_id: str = ""


class _FinalOutputCard(QFrame):
    activated = Signal(str)

    def __init__(self, result_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result_id = result_id
        self.setObjectName("finalOutputCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.result_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.activated.emit(self.result_id)
            event.accept()
            return
        super().keyPressEvent(event)


class FinalResultsWorkspace(QWidget):
    """The final-result viewer; it never infers output paths from filenames."""

    output_selected = Signal(str)
    create_more_requested = Signal()
    drafts_requested = Signal()
    projects_requested = Signal()
    rerender_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("finalResultsWorkspace")
        self._outputs: list[FinalOutput] = []
        self._active_id: str | None = None
        self._signature: tuple[tuple[str, str], ...] = ()
        self._project_directory: Path | None = None
        self._warnings: list[str] = []
        self._cards: dict[str, _FinalOutputCard] = {}
        self._thumbnail_labels: dict[str, QLabel] = {}
        self._thumbnail_pixmaps: dict[str, QPixmap] = {}
        self._thumbnail_paths: dict[str, Path] = {}
        self._thumbnail_size = (72, 128)
        self._responsive_profile: str | None = None
        self._body_layout_mode: str | None = None
        self._bottom_actions_stacked: bool | None = None
        self._body_height_floor = 0
        self._body_geometry_refresh_pending = False
        self._observed_window: QWidget | None = None
        self._thumbnail_loader = CandidateThumbnailLoader(self)
        self._thumbnail_loader.thumbnail_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.thumbnail_unavailable.connect(self._thumbnail_unavailable)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        self.heading = QLabel("Готовые ролики")
        self.heading.setObjectName("title")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        titles.addWidget(self.heading)
        titles.addWidget(self.subtitle)
        header.addLayout(titles)
        header.addStretch()
        self._root_layout.addLayout(header)

        self.stepper = QFrame()
        self.stepper.setObjectName("resultsStepper")
        self._stepper_layout = QHBoxLayout(self.stepper)
        self._stepper_layout.setContentsMargins(14, 10, 14, 10)
        for index, label in enumerate(("Источник", "Настройка", "Обработка", "4  Результаты")):
            item = QLabel(("✓  " if index < 3 else "") + label)
            item.setObjectName("resultsStepActive" if index == 3 else "resultsStep")
            self._stepper_layout.addWidget(item)
            if index < 3:
                divider = QLabel("—")
                divider.setObjectName("muted")
                self._stepper_layout.addWidget(divider)
        self._stepper_layout.addStretch()
        self._root_layout.addWidget(self.stepper)
        # ProjectScreen owns the one global four-step chrome.  Retain this
        # widget for compatibility with older integrations, but do not render
        # a second, competing stepper inside Final Results.
        self.stepper.hide()

        self.summary = QFrame()
        self.summary.setObjectName("finalSummary")
        self._summary_layout = QHBoxLayout(self.summary)
        self._summary_layout.setContentsMargins(18, 12, 18, 12)
        self.summary_values: dict[str, QLabel] = {}
        self._summary_captions: list[QLabel] = []
        for key, caption in (
            ("count", "Роликов создано"), ("duration", "Общая длительность"),
            ("format", "Формат"), ("size", "Общий размер"),
        ):
            column = QVBoxLayout()
            value = QLabel("—")
            value.setObjectName("finalSummaryValue")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption_label = QLabel(caption)
            caption_label.setObjectName("muted")
            caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption_label.setWordWrap(True)
            column.addWidget(value)
            column.addWidget(caption_label)
            self._summary_layout.addLayout(column, 1)
            self.summary_values[key] = value
            self._summary_captions.append(caption_label)
        self._root_layout.addWidget(self.summary)

        self._body_host = QWidget()
        # The outer project view can scroll the screen, while each dense area
        # below keeps its own scrolling.  This avoids a very tall final page
        # when a project has many outputs or a lengthy QualityReport warning.
        self._body_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._body_layout = QGridLayout(self._body_host)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(14)

        self._list_panel = QFrame()
        self._list_panel.setObjectName("finalOutputList")
        self._list_panel.setMinimumWidth(300)
        self._list_panel.setMaximumWidth(360)
        self._list_layout = QVBoxLayout(self._list_panel)
        self._list_layout.setContentsMargins(10, 10, 10, 10)
        self._list_heading = QLabel("Готовые ролики")
        self._list_heading.setStyleSheet("font-weight: 600;")
        self._list_layout.addWidget(self._list_heading)
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_host = QWidget()
        self.list_items = QVBoxLayout(self.list_host)
        self.list_items.setContentsMargins(0, 0, 0, 0)
        self.list_items.setSpacing(8)
        self.list_items.addStretch()
        self.list_scroll.setWidget(self.list_host)
        self._list_layout.addWidget(self.list_scroll, 1)
        self.create_more_button = QPushButton("＋  Создать ещё ролики")
        self.create_more_button.setObjectName("secondaryAction")
        self.create_more_button.setToolTip("Вернуться к моментам и выбрать другие ролики без повторного анализа.")
        self.create_more_button.clicked.connect(self.create_more_requested)
        self.create_more_button.hide()

        self.preview = VideoPreview()
        self.preview.set_vertical_frame_size(247, 440)
        self.preview.open_button.hide()
        self.preview.geometry_requirement_changed.connect(self._schedule_body_geometry_refresh)

        self._info_panel = QFrame()
        self._info_panel.setObjectName("finalInfo")
        self._info_panel.setMinimumWidth(270)
        self._info_panel.setMaximumWidth(350)
        self._info_layout = QVBoxLayout(self._info_panel)
        self._info_layout.setContentsMargins(16, 16, 16, 16)
        self._info_heading = QLabel("Информация о ролике")
        self._info_heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._info_layout.addWidget(self._info_heading)
        self.info_scroll = QScrollArea()
        self.info_scroll.setWidgetResizable(True)
        self.info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_host = QWidget()
        self._info_content_layout = QVBoxLayout(self._info_host)
        self._info_content_layout.setContentsMargins(0, 0, 0, 0)
        self._info_content_layout.setSpacing(10)
        self.info_grid = QGridLayout()
        self.info_grid.setHorizontalSpacing(14)
        self.info_grid.setVerticalSpacing(10)
        self.info_values: dict[str, QLabel] = {}
        for row, (key, label) in enumerate((
            ("source", "Исходный момент"), ("duration", "Длительность"),
            ("format", "Формат"), ("size", "Размер файла"), ("created", "Создан"),
        )):
            name = QLabel(label)
            name.setObjectName("muted")
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            make_label_shrinkable(value)
            self.info_grid.addWidget(name, row, 0)
            self.info_grid.addWidget(value, row, 1)
            self.info_values[key] = value
        self._info_content_layout.addLayout(self.info_grid)

        self.warning_box = QFrame()
        self.warning_box.setObjectName("finalWarnings")
        warning_layout = QVBoxLayout(self.warning_box)
        warning_layout.setContentsMargins(10, 10, 10, 10)
        warning_heading = QLabel("Готово с предупреждениями")
        warning_heading.setObjectName("warning")
        warning_heading.setWordWrap(True)
        warning_heading.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        warning_layout.addWidget(warning_heading)
        warning_scroll = QScrollArea()
        warning_scroll.setWidgetResizable(True)
        warning_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        warning_scroll.setFixedHeight(108)
        self._warning_scroll = warning_scroll
        warning_host = QWidget()
        warning_host_layout = QVBoxLayout(warning_host)
        warning_host_layout.setContentsMargins(0, 0, 0, 0)
        self.warning_text = QLabel()
        self.warning_text.setWordWrap(True)
        self.warning_text.setObjectName("muted")
        self.warning_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        warning_host_layout.addWidget(self.warning_text)
        warning_host_layout.addStretch()
        warning_scroll.setWidget(warning_host)
        warning_layout.addWidget(warning_scroll)
        self.warning_box.hide()
        self._info_content_layout.addWidget(self.warning_box)
        self._info_content_layout.addStretch()
        self.info_scroll.setWidget(self._info_host)
        self._actions_heading = QLabel("Действия")
        self._actions_heading.setStyleSheet("font-weight: 600;")
        self._info_layout.addWidget(self._actions_heading)
        self.open_video_button = QPushButton("▷  Открыть видео")
        # Opening the selected, metadata-bound output is the focused final
        # action.  Folder navigation and returning to projects stay quiet.
        self.open_video_button.setObjectName("secondaryAction")
        self.open_video_button.setProperty("finalAction", True)
        self.open_video_button.clicked.connect(self._open_active_video)
        self.show_folder_button = QPushButton("▢  Показать в папке")
        self.show_folder_button.setObjectName("showFinalFolder")
        self.show_folder_button.setProperty("finalAction", True)
        self.show_folder_button.clicked.connect(self._show_active_folder)
        self.rerender_button = QPushButton("↻  Собрать заново")
        self.rerender_button.setObjectName("secondaryAction")
        self.rerender_button.setProperty("finalAction", True)
        self.rerender_button.setToolTip("Повторная сборка с текущим оформлением без нового поиска моментов.")
        self.rerender_button.clicked.connect(self._request_rerender)
        self._info_layout.addWidget(self.open_video_button)
        self._info_layout.addWidget(self.show_folder_button)
        self._info_layout.addWidget(self.rerender_button)
        # Keep the selected-output actions in the initial viewport.  Metadata
        # and potentially long quality warnings remain independently scrollable
        # below, so a warning can never push the primary actions off-screen.
        self._info_layout.addWidget(self.info_scroll, 1)
        self._place_body_panels("standard")
        self._root_layout.addWidget(self._body_host)

        self._bottom = QFrame()
        self._bottom.setObjectName("finalActionBar")
        self._bottom_layout = QHBoxLayout(self._bottom)
        self._bottom_layout.setContentsMargins(12, 10, 12, 10)
        self.project_folder_button = QPushButton("▢  Открыть папку проекта")
        self.project_folder_button.clicked.connect(self._open_project_folder)
        self.drafts_button = QPushButton("←  Назад к черновикам")
        self.drafts_button.setObjectName("secondaryAction")
        self.drafts_button.clicked.connect(self.drafts_requested)
        self.create_more_primary_button = QPushButton("Создать ещё ролики  →")
        self.create_more_primary_button.setObjectName("primary")
        self.create_more_primary_button.clicked.connect(self.create_more_requested)
        self.projects_button = QPushButton("К проектам")
        self.projects_button.setObjectName("secondaryAction")
        self.projects_button.clicked.connect(self.projects_requested)
        self._bottom_layout.addWidget(self.drafts_button)
        self._bottom_layout.addStretch()
        self._bottom_layout.addWidget(self.create_more_primary_button)
        # The selected-output inspector already owns "Показать в папке".
        # Keep these compatibility actions callable but avoid duplicating them
        # in the reference bottom bar or forcing a desktop minimum width.
        self.project_folder_button.hide()
        self.projects_button.hide()
        self._root_layout.addWidget(self._bottom)
        self._apply_responsive_layout(force=True)

    @property
    def active_output_id(self) -> str | None:
        return self._active_id

    @property
    def action_bar(self) -> QFrame:
        """Expose the one action bar so a host can keep it sticky.

        The standalone workspace owns the bar by default.  ProjectScreen
        moves this same widget into its shared sticky-action slot; no button
        or signal path is duplicated.
        """

        return self._bottom

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._watch_window()
        self._apply_responsive_layout()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self._observed_window and event.type() == QEvent.Type.Resize:
            # The workspace itself is inside a scroll area and may retain its
            # content height while the application window becomes shorter.
            # Re-evaluate after Qt has updated that viewport's geometry.
            QTimer.singleShot(0, self._apply_responsive_layout)
        return super().eventFilter(watched, event)

    def _watch_window(self) -> None:
        window = self.window()
        if window is self._observed_window:
            return
        if self._observed_window is not None:
            self._observed_window.removeEventFilter(self)
        self._observed_window = window
        if window is not self:
            window.installEventFilter(self)

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        """Fit the three-column result viewer into ordinary laptop windows.

        The surrounding project screen is scrollable, but result cards and
        metadata need to remain independently usable when its viewport is
        short.  Only widget geometry changes here: the existing VideoPreview,
        QMediaPlayer and QAudioOutput remain the same persistent instances.
        """

        window_height = self.window().height() if self.window() else 0
        width = self.width()
        dense = width < 1020 or (0 < window_height <= 720)
        compact = dense or width < 1320 or (0 < window_height <= 840)
        profile = "dense" if dense else "compact" if compact else "standard"
        # Panel composition follows the actual workspace width.  Height only
        # tunes density; a wide-but-short window can use columns and let the
        # surrounding project viewport provide vertical scrolling.
        body_mode = "stacked" if width < 700 else "two_row" if width < 900 else "columns"
        # Stack early enough that the three reference actions never establish
        # an oversized minimum width before the host receives its final size.
        bottom_actions_stacked = width < 720
        if (
            not force
            and profile == self._responsive_profile
            and body_mode == self._body_layout_mode
            and bottom_actions_stacked == self._bottom_actions_stacked
        ):
            return
        self._responsive_profile = profile
        self._body_layout_mode = body_mode
        self._bottom_actions_stacked = bottom_actions_stacked

        # The inspector is intentionally narrow beside the 9:16 player.
        # Keep actions readable without allowing their full desktop copy to
        # establish a hidden horizontal minimum at laptop widths.
        if compact:
            self.open_video_button.setText("▶  Открыть")
            self.show_folder_button.setText("▢  Папка")
            self.rerender_button.setText("↻  Пересоздать")
        else:
            self.open_video_button.setText("▶  Открыть видео")
            self.show_folder_button.setText("▢  Показать в папке")
            self.rerender_button.setText("↻  Собрать заново")

        if profile == "dense":
            list_width = (260, 300)
            info_width = (250, 285)
            frame_size = (220, 391)
            body_height = 930 if body_mode == "stacked" else 500
            thumbnail_size = (52, 92)
            spacing = 8
            panel_margins = 9
            summary_margins = (10, 8)
            warning_height = 56
            self.heading.setStyleSheet("font-size: 22px; font-weight: 700;")
            summary_value_style = "font-size: 17px; font-weight: 700;"
        elif profile == "compact":
            list_width = (280, 320)
            info_width = (270, 320)
            frame_size = (252, 448)
            body_height = 560
            thumbnail_size = (60, 106)
            spacing = 10
            panel_margins = 11
            summary_margins = (14, 10)
            warning_height = 68
            self.heading.setStyleSheet("font-size: 24px; font-weight: 700;")
            summary_value_style = "font-size: 19px; font-weight: 700;"
        else:
            # 356 leaves the full-width “Create more” CTA at least its themed
            # 320px minimum after the panel's 16px side margins and frame.
            list_width = (300, 350)
            # Keep the metadata grid's two readable columns inside its own
            # viewport.  The old 420 px cap left a five-pixel hidden range at
            # full-HD once the scroll frame and its vertical bar were present.
            info_width = (300, 350)
            frame_size = (288, 512)
            body_height = 624
            thumbnail_size = (72, 128)
            spacing = 14
            panel_margins = 16
            summary_margins = (18, 12)
            warning_height = 84
            self.heading.setStyleSheet("")
            summary_value_style = ""

        self._thumbnail_size = thumbnail_size
        self._root_layout.setSpacing(spacing)
        self._body_layout.setSpacing(spacing)
        self._bottom_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if bottom_actions_stacked
            else QBoxLayout.Direction.LeftToRight
        )
        self._bottom_layout.setSpacing(8 if bottom_actions_stacked else 6)
        self._place_body_panels(body_mode)
        self._stepper_layout.setContentsMargins(
            summary_margins[0], summary_margins[1] - 2,
            summary_margins[0], summary_margins[1] - 2,
        )
        self._summary_layout.setContentsMargins(
            summary_margins[0], summary_margins[1],
            summary_margins[0], summary_margins[1],
        )
        self._list_layout.setContentsMargins(panel_margins, panel_margins, panel_margins, panel_margins)
        self._info_layout.setContentsMargins(panel_margins, panel_margins, panel_margins, panel_margins)
        self._info_content_layout.setSpacing(max(6, spacing - 2))
        self.info_grid.setHorizontalSpacing(max(8, spacing))
        self.info_grid.setVerticalSpacing(max(6, spacing - 2))
        self._warning_scroll.setFixedHeight(warning_height)
        if body_mode in {"columns", "two_row"}:
            self._list_panel.setMinimumWidth(list_width[0])
            self._list_panel.setMaximumWidth(list_width[1])
        else:
            self._list_panel.setMinimumWidth(0)
            self._list_panel.setMaximumWidth(16_777_215)
        if body_mode == "columns":
            self._info_panel.setMinimumWidth(info_width[0])
            self._info_panel.setMaximumWidth(info_width[1])
        else:
            self._info_panel.setMinimumWidth(0)
            self._info_panel.setMaximumWidth(16_777_215)
        # This is a profile floor, not a hard clipping boundary.  VideoPreview
        # owns wrapped titles/status copy and advertises their current
        # height-for-width; the grid is therefore free to grow into the outer
        # project's vertical scroll area when that copy needs more room.
        self._body_height_floor = body_height
        self._body_host.setMinimumHeight(body_height)
        self.preview.set_vertical_frame_size(*frame_size)
        for value in self.summary_values.values():
            value.setStyleSheet(summary_value_style)
        for caption in self._summary_captions:
            caption.setStyleSheet("font-size: 12px;" if compact else "")
        self._resize_output_cards()
        self._body_layout.invalidate()
        self._body_host.updateGeometry()
        self.updateGeometry()
        self._schedule_body_geometry_refresh()

    def _schedule_body_geometry_refresh(self) -> None:
        """Coalesce child HFW changes before recomputing the grid floor."""

        if self._body_geometry_refresh_pending:
            return
        self._body_geometry_refresh_pending = True
        QTimer.singleShot(0, self._refresh_body_geometry)

    def _refresh_body_geometry(self) -> None:
        """Let the outer page scroll instead of clipping overlapping panels."""

        self._body_geometry_refresh_pending = False
        self._body_layout.invalidate()
        required_height = self._body_layout.totalMinimumSize().height()
        target_height = max(self._body_height_floor, required_height)
        if self._body_host.minimumHeight() != target_height:
            self._body_host.setMinimumHeight(target_height)
        self._body_host.updateGeometry()
        self._root_layout.invalidate()
        self.updateGeometry()

    def _place_body_panels(self, mode: str) -> None:
        """Reflow result panels before their internal scroll views must clip.

        ``VideoPreview`` keeps its media objects throughout; only its owning
        layout cell changes.  A short desktop gets list + player above a
        full-width inspector, while a scaled/narrow desktop stacks all panels.
        """

        for widget in (self._list_panel, self.preview, self._info_panel):
            self._body_layout.removeWidget(widget)
        for index in range(3):
            self._body_layout.setColumnStretch(index, 0)
            self._body_layout.setRowStretch(index, 0)
        if mode == "columns":
            self._body_layout.addWidget(self._list_panel, 0, 0)
            self._body_layout.addWidget(
                self.preview, 0, 1,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            )
            self._body_layout.addWidget(self._info_panel, 0, 2)
            self._body_layout.setColumnStretch(0, 1)
            self._body_layout.setColumnStretch(1, 2)
            self._body_layout.setColumnStretch(2, 1)
            return
        if mode == "two_row":
            self._body_layout.addWidget(self._list_panel, 0, 0)
            self._body_layout.addWidget(
                self.preview, 0, 1,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            )
            self._body_layout.addWidget(self._info_panel, 1, 0, 1, 2)
            self._body_layout.setColumnStretch(0, 1)
            self._body_layout.setColumnStretch(1, 2)
            self._body_layout.setRowStretch(0, 2)
            self._body_layout.setRowStretch(1, 1)
            return
        self._body_layout.addWidget(self._list_panel, 0, 0)
        self._body_layout.addWidget(
            self.preview, 1, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
        )
        self._body_layout.addWidget(self._info_panel, 2, 0)
        self._body_layout.setRowStretch(0, 1)
        self._body_layout.setRowStretch(1, 2)
        self._body_layout.setRowStretch(2, 1)

    def _resize_output_cards(self) -> None:
        for result_id, card in self._cards.items():
            layout = card.layout()
            if layout is not None:
                margin = 7 if self._responsive_profile == "dense" else 8 if self._responsive_profile == "compact" else 9
                layout.setContentsMargins(margin, margin, margin, margin)
            thumbnail = self._thumbnail_labels.get(result_id)
            if thumbnail is None:
                continue
            thumbnail.setFixedSize(*self._thumbnail_size)
            pixmap = self._thumbnail_pixmaps.get(result_id)
            if pixmap is not None:
                self._set_thumbnail_pixmap(thumbnail, pixmap)

    @staticmethod
    def _set_thumbnail_pixmap(label: QLabel, pixmap: QPixmap) -> None:
        label.setText("")
        label.setPixmap(pixmap.scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def set_results(
        self, outputs: list[FinalOutput], *, selected_id: str | None, project_directory: Path,
        warnings: list[str] | None = None,
    ) -> None:
        signature = tuple((output.result_id, str(output.path)) for output in outputs)
        changed = signature != self._signature
        self._outputs = list(outputs)
        self._project_directory = project_directory
        self._warnings = list(warnings or [])
        self.subtitle.setText(f"{len(outputs)} {self._plural(len(outputs), 'ролик готов к публикации', 'ролика готовы к публикации', 'роликов готовы к публикации')}")
        self._update_summary()
        if changed:
            self._signature = signature
            self._rebuild_list()
        target = selected_id if any(output.result_id == selected_id for output in outputs) else (outputs[0].result_id if outputs else None)
        if target and (changed or target != self._active_id):
            self._activate(target, emit=False, warnings=self._warnings)
        elif target:
            self._render_details(self._output_for(target), self._warnings)
            self._mark_active()

    def _rebuild_list(self) -> None:
        while self.list_items.count() > 1:
            item = self.list_items.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards = {}
        self._thumbnail_labels = {}
        self._thumbnail_pixmaps = {}
        self._thumbnail_paths = {}
        for number, output in enumerate(self._outputs, start=1):
            card = _FinalOutputCard(output.result_id)
            card.activated.connect(lambda result_id: self._activate(result_id, emit=True, warnings=self._warnings))
            layout = QHBoxLayout(card)
            layout.setContentsMargins(9, 9, 9, 9)
            thumbnail = QLabel("Кадр…")
            thumbnail.setObjectName("finalThumbnail")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setFixedSize(*self._thumbnail_size)
            thumbnail.setWordWrap(True)
            thumbnail.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            layout.addWidget(thumbnail)
            self._thumbnail_labels[output.result_id] = thumbnail
            text = QVBoxLayout()
            title = QLabel()
            title.setStyleSheet("font-weight: 600;")
            make_label_shrinkable(title)
            set_responsive_text(title, f"{number}. {output.title}")
            state = QLabel("● Готово" if output.status == "completed" else "● Готово с предупреждениями")
            state.setObjectName("finalReady" if output.status == "completed" else "warning")
            make_label_shrinkable(state)
            detail = QLabel()
            detail.setObjectName("muted")
            make_label_shrinkable(detail)
            set_responsive_text(
                detail,
                f"{self._seconds(output.duration_seconds)}  ·  {self._size(self._file_size(output.path))}",
            )
            text.addWidget(title)
            text.addWidget(state)
            text.addWidget(detail)
            text.addStretch()
            layout.addLayout(text, 1)
            self.list_items.insertWidget(self.list_items.count() - 1, card)
            self._cards[output.result_id] = card
            if self._project_directory and output.path.is_file():
                thumbnail_path = self._thumbnail_loader.request(
                    cache_directory=self._project_directory / "final-thumbnails",
                    analysis_id="final-results",
                    candidate_id=output.result_id,
                    source_path=output.path,
                    timestamp_seconds=0.05,
                )
                self._thumbnail_paths[output.result_id] = thumbnail_path

    def _activate(self, result_id: str, *, emit: bool, warnings: list[str]) -> None:
        output = self._output_for(result_id)
        if output is None:
            return
        self._active_id = output.result_id
        self._render_details(output, warnings)
        self._mark_active()
        # Update card/details synchronously, then let VideoPreview queue the
        # backend source handoff.  A slow multimedia backend must never delay
        # the visible selection change.
        self.preview.show_final(
            output.path, output.title,
            poster_cache_directory=(
                self._project_directory / "preview-posters"
                if self._project_directory else None
            ),
        )
        poster_path = self._thumbnail_paths.get(output.result_id)
        if poster_path is not None and poster_path.is_file():
            self.preview.show_bound_poster(poster_path)
        if emit:
            self.output_selected.emit(output.result_id)

    def _render_details(self, output: FinalOutput | None, warnings: list[str]) -> None:
        if output is None:
            return
        source = "Не указан"
        if output.source_start_seconds is not None and output.source_end_seconds is not None:
            source = f"{self._seconds(output.source_start_seconds)} – {self._seconds(output.source_end_seconds)}"
        size = self._file_size(output.path)
        dimensions = (
            f"{output.width}×{output.height} (9:16)"
            if output.width and output.height else "Не удалось определить"
        )
        created = "—"
        try:
            created = datetime.fromtimestamp(output.path.stat().st_mtime).strftime("%d.%m.%Y, %H:%M")
        except OSError:
            pass
        values = {
            "source": source,
            "duration": self._seconds(output.duration_seconds),
            "format": dimensions,
            "size": self._size(size),
            "created": created,
        }
        for key, value in values.items():
            set_responsive_text(self.info_values[key], value)
        all_warnings = list(dict.fromkeys(warning for warning in warnings if warning.strip()))
        if output.status == "warning" and not all_warnings:
            all_warnings.append("Файл создан с предупреждением из отчёта о ролике.")
        set_responsive_text(
            self.warning_text,
            "\n".join(f"• {warning}" for warning in all_warnings),
        )
        self.warning_box.setVisible(bool(all_warnings))
        available = output.path.is_file()
        self.open_video_button.setEnabled(available)
        self.show_folder_button.setEnabled(available)
        self.rerender_button.setEnabled(bool(output.run_id))

    def _mark_active(self) -> None:
        for result_id, card in self._cards.items():
            card.setProperty("activeFinalOutput", result_id == self._active_id)
            card.style().unpolish(card)
            card.style().polish(card)

    def _thumbnail_ready(self, result_id: str, path: str) -> None:
        label = self._thumbnail_labels.get(result_id)
        if label is None:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self._thumbnail_pixmaps[result_id] = pixmap
        self._set_thumbnail_pixmap(label, pixmap)
        if result_id == self._active_id:
            self.preview.show_bound_poster(path)

    def _thumbnail_unavailable(self, result_id: str) -> None:
        label = self._thumbnail_labels.get(result_id)
        if label is not None:
            label.setText("Нет кадра")

    def _open_active_video(self) -> None:
        output = self._output_for(self._active_id)
        if output and output.path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.path)))

    def _show_active_folder(self) -> None:
        output = self._output_for(self._active_id)
        if output and output.path.parent.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.path.parent)))

    def _request_rerender(self) -> None:
        output = self._output_for(self._active_id)
        if output and output.run_id:
            self.rerender_requested.emit(output.run_id)

    def _open_project_folder(self) -> None:
        if self._project_directory and self._project_directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project_directory)))

    def _output_for(self, result_id: str | None) -> FinalOutput | None:
        return next((output for output in self._outputs if output.result_id == result_id), None)

    def _update_summary(self) -> None:
        durations = [value.duration_seconds for value in self._outputs if value.duration_seconds is not None]
        sizes = [self._file_size(value.path) for value in self._outputs]
        formats = {(value.width, value.height) for value in self._outputs if value.width and value.height}
        format_text = "—"
        if len(formats) == 1:
            width, height = next(iter(formats))
            format_text = f"{width}×{height}"
        self.summary_values["count"].setText(str(len(self._outputs)))
        self.summary_values["duration"].setText(self._seconds(sum(durations)) if durations else "—")
        self.summary_values["format"].setText(format_text)
        self.summary_values["size"].setText(self._size(sum(sizes)))

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _seconds(value: float | None) -> str:
        if value is None:
            return "—"
        whole = max(0, round(value))
        return f"{whole // 60:02d}:{whole % 60:02d}"

    @staticmethod
    def _size(value: int) -> str:
        if value < 1024 * 1024:
            return f"{value / 1024:.0f} КБ"
        if value < 1024 * 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} МБ"
        return f"{value / (1024 * 1024 * 1024):.2f} ГБ"

    @staticmethod
    def _plural(value: int, one: str, few: str, many: str) -> str:
        if value % 10 == 1 and value % 100 != 11:
            return one
        if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
            return few
        return many
