from __future__ import annotations

"""Shared visual selectors for the current creative and composition options.

The cards only present existing persisted IDs.  They deliberately have no
rendering or persistence dependency: the owning screen decides when a click
becomes a project setting or a draft-local pending override.
"""

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from app.creative_policy import CREATIVE_PRESET_DEFINITIONS
from app.gui.components.video_preview import VideoPreview
from app.settings_preview_assets import settings_preview_path


@dataclass(frozen=True, slots=True)
class VisualControlOption:
    option_id: str
    label: str
    description: str
    visual: str
    accent: str


CREATIVE_STYLE_OPTIONS = (
    VisualControlOption(
        "clean", "Clean", "Спокойная разговорная подача без лишних акцентов.",
        "ЧИСТО · РОВНО", "#7DD3FC",
    ),
    VisualControlOption(
        "dynamic", "Dynamic", "Более энергичный ритм и заметные акценты.",
        "ТЕМП · АКЦЕНТ", "#FFB36B",
    ),
    VisualControlOption(
        "documentary", "Educational", "Чёткая объясняющая подача для мысли в кадре.",
        "СМЫСЛ · ЯСНО", "#A7F3D0",
    ),
    VisualControlOption(
        "minimal", "Minimal Premium", "Минимум визуального шума, фокус на говорящем.",
        "ТИШЕ · ТОЧНЕЕ", "#D8B4FE",
    ),
)

# Fail fast if presentation gets ahead of the persisted production registry.
assert {item.option_id for item in CREATIVE_STYLE_OPTIONS} == set(CREATIVE_PRESET_DEFINITIONS)


class VisualControlCard(QFrame):
    activated = Signal(str)
    hovered = Signal(str)
    hover_left = Signal(str)

    def __init__(self, option: VisualControlOption, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.option = option
        self.setObjectName("visualControlCard")
        self.setProperty("controlOptionId", option.option_id)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(122)
        self.setAccessibleName(option.label)
        self.setToolTip(option.description)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.title = QLabel(option.label)
        self.title.setObjectName("visualControlCardTitle")
        self.badge = QLabel("Выбрать")
        self.badge.setObjectName("visualControlCardBadge")
        header.addWidget(self.title, 1)
        header.addWidget(self.badge)
        layout.addLayout(header)
        self.visual = QLabel(option.visual)
        self.visual.setObjectName("visualControlCardVisual")
        self.visual.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.visual)
        self.description = QLabel(option.description)
        self.description.setObjectName("visualControlCardDescription")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self._apply_style(False)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.setProperty("selectionTone", "#FF6846" if selected else "")
        self.badge.setText("Выбрано" if selected else "Выбрать")
        self._apply_style(selected)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.activated.emit(self.option.option_id)

    def enterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().enterEvent(event)
        self.hovered.emit(self.option.option_id)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().leaveEvent(event)
        self.hover_left.emit(self.option.option_id)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.activated.emit(self.option.option_id)
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_style(self, selected: bool) -> None:
        accent = self.option.accent
        self.setStyleSheet(
            f"QFrame#visualControlCard {{ background: {'#251A17' if selected else '#17191D'}; border: {2 if selected else 1}px solid {'#FF6846' if selected else '#3B4048'}; border-radius: 9px; }}"
            f"QFrame#visualControlCard:hover {{ border-color: {accent}; }}"
            "QLabel#visualControlCardTitle { color: #F1F4F8; font-size: 12px; font-weight: 700; background: transparent; border: 0; }"
            f"QLabel#visualControlCardBadge {{ color: {'#FF6846' if selected else accent}; font-size: 11px; font-weight: 700; background: transparent; border: 0; }}"
            f"QLabel#visualControlCardVisual {{ color: {accent}; background: #101216; border: 0; border-radius: 5px; font-size: 13px; font-weight: 700; padding: 5px; }}"
            "QLabel#visualControlCardDescription { color: #A7AFBC; background: transparent; border: 0; font-size: 11px; }"
        )


class VisualControlPicker(QWidget):
    option_selected = Signal(str)
    option_hovered = Signal(str)
    option_hover_cleared = Signal()

    def __init__(
        self, options: tuple[VisualControlOption, ...], *, columns: int, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cards: dict[str, VisualControlCard] = {}
        self.selected_option_id: str | None = None
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        for index, option in enumerate(options):
            card = VisualControlCard(option, self)
            card.activated.connect(self.choose)
            card.hovered.connect(self.option_hovered)
            card.hover_left.connect(lambda _option_id: self.option_hover_cleared.emit())
            layout.addWidget(card, index // columns, index % columns)
            self.cards[option.option_id] = card
        for column in range(columns):
            layout.setColumnStretch(column, 1)

    def choose(self, option_id: str) -> None:
        if option_id not in self.cards:
            return
        self.set_selected(option_id)
        self.option_selected.emit(option_id)

    def set_selected(self, option_id: str) -> None:
        if option_id not in self.cards:
            return
        self.selected_option_id = option_id
        for item_id, card in self.cards.items():
            card.set_selected(item_id == option_id)


class CreativeStylePicker(VisualControlPicker):
    def __init__(self, *, columns: int, parent: QWidget | None = None) -> None:
        super().__init__(CREATIVE_STYLE_OPTIONS, columns=columns, parent=parent)
        self.setObjectName("creativeStylePicker")


class CreativeStylePickerDialog(QDialog):
    """Draft-local style selection with bundled, production-rendered hover demos."""

    def __init__(
        self, current_style_id: str, caption_preset_id: str, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Стиль оформления")
        self.setModal(True)
        self.resize(1000, 590)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)
        title = QLabel("Стиль оформления")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        copy = QLabel(
            "Наведите на карточку, чтобы увидеть реальное demo. Выбор останется ожидающим до «Пересоздать черновик»."
        )
        copy.setObjectName("muted")
        copy.setWordWrap(True)
        layout.addWidget(copy)
        body = QHBoxLayout()
        body.setSpacing(14)
        self.picker = CreativeStylePicker(columns=2, parent=self)
        self.picker.set_selected(current_style_id)
        self.picker.option_hovered.connect(self._show_demo)
        self.picker.option_hover_cleared.connect(self._restore_selected_demo)
        self.picker.option_selected.connect(self._show_demo)
        body.addWidget(self.picker, 3)
        self._caption_preset_id = caption_preset_id
        self.demo_preview = VideoPreview()
        self.demo_preview.setObjectName("draftCreativeStyleDemo")
        self.demo_preview.set_vertical_frame_size(230, 409)
        self.demo_preview.set_frame_sink_output(True)
        self.demo_preview.controls_host.hide()
        self.demo_preview.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.demo_preview.player.mediaStatusChanged.connect(self._demo_media_status_changed)
        body.addWidget(self.demo_preview, 1, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(body)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.save_button = QPushButton("Сохранить выбор")
        self.save_button.setObjectName("primary")
        buttons.addButton(self.save_button, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel is not None:
            cancel.setText("Отмена")
        buttons.rejected.connect(self.reject)
        self.save_button.clicked.connect(self.accept)
        layout.addWidget(buttons)
        self.finished.connect(lambda _result: self.demo_preview.suspend())
        self._show_demo(current_style_id)

    @property
    def selected_style_id(self) -> str:
        return self.picker.selected_option_id or ""

    def _show_demo(self, style_id: str) -> None:
        path = settings_preview_path(style_id, self._caption_preset_id)
        if path is None:
            self.demo_preview.set_file(None, presentation="vertical", title="Демо недоступно")
            return
        if self.demo_preview.active_media_path == path:
            self.demo_preview._play()
            return
        self.demo_preview.set_file_when_ready(path, presentation="vertical", title="Демо оформления")

    def _restore_selected_demo(self) -> None:
        self._show_demo(self.selected_style_id)

    def _demo_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia}:
            self.demo_preview._play()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.demo_preview.suspend()
        super().closeEvent(event)
