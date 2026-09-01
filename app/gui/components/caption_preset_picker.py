from __future__ import annotations

"""Visual, keyboard-accessible selection for the seven caption presets.

The picker deliberately exposes only the existing, versioned preset identities.
It is presentation-only: callers continue to own persistence and rendering.
"""

from functools import lru_cache

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase, QKeyEvent
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from app.caption_presets import CAPTION_PRESET_DEFINITIONS, CaptionPresetDefinition
from app.font_assets import FONT_ASSET_DEFINITIONS, bundled_font_asset_path
from app.gui.components.video_preview import VideoPreview
from app.settings_preview_assets import settings_preview_path


@lru_cache(maxsize=None)
def _caption_font(preset_id: str) -> QFont:
    """Load the exact bundled face used by Preview and Final for a card sample."""

    preset = CAPTION_PRESET_DEFINITIONS[preset_id]  # type: ignore[index]
    asset = FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id]
    path = bundled_font_asset_path(asset)
    if path.is_file():
        QFontDatabase.addApplicationFont(str(path))
    font = QFont(asset.render_family)
    font.setWeight(
        QFont.Weight.Bold if asset.weight_class >= 600
        else QFont.Weight.Light if asset.weight_class <= 300
        else QFont.Weight.Normal
    )
    font.setPointSize(max(10, round(8 + preset.font_size_ratio * 120)))
    return font


def _sample_text(preset: CaptionPresetDefinition) -> str:
    """Show the same primary/highlight relationship as the compiled caption."""

    if preset.display_mode == "single_spoken_word":
        return f'<span style="color:{preset.highlight_color}">СИЛЬНАЯ</span>'
    if preset.uppercase_emphasis:
        return (
            f'<span style="color:{preset.text_color}">СИЛЬНАЯ</span><br>'
            f'<span style="color:{preset.highlight_color}">МЫСЛЬ</span>'
        )
    if preset.motion_profile_id == "semantic_karaoke":
        return (
            f'<span style="color:{preset.text_color}">Очень важная </span>'
            f'<span style="color:{preset.highlight_color}">мысль</span>'
        )
    if preset.semantic_bold:
        return (
            f'<span style="color:{preset.text_color}">Проверяйте</span><br>'
            f'<b style="color:{preset.highlight_color}">факты</b>'
        )
    return f'<span style="color:{preset.text_color}">Сильная мысль</span>'


class CaptionPresetCard(QFrame):
    """One visual preset choice with a real bundled-font sample."""

    activated = Signal(str)
    hovered = Signal(str)
    hover_left = Signal(str)

    def __init__(self, preset: CaptionPresetDefinition, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preset = preset
        self.setObjectName("captionPresetCard")
        self.setProperty("captionPresetId", preset.preset_id)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(118)
        asset = FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id]
        self.setAccessibleName(f"Стиль субтитров: {preset.label}")
        self.setToolTip(f"{preset.label} · встроенный шрифт {asset.family} ({asset.file_name})")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 10, 11, 9)
        layout.setSpacing(6)
        header = QWidget()
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setHorizontalSpacing(6)
        self.title = QLabel(preset.label)
        self.title.setObjectName("captionPresetCardTitle")
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.badge = QLabel("Выбрать")
        self.badge.setObjectName("captionPresetCardBadge")
        header_layout.addWidget(self.title, 0, 0)
        header_layout.addWidget(self.badge, 0, 1, Qt.AlignmentFlag.AlignRight)
        header_layout.setColumnStretch(0, 1)
        layout.addWidget(header)

        self.sample_surface = QFrame()
        self.sample_surface.setObjectName("captionPresetSample")
        sample_layout = QVBoxLayout(self.sample_surface)
        sample_layout.setContentsMargins(9, 7, 9, 7)
        self.sample = QLabel(_sample_text(preset))
        self.sample.setObjectName("captionPresetSampleText")
        self.sample.setTextFormat(Qt.TextFormat.RichText)
        self.sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample.setWordWrap(True)
        self.sample.setFont(_caption_font(preset.preset_id))
        sample_layout.addWidget(self.sample)
        layout.addWidget(self.sample_surface, 1)
        self._apply_style(selected=False)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.setProperty("selectionTone", "#FF6846" if selected else "")
        self.badge.setText("Выбрано" if selected else "Выбрать")
        self._apply_style(selected=selected)

    def activate(self) -> None:
        self.activated.emit(self.preset.preset_id)

    def enterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().enterEvent(event)
        self.hovered.emit(self.preset.preset_id)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().leaveEvent(event)
        self.hover_left.emit(self.preset.preset_id)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.activate()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.activate()
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_style(self, *, selected: bool) -> None:
        preset = self.preset
        accent = preset.highlight_color
        background = "#0D0F12" if preset.background_mode == "opaque_box" else "#17191D"
        selection = "#FF6846"
        border = selection if selected else "#3B4048"
        sample_background = preset.background_color if preset.background_mode == "opaque_box" else "#101216"
        font_weight = str(FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id].weight_class)
        self.setStyleSheet(
            f"QFrame#captionPresetCard {{ background: {'#251A17' if selected else background}; border: {2 if selected else 1}px solid {border}; border-radius: 9px; }}"
            "QFrame#captionPresetCard:hover { border-color: " + accent + "; }"
            "QLabel#captionPresetCardTitle { color: #F1F4F8; font-size: 12px; font-weight: 700; background: transparent; border: 0; }"
            f"QLabel#captionPresetCardBadge {{ color: {selection if selected else accent}; font-size: 11px; font-weight: 700; background: transparent; border: 0; }}"
            f"QFrame#captionPresetSample {{ background: {sample_background}; border: 0; border-radius: 6px; }}"
            f"QLabel#captionPresetSampleText {{ color: {preset.text_color}; font-family: '{FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id].render_family}'; font-weight: {font_weight}; background: transparent; border: 0; }}"
        )


class CaptionPresetPicker(QWidget):
    """A fixed seven-card grid shared by project Settings and Drafts."""

    preset_selected = Signal(str)
    preset_hovered = Signal(str)
    preset_hover_cleared = Signal()

    def __init__(self, *, columns: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("captionPresetPicker")
        self.cards: dict[str, CaptionPresetCard] = {}
        self.selected_preset_id: str | None = None
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        for index, preset in enumerate(CAPTION_PRESET_DEFINITIONS.values()):
            card = CaptionPresetCard(preset, self)
            card.activated.connect(self.choose)
            card.hovered.connect(self.preset_hovered)
            card.hover_left.connect(lambda _preset_id: self.preset_hover_cleared.emit())
            layout.addWidget(card, index // columns, index % columns)
            self.cards[preset.preset_id] = card
        for column in range(columns):
            layout.setColumnStretch(column, 1)

    def choose(self, preset_id: str) -> None:
        if preset_id not in self.cards:
            return
        self.set_selected(preset_id)
        self.preset_selected.emit(preset_id)

    def set_selected(self, preset_id: str) -> None:
        if preset_id not in self.cards:
            return
        self.selected_preset_id = preset_id
        for card_id, card in self.cards.items():
            card.set_selected(card_id == preset_id)


class CaptionPresetPickerDialog(QDialog):
    """Draft-only picker that makes the pending/rerender boundary explicit."""

    def __init__(
        self,
        current_preset_id: str,
        parent: QWidget | None = None,
        *,
        style_id: str = "documentary",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("captionPresetPickerDialog")
        self.setWindowTitle("Стиль субтитров")
        self.setModal(True)
        self._style_id = style_id
        self.resize(1120, 620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)
        title = QLabel("Стиль субтитров")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        copy = QLabel(
            "Выберите оформление для этого черновика. Сохраним изменение как ожидающее: "
            "текущий Preview останется до «Пересоздать черновик»."
        )
        copy.setObjectName("muted")
        copy.setWordWrap(True)
        layout.addWidget(copy)
        content = QHBoxLayout()
        content.setSpacing(14)
        self.picker = CaptionPresetPicker(columns=3, parent=self)
        self.picker.setObjectName("draftCaptionPresetPicker")
        self.picker.set_selected(current_preset_id)
        self.picker.preset_hovered.connect(self._show_demo)
        self.picker.preset_hover_cleared.connect(self._restore_selected_demo)
        self.picker.preset_selected.connect(self._show_demo)
        content.addWidget(self.picker, 3)
        self.demo_preview = VideoPreview()
        self.demo_preview.setObjectName("draftCaptionPresetDemo")
        self.demo_preview.set_vertical_frame_size(230, 409)
        self.demo_preview.set_frame_sink_output(True)
        self.demo_preview.controls_host.hide()
        self.demo_preview.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.demo_preview.player.mediaStatusChanged.connect(self._demo_media_status_changed)
        content.addWidget(self.demo_preview, 1, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(content)
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
        self._show_demo(current_preset_id)

    @property
    def selected_preset_id(self) -> str:
        return self.picker.selected_preset_id or ""

    def _show_demo(self, preset_id: str) -> None:
        path = settings_preview_path(self._style_id, preset_id)
        if path is None:
            self.demo_preview.set_file(None, presentation="vertical", title="Демо недоступно")
            return
        if self.demo_preview.active_media_path == path:
            self.demo_preview._play()
            return
        self.demo_preview.set_file(path, presentation="vertical", title="Демо оформления")

    def _restore_selected_demo(self) -> None:
        self._show_demo(self.selected_preset_id)

    def _demo_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in {QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia}:
            self.demo_preview._play()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.demo_preview.suspend()
        super().closeEvent(event)
