"""Native-window visual QA for caption-preset selection in Settings and Drafts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import QPoint
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication, QWidget

from app.caption_presets import CAPTION_PRESET_DEFINITIONS
from app.font_assets import FONT_ASSET_DEFINITIONS
from app.gui.components import CaptionPresetPickerDialog
from app.gui.models import DesktopProject, DesktopSettings
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme
from app.gui.viewmodels import ProjectViewModel
from app.settings_preview_assets import settings_preview_path


def _settle(app: QApplication, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


def _wait_for(app: QApplication, predicate, *, seconds: float = 8.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Settings production demo did not start playback in the native window.")


def _services(root: Path, data: Path) -> tuple[DesktopServices, DesktopProject]:
    source = data / "source.mp4"
    source.write_bytes(b"fixture source")
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    projects = DesktopProjectStore(data)
    project = projects.create(
        source,
        source_metadata={"duration": 90.0, "width": 1920, "height": 1080, "fps": 30.0},
    )
    services = DesktopServices(
        engine_root=root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(root),
        system=SystemService(root),
    )
    return services, project


def _save(widget: QWidget, target: Path) -> None:
    image = widget.grab()
    if image.isNull() or not image.save(str(target)):
        raise RuntimeError(f"Could not save native screenshot: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="caption-preset-picker-qa-") as raw_data:
        app = QApplication.instance() or QApplication([])
        if app.platformName().casefold() != "windows":
            raise RuntimeError(f"Real-window QA requires Windows Qt, got {app.platformName()!r}.")
        app.setStyleSheet(load_theme())
        services, project = _services(root, Path(raw_data))
        screen = ProjectScreen(ProjectViewModel(services))
        try:
            screen.resize(1440, 920)
            screen.open(project)
            screen.show()
            _settle(app)
            screen.content_scroll.ensureWidgetVisible(screen.setup_caption_picker, 12, 12)
            _settle(app)
            viewport = screen.content_scroll.viewport()
            picker_origin = screen.setup_caption_picker.mapTo(viewport, QPoint(0, 0))
            if picker_origin.y() < 0 or picker_origin.y() + screen.setup_caption_picker.height() > viewport.height():
                raise AssertionError("Settings caption cards are not entirely visible in the native window.")
            cards = screen.setup_caption_picker.cards
            if set(cards) != set(CAPTION_PRESET_DEFINITIONS):
                raise AssertionError("Settings does not expose all seven caption presets.")
            for preset_id, card in cards.items():
                expected_font = FONT_ASSET_DEFINITIONS[
                    CAPTION_PRESET_DEFINITIONS[preset_id].preferred_font_asset_id
                ].render_family
                if card.sample.font().family() != expected_font:
                    raise AssertionError(f"{preset_id} does not use its bundled font.")
            screen.setup_caption_picker.choose("word_pop")
            expected_caption = settings_preview_path(project.settings.subtitle_style, "word_pop")
            _wait_for(app, lambda: (
                screen.setup_demo_preview.active_media_path == expected_caption
                and screen.setup_demo_preview.poster.pixmap() is not None
                and not screen.setup_demo_preview.poster.pixmap().isNull()
                and screen.setup_demo_preview.poster.isVisible()
                and screen.setup_demo_preview.player.position() > 100
                and screen.setup_demo_preview.player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
            ))
            if cards["word_pop"].property("selectionTone") != "#FF6846":
                raise AssertionError("Selected Settings caption card is not orange.")
            _save(screen, output / "settings-caption-presets.png")

            screen._choose_setup_value(screen.setup_creative_style, "dynamic")
            expected_style = settings_preview_path("dynamic", "word_pop")
            _wait_for(app, lambda: (
                screen.setup_demo_preview.active_media_path == expected_style
                and screen.setup_demo_preview.poster.pixmap() is not None
                and not screen.setup_demo_preview.poster.pixmap().isNull()
                and screen.setup_demo_preview.poster.isVisible()
                and screen.setup_demo_preview.player.position() > 100
            ))
            if screen.setup_demo_detail.text() != "Dynamic · Word Pop":
                raise AssertionError("Creative-style selection did not update the Settings demo label.")
            _save(screen, output / "settings-creative-style-demo.png")

            dialog = CaptionPresetPickerDialog("editorial_narrow", screen)
            dialog.show()
            _settle(app)
            dialog.picker.choose("word_pop")
            _settle(app)
            if dialog.selected_preset_id != "word_pop":
                raise AssertionError("Draft picker did not retain the selected pending preset.")
            if len(dialog.picker.cards) != len(CAPTION_PRESET_DEFINITIONS):
                raise AssertionError("Draft picker does not expose all seven caption presets.")
            if dialog.save_button.text() != "Сохранить выбор":
                raise AssertionError("Draft picker save label is not localized.")
            if dialog.picker.cards["word_pop"].property("selectionTone") != "#FF6846":
                raise AssertionError("Selected Draft caption card is not orange.")
            _save(dialog, output / "drafts-caption-presets.png")
            print(output / "settings-caption-presets.png")
            print(output / "settings-creative-style-demo.png")
            print(output / "drafts-caption-presets.png")
        finally:
            dialog.close() if "dialog" in locals() else None
            screen.preview.suspend()
            screen.setup_demo_preview.suspend()
            screen.close()
            screen.deleteLater()
            _settle(app, 0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
