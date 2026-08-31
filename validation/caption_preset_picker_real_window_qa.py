"""Native-window visual QA for caption-preset selection in Settings and Drafts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import QPoint
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


def _settle(app: QApplication, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


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
            _save(screen, output / "settings-caption-presets.png")

            dialog = CaptionPresetPickerDialog("editorial_narrow", screen)
            dialog.show()
            _settle(app)
            dialog.picker.choose("word_pop")
            _settle(app)
            if dialog.selected_preset_id != "word_pop":
                raise AssertionError("Draft picker did not retain the selected pending preset.")
            if len(dialog.picker.cards) != len(CAPTION_PRESET_DEFINITIONS):
                raise AssertionError("Draft picker does not expose all seven caption presets.")
            _save(dialog, output / "drafts-caption-presets.png")
            print(output / "settings-caption-presets.png")
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
