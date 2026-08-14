from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, QRect
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from app.content_profile_taxonomy import (
    AUTO_PROFILE_INPUT,
    PROFILE_AXIS_ORDER,
    ProfileAxisId,
    profile_input_ids,
    user_overridable_values,
)
from app.gui.models import DesktopSettings
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme
from app.gui.viewmodels import ProjectViewModel
from app.utils import utc_now


SAMPLE_SELECTIONS = {
    "format": "gameplay",
    "editorial_mode": "commentary",
    "domain": "gaming",
    "traits": "visual_led",
}


def _control_snapshot(
    combo: QComboBox,
    axis_id: ProfileAxisId,
    *,
    viewport: QWidget,
) -> dict[str, object]:
    items = [
        {"label": combo.itemText(index), "value": combo.itemData(index)}
        for index in range(combo.count())
    ]
    expected_labels = [
        "Авто",
        *(item.label for item in user_overridable_values(axis_id)),
    ]
    expected_values = list(profile_input_ids(axis_id))
    actual_labels = [str(item["label"]) for item in items]
    actual_values = [str(item["value"]) for item in items]
    viewport_position = combo.mapTo(viewport, QPoint(0, 0))
    viewport_geometry = QRect(viewport_position, combo.size())
    return {
        "visible": combo.isVisible(),
        "fully_inside_scroll_viewport": viewport.rect().contains(viewport_geometry),
        "enabled": combo.isEnabled(),
        "geometry": {
            "x": combo.geometry().x(),
            "y": combo.geometry().y(),
            "width": combo.geometry().width(),
            "height": combo.geometry().height(),
        },
        "items": items,
        "expected_values": expected_values,
        "expected_labels": expected_labels,
        "values_match_contract": actual_values == expected_values,
        "labels_match_contract": actual_labels == expected_labels,
        "current_value": combo.currentData(),
        "current_label": combo.currentText(),
    }


def run(output_directory: Path) -> None:
    app = QApplication.instance() or QApplication([])
    if app.platformName().casefold() != "windows":
        raise RuntimeError(f"Real-window QA requires the Windows Qt platform, got {app.platformName()!r}.")
    app.setStyleSheet(load_theme())

    repository_root = Path(__file__).resolve().parents[1]
    output_directory.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_directory / "project-screen-profile-controls.png"
    runtime_path = output_directory / "runtime-evidence.json"

    with tempfile.TemporaryDirectory(prefix="content-factory-profile-window-qa-") as temporary:
        data_directory = Path(temporary)
        source_path = data_directory / "source.mp4"
        source_path.write_bytes(b"real-window-ui-fixture")
        projects = DesktopProjectStore(data_directory / "desktop-data")
        project = projects.create(
            source_path,
            name="Source Content Profile v2 real-window QA",
            source_metadata={"duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0},
        )
        settings = DesktopSettings.defaults(data_directory / "desktop-data")
        settings.local_test_mode = True
        settings.onboarding_completed = True
        services = DesktopServices(
            engine_root=repository_root,
            settings_store=SettingsStore(data_directory / "desktop-data"),
            settings=settings,
            projects=projects,
            runs=RunHistoryStore(projects),
            pipeline=PipelineFacade(repository_root),
            system=SystemService(repository_root),
        )
        screen = ProjectScreen(ProjectViewModel(services))
        screen.setWindowTitle("Content Factory — Source Content Profile v2 QA")
        available = app.primaryScreen().availableGeometry()
        width = min(1400, max(900, available.width() - 80))
        height = min(900, max(640, available.height() - 80))
        screen.resize(width, height)
        screen.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )
        screen.open(project)
        screen.show()
        screen.raise_()
        screen.activateWindow()
        QTest.qWait(750)
        app.processEvents()
        screen.setup_advanced_toggle.click()
        QTest.qWait(250)
        app.processEvents()

        controls = {
            "format": screen.profile_format_override,
            "editorial_mode": screen.profile_editorial_mode_override,
            "domain": screen.profile_domain_override,
            "traits": screen.profile_trait_override,
        }
        for axis_id in PROFILE_AXIS_ORDER:
            combo = controls[axis_id]
            expected_index = combo.findData(SAMPLE_SELECTIONS[axis_id])
            if expected_index < 0:
                raise AssertionError(f"Missing {axis_id} selection {SAMPLE_SELECTIONS[axis_id]!r}.")
            combo.setCurrentIndex(expected_index)
            app.processEvents()

        screen.content_scroll.ensureWidgetVisible(screen.profile_trait_override, 24, 24)
        QTest.qWait(500)
        app.processEvents()
        persisted = projects.load(project.project_id)
        persisted_values = {
            "format": persisted.settings.profile_format_override,
            "editorial_mode": persisted.settings.profile_editorial_mode_override,
            "domain": persisted.settings.profile_domain_override,
            "traits": persisted.settings.profile_traits_override[0]
            if persisted.settings.profile_traits_override
            else AUTO_PROFILE_INPUT,
        }
        control_evidence = {
            axis_id: _control_snapshot(
                controls[axis_id],
                axis_id,
                viewport=screen.content_scroll.viewport(),
            )
            for axis_id in PROFILE_AXIS_ORDER
        }
        checks = {
            "qt_windows_platform": app.platformName().casefold() == "windows",
            "native_window_created": int(screen.winId()) > 0,
            "project_screen_visible": screen.isVisible(),
            "settings_panel_visible": screen.settings_panel.isVisible(),
            "all_controls_visible": all(bool(item["visible"]) for item in control_evidence.values()),
            "all_controls_inside_scroll_viewport": all(
                bool(item["fully_inside_scroll_viewport"]) for item in control_evidence.values()
            ),
            "all_controls_enabled": all(bool(item["enabled"]) for item in control_evidence.values()),
            "all_values_match_contract": all(
                bool(item["values_match_contract"]) for item in control_evidence.values()
            ),
            "all_labels_match_contract": all(
                bool(item["labels_match_contract"]) for item in control_evidence.values()
            ),
            "interactive_selection_persisted": persisted_values == SAMPLE_SELECTIONS,
        }
        if not all(checks.values()):
            raise AssertionError(
                f"Real-window profile-control checks failed: {checks!r}; controls={control_evidence!r}"
            )

        screenshot = app.primaryScreen().grabWindow(int(screen.winId()))
        if screenshot.isNull() or not screenshot.save(str(screenshot_path), "PNG"):
            raise RuntimeError("Could not capture the real ProjectScreen window.")
        screenshot_sha256 = hashlib.sha256(screenshot_path.read_bytes()).hexdigest()
        evidence = {
            "schema_version": "source-content-profile-v2-task1-real-window.1",
            "captured_at": utc_now(),
            "os": platform.platform(),
            "qt_platform": app.platformName(),
            "screen": {
                "name": app.primaryScreen().name(),
                "available_geometry": {
                    "x": available.x(),
                    "y": available.y(),
                    "width": available.width(),
                    "height": available.height(),
                },
                "device_pixel_ratio": app.primaryScreen().devicePixelRatio(),
            },
            "window": {
                "class": type(screen).__name__,
                "title": screen.windowTitle(),
                "native_window_id": int(screen.winId()),
                "width": screen.width(),
                "height": screen.height(),
            },
            "sample_selections": SAMPLE_SELECTIONS,
            "persisted_values": persisted_values,
            "controls": control_evidence,
            "checks": checks,
            "result": "PASS",
            "screenshot": {
                "path": screenshot_path.name,
                "sha256": screenshot_sha256,
                "width": screenshot.width(),
                "height": screenshot.height(),
            },
        }
        runtime_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        screen.close()
        app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture real Windows ProjectScreen profile-control QA evidence."
    )
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    run(arguments.output_directory.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
