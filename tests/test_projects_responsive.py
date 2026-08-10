from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QPoint
from PySide6.QtWidgets import QApplication, QBoxLayout, QPushButton

from app.gui.models import DesktopSettings
from app.gui.screens.projects_screen import ProjectsScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import THEME_TOKENS, load_theme
from app.gui.viewmodels import ProjectsViewModel


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(load_theme())
    return application


def test_desktop_theme_resolves_only_the_approved_creator_tech_tokens() -> None:
    assert dict(THEME_TOKENS) == {
        "APP_BG": "#0D0F13",
        "SIDEBAR": "#11141A",
        "MAIN": "#151922",
        "ELEVATED": "#1B202C",
        "BORDER": "#2A303D",
        "PRIMARY": "#FF6A00",
        "PRIMARY_HOVER": "#FF7F33",
        "ACCENT": "#252A4A",
        "SUCCESS": "#56D6A0",
        "WARNING": "#D7A95B",
        "ERROR": "#E46B78",
        "PRIMARY_TEXT": "#F3F4F7",
        "SECONDARY": "#9299AA",
        "MUTED": "#676E7F",
    }
    theme = load_theme()
    assert "@" not in theme
    assert "#FF6A00" in theme
    assert "#FF7900" not in theme
    assert "#0B0C0E" not in theme
    assert set(re.findall(r"#[0-9A-Fa-f]{6}", theme)) <= set(THEME_TOKENS.values())


def _screen(tmp_path: Path) -> ProjectsScreen:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    project.name = "Long recent project name used to exercise real card wrapping " * 5
    projects.save(project)
    root = Path(__file__).resolve().parents[1]
    settings = DesktopSettings.defaults(data)
    services = DesktopServices(
        engine_root=root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(root),
        system=SystemService(root),
    )
    screen = ProjectsScreen(ProjectsViewModel(services))
    screen.refresh()
    return screen


@pytest.mark.parametrize(
    ("width", "height", "compact"),
    # 604 logical px is the Projects area of the 760 px application minimum
    # after its compact sidebar; it represents a 1280 px display at 150%.
    ((604, 480, True), (1072, 720, False), (1232, 900, False), (1712, 1080, False)),
)
def test_projects_source_workspace_reflows_without_hidden_horizontal_clip(
    tmp_path: Path, width: int, height: int, compact: bool,
) -> None:
    application = _application()
    screen = _screen(tmp_path)

    try:
        screen.resize(width, height)
        screen.show()
        application.processEvents()
        application.processEvents()

        viewport = screen.content_scroll.viewport()
        assert screen.content_scroll.horizontalScrollBar().maximum() == 0
        assert screen.content_scroll.widget().width() <= viewport.width()
        expected_direction = (
            QBoxLayout.Direction.TopToBottom
            if compact else QBoxLayout.Direction.LeftToRight
        )
        assert screen._top_layout.direction() == expected_direction
        assert screen._url_row_layout.direction() == expected_direction

        # A disabled horizontal scrollbar is not sufficient: every visible
        # source/recent-project action must actually fit in the viewport.
        for button in screen.findChildren(QPushButton):
            if not button.isVisible():
                continue
            left = button.mapTo(viewport, QPoint(0, 0)).x()
            assert left >= 0
            assert left + button.width() <= viewport.width()
            assert button.contentsRect().width() >= button.minimumSizeHint().width()
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()
