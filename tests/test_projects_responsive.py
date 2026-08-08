from __future__ import annotations

import os
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
from app.gui.styles import load_theme
from app.gui.viewmodels import ProjectsViewModel


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(load_theme())
    return application


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
