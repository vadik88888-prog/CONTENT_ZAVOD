from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QPoint, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QBoxLayout, QFrame, QLabel, QPushButton

from app import __version__
from app.gui.main_window import MainWindow
from app.gui.models import DesktopSettings, RunStatus
from app.gui.screens.onboarding_screen import OnboardingDialog
from app.gui.screens.projects_screen import ProjectsScreen
from app.gui.screens.settings_screen import SettingsScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import THEME_TOKENS, load_theme
from app.gui.viewmodels import ProjectsViewModel, SettingsViewModel


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
        "ACCENT": "#2B211A",
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


def _services(
    tmp_path: Path, *, project_count: int = 1, hostile_persisted_text: bool = False,
) -> DesktopServices:
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    for index in range(project_count):
        if hostile_persisted_text and index % 3 == 0:
            project = projects.create_url(
                "https://example.test/watch/" + "unbroken-url-segment-" * 6 + str(index),
                {
                    "title": "ОченьДлинныйЗаголовокИзПровереннойСсылки_" * 2 + str(index),
                },
            )
        else:
            filename = (
                f"source-{index}.mp4"
                if not hostile_persisted_text
                else (f"непрерывное_русское_имя_видео_{index}_" + "x" * 42 + ".mp4")
            )
            source = tmp_path / filename
            source.write_bytes(b"source")
            project = projects.create(source)
        project.name = (
            ("Очень_Длинное_Русское_Имя_Проекта_Без_Пробелов_" + str(index))[:79]
            if hostile_persisted_text
            else "Long recent project name used to exercise real card wrapping " * 5
        )
        projects.save(project)
    root = Path(__file__).resolve().parents[1]
    settings = DesktopSettings.defaults(data)
    settings.onboarding_completed = True
    return DesktopServices(
        engine_root=root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(root),
        system=SystemService(root),
    )


def _screen(
    tmp_path: Path, *, project_count: int = 1, hostile_persisted_text: bool = False,
) -> ProjectsScreen:
    services = _services(
        tmp_path,
        project_count=project_count,
        hostile_persisted_text=hostile_persisted_text,
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
        assert screen._source_methods_layout.direction() == expected_direction
        assert screen._url_row_layout.direction() == expected_direction
        assert screen._source_divider_layout.direction() == (
            QBoxLayout.Direction.LeftToRight
            if compact
            else QBoxLayout.Direction.TopToBottom
        )
        expected_divider = QFrame.Shape.HLine if compact else QFrame.Shape.VLine
        assert screen.source_divider_before.frameShape() == expected_divider
        assert screen.source_divider_after.frameShape() == expected_divider

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


def test_hostile_persisted_project_cards_reflow_across_logical_client_widths(tmp_path: Path) -> None:
    application = _application()
    screen = _screen(tmp_path, project_count=9, hostile_persisted_text=True)
    widths = (1280, 564, 697, 1093, 755, 1067, 840, 1024, 883, 909, 1280)

    try:
        screen.resize(widths[0], 520)
        screen.show()
        application.processEvents()

        for width in widths:
            screen.resize(width, 520)
            # Card-column reflow is deliberately queued to avoid rebuilding a
            # real project list for every intermediate native resize event.
            application.processEvents()
            application.processEvents()

            viewport = screen.content_scroll.viewport()
            host = screen.content_scroll.widget()
            assert screen.content_scroll.horizontalScrollBar().maximum() == 0, (
                width,
                screen._rendered_columns,
                viewport.width(),
                host.width(),
                host.minimumSizeHint().width(),
            )
            assert host.width() <= viewport.width()
            expected_columns = (
                4 if viewport.width() >= 1_120
                else 3 if viewport.width() >= 840
                else 2 if viewport.width() >= 600
                else 1
            )
            assert screen._rendered_columns == expected_columns

            for index in range(screen.list_layout.count()):
                card = screen.list_layout.itemAt(index).widget()
                assert card is not None
                left = card.mapTo(viewport, QPoint(0, 0)).x()
                assert left >= 0
                assert left + card.width() <= viewport.width()
                for button in card.findChildren(QPushButton):
                    assert button.width() >= button.minimumSizeHint().width()
                statuses = [
                    label for label in card.findChildren(QLabel)
                    if label.objectName() == "status"
                ]
                assert statuses and all(label.isVisible() and label.width() > 0 for label in statuses)

        first_card = screen.list_layout.itemAt(0).widget()
        assert first_card is not None
        responsive_labels = [
            label for label in first_card.findChildren(QLabel)
            if label.toolTip() and "\u200b" in label.text()
        ]
        assert responsive_labels
        assert all("\u200b" not in label.toolTip() for label in responsive_labels)

        screen.url_input.setText("https://example.test/" + "very-long-url-token-" * 30)
        application.processEvents()
        assert screen.content_scroll.widget().width() <= screen.content_scroll.viewport().width()
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()


def test_unchanged_projects_refresh_reuses_exact_cards_and_presentations(
    tmp_path: Path, monkeypatch,
) -> None:
    application = _application()
    services = _services(tmp_path, project_count=6)
    screen = ProjectsScreen(ProjectsViewModel(services))

    try:
        screen.resize(1280, 720)
        screen.refresh()
        cards = [
            screen.list_layout.itemAt(index).widget()
            for index in range(screen.list_layout.count())
        ]
        rebuilds = 0
        run_reads = 0
        original_render_cards = screen._render_cards
        original_runs_for = type(services).runs_for

        def tracked_render_cards() -> None:
            nonlocal rebuilds
            rebuilds += 1
            original_render_cards()

        def tracked_runs_for(self, project):
            nonlocal run_reads
            run_reads += 1
            return original_runs_for(self, project)

        monkeypatch.setattr(screen, "_render_cards", tracked_render_cards)
        monkeypatch.setattr(type(services), "runs_for", tracked_runs_for)

        screen.refresh()

        assert rebuilds == 0
        assert run_reads == 0
        assert [
            screen.list_layout.itemAt(index).widget()
            for index in range(screen.list_layout.count())
        ] == cards

        # A run can transition without rewriting project.json. Its exact
        # persisted identity/revision must still invalidate the card cache.
        project = services.list_projects()[0]
        run = services.runs.create(project, {}, {}, "test")
        screen.refresh()
        assert rebuilds == 1
        assert run_reads == 6
        assert screen._active_project_id == project.project_id

        run.status = RunStatus.CANCELLED
        services.runs.save(run)
        screen.refresh()
        assert rebuilds == 2
        assert run_reads == 12
        assert screen._active_project_id is None
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()


def test_shell_profiles_keep_navigation_and_brand_unclipped(tmp_path: Path) -> None:
    application = _application()
    services = _services(tmp_path, project_count=0)
    services.settings.window_geometry = None
    window = MainWindow(services)

    try:
        # Let the initial available-screen clamp finish first.  Subsequent
        # programmatic sizes exercise logical layout widths independently of
        # the fixed 800 px offscreen test display.
        window.show()
        application.processEvents()
        application.processEvents()

        for width in (720, 853, 911, 920, 1024, 1067, 1093, 1120, 1280):
            window.resize(width, 480)
            application.processEvents()

            expected_sidebar = 156 if width < 1120 else 224
            assert window.sidebar.width() == expected_sidebar
            for button in (
                window.new_button,
                window.projects_button,
                window.settings_button,
                window.help_button,
            ):
                assert button.isVisible()
                assert button.width() >= button.minimumSizeHint().width()
                assert button.toolTip()

            if width < 1120:
                assert window.brand_content.text() == "CF"
                assert not window.brand_factory.isVisible()
                assert not window.system_status.isVisible()
                assert not window.version.isVisible()
            else:
                assert window.brand_content.text() == "CONTENT"
                assert window.brand_factory.isVisible()
                assert window.brand_content.width() >= window.brand_content.minimumSizeHint().width()
                assert window.brand_factory.width() >= window.brand_factory.minimumSizeHint().width()
                assert window.system_status.isVisible()
                assert window.version.isVisible()

            assert window.stack.width() > 0

        window.resize(1280, 380)
        application.processEvents()
        assert not window.system_status.isVisible()
        assert not window.version.isVisible()
        assert window.help_button.geometry().bottom() < window.sidebar.height()
    finally:
        window.close()
        window.deleteLater()
        application.processEvents()


def test_shell_default_geometry_is_clamped_to_available_logical_work_area(tmp_path: Path) -> None:
    application = _application()
    services = _services(tmp_path, project_count=0)
    services.settings.window_geometry = None
    window = MainWindow(services)

    try:
        assert window.size().width() == 1320
        assert window.size().height() == 840
        window.show()
        application.processEvents()
        application.processEvents()

        available = application.primaryScreen().availableGeometry()
        frame = window.frameGeometry()
        assert available.contains(frame.topLeft())
        assert available.contains(frame.bottomRight())
        assert window.minimumHeight() == 380
    finally:
        window.close()
        window.deleteLater()
        application.processEvents()


def test_expanded_settings_fit_compact_shell_and_hostile_paths(tmp_path: Path) -> None:
    application = _application()
    services = _services(tmp_path, project_count=0)
    services.settings.data_directory = "C:/" + "очень-длинный-непрерывный-путь-" * 12
    services.settings.config_path = "C:/" + "unbroken-config-path-" * 24 + "config.yaml"
    screen = SettingsScreen(SettingsViewModel(services))

    try:
        screen.advanced_toggle.setChecked(True)
        screen.diagnostics.setPlainText("ОШИБКА_" * 300)
        screen.resize(588, 420)
        screen.show()
        application.processEvents()

        for width in (564, 588, 604, 697, 755, 840, 883, 909, 1024, 1067, 1093, 1280):
            screen.resize(width, 420)
            application.processEvents()
            viewport = screen.content_scroll.viewport()
            assert screen.content_scroll.horizontalScrollBar().maximum() == 0
            assert screen.content_scroll.widget().width() <= viewport.width()
            assert screen.diagnostics.horizontalScrollBar().maximum() == 0
            for button in screen.findChildren(QPushButton):
                if not button.isVisible():
                    continue
                left = button.mapTo(viewport, QPoint(0, 0)).x()
                assert left >= 0
                assert left + button.width() <= viewport.width()
                assert button.width() >= button.minimumSizeHint().width()

        assert screen.data_directory.toolTip() == services.settings.data_directory
        assert screen.config_path.toolTip() == services.settings.config_path
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()


def test_settings_keeps_support_and_key_status_visible_and_opens_telegram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = _application()
    screen = SettingsScreen(SettingsViewModel(_services(tmp_path, project_count=0)))
    opened: list[QUrl] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)

    try:
        assert screen.api_status.text().startswith("Статус ключа")
        assert screen.advanced_toggle.text() == "Расширенные настройки"
        assert screen.advanced_content.isHidden()
        assert screen.telegram_button.text() == "Написать в Telegram"
        assert any(label.text() == "Telegram: @rezvis" for label in screen.findChildren(QLabel))
        assert any(label.text() == f"Content Factory {__version__}" for label in screen.findChildren(QLabel))

        screen.telegram_button.click()
        application.processEvents()

        assert [url.toString() for url in opened] == ["https://t.me/rezvis"]
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()


def test_onboarding_actions_reflow_and_diagnostics_wrap_at_compact_widths(tmp_path: Path) -> None:
    application = _application()
    services = _services(tmp_path, project_count=0)
    dialog = OnboardingDialog(SettingsViewModel(services))

    try:
        dialog.checks.setPlainText("ДИАГНОСТИКА_" * 240)
        dialog.show()
        application.processEvents()

        for width, height in ((420, 300), (540, 340), (600, 360), (620, 360), (697, 360)):
            dialog.resize(width, height)
            application.processEvents()
            expected = (
                QBoxLayout.Direction.TopToBottom
                if width < 620
                else QBoxLayout.Direction.LeftToRight
            )
            assert dialog._actions_layout.direction() == expected
            assert dialog.checks.horizontalScrollBar().maximum() == 0
            for button in (dialog.check_button, dialog.continue_button):
                position = button.mapTo(dialog, QPoint(0, 0))
                assert position.x() >= 0
                assert position.x() + button.width() <= dialog.width()
                assert position.y() + button.height() <= dialog.height()
                assert button.width() >= button.minimumSizeHint().width()
    finally:
        dialog.close()
        dialog.deleteLater()
        application.processEvents()
