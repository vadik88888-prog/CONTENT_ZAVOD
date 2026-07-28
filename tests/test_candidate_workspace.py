from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from app.gui.components import VideoPreview
from app.gui.models import DesktopSettings, ProjectStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel
from app.utils import write_json


def _workspace(tmp_path: Path):
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source, source_metadata={"duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0})
    settings = DesktopSettings.defaults(data); settings.local_test_mode = True
    root = Path(__file__).resolve().parents[1]
    services = DesktopServices(
        engine_root=root, settings_store=SettingsStore(data), settings=settings, projects=projects,
        runs=RunHistoryStore(projects), pipeline=PipelineFacade(root), system=SystemService(root),
    )
    analysis_path = tmp_path / "analysis.json"
    write_json(analysis_path, {
        "candidates": [
            {
                "candidate_id": "candidate-recommended", "title": "Сильное начало", "start_seconds": 1.0,
                "end_seconds": 18.0, "potential": "high", "confidence": 0.9, "recommended": True,
                "reasons": ["Сильное начало."], "preview": {"thumbnail": {"timestamp_seconds": 2.0}},
            },
            {
                "candidate_id": "candidate-other", "title": "Другой момент", "start_seconds": 19.0,
                "end_seconds": 29.0, "potential": "low", "confidence": 0.6, "recommended": False,
                "reasons": ["Есть самостоятельная мысль."], "preview": {"thumbnail": {"timestamp_seconds": 20.0}},
            },
        ],
    })
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "analysis-test"
    project.status = ProjectStatus.ANALYSIS_READY
    project.candidate_states = {"candidate-recommended": "analyzed", "candidate-other": "analyzed"}
    services.projects.save(project)
    return services, project


def test_candidate_workspace_has_persistent_selection_and_disabled_delivery_cta(tmp_path: Path, monkeypatch) -> None:
    # A few non-UI tests intentionally initialise QCoreApplication first. Qt
    # cannot upgrade that singleton to QApplication in the same process; doing
    # so aborts the interpreter on Windows instead of raising a Python error.
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    preview_ranges: list[tuple[Path, float, float, Path | None]] = []

    def capture_preview_range(
        _preview, path, start_seconds, end_seconds, *, autoplay=True, cache_directory=None,
    ) -> None:
        preview_ranges.append((Path(path), float(start_seconds), float(end_seconds), cache_directory))

    monkeypatch.setattr(VideoPreview, "set_range", capture_preview_range)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)

        assert screen.content_scroll.widget() is screen.content_host
        assert screen.draft_button.isEnabled() is False
        assert screen.production_button.isEnabled() is False

        # Regression: the primary "Проверка кандидатов" action must enter the
        # review workspace without a NameError and move focus to it.
        screen.show()
        app.processEvents()
        QTest.mouseClick(screen.run_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert screen.candidate_review.hasFocus()
        assert screen.candidate_review.focusPolicy() == Qt.FocusPolicy.StrongFocus

        # Candidate selection is bound to the source player, including the
        # exact range and the project-local proxy cache destination.
        candidate_previews = [
            button for button in screen.findChildren(QPushButton)
            if button.objectName().startswith("preview-candidate-")
        ]
        assert len(candidate_previews) == 2
        first_preview, second_preview = candidate_previews
        QTest.mouseClick(first_preview, Qt.MouseButton.LeftButton)
        QTest.mouseClick(second_preview, Qt.MouseButton.LeftButton)
        assert preview_ranges == [
            (project.source, 1.0, 18.0, project.directory / "preview-proxies"),
            (project.source, 19.0, 29.0, project.directory / "preview-proxies"),
        ]
        assert screen._active_candidate_id and screen._active_candidate_id.endswith("other")

        screen._select_recommended()

        assert viewmodel.project is not None
        assert viewmodel.project.review_selected_candidate_ids == ["candidate-recommended"]
        assert screen.draft_button.isEnabled() is True
        assert screen.production_button.isEnabled() is False

        screen._change_candidate_filter("unselected")
        screen._candidate_checks["candidate-other"].setChecked(True)

        assert viewmodel.project.review_selected_candidate_ids == ["candidate-recommended", "candidate-other"]
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
