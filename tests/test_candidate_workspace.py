from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

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
    preview_ranges: list[tuple[Path, float, float, Path | None, str | None]] = []

    def capture_preview_range(
        _preview, path, start_seconds, end_seconds, *, autoplay=True, cache_directory=None, candidate_title=None,
    ) -> None:
        _preview.play_button.setEnabled(True)
        preview_ranges.append((Path(path), float(start_seconds), float(end_seconds), cache_directory, candidate_title))

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
        app.processEvents()
        assert preview_ranges == [
            (project.source, 1.0, 18.0, project.directory / "preview-proxies", "Сильное начало"),
            (project.source, 19.0, 29.0, project.directory / "preview-proxies", "Другой момент"),
        ]
        assert screen._active_candidate_id and screen._active_candidate_id.endswith("other")
        assert screen._candidate_cards[screen._active_candidate_id].property("activeCandidate") is True
        assert screen.preview.play_button.hasFocus()
        assert all(
            label.wordWrap() for label in screen.candidate_detail.findChildren(QLabel)
            if label.objectName() == "muted"
        )

        screen._select_recommended()

        assert viewmodel.project is not None
        assert viewmodel.project.review_selected_candidate_ids == ["candidate-recommended"]
        assert screen.draft_button.isEnabled() is True
        assert screen.production_button.isEnabled() is False

        screen._change_candidate_filter("unselected")
        select_other = screen._candidate_selection_buttons["candidate-other"]
        assert select_other.text() == "Добавить к черновикам"
        QTest.mouseClick(select_other, Qt.MouseButton.LeftButton)

        assert viewmodel.project.review_selected_candidate_ids == ["candidate-recommended", "candidate-other"]

        # A completed render with non-blocking warnings must still take the
        # user back to the review workspace, not offer a misleading new
        # analysis as the next action.
        project.status = ProjectStatus.COMPLETED_WITH_WARNINGS
        services.projects.save(project)
        screen._project_changed(project)
        assert screen.run_button.text() == "Посмотреть найденные моменты"
    finally:
        screen.close()


def test_source_setup_screen_persists_primary_choices_before_analysis(tmp_path: Path) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    project.analysis_artifact_path = None
    project.analysis_id = None
    project.candidate_states = {}
    project.status = ProjectStatus.SOURCE_READY
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)

    try:
        screen.open(project)
        assert not screen.setup_start_button.isHidden()
        assert screen.run_button.isHidden()
        screen.setup_processing_mode.setCurrentIndex(screen.setup_processing_mode.findData("maximum"))
        for deep_mode in ("auto", "on", "off"):
            screen.setup_deep_analysis.setCurrentIndex(screen.setup_deep_analysis.findData(deep_mode))
            app.processEvents()
            assert viewmodel.project is not None and viewmodel.project.settings.deep_analysis == deep_mode
        for platform in ("tiktok", "reels", "shorts", "universal"):
            screen.setup_platform.setCurrentIndex(screen.setup_platform.findData(platform))
            app.processEvents()
            assert viewmodel.project is not None and viewmodel.project.settings.platform == platform
        app.processEvents()

        assert viewmodel.project is not None
        assert viewmodel.project.settings.processing_mode == "maximum"
        assert viewmodel.project.settings.deep_analysis == "off"
        assert viewmodel.project.settings.platform == "universal"
        reloaded = services.projects.load(project.project_id)
        assert reloaded.setup_state.last_estimate
        assert "первому анализу" in reloaded.setup_state.change_summary
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_video_preview_frames_drafts_and_final_outputs_as_a_phone(tmp_path: Path) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    vertical = tmp_path / "vertical.mp4"; vertical.write_bytes(b"vertical")
    preview = VideoPreview()

    try:
        preview.show_source(source)
        assert preview.presentation == "source"
        assert preview.video.maximumWidth() == 840
        assert preview.media_stage.maximumWidth() == 840
        assert preview.active_candidate.text() == "Исходное видео"

        preview.show_draft(vertical, "Тестовый момент")
        assert preview.presentation == "vertical"
        assert preview.video.size().width() == 270
        assert preview.video.size().height() == 480
        assert preview.media_stage.size().width() == 270
        assert preview.media_stage.size().height() == 480
        assert preview.active_candidate.text() == "Черновик · Тестовый момент"

        preview.show_final(vertical, "Тестовый момент")
        assert preview.presentation == "vertical"
        assert preview.active_candidate.text() == "Готовый ролик · Тестовый момент"
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_av1_webm_candidate_preview_uses_existing_proxy(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.webm"; source.write_bytes(b"source")
    preview = VideoPreview()
    monkeypatch.setattr(
        "app.gui.components.video_preview.probe_video",
        lambda _path: {"video_codec": "av1"},
    )

    try:
        assert preview._qt_can_decode_source(source) is False
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_candidate_switch_ignores_stale_player_position(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    preview = VideoPreview()
    stopped: list[bool] = []
    monkeypatch.setattr(preview, "_stop_at_range_end", lambda: stopped.append(True))

    try:
        preview._range_end_ms = 10
        preview._range_media_ready = False
        preview._position_changed(20)
        assert stopped == []

        preview._range_media_ready = True
        preview._position_changed(20)
        assert stopped == [True]
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_proxy_range_is_initialised_once_without_a_zero_seek(tmp_path: Path) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    preview = VideoPreview()

    class FakePlayer:
        def __init__(self) -> None:
            self.positions: list[int] = []

        def setPosition(self, position: int) -> None:
            self.positions.append(position)

        def play(self) -> None:
            return None

    player = FakePlayer()
    preview.player = player  # type: ignore[assignment]
    preview._path = source
    preview._range_start_ms = 0
    preview._range_end_ms = 100
    preview._range_autoplay = False
    preview._range_media_ready = False

    try:
        preview._media_status_changed(QMediaPlayer.MediaStatus.LoadedMedia)
        preview._media_status_changed(QMediaPlayer.MediaStatus.BufferedMedia)
        assert player.positions == []

        preview._range_media_ready = False
        preview._range_start_ms = 125
        preview._media_status_changed(QMediaPlayer.MediaStatus.LoadedMedia)
        assert player.positions == [125]
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_workflow_explains_when_only_some_selected_candidates_need_new_drafts(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    project.review_selected_candidate_ids = ["candidate-recommended", "candidate-other"]
    project.candidate_states["candidate-recommended"] = "rendered"
    project.candidate_states["candidate-other"] = "draft_failed"
    project.candidate_errors["candidate-other"] = "Draft FinalScript was not created."
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        app.processEvents()
        assert "Выбрано 2 из 3. Для 1 из них ещё нужен черновик." in screen.workflow_hint.text()
        assert "Черновик не создан." in "\n".join(
            label.text() for label in screen.findChildren(QLabel)
        )
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_ready_draft_needs_an_explicit_confirm_or_reject_before_production(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    draft_preview = tmp_path / "draft-preview.mp4"; draft_preview.write_bytes(b"preview")
    draft_artifact = tmp_path / "draft.json"
    write_json(draft_artifact, {
        "candidates": [{
            "candidate_id": "candidate-recommended",
            "preview": {"output_file": str(draft_preview)},
        }],
    })
    project.candidate_states["candidate-recommended"] = "draft_ready"
    project.candidate_draft_artifacts["candidate-recommended"] = str(draft_artifact)
    project.review_selected_candidate_ids = ["candidate-recommended"]
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    watched: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        VideoPreview, "show_draft", lambda _preview, path, title=None: watched.append((Path(path), str(title))),
    )
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()
        assert viewmodel.project is not None
        assert viewmodel.project.selected_candidate_ids == []
        assert screen.production_button.isHidden()
        assert "Посмотрите каждый" in screen.workflow_hint.text()

        watch = next(button for button in screen.findChildren(QPushButton) if button.text() == "Смотреть черновик")
        watch.click()
        app.processEvents()
        assert watched and watched[0][0] == draft_preview
        # Confirming a draft must leave that phone preview on screen instead
        # of silently returning to the source candidate.
        screen._active_candidate_id = "candidate-recommended"
        range_updates: list[object] = []
        monkeypatch.setattr(VideoPreview, "set_range", lambda *_args, **_kwargs: range_updates.append(True))

        approve = next(button for button in screen.findChildren(QPushButton) if button.text() == "Подтвердить")
        approve.click()
        app.processEvents()
        assert viewmodel.project.selected_candidate_ids == ["candidate-recommended"]
        assert screen.production_button.isVisible() and screen.production_button.isEnabled()
        assert range_updates == []

        reject = next(button for button in screen.findChildren(QPushButton) if button.text() == "Отклонить")
        reject.click()
        app.processEvents()
        assert viewmodel.project.selected_candidate_ids == []
        assert viewmodel.project.review_selected_candidate_ids == []
        assert viewmodel.project.candidate_states["candidate-recommended"] == "draft_ready"
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
