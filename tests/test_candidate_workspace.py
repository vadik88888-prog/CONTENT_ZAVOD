from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QUrl, Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QBoxLayout, QFrame, QGridLayout, QLabel, QMessageBox, QPushButton, QWidget

from app.gui.components import VideoPreview
from app.gui.main_window import MainWindow
from app.gui.models import DesktopSettings, ProcessingPhase, ProcessingSnapshot, ProjectStatus, RunKind, RunStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel
from app.utils import read_json, write_json


def _eligibility(eligible: bool = True) -> dict[str, object]:
    return {
        "schema_version": "6D.1",
        "config_version": "test",
        "state": "assessed",
        "eligible": eligible,
        "reason_codes": [] if eligible else ["SEMANTIC_INCOMPLETE"],
        "recoverable_issues": [],
        "required_boundary_actions": [],
        "evidence_refs": [],
    }


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
                "eligibility_decision": _eligibility(),
                "reasons": ["Сильное начало."], "preview": {"thumbnail": {"timestamp_seconds": 2.0}},
            },
            {
                "candidate_id": "candidate-other", "title": "Другой момент", "start_seconds": 19.0,
                "end_seconds": 29.0, "potential": "low", "confidence": 0.6, "recommended": False,
                "eligibility_decision": _eligibility(),
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
    project.source_metadata["video_codec"] = "av1"
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    preview_ranges: list[tuple[Path, float, float, Path | None, str | None, str | None]] = []

    def capture_preview_range(
        _preview, path, start_seconds, end_seconds, *, autoplay=True, cache_directory=None, candidate_title=None,
        source_codec=None,
    ) -> None:
        # The real set_range invalidates the source request before binding its
        # new range. Keep that lifecycle invariant in this lightweight UI
        # wiring stub as well.
        _preview._expected_source = QUrl()
        _preview.play_button.setEnabled(True)
        preview_ranges.append((
            Path(path), float(start_seconds), float(end_seconds), cache_directory, candidate_title, source_codec,
        ))

    monkeypatch.setattr(VideoPreview, "set_range", capture_preview_range)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)

        assert screen.content_scroll.widget() is screen.content_host
        assert screen.draft_button.isEnabled() is True
        assert screen.draft_button.text() == "Создать 1 рекомендованных"
        assert screen.view_all_button.text() == "Посмотреть все 2"
        assert screen.production_button.isEnabled() is False

        # The saved analysis opens directly at the single-purpose moments step;
        # the setup CTA is deliberately not duplicated on this screen.
        screen.show()
        app.processEvents()
        assert screen._flow_step == "candidates"
        assert screen.candidate_review.isVisible()
        assert screen.setup_card.isHidden()
        assert screen.run_button.isHidden()
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
            (project.source, 1.0, 18.0, project.directory / "preview-proxies", "Сильное начало", "av1"),
            (project.source, 19.0, 29.0, project.directory / "preview-proxies", "Другой момент", "av1"),
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

        # A completed render takes the reopened project straight to the
        # finished videos step, without offering a misleading new analysis.
        project.status = ProjectStatus.COMPLETED_WITH_WARNINGS
        project.candidate_states["candidate-recommended"] = "rendered"
        services.projects.save(project)
        screen._project_changed(project)
        assert screen._flow_step == "finished"
        assert screen.candidate_review.isVisible()
    finally:
        screen.close()


def test_top_n_primary_cta_selects_only_recommended_and_starts_drafts_without_analysis(
    tmp_path: Path, monkeypatch,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))
    launched: list[list[str]] = []
    monkeypatch.setattr(viewmodel, "build_drafts", lambda candidate_ids: launched.append(list(candidate_ids)))
    monkeypatch.setattr(viewmodel, "start_analysis", lambda: pytest.fail("Brain/Vision must not run for Top N"))

    try:
        screen.open(project)
        app.processEvents()
        QTest.mouseClick(screen.draft_button, Qt.MouseButton.LeftButton)
        app.processEvents()

        assert viewmodel.project is not None
        assert viewmodel.project.review_selected_candidate_ids == ["candidate-recommended"]
        assert launched == [["candidate-recommended"]]
    finally:
        screen.close()


def test_moments_view_all_and_select_all_exclude_ineligible_and_legacy_candidates_without_analysis(
    tmp_path: Path, monkeypatch,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    analysis_path = Path(project.analysis_artifact_path or "")
    analysis = read_json(analysis_path, {})
    eligible_candidates = [
        {
            "candidate_id": f"candidate-{index}", "title": f"Момент {index}",
            "start_seconds": float(index), "end_seconds": float(index + 10),
            "potential": "high", "confidence": 0.9 - index / 100,
            "recommended": index < 3,
            "eligibility_decision": _eligibility(),
        }
        for index in range(7)
    ]
    ineligible_id = "candidate-ineligible"
    legacy_id = "candidate-legacy"
    analysis["candidates"] = [
        *eligible_candidates,
        {
            "candidate_id": ineligible_id, "title": "Ineligible", "start_seconds": 20.0,
            "end_seconds": 30.0, "potential": "high", "confidence": 0.99,
            "recommended": True, "eligibility_decision": _eligibility(False),
        },
        {
            "candidate_id": legacy_id, "title": "Legacy unassessed", "start_seconds": 31.0,
            "end_seconds": 41.0, "potential": "high", "confidence": 0.98,
            "recommended": True,
        },
    ]
    write_json(analysis_path, analysis)
    candidate_ids = [f"candidate-{index}" for index in range(7)]
    project.candidate_states = {
        candidate_id: "analyzed" for candidate_id in [*candidate_ids, ineligible_id, legacy_id]
    }
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))
    monkeypatch.setattr(viewmodel, "start_analysis", lambda: pytest.fail("Brain/Vision must not run for Moments selection"))

    try:
        screen.open(project)
        app.processEvents()
        screen._toggle_candidate_selection(ineligible_id)
        assert viewmodel.project is not None
        assert viewmodel.project.review_selected_candidate_ids == []
        screen._view_all_candidates()
        app.processEvents()

        assert set(screen._candidate_cards) == set(candidate_ids)
        assert ineligible_id not in screen._candidate_selection_buttons
        assert legacy_id not in screen._candidate_selection_buttons
        recommended_ids = screen._recommended_candidate_ids()
        assert recommended_ids == candidate_ids[:3]
        assert set(recommended_ids) < set(candidate_ids)

        screen._select_all_candidates()
        app.processEvents()

        assert viewmodel.project is not None
        assert viewmodel.project.review_selected_candidate_ids == candidate_ids
    finally:
        screen.close()


def test_draft_button_mouse_click_starts_selected_drafts_shows_progress_and_opens_drafts(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    analysis_path = Path(project.analysis_artifact_path or "")
    analysis = read_json(analysis_path, {})
    third_candidate = {
        "candidate_id": "candidate-third", "title": "Третий момент", "start_seconds": 5.0,
        "end_seconds": 16.0, "potential": "medium", "confidence": 0.7, "recommended": True,
        "eligibility_decision": _eligibility(),
        "reasons": ["Подходит для черновика."], "preview": {"thumbnail": {"timestamp_seconds": 6.0}},
    }
    analysis["candidates"].append(third_candidate)
    write_json(analysis_path, analysis)
    selected_ids = ["candidate-recommended", "candidate-other", "candidate-third"]
    project.candidate_states = {candidate_id: "analyzed" for candidate_id in selected_ids}
    project.review_selected_candidate_ids = list(selected_ids)
    services.projects.save(project)
    calls: list[list[str]] = []
    started: list[object] = []
    draft_artifact = tmp_path / "draft.json"
    write_json(draft_artifact, {"candidates": [{"candidate_id": item} for item in selected_ids]})

    def prepare_draft(_services, current, candidate_ids):
        calls.append(list(candidate_ids))
        run = services.runs.create(
            current, {"candidate_ids": list(candidate_ids)}, {"path": str(current.source)}, "test",
            run_kind=RunKind.DRAFT,
        )
        for candidate_id in candidate_ids:
            current.candidate_states[candidate_id] = "draft_planning"
        current.status = ProjectStatus.PROCESSING
        return run, object()

    def finish_success(_services, current, run, _prepared):
        run.status = RunStatus.DRAFT_READY
        current.status = ProjectStatus.REVIEWING_CANDIDATES
        current.draft_artifact_path = str(draft_artifact)
        current.draft_id = "draft-test"
        for candidate_id in selected_ids:
            current.candidate_states[candidate_id] = "draft_ready"
            current.candidate_draft_artifacts[candidate_id] = str(draft_artifact)
        services.runs.save(run)
        services.projects.save(current)
        return run

    monkeypatch.setattr(DesktopServices, "prepare_draft", prepare_draft)
    monkeypatch.setattr(DesktopServices, "finish_success", finish_success)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(viewmodel.runner, "start", lambda prepared: started.append(prepared))
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()
        assert screen.draft_button.isVisible() and screen.draft_button.isEnabled()
        assert screen.draft_button.text() == "Создать черновики (3)"
        assert screen.draft_button.height() > 0

        QTest.mouseClick(screen.draft_button, Qt.MouseButton.LeftButton)
        app.processEvents()

        assert calls == [selected_ids]
        assert len(started) == 1
        assert viewmodel.snapshot.phase == ProcessingPhase.PREPARING
        assert screen.progress.isVisible()
        assert screen.progress.cancel_button.isVisible()
        assert screen._flow_step == "processing"
        assert viewmodel.project is not None
        assert all(viewmodel.project.candidate_states[item] == "draft_planning" for item in selected_ids)

        viewmodel._completed(0)
        app.processEvents()

        assert screen._flow_step == "drafts"
        assert viewmodel.project is not None
        assert all(viewmodel.project.candidate_states[item] == "draft_ready" for item in selected_ids)
        assert screen.candidate_review.isVisible()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_draft_button_mouse_click_surfaces_prepare_failure(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    project.review_selected_candidate_ids = ["candidate-recommended"]
    services.projects.save(project)
    monkeypatch.setattr(
        DesktopServices,
        "prepare_draft",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("draft preparation failed")),
    )
    errors: list[tuple[str, str, str]] = []

    def capture_dialog(box: QMessageBox) -> int:
        errors.append((box.windowTitle(), box.text(), box.informativeText()))
        return 0

    monkeypatch.setattr(QMessageBox, "exec", capture_dialog)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()
        QTest.mouseClick(screen.draft_button, Qt.MouseButton.LeftButton)
        app.processEvents()

        assert errors
        assert errors[0][0] == "Не удалось создать ролик"
        assert "draft preparation failed" not in "\n".join(errors[0])
        assert "Проверьте" in errors[0][2]
        assert not viewmodel.active
        assert screen._flow_step == "candidates"
        assert screen.progress.isHidden()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


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


def test_local_video_handoff_opens_settings(monkeypatch, tmp_path: Path) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, _project = _workspace(tmp_path)
    local_video = tmp_path / "local-video.mp4"
    local_video.write_bytes(b"local video")
    monkeypatch.setattr(services.pipeline, "inspect_source", lambda _path: {
        "duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0,
    })
    project = services.create_project(local_video)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)

    try:
        screen.open(project)

        assert project.status == ProjectStatus.SOURCE_READY
        assert project.source == local_video.resolve()
        assert screen._flow_step == "settings"
        assert not screen.setup_card.isHidden()
        assert screen.download_card.isHidden()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_link_project_reopens_at_download_then_moves_to_settings(monkeypatch, tmp_path: Path) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, _local_project = _workspace(tmp_path)
    project = services.projects.create_url("https://example.test/video", {"title": "Длинное видео", "estimated_size_bytes": 1_000_000})
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)

    try:
        screen.open(project)
        assert screen._flow_step == "download"
        assert not screen.download_card.isHidden()
        assert screen.setup_card.isHidden()
        assert screen.download_button.text() == "Скачать видео"

        project.source_spec.download_state = "cancelled"
        project.source_spec.error_message = "Загрузка была прервана при закрытии приложения. Её можно начать снова."
        services.projects.save(project)
        screen._project_changed(project)
        assert screen._flow_step == "download"
        assert screen.download_button.text() == "Скачать видео ещё раз"

        snapshot = ProcessingSnapshot(
            phase=ProcessingPhase.RUNNING,
            stage="download",
            elapsed_seconds=12.0,
            progress_fraction=0.42,
            transfer_downloaded="128MiB",
            transfer_total="301MiB",
            transfer_speed="1.5MiB/s",
            eta_seconds=17,
        )
        viewmodel.snapshot = snapshot
        screen._processing_changed(snapshot)
        assert screen._flow_step == "download"
        assert screen.download_card.isHidden()
        assert not screen.progress.isHidden()
        assert screen.progress.progress.value() == 42
        assert "128MiB из 301MiB" in screen.progress.detail.text()
        assert "Скорость: 1.5MiB/s" in screen.progress.detail.text()
        assert "Осталось:" in screen.progress.detail.text()

        downloaded = project.directory / "sources" / "downloaded.mp4"
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"completed download")
        monkeypatch.setattr(services.pipeline, "inspect_source", lambda _path: {
            "duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0,
        })
        viewmodel._launching = True
        viewmodel._after_download = "none"
        viewmodel._download_completed(str(downloaded))
        app.processEvents()

        assert viewmodel.project is not None
        assert viewmodel.project.source == downloaded.resolve()
        assert viewmodel.project.source_spec.download_state == "downloaded"
        assert viewmodel.project.status == ProjectStatus.SOURCE_READY
        assert screen._flow_step == "settings"
        assert not screen.setup_card.isHidden()
        assert screen.download_card.isHidden()
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


def test_range_selection_does_not_run_ffprobe_in_the_ui_click_path(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.webm"; source.write_bytes(b"source")
    preview = VideoPreview()
    queued: list[int] = []
    monkeypatch.setattr(preview, "_queue_source_load", lambda: queued.append(preview._selection_token))

    try:
        preview.set_range(source, 1.0, 3.0, autoplay=False)
        assert queued == [preview._selection_token]
        assert preview._path == source
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
        preview._path = tmp_path / "active.mp4"
        monkeypatch.setattr(preview, "_is_current_player_source", lambda: True)
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
        assert "Выбрано 2. Для 1 из них ещё нужен черновик." in screen.workflow_hint.text()
        assert "Черновик не создан." in "\n".join(
            label.text() for label in screen.findChildren(QLabel)
        )
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_failed_draft_exposes_retry_skip_and_log_without_raw_engine_diagnostics(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    candidate_id = "candidate-other"
    raw_error = "1 validation error for ProductionPlan: BOUNDARY_PAYOFF_LOST"
    project.review_selected_candidate_ids = [candidate_id]
    project.candidate_states[candidate_id] = "draft_failed"
    project.candidate_draft_statuses[candidate_id] = "failed"
    project.candidate_errors[candidate_id] = raw_error
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()

        button_texts = [button.text() for button in screen.findChildren(QPushButton)]
        assert "Повторить черновик" in button_texts
        assert "Продолжить без этого" in button_texts
        assert "Открыть журнал" in button_texts
        visible_copy = "\n".join(label.text() for label in screen.findChildren(QLabel))
        assert raw_error not in visible_copy

        skip = screen.findChild(QPushButton, f"skip-candidate-{candidate_id}")
        assert skip is not None
        QTest.mouseClick(skip, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert candidate_id not in viewmodel.project.review_selected_candidate_ids
        button_texts = [button.text() for button in screen.findChildren(QPushButton)]
        assert "Вернуть в набор" in button_texts
        assert "Повторить черновик" not in button_texts

        restore = screen.findChild(QPushButton, f"restore-candidate-{candidate_id}")
        assert restore is not None
        screen.review_list_scroll.ensureWidgetVisible(restore)
        app.processEvents()
        QTest.mouseClick(restore, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert candidate_id in viewmodel.project.review_selected_candidate_ids
        assert "Повторить черновик" in [button.text() for button in screen.findChildren(QPushButton)]
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_missing_ready_draft_artifact_is_offered_for_individual_rebuild(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    candidate_id = "candidate-recommended"
    project.review_selected_candidate_ids = [candidate_id]
    project.candidate_states[candidate_id] = "draft_ready"
    project.candidate_draft_statuses[candidate_id] = "ready"
    project.candidate_draft_artifacts[candidate_id] = str(tmp_path / "missing-draft.json")
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    requested: list[list[str]] = []
    monkeypatch.setattr(viewmodel, "build_drafts", lambda candidate_ids: requested.append(list(candidate_ids)))
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()

        assert screen.draft_button.isVisible()
        assert screen.draft_button.isEnabled()
        QTest.mouseClick(screen.draft_button, Qt.MouseButton.LeftButton)
        assert requested == [[candidate_id]]
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_completed_single_retry_keeps_other_failed_export_in_drafts(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    ready_id = "candidate-recommended"
    failed_id = "candidate-other"
    project.review_selected_candidate_ids = [ready_id, failed_id]
    project.selected_candidate_ids = [failed_id]
    project.candidate_states = {ready_id: "rendered", failed_id: "selected"}
    project.candidate_draft_statuses = {ready_id: "ready", failed_id: "ready"}
    project.candidate_approval_states = {ready_id: "approved", failed_id: "approved"}
    project.candidate_export_statuses = {ready_id: "ready", failed_id: "failed"}
    project.status = ProjectStatus.PARTIALLY_RENDERED
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    requested: list[list[str] | None] = []
    monkeypatch.setattr(screen, "_final_output_records", lambda _project: [object()])
    monkeypatch.setattr(
        screen, "_confirm_production_render", lambda candidate_ids=None: requested.append(candidate_ids),
    )

    try:
        # A result from the successful retry exists, but the remaining item
        # must still reopen the Drafts workspace rather than disappear behind
        # the final-output screen.
        assert screen._derive_flow_step(project) == "drafts"
        screen.project = project
        screen._retry_final_export(failed_id)
        assert requested == [[failed_id]]
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


def test_final_export_cta_uses_approved_project_state_not_clicked_bool(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    candidate_id = "candidate-recommended"
    draft_preview = tmp_path / "approved-preview.mp4"
    draft_preview.write_bytes(b"preview")
    draft_artifact = tmp_path / "approved-draft.json"
    write_json(draft_artifact, {
        "candidates": [{
            "candidate_id": candidate_id,
            "preview": {"output_file": str(draft_preview)},
        }],
    })
    project.review_selected_candidate_ids = [candidate_id]
    project.selected_candidate_ids = [candidate_id]
    project.candidate_states[candidate_id] = "selected"
    project.candidate_draft_statuses[candidate_id] = "ready"
    project.candidate_approval_states[candidate_id] = "approved"
    project.candidate_draft_artifacts[candidate_id] = str(draft_artifact)
    project.status = ProjectStatus.REVIEWING_CANDIDATES
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    dispatches: list[list[str]] = []
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        viewmodel,
        "render_selected",
        lambda candidate_ids=None: dispatches.append(list(candidate_ids or [])),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()

        assert screen.production_button.isVisible() and screen.production_button.isEnabled()
        QTest.mouseClick(screen.production_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert dispatches == [[candidate_id]]

        # An accidental bool reaching the method is harmless as well: it
        # still resolves the durable approved state, never ``dict.fromkeys``.
        screen._confirm_production_render(True)  # type: ignore[arg-type]
        assert dispatches == [[candidate_id], [candidate_id]]

        screen.project.selected_candidate_ids = []
        notices: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "information",
            lambda _parent, _title, message, *_args, **_kwargs: notices.append(message),
        )
        screen._confirm_production_render()
        assert notices and "подтвердите" in notices[0].lower()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_final_export_viewmodel_reports_unavailable_dispatch(tmp_path: Path) -> None:
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    errors: list[object] = []
    viewmodel.error_occurred.connect(errors.append)

    viewmodel.render_selected()
    assert getattr(errors[-1], "error_code") == "project_not_open"

    viewmodel.project = project
    viewmodel._launching = True
    viewmodel.render_selected()
    assert getattr(errors[-1], "error_code") == "render_already_active"


def test_review_selection_is_not_capped_by_persisted_top_n(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(
        viewmodel,
        "setup_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("recommendation must not cap review selection")),
    )

    try:
        candidate_ids = [f"candidate-{index}" for index in range(7)]
        project.candidate_states = {candidate_id: "analyzed" for candidate_id in candidate_ids}
        project.settings.clip_count = "3"
        services.set_review_selection(project, candidate_ids)
        assert project.review_selected_candidate_ids == candidate_ids
        assert screen._selection_limit(project) == 7
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


@pytest.mark.parametrize(
    ("width", "height", "compact"),
    # 760×480 is the desktop shell's logical minimum, covering 1280×720 at
    # 150% Windows scaling as well as the requested 100% desktop sizes.
    ((760, 480, True), (1280, 720, True), (1440, 900, True), (1920, 1080, False)),
)
def test_review_workspace_stacks_before_laptop_cards_can_overflow(
    tmp_path: Path, monkeypatch, width: int, height: int, compact: bool,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.resize(width, height)
        screen.show()
        app.processEvents()

        assert screen._compact_stage_layout is compact
        if compact:
            assert screen.review_preview_panel.y() > screen.review_list_panel.y()
            assert screen.review_inspector_panel.y() > screen.review_preview_panel.y()
        else:
            assert screen.review_preview_panel.y() == screen.review_list_panel.y()
        assert screen.review_list_scroll.horizontalScrollBar().maximum() == 0
        assert screen.content_scroll.horizontalScrollBar().maximum() == 0
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_compact_review_reflows_candidate_actions_and_boundary_controls(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    analysis_path = Path(project.analysis_artifact_path or "")
    analysis = read_json(analysis_path, {})
    analysis["candidates"][0]["title"] = "Очень длинное название момента для проверки адаптивной карточки " * 12
    analysis["candidates"][0]["reasons"] = ["Подробное объяснение выбора должно переноситься без скрытого горизонтального переполнения."]
    write_json(analysis_path, analysis)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(VideoPreview, "set_range", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.resize(760, 480)
        screen.show()
        app.processEvents()

        assert screen._header_layout.direction() == QBoxLayout.Direction.TopToBottom
        assert screen._stepper_layout.direction() == QBoxLayout.Direction.TopToBottom
        assert screen._processing_actions_layout.direction() == QBoxLayout.Direction.TopToBottom
        assert screen.review_list_scroll.horizontalScrollBar().maximum() == 0
        assert screen.review_inspector_scroll.horizontalScrollBar().maximum() == 0
        assert screen.content_scroll.horizontalScrollBar().maximum() == 0
        card = screen._candidate_cards["candidate-recommended"]
        assert card.width() <= screen.review_list_scroll.viewport().width()
        for button in card.findChildren(QPushButton):
            assert button.width() >= button.minimumSizeHint().width()

        preview = screen.findChild(QPushButton, "preview-candidate-candidate-recommended")
        assert preview is not None
        QTest.mouseClick(preview, Qt.MouseButton.LeftButton)
        app.processEvents()
        boundary_buttons = [
            button for button in screen.candidate_detail.findChildren(QPushButton)
            if button.text().startswith(("Начало", "Конец"))
        ]
        assert len(boundary_buttons) == 8
        assert all(button.width() >= button.minimumSizeHint().width() for button in boundary_buttons)
        grids = screen.candidate_detail.findChildren(QGridLayout)
        assert grids and grids[-1].columnCount() == 2
        assert screen.review_inspector_scroll.horizontalScrollBar().maximum() == 0

        # The same long card also fits the three-column Full HD review
        # workspace; its normal action rail never forces a hidden scrollbar.
        screen.resize(1920, 1080)
        app.processEvents()
        assert screen._header_layout.direction() == QBoxLayout.Direction.LeftToRight
        assert screen.review_list_scroll.horizontalScrollBar().maximum() == 0
        assert screen.review_inspector_scroll.horizontalScrollBar().maximum() == 0
        assert screen.content_scroll.horizontalScrollBar().maximum() == 0
        controls = screen.candidate_detail.findChild(QWidget, "candidateBoundaryControls")
        assert controls is not None
        detail_layout = screen.candidate_detail.layout()
        controls_index = next(
            index for index in range(detail_layout.count())
            if detail_layout.itemAt(index).widget() is controls
        )
        descriptive_labels = [
            detail_layout.itemAt(index).widget()
            for index in range(controls_index)
            if isinstance(detail_layout.itemAt(index).widget(), QLabel)
        ]
        assert descriptive_labels
        assert max(label.geometry().bottom() for label in descriptive_labels) < controls.geometry().top()
        assert controls.geometry().bottom() <= screen.candidate_detail.contentsRect().bottom()
        assert screen.candidate_detail.minimumHeight() >= detail_layout.totalSizeHint().height()
        assert all(
            "Выберите момент в списке" not in label.text()
            for label in screen.candidate_detail.findChildren(QLabel)
        )

        # When the inspector is vertically constrained, the content keeps its
        # natural order and the dedicated scroll area exposes the remainder.
        screen.review_inspector_scroll.setFixedHeight(280)
        app.processEvents()
        app.processEvents()
        assert screen.review_inspector_scroll.verticalScrollBar().maximum() > 0
        assert max(label.geometry().bottom() for label in descriptive_labels) < controls.geometry().top()
        card = screen._candidate_cards["candidate-recommended"]
        assert card.width() <= screen.review_list_scroll.viewport().width()
        for button in card.findChildren(QPushButton):
            assert button.width() >= button.minimumSizeHint().width()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_long_candidate_title_is_elided_with_its_full_tooltip(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    long_title = "Очень длинное название момента " * 24
    analysis_path = Path(project.analysis_artifact_path or "")
    analysis = read_json(analysis_path, {})
    analysis["candidates"][0]["title"] = long_title
    write_json(analysis_path, analysis)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.resize(1280, 720)
        screen.show()
        app.processEvents()

        title = screen.findChild(QLabel, "candidateTitle")
        assert title is not None
        assert title.toolTip() == long_title
        assert title.text().endswith("…")
        assert screen.content_scroll.horizontalScrollBar().maximum() == 0
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


@pytest.mark.parametrize(("width", "height"), ((760, 480), (1280, 720), (1440, 900), (1920, 1080)))
def test_desktop_shell_keeps_navigation_controls_and_project_header_unclipped(
    tmp_path: Path, monkeypatch, width: int, height: int,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    project.name = "Очень длинное имя проекта для проверки адаптивного заголовка " * 12
    services.projects.save(project)
    services.settings.onboarding_completed = True
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    window = MainWindow(services)

    try:
        window.show_project(project)
        window.resize(width, height)
        window.show()
        app.processEvents()

        for button in (
            window.new_button,
            window.projects_button,
            window.settings_button,
            window.help_button,
        ):
            assert button.toolTip()
            assert button.contentsRect().width() >= button.minimumSizeHint().width()
        title = window.project_screen.title
        assert title.toolTip() == project.name
        assert title.text().endswith("…")
        assert window.project_screen.content_scroll.horizontalScrollBar().maximum() == 0
        if width == 760:
            assert window.project_screen._review_action_layout.direction() == QBoxLayout.Direction.TopToBottom
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_drafts_hide_moment_selection_toolbar_and_filters(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    draft_preview = tmp_path / "draft-preview.mp4"
    draft_preview.write_bytes(b"preview")
    draft_artifact = tmp_path / "draft.json"
    write_json(draft_artifact, {"candidates": [{
        "candidate_id": "candidate-recommended",
        "preview": {"output_file": str(draft_preview)},
    }]})
    project.candidate_states["candidate-recommended"] = "draft_ready"
    project.candidate_draft_artifacts["candidate-recommended"] = str(draft_artifact)
    project.review_selected_candidate_ids = ["candidate-recommended"]
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_draft", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        screen.show()
        app.processEvents()

        assert screen._flow_step == "drafts"
        assert not any(button.text() == "Выбрать рекомендованные" for button in screen.findChildren(QPushButton))
        assert not any(button.text() == "Снять выбор" for button in screen.findChildren(QPushButton))
        assert not screen.findChildren(QFrame, "reviewFilters")
        assert any(button.text() == "Исходный фрагмент" for button in screen.findChildren(QPushButton))
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_selection_update_does_not_reload_unchanged_active_preview(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    set_ranges: list[tuple[float, float]] = []
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        VideoPreview,
        "set_range",
        lambda _preview, _path, start, end, **_kwargs: set_ranges.append((float(start), float(end))),
    )
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        candidate = screen._review_candidates_by_id["candidate-recommended"]
        screen._preview_candidate(candidate)
        assert set_ranges == [(1.0, 18.0)]

        viewmodel.set_review_selection(["candidate-recommended"])
        app.processEvents()
        assert set_ranges == [(1.0, 18.0)]
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_reopened_project_restores_the_persisted_candidate_preview(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    project.active_preview_candidate_id = "candidate-other"
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    set_ranges: list[tuple[float, float, bool]] = []
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        VideoPreview,
        "set_range",
        lambda _preview, _path, start, end, *, autoplay=True, **_kwargs:
        set_ranges.append((float(start), float(end), autoplay)),
    )
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        app.processEvents()
        assert screen._active_candidate_id == "candidate-other"
        assert set_ranges == [(19.0, 29.0, False)]
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_boundary_change_replaces_a_stale_draft_preview_with_source_range(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    draft_preview = tmp_path / "draft-preview.mp4"
    draft_preview.write_bytes(b"preview")
    draft_artifact = tmp_path / "draft.json"
    write_json(draft_artifact, {"candidates": [{
        "candidate_id": "candidate-recommended",
        "preview": {"output_file": str(draft_preview)},
    }]})
    candidate_id = "candidate-recommended"
    project.candidate_states[candidate_id] = "draft_ready"
    project.candidate_draft_artifacts[candidate_id] = str(draft_artifact)
    project.review_selected_candidate_ids = [candidate_id]
    project.active_preview_candidate_id = candidate_id
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    ranges: list[tuple[float, float, bool]] = []
    monkeypatch.setattr(VideoPreview, "show_draft", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        VideoPreview,
        "set_range",
        lambda _preview, _path, start, end, *, autoplay=True, **_kwargs:
        ranges.append((float(start), float(end), autoplay)),
    )
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        assert ranges == []
        project.candidate_boundary_overrides[candidate_id] = {"start": 2.0, "end": 19.0}
        screen._project_changed(project)
        assert ranges == [(2.0, 19.0, False)]
        assert screen._active_preview_kind == "source-range"
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_late_thumbnail_failure_for_old_request_does_not_replace_current_card(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    current_thumbnail = tmp_path / "current-thumbnail.jpg"
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: current_thumbnail)

    try:
        screen.open(project)
        candidate_id = "candidate-recommended"
        label = screen._candidate_thumbnail_labels[candidate_id][0]
        assert "загружается" in label.text()

        screen._thumbnail_unavailable(candidate_id, str(tmp_path / "old-thumbnail.jpg"))
        assert "загружается" in label.text()

        screen._thumbnail_unavailable(candidate_id, str(current_thumbnail))
        assert label.text() == "Кадр\nнедоступен"
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_processing_stage_rows_follow_the_active_job_kind(tmp_path: Path) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, _project = _workspace(tmp_path)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)

    try:
        viewmodel.run = type("DraftRun", (), {"run_kind": "draft"})()  # type: ignore[assignment]
        screen._update_processing_stages(ProcessingSnapshot(
            phase=ProcessingPhase.RUNNING, stage="draft_render",
        ))
        draft_rows = [label.text() for label in screen.processing_stage_labels.values()]
        assert any("черновики" in row.lower() for row in draft_rows)
        assert not any("речь и структуру" in row.lower() for row in draft_rows)
        assert not any("сильные моменты" in row.lower() for row in draft_rows)

        viewmodel.run = type("ExportRun", (), {"run_kind": "selected_render"})()  # type: ignore[assignment]
        screen._update_processing_stages(ProcessingSnapshot(
            phase=ProcessingPhase.RUNNING, stage="production_export",
        ))
        export_rows = [label.text() for label in screen.processing_stage_labels.values()]
        assert any("готовые ролики" in row.lower() for row in export_rows)
        assert not any("речь и структуру" in row.lower() for row in export_rows)
        assert not any("сильные моменты" in row.lower() for row in export_rows)
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_failed_final_export_keeps_draft_actionable_without_raw_diagnostics(tmp_path: Path, monkeypatch) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _workspace(tmp_path)
    draft_preview = tmp_path / "draft-preview.mp4"
    draft_preview.write_bytes(b"preview")
    draft_artifact = tmp_path / "draft.json"
    write_json(draft_artifact, {"candidates": [{
        "candidate_id": "candidate-recommended",
        "preview": {"output_file": str(draft_preview)},
    }]})
    candidate_id = "candidate-recommended"
    project.candidate_states[candidate_id] = "selected"
    project.candidate_draft_artifacts[candidate_id] = str(draft_artifact)
    project.review_selected_candidate_ids = [candidate_id]
    project.selected_candidate_ids = [candidate_id]
    project.candidate_draft_statuses[candidate_id] = "ready"
    project.candidate_approval_states[candidate_id] = "approved"
    project.candidate_export_statuses[candidate_id] = "failed"
    project.candidate_errors[candidate_id] = "ffmpeg command C:/secret-args failed"
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    monkeypatch.setattr(VideoPreview, "show_draft", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(viewmodel)
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        app.processEvents()
        texts = [label.text() for label in screen.findChildren(QLabel)]
        assert any("Готовый ролик не создан" in text for text in texts)
        assert not any("secret-args" in text for text in texts)
        assert any(button.text() == "Повторить экспорт" for button in screen.findChildren(QPushButton))
        assert any(button.text() == "Открыть журнал" for button in screen.findChildren(QPushButton))
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
