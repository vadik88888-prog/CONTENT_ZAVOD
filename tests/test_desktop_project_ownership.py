from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QThreadPool
from PySide6.QtWidgets import QApplication

from app.gui.models import DesktopSettings, ProcessingPhase, ProcessingSnapshot, ProjectStatus, RunStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme
from app.gui.viewmodels.project_viewmodel import ProjectViewModel


def _drain_background() -> None:
    application = QCoreApplication.instance() or QCoreApplication([])
    assert QThreadPool.globalInstance().waitForDone(5_000)
    application.processEvents()
    application.processEvents()


def _services(tmp_path: Path) -> DesktopServices:
    data = tmp_path / "desktop-data"
    store = DesktopProjectStore(data)
    root = Path(__file__).resolve().parents[1]
    return DesktopServices(
        engine_root=root,
        settings_store=SettingsStore(data),
        settings=DesktopSettings.defaults(data),
        projects=store,
        runs=RunHistoryStore(store),
        pipeline=PipelineFacade(root),
        system=SystemService(root),
    )


def _project(services: DesktopServices, tmp_path: Path, name: str):
    source = tmp_path / f"{name}.mp4"
    source.write_bytes(b"source")
    project = services.projects.create(source)
    project.name = name
    project.status = ProjectStatus.SOURCE_READY
    services.projects.save(project)
    return project


def _active_context(tmp_path: Path):
    services = _services(tmp_path)
    owner = _project(services, tmp_path, "Проект A")
    other = _project(services, tmp_path, "Проект B")
    run = services.runs.create(owner, {}, {"path": str(owner.source)}, "test", run_kind="analysis")
    run.status = RunStatus.RUNNING
    services.runs.save(run)
    owner.status = ProjectStatus.ANALYZING
    owner.latest_run_id = run.run_id
    services.projects.save(owner)
    prepared = PreparedPipelineRun(
        program="python",
        arguments=[],
        working_directory=tmp_path,
        state_path=tmp_path / "state.json",
        report_path=tmp_path / "report.json",
        output_directory=tmp_path / "output",
        runtime_config_path=tmp_path / "runtime.yaml",
        run_id=run.run_id,
        project_id=owner.project_id,
    )
    viewmodel = ProjectViewModel(services)
    viewmodel.open(owner)
    viewmodel.run = run
    viewmodel.prepared = prepared
    viewmodel._launching = True
    viewmodel._bind_job(owner, ProcessingSnapshot(
        ProcessingPhase.RUNNING, stage="transcription", message="Распознаём речь", elapsed_seconds=19,
    ))
    return services, viewmodel, owner, other, run, prepared


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires QApplication")
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(load_theme())
    return app


def test_running_project_keeps_its_progress_across_projects_navigation(tmp_path: Path) -> None:
    services, viewmodel, owner, other, _run, _prepared = _active_context(tmp_path)
    owner_snapshot = viewmodel.snapshot
    card_presentation = services.presentation(owner)
    assert card_presentation.active
    assert card_presentation.flow_step == "processing"
    assert card_presentation.status_label == "Ищем моменты"

    viewmodel.open(other)
    assert viewmodel.project and viewmodel.project.project_id == other.project_id
    assert viewmodel.snapshot.phase == ProcessingPhase.IDLE
    assert viewmodel.blocked_by_other_project

    viewmodel._stage_changed("candidate_generation", "Ищем сильные моменты")
    assert viewmodel.snapshot.phase == ProcessingPhase.IDLE

    viewmodel.open(services.projects.load(owner.project_id))
    assert viewmodel.snapshot is owner_snapshot
    assert viewmodel.snapshot.stage == "candidate_generation"
    assert viewmodel.snapshot.message == "Ищем сильные моменты"
    assert viewmodel.owns_active_job


def test_url_download_keeps_owner_while_another_project_is_open(tmp_path: Path, monkeypatch) -> None:
    services = _services(tmp_path)
    owner = services.projects.create_url(
        "https://example.test/video",
        {"title": "URL-проект"},
    )
    other = _project(services, tmp_path, "Локальный проект")
    viewmodel = ProjectViewModel(services)
    viewmodel.open(owner)
    monkeypatch.setattr(viewmodel.source_downloader, "download", lambda *_args: None)

    viewmodel.start_download()
    presentation = services.presentation(services.projects.load(owner.project_id))
    assert presentation.active
    assert presentation.status_label == "Загружаем видео"

    viewmodel.open(other)
    viewmodel._download_progress(SimpleNamespace(
        fraction=0.4, speed="1 MiB/s", downloaded="4 MiB", total="10 MiB", eta_seconds=6,
    ))
    assert viewmodel.project and viewmodel.project.project_id == other.project_id
    assert viewmodel.snapshot.phase == ProcessingPhase.IDLE

    viewmodel._download_cancelled()
    restored_owner = services.projects.load(owner.project_id)
    restored_other = services.projects.load(other.project_id)
    assert restored_owner.source_spec.download_state == "cancelled"
    assert restored_other.status == ProjectStatus.SOURCE_READY
    assert not viewmodel.active


def test_low_confidence_language_requires_choice_before_analysis_and_persists_it(
    tmp_path: Path, monkeypatch,
) -> None:
    services = _services(tmp_path)
    project = _project(services, tmp_path, "Язык речи")
    viewmodel = ProjectViewModel(services)
    viewmodel.open(project)
    viewmodel._launching = True
    viewmodel._bind_job(project, ProcessingSnapshot(
        ProcessingPhase.PREPARING, stage="language_probe", message="Определяем язык речи",
    ))
    choices: list[object] = []
    launches: list[tuple[str, str]] = []
    viewmodel.language_choice_required.connect(choices.append)
    monkeypatch.setattr(
        viewmodel, "_start_prepared_job",
        lambda mode, owner: launches.append((mode, owner.settings.speech_language)),
    )

    viewmodel._language_probe_ready("analysis", project, SimpleNamespace(
        language="lt", confidence=0.369, is_confident=False,
    ))

    assert len(choices) == 1
    assert launches == []
    assert not viewmodel.active
    assert viewmodel.snapshot.message == "Нужно выбрать язык речи"

    viewmodel.choose_speech_language("ru")

    assert launches == [("analysis", "ru")]
    assert services.projects.load(project.project_id).settings.speech_language == "ru"


def test_processing_rows_do_not_overlap_after_dynamic_state_is_revealed(tmp_path: Path) -> None:
    app = _application()
    _services_value, viewmodel, owner, _other, _run, _prepared = _active_context(tmp_path)
    screen = ProjectScreen(viewmodel)
    try:
        screen.open(owner)
        screen.resize(1100, 720)
        screen.show()
        app.processEvents()
        app.processEvents()

        assert screen.progress.detail.isHidden() is False
        assert screen.progress.cancel_button.isHidden() is False
        assert screen.progress.geometry().bottom() < screen.processing_stages.geometry().top()
        assert screen.processing_stages.geometry().bottom() < screen.processing_actions.geometry().top()

        screen.resize(760, 480)
        app.processEvents()
        screen._processing_changed(ProcessingSnapshot(
            ProcessingPhase.RUNNING,
            stage="candidate_generation",
            message="Ищем сильные моменты",
            elapsed_seconds=1800,
            long_stage_warning="Этап выполняется дольше обычного",
        ))
        app.processEvents()
        app.processEvents()
        assert screen.progress.continue_waiting_button.isHidden() is False
        assert screen.progress.geometry().bottom() < screen.processing_stages.geometry().top()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("terminal", ("completed", "failed", "cancelled"))
def test_terminal_callback_mutates_only_owner_while_other_project_is_open(
    tmp_path: Path, monkeypatch, terminal: str,
) -> None:
    services, viewmodel, owner, other, run, _prepared = _active_context(tmp_path)
    viewmodel.open(other)
    seen_processing: list[ProcessingSnapshot] = []
    viewmodel.processing_changed.connect(seen_processing.append)

    if terminal == "completed":
        def finish_success(_services, project, current_run, _prepared):
            assert project.project_id == owner.project_id
            project.status = ProjectStatus.COMPLETED
            current_run.status = RunStatus.COMPLETED
            services.projects.save(project)
            services.runs.save(current_run)
            return current_run

        monkeypatch.setattr(DesktopServices, "finish_success", finish_success)
        viewmodel._completed(0)
    elif terminal == "failed":
        viewmodel.prepared = None

        def finish_failure(_services, project, current_run, _message, _details):
            assert project.project_id == owner.project_id
            project.status = ProjectStatus.FAILED
            current_run.status = RunStatus.FAILED
            services.projects.save(project)
            services.runs.save(current_run)
            return current_run

        monkeypatch.setattr(DesktopServices, "finish_failure", finish_failure)
        viewmodel._failed("failure")
    else:
        viewmodel.prepared = None

        def finish_cancelled(_services, project, current_run):
            assert project.project_id == owner.project_id
            project.status = ProjectStatus.CANCELLED
            current_run.status = RunStatus.CANCELLED
            services.projects.save(project)
            services.runs.save(current_run)
            return current_run

        monkeypatch.setattr(DesktopServices, "finish_cancelled", finish_cancelled)
        viewmodel._cancelled()

    _drain_background()

    assert services.projects.load(owner.project_id).status == terminal
    restored_other = services.projects.load(other.project_id)
    assert restored_other.status == ProjectStatus.SOURCE_READY
    assert viewmodel.project and viewmodel.project.project_id == other.project_id
    assert not viewmodel.active
    assert not viewmodel.blocked_by_other_project
    assert seen_processing and seen_processing[-1].phase == ProcessingPhase.IDLE
    assert services.runs.load(owner.project_id, run.run_id).status == terminal


def test_active_project_delete_and_second_heavy_job_are_blocked(tmp_path: Path) -> None:
    services, _viewmodel, owner, other, _run, _prepared = _active_context(tmp_path)

    with pytest.raises(InputValidationError, match="Нельзя удалить проект"):
        services.delete_project(owner.project_id)
    with pytest.raises(InputValidationError, match="второй тяжёлый запуск"):
        services.prepare_analysis(other)

    services.delete_project(other.project_id)
    assert not services.projects.project_path(other.project_id).exists()


def test_other_project_shows_clear_disabled_heavy_cta(tmp_path: Path) -> None:
    app = _application()
    _services_value, viewmodel, _owner, other, _run, _prepared = _active_context(tmp_path)
    screen = ProjectScreen(viewmodel)
    try:
        screen.open(other)
        screen.resize(1100, 720)
        screen.show()
        app.processEvents()
        assert screen.setup_start_button.isEnabled() is False
        assert "Проект A" in screen.flow_hint.text()
        assert "второй тяжёлый запуск" in screen.flow_hint.text()
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_project_screen_reloads_fresh_state_and_uses_shared_presentation(tmp_path: Path) -> None:
    app = _application()
    services = _services(tmp_path)
    project = _project(services, tmp_path, "Свежий проект")
    stale = services.projects.load(project.project_id)
    stale.status = ProjectStatus.ANALYZING
    project.name = "Сохранённое имя"
    project.status = ProjectStatus.SOURCE_READY
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    try:
        screen.open(stale)
        app.processEvents()
        assert viewmodel.project and viewmodel.project.name == "Сохранённое имя"
        presentation = services.presentation(viewmodel.project, snapshot=viewmodel.snapshot)
        assert screen._flow_step == presentation.flow_step == "settings"
        assert screen.status.text() == presentation.status_label == "Готов к настройке"
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_interrupted_analysis_has_consistent_recovery_title_and_cta(tmp_path: Path) -> None:
    app = _application()
    services = _services(tmp_path)
    project = _project(services, tmp_path, "Прерванный проект")
    run = services.runs.create(project, {}, {"path": str(project.source)}, "test", run_kind="analysis")
    run.status = RunStatus.INTERRUPTED
    services.runs.save(run)
    project.latest_run_id = run.run_id
    project.status = ProjectStatus.INTERRUPTED
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    try:
        screen.open(project)
        screen.show()
        app.processEvents()

        presentation = services.presentation(viewmodel.project)
        assert presentation.status_label == "Работа прервана"
        assert screen.status.text() == presentation.status_label
        assert screen.flow_title.text() == presentation.status_label
        assert screen.progress.retry_button.text() == "Повторить поиск моментов"
        assert screen.progress.retry_button.isHidden() is False
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
