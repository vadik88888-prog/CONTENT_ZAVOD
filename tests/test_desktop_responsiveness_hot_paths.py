from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import QApplication

import app.gui.services.run_projection as run_projection_module
import app.gui.services.desktop_services as desktop_services_module
from app.gui.models import (
    DesktopSettings,
    ProcessingPhase,
    ProcessingSnapshot,
    ProjectStatus,
    ProjectRun,
    RunKind,
    RunStatus,
)
from app.gui.screens.project_screen import ProjectScreen
from app.gui.screens.projects_screen import ProjectsScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices, ValidatedSource
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.run_projection import RunProjectionCache
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel, ProjectsViewModel
from app.utils import write_json


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires QApplication")
    return QApplication.instance() or QApplication([])


def _services(tmp_path: Path) -> DesktopServices:
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
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


def _process_until(application: QApplication, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        application.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for background Qt work.")
        time.sleep(0.005)


def _drain_background(application: QApplication) -> None:
    assert QThreadPool.globalInstance().waitForDone(5_000)
    application.processEvents()
    application.processEvents()


def _run(report_path: Path, *, manifest_path: Path | None = None) -> ProjectRun:
    execution: dict[str, object] = {}
    if manifest_path is not None:
        execution["engine_paths"] = {"manifest_path": str(manifest_path)}
    return ProjectRun(
        run_id="run-001",
        project_id="project-001",
        started_at="2026-08-13T00:00:00+00:00",
        finished_at="2026-08-13T00:01:00+00:00",
        status=RunStatus.COMPLETED,
        settings_snapshot={"execution": execution},
        source_snapshot={},
        pipeline_version="test",
        report_path=str(report_path),
        run_kind=RunKind.SELECTED_RENDER,
    )


def test_manifest_projection_prevents_legacy_report_read(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "result.mp4"
    output.write_bytes(b"mp4")
    write_json(report, {"production_render": {"status": "completed", "output_file": "wrong.mp4"}})
    write_json(manifest, {
        "run_id": "run-001",
        "project_id": "project-001",
        "primary_results": [{
            "candidate_id": "candidate-001",
            "output_file": str(output),
            "status": "completed",
            "run_id": "run-001",
        }],
    })
    reads: list[str] = []
    original = run_projection_module.read_json

    def tracked(path: Path, *values, **options):
        reads.append(Path(path).name)
        return original(path, *values, **options)

    monkeypatch.setattr(run_projection_module, "read_json", tracked)
    projection = RunProjectionCache().for_run(_run(report, manifest_path=manifest))

    assert [item.candidate_id for item in projection.primary_results] == ["candidate-001"]
    assert reads == ["manifest.json"]


def test_legacy_projection_cache_is_keyed_by_path_size_and_mtime(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report.json"
    write_json(report, {"production_render": {"status": "completed", "output_file": "first.mp4"}})
    reads = 0
    original = run_projection_module.read_json

    def tracked(path: Path, *values, **options):
        nonlocal reads
        reads += 1
        return original(path, *values, **options)

    monkeypatch.setattr(run_projection_module, "read_json", tracked)
    cache = RunProjectionCache()
    run = _run(report)
    first = cache.for_run(run)
    second = cache.for_run(run)
    write_json(report, {"production_render": {"status": "completed", "output_file": "second-longer.mp4"}})
    third = cache.for_run(run)

    assert reads == 2
    assert first is second
    assert first.primary_results[0].output_file == "first.mp4"
    assert third.primary_results[0].output_file == "second-longer.mp4"


def test_project_presentation_uses_project_run_without_reading_report(tmp_path: Path, monkeypatch) -> None:
    services = _services(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    project = services.projects.create(source)
    report = tmp_path / "large-report.json"
    write_json(report, {"primary_results": []})
    output = tmp_path / "final.mp4"
    output.write_bytes(b"mp4")
    run = services.runs.create(project, {}, {}, "test", run_kind=RunKind.SELECTED_RENDER)
    run.status = RunStatus.COMPLETED
    run.finished_at = "2026-08-13T00:01:00+00:00"
    run.report_path = str(report)
    run.artifact_paths = [str(output)]
    project.latest_run_id = run.run_id

    def forbidden_read(*_values, **_options):
        raise AssertionError("presentation must not read report.json")

    monkeypatch.setattr(desktop_services_module, "read_json", forbidden_read)
    presentation = services.presentation(project, runs=[run])

    assert presentation.flow_step == "finished"
    assert presentation.status_label == "Ролики готовы"


def test_hidden_projects_screen_only_marks_dirty_until_shown(tmp_path: Path, monkeypatch) -> None:
    application = _application()
    services = _services(tmp_path)
    viewmodel = ProjectsViewModel(services)
    screen = ProjectsScreen(viewmodel)
    refreshes = 0

    def refresh() -> None:
        nonlocal refreshes
        refreshes += 1

    monkeypatch.setattr(viewmodel, "refresh", refresh)
    try:
        screen.mark_dirty()
        application.processEvents()
        assert refreshes == 0

        screen.show()
        application.processEvents()
        application.processEvents()
        assert refreshes == 1
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()


def test_project_and_run_signals_coalesce_into_one_refresh(tmp_path: Path, monkeypatch) -> None:
    application = _application()
    services = _services(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    project = services.projects.create(source)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    calls = {"project": 0, "runs": 0}

    monkeypatch.setattr(
        screen,
        "_project_changed",
        lambda _project: calls.__setitem__("project", calls["project"] + 1),
    )
    monkeypatch.setattr(
        screen,
        "_runs_changed",
        lambda _runs, **_options: calls.__setitem__("runs", calls["runs"] + 1),
    )
    try:
        viewmodel.project_changed.emit(project)
        viewmodel.runs_changed.emit([])
        viewmodel.project_changed.emit(project)
        viewmodel.runs_changed.emit([])
        application.processEvents()
        assert calls == {"project": 1, "runs": 1}
    finally:
        screen.deleteLater()
        application.processEvents()


def test_progress_telemetry_tick_is_constant_projection_free_work(tmp_path: Path, monkeypatch) -> None:
    application = _application()
    services = _services(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    project = services.projects.create(source)
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    try:
        screen.open(project)
        application.processEvents()
        application.processEvents()

        def forbidden_presentation(*_values, **_options):
            raise AssertionError("progress telemetry must not derive persisted presentation")

        monkeypatch.setattr(DesktopServices, "presentation", forbidden_presentation)
        structural_updates = 0
        original_update = screen._update_stage_context

        def tracked_update(value):
            nonlocal structural_updates
            structural_updates += 1
            return original_update(value)

        monkeypatch.setattr(screen, "_update_stage_context", tracked_update)
        snapshot = ProcessingSnapshot(
            ProcessingPhase.RUNNING,
            stage="render",
            message="Создаём ролики",
            elapsed_seconds=1.0,
            progress_fraction=0.25,
        )
        viewmodel.snapshot = snapshot
        screen._processing_changed(snapshot)
        snapshot.elapsed_seconds = 1.5
        snapshot.progress_fraction = 0.5
        screen._processing_changed(snapshot)

        assert structural_updates == 1
        assert screen.progress.progress.value() == 50
    finally:
        monkeypatch.undo()
        screen.deleteLater()
        application.processEvents()


def test_local_source_probe_returns_immediately_and_keeps_gui_events_running(tmp_path: Path, monkeypatch) -> None:
    application = _application()
    services = _services(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    entered = threading.Event()
    release = threading.Event()
    worker_was_gui_thread: list[bool] = []

    def delayed_validation(_services: DesktopServices, path: str) -> ValidatedSource:
        worker_was_gui_thread.append(QThread.currentThread() == application.thread())
        entered.set()
        assert release.wait(5.0)
        return ValidatedSource(Path(path).resolve(), {
            "duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0,
        })

    monkeypatch.setattr(DesktopServices, "validate_source", delayed_validation)
    viewmodel = ProjectsViewModel(services)
    created: list[object] = []
    delivery_was_gui_thread: list[bool] = []

    def receive_project(project: object) -> None:
        delivery_was_gui_thread.append(QThread.currentThread() == application.thread())
        created.append(project)

    viewmodel.project_created.connect(receive_project)

    started = time.perf_counter()
    viewmodel.create(str(source))
    callback_seconds = time.perf_counter() - started
    _process_until(application, entered.is_set)
    gui_ticks: list[bool] = []
    QTimer.singleShot(0, lambda: gui_ticks.append(True))
    application.processEvents()

    assert callback_seconds < 0.05
    assert gui_ticks == [True]
    assert worker_was_gui_thread == [False]
    assert created == []

    release.set()
    _drain_background(application)
    assert len(created) == 1
    assert delivery_was_gui_thread == [True]


def test_completion_post_process_returns_immediately_and_delivers_only_finalized_run(
    tmp_path: Path, monkeypatch,
) -> None:
    application = _application()
    services = _services(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    project = services.projects.create(source)
    run = services.runs.create(project, {}, {"path": str(source)}, "test", run_kind=RunKind.SELECTED_RENDER)
    entered = threading.Event()
    release = threading.Event()
    worker_was_gui_thread: list[bool] = []

    def delayed_completion(_services, owner, current_run, _prepared):
        worker_was_gui_thread.append(QThread.currentThread() == application.thread())
        entered.set()
        assert release.wait(5.0)
        owner.status = ProjectStatus.COMPLETED
        current_run.status = RunStatus.COMPLETED
        services.projects.save(owner)
        services.runs.save(current_run)
        return current_run

    monkeypatch.setattr(DesktopServices, "finish_success", delayed_completion)
    viewmodel = ProjectViewModel(services)
    viewmodel.project = project
    viewmodel.run = run
    viewmodel.prepared = object()  # type: ignore[assignment]
    viewmodel._launching = True
    finished: list[ProjectRun] = []
    delivery_was_gui_thread: list[bool] = []

    def receive_run(completed: ProjectRun) -> None:
        delivery_was_gui_thread.append(QThread.currentThread() == application.thread())
        finished.append(completed)

    viewmodel.run_finished.connect(receive_run)

    started = time.perf_counter()
    viewmodel._completed(0)
    callback_seconds = time.perf_counter() - started
    _process_until(application, entered.is_set)
    gui_ticks: list[bool] = []
    QTimer.singleShot(0, lambda: gui_ticks.append(True))
    application.processEvents()

    assert callback_seconds < 0.05
    assert gui_ticks == [True]
    assert worker_was_gui_thread == [False]
    assert finished == []

    release.set()
    _drain_background(application)
    assert finished == [run]
    assert delivery_was_gui_thread == [True]
    assert viewmodel.snapshot.phase == ProcessingPhase.COMPLETED


def test_cancel_during_download_probe_discards_stale_validated_source(tmp_path: Path, monkeypatch) -> None:
    application = _application()
    services = _services(tmp_path)
    project = services.projects.create_url("https://example.test/video", {"title": "Video"})
    downloaded = project.directory / "sources" / "downloaded.mp4"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(b"source")
    entered = threading.Event()
    release = threading.Event()

    def delayed_validation(_services: DesktopServices, path: str) -> ValidatedSource:
        entered.set()
        assert release.wait(5.0)
        return ValidatedSource(Path(path).resolve(), {
            "duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0,
        })

    monkeypatch.setattr(DesktopServices, "validate_source", delayed_validation)
    viewmodel = ProjectViewModel(services)
    viewmodel.open(project)
    viewmodel._launching = True
    viewmodel._after_download = "none"
    viewmodel._download_completed(str(downloaded))
    _process_until(application, entered.is_set)

    viewmodel.cancel()
    assert viewmodel.project and viewmodel.project.source_spec.download_state == "cancelled"
    assert not viewmodel.active

    release.set()
    _drain_background(application)
    restored = services.projects.load(project.project_id)
    assert restored.source_spec.download_state == "cancelled"
    assert not restored.source_spec.is_ready
