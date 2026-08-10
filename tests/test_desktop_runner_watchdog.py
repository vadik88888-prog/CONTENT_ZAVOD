from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.gui.services import pipeline_runner as pipeline_runner_module
from app.gui.models import DesktopSettings, ProcessingPhase, ProcessingSnapshot, ProjectStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels.project_viewmodel import ProjectViewModel


def _application():
    from PySide6.QtCore import QCoreApplication

    return QCoreApplication.instance() or QCoreApplication([])


def _prepared(tmp_path: Path, arguments: list[str]) -> PreparedPipelineRun:
    return PreparedPipelineRun(
        program=sys.executable,
        arguments=arguments,
        working_directory=tmp_path,
        state_path=tmp_path / "state.json",
        report_path=tmp_path / "report.json",
        output_directory=tmp_path / "output",
        runtime_config_path=tmp_path / "runtime.yaml",
        source_path=tmp_path / "source.mp4",
    )


def _run_loop(milliseconds: int) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _services(tmp_path: Path) -> tuple[DesktopServices, object]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    services = DesktopServices(
        engine_root=tmp_path,
        settings_store=SettingsStore(data),
        settings=DesktopSettings.defaults(data),
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(tmp_path),
        system=SystemService(tmp_path),
    )
    return services, project


def test_pipeline_log_is_created_before_qprocess_start(monkeypatch, tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    prepared = _prepared(tmp_path, ["-u", "-c", "print('never started')"])
    monkeypatch.setattr(services.pipeline, "prepare", lambda *_args: prepared)

    run, returned = services.prepare_run(project)

    log = Path(run.log_path)
    assert returned.heartbeat_path == prepared.state_path.with_name("heartbeat.json")
    assert returned.log_path == log
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "Desktop pipeline launch prepared." in text
    assert f"command: {prepared.command_line()}" in text
    assert f"cwd: {tmp_path}" in text
    assert "PYTHONUNBUFFERED=1" in text


def test_viewmodel_rejects_duplicate_launch_before_qprocess_becomes_active(monkeypatch, tmp_path: Path) -> None:
    _application()
    services, project = _services(tmp_path)
    run = services.runs.create(project, {}, {"path": str(project.source)}, "0.1.0")
    prepared = _prepared(tmp_path, ["-u", "-c", "print('later')"])
    calls: list[bool] = []
    monkeypatch.setattr(DesktopServices, "prepare_run", lambda _self, _project: (calls.append(True) or (run, prepared)))
    viewmodel = ProjectViewModel(services)
    viewmodel.open(project)
    monkeypatch.setattr(viewmodel.runner, "start", lambda _prepared: None)

    viewmodel.start()
    viewmodel.start()

    assert calls == [True]
    assert viewmodel.active


def test_link_download_is_explicit_shows_transfer_details_and_can_restart(monkeypatch, tmp_path: Path) -> None:
    _application()
    services, _project = _services(tmp_path)
    project = services.projects.create_url("https://example.test/video", {"title": "Видео по ссылке"})
    viewmodel = ProjectViewModel(services)
    viewmodel.open(project)
    launches: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        viewmodel.source_downloader,
        "download",
        lambda url, directory: launches.append((url, Path(directory))),
    )

    viewmodel.start_download()

    assert viewmodel.project and viewmodel.project.source_spec.download_state == "downloading"
    assert launches == [("https://example.test/video", project.directory / "sources")]
    viewmodel._download_progress(SimpleNamespace(
        fraction=0.25, speed="2MiB/s", downloaded="50MiB", total="200MiB", eta_seconds=75,
    ))
    assert viewmodel.snapshot.progress_fraction == 0.25
    assert viewmodel.snapshot.transfer_downloaded == "50MiB"
    assert viewmodel.snapshot.transfer_total == "200MiB"
    assert viewmodel.snapshot.eta_seconds == 75

    viewmodel._download_cancelled()
    assert viewmodel.project and viewmodel.project.source_spec.download_state == "cancelled"

    viewmodel.start_download()
    assert len(launches) == 2
    assert viewmodel.project and viewmodel.project.source_spec.download_state == "downloading"


def test_completed_link_download_hands_off_without_redownloading(monkeypatch, tmp_path: Path) -> None:
    _application()
    services, _project = _services(tmp_path)
    project = services.projects.create_url("https://example.test/video", {"title": "Видео по ссылке"})
    downloaded = project.directory / "sources" / "downloaded.mp4"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"completed download")
    metadata = {"duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0, "size_bytes": downloaded.stat().st_size}
    monkeypatch.setattr(services.pipeline, "inspect_source", lambda _path: metadata)
    viewmodel = ProjectViewModel(services)
    viewmodel.open(project)
    relaunches: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        viewmodel.source_downloader,
        "download",
        lambda url, directory: relaunches.append((url, Path(directory))),
    )

    viewmodel._launching = True
    viewmodel._after_download = "none"
    viewmodel._download_completed(str(downloaded))

    assert viewmodel.project is not None
    assert viewmodel.project.status == ProjectStatus.SOURCE_READY
    assert viewmodel.project.source == downloaded.resolve()
    assert viewmodel.project.source_spec.downloaded_path == str(downloaded.resolve())
    assert viewmodel.project.source_spec.download_state == "downloaded"
    assert viewmodel.snapshot.message == "Видео загружено"
    assert not viewmodel.active
    assert relaunches == []


def test_local_source_uses_shared_validation_handoff(monkeypatch, tmp_path: Path) -> None:
    _application()
    services, _project = _services(tmp_path)
    local_video = tmp_path / "local-video.mp4"
    local_video.write_bytes(b"local video")
    metadata = {"duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0}
    monkeypatch.setattr(services.pipeline, "inspect_source", lambda _path: metadata)

    project = services.create_project(local_video)
    viewmodel = ProjectViewModel(services)
    viewmodel.open(project)

    assert project.status == ProjectStatus.SOURCE_READY
    assert project.source == local_video.resolve()
    assert project.source_metadata == metadata
    assert not viewmodel.active


def test_startup_watchdog_marks_unstarted_process_failed(tmp_path: Path) -> None:
    _application()
    failures: list[str] = []
    runner = QtPipelineRunner(startup_timeout_ms=20, stall_timeout_ms=100, kill_timeout_ms=20)
    runner.run_failed.connect(failures.append)
    runner._prepared = _prepared(tmp_path, ["-u", "-c", "pass"])

    runner._startup_timed_out()

    assert failures == ["Не удалось запустить локальный процесс обработки."]
    assert "startup watchdog timeout" in runner.failure_details
    assert "process_state=NotRunning" in runner.failure_details


def test_live_silent_process_warns_without_failure_or_auto_kill(tmp_path: Path) -> None:
    _application()
    failures: list[str] = []
    warnings: list[tuple[str, int]] = []
    completed: list[int] = []
    runner = QtPipelineRunner(
        startup_timeout_ms=1_000, stall_timeout_ms=80, long_stage_timeout_ms=80, kill_timeout_ms=50,
    )
    runner.run_failed.connect(failures.append)
    runner.stage_running_longer_than_usual.connect(lambda stage, timeout: warnings.append((stage, timeout)))
    runner.run_completed.connect(completed.append)
    runner.start(_prepared(tmp_path, ["-u", "-c", "import time; time.sleep(0.55)"]))

    _run_loop(280)

    assert runner.active
    assert failures == []
    assert warnings == [("processing", 80)]
    _run_loop(700)
    assert completed == [0]
    assert failures == []
    assert not runner.active


def test_watchdog_observes_heartbeat_log_and_artifact_changes(tmp_path: Path) -> None:
    _application()
    activities: list[str] = []
    completed: list[int] = []
    heartbeat = tmp_path / "heartbeat.json"
    log_path = tmp_path / "pipeline.log"
    output_directory = tmp_path / "output"
    script = (
        "from pathlib import Path; import time; "
        f"heartbeat=Path({str(heartbeat)!r}); log=Path({str(log_path)!r}); output=Path({str(output_directory)!r}); "
        "output.mkdir(exist_ok=True); heartbeat.write_text('first'); log.write_text('first'); "
        "(output / 'partial.bin').write_bytes(b'1'); time.sleep(0.08); "
        "heartbeat.write_text('second'); log.write_text('second'); "
        "(output / 'partial.bin').write_bytes(b'12'); time.sleep(0.08)"
    )
    runner = QtPipelineRunner(startup_timeout_ms=1_000, stall_timeout_ms=500, long_stage_timeout_ms=500)
    runner.activity_changed.connect(lambda _at, reason: activities.append(reason))
    runner.run_completed.connect(completed.append)
    prepared = replace(
        _prepared(tmp_path, ["-u", "-c", script]),
        heartbeat_path=heartbeat,
        log_path=log_path,
        output_directory=output_directory,
    )
    runner.start(prepared)

    _run_loop(600)

    assert completed == [0]
    assert any(reason == "heartbeat file updated" for reason in activities)
    assert any(reason == "pipeline log updated" for reason in activities)
    assert any(reason == "output artifact updated" for reason in activities)


def test_long_media_stages_use_extended_warning_timeout(tmp_path: Path) -> None:
    runner = QtPipelineRunner(stall_timeout_ms=80, long_stage_timeout_ms=500)

    runner._last_stage = "transcription"
    assert runner._current_timeout_seconds() == 0.5
    runner._last_stage = "metadata"
    assert runner._current_timeout_seconds() == 0.08
    runner._prepared = replace(_prepared(tmp_path, ["-u", "-c", "pass"]), source_duration_seconds=90 * 60 + 1)
    assert runner._current_timeout_seconds() == 0.5
    runner._prepared = replace(runner._prepared, source_duration_seconds=30)
    assert runner._current_timeout_seconds() == 0.08


def test_stdout_and_stderr_update_activity_and_log_without_secrets(tmp_path: Path) -> None:
    _application()
    services, project = _services(tmp_path)
    run = services.runs.create(project, {}, {"path": str(project.source)}, "0.1.0")
    activities: list[str] = []
    logs: list[str] = []
    completed: list[int] = []
    fake_key = "sk" + "-secret-value-123456"
    runner = QtPipelineRunner(startup_timeout_ms=1_000, stall_timeout_ms=5_000)
    runner.activity_changed.connect(lambda _time, reason: activities.append(reason))
    runner.log_received.connect(logs.append)
    runner.log_received.connect(lambda line: services.append_log(run, line))
    runner.run_completed.connect(completed.append)
    prepared = _prepared(
        tmp_path,
        [
            "-u", "-c",
            f"import sys, time; key_name='OPENAI_API_KEY'; print(key_name + '=' + {fake_key!r}, flush=True); "
            "print('stderr activity', file=sys.stderr, flush=True); time.sleep(0.05)",
        ],
    )
    services.record_launch_context(run, prepared)
    runner.start(prepared)

    _run_loop(500)

    assert completed == [0]
    assert "stdout output" in activities
    assert "stderr output" in activities
    combined = "\n".join(logs)
    assert "QProcess state: Starting" in combined
    assert "QProcess state: Running" in combined
    assert "QProcess state: NotRunning" in combined
    assert fake_key not in combined
    pipeline_log = Path(run.log_path).read_text(encoding="utf-8")
    assert "command:" in pipeline_log
    assert "QProcess cwd:" in pipeline_log
    assert "QProcess state: Running" in pipeline_log
    assert "OPENAI_API_KEY:" in pipeline_log
    assert fake_key not in pipeline_log


def test_gui_runner_replaces_invalid_process_bytes_in_stdout_and_stderr(tmp_path: Path) -> None:
    """Qt pipe output must still reach the UI when a tool emits arbitrary bytes."""

    _application()
    logs: list[str] = []
    completed: list[int] = []
    runner = QtPipelineRunner(startup_timeout_ms=1_000, stall_timeout_ms=5_000)
    runner.log_received.connect(logs.append)
    runner.run_completed.connect(completed.append)
    runner.start(_prepared(
        tmp_path,
        [
            "-u", "-c",
            "import sys; "
            "sys.stdout.buffer.write(b'stdout: \\x98\\n'); sys.stdout.flush(); "
            "sys.stderr.buffer.write(b'stderr: \\x98\\n'); sys.stderr.flush()",
        ],
    ))

    _run_loop(500)

    assert completed == [0]
    combined = "\n".join(logs)
    assert "stdout: �" in combined
    assert "stderr: �" in combined


def test_paths_with_cyrillic_apostrophe_and_double_spaces_are_passed_verbatim(tmp_path: Path) -> None:
    _application()
    source = tmp_path / "Клип  Тони Д'Амато.mp4"
    output: list[str] = []
    runner = QtPipelineRunner(startup_timeout_ms=1_000, stall_timeout_ms=5_000)
    runner.log_received.connect(output.append)
    runner.start(_prepared(
        tmp_path,
        ["-u", "-c", "import sys; print(sys.argv[1], flush=True)", str(source)],
    ))

    _run_loop(500)

    assert str(source) in "\n".join(output)


def test_successful_run_does_not_trigger_watchdog(tmp_path: Path) -> None:
    _application()
    completed: list[int] = []
    failures: list[str] = []
    runner = QtPipelineRunner(startup_timeout_ms=1_000, stall_timeout_ms=500)
    runner.run_completed.connect(completed.append)
    runner.run_failed.connect(failures.append)
    runner.start(_prepared(tmp_path, ["-u", "-c", "print('ok', flush=True)"]))

    _run_loop(700)

    assert completed == [0]
    assert failures == []


def test_runner_uses_close_kills_tree_windows_job_policy(monkeypatch) -> None:
    """Pipeline cleanup must retain the close-time orphan-process guarantee."""

    _application()
    runner = QtPipelineRunner()
    job = object()
    logs: list[str] = []
    released: list[object] = []
    runner._process_id = 4242
    runner.log_received.connect(logs.append)
    monkeypatch.setattr(pipeline_runner_module.sys, "platform", "win32")
    monkeypatch.setattr(
        pipeline_runner_module,
        "attach_windows_process_job",
        lambda process_id: (job, None) if process_id == 4242 else (None, "unexpected PID"),
    )
    monkeypatch.setattr(pipeline_runner_module, "close_windows_process_job", released.append)

    runner._bind_process_tree()
    runner._release_process_job()

    assert runner._job_handle is None
    assert released == [job]
    assert any("final GUI handle closes" in line for line in logs)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree cleanup is a desktop contract")
def test_cancel_terminates_child_process_tree(tmp_path: Path) -> None:
    _application()
    from PySide6.QtCore import QTimer

    marker = tmp_path / "orphan-child.txt"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text('orphan')"
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(30)"
    )
    cancelled: list[bool] = []
    runner = QtPipelineRunner(startup_timeout_ms=1_000, stall_timeout_ms=5_000, kill_timeout_ms=50)
    runner.run_cancelled.connect(lambda: cancelled.append(True))
    runner.start(_prepared(tmp_path, ["-u", "-c", parent]))
    QTimer.singleShot(100, runner.cancel)

    _run_loop(1_500)

    assert cancelled == [True]
    assert not runner.active
    assert not marker.exists()


def test_terminal_state_clears_stale_stage_after_stall_failure(tmp_path: Path) -> None:
    _application()
    services, project = _services(tmp_path)
    run = services.runs.create(project, {}, {"path": str(project.source)}, "0.1.0")
    viewmodel = ProjectViewModel(services)
    viewmodel.project = project
    viewmodel.run = run
    viewmodel.snapshot = ProcessingSnapshot(
        phase=ProcessingPhase.RUNNING,
        stage="transcription",
        message="Распознаём речь",
        elapsed_seconds=12.0,
    )
    viewmodel._activity_changed("2026-07-26T13:00:00+00:00", "stderr output")
    viewmodel.runner._failure_details = "stall watchdog timeout"

    viewmodel._failed("Обработка остановилась и не отвечает.")

    assert run.error_summary == "Обработка остановилась и не отвечает."
    assert run.technical_details == "stall watchdog timeout"
    assert viewmodel.snapshot.phase == ProcessingPhase.FAILED
    assert viewmodel.snapshot.stage is None
    assert viewmodel.snapshot.message == "Обработка остановилась и не отвечает."
    assert viewmodel.snapshot.last_activity_at == "2026-07-26T13:00:00+00:00"
