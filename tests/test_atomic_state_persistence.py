from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import pytest

import app.pipeline as pipeline_module
import app.utils as utils
from app.config import AppConfig
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner
from app.pipeline import StageTracker
from app.reporting import make_report
from app.utils import AtomicWriteError, read_json, write_json


def _permission_error() -> PermissionError:
    return PermissionError(5, "Access is denied")


def test_atomic_json_retries_first_windows_permission_error(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    original_replace = utils.os.replace
    calls = 0

    def flaky_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _permission_error()
        return original_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", flaky_replace)
    monkeypatch.setattr(utils.time, "sleep", lambda _delay: None)

    write_json(destination, {"stage": "report"})

    assert calls == 2
    assert read_json(destination) == {"stage": "report"}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_atomic_json_flushes_and_fsyncs_before_replace(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    original_fsync = utils.os.fsync
    synced: list[int] = []

    def recording_fsync(file_descriptor: int) -> None:
        synced.append(file_descriptor)
        original_fsync(file_descriptor)

    monkeypatch.setattr(utils.os, "fsync", recording_fsync)

    write_json(destination, {"stage": "report"})

    assert synced
    assert read_json(destination) == {"stage": "report"}


def test_atomic_json_retries_multiple_windows_sharing_violations(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    original_replace = utils.os.replace
    calls = 0

    class SharingViolation(OSError):
        winerror = 32

    def flaky_replace(source, target):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise SharingViolation("The process cannot access the file")
        return original_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", flaky_replace)
    monkeypatch.setattr(utils.time, "sleep", lambda _delay: None)

    write_json(destination, {"stage": "report"})

    assert calls == 4
    assert read_json(destination) == {"stage": "report"}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_permanent_atomic_json_error_keeps_old_state_and_writes_fallback(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    write_json(destination, {"stage": "previous"})
    monkeypatch.setattr(utils.os, "replace", lambda *_args: (_ for _ in ()).throw(_permission_error()))
    monkeypatch.setattr(utils.time, "sleep", lambda _delay: None)

    with pytest.raises(AtomicWriteError) as raised:
        write_json(destination, {"stage": "report"})

    error = raised.value
    assert error.attempts == 6
    assert read_json(destination) == {"stage": "previous"}
    assert error.fallback_path and error.fallback_path.is_file()
    fallback = read_json(error.fallback_path)
    assert fallback["stage"] == "report"
    assert fallback["state_persistence"]["status"] == "degraded"
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_atomic_json_serialises_parallel_thread_replacements(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    original_replace = utils.os.replace
    active_replacements = 0
    maximum_parallel_replacements = 0
    guard = threading.Lock()

    def monitored_replace(source, target):
        nonlocal active_replacements, maximum_parallel_replacements
        with guard:
            active_replacements += 1
            maximum_parallel_replacements = max(maximum_parallel_replacements, active_replacements)
        try:
            time.sleep(0.02)
            return original_replace(source, target)
        finally:
            with guard:
                active_replacements -= 1

    monkeypatch.setattr(utils.os, "replace", monitored_replace)
    first = threading.Thread(target=write_json, args=(destination, {"writer": 1}))
    second = threading.Thread(target=write_json, args=(destination, {"writer": 2}))
    first.start()
    second.start()
    first.join()
    second.join()

    assert maximum_parallel_replacements == 1
    assert read_json(destination)["writer"] in {1, 2}


def test_state_save_failure_keeps_completed_render_recoverable_in_report(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "work" / "state.json"
    fallback = state_path.with_name("state.json.fallback-test.json")
    error = AtomicWriteError(state_path, _permission_error(), attempts=6, fallback_path=fallback)
    monkeypatch.setattr(pipeline_module, "write_json", lambda *_args: (_ for _ in ()).throw(error))
    tracker = StageTracker(state_path)

    tracker.start("production_render")
    tracker.finish("production_render")
    tracker.start("report")
    tracker.finish("report")
    report_path = tmp_path / "output" / "report.json"
    make_report(
        report_path, {}, {}, AppConfig(), tracker.data, 1, 1, ["final-short.mp4"], [], [],
        {"provider": "not-called"}, False, False,
        production_render={"status": "completed", "output_file": "final-short.mp4"},
        primary_results=[{
            "candidate_id": "clip-one", "output_file": "final-short.mp4", "status": "completed", "primary": True,
        }],
    )

    assert tracker.data["stages"]["production_render"]["status"] == "completed"
    assert tracker.data["stages"]["report"]["status"] == "completed"
    assert tracker.data["state_persistence"]["status"] == "degraded"
    report = read_json(report_path)
    assert report["production_render"]["status"] == "completed"
    assert report["primary_results"][0]["candidate_id"] == "clip-one"
    assert report["state_persistence"]["status"] == "degraded"


def test_gui_state_reader_closes_its_handle(monkeypatch, tmp_path: Path) -> None:
    from PySide6.QtCore import QCoreApplication

    _application = QCoreApplication.instance() or QCoreApplication([])
    state = tmp_path / "state.json"
    state.write_text('{"stages": {"report": {"status": "running"}}}', encoding="utf-8")
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path,
        state_path=state, report_path=tmp_path / "report.json", output_directory=tmp_path,
        runtime_config_path=tmp_path / "runtime.yaml",
    )
    runner = QtPipelineRunner()
    runner._prepared = prepared
    runner._launch_wall_time = 0
    closed: list[bool] = []

    class TrackingFile(io.StringIO):
        def close(self) -> None:
            closed.append(True)
            super().close()

    def tracked_open(self, *args, **kwargs):
        assert self == state
        return TrackingFile('{"stages": {"report": {"status": "running"}}}')

    monkeypatch.setattr(Path, "open", tracked_open)

    runner._poll_stage()

    assert closed == [True]
