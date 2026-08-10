from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _prepared(tmp_path: Path, *, mode: str) -> PreparedPipelineRun:
    return PreparedPipelineRun(
        program="python",
        arguments=["-m", "app", mode],
        working_directory=tmp_path,
        state_path=tmp_path / "state.json",
        report_path=tmp_path / "report.json",
        output_directory=tmp_path / "output",
        runtime_config_path=tmp_path / "runtime.yaml",
        runtime_flags={"mode": mode},
    )


def _write_running_stages(path: Path, *stages: tuple[str, str]) -> None:
    path.write_text(
        json.dumps({
            "stages": {
                name: {"status": "running", "started_at": started_at}
                for name, started_at in stages
            },
        }),
        encoding="utf-8",
    )


def test_selected_render_ignores_stale_analysis_stage_and_keeps_final_export_stage(tmp_path: Path) -> None:
    """A final export must not switch its UI back to a prior analysis stage."""

    runner = QtPipelineRunner(stall_timeout_ms=80, long_stage_timeout_ms=500)
    prepared = _prepared(tmp_path, mode="selected_render")
    runner._prepared = prepared
    runner._launch_wall_time = time.time() - 1
    changed: list[tuple[str, str]] = []
    runner.stage_changed.connect(lambda stage, label: changed.append((stage, label)))
    _write_running_stages(
        prepared.state_path,
        # It is newer on purpose: filtering must happen before choosing the
        # latest running stage.
        ("transcription", "2026-08-01T12:00:02+00:00"),
        ("tts_generation:plan-1", "2026-08-01T12:00:01+00:00"),
    )

    runner._poll_stage()

    assert changed == [("production_render:tts_generation:plan-1", "Готовим озвучку")]
    assert runner._current_timeout_seconds() == 0.5


def test_draft_job_uses_draft_stage_and_ignores_analysis_state(tmp_path: Path) -> None:
    runner = QtPipelineRunner()
    prepared = _prepared(tmp_path, mode="draft")
    runner._prepared = prepared
    runner._launch_wall_time = time.time() - 1
    changed: list[tuple[str, str]] = []
    runner.stage_changed.connect(lambda stage, label: changed.append((stage, label)))
    _write_running_stages(
        prepared.state_path,
        ("candidate_generation", "2026-08-01T12:00:02+00:00"),
        ("draft_preview:candidate-1", "2026-08-01T12:00:01+00:00"),
    )

    runner._poll_stage()

    assert changed == [("draft:draft_preview:candidate-1", "Собираем черновик")]


@pytest.mark.parametrize(
    ("mode", "delivery_stage", "stale_stage", "expected"),
    (
        ("draft", "draft_preview:candidate-1", "report", "draft:draft_preview:candidate-1"),
        ("draft", "draft_preview:candidate-1", "terminal", "draft:draft_preview:candidate-1"),
        ("selected_render", "production_render:plan-1", "report", "production_render:production_render:plan-1"),
        ("selected_render", "production_render:plan-1", "terminal", "production_render:production_render:plan-1"),
    ),
)
def test_delivery_jobs_ignore_generic_report_and_terminal_from_reused_state(
    tmp_path: Path,
    mode: str,
    delivery_stage: str,
    stale_stage: str,
    expected: str,
) -> None:
    """A generic terminal entry has no job identity and must not mask live delivery."""

    runner = QtPipelineRunner()
    prepared = _prepared(tmp_path, mode=mode)
    runner._prepared = prepared
    runner._launch_wall_time = time.time() - 1
    changed: list[tuple[str, str]] = []
    runner.stage_changed.connect(lambda stage, label: changed.append((stage, label)))
    _write_running_stages(
        prepared.state_path,
        # It is deliberately newer than the focused delivery stage.
        (stale_stage, "2026-08-01T12:00:02+00:00"),
        (delivery_stage, "2026-08-01T12:00:01+00:00"),
    )

    runner._poll_stage()

    assert changed and changed[0][0] == expected
    assert all(stage != stale_stage for stage, _label in changed)


def test_analysis_job_rejects_delivery_stage_from_reused_state(tmp_path: Path) -> None:
    runner = QtPipelineRunner()
    prepared = _prepared(tmp_path, mode="analysis")
    runner._prepared = prepared
    runner._launch_wall_time = time.time() - 1
    changed: list[tuple[str, str]] = []
    logs: list[str] = []
    runner.stage_changed.connect(lambda stage, label: changed.append((stage, label)))
    runner.log_received.connect(logs.append)
    _write_running_stages(prepared.state_path, ("production_render:candidate-1", "2026-08-01T12:00:02+00:00"))

    runner._poll_stage()

    assert changed == []
    assert any("Ignoring stale pipeline stage for analysis job" in line for line in logs)


def test_runner_recognises_the_current_engine_stage_names_for_each_focused_job(tmp_path: Path) -> None:
    analysis = _prepared(tmp_path, mode="analysis")
    draft = _prepared(tmp_path, mode="draft")
    final = _prepared(tmp_path, mode="selected_render")

    assert QtPipelineRunner._present_stage(analysis, "candidates_v2") == (
        "candidates_v2", "Ищем сильные моменты",
    )
    assert QtPipelineRunner._present_stage(draft, "transformation_result:candidate-1") == (
        "draft:transformation_result:candidate-1", "Готовим сценарий",
    )
    assert QtPipelineRunner._present_stage(final, "audio_composition:plan-1") == (
        "production_render:audio_composition:plan-1", "Собираем звук",
    )


def test_desktop_run_reuses_one_shell_instead_of_opening_a_second_window(monkeypatch) -> None:
    """The launcher must focus an existing shell and never nest app.exec()."""

    from app.gui import application

    class FakeApplication:
        current = None

        def __init__(self, argv) -> None:
            type(self).current = self
            self.argv = argv
            self.windows: list[FakeWindow] = []
            self.exec_calls = 0

        @classmethod
        def instance(cls):
            return cls.current

        def setApplicationName(self, _value: str) -> None:
            pass

        def setOrganizationName(self, _value: str) -> None:
            pass

        def setStyleSheet(self, _value: str) -> None:
            pass

        def topLevelWidgets(self):
            return list(self.windows)

        def exec(self) -> int:
            self.exec_calls += 1
            return 17

    class FakeWindow:
        created = 0

        def __init__(self, _services) -> None:
            type(self).created += 1
            self.minimized = False
            self.show_calls = 0
            self.raise_calls = 0
            self.activate_calls = 0
            FakeApplication.current.windows.append(self)

        def windowTitle(self) -> str:
            return "Content Factory"

        def isMinimized(self) -> bool:
            return self.minimized

        def showNormal(self) -> None:
            self.minimized = False
            self.show_calls += 1

        def show(self) -> None:
            self.show_calls += 1

        def raise_(self) -> None:
            self.raise_calls += 1

        def activateWindow(self) -> None:
            self.activate_calls += 1

    class FakeServices:
        @staticmethod
        def create(_root):
            return object()

    FakeApplication.current = None
    monkeypatch.setattr(application, "QApplication", FakeApplication)
    monkeypatch.setattr(application, "QCoreApplication", FakeApplication)
    monkeypatch.setattr(application, "MainWindow", FakeWindow)
    monkeypatch.setattr(application, "DesktopServices", FakeServices)
    monkeypatch.setattr(application, "load_theme", lambda: "")
    monkeypatch.setattr(application, "_main_window", None)
    monkeypatch.setattr(application, "_start_instance_server", lambda _app: True)
    monkeypatch.setattr(application, "_release_instance_server", lambda: None)

    assert application.run(["content-factory"]) == 17
    window = FakeApplication.current.windows[0]
    assert FakeWindow.created == 1
    assert FakeApplication.current.exec_calls == 1

    assert application.run(["content-factory"]) == 0
    assert FakeWindow.created == 1
    assert FakeApplication.current.exec_calls == 1
    assert window.raise_calls == 1
    assert window.activate_calls == 1


def test_second_desktop_process_hands_off_to_the_local_instance(monkeypatch) -> None:
    """A separate process must exit before constructing another shell."""

    from app.gui import application

    seen_names: list[str] = []
    monkeypatch.setattr(application, "_instance_server", None)
    monkeypatch.setattr(
        application,
        "_notify_existing_instance",
        lambda name: seen_names.append(name) or True,
    )

    assert application._start_instance_server(object()) is False
    assert seen_names == [application._instance_server_name()]


def test_desktop_run_does_not_construct_a_shell_after_external_handoff(monkeypatch) -> None:
    from app.gui import application

    class FakeApplication:
        current = None

        def __init__(self, _argv) -> None:
            type(self).current = self

        @classmethod
        def instance(cls):
            return cls.current

        def setApplicationName(self, _value: str) -> None:
            pass

        def setOrganizationName(self, _value: str) -> None:
            pass

        def setStyleSheet(self, _value: str) -> None:
            pass

        def topLevelWidgets(self):
            return []

    class NeverConstructWindow:
        def __init__(self, _services) -> None:
            raise AssertionError("the second process must not construct a desktop shell")

    FakeApplication.current = None
    monkeypatch.setattr(application, "QApplication", FakeApplication)
    monkeypatch.setattr(application, "QCoreApplication", FakeApplication)
    monkeypatch.setattr(application, "MainWindow", NeverConstructWindow)
    monkeypatch.setattr(application, "load_theme", lambda: "")
    monkeypatch.setattr(application, "_main_window", None)
    monkeypatch.setattr(application, "_start_instance_server", lambda _app: False)

    assert application.run(["content-factory"]) == 0


def test_desktop_launcher_refuses_to_upgrade_an_existing_core_application(monkeypatch) -> None:
    from app.gui import application

    class CoreOnly:
        @classmethod
        def instance(cls):
            return object()

    class FakeApplication:
        pass

    monkeypatch.setattr(application, "QCoreApplication", CoreOnly)
    monkeypatch.setattr(application, "QApplication", FakeApplication)

    with pytest.raises(RuntimeError, match="requires QApplication"):
        application._desktop_application(["content-factory"])


def test_processing_progress_stays_indeterminate_until_a_real_fraction_is_available() -> None:
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    from app.gui.components.processing_progress import ProcessingProgress

    progress = ProcessingProgress()
    try:
        progress.set_running("Создаём черновики", "Прошло 00:04")
        assert progress.progress.minimum() == 0
        assert progress.progress.maximum() == 0
        assert not progress.progress_note.isHidden()

        progress.set_running("Загружаем видео", "Прошло 00:05", progress_fraction=0.42)
        assert progress.progress.minimum() == 0
        assert progress.progress.maximum() == 100
        assert progress.progress.value() == 42
        assert progress.progress_note.isHidden()
        assert progress.detail.isHidden()
        assert progress.cancel_button.isHidden() is False
        assert progress.minimumHeight() >= progress.layout().totalSizeHint().height()

        progress.set_finished(
            "Работа была прервана. Готовые результаты сохранены.",
            "Повторить поиск моментов",
        )
        assert progress.retry_button.isHidden() is False
        assert progress.cancel_button.isHidden()
        assert progress.minimumHeight() >= progress.layout().totalSizeHint().height()
    finally:
        progress.deleteLater()
        app.processEvents()
