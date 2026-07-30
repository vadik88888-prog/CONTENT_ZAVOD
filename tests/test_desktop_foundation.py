from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.gui.models import DesktopSettings, ProjectStatus, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError
from app.gui.services.error_mapping import map_error, redact_secrets
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.pipeline_facade import PreparedPipelineRun
from app.gui.services.pipeline_runner import QtPipelineRunner
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.system_service import SystemService


def _video(tmp_path: Path, name: str = "тестовое видео.mp4") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_bytes(b"not-a-real-video")
    return path


def test_project_persistence_run_append_and_cyrillic_path(tmp_path: Path) -> None:
    source = _video(tmp_path / "папка с пробелом")
    store = DesktopProjectStore(tmp_path / "данные")
    project = store.create(source)
    project.settings.subtitle_style = "clean"
    project.setup_state.last_estimate = {"estimated_seconds_min": 60, "estimated_seconds_max": 120}
    project.setup_state.change_summary = "Настройки сохранены."
    project.setup_state.reused_stages = ["сохранённый анализ"]
    store.save(project)
    loaded = store.load(project.project_id)
    assert loaded.source == source.resolve()
    assert loaded.settings.subtitle_style == "clean"
    assert loaded.setup_state.last_estimate["estimated_seconds_max"] == 120
    assert loaded.setup_state.reused_stages == ["сохранённый анализ"]

    history = RunHistoryStore(store)
    first = history.create(loaded, {"local_test_mode": True}, {"path": str(source)}, "0.1.0")
    first.status = RunStatus.COMPLETED
    history.save(first)
    second = history.create(loaded, {"local_test_mode": True}, {"path": str(source)}, "0.1.0")
    assert {run.run_id for run in history.list(loaded.project_id)} == {first.run_id, second.run_id}
    assert (store.project_directory(loaded.project_id) / "runs" / first.run_id / "run.json").is_file()


def test_corrupt_project_and_run_do_not_break_remaining_history(tmp_path: Path) -> None:
    store = DesktopProjectStore(tmp_path / "data")
    good = store.create(_video(tmp_path, "good.mp4"))
    corrupt = store.projects_directory / "broken"
    corrupt.mkdir(parents=True)
    (corrupt / "project.json").write_text("{bad", encoding="utf-8")
    assert [project.project_id for project in store.list()] == [good.project_id]

    history = RunHistoryStore(store)
    run = history.create(good, {}, {}, "0.1.0")
    bad_run = history.runs_directory(good.project_id) / "bad"
    bad_run.mkdir()
    (bad_run / "run.json").write_text("{bad", encoding="utf-8")
    assert [item.run_id for item in history.list(good.project_id)] == [run.run_id]


def test_active_run_is_recovered_as_interrupted(tmp_path: Path) -> None:
    store = DesktopProjectStore(tmp_path / "data")
    project = store.create(_video(tmp_path, "source.mp4"))
    project.status = ProjectStatus.PROCESSING
    store.save(project)
    history = RunHistoryStore(store)
    run = history.create(project, {}, {}, "0.1.0")
    run.status = RunStatus.RUNNING
    history.save(run)

    assert history.mark_interrupted(project)
    assert history.load(project.project_id, run.run_id).status == RunStatus.INTERRUPTED
    assert store.load(project.project_id).status == ProjectStatus.INTERRUPTED


def test_interrupted_url_download_becomes_repeatable_after_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    store = DesktopProjectStore(data)
    project = store.create_url("https://example.test/video", {"title": "Видео"})
    partial = project.directory / "sources" / "video.mp4.part"
    partial.parent.mkdir()
    partial.write_bytes(b"partial")
    project.source_spec.download_state = "downloading"
    store.save(project)
    services = DesktopServices(
        engine_root=Path(__file__).resolve().parents[1],
        settings_store=SettingsStore(data),
        settings=DesktopSettings.defaults(data),
        projects=store,
        runs=RunHistoryStore(store),
        pipeline=PipelineFacade(Path(__file__).resolve().parents[1]),
        system=SystemService(Path(__file__).resolve().parents[1]),
    )

    assert services.recover_interrupted_downloads() == 1
    restored = store.load(project.project_id)
    assert restored.source_spec.download_state == "cancelled"
    assert "начать снова" in (restored.source_spec.error_message or "")
    assert not partial.exists()


def test_settings_atomic_save_load_and_corrupt_fallback(tmp_path: Path) -> None:
    settings_store = SettingsStore(tmp_path / "ContentFactoryData")
    settings = settings_store.load()
    settings.local_test_mode = True
    settings_store.save(settings)
    assert settings_store.load().local_test_mode is True
    legacy = settings_store.load().to_dict()
    legacy.pop("schema_version")
    settings_store.path.write_text(json.dumps(legacy), encoding="utf-8")
    assert settings_store.load().schema_version == 1
    settings_store.path.write_text("not json", encoding="utf-8")
    assert settings_store.load().local_test_mode is False
    assert "OPENAI_API_KEY" not in settings_store.load().to_dict()
    with pytest.raises(ValueError):
        DesktopSettings.from_dict({"data_directory": str(tmp_path), "device_preference": "invalid"})


@pytest.mark.parametrize("name", ["video.mp4", "Видео с пробелом.mkv", "Клип.mov"])
def test_supported_input_paths_are_accepted(tmp_path: Path, name: str) -> None:
    store = DesktopProjectStore(tmp_path / "data")
    assert store.create(_video(tmp_path, name)).source.name == name


def test_unsupported_or_missing_input_is_rejected(tmp_path: Path) -> None:
    store = DesktopProjectStore(tmp_path / "data")
    unsupported = tmp_path / "file.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(InputValidationError):
        store.create(unsupported)
    with pytest.raises(InputValidationError):
        store.create(tmp_path / "missing.mp4")


def test_pipeline_facade_builds_safe_mock_command_and_runtime_config(tmp_path: Path) -> None:
    engine_root = Path(__file__).resolve().parents[1]
    store = DesktopProjectStore(tmp_path / "данные")
    source = _video(tmp_path / "путь с пробелом", "видео.mp4")
    project = store.create(source)
    history = RunHistoryStore(store)
    run = history.create(project, {}, {}, "0.1.0")
    settings = DesktopSettings.defaults(tmp_path / "данные")
    settings.local_test_mode = True
    prepared = PipelineFacade(engine_root).prepare(project, run, settings)

    assert prepared.program == sys.executable
    assert prepared.arguments[:3] == ["-u", "-m", "app"]
    assert "--mock-ai" in prepared.arguments
    assert "--no-ai-transformation" in prepared.arguments
    assert prepared.runtime_flags["mock_ai"] == "true"
    assert str(source.resolve()) in prepared.arguments
    runtime = prepared.runtime_config_path.read_text(encoding="utf-8")
    assert "provider: mock" in runtime
    assert "enabled: true" in runtime
    assert "sk-" not in runtime
    assert "device: auto" in runtime


def test_error_mapping_redacts_secrets() -> None:
    value = "sk-" + "super-secret-value"
    raw = "Authorization: Bearer " + value
    mapped = map_error(raw)
    assert value not in mapped.technical_details
    assert value not in redact_secrets(raw)


def test_pipeline_facade_reports_missing_output(tmp_path: Path) -> None:
    prepared = PreparedPipelineRun(
        program=sys.executable, arguments=[], working_directory=tmp_path,
        state_path=tmp_path / "state.json", report_path=tmp_path / "missing-report.json",
        output_directory=tmp_path, runtime_config_path=tmp_path / "runtime.yaml",
    )
    completion = PipelineFacade(Path(__file__).resolve().parents[1]).completion(prepared)
    assert completion.output_files == []
    assert completion.error_summary == "Итоговый отчёт обработки не найден."


def test_qprocess_runner_success_and_cancellation(tmp_path: Path) -> None:
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    application = QCoreApplication.instance() or QCoreApplication([])
    completed: list[int] = []
    loop = QEventLoop()
    runner = QtPipelineRunner()
    runner.run_completed.connect(lambda code: (completed.append(code), loop.quit()))
    runner.run_failed.connect(lambda _message: loop.quit())
    runner.start(PreparedPipelineRun(
        program=sys.executable, arguments=["-c", "print('desktop runner')"],
        working_directory=tmp_path, state_path=tmp_path / "state.json",
        report_path=tmp_path / "report.json", output_directory=tmp_path,
        runtime_config_path=tmp_path / "runtime.yaml",
    ))
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    assert completed == [0]

    cancelled: list[bool] = []
    loop = QEventLoop()
    runner = QtPipelineRunner()
    runner.run_cancelled.connect(lambda: (cancelled.append(True), loop.quit()))
    runner.start(PreparedPipelineRun(
        program=sys.executable, arguments=["-c", "import time; time.sleep(10)"],
        working_directory=tmp_path, state_path=tmp_path / "state.json",
        report_path=tmp_path / "report.json", output_directory=tmp_path,
        runtime_config_path=tmp_path / "runtime.yaml",
    ))
    QTimer.singleShot(100, runner.cancel)
    QTimer.singleShot(6000, loop.quit)
    loop.exec()
    assert cancelled == [True]

    failed: list[str] = []
    loop = QEventLoop()
    runner = QtPipelineRunner()
    runner.run_failed.connect(lambda message: (failed.append(message), loop.quit()))
    runner.start(PreparedPipelineRun(
        program=sys.executable, arguments=["-c", "raise SystemExit(4)"],
        working_directory=tmp_path, state_path=tmp_path / "state.json",
        report_path=tmp_path / "report.json", output_directory=tmp_path,
        runtime_config_path=tmp_path / "runtime.yaml",
    ))
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    assert failed == ["Процесс обработки завершился с кодом 4."]
