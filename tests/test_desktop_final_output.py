from __future__ import annotations

from pathlib import Path

import pytest

import app.gui.services.pipeline_facade as facade_module
from app.gui.components.video_preview import VideoPreview
from app.gui.models import DesktopSettings, ProcessingPhase, ProcessingSnapshot, ProjectStatus, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.services.pipeline_facade import STATE_PERSISTENCE_WARNING
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels.project_viewmodel import ProjectViewModel
from app.utils import read_json, write_json


def _context(tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    runs = RunHistoryStore(projects)
    run = runs.create(project, {}, {"path": str(source)}, "0.1.0")
    services = DesktopServices(
        engine_root=tmp_path,
        settings_store=SettingsStore(data),
        settings=DesktopSettings.defaults(data),
        projects=projects,
        runs=runs,
        pipeline=PipelineFacade(tmp_path),
        system=SystemService(tmp_path),
    )
    output = tmp_path / "engine-output"
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path,
        state_path=tmp_path / "state.json", report_path=output / "report.json",
        output_directory=output, runtime_config_path=tmp_path / "runtime.yaml",
    )
    return services, project, run, prepared


def _write_report(prepared: PreparedPipelineRun, final_path: Path, *, warnings: list[str] | None = None) -> None:
    write_json(prepared.report_path, {
        "output_files": [str(prepared.output_directory / "legacy-clip.mp4")],
        "warnings": warnings or [],
        "production_render": {
            "status": "warning" if warnings else "completed",
            "output_file": str(final_path),
            "warnings": warnings or [],
        },
    })


@pytest.fixture
def valid_video_probe(monkeypatch):
    monkeypatch.setattr(
        facade_module, "probe_video",
        lambda _path: {"duration": 1.0, "width": 1080, "height": 1920, "fps": 30.0},
    )


def test_zero_exit_report_without_final_mp4_is_failed_and_logged(tmp_path: Path) -> None:
    services, project, run, prepared = _context(tmp_path)
    expected = prepared.output_directory / "production-render" / "final-short.mp4"
    _write_report(prepared, expected)

    finished = services.finish_success(project, run, prepared)

    assert finished.status == RunStatus.FAILED
    assert project.status == ProjectStatus.FAILED
    assert finished.error_summary == "Не удалось создать итоговый видеофайл."
    assert "does not exist" in (finished.technical_details or "")
    assert "does not exist" in Path(finished.log_path).read_text(encoding="utf-8")
    stored = services.runs.load(project.project_id, run.run_id)
    assert stored.technical_details == finished.technical_details


def test_terminal_no_renderable_clips_exposes_the_concrete_gui_reason(tmp_path: Path) -> None:
    services, _project, _run, prepared = _context(tmp_path)
    write_json(prepared.report_path, {
        "terminal": {
            "status": "failed",
            "error_code": "NO_RENDERABLE_CLIPS",
            "message": "Не удалось подготовить ни одного ролика к созданию.",
        },
        "production_render": {"enabled": True, "status": "skipped", "reason": "no_production_plan"},
    })

    completion = services.pipeline.completion(prepared)

    assert completion.error_summary == "Не удалось подготовить ни одного ролика к созданию."
    assert "NO_RENDERABLE_CLIPS" in (completion.technical_details or "")


def test_valid_report_mp4_with_warnings_is_completed_with_warnings(tmp_path: Path, valid_video_probe) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = tmp_path / "reported-location" / "actual-final.mp4"
    final.parent.mkdir()
    final.write_bytes(b"valid final artifact")
    _write_report(prepared, final, warnings=["CPU fallback used"])

    completion = services.pipeline.completion(prepared)
    assert completion.output_files == [final]
    finished = services.finish_success(project, run, prepared)

    assert finished.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert project.status == ProjectStatus.COMPLETED_WITH_WARNINGS
    assert finished.warnings == ["CPU fallback used"]
    assert [Path(value).name for value in finished.artifact_paths] == ["report.json", "actual-final.mp4"]
    assert all(Path(value).is_file() for value in finished.artifact_paths)


def test_valid_report_mp4_without_warnings_is_completed(tmp_path: Path, valid_video_probe) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"valid final artifact")
    _write_report(prepared, final)

    finished = services.finish_success(project, run, prepared)

    assert finished.status == RunStatus.COMPLETED
    assert project.status == ProjectStatus.COMPLETED
    assert [Path(value).name for value in finished.artifact_paths] == ["report.json", "final-short.mp4"]


def test_degraded_state_persistence_is_a_success_warning_not_a_failed_render(tmp_path: Path, valid_video_probe) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"valid final artifact")
    _write_report(prepared, final)
    report = read_json(prepared.report_path)
    report["state_persistence"] = {"status": "degraded", "fallback_state_path": "state.json.fallback.json"}
    write_json(prepared.report_path, report)

    finished = services.finish_success(project, run, prepared)

    assert finished.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert project.status == ProjectStatus.COMPLETED_WITH_WARNINGS
    assert STATE_PERSISTENCE_WARNING in finished.warnings
    assert [Path(value).name for value in finished.artifact_paths] == ["report.json", "final-short.mp4"]


def test_completion_keeps_each_valid_independent_clip_result(tmp_path: Path, valid_video_probe) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    additional = prepared.output_directory / "clip-02.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"valid final artifact")
    additional.write_bytes(b"valid additional artifact")
    _write_report(prepared, final)
    report = read_json(prepared.report_path)
    report["output_files"] = [str(additional)]
    write_json(prepared.report_path, report)

    completion = services.pipeline.completion(prepared)
    finished = services.finish_success(project, run, prepared)

    assert completion.output_files == [final, additional]
    assert finished.status == RunStatus.COMPLETED
    assert [Path(value).name for value in finished.artifact_paths] == ["report.json", "final-short.mp4", "clip-02.mp4"]


def test_primary_result_registry_excludes_legacy_outputs_and_keeps_exact_count(tmp_path: Path, valid_video_probe) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    second = prepared.output_directory / "candidates" / "clip-two" / "production-render" / "clip-two.mp4"
    third = prepared.output_directory / "candidates" / "clip-three" / "production-render" / "clip-three.mp4"
    legacy = [prepared.output_directory / f"legacy-{number}.mp4" for number in range(1, 4)]
    for path in [final, second, third, *legacy]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"valid final artifact")
    _write_report(prepared, final)
    report = read_json(prepared.report_path)
    report["output_files"] = [str(path) for path in [final, second, third, *legacy]]
    report["primary_results"] = [
        {"candidate_id": "clip-one", "output_file": str(final), "status": "completed", "primary": True},
        {"candidate_id": "clip-two", "output_file": str(second), "status": "completed", "primary": True},
        {"candidate_id": "clip-three", "output_file": str(third), "status": "completed", "primary": True},
    ]
    report["produced_clips_count"] = 3
    write_json(prepared.report_path, report)

    completion = services.pipeline.completion(prepared)
    finished = services.finish_success(project, run, prepared)

    assert completion.output_files == [final, second, third]
    assert [Path(value).name for value in finished.artifact_paths] == [
        "report.json", "final-short.mp4", "clip-two.mp4", "clip-three.mp4",
    ]


def test_failed_process_with_current_canonical_results_keeps_outputs(tmp_path: Path, valid_video_probe) -> None:
    from PySide6.QtCore import QCoreApplication

    _application = QCoreApplication.instance() or QCoreApplication([])
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"valid final artifact")
    _write_report(prepared, final)
    report = read_json(prepared.report_path)
    report["primary_results"] = [{
        "candidate_id": "clip-one", "output_file": str(final), "status": "completed", "primary": True,
    }]
    write_json(prepared.report_path, report)
    viewmodel = ProjectViewModel(services)
    viewmodel.project = project
    viewmodel.run = run
    viewmodel.prepared = prepared

    viewmodel._failed("Процесс обработки завершился с кодом 1.")

    assert run.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert STATE_PERSISTENCE_WARNING in run.warnings
    assert viewmodel.snapshot.phase == ProcessingPhase.COMPLETED_WITH_WARNINGS
    assert viewmodel.snapshot.message == STATE_PERSISTENCE_WARNING
    assert [Path(value).name for value in run.artifact_paths] == ["report.json", "final-short.mp4"]


def test_cancelled_process_preserves_verified_partial_output(tmp_path: Path, valid_video_probe) -> None:
    from PySide6.QtCore import QCoreApplication

    _application = QCoreApplication.instance() or QCoreApplication([])
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"valid final artifact")
    _write_report(prepared, final, warnings=["second candidate was cancelled"])
    report = read_json(prepared.report_path)
    report["primary_results"] = [{
        "candidate_id": "clip-one", "output_file": str(final), "status": "completed", "primary": True,
    }]
    write_json(prepared.report_path, report)
    viewmodel = ProjectViewModel(services)
    viewmodel.project = project
    viewmodel.run = run
    viewmodel.prepared = prepared

    viewmodel._cancelled()

    assert run.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert viewmodel.snapshot.phase == ProcessingPhase.COMPLETED_WITH_WARNINGS
    assert [Path(value).name for value in run.artifact_paths] == ["report.json", "final-short.mp4"]


def test_startup_recovery_uses_current_report_and_canonical_results(tmp_path: Path, valid_video_probe) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"valid final artifact")
    _write_report(prepared, final)
    report = read_json(prepared.report_path)
    report["primary_results"] = [{
        "candidate_id": "clip-one", "output_file": str(final), "status": "completed", "primary": True,
    }]
    write_json(prepared.report_path, report)
    services.record_launch_context(run, prepared)
    run.status = RunStatus.RUNNING
    services.runs.save(run)
    project.status = ProjectStatus.PROCESSING
    services.projects.save(project)

    recovered = services.recover_interrupted_runs()
    restored = services.runs.load(project.project_id, run.run_id)

    assert recovered == 1
    assert restored.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert STATE_PERSISTENCE_WARNING in restored.warnings
    assert [Path(value).name for value in restored.artifact_paths] == ["report.json", "final-short.mp4"]


def test_zero_byte_final_mp4_is_failed(tmp_path: Path) -> None:
    services, project, run, prepared = _context(tmp_path)
    final = prepared.output_directory / "production-render" / "final-short.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"")
    _write_report(prepared, final)

    finished = services.finish_success(project, run, prepared)

    assert finished.status == RunStatus.FAILED
    assert "empty" in (finished.technical_details or "")


def test_zero_exit_missing_output_clears_stale_stage_and_fails(tmp_path: Path) -> None:
    from PySide6.QtCore import QCoreApplication

    _application = QCoreApplication.instance() or QCoreApplication([])
    services, project, run, prepared = _context(tmp_path)
    _write_report(prepared, prepared.output_directory / "production-render" / "final-short.mp4")
    viewmodel = ProjectViewModel(services)
    viewmodel.project = project
    viewmodel.run = run
    viewmodel.prepared = prepared
    viewmodel.snapshot = ProcessingSnapshot(
        phase=ProcessingPhase.RUNNING, stage="transcription", message="Распознаём речь", elapsed_seconds=12.0,
    )
    errors = []
    viewmodel.error_occurred.connect(errors.append)

    viewmodel._completed(0)

    assert run.status == RunStatus.FAILED
    assert viewmodel.snapshot.phase == ProcessingPhase.FAILED
    assert viewmodel.snapshot.stage is None
    assert viewmodel.snapshot.message == "Не удалось создать итоговый видеофайл"
    assert viewmodel.snapshot.elapsed_seconds == 0.0
    assert errors[0].user_message == "Не удалось создать итоговый видеофайл."


def test_missing_or_empty_output_cannot_be_used_by_preview(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")

    assert VideoPreview.usable_media_path(missing) is False
    assert VideoPreview.usable_media_path(empty) is False
