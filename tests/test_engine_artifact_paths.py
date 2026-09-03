from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.analysis_artifact import new_analysis_artifact
from app.config import AppConfig
from app.gui.models import DesktopSettings, ProjectStatus, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.pipeline import Pipeline, StageTracker
from app.run_artifacts import (
    find_run_artifact_metadata,
    make_run_artifact_metadata,
    run_metadata_path,
    write_run_artifact_metadata,
)
from app.runtime import RuntimeLayout
from app.utils import read_json, write_json


@pytest.mark.parametrize(
    "source_name,engine_slug",
    [
        ('Очень длинное имя с кириллицей, кавычками "и" #$% [локальный].mp4', "engine-local-9e6e"),
        (
            "«URL title» — 100% special characters & a very very very very very very very long name.webm",
            "different-url-title-slug-5c01",
        ),
    ],
)
def test_legacy_report_is_resolved_by_identity_not_source_slug(
    tmp_path: Path, source_name: str, engine_slug: str,
) -> None:
    """Names that cannot round-trip through a slug still restore the real output."""

    # Windows prohibits literal double quotes in local names, but URL titles
    # and imported metadata may contain them.  Keep a valid local source while
    # preserving the hostile display title in the engine report below.
    source = tmp_path / ("Кириллица_" + "длинное_" * 14 + "#$% [локальный].mp4")
    source.write_bytes(b"source")
    run_id = "run-кириллица-quote-001"
    project_id = "project-legacy-001"
    actual_output = tmp_path / "output" / engine_slug / "runs" / run_id
    analysis = actual_output / "analysis.json"
    write_json(analysis, {"project_id": project_id, "analysis_id": "analysis-ready"})
    report = actual_output / "report.json"
    write_json(report, {
        "source": {"display_name": source_name},
        "terminal": {"status": "analysis_ready"},
        "warnings": [],
        "output_files": [],
        "clip_intelligence": {"analysis_artifact_ref": str(analysis)},
        "run": {
            "run_id": run_id,
            "project_id": project_id,
            "run_directory": str(actual_output),
            "analysis_artifact_path": str(analysis),
            "analysis_id": "analysis-ready",
            "analysis_fingerprint": "fingerprint",
            "terminal_status": "analysis_ready",
        },
    })
    wrong = tmp_path / "output" / "desktop-guessed-slug" / "runs" / run_id
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path,
        state_path=wrong / "state.json", report_path=wrong / "report.json",
        output_directory=wrong, runtime_config_path=tmp_path / "runtime.yaml",
        run_id=run_id, project_id=project_id,
        artifact_metadata_path=run_metadata_path(tmp_path, run_id),
        source_path=source,
        runtime_flags={"mode": "analysis"},
    )

    facade = PipelineFacade(tmp_path)
    resolved = facade.resolve_engine_paths(prepared)
    completion = facade.completion(prepared)

    assert resolved is not None
    assert resolved.report_path == report.resolve()
    assert resolved.output_directory == actual_output.resolve()
    assert completion.error_summary is None
    assert completion.report_path == report.resolve()


def test_engine_publishes_absolute_paths_in_index_and_state(tmp_path: Path) -> None:
    source = tmp_path / "Кавычки 'и' спецсимволы #1.mp4"
    source.write_bytes(b"source")
    pipeline = Pipeline(tmp_path, AppConfig(), run_id="run-with-$-characters", project_id="project-paths")
    _source, work_directory, output_directory = pipeline._prepare_source(str(source), None)
    assert pipeline.run_work_directory is not None
    tracker = StageTracker(pipeline.run_work_directory / "state.json")

    pipeline._publish_run_paths(tracker, work_directory, output_directory)
    metadata = find_run_artifact_metadata(
        tmp_path, run_id=pipeline.run_id, project_id="project-paths",
        preferred_path=run_metadata_path(tmp_path, pipeline.run_id),
    )
    state = read_json(pipeline.run_work_directory / "state.json")

    assert metadata is not None
    assert metadata["paths"]["state_path"] == str((pipeline.run_work_directory / "state.json").resolve())
    assert metadata["paths"]["heartbeat_path"] == str((pipeline.run_work_directory / "heartbeat.json").resolve())
    assert metadata["paths"]["report_path"] == str((output_directory / "report.json").resolve())
    assert state["run"]["paths"] == metadata["paths"]


def test_same_source_runs_publish_distinct_progress_state_and_heartbeat_paths(tmp_path: Path) -> None:
    """Desktop must never bind one run to a sibling run's live progress."""

    source = tmp_path / "same-source.mp4"
    source.write_bytes(b"source")
    first = Pipeline(tmp_path, AppConfig(), run_id="run-a", project_id="project-a")
    second = Pipeline(tmp_path, AppConfig(), run_id="run-b", project_id="project-a")

    for pipeline in (first, second):
        _source, work_directory, output_directory = pipeline._prepare_source(str(source), None)
        assert pipeline.run_work_directory is not None
        pipeline._publish_run_paths(
            StageTracker(pipeline.run_work_directory / "state.json"),
            work_directory,
            output_directory,
        )

    first_metadata = find_run_artifact_metadata(tmp_path, run_id="run-a", project_id="project-a")
    second_metadata = find_run_artifact_metadata(tmp_path, run_id="run-b", project_id="project-a")

    assert first_metadata is not None and second_metadata is not None
    for path_name in ("state_path", "heartbeat_path"):
        first_path = Path(first_metadata["paths"][path_name])
        second_path = Path(second_metadata["paths"][path_name])
        assert first_path != second_path
        assert first_path.parent == first.run_work_directory
        assert second_path.parent == second.run_work_directory


def test_new_desktop_run_does_not_scan_unrelated_legacy_reports(tmp_path: Path, monkeypatch) -> None:
    """Pending indexed runs must stay cheap while the child engine starts."""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    facade = PipelineFacade(tmp_path)
    prepared = facade._pending_prepared(
        [], source, tmp_path / "runtime.yaml", "fresh-run", "fresh-project", {},
    )

    def unexpected_legacy_scan(*_args, **_kwargs):
        raise AssertionError("a fresh indexed run must not scan old reports")

    monkeypatch.setattr("app.run_artifacts._iter_run_files", unexpected_legacy_scan)

    assert prepared.allow_legacy_artifact_scan is False
    assert facade.resolve_engine_paths(prepared) is prepared


def test_legacy_lookup_still_scans_for_a_report_by_identity(tmp_path: Path, monkeypatch) -> None:
    """Old desktop records retain the report/analysis fallback when opted in."""

    run_id = "legacy-run"
    project_id = "legacy-project"
    output = tmp_path / "output" / "old-layout" / "runs" / run_id
    report = output / "report.json"
    write_json(report, {
        "run": {"run_id": run_id, "project_id": project_id, "run_directory": str(output)},
    })
    calls: list[Path] = []

    from app.run_artifacts import _iter_run_files as original_iter_run_files

    def observed_iter_run_files(root: Path, requested_run_id: str, name: str):
        calls.append(root)
        return original_iter_run_files(root, requested_run_id, name)

    monkeypatch.setattr("app.run_artifacts._iter_run_files", observed_iter_run_files)

    metadata = find_run_artifact_metadata(
        tmp_path, run_id=run_id, project_id=project_id, allow_legacy_scan=True,
    )

    assert metadata is not None
    assert metadata["paths"]["report_path"] == str(report.resolve())
    assert calls


def test_recovery_restores_ready_legacy_analysis_without_relaunching(tmp_path: Path) -> None:
    source = tmp_path / "ролик с URL title и «кавычками».mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    runs = RunHistoryStore(projects)
    run = runs.create(project, {}, {"path": str(source)}, "0.1.0", run_kind=RunKind.ANALYSIS)
    # These are precisely the stale, desktop-calculated paths this regression
    # covers.  The real engine report below uses a different URL-title slug.
    wrong = tmp_path / "output" / "wrong-desktop-slug" / "runs" / run.run_id
    run.settings_snapshot["execution"] = {
        "run_id": run.run_id,
        "project_id": project.project_id,
        "state_path": str(tmp_path / "work" / "wrong-desktop-slug" / "runs" / run.run_id / "state.json"),
        "report_path": str(wrong / "report.json"),
        "output_directory": str(wrong),
        "runtime_flags": {"mode": "analysis"},
    }
    run.status = RunStatus.FAILED
    run.error_summary = "Итоговый отчёт обработки не найден."
    run.technical_details = f"Expected report is missing: {run_metadata_path(data, run.run_id)}"
    runs.save(run)
    actual = tmp_path / "output" / "фактический-url-title-42" / "runs" / run.run_id
    analysis = actual / "analysis.json"
    write_json(analysis, {"project_id": project.project_id, "analysis_id": "analysis-restored"})
    write_json(actual / "report.json", {
        "terminal": {"status": "analysis_ready"},
        "warnings": [], "output_files": [],
        "clip_intelligence": {"analysis_artifact_ref": str(analysis), "candidates": []},
        "run": {
            "run_id": run.run_id, "project_id": project.project_id,
            "run_directory": str(actual), "analysis_artifact_path": str(analysis),
            "analysis_id": "analysis-restored", "analysis_fingerprint": "legacy-fingerprint",
            "terminal_status": "analysis_ready",
        },
    })
    services = DesktopServices(
        engine_root=tmp_path, settings_store=SettingsStore(data), settings=DesktopSettings.defaults(data),
        projects=projects, runs=runs, pipeline=PipelineFacade(tmp_path), system=SystemService(tmp_path),
    )

    assert services.recover_ready_analysis_runs() == 1
    restored = projects.load(project.project_id)
    restored_run = runs.load(project.project_id, run.run_id)
    assert restored.status == ProjectStatus.ANALYSIS_READY
    assert restored.analysis_artifact_path == str(analysis.resolve())
    assert restored_run.status == RunStatus.ANALYSIS_READY
    assert restored_run.settings_snapshot["execution"]["engine_paths"]["report_path"] == str((actual / "report.json").resolve())


def test_source_desktop_recovers_analysis_when_engine_index_is_not_under_data_root(
    tmp_path: Path, monkeypatch,
) -> None:
    """A successful source-mode CLI run must not be failed by Desktop finalization."""

    resources = tmp_path / "исходный checkout"
    data = tmp_path / "данные Desktop"
    resources.mkdir()
    source = data / "projects" / "media" / (
        "РЕКРЕНТ СЛОВИЛ ЖЕСТКИЙ ТИЛЬТ ИЗ-ЗА ТИММЕЙТОВ НА ФЕЙСИТЕ — "
        + "очень длинное имя источника " * 3
        + ".webm"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")

    runtime = RuntimeLayout.for_source(resources, data=data, program=Path(sys.executable))
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    runs = RunHistoryStore(projects)
    run = runs.create(project, {}, {"path": str(source)}, "0.1.0", run_kind=RunKind.ANALYSIS)
    facade = PipelineFacade(runtime)

    pending = facade._pending_prepared(
        ["analyze"], source, project.directory / "runtime-config.yaml",
        run.run_id, project.project_id, {"mode": "analysis"},
    )
    assert pending.artifact_metadata_path == run_metadata_path(resources, run.run_id)
    assert pending.artifact_metadata_path != run_metadata_path(data, run.run_id)

    # Recreate the persisted broken Desktop record from the real incident: it
    # points at the data-root sentinel, while the source-mode engine published
    # its canonical paths under its actual working directory.
    run.settings_snapshot["execution"] = {
        "run_id": run.run_id,
        "project_id": project.project_id,
        "artifact_metadata_path": str(run_metadata_path(data, run.run_id)),
        "allow_legacy_artifact_scan": False,
        "source_path": str(source),
        "runtime_flags": {"mode": "analysis"},
    }
    run.status = RunStatus.FAILED
    runs.save(run)

    output = resources / "output" / "engine-owned-unicode-slug" / "runs" / run.run_id
    work = resources / "work" / "engine-owned-unicode-slug"
    analysis = output / "analysis.json"
    candidate_id = "candidate-chapter-007-story-001"
    new_analysis_artifact(
        analysis_id="analysis-ready", project_id=project.project_id,
        source={"id": "source-fingerprint", "path": str(source)},
        source_fingerprint="source-fingerprint", analysis_fingerprint="analysis-fingerprint",
        work_directory=str(work), candidate_data_ref=str(work / "candidates.scored.json"),
        references={}, candidates=[{
            "candidate_id": candidate_id,
            "title": "Один найденный момент",
            "start_seconds": 10.0,
            "end_seconds": 24.0,
            "selected_by_recommendation": True,
        }],
        recommendation={"selected_candidate_ids": [candidate_id]},
        summary={"candidate_count": 1, "recommended_count": 1},
        content_profile={}, duration_seconds=90.0, candidate_count=1,
        recommended_count={"min": 1, "max": 3, "default": 3},
    ).write(analysis)
    warning = "Найдено только 1 достаточно разных сильных фрагмента из запрошенных 3."
    report = output / "report.json"
    write_json(report, {
        "terminal": {"status": "analysis_ready"},
        "warnings": [warning],
        "output_files": [],
        "clip_intelligence": {"candidates": [{"id": candidate_id}]},
        "run": {
            "run_id": run.run_id,
            "project_id": project.project_id,
            "run_directory": str(output),
            "analysis_artifact_path": str(analysis),
            "analysis_id": "analysis-ready",
            "analysis_fingerprint": "analysis-fingerprint",
            "terminal_status": "analysis_ready",
        },
    })
    write_run_artifact_metadata(resources, make_run_artifact_metadata(
        engine_root=resources,
        run_id=run.run_id,
        project_id=project.project_id,
        work_directory=work,
        output_directory=output,
        report_path=report,
        analysis_artifact_path=analysis,
        terminal_status="analysis_ready",
    ))

    settings = DesktopSettings.defaults(data)
    services = DesktopServices(
        engine_root=data,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=runs,
        pipeline=facade,
        system=SystemService(runtime),
        runtime=runtime,
    )
    monkeypatch.setattr(
        services.pipeline, "prepare_analysis",
        lambda *_args, **_kwargs: pytest.fail("persisted analysis must not be launched again"),
    )

    assert services.recover_ready_analysis_runs() == 1
    restored = projects.load(project.project_id)
    restored_run = runs.load(project.project_id, run.run_id)

    assert restored.status == ProjectStatus.ANALYSIS_READY
    assert restored.analysis_artifact_path == str(analysis.resolve())
    assert restored.candidate_states == {candidate_id: "analyzed"}
    assert restored_run.status == RunStatus.ANALYSIS_READY
    assert restored_run.warnings == [warning]
    assert restored_run.error_summary is None
    assert restored_run.technical_details is None
    assert restored_run.settings_snapshot["execution"]["engine_paths"]["report_path"] == str(report.resolve())
    assert read_json(Path(restored.analysis_artifact_path))["candidates"][0]["title"] == "Один найденный момент"
