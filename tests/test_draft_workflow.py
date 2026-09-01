from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.analysis_artifact import new_analysis_artifact
from app.cli import _apply_draft_command_arguments, _apply_render_command_arguments
from app.clip_results import ClipResult
from app.config import AppConfig, load_config
from app.draft_artifact import new_draft_artifact
from app.gui.models import DesktopSettings, ProjectStatus, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineCompletion, PipelineFacade, PreparedPipelineRun
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.utils import read_json, write_json


def _services(tmp_path: Path):
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
    return services, project, source


def _write_analysis_artifact(
    path: Path,
    project_id: str,
    analysis_id: str,
    fingerprint: str,
    candidate_ids: list[str],
    *,
    analysis_run_id: str = "",
) -> None:
    new_analysis_artifact(
        analysis_id=analysis_id, project_id=project_id,
        source={"id": "source-fingerprint"}, source_fingerprint="source-fingerprint",
        analysis_fingerprint=fingerprint, work_directory=str(path.parent),
        candidate_data_ref=str(path.parent / "candidate-data.json"), references={},
        candidates=[{"candidate_id": candidate_id} for candidate_id in candidate_ids],
        recommendation={}, summary={}, content_profile={}, duration_seconds=30.0,
        candidate_count=len(candidate_ids),
        analysis_run_id=analysis_run_id,
        schema_version="1.0",
    ).write(path)


def test_analysis_start_reuses_only_a_verified_current_artifact(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis_path, project.project_id, "analysis-current", "fingerprint-current", ["candidate-a"],
    )
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "analysis-current"
    project.analysis_fingerprint = "fingerprint-current"
    services.projects.save(project)

    with pytest.raises(InputValidationError, match="Сохранённый анализ уже готов"):
        services.prepare_analysis(project)

    analysis_path.write_text("{}", encoding="utf-8")
    retry_run, retry = services.prepare_analysis(project)

    assert retry_run.run_kind == RunKind.ANALYSIS
    assert "analyze" in retry.arguments


def test_analysis_dependency_change_allows_refresh_without_recompute_all(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis_path, project.project_id, "analysis-current", "fingerprint-current", ["candidate-a"],
    )
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "analysis-current"
    project.analysis_fingerprint = "fingerprint-current"
    project.setup_state.needs_new_analysis = True
    services.projects.save(project)

    run, prepared = services.prepare_analysis(project)

    assert run.run_kind == RunKind.ANALYSIS
    assert project.settings.recompute_all is False
    assert "analyze" in prepared.arguments


def test_full_run_can_replace_a_corrupt_analysis_handoff(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text("{}", encoding="utf-8")
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "stale-analysis"
    project.analysis_fingerprint = "stale-fingerprint"
    services.projects.save(project)

    run, prepared = services.prepare_run(project)

    assert run.run_kind == RunKind.FULL
    assert "process" in prepared.arguments


def test_normal_analysis_completion_rejects_foreign_run_identity(tmp_path: Path) -> None:
    output = tmp_path / "run-output"
    output.mkdir()
    artifact_path = output / "analysis.json"
    _write_analysis_artifact(
        artifact_path, "project-a", "analysis-a", "fingerprint-a", ["candidate-a"],
    )
    report_path = output / "report.json"
    write_json(report_path, {
        "terminal": {"status": "analysis_ready"},
        "run": {
            "run_id": "run-other", "project_id": "project-a",
            "analysis_id": "analysis-a", "analysis_fingerprint": "fingerprint-a",
            "analysis_artifact_path": str(artifact_path),
        },
    })
    prepared = PreparedPipelineRun(
        program="", arguments=[], working_directory=tmp_path,
        state_path=output / "state.json", report_path=report_path,
        output_directory=output, runtime_config_path=tmp_path / "runtime.yaml",
        run_id="run-a", project_id="project-a", runtime_flags={"mode": "analysis"},
        allow_legacy_artifact_scan=False,
    )

    completion = PipelineFacade(tmp_path).completion(prepared)

    assert completion.error_summary is not None
    assert completion.technical_details == "Stage report run_id does not match the current Desktop run."


def test_normal_draft_completion_rejects_foreign_artifact_run(tmp_path: Path) -> None:
    output = tmp_path / "run-output"
    output.mkdir()
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis_path, "project-a", "analysis-a", "fingerprint-a", ["candidate-a"],
    )
    draft_path = output / "draft.json"
    new_draft_artifact(
        draft_id="draft-a", analysis_id="analysis-a", analysis_fingerprint="fingerprint-a",
        analysis_artifact_path=str(analysis_path), project_id="project-a",
        source_fingerprint="source-fingerprint", run_id="run-other",
        candidates=[{
            "candidate_id": "candidate-a", "state": "draft_failed",
            "requested_index": 1, "source_start_seconds": 1.0,
            "source_end_seconds": 18.0, "output_file": None,
        }],
    ).write(draft_path)
    report_path = output / "report.json"
    write_json(report_path, {
        "terminal": {"status": "draft_ready"},
        "run": {
            "run_id": "run-a", "project_id": "project-a", "analysis_id": "analysis-a",
            "draft_id": "draft-a", "draft_artifact_path": str(draft_path),
        },
        "candidate_flow": {"draft_candidates": [{
            "candidate_id": "candidate-a", "state": "draft_failed",
        }]},
    })
    prepared = PreparedPipelineRun(
        program="", arguments=[], working_directory=tmp_path,
        state_path=output / "state.json", report_path=report_path,
        output_directory=output, runtime_config_path=tmp_path / "runtime.yaml",
        run_id="run-a", project_id="project-a",
        runtime_flags={"mode": "draft", "analysis_id": "analysis-a"},
        expected_candidate_ids=("candidate-a",), allow_legacy_artifact_scan=False,
    )

    completion = PipelineFacade(tmp_path).completion(prepared)

    assert completion.error_summary is not None
    assert completion.technical_details == "Draft artifact belongs to another run."


def test_normal_draft_completion_accepts_verified_partial_success(
    tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "run-output"
    output.mkdir()
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis_path, "project-a", "analysis-a", "fingerprint-a",
        ["candidate-a", "candidate-b"],
    )
    preview = output / "drafts" / "01-candidate-a" / "draft-preview.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"preview")
    records = [
        {
            "candidate_id": "candidate-a", "state": "draft_ready", "requested_index": 1,
            "source_start_seconds": 1.0, "source_end_seconds": 18.0,
            "draft_final_script": {"candidate_id": "candidate-a"},
            "draft_production_plan": {"metadata": {"candidate_id": "candidate-a"}},
            "output_file": str(preview),
            "preview": {
                "status": "draft_ready", "output_file": str(preview),
                "segments": [{"source_start_seconds": 1.0, "source_end_seconds": 18.0}],
            },
        },
        {
            "candidate_id": "candidate-b", "state": "draft_failed", "requested_index": 2,
            "source_start_seconds": 20.0, "source_end_seconds": 29.0, "output_file": None,
        },
    ]
    draft_path = output / "draft.json"
    new_draft_artifact(
        draft_id="draft-a", analysis_id="analysis-a", analysis_fingerprint="fingerprint-a",
        analysis_artifact_path=str(analysis_path), project_id="project-a",
        source_fingerprint="source-fingerprint", run_id="run-a", status="draft_partial",
        candidates=records,
    ).write(draft_path)
    report_path = output / "report.json"
    write_json(report_path, {
        "terminal": {"status": "draft_ready"},
        "run": {
            "run_id": "run-a", "project_id": "project-a", "analysis_id": "analysis-a",
            "draft_id": "draft-a", "draft_artifact_path": str(draft_path),
        },
        "candidate_flow": {"draft_candidates": [
            {"candidate_id": record["candidate_id"], "state": record["state"]}
            for record in records
        ]},
    })
    prepared = PreparedPipelineRun(
        program="", arguments=[], working_directory=tmp_path,
        state_path=output / "state.json", report_path=report_path,
        output_directory=output, runtime_config_path=tmp_path / "runtime.yaml",
        run_id="run-a", project_id="project-a",
        runtime_flags={"mode": "draft", "analysis_id": "analysis-a"},
        expected_candidate_ids=("candidate-a", "candidate-b"),
        allow_legacy_artifact_scan=False,
    )
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    completion = PipelineFacade(tmp_path).completion(prepared)

    assert completion.error_summary is None
    assert completion.output_files == [preview]


def test_desktop_flow_prepares_analysis_then_draft_then_confirmed_production(
    tmp_path: Path, monkeypatch,
) -> None:
    services, project, source = _services(tmp_path)

    analysis_run, analysis_prepared = services.prepare_analysis(project)
    assert analysis_run.run_kind == RunKind.ANALYSIS
    assert "analyze" in analysis_prepared.arguments
    assert "--transform-script" not in analysis_prepared.arguments
    assert project.status == ProjectStatus.ANALYZING

    analysis_path = analysis_prepared.output_directory / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    _write_analysis_artifact(
        analysis_path, project.project_id, "analysis-001", "fingerprint-001",
        ["candidate-a", "candidate-b"], analysis_run_id=analysis_run.run_id,
    )
    write_json(analysis_prepared.report_path, {
        "terminal": {"status": "analysis_ready"}, "output_files": [], "warnings": [],
        "run": {
            "run_id": analysis_run.run_id, "project_id": project.project_id,
            "analysis_id": "analysis-001", "analysis_fingerprint": "fingerprint-001",
            "analysis_artifact_path": str(analysis_path),
        },
        "clip_intelligence": {"candidates": [{"id": "candidate-a"}, {"id": "candidate-b"}]},
    })
    finished_analysis = services.finish_success(project, analysis_run, analysis_prepared)
    assert finished_analysis.status == RunStatus.ANALYSIS_READY
    assert project.status == ProjectStatus.ANALYSIS_READY
    assert project.candidate_states == {"candidate-a": "analyzed", "candidate-b": "analyzed"}
    services.set_review_selection(project, ["candidate-a"])
    assert services.projects.load(project.project_id).review_selected_candidate_ids == ["candidate-a"]

    draft_run, draft_prepared = services.prepare_draft(project, ["candidate-a"])
    assert draft_run.run_kind == RunKind.DRAFT
    assert "draft" in draft_prepared.arguments
    assert "render" not in draft_prepared.arguments
    assert project.candidate_states["candidate-a"] == "draft_planning"

    preview = draft_prepared.output_directory / "drafts" / "01-candidate-a" / "draft-preview.mp4"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"preview")
    draft_path = draft_prepared.output_directory / "draft.json"
    new_draft_artifact(
        draft_id="draft-001", analysis_id="analysis-001", analysis_fingerprint="fingerprint-001",
        analysis_artifact_path=str(analysis_path), project_id=project.project_id,
        source_fingerprint="source-fingerprint", run_id=draft_run.run_id, candidates=[{
            "candidate_id": "candidate-a", "state": "draft_ready",
            "requested_index": 1, "source_start_seconds": 1.0, "source_end_seconds": 18.0,
            "draft_final_script": {"candidate_id": "candidate-a"},
            "draft_production_plan": {"plan_id": "plan-a", "metadata": {"candidate_id": "candidate-a"}},
            "output_file": str(preview),
            "preview": {
                "status": "draft_ready", "output_file": str(preview),
                "segments": [{"source_start_seconds": 1.0, "source_end_seconds": 18.0}],
            },
        }],
    ).write(draft_path)
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))
    write_json(draft_prepared.report_path, {
        "terminal": {"status": "draft_ready"}, "output_files": [str(preview)], "warnings": [],
        "run": {
            "run_id": draft_run.run_id, "project_id": project.project_id,
            "analysis_id": "analysis-001", "draft_id": "draft-001",
            "draft_artifact_path": str(draft_path),
        },
        "candidate_flow": {"draft_candidates": [{"candidate_id": "candidate-a", "state": "draft_ready"}]},
    })
    finished_draft = services.finish_success(project, draft_run, draft_prepared)
    assert finished_draft.status == RunStatus.DRAFT_READY
    assert project.candidate_draft_artifacts == {"candidate-a": str(draft_path)}
    recovered = services.projects.load(project.project_id)
    assert recovered.candidate_states["candidate-a"] == "draft_ready"
    assert recovered.candidate_draft_artifacts == {"candidate-a": str(draft_path)}

    # A later draft can be prepared independently.  The final hand-off joins
    # both immutable draft plans in the user's selected order.
    draft_b_path = tmp_path / "draft-b.json"
    new_draft_artifact(
        draft_id="draft-002", analysis_id="analysis-001", analysis_fingerprint="fingerprint-001",
        analysis_artifact_path=str(analysis_path), project_id=project.project_id,
        source_fingerprint="source-fingerprint", candidates=[{
            "candidate_id": "candidate-b", "state": "draft_ready",
            "draft_production_plan": {"plan_id": "plan-b"},
        }],
    ).write(draft_b_path)
    project.candidate_states["candidate-b"] = "draft_ready"
    project.candidate_draft_artifacts["candidate-b"] = str(draft_b_path)
    services.projects.save(project)
    services.select_draft_candidates(project, ["candidate-b", "candidate-a"])
    assert project.review_selected_candidate_ids == ["candidate-b", "candidate-a"]
    production_run, production_prepared = services.prepare_selected_render(project)
    assert production_run.run_kind == RunKind.SELECTED_RENDER
    assert "render" in production_prepared.arguments
    assert "--draft" in production_prepared.arguments
    assert "--confirm-production" in production_prepared.arguments
    assert project.status == ProjectStatus.RENDERING_SELECTED
    approved_path = Path(production_prepared.arguments[production_prepared.arguments.index("--draft") + 1])
    assert [item["candidate_id"] for item in read_json(approved_path, {})["candidates"]] == ["candidate-b", "candidate-a"]


def test_targeted_preview_failure_keeps_previous_valid_candidate_preview(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis_path, project.project_id, "analysis-targeted", "fingerprint-targeted", ["candidate-a"],
    )
    previous_draft = tmp_path / "previous-draft.json"
    previous_draft.write_text("{}", encoding="utf-8")
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "analysis-targeted"
    project.analysis_fingerprint = "fingerprint-targeted"
    project.status = ProjectStatus.REVIEWING_CANDIDATES
    project.candidate_states = {"candidate-a": "draft_ready"}
    project.candidate_draft_statuses = {"candidate-a": "ready"}
    project.candidate_approval_states = {"candidate-a": "pending"}
    project.candidate_export_statuses = {"candidate-a": "pending"}
    project.review_selected_candidate_ids = ["candidate-a"]
    project.candidate_draft_artifacts = {"candidate-a": str(previous_draft)}
    services.projects.save(project)

    services.update_project_options(project, same_source_broll_allowed=True)
    assert project.setup_state.needs_new_analysis is False
    assert project.candidate_draft_statuses["candidate-a"] == "pending"
    assert project.candidate_draft_artifacts["candidate-a"] == str(previous_draft)

    run, prepared = services.prepare_draft(project, ["candidate-a"])
    assert run.settings_snapshot["previous_draft_artifacts"] == {
        "candidate-a": str(previous_draft),
    }
    assert project.candidate_draft_artifacts["candidate-a"] == str(previous_draft)

    failed_artifact = prepared.output_directory / "failed-revision.json"
    failed_artifact.parent.mkdir(parents=True, exist_ok=True)
    failed_artifact.write_text("{}", encoding="utf-8")
    write_json(prepared.report_path, {
        "terminal": {
            "status": "failed", "error_code": "NO_DRAFT_PREVIEWS",
            "message": "new revision failed",
        },
        "output_files": [],
        "warnings": [],
        "run": {
            "run_id": run.run_id, "project_id": project.project_id,
            "analysis_id": project.analysis_id,
            "draft_id": "draft-targeted", "draft_artifact_path": str(failed_artifact),
        },
        "candidate_flow": {"draft_candidates": [{
            "candidate_id": "candidate-a", "state": "draft_failed",
            "error": "new revision failed",
        }]},
    })
    assert services.recover_reported_failure(project, run, prepared) is run

    assert project.candidate_states["candidate-a"] == "draft_ready"
    assert project.candidate_draft_artifacts["candidate-a"] == str(previous_draft)
    assert "Предыдущая готовая версия сохранена" in project.candidate_errors["candidate-a"]


def test_same_source_broll_is_explicit_opt_in_in_runtime_config(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    assert project.settings.same_source_broll_allowed is False

    services.update_project_options(project, same_source_broll_allowed=True)
    _run, prepared = services.prepare_analysis(project)
    runtime = load_config(prepared.runtime_config_path)

    assert project.settings.same_source_broll_allowed is True
    assert runtime.production_render.same_source_broll_allowed is True


def test_approved_render_enables_the_required_delivery_stages() -> None:
    config = AppConfig()

    _apply_render_command_arguments(config, SimpleNamespace(
        output_width=None,
        output_height=None,
        output_fps=None,
        crop_strategy=None,
        subtitle_style=None,
        disable_subtitles=False,
        video_encoder=None,
    ))

    assert config.production.enabled is True
    assert config.tts.enabled is True
    assert config.audio_composition.enabled is True
    assert config.production_render.enabled is True


def test_draft_command_enables_real_creative_preview_stages() -> None:
    config = AppConfig()

    _apply_draft_command_arguments(config)

    assert config.transformation.enabled is True
    assert config.production.enabled is True
    assert config.tts.enabled is True
    assert config.audio_composition.enabled is True
    assert config.production_render.enabled is True


def test_individual_draft_approval_is_persistent_and_does_not_change_review_selection(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    draft_a = tmp_path / "draft-a.json"; draft_a.write_text("{}", encoding="utf-8")
    draft_b = tmp_path / "draft-b.json"; draft_b.write_text("{}", encoding="utf-8")
    project.candidate_states = {"candidate-a": "draft_ready", "candidate-b": "draft_ready"}
    project.candidate_draft_artifacts = {"candidate-a": str(draft_a), "candidate-b": str(draft_b)}
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    services.projects.save(project)

    services.set_draft_approval(project, "candidate-a", True)

    assert project.review_selected_candidate_ids == ["candidate-a", "candidate-b"]
    assert project.selected_candidate_ids == ["candidate-a"]
    assert project.candidate_states["candidate-a"] == "selected"
    restored = services.projects.load(project.project_id)
    assert restored.selected_candidate_ids == ["candidate-a"]

    services.set_draft_approval(project, "candidate-a", False)

    assert project.review_selected_candidate_ids == ["candidate-a", "candidate-b"]
    assert project.selected_candidate_ids == []
    assert project.candidate_states["candidate-a"] == "draft_ready"


def test_candidate_lifecycle_axes_and_active_preview_survive_project_reload(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    project.candidate_states = {
        "candidate-ready": "draft_ready",
        "candidate-exporting": "production_rendering",
        "candidate-failed": "draft_failed",
    }
    project.candidate_draft_statuses = {
        "candidate-ready": "ready",
        "candidate-exporting": "ready",
        "candidate-failed": "failed",
    }
    project.candidate_approval_states = {
        "candidate-ready": "pending",
        "candidate-exporting": "approved",
        "candidate-failed": "rejected",
    }
    project.candidate_export_statuses = {
        "candidate-ready": "pending",
        "candidate-exporting": "running",
        "candidate-failed": "pending",
    }
    project.review_selected_candidate_ids = ["candidate-ready", "candidate-exporting", "candidate-failed"]
    project.selected_candidate_ids = ["candidate-exporting"]
    project.active_preview_candidate_id = "candidate-ready"
    services.projects.save(project)

    restored = services.projects.load(project.project_id)

    assert restored.candidate_draft_statuses == project.candidate_draft_statuses
    assert restored.candidate_approval_states == project.candidate_approval_states
    assert restored.candidate_export_statuses == project.candidate_export_statuses
    assert restored.active_preview_candidate_id == "candidate-ready"


def test_draft_retry_omits_ready_artifacts_and_launches_only_failed_candidates(tmp_path: Path, monkeypatch) -> None:
    services, project, _source = _services(tmp_path)
    analysis = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis, project.project_id, "analysis-a", "fingerprint-a", ["candidate-ready", "candidate-failed"],
    )
    ready_draft = tmp_path / "ready-draft.json"
    ready_draft.write_text("{}", encoding="utf-8")
    project.analysis_artifact_path = str(analysis)
    project.analysis_id = "analysis-a"
    project.review_selected_candidate_ids = ["candidate-ready", "candidate-failed"]
    project.candidate_states = {"candidate-ready": "draft_ready", "candidate-failed": "draft_failed"}
    project.candidate_draft_statuses = {"candidate-ready": "ready", "candidate-failed": "failed"}
    project.candidate_draft_artifacts = {"candidate-ready": str(ready_draft)}
    services.projects.save(project)
    launched: list[list[str]] = []

    def fake_prepare(current, run, _settings, candidate_ids):
        launched.append(list(candidate_ids))
        return PreparedPipelineRun(
            program="python", arguments=["-m", "app", "draft"], working_directory=tmp_path,
            state_path=tmp_path / "state.json", report_path=tmp_path / "report.json",
            output_directory=tmp_path / "output", runtime_config_path=tmp_path / "runtime.yaml",
            run_id=run.run_id, project_id=current.project_id, runtime_flags={"mode": "draft"},
        )

    monkeypatch.setattr(services.pipeline, "prepare_draft", fake_prepare)

    run, _prepared = services.prepare_draft(project, ["candidate-ready", "candidate-failed"])

    assert launched == [["candidate-failed"]]
    assert run.settings_snapshot["candidate_ids"] == ["candidate-failed"]
    assert project.candidate_draft_statuses["candidate-ready"] == "ready"
    assert project.candidate_states["candidate-ready"] == "draft_ready"
    assert project.candidate_draft_statuses["candidate-failed"] == "running"


def test_run_history_keeps_each_candidate_preview_when_file_names_match(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    run = services.runs.create(project, {}, {}, "test", run_kind=RunKind.DRAFT)
    previews = []
    for index, payload in enumerate((b"candidate-a", b"candidate-b", b"candidate-c"), start=1):
        preview = tmp_path / f"candidate-{index}" / "draft-preview.mp4"
        preview.parent.mkdir(parents=True)
        preview.write_bytes(payload)
        previews.append(preview)

    services.runs.snapshot_report_and_outputs(run, tmp_path / "missing-report.json", previews)

    stored = [Path(path) for path in run.artifact_paths]
    assert [path.name for path in stored] == [
        "draft-preview.mp4", "02-draft-preview.mp4", "03-draft-preview.mp4",
    ]
    assert [path.read_bytes() for path in stored] == [b"candidate-a", b"candidate-b", b"candidate-c"]


def test_completed_selected_render_with_warnings_is_not_marked_partial(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    project.selected_candidate_ids = ["candidate-a"]
    project.candidate_states = {"candidate-a": "production_rendering"}
    services.projects.save(project)
    run = services.runs.create(
        project, {}, {}, "test", run_kind=RunKind.SELECTED_RENDER,
    )
    final = tmp_path / "final.mp4"
    final.write_bytes(b"final")
    report_path = tmp_path / "report.json"
    write_json(report_path, {
        "production_render": {
            "items": [{"candidate_id": "candidate-a", "status": "warning"}],
        },
    })
    completion = PipelineCompletion(
        report_path, [final], ["Subtitle fallback fit used"], None, None, 0.0,
    )

    finished = services._finish_completion(project, run, completion)

    assert finished.status == RunStatus.COMPLETED_WITH_WARNINGS
    assert project.status == ProjectStatus.COMPLETED_WITH_WARNINGS
    assert project.selected_candidate_ids == []
    assert project.candidate_states["candidate-a"] == "rendered"


def test_canonical_final_results_override_divergent_report_projection(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    project.selected_candidate_ids = ["candidate-a"]
    project.candidate_states = {"candidate-a": "production_rendering"}
    project.candidate_draft_statuses = {"candidate-a": "ready"}
    project.candidate_approval_states = {"candidate-a": "approved"}
    project.candidate_export_statuses = {"candidate-a": "running"}
    services.projects.save(project)
    run = services.runs.create(
        project, {"candidate_ids": ["candidate-a"]}, {}, "test",
        run_kind=RunKind.SELECTED_RENDER,
    )
    final = tmp_path / "canonical-final.mp4"
    final.write_bytes(b"canonical final")
    untrusted = tmp_path / "untrusted-report-final.mp4"
    untrusted.write_bytes(b"untrusted report final")
    report_path = tmp_path / "divergent-report.json"
    write_json(report_path, {
        "production_render": {
            "status": "partial",
            "items": [{
                "candidate_id": "candidate-a", "status": "failed",
                "error": "stale report projection",
            }],
        },
        "candidate_flow": {"items": [{
            "candidate_id": "candidate-a", "status": "failed",
            "error": "stale report projection",
        }]},
        "primary_results": [{
            "clip_result_id": "untrusted-report-result",
            "candidate_id": "candidate-a",
            "production_plan_id": "untrusted-plan",
            "output_file": str(untrusted),
            "run_id": run.run_id,
            "status": "completed",
            "primary": True,
        }],
    })
    canonical = ClipResult(
        "candidate-a", str(final),
        clip_result_id="canonical-result",
        production_plan_id="canonical-plan",
        run_id=run.run_id,
        revision_id=f"{run.run_id}:canonical-revision",
    )
    completion = PipelineCompletion(
        report_path, [final], [], None, None, 0.0,
        canonical_results=True,
        validated_results=(canonical,),
    )

    finished = services._finish_completion(project, run, completion)

    assert finished.status == RunStatus.COMPLETED
    assert project.status == ProjectStatus.COMPLETED
    assert project.candidate_states["candidate-a"] == "rendered"
    assert project.candidate_export_statuses["candidate-a"] == "ready"
    assert project.candidate_errors.get("candidate-a") is None
    assert project.selected_candidate_ids == []
    assert project.last_final_result_id == "canonical-result"
    assert services.projects.load(project.project_id).last_final_result_id == "canonical-result"


def _reported_failure_prepared(tmp_path: Path, report_path: Path, mode: str) -> PreparedPipelineRun:
    return PreparedPipelineRun(
        program="", arguments=[], working_directory=tmp_path,
        state_path=tmp_path / "state.json", report_path=report_path,
        output_directory=tmp_path, runtime_config_path=tmp_path / "runtime-config.yaml",
        runtime_flags={"mode": mode},
    )


def test_terminal_draft_failure_report_keeps_item_level_retry_state(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis, project.project_id, "analysis-a", "fingerprint-a", ["candidate-a", "candidate-b"],
    )
    project.analysis_artifact_path = str(analysis)
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.candidate_states = {"candidate-a": "draft_planning", "candidate-b": "draft_planning"}
    project.status = ProjectStatus.PROCESSING
    services.projects.save(project)
    run = services.runs.create(
        project, {"candidate_ids": ["candidate-a", "candidate-b"]}, {}, "test", run_kind=RunKind.DRAFT,
    )
    report_path = tmp_path / "failed-draft-report.json"
    write_json(report_path, {
        "terminal": {
            "status": "failed", "error_code": "NO_DRAFT_PREVIEWS",
            "message": "No candidate draft could be assembled.",
        },
        "run": {"run_id": run.run_id, "project_id": project.project_id},
        "candidate_flow": {"draft_candidates": [
            {
                "candidate_id": "candidate-a", "state": "draft_failed",
                "error": "Boundary completion was lost.", "stage": "draft_generation:candidate-a",
            },
            {
                "candidate_id": "candidate-b", "state": "draft_failed",
                "error": "Boundary payoff was lost.", "stage": "draft_generation:candidate-b",
            },
        ]},
    })

    restored = services.recover_reported_failure(
        project, run, _reported_failure_prepared(tmp_path, report_path, "draft"),
    )

    assert restored is run
    assert run.status == RunStatus.FAILED
    assert Path(run.report_path or "").is_file()
    assert project.status == ProjectStatus.REVIEWING_CANDIDATES
    assert project.candidate_states == {"candidate-a": "draft_failed", "candidate-b": "draft_failed"}
    assert project.candidate_draft_statuses == {"candidate-a": "failed", "candidate-b": "failed"}
    assert project.review_selected_candidate_ids == []
    assert "draft_generation:candidate-a" in project.candidate_errors["candidate-a"]
    finished_at = run.finished_at
    assert services.recover_reported_failure(
        project, run, _reported_failure_prepared(tmp_path, report_path, "draft"),
    ) is run
    assert run.finished_at == finished_at
    run.settings_snapshot["execution"] = {
        "state_path": str(tmp_path / "state.json"),
        "report_path": str(report_path),
        "output_directory": str(tmp_path),
        "runtime_flags": {"mode": "draft"},
        "run_id": run.run_id,
        "project_id": project.project_id,
    }
    services.runs.save(run)
    # Simulate a shutdown precisely after the terminal run snapshot but before
    # its user-facing project projection was persisted.  Restart recovery must
    # replay the report rather than collapsing the candidates into a generic
    # process failure.
    project.status = ProjectStatus.PROCESSING
    project.candidate_states = {"candidate-a": "draft_planning", "candidate-b": "draft_planning"}
    project.candidate_draft_statuses = {"candidate-a": "running", "candidate-b": "running"}
    services.projects.save(project)
    assert services.recover_interrupted_runs() == 0
    restored_project = services.projects.load(project.project_id)
    assert services.runs.load(project.project_id, run.run_id).finished_at == finished_at
    assert restored_project.status == ProjectStatus.REVIEWING_CANDIDATES
    assert restored_project.candidate_draft_statuses == {"candidate-a": "failed", "candidate-b": "failed"}
    assert restored_project.review_selected_candidate_ids == []


def test_boundary_terminal_failure_requires_boundary_change_before_requeue(tmp_path: Path, monkeypatch) -> None:
    services, project, _source = _services(tmp_path)
    candidate_id = "candidate-a"
    project.candidate_states = {candidate_id: "draft_failed"}
    project.candidate_draft_statuses = {candidate_id: "failed"}
    project.candidate_approval_states = {candidate_id: "pending"}
    project.candidate_export_statuses = {candidate_id: "pending"}
    project.candidate_errors[candidate_id] = "BOUNDARY_DECISION_REQUIRED: missing evidence"
    project.status = ProjectStatus.REVIEWING_CANDIDATES
    services.projects.save(project)

    # Caption, style, and crop edits remain durable, but cannot erase evidence
    # that the source interval itself still needs a boundary decision.
    services.update_candidate_creative_override(
        project,
        candidate_id,
        creative_style="dynamic",
        caption_preset_id="clean_white",
    )
    assert project.candidate_states[candidate_id] == "draft_failed"
    assert project.candidate_draft_statuses[candidate_id] == "failed"
    assert project.review_selected_candidate_ids == []
    assert "BOUNDARY_DECISION_REQUIRED" in project.candidate_errors[candidate_id]
    restored = services.projects.load(project.project_id)
    assert restored.candidate_draft_statuses[candidate_id] == "failed"
    assert restored.review_selected_candidate_ids == []
    assert restored.candidate_creative_overrides[candidate_id] == {
        "creative_style": "dynamic",
        "caption_preset_id": "clean_white",
    }

    # Restoring the card or calling the service directly cannot create a run.
    services.set_review_selection(project, [candidate_id])
    assert project.review_selected_candidate_ids == []

    analysis = SimpleNamespace(
        candidates=[{"candidate_id": candidate_id, "start": 2.0, "end": 20.0}],
        load_reference=lambda _name: {},
    )
    project.analysis_artifact_path = str(tmp_path / "analysis.json")
    monkeypatch.setattr(
        services.pipeline, "load_verified_analysis", lambda *_args, **_kwargs: analysis,
    )
    with pytest.raises(InputValidationError, match="Сначала измените начало или конец"):
        services.prepare_draft(project, [candidate_id])
    assert services.runs.list(project.project_id) == []
    monkeypatch.setattr(
        "app.gui.services.desktop_services.validate_boundary_override",
        lambda *_args, **_kwargs: {
            "valid": True, "start": 2.5, "end": 20.0,
            "warnings": [], "revalidation": {"status": "valid"},
        },
    )

    # A relevant, validated boundary change clears exactly this retry gate.
    services.adjust_candidate_boundary(project, candidate_id, "start", 0.5)
    assert project.candidate_states[candidate_id] == "analyzed"
    assert project.candidate_draft_statuses[candidate_id] == "pending"
    assert project.review_selected_candidate_ids == [candidate_id]
    assert candidate_id not in project.candidate_errors


def test_reported_failure_requires_current_run_and_project_identity(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    write_json(report_path, {
        "terminal": {"status": "failed", "error_code": "NO_DRAFT_PREVIEWS"},
        "run": {"run_id": "other-run", "project_id": "project-a"},
    })
    prepared = PreparedPipelineRun(
        program="", arguments=[], working_directory=tmp_path,
        state_path=tmp_path / "state.json", report_path=report_path,
        output_directory=tmp_path, runtime_config_path=tmp_path / "runtime-config.yaml",
        runtime_flags={"mode": "draft"}, run_id="expected-run", project_id="project-a",
    )

    assert PipelineFacade(tmp_path).reported_failure(prepared, "2026-01-01T00:00:00+00:00") is None

    write_json(report_path, {
        "terminal": {"status": "failed", "error_code": "NO_DRAFT_PREVIEWS"},
        "run": {"run_id": "expected-run", "project_id": "project-a"},
    })
    assert PipelineFacade(tmp_path).reported_failure(prepared, "2026-01-01T00:00:00+00:00") is not None

    write_json(report_path, {
        "terminal": {"status": "failed", "error_code": "QUALITY_GATE_BLOCKED"},
        "run": {"run_id": "expected-run", "project_id": "project-a"},
    })
    assert PipelineFacade(tmp_path).reported_failure(prepared, "2026-01-01T00:00:00+00:00") is None


def test_cancelled_draft_retry_reconciles_only_its_persisted_candidate_ids(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    project.analysis_artifact_path = str(tmp_path / "analysis.json")
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.candidate_states = {
        "candidate-a": "draft_planning",
        "candidate-b": "draft_planning",
    }
    project.candidate_draft_statuses = {
        "candidate-a": "running",
        "candidate-b": "running",
    }
    services.projects.save(project)
    retry = services.runs.create(
        project, {"candidate_ids": ["candidate-a"]}, {}, "test", run_kind=RunKind.DRAFT,
    )

    services.finish_cancelled(project, retry)

    assert project.candidate_states["candidate-a"] == "analyzed"
    assert project.candidate_draft_statuses["candidate-a"] == "pending"
    assert project.candidate_states["candidate-b"] == "draft_planning"
    assert project.candidate_draft_statuses["candidate-b"] == "running"


def test_terminal_selected_render_failure_returns_only_invalid_drafts_to_retry(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    project.selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.candidate_states = {"candidate-a": "production_rendering", "candidate-b": "production_rendering"}
    project.status = ProjectStatus.RENDERING_SELECTED
    services.projects.save(project)
    run = services.runs.create(
        project, {"candidate_ids": ["candidate-a", "candidate-b"]}, {}, "test",
        run_kind=RunKind.SELECTED_RENDER,
    )
    report_path = tmp_path / "failed-render-report.json"
    write_json(report_path, {
        "terminal": {
            "status": "failed", "error_code": "NO_RENDERABLE_CLIPS",
            "message": "No approved draft could be rendered.",
        },
        "candidate_flow": {"items": [
            {
                "candidate_id": "candidate-a", "outcome": "failed", "reason": "production_plan_failed",
                "message": "Draft ProductionPlan is invalid.", "stage": "approved_draft_plan:candidate-a",
            },
            {
                "candidate_id": "candidate-b", "outcome": "failed", "reason": "production_plan_failed",
                "message": "Draft ProductionPlan is missing.",
            },
        ]},
    })

    restored = services.recover_reported_failure(
        project, run, _reported_failure_prepared(tmp_path, report_path, "selected_render"),
    )

    assert restored is run
    assert run.status == RunStatus.FAILED
    assert project.status == ProjectStatus.REVIEWING_CANDIDATES
    assert project.selected_candidate_ids == []
    assert project.candidate_states == {"candidate-a": "draft_failed", "candidate-b": "draft_failed"}
    assert project.candidate_approval_states == {"candidate-a": "pending", "candidate-b": "pending"}
    assert "Draft ProductionPlan is missing" in project.candidate_errors["candidate-b"]


def test_selected_render_skips_only_missing_approved_draft(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis, project.project_id, "analysis-a", "fingerprint-a", ["candidate-a", "candidate-b"],
    )
    ready_draft = tmp_path / "draft-a.json"
    new_draft_artifact(
        draft_id="draft-a", analysis_id="analysis-a", analysis_fingerprint="fingerprint-a",
        analysis_artifact_path=str(analysis), project_id=project.project_id,
        source_fingerprint="source-fingerprint", candidates=[{
            "candidate_id": "candidate-a", "state": "draft_ready",
            "draft_production_plan": {"plan_id": "plan-a"},
        }],
    ).write(ready_draft)
    stale_draft = tmp_path / "draft-b-stale-analysis.json"
    new_draft_artifact(
        draft_id="draft-b", analysis_id="analysis-a", analysis_fingerprint="fingerprint-a",
        analysis_artifact_path=str(tmp_path / "missing-analysis.json"), project_id=project.project_id,
        source_fingerprint="source-fingerprint", candidates=[{
            "candidate_id": "candidate-b", "state": "draft_ready",
            "draft_production_plan": {"plan_id": "plan-b"},
        }],
    ).write(stale_draft)
    project.analysis_artifact_path = str(analysis)
    project.analysis_id = "analysis-a"
    project.analysis_fingerprint = "fingerprint-a"
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.candidate_states = {"candidate-a": "selected", "candidate-b": "selected"}
    project.candidate_draft_artifacts = {
        "candidate-a": str(ready_draft), "candidate-b": str(stale_draft),
    }
    services.projects.save(project)

    run, prepared = services.prepare_selected_render(project)

    assert run.settings_snapshot["candidate_ids"] == ["candidate-a"]
    assert prepared.arguments.count("--candidate-id") == 1
    assert "candidate-a" in prepared.arguments
    assert "candidate-b" not in prepared.arguments
    assert project.selected_candidate_ids == ["candidate-a"]
    assert project.candidate_states["candidate-a"] == "production_rendering"
    assert project.candidate_states["candidate-b"] == "draft_failed"
    assert "исходный анализ" in project.candidate_errors["candidate-b"]
    restored = services.projects.load(project.project_id)
    assert restored.candidate_states["candidate-b"] == "draft_failed"


def test_single_final_export_retry_keeps_other_approved_candidates_queued(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis, project.project_id, "analysis-a", "fingerprint-a", ["candidate-a", "candidate-b"],
    )
    artifacts: dict[str, str] = {}
    for candidate_id in ("candidate-a", "candidate-b"):
        draft_path = tmp_path / f"draft-{candidate_id}.json"
        new_draft_artifact(
            draft_id=f"draft-{candidate_id}", analysis_id="analysis-a", analysis_fingerprint="fingerprint-a",
            analysis_artifact_path=str(analysis), project_id=project.project_id,
            source_fingerprint="source-fingerprint", candidates=[{
                "candidate_id": candidate_id, "state": "draft_ready",
                "draft_production_plan": {"plan_id": f"plan-{candidate_id}"},
            }],
        ).write(draft_path)
        artifacts[candidate_id] = str(draft_path)
    project.analysis_artifact_path = str(analysis)
    project.analysis_id = "analysis-a"
    project.analysis_fingerprint = "fingerprint-a"
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.candidate_states = {"candidate-a": "selected", "candidate-b": "selected"}
    project.candidate_draft_artifacts = artifacts
    project.candidate_draft_statuses = {"candidate-a": "ready", "candidate-b": "ready"}
    project.candidate_approval_states = {"candidate-a": "approved", "candidate-b": "approved"}
    project.candidate_export_statuses = {"candidate-a": "failed", "candidate-b": "failed"}
    services.projects.save(project)

    run, prepared = services.prepare_selected_render(project, ["candidate-a"])

    assert run.settings_snapshot["candidate_ids"] == ["candidate-a"]
    assert prepared.arguments.count("--candidate-id") == 1
    assert "candidate-a" in prepared.arguments
    assert "candidate-b" not in prepared.arguments
    assert project.selected_candidate_ids == ["candidate-a", "candidate-b"]
    assert project.candidate_export_statuses["candidate-a"] == "running"
    assert project.candidate_export_statuses["candidate-b"] == "failed"

    final = tmp_path / "final-a.mp4"
    final.write_bytes(b"valid final artifact")
    result_id = "candidate-a:plan-a"
    write_json(prepared.report_path, {
        "terminal": {"status": "completed"},
        "output_files": [str(final)],
        "warnings": [],
        "production_render": {"status": "completed", "output_file": str(final), "items": [{
            "candidate_id": "candidate-a", "status": "completed", "output_file": str(final),
        }]},
        "primary_results": [{
            "clip_result_id": result_id,
            "candidate_id": "candidate-a",
            "production_plan_id": "plan-a",
            "output_file": str(final),
            "run_id": run.run_id,
            "status": "completed",
            "primary": True,
        }],
        "candidate_flow": {"items": [{"candidate_id": "candidate-a", "status": "completed"}]},
    })

    finished = services._finish_completion(
        project, run, PipelineCompletion(prepared.report_path, [final], [], None, None, 0.0),
    )

    assert finished.status == RunStatus.COMPLETED
    assert project.status == ProjectStatus.PARTIALLY_RENDERED
    assert project.selected_candidate_ids == ["candidate-b"]
    assert project.candidate_export_statuses["candidate-a"] == "ready"
    assert project.candidate_export_statuses["candidate-b"] == "failed"
    assert project.last_final_result_id == result_id
    assert services.projects.load(project.project_id).last_final_result_id == result_id


def test_stale_ready_draft_is_regenerated_without_rebuilding_other_ready_drafts(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis, project.project_id, "analysis-a", "fingerprint-a", ["candidate-a", "candidate-b"],
    )
    ready_artifact = tmp_path / "draft-ready.json"
    ready_artifact.write_text("{}", encoding="utf-8")
    missing_artifact = tmp_path / "deleted-draft.json"
    project.analysis_artifact_path = str(analysis)
    project.analysis_id = "analysis-a"
    project.analysis_fingerprint = "fingerprint-a"
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.selected_candidate_ids = ["candidate-b"]
    project.candidate_states = {"candidate-a": "draft_ready", "candidate-b": "selected"}
    project.candidate_draft_statuses = {"candidate-a": "ready", "candidate-b": "ready"}
    project.candidate_approval_states = {"candidate-a": "pending", "candidate-b": "approved"}
    project.candidate_export_statuses = {"candidate-a": "pending", "candidate-b": "failed"}
    project.candidate_draft_artifacts = {"candidate-a": str(ready_artifact), "candidate-b": str(missing_artifact)}
    services.projects.save(project)

    run, prepared = services.prepare_draft(project, ["candidate-a", "candidate-b"])

    assert run.settings_snapshot["candidate_ids"] == ["candidate-b"]
    assert prepared.arguments.count("--candidate-id") == 1
    assert "candidate-b" in prepared.arguments
    assert "candidate-a" not in prepared.arguments
    assert project.selected_candidate_ids == []
    assert project.candidate_states["candidate-a"] == "draft_ready"
    assert project.candidate_states["candidate-b"] == "draft_planning"


def test_restart_migrates_legacy_partial_draft_artifact_to_individual_retry(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    analysis = tmp_path / "analysis.json"
    _write_analysis_artifact(
        analysis, project.project_id, "analysis-a", "fingerprint-a", ["candidate-a", "candidate-b", "candidate-c"],
    )
    draft_path = tmp_path / "legacy-partial-draft.json"
    new_draft_artifact(
        draft_id="draft-partial", analysis_id="analysis-a", analysis_fingerprint="fingerprint-a",
        analysis_artifact_path=str(analysis), project_id=project.project_id,
        source_fingerprint="source-fingerprint", candidates=[
            {"candidate_id": "candidate-a", "state": "draft_ready", "draft_production_plan": {"plan_id": "plan-a"}},
            {"candidate_id": "candidate-b", "state": "draft_ready", "draft_production_plan": {"plan_id": "plan-b"}},
            {"candidate_id": "candidate-c", "state": "draft_failed", "error": "BOUNDARY_PAYOFF_LOST"},
        ],
    ).write(draft_path)
    project.analysis_artifact_path = str(analysis)
    project.analysis_id = "analysis-a"
    project.analysis_fingerprint = "fingerprint-a"
    project.draft_artifact_path = str(draft_path)
    project.review_selected_candidate_ids = ["candidate-a", "candidate-b"]
    project.candidate_states = {
        "candidate-a": "rendered", "candidate-b": "rendered", "candidate-c": "analyzed",
    }
    project.status = ProjectStatus.FAILED
    services.projects.save(project)

    # Simulate the pre-lifecycle project representation: only combined
    # states and the aggregate draft artifact survived the earlier code.
    raw = read_json(services.projects.project_path(project.project_id), {})
    for key in (
        "candidate_draft_artifacts", "candidate_draft_statuses",
        "candidate_approval_states", "candidate_export_statuses",
    ):
        raw.pop(key, None)
    write_json(services.projects.project_path(project.project_id), raw)

    assert services.recover_interrupted_runs() == 0

    restored = services.projects.load(project.project_id)
    assert restored.status == ProjectStatus.REVIEWING_CANDIDATES
    assert restored.review_selected_candidate_ids == ["candidate-a", "candidate-b"]
    assert restored.candidate_states["candidate-c"] == "draft_failed"
    assert restored.candidate_draft_statuses["candidate-c"] == "failed"
    assert "границ" in restored.candidate_errors["candidate-c"]

    # The migration keeps terminal failures out of the persisted Draft queue.
    # A later restart must not silently re-add the failed candidate.
    services.recover_interrupted_runs()
    assert "candidate-c" not in services.projects.load(project.project_id).review_selected_candidate_ids
