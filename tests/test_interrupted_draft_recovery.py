from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis_artifact import AnalysisArtifact
from app.draft_artifact import new_draft_artifact
from app.gui.models import ProcessingPhase, ProjectStatus, RunKind, RunStatus
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.viewmodels import ProjectViewModel

from test_draft_workflow import _services
from test_gui_analysis_integrity import _write_verified_analysis


def _progress_candidate(
    candidate_id: str,
    index: int,
    *,
    state: str = "draft_planning",
    output_file: Path | None = None,
) -> dict:
    record = {
        "candidate_id": candidate_id,
        "state": state,
        "requested_index": index,
        "source_start_seconds": 10.0 * index,
        "source_end_seconds": 10.0 * index + 5.0,
        "output_file": None,
    }
    if state == "draft_ready":
        assert output_file is not None
        record.update({
            "output_file": str(output_file),
            "draft_final_script": {"candidate_id": candidate_id},
            "draft_production_plan": {"metadata": {"candidate_id": candidate_id}},
            "preview": {
                "status": "draft_ready",
                "output_file": str(output_file),
                "segments": [{
                    "source_start_seconds": record["source_start_seconds"],
                    "source_end_seconds": record["source_end_seconds"],
                }],
            },
        })
    return record


def _interrupted_draft_context(tmp_path: Path, candidate_ids: list[str]):
    services, project, _source = _services(tmp_path)
    analysis_path, analysis = _write_verified_analysis(tmp_path, project)
    assert isinstance(analysis, AnalysisArtifact)
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = analysis.analysis_id
    project.analysis_fingerprint = analysis.analysis_fingerprint
    project.review_selected_candidate_ids = list(candidate_ids)
    project.candidate_states = {candidate_id: "draft_planning" for candidate_id in candidate_ids}
    project.status = ProjectStatus.PROCESSING
    run = services.runs.create(
        project,
        {"analysis_id": project.analysis_id, "candidate_ids": list(candidate_ids)},
        {"path": project.source_path},
        "test",
        run_kind=RunKind.DRAFT,
    )
    output = tmp_path / "engine-output"
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path,
        state_path=tmp_path / "state.json", report_path=output / "report.json",
        output_directory=output, runtime_config_path=tmp_path / "runtime.yaml",
        run_id=run.run_id, project_id=project.project_id,
        runtime_flags={"mode": "draft", "analysis_id": project.analysis_id},
    )
    services.record_launch_context(run, prepared)
    run.status = RunStatus.RUNNING
    services.runs.save(run)
    services.projects.save(project)
    return services, project, run, prepared


def test_interrupted_draft_restores_only_bound_preview_and_resumes_missing(tmp_path: Path, monkeypatch) -> None:
    candidate_ids = ["candidate-a", "candidate-b", "candidate-c"]
    services, project, run, prepared = _interrupted_draft_context(tmp_path, candidate_ids)
    preview = prepared.output_directory / "drafts" / "01-candidate-a" / "draft-preview.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"verified preview")
    new_draft_artifact(
        draft_id=f"draft-progress-{run.run_id}",
        analysis_id=project.analysis_id, analysis_fingerprint=project.analysis_fingerprint,
        analysis_artifact_path=project.analysis_artifact_path, project_id=project.project_id,
        source_fingerprint="source-fingerprint", status="draft_partial", run_id=run.run_id,
        candidates=[
            _progress_candidate(candidate_ids[0], 1, state="draft_ready", output_file=preview),
            _progress_candidate(candidate_ids[1], 2),
            _progress_candidate(candidate_ids[2], 3),
        ],
    ).write(prepared.output_directory / "draft-progress.json")
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    assert services.recover_interrupted_runs() == 1

    restored = services.projects.load(project.project_id)
    restored_run = services.runs.load(project.project_id, run.run_id)
    assert restored.status == ProjectStatus.INTERRUPTED
    assert restored_run.status == RunStatus.INTERRUPTED
    assert restored.review_selected_candidate_ids == ["candidate-a"]
    assert restored.candidate_states == {
        "candidate-a": "draft_ready", "candidate-b": "draft_failed", "candidate-c": "draft_failed",
    }
    assert restored.candidate_draft_statuses == {
        "candidate-a": "ready", "candidate-b": "failed", "candidate-c": "failed",
    }
    assert restored.candidate_draft_artifacts == {"candidate-a": str((prepared.output_directory / "draft-progress.json").resolve())}

    restored = services.set_review_selection(restored, candidate_ids)
    resume_run, resume = services.prepare_draft(restored, ["candidate-b", "candidate-c"])

    assert resume_run.run_kind == RunKind.DRAFT
    assert "analyze" not in resume.arguments
    assert [resume.arguments[index + 1] for index, value in enumerate(resume.arguments) if value == "--candidate-id"] == [
        "candidate-b", "candidate-c",
    ]
    assert restored.candidate_states["candidate-a"] == "draft_ready"


def test_live_draft_failure_preserves_verified_sibling_progress(tmp_path: Path, monkeypatch) -> None:
    candidate_ids = ["candidate-a", "candidate-b"]
    services, project, run, prepared = _interrupted_draft_context(tmp_path, candidate_ids)
    preview = prepared.output_directory / "drafts" / "01-candidate-a" / "draft-preview.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"verified preview")
    new_draft_artifact(
        draft_id=f"draft-progress-{run.run_id}",
        analysis_id=project.analysis_id, analysis_fingerprint=project.analysis_fingerprint,
        analysis_artifact_path=project.analysis_artifact_path, project_id=project.project_id,
        source_fingerprint="source-fingerprint", status="draft_partial", run_id=run.run_id,
        candidates=[
            _progress_candidate(candidate_ids[0], 1, state="draft_ready", output_file=preview),
            _progress_candidate(candidate_ids[1], 2),
        ],
    ).write(prepared.output_directory / "draft-progress.json")
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    result = ProjectViewModel(services)._finalize_failure(
        project, run, prepared, "renderer crashed", "renderer crashed",
    )

    restored = services.projects.load(project.project_id)
    assert result.phase == ProcessingPhase.FAILED
    assert result.run.status == RunStatus.FAILED
    assert restored.candidate_states == {
        "candidate-a": "draft_ready", "candidate-b": "draft_failed",
    }
    assert restored.candidate_draft_artifacts == {
        "candidate-a": str((prepared.output_directory / "draft-progress.json").resolve()),
    }


def test_live_draft_cancel_preserves_verified_sibling_and_retries_only_missing(
    tmp_path: Path, monkeypatch,
) -> None:
    candidate_ids = ["candidate-a", "candidate-b"]
    services, project, run, prepared = _interrupted_draft_context(tmp_path, candidate_ids)
    preview = prepared.output_directory / "drafts" / "01-candidate-a" / "draft-preview.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"verified preview")
    new_draft_artifact(
        draft_id=f"draft-progress-{run.run_id}",
        analysis_id=project.analysis_id, analysis_fingerprint=project.analysis_fingerprint,
        analysis_artifact_path=project.analysis_artifact_path, project_id=project.project_id,
        source_fingerprint="source-fingerprint", status="draft_partial", run_id=run.run_id,
        candidates=[
            _progress_candidate(candidate_ids[0], 1, state="draft_ready", output_file=preview),
            _progress_candidate(candidate_ids[1], 2),
        ],
    ).write(prepared.output_directory / "draft-progress.json")
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    result = ProjectViewModel(services)._finalize_cancelled(project, run, prepared)

    restored = services.projects.load(project.project_id)
    assert result.phase == ProcessingPhase.CANCELLED
    assert result.run.status == RunStatus.CANCELLED
    assert restored.candidate_states == {
        "candidate-a": "draft_ready", "candidate-b": "analyzed",
    }
    assert restored.candidate_draft_statuses == {
        "candidate-a": "ready", "candidate-b": "pending",
    }
    retry_run, retry = services.prepare_draft(restored, ["candidate-b"])
    assert retry_run.run_kind == RunKind.DRAFT
    assert [retry.arguments[index + 1] for index, value in enumerate(retry.arguments) if value == "--candidate-id"] == [
        "candidate-b",
    ]


def test_obsolete_interrupted_draft_does_not_override_a_later_ready_run(tmp_path: Path) -> None:
    services, project, old_run, _prepared = _interrupted_draft_context(tmp_path, ["candidate-a"])
    old_run.status = RunStatus.INTERRUPTED
    services.runs.save(old_run)
    latest = services.runs.create(project, {}, {}, "test", run_kind=RunKind.DRAFT)
    latest.status = RunStatus.DRAFT_READY
    services.runs.save(latest)
    project.latest_run_id = latest.run_id
    project.status = ProjectStatus.REVIEWING_CANDIDATES
    project.candidate_states["candidate-a"] = "draft_ready"
    services.projects.save(project)

    assert services.recover_interrupted_runs() == 0

    restored = services.projects.load(project.project_id)
    assert restored.status == ProjectStatus.REVIEWING_CANDIDATES
    assert restored.candidate_states["candidate-a"] == "draft_ready"


@pytest.mark.parametrize("untrusted_output", ["wrong_path", "empty_file"])
def test_interrupted_draft_rejects_random_or_partial_mp4(
    tmp_path: Path,
    untrusted_output: str,
) -> None:
    services, project, run, prepared = _interrupted_draft_context(tmp_path, ["candidate-a"])
    expected = prepared.output_directory / "drafts" / "01-candidate-a" / "draft-preview.mp4"
    output = prepared.output_directory / "random.mp4" if untrusted_output == "wrong_path" else expected
    output.parent.mkdir(parents=True)
    output.write_bytes(b"not a trusted preview" if untrusted_output == "wrong_path" else b"")
    new_draft_artifact(
        draft_id=f"draft-progress-{run.run_id}",
        analysis_id=project.analysis_id, analysis_fingerprint=project.analysis_fingerprint,
        analysis_artifact_path=project.analysis_artifact_path, project_id=project.project_id,
        source_fingerprint="source-fingerprint", status="draft_partial", run_id=run.run_id,
        candidates=[_progress_candidate("candidate-a", 1, state="draft_ready", output_file=output)],
    ).write(prepared.output_directory / "draft-progress.json")

    assert services.recover_interrupted_runs() == 1

    restored = services.projects.load(project.project_id)
    assert restored.status == ProjectStatus.INTERRUPTED
    assert restored.candidate_states["candidate-a"] == "draft_failed"
    assert restored.candidate_draft_statuses["candidate-a"] == "failed"
    assert restored.candidate_draft_artifacts == {}
    assert "Неполный файл" in restored.candidate_errors["candidate-a"]


def test_interrupted_selected_render_keeps_each_approved_draft_retryable(tmp_path: Path) -> None:
    services, project, _source = _services(tmp_path)
    candidate_ids = ["candidate-a", "candidate-b"]
    project.review_selected_candidate_ids = list(candidate_ids)
    project.selected_candidate_ids = list(candidate_ids)
    project.candidate_states = {candidate_id: "production_rendering" for candidate_id in candidate_ids}
    project.candidate_draft_statuses = {candidate_id: "ready" for candidate_id in candidate_ids}
    project.candidate_approval_states = {candidate_id: "approved" for candidate_id in candidate_ids}
    project.candidate_export_statuses = {candidate_id: "running" for candidate_id in candidate_ids}
    project.status = ProjectStatus.RENDERING_SELECTED
    run = services.runs.create(
        project,
        {"candidate_ids": list(candidate_ids)},
        {"path": project.source_path},
        "test",
        run_kind=RunKind.SELECTED_RENDER,
    )
    run.status = RunStatus.RUNNING
    services.runs.save(run)
    services.projects.save(project)

    assert services.recover_interrupted_runs() == 1

    restored = services.projects.load(project.project_id)
    restored_run = services.runs.load(project.project_id, run.run_id)
    assert restored_run.status == RunStatus.INTERRUPTED
    assert restored.status == ProjectStatus.REVIEWING_CANDIDATES
    assert restored.review_selected_candidate_ids == candidate_ids
    assert restored.selected_candidate_ids == candidate_ids
    assert restored.candidate_states == {candidate_id: "selected" for candidate_id in candidate_ids}
    assert restored.candidate_draft_statuses == {candidate_id: "ready" for candidate_id in candidate_ids}
    assert restored.candidate_approval_states == {candidate_id: "approved" for candidate_id in candidate_ids}
    assert restored.candidate_export_statuses == {candidate_id: "failed" for candidate_id in candidate_ids}
    assert all("прерван" in restored.candidate_errors[candidate_id] for candidate_id in candidate_ids)
