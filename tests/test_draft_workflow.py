from __future__ import annotations

from pathlib import Path

from app.draft_artifact import new_draft_artifact
from app.gui.models import DesktopSettings, ProjectStatus, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
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


def test_desktop_flow_prepares_analysis_then_draft_then_confirmed_production(tmp_path: Path) -> None:
    services, project, source = _services(tmp_path)

    analysis_run, analysis_prepared = services.prepare_analysis(project)
    assert analysis_run.run_kind == RunKind.ANALYSIS
    assert "analyze" in analysis_prepared.arguments
    assert "--transform-script" not in analysis_prepared.arguments
    assert project.status == ProjectStatus.ANALYZING

    analysis_path = tmp_path / "analysis.json"
    write_json(analysis_path, {"placeholder": True})
    write_json(analysis_prepared.report_path, {
        "terminal": {"status": "analysis_ready"}, "output_files": [], "warnings": [],
        "run": {"analysis_id": "analysis-001", "analysis_fingerprint": "fingerprint-001", "analysis_artifact_path": str(analysis_path)},
        "clip_intelligence": {"candidates": [{"id": "candidate-a"}, {"id": "candidate-b"}]},
    })
    finished_analysis = services.finish_success(project, analysis_run, analysis_prepared)
    assert finished_analysis.status == RunStatus.ANALYSIS_READY
    assert project.status == ProjectStatus.ANALYSIS_READY
    assert project.candidate_states == {"candidate-a": "analyzed", "candidate-b": "analyzed"}

    draft_run, draft_prepared = services.prepare_draft(project, ["candidate-a"])
    assert draft_run.run_kind == RunKind.DRAFT
    assert "draft" in draft_prepared.arguments
    assert "render" not in draft_prepared.arguments
    assert project.candidate_states["candidate-a"] == "draft_planning"

    preview = tmp_path / "draft-preview.mp4"; preview.write_bytes(b"preview")
    draft_path = tmp_path / "draft-a.json"
    new_draft_artifact(
        draft_id="draft-001", analysis_id="analysis-001", analysis_fingerprint="fingerprint-001",
        analysis_artifact_path=str(analysis_path), project_id=project.project_id,
        source_fingerprint="source-fingerprint", candidates=[{
            "candidate_id": "candidate-a", "state": "draft_ready",
            "draft_production_plan": {"plan_id": "plan-a"},
            "preview": {"output_file": str(preview)},
        }],
    ).write(draft_path)
    write_json(draft_prepared.report_path, {
        "terminal": {"status": "draft_ready"}, "output_files": [str(preview)], "warnings": [],
        "run": {"draft_id": "draft-001", "draft_artifact_path": str(draft_path)},
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
    production_run, production_prepared = services.prepare_selected_render(project)
    assert production_run.run_kind == RunKind.SELECTED_RENDER
    assert "render" in production_prepared.arguments
    assert "--draft" in production_prepared.arguments
    assert "--confirm-production" in production_prepared.arguments
    assert project.status == ProjectStatus.RENDERING_SELECTED
    approved_path = Path(production_prepared.arguments[production_prepared.arguments.index("--draft") + 1])
    assert [item["candidate_id"] for item in read_json(approved_path, {})["candidates"]] == ["candidate-b", "candidate-a"]
