from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QLabel

from app.analysis_artifact import AnalysisArtifact
from app.draft_artifact import new_draft_artifact
from app.gui.components import VideoPreview
from app.gui.models import DesktopProject, DesktopSettings, ProjectStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel
from app.utils import read_json, stable_file_hash, write_json


def _services(tmp_path: Path) -> tuple[DesktopServices, DesktopProject]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(
        source,
        source_metadata={"duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0},
    )
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    root = Path(__file__).resolve().parents[1]
    services = DesktopServices(
        engine_root=root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(root),
        system=SystemService(root),
    )
    return services, project


def _write_verified_analysis(tmp_path: Path, project: DesktopProject) -> tuple[Path, AnalysisArtifact]:
    snapshot = tmp_path / "analysis-snapshot"
    snapshot.mkdir()
    objects = {
        "final_selection": {"selected_ids": ["candidate-a"]},
        "candidate_data": {"candidates": [{"id": "candidate-a"}]},
        "transcript_features": {"segments": [{"start": 1.0, "end": 24.0}]},
        "scene_boundaries": {"boundaries": [{"timestamp": 2.0}]},
        "content_profile": {"detected_content_type": "podcast"},
    }
    producer = {"name": "gui-integrity-test", "version": "1", "analysis_run_id": "analysis-run"}
    references: dict[str, str] = {}
    integrity: dict[str, dict[str, object]] = {}
    for name, value in objects.items():
        path = snapshot / f"{name}.json"
        write_json(path, value)
        references[name] = str(path)
        integrity[name] = {
            "sha256": stable_file_hash(path),
            "byte_size": path.stat().st_size,
            "producer": {**producer, "artifact_name": name},
        }
    analysis = AnalysisArtifact(
        analysis_id="analysis-gui",
        project_id=project.project_id,
        created_at="2026-08-15T00:00:00+00:00",
        source={"id": "source-fingerprint"},
        source_fingerprint="source-fingerprint",
        analysis_fingerprint="analysis-fingerprint",
        work_directory=str(snapshot),
        candidate_data_ref=references["candidate_data"],
        references=references,
        candidates=[{
            "candidate_id": "candidate-a",
            "title": "Candidate A",
            "start": 1.0,
            "end": 24.0,
            "eligibility_decision": {
                "schema_version": "6D.1",
                "config_version": "test",
                "state": "assessed",
                "eligible": True,
                "reason_codes": [],
                "recoverable_issues": [],
                "required_boundary_actions": [],
                "evidence_refs": [],
            },
        }],
        recommendation={},
        summary={},
        content_profile=objects["content_profile"],
        duration_seconds=30.0,
        analysis_run_id="analysis-run",
        snapshot_directory=str(snapshot),
        reference_integrity=integrity,
        producer=producer,
        candidate_count=1,
    )
    path = tmp_path / "analysis.json"
    analysis.write_with_integrity(path)
    project.analysis_artifact_path = str(path)
    project.analysis_id = analysis.analysis_id
    project.analysis_fingerprint = analysis.analysis_fingerprint
    project.status = ProjectStatus.ANALYSIS_READY
    project.candidate_states = {"candidate-a": "analyzed"}
    return path, analysis


def test_candidate_review_rechecks_reference_integrity_before_using_cached_analysis(tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    _path, analysis = _write_verified_analysis(tmp_path, project)
    fake_screen = SimpleNamespace(
        viewmodel=SimpleNamespace(services=services),
        _analysis_cache_key=None,
        _analysis_cache={},
        _analysis_load_error=None,
    )

    first = ProjectScreen._analysis_artifact(fake_screen, project)
    assert [item["candidate_id"] for item in first["candidates"]] == ["candidate-a"]

    write_json(Path(analysis.references["content_profile"]), {"detected_content_type": "gameplay"})
    second = ProjectScreen._analysis_artifact(fake_screen, project)

    assert second == {}
    assert "ANALYSIS_INTEGRITY_MISMATCH" in fake_screen._analysis_load_error


def test_candidate_review_window_shows_blocking_integrity_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    application = QApplication.instance() or QApplication([])
    services, project = _services(tmp_path)
    _analysis_path, analysis = _write_verified_analysis(tmp_path, project)
    write_json(Path(analysis.references["content_profile"]), {"detected_content_type": "gameplay"})
    services.projects.save(project)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(ProjectViewModel(services))

    try:
        screen.open(project)
        screen.show()
        application.processEvents()

        assert screen.isVisible()
        assert screen._review_candidates_by_id == {}
        assert "ANALYSIS_INTEGRITY_MISMATCH" in screen.workflow_hint.text()
        integrity_notice = screen.findChild(QLabel, "analysisIntegrityError")
        assert integrity_notice is not None
        assert "повреждён или изменён" in integrity_notice.text()
        assert screen.draft_button.isHidden()
        assert screen.production_button.isHidden()
        screenshot = os.environ.get("GUI_INTEGRITY_SCREENSHOT")
        if screenshot:
            assert screen.grab().save(screenshot)
    finally:
        screen.close()
        screen.deleteLater()
        application.processEvents()


def test_draft_preparation_blocks_changed_analysis_before_run_creation(tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    analysis_path, _analysis = _write_verified_analysis(tmp_path, project)
    project.review_selected_candidate_ids = ["candidate-a"]
    services.projects.save(project)
    raw = read_json(analysis_path, {})
    raw["summary"]["tampered"] = True
    write_json(analysis_path, raw)

    with pytest.raises(InputValidationError, match="ANALYSIS_INTEGRITY_MISMATCH"):
        services.prepare_draft(project, ["candidate-a"])

    assert services.runs.list(project.project_id) == []
    assert project.status == ProjectStatus.ANALYSIS_READY
    assert project.candidate_states == {"candidate-a": "analyzed"}


def test_boundary_edit_loads_features_and_scenes_through_verified_references(tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    _analysis_path, analysis = _write_verified_analysis(tmp_path, project)
    write_json(Path(analysis.references["transcript_features"]), {"segments": []})

    with pytest.raises(InputValidationError, match="ANALYSIS_INTEGRITY_MISMATCH"):
        services.adjust_candidate_boundary(project, "candidate-a", "start", 0.5)

    assert project.candidate_boundary_overrides == {}


def test_final_facade_blocks_changed_reference_before_preparation(tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    analysis_path, analysis = _write_verified_analysis(tmp_path, project)
    draft_path = tmp_path / "draft.json"
    new_draft_artifact(
        draft_id="draft-gui",
        analysis_id=analysis.analysis_id,
        analysis_fingerprint=analysis.analysis_fingerprint,
        analysis_artifact_path=str(analysis_path),
        project_id=project.project_id,
        source_fingerprint=analysis.source_fingerprint,
        candidates=[{"candidate_id": "candidate-a", "state": "draft_ready"}],
        analysis_run_id=analysis.analysis_run_id,
        analysis_artifact_sha256=stable_file_hash(analysis_path),
    ).write(draft_path)
    project.candidate_draft_artifacts = {"candidate-a": str(draft_path)}
    write_json(Path(analysis.references["scene_boundaries"]), {"boundaries": []})

    with pytest.raises(InputValidationError, match="ANALYSIS_INTEGRITY_MISMATCH"):
        services.pipeline.prepare_selected_render(
            project,
            SimpleNamespace(run_id="final-run"),
            services.settings,
            ["candidate-a"],
        )

    assert not (project.directory / "runs" / "final-run").exists()


def test_gui_verified_loader_keeps_legacy_v10_readable_with_warnings(tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    candidate_data = tmp_path / "candidate-data.json"
    write_json(candidate_data, {"candidates": []})
    analysis_path = tmp_path / "legacy-analysis.json"
    AnalysisArtifact(
        analysis_id="analysis-legacy",
        project_id=project.project_id,
        created_at="2026-08-15T00:00:00+00:00",
        source={"id": "source-legacy"},
        source_fingerprint="source-legacy",
        analysis_fingerprint="legacy-fingerprint",
        work_directory=str(tmp_path),
        candidate_data_ref=str(candidate_data),
        references={"candidate_data": str(candidate_data)},
        candidates=[],
        recommendation={},
        summary={},
        content_profile={"detected_content_type": "podcast"},
        duration_seconds=30.0,
        schema_version="1.0",
    ).write(analysis_path)
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "analysis-legacy"
    project.analysis_fingerprint = "legacy-fingerprint"

    artifact = services.pipeline.load_verified_analysis(project, required=True)
    assert artifact is not None
    assert any("LEGACY_ANALYSIS_ARTIFACT_1_0" in warning for warning in artifact.warnings)
    assert any("LEGACY_ANALYSIS_CHECKSUM_ONLY" in warning for warning in artifact.warnings)
    services.pipeline.plan_processing(project, services.settings)
