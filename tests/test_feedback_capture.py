from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from app.analysis_artifact import new_analysis_artifact
from app.clip_results import ClipResult
from app.feedback_contracts import FeedbackDomain
from app.feedback_store import (
    CREATIVE_FEEDBACK_FILE_NAME,
    EDITORIAL_FEEDBACK_FILE_NAME,
    OUTCOME_FEEDBACK_FILE_NAME,
    FeedbackStore,
)
from app.gui.models import ProjectStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_services import DesktopServices
from app.gui.viewmodels.project_viewmodel import ProjectViewModel
from app.runtime import RuntimeLayout


def _services(tmp_path: Path) -> tuple[DesktopServices, Path]:
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "desktop-data"
    services = DesktopServices.create(RuntimeLayout.for_source(root, data=data))
    source = tmp_path / "private-source.mp4"
    source.write_bytes(b"private video must never be exported")
    project = services.projects.create(
        source,
        source_metadata={"duration": 90.0, "width": 1920, "height": 1080, "fps": 30.0},
    )
    project.analysis_id = "analysis-1"
    project.draft_id = "draft-1"
    project.status = ProjectStatus.REVIEWING_CANDIDATES
    project.candidate_states = {"candidate-1": "draft_ready"}
    project.candidate_draft_statuses = {"candidate-1": "ready"}
    project.candidate_approval_states = {"candidate-1": "pending"}
    project.candidate_export_statuses = {"candidate-1": "pending"}
    services.projects.save(project)
    return services, project.project_id


def test_moment_preview_interaction_writes_editorial_feedback(tmp_path: Path) -> None:
    services, project_id = _services(tmp_path)
    project = services.projects.load(project_id)
    candidate_id = "candidate-ui-feedback"
    artifact_path = tmp_path / "analysis.json"
    new_analysis_artifact(
        analysis_id="analysis-ui-feedback",
        project_id=project.project_id,
        source={"id": "source-ui-feedback"},
        source_fingerprint="source-ui-feedback",
        analysis_fingerprint="analysis-ui-feedback-fingerprint",
        work_directory=str(tmp_path),
        candidate_data_ref=str(tmp_path / "candidate-data.json"),
        references={},
        candidates=[{
            "candidate_id": candidate_id,
            "title": "Moment для проверки UI",
            "start_seconds": 4.0,
            "end_seconds": 22.0,
            "potential": "high",
            "confidence": 0.9,
            "recommended": True,
            "eligibility_decision": {
                "schema_version": "6D.1", "config_version": "test", "state": "assessed",
                "eligible": True, "reason_codes": [], "recoverable_issues": [],
                "required_boundary_actions": [], "evidence_refs": [],
            },
            "reasons": ["Проверка UI feedback."],
        }],
        recommendation={}, summary={}, content_profile={}, duration_seconds=90.0, candidate_count=1,
    ).write(artifact_path)
    project.analysis_artifact_path = str(artifact_path)
    project.analysis_id = "analysis-ui-feedback"
    project.analysis_fingerprint = "analysis-ui-feedback-fingerprint"
    project.status = ProjectStatus.ANALYSIS_READY
    project.candidate_states = {candidate_id: "analyzed"}
    services.projects.save(project)

    application = QApplication.instance() or QApplication([])
    viewmodel = ProjectViewModel(services)
    screen = ProjectScreen(viewmodel)
    screen.open(project)
    application.processEvents()
    preview = screen.findChild(QPushButton, f"preview-candidate-{candidate_id}")
    assert preview is not None
    preview.click()
    application.processEvents()

    events = FeedbackStore(project.directory, file_name=EDITORIAL_FEEDBACK_FILE_NAME).read_events()
    assert [(event.name, dict(event.payload)) for event in events] == [
        ("moment_shown", {"rank": 0, "recommended": True}),
    ]
    screen.deleteLater()


def test_feedback_capture_survives_restart_and_exports_only_safe_structured_data(tmp_path: Path) -> None:
    services, project_id = _services(tmp_path)
    project = services.projects.load(project_id)

    services.record_moment_shown(project, "candidate-1", rank=0, recommended=True)
    project = services.set_review_selection(project, ["candidate-1"])
    project = services.set_review_selection(project, [])
    services.record_boundary_changed(
        project, "candidate-1", boundary="start", old_start=12.0, old_end=32.0,
        new_start=11.5, new_end=32.0, delta=-0.5,
    )
    services.record_draft_shown(project, "candidate-1", rank=0)
    services.record_draft_approval(project, "candidate-1", approved=True)
    services.record_draft_approval(project, "candidate-1", approved=False, reason="captions")
    services.record_creative_change(
        project, field_name="caption_preset_id", old_value="editorial_narrow",
        new_value="clean_white", candidate_id="candidate-1",
    )
    project = services.update_project_options(project, subtitle_style="clean")
    project = services.update_project_options(project, reduced_motion=True)
    services.record_final_created(
        project,
        ClipResult(
            candidate_id="candidate-1",
            clip_result_id="result-1",
            output_file=str(tmp_path / "private-final.mp4"),
            run_id="run-1",
        ),
        "run-1",
    )

    assert {
        path.name for path in (project.directory / "feedback").iterdir()
    } == {EDITORIAL_FEEDBACK_FILE_NAME, CREATIVE_FEEDBACK_FILE_NAME, OUTCOME_FEEDBACK_FILE_NAME}

    restarted = DesktopServices.create(RuntimeLayout.for_source(Path(__file__).resolve().parents[1], data=tmp_path / "desktop-data"))
    restored = restarted.projects.load(project_id)
    editorial = FeedbackStore(restored.directory, file_name=EDITORIAL_FEEDBACK_FILE_NAME).read_events()
    creative = FeedbackStore(restored.directory, file_name=CREATIVE_FEEDBACK_FILE_NAME).read_events()
    outcome = FeedbackStore(restored.directory, file_name=OUTCOME_FEEDBACK_FILE_NAME).read_events()
    assert {event.name for event in editorial} == {
        "moment_shown", "moment_selected", "moment_rejected", "boundary_changed",
    }
    assert next(event for event in editorial if event.name == "moment_rejected").payload["reason"] == "other"
    assert {event.name for event in creative} == {
        "draft_shown", "draft_approved", "draft_rejected", "creative_override_changed",
    }
    assert any(
        event.payload.get("field") == "reduced_motion"
        for event in creative
        if event.name == "creative_override_changed"
    )
    assert {event.name for event in outcome} == {"final_created", "final_exported"}
    assert {event.domain for event in editorial} == {FeedbackDomain.EDITORIAL}
    assert {event.domain for event in creative} == {FeedbackDomain.CREATIVE}

    result = restarted.export_feedback_data()
    assert result.path.is_file()
    with zipfile.ZipFile(result.path) as archive:
        assert set(archive.namelist()) == {
            "summary.json",
            f"events/{EDITORIAL_FEEDBACK_FILE_NAME}",
            f"events/{CREATIVE_FEEDBACK_FILE_NAME}",
            f"events/{OUTCOME_FEEDBACK_FILE_NAME}",
        }
        summary = json.loads(archive.read("summary.json"))
        payload = "\n".join(
            archive.read(name).decode("utf-8") for name in archive.namelist()
        ).casefold()
    assert summary["event_count"] == len(editorial) + len(creative) + len(outcome)
    assert summary["privacy"] == {
        "includes_api_keys": False,
        "includes_media": False,
        "includes_full_transcripts": False,
        "includes_project_paths": False,
    }
    assert "private-source" not in payload
    assert "private-final" not in payload
    assert "sk-secret" not in payload
    assert "transcript_text" not in payload
