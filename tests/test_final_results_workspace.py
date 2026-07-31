from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.gui.components.final_results import FinalOutput, FinalResultsWorkspace
from app.gui.models import DesktopProject, ProjectOptions, ProjectStatus
from app.source_models import SourceSpec


def _project(tmp_path: Path) -> DesktopProject:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    return DesktopProject(
        project_id="project-final-results",
        name="Final results",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        source_path=str(source),
        project_directory=str(tmp_path),
        status=ProjectStatus.COMPLETED_WITH_WARNINGS,
        settings=ProjectOptions(),
        source_spec=SourceSpec.local(str(source)),
        last_final_result_id="candidate-two:plan-two:two.mp4",
    )


def test_last_final_result_identity_round_trips_with_project_metadata(tmp_path: Path) -> None:
    project = _project(tmp_path)

    restored = DesktopProject.from_dict(project.to_dict())

    assert restored.last_final_result_id == "candidate-two:plan-two:two.mp4"


def test_final_results_workspace_switches_only_between_bound_outputs(tmp_path: Path, monkeypatch) -> None:
    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    first = tmp_path / "one.mp4"; first.write_bytes(b"first output")
    second = tmp_path / "two.mp4"; second.write_bytes(b"second output")
    monkeypatch.setattr(
        "app.gui.components.final_results.CandidateThumbnailLoader.request",
        lambda *_args, **_kwargs: tmp_path / "unused.jpg",
    )
    workspace = FinalResultsWorkspace()
    outputs = [
        FinalOutput("candidate-one:plan-one:one.mp4", "candidate-one", first, "Первый ролик", 12.0, 1080, 1920),
        FinalOutput("candidate-two:plan-two:two.mp4", "candidate-two", second, "Второй ролик", 18.0, 1080, 1920),
    ]
    selected: list[str] = []
    workspace.output_selected.connect(selected.append)

    try:
        workspace.set_results(
            outputs, selected_id=outputs[0].result_id, project_directory=tmp_path,
            warnings=["Реальное предупреждение из отчёта"],
        )
        assert workspace.preview.active_media_path == first
        assert workspace.warning_box.isHidden() is False
        assert "Реальное предупреждение" in workspace.warning_text.text()

        workspace._activate(outputs[1].result_id, emit=True, warnings=workspace._warnings)
        assert workspace.active_output_id == outputs[1].result_id
        assert workspace.preview.active_media_path == second
        assert selected == [outputs[1].result_id]
        assert workspace._cards[outputs[1].result_id].property("activeFinalOutput") is True
    finally:
        workspace.close()
        workspace.deleteLater()
        app.processEvents()
