from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QBoxLayout, QVBoxLayout, QWidget

from app.gui.components.final_results import FinalOutput, FinalResultsWorkspace
from app.gui.components.video_preview import VideoPreview
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


@pytest.mark.parametrize(
    ("width", "height", "profile", "body_mode", "bottom_stacked"),
    (
        (542, 480, "dense", "stacked", True),
        (1026, 720, "compact", "two_row", False),
        (1186, 900, "compact", "two_row", False),
        (1660, 1080, "standard", "columns", False),
    ),
)
def test_final_results_reflows_without_hidden_horizontal_clipping(
    tmp_path: Path,
    monkeypatch,
    width: int,
    height: int,
    profile: str,
    body_mode: str,
    bottom_stacked: bool,
) -> None:
    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    output = tmp_path / "result.mp4"
    output.write_bytes(b"result")
    monkeypatch.setattr(
        VideoPreview,
        "show_final",
        lambda preview, *_args, **_kwargs: preview._set_presentation("vertical"),
    )
    workspace = FinalResultsWorkspace()
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(workspace)

    try:
        workspace.set_results(
            [FinalOutput("result", "candidate", output, "Long final title " * 12, 28.0, 1080, 1920)],
            selected_id="result",
            project_directory=tmp_path,
            warnings=["A detailed warning that must wrap rather than widen the metadata panel. " * 8],
        )
        host.resize(width, height)
        host.show()
        app.processEvents()
        app.processEvents()

        assert workspace._responsive_profile == profile
        assert workspace._body_layout_mode == body_mode
        assert workspace.minimumSizeHint().width() <= width
        assert workspace.list_scroll.horizontalScrollBar().maximum() == 0
        assert workspace.info_scroll.horizontalScrollBar().maximum() == 0
        expected_direction = (
            QBoxLayout.Direction.TopToBottom
            if bottom_stacked else QBoxLayout.Direction.LeftToRight
        )
        assert workspace._bottom_layout.direction() == expected_direction
        for button in workspace.findChildren(type(workspace.create_more_button)):
            if button.isVisible():
                assert button.width() >= button.minimumSizeHint().width()
    finally:
        host.close()
        workspace.deleteLater()
        host.deleteLater()
        app.processEvents()
