from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QBoxLayout, QVBoxLayout, QWidget

from app.gui.components.final_results import FinalOutput, FinalResultsWorkspace
from app.gui.components.video_preview import VideoPreview
from app.gui.models import DesktopProject, ProjectOptions, ProjectStatus
from app.gui.styles import load_theme
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
        (1026, 720, "compact", "columns", False),
        (1186, 900, "compact", "columns", False),
        (1338, 900, "standard", "columns", False),
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
    app.setStyleSheet(load_theme())
    output = tmp_path / "result.mp4"
    output.write_bytes(b"result")

    def show_final_with_hostile_copy(
        preview: VideoPreview, _path: Path, title: str, **_kwargs: object,
    ) -> None:
        preview._set_presentation("vertical")
        preview.active_candidate.setText(f"Готовый ролик · {title}")
        preview.active_candidate.show()
        preview._show_status(
            "Preview unavailable: " + "https://example.test/very-long-error-token/" * 12
        )

    monkeypatch.setattr(
        VideoPreview,
        "show_final",
        show_final_with_hostile_copy,
    )
    workspace = FinalResultsWorkspace()
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(workspace)

    try:
        workspace.set_results(
            [FinalOutput(
                "result",
                "candidate",
                output,
                "ОченьДлинныйФинальныйЗаголовок_" * 12,
                28.0,
                1080,
                1920,
            )],
            selected_id="result",
            project_directory=tmp_path,
            warnings=["https://example.test/very-long-warning-token/" * 16],
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
        assert workspace._info_layout.indexOf(
            workspace.open_video_button,
        ) < workspace._info_layout.indexOf(workspace.info_scroll)
        assert workspace.preview.video.width() >= {
            "dense": 220,
            "compact": 252,
            "standard": 288,
        }[profile]
        assert workspace.preview.active_candidate.geometry().bottom() < workspace.preview.media_stage.geometry().top()
        assert workspace.preview.media_stage.geometry().bottom() < workspace.preview.preview_status.geometry().top()
        assert workspace.preview.preview_status.geometry().bottom() < workspace.preview.controls_host.geometry().top()

        preview_position = workspace.preview.mapTo(workspace._body_host, QPoint(0, 0))
        assert preview_position.y() + workspace.preview.height() <= workspace._body_host.height()
        if body_mode == "stacked":
            assert workspace._list_panel.geometry().bottom() < workspace.preview.geometry().top()
            assert workspace.preview.geometry().bottom() < workspace._info_panel.geometry().top()
        elif body_mode == "two_row":
            assert workspace._list_panel.geometry().bottom() < workspace._info_panel.geometry().top()
            assert workspace.preview.geometry().bottom() < workspace._info_panel.geometry().top()
        else:
            assert workspace._list_panel.geometry().right() < workspace.preview.geometry().left()
            assert workspace.preview.geometry().right() < workspace._info_panel.geometry().left()
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


def test_vertical_preview_height_for_width_shrinks_and_controls_stay_inside() -> None:
    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(load_theme())
    preview = VideoPreview()
    preview.set_vertical_frame_size(172, 306)
    preview._set_presentation("vertical")
    preview.open_button.hide()
    preview.active_candidate.setText("Very long persisted final title " * 20)
    preview.active_candidate.show()
    preview._show_status(
        "Preview unavailable: " + "https://example.test/very-long-error-token/" * 12
    )

    try:
        heights: list[int] = []
        for width in (360, 500, 360):
            preview.resize(width, max(1, preview.minimumHeight()))
            preview.show()
            app.processEvents()
            app.processEvents()
            preview._refresh_layout_geometry()
            app.processEvents()
            heights.append(preview.minimumHeight())

            assert preview.minimumHeight() == preview.layout().totalHeightForWidth(
                preview.contentsRect().width()
            )
            assert preview.active_candidate.geometry().bottom() < preview.media_stage.geometry().top()
            assert preview.media_stage.geometry().bottom() < preview.preview_status.geometry().top()
            assert preview.preview_status.geometry().bottom() < preview.controls_host.geometry().top()
            for control in (
                preview.play_button,
                preview.time_label,
                preview.seek_slider,
                preview.volume_button,
                preview.volume_slider,
                preview.fullscreen_button,
            ):
                position = control.mapTo(preview.controls_host, QPoint(0, 0))
                assert position.x() >= 0
                assert position.x() + control.width() <= preview.controls_host.width()
                assert control.width() >= control.minimumSizeHint().width()

        assert heights[1] < heights[0]
        assert heights[2] == heights[0]
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_source_preview_controls_reflow_after_compact_resize() -> None:
    existing = QApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(load_theme())
    preview = VideoPreview()
    preview._set_presentation("source")

    try:
        for width, compact in (
            (800, False), (770, True), (760, True), (360, True), (800, False),
        ):
            preview.resize(width, 620)
            preview.show()
            app.processEvents()
            app.processEvents()
            assert preview._compact_controls is compact
            assert preview._controls_layout.totalMinimumSize().width() <= preview.controls_host.width()
            for control in (
                preview.play_button,
                preview.time_label,
                preview.seek_slider,
                preview.volume_button,
                preview.volume_slider,
                preview.fullscreen_button,
                preview.open_button,
            ):
                position = control.mapTo(preview.controls_host, QPoint(0, 0))
                assert position.x() >= 0
                assert position.x() + control.width() <= preview.controls_host.width()
                assert control.width() >= control.minimumSizeHint().width()
        assert preview.open_button.text() == "Открыть в проигрывателе"
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()
