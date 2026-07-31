"""Exercise the real Windows Qt video widget against three finished MP4 files.

Run with the project virtual environment:

    .venv\\Scripts\\python.exe validation\\windows_media_smoke.py path\\to\\one.mp4 ...

The script deliberately uses the same VideoPreview QWidget as the final-results
screen.  It keeps the widget visible while testing playback, seek, volume,
mute, source changes and QVideoWidget fullscreen.  It prints JSON so a CI or
manual desktop run has an auditable result without touching production files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QUrl, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.gui.components.final_results import FinalOutput, FinalResultsWorkspace


def _wait(app: QApplication, predicate, timeout_ms: int, message: str) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(25)
    raise RuntimeError(message)


def _default_paths(root: Path) -> list[Path]:
    project = root / "output" / "long-scene-185s-ee7b94d5b59b"
    return list(project.glob("clip-*.mp4"))[:3]


def _select_output(workspace: FinalResultsWorkspace, output: FinalOutput) -> None:
    QTest.mouseClick(workspace._cards[output.result_id], Qt.MouseButton.LeftButton)


def _exercise_one(
    app: QApplication, workspace: FinalResultsWorkspace, output: FinalOutput, index: int,
) -> dict[str, object]:
    preview = workspace.preview
    path = output.path
    errors: list[str] = []
    frames: list[object] = []
    preview.preview_error.connect(errors.append)
    preview.video.videoSink().videoFrameChanged.connect(frames.append)
    try:
        _select_output(workspace, output)
        _wait(
            app,
            lambda: not preview._media_loading and preview.player.source() == QUrl.fromLocalFile(str(path)),
            12_000,
            f"source did not load: {path}",
        )
        duration = preview.player.duration()
        if duration <= 1_000:
            raise RuntimeError(f"invalid duration reported for {path}: {duration}")

        QTest.mouseClick(preview.play_button, Qt.MouseButton.LeftButton)
        _wait(app, lambda: preview.player.position() > 250, 8_000, f"playback did not advance: {path}")
        _wait(app, lambda: len(frames) > 2, 8_000, f"video frames did not arrive: {path}")
        playing_position = preview.player.position()

        QTest.mouseClick(preview.play_button, Qt.MouseButton.LeftButton)
        _wait(
            app,
            lambda: preview.player.playbackState().name == "PausedState",
            3_000,
            f"pause failed: {path}",
        )
        QTest.mouseClick(preview.play_button, Qt.MouseButton.LeftButton)
        _wait(app, lambda: preview.player.position() > playing_position, 5_000, f"resume failed: {path}")

        preview.player.pause()
        _wait(
            app,
            lambda: preview.player.playbackState().name == "PausedState",
            3_000,
            f"pause before seek failed: {path}",
        )
        target = min(max(500, duration // 2), duration - 250)
        preview.player.setPosition(target)
        _wait(app, lambda: abs(preview.player.position() - target) < 800, 5_000, f"forward seek failed: {path}")
        preview.player.setPosition(100)
        _wait(app, lambda: preview.player.position() < 1_000, 5_000, f"backward seek failed: {path}")
        QTest.mouseClick(preview.play_button, Qt.MouseButton.LeftButton)

        preview.volume_slider.setValue(37)
        if abs(preview.audio.volume() - 0.37) > 0.01:
            raise RuntimeError(f"volume mapping failed: {preview.audio.volume()}")
        QTest.mouseClick(preview.volume_button, Qt.MouseButton.LeftButton)
        if not preview.audio.isMuted():
            raise RuntimeError("mute did not reach QAudioOutput")
        QTest.mouseClick(preview.volume_button, Qt.MouseButton.LeftButton)
        if preview.audio.isMuted() or abs(preview.audio.volume() - 0.37) > 0.01:
            raise RuntimeError("unmute did not restore the prior level")

        frames_before_fullscreen = len(frames)
        QTest.mouseClick(preview.fullscreen_button, Qt.MouseButton.LeftButton)
        _wait(app, preview.video.isFullScreen, 3_000, f"fullscreen did not open: {path}")
        if preview.player.videoOutput() is not preview.video:
            raise RuntimeError("fullscreen detached the active video output")
        _wait(
            app,
            lambda: len(frames) > frames_before_fullscreen,
            3_000,
            f"fullscreen stopped video frames: {path}",
        )
        QTest.keyClick(preview.video, Qt.Key.Key_Escape)
        _wait(app, lambda: not preview.video.isFullScreen(), 3_000, f"Escape did not leave fullscreen: {path}")

        if errors:
            raise RuntimeError("media error: " + " | ".join(errors))
        return {
            "file": str(path),
            "duration_ms": duration,
            "video_frames": len(frames),
            "position_after_play_ms": playing_position,
            "status": "passed",
        }
    finally:
        try:
            preview.preview_error.disconnect(errors.append)
            preview.video.videoSink().videoFrameChanged.disconnect(frames.append)
        except (RuntimeError, TypeError):
            pass


def _rapid_switch(app: QApplication, workspace: FinalResultsWorkspace, outputs: list[FinalOutput]) -> float:
    preview = workspace.preview
    started = time.perf_counter()
    for output in outputs:
        _select_output(workspace, output)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _wait(
        app,
        lambda: not preview._media_loading and preview.player.source() == QUrl.fromLocalFile(str(outputs[-1].path)),
        12_000,
        "last rapid selection did not win the source handoff",
    )
    if elapsed_ms > 500:
        raise RuntimeError(f"card switch blocked the UI for {elapsed_ms:.0f} ms")
    return round(elapsed_ms, 1)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = args.paths or _default_paths(Path(__file__).resolve().parents[1])
    if len(paths) < 3:
        raise SystemExit("Pass three finished MP4 paths, or keep the current long-scene smoke artifacts.")
    paths = paths[:3]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("Unreadable smoke input: " + ", ".join(missing))

    app = QApplication.instance() or QApplication(sys.argv)
    outputs = [
        FinalOutput(
            result_id=f"windows-smoke-{index}", candidate_id=f"windows-smoke-{index}", path=path,
            title=f"Windows smoke {index}", duration_seconds=30.0, width=1080, height=1920,
        )
        for index, path in enumerate(paths, start=1)
    ]
    workspace = FinalResultsWorkspace()
    workspace.set_results(outputs, selected_id=outputs[0].result_id, project_directory=paths[0].parent)
    workspace.resize(1260, 760)
    workspace.show()
    try:
        rapid_switch_ms = _rapid_switch(app, workspace, outputs)
        results = []
        for index, output in enumerate(outputs, start=1):
            preview = workspace.preview
            if index > 1 and (preview.volume_slider.value() != 37 or abs(preview.audio.volume() - 0.37) > 0.01):
                raise RuntimeError("volume was not preserved across source switches")
            results.append(_exercise_one(app, workspace, output, index))
    finally:
        workspace.close()
        workspace.deleteLater()
        app.processEvents()
    print(json.dumps({
        "backend": "Qt Multimedia FFmpeg plugin on Windows",
        "rapid_switch_ms": rapid_switch_ms,
        "results": results,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
