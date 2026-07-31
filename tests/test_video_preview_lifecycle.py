from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.gui.components.video_preview import VideoPreview


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    return QApplication.instance() or QApplication([])


def test_rapid_file_selection_is_deferred_and_coalesced(tmp_path: Path, monkeypatch) -> None:
    app = _application()
    first = tmp_path / "first.mp4"; first.write_bytes(b"first")
    second = tmp_path / "second.mp4"; second.write_bytes(b"second")
    preview = VideoPreview()
    loads: list[int] = []
    monkeypatch.setattr(preview, "_request_poster", lambda _path: None)

    def capture_current_load(token: int) -> None:
        if token == preview._selection_token:
            loads.append(token)

    monkeypatch.setattr(preview, "_load_selected_source", capture_current_load)

    try:
        preview.show_final(first, "Первый")
        preview.show_final(second, "Второй")

        assert loads == []
        assert preview.active_media_path == second
        assert preview.player.videoOutput() is preview.video

        QTest.qWait(10)
        assert loads == [preview._selection_token]
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_volume_and_mute_restore_the_last_audible_level() -> None:
    app = _application()
    preview = VideoPreview()

    try:
        preview.volume_slider.setValue(37)
        assert preview.audio.volume() == pytest.approx(0.37)
        assert preview.audio.isMuted() is False

        preview._toggle_mute()
        assert preview.audio.isMuted() is True
        assert preview.volume_slider.value() == 37

        preview._toggle_mute()
        assert preview.audio.isMuted() is False
        assert preview.audio.volume() == pytest.approx(0.37)
        assert preview.volume_slider.value() == 37

        preview.volume_slider.setValue(0)
        assert preview.audio.volume() == pytest.approx(0.0)
        preview._toggle_mute()
        assert preview.audio.isMuted() is False
        assert preview.audio.volume() == pytest.approx(0.37)
        assert preview.volume_slider.value() == 37
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_repeated_buffering_status_does_not_rewind_a_non_range_seek(tmp_path: Path) -> None:
    app = _application()
    source = tmp_path / "finished.mp4"; source.write_bytes(b"finished")
    preview = VideoPreview()

    class FakePlayer:
        def __init__(self) -> None:
            self.positions: list[int] = []

        def setPosition(self, value: int) -> None:
            self.positions.append(value)

    preview.player = FakePlayer()  # type: ignore[assignment]
    preview._path = source

    try:
        preview._media_status_changed(QMediaPlayer.MediaStatus.LoadedMedia)
        preview._media_status_changed(QMediaPlayer.MediaStatus.BufferedMedia)
        assert preview.player.positions == [0]
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()
