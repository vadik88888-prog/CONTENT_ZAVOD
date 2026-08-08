from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QUrl
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


def test_same_source_stale_position_is_ignored_until_current_range_seek(tmp_path: Path, monkeypatch) -> None:
    """A and B may share a URL, so URL matching alone cannot identify a callback."""

    app = _application()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    preview = VideoPreview()

    class FakePlayer:
        def source(self) -> QUrl:
            return QUrl.fromLocalFile(str(source))

    preview.player = FakePlayer()  # type: ignore[assignment]
    preview._path = source
    preview._expected_source = QUrl.fromLocalFile(str(source))
    preview._selection_token = 2
    preview._range_start_ms = 30_000
    preview._range_end_ms = 40_000
    preview._range_media_ready = True
    preview._range_ready_token = 2
    preview._range_seek_pending_token = 2
    preview._range_seek_target_ms = 30_000
    timeline_positions: list[int] = []
    monkeypatch.setattr(preview, "_update_timeline", timeline_positions.append)

    try:
        # Queued from range A (same media path, wrong interval).
        preview._position_changed(18_000)
        assert timeline_positions == []
        assert preview._range_seek_pending_token == 2

        # Range B's own seek acknowledgement now makes callbacks trustworthy.
        preview._position_changed(30_000)
        assert timeline_positions == [30_000]
        assert preview._range_seek_pending_token is None
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_finished_source_is_reused_for_a_new_range(tmp_path: Path) -> None:
    app = _application()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    preview = VideoPreview()
    expected = QUrl.fromLocalFile(str(source))

    class FakePlayer:
        def source(self) -> QUrl:
            return expected

        @staticmethod
        def mediaStatus() -> QMediaPlayer.MediaStatus:
            return QMediaPlayer.MediaStatus.EndOfMedia

    preview.player = FakePlayer()  # type: ignore[assignment]

    try:
        assert preview._player_has_loaded_source(expected)
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()
