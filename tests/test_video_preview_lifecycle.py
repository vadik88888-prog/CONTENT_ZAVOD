from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QProcess, QUrl
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.gui.components import video_preview as video_preview_module
from app.gui.components.video_preview import VideoPreview, _ProxyRequest


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


def test_av1_source_never_hands_the_full_file_to_qt_multimedia(tmp_path: Path, monkeypatch) -> None:
    app = _application()
    source = tmp_path / "source-av1.webm"
    source.write_bytes(b"av1")
    preview = VideoPreview()
    poster_requests: list[Path] = []
    direct_attempts: list[bool] = []

    class FakePlayer:
        def __init__(self) -> None:
            self.sources: list[QUrl] = []

        @staticmethod
        def playbackState() -> QMediaPlayer.PlaybackState:
            return QMediaPlayer.PlaybackState.StoppedState

        def setSource(self, value: QUrl) -> None:
            self.sources.append(value)

    preview.player = FakePlayer()  # type: ignore[assignment]
    monkeypatch.setattr(preview, "_request_poster", poster_requests.append)
    monkeypatch.setattr(preview, "_queue_source_load", lambda: direct_attempts.append(True))

    try:
        preview.show_source(source, source_codec="AV01")

        assert poster_requests == [source]
        assert direct_attempts == []
        assert preview.player.sources and not preview.player.sources[-1].isValid()
        assert preview.open_button.isEnabled()
        assert not preview.play_button.isEnabled()
        assert not preview.preview_status.isHidden()
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_av1_candidate_range_starts_compatible_proxy_without_direct_decoder(tmp_path: Path, monkeypatch) -> None:
    app = _application()
    source = tmp_path / "source-av1.webm"
    source.write_bytes(b"av1")
    preview = VideoPreview()
    proxy_requests: list[tuple[Path, str]] = []
    direct_attempts: list[bool] = []
    monkeypatch.setattr(preview, "_request_proxy", lambda cache, reason: proxy_requests.append((cache, reason)))
    monkeypatch.setattr(preview, "_activate_direct_source", lambda: direct_attempts.append(True))

    try:
        preview.set_range(source, 12.0, 20.0, source_codec="av1")

        assert direct_attempts == []
        assert len(proxy_requests) == 1
        assert proxy_requests[0][0] == preview._proxy_cache_directory
        assert preview.source_range_seconds == (12.0, 20.0)
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_proxy_limits_both_av1_decode_and_h264_encode_threads(tmp_path: Path, monkeypatch) -> None:
    app = _application()
    source = tmp_path / "source-av1.webm"
    source.write_bytes(b"av1")
    destination = tmp_path / "cache" / "preview.mp4"
    request = _ProxyRequest(
        token=1,
        source_path=source,
        start_seconds=1.0,
        end_seconds=4.0,
        destination=destination,
        temporary=destination.with_name("preview.part.mp4"),
    )
    preview = VideoPreview()

    class FakeProcess:
        def __init__(self) -> None:
            self.program = ""
            self.arguments: list[str] = []
            self.started = False

        def setProgram(self, value: str) -> None:
            self.program = value

        def setArguments(self, value: list[str]) -> None:
            self.arguments = list(value)

        def start(self) -> None:
            self.started = True

        @staticmethod
        def state() -> QProcess.ProcessState:
            return QProcess.ProcessState.NotRunning

    process = FakeProcess()
    preview._proxy_process = process  # type: ignore[assignment]
    monkeypatch.setattr(video_preview_module.shutil, "which", lambda _name: "ffmpeg")

    try:
        preview._start_proxy(request)

        assert process.started and process.program == "ffmpeg"
        assert process.arguments.count("-threads") == 2
        assert "-ss" not in process.arguments
        assert "-t" not in process.arguments
        input_threads = process.arguments.index("-threads")
        assert process.arguments[input_threads + 1] == "2"
        encoder = process.arguments.index("-c:v")
        assert process.arguments[encoder + 2:encoder + 4] == ["-threads", "2"]
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_av1_rapid_range_switch_reuses_one_inflight_source_proxy(tmp_path: Path, monkeypatch) -> None:
    app = _application()
    source = tmp_path / "source-av1.webm"
    source.write_bytes(b"av1")
    preview = VideoPreview()
    started: list[_ProxyRequest] = []
    monkeypatch.setattr(preview, "_request_poster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preview, "_start_proxy", started.append)

    try:
        preview.set_range(source, 12.0, 20.0, source_codec="av1")
        assert len(started) == 1
        preview._active_proxy = started[0]

        preview.set_range(source, 42.0, 54.0, source_codec="av1")

        assert len(started) == 1
        assert preview._active_proxy is not None
        assert preview._active_proxy.destination == started[0].destination
        assert (preview._active_proxy.start_seconds, preview._active_proxy.end_seconds) == (42.0, 54.0)
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_direct_range_load_timeout_falls_back_to_proxy_instead_of_spinning(tmp_path: Path, monkeypatch) -> None:
    app = _application()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    preview = VideoPreview()
    proxy_requests: list[str] = []
    monkeypatch.setattr(preview, "_request_proxy", lambda _cache, reason: proxy_requests.append(reason))
    preview._source_path = source
    preview._source_range_seconds = (5.0, 12.0)
    preview._path = source
    preview._selection_token = 7
    preview._media_load_token = 7
    preview._media_loading = True

    try:
        preview._media_load_timed_out()

        assert len(proxy_requests) == 1
        assert preview._media_loading is False
        assert preview._media_load_token is None
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()


def test_partial_proxy_is_deleted_and_error_releases_stale_player_source(tmp_path: Path) -> None:
    app = _application()
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    destination = tmp_path / "cache" / "preview.mp4"
    temporary = tmp_path / "cache" / "preview.part.mp4"
    temporary.parent.mkdir()
    temporary.write_bytes(b"interrupted proxy")
    preview = VideoPreview()

    class FakePlayer:
        def __init__(self) -> None:
            self.sources: list[QUrl] = []

        @staticmethod
        def playbackState() -> QMediaPlayer.PlaybackState:
            return QMediaPlayer.PlaybackState.StoppedState

        def setSource(self, value: QUrl) -> None:
            self.sources.append(value)

    preview.player = FakePlayer()  # type: ignore[assignment]
    request = _ProxyRequest(
        token=preview._selection_token,
        source_path=source,
        start_seconds=1.0,
        end_seconds=4.0,
        destination=destination,
        temporary=temporary,
    )
    preview._active_proxy = request
    preview._path = source
    preview._source_path = source
    preview._expected_source = QUrl.fromLocalFile(str(source))

    try:
        preview._complete_proxy(request, False, "FFmpeg stopped")

        assert not temporary.exists()
        assert not destination.exists()
        assert preview.active_media_path is None
        assert preview.player.sources and not preview.player.sources[-1].isValid()
        assert not preview.preview_status.isHidden()
    finally:
        preview.close()
        preview.deleteLater()
        app.processEvents()
