from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QProcess

import app.source_download as download_module
from app.gui.services import url_source_service
from app.errors import SourceError
from app.source_download import (
    DownloadCancelled,
    YtDlpSource,
    cleanup_partial_downloads,
    describe_public_url_failure,
    find_ytdlp_executable,
    parse_download_progress,
    parse_url_metadata,
    validate_public_video_url,
)
from app.source_models import SourceSpec
from app.gui.services.url_source_service import URLSourceService


@pytest.mark.parametrize("url", ["file:///C:/video.mp4", "ftp://example.test/video", "http://localhost/video", "http://127.0.0.1/video"])
def test_unsafe_url_schemes_and_local_hosts_are_rejected(url: str) -> None:
    with pytest.raises(SourceError):
        validate_public_video_url(url)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ERROR: Private video. Sign in required", "входа"),
        ("ERROR: This video is DRM protected", "защищено"),
        ("ERROR: Video unavailable", "недоступно"),
    ],
)
def test_public_url_failure_is_explained_without_suggesting_a_bypass(raw: str, expected: str) -> None:
    message = describe_public_url_failure(raw)
    assert expected in message
    assert "cookies" not in message.lower()


def test_metadata_and_progress_are_parsed_without_exposing_internal_ytdlp_output() -> None:
    metadata = parse_url_metadata("https://example.test/video", '{"title":"Видео — тест","duration":91.2,"filesize_approx":1234,"thumbnail":"https://image.test/a.jpg","width":1920,"height":1080,"ext":"mp4"}')
    progress = parse_download_progress("download: 42.5%|1.5MiB/s|00:17")

    assert metadata.title == "Видео — тест"
    assert metadata.duration == 91.2
    assert metadata.estimated_size_bytes == 1234
    assert (metadata.width, metadata.height) == (1920, 1080)
    assert metadata.format == "MP4"
    assert progress is not None
    assert progress.fraction == 0.425
    assert progress.speed == "1.5MiB/s"
    assert progress.eta_seconds == 17


def test_progress_parses_real_transferred_and_expected_volume() -> None:
    progress = parse_download_progress("download: 42.5%|128.0MiB|301.4MiB|NA|1.5MiB/s|00:17")

    assert progress is not None
    assert progress.fraction == 0.425
    assert progress.downloaded == "128.0MiB"
    assert progress.total == "301.4MiB"
    assert progress.speed == "1.5MiB/s"
    assert progress.eta_seconds == 17


def test_ytdlp_is_found_beside_active_virtual_environment_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    executable = tmp_path / "yt-dlp.exe"
    executable.write_bytes(b"yt-dlp")
    monkeypatch.setattr(download_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(download_module.sys, "executable", str(python))
    monkeypatch.setattr(download_module.sys, "platform", "win32")

    assert find_ytdlp_executable() == str(executable)
    assert YtDlpSource().executable == str(executable)


def test_ytdlp_download_uses_argument_list_and_reports_progress(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "источники"
    target.mkdir()
    downloaded = target / "Видео.mp4"
    downloaded.write_bytes(b"video")
    received: dict[str, object] = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO(f"download: 50.0%|2MiB/s|00:04\n{downloaded}\n")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError("download should not be cancelled")

    def fake_popen(arguments, **kwargs):
        received["arguments"] = arguments
        received["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(download_module.subprocess, "Popen", fake_popen)
    updates = []
    result = YtDlpSource("yt-dlp.exe").download("https://example.test/a video?x=1;not-a-command", target, on_progress=updates.append)

    assert result == downloaded.resolve()
    assert received["arguments"][0] == "yt-dlp.exe"
    assert received["arguments"][-1] == "https://example.test/a video?x=1;not-a-command"
    assert "shell" not in received["kwargs"]
    assert updates[0].fraction == 0.5
    assert "--progress" in received["arguments"]
    assert any("_downloaded_bytes_str" in str(item) for item in received["arguments"])
    assert any(str(item).startswith("download:download:") for item in received["arguments"])


def test_ytdlp_download_recovers_a_completed_direct_file_without_after_move_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "sources"
    target.mkdir()
    completed = target / "direct-video.mp4"
    completed.write_bytes(b"video")

    class FakeProcess:
        stdout = io.StringIO("download: 100.0%|2MiB/s|NA\n")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(download_module.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    assert YtDlpSource("yt-dlp.exe").download("https://example.test/video.mp4", target) == completed.resolve()


def test_cancelled_download_removes_only_partial_markers(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "sources"
    target.mkdir()
    partial = target / "Видео.mp4.part"
    partial.write_bytes(b"part")
    unrelated = target / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.StringIO("download: 1.0%|1MiB/s|00:30\n")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(download_module.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(DownloadCancelled):
        YtDlpSource("yt-dlp.exe").download("https://example.test/video", target, cancel_event=cancelled)
    assert not partial.exists()
    assert unrelated.exists()


def test_source_spec_keeps_url_pending_until_a_project_local_file_exists() -> None:
    source = SourceSpec.url("https://example.test/video", {"title": "Видео"})
    assert source.is_ready is False
    source.download_state = "downloaded"
    source.downloaded_path = "C:/project/sources/Видео.mp4"
    assert source.is_ready is True
    source.validate()


def test_cleanup_partial_downloads_leaves_completed_and_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4.part").write_bytes(b"part")
    (tmp_path / "clip.ytdl").write_bytes(b"marker")
    completed = tmp_path / "clip.mp4"
    completed.write_bytes(b"complete")
    other = tmp_path / "notes.txt"
    other.write_text("keep", encoding="utf-8")

    cleanup_partial_downloads(tmp_path)

    assert not (tmp_path / "clip.mp4.part").exists()
    assert not (tmp_path / "clip.ytdl").exists()
    assert completed.exists() and other.exists()


def test_qt_url_download_recovers_direct_media_without_after_move_output(tmp_path: Path) -> None:
    application = QCoreApplication.instance() or QCoreApplication([])
    completed = tmp_path / "direct-video.mp4"
    completed.write_bytes(b"video")
    service = URLSourceService()
    received: list[str] = []
    service.download_completed.connect(received.append)
    service._mode = "download"
    service._url = "https://example.test/video.mp4"
    service._target_directory = tmp_path

    service._finished(0, QProcess.ExitStatus.NormalExit)

    assert received == [str(completed.resolve())]


def test_qt_url_service_releases_its_windows_job_on_terminal_state(monkeypatch) -> None:
    """A completed yt-dlp parent must not leave an FFmpeg helper behind."""

    QCoreApplication.instance() or QCoreApplication([])
    service = URLSourceService()
    job = object()
    released: list[object] = []
    monkeypatch.setattr(url_source_service, "close_windows_process_job", released.append)
    service._job_handle = job
    service._mode = "metadata"
    service._url = "https://example.test/video"

    service._finished(1, QProcess.ExitStatus.NormalExit)

    assert released == [job]
    assert service._job_handle is None
