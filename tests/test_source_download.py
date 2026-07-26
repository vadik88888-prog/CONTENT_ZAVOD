from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

import app.source_download as download_module
from app.errors import SourceError
from app.source_download import (
    DownloadCancelled,
    YtDlpSource,
    cleanup_partial_downloads,
    parse_download_progress,
    parse_url_metadata,
    validate_public_video_url,
)
from app.source_models import SourceSpec


@pytest.mark.parametrize("url", ["file:///C:/video.mp4", "ftp://example.test/video", "http://localhost/video", "http://127.0.0.1/video"])
def test_unsafe_url_schemes_and_local_hosts_are_rejected(url: str) -> None:
    with pytest.raises(SourceError):
        validate_public_video_url(url)


def test_metadata_and_progress_are_parsed_without_exposing_internal_ytdlp_output() -> None:
    metadata = parse_url_metadata("https://example.test/video", '{"title":"Видео — тест","duration":91.2,"filesize_approx":1234,"thumbnail":"https://image.test/a.jpg","width":1920,"height":1080}')
    progress = parse_download_progress("download: 42.5%|1.5MiB/s|00:17")

    assert metadata.title == "Видео — тест"
    assert metadata.duration == 91.2
    assert metadata.estimated_size_bytes == 1234
    assert (metadata.width, metadata.height) == (1920, 1080)
    assert progress is not None
    assert progress.fraction == 0.425
    assert progress.speed == "1.5MiB/s"
    assert progress.eta_seconds == 17


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
