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
    BGUTIL_PROVIDER_VERSION,
    build_ytdlp_download_arguments,
    build_ytdlp_inspect_arguments,
    classify_ytdlp_failure,
    DownloadCancelled,
    YtDlpSource,
    YtDlpCapabilities,
    YtDlpFailureReason,
    YtDlpSourceError,
    cleanup_partial_downloads,
    describe_public_url_failure,
    find_ytdlp_executable,
    parse_download_progress,
    parse_url_metadata,
    normalize_ytdlp_diagnostics,
    sanitize_public_url_for_diagnostics,
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ERROR: Sign in to confirm you’re not a bot", YtDlpFailureReason.BOT_CHECK),
        ("WARNING: No supported JavaScript runtime could be found", YtDlpFailureReason.JS_RUNTIME_MISSING),
        ("ERROR: This client requires a PO Token", YtDlpFailureReason.PO_TOKEN_REQUIRED),
        ("ERROR: Private video. Sign in required", YtDlpFailureReason.LOGIN_REQUIRED),
        ("ERROR: Unsupported URL", YtDlpFailureReason.UNSUPPORTED),
        ("ERROR: Video unavailable", YtDlpFailureReason.UNAVAILABLE),
    ],
)
def test_ytdlp_failure_categories_keep_diagnostics_but_expose_safe_text(
    raw: str, expected: YtDlpFailureReason,
) -> None:
    error = classify_ytdlp_failure(raw)

    assert error.reason == expected
    assert error.diagnostics == raw
    assert "ERROR:" not in str(error)
    assert "WARNING:" not in str(error)


def test_ytdlp_diagnostic_redaction_removes_cookie_and_url_query_secrets() -> None:
    diagnostics = normalize_ytdlp_diagnostics(
        "Cookie: session=never-persist; csrf=also-secret\n"
        "ERROR: https://example.test/video?api_key=never-persist&signature=also-secret\n",
    )

    assert diagnostics == "Cookie: [redacted]\nERROR: https://example.test/video"
    assert sanitize_public_url_for_diagnostics(
        "https://user:never-persist@example.test/video?token=also-secret",
    ) == "https://example.test/video"


def test_engine_and_desktop_share_public_only_ytdlp_contract(tmp_path: Path) -> None:
    capabilities = YtDlpCapabilities("yt-dlp.exe", "C:/portable/tools/deno.exe")
    url = "https://example.test/video"
    inspect = build_ytdlp_inspect_arguments(capabilities, url)
    download = build_ytdlp_download_arguments(capabilities, url, tmp_path)

    shared = [
        "--ignore-config", "--no-playlist", "--js-runtimes", "deno:C:/portable/tools/deno.exe",
    ]
    assert inspect[:5] == [*shared, "--skip-download"]
    assert download[:5] == [*shared, "--newline"]
    for arguments in (inspect, download):
        assert "--no-warnings" not in arguments
        assert not any("cookie" in argument.casefold() for argument in arguments)
        assert not any(
            argument in {"-f", "--format", "--remux-video", "--merge-output-format"}
            for argument in arguments
        )
    assert inspect[-1] == download[-1] == url


def test_youtube_mweb_provider_contract_is_pinned_and_never_uses_cookies(tmp_path: Path) -> None:
    plugin_directory = tmp_path
    server_home = tmp_path / "server"
    capabilities = YtDlpCapabilities(
        "yt-dlp.exe",
        "C:/portable/tools/deno.exe",
        po_token_provider=True,
        plugin_directory=str(plugin_directory),
        po_token_server_home=str(server_home),
        runtime_cache_directory=str(tmp_path / "cache"),
    )

    arguments = build_ytdlp_download_arguments(
        capabilities, "https://www.youtube.com/watch?v=_PCWk_GD9c4", tmp_path,
    )
    environment = capabilities.process_environment({"PATH": "C:/portable/tools"})

    assert BGUTIL_PROVIDER_VERSION == "1.3.2"
    assert [
        "--plugin-dirs", str(plugin_directory),
        "--remote-components", "ejs:github",
        "--extractor-args", "youtube:player_client=mweb",
        "--extractor-args", f"youtubepot-bgutilscript:server_home={server_home}",
    ] == arguments[4:12]
    assert environment["XDG_CACHE_HOME"] == str((tmp_path / "cache").resolve())
    assert environment["DENO_DIR"] == str((tmp_path / "cache" / "deno").resolve())
    assert (tmp_path / "cache" / "bgutil-ytdlp-pot-provider").is_dir()
    assert not any("cookie" in argument.casefold() for argument in arguments)
    assert not any(argument in {"-f", "--format", "--remux-video", "--merge-output-format"} for argument in arguments)


def test_inspect_parses_only_stdout_and_retains_stderr_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = '{"title":"Public video","duration":12,"ext":"mp4"}'
        stderr = "WARNING: [youtube] diagnostic warning\n"

    def fake_run(arguments, **kwargs):
        received["arguments"] = arguments
        received["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    source = YtDlpSource(
        capabilities=YtDlpCapabilities("yt-dlp.exe", "C:/portable/tools/deno.exe"),
    )

    metadata = source.inspect("https://example.test/video")

    assert metadata.title == "Public video"
    assert source.last_diagnostics == "WARNING: [youtube] diagnostic warning"
    assert received["arguments"] == ["yt-dlp.exe", *build_ytdlp_inspect_arguments(source.capabilities, metadata.url)]
    assert received["kwargs"]["check"] is False


def test_inspect_failure_keeps_raw_diagnostics_on_safe_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stdout = ""
        stderr = "ERROR: This client requires a PO Token"

    monkeypatch.setattr(download_module.subprocess, "run", lambda *_args, **_kwargs: Result())
    source = YtDlpSource(capabilities=YtDlpCapabilities("yt-dlp.exe", "deno.exe"))

    with pytest.raises(YtDlpSourceError) as captured:
        source.inspect("https://example.test/video")

    assert captured.value.reason == YtDlpFailureReason.PO_TOKEN_REQUIRED
    assert captured.value.diagnostics == Result.stderr
    assert "PO Token" in str(captured.value)


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
    monkeypatch.setattr(download_module, "_managed_tools_directory", lambda: tmp_path / "missing-tools")
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


def test_qt_metadata_keeps_warnings_out_of_json() -> None:
    QCoreApplication.instance() or QCoreApplication([])
    service = URLSourceService()
    received: list[dict] = []
    service.metadata_ready.connect(received.append)
    service._mode = "metadata"
    service._url = "https://example.test/video"
    service._stdout_chunks = ['{"title":"Public video","duration":9,"ext":"mp4"}\n']
    service._stderr_chunks = ["WARNING: useful extractor diagnostic\n"]

    service._finished(0, QProcess.ExitStatus.NormalExit)

    assert received[0]["title"] == "Public video"
    assert service.last_diagnostics == "WARNING: useful extractor diagnostic"


def test_qt_download_failure_keeps_safe_exit_evidence() -> None:
    QCoreApplication.instance() or QCoreApplication([])
    service = URLSourceService()
    received: list[str] = []
    service.failed.connect(received.append)
    service._mode = "download"
    service._url = "https://example.test/video?token=never-persist"
    service._stderr_chunks = ["ERROR: This client requires a PO Token token=never-persist\n"]

    service._finished(1, QProcess.ExitStatus.NormalExit)

    assert received and "PO Token" in received[0]
    assert service.last_failure is not None
    assert service.last_failure.exit_code == 1
    assert service.last_failure.reason == YtDlpFailureReason.PO_TOKEN_REQUIRED.value
    assert service.last_failure.url == "https://example.test/video"
    assert "never-persist" not in service.last_failure.last_diagnostics
    assert "token=[redacted]" in service.last_failure.last_diagnostics


def test_qt_service_uses_shared_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    QCoreApplication.instance() or QCoreApplication([])
    capabilities = YtDlpCapabilities("yt-dlp.exe", "C:/portable/tools/deno.exe")
    monkeypatch.setattr(url_source_service, "detect_ytdlp_capabilities", lambda: capabilities)
    service = URLSourceService()
    received: list[list[str]] = []
    monkeypatch.setattr(service, "_start", received.append)

    service.inspect("https://example.test/video")

    assert received == [build_ytdlp_inspect_arguments(capabilities, "https://example.test/video")]
    assert service.process.program() == "yt-dlp.exe"


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
