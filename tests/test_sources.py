import pytest

from app.errors import SourceError
from app.sources import url_source, validate_source_arguments


@pytest.mark.parametrize(
    ("input_path", "url"),
    [(None, None), ("video.mp4", "https://example.com/video")],
)
def test_exactly_one_source_is_required(input_path, url) -> None:
    with pytest.raises(SourceError, match="ровно один"):
        validate_source_arguments(input_path, url)


def test_url_source_runs_ytdlp_without_shell_interpolation(tmp_path, monkeypatch) -> None:
    downloaded = tmp_path / "downloaded.mp4"
    downloaded.write_bytes(b"video")
    received: dict[str, object] = {}

    class FakeDownloader:
        def download(self, url, target_directory):
            received["url"] = url
            received["target_directory"] = target_directory
            return downloaded

    monkeypatch.setattr("app.sources.YtDlpSource", FakeDownloader)

    source = url_source("https://example.test/a video?x=1;not-a-command", tmp_path)

    assert source.path == downloaded.resolve()
    assert received["url"] == "https://example.test/a video?x=1;not-a-command"
    assert received["target_directory"] == tmp_path
