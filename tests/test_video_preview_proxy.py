from __future__ import annotations

from pathlib import Path

from app.gui.components.video_preview import preview_proxy_path


def test_preview_proxy_path_is_specific_to_source_revision_and_range(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"first")
    cache = tmp_path / "preview-proxies"

    first = preview_proxy_path(cache, source, 10.0, 18.0)
    same = preview_proxy_path(cache, source, 10.0, 18.0)
    other_range = preview_proxy_path(cache, source, 11.0, 18.0)
    source.write_bytes(b"changed source revision")
    changed_source = preview_proxy_path(cache, source, 10.0, 18.0)

    assert first == same
    assert first.suffix == ".mp4"
    assert first != other_range
    assert first != changed_source
