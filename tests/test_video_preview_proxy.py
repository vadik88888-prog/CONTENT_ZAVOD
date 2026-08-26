from __future__ import annotations

from pathlib import Path

from app.gui.components.video_preview import preview_proxy_path, preview_proxy_temporary_path


def test_preview_proxy_path_is_specific_to_source_revision_not_candidate_range(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"first")
    cache = tmp_path / "preview-proxies"

    first = preview_proxy_path(cache, source)
    same = preview_proxy_path(cache, source)
    source.write_bytes(b"changed source revision")
    changed_source = preview_proxy_path(cache, source)

    assert first == same
    assert first.suffix == ".mp4"
    assert first != changed_source


def test_preview_proxy_temporary_path_keeps_the_mp4_muxer_suffix(tmp_path: Path) -> None:
    destination = tmp_path / "preview.mp4"

    temporary = preview_proxy_temporary_path(destination)

    assert temporary.name == "preview.part.mp4"
    assert temporary.parent == destination.parent
