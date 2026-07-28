from __future__ import annotations

from pathlib import Path

from app.gui.components.candidate_thumbnail import thumbnail_path


def test_candidate_thumbnail_path_changes_for_source_revision_and_analysis(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first")
    cache = tmp_path / "candidate-thumbnails"

    first = thumbnail_path(cache, "analysis-one", "candidate-001", source, 12.5)
    second = thumbnail_path(cache, "analysis-two", "candidate-001", source, 12.5)
    source.write_bytes(b"second source revision")
    third = thumbnail_path(cache, "analysis-one", "candidate-001", source, 12.5)

    assert first.parent.name == "analysis-one"
    assert first.suffix == ".jpg"
    assert first != second
    assert first != third
