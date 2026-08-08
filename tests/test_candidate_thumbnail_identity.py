from __future__ import annotations

from pathlib import Path

from app.gui.components.candidate_thumbnail import CandidateThumbnailLoader


def test_thumbnail_queue_keys_requests_by_generated_destination(tmp_path: Path, monkeypatch) -> None:
    """A repeated candidate id must not discard a newer analysis/frame request."""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    loader = CandidateThumbnailLoader()
    monkeypatch.setattr(loader, "_start_next", lambda: None)

    first = loader.request(
        cache_directory=tmp_path / "thumbnails",
        analysis_id="first-analysis",
        candidate_id="candidate-001",
        source_path=source,
        timestamp_seconds=3.0,
    )
    second = loader.request(
        cache_directory=tmp_path / "thumbnails",
        analysis_id="second-analysis",
        candidate_id="candidate-001",
        source_path=source,
        timestamp_seconds=4.0,
    )

    assert first != second
    assert len(loader._queue) == 2
    assert {request.destination for request in loader._queue} == {first, second}
