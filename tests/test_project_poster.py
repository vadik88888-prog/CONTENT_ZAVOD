from __future__ import annotations

from pathlib import Path

from app.gui.components.candidate_thumbnail import thumbnail_path
from app.gui.components.project_poster import (
    PROJECT_POSTER_FILTER,
    PROJECT_POSTER_PROFILE_ID,
    ProjectPosterLoader,
    ProjectPosterRequest,
    project_poster_ffmpeg_arguments,
    project_poster_has_input,
    project_poster_path,
    project_youtube_thumbnail_url,
    youtube_thumbnail_url,
)
from app.gui.services.desktop_project_store import DesktopProjectStore


def test_local_project_poster_uses_new_profile_and_source_revision(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first source revision")
    project = DesktopProjectStore(tmp_path / "data").create(source)

    first = project_poster_path(project)
    old_candidate_profile = thumbnail_path(
        project.directory / "thumbnails",
        "source-poster-v1",
        project.project_id,
        source,
        1.0,
    )
    source.write_bytes(b"second, longer source revision")
    second = project_poster_path(project)

    assert first.parent.name == PROJECT_POSTER_PROFILE_ID
    assert first.suffix == ".jpg"
    assert first != old_candidate_profile
    assert first != second


def test_youtube_project_prefers_inspected_real_thumbnail(tmp_path: Path, monkeypatch) -> None:
    metadata = {
        "title": "Public video",
        "extractor": "Youtube",
        "thumbnail_url": "https://i.ytimg.com/vi/example/maxresdefault.jpg",
    }
    project = DesktopProjectStore(tmp_path / "data").create_url(
        "https://www.youtube.com/watch?v=example",
        metadata,
    )
    loader = ProjectPosterLoader()
    monkeypatch.setattr(loader, "_start_next", lambda: None)

    destination = loader.request(project)
    request = loader._queue[0]

    assert project_poster_has_input(project)
    assert project_youtube_thumbnail_url(project) == metadata["thumbnail_url"]
    assert request.thumbnail_url == metadata["thumbnail_url"]
    assert request.source_path is None
    assert destination.parent.name == PROJECT_POSTER_PROFILE_ID


def test_project_poster_profile_is_hq_while_candidate_profile_stays_lightweight(tmp_path: Path) -> None:
    request = ProjectPosterRequest(
        project_id="project-001",
        source_path=tmp_path / "source.mp4",
        thumbnail_url=None,
        timestamp_seconds=1.0,
        destination=tmp_path / "poster.jpg",
        temporary_path=tmp_path / ".poster.test.tmp.jpg",
    )

    arguments = project_poster_ffmpeg_arguments(request, use_thumbnail_url=False)

    assert PROJECT_POSTER_FILTER in arguments
    assert "-q:v" in arguments
    assert arguments[arguments.index("-q:v") + 1] == "2"
    candidate_source = Path("app/gui/components/candidate_thumbnail.py").read_text(encoding="utf-8")
    assert '"scale=240:-2"' in candidate_source
    assert '"-q:v", "5"' in candidate_source


def test_concurrent_project_loaders_use_unique_temps_and_accept_shared_final(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    project = DesktopProjectStore(tmp_path / "data").create(source)
    first_loader = ProjectPosterLoader()
    second_loader = ProjectPosterLoader()
    monkeypatch.setattr(first_loader, "_start_next", lambda: None)
    monkeypatch.setattr(second_loader, "_start_next", lambda: None)

    first_destination = first_loader.request(project)
    second_destination = second_loader.request(project)
    first_request = first_loader._queue.popleft()
    second_request = second_loader._queue.popleft()

    assert first_destination == second_destination
    assert first_request.temporary_path != second_request.temporary_path

    ready: list[tuple[str, str]] = []
    first_loader.poster_ready.connect(lambda project_id, path: ready.append((project_id, path)))
    first_loader._active = first_request
    first_destination.parent.mkdir(parents=True)
    first_destination.write_bytes(b"poster completed by the other screen")
    first_loader._finish_attempt(False)

    assert ready == [(project.project_id, str(first_destination))]


def test_only_real_youtube_thumbnail_hosts_are_accepted() -> None:
    assert youtube_thumbnail_url({
        "extractor": "Youtube",
        "thumbnail_url": "https://i.ytimg.com/vi/id/maxresdefault.jpg",
    }) == "https://i.ytimg.com/vi/id/maxresdefault.jpg"
    assert youtube_thumbnail_url({
        "extractor": "Vimeo",
        "thumbnail_url": "https://i.ytimg.com/vi/id/maxresdefault.jpg",
    }) is None
    assert youtube_thumbnail_url({
        "extractor": "Youtube",
        "thumbnail_url": "https://example.test/poster.jpg",
    }) is None
