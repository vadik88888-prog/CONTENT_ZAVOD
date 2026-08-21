from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.errors import SourceError
from app.source_download import validate_public_video_url, YtDlpSource
from app.utils import safe_name, stable_file_hash, stable_text_hash


@dataclass(slots=True)
class Source:
    id: str
    path: Path
    display_name: str
    origin: str
    downloaded: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": str(self.path),
            "display_name": self.display_name,
            "origin": self.origin,
            "downloaded": self.downloaded,
        }


def validate_source_arguments(input_path: str | None, url: str | None) -> None:
    if bool(input_path) == bool(url):
        raise SourceError("Укажите ровно один источник: --input или --url.")


def local_source(value: str) -> Source:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SourceError(f"Локальный видеофайл не найден: {path}")
    return Source(
        id=stable_file_hash(path)[:24],
        path=path,
        display_name=safe_name(path.name),
        origin=str(path),
    )


def url_source(url: str, target_directory: Path) -> Source:
    safe_url = validate_public_video_url(url)
    path = YtDlpSource().download(safe_url, target_directory)
    return Source(
        id=stable_text_hash(safe_url + stable_file_hash(path))[:24],
        path=path.resolve(),
        display_name=safe_name(path.name),
        origin=safe_url,
        downloaded=True,
    )
