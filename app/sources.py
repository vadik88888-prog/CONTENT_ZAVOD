from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.errors import DependencyError, SourceError
from app.source_download import validate_public_video_url
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
    executable = shutil.which("yt-dlp")
    if not executable:
        raise DependencyError("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")
    target_directory.mkdir(parents=True, exist_ok=True)
    template = str(target_directory / "%(title).80B-%(id)s.%(ext)s")
    command = [
        executable, "--no-playlist", "--restrict-filenames", "--no-progress",
        "--print", "after_move:filepath", "-o", template, safe_url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3600)
    except subprocess.CalledProcessError as error:
        raise SourceError("Не удалось получить видео по этой ссылке.") from error
    except subprocess.TimeoutExpired as error:
        raise SourceError("Загрузка заняла слишком много времени и была остановлена.") from error
    candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    path = next((item for item in reversed(candidates) if item.is_file()), None)
    if path is None:
        raise SourceError("yt-dlp завершился, но не вернул путь к загруженному видео.")
    return Source(
        id=stable_text_hash(safe_url + stable_file_hash(path))[:24],
        path=path.resolve(),
        display_name=safe_name(path.name),
        origin=safe_url,
        downloaded=True,
    )
