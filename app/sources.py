from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.errors import DependencyError, SourceError
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
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SourceError("Нужна полная публичная ссылка, начинающаяся с https:// или http://.")
    executable = shutil.which("yt-dlp")
    if not executable:
        raise DependencyError("yt-dlp не найден. Установите зависимости и повторите запуск.")
    target_directory.mkdir(parents=True, exist_ok=True)
    template = str(target_directory / "%(title).80B-%(id)s.%(ext)s")
    command = [
        executable, "--no-playlist", "--restrict-filenames", "--no-progress",
        "--print", "after_move:filepath", "-o", template, url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3600)
    except subprocess.CalledProcessError as error:
        message = (error.stderr or error.stdout or "").strip().splitlines()[-1:]
        raise SourceError(
            "Не удалось получить видео. Проверьте, что ссылка публичная и не защищена "
            f"авторизацией, DRM или paywall. {' '.join(message)}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise SourceError("Загрузка заняла слишком много времени и была остановлена.") from error
    candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    path = next((item for item in reversed(candidates) if item.is_file()), None)
    if path is None:
        raise SourceError("yt-dlp завершился, но не вернул путь к загруженному видео.")
    return Source(
        id=stable_text_hash(url + stable_file_hash(path))[:24],
        path=path.resolve(),
        display_name=safe_name(path.name),
        origin=url,
        downloaded=True,
    )
