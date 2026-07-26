"""Safe yt-dlp metadata and download support for public video sources."""

from __future__ import annotations

import ipaddress
import json
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from app.errors import DependencyError, SourceError


@dataclass(frozen=True, slots=True)
class URLMetadata:
    url: str
    title: str
    duration: float | None
    thumbnail_url: str | None
    estimated_size_bytes: int | None
    extractor: str | None
    width: int | None
    height: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    fraction: float | None
    speed: str | None
    eta_seconds: int | None


class DownloadCancelled(SourceError):
    """The user intentionally stopped a source download."""


ProgressCallback = Callable[[DownloadProgress], None]


def find_ytdlp_executable() -> str | None:
    """Find yt-dlp from PATH or beside the active virtual-environment Python."""

    from_path = shutil.which("yt-dlp")
    if from_path:
        return from_path
    scripts = Path(sys.executable).resolve().parent
    names = ("yt-dlp.exe", "yt-dlp") if sys.platform == "win32" else ("yt-dlp", "yt-dlp.exe")
    for name in names:
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    return None


def validate_public_video_url(value: str) -> str:
    """Accept only absolute HTTP(S) URLs that cannot directly address local hosts."""

    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise SourceError("Укажите полную публичную ссылку, начинающуюся с https:// или http://.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise SourceError("Можно использовать только публичную ссылку на видео.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved):
        raise SourceError("Можно использовать только публичную ссылку на видео.")
    return parsed.geturl()


class YtDlpSource:
    """A process-only adapter. It never reads browser cookies or invokes a shell."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or find_ytdlp_executable()

    @property
    def available(self) -> bool:
        return bool(self.executable)

    def inspect(self, url: str) -> URLMetadata:
        safe_url = validate_public_video_url(url)
        executable = self._require_executable()
        command = [
            executable, "--no-playlist", "--skip-download", "--no-warnings",
            "--dump-single-json", safe_url,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90, check=True)
            return parse_url_metadata(safe_url, result.stdout)
        except subprocess.TimeoutExpired as error:
            raise SourceError("Не удалось получить информацию о видео: запрос занял слишком много времени.") from error
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            raise SourceError("Не удалось получить видео по этой ссылке.") from error

    def download(
        self,
        url: str,
        target_directory: Path,
        *,
        on_progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        safe_url = validate_public_video_url(url)
        executable = self._require_executable()
        destination = target_directory.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output_template = str(destination / "%(title).120B-%(id)s.%(ext)s")
        command = [
            executable, "--no-playlist", "--newline", "--no-warnings", "--no-overwrites",
            "--progress-template", "download:%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
            "--print", "after_move:filepath", "-o", output_template, safe_url,
        ]
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace",
            )
        except OSError as error:
            raise DependencyError("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.") from error
        paths: list[Path] = []
        output: list[str] = []
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if line:
                    output.append(line)
                progress = parse_download_progress(line)
                if progress and on_progress:
                    on_progress(progress)
                candidate = Path(line)
                if line and _is_child(candidate, destination):
                    paths.append(candidate)
                if cancel_event and cancel_event.is_set():
                    self._cancel_process(process)
                    cleanup_partial_downloads(destination)
                    raise DownloadCancelled("Загрузка видео отменена.")
            exit_code = process.wait(timeout=20)
        except DownloadCancelled:
            raise
        except subprocess.TimeoutExpired as error:
            self._cancel_process(process)
            raise SourceError("Загрузка видео заняла слишком много времени и была остановлена.") from error
        finally:
            if process.stdout:
                process.stdout.close()
        if cancel_event and cancel_event.is_set():
            cleanup_partial_downloads(destination)
            raise DownloadCancelled("Загрузка видео отменена.")
        if exit_code != 0:
            cleanup_partial_downloads(destination)
            raise SourceError("Не удалось получить видео по этой ссылке.")
        downloaded = next(
            (path.resolve() for path in reversed(paths) if path.is_file() and _is_child(path, destination)),
            None,
        )
        if downloaded is None:
            # Generic/direct-media extractors do not always emit after_move even
            # though they return zero and leave a complete local file. Restrict
            # the recovery to completed files inside this known project folder.
            completed = _completed_downloads(destination)
            downloaded = completed[-1] if completed else None
        if downloaded is None:
            raise SourceError("Загрузка завершилась, но итоговый видеофайл не найден.")
        return downloaded

    def _require_executable(self) -> str:
        if not self.executable:
            raise DependencyError("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.")
        return self.executable

    @staticmethod
    def _cancel_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

_PROGRESS = re.compile(r"^download:\s*(?P<percent>[\d.]+)%\|(?P<speed>[^|]*)\|(?P<eta>[^|]*)$")


def parse_url_metadata(url: str, stdout: str) -> URLMetadata:
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise SourceError("Не удалось получить информацию о видео по этой ссылке.") from error
    if not isinstance(raw, dict):
        raise SourceError("Не удалось получить информацию о видео по этой ссылке.")
    duration = _number(raw.get("duration"))
    size = _integer(raw.get("filesize")) or _integer(raw.get("filesize_approx"))
    return URLMetadata(
        url=url,
        title=str(raw.get("title") or "Видео по ссылке"),
        duration=duration,
        thumbnail_url=str(raw["thumbnail"]) if raw.get("thumbnail") else None,
        estimated_size_bytes=size,
        extractor=str(raw["extractor_key"] or raw["extractor"]) if raw.get("extractor_key") or raw.get("extractor") else None,
        width=_integer(raw.get("width")),
        height=_integer(raw.get("height")),
    )


def parse_download_progress(line: str) -> DownloadProgress | None:
    matched = _PROGRESS.match(line)
    if not matched:
        return None
    try:
        fraction = max(0.0, min(1.0, float(matched.group("percent")) / 100.0))
    except ValueError:
        fraction = None
    eta_text = matched.group("eta").strip()
    return DownloadProgress(fraction, matched.group("speed").strip() or None, _eta_seconds(eta_text))


def cleanup_partial_downloads(directory: Path) -> None:
    """Delete only yt-dlp partial markers inside the known project-local source folder."""

    root = directory.expanduser().resolve()
    if not root.is_dir():
        return
    for item in root.glob("*"):
        if item.is_file() and (item.name.endswith(".part") or item.name.endswith(".ytdl")) and _is_child(item, root):
            item.unlink(missing_ok=True)


def _completed_downloads(directory: Path) -> list[Path]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = [
        item.resolve()
        for item in root.iterdir()
        if item.is_file()
        and not item.name.endswith(".part")
        and not item.name.endswith(".ytdl")
        and _is_child(item, root)
    ]
    return sorted(candidates, key=lambda item: item.stat().st_mtime_ns)


def _eta_seconds(value: str) -> int | None:
    if not value or value.upper() == "NA":
        return None
    parts = value.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return None


def _is_child(path: Path, directory: Path) -> bool:
    try:
        return path.expanduser().resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _integer(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None
