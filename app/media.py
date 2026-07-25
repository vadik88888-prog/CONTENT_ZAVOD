from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.errors import DependencyError, StageError
from app.utils import write_json


def _tool(name: str) -> str:
    value = shutil.which(name)
    if not value:
        raise DependencyError(f"{name} не найден в PATH. Установите FFmpeg и повторите запуск.")
    return value


def run_checked(arguments: list[str], timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise StageError(f"FFmpeg завершился с ошибкой: {details[-1200:]}") from error
    except subprocess.TimeoutExpired as error:
        raise StageError("FFmpeg работал слишком долго и был остановлен.") from error


def probe_video(path: Path) -> dict[str, Any]:
    result = run_checked([
        _tool("ffprobe"), "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(path),
    ], timeout=120)
    try:
        raw = json.loads(result.stdout)
        streams = raw.get("streams", [])
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        duration = float(raw.get("format", {}).get("duration") or video.get("duration") or 0)
    except (ValueError, KeyError, StopIteration, TypeError) as error:
        raise StageError("Файл не похож на читаемое видео с видеопотоком.") from error
    if duration <= 0:
        raise StageError("У видео не удалось определить положительную длительность.")
    fps = _fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    return {
        "duration": duration,
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "fps": fps,
        "video_codec": video.get("codec_name"),
        "audio_streams": len(audio),
        "audio_codecs": [item.get("codec_name") for item in audio],
        "format": raw.get("format", {}).get("format_name"),
        "source_path": str(path),
    }


def _fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        numerator, denominator = value.split("/", 1)
        return round(float(numerator) / float(denominator), 3)
    except (ValueError, ZeroDivisionError):
        return None


def prepare_media(source_path: Path, work_directory: Path) -> dict[str, Any]:
    metadata = probe_video(source_path)
    audio_path = work_directory / "audio_16khz_mono.wav"
    if metadata["audio_streams"]:
        run_checked([
            _tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i",
            str(source_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a",
            "pcm_s16le", str(audio_path),
        ])
        metadata["audio_path"] = str(audio_path)
    else:
        metadata["audio_path"] = None
        metadata["warning"] = "В видео нет аудиодорожки: распознавание речи и клипы недоступны."
    write_json(work_directory / "metadata.json", metadata)
    return metadata
