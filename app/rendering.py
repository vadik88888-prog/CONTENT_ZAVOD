from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.config import AppConfig
from app.errors import DependencyError, StageError
from app.models import ScoredCandidate


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise DependencyError("FFmpeg не найден в PATH. Установите FFmpeg и повторите запуск.")
    return executable


def nvenc_available() -> bool:
    try:
        result = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "h264_nvenc" in result.stdout


def render_clip(
    source: Path,
    item: ScoredCandidate,
    ass_path: Path | None,
    destination: Path,
    config: AppConfig,
) -> tuple[Path, bool, str | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = _filters(config, ass_path)
    first_encoder = "h264_nvenc" if config.encoder_preference != "cpu" and nvenc_available() else "libx264"
    try:
        _render_with_encoder(source, item, destination, filters, first_encoder)
        return destination, first_encoder == "h264_nvenc", None
    except StageError as error:
        if first_encoder != "h264_nvenc":
            raise
        _render_with_encoder(source, item, destination, filters, "libx264")
        return destination, False, f"NVENC недоступен для рендера, использован CPU: {error}"


def _render_with_encoder(
    source: Path, item: ScoredCandidate, destination: Path, filters: str, encoder: str
) -> None:
    command = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{item.candidate.start:.3f}", "-i", str(source),
        "-t", f"{item.candidate.duration:.3f}", "-filter_complex", filters,
        "-map", "[vout]", "-map", "0:a?", "-c:v", encoder,
        "-preset", "p4" if encoder == "h264_nvenc" else "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=7200)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise StageError(f"Рендер {item.candidate.id} не выполнен: {details[-1200:]}") from error
    except subprocess.TimeoutExpired as error:
        raise StageError(f"Рендер {item.candidate.id} занял слишком много времени.") from error


def _filters(config: AppConfig, ass_path: Path | None) -> str:
    width, height = config.output_width, config.output_height
    if config.render_mode == "center-crop":
        video = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1[vbase]"
        )
    else:
        video = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:10[blur];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fit];"
            f"[blur][fit]overlay=(W-w)/2:(H-h),setsar=1[vbase]"
        )
    if ass_path is None:
        return video.replace("[vbase]", "[vout]")
    return video + f";[vbase]ass='{_filter_path(ass_path)}'[vout]"


def _filter_path(path: Path) -> str:
    # В filtergraph двоеточие диска Windows и обратные слэши необходимо экранировать.
    value = path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
    return value
