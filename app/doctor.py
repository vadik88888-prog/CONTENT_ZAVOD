from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig


@dataclass(slots=True)
class Check:
    label: str
    status: str
    detail: str


def _run(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            arguments, check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _nvidia_checks() -> list[Check]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [
            Check("NVIDIA GPU", "warn", "nvidia-smi не найден; возможна работа только на CPU."),
        ]
    output = _run([
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return [
            Check("NVIDIA GPU", "warn", "nvidia-smi есть, но не вернул данные."),
        ]
    return [
        Check("NVIDIA GPU", "ok", output.replace(",", " ·")),
    ]


def _cuda_check() -> Check:
    try:
        import ctranslate2
    except ImportError:
        return Check(
            "CUDA",
            "warn",
            "Нельзя проверить без faster-whisper/CTranslate2; после установки выполните doctor ещё раз.",
        )
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as error:
        return Check("CUDA", "warn", f"CTranslate2 не смог проверить CUDA: {error}")
    if count:
        return Check("CUDA", "ok", f"CTranslate2 видит CUDA-устройств: {count}.")
    return Check(
        "CUDA",
        "warn",
        "CTranslate2 не видит доступную CUDA; faster-whisper будет использовать CPU.",
    )


def collect_checks(root: Path, config: AppConfig | None = None) -> list[Check]:
    config = config or AppConfig()
    checks = [
        Check(
            "Python",
            "ok" if sys.version_info >= (3, 11) else "error",
            f"{sys.version.split()[0]} (требуется 3.11+)",
        ),
    ]
    for command, title in (("ffmpeg", "FFmpeg"), ("ffprobe", "FFprobe"), ("yt-dlp", "yt-dlp")):
        executable = shutil.which(command)
        checks.append(
            Check(title, "ok" if executable else "error", executable or "Не найден в PATH.")
        )
    checks.extend(_nvidia_checks())
    checks.append(_cuda_check())
    has_whisper = importlib.util.find_spec("faster_whisper") is not None
    checks.append(Check(
        "faster-whisper",
        "ok" if has_whisper else "error",
        "Установлен." if has_whisper else "Не установлен: pip install -r requirements.txt",
    ))
    provider = config.ai.provider
    checks.append(Check("AI provider", "ok", f"{provider} · {config.ai.model}"))
    if provider == "mock":
        checks.append(Check("AI API key", "ok", "Не требуется для mock-провайдера."))
    else:
        variable = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        label = "OpenAI API key" if provider == "openai" else "Gemini API key"
        checks.append(Check(
            label,
            "ok" if os.getenv(variable) else "warn",
            "Найден." if os.getenv(variable) else "Не задан — используйте --mock-ai или добавьте ключ в .env.",
        ))
    if config.tts.enabled:
        tts_provider = config.tts.provider
        checks.append(Check("TTS provider", "ok", f"{tts_provider} · {config.tts.model}"))
        if tts_provider == "openai":
            checks.append(Check(
                "OpenAI TTS API key",
                "ok" if os.getenv("OPENAI_API_KEY") else "warn",
                "Найден." if os.getenv("OPENAI_API_KEY") else "Не задан — используйте tts.provider=mock/local или добавьте ключ в .env.",
            ))
    for folder in ("input", "work", "output"):
        path = root / folder
        try:
            path.mkdir(parents=True, exist_ok=True)
            writable = os.access(path, os.W_OK)
        except OSError:
            writable = False
        checks.append(Check(
            f"Папка {folder}",
            "ok" if writable else "error",
            str(path) if writable else f"Нет доступа: {path}",
        ))
    return checks


def format_report(checks: list[Check]) -> str:
    # ASCII-метки работают и в стандартной Windows-консоли cp1251.
    glyph = {"ok": "OK", "warn": "!", "error": "X"}
    lines = ["Content Factory — проверка окружения", ""]
    for check in checks:
        lines.append(f"{glyph[check.status]} {check.label}: {check.detail}")
    errors = sum(item.status == "error" for item in checks)
    warnings = sum(item.status == "warn" for item in checks)
    lines.extend([
        "",
        f"Итог: ошибок — {errors}, предупреждений — {warnings}.",
        "Ключ AI не обязателен: локальный режим с --mock-ai доступен без него.",
    ])
    return "\n".join(lines)
