from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.config import AppConfig
from app.cuda_runtime import probe_cuda_runtime
from app.runtime import RuntimeLayout
from app.source_download import find_deno_executable, find_ytdlp_executable
from app.subprocess_utils import UTF8_REPLACE_TEXT


class DoctorReadiness(StrEnum):
    READY = "ready"
    LIMITED = "limited"
    SETUP_REQUIRED = "setup_required"


class CredentialProbeStatus(StrEnum):
    CONFIGURED = "configured"
    AUTH_REJECTED = "auth_rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Check:
    """One bounded diagnostic result.

    ``status`` retains the established ``ok/warn/error`` values for callers;
    in friend-beta language they mean ready, WARNING and BLOCKING.
    """

    label: str
    status: str
    detail: str
    action: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"ok", "warn", "error"}:
            raise ValueError("Unknown doctor check status.")
        if self.status != "ok" and not self.action.strip():
            raise ValueError("A non-OK doctor check must provide a recovery action.")

    @property
    def blocking(self) -> bool:
        return self.status == "error"

    @property
    def warning(self) -> bool:
        return self.status == "warn"


@dataclass(frozen=True, slots=True)
class DoctorSummary:
    readiness: DoctorReadiness
    title: str
    detail: str
    blocking_count: int
    warning_count: int


def _run(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            arguments, check=True, capture_output=True, timeout=10, **UTF8_REPLACE_TEXT
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _nvidia_checks() -> list[Check]:
    executable = shutil.which("nvidia-smi")
    action = "Можно продолжить на CPU или установить совместимый драйвер NVIDIA."
    if not executable:
        return [Check("NVIDIA GPU", "warn", "nvidia-smi не найден; доступна работа на CPU.", action)]
    output = _run([
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return [Check("NVIDIA GPU", "warn", "Драйвер найден, но не вернул сведения о GPU.", action)]
    return [Check("NVIDIA GPU", "ok", output.replace(",", " ·"))]


def _cuda_check() -> Check:
    action = "Оставьте устройство в режиме «Автоматически»/CPU или установите совместимые CUDA-компоненты."
    probe = probe_cuda_runtime()
    if probe.usable:
        return Check("CUDA", "ok", f"CTranslate2 и CUDA runtime готовы; устройств: {probe.device_count}.")
    return Check(
        "CUDA",
        "warn",
        f"{probe.fallback_reason}; faster-whisper сразу будет использовать CPU int8.",
        action,
    )


def _layout(value: Path | RuntimeLayout) -> RuntimeLayout:
    if isinstance(value, RuntimeLayout):
        return value
    # RuntimeLayout remains the sole source/frozen location resolver.  The
    # explicit path is the writable data root used by CLI and test callers.
    return RuntimeLayout.detect(data=Path(value))


def _find_executable(runtime: RuntimeLayout, command: str) -> str | None:
    environment = runtime.process_environment()
    executable = shutil.which(command, path=environment.get("PATH"))
    if executable:
        return executable
    if command == "yt-dlp":
        fallback = find_ytdlp_executable()
        return str(fallback) if fallback else None
    if command == "deno":
        fallback = find_deno_executable(find_ytdlp_executable())
        return str(fallback) if fallback else None
    return None


def _tool_check(runtime: RuntimeLayout, command: str, label: str, *, blocking: bool) -> Check:
    executable = _find_executable(runtime, command)
    status = "error" if blocking else "warn"
    if not executable:
        if blocking:
            action = "Переустановите portable-сборку или добавьте программу в PATH, затем повторите проверку."
        elif command == "deno":
            action = "Для полной поддержки YouTube обновите portable-сборку или используйте локальный видеофайл."
        else:
            action = "Для загрузки по ссылке установите yt-dlp или используйте локальный видеофайл."
        return Check(label, status, "Исполняемый файл не найден.", action)
    version = _run(
        [executable, "--version"] if command in {"yt-dlp", "deno"} else [executable, "-version"]
    )
    if version is None:
        if blocking:
            action = "Переустановите portable-сборку или исправьте PATH, затем повторите проверку."
        elif command == "deno":
            action = "Обновите portable-сборку с исправным Deno или используйте локальный видеофайл."
        else:
            action = "Обновите yt-dlp или используйте локальный видеофайл."
        return Check(label, status, "Исполняемый файл найден, но не запускается.", action)
    return Check(label, "ok", f"Доступен: {Path(executable).name}.")


def _writable_directory_check(path: Path, label: str) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".doctor-", dir=path, delete=True):
            pass
    except OSError:
        return Check(
            label,
            "error",
            "Папка недоступна для записи.",
            "Выберите доступную локальную папку данных и повторите проверку.",
        )
    return Check(label, "ok", f"Доступна для записи: {path}")


def _probe_api_credential(provider: str, secret: str) -> str:
    """Validate one real-provider credential without exposing its value.

    Both supported providers accept credentials in HTTPS headers for their
    read-only model-list endpoint.  Only an explicit 401/403 is an auth
    rejection; transport failures, rate limits and provider 5xx responses are
    temporary availability warnings.
    """

    if provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {secret}"}
    elif provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"
        headers = {"x-goog-api-key": secret}
    else:
        return CredentialProbeStatus.UNAVAILABLE
    request = urllib.request.Request(
        url,
        headers={
            **headers,
            "Accept": "application/json",
            "User-Agent": "ContentFactory-Doctor/1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status_value = getattr(response, "status", None)
            status = int(status_value if status_value is not None else response.getcode())
    except urllib.error.HTTPError as error:
        status = int(error.code)
        error.close()
    except (urllib.error.URLError, TimeoutError, OSError):
        return CredentialProbeStatus.UNAVAILABLE
    except Exception:
        # Provider libraries and system TLS stacks can raise implementation-
        # specific errors.  Their text may contain proxy or credential data,
        # so diagnostics deliberately collapse them to one safe state.
        return CredentialProbeStatus.UNAVAILABLE
    if status in {401, 403}:
        return CredentialProbeStatus.AUTH_REJECTED
    if 200 <= status < 300:
        return CredentialProbeStatus.CONFIGURED
    return CredentialProbeStatus.UNAVAILABLE


def _ai_provider_checks(runtime: RuntimeLayout, config: AppConfig) -> list[Check]:
    provider = config.ai.provider
    checks = [Check("AI provider", "ok", f"{provider} · {config.ai.model}")]
    if provider == "mock":
        checks.append(Check(
            "Локальный тестовый режим",
            "warn",
            "Используется mock-провайдер; этот режим не подтверждает готовность реальной AI-обработки.",
            "Для реальной обработки выберите OpenAI или Gemini и подтвердите рабочий API-ключ.",
        ))
        return checks

    from app.secure_secrets import api_key_state

    label = "OpenAI API key" if provider == "openai" else "Gemini API key"
    state = api_key_state(
        provider,
        runtime.data,
        probe=lambda secret: _probe_api_credential(provider, secret),
    )
    if state == CredentialProbeStatus.CONFIGURED:
        checks.append(Check(label, "ok", "Ключ подтверждён провайдером; значение скрыто."))
    elif state == CredentialProbeStatus.AUTH_REJECTED:
        checks.append(Check(
            label,
            "error",
            "Провайдер отклонил ключ (401/403); реальная AI-обработка недоступна.",
            "Замените ключ в настройках и повторите проверку либо явно включите локальный тестовый режим.",
        ))
    elif state == "invalid":
        checks.append(Check(
            label,
            "error",
            "Ключ найден, но не прошёл локальную проверку формата; реальная AI-обработка недоступна.",
            "Введите корректный ключ в настройках либо явно включите локальный тестовый режим.",
        ))
    elif state == CredentialProbeStatus.UNAVAILABLE:
        checks.append(Check(
            label,
            "warn",
            "Ключ настроен, но провайдер временно недоступен; подтвердить ключ сейчас не удалось.",
            "Повторите проверку позже; при необходимости явно включите локальный тестовый режим.",
        ))
    else:
        checks.append(Check(
            label,
            "error",
            "Ключ не настроен; реальная AI-обработка недоступна.",
            "Добавьте ключ в настройках либо явно включите локальный тестовый режим.",
        ))
    return checks


def collect_checks(root: Path | RuntimeLayout, config: AppConfig | None = None) -> list[Check]:
    runtime = _layout(root)
    config = config or AppConfig()
    if runtime.frozen:
        runtime_check = Check("Runtime", "ok", "Встроенный runtime portable-сборки доступен.")
    else:
        runtime_check = Check(
            "Python",
            "ok" if sys.version_info >= (3, 11) else "error",
            f"{sys.version.split()[0]} (требуется 3.11+)",
            "Установите Python 3.11+ или используйте portable-сборку."
            if sys.version_info < (3, 11) else "",
        )
    checks = [runtime_check]
    if not runtime.resources.is_dir():
        checks.append(Check(
            "Ресурсы приложения",
            "error",
            "Папка ресурсов не найдена.",
            "Распакуйте portable-сборку полностью и запустите её из новой папки.",
        ))
    else:
        checks.append(Check("Ресурсы приложения", "ok", "Папка ресурсов доступна."))

    checks.extend([
        _tool_check(runtime, "ffmpeg", "FFmpeg", blocking=True),
        _tool_check(runtime, "ffprobe", "FFprobe", blocking=True),
        _tool_check(runtime, "yt-dlp", "yt-dlp", blocking=False),
        _tool_check(runtime, "deno", "Deno", blocking=False),
    ])
    checks.extend(_nvidia_checks())
    checks.append(_cuda_check())

    try:
        has_whisper = importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        has_whisper = False
    checks.append(Check(
        "faster-whisper",
        "ok" if has_whisper else "error",
        "Модуль транскрибации доступен." if has_whisper else "Модуль транскрибации не найден.",
        "Переустановите portable-сборку или зависимости из requirements.txt."
        if not has_whisper else "",
    ))

    checks.extend(_ai_provider_checks(runtime, config))

    if config.tts.enabled:
        tts_provider = config.tts.provider
        checks.append(Check("TTS provider", "ok", f"{tts_provider} · {config.tts.model}"))
        if tts_provider == "openai":
            from app.secure_secrets import api_key_state

            state = api_key_state("openai", runtime.data)
            checks.append(Check(
                "OpenAI TTS API key",
                "ok" if state == "configured" else "warn",
                "Ключ настроен; значение скрыто."
                if state == "configured" else "Ключ для облачной озвучки не настроен.",
                "Добавьте OpenAI-ключ или выберите mock/local TTS."
                if state != "configured" else "",
            ))

    directory_results: list[Check] = []
    for folder in ("input", "work", "output"):
        directory_results.append(_writable_directory_check(runtime.data / folder, f"Папка {folder}"))
    checks.extend(directory_results)

    if not any(item.blocking and item.label.startswith("Папка ") for item in directory_results):
        try:
            free_bytes = shutil.disk_usage(runtime.data).free
        except OSError:
            checks.append(Check(
                "Свободное место",
                "warn",
                "Не удалось определить свободное место.",
                "Проверьте свободное место в папке данных перед обработкой большого видео.",
            ))
        else:
            minimum = 5 * 1024**3
            if free_bytes < minimum:
                checks.append(Check(
                    "Свободное место",
                    "warn",
                    f"Доступно меньше 5 ГБ ({free_bytes / 1024**3:.1f} ГБ).",
                    "Освободите место или выберите другой локальный диск для папки данных.",
                ))
            else:
                checks.append(Check("Свободное место", "ok", f"Доступно {free_bytes / 1024**3:.1f} ГБ."))
    return checks


def summarize_checks(checks: list[Check]) -> DoctorSummary:
    blocking = sum(item.blocking for item in checks)
    warnings = sum(item.warning for item in checks)
    if blocking:
        return DoctorSummary(
            DoctorReadiness.SETUP_REQUIRED,
            "Требуется настройка",
            f"Критических проблем: {blocking}. Откройте диагностику и выполните указанные действия.",
            blocking,
            warnings,
        )
    if warnings:
        return DoctorSummary(
            DoctorReadiness.LIMITED,
            "С ограничениями",
            f"Работа доступна; предупреждений: {warnings}.",
            0,
            warnings,
        )
    return DoctorSummary(DoctorReadiness.READY, "Готов", "Все обязательные проверки пройдены.", 0, 0)


def has_blocking_checks(checks: list[Check]) -> bool:
    return any(item.blocking for item in checks)


def format_report(checks: list[Check]) -> str:
    glyph = {"ok": "OK", "warn": "WARNING", "error": "BLOCKING"}
    lines = ["Content Factory — проверка окружения", ""]
    for check in checks:
        lines.append(f"{glyph[check.status]} {check.label}: {check.detail}")
        if check.action:
            lines.append(f"  Что сделать: {check.action}")
    summary = summarize_checks(checks)
    lines.extend([
        "",
        f"Итог: {summary.title}. Критических проблем — {summary.blocking_count}, "
        f"предупреждений — {summary.warning_count}.",
        "WARNING не блокирует работу; BLOCKING нужно исправить до завершения первого запуска.",
    ])
    return "\n".join(lines)


__all__ = [
    "Check", "CredentialProbeStatus", "DoctorReadiness", "DoctorSummary", "collect_checks",
    "format_report", "has_blocking_checks", "summarize_checks",
]
