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
from enum import Enum
from pathlib import Path
from typing import Callable, TextIO
from urllib.parse import urlparse, urlsplit, urlunsplit

from app.errors import DependencyError, SourceError
from app.subprocess_utils import UTF8_REPLACE_TEXT


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
    format: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    fraction: float | None
    speed: str | None
    eta_seconds: int | None
    downloaded: str | None = None
    total: str | None = None


class DownloadCancelled(SourceError):
    """The user intentionally stopped a source download."""


ProgressCallback = Callable[[DownloadProgress], None]

# The first ``download:`` selects yt-dlp's download-progress output type; the
# second is our stable, parseable marker. Keeping the marker explicit avoids
# confusing an output-type prefix with user-facing progress data.
YTDLP_DOWNLOAD_PROGRESS_TEMPLATE = (
    "download:download:%(progress._percent_str)s|%(progress._downloaded_bytes_str)s|"
    "%(progress._total_bytes_str)s|%(progress._total_bytes_estimate_str)s|"
    "%(progress._speed_str)s|%(progress._eta_str)s"
)
YTDLP_OUTPUT_TEMPLATE = "%(title).120B-%(id)s.%(ext)s"


class YtDlpFailureReason(str, Enum):
    LOGIN_REQUIRED = "login_required"
    BOT_CHECK = "bot_check_or_rate_limit"
    JS_RUNTIME_MISSING = "js_runtime_missing"
    PO_TOKEN_REQUIRED = "po_token_required"
    UNAVAILABLE = "video_unavailable"
    UNSUPPORTED = "unsupported_video"
    PROTECTED = "protected_video"
    UNKNOWN = "unknown"


_FAILURE_MESSAGES = {
    YtDlpFailureReason.LOGIN_REQUIRED: (
        "Видео требует входа или имеет ограниченный доступ. Friend Beta загружает только публичные видео, "
        "которые открываются без авторизации."
    ),
    YtDlpFailureReason.BOT_CHECK: (
        "Сайт отклонил запрос из-за проверки на автоматические запросы или ограничения частоты. "
        "Подождите и повторите позже либо попробуйте другую сеть."
    ),
    YtDlpFailureReason.JS_RUNTIME_MISSING: (
        "Для этой публичной YouTube-ссылки нужен JavaScript runtime Deno, но он не найден. "
        "Переустановите или обновите portable-сборку Content Factory."
    ),
    YtDlpFailureReason.PO_TOKEN_REQUIRED: (
        "YouTube требует PO Token для этого видео. Friend Beta пока не поддерживает такой способ доступа; "
        "нужна отдельная совместимая интеграция."
    ),
    YtDlpFailureReason.UNAVAILABLE: (
        "Видео по этой ссылке недоступно или удалено. Проверьте ссылку и доступность ролика без входа."
    ),
    YtDlpFailureReason.UNSUPPORTED: (
        "Эта ссылка или доступный на ней видеоформат пока не поддерживается. Попробуйте другую публичную ссылку."
    ),
    YtDlpFailureReason.PROTECTED: (
        "Видео защищено и не поддерживается. Friend Beta не обходит DRM, авторизацию или другие ограничения доступа."
    ),
    YtDlpFailureReason.UNKNOWN: (
        "Не удалось получить видео по этой ссылке. Проверьте, что она публичная и открывается без входа."
    ),
}


@dataclass(frozen=True, slots=True)
class YtDlpCapabilities:
    """Resolved executables and deliberately unsupported access mechanisms."""

    executable: str | None
    deno_executable: str | None
    reads_browser_cookies: bool = False
    po_token_provider: bool = False

    @property
    def available(self) -> bool:
        return bool(self.executable)

    @property
    def javascript_runtime_available(self) -> bool:
        return bool(self.deno_executable)

    def common_arguments(self) -> list[str]:
        # Ignoring ambient yt-dlp config is intentional: a user-level config
        # must not silently opt this public-only application into browser
        # cookies, authentication, a downloader, or a format/remux override.
        arguments = ["--ignore-config", "--no-playlist"]
        if self.deno_executable:
            arguments.extend(["--js-runtimes", f"deno:{self.deno_executable}"])
        return arguments


class YtDlpSourceError(SourceError):
    """Safe user text plus captured yt-dlp diagnostics for logs and tests."""

    def __init__(self, reason: YtDlpFailureReason, diagnostics: str) -> None:
        self.reason = reason
        self.diagnostics = diagnostics
        super().__init__(_FAILURE_MESSAGES[reason])


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


def find_deno_executable(ytdlp_executable: str | None = None) -> str | None:
    """Find Deno beside yt-dlp first, then in the active process environment."""

    resolved_ytdlp = shutil.which(ytdlp_executable) if ytdlp_executable else None
    executable_path = Path(resolved_ytdlp or ytdlp_executable or "")
    if executable_path.name:
        sibling = executable_path.expanduser().resolve().parent / (
            "deno.exe" if sys.platform == "win32" else "deno"
        )
        if sibling.is_file():
            return str(sibling)
    from_path = shutil.which("deno")
    if from_path:
        return from_path
    scripts = Path(sys.executable).resolve().parent
    for name in (("deno.exe", "deno") if sys.platform == "win32" else ("deno", "deno.exe")):
        candidate = scripts / name
        if candidate.is_file():
            return str(candidate)
    return None


def detect_ytdlp_capabilities(executable: str | None = None) -> YtDlpCapabilities:
    resolved = executable or find_ytdlp_executable()
    return YtDlpCapabilities(
        executable=resolved,
        deno_executable=find_deno_executable(resolved),
    )


def build_ytdlp_inspect_arguments(capabilities: YtDlpCapabilities, url: str) -> list[str]:
    """Return the canonical metadata command arguments (program excluded)."""

    return [
        *capabilities.common_arguments(),
        "--skip-download",
        "--dump-single-json",
        url,
    ]


def build_ytdlp_download_arguments(
    capabilities: YtDlpCapabilities, url: str, destination: Path,
) -> list[str]:
    """Return the canonical download arguments without changing yt-dlp format selection."""

    output_template = str(destination / YTDLP_OUTPUT_TEMPLATE)
    return [
        *capabilities.common_arguments(),
        "--newline",
        "--no-colors",
        "--no-overwrites",
        "--progress",
        "--progress-template",
        YTDLP_DOWNLOAD_PROGRESS_TEMPLATE,
        "--print",
        "after_move:filepath",
        "-o",
        output_template,
        url,
    ]


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


def sanitize_public_url_for_diagnostics(value: str) -> str:
    """Keep an identifiable public URL without persisting query secrets."""

    parsed = urlsplit(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_DIAGNOSTIC_SECRET = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+|cookie\s*:\s*|(?:api[_-]?key|token)\s*[=:]\s*)[^\s;]+"
)
_DIAGNOSTIC_COOKIE = re.compile(r"(?i)((?:set-)?cookie\s*:\s*).*$")
_DIAGNOSTIC_URL_QUERY = re.compile(r"(?i)(https?://[^\s?#]+(?:/[^\s?#]*)?)\?[^\s#]+")


def normalize_ytdlp_diagnostics(*outputs: str, limit: int = 16_000) -> str:
    """Keep actionable stderr while bounding and redacting diagnostic storage."""

    lines: list[str] = []
    for output in outputs:
        for raw_line in str(output or "").splitlines():
            line = _ANSI_ESCAPE.sub("", raw_line).strip()
            if line:
                line = _DIAGNOSTIC_COOKIE.sub(r"\1[redacted]", line)
                line = _DIAGNOSTIC_URL_QUERY.sub(r"\1", line)
                lines.append(_DIAGNOSTIC_SECRET.sub(r"\1[redacted]", line))
    return "\n".join(lines)[-limit:]


def classify_ytdlp_failure(output: str) -> YtDlpSourceError:
    """Classify yt-dlp evidence before generic login wording can mask the cause."""

    diagnostics = normalize_ytdlp_diagnostics(output)
    message = diagnostics.casefold()
    patterns = (
        (YtDlpFailureReason.BOT_CHECK, (
            "confirm you’re not a bot", "confirm you're not a bot", "confirm you are not a bot",
            "unusual traffic", "too many requests", "http error 429", "status code 429", "rate limit",
        )),
        (YtDlpFailureReason.PO_TOKEN_REQUIRED, (
            "po token", "po_token", "proof of origin token", "proof-of-origin token",
        )),
        (YtDlpFailureReason.JS_RUNTIME_MISSING, (
            "no supported javascript runtime", "javascript runtime could not be found",
            "javascript runtime is not available", "external javascript runtime",
            "challenge solver was not found", "no usable javascript runtime",
        )),
        (YtDlpFailureReason.LOGIN_REQUIRED, (
            "private video", "private content", "login required", "sign in required",
            "members-only", "members only", "join this channel", "age-restricted",
            "authentication required", "requires authentication", "not a public video",
        )),
        (YtDlpFailureReason.PROTECTED, (
            "copyright-protected", "protected content", "drm protected", "this video is drm",
        )),
        (YtDlpFailureReason.UNSUPPORTED, (
            "unsupported url", "unsupported site", "no suitable formats", "no video formats found",
        )),
        (YtDlpFailureReason.UNAVAILABLE, (
            "video unavailable", "video is unavailable", "has been removed", "does not exist",
            "not available in your country", "not available in your region", "geo-restricted",
        )),
    )
    reason = next(
        (candidate for candidate, markers in patterns if any(marker in message for marker in markers)),
        YtDlpFailureReason.UNKNOWN,
    )
    return YtDlpSourceError(reason, diagnostics)


def describe_public_url_failure(output: str) -> str:
    """Compatibility wrapper returning only safe user-facing text."""

    return str(classify_ytdlp_failure(output))


class YtDlpSource:
    """A process-only adapter. It never reads browser cookies or invokes a shell."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        capabilities: YtDlpCapabilities | None = None,
    ) -> None:
        self.capabilities = capabilities or detect_ytdlp_capabilities(executable)
        self.executable = self.capabilities.executable
        self.last_diagnostics = ""

    @property
    def available(self) -> bool:
        return bool(self.executable)

    def inspect(self, url: str) -> URLMetadata:
        safe_url = validate_public_video_url(url)
        executable = self._require_executable()
        command = [executable, *build_ytdlp_inspect_arguments(self.capabilities, safe_url)]
        self.last_diagnostics = ""
        try:
            result = subprocess.run(command, capture_output=True, timeout=90, check=False, **UTF8_REPLACE_TEXT)
            stdout = str(getattr(result, "stdout", "") or "")
            stderr = str(getattr(result, "stderr", "") or "")
            self.last_diagnostics = normalize_ytdlp_diagnostics(stderr)
            if int(getattr(result, "returncode", 0)) != 0:
                raise classify_ytdlp_failure(self.last_diagnostics or stdout)
            try:
                return parse_url_metadata(safe_url, stdout)
            except SourceError as error:
                details = normalize_ytdlp_diagnostics(
                    self.last_diagnostics,
                    "yt-dlp returned invalid metadata JSON on stdout.",
                )
                self.last_diagnostics = details
                raise YtDlpSourceError(YtDlpFailureReason.UNKNOWN, details) from error
        except subprocess.TimeoutExpired as error:
            raise SourceError("Не удалось получить информацию о видео: запрос занял слишком много времени.") from error
        except YtDlpSourceError:
            raise
        except OSError as error:
            raise DependencyError("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.") from error

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
        command = [executable, *build_ytdlp_download_arguments(self.capabilities, safe_url, destination)]
        self.last_diagnostics = ""
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **UTF8_REPLACE_TEXT,
            )
        except OSError as error:
            raise DependencyError("Для загрузки по ссылке требуется дополнительный компонент yt-dlp.") from error
        paths: list[Path] = []
        output: list[str] = []
        diagnostic_output: list[str] = []
        stderr = getattr(process, "stderr", None)
        diagnostic_reader = (
            threading.Thread(
                target=_drain_diagnostics,
                args=(stderr, diagnostic_output),
                name="yt-dlp-diagnostics",
                daemon=True,
            )
            if stderr is not None else None
        )
        if diagnostic_reader:
            diagnostic_reader.start()
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
            if diagnostic_reader:
                diagnostic_reader.join(timeout=10)
            if stderr:
                stderr.close()
            self.last_diagnostics = normalize_ytdlp_diagnostics(*diagnostic_output)
        if cancel_event and cancel_event.is_set():
            cleanup_partial_downloads(destination)
            raise DownloadCancelled("Загрузка видео отменена.")
        if exit_code != 0:
            cleanup_partial_downloads(destination)
            raise classify_ytdlp_failure(self.last_diagnostics or "\n".join(output))
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


def _drain_diagnostics(stream: TextIO, target: list[str]) -> None:
    try:
        for chunk in stream:
            target.append(chunk)
    except (OSError, ValueError):
        return


_PROGRESS_PREFIX = re.compile(r"^download:\s*(?P<percent>[\d.]+)%\|(?P<values>.*)$")


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
        format=str(raw["ext"]).upper() if raw.get("ext") else None,
    )


def parse_download_progress(line: str) -> DownloadProgress | None:
    matched = _PROGRESS_PREFIX.match(line)
    if not matched:
        return None
    try:
        fraction = max(0.0, min(1.0, float(matched.group("percent")) / 100.0))
    except ValueError:
        fraction = None
    values = matched.group("values").split("|")
    if len(values) == 2:
        # Kept for yt-dlp output recorded by earlier app versions and for
        # third-party wrappers that only emit speed and ETA.
        speed, eta = values
        return DownloadProgress(fraction, _progress_value(speed), _eta_seconds(eta.strip()))
    if len(values) != 5:
        return None
    downloaded, total, estimated_total, speed, eta = values
    return DownloadProgress(
        fraction,
        _progress_value(speed),
        _eta_seconds(eta.strip()),
        downloaded=_progress_value(downloaded),
        total=_progress_value(total) or _progress_value(estimated_total),
    )


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


def _progress_value(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned if cleaned and cleaned.upper() not in {"NA", "N/A", "UNKNOWN"} else None


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
