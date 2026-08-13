from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_PROVIDER_VARIABLES = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


@dataclass(frozen=True, slots=True)
class ApiKeySaveResult:
    saved: bool
    message: str


def _variable(provider: str) -> str | None:
    return _PROVIDER_VARIABLES.get(str(provider).strip().casefold())


def _read_dotenv_value(path: Path, variable: str) -> str | None:
    if not path.is_file():
        return None
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = raw.partition("=")
            if separator and key.strip() == variable:
                return value.strip().strip("'\"")
    except (OSError, UnicodeError):
        return None
    return None


def validate_api_key(provider: str, value: str) -> str | None:
    """Return a safe validation message, never the submitted secret."""

    normalized_provider = str(provider).strip().casefold()
    variable = _variable(normalized_provider)
    if variable is None:
        return "Выбранный AI-провайдер не поддерживает настройку ключа."
    if not isinstance(value, str):
        return "Введите ключ как одну строку."
    secret = value.strip()
    if not 20 <= len(secret) <= 512:
        return "Ключ выглядит слишком коротким или слишком длинным."
    if any(character.isspace() or ord(character) < 32 for character in secret):
        return "Ключ должен быть одной строкой без пробелов."
    expected_prefix = "sk-" if normalized_provider == "openai" else "AIza"
    if not secret.startswith(expected_prefix):
        return f"Ключ не похож на ключ {normalized_provider.title()}. Проверьте выбранный провайдер."
    return None


def api_key_state(
    provider: str,
    root: Path | None = None,
    *,
    probe: Callable[[str], str] | None = None,
) -> str:
    """Return credential state without ever returning the credential itself.

    A caller may provide a bounded probe.  The secret is passed only to that
    in-memory callback; exceptions and unknown results collapse to the safe
    ``unavailable`` state and never escape with provider response details.
    """

    normalized_provider = str(provider).strip().casefold()
    variable = _variable(normalized_provider)
    if variable is None:
        return "not_required" if normalized_provider == "mock" else "missing"
    value = os.getenv(variable)
    if value is None and root is not None:
        value = _read_dotenv_value(Path(root) / ".env", variable)
    if not value:
        return "missing"
    if validate_api_key(normalized_provider, value):
        return "invalid"
    if probe is None:
        return "configured"
    try:
        state = probe(value)
    except Exception:
        return "unavailable"
    return state if state in {"configured", "auth_rejected", "unavailable"} else "unavailable"


def key_configured(provider: str, root: Path | None = None) -> bool:
    """Expose only a boolean; secret values never enter settings or UI state."""

    return api_key_state(provider, root) == "configured"


def load_runtime_secrets(root: Path) -> None:
    """Load supported local secrets into this process without logging values."""

    path = Path(root) / ".env"
    for variable in _PROVIDER_VARIABLES.values():
        if variable in os.environ:
            continue
        value = _read_dotenv_value(path, variable)
        if value:
            os.environ[variable] = value


def save_api_key(provider: str, value: str, root: Path) -> ApiKeySaveResult:
    """Atomically persist one supported key in runtime data, never in settings."""

    normalized_provider = str(provider).strip().casefold()
    variable = _variable(normalized_provider)
    error = validate_api_key(normalized_provider, value)
    if variable is None or error:
        return ApiKeySaveResult(False, error or "Провайдер не поддерживается.")
    secret = value.strip()
    directory = Path(root)
    path = directory / ".env"
    temporary_path: Path | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        output: list[str] = []
        replaced = False
        for raw in existing:
            key, separator, _old_value = raw.partition("=")
            if separator and key.strip() == variable:
                if not replaced:
                    output.append(f"{variable}={secret}")
                    replaced = True
                continue
            output.append(raw)
        if not replaced:
            output.append(f"{variable}={secret}")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".env.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write("\n".join(output) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary_path, path)
        os.environ[variable] = secret
    except (OSError, UnicodeError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return ApiKeySaveResult(False, "Не удалось безопасно сохранить ключ в локальной папке данных.")
    return ApiKeySaveResult(True, "Ключ сохранён локально. Значение скрыто и не попадёт в настройки или логи.")


__all__ = [
    "ApiKeySaveResult", "api_key_state", "key_configured", "load_runtime_secrets",
    "save_api_key", "validate_api_key",
]
