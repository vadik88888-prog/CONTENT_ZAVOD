"""Offline, device-bound Friend Beta licensing.

The desktop client contains only ``PUBLIC_VERIFICATION_KEY_B64``.  The
separate administrator tool signs canonical JSON payloads with the matching
private key, which is deliberately kept outside the repository and package.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import secrets
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LICENSE_SCHEMA_VERSION = 1
DEVICE_CODE_PREFIX = "CFB1"
DEVICE_IDENTITY_FILE = "device.identity"
LICENSE_FILE = "friend-beta-license.json"

# This is intentionally a verification key, not a signing key.
PUBLIC_VERIFICATION_KEY_B64 = "5B/57yqSumkvzWeMp2NcbUfkhddsdsBQOPu2jrl64Ks="


class LicensingError(ValueError):
    """A safe, user-presentable activation error."""


@dataclass(frozen=True, slots=True)
class LicenseStatus:
    active: bool
    code: str
    message: str
    expires_at: datetime | None = None


# Minimal RFC 8032 Ed25519 implementation.  Keeping it here avoids putting a
# heavy crypto dependency into the portable beta solely for signature verify.
_Q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _Q - 2, _Q)) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)
_B_Y = (4 * pow(5, _Q - 2, _Q)) % _Q


def _x_recover(y: int) -> int:
    x2 = (y * y - 1) * pow(_D * y * y + 1, _Q - 2, _Q) % _Q
    x = pow(x2, (_Q + 3) // 8, _Q)
    if (x * x - x2) % _Q:
        x = x * _I % _Q
    if x & 1:
        x = _Q - x
    return x


_B = (_x_recover(_B_Y), _B_Y)


def _edwards(point_a: tuple[int, int], point_b: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = point_a
    x2, y2 = point_b
    denominator = _D * x1 * x2 * y1 * y2 % _Q
    return (
        (x1 * y2 + x2 * y1) * pow(1 + denominator, _Q - 2, _Q) % _Q,
        (y1 * y2 + x1 * x2) * pow(1 - denominator, _Q - 2, _Q) % _Q,
    )


def _scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = (0, 1)
    while scalar:
        if scalar & 1:
            result = _edwards(result, point)
        point = _edwards(point, point)
        scalar >>= 1
    return result


def _encode_point(point: tuple[int, int]) -> bytes:
    x, y = point
    encoded = bytearray(y.to_bytes(32, "little"))
    encoded[31] |= (x & 1) << 7
    return bytes(encoded)


def _decode_point(value: bytes) -> tuple[int, int]:
    if len(value) != 32:
        raise LicensingError("Ключ лицензии имеет неверный формат.")
    encoded = bytearray(value)
    sign = encoded[31] >> 7
    encoded[31] &= 0x7F
    y = int.from_bytes(encoded, "little")
    if y >= _Q:
        raise LicensingError("Ключ лицензии имеет неверный формат.")
    x = _x_recover(y)
    if x & 1 != sign:
        x = _Q - x
    point = (x, y)
    if _encode_point(point) != value:
        raise LicensingError("Ключ лицензии имеет неверный формат.")
    return point


def _sha512_mod_l(value: bytes) -> int:
    return int.from_bytes(hashlib.sha512(value).digest(), "little") % _L


def public_key_from_seed(seed: bytes) -> bytes:
    if len(seed) != 32:
        raise LicensingError("Приватный ключ имеет неверный формат.")
    hashed = bytearray(hashlib.sha512(seed).digest()[:32])
    hashed[0] &= 248
    hashed[31] &= 63
    hashed[31] |= 64
    return _encode_point(_scalar_mult(_B, int.from_bytes(hashed, "little")))


def sign_message(seed: bytes, message: bytes) -> bytes:
    if len(seed) != 32:
        raise LicensingError("Приватный ключ имеет неверный формат.")
    digest = hashlib.sha512(seed).digest()
    scalar_bytes = bytearray(digest[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    public_key = public_key_from_seed(seed)
    nonce = _sha512_mod_l(digest[32:] + message)
    encoded_nonce_point = _encode_point(_scalar_mult(_B, nonce))
    challenge = _sha512_mod_l(encoded_nonce_point + public_key + message)
    response = (nonce + challenge * scalar) % _L
    return encoded_nonce_point + response.to_bytes(32, "little")


def verify_message(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    try:
        public_point = _decode_point(public_key)
        nonce_point = _decode_point(signature[:32])
    except LicensingError:
        return False
    response = int.from_bytes(signature[32:], "little")
    if response >= _L:
        return False
    challenge = _sha512_mod_l(signature[:32] + public_key + message)
    # Cofactor multiplication rejects small-order points under the RFC 8032
    # verification equation while preserving valid Ed25519 signatures.
    left = _scalar_mult(_scalar_mult(_B, response), 8)
    right = _scalar_mult(_edwards(nonce_point, _scalar_mult(public_point, challenge)), 8)
    return hmac.compare_digest(_encode_point(left), _encode_point(right))


def generate_signing_seed() -> bytes:
    return secrets.token_bytes(32)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalize_device_code(value: str) -> str:
    normalized = "".join(str(value).strip().upper().split())
    chunks = normalized.split("-")
    if len(chunks) != 5 or chunks[0] != DEVICE_CODE_PREFIX or any(
        len(chunk) != 13 or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for character in chunk)
        for chunk in chunks[1:]
    ):
        raise LicensingError("Код устройства имеет неверный формат.")
    compact = "".join(chunks[1:])
    raw = base64.b32decode(compact + "=" * (-len(compact) % 8), casefold=True)
    if len(raw) != 32:
        raise LicensingError("Код устройства имеет неверный формат.")
    return normalized


def device_code_from_secret(secret: bytes) -> str:
    if len(secret) != 32:
        raise LicensingError("Идентификатор устройства повреждён.")
    encoded = base64.b32encode(secret).decode("ascii").rstrip("=")
    return DEVICE_CODE_PREFIX + "-" + "-".join(encoded[index:index + 13] for index in range(0, len(encoded), 13))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_expiration(value: object) -> datetime:
    if not isinstance(value, str):
        raise LicensingError("В лицензии отсутствует срок действия.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LicensingError("Срок действия лицензии имеет неверный формат.") from error
    if parsed.tzinfo is None:
        raise LicensingError("Срок действия лицензии должен содержать часовой пояс.")
    return parsed.astimezone(timezone.utc)


def _protected_blob(data: bytes) -> bytes:
    if sys.platform != "win32":
        return data

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_byte))]

    source_buffer = ctypes.create_string_buffer(data)
    source = DATA_BLOB(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
    output = DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    if not crypt32.CryptProtectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
        raise OSError("Windows DPAPI could not protect the device identity.")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def _unprotected_blob(data: bytes) -> bytes:
    if sys.platform != "win32":
        return data

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_byte))]

    source_buffer = ctypes.create_string_buffer(data)
    source = DATA_BLOB(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
    output = DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    if not crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
        raise LicensingError("Не удалось прочитать защищённый идентификатор этого устройства.")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def _write_private_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def admin_key_directory() -> Path:
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return Path(local) / "ContentFactoryAdmin" if local else Path.home() / ".content-factory-admin"


def default_admin_key_path() -> Path:
    return admin_key_directory() / "friend_beta_signing.seed"


def require_processing_license(
    data_directory: Path, *, public_key_b64: str = PUBLIC_VERIFICATION_KEY_B64,
) -> LicenseStatus:
    """The one executable permission boundary for every local processing path."""

    activation = ActivationService(data_directory, public_key_b64=public_key_b64)
    status = activation.status()
    if not status.active:
        raise LicensingError(status.message + " Установите лицензию, чтобы запустить новую обработку.")
    return status


class ActivationService:
    """Owns only activation state; it never touches project data or engine state."""

    def __init__(self, data_directory: Path, *, public_key_b64: str = PUBLIC_VERIFICATION_KEY_B64) -> None:
        self.data_directory = Path(data_directory).expanduser().resolve()
        self.directory = self.data_directory / "activation"
        self.identity_path = self.directory / DEVICE_IDENTITY_FILE
        self.license_path = self.directory / LICENSE_FILE
        self.public_key_b64 = public_key_b64

    @property
    def device_code(self) -> str:
        return device_code_from_secret(self._device_secret())

    def _device_secret(self) -> bytes:
        if self.identity_path.is_file():
            try:
                secret = _unprotected_blob(self.identity_path.read_bytes())
            except (OSError, LicensingError) as error:
                raise LicensingError("Не удалось прочитать идентификатор этого устройства.") from error
            if len(secret) != 32:
                raise LicensingError("Идентификатор этого устройства повреждён.")
            return secret
        secret = secrets.token_bytes(32)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _write_private_file(self.identity_path, _protected_blob(secret))
        except OSError as error:
            raise LicensingError("Не удалось безопасно сохранить идентификатор устройства.") from error
        return secret

    def status(self, *, now: datetime | None = None) -> LicenseStatus:
        try:
            self._device_secret()
        except LicensingError as error:
            return LicenseStatus(False, "device_error", str(error))
        if not self.license_path.is_file():
            return LicenseStatus(False, "missing", "Лицензия для этого устройства ещё не установлена.")
        try:
            raw = json.loads(self.license_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise LicensingError("Файл лицензии повреждён.")
            payload = raw.get("payload")
            signature_text = raw.get("signature")
            if not isinstance(payload, dict) or not isinstance(signature_text, str):
                raise LicensingError("Файл лицензии повреждён.")
            if payload.get("schema_version") != LICENSE_SCHEMA_VERSION:
                raise LicensingError("Версия лицензии не поддерживается.")
            public_key = base64.b64decode(self.public_key_b64, validate=True)
            signature = base64.b64decode(signature_text, validate=True)
            if not verify_message(public_key, _canonical_json(payload), signature):
                raise LicensingError("Подпись лицензии не прошла проверку.")
            if normalize_device_code(str(payload.get("device_code") or "")) != self.device_code:
                return LicenseStatus(False, "wrong_device", "Эта лицензия выпущена для другого устройства.")
            expires_at = _parse_expiration(payload.get("expires_at"))
            if expires_at < (now or _utc_now()):
                return LicenseStatus(False, "expired", "Срок действия лицензии истёк.", expires_at)
            return LicenseStatus(True, "active", "Лицензия активна.", expires_at)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, LicensingError):
            return LicenseStatus(False, "invalid", "Лицензия недействительна или повреждена.")

    def install_license(self, source: Path | str | bytes) -> LicenseStatus:
        try:
            if isinstance(source, bytes):
                raw_bytes = source
            else:
                raw_bytes = Path(source).read_bytes()
            # Validate in memory before replacing a valid local licence.
            candidate = self.directory / ".candidate-license.json"
            self.directory.mkdir(parents=True, exist_ok=True)
            _write_private_file(candidate, raw_bytes)
            previous = self.license_path.read_bytes() if self.license_path.is_file() else None
            os.replace(candidate, self.license_path)
            result = self.status()
            if result.active:
                return result
            if previous is None:
                self.license_path.unlink(missing_ok=True)
            else:
                _write_private_file(self.license_path, previous)
            return result
        except (OSError, TypeError):
            return LicenseStatus(False, "invalid", "Не удалось установить файл лицензии.")

    def require_processing(self) -> None:
        require_processing_license(self.data_directory, public_key_b64=self.public_key_b64)


def create_signed_license(seed: bytes, device_code: str, expires_at: datetime, *, issued_at: datetime | None = None) -> dict[str, Any]:
    if expires_at.tzinfo is None:
        raise LicensingError("Срок лицензии должен содержать часовой пояс.")
    payload = {
        "schema_version": LICENSE_SCHEMA_VERSION,
        "device_code": normalize_device_code(device_code),
        "issued_at": (issued_at or _utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return {
        "payload": payload,
        "signature": base64.b64encode(sign_message(seed, _canonical_json(payload))).decode("ascii"),
    }


__all__ = [
    "ActivationService", "DEVICE_CODE_PREFIX", "LICENSE_FILE", "LicenseStatus", "LicensingError",
    "PUBLIC_VERIFICATION_KEY_B64", "admin_key_directory", "create_signed_license",
    "default_admin_key_path", "device_code_from_secret", "generate_signing_seed", "normalize_device_code",
    "public_key_from_seed", "require_processing_license", "sign_message", "verify_message",
]
