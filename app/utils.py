from __future__ import annotations

import hashlib
import json
import errno
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# A short, bounded retry window is enough for the usual Windows Defender,
# Explorer-preview, and just-closed-reader sharing violations.  It keeps a
# state save from becoming an unbounded stall while making the normal race
# invisible to callers.
ATOMIC_REPLACE_BACKOFF_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)
_TRANSIENT_WINDOWS_ERRORS = {5, 32, 33}  # access denied, sharing, lock violation
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class AtomicWriteError(OSError):
    """An atomic JSON replacement failed after its bounded retry budget.

    ``fallback_path`` points to an independently-written diagnostic state
    snapshot when that write succeeded.  The old destination is intentionally
    left untouched in every failure path.
    """

    def __init__(
        self,
        path: Path,
        cause: OSError,
        *,
        attempts: int,
        fallback_path: Path | None = None,
    ) -> None:
        self.path = path
        self.cause = cause
        self.attempts = attempts
        self.fallback_path = fallback_path
        fallback = f"; diagnostic fallback: {fallback_path}" if fallback_path else ""
        super().__init__(
            f"Could not atomically replace {path} after {attempts} attempts: {cause}{fallback}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str, fallback: str = "source") -> str:
    stem = Path(value).stem if value else fallback
    cleaned = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", stem).strip("._-")
    return (cleaned or fallback)[:80]


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    """Atomically write JSON without exposing partial content to readers.

    The replace itself is protected by a process/thread lock and retried for
    short-lived Windows sharing violations.  A failed replace never removes
    the previous file; a diagnostic fallback is saved under a unique name so
    an otherwise-successful production run remains recoverable.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    attempts = 0
    last_error: OSError | None = None

    try:
        with _exclusive_path_write_lock(path):
            for attempt in range(len(ATOMIC_REPLACE_BACKOFF_SECONDS) + 1):
                attempts = attempt + 1
                temporary = _write_json_temporary(path, payload)
                try:
                    os.replace(temporary, path)
                    return
                except OSError as error:
                    last_error = error
                    _remove_temporary(temporary)
                    if not _is_transient_replace_error(error) or attempt == len(ATOMIC_REPLACE_BACKOFF_SECONDS):
                        break
                    time.sleep(ATOMIC_REPLACE_BACKOFF_SECONDS[attempt])
    except OSError as error:
        last_error = error
        attempts = max(attempts, 1)

    assert last_error is not None
    fallback_path = _write_json_fallback(path, data, last_error)
    raise AtomicWriteError(path, last_error, attempts=attempts, fallback_path=fallback_path) from last_error


def _write_json_temporary(path: Path, payload: str) -> Path:
    """Write one complete, durable replacement candidate beside ``path``."""

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False,
        prefix=f".{path.name}.", suffix=".tmp",
    ) as file:
        file.write(payload)
        file.flush()
        _fsync_if_supported(file)
        return Path(file.name)


def _write_json_fallback(path: Path, data: Any, error: OSError) -> Path | None:
    """Persist a distinct state snapshot when replacing the canonical file fails."""

    fallback = path.with_name(f"{path.name}.fallback-{uuid.uuid4().hex}.json")
    diagnostic: Any
    if isinstance(data, dict):
        diagnostic = dict(data)
        diagnostic["state_persistence"] = {
            "status": "degraded",
            "target_path": str(path),
            "error_type": type(error).__name__,
            "error": str(error),
            "winerror": getattr(error, "winerror", None),
            "saved_at": utc_now(),
        }
    else:
        diagnostic = {
            "data": data,
            "state_persistence": {
                "status": "degraded",
                "target_path": str(path),
                "error_type": type(error).__name__,
                "error": str(error),
                "winerror": getattr(error, "winerror", None),
                "saved_at": utc_now(),
            },
        }
    try:
        payload = json.dumps(diagnostic, ensure_ascii=False, indent=2)
        with fallback.open("x", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            _fsync_if_supported(file)
        return fallback
    except OSError:
        return None


def _fsync_if_supported(file: Any) -> None:
    try:
        os.fsync(file.fileno())
    except OSError as error:
        # Some virtual and network-backed filesystems do not implement fsync.
        # Other errors (for example ENOSPC) must abort the replacement rather
        # than claiming that a durable state snapshot was written.
        if error.errno in {errno.EINVAL, errno.ENOTSUP}:
            return
        raise


def _remove_temporary(path: Path) -> None:
    for attempt in range(len(ATOMIC_REPLACE_BACKOFF_SECONDS) + 1):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            if attempt == len(ATOMIC_REPLACE_BACKOFF_SECONDS):
                return
            time.sleep(ATOMIC_REPLACE_BACKOFF_SECONDS[attempt])


def _is_transient_replace_error(error: OSError) -> bool:
    if isinstance(error, PermissionError):
        return True
    if getattr(error, "winerror", None) in _TRANSIENT_WINDOWS_ERRORS:
        return True
    return getattr(error, "errno", None) in {errno.EACCES, errno.EPERM, errno.EBUSY}


def _thread_lock_for(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.resolve()))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_path_write_lock(path: Path):
    """Serialise JSON replacements across threads and local processes."""

    thread_lock = _thread_lock_for(path)
    thread_lock.acquire()
    lock_file = path.with_name(f".{path.name}.write.lock")
    handle = None
    try:
        handle = lock_file.open("a+b")
        _acquire_process_lock(handle)
        yield
    finally:
        if handle is not None:
            _release_process_lock(handle)
            handle.close()
        thread_lock.release()


def _acquire_process_lock(handle: Any) -> None:
    last_error: OSError | None = None
    for attempt in range(len(ATOMIC_REPLACE_BACKOFF_SECONDS) + 1):
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as error:
            last_error = error
            if attempt == len(ATOMIC_REPLACE_BACKOFF_SECONDS):
                raise
            time.sleep(ATOMIC_REPLACE_BACKOFF_SECONDS[attempt])
    assert last_error is not None
    raise last_error


def _release_process_lock(handle: Any) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write binary artifacts without exposing a partial audio file to cache readers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as file:
        file.write(data)
        file.flush()
        temporary = Path(file.name)
    temporary.replace(path)


def format_seconds(value: float | None) -> str:
    if value is None:
        return "н/д"
    minutes, seconds = divmod(max(0, value), 60)
    return f"{int(minutes)} мин {seconds:04.1f} с" if minutes else f"{seconds:.1f} с"
