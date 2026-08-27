from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import uuid

from PySide6.QtCore import QLockFile

from app.utils import safe_name, stable_text_hash


PREVIEW_PROXY_FORMAT_VERSION = "h264-30fps-source-v5"
PROXY_LEASE_STALE_SECONDS = 45.0


@dataclass(frozen=True, slots=True)
class SourceRevision:
    key: str
    resolved_path: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class PreviewProxyLease:
    owner_id: str
    lock_path: Path
    temporary: Path
    lock_file: QLockFile = field(repr=False, compare=False)


def source_revision(source_path: Path) -> SourceRevision:
    source = source_path.expanduser().resolve()
    stat = source.stat()
    identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    return SourceRevision(
        key=stable_text_hash(f"{PREVIEW_PROXY_FORMAT_VERSION}:{identity}"),
        resolved_path=str(source),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def preview_proxy_path(cache_directory: Path, source_path: Path) -> Path:
    try:
        revision = source_revision(source_path)
        digest = revision.key[:20]
    except OSError:
        digest = stable_text_hash(
            f"{PREVIEW_PROXY_FORMAT_VERSION}:{source_path}"
        )[:20]
    return cache_directory / f"{safe_name(source_path.stem, 'source')}-{digest}.mp4"


def preview_proxy_temporary_path(destination: Path, owner_id: str | None = None) -> Path:
    owner_suffix = f".{owner_id}" if owner_id else ""
    return destination.with_name(f"{destination.stem}{owner_suffix}.part{destination.suffix}")


def preview_proxy_manifest_path(destination: Path) -> Path:
    return destination.with_suffix(f"{destination.suffix}.manifest.json")


def preview_proxy_lock_path(destination: Path) -> Path:
    return destination.with_suffix(f"{destination.suffix}.lock")


def acquire_preview_proxy_lease(
    destination: Path, source_path: Path,
) -> PreviewProxyLease | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resolve the source before claiming the cache identity. A missing or
    # changed source must fail without leaving an orphaned owner record.
    source_revision(source_path)
    owner_id = uuid.uuid4().hex
    lock_path = preview_proxy_lock_path(destination)
    temporary = preview_proxy_temporary_path(destination, owner_id)
    lock_file = QLockFile(str(lock_path))
    lock_file.setStaleLockTime(round(PROXY_LEASE_STALE_SECONDS * 1000))
    if not lock_file.tryLock(0):
        return None
    return PreviewProxyLease(owner_id, lock_path, temporary, lock_file)


def owns_preview_proxy_lease(lease: PreviewProxyLease) -> bool:
    return lease.lock_file.isLocked()


def refresh_preview_proxy_lease(lease: PreviewProxyLease) -> bool:
    # QLockFile keeps the OS/process-aware ownership for the lease lifetime;
    # there is no heartbeat file to race with another process.
    return owns_preview_proxy_lease(lease)


def release_preview_proxy_lease(lease: PreviewProxyLease) -> None:
    if owns_preview_proxy_lease(lease):
        lease.lock_file.unlock()


def reclaim_stale_preview_proxy_lease(
    destination: Path, *, stale_after_seconds: float = PROXY_LEASE_STALE_SECONDS,
) -> bool:
    lock_path = preview_proxy_lock_path(destination)
    lock_file = QLockFile(str(lock_path))
    lock_file.setStaleLockTime(round(max(0.0, stale_after_seconds) * 1000))
    return lock_file.removeStaleLockFile()


def write_preview_proxy_manifest(
    destination: Path, source_path: Path, probe: dict[str, object],
) -> None:
    revision = source_revision(source_path)
    proxy_stat = destination.stat()
    manifest = {
        "format_version": PREVIEW_PROXY_FORMAT_VERSION,
        "source_revision": revision.key,
        "source_size_bytes": revision.size_bytes,
        "source_mtime_ns": revision.mtime_ns,
        "proxy_size_bytes": proxy_stat.st_size,
        "proxy_mtime_ns": proxy_stat.st_mtime_ns,
        "probe": probe,
    }
    target = preview_proxy_manifest_path(destination)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def validated_preview_proxy(
    destination: Path, source_path: Path, *, required_end_seconds: float = 0.0,
) -> bool:
    try:
        proxy_stat = destination.stat()
        manifest = json.loads(
            preview_proxy_manifest_path(destination).read_text(encoding="utf-8")
        )
        revision = source_revision(source_path)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(manifest, dict) or proxy_stat.st_size <= 0:
        return False
    if manifest.get("format_version") != PREVIEW_PROXY_FORMAT_VERSION:
        return False
    if manifest.get("source_revision") != revision.key:
        return False
    if manifest.get("proxy_size_bytes") != proxy_stat.st_size:
        return False
    if manifest.get("proxy_mtime_ns") != proxy_stat.st_mtime_ns:
        return False
    probe = manifest.get("probe")
    if not isinstance(probe, dict):
        return False
    try:
        width = int(probe.get("width") or 0)
        height = int(probe.get("height") or 0)
        duration = float(probe.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        return False
    if probe.get("video_codec") != "h264" or width <= 0 or height != 480:
        return False
    return duration > 0.0 and duration + 0.25 >= max(0.0, required_end_seconds)
