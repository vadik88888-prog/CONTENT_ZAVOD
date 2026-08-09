from __future__ import annotations

"""Atomic, checksum-validated cache for the Phase 7 render dependency graph."""

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.utils import (
    ATOMIC_REPLACE_BACKOFF_SECONDS,
    _exclusive_path_write_lock,
    _is_transient_replace_error,
    stable_file_hash,
    stable_text_hash,
    utc_now,
    write_json,
)


RENDER_CACHE_SCHEMA_VERSION = "7G.render-cache.1"
_NODE_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_KEY_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class CacheArtifact:
    node_id: str
    cache_key: str
    path: Path
    checksum: str
    byte_size: int
    producer_version: str


def runtime_cache_key(
    plan_node_key: str,
    *,
    profile: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> str:
    """Add runtime-only quality/input identity without polluting plan parity."""

    return stable_text_hash(json.dumps({
        "plan_node_key": plan_node_key,
        "profile": dict(profile or {}),
        "inputs": dict(inputs or {}),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class GranularRenderCache:
    """Content-addressed cache whose manifest is the commit marker.

    An artifact without a complete, matching manifest is always a miss.  The
    artifact is replaced first and its manifest second, so interruption can at
    worst cause recomputation; it cannot bless partial or stale bytes.
    """

    def __init__(self, root: Path, *, producer_version: str) -> None:
        self.root = root.resolve()
        self.producer_version = producer_version

    def load(
        self,
        node_id: str,
        cache_key: str,
        *,
        suffix: str,
        dependency_checksums: Mapping[str, str] | None = None,
    ) -> CacheArtifact | None:
        artifact_path, manifest_path = self._paths(node_id, cache_key, suffix)
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        expected_dependencies = dict(sorted((dependency_checksums or {}).items()))
        if not isinstance(manifest, dict) or any((
            manifest.get("schema_version") != RENDER_CACHE_SCHEMA_VERSION,
            manifest.get("status") != "complete",
            manifest.get("node_id") != node_id,
            manifest.get("cache_key") != cache_key,
            manifest.get("producer_version") != self.producer_version,
            manifest.get("dependency_checksums") != expected_dependencies,
            manifest.get("artifact_name") != artifact_path.name,
        )):
            return None
        try:
            byte_size = artifact_path.stat().st_size
            checksum = stable_file_hash(artifact_path)
        except OSError:
            return None
        if byte_size <= 0 or manifest.get("byte_size") != byte_size or manifest.get("checksum") != checksum:
            return None
        return CacheArtifact(
            node_id=node_id,
            cache_key=cache_key,
            path=artifact_path,
            checksum=checksum,
            byte_size=byte_size,
            producer_version=self.producer_version,
        )

    def store_file(
        self,
        node_id: str,
        cache_key: str,
        source: Path,
        *,
        suffix: str,
        dependency_checksums: Mapping[str, str] | None = None,
    ) -> CacheArtifact:
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError("cache source artifact must be a non-empty file")
        artifact_path, manifest_path = self._paths(node_id, cache_key, suffix)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        dependencies = dict(sorted((dependency_checksums or {}).items()))
        # Serialise the bytes under the artifact path.  ``write_json`` then
        # commits the independent manifest marker under its own lock.
        with _exclusive_path_write_lock(artifact_path):
            temporary: Path | None = None
            try:
                with source.open("rb") as incoming, tempfile.NamedTemporaryFile(
                    "wb", dir=artifact_path.parent, delete=False,
                    prefix=f".{artifact_path.name}.", suffix=".tmp",
                ) as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                    temporary = Path(outgoing.name)
                checksum = stable_file_hash(temporary)
                byte_size = temporary.stat().st_size
                if byte_size <= 0:
                    raise ValueError("cache artifact cannot be empty")
                _replace_with_retry(temporary, artifact_path)
                temporary = None
                write_json(manifest_path, {
                    "schema_version": RENDER_CACHE_SCHEMA_VERSION,
                    "status": "complete",
                    "node_id": node_id,
                    "cache_key": cache_key,
                    "producer_version": self.producer_version,
                    "dependency_checksums": dependencies,
                    "artifact_name": artifact_path.name,
                    "checksum": checksum,
                    "byte_size": byte_size,
                    "created_at": utc_now(),
                })
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        result = self.load(
            node_id, cache_key, suffix=suffix,
            dependency_checksums=dependencies,
        )
        if result is None:
            raise OSError(f"cache commit validation failed for {node_id}:{cache_key}")
        return result

    def store_json(
        self,
        node_id: str,
        cache_key: str,
        payload: Mapping[str, Any],
        *,
        dependency_checksums: Mapping[str, str] | None = None,
    ) -> CacheArtifact:
        directory = self.root / ".staging"
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False,
            prefix=f"{node_id}-{cache_key}.", suffix=".json",
        )
        temporary = Path(handle.name)
        try:
            with handle:
                json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            return self.store_file(
                node_id, cache_key, temporary, suffix=".json",
                dependency_checksums=dependency_checksums,
            )
        finally:
            temporary.unlink(missing_ok=True)

    def materialize(self, artifact: CacheArtifact, destination: Path) -> Path:
        """Publish validated cache bytes atomically and verify the copy."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_path_write_lock(destination):
            with tempfile.NamedTemporaryFile(
                "wb", dir=destination.parent, delete=False,
                prefix=f".{destination.name}.", suffix=".tmp",
            ) as outgoing, artifact.path.open("rb") as incoming:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
                temporary = Path(outgoing.name)
            try:
                if stable_file_hash(temporary) != artifact.checksum:
                    raise OSError("materialized cache checksum mismatch")
                _replace_with_retry(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    def _paths(self, node_id: str, cache_key: str, suffix: str) -> tuple[Path, Path]:
        if not _NODE_PATTERN.fullmatch(node_id):
            raise ValueError(f"invalid cache node id: {node_id!r}")
        if not _KEY_PATTERN.fullmatch(cache_key):
            raise ValueError("invalid render cache key")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("cache suffix must be a simple file extension")
        directory = self.root / node_id
        return directory / f"{cache_key}{suffix}", directory / f"{cache_key}.manifest.json"


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(len(ATOMIC_REPLACE_BACKOFF_SECONDS) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (
                not _is_transient_replace_error(error)
                or attempt == len(ATOMIC_REPLACE_BACKOFF_SECONDS)
            ):
                raise
            time.sleep(ATOMIC_REPLACE_BACKOFF_SECONDS[attempt])
