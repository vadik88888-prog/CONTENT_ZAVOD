from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.render_cache import GranularRenderCache, runtime_cache_key
from app.utils import stable_file_hash


def _key(label: str) -> str:
    return runtime_cache_key("a" * 64, inputs={"label": label})


def test_cache_requires_matching_version_dependencies_size_and_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"complete artifact")
    dependencies = {"base": "b" * 64}
    cache = GranularRenderCache(tmp_path / "cache", producer_version="renderer-1")
    stored = cache.store_file(
        "encode", _key("valid"), source, suffix=".mp4",
        dependency_checksums=dependencies,
    )

    assert cache.load(
        "encode", stored.cache_key, suffix=".mp4",
        dependency_checksums=dependencies,
    ) == stored
    assert cache.load(
        "encode", stored.cache_key, suffix=".mp4",
        dependency_checksums={"base": "c" * 64},
    ) is None
    assert GranularRenderCache(
        tmp_path / "cache", producer_version="renderer-2",
    ).load(
        "encode", stored.cache_key, suffix=".mp4",
        dependency_checksums=dependencies,
    ) is None

    stored.path.write_bytes(b"truncated")
    assert cache.load(
        "encode", stored.cache_key, suffix=".mp4",
        dependency_checksums=dependencies,
    ) is None


def test_partial_or_corrupt_manifest_never_blesses_stale_artifact(tmp_path: Path) -> None:
    cache = GranularRenderCache(tmp_path / "cache", producer_version="renderer-1")
    key = _key("partial")
    node = tmp_path / "cache" / "qc"
    node.mkdir(parents=True)
    artifact = node / f"{key}.json"
    artifact.write_text('{"old": true}', encoding="utf-8")
    assert cache.load("qc", key, suffix=".json") is None

    manifest = node / f"{key}.manifest.json"
    manifest.write_text("{truncated", encoding="utf-8")
    assert cache.load("qc", key, suffix=".json") is None

    manifest.write_text(json.dumps({
        "schema_version": "7G.render-cache.1",
        "status": "failed",
        "node_id": "qc",
        "cache_key": key,
        "producer_version": "renderer-1",
        "dependency_checksums": {},
        "artifact_name": artifact.name,
        "checksum": stable_file_hash(artifact),
        "byte_size": artifact.stat().st_size,
    }), encoding="utf-8")
    assert cache.load("qc", key, suffix=".json") is None


def test_concurrent_same_key_commit_is_atomic_and_materialization_is_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"deterministic bytes" * 1024)
    cache = GranularRenderCache(tmp_path / "cache", producer_version="renderer-1")
    key = _key("concurrent")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda _: cache.store_file("base-visual", key, source, suffix=".mp4"),
            range(8),
        ))

    assert {item.checksum for item in results} == {stable_file_hash(source)}
    loaded = cache.load("base-visual", key, suffix=".mp4")
    assert loaded is not None
    destination = tmp_path / "published" / "preview.mp4"
    cache.materialize(loaded, destination)
    assert stable_file_hash(destination) == loaded.checksum
