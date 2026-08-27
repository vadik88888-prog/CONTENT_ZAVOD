from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from app.gui.components.video_preview import preview_proxy_path, preview_proxy_temporary_path
from app.gui.services.preview_proxy_cache import (
    acquire_preview_proxy_lease,
    preview_proxy_lock_path,
    reclaim_stale_preview_proxy_lease,
    release_preview_proxy_lease,
    validated_preview_proxy,
    write_preview_proxy_manifest,
)


def test_preview_proxy_path_is_specific_to_source_revision_not_candidate_range(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"first")
    cache = tmp_path / "preview-proxies"

    first = preview_proxy_path(cache, source)
    same = preview_proxy_path(cache, source)
    source.write_bytes(b"changed source revision")
    changed_source = preview_proxy_path(cache, source)

    assert first == same
    assert first.suffix == ".mp4"
    assert first != changed_source


def test_preview_proxy_temporary_path_keeps_the_mp4_muxer_suffix(tmp_path: Path) -> None:
    destination = tmp_path / "preview.mp4"

    temporary = preview_proxy_temporary_path(destination)

    assert temporary.name == "preview.part.mp4"
    assert temporary.parent == destination.parent


def test_proxy_lease_deduplicates_across_cache_clients_and_uses_unique_temp(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    destination = preview_proxy_path(tmp_path / "cache", source)

    owner = acquire_preview_proxy_lease(destination, source)
    follower = acquire_preview_proxy_lease(destination, source)

    assert owner is not None
    assert follower is None
    assert not reclaim_stale_preview_proxy_lease(destination, stale_after_seconds=0)
    assert owner.temporary.name.endswith(f".{owner.owner_id}.part.mp4")
    assert owner.temporary != preview_proxy_temporary_path(destination)

    release_preview_proxy_lease(owner)
    next_owner = acquire_preview_proxy_lease(destination, source)
    assert next_owner is not None
    assert next_owner.owner_id != owner.owner_id
    release_preview_proxy_lease(next_owner)


def test_proxy_lease_is_exclusive_across_python_processes(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    destination = preview_proxy_path(tmp_path / "cache", source)
    script = "\n".join([
        "import sys",
        "from pathlib import Path",
        "from app.gui.services.preview_proxy_cache import acquire_preview_proxy_lease, release_preview_proxy_lease",
        "lease = acquire_preview_proxy_lease(Path(sys.argv[1]), Path(sys.argv[2]))",
        "print(lease.owner_id if lease else 'FOLLOWER', flush=True)",
        "sys.stdin.readline()",
        "release_preview_proxy_lease(lease) if lease else None",
    ])
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(destination), str(source)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    try:
        owner_id = process.stdout.readline().strip()
        assert owner_id and owner_id != "FOLLOWER"
        assert acquire_preview_proxy_lease(destination, source) is None
    finally:
        process.stdin.write("release\n")
        process.stdin.flush()
        stderr = process.communicate(timeout=10)[1]
    assert process.returncode == 0, stderr

    next_owner = acquire_preview_proxy_lease(destination, source)
    assert next_owner is not None
    release_preview_proxy_lease(next_owner)


def test_proxy_lease_recovers_after_owner_process_crash(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    destination = preview_proxy_path(tmp_path / "cache", source)
    script = "\n".join([
        "import os, sys",
        "from pathlib import Path",
        "from app.gui.services.preview_proxy_cache import acquire_preview_proxy_lease",
        "lease = acquire_preview_proxy_lease(Path(sys.argv[1]), Path(sys.argv[2]))",
        "print(lease.owner_id if lease else 'FOLLOWER', flush=True)",
        "os._exit(0)",
    ])
    process = subprocess.run(
        [sys.executable, "-c", script, str(destination), str(source)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() not in {"", "FOLLOWER"}
    assert preview_proxy_lock_path(destination).exists()

    recovered = acquire_preview_proxy_lease(destination, source)
    assert recovered is not None
    release_preview_proxy_lease(recovered)


def test_proxy_cache_requires_matching_probe_manifest_and_immutable_file(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    destination = preview_proxy_path(tmp_path / "cache", source)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"validated proxy bytes")
    probe = {
        "video_codec": "h264", "width": 854, "height": 480,
        "fps": "30/1", "duration_seconds": 120.0, "has_audio": True,
    }

    assert not validated_preview_proxy(destination, source)
    write_preview_proxy_manifest(destination, source, probe)
    assert validated_preview_proxy(destination, source, required_end_seconds=119.9)

    destination.write_bytes(b"corrupt replacement with a different size")
    assert not validated_preview_proxy(destination, source)
