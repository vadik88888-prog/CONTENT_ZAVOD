from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile


ARTIFACT_NAME = "ContentFactory-beta-win-x64"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _reset_directory(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or not resolved.is_relative_to(parent):
        raise RuntimeError(f"Refusing to reset path outside {parent}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _verify_runtime(lock: dict) -> dict[str, str]:
    expected_python = lock["python"]
    actual_python = platform.python_version()
    actual_architecture = platform.machine()
    actual_bits = 64 if sys.maxsize > 2**32 else 32
    if actual_python != expected_python["version"]:
        raise RuntimeError(
            f"Python version mismatch: expected {expected_python['version']}, got {actual_python}"
        )
    if actual_architecture.casefold() != str(expected_python["architecture"]).casefold():
        raise RuntimeError(
            f"Python architecture mismatch: expected {expected_python['architecture']}, got {actual_architecture}"
        )
    if actual_bits != int(expected_python["bits"]):
        raise RuntimeError(f"Python bitness mismatch: expected {expected_python['bits']}, got {actual_bits}")
    resolved: dict[str, str] = {}
    for distribution, expected in lock["distributions"].items():
        actual = distribution_version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"Distribution mismatch for {distribution}: expected {expected}, got {actual}"
            )
        resolved[distribution] = actual
    return resolved


def _verify_binaries(tools: Path, lock: dict) -> list[dict]:
    verified: list[dict] = []
    for item in lock["binaries"]:
        path = tools / str(item["name"])
        if not path.is_file():
            raise RuntimeError(f"Required portable binary is missing: {path}")
        size = path.stat().st_size
        checksum = _sha256(path)
        if size != int(item["size_bytes"]):
            raise RuntimeError(f"Size mismatch for {path.name}: expected {item['size_bytes']}, got {size}")
        if checksum.casefold() != str(item["sha256"]).casefold():
            raise RuntimeError(f"SHA-256 mismatch for {path.name}: {checksum}")
        verified.append({**item, "sha256": checksum, "size_bytes": size})
    return verified


def _zip_directory(source: Path, destination: Path) -> int:
    if destination.exists():
        destination.unlink()
    file_count = 0
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True,
    ) as archive:
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            archive.write(path, (Path(ARTIFACT_NAME) / relative).as_posix())
            file_count += 1
    return file_count


def _git_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pinned Windows x64 portable beta.")
    parser.add_argument("--skip-clean", action="store_true", help="Reuse PyInstaller analysis cache.")
    parser.add_argument("--base-git-commit", help="Commit exported into a clean source snapshot.")
    parser.add_argument(
        "--source-mode", choices=("worktree", "exported_git_snapshot"), default="worktree",
    )
    args = parser.parse_args()

    windows = Path(__file__).resolve().parent
    root = windows.parents[1]
    runtime_lock_path = windows / "runtime.lock.json"
    binary_lock_path = windows / "binaries.lock.json"
    runtime_lock = _read_json(runtime_lock_path)
    binary_lock = _read_json(binary_lock_path)
    resolved_distributions = _verify_runtime(runtime_lock)
    verified_binaries = _verify_binaries(windows / "tools", binary_lock)

    build_root = root / "build" / "windows-portable"
    work_path = build_root / "pyinstaller"
    dist_path = build_root / "dist"
    artifacts = windows / "artifacts"
    unpacked = artifacts / ARTIFACT_NAME
    reports = windows / "reports"
    zip_path = artifacts / f"{ARTIFACT_NAME}.zip"
    report_path = reports / f"{ARTIFACT_NAME}.build.json"
    sums_path = reports / "SHA256SUMS"

    artifacts.mkdir(parents=True, exist_ok=True)
    if not args.skip_clean:
        _reset_directory(work_path, build_root)
        _reset_directory(dist_path, build_root)
    else:
        work_path.mkdir(parents=True, exist_ok=True)
        dist_path.mkdir(parents=True, exist_ok=True)
    if unpacked.exists():
        if not unpacked.resolve().is_relative_to(artifacts.resolve()):
            raise RuntimeError(f"Unexpected artifact path: {unpacked}")
        shutil.rmtree(unpacked)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        *(tuple() if args.skip_clean else ("--clean",)),
        "--workpath",
        str(work_path),
        "--distpath",
        str(dist_path),
        str(windows / "ContentFactory.spec"),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=root, check=False)
    build_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"PyInstaller failed with exit code {completed.returncode}")

    collected = dist_path / "ContentFactory"
    executable = collected / "ContentFactory.exe"
    required = [
        executable,
        collected / "_internal" / "config.example.yaml",
        collected / "_internal" / "app" / "gui" / "styles" / "theme.qss",
        *(collected / "_internal" / "tools" / str(item["name"]) for item in verified_binaries),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("PyInstaller output is incomplete: " + ", ".join(missing))
    shutil.move(str(collected), str(unpacked))

    manifests = unpacked / "manifests"
    licenses = unpacked / "licenses"
    manifests.mkdir()
    licenses.mkdir()
    for path in (runtime_lock_path, binary_lock_path):
        shutil.copy2(path, manifests / path.name)
    shutil.copy2(windows / "THIRD_PARTY_NOTICES.md", unpacked / "THIRD_PARTY_NOTICES.md")
    for path in sorted((windows / "licenses").glob("*")):
        if path.is_file():
            shutil.copy2(path, licenses / path.name)

    source_inputs = [
        windows / "ContentFactory.spec",
        windows / "desktop_entrypoint.py",
        windows / "build_portable.py",
        windows / "smoke_portable.py",
        runtime_lock_path,
        binary_lock_path,
        root / "app" / "runtime.py",
        root / "app" / "frozen_entrypoint.py",
        root / "app" / "source_download.py",
        windows / "THIRD_PARTY_NOTICES.md",
        windows / "licenses" / "LICENSE-deno-MIT.txt",
    ]
    build_info = {
        "schema_version": 1,
        "artifact": ARTIFACT_NAME,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "base_git_commit": args.base_git_commit or _git_value(root, "rev-parse", "HEAD"),
        "source_mode": args.source_mode,
        "python": runtime_lock["python"],
        "distributions": resolved_distributions,
        "binaries": verified_binaries,
        "inputs": {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in source_inputs
        },
    }
    _write_json(unpacked / "BUILD-INFO.json", build_info)

    folder_files = [path for path in unpacked.rglob("*") if path.is_file()]
    folder_size = sum(path.stat().st_size for path in folder_files)
    zip_started = time.perf_counter()
    zip_file_count = _zip_directory(unpacked, zip_path)
    zip_seconds = time.perf_counter() - zip_started
    zip_checksum = _sha256(zip_path)
    report = {
        **build_info,
        "status": "built",
        "build_seconds": round(build_seconds, 3),
        "zip_seconds": round(zip_seconds, 3),
        "onedir": {
            "path": str(unpacked.relative_to(root)).replace("\\", "/"),
            "file_count": len(folder_files),
            "size_bytes": folder_size,
            "executable_sha256": _sha256(unpacked / "ContentFactory.exe"),
        },
        "zip": {
            "path": str(zip_path.relative_to(root)).replace("\\", "/"),
            "file_count": zip_file_count,
            "size_bytes": zip_path.stat().st_size,
            "sha256": zip_checksum,
        },
        "smoke": None,
    }
    _write_json(report_path, report)
    sums_path.parent.mkdir(parents=True, exist_ok=True)
    sums_path.write_text(f"{zip_checksum}  {zip_path.name}\n", encoding="ascii")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
