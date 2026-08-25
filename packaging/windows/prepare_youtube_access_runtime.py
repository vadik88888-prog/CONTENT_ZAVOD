"""Stage the pinned BGutil plugin and Deno server for the Windows portable app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = _sha256(destination)
    if actual.casefold() != expected_sha256.casefold():
        raise RuntimeError(f"SHA-256 mismatch for {url}: {actual}")


def _safe_members(archive: zipfile.ZipFile, prefix: str) -> list[str]:
    members: list[str] = []
    for name in archive.namelist():
        relative = Path(name)
        if not name.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
            continue
        members.append(name)
    return members


def _extract_plugin(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = _safe_members(archive, "yt_dlp_plugins/")
        if not members:
            raise RuntimeError("BGutil wheel does not contain yt-dlp plugin files.")
        for member in members:
            target = destination / Path(member)
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _extract_server(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        root = next((name.split("/", 1)[0] for name in archive.namelist() if "/server/" in name), None)
        if not root:
            raise RuntimeError("BGutil source archive does not contain the server directory.")
        prefix = f"{root}/server/"
        members = _safe_members(archive, prefix)
        if not members:
            raise RuntimeError("BGutil source archive does not contain server files.")
        for member in members:
            relative = Path(member.removeprefix(prefix))
            if not relative.parts:
                continue
            target = destination / relative
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as input_file, target.open("wb") as output:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
        license_member = f"{root}/LICENSE"
        if license_member not in archive.namelist():
            raise RuntimeError("BGutil source archive does not include its GPL license.")
        with archive.open(license_member) as input_file, (destination.parent / "LICENSE-bgutil-GPL-3.0-only.txt").open("wb") as output:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)


def _deno_command(windows: Path) -> str:
    bundled = windows / "tools" / "deno.exe"
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("deno")
    if found:
        return found
    raise RuntimeError("Deno 2.9.5 is required; run prepare_binaries.py first.")


def _verify_runtime(root: Path, lock: dict) -> None:
    required = (
        root / "yt-dlp-plugins" / "yt_dlp_plugins" / "extractor" / "getpot_bgutil_script.py",
        root / "server" / "src" / "generate_once.ts",
        root / "server" / "package-lock.json",
        root / "server" / "node_modules",
        root / "LICENSE-bgutil-GPL-3.0-only.txt",
        root / "deno-junctions.json",
        root / "runtime.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("YouTube access runtime is incomplete: " + ", ".join(missing))
    marker = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    if marker != lock:
        raise RuntimeError("YouTube access runtime lock does not match staged assets.")


def _is_windows_reparse_directory(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        return False
    return path.is_dir() and bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _write_deno_junction_manifest(node_modules: Path, destination: Path) -> None:
    """Record Deno's local link graph for reconstruction inside the onedir."""

    links: list[dict[str, str]] = []
    for junction in node_modules.rglob("*"):
        if not _is_windows_reparse_directory(junction):
            continue
        target = junction.resolve(strict=True)
        if not target.is_relative_to(node_modules):
            raise RuntimeError(f"Deno junction escapes the staged server: {junction}")
        links.append({
            "path": str(junction.relative_to(node_modules)).replace("\\", "/"),
            "target": str(target.relative_to(node_modules)).replace("\\", "/"),
        })
    destination.write_text(
        json.dumps({"schema_version": 1, "links": links}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage pinned BGutil mweb PO Token runtime.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    windows = Path(__file__).resolve().parent
    lock = json.loads((windows / "youtube-access-runtime.lock.json").read_text(encoding="utf-8"))
    runtime = windows / "youtube-access-runtime"
    if args.verify_only:
        _verify_runtime(runtime, lock)
        print("Verified pinned YouTube access runtime.")
        return 0

    deno = _deno_command(windows)
    with tempfile.TemporaryDirectory(prefix="content-factory-youtube-access-") as temporary:
        temporary_root = Path(temporary)
        wheel = temporary_root / "provider.whl"
        source = temporary_root / "provider-source.zip"
        _download(lock["provider"]["wheel"]["url"], wheel, lock["provider"]["wheel"]["sha256"])
        _download(lock["server"]["archive"]["url"], source, lock["server"]["archive"]["sha256"])
        # Deno's Windows node_modules entries are junctions.  They must be
        # created at their final package path: moving a temporary tree would
        # leave junctions pointing at the deleted temporary directory.
        if runtime.exists():
            shutil.rmtree(runtime)
        _extract_plugin(wheel, runtime / "yt-dlp-plugins")
        _extract_server(source, runtime / "server")
        completed = subprocess.run(
            [deno, "install", "--allow-scripts=npm:canvas", "--frozen"],
            cwd=runtime / "server",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Deno failed to install pinned BGutil server dependencies: {completed.returncode}")
        _write_deno_junction_manifest(
            runtime / "server" / "node_modules", runtime / "deno-junctions.json",
        )
        (runtime / "runtime.json").write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        _verify_runtime(runtime, lock)
    print(f"Staged pinned YouTube access runtime: {runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
