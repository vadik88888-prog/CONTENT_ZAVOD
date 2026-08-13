from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, item: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(item["size_bytes"])
        and _sha256(path).casefold() == str(item["sha256"]).casefold()
    )


def _download(url: str, destination: Path) -> None:
    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _extract_named(archive_path: Path, filename: str, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        member = next(
            (name for name in archive.namelist() if Path(name).name.casefold() == filename.casefold()),
            None,
        )
        if member is None:
            raise RuntimeError(f"{filename} not found in {archive_path.name}")
        with archive.open(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify pinned Windows tool binaries.")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    windows = Path(__file__).resolve().parent
    tools = windows / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    lock = json.loads((windows / "binaries.lock.json").read_text(encoding="utf-8"))
    items = lock["binaries"]
    pending = [item for item in items if args.force or not _matches(tools / item["name"], item)]
    if args.verify_only:
        if pending:
            raise RuntimeError(
                "Pinned binaries missing or invalid: " + ", ".join(item["name"] for item in pending)
            )
    elif pending:
        with tempfile.TemporaryDirectory(prefix="content-factory-tools-") as temporary:
            temporary_root = Path(temporary)
            downloads: dict[str, Path] = {}
            for item in pending:
                url = str(item["source"])
                downloaded = downloads.get(url)
                if downloaded is None:
                    downloaded = temporary_root / f"download-{len(downloads)}{Path(url).suffix}"
                    _download(url, downloaded)
                    downloads[url] = downloaded
                destination = tools / str(item["name"])
                staged = temporary_root / f"staged-{item['name']}"
                if downloaded.suffix.casefold() == ".zip":
                    _extract_named(downloaded, str(item["name"]), staged)
                else:
                    shutil.copy2(downloaded, staged)
                if not _matches(staged, item):
                    raise RuntimeError(f"Downloaded binary does not match lock: {item['name']}")
                staged.replace(destination)
    invalid = [item["name"] for item in items if not _matches(tools / item["name"], item)]
    if invalid:
        raise RuntimeError("Pinned binary verification failed: " + ", ".join(invalid))
    print("Verified pinned binaries: " + ", ".join(item["name"] for item in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
