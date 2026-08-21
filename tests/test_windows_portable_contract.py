from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging" / "windows"


def test_windows_portable_pins_supported_deno_with_notice_and_license() -> None:
    lock = json.loads((WINDOWS / "binaries.lock.json").read_text(encoding="utf-8"))
    binaries = {str(item["name"]): item for item in lock["binaries"]}
    deno = binaries["deno.exe"]

    assert deno == {
        "name": "deno.exe",
        "version": "2.9.5",
        "size_bytes": 97_408_288,
        "sha256": "98f8c2a2d470e4ccb04c935c86ff8050817d877762aec5eaeeb9e409ccb3b9fd",
        "source": "https://github.com/denoland/deno/releases/download/v2.9.5/deno-x86_64-pc-windows-msvc.zip",
        "license": "MIT",
    }
    notice = (WINDOWS / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    license_text = (WINDOWS / "licenses" / "LICENSE-deno-MIT.txt").read_text(encoding="utf-8")
    assert "Deno 2.9.5" in notice
    assert "MIT License" in license_text


def test_fresh_zip_smoke_requires_deno_runtime_and_doctor_capability() -> None:
    smoke = (WINDOWS / "smoke_portable.py").read_text(encoding="utf-8")

    assert '"deno.exe"' in smoke
    assert '[str(deno), "--version"]' in smoke
    assert '"OK Deno"' in smoke
    assert 'smoke_settings["ai"]["provider"] = "mock"' in smoke
