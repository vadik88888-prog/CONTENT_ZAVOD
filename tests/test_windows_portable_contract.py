from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging" / "windows"


def _build_module():
    spec = importlib.util.spec_from_file_location(
        "windows_portable_build", WINDOWS / "build_portable.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_windows_portable_keeps_mweb_provider_build_reference_out_of_portable_runtime() -> None:
    lock = json.loads((WINDOWS / "youtube-access-runtime.lock.json").read_text(encoding="utf-8"))
    spec = (WINDOWS / "ContentFactory.spec").read_text(encoding="utf-8")
    preparation = (WINDOWS / "prepare_youtube_access_runtime.py").read_text(encoding="utf-8")

    assert lock["provider"]["version"] == lock["server"]["version"] == "1.3.2"
    assert lock["runtime"] == {
        "yt_dlp": "2026.08.19",
        "deno": "2.9.5",
        "player_client": "mweb",
        "remote_component": "ejs:github",
    }
    assert "--allow-scripts=npm:canvas" in preparation
    assert "_write_deno_junction_manifest" in preparation
    assert "FILE_ATTRIBUTE_REPARSE_POINT" in preparation
    build = (WINDOWS / "build_portable.py").read_text(encoding="utf-8")
    assert "_restore_deno_junctions" in build
    assert "staged_youtube_runtime.is_dir()" in build
    assert "youtube_access_runtime =" not in spec
    assert 'datas.append((str(youtube_access_runtime), "youtube-access-runtime"))' not in spec


def test_fresh_zip_smoke_requires_deno_runtime_and_doctor_capability() -> None:
    smoke = (WINDOWS / "smoke_portable.py").read_text(encoding="utf-8")

    assert '"deno.exe"' in smoke
    assert '[str(deno), "--version"]' in smoke
    assert '"OK Deno"' in smoke
    assert 'smoke_settings["ai"]["provider"] = "mock"' in smoke
    assert '$null -ne $process.MainWindowHandle' in smoke


def test_portable_collects_shiboken_runtime_for_frozen_qt_startup() -> None:
    spec = (WINDOWS / "ContentFactory.spec").read_text(encoding="utf-8")
    hook = (WINDOWS / "pyside_runtime_hook.py").read_text(encoding="utf-8")

    assert '"shiboken6"' in spec
    assert '"PySide6"' in spec
    assert 'pyside_runtime_hook.py' in spec
    assert 'SetDllDirectoryW' in hook


def test_portable_build_rejects_friend_beta_private_signing_material(tmp_path: Path) -> None:
    package = tmp_path / "ContentFactory"
    package.mkdir()
    (package / "friend_beta_signing.seed").write_bytes(b"never package a signing key")

    try:
        _build_module()._assert_no_private_signing_material(package)
    except RuntimeError as error:
        assert "forbidden Friend Beta signing material" in str(error)
    else:
        raise AssertionError("portable build must reject a private signing key")

    (package / "friend_beta_signing.seed").unlink()
    (package / "public-verification-key.txt").write_text("public only", encoding="utf-8")
    _build_module()._assert_no_private_signing_material(package)


def test_portable_build_rejects_private_key_bytes_and_private_input(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "ContentFactory"
    package.mkdir()
    module = _build_module()
    monkeypatch.setattr(module, "_private_signing_material", lambda: (b"friend-beta-private-seed",))
    (package / "payload.bin").write_bytes(b"prefixfriend-beta-private-seedsuffix")
    with pytest.raises(RuntimeError, match="private Friend Beta signing material"):
        module._assert_no_private_signing_material(package)

    (package / "payload.bin").write_bytes(b"prefix-----BEGIN PRIVATE KEY-----suffix")
    with pytest.raises(RuntimeError, match="private Friend Beta signing material"):
        module._assert_no_private_signing_material(package)

    (package / "payload.bin").unlink()
    private_input = tmp_path / "friend_beta_signing.seed"
    private_input.write_bytes(b"not a package input")
    with pytest.raises(RuntimeError, match="package input"):
        module._assert_no_private_signing_material(package, package_inputs=(private_input,))


def test_materialized_deno_junction_is_accepted_in_portable_onedir(tmp_path: Path) -> None:
    runtime = tmp_path / "youtube-access-runtime"
    node_modules = runtime / "server" / "node_modules"
    target = node_modules / ".deno" / "axios" / "node_modules" / "axios"
    target.mkdir(parents=True)
    (target / "package.json").write_text("{}", encoding="utf-8")
    # PyInstaller dereferences the original junction when collecting datas.
    # Model the resulting bundled directory rather than relying on junction
    # creation privileges in the test environment.
    materialized = node_modules / "axios"
    materialized.mkdir()
    (materialized / "package.json").write_text("{}", encoding="utf-8")
    (runtime / "deno-junctions.json").write_text(
        json.dumps({
            "schema_version": 1,
            "links": [{"path": "axios", "target": ".deno/axios/node_modules/axios"}],
        }),
        encoding="utf-8",
    )

    _build_module()._restore_deno_junctions(runtime)

    assert materialized.is_dir()
