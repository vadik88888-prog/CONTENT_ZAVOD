from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import yaml


ARTIFACT_NAME = "ContentFactory-beta-win-x64"
INTERNAL_CLI_SWITCH = "--content-factory-internal-cli"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _decode_output(value: bytes) -> str:
    """Decode redirected output from either UTF-8 or a Russian Windows console."""

    for encoding in ("utf-8", "cp1251"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def _reset_directory(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or not resolved.is_relative_to(parent):
        raise RuntimeError(f"Refusing to reset path outside {parent}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke the portable ZIP from a fresh extraction.")
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()

    windows = Path(__file__).resolve().parent
    root = windows.parents[1]
    build_root = root / "build" / "windows-portable-smoke"
    extracted_root = build_root / "extracted"
    profile = build_root / "profile"
    zip_path = windows / "artifacts" / f"{ARTIFACT_NAME}.zip"
    report_path = windows / "reports" / f"{ARTIFACT_NAME}.build.json"
    if not zip_path.is_file() or not report_path.is_file():
        raise RuntimeError("Build the portable ZIP before running smoke.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_deno = next(
        (item for item in report.get("binaries", []) if item.get("name") == "deno.exe"),
        None,
    )
    if expected_deno is None:
        raise RuntimeError("Build report does not contain the pinned Deno runtime.")
    _reset_directory(extracted_root, build_root)
    _reset_directory(profile, build_root)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted_root)

    portable = extracted_root / ARTIFACT_NAME
    executable = portable / "ContentFactory.exe"
    config = portable / "_internal" / "config.example.yaml"
    deno = portable / "_internal" / "tools" / "deno.exe"
    youtube_runtime = portable / "_internal" / "youtube-access-runtime"
    required_youtube_runtime_files = (
        youtube_runtime / "runtime.json",
        youtube_runtime / "yt-dlp-plugins" / "yt_dlp_plugins" / "extractor" / "getpot_bgutil_script.py",
        youtube_runtime / "server" / "src" / "generate_once.ts",
    )
    if (
        not executable.is_file()
        or not config.is_file()
        or not deno.is_file()
        or any(not path.is_file() for path in required_youtube_runtime_files)
    ):
        raise RuntimeError("Freshly extracted portable folder is incomplete.")
    environment = dict(os.environ)
    environment.update({
        "LOCALAPPDATA": str(profile),
        "APPDATA": str(profile),
        "PYTHONIOENCODING": "utf-8",
    })
    smoke_config = profile / "portable-smoke-config.yaml"
    smoke_settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(smoke_settings, dict) or not isinstance(smoke_settings.get("ai"), dict):
        raise RuntimeError("Bundled config does not contain the expected AI settings.")
    smoke_settings["ai"]["provider"] = "mock"
    smoke_config.write_text(
        yaml.safe_dump(smoke_settings, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    deno_started = time.perf_counter()
    deno_check = subprocess.run(
        [str(deno), "--version"],
        cwd=portable,
        env=environment,
        capture_output=True,
        timeout=30,
    )
    deno_seconds = time.perf_counter() - deno_started
    deno_output = "\n".join(
        _decode_output(part) for part in (deno_check.stdout, deno_check.stderr) if part
    ).strip()
    if deno_check.returncode != 0 or not deno_output.startswith(f"deno {expected_deno['version']}"):
        raise RuntimeError(
            f"Bundled Deno smoke failed ({deno_check.returncode}); output:\n{deno_output[-2000:]}"
        )

    cli_started = time.perf_counter()
    cli = subprocess.run(
        [str(executable), INTERNAL_CLI_SWITCH, "doctor", "--config", str(smoke_config)],
        cwd=portable,
        env=environment,
        capture_output=True,
        timeout=120,
    )
    cli_seconds = time.perf_counter() - cli_started
    cli_output = "\n".join(
        _decode_output(part) for part in (cli.stdout, cli.stderr) if part
    ).strip()
    required_markers = ("Content Factory", "OK FFmpeg", "OK FFprobe", "OK yt-dlp", "OK Deno")
    if cli.returncode != 0 or any(marker not in cli_output for marker in required_markers):
        raise RuntimeError(
            f"Frozen CLI smoke failed ({cli.returncode}); output:\n{cli_output[-4000:]}"
        )

    escaped_executable = str(executable).replace("'", "''")
    escaped_portable = str(portable).replace("'", "''")
    native_script = f"""
$ErrorActionPreference = 'Stop'
$process = Start-Process -FilePath '{escaped_executable}' -WorkingDirectory '{escaped_portable}' -PassThru
$visible = $false
$title = ''
$handle = 0
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ([DateTime]::UtcNow -lt $deadline -and -not $process.HasExited) {{
    Start-Sleep -Milliseconds 100
    $process.Refresh()
    if ($null -ne $process.MainWindowHandle -and [int64]$process.MainWindowHandle -ne 0) {{
        $visible = $true
        $title = $process.MainWindowTitle
        $handle = [int64]$process.MainWindowHandle
        break
    }}
}}
$closeRequested = $false
$survivedSettle = $false
if ($visible -and -not $process.HasExited) {{
    Start-Sleep -Seconds 2
    $process.Refresh()
    $survivedSettle = -not $process.HasExited
}}
if (-not $process.HasExited) {{
    $closeRequested = $process.CloseMainWindow()
    if (-not $process.WaitForExit(5000)) {{ Stop-Process -Id $process.Id -Force; $process.WaitForExit() }}
}}
[pscustomobject]@{{
    process_id = $process.Id
    window_visible = $visible
    window_title = $title
    window_handle = $handle
    survived_settle = $survivedSettle
    close_requested = $closeRequested
    exited = $process.HasExited
}} | ConvertTo-Json -Compress
"""
    native_started = time.perf_counter()
    native = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", native_script],
        cwd=portable,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    native_seconds = time.perf_counter() - native_started
    if native.returncode != 0:
        raise RuntimeError(f"Native smoke failed: {native.stderr.strip()}")
    native_result = json.loads(native.stdout.strip().splitlines()[-1])
    if (
        not native_result.get("window_visible")
        or native_result.get("window_title") != "Content Factory"
        or not native_result.get("survived_settle")
    ):
        raise RuntimeError(f"Native Content Factory window was not observed: {native_result}")

    report["status"] = "smoke_passed"
    report["smoke"] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": "fresh_zip_extraction",
        "extracted_path": str(portable.relative_to(root)).replace("\\", "/"),
        "deno": {
            "exit_code": deno_check.returncode,
            "seconds": round(deno_seconds, 3),
            "version_line": deno_output.splitlines()[0],
        },
        "youtube_access_runtime": {
            "path": str(youtube_runtime.relative_to(portable)).replace("\\", "/"),
            "required_files": [
                str(path.relative_to(portable)).replace("\\", "/")
                for path in required_youtube_runtime_files
            ],
        },
        "frozen_cli": {
            "exit_code": cli.returncode,
            "seconds": round(cli_seconds, 3),
            "ai_mode": "mock",
            "markers": list(required_markers),
            "output_tail": cli_output[-2000:],
        },
        "native_window": {
            **native_result,
            "seconds": round(native_seconds, 3),
            "launch_method": "Start-Process",
        },
    }
    _write_json(report_path, report)
    if not args.keep_extracted:
        shutil.rmtree(extracted_root)
        shutil.rmtree(profile)
    print(json.dumps(report["smoke"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
