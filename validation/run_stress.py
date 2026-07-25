"""Run a bounded, isolated production-render cache stress check for Goal 3E."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run repeated isolated production-render cache checks.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "metrics" / "stress.json")
    arguments = parser.parse_args(argv)
    if arguments.iterations < 1:
        raise SystemExit("--iterations must be positive")
    command = [
        sys.executable, "-m", "app", "process", "--input", str(arguments.input), "--config", str(arguments.config),
        "--mock-ai", "--production-render-only",
    ]
    runs = [_run(command) for _ in range(arguments.iterations)]
    durations = [run["duration_seconds"] for run in runs]
    peaks = [run["peak_working_set_bytes"] for run in runs if isinstance(run["peak_working_set_bytes"], int)]
    report: dict[str, Any] = {
        "schema_version": "3E.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "iterations": arguments.iterations,
        "mode": "isolated_production_render_cache",
        "command": command[1:],
        "all_succeeded": all(run["return_code"] == 0 for run in runs),
        "duration_seconds": {"min": min(durations), "max": max(durations), "average": round(sum(durations) / len(durations), 3)},
        "peak_working_set_bytes": {
            "available": bool(peaks), "min": min(peaks) if peaks else None,
            "max": max(peaks) if peaks else None,
            "trend_bytes": (peaks[-1] - peaks[0]) if len(peaks) > 1 else 0 if peaks else None,
        },
        "runs": runs,
        "limitations": [
            "This measures a fresh CLI process per iteration; it detects repeat-run failures and abnormal per-process memory growth, not a long-lived service leak.",
            "No API key or provider request is included in this command or report.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(arguments.output)
    return 0 if report["all_succeeded"] else 1


def _run(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0
    while process.poll() is None:
        peak = max(peak, _working_set_bytes(process.pid) or 0)
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    peak = max(peak, _working_set_bytes(process.pid) or 0)
    return {
        "return_code": process.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "peak_working_set_bytes": peak or None,
        "stdout_tail": _safe_tail(stdout),
        "stderr_tail": _safe_tail(stderr),
    }


def _safe_tail(value: str, limit: int = 600) -> str:
    # No command accepts secrets; still avoid retaining a large raw subprocess log.
    return value[-limit:].replace("sk-", "[redacted]-")


def _working_set_bytes(pid: int) -> int | None:
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process:
        return None
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


if __name__ == "__main__":
    raise SystemExit(main())
