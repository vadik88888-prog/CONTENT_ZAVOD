"""Small Windows Job Object wrapper for GUI-owned child process trees.

The desktop client launches a few tools through ``QProcess``.  Some of those
tools (the pipeline CLI and yt-dlp) can launch FFmpeg children of their own.
On Windows a Job Object gives the GUI one handle for that whole tree.  The
``KILL_ON_JOB_CLOSE`` limit is intentionally set before assignment: closing
the application or releasing a completed run cannot leave a detached encoder
behind consuming CPU, RAM, or a project output file.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any


PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _set_signature(function: Any, argtypes: tuple[object, ...], restype: object) -> None:
    """Set ctypes metadata while keeping the helper easy to mock in tests."""

    try:
        function.argtypes = argtypes
        function.restype = restype
    except AttributeError:
        pass


def configure_job_kill_on_close(kernel32: Any, job: object) -> tuple[bool, int]:
    """Set the Job Object policy that kills every member on last handle close."""

    setter = kernel32.SetInformationJobObject
    _set_signature(
        setter,
        (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32),
        ctypes.c_int,
    )
    limits = _JobObjectExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = bool(
        setter(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
    )
    return configured, 0 if configured else ctypes.get_last_error()


def attach_windows_process_job(process_id: int) -> tuple[object | None, str | None]:
    """Attach a running Windows process to a close-kills-tree Job Object.

    The caller owns the returned Job handle and must call
    :func:`close_windows_process_job` in every terminal path.  On an ordinary
    successful run that close is also useful: it prevents a CLI that exits
    before one of its helpers from leaking the helper into the desktop.
    """

    if sys.platform != "win32" or not process_id:
        return None, None
    kernel32: Any | None = None
    job: object | None = None
    process: object | None = None
    attached = False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _set_signature(kernel32.CreateJobObjectW, (ctypes.c_void_p, ctypes.c_wchar_p), ctypes.c_void_p)
        _set_signature(kernel32.OpenProcess, (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32), ctypes.c_void_p)
        _set_signature(kernel32.AssignProcessToJobObject, (ctypes.c_void_p, ctypes.c_void_p), ctypes.c_int)
        _set_signature(kernel32.CloseHandle, (ctypes.c_void_p,), ctypes.c_int)

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None, f"Windows job object creation failed for PID {process_id}; error={ctypes.get_last_error()}."
        process = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, process_id)
        if not process:
            return None, (
                f"Windows job process handle unavailable for PID {process_id}; "
                f"error={ctypes.get_last_error()}."
            )
        configured, error = configure_job_kill_on_close(kernel32, job)
        if not configured:
            return None, f"Windows job close policy failed for PID {process_id}; error={error}."
        assigned = bool(kernel32.AssignProcessToJobObject(job, process))
        if not assigned:
            return None, (
                f"Windows job object assignment failed for PID {process_id}; "
                f"error={ctypes.get_last_error()}."
            )
        attached = True
        return job, None
    except (AttributeError, OSError) as error:
        return None, f"Windows job object setup failed for PID {process_id}: {error}"
    finally:
        if kernel32 is not None and process:
            try:
                kernel32.CloseHandle(process)
            except (AttributeError, OSError):
                pass
        if kernel32 is not None and job and not attached:
            try:
                kernel32.CloseHandle(job)
            except (AttributeError, OSError):
                pass


def terminate_windows_process_job(job: object | None, exit_code: int = 1) -> tuple[bool, int]:
    """Terminate all processes in an attached Job Object without closing it."""

    if sys.platform != "win32" or job is None:
        return False, 0
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _set_signature(kernel32.TerminateJobObject, (ctypes.c_void_p, ctypes.c_uint32), ctypes.c_int)
        terminated = bool(kernel32.TerminateJobObject(job, exit_code))
        return terminated, 0 if terminated else ctypes.get_last_error()
    except (AttributeError, OSError):
        return False, ctypes.get_last_error()


def close_windows_process_job(job: object | None) -> None:
    """Release the final Job handle, enforcing its kill-on-close policy."""

    if sys.platform != "win32" or job is None:
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _set_signature(kernel32.CloseHandle, (ctypes.c_void_p,), ctypes.c_int)
        kernel32.CloseHandle(job)
    except (AttributeError, OSError):
        # Explicit cancellation still has the QProcess/taskkill fallback;
        # closing a best-effort OS handle must never crash the desktop UI.
        return
