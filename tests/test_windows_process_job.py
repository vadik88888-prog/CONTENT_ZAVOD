from __future__ import annotations

import ctypes

from app.gui.services import windows_process_job


class _FakeApi:
    """Callable object that accepts ctypes signature attributes."""

    def __init__(self, callback) -> None:
        self._callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._callback(*args)


def test_windows_job_attaches_kill_on_close_before_assigning_tree(monkeypatch) -> None:
    """The job must keep the crash/close containment policy, not only cancel support."""

    events: list[str] = []
    job = object()
    process = object()

    def set_information(_job, info_class, info_pointer, info_size) -> int:
        limits = ctypes.cast(
            info_pointer,
            ctypes.POINTER(windows_process_job._JobObjectExtendedLimitInformation),
        ).contents
        assert info_class == windows_process_job.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION
        assert info_size == ctypes.sizeof(windows_process_job._JobObjectExtendedLimitInformation)
        assert limits.BasicLimitInformation.LimitFlags & windows_process_job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        events.append("kill-on-close")
        return 1

    class FakeKernel32:
        CreateJobObjectW = _FakeApi(lambda *_args: (events.append("create") or job))
        OpenProcess = _FakeApi(lambda *_args: (events.append("open") or process))
        SetInformationJobObject = _FakeApi(set_information)
        AssignProcessToJobObject = _FakeApi(
            lambda *_args: (events.append("assign") or 1)
        )
        CloseHandle = _FakeApi(lambda handle: (events.append("close-job" if handle is job else "close-process") or 1))

    monkeypatch.setattr(windows_process_job.sys, "platform", "win32")
    monkeypatch.setattr(windows_process_job.ctypes, "WinDLL", lambda *_args, **_kwargs: FakeKernel32())

    attached, problem = windows_process_job.attach_windows_process_job(4242)

    assert attached is job
    assert problem is None
    assert events[:5] == ["create", "open", "kill-on-close", "assign", "close-process"]

    windows_process_job.close_windows_process_job(attached)
    assert events[-1] == "close-job"
