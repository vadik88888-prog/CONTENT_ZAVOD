"""PyInstaller entry script for the Content Factory desktop executable."""

import sys
from multiprocessing import freeze_support


_INTERNAL_CLI_SWITCH = "--content-factory-internal-cli"


def _restore_internal_cli_stdio(arguments: list[str]) -> None:
    """Rebuild QProcess pipes lost by PyInstaller's windowed bootloader.

    A windowed Windows executable deliberately has no console and PyInstaller
    sets ``sys.stdout``/``sys.stderr`` to ``None``.  The desktop starts its
    private worker with QProcess, which supplies valid inherited pipe handles;
    restore only those handles so ordinary GUI launches cannot create a console.
    """

    if (
        arguments[:1] != [_INTERNAL_CLI_SWITCH]
        or sys.platform != "win32"
        or not getattr(sys, "frozen", False)
    ):
        return
    import ctypes
    import msvcrt
    import os

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    encoding = os.environ.get("PYTHONIOENCODING", "utf-8").split(":", 1)[0] or "utf-8"
    for attribute, handle_code in (("stdout", -11), ("stderr", -12)):
        if getattr(sys, attribute, None) is not None:
            continue
        handle = kernel32.GetStdHandle(handle_code)
        if handle in (None, 0, ctypes.c_void_p(-1).value):
            continue
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY)
            stream = os.fdopen(
                descriptor, "w", buffering=1, encoding=encoding,
                errors="backslashreplace",
            )
        except OSError:
            continue
        setattr(sys, attribute, stream)


if __name__ == "__main__":
    # PyInstaller's override diverts multiprocessing/resource-tracker children
    # before application imports can mistake them for another desktop launch.
    freeze_support()
    _restore_internal_cli_stdio(sys.argv[1:])
    from app.frozen_entrypoint import main

    raise SystemExit(main())
