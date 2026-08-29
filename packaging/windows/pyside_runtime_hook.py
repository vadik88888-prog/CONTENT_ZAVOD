"""Make PySide6's own runtime DLL directory win during frozen startup."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys


if bool(getattr(sys, "frozen", False)):
    runtime = Path(getattr(sys, "_MEIPASS")) / "PySide6"
    if runtime.is_dir():
        # PyInstaller otherwise makes the broad _internal directory the DLL
        # lookup root.  That can contain a different MSVC runtime from another
        # extension, which makes QtCore fail before the desktop can start.
        if not ctypes.windll.kernel32.SetDllDirectoryW(str(runtime)):
            raise OSError("Could not configure the bundled PySide6 runtime directory.")
