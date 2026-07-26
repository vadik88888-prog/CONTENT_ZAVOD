from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow
from app.gui.services.desktop_services import DesktopServices
from app.gui.styles import load_theme


def application_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Content Factory")
    app.setOrganizationName("Content Factory")
    app.setStyleSheet(load_theme())
    window = MainWindow(DesktopServices.create(application_root()))
    window.show()
    return app.exec()
