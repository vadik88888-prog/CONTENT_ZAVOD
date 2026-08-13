"""Run a short native Qt navigation smoke against an existing desktop dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow
from app.gui.services.settings_store import default_data_directory
from desktop_responsiveness_probe import _services


def _settle(application: QApplication, turns: int = 4) -> None:
    for _ in range(turns):
        application.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=Path, default=default_data_directory())
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    services = _services(root, args.data_directory.expanduser().resolve())
    application = QApplication.instance() or QApplication([])
    window = MainWindow(services)
    window.show()
    _settle(application)

    started = time.perf_counter()
    window.show_projects()
    _settle(application)
    projects_seconds = time.perf_counter() - started
    assert window.stack.currentIndex() == window.projects_index
    assert window.projects_screen._projects

    project = services.projects.load(args.project_id)
    started = time.perf_counter()
    window.show_project(project)
    _settle(application)
    project_seconds = time.perf_counter() - started
    assert window.stack.currentIndex() == window.project_index
    assert window.project_screen.project is not None
    assert window.project_screen.project.project_id == project.project_id

    started = time.perf_counter()
    window.show_projects()
    _settle(application)
    back_seconds = time.perf_counter() - started
    assert window.stack.currentIndex() == window.projects_index

    if args.screenshot:
        screenshot = args.screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(screenshot)):
            raise RuntimeError(f"Could not save smoke screenshot: {screenshot}")

    print(json.dumps({
        "projects_seconds": round(projects_seconds, 3),
        "project_seconds": round(project_seconds, 3),
        "back_seconds": round(back_seconds, 3),
        "project_count": len(window.projects_screen._projects),
        "flow_step": window.project_screen._flow_step,
        "window_visible": window.isVisible(),
    }))
    window.close()
    _settle(application, 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
