"""Measure desktop UI hot paths against an existing local project dataset.

The probe is read-only: it constructs the desktop services without startup
recovery, opens Qt offscreen, and reports elapsed time plus legacy report reads.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import app.gui.screens.project_screen as project_screen_module
import app.gui.services.desktop_services as desktop_services_module
import app.gui.services.run_projection as run_projection_module
from app.gui.models import DesktopSettings, ProcessingPhase, ProcessingSnapshot
from app.gui.screens.project_screen import ProjectScreen
from app.gui.screens.projects_screen import ProjectsScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore, default_data_directory
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel, ProjectsViewModel


def _services(engine_root: Path, data_directory: Path) -> DesktopServices:
    settings_path = data_directory / "settings.json"
    settings = DesktopSettings.from_dict(json.loads(settings_path.read_text(encoding="utf-8")))
    projects = DesktopProjectStore(data_directory)
    return DesktopServices(
        engine_root=engine_root,
        settings_store=SettingsStore(data_directory),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(engine_root),
        system=SystemService(engine_root),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=Path, default=default_data_directory())
    parser.add_argument("--project-id", required=True)
    args = parser.parse_args()

    engine_root = Path(__file__).resolve().parents[1]
    data_directory = args.data_directory.expanduser().resolve()
    services = _services(engine_root, data_directory)
    application = QApplication.instance() or QApplication([])
    counters = {"reads": 0, "bytes": 0}

    original_service_read = desktop_services_module.read_json
    original_screen_read = project_screen_module.read_json
    original_projection_read = run_projection_module.read_json

    def tracked(original: Callable):
        def wrapper(path: Path, *values, **options):
            candidate = Path(path)
            if candidate.name == "report.json" and candidate.is_file():
                counters["reads"] += 1
                counters["bytes"] += candidate.stat().st_size
            return original(path, *values, **options)

        return wrapper

    desktop_services_module.read_json = tracked(original_service_read)
    project_screen_module.read_json = tracked(original_screen_read)
    run_projection_module.read_json = tracked(original_projection_read)

    def measure(label: str, action: Callable[[], object], *, settle_events: bool = False) -> None:
        counters["reads"] = 0
        counters["bytes"] = 0
        started = time.perf_counter()
        action()
        if settle_events:
            application.processEvents()
            application.processEvents()
        elapsed = time.perf_counter() - started
        print(json.dumps({
            "label": label,
            "seconds": round(elapsed, 3),
            "report_reads": counters["reads"],
            "report_gib": round(counters["bytes"] / (1024 ** 3), 3),
        }))

    projects_screen = ProjectsScreen(ProjectsViewModel(services))
    measure("projects_refresh", projects_screen.refresh)

    project = services.projects.load(args.project_id)
    project_viewmodel = ProjectViewModel(services)
    project_screen = ProjectScreen(project_viewmodel)
    measure("project_open", lambda: project_screen.open(project), settle_events=True)
    snapshot = ProcessingSnapshot(
        ProcessingPhase.RUNNING,
        stage="render",
        message="Создаём ролики",
        elapsed_seconds=12.0,
        progress_fraction=0.4,
    )
    measure("progress_tick", lambda: project_screen._processing_changed(snapshot))

    projects_screen.deleteLater()
    project_screen.deleteLater()
    application.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
