"""Native-window responsiveness smoke for source probe and run completion."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow
from app.gui.models import DesktopProject, DesktopSettings, ProcessingPhase
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.utils import write_json


def _services(engine_root: Path, data_directory: Path) -> DesktopServices:
    projects = DesktopProjectStore(data_directory)
    settings = DesktopSettings.defaults(data_directory)
    settings.onboarding_completed = True
    return DesktopServices(
        engine_root=engine_root,
        settings_store=SettingsStore(data_directory),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(engine_root),
        system=SystemService(engine_root),
    )


def _wait_for(application: QApplication, predicate, *, timeout: float = 15.0) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    while not predicate():
        application.processEvents()
        if time.perf_counter() >= deadline:
            raise TimeoutError("Native-window smoke timed out.")
        time.sleep(0.001)
    application.processEvents()
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    application = cast(QApplication, QApplication.instance() or QApplication([]))

    with tempfile.TemporaryDirectory(prefix="desktop-p1-", dir=root / "validation") as temporary:
        services = _services(root, Path(temporary))
        window = MainWindow(services)
        window.show()
        application.processEvents()
        assert window.isVisible()

        ticks: list[float] = []
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        timer.start()

        created: list[DesktopProject] = []
        window.projects_viewmodel.project_created.connect(created.append)
        started = time.perf_counter()
        window.projects_viewmodel.create(str(source))
        source_callback_seconds = time.perf_counter() - started
        source_total_seconds = _wait_for(application, lambda: bool(created))
        source_timer_ticks = len(ticks)
        project = created[0]
        assert window.project_screen.project is not None
        assert window.project_screen.project.project_id == project.project_id

        run = services.runs.create(project, {}, {"path": str(source)}, "p1-smoke")
        output_directory = Path(temporary) / "engine-output"
        prepared = PreparedPipelineRun(
            program="python",
            arguments=[],
            working_directory=root,
            state_path=output_directory / "state.json",
            report_path=output_directory / "report.json",
            output_directory=output_directory,
            runtime_config_path=output_directory / "runtime.yaml",
        )
        write_json(prepared.report_path, {
            "output_files": [],
            "warnings": [],
            "production_render": {
                "status": "completed",
                "output_file": str(source),
                "warnings": [],
            },
        })
        viewmodel = window.project_viewmodel
        viewmodel.project = project
        viewmodel.run = run
        viewmodel.prepared = prepared
        viewmodel._launching = True
        finished: list[object] = []
        viewmodel.run_finished.connect(finished.append)

        started = time.perf_counter()
        viewmodel._completed(0)
        completion_callback_seconds = time.perf_counter() - started
        completion_total_seconds = _wait_for(application, lambda: bool(finished))
        completion_timer_ticks = len(ticks) - source_timer_ticks
        timer.stop()

        assert viewmodel.snapshot.phase == ProcessingPhase.COMPLETED
        assert Path(run.artifact_paths[-1]).is_file()
        previews = (window.project_screen.preview, window.project_screen.final_results.preview)
        _wait_for(application, lambda: all(
            preview._poster_process.state() == QProcess.ProcessState.NotRunning
            and preview._proxy_process.state() == QProcess.ProcessState.NotRunning
            for preview in previews
        ))
        if args.screenshot:
            screenshot = args.screenshot.expanduser().resolve()
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(screenshot)):
                raise RuntimeError(f"Could not save smoke screenshot: {screenshot}")

        print(json.dumps({
            "source_callback_seconds": round(source_callback_seconds, 4),
            "source_total_seconds": round(source_total_seconds, 4),
            "completion_callback_seconds": round(completion_callback_seconds, 4),
            "completion_total_seconds": round(completion_total_seconds, 4),
            "source_timer_ticks": source_timer_ticks,
            "completion_timer_ticks": completion_timer_ticks,
            "window_visible": window.isVisible(),
            "source_metadata_duration": project.source_metadata.get("duration"),
            "snapshot_artifacts": len(run.artifact_paths),
        }, ensure_ascii=False))
        window.close()
        application.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
