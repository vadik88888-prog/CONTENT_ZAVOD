"""Native Windows QA for the user-facing AI cost states.

No provider request is made: a process-local syntactically valid key only
admits the existing preflight estimator for the estimate-state check.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtWidgets import QApplication, QFrame

from app.gui.models import DesktopSettings, ProjectStatus, RunKind, RunStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme
from app.gui.viewmodels import ProjectViewModel
from app.utils import utc_now


def _settle(application: QApplication, seconds: float = 0.35) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.02)


def _card_lines(card: QFrame) -> list[str]:
    layout = card.layout()
    return [
        str(item.widget().text())
        for index in range(1, layout.count())
        if (item := layout.itemAt(index)).widget() is not None
    ]


def _capture(screen: ProjectScreen, target: Path) -> None:
    image = screen.grab()
    if image.isNull() or not image.save(str(target), "PNG"):
        raise RuntimeError(f"Could not capture native AI cost window: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    application = QApplication.instance() or QApplication([])
    if application.platformName().casefold() != "windows":
        raise RuntimeError(f"Real-window QA requires Windows Qt, got {application.platformName()!r}.")
    application.setStyleSheet(load_theme())
    previous_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-" + "x" * 32
    try:
        with tempfile.TemporaryDirectory(prefix="content-factory-ai-cost-qa-") as raw_data:
            data = Path(raw_data)
            source = data / "source.mp4"
            source.write_bytes(b"fixture source")
            projects = DesktopProjectStore(data)
            project = projects.create(
                source,
                source_metadata={"duration": 90.0, "width": 1920, "height": 1080, "fps": 30.0},
            )
            settings = DesktopSettings.defaults(data)
            services = DesktopServices(
                engine_root=root,
                settings_store=SettingsStore(data),
                settings=settings,
                projects=projects,
                runs=RunHistoryStore(projects),
                pipeline=PipelineFacade(root),
                system=SystemService(root),
            )
            screen = ProjectScreen(ProjectViewModel(services))
            screen.resize(1280, 860)
            screen.open(project)
            screen.show()
            _settle(application)
            estimate_lines = _card_lines(screen.estimate)
            expected_line = next((line for line in estimate_lines if line.startswith("Ожидаемая стоимость AI ≈ $")), "")
            if not expected_line:
                raise AssertionError(f"AI estimate is not visible before launch: {estimate_lines!r}")
            screen.content_scroll.ensureWidgetVisible(screen.setup_estimate, 16, 80)
            _settle(application)
            _capture(screen, output / "ai-cost-estimate.png")

            run = services.runs.create(project, {}, {}, "0.1.0", run_kind=RunKind.ANALYSIS)
            run.status = RunStatus.ANALYSIS_READY
            run.finished_at = utc_now()
            run.actual_cost = 0.1275
            services.runs.save(run)
            project.latest_run_id = run.run_id
            project.status = ProjectStatus.ANALYSIS_READY
            projects.save(project)
            screen.open(project)
            _settle(application)
            actual_lines = _card_lines(screen.estimate)
            if actual_lines != ["Фактическая стоимость AI: $0.13"]:
                raise AssertionError(f"Actual provider cost is not shown honestly: {actual_lines!r}")
            if RunHistoryStore(projects).load(project.project_id, run.run_id).actual_cost != 0.1275:
                raise AssertionError("Actual provider cost was not retained after run-history restart.")
            if screen.review_ai_cost.text() != "Фактическая стоимость AI: $0.13":
                raise AssertionError("Results AI cost component did not receive the actual provider cost.")
            screen.content_scroll.ensureWidgetVisible(screen.setup_estimate, 16, 80)
            _settle(application)
            _capture(screen, output / "ai-cost-actual.png")

            rerender = services.runs.create(project, {}, {}, "0.1.0", run_kind=RunKind.RENDER_REVISION, parent_run_id=run.run_id)
            rerender.status = RunStatus.COMPLETED
            rerender.finished_at = utc_now()
            services.runs.save(rerender)
            project.latest_run_id = rerender.run_id
            project.status = ProjectStatus.COMPLETED
            projects.save(project)
            screen.open(project)
            _settle(application)
            rerender_lines = _card_lines(screen.estimate)
            if rerender_lines != ["AI не используется"]:
                raise AssertionError(f"Render-only run shows a fictional AI charge: {rerender_lines!r}")
            if screen.review_ai_cost.text() != "AI не используется":
                raise AssertionError("Results AI cost component did not clear for a render-only run.")
            screen.content_scroll.ensureWidgetVisible(screen.setup_estimate, 16, 80)
            _settle(application)
            _capture(screen, output / "ai-cost-rerender.png")
            screen.close()
    finally:
        if previous_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_key
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
