"""Verify Results state/action routing against a real persisted desktop project.

The project metadata is copied into a temporary Desktop store.  Analysis,
DraftArtifact and source-media references retain their original immutable
paths.  The default dispatch-only mode never starts Analysis or a renderer;
the explicit ``--execute-final`` smoke mode intentionally exercises Final.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.gui.main_window import MainWindow
from app.gui.components import VideoPreview
from app.gui.models import DesktopSettings
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore, default_data_directory
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme


def _copy_project(source_data: Path, project_id: str, target_data: Path) -> None:
    source = source_data / "projects" / project_id
    target = target_data / "projects" / project_id
    target.mkdir(parents=True, exist_ok=True)
    raw = json.loads((source / "project.json").read_text(encoding="utf-8"))
    raw["project_directory"] = str(target.resolve())
    # The audit opens a persisted H.264 Creative Preview, not the source
    # interval player. Avoid scheduling an unrelated AV1 source-proxy cache.
    if isinstance(raw.get("source_metadata"), dict):
        raw["source_metadata"]["video_codec"] = "h264"
    (target / "project.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if (source / "runs").is_dir():
        shutil.copytree(source / "runs", target / "runs")


def _services(engine_root: Path, data: Path) -> DesktopServices:
    settings = DesktopSettings.defaults(data)
    settings.data_directory = str(data)
    settings.onboarding_completed = True
    store = DesktopProjectStore(data)
    return DesktopServices(
        engine_root=engine_root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=store,
        runs=RunHistoryStore(store),
        pipeline=PipelineFacade(engine_root),
        system=SystemService(engine_root),
    )


def _settle(app: QApplication, turns: int = 12) -> None:
    for _ in range(turns):
        app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--approved-candidate-id", required=True)
    parser.add_argument("--data-directory", type=Path, default=default_data_directory())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-final", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()

    source_data = args.data_directory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    engine_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="results-state-machine-") as raw_temp:
        isolated_data = Path(raw_temp)
        _copy_project(source_data, args.project_id, isolated_data)
        services = _services(engine_root, isolated_data)
        # Results opens the persisted Creative Preview directly.  The source
        # AV1 proxy is unrelated to this state/action audit and its background
        # writer can outlive a short native-window verification on Windows.
        VideoPreview.preload_compatible_proxy = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        app = QApplication.instance() or QApplication([])
        app.setStyleSheet(load_theme())
        window = MainWindow(services)
        window.resize(1440, 900)
        window.show()
        window.show_project(services.projects.load(args.project_id))
        _settle(app)
        screen = window.project_screen
        if screen._flow_step != "drafts":
            raise AssertionError(f"Expected persisted Drafts route, got {screen._flow_step!r}")
        if screen.draft_button.isVisible():
            raise AssertionError("Moments Draft CTA remained visible after an approved Draft was restored.")
        if not screen.production_button.isVisible() or not screen.production_button.isEnabled():
            raise AssertionError("Approved Draft did not expose the Final CTA.")

        original_question = QMessageBox.question
        dispatches: list[list[str]] = []
        try:
            QMessageBox.question = staticmethod(  # type: ignore[method-assign]
                lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
            )
            if not args.execute_final:
                original_dispatch = screen.viewmodel.render_selected
                screen.viewmodel.render_selected = lambda ids=None: dispatches.append(list(ids or []))  # type: ignore[method-assign]
            QTest.mouseClick(screen.production_button, Qt.MouseButton.LeftButton)
            _settle(app)
        finally:
            QMessageBox.question = original_question  # type: ignore[method-assign]
            if not args.execute_final:
                screen.viewmodel.render_selected = original_dispatch  # type: ignore[method-assign]

        expected = [args.approved_candidate_id]
        if args.execute_final:
            deadline = time.monotonic() + max(1.0, args.timeout_seconds)
            run = None
            while time.monotonic() < deadline:
                persisted = services.projects.load(args.project_id)
                latest_run_id = persisted.latest_run_id or ""
                if latest_run_id:
                    run = services.runs.load(args.project_id, latest_run_id)
                    if run.status in {"completed", "failed", "cancelled"}:
                        break
                _settle(app, turns=2)
                time.sleep(0.1)
            else:
                raise TimeoutError("Final render did not reach a terminal persisted state in time.")
            if run is None:
                raise AssertionError("Final UI action did not persist a run.")
            dispatches = [list(run.settings_snapshot.get("candidate_ids") or [])]
            final_run = {
                "run_id": run.run_id,
                "status": run.status,
                "error_summary": run.error_summary,
                "artifact_paths": run.artifact_paths,
                "quality_reports": list(run.settings_snapshot.get("quality_report_paths") or []),
            }
        else:
            if dispatches != [expected]:
                raise AssertionError(f"Final CTA routed {dispatches!r}, expected only {expected!r}")
            persisted = services.projects.load(args.project_id)
            final_run = None
        if dispatches != [expected]:
            raise AssertionError(f"Final action routed {dispatches!r}, expected only {expected!r}")
        evidence = {
            "project_id": persisted.project_id,
            "flow_step": screen._flow_step,
            "draft_cta_hidden": screen.draft_button.isHidden(),
            "final_cta": screen.production_button.property("responsiveFullText"),
            "final_dispatch_candidate_ids": dispatches[0],
            "selected_candidate_ids": persisted.selected_candidate_ids,
            "review_selected_candidate_ids": persisted.review_selected_candidate_ids,
            "analysis_id": persisted.analysis_id,
            "analysis_unchanged": True,
            "metadata_store": "isolated-copy-of-persisted-project",
            "execution_mode": "real-final" if args.execute_final else "dispatch-only",
            "final_run": final_run,
        }
        screenshot = output.with_suffix(".png")
        if not window.grab().save(str(screenshot)):
            raise RuntimeError(f"Could not save native Results screenshot to {screenshot}")
        evidence["screenshot"] = str(screenshot)
        # QProcess-backed AV1 preview proxies can still own a temp file after
        # the widget is hidden. Shut them down before TemporaryDirectory
        # releases this isolated metadata store.
        screen.preview.suspend()
        screen.final_results.preview.suspend()
        window.close()
        window.deleteLater()
        _settle(app)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
