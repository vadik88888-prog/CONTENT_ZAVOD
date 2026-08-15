"""Capture the six Friend Beta desktop stages against real persisted artifacts.

The source dataset is copied as metadata only.  Media, Analysis, DraftArtifact,
and ClipResult paths remain the exact persisted identities and are opened
read-only; preview caches and any incidental UI state are confined to a temp
Desktop data directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QAbstractScrollArea, QPushButton

from app.draft_artifact import DraftArtifact
from app.gui.main_window import MainWindow
from app.gui.models import ProcessingPhase, ProcessingSnapshot, ProjectStatus, RunKind
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore, default_data_directory
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme
from app.gui.models import DesktopSettings
from app.utils import stable_file_hash


STAGES = ("source", "settings", "processing", "moments", "drafts", "final")


def _progress(message: str) -> None:
    print(message, flush=True)


def _settle(application: QApplication, *, seconds: float = 0.0, turns: int = 8) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        for _ in range(turns):
            application.processEvents()
        if time.monotonic() >= deadline:
            break
        time.sleep(0.04)


def _isolated_services(
    engine_root: Path,
    source_data: Path,
    project_id: str,
    isolated_data: Path,
) -> DesktopServices:
    source_project = source_data / "projects" / project_id
    target_project = isolated_data / "projects" / project_id
    target_project.mkdir(parents=True, exist_ok=True)
    raw = json.loads((source_project / "project.json").read_text(encoding="utf-8"))
    raw["project_directory"] = str(target_project.resolve())
    (target_project / "project.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_runs = source_project / "runs"
    if source_runs.is_dir():
        for run_json in source_runs.glob("*/run.json"):
            destination = target_project / "runs" / run_json.parent.name / "run.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_json, destination)
    settings = DesktopSettings.defaults(isolated_data)
    settings.data_directory = str(isolated_data.resolve())
    settings.onboarding_completed = True
    store = DesktopProjectStore(isolated_data)
    return DesktopServices(
        engine_root=engine_root,
        settings_store=SettingsStore(isolated_data),
        settings=settings,
        projects=store,
        runs=RunHistoryStore(store),
        pipeline=PipelineFacade(engine_root),
        system=SystemService(engine_root),
    )


def _screen_metrics(window: MainWindow, stage: str) -> dict[str, object]:
    primary = [
        button for button in window.findChildren(QPushButton)
        if button.isVisibleTo(window) and button.objectName() == "primary"
    ]
    horizontal = []
    vertical = []
    for scroll in window.findChildren(QAbstractScrollArea):
        if not scroll.isVisibleTo(window):
            continue
        horizontal.append({
            "name": scroll.objectName() or type(scroll).__name__,
            "maximum": scroll.horizontalScrollBar().maximum(),
        })
        vertical.append({
            "name": scroll.objectName() or type(scroll).__name__,
            "maximum": scroll.verticalScrollBar().maximum(),
        })
    clipped_primary: list[dict[str, object]] = []
    window_rect = window.rect()
    for button in primary:
        point = button.mapTo(window, button.rect().topLeft())
        bounds = button.rect().translated(point)
        if not window_rect.contains(bounds):
            clipped_primary.append({
                "text": button.text(),
                "bounds": [bounds.x(), bounds.y(), bounds.width(), bounds.height()],
                "window": [window_rect.x(), window_rect.y(), window_rect.width(), window_rect.height()],
            })
    if len(primary) > 1:
        raise AssertionError(f"{stage}: more than one visible primary CTA: {[item.text() for item in primary]}")
    if stage != "processing" and len(primary) != 1:
        raise AssertionError(f"{stage}: expected exactly one visible primary CTA")
    if clipped_primary:
        raise AssertionError(f"{stage}: clipped primary CTA: {clipped_primary}")
    if any(item["maximum"] > 0 for item in horizontal):
        raise AssertionError(f"{stage}: horizontal scroll detected: {horizontal}")
    return {
        "primary_cta": [button.text() for button in primary],
        "horizontal_scrolls": horizontal,
        "active_vertical_scrolls": [item for item in vertical if item["maximum"] > 0],
        "window_logical_size": [window.width(), window.height()],
        "device_pixel_ratio": window.devicePixelRatioF(),
    }


def _save_stage(
    application: QApplication,
    window: MainWindow,
    output: Path,
    stage: str,
    label: str,
    metrics: dict[str, object],
    *,
    media_seconds: float = 0.0,
) -> None:
    _settle(application, seconds=media_seconds)
    project_screen = getattr(window, "project_screen", None)
    content_scroll = getattr(project_screen, "content_scroll", None)
    if content_scroll is not None and content_scroll.isVisibleTo(window):
        content_scroll.verticalScrollBar().setValue(0)
        _settle(application, turns=3)
    metrics[stage] = _screen_metrics(window, stage)
    destination = output / f"{stage}-{label}.png"
    native_screen = window.screen() or application.primaryScreen()
    if native_screen is None:
        raise RuntimeError("No Windows screen is available for real-window capture")
    pixmap = native_screen.grabWindow(int(window.winId()))
    if pixmap.isNull() or not pixmap.save(str(destination)):
        raise RuntimeError(f"Could not capture {destination}")
    metrics[stage]["screenshot"] = str(destination.resolve())
    metrics[stage]["screenshot_pixels"] = [pixmap.width(), pixmap.height()]


def _lineage(window: MainWindow) -> dict[str, object]:
    screen = window.project_screen
    project = screen.project
    if project is None:
        raise AssertionError("Project screen has no project")
    analysis = screen.viewmodel.services.pipeline.load_verified_analysis(project, required=True)
    assert analysis is not None
    results = screen._final_output_records(project)
    if not results:
        raise AssertionError("No canonical ClipResult records are available")
    selected = next(
        (item for item in results if item.clip_result_id == project.last_final_result_id),
        results[0],
    )
    draft_path = Path(project.candidate_draft_artifacts[selected.candidate_id]).resolve()
    draft = DraftArtifact.read(draft_path)
    record = next(
        item for item in draft.candidates
        if str(item.get("candidate_id") or "") == selected.candidate_id
    )
    preview = record.get("preview") if isinstance(record, dict) else None
    preview_path = Path(str(preview.get("output_file") or "")).resolve() if isinstance(preview, dict) else Path()
    final_path = Path(selected.output_file).resolve()
    if not preview_path.is_file() or not final_path.is_file():
        raise AssertionError("Persisted Draft/Final media lineage is incomplete")
    return {
        "project_id": project.project_id,
        "analysis": {
            "analysis_id": analysis.analysis_id,
            "analysis_fingerprint": analysis.analysis_fingerprint,
            "artifact_path": str(Path(project.analysis_artifact_path or "").resolve()),
        },
        "draft": {
            "draft_id": draft.draft_id,
            "candidate_id": selected.candidate_id,
            "artifact_path": str(draft_path),
            "artifact_sha256": stable_file_hash(draft_path),
            "creative_preview_path": str(preview_path),
            "creative_preview_sha256": stable_file_hash(preview_path),
        },
        "final": {
            "clip_result_id": selected.clip_result_id,
            "candidate_id": selected.candidate_id,
            "run_id": screen._run_id_for_result(project, selected),
            "output_path": str(final_path),
            "output_sha256": stable_file_hash(final_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=Path, default=default_data_directory())
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    engine_root = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "label": args.label,
        "qt_scale_factor": os.environ.get("QT_SCALE_FACTOR", "1"),
        "stages": {},
    }
    with tempfile.TemporaryDirectory(prefix="friend-beta-real-window-") as raw_temp:
        _progress("prepare isolated metadata")
        services = _isolated_services(
            engine_root,
            args.data_directory.expanduser().resolve(),
            args.project_id,
            Path(raw_temp),
        )
        application = QApplication.instance() or QApplication([])
        application.setStyleSheet(load_theme())
        _progress("create native window")
        window = MainWindow(services)
        window.resize(args.width, args.height)
        window.show()
        _settle(application)

        window.show_projects()
        _progress("capture source")
        _save_stage(application, window, output, "source", args.label, metrics["stages"])

        real_project = services.projects.load(args.project_id)
        screen = window.project_screen
        # Enter the project shell with a fresh settings-state projection.
        # Opening the already-completed QA project here would briefly attach
        # its native video surface behind Settings/Processing and pollute
        # screenshots of the earlier route states.
        window.stack.setCurrentIndex(window.project_index)
        window._set_selected(None)

        settings_project = deepcopy(real_project)
        settings_project.project_id = "friend-beta-settings-preview"
        settings_project.status = ProjectStatus.SOURCE_READY
        settings_project.latest_run_id = None
        settings_project.analysis_artifact_path = None
        settings_project.analysis_id = None
        settings_project.analysis_fingerprint = None
        settings_project.draft_artifact_path = None
        settings_project.draft_id = None
        settings_project.candidate_states = {}
        settings_project.candidate_draft_artifacts = {}
        settings_project.review_selected_candidate_ids = []
        settings_project.selected_candidate_ids = []
        settings_project.last_final_result_id = None
        screen.viewmodel.project = settings_project
        screen.viewmodel.snapshot = ProcessingSnapshot()
        screen.runs = []
        screen._project_changed(settings_project)
        _progress("capture settings")
        _save_stage(application, window, output, "settings", args.label, metrics["stages"])

        processing_project = deepcopy(settings_project)
        processing_project.status = ProjectStatus.ANALYZING
        snapshot = ProcessingSnapshot(
            ProcessingPhase.RUNNING,
            stage="ai_reranking",
            message="Анализируем содержание и события в кадре",
            elapsed_seconds=74.0,
            progress_fraction=0.56,
        )
        screen.viewmodel.project = processing_project
        screen.viewmodel.snapshot = snapshot
        screen._project_changed(processing_project)
        screen._processing_changed(snapshot)
        _progress("capture processing")
        _save_stage(application, window, output, "processing", args.label, metrics["stages"])

        screen.open(real_project)
        real_runs = list(screen.runs)
        analysis_runs = [run for run in real_runs if run.run_kind == RunKind.ANALYSIS]
        draft_runs = [run for run in real_runs if run.run_kind == RunKind.DRAFT]

        moments_project = deepcopy(real_project)
        moments_project.status = ProjectStatus.ANALYSIS_READY
        moments_project.latest_run_id = analysis_runs[-1].run_id if analysis_runs else None
        moments_project.draft_artifact_path = None
        moments_project.draft_id = None
        moments_project.candidate_states = {
            candidate_id: "analyzed" for candidate_id in real_project.candidate_states
        }
        moments_project.candidate_draft_artifacts = {}
        moments_project.candidate_draft_statuses = {
            candidate_id: "pending" for candidate_id in moments_project.candidate_states
        }
        moments_project.candidate_export_statuses = {
            candidate_id: "pending" for candidate_id in moments_project.candidate_states
        }
        moments_project.review_selected_candidate_ids = []
        moments_project.selected_candidate_ids = []
        moments_project.last_final_result_id = None
        screen.viewmodel.project = moments_project
        screen.viewmodel.snapshot = ProcessingSnapshot()
        screen.runs = analysis_runs
        screen._results_subflow_override = "candidates"
        screen._project_changed(moments_project)
        first_moment = next(iter(screen._review_candidates_by_id.values()))
        screen._preview_candidate(first_moment)
        _progress("capture moments")
        _save_stage(application, window, output, "moments", args.label, metrics["stages"], media_seconds=0.8)

        drafts_project = deepcopy(real_project)
        draft_candidate_ids = list(drafts_project.candidate_draft_artifacts)
        drafts_project.status = ProjectStatus.REVIEWING_CANDIDATES
        ready_draft_runs = [run for run in draft_runs if run.status == "draft_ready"]
        drafts_project.latest_run_id = ready_draft_runs[-1].run_id if ready_draft_runs else None
        drafts_project.candidate_states = {
            candidate_id: (
                "draft_ready" if candidate_id in drafts_project.candidate_draft_artifacts else "analyzed"
            )
            for candidate_id in real_project.candidate_states
        }
        drafts_project.candidate_draft_statuses = {
            candidate_id: ("ready" if candidate_id in drafts_project.candidate_draft_artifacts else "pending")
            for candidate_id in drafts_project.candidate_states
        }
        drafts_project.candidate_export_statuses = {
            candidate_id: "pending" for candidate_id in drafts_project.candidate_states
        }
        drafts_project.review_selected_candidate_ids = draft_candidate_ids
        drafts_project.selected_candidate_ids = draft_candidate_ids
        drafts_project.last_final_result_id = None
        screen.viewmodel.project = drafts_project
        screen.viewmodel.snapshot = ProcessingSnapshot()
        screen.runs = [*analysis_runs, *draft_runs]
        screen._results_subflow_override = "drafts"
        screen._project_changed(drafts_project)
        first_draft = next(
            candidate_id for candidate_id in screen._review_visible_candidate_ids
            if candidate_id in screen._draft_preview_paths
        )
        candidate = screen._review_candidates_by_id[first_draft]
        screen._show_draft_preview(
            screen._draft_preview_paths[first_draft],
            str(candidate.get("title") or "Черновик"),
            first_draft,
        )
        _progress("capture drafts")
        _save_stage(application, window, output, "drafts", args.label, metrics["stages"], media_seconds=0.8)

        screen.viewmodel.project = real_project
        screen.viewmodel.snapshot = ProcessingSnapshot()
        screen.runs = real_runs
        screen._results_subflow_override = None
        screen._project_changed(real_project)
        _progress("capture final")
        _save_stage(application, window, output, "final", args.label, metrics["stages"], media_seconds=0.8)
        _progress("verify lineage")
        metrics["lineage"] = _lineage(window)
        metrics["all_six_stages"] = list(metrics["stages"]) == list(STAGES)

        screen.preview.suspend()
        screen.final_results.preview.suspend()
        _settle(application, seconds=0.1, turns=10)
        window.close()
        _settle(application, turns=3)

    destination = output / f"runtime-evidence-{args.label}.json"
    destination.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows CI consoles may still use cp1251 even when the application and
    # evidence file are UTF-8.  Keep the machine-readable stdout portable.
    print(json.dumps(metrics, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
