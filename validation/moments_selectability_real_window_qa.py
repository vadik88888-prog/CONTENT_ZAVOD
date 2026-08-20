from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from app.gui.components.video_preview import VideoPreview
from app.gui.models import DesktopProject, DesktopSettings
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.styles import load_theme
from app.gui.viewmodels import ProjectViewModel
from app.utils import read_json, utc_now


PROJECT_ID = "de57e7edcd884509868610aee84827c5"
EXPECTED_COUNTS = {"RECOMMENDED": 1, "AVAILABLE": 13, "BLOCKED": 0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _original_data_directory() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if not root:
        raise RuntimeError("Windows application data directory is unavailable.")
    return Path(root) / "ContentFactoryData"


def run(output_directory: Path, project_id: str = PROJECT_ID) -> None:
    app = QApplication.instance() or QApplication([])
    if app.platformName().casefold() != "windows":
        raise RuntimeError(f"Real-window QA requires Windows Qt, got {app.platformName()!r}.")
    app.setStyleSheet(load_theme())

    repository_root = Path(__file__).resolve().parents[1]
    output_directory.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_directory / "moments-gameplay-all-selectable.png"
    card_path = output_directory / "moments-gameplay-risk-card.png"
    evidence_path = output_directory / "runtime-evidence.json"

    original_store = DesktopProjectStore(_original_data_directory())
    original = original_store.load(project_id)
    original_project_path = original_store.project_path(project_id)
    analysis_path = Path(original.analysis_artifact_path or "").resolve()
    if not analysis_path.is_file():
        raise RuntimeError(f"Project analysis artifact is unavailable: {analysis_path}")
    original_project_sha256 = _sha256(original_project_path)
    analysis_sha256 = _sha256(analysis_path)
    analysis = read_json(analysis_path, {})
    candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
    if not isinstance(candidates, list) or len(candidates) != 14:
        raise RuntimeError("Expected the current 14-candidate gameplay analysis artifact.")
    persisted_counts = Counter(
        str(candidate.get("surfacing_state") or "")
        for candidate in candidates if isinstance(candidate, dict)
    )

    with tempfile.TemporaryDirectory(prefix="content-factory-moments-window-qa-") as temporary:
        data_directory = Path(temporary)
        projects = DesktopProjectStore(data_directory)
        clone = DesktopProject.from_dict(original.to_dict())
        clone.project_directory = str(projects.project_directory(project_id))
        clone.review_selected_candidate_ids = []
        clone.selected_candidate_ids = []
        Path(clone.project_directory).mkdir(parents=True, exist_ok=False)
        projects.save(clone)

        settings = DesktopSettings.defaults(data_directory)
        settings.local_test_mode = True
        settings.onboarding_completed = True
        services = DesktopServices(
            engine_root=repository_root,
            settings_store=SettingsStore(data_directory),
            settings=settings,
            projects=projects,
            runs=RunHistoryStore(projects),
            pipeline=PipelineFacade(repository_root),
            system=SystemService(repository_root),
        )
        viewmodel = ProjectViewModel(services)
        original_show_source = VideoPreview.show_source
        VideoPreview.show_source = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        screen = ProjectScreen(viewmodel)
        screen.setWindowTitle("Content Factory — Moments selectability QA")
        screen._results_subflow_override = "candidates"
        screen._bind_source_candidate = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        screen._thumbnail_loader.request = lambda **_kwargs: output_directory / "thumbnail-not-generated.jpg"  # type: ignore[method-assign]

        available = app.primaryScreen().availableGeometry()
        width = min(1500, max(1100, available.width() - 80))
        height = min(920, max(700, available.height() - 80))
        screen.resize(width, height)
        screen.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )
        screen.open(clone)
        screen._results_subflow_override = "candidates"
        screen._project_changed(screen.project)
        screen.show()
        screen.raise_()
        screen.activateWindow()
        QTest.qWait(1200)
        app.processEvents()

        initial_card_count = len(screen._candidate_cards)
        view_all_visible = screen.view_all_button.isVisible()
        QTest.mouseClick(screen.view_all_button, Qt.MouseButton.LeftButton)
        QTest.qWait(700)
        app.processEvents()

        screenshot = app.primaryScreen().grabWindow(int(screen.winId()))
        if screenshot.isNull() or not screenshot.save(str(screenshot_path), "PNG"):
            raise RuntimeError("Could not capture the real ProjectScreen window.")

        quality_notice = screen.findChild(QLabel, "candidateQualityNotice")
        select_recommended = screen.findChild(QPushButton, "selectRecommendedCandidates")
        select_all = screen.findChild(QPushButton, "selectAllCandidates")
        visible_counts = Counter(
            str(item.get("surfacing_state") or "")
            for item in screen._all_candidates_by_id.values()
        )
        recommended_ids = screen._recommended_candidate_ids()
        available_risk_id = next(
            candidate_id
            for candidate_id, item in screen._all_candidates_by_id.items()
            if item.get("surfacing_state") == "AVAILABLE"
            and "SENTENCE_BOUNDARY_UNRECOVERABLE"
            in item.get("editorial_decision", {}).get("soft_issues", [])
        )
        risk_button = screen._candidate_selection_buttons[available_risk_id]
        QTest.mouseClick(risk_button, Qt.MouseButton.LeftButton)
        QTest.qWait(150)
        individual_selection = list(viewmodel.project.review_selected_candidate_ids)

        risk_candidate = screen._all_candidates_by_id[available_risk_id]
        start, end = screen._candidate_range(risk_candidate)
        screen._show_candidate_detail(risk_candidate, start, end)
        QTest.qWait(250)
        app.processEvents()
        risk_card = screen._candidate_cards[available_risk_id].grab()
        if risk_card.isNull() or not risk_card.save(str(card_path), "PNG"):
            raise RuntimeError("Could not capture the selectable risk candidate card.")

        QTest.mouseClick(select_recommended, Qt.MouseButton.LeftButton)
        QTest.qWait(150)
        recommended_selection = list(viewmodel.project.review_selected_candidate_ids)
        QTest.mouseClick(select_all, Qt.MouseButton.LeftButton)
        QTest.qWait(150)
        all_selection = list(viewmodel.project.review_selected_candidate_ids)

        first_candidate = next(iter(screen._all_candidates_by_id.values()))
        feasibility = first_candidate.get("production_feasibility", {})
        blocked_buttons = [
            button for button in screen.findChildren(QPushButton)
            if button.objectName().startswith("blocked-candidate-")
        ]
        reason_labels = [
            label for label in screen.findChildren(QLabel)
            if label.objectName() == "candidateBlockedReason"
        ]
        checks = {
            "qt_windows_platform": app.platformName().casefold() == "windows",
            "native_window_created": int(screen.winId()) > 0,
            "project_screen_visible": screen.isVisible(),
            "current_gameplay_candidates_14": len(screen._all_candidates_by_id) == 14,
            "all_candidates_draftable": len(screen._draftable_candidates_by_id) == 14,
            "initial_cards_paginated": initial_card_count == 12,
            "view_all_visible": view_all_visible,
            "all_14_cards_loaded": len(screen._candidate_cards) == 14,
            "moments_counts_1_13_0": all(
                visible_counts[state] == expected
                for state, expected in EXPECTED_COUNTS.items()
            ),
            "quality_notice_absent": quality_notice is None,
            "bulk_selection_enabled": bool(
                select_recommended and select_all
                and select_recommended.isEnabled() and select_all.isEnabled()
            ),
            "all_candidates_have_selection_buttons": len(screen._candidate_selection_buttons) == 14,
            "no_blocked_buttons": not blocked_buttons,
            "no_blocked_reason_labels": not reason_labels,
            "available_risk_individually_selectable": individual_selection == [available_risk_id],
            "recommended_flow_preserved": recommended_selection == recommended_ids,
            "select_all_flow_preserved": all_selection == list(screen._draftable_candidates_by_id),
            "risk_detail_has_boundary_controls": screen.candidate_detail.findChild(
                QWidget, "candidateBoundaryControls"
            ) is not None,
            "cps_diagnostic_preserved": (
                feasibility.get("status") == "ADVISORY"
                and feasibility.get("diagnostic_status") == "GUARANTEED_BLOCKED"
                and feasibility.get("reason_code") == "CAPTION_CPS_INFEASIBLE"
            ),
            "original_project_unchanged": _sha256(original_project_path) == original_project_sha256,
            "analysis_artifact_unchanged": _sha256(analysis_path) == analysis_sha256,
        }
        if not all(checks.values()):
            raise AssertionError(f"Moments real-window checks failed: {checks!r}")

        evidence = {
            "schema_version": "moments-selectability-real-window.1",
            "captured_at": utc_now(),
            "os": platform.platform(),
            "qt_platform": app.platformName(),
            "source_project": {
                "project_id": project_id,
                "analysis_path": str(analysis_path.relative_to(repository_root)),
                "analysis_sha256": analysis_sha256,
                "analysis_run_id": analysis.get("analysis_run_id"),
                "candidate_count": len(candidates),
                "persisted_editorial_counts": dict(persisted_counts),
            },
            "observed": {
                "visible_counts": dict(visible_counts),
                "draftable_count": len(screen._draftable_candidates_by_id),
                "selection_button_count": len(screen._candidate_selection_buttons),
                "recommended_ids": recommended_ids,
                "available_risk_id": available_risk_id,
                "feasibility": feasibility,
            },
            "window": {
                "class": type(screen).__name__,
                "title": screen.windowTitle(),
                "native_window_id": int(screen.winId()),
                "width": screen.width(),
                "height": screen.height(),
            },
            "checks": checks,
            "result": "PASS",
            "screenshot": {
                "path": screenshot_path.name,
                "sha256": _sha256(screenshot_path),
                "width": screenshot.width(),
                "height": screenshot.height(),
            },
            "risk_card_screenshot": {
                "path": card_path.name,
                "sha256": _sha256(card_path),
                "width": risk_card.width(),
                "height": risk_card.height(),
            },
        }
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        screen.close()
        VideoPreview.show_source = original_show_source  # type: ignore[method-assign]
        app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture real Windows QA for selectable Moments candidates."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--project-id", default=PROJECT_ID)
    arguments = parser.parse_args()
    run(arguments.output_directory.resolve(), arguments.project_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
