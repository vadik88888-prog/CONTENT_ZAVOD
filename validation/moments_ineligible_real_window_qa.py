from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

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


PROJECT_ID = "3c21b1579922410b9b2f33080356f34f"
EXPECTED_MESSAGE = "Найдено 95 моментов, но ни один пока не прошёл проверку качества"


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
    screenshot_path = output_directory / "moments-95-ineligible.png"
    reasons_screenshot_path = output_directory / "moments-blocked-reasons.png"
    blocked_card_path = output_directory / "blocked-card-detail.png"
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
    if not isinstance(candidates, list):
        raise RuntimeError("Verified analysis candidate collection is invalid.")
    reason_counts = Counter(
        str(reason_codes[0])
        for candidate in candidates
        if isinstance(candidate, dict)
        for decision in [candidate.get("eligibility_decision")]
        if isinstance(decision, dict)
        for reason_codes in [decision.get("reason_codes")]
        if isinstance(reason_codes, list) and reason_codes
    )

    with tempfile.TemporaryDirectory(prefix="content-factory-moments-window-qa-") as temporary:
        data_directory = Path(temporary)
        projects = DesktopProjectStore(data_directory)
        clone = DesktopProject.from_dict(original.to_dict())
        clone.project_directory = str(projects.project_directory(project_id))
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
        screen = ProjectScreen(viewmodel)
        screen.setWindowTitle("Content Factory — Moments blocked-candidates QA")
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
        screen.show()
        screen.raise_()
        screen.activateWindow()
        QTest.qWait(900)
        app.processEvents()

        initial_card_count = len(screen._candidate_cards)
        review_scroll = screen.content_scroll.verticalScrollBar()
        review_scroll.setValue(
            max(review_scroll.minimum(), min(review_scroll.maximum(), screen.candidate_review.y() - 16))
        )
        QTest.qWait(500)
        app.processEvents()
        screenshot = app.primaryScreen().grabWindow(int(screen.winId()))
        if screenshot.isNull() or not screenshot.save(str(screenshot_path), "PNG"):
            raise RuntimeError("Could not capture the real ProjectScreen window.")
        screenshot_sha256 = _sha256(screenshot_path)
        first_initial_card = next(iter(screen._candidate_cards.values()))
        list_scroll = screen.review_list_scroll.verticalScrollBar()
        list_scroll.setValue(
            max(list_scroll.minimum(), min(list_scroll.maximum(), first_initial_card.y()))
        )
        QTest.qWait(300)
        app.processEvents()
        reasons_screenshot = app.primaryScreen().grabWindow(int(screen.winId()))
        if reasons_screenshot.isNull() or not reasons_screenshot.save(str(reasons_screenshot_path), "PNG"):
            raise RuntimeError("Could not capture blocked-reason cards in the real ProjectScreen window.")
        reasons_screenshot_sha256 = _sha256(reasons_screenshot_path)
        blocked_card = first_initial_card.grab()
        if blocked_card.isNull() or not blocked_card.save(str(blocked_card_path), "PNG"):
            raise RuntimeError("Could not capture the rendered blocked candidate card.")
        blocked_card_sha256 = _sha256(blocked_card_path)
        view_all_visible = screen.view_all_button.isVisible()
        QTest.mouseClick(screen.view_all_button, Qt.MouseButton.LeftButton)
        if screen._candidate_visible_limit < len(screen._all_candidates_by_id):
            screen.view_all_button.click()
        QTest.qWait(1000)
        app.processEvents()

        quality_notice = screen.findChild(QLabel, "candidateQualityNotice")
        select_recommended = screen.findChild(QPushButton, "selectRecommendedCandidates")
        select_all = screen.findChild(QPushButton, "selectAllCandidates")
        blocked_buttons = [
            screen.findChild(QPushButton, f"blocked-candidate-{candidate_id}")
            for candidate_id in screen._all_candidates_by_id
        ]
        reason_labels = [
            label.text()
            for label in screen.findChildren(QLabel)
            if label.objectName() == "candidateBlockedReason"
        ]
        first_candidate_id = next(iter(screen._all_candidates_by_id))
        first_blocked = screen.findChild(QPushButton, f"blocked-candidate-{first_candidate_id}")
        if first_blocked is None:
            raise AssertionError("First blocked candidate has no blocked action marker.")
        QTest.mouseClick(first_blocked, Qt.MouseButton.LeftButton)
        first_candidate = screen._all_candidates_by_id[first_candidate_id]
        start, end = screen._candidate_range(first_candidate)
        screen._show_candidate_detail(first_candidate, start, end)
        QTest.qWait(250)
        app.processEvents()

        project_unchanged = _sha256(original_project_path) == original_project_sha256
        analysis_unchanged = _sha256(analysis_path) == analysis_sha256
        checks = {
            "qt_windows_platform": app.platformName().casefold() == "windows",
            "native_window_created": int(screen.winId()) > 0,
            "project_screen_visible": screen.isVisible(),
            "source_project_id_matches": clone.project_id == project_id,
            "all_candidates_95": len(screen._all_candidates_by_id) == 95,
            "draftable_candidates_0": len(screen._draftable_candidates_by_id) == 0,
            "initial_cards_paginated": initial_card_count == 12,
            "view_all_visible": view_all_visible,
            "all_95_cards_loaded": len(screen._candidate_cards) == 95,
            "quality_message_visible": bool(quality_notice and quality_notice.isVisible()),
            "quality_message_exact": bool(quality_notice and quality_notice.text() == EXPECTED_MESSAGE),
            "bulk_selection_disabled": bool(
                select_recommended and select_all
                and not select_recommended.isEnabled() and not select_all.isEnabled()
            ),
            "draft_action_hidden": screen.draft_button.isHidden(),
            "no_selectable_candidate_buttons": len(screen._candidate_selection_buttons) == 0,
            "all_blocked_buttons_disabled": len(blocked_buttons) == 95 and all(
                button is not None and not button.isEnabled() for button in blocked_buttons
            ),
            "all_cards_have_primary_reason": len(reason_labels) == 95 and all(
                value.startswith("Почему нельзя создать черновик: ") for value in reason_labels
            ),
            "selection_remains_empty": not clone.review_selected_candidate_ids,
            "blocked_detail_has_no_boundary_controls": screen.candidate_detail.findChild(
                QWidget, "candidateBoundaryControls"
            ) is None,
            "original_project_unchanged": project_unchanged,
            "analysis_artifact_unchanged": analysis_unchanged,
        }
        if not all(checks.values()):
            raise AssertionError(f"Moments real-window checks failed: {checks!r}")

        evidence = {
            "schema_version": "moments-ineligible-real-window.1",
            "captured_at": utc_now(),
            "os": platform.platform(),
            "qt_platform": app.platformName(),
            "source_project": {
                "project_id": project_id,
                "analysis_path": str(analysis_path.relative_to(repository_root)),
                "analysis_sha256": analysis_sha256,
                "candidate_count": len(candidates),
                "draftable_count": len(screen._draftable_candidates_by_id),
                "primary_reason_counts": dict(reason_counts.most_common()),
            },
            "window": {
                "class": type(screen).__name__,
                "title": screen.windowTitle(),
                "native_window_id": int(screen.winId()),
                "width": screen.width(),
                "height": screen.height(),
                "quality_notice_position": {
                    "x": quality_notice.mapTo(screen, QPoint(0, 0)).x(),
                    "y": quality_notice.mapTo(screen, QPoint(0, 0)).y(),
                },
            },
            "observed": {
                "initial_card_count": initial_card_count,
                "loaded_card_count": len(screen._candidate_cards),
                "blocked_button_count": len(blocked_buttons),
                "primary_reason_label_count": len(reason_labels),
                "message": quality_notice.text(),
            },
            "checks": checks,
            "result": "PASS",
            "screenshot": {
                "path": screenshot_path.name,
                "sha256": screenshot_sha256,
                "width": screenshot.width(),
                "height": screenshot.height(),
            },
            "blocked_reasons_screenshot": {
                "path": reasons_screenshot_path.name,
                "sha256": reasons_screenshot_sha256,
                "width": reasons_screenshot.width(),
                "height": reasons_screenshot.height(),
            },
            "blocked_card_screenshot": {
                "path": blocked_card_path.name,
                "sha256": blocked_card_sha256,
                "width": blocked_card.width(),
                "height": blocked_card.height(),
            },
        }
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        screen.close()
        app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture real Windows QA for blocked Moments candidates.")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--project-id", default=PROJECT_ID)
    arguments = parser.parse_args()
    run(arguments.output_directory.resolve(), arguments.project_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
