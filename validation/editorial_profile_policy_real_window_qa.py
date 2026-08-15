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
from PySide6.QtWidgets import QApplication, QPushButton

from app.analysis_artifact import AnalysisArtifact
from app.editorial_profile_policy import EDITORIAL_PROFILE_POLICY_VERSION
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
from app.media import probe_video
from app.utils import utc_now, write_json


PROJECT_ID = "3c21b1579922410b9b2f33080356f34f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _data_directory() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if not root:
        raise RuntimeError("Windows application data directory is unavailable.")
    return Path(root) / "ContentFactoryData"


def run(output_directory: Path, project_id: str = PROJECT_ID) -> None:
    application = QApplication.instance() or QApplication([])
    if application.platformName().casefold() != "windows":
        raise RuntimeError(f"Real-window QA requires Windows Qt, got {application.platformName()!r}.")
    application.setStyleSheet(load_theme())
    repository_root = Path(__file__).resolve().parents[1]
    output_directory.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_directory / "moments-profile-aware-95.png"
    available_card_path = output_directory / "available-editorial-weakness-card.png"
    blocked_card_path = output_directory / "blocked-integrity-card.png"
    evidence_path = output_directory / "runtime-evidence.json"

    originals = DesktopProjectStore(_data_directory())
    original = originals.load(project_id)
    original_project_path = originals.project_path(project_id)
    analysis_path = Path(original.analysis_artifact_path or "").resolve()
    source_path = original.source.resolve()
    project_sha_before = _sha256(original_project_path)
    analysis_sha_before = _sha256(analysis_path)
    analysis = AnalysisArtifact.read_verified(analysis_path)
    candidate_data = analysis.load_reference("candidate_data")
    raw_candidates = [item for item in candidate_data.get("candidates", []) if isinstance(item, dict)]
    boundary_hashes_before = {
        str(item.get("id") or ""): _json_sha256((item.get("boundary_diagnostics") or {}).get("boundary_decision"))
        for item in raw_candidates
    }

    with tempfile.TemporaryDirectory(prefix="content-factory-editorial-policy-qa-") as temporary:
        data_directory = Path(temporary)
        projects = DesktopProjectStore(data_directory)
        clone = DesktopProject.from_dict(original.to_dict())
        clone.project_directory = str(projects.project_directory(project_id))
        Path(clone.project_directory).mkdir(parents=True, exist_ok=False)
        clone.review_selected_candidate_ids = []
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
        screen.setWindowTitle("Content Factory — Profile-aware Editorial Policy QA")
        screen._thumbnail_loader.request = lambda **_kwargs: output_directory / "thumbnail-not-generated.jpg"  # type: ignore[method-assign]
        available = application.primaryScreen().availableGeometry()
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
        QTest.qWait(1000)
        application.processEvents()

        states = Counter(
            str(item.get("surfacing_state") or "")
            for item in screen._all_candidates_by_id.values()
        )
        profiles = Counter(
            str((item.get("editorial_decision") or {}).get("profile_id") or "")
            for item in screen._all_candidates_by_id.values()
        )
        available_candidate_id = next(
            candidate_id
            for candidate_id, item in screen._all_candidates_by_id.items()
            if item.get("surfacing_state") == "AVAILABLE"
            and "NO_PAYOFF" in (item.get("editorial_decision") or {}).get("soft_issues", [])
        )
        screen._toggle_candidate_selection(available_candidate_id)
        application.processEvents()
        selected_after_editorial_weakness = list(viewmodel.project.review_selected_candidate_ids) if viewmodel.project else []

        screen._view_all_candidates()
        QTest.qWait(900)
        application.processEvents()
        review_scroll = screen.content_scroll.verticalScrollBar()
        review_scroll.setValue(
            max(review_scroll.minimum(), min(review_scroll.maximum(), screen.candidate_review.y() - 16))
        )
        screen.review_list_scroll.verticalScrollBar().setValue(0)
        QTest.qWait(500)
        application.processEvents()
        screenshot = application.primaryScreen().grabWindow(int(screen.winId()))
        if screenshot.isNull() or not screenshot.save(str(screenshot_path), "PNG"):
            raise RuntimeError("Could not capture profile-aware Moments window.")
        blocked_candidate_id = next(
            candidate_id
            for candidate_id, item in screen._all_candidates_by_id.items()
            if item.get("surfacing_state") == "BLOCKED"
        )
        available_card = screen._candidate_cards[available_candidate_id].grab()
        blocked_card = screen._candidate_cards[blocked_candidate_id].grab()
        if available_card.isNull() or not available_card.save(str(available_card_path), "PNG"):
            raise RuntimeError("Could not capture AVAILABLE candidate card.")
        if blocked_card.isNull() or not blocked_card.save(str(blocked_card_path), "PNG"):
            raise RuntimeError("Could not capture BLOCKED candidate card.")
        selectable_buttons = len(screen._candidate_selection_buttons)
        blocked_buttons = sum(
            screen.findChild(QPushButton, f"blocked-candidate-{candidate_id}") is not None
            for candidate_id, item in screen._all_candidates_by_id.items()
            if item.get("surfacing_state") == "BLOCKED"
        )
        summary_text = screen.review_metrics_text.text()
        screen.close()

    analysis_after = AnalysisArtifact.read_verified(analysis_path)
    candidate_data_after = analysis_after.load_reference("candidate_data")
    boundary_hashes_after = {
        str(item.get("id") or ""): _json_sha256((item.get("boundary_diagnostics") or {}).get("boundary_decision"))
        for item in candidate_data_after.get("candidates", []) if isinstance(item, dict)
    }
    checks = {
        "qt_windows_platform": application.platformName().casefold() == "windows",
        "native_window_created": bool(screenshot_path.is_file()),
        "real_source_probe_valid": probe_video(source_path).get("duration", 0) > 0,
        "candidate_count_95": len(analysis.candidates) == 95,
        "recommended_present": states["RECOMMENDED"] > 0,
        "available_present": states["AVAILABLE"] > 0,
        "technical_integrity_failures_6": states["BLOCKED"] == 6,
        "selectable_89": selectable_buttons == 89,
        "blocked_buttons_6": blocked_buttons == 6,
        "auto_resolved_movie_series": profiles == {"movie_series": 95},
        "editorial_weakness_selected": selected_after_editorial_weakness == [available_candidate_id],
        "analysis_checksum_unchanged": analysis_sha_before == _sha256(analysis_path),
        "analysis_snapshot_still_verified": analysis_after.verified_sha256 == analysis.verified_sha256,
        "boundary_decisions_unchanged": boundary_hashes_before == boundary_hashes_after,
        "original_project_unchanged": project_sha_before == _sha256(original_project_path),
    }
    if not all(checks.values()):
        raise AssertionError(f"Profile-aware real-window checks failed: {checks!r}")
    write_json(evidence_path, {
        "schema_version": "editorial-profile-policy-real-window.1",
        "captured_at": utc_now(),
        "policy_version": EDITORIAL_PROFILE_POLICY_VERSION,
        "os": platform.platform(),
        "qt_platform": application.platformName(),
        "source_project": {
            "project_id": project_id,
            "analysis_path": str(analysis_path),
            "analysis_sha256": analysis_sha_before,
            "source_path": str(source_path),
            "source_probe": probe_video(source_path),
        },
        "regression": {
            "candidate_count": len(analysis.candidates),
            "surfacing_counts": dict(states),
            "selectable_count": selectable_buttons,
            "technical_integrity_failure_count": states["BLOCKED"],
            "profile_counts": dict(profiles),
            "selected_editorial_weakness_candidate_id": available_candidate_id,
            "summary_text": summary_text,
        },
        "checks": checks,
        "result": "PASS",
        "screenshot": {
            "path": screenshot_path.name,
            "sha256": _sha256(screenshot_path),
            "width": screenshot.width(),
            "height": screenshot.height(),
        },
        "candidate_cards": {
            "available": {
                "path": available_card_path.name,
                "sha256": _sha256(available_card_path),
                "width": available_card.width(),
                "height": available_card.height(),
            },
            "blocked": {
                "path": blocked_card_path.name,
                "sha256": _sha256(blocked_card_path),
                "width": blocked_card.width(),
                "height": blocked_card.height(),
            },
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run profile-aware Editorial Policy QA on the real 95-candidate series.")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--project-id", default=PROJECT_ID)
    arguments = parser.parse_args()
    run(arguments.output_directory, arguments.project_id)
