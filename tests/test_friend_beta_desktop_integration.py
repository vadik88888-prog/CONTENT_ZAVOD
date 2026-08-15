from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app.analysis_artifact import new_analysis_artifact
from app.caption_presets import CAPTION_PRESET_DEFINITIONS
from app.content_profile_taxonomy import CONTENT_PROFILE_PRESETS
from app.font_assets import FONT_ASSET_DEFINITIONS, bundled_font_asset_path
from app.gui.components import VideoPreview
from app.gui.models import DesktopProject, DesktopSettings, ProjectStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel


def _eligibility() -> dict[str, object]:
    return {
        "schema_version": "6D.1",
        "config_version": "friend-beta-test",
        "state": "assessed",
        "eligible": True,
        "reason_codes": [],
        "recoverable_issues": [],
        "required_boundary_actions": [],
        "evidence_refs": [],
    }


def _services(tmp_path: Path) -> tuple[DesktopServices, DesktopProject]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(
        source,
        source_metadata={"duration": 90.0, "width": 1920, "height": 1080, "fps": 30.0},
    )
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    root = Path(__file__).resolve().parents[1]
    services = DesktopServices(
        engine_root=root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(root),
        system=SystemService(root),
    )
    return services, project


def _attach_analysis(tmp_path: Path, project: DesktopProject, *, count: int = 25) -> Path:
    candidates = [
        {
            "candidate_id": f"candidate-{index:03d}",
            "title": f"Момент {index + 1}",
            "start_seconds": float(index),
            "end_seconds": float(index + 18),
            "potential": "high" if index < 3 else "medium",
            "confidence": 0.9,
            "recommended": index < 3,
            "eligibility_decision": _eligibility(),
            "reasons": ["Самостоятельная мысль."],
        }
        for index in range(count)
    ]
    path = tmp_path / "analysis.json"
    new_analysis_artifact(
        analysis_id="analysis-friend-beta",
        project_id=project.project_id,
        source={"id": "source-friend-beta"},
        source_fingerprint="source-friend-beta",
        analysis_fingerprint="analysis-friend-beta-fingerprint",
        work_directory=str(tmp_path),
        candidate_data_ref=str(tmp_path / "candidate-data.json"),
        references={},
        candidates=candidates,
        recommendation={},
        summary={},
        content_profile={},
        duration_seconds=90.0,
        candidate_count=len(candidates),
    ).write(path)
    project.analysis_artifact_path = str(path)
    project.analysis_id = "analysis-friend-beta"
    project.analysis_fingerprint = "analysis-friend-beta-fingerprint"
    project.status = ProjectStatus.ANALYSIS_READY
    project.candidate_states = {
        str(candidate["candidate_id"]): "analyzed" for candidate in candidates
    }
    return path


def test_friend_beta_uses_all_canonical_profiles_and_bundled_caption_fonts() -> None:
    assert len(CONTENT_PROFILE_PRESETS) == 15
    assert len(CAPTION_PRESET_DEFINITIONS) == 7
    for preset in CAPTION_PRESET_DEFINITIONS.values():
        asset = FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id]
        assert asset.redistribution_allowed is True
        assert bundled_font_asset_path(asset).is_file()
        if preset.semantic_font_asset_id:
            semantic = FONT_ASSET_DEFINITIONS[preset.semantic_font_asset_id]
            assert bundled_font_asset_path(semantic).is_file()


def test_legacy_project_caption_migration_follows_existing_creative_family(
    tmp_path: Path,
) -> None:
    services, project = _services(tmp_path)
    raw = project.to_dict()
    raw["settings"].pop("caption_preset_id", None)
    raw["settings"]["subtitle_style"] = "dynamic"

    restored = DesktopProject.from_dict(raw)

    assert restored.settings.caption_preset_id == "accent_yellow"


def test_candidate_override_invalidates_only_its_draft_and_preserves_analysis(
    tmp_path: Path,
) -> None:
    services, project = _services(tmp_path)
    _attach_analysis(tmp_path, project, count=2)
    project.candidate_draft_artifacts = {
        "candidate-000": str(tmp_path / "draft-a.json"),
        "candidate-001": str(tmp_path / "draft-b.json"),
    }
    project.candidate_draft_statuses = {"candidate-000": "ready", "candidate-001": "ready"}
    project.candidate_approval_states = {"candidate-000": "approved", "candidate-001": "approved"}
    project.candidate_export_statuses = {"candidate-000": "ready", "candidate-001": "ready"}
    project.candidate_states = {"candidate-000": "rendered", "candidate-001": "rendered"}
    project.selected_candidate_ids = ["candidate-000", "candidate-001"]
    services.projects.save(project)
    analysis_identity = (
        project.analysis_artifact_path,
        project.analysis_id,
        project.analysis_fingerprint,
    )

    services.update_candidate_creative_override(
        project,
        "candidate-000",
        creative_style="minimal",
        caption_preset_id="word_pop",
        composition_strategy="fit_blur_background",
        same_source_broll_allowed=True,
    )

    assert project.candidate_creative_overrides["candidate-000"] == {
        "creative_style": "minimal",
        "caption_preset_id": "word_pop",
        "composition_strategy": "fit_blur_background",
        "same_source_broll_allowed": True,
    }
    assert project.candidate_draft_statuses == {
        "candidate-000": "pending",
        "candidate-001": "ready",
    }
    assert project.candidate_approval_states["candidate-001"] == "approved"
    assert project.candidate_export_statuses["candidate-001"] == "ready"
    assert project.selected_candidate_ids == ["candidate-001"]
    assert set(project.candidate_draft_artifacts) == {"candidate-000", "candidate-001"}
    assert (
        project.analysis_artifact_path,
        project.analysis_id,
        project.analysis_fingerprint,
    ) == analysis_identity
    assert project.setup_state.needs_new_analysis is False


def test_candidate_override_is_an_isolated_draft_config_overlay(tmp_path: Path) -> None:
    _services_value, project = _services(tmp_path)
    project.candidate_states = {"candidate-a": "draft_ready", "candidate-b": "draft_ready"}
    project.candidate_creative_overrides = {
        "candidate-a": {
            "creative_style": "clean",
            "caption_preset_id": "contrast_box",
            "composition_strategy": "center_crop",
            "same_source_broll_allowed": True,
            "reduced_motion": True,
        }
    }

    effective = PipelineFacade._project_with_candidate_options(project, ["candidate-a"])

    assert effective is not project
    assert effective.settings.subtitle_style == "clean"
    assert effective.settings.caption_preset_id == "contrast_box"
    assert effective.settings.composition_strategy == "center_crop"
    assert effective.settings.same_source_broll_allowed is True
    assert effective.settings.reduced_motion is True
    assert project.settings.subtitle_style == "documentary"
    assert project.settings.same_source_broll_allowed is False
    assert PipelineFacade._project_with_candidate_options(project, ["candidate-a", "candidate-b"]) is project


def test_single_candidate_final_reuses_the_draft_override_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, project = _services(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text("{}", encoding="utf-8")
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = "analysis-friend-beta"
    project.candidate_draft_artifacts = {"candidate-a": str(tmp_path / "draft.json")}
    project.candidate_creative_overrides = {
        "candidate-a": {
            "creative_style": "dynamic",
            "caption_preset_id": "word_pop",
            "composition_strategy": "fit_blur_background",
            "same_source_broll_allowed": False,
            "reduced_motion": True,
        }
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(services.pipeline, "load_verified_analysis", lambda *_args, **_kwargs: object())

    def prepare_paths(effective, _run, _settings):
        captured["settings"] = effective.settings
        config_path = tmp_path / "run" / "runtime-config.yaml"
        config_path.parent.mkdir()
        return (
            project.source,
            SimpleNamespace(production_render=SimpleNamespace(encoder="cpu")),
            SimpleNamespace(processing_mode="deep", platform=SimpleNamespace(platform="universal")),
            config_path,
        )

    monkeypatch.setattr(services.pipeline, "_prepare_mode_paths", prepare_paths)
    monkeypatch.setattr(
        services.pipeline,
        "_compose_approved_draft",
        lambda *_args: tmp_path / "run" / "approved-draft.json",
    )

    services.pipeline.prepare_selected_render(
        project, SimpleNamespace(run_id="final-run"), services.settings, ["candidate-a"],
    )

    effective = captured["settings"]
    assert effective.subtitle_style == "dynamic"
    assert effective.caption_preset_id == "word_pop"
    assert effective.composition_strategy == "fit_blur_background"
    assert effective.same_source_broll_allowed is False
    assert effective.reduced_motion is True


def test_project_open_and_incremental_cards_share_one_verified_analysis_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _services(tmp_path)
    _attach_analysis(tmp_path, project)
    services.projects.save(project)
    loads = 0
    original = services.pipeline.load_verified_analysis

    def counted(current: DesktopProject, *, required: bool = True):
        nonlocal loads
        loads += 1
        return original(current, required=required)

    monkeypatch.setattr(services.pipeline, "load_verified_analysis", counted)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    screen = ProjectScreen(ProjectViewModel(services))
    monkeypatch.setattr(screen._thumbnail_loader, "request", lambda **_kwargs: Path("thumbnail.jpg"))

    try:
        screen.open(project)
        assert loads == 1
        assert len(screen._candidate_cards) == 12

        screen._show_more_candidates()
        app.processEvents()
        assert loads == 1
        assert len(screen._candidate_cards) == 24

        screen._toggle_candidate_selection("candidate-010")
        app.processEvents()
        assert loads == 1
        assert screen.project is not None
        assert screen.project.review_selected_candidate_ids == ["candidate-010"]
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
