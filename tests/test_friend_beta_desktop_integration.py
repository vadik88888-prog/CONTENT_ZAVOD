from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QLabel

from app.analysis_artifact import new_analysis_artifact
from app.caption_presets import CAPTION_PRESET_DEFINITIONS
from app.clip_results import ClipResult
from app.config import load_config
from app.content_profile_taxonomy import CONTENT_PROFILE_PRESETS
from app.font_assets import FONT_ASSET_DEFINITIONS, bundled_font_asset_path
from app.gui.components import VideoPreview
from app.gui.components.project_poster import project_poster_path
from app.gui.models import DesktopProject, DesktopSettings, ProjectStatus
from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices, ValidatedSource
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.gui.viewmodels import ProjectViewModel
from app.settings_preview_assets import settings_preview_manifest, settings_preview_path
from app.utils import stable_file_hash


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


def test_settings_manifest_covers_every_current_style_caption_identity() -> None:
    manifest = settings_preview_manifest()
    expected = {
        (style_id, preset_id)
        for style_id in ("minimal", "documentary", "dynamic", "clean")
        for preset_id in CAPTION_PRESET_DEFINITIONS
    }
    records = {
        (item["creative_style_id"], item["caption_preset_id"]): item
        for item in manifest["items"]
    }

    assert set(records) == expected
    assert manifest["provider_calls"] == 0
    assert manifest["brain_rerun"] is False
    assert manifest["vision_rerun"] is False
    for identity, item in records.items():
        path = settings_preview_path(*identity)
        assert path is not None and path.is_file()
        assert stable_file_hash(path) == item["sha256"]
        preset = CAPTION_PRESET_DEFINITIONS[identity[1]]
        assert item["caption_preset_version"] == preset.preset_version
        assert preset.preferred_font_asset_id in item["font_asset_ids"]


def test_project_thumbnail_identity_persists_across_store_restart(tmp_path: Path) -> None:
    services, project = _services(tmp_path)
    project.analysis_artifact_path = str(tmp_path / "analysis.json")
    project.analysis_id = "analysis-kept"
    project.analysis_fingerprint = "analysis-fingerprint-kept"
    poster = project_poster_path(project)
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"exact persisted source frame")

    services.update_project_thumbnail(project, poster)
    reopened_store = DesktopProjectStore(Path(services.settings.data_directory))
    reopened = reopened_store.load(project.project_id)

    assert reopened.thumbnail_path == str(poster.resolve())
    assert Path(reopened.thumbnail_path).read_bytes() == b"exact persisted source frame"
    assert reopened.analysis_artifact_path == str(tmp_path / "analysis.json")
    assert reopened.analysis_id == "analysis-kept"
    assert reopened.analysis_fingerprint == "analysis-fingerprint-kept"


def test_url_download_keeps_youtube_thumbnail_metadata_for_project_poster(tmp_path: Path) -> None:
    services, _project = _services(tmp_path)
    thumbnail_url = "https://i.ytimg.com/vi/example/maxresdefault.jpg"
    project = services.create_url_project(
        "https://www.youtube.com/watch?v=example",
        {
            "title": "Public video",
            "duration": 90.0,
            "extractor": "Youtube",
            "thumbnail_url": thumbnail_url,
            "width": 1280,
            "height": 720,
        },
    )
    downloaded = project.directory / "sources" / "public-video.mp4"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(b"downloaded source")

    services.complete_validated_url_download(
        project,
        ValidatedSource(downloaded.resolve(), {
            "duration": 91.0,
            "width": 1920,
            "height": 1080,
            "video_codec": "h264",
        }),
    )

    reopened = services.projects.load(project.project_id)
    assert reopened.source_metadata["thumbnail_url"] == thumbnail_url
    assert reopened.source_metadata["extractor"] == "Youtube"
    assert reopened.source_metadata["width"] == 1920
    assert reopened.source_spec.metadata["thumbnail_url"] == thumbnail_url


def test_settings_exposes_seven_real_font_cards_and_exact_production_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _services(tmp_path)
    monkeypatch.setattr(VideoPreview, "show_source", lambda *_args, **_kwargs: None)
    media_calls: list[Path | None] = []
    monkeypatch.setattr(
        VideoPreview, "set_file",
        lambda _self, path, **_kwargs: media_calls.append(Path(path) if path else None),
    )
    screen = ProjectScreen(ProjectViewModel(services))
    try:
        screen.open(project)
        screen.show()
        app.processEvents()
        cards = screen._setup_choice_buttons["caption"]
        assert set(cards) == set(CAPTION_PRESET_DEFINITIONS)
        assert screen.setup_caption_preset.isHidden()
        for preset_id, preset in CAPTION_PRESET_DEFINITIONS.items():
            card = cards[preset_id]
            asset = FONT_ASSET_DEFINITIONS[preset.preferred_font_asset_id]
            assert card.font().family() == asset.render_family
            assert asset.file_name in card.toolTip()
            assert preset.label in card.text()

        screen._choose_setup_value(screen.setup_caption_preset, "word_pop")
        app.processEvents()
        expected = settings_preview_path(project.settings.subtitle_style, "word_pop")
        assert expected is not None and expected.is_file()
        assert media_calls[-1] == expected
        record = next(
            item for item in settings_preview_manifest()["items"]
            if item["creative_style_id"] == project.settings.subtitle_style
            and item["caption_preset_id"] == "word_pop"
        )
        assert record["caption_token_id"] == "caption-preset:word_pop:2.1.0"
        assert "font.unbounded.bold" in record["font_asset_ids"]
        assert any(
            "без обработки вашего видео" in label.text()
            for label in screen.setup_summary.findChildren(QLabel)
        )
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


def test_final_metadata_and_warnings_use_the_exact_bound_quality_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process, not an existing QCoreApplication")
    app = QApplication.instance() or QApplication([])
    services, project = _services(tmp_path)
    output = tmp_path / "final.mp4"
    output.write_bytes(b"final")
    report_path = tmp_path / "quality.json"
    report = {
        "artifact_id": "artifact-exact",
        "artifact_path": str(output.resolve()),
        "project_id": project.project_id,
        "candidate_id": "candidate-exact",
        "status": "PASS_WITH_WARNINGS",
        "metrics": {"technical": {"duration": 28.3, "resolution": "1080x1920"}},
        "findings": [{
            "code": "CAPTION_READABILITY_FALLBACK",
            "severity": "warning",
            "user_message": "Raw internal English must not reach Final.",
        }],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = ClipResult(
        candidate_id="candidate-exact",
        output_file=str(output),
        artifact_id="artifact-exact",
        quality_report_path=str(report_path),
        quality_status="PASS_WITH_WARNINGS",
    )
    screen = ProjectScreen(ProjectViewModel(services))
    try:
        assert screen._quality_media_for_result(project, result) == {
            "duration": 28.3, "width": 1080, "height": 1920,
        }
        assert "Субтитры упрощены" in screen._quality_finding_message(report["findings"][0])
        assert "Raw internal" not in screen._quality_finding_message(report["findings"][0])

        report["candidate_id"] = "candidate-other"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        assert screen._quality_media_for_result(project, result) == {}
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


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


def test_draft_override_is_transactional_until_explicit_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, project = _services(tmp_path)
    _attach_analysis(tmp_path, project, count=1)
    previous = tmp_path / "previous-draft.json"
    previous.write_text("{}", encoding="utf-8")
    project.candidate_draft_artifacts = {"candidate-000": str(previous)}
    project.candidate_draft_statuses = {"candidate-000": "ready"}
    project.candidate_approval_states = {"candidate-000": "pending"}
    project.candidate_export_statuses = {"candidate-000": "pending"}
    project.candidate_states = {"candidate-000": "draft_ready"}
    services.projects.save(project)
    viewmodel = ProjectViewModel(services)
    viewmodel.project = project
    launches: list[list[str]] = []
    monkeypatch.setattr(viewmodel, "build_drafts", lambda ids: launches.append(list(ids)))

    viewmodel.revise_draft("candidate-000", caption_preset_id="word_pop")

    assert launches == []
    assert project.candidate_creative_overrides["candidate-000"] == {
        "caption_preset_id": "word_pop",
    }
    assert project.candidate_draft_statuses["candidate-000"] == "pending"
    assert project.candidate_draft_artifacts["candidate-000"] == str(previous)
    assert previous.is_file()


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


def test_advanced_controls_persist_and_reach_existing_runtime_owners(tmp_path: Path) -> None:
    services, project = _services(tmp_path)

    services.update_project_options(
        project,
        processing_mode="maximum",
        deep_analysis="on",
        platform="reels",
        clip_count="5",
        audio_mode="original_enhanced",
        composition_strategy="fit_blur_background",
        same_source_broll_allowed=True,
        subtitles_enabled=False,
        reduced_motion=True,
        use_cache=False,
    )

    persisted = services.projects.load(project.project_id)
    _intent, resolved, _estimate = services.pipeline.plan_processing(persisted, services.settings)
    config = load_config(services.pipeline._base_config(services.settings))
    services.pipeline._apply_project_options(config, persisted, services.settings, resolved)

    assert persisted.settings.processing_mode == "maximum"
    assert persisted.settings.deep_analysis == "on"
    assert persisted.settings.platform == "reels"
    assert persisted.settings.clip_count == "5"
    assert persisted.settings.audio_mode == "original_enhanced"
    assert persisted.settings.composition_strategy == "fit_blur_background"
    assert persisted.settings.same_source_broll_allowed is True
    assert persisted.settings.subtitles_enabled is False
    assert persisted.settings.reduced_motion is True
    assert persisted.settings.use_cache is False
    assert config.product_flow.processing_mode == "maximum"
    assert config.product_flow.deep_analysis_requested == "on"
    assert config.product_flow.deep_analysis_resolved is True
    assert config.product_flow.platform == "reels"
    assert config.product_flow.clip_count == 5
    assert config.product_flow.audio_mode == "original_enhanced"
    assert config.product_flow.reduced_motion is True
    assert config.production.audio_mode == "original_enhanced"
    assert config.production_render.crop_strategy == "fit_blur_background"
    assert config.production_render.same_source_broll_allowed is True
    assert config.production_render.subtitles_enabled is False
    assert config.production_render.cache_enabled is False


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
