from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.config import load_config
from app.gui.models import DesktopSettings, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.product_flow import (
    ProcessingIntent,
    apply_resolved_processing_config,
    calibrate_processing_estimate,
    estimate_processing,
    resolve_processing_intent,
)


def _metadata(**extra: object) -> dict[str, object]:
    return {"duration": 600.0, "width": 1920, "height": 1080, "fps": 30.0, **extra}


def test_presets_resolve_to_distinct_real_pipeline_values() -> None:
    fast = resolve_processing_intent(ProcessingIntent(processing_mode="fast", clip_count="1"), _metadata())
    standard = resolve_processing_intent(ProcessingIntent(processing_mode="standard", clip_count="3"), _metadata())
    maximum = resolve_processing_intent(ProcessingIntent(processing_mode="maximum", clip_count="5"), _metadata())

    assert fast.ai_reranking_enabled is False
    assert fast.transformation_strategy == "local_only"
    assert standard.ai_reranking_enabled is True
    assert maximum.candidate_limit > standard.candidate_limit > fast.candidate_limit
    assert maximum.shortlist_size > standard.shortlist_size > fast.shortlist_size
    assert fast.crop_strategy == "fit_blur_background"
    assert standard.crop_strategy == maximum.crop_strategy == "center_crop"

    config = load_config()
    apply_resolved_processing_config(config, maximum)
    assert config.ai_reranking.final_clip_count == 5
    assert config.candidate_generation.max_candidates == maximum.candidate_limit
    assert config.transformation.ai_strategy == "staged"
    assert config.production_render.crop_strategy == "center_crop"
    assert config.product_flow.processing_mode == "maximum"


def test_deep_analysis_auto_is_conservative_and_manual_choice_wins() -> None:
    static = resolve_processing_intent(
        ProcessingIntent(deep_analysis="auto"), _metadata(content_kind="podcast", visual_activity_score=0.1)
    )
    active = resolve_processing_intent(
        ProcessingIntent(deep_analysis="auto"), _metadata(visual_activity_score=0.9)
    )
    manual_off = resolve_processing_intent(
        ProcessingIntent(deep_analysis="off"), _metadata(visual_activity_score=0.9)
    )

    assert static.deep_analysis.resolved is False
    assert "разговорный" in static.deep_analysis.reason
    assert active.deep_analysis.resolved is True
    assert manual_off.deep_analysis.resolved is False
    assert "вашему выбору" in manual_off.deep_analysis.reason


def test_estimate_is_a_range_and_does_not_invent_local_ai_cost() -> None:
    resolved = resolve_processing_intent(ProcessingIntent(processing_mode="standard", clip_count="3"), _metadata())
    local = estimate_processing(resolved, _metadata(), paid_ai_available=False)
    paid = estimate_processing(resolved, _metadata(), paid_ai_available=True)
    longer = estimate_processing(resolved, _metadata(duration=1800.0), paid_ai_available=True)

    assert local.estimated_ai_cost_min is None and local.estimated_ai_cost_max is None
    assert paid.estimated_seconds_min < paid.estimated_seconds_max
    assert paid.estimated_ai_cost_min is not None
    assert longer.estimated_seconds_min > paid.estimated_seconds_min


def test_estimate_calibrates_from_persisted_completed_run_history() -> None:
    resolved = resolve_processing_intent(ProcessingIntent(processing_mode="standard", clip_count="3"), _metadata())
    base = estimate_processing(resolved, _metadata(), paid_ai_available=False)
    history = [
        SimpleNamespace(
            status="completed", started_at="2026-07-01T10:00:00+00:00", finished_at="2026-07-01T10:06:00+00:00",
            settings_snapshot={"product_flow": {"estimate": base.to_dict()}},
        ),
        SimpleNamespace(
            status="completed_with_warnings", started_at="2026-07-02T10:00:00+00:00", finished_at="2026-07-02T10:05:00+00:00",
            settings_snapshot={"product_flow": {"estimate": base.to_dict()}},
        ),
    ]

    calibrated = calibrate_processing_estimate(base, history)

    assert calibrated.confidence == "calibrated"
    assert calibrated.estimated_seconds_max != base.estimated_seconds_max


def test_legacy_project_migrates_to_product_flow_defaults(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    store = DesktopProjectStore(tmp_path / "data")
    project = store.create(source)
    raw = project.to_dict()
    raw["schema_version"] = 1
    raw["settings"] = {"subtitle_style": "clean", "use_cache": True}
    path = store.project_path(project.project_id)
    import json
    path.write_text(json.dumps(raw), encoding="utf-8")

    migrated = store.load(project.project_id)
    assert migrated.schema_version == 3
    assert migrated.settings.processing_mode == "standard"
    assert migrated.settings.subtitle_style == "clean"


def test_desktop_run_persists_intent_resolved_config_and_estimate(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source, source_metadata=_metadata(visual_activity_score=0.8))
    project.settings.processing_mode = "maximum"
    project.settings.deep_analysis = "auto"
    project.settings.platform = "shorts"
    project.settings.clip_count = "5"
    projects.save(project)
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    engine_root = Path(__file__).resolve().parents[1]
    services = DesktopServices(
        engine_root=engine_root,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(engine_root),
        system=SystemService(engine_root),
    )
    monkeypatch.setattr(services.pipeline, "inspect_source", lambda _source: _metadata())

    run, prepared = services.prepare_run(project)

    flow = run.settings_snapshot["product_flow"]
    assert flow["user_intent"]["processing_mode"] == "maximum"
    assert flow["resolved_config"]["platform"]["platform"] == "shorts"
    assert flow["estimate"]["estimated_ai_cost_max"] is None
    runtime = prepared.runtime_config_path.read_text(encoding="utf-8")
    assert "processing_mode: maximum" in runtime
    assert "platform: shorts" in runtime
    assert "final_clip_count: 5" in runtime


def test_render_revision_is_append_only_and_runs_render_stage_only(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source, source_metadata=_metadata())
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    engine_root = Path(__file__).resolve().parents[1]
    history = RunHistoryStore(projects)
    parent = history.create(project, {}, {"path": str(source)}, "0.1.0")
    parent.status = RunStatus.COMPLETED
    parent_output = tmp_path / "parent-run-output"
    parent_output.mkdir()
    parent.settings_snapshot["execution"] = {"output_directory": str(parent_output)}
    history.save(parent)
    services = DesktopServices(
        engine_root=engine_root, settings_store=SettingsStore(data), settings=settings,
        projects=projects, runs=history, pipeline=PipelineFacade(engine_root), system=SystemService(engine_root),
    )
    project.settings.subtitle_style = "dynamic"
    project.settings.platform = "shorts"

    revision, prepared = services.prepare_render_revision(project, parent)

    assert revision.run_kind == RunKind.RENDER_REVISION
    assert revision.parent_run_id == parent.run_id
    assert revision.invalidated_stages == ["production_render"]
    assert revision.cost_estimate == 0.0
    assert "--production-render-only" in prepared.arguments
    assert "--recompute-production-render" in prepared.arguments
    assert "--transform-script" not in prepared.arguments
    assert prepared.runtime_flags["render_only"] == "true"


def test_render_revision_rejects_audio_mode_change(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source, source_metadata=_metadata())
    settings = DesktopSettings.defaults(data); settings.local_test_mode = True
    history = RunHistoryStore(projects)
    parent = history.create(
        project, {"project_options": {"audio_mode": "original"}}, {"path": str(source)}, "0.1.0",
    )
    parent.status = RunStatus.COMPLETED; history.save(parent)
    services = DesktopServices(
        engine_root=Path(__file__).resolve().parents[1], settings_store=SettingsStore(data), settings=settings,
        projects=projects, runs=history, pipeline=PipelineFacade(Path(__file__).resolve().parents[1]),
        system=SystemService(Path(__file__).resolve().parents[1]),
    )
    project.settings.audio_mode = "voiceover"

    import pytest
    from app.gui.services.desktop_project_store import InputValidationError
    with pytest.raises(InputValidationError, match="аудиорежима"):
        services.prepare_render_revision(project, parent)
