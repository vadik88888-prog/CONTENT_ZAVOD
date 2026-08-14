from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.analysis_artifact import AnalysisArtifact
from app.config import load_config
from app.gui.models import DesktopSettings, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.product_flow import (
    CostPricing,
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
    assert config.virality.enabled is True
    assert config.virality.semantic_ai_mode == "auto"


def test_editorial_intent_and_profile_override_resolve_into_existing_pipeline_config() -> None:
    intent = ProcessingIntent(
        editorial_intent="  Найти практические ошибки и сильный вывод  ",
        profile_format_override="gameplay",
        profile_editorial_mode_override="commentary",
        profile_domain_override="gaming",
        profile_traits_override=("visual_led", "high_pacing"),
    )
    resolved = resolve_processing_intent(intent, _metadata())
    config = load_config()
    apply_resolved_processing_config(config, resolved)

    assert resolved.editorial_intent == "Найти практические ошибки и сильный вывод"
    assert config.content_understanding.editorial_intent == resolved.editorial_intent
    assert config.content_understanding.manual_override == {
        "format": "gameplay", "editorial_mode": "commentary", "domain": "gaming",
        "traits": ["visual_led", "high_pacing"],
    }
    config.validate()


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


def test_cost_preview_uses_active_tariffs_and_explains_its_drivers() -> None:
    resolved = resolve_processing_intent(
        ProcessingIntent(processing_mode="maximum", deep_analysis="on", clip_count="5", audio_mode="voiceover"),
        _metadata(),
    )
    low_tariff = CostPricing(0.0000001, 0.000001, 5.0, ai_available=True, tts_available=True)
    high_tariff = CostPricing(0.0000005, 0.000005, 25.0, ai_available=True, tts_available=True)

    low = estimate_processing(resolved, _metadata(), paid_ai_available=True, pricing=low_tariff)
    high = estimate_processing(resolved, _metadata(), paid_ai_available=True, pricing=high_tariff)

    assert low.estimated_ai_cost_min is not None
    assert high.estimated_ai_cost_max is not None
    assert high.estimated_ai_cost_max > low.estimated_ai_cost_max
    assert any("кадр" in item for item in high.cost_drivers)
    assert "тариф" in high.cost_note.lower()


def test_auto_recommendation_uses_available_content_signals_and_stays_conservative_when_unknown() -> None:
    gameplay = resolve_processing_intent(
        ProcessingIntent(deep_analysis="auto"), _metadata(title="PUBG gameplay decisive match")
    )
    speech = resolve_processing_intent(
        ProcessingIntent(deep_analysis="auto"), _metadata(speech_ratio=0.9, visual_activity_score=0.2)
    )
    unknown = resolve_processing_intent(ProcessingIntent(deep_analysis="auto"), {})

    assert gameplay.deep_analysis.resolved is True
    assert speech.deep_analysis.resolved is False
    assert unknown.deep_analysis.estimated_benefit == "unknown"


def test_auto_preset_recommendation_resolves_effective_preset_with_provenance() -> None:
    resolved = resolve_processing_intent(
        ProcessingIntent(
            subtitle_preset="documentary",
            preset_selection_mode="auto",
        ),
        _metadata(detected_content_type="gameplay"),
    )

    assert resolved.configured_subtitle_preset == "documentary"
    assert resolved.recommended_subtitle_preset == "minimal"
    assert resolved.subtitle_preset == "minimal"
    assert resolved.preset_selection_mode == "auto"
    assert resolved.preset_provenance == "content_recommendation"
    assert resolved.to_dict()["effective_subtitle_preset"] == "minimal"


def test_auto_preset_uses_structured_effective_profile_before_legacy_projection() -> None:
    resolved = resolve_processing_intent(
        ProcessingIntent(preset_selection_mode="auto"),
        _metadata(
            detected_content_type="podcast",
            effective_profile={
                "format": "gameplay",
                "editorial_mode": "commentary",
                "domain": "gaming",
                "traits": ["visual_led"],
            },
        ),
    )

    assert resolved.recommended_subtitle_preset == "minimal"
    assert resolved.subtitle_preset == "minimal"


def test_explicit_preset_beats_content_recommendation_and_is_applied_to_runtime() -> None:
    resolved = resolve_processing_intent(
        ProcessingIntent(
            subtitle_preset="dynamic",
            preset_selection_mode="explicit",
        ),
        _metadata(detected_content_type="podcast"),
    )
    config = load_config()
    apply_resolved_processing_config(config, resolved)

    assert resolved.recommended_subtitle_preset == "clean"
    assert resolved.subtitle_preset == "dynamic"
    assert resolved.preset_provenance == "explicit_selection"
    assert config.production_render.subtitle_style == "dynamic"
    assert config.product_flow.configured_subtitle_preset == "dynamic"
    assert config.product_flow.subtitle_preset == "dynamic"
    assert config.product_flow.preset_selection_mode == "explicit"


def test_legacy_processing_intent_without_mode_remains_explicit_and_pinned() -> None:
    intent = ProcessingIntent.from_dict({"subtitle_preset": "clean"})
    resolved = resolve_processing_intent(
        intent,
        _metadata(detected_content_type="gameplay"),
    )

    assert intent.preset_selection_mode == "explicit"
    assert resolved.recommended_subtitle_preset == "minimal"
    assert resolved.subtitle_preset == "clean"


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
    assert migrated.settings.preset_selection_mode == "explicit"


def test_desktop_run_persists_intent_resolved_config_and_estimate(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(
        source,
        source_metadata=_metadata(
            visual_activity_score=0.8,
            detected_content_type="vlog",
        ),
    )
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
    assert flow["resolved_config"]["effective_subtitle_preset"] == "dynamic"
    assert flow["resolved_config"]["preset_provenance"] == "content_recommendation"
    assert flow["estimate"]["estimated_ai_cost_max"] is None
    runtime = prepared.runtime_config_path.read_text(encoding="utf-8")
    assert "processing_mode: maximum" in runtime
    assert "platform: shorts" in runtime
    assert "subtitle_style: dynamic" in runtime
    assert "preset_selection_mode: auto" in runtime
    assert "final_clip_count: 5" in runtime


def test_manual_desktop_preset_choice_persists_explicit_override(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(
        source,
        source_metadata=_metadata(detected_content_type="podcast"),
    )
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    root = Path(__file__).resolve().parents[1]
    services = DesktopServices(
        engine_root=root, settings_store=SettingsStore(data), settings=settings,
        projects=projects, runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(root), system=SystemService(root),
    )

    services.update_project_options(project, subtitle_style="minimal")
    _intent, resolved, _estimate = services.pipeline.plan_processing(project, settings)
    reloaded = projects.load(project.project_id)

    assert project.settings.preset_selection_mode == "explicit"
    assert resolved.recommended_subtitle_preset == "clean"
    assert resolved.subtitle_preset == "minimal"
    assert reloaded.settings.preset_selection_mode == "explicit"


def test_normal_draft_planning_uses_persisted_content_profile_for_auto_preset(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(
        source,
        source_metadata=_metadata(title="gameplay filename"),
    )
    analysis_path = tmp_path / "analysis.json"
    artifact = AnalysisArtifact(
        analysis_id="analysis-preset",
        project_id=project.project_id,
        created_at="2026-08-12T00:00:00+00:00",
        source={"id": "source-preset"},
        source_fingerprint="source-fingerprint",
        analysis_fingerprint="analysis-fingerprint",
        work_directory=str(tmp_path / "work"),
        candidate_data_ref=str(tmp_path / "candidates.json"),
        references={},
        candidates=[],
        recommendation={},
        summary={},
        content_profile={"detected_content_type": "podcast"},
        duration_seconds=600.0,
    )
    artifact.write(analysis_path)
    project.analysis_artifact_path = str(analysis_path)
    project.analysis_id = artifact.analysis_id
    projects.save(project)
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    root = Path(__file__).resolve().parents[1]
    pipeline = PipelineFacade(root)

    _intent, resolved, _estimate = pipeline.plan_processing(project, settings)

    assert resolved.recommended_subtitle_preset == "clean"
    assert resolved.subtitle_preset == "clean"
    assert resolved.preset_provenance == "content_recommendation"


def test_setup_changes_explain_when_analysis_is_reused_or_needed_again(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "desktop-data"
    projects = DesktopProjectStore(data)
    project = projects.create(source, source_metadata=_metadata())
    project.analysis_artifact_path = str(tmp_path / "analysis.json")
    settings = DesktopSettings.defaults(data)
    settings.local_test_mode = True
    root = Path(__file__).resolve().parents[1]
    services = DesktopServices(
        engine_root=root, settings_store=SettingsStore(data), settings=settings, projects=projects,
        runs=RunHistoryStore(projects), pipeline=PipelineFacade(root), system=SystemService(root),
    )

    services.update_project_options(project, processing_mode="maximum")
    assert project.setup_state.needs_new_analysis is True
    assert "новый анализ" in project.setup_state.change_summary

    services.update_project_options(project, platform="shorts")
    assert project.setup_state.needs_new_analysis is False
    assert project.setup_state.reused_stages == ["сохранённый анализ", "найденные моменты"]


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
    assert prepared.arguments[prepared.arguments.index("--project-id") + 1] == project.project_id
    assert prepared.arguments[prepared.arguments.index("--upstream-run-directory") + 1] == str(parent_output)
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
