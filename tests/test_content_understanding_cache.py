from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.config import AppConfig
from app.pipeline import Pipeline
from app.utils import read_json, write_json


def _wire_pipeline(monkeypatch, calls: dict[str, int]) -> None:
    import app.pipeline as pipeline_module

    real_profile = pipeline_module.build_video_content_profile
    real_map = pipeline_module.build_global_content_map
    real_boundaries = pipeline_module.generate_semantic_candidates

    def prepare(_source: Path, work: Path) -> dict:
        audio = work / "audio.wav"; audio.write_bytes(b"wav")
        result = {"duration": 25.0, "width": 1920, "height": 1080, "fps": 30, "audio_streams": 1, "audio_path": str(audio)}
        write_json(work / "metadata.json", result)
        return result

    def transcribe(_audio, source_id, duration, config, destination):
        result = {
            "source_id": source_id, "language": "ru", "duration": duration,
            "segments": [{"start": 1.0, "end": 20.0, "text": "Завершённая самостоятельная мысль с понятным итогом."}],
            "words": [], "model": config.whisper_model, "runtime": {"device": "cpu"},
        }
        write_json(destination, result)
        destination.with_suffix(".txt").write_text(result["segments"][0]["text"], encoding="utf-8")
        return result

    def render(_source, _item, _ass, destination, _config):
        destination.write_bytes(b"mp4")
        return destination, False, None

    def profile(*args, **kwargs):
        calls["profile"] += 1
        return real_profile(*args, **kwargs)

    def content_map(*args, **kwargs):
        calls["map"] += 1
        return real_map(*args, **kwargs)

    def boundaries(*args, **kwargs):
        calls["boundaries"] += 1
        return real_boundaries(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "prepare_media", prepare)
    monkeypatch.setattr(pipeline_module, "transcribe", transcribe)
    monkeypatch.setattr(pipeline_module, "render_clip", render)
    monkeypatch.setattr(pipeline_module, "build_video_content_profile", profile)
    monkeypatch.setattr(pipeline_module, "build_global_content_map", content_map)
    monkeypatch.setattr(pipeline_module, "generate_semantic_candidates", boundaries)


def test_content_artifacts_are_source_cached_and_boundary_config_isolated(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)

    first = AppConfig(score_threshold=0)
    first_result = Pipeline(tmp_path, first, mock_ai=True).run(input_path=str(source))
    render_only_options = AppConfig(score_threshold=0)
    render_only_options.production_render.subtitle_style = "clean"
    render_only_options.production_render.crop_strategy = "center_crop"
    Pipeline(tmp_path, render_only_options, mock_ai=True).run(input_path=str(source))
    changed_boundary = AppConfig(score_threshold=0)
    changed_boundary.content_understanding.target_tail_padding_seconds = 0.9
    Pipeline(tmp_path, changed_boundary, mock_ai=True).run(input_path=str(source))

    assert calls == {"profile": 1, "map": 1, "boundaries": 2}
    report = read_json(first_result.report_path, {})
    understanding = report["content_understanding"]
    assert understanding["coverage_map"]["schema_version"] == "5A.1"
    assert understanding["clip_count_recommendation"]["post_analysis"] is True
    assert Path(understanding["coverage_map_ref"]).is_file()
    manifest = read_json(first_result.output_directory / "manifest.json", {})
    manifest_analysis = manifest["content_understanding"]
    assert manifest_analysis["content_profile_ref"].endswith("video_content_profile.json")
    assert manifest_analysis["analysis_fingerprint"]

    changed_transcript = AppConfig(score_threshold=0, whisper_model="base")
    Pipeline(tmp_path, changed_transcript, mock_ai=True).run(input_path=str(source))
    assert calls["profile"] == 2
    assert calls["map"] == 2


def test_analysis_only_writes_versioned_review_artifact_without_delivery(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)

    def delivery_must_not_start(*_args, **_kwargs):
        raise AssertionError("Analysis-only mode must not start delivery work.")

    import app.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module.Pipeline, "_transform_selected", delivery_must_not_start)

    result = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))

    assert result.terminal_status == "analysis_ready"
    assert result.output_files == []
    assert result.analysis_path and result.analysis_path.is_file()
    analysis = read_json(result.analysis_path, {})
    assert analysis["schema_version"] == "1.0"
    assert analysis["status"] == "analysis_ready"
    assert analysis["analysis_id"] == result.analysis_id
    assert analysis["source_fingerprint"]
    assert analysis["candidate_data_ref"].endswith("candidates.scored.json")
    assert Path(analysis["candidate_data_ref"]).is_file()
    assert analysis["candidates"]
    first = analysis["candidates"][0]
    assert {"story_unit_id", "chapter_id", "start_seconds", "end_seconds", "duration_seconds"} <= set(first)
    assert {"title", "core_idea", "hook_summary", "payoff_summary", "confidence", "potential"} <= set(first)
    assert {"reasons", "risks", "feature_profile", "boundary_evidence", "preview", "recommended"} <= set(first)
    assert first["preview"]["thumbnail"]["kind"] == "lazy_source_frame"
    assert analysis["summary"]["potential_counts"]
    assert analysis["content_profile"]
    assert analysis["duration_seconds"] == 25.0
    assert analysis["candidate_count"] == len(analysis["candidates"])
    assert {"min", "max", "default"} <= set(analysis["recommended_count"])
    report = read_json(result.report_path, {})
    assert report["terminal"]["status"] == "analysis_ready"
    assert report["production_render"]["status"] == "skipped"


def test_repeated_analysis_reuses_source_intelligence_cache(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)

    first = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))
    second = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))

    assert calls == {"profile": 1, "map": 1, "boundaries": 1}
    assert first.analysis_id == second.analysis_id
    report = read_json(second.report_path, {})
    assert report["stages"]["transcription"]["cache_hit"] is True
    assert report["terminal"]["status"] == "analysis_ready"


def test_draft_preview_uses_analysis_artifact_and_preserves_exact_requested_order(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))
    artifact = read_json(analysis.analysis_path, {})
    candidate_ids = [item["candidate_id"] for item in artifact["candidates"][:2]]
    assert candidate_ids
    requested_ids = list(reversed(candidate_ids))
    before_render = dict(calls)

    import app.pipeline as pipeline_module

    def preview(_self, _plan, _source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "draft-preview.mp4"; output.write_bytes(b"preview")
        return SimpleNamespace(
            output_file=output,
            to_dict=lambda: {
                "status": "draft_ready", "output_file": str(output),
                "segments": [{"order": 1, "role": "hook"}], "estimated_duration_seconds": 12,
            },
        )

    monkeypatch.setattr(pipeline_module.DraftPreviewService, "render", preview)
    draft_config = AppConfig(score_threshold=0)
    draft_config.transformation.enabled = True
    draft_config.production.enabled = True
    result = Pipeline(
        tmp_path, draft_config, mock_ai=True,
        analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=requested_ids,
        draft_only=True,
    ).run(input_path=str(source))

    assert calls == before_render  # analysis functions were not even cache-read by the draft command
    assert result.selected_clips == len(requested_ids)
    assert len(result.output_files) == len(requested_ids)
    assert result.draft_path and result.draft_path.is_file()
    draft_artifact = read_json(result.draft_path, {})
    first_draft = draft_artifact["candidates"][0]
    progress = read_json(result.output_directory / "draft-progress.json", {})
    assert progress["run_id"] == result.output_directory.name
    assert [item["candidate_id"] for item in progress["candidates"]] == requested_ids
    assert all(item["source_end_seconds"] > item["source_start_seconds"] for item in progress["candidates"])
    assert all(item["output_file"] for item in progress["candidates"])
    assert first_draft["candidate_boundary_fingerprint"]
    assert first_draft["transformation_fingerprint"]
    assert first_draft["production_plan_fingerprint"]
    report = read_json(result.report_path, {})
    assert report["terminal"]["status"] == "draft_ready"
    assert [item["candidate_id"] for item in report["candidate_flow"]["draft_candidates"]] == requested_ids
    assert report["run"]["analysis_id"] == analysis.analysis_id


def test_production_cannot_start_directly_from_analysis(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))

    import pytest
    from app.errors import ClipEngineError
    with pytest.raises(ClipEngineError, match="draft preview first"):
        Pipeline(
            tmp_path, AppConfig(score_threshold=0), mock_ai=True,
            analysis_artifact_path=analysis.analysis_path,
            selected_candidate_ids=[read_json(analysis.analysis_path, {})["candidates"][0]["candidate_id"]],
        ).run(input_path=str(source))


def test_changed_source_fingerprint_requires_fresh_analysis_before_draft(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))
    candidate_id = read_json(analysis.analysis_path, {})["candidates"][0]["candidate_id"]
    source.write_bytes(b"changed source fingerprint")

    import pytest
    from app.errors import ClipEngineError

    with pytest.raises(ClipEngineError, match="different source file"):
        Pipeline(
            tmp_path, AppConfig(score_threshold=0), mock_ai=True,
            analysis_artifact_path=analysis.analysis_path,
            selected_candidate_ids=[candidate_id], draft_only=True,
        ).run(input_path=str(source))


def test_approved_draft_reuses_its_plan_and_reports_a_completed_candidate_flow(tmp_path: Path, monkeypatch) -> None:
    """The post-review render must not lose already-completed draft stages."""

    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))
    candidate_id = read_json(analysis.analysis_path, {})["candidates"][0]["candidate_id"]

    import app.pipeline as pipeline_module

    def preview(_self, _plan, _source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "draft-preview.mp4"; output.write_bytes(b"preview")
        return SimpleNamespace(
            output_file=output,
            to_dict=lambda: {
                "status": "draft_ready", "output_file": str(output),
                "segments": [{"order": 1, "role": "hook"}], "estimated_duration_seconds": 12,
            },
        )

    monkeypatch.setattr(pipeline_module.DraftPreviewService, "render", preview)
    draft_config = AppConfig(score_threshold=0)
    draft_config.transformation.enabled = True
    draft_config.production.enabled = True
    draft = Pipeline(
        tmp_path, draft_config, mock_ai=True,
        analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=[candidate_id], draft_only=True,
    ).run(input_path=str(source))

    def tts(_self, _tracker, _production, _work, _output):
        return {"enabled": False, "status": "skipped", "items": []}

    def audio(_self, _tracker, _production, _tts, _source, _transcript, _work, _output, _prepared=None):
        return {"enabled": False, "status": "skipped", "items": []}

    def render(_self, _tracker, production, _audio, _source, _transcript, _work, output, _visual=None):
        item = production["items"][0]
        result = output / "results" / "final-short-01.mp4"
        result.parent.mkdir(parents=True, exist_ok=True); result.write_bytes(b"mp4")
        return {
            "enabled": True, "status": "completed", "items": [{
                "candidate_id": item["candidate_id"], "status": "completed", "output_file": str(result),
                "production_plan_id": item["production_plan_id"], "clip_result_id": "approved-result",
                "run_id": _self.run_id, "revision_id": f"{_self.run_id}:render-01", "primary": True,
            }],
        }

    monkeypatch.setattr(Pipeline, "_run_tts", tts)
    monkeypatch.setattr(Pipeline, "_run_audio", audio)
    monkeypatch.setattr(Pipeline, "_run_production_render", render)
    before_production = dict(calls)
    result = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True,
        draft_artifact_path=draft.draft_path, selected_candidate_ids=[candidate_id],
    ).run(input_path=str(source))

    assert calls == before_production
    report = read_json(result.report_path, {})
    flow = report["candidate_flow"]
    assert report["terminal"]["status"] == "completed"
    assert flow["transformed"] == flow["production_plans"] == flow["rendered"] == 1
    assert flow["items"] == [{
        "candidate_id": candidate_id, "outcome": "selected", "reason": "rendered",
        "production_plan_id": report["production_plan"]["items"][0]["production_plan_id"],
        "clip_result_id": "approved-result",
    }]
    assert report["run"]["render_settings_fingerprint"]
    assert report["production_render"]["render_settings_fingerprint"] == report["run"]["render_settings_fingerprint"]


def test_content_cache_is_never_shared_between_sources(tmp_path: Path, monkeypatch) -> None:
    first_source = tmp_path / "first.mp4"; first_source.write_bytes(b"first")
    second_source = tmp_path / "second.mp4"; second_source.write_bytes(b"second")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)

    Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True).run(input_path=str(first_source))
    Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True).run(input_path=str(second_source))

    assert calls == {"profile": 2, "map": 2, "boundaries": 2}


def test_virality_cache_ignores_render_revisions_but_respects_scoring_weights(tmp_path: Path, monkeypatch) -> None:
    import app.pipeline as pipeline_module

    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0, "virality_profiles": 0, "virality_ranking": 0}
    _wire_pipeline(monkeypatch, calls)
    real_profiles = pipeline_module.build_virality_assessments
    real_ranking = pipeline_module.apply_virality_ranking

    def virality_profiles(*args, **kwargs):
        calls["virality_profiles"] += 1
        return real_profiles(*args, **kwargs)

    def virality_ranking(*args, **kwargs):
        calls["virality_ranking"] += 1
        return real_ranking(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "build_virality_assessments", virality_profiles)
    monkeypatch.setattr(pipeline_module, "apply_virality_ranking", virality_ranking)

    first = AppConfig(score_threshold=0); first.virality.enabled = True
    result = Pipeline(tmp_path, first, mock_ai=True).run(input_path=str(source))
    assert (result.work_directory / "virality_profiles.json").is_file()
    assert (result.work_directory / "virality_ranking.json").is_file()
    report = read_json(result.report_path, {})
    assert report["virality"]["enabled"] is True
    assert Path(report["virality"]["profiles_ref"]).is_file()
    assert report["virality"]["cost"]["actual_ai_cost"] == 0
    manifest = read_json(result.output_directory / "manifest.json", {})
    assert manifest["virality"]["analysis_fingerprint"]
    assert calls["virality_profiles"] == 1 and calls["virality_ranking"] == 1

    render_only = AppConfig(score_threshold=0); render_only.virality.enabled = True
    render_only.production_render.subtitle_style = "clean"
    Pipeline(tmp_path, render_only, mock_ai=True).run(input_path=str(source))
    assert calls["virality_profiles"] == 1 and calls["virality_ranking"] == 1

    changed_weights = AppConfig(score_threshold=0); changed_weights.virality.enabled = True
    changed_weights.virality.strategy_weights["motivational_monologue"]["hook"] += 0.01
    changed_weights.virality.strategy_weights["motivational_monologue"]["payoff"] -= 0.01
    Pipeline(tmp_path, changed_weights, mock_ai=True).run(input_path=str(source))
    assert calls["virality_profiles"] == 1
    assert calls["virality_ranking"] == 2
