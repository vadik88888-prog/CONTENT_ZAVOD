from __future__ import annotations

from pathlib import Path

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
