from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import main
from app.config import AppConfig
from app.errors import ClipEngineError
from app.pipeline import INTELLIGENCE_ENGINE_VERSION, Pipeline, PipelineResult, StageTracker, _hash
from app.run_artifacts import run_metadata_path
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


def _wire_two_candidate_pipeline(monkeypatch, calls: dict[str, int]) -> None:
    """Keep two independently valid candidates available for approved-draft tests."""

    _wire_pipeline(monkeypatch, calls)
    import app.pipeline as pipeline_module

    def prepare(_source: Path, work: Path) -> dict:
        audio = work / "audio.wav"; audio.write_bytes(b"wav")
        result = {"duration": 50.0, "width": 1920, "height": 1080, "fps": 30, "audio_streams": 1, "audio_path": str(audio)}
        write_json(work / "metadata.json", result)
        return result

    def transcribe(_audio, source_id, duration, config, destination):
        result = {
            "source_id": source_id, "language": "en", "duration": duration,
            "segments": [
                {"start": 1.0, "end": 19.0, "text": "The first independent story develops. In the end, the result is clear."},
                {"start": 25.0, "end": 43.0, "text": "A separate second story develops. In the end, the result is different."},
            ],
            "words": [], "model": config.whisper_model, "runtime": {"device": "cpu"},
        }
        write_json(destination, result)
        destination.with_suffix(".txt").write_text(
            "\n".join(item["text"] for item in result["segments"]), encoding="utf-8",
        )
        return result

    monkeypatch.setattr(pipeline_module, "prepare_media", prepare)
    monkeypatch.setattr(pipeline_module, "transcribe", transcribe)


def _wire_three_candidate_pipeline(monkeypatch, calls: dict[str, int]) -> None:
    """Provide three independent stories for the partial-success draft flow."""

    _wire_pipeline(monkeypatch, calls)
    import app.pipeline as pipeline_module

    def prepare(_source: Path, work: Path) -> dict:
        audio = work / "audio.wav"; audio.write_bytes(b"wav")
        result = {"duration": 80.0, "width": 1920, "height": 1080, "fps": 30, "audio_streams": 1, "audio_path": str(audio)}
        write_json(work / "metadata.json", result)
        return result

    def transcribe(_audio, source_id, duration, config, destination):
        result = {
            "source_id": source_id, "language": "en", "duration": duration,
            "segments": [
                {"start": 1.0, "end": 19.0, "text": "The first independent story develops. In the end, the result is clear."},
                {"start": 27.0, "end": 45.0, "text": "The second independent story develops. In the end, the result is different."},
                {"start": 53.0, "end": 71.0, "text": "The third independent story develops. In the end, the result is distinct."},
            ],
            "words": [], "model": config.whisper_model, "runtime": {"device": "cpu"},
        }
        write_json(destination, result)
        destination.with_suffix(".txt").write_text(
            "\n".join(item["text"] for item in result["segments"]), encoding="utf-8",
        )
        return result

    monkeypatch.setattr(pipeline_module, "prepare_media", prepare)
    monkeypatch.setattr(pipeline_module, "transcribe", transcribe)


def _fake_creative_previews(monkeypatch) -> None:
    """Replace the current Creative Preview executor, not the removed legacy service."""

    def tts(_self, _tracker, _production, _work, _output):
        return {"enabled": False, "status": "skipped", "items": []}

    def audio(_self, _tracker, _production, _tts, _source, _transcript, _work, _output, _prepared=None):
        return {"enabled": False, "status": "skipped", "items": []}

    def render(
        _self, _tracker, production, _audio, _source, _transcript, _work, output, _visual=None,
        **_kwargs,
    ):
        outcomes = []
        for index, item in enumerate(production["items"], start=1):
            if item.get("status") != "completed":
                continue
            candidate_id = str(item["candidate_id"])
            root = output / "creative-previews" / f"candidate-{index:02d}"
            root.mkdir(parents=True, exist_ok=True)
            preview = root / "creative-preview.mp4"
            preview.write_bytes(b"creative-preview")
            write_json(root / "parity-manifest.json", {"candidate_id": candidate_id})
            write_json(root / "compiled-render-plan.json", {"candidate_id": candidate_id})
            outcomes.append({
                "candidate_id": candidate_id,
                "status": "completed",
                "output_file": str(preview),
                "report": {
                    "render_profile": "creative_preview",
                    "compiled_plan_hash": f"compiled-{candidate_id}",
                    "parity_signature": f"parity-{candidate_id}",
                },
            })
        return {"enabled": True, "status": "completed", "items": outcomes}

    monkeypatch.setattr(Pipeline, "_run_tts", tts)
    monkeypatch.setattr(Pipeline, "_run_audio", audio)
    monkeypatch.setattr(Pipeline, "_run_production_render", render)


def _two_candidate_draft(tmp_path: Path, monkeypatch) -> tuple[Path, PipelineResult, list[str]]:
    """Produce a trusted two-candidate draft artifact, then let tests corrupt one hand-off."""

    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_two_candidate_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
        run_id="analysis-run", project_id="project",
    ).run(input_path=str(source))
    candidate_ids = [item["candidate_id"] for item in read_json(analysis.analysis_path, {})["candidates"][:2]]
    assert len(candidate_ids) == 2

    _fake_creative_previews(monkeypatch)
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    draft = Pipeline(
        tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=candidate_ids, draft_only=True, run_id="draft-run", project_id="project",
    ).run(input_path=str(source))
    return source, draft, candidate_ids


def _fake_approved_delivery(monkeypatch) -> None:
    """Make the expensive post-approval stages deterministic while retaining their inputs."""

    def tts(_self, _tracker, _production, _work, _output):
        return {"enabled": False, "status": "skipped", "items": []}

    def audio(_self, _tracker, _production, _tts, _source, _transcript, _work, _output, _prepared=None):
        return {"enabled": False, "status": "skipped", "items": []}

    def render(self, _tracker, production, _audio, _source, _transcript, _work, output, _visual=None, **_kwargs):
        outcomes = []
        for item in production["items"]:
            if item.get("status") != "completed":
                continue
            index = int(item["requested_index"])
            result = output / "results" / f"final-short-{index:02d}.mp4"
            result.parent.mkdir(parents=True, exist_ok=True); result.write_bytes(b"mp4")
            outcomes.append({
                "candidate_id": item["candidate_id"], "status": "completed", "output_file": str(result),
                "production_plan_id": item["production_plan_id"],
                "clip_result_id": f"approved-result-{index}", "run_id": self.run_id,
                "revision_id": f"{self.run_id}:render-{index:02d}", "primary": True,
                "source_start_seconds": item.get("source_start_seconds"),
                "source_end_seconds": item.get("source_end_seconds"),
            })
        return {"enabled": True, "status": "completed", "items": outcomes}

    monkeypatch.setattr(Pipeline, "_run_tts", tts)
    monkeypatch.setattr(Pipeline, "_run_audio", audio)
    monkeypatch.setattr(Pipeline, "_run_production_render", render)


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

    editorial_intent = AppConfig(score_threshold=0)
    editorial_intent.content_understanding.target_tail_padding_seconds = 0.9
    editorial_intent.content_understanding.editorial_intent = "найти практический вывод"
    Pipeline(tmp_path, editorial_intent, mock_ai=True).run(input_path=str(source))

    profile_override = AppConfig(score_threshold=0)
    profile_override.content_understanding.target_tail_padding_seconds = 0.9
    profile_override.content_understanding.manual_override = {
        "format": "gameplay", "editorial_mode": "commentary", "domain": "gaming",
    }
    Pipeline(tmp_path, profile_override, mock_ai=True).run(input_path=str(source))

    # The manual gameplay profile preserves the evidence-only content map, but
    # its accepted Vision admission invalidates the dependent boundaries.
    assert calls == {"profile": 2, "map": 1, "boundaries": 3}
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
    assert calls["profile"] == 3
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
    assert analysis["schema_version"] == "1.1"
    assert analysis["status"] == "analysis_ready"
    assert analysis["analysis_id"] == result.analysis_id
    assert analysis["source_fingerprint"]
    assert analysis["candidate_data_ref"].endswith("candidate_data.json")
    assert Path(analysis["candidate_data_ref"]).is_file()
    assert analysis["analysis_run_id"] == result.output_directory.name
    assert "final_selection" in analysis["references"]
    assert set(analysis["references"]) == set(analysis["reference_integrity"])
    assert analysis["candidates"]
    first = analysis["candidates"][0]
    assert {"story_unit_id", "chapter_id", "start_seconds", "end_seconds", "duration_seconds"} <= set(first)
    assert {"title", "core_idea", "hook_summary", "payoff_summary", "confidence", "potential"} <= set(first)
    assert {
        "reasons", "risks", "feature_profile", "eligibility_decision",
        "boundary_evidence", "preview", "recommended",
    } <= set(first)
    assert first["eligibility_decision"]["state"] == "assessed"
    assert first["eligibility_decision"]["eligible"] is True
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
    assert report["stages"]["production_feasibility"]["cache_hit"] is True
    feasibility = read_json(first.work_directory / "production_feasibility.json", {})
    assert feasibility["provider_mode"] == "local_only"
    assert feasibility["provider_calls"] == {"brain": 0, "vision": 0, "transformation": 0}
    assert report["terminal"]["status"] == "analysis_ready"


@pytest.mark.parametrize(
    ("stage", "artifact_name"),
    (("vision_pass1", "vision-observations.json"), ("vision_pass2", "shortlist.vision.json")),
)
def test_vision_admission_fingerprint_rejects_legacy_skipped_cache(
    tmp_path: Path, stage: str, artifact_name: str,
) -> None:
    artifact = tmp_path / artifact_name
    legacy_result = {"status": "skipped", "failure_reason": "vision_not_opted_in", "marker": "legacy"}
    write_json(artifact, legacy_result)
    legacy_fingerprint = {
        "source": "source-1", "processing_mode": "standard",
        "vision": {"enabled": True}, "provider": "mock", "model": "gpt-5-mini",
    }
    legacy_key = _hash({"engine_version": INTELLIGENCE_ENGINE_VERSION, "input": legacy_fingerprint})
    source_cache = StageTracker(tmp_path / "source-cache.json")
    source_cache.start(stage, legacy_key)
    source_cache.finish(stage)
    run_tracker = StageTracker(tmp_path / "run-state.json")
    calls: list[str] = []

    def recompute() -> dict:
        calls.append(stage)
        fresh = {"status": "completed", "marker": "fresh"}
        write_json(artifact, fresh)
        return fresh

    current_fingerprint = {
        **legacy_fingerprint,
        "deep_analysis": {
            "requested": "auto", "resolved": True, "reason": "accepted gameplay profile",
            "evidence": {"profile_format": "gameplay", "profile_format_resolution": "detected"},
            "estimated_benefit": "high",
        },
    }
    result = Pipeline(tmp_path, AppConfig(), mock_ai=True)._cached(
        run_tracker, stage, artifact, current_fingerprint, recompute, cache_tracker=source_cache,
    )

    assert calls == [stage]
    assert result == {"status": "completed", "marker": "fresh"}
    assert run_tracker.data["stages"][stage]["cache_hit"] is False


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

    _fake_creative_previews(monkeypatch)
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
    assert first_draft["eligibility_decision"]["state"] == "assessed"
    assert first_draft["eligibility_decision"]["eligible"] is True
    report = read_json(result.report_path, {})
    assert report["terminal"]["status"] == "draft_ready"
    assert [item["candidate_id"] for item in report["candidate_flow"]["draft_candidates"]] == requested_ids
    assert report["run"]["analysis_id"] == analysis.analysis_id


def test_draft_rejects_tampered_overlong_snapshot_before_transformation(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
    ).run(input_path=str(source))
    artifact = read_json(analysis.analysis_path, {})
    candidate_id = artifact["candidates"][0]["candidate_id"]
    candidate_path = Path(artifact["candidate_data_ref"])
    candidate_data = read_json(candidate_path, {})
    candidate = next(item for item in candidate_data["candidates"] if item["id"] == candidate_id)
    candidate["end"] = float(candidate["start"]) + 106.94
    write_json(candidate_path, candidate_data)
    transformed: list[list[str]] = []

    def transform(_self, _tracker, _source, _metadata, selected, *_args, **_kwargs):
        transformed.append([item.candidate.id for item in selected])
        return {"enabled": True, "status": "failed", "items": [], "warnings": []}

    monkeypatch.setattr(Pipeline, "_transform_selected", transform)
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    with pytest.raises(ClipEngineError, match="ANALYSIS_INTEGRITY_MISMATCH"):
        Pipeline(
            tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
            selected_candidate_ids=[candidate_id], draft_only=True,
        ).run(input_path=str(source))

    assert transformed == []


def test_draft_rejects_tampered_incomplete_story_snapshot_before_transformation(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
    ).run(input_path=str(source))
    artifact = read_json(analysis.analysis_path, {})
    candidate_id = artifact["candidates"][0]["candidate_id"]
    candidate_path = Path(artifact["candidate_data_ref"])
    candidate_data = read_json(candidate_path, {})
    candidate = next(item for item in candidate_data["candidates"] if item["id"] == candidate_id)
    candidate["rejection_reason"] = "incomplete_story"
    candidate.setdefault("virality", {})["eligibility"] = {
        "status": "rejected",
        "critical_failures": ["incomplete_story"],
    }
    write_json(candidate_path, candidate_data)
    transformed: list[list[str]] = []
    rendered: list[list[str]] = []

    def transform(_self, _tracker, _source, _metadata, selected, *_args, **_kwargs):
        transformed.append([item.candidate.id for item in selected])
        return {"enabled": True, "status": "completed", "items": [], "warnings": []}

    def render(_self, _tracker, production, *_args, **_kwargs):
        rendered.append([
            str(item["candidate_id"])
            for item in production.get("items", []) if item.get("status") == "completed"
        ])
        return {"enabled": True, "status": "completed", "items": []}

    monkeypatch.setattr(Pipeline, "_transform_selected", transform)
    monkeypatch.setattr(Pipeline, "_run_production_render", render)
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    with pytest.raises(ClipEngineError, match="ANALYSIS_INTEGRITY_MISMATCH"):
        Pipeline(
            tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
            selected_candidate_ids=[candidate_id], draft_only=True,
        ).run(input_path=str(source))

    assert transformed == []
    assert rendered == []


def test_three_selected_drafts_keep_two_ready_when_one_boundary_override_is_invalid(tmp_path: Path, monkeypatch) -> None:
    """One bad review edit must be a candidate retry, never a batch code-2 loss."""

    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_three_candidate_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
        run_id="analysis-run", project_id="project",
    ).run(input_path=str(source))
    candidate_ids = [item["candidate_id"] for item in read_json(analysis.analysis_path, {})["candidates"][:3]]
    assert len(candidate_ids) == 3

    _fake_creative_previews(monkeypatch)
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    rejected_id = candidate_ids[1]
    result = Pipeline(
        tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=candidate_ids, draft_only=True, run_id="draft-run", project_id="project",
        candidate_boundary_overrides={rejected_id: {"start": -1.0, "end": 20.0}},
    ).run(input_path=str(source))

    report = read_json(result.report_path, {})
    drafts = {
        item["candidate_id"]: item
        for item in report["candidate_flow"]["draft_candidates"]
    }
    assert report["terminal"]["status"] == "draft_ready"
    assert report["candidate_flow"]["draft_summary"] == {"requested": 3, "ready": 2, "failed": 1}
    assert len(result.output_files) == 2
    assert drafts[rejected_id]["state"] == "draft_failed"
    assert drafts[rejected_id]["stage"] == f"boundary_override:{rejected_id}"
    assert all(drafts[candidate_id]["state"] == "draft_ready" for candidate_id in candidate_ids if candidate_id != rejected_id)
    assert any("partial success" in warning for warning in report["warnings"])


def test_failed_creative_preview_reports_candidate_level_executor_reason(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
        run_id="analysis-run", project_id="project",
    ).run(input_path=str(source))
    candidate_id = read_json(analysis.analysis_path, {})["candidates"][0]["candidate_id"]

    def tts(_self, _tracker, _production, _work, _output):
        return {"enabled": False, "status": "skipped", "items": []}

    def audio(_self, _tracker, _production, _tts, _source, _transcript, _work, _output, _prepared=None):
        return {"enabled": False, "status": "skipped", "items": []}

    def render(_self, _tracker, _production, _audio, _source, _transcript, _work, _output, _visual=None, **_kwargs):
        return {
            "enabled": True,
            "status": "failed",
            "items": [{
                "candidate_id": candidate_id,
                "status": "failed",
                "stage": f"creative_execution:{candidate_id}",
                "report": {"errors": ["SOURCE_OUTPUT_TIME_MAP_REJECTED: destination frame 61 overlaps"]},
            }],
        }

    monkeypatch.setattr(Pipeline, "_run_tts", tts)
    monkeypatch.setattr(Pipeline, "_run_audio", audio)
    monkeypatch.setattr(Pipeline, "_run_production_render", render)
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    result = Pipeline(
        tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=[candidate_id], draft_only=True,
        run_id="draft-run", project_id="project",
    ).run(input_path=str(source))

    report = read_json(result.report_path, {})
    failed = report["candidate_flow"]["draft_candidates"][0]
    assert failed["state"] == "draft_failed"
    assert failed["stage"] == f"creative_execution:{candidate_id}"
    assert "SOURCE_OUTPUT_TIME_MAP_REJECTED" in failed["error"]
    assert report["terminal"]["error_code"] == "NO_DRAFT_PREVIEWS"
    assert report["terminal"]["candidate_failures"] == [{
        "candidate_id": candidate_id,
        "stage": f"creative_execution:{candidate_id}",
        "error": failed["error"],
    }]
    assert "SOURCE_OUTPUT_TIME_MAP_REJECTED" in report["terminal"]["message"]


def test_failed_transformation_reports_nested_candidate_validation_reason(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
        run_id="analysis-run", project_id="project",
    ).run(input_path=str(source))
    candidate_id = read_json(analysis.analysis_path, {})["candidates"][0]["candidate_id"]
    cause = "FinalScript contract validation failed: required payoff evidence is missing"

    def transform(_self, *_args, **_kwargs):
        return {
            "enabled": True,
            "status": "failed",
            "items": [{
                "candidate_id": candidate_id,
                "status": "failed",
                "validation": {
                    "final_script": {"passed": False, "errors": [cause]},
                },
            }],
            "warnings": [],
        }

    monkeypatch.setattr(Pipeline, "_transform_selected", transform)
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    result = Pipeline(
        tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=[candidate_id], draft_only=True,
        run_id="draft-run", project_id="project",
    ).run(input_path=str(source))

    draft = read_json(result.draft_path, {})
    failed = draft["candidates"][0]
    report = read_json(result.report_path, {})
    assert failed["state"] == "draft_failed"
    assert failed["error"] == cause
    assert report["terminal"]["candidate_failures"] == [{
        "candidate_id": candidate_id,
        "stage": f"transformation_result:{candidate_id}",
        "error": cause,
    }]
    assert cause in report["terminal"]["message"]


def test_failed_tts_reason_survives_audio_and_render_skips_to_terminal(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True,
        run_id="analysis-run", project_id="project",
    ).run(input_path=str(source))
    candidate_id = read_json(analysis.analysis_path, {})["candidates"][0]["candidate_id"]
    cause = "TTS provider rejected the candidate voice request"
    stage = f"tts_generation:{candidate_id}"

    def tts(_self, _tracker, _production, _work, _output):
        return {
            "enabled": True,
            "status": "failed",
            "items": [{
                "candidate_id": candidate_id,
                "status": "failed",
                "stage": stage,
                "error": cause,
                "errors": [cause],
            }],
        }

    monkeypatch.setattr(Pipeline, "_run_tts", tts)
    monkeypatch.setattr("app.pipeline.tts_eligibility", lambda _plan: (True, "test"))
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.production.enabled = True
    config.production.audio_mode = "voiceover"
    config.audio_composition.enabled = True
    config.production_render.enabled = True
    result = Pipeline(
        tmp_path, config, mock_ai=True, analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=[candidate_id], draft_only=True,
        run_id="draft-run", project_id="project",
    ).run(input_path=str(source))

    draft = read_json(result.draft_path, {})
    failed = draft["candidates"][0]
    report = read_json(result.report_path, {})
    assert failed["state"] == "draft_failed"
    assert failed["stage"] == stage
    assert failed["error"] == cause
    assert report["terminal"]["candidate_failures"] == [{
        "candidate_id": candidate_id,
        "stage": stage,
        "error": cause,
    }]
    assert cause in report["terminal"]["message"]


def test_production_cannot_start_directly_from_analysis(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    calls = {"profile": 0, "map": 0, "boundaries": 0}
    _wire_pipeline(monkeypatch, calls)
    analysis = Pipeline(tmp_path, AppConfig(score_threshold=0), mock_ai=True, analysis_only=True).run(input_path=str(source))

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

    _fake_creative_previews(monkeypatch)
    draft_config = AppConfig(score_threshold=0)
    draft_config.transformation.enabled = True
    draft_config.production.enabled = True
    draft = Pipeline(
        tmp_path, draft_config, mock_ai=True,
        analysis_artifact_path=analysis.analysis_path,
        selected_candidate_ids=[candidate_id], draft_only=True,
    ).run(input_path=str(source))
    draft_data = read_json(draft.draft_path, {})
    persisted_decision = draft_data["candidates"][0]["eligibility_decision"]
    persisted_editorial = draft_data["candidates"][0]["editorial_decision"]
    assert persisted_decision["state"] == "assessed"
    assert persisted_editorial["surfacing_state"] in {"RECOMMENDED", "AVAILABLE"}
    assert persisted_editorial["selectable"] is True

    # Mutate only the reusable source cache after Draft(A). Final must still
    # read Analysis A's immutable snapshot and the Draft-owned decision.
    candidate_path = analysis.work_directory / "candidates.scored.json"
    candidate_data = read_json(candidate_path, {})
    cached_candidate = next(item for item in candidate_data["candidates"] if item["id"] == candidate_id)
    cached_candidate.pop("eligibility_decision", None)
    cached_candidate.pop("editorial_decision", None)
    write_json(candidate_path, candidate_data)

    def tts(_self, _tracker, _production, _work, _output):
        return {"enabled": False, "status": "skipped", "items": []}

    def audio(_self, _tracker, _production, _tts, _source, _transcript, _work, _output, _prepared=None):
        return {"enabled": False, "status": "skipped", "items": []}

    def render(_self, _tracker, production, _audio, _source, _transcript, _work, output, _visual=None, **_kwargs):
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
    persisted_at_final: list[tuple[dict, dict | None, dict]] = []
    original_quality_reports = Pipeline._persist_quality_reports

    def persist_quality_reports(self, *args, **kwargs):
        item = next(
            scored for scored in kwargs["final_scored"]
            if scored.candidate.id == candidate_id
        )
        override = kwargs["candidate_overrides"][candidate_id]
        persisted_at_final.append((
            item.candidate.eligibility_decision.to_dict(),
            item.candidate.editorial_decision.to_dict() if item.candidate.editorial_decision else None,
            override["editorial_final_handoff"],
        ))
        return original_quality_reports(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "_persist_quality_reports", persist_quality_reports)
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
    assert persisted_at_final[0][0] == persisted_decision
    assert persisted_at_final[0][1] == persisted_editorial
    assert persisted_at_final[0][2]["status"] == "passed"
    assert persisted_at_final[0][2]["candidate_id"] == candidate_id


def test_approved_draft_render_keeps_valid_candidate_when_another_plan_is_malformed(tmp_path: Path, monkeypatch) -> None:
    source, draft, candidate_ids = _two_candidate_draft(tmp_path, monkeypatch)
    invalid_id, valid_id = candidate_ids
    artifact = read_json(draft.draft_path, {})
    invalid = next(item for item in artifact["candidates"] if item["candidate_id"] == invalid_id)
    invalid["draft_production_plan"] = {"plan_id": "malformed"}
    write_json(draft.draft_path, artifact)
    _fake_approved_delivery(monkeypatch)

    result = Pipeline(
        tmp_path, AppConfig(score_threshold=0), mock_ai=True,
        draft_artifact_path=draft.draft_path, selected_candidate_ids=candidate_ids,
        run_id="partial-approved-render", project_id="project",
    ).run(input_path=str(source))

    report = read_json(result.report_path, {})
    outcomes = {item["candidate_id"]: item for item in report["production_plan"]["items"]}
    flow = {item["candidate_id"]: item for item in report["candidate_flow"]["items"]}
    manifest = read_json(result.output_directory / "manifest.json", {})

    assert result.terminal_status == "completed_with_warnings"
    assert result.selected_clips == 2
    assert [path.name for path in result.output_files] == ["final-short-02.mp4"]
    assert outcomes[invalid_id]["status"] == "failed"
    assert outcomes[invalid_id]["reason"] == "approved_draft_plan_invalid"
    assert outcomes[invalid_id]["stage"] == f"approved_draft_plan:{invalid_id}"
    assert outcomes[valid_id]["status"] == "completed"
    assert flow[invalid_id] == {
        "candidate_id": invalid_id,
        "outcome": "failed",
        "reason": "production_plan_failed",
        "message": outcomes[invalid_id]["error"],
        "stage": f"approved_draft_plan:{invalid_id}",
    }
    assert flow[valid_id]["outcome"] == "selected"
    assert manifest["requested_clip_count"] == 2
    assert manifest["completed_clip_count"] == 1
    assert manifest["terminal"]["status"] == "completed_with_warnings"


def test_all_invalid_approved_plans_persist_terminal_report_before_render_cli_exits_two(tmp_path: Path, monkeypatch, capsys) -> None:
    source, draft, candidate_ids = _two_candidate_draft(tmp_path, monkeypatch)
    missing_id, stale_id = candidate_ids
    artifact = read_json(draft.draft_path, {})
    missing = next(item for item in artifact["candidates"] if item["candidate_id"] == missing_id)
    stale = next(item for item in artifact["candidates"] if item["candidate_id"] == stale_id)
    missing.pop("draft_production_plan")
    stale["draft_production_plan"]["envelope"]["input_fingerprints"]["analysis_sha256"] = "0" * 64
    write_json(draft.draft_path, artifact)
    monkeypatch.setattr("app.cli.load_config", lambda _path: AppConfig(score_threshold=0))
    monkeypatch.chdir(tmp_path)

    run_id = "all-invalid-approved-render"
    assert main([
        "render", "--input", str(source), "--draft", str(draft.draft_path),
        "--run-id", run_id, "--project-id", "project", "--confirm-production",
        "--candidate-id", missing_id, "--candidate-id", stale_id,
    ]) == 2

    metadata = read_json(run_metadata_path(tmp_path, run_id), {})
    report_path = Path(metadata["paths"]["report_path"])
    report = read_json(report_path, {})
    manifest = read_json(report_path.with_name("manifest.json"), {})
    outcomes = {item["candidate_id"]: item for item in report["production_plan"]["items"]}
    flow = {item["candidate_id"]: item for item in report["candidate_flow"]["items"]}

    assert "Error:" in capsys.readouterr().err
    assert report["terminal"]["status"] == "failed"
    assert report["terminal"]["error_code"] == "NO_RENDERABLE_CLIPS"
    assert report["selected_clips_count"] == 2
    assert outcomes[missing_id]["reason"] == "approved_draft_plan_missing"
    assert outcomes[stale_id]["reason"] == "approved_draft_plan_stale"
    for candidate_id in candidate_ids:
        assert flow[candidate_id]["outcome"] == "failed"
        assert flow[candidate_id]["reason"] == "production_plan_failed"
        assert flow[candidate_id]["stage"] == f"approved_draft_plan:{candidate_id}"
    assert manifest["requested_clip_count"] == 2
    assert manifest["completed_clip_count"] == 0
    assert manifest["terminal"]["status"] == "failed"
    assert manifest["terminal"]["error_code"] == "NO_RENDERABLE_CLIPS"


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
