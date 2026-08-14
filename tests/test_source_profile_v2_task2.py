from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis_artifact import AnalysisArtifact, AnalysisArtifactError
from app.config import AppConfig
from app.continuity import build_continuity_decision
from app.draft_artifact import DraftArtifact, new_draft_artifact
from app.errors import ClipEngineError
from app.pipeline import Pipeline
from app.utils import read_json, stable_file_hash, write_json


def _install_analysis_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_prepare_media(_source_path: Path, work_directory: Path) -> dict:
        audio = work_directory / "audio.wav"
        audio.write_bytes(b"wav")
        metadata = {
            "duration": 42.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "audio_streams": 1,
            "audio_path": str(audio),
        }
        write_json(work_directory / "metadata.json", metadata)
        return metadata

    def fake_transcribe(
        _audio_path: Path, source_id: str, source_duration: float, _config: AppConfig, destination: Path,
    ) -> dict:
        sentences = (
            "Why do projects fail before launch?",
            "They fail when teams skip the smallest validation step.",
            "The practical fix is to test one assumption before scaling.",
        )
        segments = []
        words = []
        cursor = 1.0
        for segment_id, sentence in enumerate(sentences):
            start = cursor
            for token in sentence.split():
                words.append({"start": cursor, "end": cursor + 0.45, "text": token, "confidence": 0.99})
                cursor += 0.55
            segments.append({
                "id": segment_id,
                "start": start,
                "end": cursor,
                "text": sentence,
                "speaker": "speaker-1",
            })
            cursor += 0.8
        transcript = {
            "source_id": source_id,
            "language": "en",
            "duration": source_duration,
            "segments": segments,
            "words": words,
            "model": "task2-fake",
            "runtime": {"device": "cpu"},
            "processing_duration_seconds": 0.01,
        }
        write_json(destination, transcript)
        destination.with_suffix(".txt").write_text(" ".join(sentences), encoding="utf-8")
        return transcript

    monkeypatch.setattr("app.pipeline.prepare_media", fake_prepare_media)
    monkeypatch.setattr("app.pipeline.transcribe", fake_transcribe)


def _config(*, profile: dict[str, object], intent: str) -> AppConfig:
    config = AppConfig(score_threshold=0)
    config.content_understanding.manual_override = profile
    config.content_understanding.editorial_intent = intent
    config.validate()
    return config


def _analysis_run(
    root: Path, source: Path, config: AppConfig, run_id: str,
) -> tuple[object, AnalysisArtifact]:
    result = Pipeline(
        root, config, mock_ai=True, analysis_only=True, run_id=run_id, project_id="project-task2",
    ).run(input_path=str(source))
    assert result.analysis_path is not None
    artifact = AnalysisArtifact.read_verified(result.analysis_path)
    return result, artifact


def _candidate_boundary(artifact: AnalysisArtifact) -> tuple[dict, dict]:
    semantic = artifact.load_reference("semantic_boundaries")
    candidate = semantic["candidates"][0]
    return candidate, candidate["boundary_diagnostics"]["boundary_decision"]


def _primary_evidence(candidate: dict) -> list[dict]:
    return [
        {
            "start": item["start_seconds"],
            "end": item["end_seconds"],
            "segment_id": item.get("observation", {}).get("segment_id"),
        }
        for item in candidate["multimodal_provenance"]["transcript_evidence"]
    ]


def test_profile_and_editorial_intent_do_not_change_boundaries_or_a2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_fakes(monkeypatch)
    source = tmp_path / "same-source.mp4"
    source.write_bytes(b"same source bytes")
    config_a = _config(
        profile={
            "format": "dialogue", "editorial_mode": "interview", "domain": "business",
            "traits": ["speech_led", "question_answer"],
        },
        intent="Find the business lesson.",
    )
    config_b = _config(
        profile={
            "format": "gameplay", "editorial_mode": "commentary", "domain": "gaming",
            "traits": ["visual_led", "high_pacing"],
        },
        intent="Find the most entertaining reaction.",
    )

    result_a, analysis_a = _analysis_run(tmp_path, source, config_a, "analysis-a")
    result_b, analysis_b = _analysis_run(tmp_path, source, config_b, "analysis-b")
    candidate_a, boundary_a = _candidate_boundary(analysis_a)
    candidate_b, boundary_b = _candidate_boundary(analysis_b)

    assert candidate_a["id"] == candidate_b["id"]
    assert analysis_a.load_reference("content_map") == analysis_b.load_reference("content_map")
    assert boundary_a == boundary_b
    report_b = read_json(result_b.report_path, {})
    for stage in ("vision_pass1", "global_content_map", "semantic_boundaries"):
        assert report_b["stages"][stage]["cache_hit"] is True

    continuity_a = build_continuity_decision(
        candidate_id=candidate_a["id"],
        boundary_decision=boundary_a,
        primary_evidence=_primary_evidence(candidate_a),
        multimodal_context={},
    )
    continuity_b = build_continuity_decision(
        candidate_id=candidate_b["id"],
        boundary_decision=boundary_b,
        primary_evidence=_primary_evidence(candidate_b),
        multimodal_context={},
    )
    assert continuity_a is not None and continuity_b is not None
    assert continuity_a.model_dump(mode="json") == continuity_b.model_dump(mode="json")
    assert result_a.work_directory == result_b.work_directory  # expensive source cache is preserved


def test_analysis_a_snapshot_survives_analysis_b_and_draft_binds_to_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_fakes(monkeypatch)
    source = tmp_path / "same-source.mp4"
    source.write_bytes(b"same source bytes")
    config_a = _config(
        profile={"format": "dialogue", "editorial_mode": "interview", "domain": "business", "traits": []},
        intent="Analysis A intent",
    )
    config_b = _config(
        profile={"format": "gameplay", "editorial_mode": "commentary", "domain": "gaming", "traits": ["visual_led"]},
        intent="Analysis B intent",
    )
    result_a, analysis_a = _analysis_run(tmp_path, source, config_a, "analysis-a")
    profile_a = analysis_a.load_reference("content_profile")
    candidate_id = analysis_a.load_reference("candidate_data")["candidates"][0]["id"]
    result_b, analysis_b = _analysis_run(tmp_path, source, config_b, "analysis-b")

    assert analysis_a.analysis_run_id == "analysis-a"
    assert analysis_b.analysis_run_id == "analysis-b"
    assert Path(analysis_a.snapshot_directory).is_relative_to(result_a.output_directory)
    assert Path(analysis_b.snapshot_directory).is_relative_to(result_b.output_directory)
    assert analysis_a.references["content_profile"] != analysis_b.references["content_profile"]
    assert analysis_a.load_reference("content_profile") == profile_a
    assert analysis_a.load_reference("content_profile")["manual_override"]["format"] == "dialogue"
    assert analysis_b.load_reference("content_profile")["manual_override"]["format"] == "gameplay"
    assert "final_selection" in analysis_a.references
    assert set(analysis_a.references) == set(analysis_a.reference_integrity)

    draft_result = Pipeline(
        tmp_path,
        config_b,
        mock_ai=True,
        analysis_artifact_path=result_a.analysis_path,
        selected_candidate_ids=[candidate_id],
        expected_analysis_id=analysis_a.analysis_id,
        expected_analysis_fingerprint=analysis_a.analysis_fingerprint,
        draft_only=True,
        run_id="draft-from-a",
        project_id="project-task2",
    ).run(input_path=str(source))
    assert draft_result.draft_path is not None
    draft = DraftArtifact.read(draft_result.draft_path)
    assert draft.schema_version == "1.1"
    assert draft.analysis_run_id == "analysis-a"
    assert draft.analysis_artifact_sha256 == stable_file_hash(result_a.analysis_path)
    draft_report = read_json(draft_result.report_path, {})
    assert draft_report["content_understanding"]["profile"]["manual_override"]["format"] == "dialogue"


def test_integrity_mismatch_stops_draft_before_transformation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_fakes(monkeypatch)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    config = _config(profile={}, intent="")
    result, analysis = _analysis_run(tmp_path, source, config, "analysis-integrity")
    profile_path = Path(analysis.references["content_profile"])
    profile = read_json(profile_path, {})
    profile["manual_override"] = {"format": "gameplay"}
    write_json(profile_path, profile)
    called = False

    def forbidden_transform(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("transformation must not start after an integrity mismatch")

    monkeypatch.setattr(Pipeline, "_transform_selected", forbidden_transform)
    with pytest.raises(ClipEngineError, match="ANALYSIS_INTEGRITY_MISMATCH"):
        Pipeline(
            tmp_path,
            config,
            mock_ai=True,
            analysis_artifact_path=result.analysis_path,
            selected_candidate_ids=["candidate-does-not-matter"],
            draft_only=True,
            run_id="draft-blocked",
            project_id="project-task2",
        ).run(input_path=str(source))
    assert called is False


def test_v10_artifacts_remain_readable_with_explicit_legacy_warnings(tmp_path: Path) -> None:
    analysis_path = tmp_path / "legacy-analysis.json"
    write_json(analysis_path, {
        "schema_version": "1.0",
        "analysis_id": "analysis-legacy",
        "project_id": None,
        "created_at": "2026-08-14T00:00:00+00:00",
        "source": {"id": "source-legacy"},
        "source_fingerprint": "source-legacy",
        "analysis_fingerprint": "fingerprint-legacy",
        "work_directory": str(tmp_path),
        "candidate_data_ref": str(tmp_path / "candidate-data.json"),
        "references": {},
        "candidates": [],
        "recommendation": {},
        "summary": {},
        "content_profile": {},
        "duration_seconds": 1.0,
        "candidate_count": 0,
        "recommended_count": {},
        "status": "analysis_ready",
        "warnings": [],
    })
    artifact = AnalysisArtifact.read_verified(analysis_path)
    assert artifact.schema_version == "1.0"
    assert any("LEGACY_ANALYSIS_ARTIFACT_1_0" in warning for warning in artifact.warnings)
    assert any("LEGACY_ANALYSIS_CHECKSUM_ONLY" in warning for warning in artifact.warnings)

    draft_path = tmp_path / "legacy-draft.json"
    write_json(draft_path, {
        "schema_version": "1.0",
        "draft_id": "draft-legacy",
        "analysis_id": "analysis-legacy",
        "analysis_fingerprint": "fingerprint-legacy",
        "analysis_artifact_path": str(analysis_path),
        "project_id": None,
        "source_fingerprint": "source-legacy",
        "created_at": "2026-08-14T00:00:00+00:00",
        "candidates": [],
        "status": "draft_ready",
        "warnings": [],
    })
    draft = DraftArtifact.read(draft_path)
    assert draft.schema_version == "1.0"
    assert any("LEGACY_DRAFT_ARTIFACT_1_0" in warning for warning in draft.warnings)


def test_final_checksum_binding_rejects_changed_analysis_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_fakes(monkeypatch)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    config = _config(profile={}, intent="")
    result, analysis = _analysis_run(tmp_path, source, config, "analysis-final")
    original_sha = stable_file_hash(result.analysis_path)
    draft_path = tmp_path / "approved-draft.json"
    new_draft_artifact(
        draft_id="draft-final",
        analysis_id=analysis.analysis_id,
        analysis_fingerprint=analysis.analysis_fingerprint,
        analysis_artifact_path=str(result.analysis_path),
        project_id="project-task2",
        source_fingerprint=analysis.source_fingerprint,
        candidates=[{"candidate_id": "candidate-1", "state": "draft_ready"}],
        analysis_run_id=analysis.analysis_run_id,
        analysis_artifact_sha256=original_sha,
    ).write(draft_path)
    raw = read_json(result.analysis_path, {})
    raw["summary"]["tampered"] = True
    write_json(result.analysis_path, raw)

    with pytest.raises(AnalysisArtifactError, match="checksum mismatch"):
        AnalysisArtifact.read_verified(result.analysis_path, expected_sha256=original_sha)
    with pytest.raises(ClipEngineError, match="checksum mismatch"):
        Pipeline(
            tmp_path,
            config,
            mock_ai=True,
            draft_artifact_path=draft_path,
            selected_candidate_ids=["candidate-1"],
            run_id="final-blocked",
            project_id="project-task2",
        ).run(input_path=str(source))
    assert analysis.verified_sha256 == original_sha
