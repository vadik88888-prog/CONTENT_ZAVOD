from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from app.config import AppConfig
from app.creative_evidence import has_usable_composition_evidence
from app.models import Candidate
from app.multimodal_evidence import build_multimodal_timeline
from app.pipeline import Pipeline, StageTracker
from app.sources import Source
from app.utils import read_json
from app.vision_intelligence import build_candidate_bounded_pass2_timeline


def _sparse_timeline() -> dict:
    transcript = {
        "source_id": "source-pass2",
        "duration": 90.0,
        "language": "en",
        "words": [],
        "segments": [{
            "id": 0,
            "start": 1.0,
            "end": 8.0,
            "text": "A complete source thought outside the selected candidate range.",
        }],
    }
    return build_multimodal_timeline(
        source_id="source-pass2",
        source_duration_seconds=90.0,
        transcript=transcript,
        audio_features={"window_seconds": 0.5, "energy_frames": [], "silence_intervals": []},
        scenes={"enabled": True, "boundaries": [
            {"timestamp": 20.0, "scene_change_score": 0.9},
            {"timestamp": 70.0, "scene_change_score": 0.9},
        ]},
        visual_analysis={
            "schema_version": "5D.0",
            "enabled": False,
            "status": "fallback",
            "evidence_status": "fallback",
            "reason": "not_sampled",
            "subject_keyframes": [],
            "sample_count": 0,
        },
    )


def _candidate() -> Candidate:
    return Candidate(
        id="candidate-podcast-gap",
        start=40.0,
        end=50.0,
        text="A selected podcast moment.",
        composition_intent={"active_speaker": {"value": True}},
        multimodal_provenance={
            "generation": {"anchors": {
                "hook": 40.5,
                "action": 43.0,
                "reaction": 46.0,
                "payoff": 49.5,
            }},
        },
        vision_pass2_evidence={
            "schema_version": "6C.pass2-evidence.1",
            "status": "not_requested",
            "reason": "pass2_shortlist_budget_limit",
            "result": None,
        },
    )


def test_candidate_bounded_timeline_fills_sparse_range_without_mutating_snapshot() -> None:
    timeline = _sparse_timeline()
    original = repr(timeline)
    candidate = _candidate()
    anchors = candidate.multimodal_provenance["generation"]["anchors"]

    bounded = build_candidate_bounded_pass2_timeline(
        timeline,
        candidate_id=candidate.id,
        window_start=candidate.start,
        window_end=candidate.end,
        anchors=anchors,
        max_frames=7,
    )

    assert repr(timeline) == original
    assert not [
        item for item in timeline["keyframes"]
        if candidate.start <= item["time_seconds"] <= candidate.end
    ]
    planned = [
        item for item in bounded["keyframes"]
        if candidate.start <= item["time_seconds"] <= candidate.end
    ]
    assert 3 <= len(planned) <= 7
    assert all("draft_candidate_composition_gap" in item["selection_reasons"] for item in planned)
    assert bounded["analysis_run_id"] == timeline["analysis_run_id"]


def test_draft_composition_pass2_is_candidate_cached_and_reused(
    tmp_path: Path, monkeypatch,
) -> None:
    import app.pipeline as pipeline_module

    timeline = _sparse_timeline()
    source_path = tmp_path / "podcast.mp4"
    source_path.write_bytes(b"real-source-placeholder")
    source = Source(
        id=timeline["source_id"],
        path=source_path,
        display_name="podcast.mp4",
        origin=str(source_path),
    )
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_bytes(b'{"immutable":true}')
    analysis_sha = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    analysis = SimpleNamespace(
        analysis_id="analysis-podcast",
        verified_sha256=analysis_sha,
    )
    calls: list[dict] = []

    class FakeVisionGateway:
        def __init__(self, **_kwargs) -> None:
            pass

        def analyze_pass2(self, *, source: Path, timeline: dict, request: dict) -> dict:
            calls.append(request)
            observations = [{
                "observation_schema_version": "6B.1",
                "keyframe_id": frame["keyframe_id"],
                "timestamp": frame["timestamp"],
                "scene_type": "PODCAST",
                "primary_subject": "face",
                "normalized_center_x": 0.68,
                "normalized_center_y": 0.42,
                "visible_face_count": 1,
                "action": "speaking",
                "reaction": "attention_shift",
                "payoff_signal": "result",
                "on_screen_text": "",
                "composition_risk": "none",
                "confidence": 0.92,
                "missing_evidence": [],
                "origin": "provider",
                "provenance": {
                    "provider": "mock",
                    "model": "gpt-5.6-terra",
                    "detail": "high",
                    "prompt_version": "6B.pass2.1",
                    "schema_version": "6B.1",
                    "frame_hash": "a" * 64,
                    "cache_key": "b" * 64,
                    "request_id": "request-test",
                },
            } for frame in request["frames"]]
            return {
                "schema_version": "6B.pass2-result.1",
                "candidate_id": request["candidate_id"],
                "analysis_run_id": timeline["analysis_run_id"],
                "request": request,
                "status": "completed",
                "verification": {
                    "hook_visible": True,
                    "action_visible": True,
                    "reaction_visible": True,
                    "payoff_visible": True,
                    "continuity_risk": "low",
                    "confidence": 0.92,
                },
                "observations": observations,
                "diagnostics": {"frames_sent": len(observations)},
            }

    monkeypatch.setattr(pipeline_module, "VisionGateway", FakeVisionGateway)
    monkeypatch.setattr(pipeline_module, "get_vision_provider", lambda *_args, **_kwargs: object())

    config = AppConfig(optional_visual_features=True)
    first_candidate = _candidate()
    first = Pipeline(
        tmp_path, config, mock_ai=True,
        analysis_artifact_path=analysis_path,
    )._ensure_draft_composition_evidence(
        [SimpleNamespace(candidate=first_candidate)],
        source=source,
        timeline=timeline,
        analysis=analysis,
        work_directory=tmp_path / "source-work",
        tracker=StageTracker(tmp_path / "first-state.json"),
    )

    assert len(calls) == 1
    request = calls[0]
    assert request["window"] == {"start_seconds": 40.0, "end_seconds": 50.0}
    assert all(40.0 <= frame["timestamp"] <= 50.0 for frame in request["frames"])
    assert has_usable_composition_evidence(first_candidate.to_dict(), timeline)
    first_summary = first[first_candidate.id]
    assert first_summary["model"] == "gpt-5.6-terra"
    assert first_summary["cache_hit"] is False
    assert first_summary["usable_composition_evidence"] is True
    artifact_path = Path(first_summary["artifact_ref"])
    artifact = read_json(artifact_path, {})
    assert artifact["lineage"]["analysis_snapshot_mutated"] is False
    assert artifact["lineage"]["trigger"] == "draft_candidate_composition_evidence_gap"
    assert analysis_path.read_bytes() == b'{"immutable":true}'

    config.production_render.subtitle_style = "minimal"
    config.production_render.crop_strategy = "center_crop"
    config.validate()
    second_candidate = _candidate()
    second = Pipeline(
        tmp_path, config, mock_ai=True,
        analysis_artifact_path=analysis_path,
    )._ensure_draft_composition_evidence(
        [SimpleNamespace(candidate=second_candidate)],
        source=source,
        timeline=timeline,
        analysis=analysis,
        work_directory=tmp_path / "source-work",
        tracker=StageTracker(tmp_path / "second-state.json"),
    )

    assert len(calls) == 1
    assert second[second_candidate.id]["cache_hit"] is True
    assert second[second_candidate.id]["artifact_ref"] == str(artifact_path)
    assert has_usable_composition_evidence(second_candidate.to_dict(), timeline)
    assert analysis_path.read_bytes() == b'{"immutable":true}'


def test_draft_composition_pass2_failure_is_isolated_per_candidate(
    tmp_path: Path, monkeypatch,
) -> None:
    first = _candidate()
    first.id = "candidate-failed"
    second = _candidate()
    second.id = "candidate-valid"
    pipeline = Pipeline(tmp_path, AppConfig(), mock_ai=True)
    tracker = StageTracker(tmp_path / "state.json")

    def fake_ensure(
        _self, selected, **_kwargs,
    ) -> dict[str, dict]:
        candidate_id = selected[0].candidate.id
        if candidate_id == first.id:
            raise ValueError("candidate-only contract failure")
        return {candidate_id: {"status": "completed"}}

    monkeypatch.setattr(Pipeline, "_ensure_draft_composition_evidence", fake_ensure)
    outcomes = pipeline._ensure_draft_composition_evidence_isolated(
        [SimpleNamespace(candidate=first), SimpleNamespace(candidate=second)],
        source=SimpleNamespace(),
        timeline={},
        analysis=SimpleNamespace(),
        work_directory=tmp_path,
        tracker=tracker,
    )

    assert outcomes[first.id]["status"] == "skipped"
    assert outcomes[first.id]["usable_composition_evidence"] is False
    assert outcomes[second.id] == {"status": "completed"}
    state = read_json(tmp_path / "state.json", {})
    assert state["stages"][f"draft_composition_vision:{first.id}"]["status"] == "warning"
