from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.content_understanding import build_global_content_map, build_video_content_profile
from app.models import Candidate, candidate_from_dict
from app.multimodal_candidates import enrich_shortlist_with_pass2, generate_multimodal_candidates
from app.multimodal_evidence import build_multimodal_timeline
from app.transcript_features import analyse_transcript


def _candidate_inputs(
    *, audio_peak: bool = False, visual_action: bool = False, visual_payoff: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], AppConfig, dict[str, Any]]:
    transcript = {
        "source_id": "source-mm", "duration": 34.0, "language": "en", "words": [],
        "segments": [
            {"id": 1, "start": 0.0, "end": 16.0, "text": "And it.", "confidence": 0.91},
            {"id": 2, "start": 16.5, "end": 33.0, "text": "Then it.", "confidence": 0.91},
        ],
    }
    audio = {
        "window_seconds": 0.5,
        "energy_frames": (
            [{"time": 8.0, "normalized_loudness": 0.98, "audio_energy": 0.8}]
            if audio_peak else []
        ),
        "silence_intervals": [],
    }
    scenes = {"enabled": True, "boundaries": []}
    visual = {
        "schema_version": "5D.0", "enabled": False, "status": "fallback",
        "evidence_status": "fallback", "reason": "vision_gateway", "subject_keyframes": [],
        "sample_count": 0,
    }
    config = AppConfig(optional_visual_features=True)
    config.content_understanding.min_story_unit_seconds = 10.0
    config.content_understanding.target_story_unit_seconds = 15.0
    config.content_understanding.max_story_unit_seconds = 30.0
    config.candidate_generation.max_duration_seconds = 60.0
    config.validate()
    vision_observations = []
    if visual_action:
        vision_observations.append({
            "keyframe_id": "action-frame", "timestamp": 5.0, "origin": "provider",
            "confidence": 0.94, "action": "interaction", "reaction": "none",
            "payoff_signal": "none", "primary_subject": "object",
        })
    if visual_payoff:
        vision_observations.append({
            "keyframe_id": "payoff-frame", "timestamp": 24.0, "origin": "provider",
            "confidence": 0.92, "action": "none", "reaction": "surprise",
            "payoff_signal": "result", "primary_subject": "object",
        })
    vision = {"status": "completed" if vision_observations else "fallback", "observations": vision_observations}
    return transcript, audio, scenes, visual, config, vision


def _generate(
    *, audio_peak: bool = False, visual_action: bool = False, visual_payoff: bool = False,
) -> list[Candidate]:
    transcript, audio, scenes, visual, config, vision = _candidate_inputs(
        audio_peak=audio_peak, visual_action=visual_action, visual_payoff=visual_payoff,
    )
    features = analyse_transcript(transcript, config.transcript_features)
    profile = build_video_content_profile(
        {"id": "source-mm", "display_name": "source.mp4"}, {"duration": 34.0}, transcript,
        features, audio, scenes, visual, config,
    )
    timeline = build_multimodal_timeline(
        source_id="source-mm", source_duration_seconds=34.0, transcript=transcript,
        audio_features=audio, scenes=scenes, visual_analysis=visual,
    )
    content_map = build_global_content_map(
        {"id": "source-mm", "display_name": "source.mp4"}, {"duration": 34.0}, transcript,
        features, audio, scenes, visual, profile, config, timeline,
    )
    return generate_multimodal_candidates(
        content_map, transcript, features, scenes, timeline, vision, config,
    )[0]


def test_text_audio_visual_and_multimodal_generation_paths() -> None:
    assert {item.candidate_kind for item in _generate()} == {"transcript"}
    assert "audio" in {item.candidate_kind for item in _generate(audio_peak=True)}
    assert "visual" in {item.candidate_kind for item in _generate(visual_action=True)}
    assert "multimodal" in {
        item.candidate_kind for item in _generate(audio_peak=True, visual_action=True)
    }


def test_combined_evidence_expands_story_range_and_preserves_full_provenance() -> None:
    candidates = _generate(audio_peak=True, visual_payoff=True)
    expanded = next(item for item in candidates if len(item.story_unit_ids) == 2)

    provenance = expanded.multimodal_provenance
    assert expanded.candidate_kind == "multimodal"
    assert provenance["schema_version"] == "6C.1"
    assert provenance["story_unit_ids"] == expanded.story_unit_ids
    assert provenance["transcript_evidence"]
    assert provenance["audio_evidence"]
    assert provenance["visual_evidence"]
    assert provenance["generation"]["range_expanded"] is True
    assert "range_expanded_to_preserve_linked_action_reaction_or_payoff" in provenance["generation"]["reasons"]
    assert expanded.boundary_diagnostics["boundary_decision"]["candidate_id"] == expanded.id

    restored = candidate_from_dict(expanded.to_dict())
    assert restored.candidate_kind == "multimodal"
    assert restored.story_unit_ids == expanded.story_unit_ids
    assert restored.multimodal_provenance == provenance


def _pass2_timeline() -> dict[str, Any]:
    transcript = {
        "source_id": "source-pass2", "duration": 90.0, "language": "en", "words": [],
        "segments": [
            {"id": index, "start": float(index * 10 + 1), "end": float(index * 10 + 8),
             "text": f"Complete moment {index}."}
            for index in range(8)
        ],
    }
    return build_multimodal_timeline(
        source_id="source-pass2", source_duration_seconds=90.0, transcript=transcript,
        audio_features={"window_seconds": 0.5, "energy_frames": [], "silence_intervals": []},
        scenes={"enabled": True, "boundaries": [
            {"timestamp": float(value), "scene_change_score": 0.9} for value in (15, 30, 45, 60, 75)
        ]},
        visual_analysis={
            "schema_version": "5D.0", "enabled": False, "status": "fallback",
            "evidence_status": "fallback", "reason": "gateway", "subject_keyframes": [], "sample_count": 0,
        },
    )


class _Pass2Spy:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def analyze_pass2(
        self, *, source: Path, timeline: dict[str, Any], request: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(str(request["candidate_id"]))
        if self.fail:
            raise RuntimeError("vision unavailable")
        return {
            "schema_version": "6B.pass2-result.1", "candidate_id": request["candidate_id"],
            "analysis_run_id": timeline["analysis_run_id"], "request": request, "status": "completed",
            "verification": {
                "hook_visible": True, "action_visible": True, "reaction_visible": True,
                "payoff_visible": True, "continuity_risk": "low", "confidence": 0.9,
            },
            "observations": [], "diagnostics": {},
        }


def test_pass2_only_calls_budgeted_shortlist_prefix_and_failure_keeps_candidates(tmp_path: Path) -> None:
    timeline = _pass2_timeline()
    config = AppConfig(optional_visual_features=True)
    config.product_flow.processing_mode = "maximum"
    config.vision.pass2_max_candidates = 2
    candidates = [
        Candidate(id=f"candidate-{index}", start=0.0, end=80.0, text="Complete moment.")
        for index in range(3)
    ]
    spy = _Pass2Spy()

    result = enrich_shortlist_with_pass2(
        candidates, source=tmp_path / "source.mp4", timeline=timeline, gateway=spy, config=config,
    )

    assert spy.calls == ["candidate-0", "candidate-1"]
    assert result[0].vision_pass2_evidence["result"]["verification"]["payoff_visible"] is True
    assert result[2].vision_pass2_evidence["status"] == "not_requested"

    failing_candidate = Candidate(id="candidate-fallback", start=0.0, end=80.0, text="Still valid.")
    failed = enrich_shortlist_with_pass2(
        [failing_candidate], source=tmp_path / "source.mp4", timeline=timeline,
        gateway=_Pass2Spy(fail=True), config=config,
    )
    assert failed == [failing_candidate]
    assert failed[0].vision_pass2_evidence["status"] == "skipped"
    assert "vision unavailable" in failed[0].vision_pass2_evidence["reason"]
