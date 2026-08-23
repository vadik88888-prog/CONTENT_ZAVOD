from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio_features import analyse_audio
from app.audio_semantics import (
    AUDIO_PROFILE_IDS,
    analyse_semantic_audio,
    project_semantic_audio_event,
    select_semantic_audio_regions,
)
from app.candidate_quality import assess_sparse_multimodal_content
from app.config import AppConfig, AudioAnalysisConfig
from app.models import Candidate


def _write_signal(path: Path, sections: list[tuple[float, float]]) -> None:
    sample_rate = 16000
    samples: list[int] = []
    for duration, amplitude in sections:
        for index in range(round(duration * sample_rate)):
            samples.append(round(amplitude * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate)))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _bounded_audio() -> dict:
    return {
        "duration_seconds": 60.0,
        "window_seconds": 0.1,
        "energy_frames": [
            {
                "time": round(index / 10, 1), "normalized_loudness": 0.8,
                "relative_loudness": 0.8, "activity_score": 0.8,
            }
            for index in range(600)
        ],
        "peak_regions": [
            {"region_id": "peak-a", "start": 8.0, "end": 20.0, "peak_time": 14.0, "score": 0.95},
            {"region_id": "peak-b", "start": 40.0, "end": 52.0, "peak_time": 46.0, "score": 0.85},
        ],
        "activity_intervals": [{"start": 0.0, "end": 60.0}],
        "dead_zones": [],
    }


def _semantic_audio() -> dict:
    return {
        "duration_seconds": 2.0,
        "window_seconds": 0.1,
        "energy_frames": [
            {
                "time": round(index / 10, 1), "normalized_loudness": 0.8,
                "relative_loudness": 0.8, "activity_score": 0.8,
            }
            for index in range(20)
        ],
        "peak_regions": [
            {"region_id": "peak-a", "start": 0.0, "end": 2.0, "peak_time": 1.0, "score": 0.95},
        ],
        "activity_intervals": [{"start": 0.0, "end": 2.0}],
        "dead_zones": [],
    }


def _fake_scores(class_index: int):
    def inference(_waveform):
        result = np.zeros((1, 521), dtype=np.float32)
        result[0, class_index] = 0.96
        return result

    return inference


def _candidate(profile_id: str) -> Candidate:
    candidate = Candidate(
        id=f"candidate-{profile_id}", start=0.0, end=30.0, text="A self-contained moment.",
    )
    candidate.multimodal_provenance = {
        "audio_summary": {
            "schema_version": "audio-range-summary.1",
            "longest_speech_gap_seconds": 26.0,
            "activity_ratio": 0.04,
            "dead_zone_ratio": 0.82,
            "longest_audio_dead_zone_seconds": 24.0,
            "meaningful_event_count": 0,
            "spike_count": 0,
            "background_music_only": False,
        },
        "visual_evidence": [],
    }
    return candidate


def test_signal_analysis_is_one_pass_and_finds_relative_peaks_and_dead_zones(tmp_path: Path) -> None:
    wav = tmp_path / "signals.wav"
    _write_signal(wav, [(2.0, 0.0), (0.8, 0.85), (3.2, 0.0)])
    config = AudioAnalysisConfig(
        window_seconds=0.1, dead_zone_seconds=0.5, min_activity_seconds=0.1,
        peak_region_seconds=2.0, peak_min_separation_seconds=0.5,
    )

    evidence = analyse_audio(wav, config)

    assert evidence["analysis_passes"] == 1
    assert evidence["prepared_pcm_reads"] == 1
    assert evidence["source_video_decodes"] == 0
    assert evidence["activity_intervals"]
    assert evidence["dead_zones"]
    assert evidence["peak_regions"]
    assert evidence["signal_summary"]["onset_count"] >= 1


def test_semantic_regions_are_peak_and_shortlist_bounded() -> None:
    config = AudioAnalysisConfig(
        semantic_max_peak_regions=1, semantic_max_shortlist_regions=1,
        semantic_max_region_seconds=6.0, semantic_max_total_seconds=12.0,
    )
    regions = select_semantic_audio_regions(
        _bounded_audio(), [Candidate("short", 25.0, 35.0, "Candidate")], config,
    )

    assert len(regions) == 2
    assert all(item["end"] - item["start"] <= 6.0 for item in regions)
    assert sum(item["end"] - item["start"] for item in regions) <= 12.0
    assert {source for item in regions for source in item["sources"]} == {"audio_peak", "shortlist"}


@pytest.mark.parametrize(
    ("profile_id", "class_index", "expected_group"),
    [
        ("podcast", 13, "reaction"),
        ("interview", 62, "reaction"),
        ("gameplay", 421, "impact"),
        ("food", 484, "cooking"),
        ("sports_fitness", 459, "sports"),
        ("tutorial_education", 412, "tool"),
        ("movie_series", 420, "impact"),
    ],
)
def test_profile_regressions_interpret_meaningful_nonverbal_events(
    tmp_path: Path, profile_id: str, class_index: int, expected_group: str,
) -> None:
    wav = tmp_path / f"{profile_id}.wav"
    _write_signal(wav, [(2.0, 0.25)])
    artifact = analyse_semantic_audio(
        wav, _semantic_audio(), [], profile_id, AudioAnalysisConfig(),
        inference=_fake_scores(class_index),
    )

    assert artifact["status"] == "completed"
    assert artifact["events"]
    event = project_semantic_audio_event(artifact["events"][0], profile_id)
    assert event["observation"]["event_group"] == expected_group
    assert event["observation"]["meaningful_for_profile"] is True
    assert event["observation"]["payoff_claim"] is False


def test_all_fifteen_profiles_have_explicit_audio_interpretation() -> None:
    assert len(AUDIO_PROFILE_IDS) == 15
    assert {"podcast", "gameplay", "food", "sports_fitness", "tutorial_education", "movie_series"} <= AUDIO_PROFILE_IDS


def test_background_music_is_evidence_but_never_meaningful_or_payoff(tmp_path: Path) -> None:
    wav = tmp_path / "music.wav"
    _write_signal(wav, [(2.0, 0.25)])
    artifact = analyse_semantic_audio(
        wav, _semantic_audio(), [], "gameplay", AudioAnalysisConfig(),
        inference=_fake_scores(132),
    )

    event = project_semantic_audio_event(artifact["events"][0], "gameplay")
    assert event["event_type"] == "background_music"
    assert event["observation"]["meaningful_for_profile"] is False
    assert event["observation"]["editorial_roles"] == []
    assert event["observation"]["payoff_claim"] is False


def test_sparse_content_is_strong_soft_downgrade_not_block() -> None:
    candidate = _candidate("podcast")

    assessment = assess_sparse_multimodal_content(candidate)

    assert assessment["applies"] is True
    assert assessment["penalty"] >= 30
    assert assessment["surfacing_effect"] == "strong_soft_downgrade_only"
    assert assessment["blocked"] is False


def test_low_speech_with_meaningful_audio_or_visual_action_is_not_sparse() -> None:
    audio_candidate = _candidate("gameplay")
    audio_candidate.multimodal_provenance["audio_summary"].update({
        "activity_ratio": 0.5, "dead_zone_ratio": 0.1,
        "longest_audio_dead_zone_seconds": 2.0, "meaningful_event_count": 1,
    })
    visual_candidate = _candidate("food")
    visual_candidate.multimodal_provenance["visual_evidence"] = [{
        "confidence": 0.9, "action": "demonstration", "reaction": "none",
        "payoff_signal": "none", "missing_evidence": [],
    }]

    assert assess_sparse_multimodal_content(audio_candidate)["applies"] is False
    assert assess_sparse_multimodal_content(visual_candidate)["applies"] is False


def test_context_only_action_without_grounded_result_is_downgraded() -> None:
    candidate = _candidate("gameplay")
    candidate.multimodal_provenance["audio_summary"].update({
        "longest_speech_gap_seconds": 6.5,
        "activity_ratio": 0.30,
        "dead_zone_ratio": 0.42,
        "longest_audio_dead_zone_seconds": 5.0,
        "meaningful_event_count": 5,
    })
    candidate.content_signature = {"narrative_function": "context"}
    candidate.semantic_evidence = {"setup": "", "payoff": ""}
    candidate.multimodal_provenance["visual_evidence"] = [{
        "confidence": 0.9, "action": "interaction", "reaction": "none",
        "payoff_signal": "none", "missing_evidence": ["payoff"],
    }]

    assessment = assess_sparse_multimodal_content(candidate)

    assert assessment["applies"] is True
    assert assessment["reason"] == "context_only_without_grounded_result_or_payoff"
    assert assessment["meaningful_visual_action"] is True
    assert assessment["meaningful_visual_payoff"] is False
    assert assessment["blocked"] is False


def test_grounded_nonverbal_result_is_not_downgraded_as_context_only() -> None:
    candidate = _candidate("gameplay")
    candidate.content_signature = {"narrative_function": "context"}
    candidate.semantic_evidence = {"setup": "", "payoff": ""}
    candidate.multimodal_provenance["visual_evidence"] = [{
        "confidence": 0.9, "action": "interaction", "reaction": "surprise",
        "payoff_signal": "result", "missing_evidence": [],
    }]

    assessment = assess_sparse_multimodal_content(candidate)

    assert assessment["applies"] is False
    assert assessment["meaningful_visual_payoff"] is True


def test_bundled_onnx_runtime_loads_without_tensorflow_and_stays_bounded(tmp_path: Path) -> None:
    wav = tmp_path / "runtime.wav"
    _write_signal(wav, [(2.0, 0.25)])

    artifact = analyse_semantic_audio(wav, _semantic_audio(), [], "gameplay", AudioAnalysisConfig())

    assert artifact["status"] == "completed"
    assert artifact["runtime"]["backend"] == "onnxruntime-cpu"
    assert artifact["runtime"]["tensorflow_required"] is False
    assert artifact["diagnostics"]["full_source_scan"] is False
    assert artifact["diagnostics"]["classified_seconds"] <= 2.0
    assert artifact["diagnostics"]["prepared_pcm_opens"] == 1
    assert artifact["diagnostics"]["source_video_decodes"] == 0
