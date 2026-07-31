from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.audio_models import NarrationClip
from app.audio_service import AudioCompositionService, audio_report_section
from app.errors import DUPLICATE_EXACT_SOURCE_RANGE, ProductionPlanHandoffError
from app.cli import build_parser
from app.config import AppConfig
from app.pipeline import Pipeline, StageTracker
from app.production_models import ProductionPlan, validate_audio_handoff
from app.sources import Source, local_source
from app.tts_providers import MockTTSProvider
from app.tts_service import TTSService
from app.utils import read_json, write_json


def _audio_config() -> AppConfig:
    config = AppConfig()
    config.tts.enabled = True
    config.tts.provider = "mock"
    config.tts.model = "mock-tts"
    config.tts.duration_warning_ratio = 1.0
    config.tts.duration_error_ratio = 2.0
    config.audio_composition.enabled = True
    config.validate()
    return config


def _plan(*, narration: bool = True, dialogue: bool = True) -> ProductionPlan:
    segments: list[dict] = []
    entries: list[dict] = []
    dialogue_mappings: list[dict] = []
    if narration:
        segments.append({
            "segment_id": "narration-001", "segment_type": "narration", "order": 1,
            "estimated_duration_seconds": 1.0, "timeline_included": True,
            "linked_segment_ids": ["dialogue-001"] if dialogue else [],
            "text": "Hello from a deterministic audio composition test.", "narration_role": "intro",
            "source_sentence_id": "sentence-001", "fact_ids": ["fact-001"], "source_segment_ids": [0],
            "word_count": 7, "words_per_second": 2.5, "voice_profile_id": "voice-test",
        })
        entries.append({
            "segment_id": "narration-001", "order": 1, "estimated_start_seconds": 0,
            "estimated_end_seconds": 1, "included_in_master_timeline": True,
            "linked_segment_ids": ["dialogue-001"] if dialogue else [],
        })
    if dialogue:
        item = {
            "segment_id": "dialogue-001", "segment_type": "original_dialogue", "order": 2,
            "estimated_duration_seconds": 1.0, "timeline_included": False,
            "linked_segment_ids": ["narration-001"] if narration else [], "fact_id": "fact-001",
            "transcript_segment_id": 0, "source_start_seconds": 1.0, "source_end_seconds": 2.0,
            "source_text": "Source dialogue remains audible.", "speaker": "source-speaker",
            "confidence": 0.9, "is_placeholder": True,
        }
        segments.append(item)
        dialogue_mappings.append(item.copy())
        entries.append({
            "segment_id": "dialogue-001", "order": 2, "estimated_start_seconds": 0,
            "estimated_end_seconds": 1, "included_in_master_timeline": False,
            "linked_segment_ids": ["narration-001"] if narration else [],
        })
    segments.append({
        "segment_id": "pause-001", "segment_type": "pause", "order": 3,
        "estimated_duration_seconds": 0.25, "timeline_included": True,
        "linked_segment_ids": [], "reason": "narration_transition",
    })
    entries.append({
        "segment_id": "pause-001", "order": 3, "estimated_start_seconds": 1,
        "estimated_end_seconds": 1.25, "included_in_master_timeline": True, "linked_segment_ids": [],
    })
    return ProductionPlan.model_validate({
        "plan_id": "audio-plan-001", "status": "draft", "segments": segments,
        "dialogue_mappings": dialogue_mappings,
        "timeline": {
            "timeline_version": "3A.0", "estimated_duration_seconds": 1.25,
            "narration_count": int(narration), "dialogue_count": int(dialogue), "pause_count": 1,
            "entries": entries,
        },
        "voice_profile": {"profile_id": "voice-test", "gender": "neutral", "style": "documentary", "language": "en"},
        "audio_layers": [
            {"layer_id": "narration", "layer_type": "narration"},
            {"layer_id": "dialogue", "layer_type": "original_dialogue"},
            {"layer_id": "music", "layer_type": "music"},
            {"layer_id": "effects", "layer_type": "effects"},
        ],
        "subtitle_track": {"track_id": "subtitle", "language": "en", "cues": [], "status": "placeholder"},
        "metadata": {
            "plan_version": "3A.0", "candidate_id": "candidate-audio", "source_id": "source-audio",
            "final_script_hash": "a" * 64,
        },
    })


def _source_wav(path: Path) -> Path:
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("ffmpeg is required for audio composition tests")
    subprocess.run(
        [executable, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def _tts_result(root: Path, config: AppConfig, plan: ProductionPlan) -> dict:
    result = TTSService(root, config).generate(plan, root / "run", root / "out", provider=MockTTSProvider())
    return result.model_dump(mode="json")


def _plan_with_adjacent_dialogues(*, duplicate_exact_range: bool) -> ProductionPlan:
    raw = _plan(narration=False, dialogue=True).model_dump(mode="json")
    first = dict(raw["dialogue_mappings"][0])
    first["order"] = 1
    second = dict(first)
    second["segment_id"] = "dialogue-002"
    second["order"] = 2
    second["fact_id"] = "fact-002"
    second["source_start_seconds"] = 1.0 if duplicate_exact_range else 2.0
    second["source_end_seconds"] = 2.0 if duplicate_exact_range else 3.0
    raw["segments"] = [dict(first), second, {**raw["segments"][-1], "order": 3}]
    raw["dialogue_mappings"] = [first, dict(second)]
    raw["timeline"]["dialogue_count"] = 2
    raw["timeline"]["entries"] = [
        {**raw["timeline"]["entries"][0], "order": 1},
        {
            "segment_id": "dialogue-002", "order": 2, "estimated_start_seconds": 0,
            "estimated_end_seconds": 0, "included_in_master_timeline": False, "linked_segment_ids": [],
        },
        {**raw["timeline"]["entries"][-1], "order": 3},
    ]
    return ProductionPlan.model_validate(raw)


def test_duplicate_exact_source_range_is_blocked_before_audio_composition(tmp_path: Path) -> None:
    plan = _plan_with_adjacent_dialogues(duplicate_exact_range=True)
    missing_source = Source("missing-source", tmp_path / "does-not-exist.wav", "missing.wav", "test")

    with pytest.raises(ProductionPlanHandoffError) as raised:
        AudioCompositionService(tmp_path, _audio_config()).compose(
            plan, missing_source, {"segments": []}, None, tmp_path / "work", tmp_path / "out",
        )

    assert raised.value.code == DUPLICATE_EXACT_SOURCE_RANGE
    assert raised.value.evidence == {
        "candidate_id": "candidate-audio",
        "segment_ids": ["dialogue-001", "dialogue-002"],
        "source_start": 1.0,
        "source_end": 2.0,
        "source_start_seconds": 1.0,
        "source_end_seconds": 2.0,
    }


def test_adjacent_nonidentical_source_ranges_are_allowed_by_duplicate_gate() -> None:
    plan = _plan_with_adjacent_dialogues(duplicate_exact_range=False)

    assert validate_audio_handoff(plan) is None


def test_audio_project_builds_dialogue_ducked_narration_and_artifacts(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan()
    source_path = _source_wav(tmp_path / "source.wav")
    project = AudioCompositionService(tmp_path, config).compose(
        plan, local_source(str(source_path)), {"segments": [{"id": 0, "start": 0, "end": 3, "text": "source"}]},
        _tts_result(tmp_path, config, plan), tmp_path / "work-run", tmp_path / "out",
    )
    assert project.status == "completed"
    assert [clip.clip_type for clip in project.timeline.clips] == ["narration", "dialogue", "silence"]
    narration = next(track for track in project.tracks if track.track_type == "narration").clips[0]
    assert narration.loudness_normalized and narration.source_bed_path and narration.ducked_source_bed_path
    assert project.mix.ducking.duck_level == config.audio_composition.duck_level
    assert Path(project.mix.mixed_audio_path or "").is_file()
    for name in ("audio-project.json", "audio-manifest.json", "audio-summary.txt"):
        assert (tmp_path / "out" / "audio" / name).is_file()
    report = audio_report_section(project)
    assert report["dialogue_count"] == 1 and report["narration_count"] == 1
    assert "sk-" not in json.dumps(report)


def test_audio_cache_reuses_dialogue_narration_and_source_bed(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan()
    source_path = _source_wav(tmp_path / "source.wav")
    source = local_source(str(source_path))
    tts = _tts_result(tmp_path, config, plan)
    service = AudioCompositionService(tmp_path, config)
    service.compose(plan, source, {"segments": [{"id": 0, "start": 0, "end": 3}]}, tts, tmp_path / "run-a", tmp_path / "out-a")
    second = service.compose(plan, source, {"segments": [{"id": 0, "start": 0, "end": 3}]}, tts, tmp_path / "run-b", tmp_path / "out-b")
    assert second.cache["dialogue_hit_count"] == 1
    assert second.cache["narration_hit_count"] == 1
    assert second.cache["source_bed_hit_count"] == 1


def test_prepared_source_audio_fallback_handles_broken_direct_media_seek(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan()
    prepared_audio = _source_wav(tmp_path / "prepared.wav")
    broken_media = tmp_path / "broken-source.mp4"
    broken_media.write_bytes(b"not a media file")
    project = AudioCompositionService(tmp_path, config).compose(
        plan, local_source(str(broken_media)), {"segments": [{"id": 0, "start": 0, "end": 3}]},
        _tts_result(tmp_path, config, plan), tmp_path / "work", tmp_path / "out",
        prepared_source_audio_path=prepared_audio,
    )
    assert project.status == "completed"
    assert [clip.clip_type for clip in project.timeline.clips] == ["narration", "dialogue", "silence"]


@pytest.mark.parametrize(("narration", "dialogue", "expected"), [
    (False, True, ["dialogue", "silence"]),
    (True, False, ["narration", "silence"]),
])
def test_audio_composition_handles_missing_optional_tracks(
    tmp_path: Path, narration: bool, dialogue: bool, expected: list[str],
) -> None:
    config = _audio_config()
    plan = _plan(narration=narration, dialogue=dialogue)
    source_path = _source_wav(tmp_path / "source.wav")
    tts = _tts_result(tmp_path, config, plan) if narration else {}
    project = AudioCompositionService(tmp_path, config).compose(
        plan, local_source(str(source_path)), {"segments": [{"id": 0, "start": 0, "end": 3}]}, tts,
        tmp_path / "work", tmp_path / "out",
    )
    assert [clip.clip_type for clip in project.timeline.clips] == expected
    assert Path(project.mix.mixed_audio_path or "").is_file()


def test_source_audio_mode_composes_dialogue_without_any_tts_artifact(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan(narration=False, dialogue=True)
    source_path = _source_wav(tmp_path / "source.wav")
    provider = MockTTSProvider()

    tts = TTSService(tmp_path, config).generate(plan, tmp_path / "run", tmp_path / "tts-out", provider=provider)
    project = AudioCompositionService(tmp_path, config).compose(
        plan, local_source(str(source_path)), {"segments": [{"id": 0, "start": 0, "end": 3}]},
        None, tmp_path / "work", tmp_path / "out",
    )

    assert plan.audio_mode == "original"
    assert tts.status == "skipped" and provider.call_count == 0 and not tts.artifacts
    assert project.status == "completed"
    assert [clip.clip_type for clip in project.timeline.clips] == ["dialogue", "silence"]
    assert Path(project.mix.mixed_audio_path or "").is_file()


def test_pipeline_composes_source_audio_after_a_skipped_tts_stage(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan(narration=False, dialogue=True)
    source_path = _source_wav(tmp_path / "source.wav")
    pipeline = Pipeline(tmp_path, config)
    tracker = StageTracker(tmp_path / "state.json")
    production = {
        "items": [{"candidate_id": "candidate-audio", "status": "completed", "plan": plan.model_dump(mode="json")}],
    }

    tts = pipeline._run_tts(tracker, production, tmp_path / "work", tmp_path / "out")
    audio = pipeline._run_audio(
        tracker, production, tts, local_source(str(source_path)), {"segments": [{"id": 0, "start": 0, "end": 3}]},
        tmp_path / "work", tmp_path / "out",
    )

    assert tts["status"] == "skipped"
    assert audio["status"] == "completed"
    assert (tmp_path / "out" / "audio" / "mixed_audio.wav").is_file()
    assert not (tmp_path / "out" / "tts").exists()


def test_tts_provider_fallback_does_not_block_dialogue_audio(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan()
    source_path = _source_wav(tmp_path / "source.wav")
    failed_tts = TTSService(tmp_path, config).generate(
        plan, tmp_path / "run", tmp_path / "tts-out", provider=MockTTSProvider("provider_error"),
    )
    assert failed_tts.api_call_count >= 1
    project = AudioCompositionService(tmp_path, config).compose(
        plan, local_source(str(source_path)), {"segments": [{"id": 0, "start": 0, "end": 3}]},
        failed_tts.model_dump(mode="json"), tmp_path / "work", tmp_path / "out",
    )
    assert project.status == "partial"
    assert [clip.clip_type for clip in project.timeline.clips] == ["dialogue", "silence"]
    assert Path(project.mix.mixed_audio_path or "").is_file()


def test_audio_only_preserves_existing_mp4_and_cli_flags(tmp_path: Path) -> None:
    config = _audio_config()
    plan = _plan()
    source_path = _source_wav(tmp_path / "source.wav")
    pipeline = Pipeline(tmp_path, config, audio_only=True)
    source, work_directory, output_directory = pipeline._prepare_source(str(source_path), None)
    write_json(output_directory / "production-plan.json", plan.model_dump(mode="json"))
    write_json(work_directory / "transcript.json", {"segments": [{"id": 0, "start": 0, "end": 3}]})
    tts_result = TTSService(tmp_path, config).generate(plan, work_directory, output_directory, provider=MockTTSProvider())
    assert tts_result.status == "completed"
    old_mp4 = output_directory / "old.mp4"
    old_mp4.write_bytes(b"old-video")
    write_json(output_directory / "report.json", {"output_files": [str(old_mp4)], "selected_clips_count": 1})
    result = pipeline._run_audio_only(StageTracker(work_directory / "state.json"), source, work_directory, output_directory)
    assert result.output_files == [old_mp4] and old_mp4.read_bytes() == b"old-video"
    assert read_json(result.report_path, {})["audio"]["status"] == "completed"
    arguments = build_parser().parse_args(["process", "--input", "source.mp4", "--audio-only", "--recompute-audio"])
    assert arguments.audio_only and arguments.recompute_audio


def test_audio_models_reject_invalid_timeline_range() -> None:
    with pytest.raises(ValidationError):
        NarrationClip(
            clip_id="bad", order=1, production_segment_id="n", timeline_start_seconds=1,
            timeline_end_seconds=0, duration_seconds=1, audio_file_path="a.wav", checksum="a" * 64,
            status="ready", tts_segment_id="n", normalized_tts_path="a.wav", loudness_normalized=True,
            target_lufs=-16, duck_level=0.5, attack_seconds=0, release_seconds=0,
        )
