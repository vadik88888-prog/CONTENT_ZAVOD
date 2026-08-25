from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.cli import build_parser
from app.config import AppConfig
from app.content_transformation import run_content_transformation
from app.errors import TTSCredentialError, TTSError as TTSRuntimeError
from app.models import Candidate
from app.pipeline import Pipeline, StageTracker
from app.production_models import NarrationSegment, ProductionPlan
from app.production_plan import build_production_plan
from app.semantic_extraction import build_source_context
from app.tts_models import TTSSegmentRequest, TTSProviderConfig, TTSVoiceConfig
from app.tts_providers import MockTTSProvider, OpenAITTSProvider
from app.tts_service import TTSService, normalize_audio, normalize_narration_text, resolve_voice, tts_report_section, validate_audio
from app.utils import read_json, write_json


def _plan(audio_mode: str = "voiceover") -> ProductionPlan:
    config = AppConfig()
    config.transformation.ai_strategy = "local_only"
    text = "First, measure the source data. Then keep the original claim without changing the facts."
    candidate = Candidate("candidate-tts-001", 4, 20, text, transcript_segment_ids=[0])
    transcript = {"language": "en", "segments": [{"start": 4, "end": 20, "text": text}]}
    features = {"segments": [{
        "id": 0, "start": 4, "end": 20, "sentence_start": True, "sentence_end": True,
        "speech_density": 0.6, "pause_before_seconds": 0.1, "pause_after_seconds": 0.2,
        "filler_word_ratio": 0.0, "repetition_score": 0.0,
    }]}
    context = build_source_context(
        {"id": "source-tts", "path": "source.mp4"}, {}, candidate, transcript,
        features, {}, {"boundaries": []}, config.transformation,
    )
    outcome = run_content_transformation(context, config.transformation, None, force_local=True)
    config.production.audio_mode = audio_mode
    return build_production_plan(outcome, config.production)


def _tts_config() -> AppConfig:
    config = AppConfig()
    config.tts.enabled = True
    config.tts.provider = "mock"
    config.tts.model = "mock-tts"
    config.tts.duration_warning_ratio = 1.0
    config.tts.duration_error_ratio = 2.0
    config.validate()
    return config


def _request(tmp_path: Path) -> TTSSegmentRequest:
    provider = TTSProviderConfig(
        provider="mock", model="mock-tts", timeout_seconds=5, max_retries=0,
        speed=1, sample_rate=48000, provider_config_version="test",
    )
    voice = TTSVoiceConfig(
        voice="cedar", gender="neutral", style="documentary", language="en",
        instructions="Preserve wording.", mapping_source="voice_registry",
    )
    return TTSSegmentRequest(
        production_plan_id="plan-test", segment_id="narration-001",
        narration_text="Hello world", normalized_text="Hello world", language="en",
        voice=voice, provider_config=provider, estimated_duration_seconds=1,
        cache_key="a" * 64, raw_output_path=str(tmp_path / "raw.wav"),
    )


def test_tts_models_validate_and_serialize(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert json.loads(request.model_dump_json())["segment_id"] == "narration-001"
    with pytest.raises(ValidationError):
        TTSSegmentRequest(**{**request.model_dump(), "segment_id": "bad / name"})


def test_segment_preparation_uses_only_narration_and_preserves_words(tmp_path: Path) -> None:
    plan = _plan()
    service = TTSService(tmp_path, _tts_config())
    provider_config = service._provider_config()
    voice, _ = resolve_voice(plan, service.config)
    requests, _ = service._prepare_requests(plan, provider_config, voice, tmp_path / "tmp")
    narration = [item for item in plan.segments if isinstance(item, NarrationSegment)]
    assert [item.segment_id for item in requests] == [item.segment_id for item in narration]
    assert all(item.narration_text == segment.text for item, segment in zip(requests, narration))
    assert normalize_narration_text("  keep\r\n words\tunchanged  ") == "keep words unchanged"


def test_voice_mapping_has_override_and_safe_default() -> None:
    plan = _plan()
    config = _tts_config()
    voice, warning = resolve_voice(plan, config)
    assert voice.voice == "cedar" and warning is None
    config.tts.voice = "coral"
    override, _ = resolve_voice(plan, config)
    assert override.voice == "coral" and override.mapping_source == "config_override"
    unknown = plan.model_copy(update={
        "voice_profile": plan.voice_profile.model_copy(update={"gender": "unknown", "style": "mystery"}),
    })
    default, warning = resolve_voice(unknown, _tts_config())
    assert default.voice == "cedar" and default.mapping_source == "safe_default" and warning
    conversational = plan.model_copy(update={
        "voice_profile": plan.voice_profile.model_copy(update={"gender": "male", "style": "conversational"}),
    })
    assert resolve_voice(conversational, _tts_config())[0].voice == "echo"


def test_mock_tts_generates_normalized_wav_and_manifest(tmp_path: Path) -> None:
    plan = _plan()
    provider = MockTTSProvider()
    result = TTSService(tmp_path, _tts_config()).generate(plan, tmp_path / "run", tmp_path / "out", provider=provider)
    assert result.status == "completed"
    assert provider.call_count == plan.timeline.narration_count
    assert result.fallback_count == 0
    assert all(item.status == "generated" for item in result.segments)
    assert all(item.artifact and Path(item.artifact.audio_file_path or "").is_file() for item in result.segments)
    assert all(item.artifact and item.artifact.sample_rate == 48000 for item in result.segments)
    for name in ("tts-result.json", "tts-manifest.json", "tts-summary.txt"):
        assert (tmp_path / "out" / "tts" / name).is_file()


def test_source_audio_mode_skips_tts_without_provider_call_or_artifacts(tmp_path: Path) -> None:
    plan = _plan("original")
    provider = MockTTSProvider()

    result = TTSService(tmp_path, _tts_config()).generate(
        plan, tmp_path / "run", tmp_path / "out", provider=provider,
    )

    assert plan.tts_eligible is False
    assert result.status == "skipped"
    assert result.skip_reason == "source_audio_mode"
    assert provider.call_count == 0
    assert result.segments == [] and result.artifacts == []
    assert not (tmp_path / "out" / "tts").exists()
    report = tts_report_section(result)
    assert report["tts_invoked"] is False and report["estimated_cost"] == 0


def test_source_audio_mode_does_not_require_cloud_tts_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _tts_config()
    config.tts.provider = "openai"

    result = TTSService(tmp_path, config).generate(
        _plan("original"), tmp_path / "run", tmp_path / "out",
    )

    assert result.status == "skipped"
    assert result.skip_reason == "source_audio_mode"


def test_pipeline_source_audio_mode_has_no_tts_stage_artifacts(tmp_path: Path) -> None:
    plan = _plan("original")
    pipeline = Pipeline(tmp_path, _tts_config())
    result = pipeline._run_tts(
        StageTracker(tmp_path / "state.json"),
        {"items": [{"candidate_id": "candidate-tts-001", "status": "completed", "plan": plan.model_dump(mode="json")} ]},
        tmp_path / "run", tmp_path / "out",
    )

    assert result["status"] == "skipped"
    assert result["tts_invoked"] is False
    assert result["items"][0]["reason"] == "source_audio_mode"
    assert not (tmp_path / "out" / "tts").exists()


def test_final_tts_admission_is_per_candidate_and_only_when_cloud_audio_is_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _tts_config()
    config.tts.provider = "openai"
    original = _plan("original").model_copy(update={"plan_id": "plan-original"})
    voiceover = _plan("voiceover").model_copy(update={"plan_id": "plan-voiceover"})
    pipeline = Pipeline(tmp_path, config)

    result = pipeline._run_tts(
        StageTracker(tmp_path / "state.json"),
        {"items": [
            {
                "candidate_id": "original-candidate", "requested_index": 1,
                "status": "completed", "plan": original.model_dump(mode="json"),
            },
            {
                "candidate_id": "voiceover-candidate", "requested_index": 2,
                "status": "completed", "plan": voiceover.model_dump(mode="json"),
            },
        ]},
        tmp_path / "run", tmp_path / "out",
    )
    by_candidate = {item["candidate_id"]: item for item in result["items"]}

    assert by_candidate["original-candidate"]["status"] == "skipped"
    assert by_candidate["original-candidate"]["reason"] == "source_audio_mode"
    assert by_candidate["voiceover-candidate"]["status"] == "failed"
    assert "TTS_CREDENTIAL_MISSING" in by_candidate["voiceover-candidate"]["error"]


def test_tts_cache_is_segment_based_and_force_recompute_bypasses_it(tmp_path: Path) -> None:
    plan = _plan()
    config = _tts_config()
    first_provider = MockTTSProvider()
    first = TTSService(tmp_path, config).generate(plan, tmp_path / "run-a", tmp_path / "out-a", provider=first_provider)
    second_provider = MockTTSProvider()
    second = TTSService(tmp_path, config).generate(plan, tmp_path / "run-b", tmp_path / "out-b", provider=second_provider)
    assert first_provider.call_count == plan.timeline.narration_count
    assert second_provider.call_count == 0
    assert second.cache_hit_count == plan.timeline.narration_count
    assert [item.artifact.checksum for item in first.segments] == [item.artifact.checksum for item in second.segments]
    third_provider = MockTTSProvider()
    forced = TTSService(tmp_path, config).generate(plan, tmp_path / "run-c", tmp_path / "out-c", force_recompute=True, provider=third_provider)
    assert third_provider.call_count == plan.timeline.narration_count
    assert forced.cache_hit_count == 0


def test_tts_cache_rebinds_results_to_the_current_plan_segment_ids(tmp_path: Path) -> None:
    plan = _plan()
    config = _tts_config()
    service = TTSService(tmp_path, config)
    service.generate(plan, tmp_path / "first", tmp_path / "first-out", provider=MockTTSProvider())
    renamed_segments = [
        segment.model_copy(update={"segment_id": f"{segment.segment_id}-rerun"})
        if isinstance(segment, NarrationSegment) else segment
        for segment in plan.segments
    ]
    reused_plan = plan.model_copy(update={
        "plan_id": "production-candidate-tts-001-rerun", "segments": renamed_segments,
    })
    provider = MockTTSProvider()
    result = service.generate(reused_plan, tmp_path / "rerun", tmp_path / "rerun-out", provider=provider)

    expected_ids = [segment.segment_id for segment in renamed_segments if isinstance(segment, NarrationSegment)]
    assert provider.call_count == 0
    assert result.cache_hit_count == len(expected_ids)
    assert [segment.segment_id for segment in result.segments] == expected_ids
    assert all(segment.production_plan_id == reused_plan.plan_id for segment in result.segments)
    assert all(segment.artifact and segment.artifact.segment_id == segment.segment_id for segment in result.segments)


def test_cache_key_changes_for_text_voice_provider_and_model(tmp_path: Path) -> None:
    plan = _plan()
    config = _tts_config()
    service = TTSService(tmp_path, config)
    first = service.generate(plan, tmp_path / "first", tmp_path / "first-out", provider=MockTTSProvider())
    data = plan.model_dump(mode="json")
    data["segments"][0]["text"] += " Again."
    changed_plan = ProductionPlan.model_validate(data)
    text_provider = MockTTSProvider()
    text_result = service.generate(changed_plan, tmp_path / "text", tmp_path / "text-out", provider=text_provider)
    assert text_provider.call_count == 1
    assert text_result.cache_hit_count == len(first.segments) - 1
    config.tts.voice = "coral"
    voice_provider = MockTTSProvider()
    assert service.generate(plan, tmp_path / "voice", tmp_path / "voice-out", provider=voice_provider).cache_hit_count == 0
    assert voice_provider.call_count == len(first.segments)
    config.tts.model = "mock-tts-v2"
    model_provider = MockTTSProvider()
    service.generate(plan, tmp_path / "model", tmp_path / "model-out", provider=model_provider)
    assert model_provider.call_count == len(first.segments)
    config.tts.provider = "openai"
    provider_provider = MockTTSProvider()
    service.generate(plan, tmp_path / "provider", tmp_path / "provider-out", provider=provider_provider)
    assert provider_provider.call_count == len(first.segments)


def test_corrupt_cache_is_not_reused(tmp_path: Path) -> None:
    plan = _plan()
    config = _tts_config()
    TTSService(tmp_path, config).generate(plan, tmp_path / "first", tmp_path / "first-out", provider=MockTTSProvider())
    cache_file = next((tmp_path / "work" / "tts-cache").glob("*.wav"))
    cache_file.write_bytes(b"corrupt")
    provider = MockTTSProvider()
    result = TTSService(tmp_path, config).generate(plan, tmp_path / "second", tmp_path / "second-out", provider=provider)
    assert provider.call_count >= 1
    assert result.generated_count >= 1


def test_budget_and_disabled_tts_prevent_provider_calls(tmp_path: Path) -> None:
    plan = _plan()
    config = _tts_config()
    config.tts.budget_limit = 0
    provider = MockTTSProvider()
    budget = TTSService(tmp_path, config).generate(plan, tmp_path / "budget", tmp_path / "budget-out", provider=provider)
    assert provider.call_count == 0
    assert {item.fallback_reason for item in budget.segments} == {"budget_exceeded"}
    config.tts.enabled = False
    disabled_provider = MockTTSProvider()
    disabled = TTSService(tmp_path, config).generate(plan, tmp_path / "disabled", tmp_path / "disabled-out", provider=disabled_provider)
    assert disabled_provider.call_count == 0
    assert {item.fallback_reason for item in disabled.segments} == {"ai_disabled"}


def test_mock_tts_reports_no_paid_cost(tmp_path: Path) -> None:
    result = TTSService(tmp_path, _tts_config()).generate(
        _plan(), tmp_path / "mock", tmp_path / "mock-out", provider=MockTTSProvider(),
    )

    assert result.estimated_cost == 0.0
    assert result.actual_cost is None


@pytest.mark.parametrize(
    ("credential", "error_code"),
    [(None, "MISSING"), ("invalid-key", "INVALID")],
)
def test_unusable_openai_key_blocks_an_uncached_cloud_tts_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    credential: str | None, error_code: str,
) -> None:
    if credential is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", credential)
    config = _tts_config()
    config.tts.provider = "openai"
    with pytest.raises(TTSCredentialError, match=f"TTS_CREDENTIAL_{error_code}"):
        TTSService(tmp_path, config).generate(_plan(), tmp_path / "run", tmp_path / "out")

    assert not (tmp_path / "out" / "tts").exists()


def test_complete_cloud_tts_cache_hit_does_not_require_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tts_config()
    config.tts.provider = "openai"
    plan = _plan()
    service = TTSService(tmp_path, config)
    seeded = service.generate(
        plan, tmp_path / "seed-run", tmp_path / "seed-out", provider=MockTTSProvider(),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cached = service.generate(plan, tmp_path / "retry-run", tmp_path / "retry-out")

    assert cached.cache_hit_count == len(seeded.segments)
    assert cached.api_call_count == 0


@pytest.mark.parametrize("mode", ["provider_error", "timeout", "empty_audio", "malformed_response"])
def test_provider_failures_and_invalid_audio_fall_back_without_cache(tmp_path: Path, mode: str) -> None:
    plan = _plan()
    config = _tts_config()
    failed = TTSService(tmp_path, config).generate(plan, tmp_path / mode, tmp_path / f"{mode}-out", provider=MockTTSProvider(mode))
    assert failed.status == "fallback"
    assert all(item.status == "fallback" for item in failed.segments)
    assert all(item.fallback_reason in {"provider_failure", "invalid_audio"} for item in failed.segments)
    assert not list((tmp_path / "work" / "tts-cache").glob("*.json"))


def test_openai_provider_maps_streaming_response_without_secret(tmp_path: Path) -> None:
    request = _request(tmp_path)
    audio = MockTTSProvider().synthesize(request).artifact
    assert audio and audio.audio_file_path
    payload = Path(audio.audio_file_path).read_bytes()

    class Response:
        _request_id = "req_tts_test"
        status_code = 200
        headers = {"x-request-id": "req_tts_test"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(with_streaming_response=SimpleNamespace(create=lambda **kwargs: Response()))))
    provider = OpenAITTSProvider("sk-test-not-real", 5, 0, client_factory=lambda **kwargs: client)
    result = provider.synthesize(request)
    assert result.status == "generated"
    assert result.usage.provider_request_id == "req_tts_test"
    assert result.usage.http_status == 200
    assert "sk-test-not-real" not in result.model_dump_json()


def test_openai_provider_redacts_secret_from_exception(tmp_path: Path) -> None:
    request = _request(tmp_path)
    secret = "sk-fake"
    broken = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(with_streaming_response=SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError(f"Authorization: Bearer {secret}"))
    ))))
    result = OpenAITTSProvider(secret, 5, 0, client_factory=lambda **kwargs: broken).synthesize(request)
    assert result.status == "failed"
    assert secret not in result.model_dump_json()
    assert "[REDACTED]" in result.error.message


def test_openai_provider_auth_rejection_is_a_credential_error(tmp_path: Path) -> None:
    request = _request(tmp_path)

    def reject(**_kwargs):
        error = RuntimeError("unauthorized")
        error.status_code = 401  # type: ignore[attr-defined]
        raise error

    broken = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(
        with_streaming_response=SimpleNamespace(create=reject),
    )))

    with pytest.raises(TTSCredentialError, match="TTS_CREDENTIAL_AUTH_REJECTED"):
        OpenAITTSProvider(
            "sk-fake", 5, 3, client_factory=lambda **_kwargs: broken,
        ).synthesize(request)


def test_tts_only_preserves_existing_mp4_and_cli_flags(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    config = _tts_config()
    pipeline = Pipeline(tmp_path, config, tts_only=True)
    source, work_directory, output_directory = pipeline._prepare_source(str(source_path), None)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "production-plan.json", _plan().model_dump(mode="json"))
    mp4 = output_directory / "old.mp4"
    mp4.write_bytes(b"old-video")
    write_json(output_directory / "report.json", {"output_files": [str(mp4)], "selected_clips_count": 1})
    result = pipeline._run_tts_only(StageTracker(work_directory / "state.json"), source, work_directory, output_directory)
    assert result.output_files == [mp4]
    assert mp4.read_bytes() == b"old-video"
    assert read_json(result.report_path, {})["tts"]["status"] == "completed"
    arguments = build_parser().parse_args([
        "process", "--input", "source.mp4", "--tts-only", "--recompute-tts", "--disable-tts",
        "--tts-provider", "mock", "--tts-voice", "cedar", "--tts-model", "mock-tts", "--tts-budget-limit", "0.01",
    ])
    assert arguments.tts_only and arguments.recompute_tts and arguments.disable_tts


def test_tts_only_requires_existing_plan(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    pipeline = Pipeline(tmp_path, _tts_config(), tts_only=True)
    with pytest.raises(TTSRuntimeError, match="ProductionPlan"):
        pipeline.run(input_path=str(source_path))


def test_duration_validation_rejects_non_audio(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wav")
    validation = validate_audio(path, 1, _tts_config())
    assert validation.status == "invalid"


def test_missing_ffmpeg_and_ffprobe_are_safe_tts_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    with monkeypatch.context() as context:
        context.setattr("app.tts_service.shutil.which", lambda _name: None)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            normalize_audio(source, tmp_path / "normalized.wav", 48000)

    result = TTSService(tmp_path, _tts_config()).generate(_plan(), tmp_path / "run", tmp_path / "out", provider=MockTTSProvider())
    audio = Path(result.segments[0].artifact.audio_file_path or "")
    monkeypatch.setattr("app.tts_service.shutil.which", lambda _name: None)
    validation = validate_audio(audio, 1, _tts_config())
    assert validation.status == "invalid" and validation.message == "ffprobe is unavailable."


def test_duration_validation_has_valid_warning_invalid_states(tmp_path: Path) -> None:
    plan = _plan()
    config = _tts_config()
    result = TTSService(tmp_path, config).generate(plan, tmp_path / "run", tmp_path / "out", provider=MockTTSProvider())
    path = Path(result.segments[0].artifact.audio_file_path)
    actual = result.segments[0].artifact.actual_duration_seconds
    assert validate_audio(path, actual, config).status == "valid"
    config.tts.duration_warning_ratio = 0.05
    config.tts.duration_error_ratio = 0.5
    assert validate_audio(path, actual * 1.2, config).status == "warning"
    assert validate_audio(path, actual * 3, config).status == "invalid"


def test_tts_report_serialization_has_cache_stats_and_no_raw_response(tmp_path: Path) -> None:
    result = TTSService(tmp_path, _tts_config()).generate(_plan(), tmp_path / "run", tmp_path / "out", provider=MockTTSProvider())
    report = tts_report_section(result)
    assert report["status"] == "completed"
    assert report["cache_hit_count"] == 0
    assert report["generated_count"] == len(result.segments)
    assert "base64" not in json.dumps(report).lower()


def test_tts_report_represents_partial_fallback(tmp_path: Path) -> None:
    class PartialProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.good = MockTTSProvider()
            self.bad = MockTTSProvider("provider_error")

        def synthesize(self, request):
            self.calls += 1
            return self.good.synthesize(request) if self.calls == 1 else self.bad.synthesize(request)

    result = TTSService(tmp_path, _tts_config()).generate(_plan(), tmp_path / "run", tmp_path / "out", provider=PartialProvider())
    report = tts_report_section(result)
    assert report["status"] == "partial"
    assert report["generated_count"] == 1
    assert report["fallback_count"] == len(result.segments) - 1
