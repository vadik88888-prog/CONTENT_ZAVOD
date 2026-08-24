from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app.audio_features import analyse_audio, window_audio_features
from app.config import AppConfig, AudioAnalysisConfig
from app.errors import SemanticCredentialError
from app.intelligence import local_rank, merge_ai_ranking, shortlist
from app.intelligence_candidates import generate_candidates
from app.local_scoring import score_candidates
from app.models import Candidate
from app.pipeline import Pipeline
from app.scene_detection import parse_scene_output
from app.transcript_features import analyse_transcript, candidate_transcript_features


def _transcript() -> dict:
    return {
        "language": "ru",
        "duration": 45,
        "segments": [
            {"start": 0, "end": 4, "text": "Почему это важная ошибка?"},
            {"start": 5, "end": 20, "text": "Сначала проверяем результат, а затем исправляем процесс."},
            {"start": 21, "end": 38, "text": "Главная ошибка — делать всё без понятного плана."},
        ],
    }


def test_transcript_features_detect_boundaries_pause_hook_and_fillers() -> None:
    features = analyse_transcript(_transcript(), AppConfig().transcript_features)

    assert features["segments"][0]["sentence_start"]
    assert features["segments"][0]["sentence_end"]
    assert features["segments"][1]["pause_before_seconds"] == 1.0
    assert features["segments"][0]["hook_phrase_score"] >= 20
    assert features["segments"][2]["hook_phrase_score"] >= 20
    assert candidate_transcript_features(0, 20, features)["word_count"] > 0


def test_transcript_features_support_english_and_mid_sentence() -> None:
    transcript = {"language": "en", "segments": [
        {"start": 0, "end": 2, "text": "and this continues"},
        {"start": 2, "end": 5, "text": "Why does this matter?"},
    ]}
    features = analyse_transcript(transcript, AppConfig().transcript_features)

    assert features["segments"][0]["starts_mid_sentence"]
    assert features["segments"][1]["hook_phrase_score"] >= 20


def test_audio_analysis_finds_silence_and_energy(tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(1000)
        output.writeframes((b"\x00\x00" * 500) + (b"\xff\x3f" * 500) + (b"\x00\x00" * 500))
    features = analyse_audio(wav, AudioAnalysisConfig(window_seconds=0.1, min_silence_seconds=0.2))

    assert features["silence_intervals"]
    assert features["energy_peak"] > 0
    assert window_audio_features(0.5, 1.0, features)["audio_energy"] > 0


def test_scene_parser_reads_multiple_boundaries() -> None:
    output = "pts_time:10.0\nlavfi.scene_score=0.42\npts_time:22.5\nlavfi.scene_score=0.71"
    boundaries = parse_scene_output(output, 30)

    assert [item["timestamp"] for item in boundaries] == [10.0, 22.5]
    assert boundaries[0]["distance_to_next_scene"] == 12.5


def test_scene_parser_accepts_empty_output() -> None:
    assert parse_scene_output("", 30) == []


def test_candidate_generation_and_local_scoring_are_deterministic() -> None:
    transcript = _transcript()
    config = AppConfig()
    transcript_data = analyse_transcript(transcript, config.transcript_features)
    audio = {"energy_frames": [], "silence_intervals": []}
    scenes = {"boundaries": [{"timestamp": 20, "scene_change_score": 0.5}]}
    candidates = generate_candidates(transcript, transcript_data, audio, scenes, config.candidate_generation)
    scored = score_candidates(candidates, audio, scenes, config.scoring)

    assert scored
    assert all(config.candidate_generation.min_duration_seconds <= item.duration <= config.candidate_generation.max_duration_seconds for item in scored)
    assert all(0 <= item.local_quality_score <= 100 for item in scored)
    assert all(item.explanations for item in scored)
    assert shortlist(scored, 1) == shortlist(scored, 1)


def test_local_fallback_keeps_candidates_when_ai_fails() -> None:
    candidate = Candidate("one", 0, 20, "Why this works?", local_quality_score=72, local_scores={"hook": 70, "completeness": 80, "clarity": 70, "context_independence": 70})
    ranked = merge_ai_ranking([candidate], [], ai_ok=False)

    assert ranked[0].score == 72
    assert ranked[0].selected


def test_scoring_weights_must_sum_to_one() -> None:
    config = AppConfig()
    config.scoring.weights["hook"] = 0.5

    try:
        config.validate()
    except Exception as error:
        assert "weights" in str(error)
    else:
        raise AssertionError("Invalid scoring weights must be rejected")


def test_pipeline_ai_error_uses_local_fallback(tmp_path: Path, monkeypatch) -> None:
    candidate = Candidate(
        "one", 0, 20, "Why this works?", local_quality_score=72,
        local_scores={"hook": 70, "completeness": 80, "clarity": 70, "context_independence": 70},
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "a" * 40)
    monkeypatch.setattr("app.pipeline.get_scorer", unavailable)
    pipeline = Pipeline(tmp_path, AppConfig())
    data = pipeline._ai_rerank([candidate], [candidate], {"segments": []}, tmp_path / "ai.json")

    assert data["ai_fallback_used"]
    assert data["selection_mode"] == "local-fallback"
    assert data["candidates"][0]["score"] == 72
    assert data["ai"]["provider"] == "openai"
    assert data["ai"]["reason"] == "provider_temporarily_unavailable"
    assert data["ai"]["execution_state"] == "degraded"
    assert data["ai"]["retryable"] is True
    assert any("degraded" in warning for warning in pipeline.warnings)


def test_semantic_ai_auto_reaches_configured_provider_with_virality_enabled(tmp_path: Path, monkeypatch) -> None:
    candidate = Candidate(
        "one", 0, 20, "Why this works?", local_quality_score=72,
        local_scores={"hook": 70, "completeness": 80, "clarity": 70, "context_independence": 70},
    )
    calls: list[list[str]] = []

    class Provider:
        def score(self, candidates, transcript):
            calls.append([item.id for item in candidates])
            return local_rank(candidates), {
                "provider": "openai", "model": "gpt-5-mini",
                "input_tokens": 12, "output_tokens": 8, "retries": 0, "api_errors": [],
            }

    config = AppConfig()
    config.virality.enabled = True
    config.virality.semantic_ai_mode = "auto"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "b" * 40)
    monkeypatch.setattr("app.pipeline.get_scorer", lambda *_args, **_kwargs: Provider())

    data = Pipeline(tmp_path, config)._ai_rerank(
        [candidate], [candidate], {"segments": []}, tmp_path / "ai-auto.json",
    )

    assert calls == [["one"]]
    assert data["ai_reranking_used"] is True
    assert data["selection_mode"] == "ai-reranked"
    assert data["ai"]["provider"] == "openai"
    assert data["ai"]["execution_state"] == "completed"
    assert data["ai"]["reason"] == "semantic_ai_completed"


@pytest.mark.parametrize(
    ("credential", "reason"),
    [(None, "MISSING"), ("invalid-key-format", "INVALID")],
)
def test_unusable_semantic_credentials_block_analysis_before_provider_call(
    tmp_path: Path, monkeypatch, credential: str | None, reason: str,
) -> None:
    candidate = Candidate(
        "one", 0, 20, "Why this works?", local_quality_score=72,
        local_scores={"hook": 70, "completeness": 80, "clarity": 70, "context_independence": 70},
    )
    config = AppConfig()
    config.virality.enabled = True
    config.virality.semantic_ai_mode = "auto"
    if credential is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", credential)
    monkeypatch.setattr(
        "app.pipeline.get_scorer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )

    path = tmp_path / "ai-unusable.json"
    with pytest.raises(SemanticCredentialError, match=f"SEMANTIC_CREDENTIAL_{reason}"):
        Pipeline(tmp_path, config)._ai_rerank(
            [candidate], [candidate], {"segments": []}, path,
        )

    assert not path.exists()


def test_provider_rejected_semantic_credential_blocks_without_local_success(
    tmp_path: Path, monkeypatch,
) -> None:
    candidate = Candidate(
        "one", 0, 20, "Why this works?", local_quality_score=72,
        local_scores={"hook": 70, "completeness": 80, "clarity": 70, "context_independence": 70},
    )

    class RejectedProvider:
        @staticmethod
        def score(candidates, _transcript):
            return local_rank(candidates), {
                "provider": "openai",
                "model": "gpt-5.6-terra",
                "api_errors": ["request_failure: rejected"],
                "failure_kind": "auth_rejected",
            }

    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "r" * 40)
    monkeypatch.setattr("app.pipeline.get_scorer", lambda *_args, **_kwargs: RejectedProvider())
    path = tmp_path / "ai-rejected.json"

    with pytest.raises(SemanticCredentialError, match="AUTH_REJECTED"):
        Pipeline(tmp_path, AppConfig())._ai_rerank(
            [candidate], [candidate], {"segments": []}, path,
        )

    assert not path.exists()
