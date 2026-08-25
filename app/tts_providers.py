from __future__ import annotations

import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any, Callable, Protocol

from app.ai import sanitize_api_error, semantic_failure_kind
from app.config import AppConfig
from app.errors import TTSCredentialError
from app.tts_models import (
    TTSAudioArtifact,
    TTSError as TTSErrorRecord,
    TTSSegmentRequest,
    TTSSegmentResult,
    TTSUsage,
)
from app.utils import utc_now, write_bytes_atomic


class TTSProvider(Protocol):
    """Provider boundary: one request produces audio for one narration segment."""

    def synthesize(self, request: TTSSegmentRequest) -> TTSSegmentResult:
        ...


def _failure(
    request: TTSSegmentRequest, provider: str, error: BaseException | str,
    retries: int = 0, retryable: bool = False, api_call_count: int = 1,
    *secrets: str | None,
) -> TTSSegmentResult:
    return TTSSegmentResult(
        production_plan_id=request.production_plan_id,
        segment_id=request.segment_id,
        status="failed",
        provider=provider,
        model=request.provider_config.model,
        voice=request.voice.voice,
        language=request.language,
        source_text=request.narration_text,
        cache_key=request.cache_key,
        usage=TTSUsage(
            character_count=len(request.normalized_text), api_call_count=api_call_count,
            retries=retries,
        ),
        error=TTSErrorRecord(
            error_type=type(error).__name__ if not isinstance(error, str) else "ProviderError",
            message=sanitize_api_error(error), retryable=retryable, occurred_at=utc_now(),
        ),
        created_at=utc_now(),
    )


class OpenAITTSProvider:
    """Official OpenAI Audio Speech SDK provider with bounded application retries."""

    name = "openai"

    def __init__(
        self, api_key: str, timeout_seconds: float, max_retries: int,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client_factory = client_factory

    def synthesize(self, request: TTSSegmentRequest) -> TTSSegmentResult:
        if not self.api_key:
            return _failure(request, self.name, "OPENAI_API_KEY is not configured", api_call_count=0)
        try:
            client = self._client()
        except Exception as error:
            return _failure(request, self.name, error, 0, False, 0, self.api_key)
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.audio.speech.with_streaming_response.create(
                    model=request.provider_config.model,
                    voice=request.voice.voice,
                    input=request.normalized_text,
                    instructions=request.voice.instructions,
                    response_format="wav",
                    speed=request.provider_config.speed,
                )
                with response as streamed:
                    audio = streamed.read()
                    request_id, http_status = _response_trace(streamed)
                if not isinstance(audio, bytes) or not audio:
                    return _failure(request, self.name, "OpenAI Speech returned empty audio", attempt, False, 1, self.api_key)
                raw_path = Path(request.raw_output_path)
                write_bytes_atomic(raw_path, audio)
                artifact = TTSAudioArtifact(
                    production_plan_id=request.production_plan_id,
                    segment_id=request.segment_id,
                    audio_file_path=str(raw_path),
                    raw_audio_file_path=str(raw_path),
                    output_format="wav",
                    sample_rate=24000,
                    byte_size=len(audio),
                    estimated_duration_seconds=request.estimated_duration_seconds,
                    cache_key=request.cache_key,
                    origin="provider",
                    created_at=utc_now(),
                )
                return TTSSegmentResult(
                    production_plan_id=request.production_plan_id,
                    segment_id=request.segment_id,
                    status="generated",
                    provider=self.name,
                    model=request.provider_config.model,
                    voice=request.voice.voice,
                    language=request.language,
                    source_text=request.narration_text,
                    cache_key=request.cache_key,
                    artifact=artifact,
                    usage=TTSUsage(
                        character_count=len(request.normalized_text), api_call_count=1,
                        retries=attempt, provider_request_id=request_id, http_status=http_status,
                    ),
                    created_at=utc_now(),
                )
            except Exception as error:
                if semantic_failure_kind(error) == "auth_rejected":
                    raise TTSCredentialError(
                        "TTS_CREDENTIAL_AUTH_REJECTED: OpenAI rejected the TTS credential."
                    ) from error
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2 ** attempt), 1.0))
        assert last_error is not None
        return _failure(
            request, self.name, last_error, self.max_retries, True,
            self.max_retries + 1, self.api_key,
        )

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)
        from openai import OpenAI

        # SDK retries are disabled here: retry accounting belongs to the typed result.
        return OpenAI(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)


class MockTTSProvider:
    """Deterministic offline provider that generates a non-silent WAV fixture."""

    name = "mock"

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.call_count = 0

    def synthesize(self, request: TTSSegmentRequest) -> TTSSegmentResult:
        self.call_count += 1
        if self.mode == "timeout":
            return _failure(request, self.name, TimeoutError("mock TTS timeout"), retryable=True)
        if self.mode in {"provider_error", "malformed_response"}:
            return _failure(request, self.name, RuntimeError("mock TTS provider failure"))
        raw_path = Path(request.raw_output_path)
        if self.mode == "empty_audio":
            write_bytes_atomic(raw_path, b"")
        else:
            write_bytes_atomic(raw_path, _mock_wav(request.estimated_duration_seconds))
        artifact = TTSAudioArtifact(
            production_plan_id=request.production_plan_id,
            segment_id=request.segment_id,
            audio_file_path=str(raw_path),
            raw_audio_file_path=str(raw_path),
            output_format="wav",
            sample_rate=24000,
            byte_size=raw_path.stat().st_size,
            estimated_duration_seconds=request.estimated_duration_seconds,
            cache_key=request.cache_key,
            origin="provider",
            created_at=utc_now(),
        )
        return TTSSegmentResult(
            production_plan_id=request.production_plan_id,
            segment_id=request.segment_id,
            status="generated",
            provider=self.name,
            model=request.provider_config.model,
            voice=request.voice.voice,
            language=request.language,
            source_text=request.narration_text,
            cache_key=request.cache_key,
            artifact=artifact,
            usage=TTSUsage(character_count=len(request.normalized_text), api_call_count=1, retries=0, http_status=200),
            created_at=utc_now(),
        )


class LocalFallbackTTSProvider:
    """Explicit no-audio fallback. It never fabricates a successful silent narration."""

    name = "local"

    def synthesize(self, request: TTSSegmentRequest) -> TTSSegmentResult:
        return TTSSegmentResult(
            production_plan_id=request.production_plan_id,
            segment_id=request.segment_id,
            status="fallback",
            provider=self.name,
            model=request.provider_config.model,
            voice=request.voice.voice,
            language=request.language,
            source_text=request.narration_text,
            cache_key=request.cache_key,
            artifact=TTSAudioArtifact(
                production_plan_id=request.production_plan_id,
                segment_id=request.segment_id,
                output_format="wav",
                sample_rate=request.provider_config.sample_rate,
                byte_size=0,
                estimated_duration_seconds=request.estimated_duration_seconds,
                cache_key=request.cache_key,
                origin="silent_placeholder",
                created_at=utc_now(),
            ),
            usage=TTSUsage(character_count=len(request.normalized_text), api_call_count=0, retries=0),
            fallback_reason="local_placeholder",
            created_at=utc_now(),
        )


def get_tts_provider(config: AppConfig) -> TTSProvider:
    provider = config.tts.provider
    if provider == "mock":
        return MockTTSProvider(config.tts.mock_mode)
    if provider == "local":
        return LocalFallbackTTSProvider()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise TTSCredentialError("TTS_CREDENTIAL_MISSING: OPENAI_API_KEY is not configured")
    return OpenAITTSProvider(api_key, config.tts.timeout_seconds, config.tts.max_retries)


def _mock_wav(estimated_seconds: float) -> bytes:
    """A short deterministic tone makes mock artifacts detectable as real WAV test data."""

    sample_rate = 24000
    duration = max(0.10, min(float(estimated_seconds), 12.0))
    frames = int(sample_rate * duration)
    payload = bytearray()
    for index in range(frames):
        sample = int(1000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        payload.extend(struct.pack("<h", sample))
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(bytes(payload))
    return buffer.getvalue()


def _response_trace(response: Any) -> tuple[str | None, int | None]:
    request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        request_id = request_id or headers.get("x-request-id")
    raw = getattr(response, "http_response", None) or getattr(response, "_response", None)
    status = getattr(raw, "status_code", None) or getattr(response, "status_code", None)
    return (str(request_id) if request_id else None, int(status) if isinstance(status, int) else None)
