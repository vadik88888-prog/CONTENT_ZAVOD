from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TTS_SCHEMA_VERSION = "3B.0"


class TTSVoiceConfig(BaseModel):
    """Resolved provider voice; never a cloned or user-supplied voice asset."""

    model_config = ConfigDict(extra="forbid")

    voice: str = Field(min_length=1, max_length=80)
    gender: str = Field(min_length=1, max_length=32)
    style: str = Field(min_length=1, max_length=32)
    language: str = Field(min_length=1, max_length=16)
    instructions: str = Field(min_length=1, max_length=500)
    mapping_source: Literal["config_override", "voice_registry", "safe_default"]


class TTSProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "mock", "local"]
    model: str = Field(min_length=1, max_length=120)
    timeout_seconds: float = Field(gt=0, le=300)
    max_retries: int = Field(ge=0, le=5)
    speed: float = Field(gt=0, le=4)
    output_format: Literal["wav"] = "wav"
    sample_rate: int = Field(ge=8000, le=192000)
    provider_config_version: str = Field(min_length=1, max_length=64)


class TTSRequest(BaseModel):
    """A batch-level trace; segment requests remain independently cacheable."""

    model_config = ConfigDict(extra="forbid")

    production_plan_id: str = Field(min_length=1)
    provider_config: TTSProviderConfig
    voice: TTSVoiceConfig
    created_at: str = Field(min_length=1)


class TTSSegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    production_plan_id: str = Field(min_length=1)
    segment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
    narration_text: str = Field(min_length=1, max_length=4000)
    normalized_text: str = Field(min_length=1, max_length=4000)
    language: str = Field(min_length=1, max_length=16)
    voice: TTSVoiceConfig
    provider_config: TTSProviderConfig
    estimated_duration_seconds: float = Field(ge=0)
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_output_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _text_is_preserved_except_whitespace(self) -> "TTSSegmentRequest":
        if "\x00" in self.normalized_text:
            raise ValueError("normalized text cannot contain NUL")
        return self


class TTSUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_count: int = Field(ge=0)
    api_call_count: int = Field(ge=0)
    retries: int = Field(ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    actual_cost: float | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)


class TTSError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool = False
    occurred_at: str = Field(min_length=1)


class TTSValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["valid", "warning", "invalid"]
    estimated_duration_seconds: float = Field(ge=0)
    actual_duration_seconds: float | None = Field(default=None, ge=0)
    difference_seconds: float | None = None
    difference_ratio: float | None = Field(default=None, ge=0)
    minimum_audio_duration: float = Field(ge=0)
    maximum_segment_duration: float = Field(gt=0)
    message: str | None = None


class TTSAudioArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    production_plan_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    audio_file_path: str | None = None
    raw_audio_file_path: str | None = None
    checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    output_format: Literal["wav"] = "wav"
    sample_rate: int = Field(ge=8000, le=192000)
    channels: Literal[1] = 1
    bit_depth: Literal[16] = 16
    byte_size: int = Field(ge=0)
    estimated_duration_seconds: float = Field(ge=0)
    actual_duration_seconds: float | None = Field(default=None, ge=0)
    cache_key: str = Field(min_length=1)
    origin: Literal["provider", "cache", "silent_placeholder"]
    created_at: str = Field(min_length=1)


class TTSSegmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    production_plan_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    status: Literal["generated", "cached", "fallback", "failed"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    cache_key: str = Field(min_length=1)
    artifact: TTSAudioArtifact | None = None
    validation: TTSValidationResult | None = None
    usage: TTSUsage
    fallback_reason: str | None = None
    error: TTSError | None = None
    created_at: str = Field(min_length=1)


class TTSMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = TTS_SCHEMA_VERSION
    production_plan_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    completed_at: str = Field(min_length=1)
    source_of_truth: Literal["production_plan"] = "production_plan"
    production_plan_mutated: Literal[False] = False
    normalized_format: Literal["wav_pcm_s16le"] = "wav_pcm_s16le"
    normalized_sample_rate: int = Field(ge=8000, le=192000)


class TTSGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: TTSRequest
    metadata: TTSMetadata
    status: Literal["completed", "partial", "fallback", "skipped", "failed"]
    segments: list[TTSSegmentResult]
    estimated_cost: float = Field(ge=0)
    actual_cost: float | None = Field(default=None, ge=0)
    total_duration_seconds: float = Field(ge=0)
    validation_status: Literal["valid", "warning", "invalid", "not_applicable"]
    cache_hit_count: int = Field(ge=0)
    generated_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    api_call_count: int = Field(ge=0)
    api_errors: list[TTSError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    artifacts: list[str] = Field(default_factory=list)
