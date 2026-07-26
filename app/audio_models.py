from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AUDIO_PROJECT_SCHEMA_VERSION = "3C.0"
AUDIO_TIMELINE_VERSION = "3C.0"


class AudioClip(BaseModel):
    """A concrete, deterministic item in the final audio timeline."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
    clip_type: str
    order: int = Field(ge=1)
    track_type: Literal["narration", "dialogue", "music", "effects", "silence"]
    production_segment_id: str | None = None
    timeline_start_seconds: float = Field(ge=0)
    timeline_end_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    audio_file_path: str | None = None
    checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    status: Literal["ready", "placeholder", "skipped", "failed"]
    cache_key: str | None = None

    @model_validator(mode="after")
    def _valid_duration(self) -> "AudioClip":
        if self.timeline_end_seconds < self.timeline_start_seconds:
            raise ValueError("timeline end must not precede start")
        if abs((self.timeline_end_seconds - self.timeline_start_seconds) - self.duration_seconds) > 0.02:
            raise ValueError("timeline range must match clip duration")
        return self


class NarrationClip(AudioClip):
    clip_type: Literal["narration"] = "narration"
    track_type: Literal["narration"] = "narration"
    tts_segment_id: str
    normalized_tts_path: str
    source_bed_path: str | None = None
    ducked_source_bed_path: str | None = None
    loudness_normalized: bool
    target_lufs: float
    duck_level: float = Field(gt=0, le=1)
    attack_seconds: float = Field(ge=0)
    release_seconds: float = Field(ge=0)


class DialogueClip(AudioClip):
    clip_type: Literal["dialogue"] = "dialogue"
    track_type: Literal["dialogue"] = "dialogue"
    fact_id: str
    transcript_segment_id: int = Field(ge=0)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    speaker: str

    @model_validator(mode="after")
    def _ordered_source_timestamps(self) -> "DialogueClip":
        if self.source_end_seconds < self.source_start_seconds:
            raise ValueError("source_end_seconds must not precede source_start_seconds")
        return self


class MusicClip(AudioClip):
    clip_type: Literal["music"] = "music"
    track_type: Literal["music"] = "music"
    status: Literal["placeholder"] = "placeholder"


class EffectClip(AudioClip):
    clip_type: Literal["effect"] = "effect"
    track_type: Literal["effects"] = "effects"
    status: Literal["placeholder"] = "placeholder"


class SilenceClip(AudioClip):
    clip_type: Literal["silence"] = "silence"
    track_type: Literal["silence"] = "silence"
    pause_reason: str


AudioClipModel = Annotated[
    NarrationClip | DialogueClip | MusicClip | EffectClip | SilenceClip,
    Field(discriminator="clip_type"),
]


class AudioTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    track_type: Literal["narration", "dialogue", "music", "effects"]
    clips: list[AudioClipModel] = Field(default_factory=list)
    status: Literal["ready", "empty", "placeholder"]
    gain_db: float = 0.0

    @model_validator(mode="after")
    def _track_clip_types_match(self) -> "AudioTrack":
        expected = {"effects": "effects"}.get(self.track_type, self.track_type)
        if any(clip.track_type != expected for clip in self.clips):
            raise ValueError("audio track contains a clip for another track")
        return self


class AudioTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_version: str = AUDIO_TIMELINE_VERSION
    clips: list[AudioClipModel]
    duration_seconds: float = Field(ge=0)
    source_of_truth: Literal["production_plan"] = "production_plan"

    @model_validator(mode="after")
    def _timeline_is_sequential(self) -> "AudioTimeline":
        if [clip.order for clip in self.clips] != sorted(clip.order for clip in self.clips):
            raise ValueError("audio timeline clips must be ordered")
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise ValueError("audio timeline clip ids must be unique")
        previous = 0.0
        for clip in self.clips:
            if abs(clip.timeline_start_seconds - previous) > 0.02:
                raise ValueError("audio timeline cannot contain an implicit gap")
            previous = clip.timeline_end_seconds
        if abs(previous - self.duration_seconds) > 0.02:
            raise ValueError("timeline duration must match final clip")
        return self


class DuckingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    attack_seconds: float = Field(ge=0)
    release_seconds: float = Field(ge=0)
    duck_level: float = Field(gt=0, le=1)
    preserve_original_events: bool
    policy: Literal["audible_source_bed_with_smooth_ducking"]


class LoudnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narration_target_lufs: float
    narration_true_peak_db: float
    narration_lra: float = Field(gt=0)
    normalized_narration_count: int = Field(ge=0)


class AudioMix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mixed_audio_path: str | None = None
    checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    duration_seconds: float = Field(ge=0)
    sample_rate: int = Field(ge=8000, le=192000)
    channels: Literal[1] = 1
    bit_depth: Literal[16] = 16
    ducking: DuckingConfig
    loudness: LoudnessConfig
    status: Literal["completed", "partial", "skipped", "failed"]


class AudioExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["wav"] = "wav"
    codec: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate: int = Field(ge=8000, le=192000)
    channels: Literal[1] = 1
    path: str | None = None
    byte_size: int = Field(ge=0)
    status: Literal["completed", "skipped", "failed"]


class AudioValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["valid", "warning", "invalid", "not_applicable"]
    checked_clip_count: int = Field(ge=0)
    failed_clip_count: int = Field(ge=0)
    mix_duration_seconds: float | None = Field(default=None, ge=0)
    messages: list[str] = Field(default_factory=list)


class AudioMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = AUDIO_PROJECT_SCHEMA_VERSION
    audio_project_id: str
    production_plan_id: str
    source_id: str
    source_media_path: str
    audio_mode: Literal["original", "original_enhanced", "voiceover", "replace_voice", "mixed"] = "original"
    tts_result_path: str | None = None
    transcript_path: str | None = None
    created_at: str
    completed_at: str
    production_plan_mutated: Literal[False] = False
    tts_artifacts_mutated: Literal[False] = False
    video_rendered: Literal[False] = False


class AudioProject(BaseModel):
    """The typed source of truth for the Goal 3C audio-only handoff."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    audio_mode: Literal["original", "original_enhanced", "voiceover", "replace_voice", "mixed"] = "original"
    status: Literal["completed", "partial", "skipped", "failed"]
    timeline: AudioTimeline
    tracks: list[AudioTrack]
    mix: AudioMix
    export: AudioExport
    metadata: AudioMetadata
    validation: AudioValidation
    cache: dict[str, int | bool | list[bool]]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _required_tracks_exist(self) -> "AudioProject":
        if {track.track_type for track in self.tracks} != {"narration", "dialogue", "music", "effects"}:
            raise ValueError("AudioProject must expose narration, dialogue, music and effects tracks")
        return self
