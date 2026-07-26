from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VoiceProfile(BaseModel):
    """Placeholder only; Goal 3A never synthesizes or clones a voice."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = "default-documentary"
    gender: Literal["male", "female", "neutral"] = "neutral"
    style: Literal["calm", "energetic", "documentary", "conversational"] = "documentary"
    language: str
    is_placeholder: Literal[True] = True


class AudioLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_id: str
    layer_type: Literal["narration", "original_dialogue", "music", "effects"]
    status: Literal["placeholder"] = "placeholder"
    source_asset: None = None


class ProductionSegment(BaseModel):
    """Common contract for every future production segment."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    segment_type: str
    order: int = Field(ge=1)
    estimated_duration_seconds: float = Field(ge=0)
    timeline_included: bool
    linked_segment_ids: list[str] = Field(default_factory=list)


class NarrationSegment(ProductionSegment):
    segment_type: Literal["narration"] = "narration"
    text: str = Field(min_length=1, max_length=4000)
    narration_role: Literal["intro", "body", "outro", "cta"]
    source_sentence_id: str
    fact_ids: list[str] = Field(min_length=1)
    source_segment_ids: list[int] = Field(min_length=1)
    word_count: int = Field(ge=1)
    words_per_second: float = Field(gt=0, le=8)
    voice_profile_id: str
    timeline_included: Literal[True] = True


class DialogueSegment(ProductionSegment):
    """A mapping placeholder, not an extracted or mixed audio asset."""

    segment_type: Literal["original_dialogue"] = "original_dialogue"
    fact_id: str
    transcript_segment_id: int = Field(ge=0)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    source_text: str = Field(min_length=1)
    speaker: str
    confidence: float = Field(ge=0, le=1)
    is_placeholder: Literal[True] = True
    timeline_included: Literal[False] = False

    @model_validator(mode="after")
    def _timestamps_are_ordered(self) -> "DialogueSegment":
        if self.source_end_seconds < self.source_start_seconds:
            raise ValueError("source_end_seconds must not precede source_start_seconds")
        return self


class PauseSegment(ProductionSegment):
    segment_type: Literal["pause"] = "pause"
    reason: Literal["narration_transition", "intro_breath", "outro_breath"]
    timeline_included: Literal[True] = True


ProductionSegmentModel = Annotated[
    NarrationSegment | DialogueSegment | PauseSegment,
    Field(discriminator="segment_type"),
]


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    order: int = Field(ge=1)
    estimated_start_seconds: float = Field(ge=0)
    estimated_end_seconds: float = Field(ge=0)
    included_in_master_timeline: bool
    linked_segment_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "TimelineEntry":
        if self.estimated_end_seconds < self.estimated_start_seconds:
            raise ValueError("timeline end must not precede start")
        return self


class TimelineEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_version: str
    estimated_duration_seconds: float = Field(ge=0)
    narration_count: int = Field(ge=0)
    dialogue_count: int = Field(ge=0)
    pause_count: int = Field(ge=0)
    entries: list[TimelineEntry]


class SubtitleCue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cue_id: str
    text: str = Field(min_length=1)
    estimated_start_seconds: float = Field(ge=0)
    estimated_end_seconds: float = Field(ge=0)
    speaker: str
    segment_id: str
    is_placeholder: Literal[True] = True


class SubtitleTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    language: str
    cues: list[SubtitleCue]
    status: Literal["placeholder"] = "placeholder"


class ProductionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_version: str
    candidate_id: str
    source_id: str
    final_script_hash: str
    build_strategy: Literal["deterministic_local"] = "deterministic_local"
    original_audio_preserved: Literal[True] = True
    original_subtitles_preserved: Literal[True] = True
    tts_generated: Literal[False] = False
    audio_mix_generated: Literal[False] = False
    render_generated: Literal[False] = False


class ProductionPlan(BaseModel):
    """Single source of truth for future TTS/mix/subtitle-sync phases."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    status: Literal["draft", "skipped"] = "draft"
    segments: list[ProductionSegmentModel]
    dialogue_mappings: list[DialogueSegment]
    timeline: TimelineEstimate
    voice_profile: VoiceProfile
    audio_layers: list[AudioLayer]
    subtitle_track: SubtitleTrack
    metadata: ProductionMetadata
    audio_mode: Literal["original", "original_enhanced", "voiceover", "replace_voice", "mixed"] = "original"
    tts_eligible: bool = False
    audio_mode_reason: str = "source_audio_mode"

    @model_validator(mode="before")
    @classmethod
    def _migrate_pre_audio_mode_plans(cls, value: object) -> object:
        """Old cached plans with narration retain their historical voiceover intent."""

        if not isinstance(value, dict) or "audio_mode" in value:
            return value
        migrated = dict(value)
        has_narration = any(
            isinstance(item, dict) and item.get("segment_type") == "narration"
            for item in migrated.get("segments", [])
        )
        migrated["audio_mode"] = "voiceover" if has_narration else "original"
        migrated["tts_eligible"] = has_narration
        migrated["audio_mode_reason"] = "legacy_plan_migration"
        return migrated

    @model_validator(mode="after")
    def _validate_relationships(self) -> "ProductionPlan":
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("ProductionPlan segment ids must be unique")
        if [segment.order for segment in self.segments] != sorted(segment.order for segment in self.segments):
            raise ValueError("ProductionPlan segments must be ordered")
        dialogue_ids = {segment.segment_id for segment in self.segments if isinstance(segment, DialogueSegment)}
        if {segment.segment_id for segment in self.dialogue_mappings} != dialogue_ids:
            raise ValueError("dialogue_mappings must mirror dialogue placeholder segments")
        if {entry.segment_id for entry in self.timeline.entries} != set(segment_ids):
            raise ValueError("Timeline must contain every production segment")
        has_narration = any(isinstance(segment, NarrationSegment) for segment in self.segments)
        if self.audio_mode in {"original", "original_enhanced"} and has_narration:
            raise ValueError("source audio modes cannot contain generated narration segments")
        if self.tts_eligible and (not has_narration or self.audio_mode not in {"voiceover", "replace_voice", "mixed"}):
            raise ValueError("TTS eligibility requires explicit voiceover intent and narration")
        return self
