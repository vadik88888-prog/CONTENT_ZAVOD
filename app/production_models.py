from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BOUNDARY_EPSILON_SECONDS = 0.001


class BoundaryRange(BaseModel):
    """A source-time interval accepted by the semantic boundary decision."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "BoundaryRange":
        if self.end_seconds < self.start_seconds:
            raise ValueError("boundary range end must not precede start")
        return self


class BoundaryRequirement(BaseModel):
    """Source evidence which must survive the hand-off into a production plan."""

    model_config = ConfigDict(extra="forbid")

    requirement_type: Literal["hook", "completion", "payoff"]
    required: bool = True
    source_range: BoundaryRange
    transcript_segment_id: int = Field(ge=0)
    reason: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)


class BoundaryDecision(BaseModel):
    """Versioned, persisted semantic-boundary contract for one candidate.

    It deliberately embeds the small set of boundary evidence that a later
    ProductionPlan needs.  This keeps render-only reuse deterministic and does
    not create a parallel boundary pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["5C.1"]
    decision_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    rough_range: BoundaryRange
    refined_range: BoundaryRange
    allowed_source_range: BoundaryRange
    start_reason: str = Field(min_length=1)
    end_reason: str = Field(min_length=1)
    word_integrity: bool
    sentence_integrity: bool
    semantic_completion: bool
    payoff_preserved: bool
    continuation_risk: float = Field(ge=0, le=1)
    continuation_risk_threshold: float = Field(ge=0, le=1)
    pre_roll_seconds: float = Field(ge=0)
    post_roll_seconds: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    start_evidence: dict[str, Any] = Field(default_factory=dict)
    end_evidence: dict[str, Any] = Field(default_factory=dict)
    pause_evidence: dict[str, Any] = Field(default_factory=dict)
    question_context: dict[str, Any] = Field(default_factory=dict)
    required_evidence: list[BoundaryRequirement] = Field(default_factory=list)
    safe_start_points: list[float] = Field(default_factory=list)
    safe_end_points: list[float] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "BoundaryDecision":
        allowed = self.allowed_source_range
        refined = self.refined_range
        if (
            refined.start_seconds < allowed.start_seconds - BOUNDARY_EPSILON_SECONDS
            or refined.end_seconds > allowed.end_seconds + BOUNDARY_EPSILON_SECONDS
        ):
            raise ValueError("refined boundary range must stay within allowed source range")
        for requirement in self.required_evidence:
            source_range = requirement.source_range
            if (
                source_range.start_seconds < allowed.start_seconds - BOUNDARY_EPSILON_SECONDS
                or source_range.end_seconds > allowed.end_seconds + BOUNDARY_EPSILON_SECONDS
            ):
                raise ValueError("boundary requirement range must stay within allowed source range")
        return self


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


class SourceSegmentRange(BaseModel):
    """A transcript-backed source interval used by an existing plan segment."""

    model_config = ConfigDict(extra="forbid")

    transcript_segment_id: int = Field(ge=0)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _timestamps_are_ordered(self) -> "SourceSegmentRange":
        if self.source_end_seconds < self.source_start_seconds:
            raise ValueError("source_end_seconds must not precede source_start_seconds")
        return self


class NarrationSegment(ProductionSegment):
    segment_type: Literal["narration"] = "narration"
    text: str = Field(min_length=1, max_length=4000)
    narration_role: Literal["intro", "body", "outro", "cta"]
    source_sentence_id: str
    fact_ids: list[str] = Field(min_length=1)
    source_segment_ids: list[int] = Field(min_length=1)
    source_ranges: list[SourceSegmentRange] = Field(default_factory=list)
    word_count: int = Field(ge=1)
    words_per_second: float = Field(gt=0, le=8)
    voice_profile_id: str
    boundary_decision_id: str | None = None
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
    # A 5C plan binds every extracted source interval to the decision that
    # authorized it. None is retained only for pre-5C cached plans.
    boundary_decision_id: str | None = None
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
    # None is the explicit migration state for 3A/5A artifacts.  New 5C plans
    # persist the full decision so a render-only run never has to recompute it.
    boundary_decision: BoundaryDecision | None = None

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
        for segment in self.segments:
            if isinstance(segment, NarrationSegment) and segment.source_ranges:
                if not {item.transcript_segment_id for item in segment.source_ranges}.issubset(segment.source_segment_ids):
                    raise ValueError("narration source_ranges must reference source_segment_ids")
        has_narration = any(isinstance(segment, NarrationSegment) for segment in self.segments)
        if self.audio_mode in {"original", "original_enhanced"} and has_narration:
            raise ValueError("source audio modes cannot contain generated narration segments")
        if self.tts_eligible and (not has_narration or self.audio_mode not in {"voiceover", "replace_voice", "mixed"}):
            raise ValueError("TTS eligibility requires explicit voiceover intent and narration")
        if self.boundary_decision is not None:
            boundary_errors = _boundary_handoff_errors(self)
            if boundary_errors:
                raise ValueError("; ".join(boundary_errors))
        return self


@dataclass(frozen=True, slots=True)
class ProductionPlanHandoffFailure:
    """A deterministic precondition for the existing AudioCompositionService."""

    code: str
    evidence: dict[str, object]


def validate_audio_handoff(plan: ProductionPlan) -> ProductionPlanHandoffFailure | None:
    """Reject only adjacent, exactly equal dialogue source intervals.

    The plan remains untouched: a later contract evolution may intentionally
    model a shared source range, but this V2 foundation must never silently
    delete or merge either dialogue segment.
    """

    from app.errors import DUPLICATE_EXACT_SOURCE_RANGE

    dialogues = sorted(plan.dialogue_mappings, key=lambda item: item.order)
    for first, second in zip(dialogues, dialogues[1:]):
        first_range = (first.source_start_seconds, first.source_end_seconds)
        second_range = (second.source_start_seconds, second.source_end_seconds)
        if first_range == second_range:
            return ProductionPlanHandoffFailure(
                code=DUPLICATE_EXACT_SOURCE_RANGE,
                evidence={
                    "candidate_id": plan.metadata.candidate_id,
                    "segment_ids": [first.segment_id, second.segment_id],
                    "source_start": first.source_start_seconds,
                    "source_end": first.source_end_seconds,
                    "source_start_seconds": first.source_start_seconds,
                    "source_end_seconds": first.source_end_seconds,
                },
            )
    return None


def _boundary_handoff_errors(plan: ProductionPlan) -> list[str]:
    """Return deterministic post-edit boundary failures for a 5C plan.

    Dialogue ranges are the actual source material consumed by the existing
    audio/video stages, so they must remain inside the selected candidate
    boundary and retain its required hook/completion/payoff evidence.
    """

    decision = plan.boundary_decision
    assert decision is not None
    errors: list[str] = []
    if decision.candidate_id != plan.metadata.candidate_id:
        errors.append("BOUNDARY_CANDIDATE_MISMATCH")
    if not decision.word_integrity:
        errors.append("BOUNDARY_WORD_CUT")
    if not decision.sentence_integrity or not decision.semantic_completion:
        errors.append("BOUNDARY_INCOMPLETE_THOUGHT")
    if decision.continuation_risk > decision.continuation_risk_threshold + BOUNDARY_EPSILON_SECONDS:
        errors.append("BOUNDARY_CONTINUATION_RISK")
    if bool(decision.question_context.get("end_is_question")) and not bool(
        decision.question_context.get("answer_or_completion_included")
    ):
        errors.append("BOUNDARY_QUESTION_CONTEXT_MISSING")
    if not decision.payoff_preserved:
        errors.append("BOUNDARY_PAYOFF_MISSING")

    allowed = decision.allowed_source_range
    source_ranges = [
        BoundaryRange(start_seconds=item.source_start_seconds, end_seconds=item.source_end_seconds)
        for item in plan.dialogue_mappings
    ]
    for dialogue in plan.dialogue_mappings:
        if dialogue.boundary_decision_id != decision.decision_id:
            errors.append(f"BOUNDARY_DECISION_REFERENCE_MISMATCH:{dialogue.segment_id}")
        if (
            dialogue.source_start_seconds < allowed.start_seconds - BOUNDARY_EPSILON_SECONDS
            or dialogue.source_end_seconds > allowed.end_seconds + BOUNDARY_EPSILON_SECONDS
        ):
            errors.append(f"BOUNDARY_SOURCE_RANGE_OUTSIDE:{dialogue.segment_id}")
        if not _is_safe_point(dialogue.source_start_seconds, decision.safe_start_points):
            errors.append(f"BOUNDARY_WORD_CUT:{dialogue.segment_id}:start")
        if not _is_safe_point(dialogue.source_end_seconds, decision.safe_end_points):
            errors.append(f"BOUNDARY_WORD_CUT:{dialogue.segment_id}:end")

    for narration in (item for item in plan.segments if isinstance(item, NarrationSegment)):
        if narration.source_ranges and narration.boundary_decision_id != decision.decision_id:
            errors.append(f"BOUNDARY_DECISION_REFERENCE_MISMATCH:{narration.segment_id}")
        for source in narration.source_ranges:
            source_ranges.append(BoundaryRange(
                start_seconds=source.source_start_seconds,
                end_seconds=source.source_end_seconds,
            ))
            if (
                source.source_start_seconds < allowed.start_seconds - BOUNDARY_EPSILON_SECONDS
                or source.source_end_seconds > allowed.end_seconds + BOUNDARY_EPSILON_SECONDS
            ):
                errors.append(f"BOUNDARY_SOURCE_RANGE_OUTSIDE:{narration.segment_id}")
            if not _is_safe_point(source.source_start_seconds, decision.safe_start_points):
                errors.append(f"BOUNDARY_WORD_CUT:{narration.segment_id}:start")
            if not _is_safe_point(source.source_end_seconds, decision.safe_end_points):
                errors.append(f"BOUNDARY_WORD_CUT:{narration.segment_id}:end")

    for requirement in decision.required_evidence:
        if requirement.required and not _range_is_covered(requirement.source_range, source_ranges):
            errors.append(f"BOUNDARY_{requirement.requirement_type.upper()}_LOST")
    return errors


def _is_safe_point(timestamp: float, safe_points: list[float]) -> bool:
    return any(abs(timestamp - point) <= BOUNDARY_EPSILON_SECONDS for point in safe_points)


def _range_is_covered(required: BoundaryRange, ranges: list[BoundaryRange]) -> bool:
    """Check coverage by one or more touching dialogue ranges without gaps."""

    cursor = required.start_seconds
    for source_range in sorted(ranges, key=lambda item: (item.start_seconds, item.end_seconds)):
        if source_range.end_seconds < cursor - BOUNDARY_EPSILON_SECONDS:
            continue
        if source_range.start_seconds > cursor + BOUNDARY_EPSILON_SECONDS:
            break
        cursor = max(cursor, source_range.end_seconds)
        if cursor >= required.end_seconds - BOUNDARY_EPSILON_SECONDS:
            return True
    return False
