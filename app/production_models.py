from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.utils import stable_text_hash


BOUNDARY_EPSILON_SECONDS = 0.001
PRODUCTION_PLAN_ENVELOPE_VERSION = "5F.1"
SUPPORTED_LEGACY_PLAN_VERSIONS = frozenset({"3A.0", "3A.1", "3A.2"})
CONTINUITY_DECISION_VERSION = "A-2.continuity.1"


class ProductionPlanIdentity(BaseModel):
    """Immutable parents of a plan; paths are deliberately not identities."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class ProductionPlanPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset_id: str = Field(min_length=1)
    preset_version: str = Field(min_length=1)
    platform: Literal["tiktok", "reels", "shorts", "universal"]


class ProductionPlanTarget(BaseModel):
    """The stable output contract, not a replacement composition plan."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=2, le=7680)
    height: int = Field(ge=2, le=7680)
    fps: float = Field(gt=1, le=120)
    video_codec: Literal["h264"] = "h264"
    pixel_format: Literal["yuv420p"] = "yuv420p"

    @model_validator(mode="after")
    def _vertical_target_is_valid(self) -> "ProductionPlanTarget":
        if self.width % 2 or self.height % 2:
            raise ValueError("ProductionPlan target dimensions must be even")
        if abs((self.width / self.height) - (9 / 16)) > 0.002:
            raise ValueError("ProductionPlan target must use a 9:16 aspect ratio")
        return self


class ProductionPlanInputFingerprints(BaseModel):
    """Fingerprints for the immutable inputs a later render must not replace."""

    model_config = ConfigDict(extra="forbid")

    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    transcript_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    analysis_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    final_script_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    boundary_decision_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # A-2 keeps a separate cache identity for editorial continuity.  ``None``
    # is retained only while reading previously persisted 5F plans.
    continuity_decision_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ProductionPlanEnvelope(BaseModel):
    """Versioned V2 envelope around the existing ProductionPlan contract.

    ``legacy_adapter`` is intentionally explicit.  It keeps known cached 3A
    plans inspectable and renderable through the old typed path while making an
    unrecognised historical schema a visible hand-off failure.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["5F.1"] = PRODUCTION_PLAN_ENVELOPE_VERSION
    identity: ProductionPlanIdentity
    boundary_decision_ref: str | None = Field(default=None, min_length=1)
    continuity_decision_ref: str | None = Field(default=None, min_length=1)
    preset: ProductionPlanPreset
    target: ProductionPlanTarget
    input_fingerprints: ProductionPlanInputFingerprints
    created_at: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    compatibility_mode: Literal["native", "legacy_adapter"] = "native"
    legacy_source_version: str | None = None

    @model_validator(mode="after")
    def _legacy_marker_is_consistent(self) -> "ProductionPlanEnvelope":
        if self.compatibility_mode == "native" and self.legacy_source_version is not None:
            raise ValueError("native ProductionPlan envelope cannot carry a legacy source version")
        if self.compatibility_mode == "legacy_adapter" and self.legacy_source_version not in SUPPORTED_LEGACY_PLAN_VERSIONS:
            raise ValueError("UNSUPPORTED_LEGACY_PLAN_VERSION")
        return self


class ProductionPlanReference(BaseModel):
    """A compact immutable link used by Audio/Subtitle/Reframe/Video artifacts."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    schema_version: Literal["5F.1"] = PRODUCTION_PLAN_ENVELOPE_VERSION
    plan_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    identity: ProductionPlanIdentity


@dataclass(frozen=True, slots=True)
class ProductionPlanValidationFailure:
    """Machine-readable rejection returned before the existing renderer runs."""

    code: str
    evidence: dict[str, object]


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
    # Goal 6D: observed audio/visual payoff timing accompanies the semantic
    # text decision. Legacy 5C.1 artifacts load with an explicit empty state.
    multimodal_context: dict[str, Any] = Field(default_factory=dict)

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


class ContinuityRequiredSpan(BaseModel):
    """A non-dialogue source span that must remain in the approved boundary."""

    model_config = ConfigDict(extra="forbid")

    requirement_type: Literal["visual_action", "semantic_bridge", "reaction", "payoff"]
    source_range: BoundaryRange
    rationale: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(min_length=1)


class ContinuityOmittedSpan(BaseModel):
    """An intentional or unresolved omission inside the approved boundary."""

    model_config = ConfigDict(extra="forbid")

    source_range: BoundaryRange
    rationale_type: Literal[
        "silence", "dialogue_compaction", "editorially_redundant", "unexplained",
    ]
    rationale: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ContinuityDecision(BaseModel):
    """Versioned candidate-owned decision for non-dialogue continuity.

    The decision is deliberately distinct from ``BoundaryDecision``: the
    latter approves *where* production may source material, while this
    artifact records whether non-dialogue spans inside that boundary may be
    compacted, must be retained, or remain too weakly evidenced to publish.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["A-2.continuity.1"] = CONTINUITY_DECISION_VERSION
    decision_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    boundary_decision_id: str = Field(min_length=1)
    boundary_decision_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    approved_source_range: BoundaryRange
    mode: Literal["compact_dialogue", "preserve_required_spans", "uncertain"]
    required_spans: list[ContinuityRequiredSpan] = Field(default_factory=list)
    omitted_spans: list[ContinuityOmittedSpan] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spans_stay_within_approved_boundary(self) -> "ContinuityDecision":
        approved = self.approved_source_range
        for span in [*self.required_spans, *self.omitted_spans]:
            source_range = span.source_range
            if (
                source_range.start_seconds < approved.start_seconds - BOUNDARY_EPSILON_SECONDS
                or source_range.end_seconds > approved.end_seconds + BOUNDARY_EPSILON_SECONDS
            ):
                raise ValueError("continuity span must stay within approved source range")
        if self.mode == "compact_dialogue" and self.required_spans:
            raise ValueError("compact_dialogue cannot declare required continuity spans")
        if self.mode == "preserve_required_spans" and not self.required_spans:
            raise ValueError("preserve_required_spans requires evidence-backed spans")
        return self

    def fingerprint(self) -> str:
        return stable_text_hash(self.model_dump_json())


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


class DialogueEvidenceMapping(BaseModel):
    """Exact fact/ASR provenance retained independently from media cuts."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(min_length=1)
    transcript_segment_id: int = Field(ge=0)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    source_text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _timestamps_are_ordered(self) -> "DialogueEvidenceMapping":
        if self.source_end_seconds < self.source_start_seconds:
            raise ValueError("source_end_seconds must not precede source_start_seconds")
        return self


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
    # Physical source media may span natural pauses between exact ASR/fact
    # evidence.  These mappings preserve the original evidence geometry without
    # turning every evidence edge into an edit point.  Empty means a legacy plan.
    evidence_mappings: list[DialogueEvidenceMapping] = Field(default_factory=list)
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
    # ``schema_version`` and ``envelope`` turn the established ProductionPlan
    # into the V2 migration base.  They deliberately do not introduce another
    # EditPlan class or renderer.
    schema_version: str = PRODUCTION_PLAN_ENVELOPE_VERSION
    envelope: ProductionPlanEnvelope | None = None
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
    # A-2 is a separate candidate-owned decision.  It never mutates the
    # semantic BoundaryDecision and lets downstream stages distinguish a safe
    # dialogue compaction from mandatory visual/causal continuity.
    continuity_decision: ContinuityDecision | None = None
    # Evidence-bearing editorial target hints only; the existing composition
    # engine remains responsible for safe crop/tracking decisions.
    composition_intent: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_pre_audio_mode_plans(cls, value: object) -> object:
        """Adapt known 3A plans and retain their historical audio intent.

        Models in this repository use ``extra=forbid``, so loading historical
        JSON needs to happen before field validation.  The adapter never
        invents a boundary decision; it records its compatibility mode instead.
        """

        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "audio_mode" not in migrated:
            has_narration = any(
                isinstance(item, dict) and item.get("segment_type") == "narration"
                for item in migrated.get("segments", [])
            )
            migrated["audio_mode"] = "voiceover" if has_narration else "original"
            migrated["tts_eligible"] = has_narration
            migrated["audio_mode_reason"] = "legacy_plan_migration"
        envelope = migrated.get("envelope")
        mode = envelope.get("compatibility_mode") if isinstance(envelope, dict) else None
        if envelope is None or mode == "legacy_adapter":
            legacy_source_version = _legacy_plan_version(migrated)
            migrated["schema_version"] = PRODUCTION_PLAN_ENVELOPE_VERSION
            migrated["envelope"] = _legacy_envelope_payload(
                migrated,
                legacy_source_version=legacy_source_version,
            )
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
        envelope = self.envelope
        if self.schema_version != PRODUCTION_PLAN_ENVELOPE_VERSION or envelope is None:
            raise ValueError("EDIT_PLAN_SCHEMA_INVALID")
        if envelope.identity.candidate_id != self.metadata.candidate_id:
            raise ValueError("IDENTITY_MISMATCH: candidate_id")
        if envelope.identity.source_id != self.metadata.source_id:
            raise ValueError("IDENTITY_MISMATCH: source_id")
        if envelope.compatibility_mode == "native":
            if self.boundary_decision is None or not envelope.boundary_decision_ref:
                raise ValueError("EDIT_PLAN_SCHEMA_INVALID: boundary_decision_ref")
            if envelope.boundary_decision_ref != self.boundary_decision.decision_id:
                raise ValueError("IDENTITY_MISMATCH: boundary_decision_ref")
            if self.continuity_decision is None or not envelope.continuity_decision_ref:
                raise ValueError("EDIT_PLAN_SCHEMA_INVALID: continuity_decision_ref")
            if envelope.continuity_decision_ref != self.continuity_decision.decision_id:
                raise ValueError("IDENTITY_MISMATCH: continuity_decision_ref")
            if envelope.input_fingerprints.final_script_sha256 != self.metadata.final_script_hash:
                raise ValueError("EDIT_PLAN_SCHEMA_INVALID: final_script fingerprint")
        if self.continuity_decision is not None:
            continuity_errors = _continuity_handoff_errors(self)
            if continuity_errors:
                raise ValueError("; ".join(continuity_errors))
        if self.boundary_decision is not None:
            boundary_errors = _boundary_handoff_errors(self)
            if boundary_errors:
                raise ValueError("; ".join(boundary_errors))
        return self

    def plan_fingerprint(self) -> str:
        """Stable plan identity excluding its wall-clock creation timestamp.

        Rebuilding the same semantic plan in another run therefore produces the
        same fingerprint while ``created_at`` remains an audit fact rather than
        a cache-invalidating input.
        """

        value = self.model_dump(mode="json")
        envelope = value.get("envelope")
        if isinstance(envelope, dict):
            envelope = dict(envelope)
            envelope.pop("created_at", None)
            value["envelope"] = envelope
        return stable_text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def reference(self) -> ProductionPlanReference:
        assert self.envelope is not None
        return ProductionPlanReference(
            plan_id=self.plan_id,
            plan_fingerprint=self.plan_fingerprint(),
            identity=self.envelope.identity,
        )


@dataclass(frozen=True, slots=True)
class ProductionPlanHandoffFailure:
    """A deterministic precondition for the existing AudioCompositionService."""

    code: str
    evidence: dict[str, object]


def validate_renderer_handoff(
    plan: ProductionPlan,
    audio_project: Any,
    *,
    source_id: str,
    source_sha256: str,
    transcript: dict[str, Any],
    expected_project_id: str | None = None,
    expected_analysis_id: str | None = None,
    expected_plan_run_id: str | None = None,
    expected_candidate_id: str | None = None,
    expected_preset_id: str | None = None,
    expected_preset_version: str | None = None,
    expected_platform: str | None = None,
    expected_target: tuple[int, int, float] | None = None,
) -> ProductionPlanValidationFailure | None:
    """Validate the V2 envelope before the existing renderer touches media.

    This is intentionally a pure hand-off check.  It does not analyse source
    media, regenerate TTS/audio, or mutate the plan, which keeps render-only
    and visual rerender flows independent from expensive analysis.
    """

    envelope = plan.envelope
    if plan.schema_version != PRODUCTION_PLAN_ENVELOPE_VERSION or envelope is None:
        return ProductionPlanValidationFailure(
            "EDIT_PLAN_SCHEMA_INVALID", {"plan_id": plan.plan_id, "schema_version": plan.schema_version},
        )
    if envelope.compatibility_mode == "legacy_adapter" and envelope.legacy_source_version not in SUPPORTED_LEGACY_PLAN_VERSIONS:
        return ProductionPlanValidationFailure(
            "UNSUPPORTED_LEGACY_PLAN_VERSION",
            {"plan_id": plan.plan_id, "legacy_source_version": envelope.legacy_source_version},
        )
    identity = envelope.identity
    identity_pairs = {
        "candidate_id": (identity.candidate_id, plan.metadata.candidate_id),
        "source_id": (identity.source_id, plan.metadata.source_id),
        "project_id": (identity.project_id, expected_project_id),
        "analysis_id": (identity.analysis_id, expected_analysis_id),
        "run_id": (identity.run_id, expected_plan_run_id),
        "candidate_id_runtime": (identity.candidate_id, expected_candidate_id),
    }
    if envelope.compatibility_mode == "native":
        identity_pairs["source_id_runtime"] = (identity.source_id, source_id)
    for name, (actual, expected) in identity_pairs.items():
        if expected is not None and actual != expected:
            return ProductionPlanValidationFailure(
                "IDENTITY_MISMATCH",
                {"plan_id": plan.plan_id, "field": name, "expected": expected, "actual": actual},
            )
    if envelope.compatibility_mode == "native":
        if plan.boundary_decision is None or envelope.boundary_decision_ref != plan.boundary_decision.decision_id:
            return ProductionPlanValidationFailure(
                "IDENTITY_MISMATCH",
                {
                    "plan_id": plan.plan_id,
                    "field": "boundary_decision_ref",
                    "expected": plan.boundary_decision.decision_id if plan.boundary_decision else None,
                    "actual": envelope.boundary_decision_ref,
                },
            )
        continuity = plan.continuity_decision
        if continuity is None or envelope.continuity_decision_ref != continuity.decision_id:
            return ProductionPlanValidationFailure(
                "CONTINUITY_DECISION_MISSING",
                {
                    "plan_id": plan.plan_id,
                    "expected": continuity.decision_id if continuity else None,
                    "actual": envelope.continuity_decision_ref,
                },
            )
        if continuity.boundary_decision_id != plan.boundary_decision.decision_id:
            return ProductionPlanValidationFailure(
                "IDENTITY_MISMATCH",
                {
                    "plan_id": plan.plan_id,
                    "field": "continuity_decision.boundary_decision_id",
                    "expected": plan.boundary_decision.decision_id,
                    "actual": continuity.boundary_decision_id,
                },
            )
        boundary_sha256 = stable_text_hash(plan.boundary_decision.model_dump_json())
        if envelope.input_fingerprints.boundary_decision_sha256 != boundary_sha256:
            return ProductionPlanValidationFailure(
                "STALE_INPUTS",
                {
                    "plan_id": plan.plan_id,
                    "input": "boundary_decision",
                    "expected": envelope.input_fingerprints.boundary_decision_sha256,
                    "actual": boundary_sha256,
                },
            )
        continuity_sha256 = continuity.fingerprint()
        if envelope.input_fingerprints.continuity_decision_sha256 != continuity_sha256:
            return ProductionPlanValidationFailure(
                "STALE_INPUTS",
                {
                    "plan_id": plan.plan_id,
                    "input": "continuity_decision",
                    "expected": envelope.input_fingerprints.continuity_decision_sha256,
                    "actual": continuity_sha256,
                },
            )
        if envelope.input_fingerprints.source_sha256 != source_sha256:
            return ProductionPlanValidationFailure(
                "SOURCE_FINGERPRINT_MISMATCH",
                {
                    "plan_id": plan.plan_id,
                    "expected": envelope.input_fingerprints.source_sha256,
                    "actual": source_sha256,
                },
            )
        transcript_sha256 = _json_fingerprint(transcript)
        if envelope.input_fingerprints.transcript_sha256 != transcript_sha256:
            return ProductionPlanValidationFailure(
                "STALE_INPUTS",
                {
                    "plan_id": plan.plan_id,
                    "input": "transcript",
                    "expected": envelope.input_fingerprints.transcript_sha256,
                    "actual": transcript_sha256,
                },
            )
        preset_pairs = {
            "preset_id": (envelope.preset.preset_id, expected_preset_id),
            "preset_version": (envelope.preset.preset_version, expected_preset_version),
            "platform": (envelope.preset.platform, expected_platform),
        }
        for name, (actual, expected) in preset_pairs.items():
            if expected is not None and actual != expected:
                return ProductionPlanValidationFailure(
                    "PRESET_CONSTRAINT_VIOLATION",
                    {"plan_id": plan.plan_id, "field": name, "expected": expected, "actual": actual},
                )
        if expected_target is not None:
            target = (envelope.target.width, envelope.target.height, envelope.target.fps)
            if target != expected_target:
                return ProductionPlanValidationFailure(
                    "OUTPUT_CONTRACT_INVALID",
                    {"plan_id": plan.plan_id, "expected": expected_target, "actual": target},
                )
    audio_metadata = getattr(audio_project, "metadata", None)
    audio_reference = getattr(audio_metadata, "plan_reference", None)
    if envelope.compatibility_mode == "native" and audio_reference is None:
        return ProductionPlanValidationFailure(
            "EDIT_PLAN_SCHEMA_INVALID", {"plan_id": plan.plan_id, "missing": "AudioProject.metadata.plan_reference"},
        )
    if audio_reference is not None:
        expected_reference = plan.reference()
        if (
            audio_reference.plan_id != expected_reference.plan_id
            or audio_reference.plan_fingerprint != expected_reference.plan_fingerprint
            or audio_reference.identity != expected_reference.identity
        ):
            return ProductionPlanValidationFailure(
                "IDENTITY_MISMATCH",
                {
                    "plan_id": plan.plan_id,
                    "field": "audio_project.plan_reference",
                    "audio_project_id": getattr(audio_project, "project_id", None),
                },
            )
    if getattr(audio_metadata, "production_plan_id", None) != plan.plan_id:
        return ProductionPlanValidationFailure(
            "IDENTITY_MISMATCH",
            {
                "plan_id": plan.plan_id,
                "field": "audio_project.production_plan_id",
                "actual": getattr(audio_metadata, "production_plan_id", None),
            },
        )
    if getattr(audio_metadata, "source_id", None) != source_id:
        return ProductionPlanValidationFailure(
            "IDENTITY_MISMATCH",
            {
                "plan_id": plan.plan_id,
                "field": "audio_project.source_id",
                "expected": source_id,
                "actual": getattr(audio_metadata, "source_id", None),
            },
        )
    return None


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


def _continuity_handoff_errors(plan: ProductionPlan) -> list[str]:
    """Validate the immutable A-2 decision against its production consumer."""

    decision = plan.continuity_decision
    assert decision is not None
    errors: list[str] = []
    if decision.candidate_id != plan.metadata.candidate_id:
        errors.append("CONTINUITY_CANDIDATE_MISMATCH")
    boundary = plan.boundary_decision
    if boundary is None:
        errors.append("CONTINUITY_BOUNDARY_MISSING")
        return errors
    if decision.boundary_decision_id != boundary.decision_id:
        errors.append("CONTINUITY_BOUNDARY_REFERENCE_MISMATCH")
    if decision.boundary_decision_sha256 != stable_text_hash(boundary.model_dump_json()):
        errors.append("CONTINUITY_BOUNDARY_FINGERPRINT_MISMATCH")
    source_ranges = [
        BoundaryRange(start_seconds=item.source_start_seconds, end_seconds=item.source_end_seconds)
        for item in plan.dialogue_mappings
    ]
    if plan.envelope and plan.envelope.compatibility_mode == "native" and plan.audio_mode in {
        "original", "original_enhanced",
    }:
        if not any(item.evidence_mappings for item in plan.dialogue_mappings):
            errors.append("DIALOGUE_EVIDENCE_MAPPING_MISSING")
        explained = [
            item.source_range for item in decision.omitted_spans
            if item.rationale_type != "unexplained"
        ]
        approved = decision.approved_source_range
        cursor = approved.start_seconds
        for source_range in sorted(source_ranges, key=lambda item: (item.start_seconds, item.end_seconds)):
            if source_range.end_seconds <= approved.start_seconds + BOUNDARY_EPSILON_SECONDS:
                continue
            if source_range.start_seconds >= approved.end_seconds - BOUNDARY_EPSILON_SECONDS:
                break
            gap_end = min(source_range.start_seconds, approved.end_seconds)
            if gap_end > cursor + BOUNDARY_EPSILON_SECONDS and not _range_is_covered(
                BoundaryRange(start_seconds=cursor, end_seconds=gap_end), explained,
            ):
                errors.append("CONTINUITY_UNEXPLAINED_MEDIA_CUT")
            cursor = max(cursor, min(source_range.end_seconds, approved.end_seconds))
        if cursor < approved.end_seconds - BOUNDARY_EPSILON_SECONDS and not _range_is_covered(
            BoundaryRange(start_seconds=cursor, end_seconds=approved.end_seconds), explained,
        ):
            errors.append("CONTINUITY_UNEXPLAINED_MEDIA_CUT")
    for requirement in decision.required_spans:
        if not _range_is_covered(requirement.source_range, source_ranges):
            errors.append(f"CONTINUITY_{requirement.requirement_type.upper()}_LOST")
    return errors


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
        if not _is_safe_point(
            dialogue.source_start_seconds, decision.safe_start_points, plan.continuity_decision,
        ):
            errors.append(f"BOUNDARY_WORD_CUT:{dialogue.segment_id}:start")
        if not _is_safe_point(
            dialogue.source_end_seconds, decision.safe_end_points, plan.continuity_decision,
        ):
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
            if not _is_safe_point(
                source.source_start_seconds, decision.safe_start_points, plan.continuity_decision,
            ):
                errors.append(f"BOUNDARY_WORD_CUT:{narration.segment_id}:start")
            if not _is_safe_point(
                source.source_end_seconds, decision.safe_end_points, plan.continuity_decision,
            ):
                errors.append(f"BOUNDARY_WORD_CUT:{narration.segment_id}:end")

    for requirement in decision.required_evidence:
        if requirement.required and not _range_is_covered(requirement.source_range, source_ranges):
            errors.append(f"BOUNDARY_{requirement.requirement_type.upper()}_LOST")
    return errors


def _is_safe_point(
    timestamp: float,
    safe_points: list[float],
    continuity: ContinuityDecision | None = None,
) -> bool:
    if any(abs(timestamp - point) <= BOUNDARY_EPSILON_SECONDS for point in safe_points):
        return True
    # A required continuity span is a distinct evidence-backed edit decision.
    # Its endpoints are valid even when they are not transcript word edges;
    # this keeps visual action/reaction preservation from mutating 5C data.
    if continuity is None:
        return False
    if any(
        abs(timestamp - point) <= BOUNDARY_EPSILON_SECONDS
        for span in continuity.required_spans
        for point in (span.source_range.start_seconds, span.source_range.end_seconds)
    ):
        return True
    # A persisted typed omission is an explicit edit decision.  Unexplained
    # gaps never authorize a physical cut.
    return any(
        span.rationale_type != "unexplained"
        and abs(timestamp - point) <= BOUNDARY_EPSILON_SECONDS
        for span in continuity.omitted_spans
        for point in (span.source_range.start_seconds, span.source_range.end_seconds)
    )


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


def _legacy_envelope_payload(
    value: dict[str, Any], *, legacy_source_version: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic compatibility envelope for known 3A JSON.

    The data is intentionally marked as an adapter output.  Native plans are
    created by ``build_production_plan`` with real project/run/analysis and
    source fingerprints; this path is solely for persisted historical drafts
    and test fixtures that still use the 3A contract.
    """

    metadata_value = value.get("metadata")
    if isinstance(metadata_value, dict):
        metadata = metadata_value
    elif isinstance(metadata_value, BaseModel):
        metadata = metadata_value.model_dump(mode="json")
    else:
        metadata = {}
    candidate_id = str(metadata.get("candidate_id") or "legacy-candidate")
    source_id = str(metadata.get("source_id") or "legacy-source")
    plan_id = str(value.get("plan_id") or "legacy-plan")
    source_version = legacy_source_version or str(metadata.get("plan_version") or value.get("schema_version") or "")
    decision = value.get("boundary_decision") if isinstance(value.get("boundary_decision"), dict) else {}
    decision_id = str(decision.get("decision_id") or "") or None
    final_script_hash = str(metadata.get("final_script_hash") or "")
    if len(final_script_hash) != 64:
        final_script_hash = stable_text_hash(final_script_hash or plan_id)
    digest = lambda label: stable_text_hash(f"{label}:{plan_id}:{candidate_id}:{source_id}")
    return {
        "schema_version": PRODUCTION_PLAN_ENVELOPE_VERSION,
        "identity": {
            "project_id": f"legacy-project-{stable_text_hash(source_id)[:12]}",
            "run_id": f"legacy-run-{stable_text_hash(plan_id)[:12]}",
            "analysis_id": f"legacy-analysis-{stable_text_hash(source_id)[:12]}",
            "candidate_id": candidate_id,
            "source_id": source_id,
        },
        "boundary_decision_ref": decision_id,
        "preset": {"preset_id": "legacy", "preset_version": "3A", "platform": "universal"},
        "target": {"width": 1080, "height": 1920, "fps": 30.0, "video_codec": "h264", "pixel_format": "yuv420p"},
        "input_fingerprints": {
            "source_sha256": digest("source"),
            "transcript_sha256": digest("transcript"),
            "analysis_sha256": digest("analysis"),
            "final_script_sha256": final_script_hash,
            "boundary_decision_sha256": stable_text_hash(decision_id or "legacy-boundary-unavailable"),
        },
        "created_at": "1970-01-01T00:00:00Z",
        "warnings": ["LEGACY_PLAN_ADAPTER"],
        "compatibility_mode": "legacy_adapter",
        "legacy_source_version": source_version,
    }


def _json_fingerprint(value: dict[str, Any]) -> str:
    # Match the pipeline's established ``_hash`` serialization so a native
    # plan does not appear stale merely because the hand-off uses it later.
    return stable_text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _legacy_plan_version(value: dict[str, Any]) -> str:
    raw_schema_version = str(value.get("schema_version") or "")
    # A historical 3A record had no top-level envelope version.  If one is
    # present and is not the value injected by this migration reader, it is the
    # authoritative claim and must not be silently downgraded via metadata.
    if raw_schema_version and raw_schema_version != PRODUCTION_PLAN_ENVELOPE_VERSION:
        return raw_schema_version
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("plan_version") or raw_schema_version)
    if isinstance(metadata, BaseModel):
        return str(getattr(metadata, "plan_version", "") or raw_schema_version)
    return raw_schema_version
