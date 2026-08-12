from __future__ import annotations

"""Phase 7A creative contracts.

The module is deliberately a compiler boundary, not a renderer.  Untrusted
creative proposals are reduced to evidence-backed intent, and only immutable,
bounded plans can cross the future renderer hand-off.  The existing
``ProductionPlan`` remains the production lifecycle root.
"""

from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.production_models import ContinuityDecision, ProductionPlan, ProductionPlanReference
from app.video_models import ReframePlan, SubtitleProject, VideoTimeline


CREATIVE_PROPOSAL_SCHEMA_VERSION = "7A.proposal.1"
CREATIVE_INTENT_SCHEMA_VERSION = "7A.intent.1"
TIME_MAPPING_SCHEMA_VERSION = "7A.time-map.1"
CAPTION_PLAN_SCHEMA_VERSION = "7C.caption-plan.1"
COMPOSITION_PLAN_SCHEMA_VERSION: Literal["7D.composition-plan.1"] = "7D.composition-plan.1"
MOTION_PLAN_SCHEMA_VERSION: Literal["7F.motion-plan.1"] = "7F.motion-plan.1"
SOURCE_BROLL_PLAN_SCHEMA_VERSION: Literal["7E.source-broll-plan.1"] = "7E.source-broll-plan.1"
COMPILED_RENDER_PLAN_SCHEMA_VERSION = "7G.compiled-render-plan.1"
PARITY_SIGNATURE_SCHEMA_VERSION = "7G.parity.1"
RENDER_PROFILE_SCHEMA_VERSION = "7G.render-profile.1"
PARITY_MANIFEST_SCHEMA_VERSION = "7G.parity-manifest.1"

SOURCE_TICKS_PER_SECOND = 1_000_000
OUTPUT_FPS = 30
HASH_PATTERN = r"^[a-f0-9]{64}$"
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$"

T = TypeVar("T")


def _present(values: Iterable[T | None]) -> tuple[T, ...]:
    return tuple(value for value in values if value is not None)


class FrozenContract(BaseModel):
    """Strict, assignment-frozen base for durable contract objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def canonical_hash(self) -> str:
        return canonical_hash(self)


class ImmutableProductionIdentity(FrozenContract):
    project_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class ImmutableProductionPlanLink(FrozenContract):
    """Deeply immutable snapshot of the established ProductionPlan reference."""

    plan_id: str = Field(min_length=1)
    schema_version: Literal["5F.1"] = "5F.1"
    plan_fingerprint: str = Field(pattern=HASH_PATTERN)
    identity: ImmutableProductionIdentity

    @classmethod
    def from_reference(cls, reference: ProductionPlanReference) -> "ImmutableProductionPlanLink":
        return cls.model_validate(reference.model_dump(mode="json"))


def canonical_json(value: Any) -> str:
    """Return the sole canonical JSON representation used by Phase 7A.

    Pydantic JSON mode normalises enums and tuples first.  NaN/Infinity and
    arbitrary Python objects are rejected rather than stringified, keeping
    hashes replayable across processes and machines.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def seconds_to_source_tick(value: float) -> int:
    """Convert seconds to the explicit source time base without float drift."""

    if not math.isfinite(value) or value < 0:
        raise ValueError("source timestamp must be finite and non-negative")
    return int(round(value * SOURCE_TICKS_PER_SECOND))


def seconds_to_output_frame(value: float, *, end: bool = False) -> int:
    """Quantise seconds to the fixed 30 fps output base.

    Starts use floor and exclusive ends use ceil so a positive interval never
    disappears during conversion.
    """

    if not math.isfinite(value) or value < 0:
        raise ValueError("output timestamp must be finite and non-negative")
    frames = value * OUTPUT_FPS
    return math.ceil(frames - 1e-9) if end else math.floor(frames + 1e-9)


class SourceInterval(FrozenContract):
    start_tick: int = Field(ge=0)
    end_tick: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "SourceInterval":
        if self.end_tick <= self.start_tick:
            raise ValueError("source interval must be start-inclusive/end-exclusive and positive")
        return self

    @classmethod
    def from_seconds(cls, start: float, end: float) -> "SourceInterval":
        return cls(start_tick=seconds_to_source_tick(start), end_tick=seconds_to_source_tick(end))

    def contains(self, other: "SourceInterval") -> bool:
        return self.start_tick <= other.start_tick and other.end_tick <= self.end_tick

    def overlaps(self, other: "SourceInterval") -> bool:
        return self.start_tick < other.end_tick and other.start_tick < self.end_tick


class OutputInterval(FrozenContract):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "OutputInterval":
        if self.end_frame <= self.start_frame:
            raise ValueError("output interval must be start-inclusive/end-exclusive and positive")
        return self

    def contains(self, other: "OutputInterval") -> bool:
        return self.start_frame <= other.start_frame and other.end_frame <= self.end_frame

    def overlaps(self, other: "OutputInterval") -> bool:
        return self.start_frame < other.end_frame and other.start_frame < self.end_frame

    @classmethod
    def from_seconds(cls, start: float, end: float) -> "OutputInterval":
        start_frame = seconds_to_output_frame(start)
        end_frame = seconds_to_output_frame(end, end=True)
        return cls(start_frame=start_frame, end_frame=max(start_frame + 1, end_frame))


class EditMapSegment(FrozenContract):
    map_id: str = Field(pattern=ID_PATTERN)
    source: SourceInterval
    output: OutputInterval


class SourceOutputTimeMap(FrozenContract):
    """Explicit source-to-output mapping for cuts and reorder.

    Destination frames may not overlap.  A repeated source range is representable
    but source lookup then becomes deliberately ambiguous and is rejected by
    ``map_interval`` unless a unique segment contains the requested interval.
    """

    schema_version: Literal["7A.time-map.1"] = "7A.time-map.1"
    source_ticks_per_second: Literal[1_000_000] = 1_000_000
    output_fps: Literal[30] = 30
    segments: tuple[EditMapSegment, ...] = Field(min_length=1)
    # A-2 binds the actual source/output map to the candidate-owned continuity
    # decision, so render cache and final QC cannot treat it as an untracked
    # dialogue-only timeline.
    continuity_decision_id: str | None = Field(default=None, pattern=ID_PATTERN)
    continuity_decision_version: str | None = None
    continuity_decision_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _unambiguous_destination(self) -> "SourceOutputTimeMap":
        ids = [item.map_id for item in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("edit mapping segment ids must be unique")
        ordered = sorted(self.segments, key=lambda item: (item.output.start_frame, item.output.end_frame))
        if tuple(ordered) != self.segments:
            raise ValueError("edit mapping segments must be ordered by output frame")
        if any(right.output.start_frame < left.output.end_frame for left, right in zip(ordered, ordered[1:])):
            raise ValueError("edit mapping cannot assign two sources to the same destination frame")
        continuity_values = (
            self.continuity_decision_id,
            self.continuity_decision_version,
            self.continuity_decision_sha256,
        )
        if any(value is not None for value in continuity_values) and any(value is None for value in continuity_values):
            raise ValueError("edit mapping continuity identity must be complete")
        return self

    @property
    def fingerprint(self) -> str:
        return self.canonical_hash()

    def map_interval(self, source: SourceInterval) -> OutputInterval | None:
        matches = [item for item in self.segments if item.source.contains(source)]
        if len(matches) != 1:
            return None
        segment = matches[0]
        source_duration = segment.source.end_tick - segment.source.start_tick
        output_duration = segment.output.end_frame - segment.output.start_frame
        start_offset = source.start_tick - segment.source.start_tick
        end_offset = source.end_tick - segment.source.start_tick
        start_frame = segment.output.start_frame + (start_offset * output_duration) // source_duration
        end_frame = segment.output.start_frame + (
            end_offset * output_duration + source_duration - 1
        ) // source_duration
        return OutputInterval(start_frame=start_frame, end_frame=max(start_frame + 1, end_frame))


def source_output_map_from_legacy_timeline(
    timeline: VideoTimeline,
    continuity_decision: ContinuityDecision | None = None,
) -> SourceOutputTimeMap:
    """Adapt the current VideoTimeline's persisted clip decisions to 30 fps."""

    segments: list[EditMapSegment] = []
    destination_cursor = 0
    for clip in timeline.clips:
        # ``OutputInterval.from_seconds`` deliberately widens an isolated
        # interval (floor start / ceil end).  Applying that conversion to both
        # sides of a non-frame-aligned cut assigns the boundary frame twice.
        # Quantise the complete timeline as one ordered partition instead, and
        # reserve frames for source-less clips before omitting them from the map.
        output_start = max(
            destination_cursor,
            seconds_to_output_frame(clip.timeline_start_seconds),
        )
        output_end = max(
            output_start + 1,
            seconds_to_output_frame(clip.timeline_end_seconds, end=True),
        )
        destination_cursor = output_end
        if clip.source_start_seconds is None or clip.source_end_seconds is None:
            continue
        if clip.source_end_seconds <= clip.source_start_seconds:
            continue
        segments.append(EditMapSegment(
            map_id=f"legacy-{clip.clip_id}",
            source=SourceInterval.from_seconds(
                clip.source_start_seconds,
                clip.source_end_seconds,
            ),
            output=OutputInterval(start_frame=output_start, end_frame=output_end),
        ))
    if not segments:
        raise ValueError("LEGACY_TIMELINE_HAS_NO_SOURCE_MAPPING")
    return SourceOutputTimeMap(
        segments=tuple(segments),
        continuity_decision_id=(continuity_decision.decision_id if continuity_decision else None),
        continuity_decision_version=(continuity_decision.schema_version if continuity_decision else None),
        continuity_decision_sha256=(continuity_decision.fingerprint() if continuity_decision else None),
    )


class BeatRole(StrEnum):
    HOOK = "hook"
    SETUP = "setup"
    CLAIM = "claim"
    ACTION = "action"
    REACTION = "reaction"
    PAYOFF = "payoff"


class AttentionTarget(StrEnum):
    SPEAKER = "speaker"
    SUBJECT = "subject"
    OBJECT = "object"
    PRODUCT = "product"
    SCREEN = "screen"
    REACTION = "reaction"
    GROUP = "group"
    STABLE_SOURCE = "stable_source"


class LayoutFamily(StrEnum):
    SINGLE_SUBJECT = "single_subject"
    STABLE_SPEAKER = "stable_speaker"
    WIDE_GROUP = "wide_group"
    FIT_BACKGROUND = "fit_background"
    SPLIT = "split"
    STACKED = "stacked"
    SCREEN_PRIORITY = "screen_priority"
    SCREEN_PRODUCT = "screen_product"
    LEGACY_PASSTHROUGH = "legacy_passthrough"


class MotionPurpose(StrEnum):
    HOOK = "hook"
    CLAIM_CHANGE = "claim_change"
    EVIDENCE_REVEAL = "evidence_reveal"
    REACTION = "reaction"
    PAYOFF = "payoff"


class MotionDomain(StrEnum):
    CAPTION = "caption"
    COMPOSITION = "composition"
    TRANSITION = "transition"


class Intensity(StrEnum):
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"


class SemanticClass(StrEnum):
    CLAIM = "claim"
    NUMBER = "number"
    PRODUCT = "product"
    ACTION = "action"
    CONTRAST = "contrast"
    PAYOFF = "payoff"


class SourceBRollSemanticKind(StrEnum):
    ACTION = "action"
    OBJECT = "object"
    PRODUCT = "product"
    SCREEN = "screen"
    REACTION = "reaction"
    CONTEXT = "context"


class EvidenceItem(FrozenContract):
    evidence_ref: str = Field(pattern=ID_PATTERN)
    evidence_kind: Literal[
        "transcript", "audio", "visual", "scene", "story_unit", "boundary", "user_override"
    ]
    source: SourceInterval
    confidence: float = Field(ge=0, le=1)
    artifact_fingerprint: str = Field(pattern=HASH_PATTERN)
    provenance: str = Field(min_length=1, max_length=240)


class EvidenceBundle(FrozenContract):
    production_plan: ImmutableProductionPlanLink
    source_range: SourceInterval
    candidate_source_range: SourceInterval
    items: tuple[EvidenceItem, ...] = ()

    @model_validator(mode="after")
    def _unique_and_bounded(self) -> "EvidenceBundle":
        refs = [item.evidence_ref for item in self.items]
        if len(refs) != len(set(refs)):
            raise ValueError("evidence refs must be unique")
        if not self.source_range.contains(self.candidate_source_range):
            raise ValueError("candidate range must stay inside the source range")
        if any(not self.source_range.contains(item.source) for item in self.items):
            raise ValueError("evidence must stay inside the source range")
        return self

    @property
    def fingerprint(self) -> str:
        return self.canonical_hash()


class ProposalDecision(FrozenContract):
    decision_id: str = Field(pattern=ID_PATTERN)
    source: SourceInterval
    confidence: float = Field(ge=0, le=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_evidence(self) -> "ProposalDecision":
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("proposal evidence refs must be unique")
        return self


class BeatProposal(ProposalDecision):
    role: BeatRole
    importance: float = Field(ge=0, le=1)
    payoff_must_not_precede: SourceInterval | None = None


class EmphasisProposal(ProposalDecision):
    text_span: str = Field(min_length=1, max_length=240)
    semantic_class: SemanticClass
    importance: float = Field(ge=0, le=1)


class CompositionTargetProposal(ProposalDecision):
    target: AttentionTarget
    target_ref: str | None = Field(default=None, pattern=ID_PATTERN)
    priority: int = Field(ge=1, le=100)
    allowed_layouts: tuple[LayoutFamily, ...] = Field(min_length=1)


class MotionEventProposal(ProposalDecision):
    purpose: MotionPurpose
    domain: MotionDomain
    intensity: Intensity


class SourceBRollProposal(ProposalDecision):
    source_cutaway: SourceInterval
    source_cutaway_evidence_refs: tuple[str, ...] = Field(min_length=1)
    story_unit_id: str = Field(pattern=ID_PATTERN)
    story_unit_evidence_ref: str = Field(pattern=ID_PATTERN)
    semantic_kind: SourceBRollSemanticKind = SourceBRollSemanticKind.CONTEXT
    retain_source_audio: Literal[False] = False

    @model_validator(mode="after")
    def _unique_cutaway_evidence(self) -> "SourceBRollProposal":
        if len(self.source_cutaway_evidence_refs) != len(set(self.source_cutaway_evidence_refs)):
            raise ValueError("source cutaway evidence refs must be unique")
        if self.story_unit_evidence_ref not in self.evidence_refs:
            raise ValueError("source B-roll must link its StoryUnit evidence ref")
        return self


class CreativeProposal(FrozenContract):
    """Bounded, untrusted Brain output.  It is never a renderer input."""

    schema_version: Literal["7A.proposal.1"] = "7A.proposal.1"
    proposal_id: str = Field(pattern=ID_PATTERN)
    production_plan: ImmutableProductionPlanLink
    revision: int = Field(ge=1)
    beats: tuple[BeatProposal, ...] = ()
    emphasis: tuple[EmphasisProposal, ...] = ()
    composition: tuple[CompositionTargetProposal, ...] = ()
    motion: tuple[MotionEventProposal, ...] = ()
    source_broll: tuple[SourceBRollProposal, ...] = ()

    @model_validator(mode="after")
    def _unique_decisions(self) -> "CreativeProposal":
        decisions: list[ProposalDecision] = [
            *self.beats, *self.emphasis, *self.composition, *self.motion, *self.source_broll,
        ]
        ids = [item.decision_id for item in decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("creative proposal decision ids must be unique")
        return self


class CreativeProposalNormalizer:
    """Bind untrusted JSON to trusted ProductionPlan identity and strict enums."""

    def normalize(
        self,
        raw: Mapping[str, Any],
        production_plan: ProductionPlanReference | ImmutableProductionPlanLink,
    ) -> CreativeProposal:
        payload = dict(raw)
        claimed = payload.get("production_plan")
        trusted = production_plan.model_dump(mode="json")
        if claimed is not None and claimed != trusted:
            raise ValueError("IDENTITY_MISMATCH: raw CreativeProposal ProductionPlan")
        payload["production_plan"] = trusted
        return CreativeProposal.model_validate(payload)


class CreativeDiagnostic(FrozenContract):
    code: Literal[
        "MISSING_EVIDENCE",
        "EVIDENCE_OUTSIDE_DECISION",
        "DECISION_OUTSIDE_CANDIDATE",
        "UNMAPPED_SOURCE_INTERVAL",
        "AMBIGUOUS_SOURCE_INTERVAL",
        "CUTAWAY_OUTSIDE_SOURCE",
        "STORY_UNIT_EVIDENCE_MISSING",
        "SAFE_FALLBACK_APPLIED",
    ]
    severity: Literal["warning", "blocked"] = "warning"
    decision_id: str = Field(pattern=ID_PATTERN)
    evidence_refs: tuple[str, ...] = ()
    fallback: Literal["drop_emphasis", "stable_source", "static_state", "a_roll", "drop_beat"]


class ResolvedDecision(FrozenContract):
    decision_id: str = Field(pattern=ID_PATTERN)
    source: SourceInterval
    output: OutputInterval
    confidence: float = Field(ge=0, le=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class ResolvedBeat(ResolvedDecision):
    role: BeatRole
    importance: float = Field(ge=0, le=1)


class ResolvedEmphasis(ResolvedDecision):
    text_span: str
    semantic_class: SemanticClass
    importance: float = Field(ge=0, le=1)


class ResolvedCompositionTarget(ResolvedDecision):
    target: AttentionTarget
    target_ref: str | None = None
    priority: int
    allowed_layouts: tuple[LayoutFamily, ...]


class ResolvedMotionEvent(ResolvedDecision):
    purpose: MotionPurpose
    domain: MotionDomain
    intensity: Intensity


class ResolvedSourceBRoll(ResolvedDecision):
    source_cutaway: SourceInterval
    source_cutaway_evidence_refs: tuple[str, ...]
    story_unit_id: str
    story_unit_evidence_ref: str
    semantic_kind: SourceBRollSemanticKind = SourceBRollSemanticKind.CONTEXT
    retain_source_audio: Literal[False] = False


class EvidenceResolution(FrozenContract):
    proposal_hash: str = Field(pattern=HASH_PATTERN)
    evidence_fingerprint: str = Field(pattern=HASH_PATTERN)
    mapping_fingerprint: str = Field(pattern=HASH_PATTERN)
    evidence_items: tuple[EvidenceItem, ...] = ()
    beats: tuple[ResolvedBeat, ...] = ()
    emphasis: tuple[ResolvedEmphasis, ...] = ()
    composition: tuple[ResolvedCompositionTarget, ...] = ()
    motion: tuple[ResolvedMotionEvent, ...] = ()
    source_broll: tuple[ResolvedSourceBRoll, ...] = ()
    diagnostics: tuple[CreativeDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _accepted_decisions_have_evidence(self) -> "EvidenceResolution":
        catalog = {item.evidence_ref: item for item in self.evidence_items}
        decisions: tuple[ResolvedDecision, ...] = (
            *self.beats,
            *self.emphasis,
            *self.composition,
            *self.motion,
            *self.source_broll,
        )
        for decision in decisions:
            if any(ref not in catalog for ref in decision.evidence_refs):
                raise ValueError("resolved decision references missing evidence")
            if any(not catalog[ref].source.overlaps(decision.source) for ref in decision.evidence_refs):
                raise ValueError("resolved decision evidence does not overlap its source interval")
        for decision in self.source_broll:
            if any(ref not in catalog for ref in decision.source_cutaway_evidence_refs):
                raise ValueError("resolved source B-roll references missing cutaway evidence")
            if any(
                not catalog[ref].source.overlaps(decision.source_cutaway)
                for ref in decision.source_cutaway_evidence_refs
            ):
                raise ValueError("source B-roll evidence does not overlap its cutaway")
            story = catalog.get(decision.story_unit_evidence_ref)
            if story is None or story.evidence_kind != "story_unit":
                raise ValueError("resolved source B-roll requires StoryUnit evidence")
        return self


class EvidenceResolver:
    """Resolve bounded proposals against persisted evidence and edit mapping."""

    def resolve(
        self,
        proposal: CreativeProposal,
        evidence: EvidenceBundle,
        mapping: SourceOutputTimeMap,
    ) -> EvidenceResolution:
        if proposal.production_plan != evidence.production_plan:
            raise ValueError("IDENTITY_MISMATCH: CreativeProposal/EvidenceBundle ProductionPlan")

        catalog = {item.evidence_ref: item for item in evidence.items}
        diagnostics: list[CreativeDiagnostic] = []

        beats = _present(
            self._resolve_beat(decision, evidence, catalog, mapping, diagnostics)
            for decision in proposal.beats
        )
        emphasis = _present(
            self._resolve_emphasis(decision, evidence, catalog, mapping, diagnostics)
            for decision in proposal.emphasis
        )
        composition = _present(
            self._resolve_composition(decision, evidence, catalog, mapping, diagnostics)
            for decision in proposal.composition
        )
        motion = _present(
            self._resolve_motion(decision, evidence, catalog, mapping, diagnostics)
            for decision in proposal.motion
        )
        source_broll = _present(
            self._resolve_broll(decision, evidence, catalog, mapping, diagnostics)
            for decision in proposal.source_broll
        )
        return EvidenceResolution(
            proposal_hash=proposal.canonical_hash(),
            evidence_fingerprint=evidence.fingerprint,
            mapping_fingerprint=mapping.fingerprint,
            evidence_items=evidence.items,
            beats=beats,
            emphasis=emphasis,
            composition=composition,
            motion=motion,
            source_broll=source_broll,
            diagnostics=tuple(diagnostics),
        )

    def _resolved_base(
        self,
        decision: ProposalDecision,
        evidence: EvidenceBundle,
        catalog: Mapping[str, EvidenceItem],
        mapping: SourceOutputTimeMap,
        diagnostics: list[CreativeDiagnostic],
        fallback: Literal["drop_emphasis", "stable_source", "static_state", "a_roll", "drop_beat"],
        *,
        evidence_refs: Sequence[str] | None = None,
        evidence_interval: SourceInterval | None = None,
    ) -> dict[str, Any] | None:
        refs = tuple(evidence_refs or decision.evidence_refs)
        if not evidence.candidate_source_range.contains(decision.source):
            diagnostics.append(CreativeDiagnostic(
                code="DECISION_OUTSIDE_CANDIDATE", decision_id=decision.decision_id,
                evidence_refs=refs, fallback=fallback,
            ))
            return None
        missing = tuple(ref for ref in refs if ref not in catalog)
        if missing:
            diagnostics.append(CreativeDiagnostic(
                code="MISSING_EVIDENCE", decision_id=decision.decision_id,
                evidence_refs=missing, fallback=fallback,
            ))
            return None
        interval = evidence_interval or decision.source
        if any(not catalog[ref].source.overlaps(interval) for ref in refs):
            diagnostics.append(CreativeDiagnostic(
                code="EVIDENCE_OUTSIDE_DECISION", decision_id=decision.decision_id,
                evidence_refs=refs, fallback=fallback,
            ))
            return None
        output = mapping.map_interval(decision.source)
        if output is None:
            containing = sum(1 for segment in mapping.segments if segment.source.contains(decision.source))
            diagnostics.append(CreativeDiagnostic(
                code="AMBIGUOUS_SOURCE_INTERVAL" if containing > 1 else "UNMAPPED_SOURCE_INTERVAL",
                decision_id=decision.decision_id, evidence_refs=refs, fallback=fallback,
            ))
            return None
        return {
            "decision_id": decision.decision_id,
            "source": decision.source,
            "output": output,
            "confidence": decision.confidence,
            "evidence_refs": decision.evidence_refs,
        }

    def _resolve_beat(
        self,
        decision: BeatProposal,
        evidence: EvidenceBundle,
        catalog: Mapping[str, EvidenceItem],
        mapping: SourceOutputTimeMap,
        diagnostics: list[CreativeDiagnostic],
    ) -> ResolvedBeat | None:
        base = self._resolved_base(decision, evidence, catalog, mapping, diagnostics, "drop_beat")
        return None if base is None else ResolvedBeat(**base, role=decision.role, importance=decision.importance)

    def _resolve_emphasis(
        self,
        decision: EmphasisProposal,
        evidence: EvidenceBundle,
        catalog: Mapping[str, EvidenceItem],
        mapping: SourceOutputTimeMap,
        diagnostics: list[CreativeDiagnostic],
    ) -> ResolvedEmphasis | None:
        base = self._resolved_base(decision, evidence, catalog, mapping, diagnostics, "drop_emphasis")
        return None if base is None else ResolvedEmphasis(
            **base,
            text_span=decision.text_span,
            semantic_class=decision.semantic_class,
            importance=decision.importance,
        )

    def _resolve_composition(
        self,
        decision: CompositionTargetProposal,
        evidence: EvidenceBundle,
        catalog: Mapping[str, EvidenceItem],
        mapping: SourceOutputTimeMap,
        diagnostics: list[CreativeDiagnostic],
    ) -> ResolvedCompositionTarget | None:
        base = self._resolved_base(decision, evidence, catalog, mapping, diagnostics, "stable_source")
        return None if base is None else ResolvedCompositionTarget(
            **base,
            target=decision.target,
            target_ref=decision.target_ref,
            priority=decision.priority,
            allowed_layouts=decision.allowed_layouts,
        )

    def _resolve_motion(
        self,
        decision: MotionEventProposal,
        evidence: EvidenceBundle,
        catalog: Mapping[str, EvidenceItem],
        mapping: SourceOutputTimeMap,
        diagnostics: list[CreativeDiagnostic],
    ) -> ResolvedMotionEvent | None:
        base = self._resolved_base(decision, evidence, catalog, mapping, diagnostics, "static_state")
        return None if base is None else ResolvedMotionEvent(
            **base,
            purpose=decision.purpose,
            domain=decision.domain,
            intensity=decision.intensity,
        )

    def _resolve_broll(
        self,
        decision: SourceBRollProposal,
        evidence: EvidenceBundle,
        catalog: Mapping[str, EvidenceItem],
        mapping: SourceOutputTimeMap,
        diagnostics: list[CreativeDiagnostic],
    ) -> ResolvedSourceBRoll | None:
        base = self._resolved_base(decision, evidence, catalog, mapping, diagnostics, "a_roll")
        if base is None:
            return None
        story_evidence = catalog.get(decision.story_unit_evidence_ref)
        if story_evidence is None or story_evidence.evidence_kind != "story_unit":
            diagnostics.append(CreativeDiagnostic(
                code="STORY_UNIT_EVIDENCE_MISSING",
                decision_id=decision.decision_id,
                evidence_refs=(decision.story_unit_evidence_ref,),
                fallback="a_roll",
            ))
            return None
        if not evidence.source_range.contains(decision.source_cutaway):
            diagnostics.append(CreativeDiagnostic(
                code="CUTAWAY_OUTSIDE_SOURCE",
                decision_id=decision.decision_id,
                evidence_refs=decision.source_cutaway_evidence_refs,
                fallback="a_roll",
            ))
            return None
        cutaway_check = self._resolved_base(
            decision, evidence, catalog, mapping, diagnostics, "a_roll",
            evidence_refs=decision.source_cutaway_evidence_refs,
            evidence_interval=decision.source_cutaway,
        )
        if cutaway_check is None:
            return None
        return ResolvedSourceBRoll(
            **base,
            source_cutaway=decision.source_cutaway,
            source_cutaway_evidence_refs=decision.source_cutaway_evidence_refs,
            story_unit_id=decision.story_unit_id,
            story_unit_evidence_ref=decision.story_unit_evidence_ref,
            semantic_kind=decision.semantic_kind,
            retain_source_audio=False,
        )


class CreativePolicy(FrozenContract):
    preset_id: str = Field(min_length=1, max_length=160)
    preset_version: str = Field(min_length=1, max_length=80)
    platform: Literal["tiktok", "reels", "shorts", "universal"]
    caption_style_family: Literal["clean", "emphasis", "minimal", "editorial"] = "clean"
    caption_density: Literal["low", "balanced", "high"] = "balanced"
    intensity: Intensity = Intensity.BALANCED
    reduced_motion: bool = False
    source_broll_enabled: bool = False
    user_override_ids: tuple[str, ...] = ()


class CreativeIntent(FrozenContract):
    schema_version: Literal["7A.intent.1"] = "7A.intent.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    revision: int = Field(ge=1)
    production_plan: ImmutableProductionPlanLink
    source_output_mapping: SourceOutputTimeMap
    evidence_fingerprint: str = Field(pattern=HASH_PATTERN)
    evidence_manifest: tuple[EvidenceItem, ...] = ()
    proposal_hash: str = Field(pattern=HASH_PATTERN)
    policy: CreativePolicy
    confidence: float = Field(ge=0, le=1)
    provenance: tuple[str, ...] = ()
    beats: tuple[ResolvedBeat, ...] = ()
    semantic_emphasis: tuple[ResolvedEmphasis, ...] = ()
    composition_targets: tuple[ResolvedCompositionTarget, ...] = ()
    motion_events: tuple[ResolvedMotionEvent, ...] = ()
    source_broll: tuple[ResolvedSourceBRoll, ...] = ()
    forbidden_behaviours: tuple[Literal[
        "raw_commands", "unknown_primitives", "unsupported_broll", "premature_payoff",
        "random_motion", "unverified_fonts", "pixel_coordinates",
    ], ...] = (
        "raw_commands", "unknown_primitives", "unsupported_broll", "premature_payoff",
        "random_motion", "unverified_fonts", "pixel_coordinates",
    )
    ordered_fallbacks: tuple[Literal[
        "drop_emphasis", "stable_source", "static_state", "a_roll", "approved_font", "block",
    ], ...] = (
        "drop_emphasis", "stable_source", "static_state", "a_roll", "approved_font", "block",
    )
    diagnostics: tuple[CreativeDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _evidence_manifest_covers_intent(self) -> "CreativeIntent":
        refs = {item.evidence_ref for item in self.evidence_manifest}
        decisions: tuple[ResolvedDecision, ...] = (
            *self.beats,
            *self.semantic_emphasis,
            *self.composition_targets,
            *self.motion_events,
            *self.source_broll,
        )
        if any(not set(item.evidence_refs).issubset(refs) for item in decisions):
            raise ValueError("CreativeIntent decision is not covered by its evidence manifest")
        return self


class CreativeIntentCompiler:
    def compile(
        self,
        proposal: CreativeProposal,
        resolution: EvidenceResolution,
        mapping: SourceOutputTimeMap,
        policy: CreativePolicy,
    ) -> CreativeIntent:
        if resolution.proposal_hash != proposal.canonical_hash():
            raise ValueError("STALE_EVIDENCE_RESOLUTION: proposal hash")
        if resolution.mapping_fingerprint != mapping.fingerprint:
            raise ValueError("STALE_EVIDENCE_RESOLUTION: source/output mapping")
        broll = resolution.source_broll if policy.source_broll_enabled else ()
        motion = () if policy.reduced_motion else resolution.motion
        accepted: tuple[ResolvedDecision, ...] = (
            *resolution.beats,
            *resolution.emphasis,
            *resolution.composition,
            *motion,
            *broll,
        )
        confidence = (
            sum(item.confidence for item in accepted) / len(accepted)
            if accepted
            else 0.0
        )
        identity_payload = {
            "schema_version": CREATIVE_INTENT_SCHEMA_VERSION,
            "proposal_hash": resolution.proposal_hash,
            "evidence_fingerprint": resolution.evidence_fingerprint,
            "mapping_fingerprint": mapping.fingerprint,
            "policy": policy.model_dump(mode="json"),
            "revision": proposal.revision,
        }
        return CreativeIntent(
            intent_id=f"intent-{canonical_hash(identity_payload)[:24]}",
            revision=proposal.revision,
            production_plan=proposal.production_plan,
            source_output_mapping=mapping,
            evidence_fingerprint=resolution.evidence_fingerprint,
            evidence_manifest=resolution.evidence_items,
            proposal_hash=resolution.proposal_hash,
            policy=policy,
            confidence=confidence,
            provenance=tuple(sorted({item.provenance for item in resolution.evidence_items})),
            beats=resolution.beats,
            semantic_emphasis=resolution.emphasis,
            composition_targets=resolution.composition,
            motion_events=motion,
            source_broll=broll,
            diagnostics=resolution.diagnostics,
        )


def compile_creative_intent(
    proposal: CreativeProposal,
    evidence: EvidenceBundle,
    mapping: SourceOutputTimeMap,
    policy: CreativePolicy,
) -> CreativeIntent:
    resolution = EvidenceResolver().resolve(proposal, evidence, mapping)
    return CreativeIntentCompiler().compile(proposal, resolution, mapping, policy)


class NormalizedRect(FrozenContract):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside_canvas(self) -> "NormalizedRect":
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("normalized rectangle must stay inside the canvas")
        return self


class CaptionWordPlan(FrozenContract):
    """One display word with immutable output-frame timing."""

    word_id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1, max_length=400)
    output: OutputInterval
    timing_source: Literal["verified", "aligned", "phrase", "estimated"]
    confidence: float = Field(ge=0, le=1)


class CaptionEmphasisPlan(FrozenContract):
    """A single evidence-backed semantic treatment inside a cue."""

    emphasis_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    word_indexes: tuple[int, ...] = Field(min_length=1)
    semantic_class: SemanticClass
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    treatment: Literal["color", "phrase_color", "karaoke", "bounded_scale"]

    @model_validator(mode="after")
    def _ordered_indexes(self) -> "CaptionEmphasisPlan":
        if tuple(sorted(set(self.word_indexes))) != self.word_indexes:
            raise ValueError("caption emphasis word indexes must be unique and ordered")
        return self


class CaptionFontManifest(FrozenContract):
    """Deterministic font identity used by both Preview and Final."""

    font_id: str = Field(pattern=ID_PATTERN)
    requested_family: str = Field(min_length=1, max_length=160)
    resolved_family: str = Field(min_length=1, max_length=160)
    style: Literal["normal", "italic"] = "normal"
    weight: Literal["normal", "bold"] = "bold"
    file_name: str | None = Field(default=None, max_length=260)
    file_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    supported_scripts: tuple[Literal["latin", "cyrillic", "unknown"], ...] = ("unknown",)
    metrics_backend: Literal["gdi_file_metrics", "qt_file_metrics", "qt_family_metrics", "heuristic"] = "heuristic"
    shaping_backend: str = Field(default="libass-harfbuzz", min_length=1, max_length=120)
    fallback_chain: tuple[str, ...] = ()
    deployment_status: Literal["system", "bundled", "unverified"] = "unverified"
    fallback_used: bool = False

    @model_validator(mode="after")
    def _exact_identity_is_complete(self) -> "CaptionFontManifest":
        if (self.file_name is None) != (self.file_sha256 is None):
            raise ValueError("font file name and checksum must be provided together")
        if self.metrics_backend in {"gdi_file_metrics", "qt_file_metrics"} and self.file_sha256 is None:
            raise ValueError("file metrics require an exact font checksum")
        return self


class CaptionTypographyToken(FrozenContract):
    token_id: str = Field(pattern=ID_PATTERN)
    font_size_ratio: float = Field(gt=0, le=0.2)
    minimum_font_size_ratio: float = Field(gt=0, le=0.2)
    line_height: float = Field(default=1.22, ge=1, le=2)
    font_weight: Literal["normal", "bold"] = "bold"
    text_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    highlight_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_width_ratio: float = Field(ge=0, le=0.02)
    shadow_ratio: float = Field(ge=0, le=0.02)
    max_width_ratio: float = Field(gt=0.2, le=0.95)
    alignment: Literal["center", "left"] = "center"
    uppercase_emphasis: bool = False


class CaptionCollisionDecision(FrozenContract):
    lane: Literal["lower", "lower_mid", "upper_mid", "upper"]
    candidate_costs: tuple[tuple[str, float], ...] = ()
    overlap_ratio: float = Field(default=0, ge=0, le=1)
    protected_region_ids: tuple[str, ...] = ()
    safe_zone_valid: bool = True
    switched_lane: bool = False
    reason: Literal[
        "preferred_lane", "protected_region_avoidance", "platform_safe_zone",
        "stable_lane", "least_overlap_fallback",
    ] = "preferred_lane"


class CaptionQualityFinding(FrozenContract):
    code: Literal[
        "CAPTION_CPS_HIGH", "CAPTION_TIMING_WEAK", "CAPTION_FONT_FALLBACK",
        "CAPTION_METRICS_FALLBACK", "CAPTION_INTENSITY_DEGRADED",
        "CAPTION_SAFE_ZONE_VIOLATION", "CAPTION_PROTECTED_REGION_OVERLAP",
        "CAPTION_LANE_SWITCH_RATE_HIGH", "CAPTION_LINE_OVERFLOW",
        "CAPTION_READABILITY_FALLBACK",
    ]
    severity: Literal["warning", "blocker"]
    cue_id: str | None = Field(default=None, pattern=ID_PATTERN)
    measured_value: float | int | str | bool | None = None
    threshold: float | int | str | bool | None = None
    message: str = Field(min_length=1, max_length=1200)


class CaptionQualityMetrics(FrozenContract):
    cue_count: int = Field(default=0, ge=0)
    word_timed_cue_count: int = Field(default=0, ge=0)
    weak_timing_cue_count: int = Field(default=0, ge=0)
    semantic_emphasis_count: int = Field(default=0, ge=0)
    motion_cue_count: int = Field(default=0, ge=0)
    lane_switch_count: int = Field(default=0, ge=0)
    protected_overlap_count: int = Field(default=0, ge=0)
    safe_zone_violation_count: int = Field(default=0, ge=0)
    max_cps: float = Field(default=0, ge=0)
    font_exact: bool = False
    metrics_exact: bool = False


class CaptionQualityProvenance(FrozenContract):
    producer: str = Field(default="legacy_caption_contract", min_length=1, max_length=240)
    planner_version: str = Field(default="legacy", min_length=1, max_length=120)
    backend: str = Field(default="legacy", min_length=1, max_length=120)
    intent_id: str = Field(default="legacy", min_length=1, max_length=160)


class CaptionQualityReport(FrozenContract):
    schema_version: Literal["7C.caption-quality.1"] = "7C.caption-quality.1"
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED", "LEGACY_UNASSESSED"] = "LEGACY_UNASSESSED"
    findings: tuple[CaptionQualityFinding, ...] = ()
    metrics: CaptionQualityMetrics = Field(default_factory=CaptionQualityMetrics)
    provenance: CaptionQualityProvenance = Field(default_factory=CaptionQualityProvenance)

    @model_validator(mode="after")
    def _status_matches_findings(self) -> "CaptionQualityReport":
        has_blocker = any(item.severity == "blocker" for item in self.findings)
        has_warning = any(item.severity == "warning" for item in self.findings)
        expected = "BLOCKED" if has_blocker else "PASS_WITH_WARNINGS" if has_warning else "PASS"
        if self.status != "LEGACY_UNASSESSED" and self.status != expected:
            raise ValueError("caption quality status does not match findings")
        return self


class CaptionCuePlan(FrozenContract):
    cue_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    resolved_lines: tuple[str, ...] = Field(min_length=1, max_length=2)
    lane: Literal["lower", "lower_mid", "upper_mid", "upper"]
    typography_token_id: str = Field(pattern=ID_PATTERN)
    semantic_class: SemanticClass | None = None
    evidence_refs: tuple[str, ...] = ()
    primitive_id: Literal["legacy_passthrough", "static", "fade", "scale", "slide", "karaoke"] = "static"
    easing_id: Literal["none", "linear", "ease_in_out"] = "none"
    normalized_bounds: NormalizedRect | None = None
    words: tuple[CaptionWordPlan, ...] = ()
    timing_mode: Literal["word", "phrase", "static"] = "static"
    timing_confidence: float = Field(default=0, ge=0, le=1)
    emphasis: CaptionEmphasisPlan | None = None
    beat_role: BeatRole | None = None
    collision: CaptionCollisionDecision | None = None
    resolved_font_size_ratio: float | None = Field(default=None, gt=0, le=0.2)
    motion_duration_frames: int = Field(default=0, ge=0, le=30)
    scale_percent: int = Field(default=100, ge=94, le=108)
    slide_distance_ratio: float = Field(default=0, ge=0, le=0.05)
    fallback_reason: Literal[
        "weak_timing", "missing_font", "metrics_unavailable", "readability",
        "collision", "unsupported_primitive",
    ] | None = None

    @model_validator(mode="after")
    def _bounded_caption_cue(self) -> "CaptionCuePlan":
        if any(not self.output.contains(word.output) for word in self.words):
            raise ValueError("caption word timing must stay inside its cue")
        if self.emphasis is not None:
            if not self.output.contains(self.emphasis.output):
                raise ValueError("caption emphasis timing must stay inside its cue")
            if any(index >= len(self.words) for index in self.emphasis.word_indexes):
                raise ValueError("caption emphasis index is outside cue words")
        if self.timing_mode != "word" and self.primitive_id == "karaoke":
            raise ValueError("karaoke requires trusted word timing")
        if self.primitive_id == "scale" and self.scale_percent == 100:
            raise ValueError("scale primitive requires a bounded scale change")
        if self.primitive_id == "slide" and self.slide_distance_ratio == 0:
            raise ValueError("slide primitive requires a bounded distance")
        return self


class CaptionPlan(FrozenContract):
    # Default remains the frozen 7A shape for compatibility with persisted
    # foundation plans. The 7C planner always opts into the production schema.
    schema_version: Literal["7A.caption-plan.1", "7C.caption-plan.1"] = "7A.caption-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    cues: tuple[CaptionCuePlan, ...] = ()
    backend_id: Literal["none", "libass", "legacy_passthrough"] = "none"
    intensity: Intensity = Intensity.LOW
    font_manifest: CaptionFontManifest | None = None
    typography: CaptionTypographyToken | None = None
    quality_report: CaptionQualityReport = Field(default_factory=CaptionQualityReport)
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_cues(self) -> "CaptionPlan":
        if any(right.output.start_frame < left.output.start_frame for left, right in zip(self.cues, self.cues[1:])):
            raise ValueError("caption cues must be ordered")
        if self.schema_version == "7C.caption-plan.1" and self.backend_id == "libass":
            if self.font_manifest is None or self.typography is None:
                raise ValueError("7C libass captions require deterministic font and typography manifests")
            if self.quality_report.status == "LEGACY_UNASSESSED":
                raise ValueError("7C libass captions require an assessed quality report")
            if any(cue.normalized_bounds is None or cue.collision is None for cue in self.cues):
                raise ValueError("7C libass caption cues require frozen geometry and collision decisions")
        return self


class CompositionCropKeyframe(FrozenContract):
    """One deterministic crop state in normalized source coordinates."""

    frame: int = Field(ge=0)
    crop: NormalizedRect
    velocity_per_frame: float = Field(default=0, ge=0, le=1)
    acceleration_per_frame_sq: float = Field(default=0, ge=0, le=1)
    reason: Literal[
        "static", "target_acquired", "target_switch", "editorial_punch_in",
        "punch_out", "scene_reset", "safe_fallback",
    ] = "static"


class CompositionGeometryRegion(FrozenContract):
    """Stable collision-resolver input expressed on the output canvas."""

    region_id: str = Field(pattern=ID_PATTERN)
    kind: Literal["face", "subject", "object", "product", "screen", "reaction", "group", "overlay"]
    bounds: NormalizedRect
    target_ref: str | None = Field(default=None, pattern=ID_PATTERN)
    confidence: float = Field(default=1, ge=0, le=1)
    importance: float = Field(default=1, ge=0, le=1)


class CompositionGeometryContract(FrozenContract):
    """Coordinate-space hand-off for the future caption collision resolver.

    ``source_crop`` uses normalized source coordinates.  Every other rectangle
    uses normalized output-canvas coordinates, independent of render profile.
    """

    schema_version: Literal["7D.composition-geometry.1"] = "7D.composition-geometry.1"
    source_coordinate_space: Literal["normalized_source"] = "normalized_source"
    output_coordinate_space: Literal["normalized_output"] = "normalized_output"
    source_crop: NormalizedRect
    output_content_bounds: NormalizedRect
    target_regions: tuple[CompositionGeometryRegion, ...] = ()
    protected_regions: tuple[CompositionGeometryRegion, ...] = ()

    @model_validator(mode="after")
    def _unique_regions(self) -> "CompositionGeometryContract":
        for regions in (self.target_regions, self.protected_regions):
            ids = [item.region_id for item in regions]
            if len(ids) != len(set(ids)):
                raise ValueError("composition geometry region ids must be unique")
        return self


class CompositionPunchIn(FrozenContract):
    event_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    scale: float = Field(gt=1, le=1.12)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class CompositionQualityFinding(FrozenContract):
    code: Literal[
        "COMPOSITION_JITTER", "COMPOSITION_TARGET_CLIPPED", "COMPOSITION_UNSAFE_CROP",
        "COMPOSITION_VELOCITY_LIMIT", "COMPOSITION_ACCELERATION_LIMIT",
        "COMPOSITION_SWITCH_RATE_HIGH", "COMPOSITION_LAYOUT_SWITCH_RATE_HIGH",
        "COMPOSITION_MINIMUM_HOLD_VIOLATION",
        "COMPOSITION_LOW_CONFIDENCE", "COMPOSITION_SAFE_FALLBACK",
    ]
    severity: Literal["warning", "blocker"]
    segment_id: str | None = Field(default=None, pattern=ID_PATTERN)
    measured_value: float | int | str | bool | None = None
    threshold: float | int | str | bool | None = None
    message: str = Field(min_length=1, max_length=1200)


class CompositionQualityMetrics(FrozenContract):
    segment_count: int = Field(default=0, ge=0)
    target_switch_count: int = Field(default=0, ge=0)
    layout_switch_count: int = Field(default=0, ge=0)
    suppressed_switch_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    punch_in_count: int = Field(default=0, ge=0)
    clipped_target_count: int = Field(default=0, ge=0)
    unsafe_crop_count: int = Field(default=0, ge=0)
    jitter_event_count: int = Field(default=0, ge=0)
    minimum_hold_violation_count: int = Field(default=0, ge=0)
    max_velocity_per_frame: float = Field(default=0, ge=0)
    max_acceleration_per_frame_sq: float = Field(default=0, ge=0)
    switches_per_minute: float = Field(default=0, ge=0)
    layout_switches_per_minute: float = Field(default=0, ge=0)
    minimum_target_containment: float = Field(default=1, ge=0, le=1)


class CompositionQualityProvenance(FrozenContract):
    producer: str = Field(default="legacy_composition_contract", min_length=1, max_length=240)
    planner_version: str = Field(default="legacy", min_length=1, max_length=120)
    intent_id: str = Field(default="legacy", min_length=1, max_length=160)


class CompositionQualityReport(FrozenContract):
    schema_version: Literal["7D.composition-quality.1"] = "7D.composition-quality.1"
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED", "LEGACY_UNASSESSED"] = "LEGACY_UNASSESSED"
    findings: tuple[CompositionQualityFinding, ...] = ()
    metrics: CompositionQualityMetrics = Field(default_factory=CompositionQualityMetrics)
    provenance: CompositionQualityProvenance = Field(default_factory=CompositionQualityProvenance)

    @model_validator(mode="after")
    def _status_matches_findings(self) -> "CompositionQualityReport":
        has_blocker = any(item.severity == "blocker" for item in self.findings)
        has_warning = any(item.severity == "warning" for item in self.findings)
        expected = "BLOCKED" if has_blocker else "PASS_WITH_WARNINGS" if has_warning else "PASS"
        if self.status != "LEGACY_UNASSESSED" and self.status != expected:
            raise ValueError("composition quality status does not match findings")
        return self


class CompositionSegmentPlan(FrozenContract):
    segment_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    layout: LayoutFamily
    target: AttentionTarget = AttentionTarget.STABLE_SOURCE
    target_ref: str | None = Field(default=None, pattern=ID_PATTERN)
    crop: NormalizedRect
    target_confidence: float = Field(default=0, ge=0, le=1)
    target_bounds: NormalizedRect | None = None
    protected_regions: tuple[NormalizedRect, ...] = ()
    geometry: CompositionGeometryContract | None = None
    crop_keyframes: tuple[CompositionCropKeyframe, ...] = ()
    movement_reason: Literal[
        "none", "target_acquired", "target_switch", "editorial_punch_in",
        "punch_out", "scene_reset", "safe_fallback",
    ] = "none"
    punch_in: CompositionPunchIn | None = None
    easing_id: Literal["none", "linear", "ease_in_out"] = "none"
    evidence_refs: tuple[str, ...] = ()
    fallback: Literal["none", "wider_crop", "stable_source", "fit_background"] = "none"

    @model_validator(mode="after")
    def _valid_geometry_and_track(self) -> "CompositionSegmentPlan":
        if self.geometry is not None and self.geometry.source_crop != self.crop:
            raise ValueError("composition geometry source crop must match segment crop")
        if self.crop_keyframes:
            frames = [item.frame for item in self.crop_keyframes]
            if frames != sorted(set(frames)):
                raise ValueError("composition crop keyframes must be unique and ordered")
            if any(frame < self.output.start_frame or frame >= self.output.end_frame for frame in frames):
                raise ValueError("composition crop keyframe must stay inside its segment")
        if self.punch_in is not None and not self.output.contains(self.punch_in.output):
            raise ValueError("composition punch-in must stay inside its segment")
        if (self.punch_in is not None) != (self.movement_reason == "editorial_punch_in"):
            raise ValueError("composition punch-in requires an explicit editorial movement reason")
        return self


class CompositionPlan(FrozenContract):
    # 7A remains readable for persisted compatibility plans.  The 7D planner
    # always emits assessed geometry using the production schema.
    schema_version: Literal["7A.composition-plan.1", "7D.composition-plan.1"] = "7A.composition-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    segments: tuple[CompositionSegmentPlan, ...] = ()
    ordered_fallbacks: tuple[Literal["wider_crop", "stable_source", "fit_background"], ...] = (
        "wider_crop", "stable_source", "fit_background",
    )
    quality_report: CompositionQualityReport = Field(default_factory=CompositionQualityReport)
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_segments(self) -> "CompositionPlan":
        if any(
            right.output.start_frame < left.output.end_frame
            for left, right in zip(self.segments, self.segments[1:])
        ):
            raise ValueError("composition segments must be ordered without overlap")
        if self.schema_version == "7D.composition-plan.1":
            if self.ordered_fallbacks != ("wider_crop", "stable_source", "fit_background"):
                raise ValueError("7D composition fallback order is fixed for safety")
            if self.quality_report.status == "LEGACY_UNASSESSED":
                raise ValueError("7D composition requires an assessed quality report")
            if any(segment.geometry is None or not segment.crop_keyframes for segment in self.segments):
                raise ValueError("7D composition segments require frozen geometry and crop keyframes")
        return self


class MotionEventPlan(FrozenContract):
    event_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    purpose: MotionPurpose
    domain: MotionDomain
    primitive_id: Literal[
        "static", "fade", "scale", "slide", "dissolve", "crop_translate", "punch_in",
    ]
    easing_id: Literal["none", "linear", "ease_in_out"] = "none"
    intensity: Intensity
    evidence_refs: tuple[str, ...]
    fallback_primitive_id: Literal["static", "fade"] = "static"
    requested_primitive_id: Literal[
        "static", "fade", "scale", "slide", "dissolve", "crop_translate", "punch_in",
    ] | None = None
    backend_id: Literal["none", "libass", "ffmpeg"] = "none"
    duration_frames: int = Field(default=0, ge=0, le=30)
    scale_from: float = Field(default=1, ge=0.94, le=1.12)
    scale_to: float = Field(default=1, ge=0.94, le=1.12)
    translate_x_ratio: float = Field(default=0, ge=-0.05, le=0.05)
    translate_y_ratio: float = Field(default=0, ge=-0.05, le=0.05)
    opacity_from: float = Field(default=1, ge=0, le=1)
    opacity_to: float = Field(default=1, ge=0, le=1)
    target_plan_ids: tuple[str, ...] = ()
    budget_points: int = Field(default=0, ge=0, le=8)
    reduced_motion_fallback: bool = False
    fallback_reason: Literal[
        "reduced_motion", "readability", "cooldown", "concurrency",
        "animation_budget", "unsupported_primitive", "missing_domain_target", "short_event",
    ] | None = None

    @model_validator(mode="after")
    def _bounded_registered_motion(self) -> "MotionEventPlan":
        span = self.output.end_frame - self.output.start_frame
        if self.duration_frames > span:
            raise ValueError("motion duration must stay inside its editorial event")
        if self.primitive_id == "scale" and self.scale_from == self.scale_to:
            raise ValueError("scale motion requires a bounded scale change")
        if self.primitive_id == "slide" and self.translate_x_ratio == 0 and self.translate_y_ratio == 0:
            raise ValueError("slide motion requires a bounded translation")
        if self.primitive_id == "fade" and self.opacity_from == self.opacity_to:
            raise ValueError("fade motion requires a bounded opacity change")
        if self.primitive_id == "static" and self.budget_points:
            raise ValueError("static fallback cannot consume animation budget")
        if self.reduced_motion_fallback and self.primitive_id not in {"static", "fade"}:
            raise ValueError("reduced motion permits only static or fade")
        return self


class MotionAnimationBudget(FrozenContract):
    intensity: Intensity = Intensity.LOW
    point_limit: int = Field(default=0, ge=0)
    points_used: int = Field(default=0, ge=0)
    animated_frame_limit: int = Field(default=0, ge=0)
    animated_frames_used: int = Field(default=0, ge=0)
    cooldown_frames: int = Field(default=0, ge=0)
    max_concurrent_layers: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def _within_limits(self) -> "MotionAnimationBudget":
        if self.points_used > self.point_limit:
            raise ValueError("motion point budget exceeded")
        if self.animated_frames_used > self.animated_frame_limit:
            raise ValueError("motion frame budget exceeded")
        return self


class MotionQualityFinding(FrozenContract):
    code: Literal[
        "MOTION_COOLDOWN_SUPPRESSED", "MOTION_CONCURRENCY_SUPPRESSED",
        "MOTION_BUDGET_SUPPRESSED", "MOTION_READABILITY_SUPPRESSED",
        "MOTION_PRIMITIVE_FALLBACK", "MOTION_DOMAIN_TARGET_MISSING",
        "MOTION_REDUCED_MOTION_FALLBACK", "MOTION_SHORT_EVENT_FALLBACK",
    ]
    severity: Literal["warning", "blocker"]
    event_id: str | None = Field(default=None, pattern=ID_PATTERN)
    measured_value: float | int | str | bool | None = None
    threshold: float | int | str | bool | None = None
    message: str = Field(min_length=1, max_length=1200)


class MotionQualityMetrics(FrozenContract):
    requested_event_count: int = Field(default=0, ge=0)
    emitted_event_count: int = Field(default=0, ge=0)
    animated_event_count: int = Field(default=0, ge=0)
    suppressed_event_count: int = Field(default=0, ge=0)
    cooldown_suppression_count: int = Field(default=0, ge=0)
    concurrency_suppression_count: int = Field(default=0, ge=0)
    budget_suppression_count: int = Field(default=0, ge=0)
    readability_suppression_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    max_concurrent_layers: int = Field(default=0, ge=0, le=3)
    animation_points_used: int = Field(default=0, ge=0)
    animated_frames_used: int = Field(default=0, ge=0)
    events_per_minute: float = Field(default=0, ge=0)


class MotionQualityProvenance(FrozenContract):
    producer: str = Field(default="legacy_motion_contract", min_length=1, max_length=240)
    planner_version: str = Field(default="legacy", min_length=1, max_length=120)
    capability_registry_version: str = Field(default="legacy", min_length=1, max_length=120)
    intent_id: str = Field(default="legacy", min_length=1, max_length=160)
    caption_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    composition_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    source_broll_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)


class MotionQualityReport(FrozenContract):
    schema_version: Literal["7F.motion-quality.1"] = "7F.motion-quality.1"
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED", "LEGACY_UNASSESSED"] = "LEGACY_UNASSESSED"
    findings: tuple[MotionQualityFinding, ...] = ()
    metrics: MotionQualityMetrics = Field(default_factory=MotionQualityMetrics)
    provenance: MotionQualityProvenance = Field(default_factory=MotionQualityProvenance)

    @model_validator(mode="after")
    def _status_matches_findings(self) -> "MotionQualityReport":
        expected = (
            "BLOCKED" if any(item.severity == "blocker" for item in self.findings)
            else "PASS_WITH_WARNINGS" if self.findings else "PASS"
        )
        if self.status != "LEGACY_UNASSESSED" and self.status != expected:
            raise ValueError("motion quality status does not match findings")
        return self


class MotionPlan(FrozenContract):
    schema_version: Literal["7A.motion-plan.1", "7F.motion-plan.1"] = "7A.motion-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    events: tuple[MotionEventPlan, ...] = ()
    intensity: Intensity = Intensity.LOW
    reduced_motion: bool = False
    capability_registry_version: str = "legacy"
    animation_budget: MotionAnimationBudget = Field(default_factory=MotionAnimationBudget)
    caption_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    composition_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    source_broll_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    quality_report: MotionQualityReport = Field(default_factory=MotionQualityReport)
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_assessed_motion(self) -> "MotionPlan":
        if any(
            (right.output.start_frame, right.event_id) < (left.output.start_frame, left.event_id)
            for left, right in zip(self.events, self.events[1:])
        ):
            raise ValueError("motion events must be deterministically ordered")
        if self.schema_version == "7F.motion-plan.1":
            if None in {
                self.caption_plan_sha256,
                self.composition_plan_sha256,
                self.source_broll_plan_sha256,
            }:
                raise ValueError("7F motion requires current domain plan identities")
            if self.quality_report.status == "LEGACY_UNASSESSED":
                raise ValueError("7F motion requires an assessed quality report")
            if self.capability_registry_version != "7B.capability-registry.1":
                raise ValueError("7F motion requires the frozen 7B capability registry")
            if self.animation_budget.intensity != self.intensity:
                raise ValueError("motion intensity and animation budget must agree")
            if self.animation_budget.points_used != sum(item.budget_points for item in self.events):
                raise ValueError("motion budget accounting must match emitted events")
            animated_frames = sum(
                item.duration_frames for item in self.events if item.primitive_id != "static"
            )
            if self.animation_budget.animated_frames_used != animated_frames:
                raise ValueError("motion frame accounting must match emitted events")
            if any(
                (item.primitive_id == "static") != (item.duration_frames == 0)
                for item in self.events
            ):
                raise ValueError("7F static and animated timing must be explicit")
            if any(
                item.primitive_id == "punch_in" and item.scale_to <= item.scale_from
                for item in self.events
            ):
                raise ValueError("7F punch-in requires bounded increasing scale")
            if self.reduced_motion and any(
                item.primitive_id not in {"static", "fade"} for item in self.events
            ):
                raise ValueError("reduced MotionPlan permits only static or fade")
            metrics = self.quality_report.metrics
            if metrics.emitted_event_count != len(self.events):
                raise ValueError("motion quality event accounting must match MotionPlan")
            if metrics.animation_points_used != self.animation_budget.points_used:
                raise ValueError("motion quality point accounting must match MotionPlan")
            if metrics.animated_frames_used != self.animation_budget.animated_frames_used:
                raise ValueError("motion quality frame accounting must match MotionPlan")
        return self


class SourceBRollSafetyChecks(FrozenContract):
    evidence_relevance: Literal[True] = True
    story_unit_linked: Literal[True] = True
    beat_linked: Literal[True] = True
    source_identity_verified: Literal[True] = True
    attribution_verified: Literal[True] = True
    chronology_safe: Literal[True] = True
    causality_safe: Literal[True] = True
    payoff_timing_safe: Literal[True] = True
    screen_text_safe: Literal[True] = True
    source_rights_verified: Literal[True] = True
    lip_sync_not_required: Literal[True] = True
    filler_range_unique: Literal[True] = True


class SourceBRollQualityFinding(FrozenContract):
    code: str = Field(pattern=r"^SOURCE_BROLL_[A-Z0-9_]+$")
    severity: Literal["warning", "blocker"]
    decision_id: str | None = Field(default=None, pattern=ID_PATTERN)
    scene_id: str | None = Field(default=None, pattern=ID_PATTERN)
    measured_value: float | int | str | bool | None = None
    threshold: float | int | str | bool | None = None
    message: str = Field(min_length=1, max_length=1200)


class SourceBRollQualityMetrics(FrozenContract):
    proposal_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    a_roll_fallback_count: int = Field(default=0, ge=0)
    repeated_range_count: int = Field(default=0, ge=0)
    premature_reveal_count: int = Field(default=0, ge=0)
    attribution_violation_count: int = Field(default=0, ge=0)
    causality_violation_count: int = Field(default=0, ge=0)
    chronology_violation_count: int = Field(default=0, ge=0)
    selected_duration_frames: int = Field(default=0, ge=0)
    semantic_kind_counts: tuple[tuple[SourceBRollSemanticKind, int], ...] = ()


class SourceBRollQualityProvenance(FrozenContract):
    producer: str = Field(default="legacy_source_broll_contract", min_length=1, max_length=240)
    planner_version: str = Field(default="legacy", min_length=1, max_length=120)
    intent_id: str = Field(default="legacy", min_length=1, max_length=160)
    evidence_fingerprint: str | None = Field(default=None, pattern=HASH_PATTERN)
    composition_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)


class SourceBRollQualityReport(FrozenContract):
    schema_version: Literal["7E.source-broll-quality.1"] = "7E.source-broll-quality.1"
    status: Literal["PASS", "PASS_WITH_WARNINGS", "BLOCKED", "LEGACY_UNASSESSED"] = "LEGACY_UNASSESSED"
    findings: tuple[SourceBRollQualityFinding, ...] = ()
    metrics: SourceBRollQualityMetrics = Field(default_factory=SourceBRollQualityMetrics)
    provenance: SourceBRollQualityProvenance = Field(default_factory=SourceBRollQualityProvenance)

    @model_validator(mode="after")
    def _status_matches_findings(self) -> "SourceBRollQualityReport":
        expected = (
            "BLOCKED" if any(item.severity == "blocker" for item in self.findings)
            else "PASS_WITH_WARNINGS" if self.findings else "PASS"
        )
        if self.status != "LEGACY_UNASSESSED" and self.status != expected:
            raise ValueError("source B-roll quality status does not match findings")
        return self


class SourceBRollSegmentPlan(FrozenContract):
    segment_id: str = Field(pattern=ID_PATTERN)
    decision_id: str = Field(default="legacy", pattern=ID_PATTERN)
    destination: OutputInterval
    source_cutaway: SourceInterval
    source_crop: NormalizedRect | None = None
    source_target: AttentionTarget | None = None
    source_scene_id: str = Field(default="legacy", pattern=ID_PATTERN)
    story_unit_id: str = Field(pattern=ID_PATTERN)
    beat_role: BeatRole | None = None
    semantic_kind: SourceBRollSemanticKind = SourceBRollSemanticKind.CONTEXT
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    scene_evidence_refs: tuple[str, ...] = ()
    relevance_confidence: float = Field(default=0, ge=0, le=1)
    retain_source_audio: Literal[False] = False
    audio_timeline: Literal["a_roll_master"] = "a_roll_master"
    transition: Literal["cut", "short_dissolve"] = "cut"
    fallback: Literal["a_roll"] = "a_roll"
    fallback_composition_segment_ids: tuple[str, ...] = ()
    safety_checks: SourceBRollSafetyChecks | None = None
    provenance: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _evidence_is_unique(self) -> "SourceBRollSegmentPlan":
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("source B-roll evidence refs must be unique")
        if len(self.scene_evidence_refs) != len(set(self.scene_evidence_refs)):
            raise ValueError("source B-roll scene evidence refs must be unique")
        if not set(self.scene_evidence_refs).issubset(self.evidence_refs):
            raise ValueError("source B-roll scene evidence must be included in provenance refs")
        return self


class SourceBRollPlan(FrozenContract):
    schema_version: Literal["7A.source-broll-plan.1", "7E.source-broll-plan.1"] = "7A.source-broll-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    segments: tuple[SourceBRollSegmentPlan, ...] = ()
    default_visual: Literal["a_roll"] = "a_roll"
    default_audio: Literal["a_roll_master"] = "a_roll_master"
    fallback_policy: Literal["a_roll_current_composition"] = "a_roll_current_composition"
    composition_plan_sha256: str | None = Field(default=None, pattern=HASH_PATTERN)
    quality_report: SourceBRollQualityReport = Field(default_factory=SourceBRollQualityReport)
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _safe_ordered_segments(self) -> "SourceBRollPlan":
        if any(
            right.destination.start_frame < left.destination.end_frame
            for left, right in zip(self.segments, self.segments[1:])
        ):
            raise ValueError("source B-roll destinations must be ordered without overlap")
        ranges = [item.source_cutaway for item in self.segments]
        if any(left.overlaps(right) for index, left in enumerate(ranges) for right in ranges[index + 1:]):
            raise ValueError("source B-roll cannot repeat a filler range")
        if self.schema_version == "7E.source-broll-plan.1":
            if self.composition_plan_sha256 is None:
                raise ValueError("7E source B-roll requires current CompositionPlan identity")
            if self.quality_report.status == "LEGACY_UNASSESSED":
                raise ValueError("7E source B-roll requires an assessed quality report")
            if any(item.safety_checks is None or item.beat_role is None for item in self.segments):
                raise ValueError("7E source B-roll segments require beat and safety evidence")
        return self


class CanvasPlan(FrozenContract):
    width: int = Field(ge=2, le=7680)
    height: int = Field(ge=2, le=7680)
    fps: Literal[30] = 30
    sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)

    @model_validator(mode="after")
    def _valid_canvas(self) -> "CanvasPlan":
        if self.width % 2 or self.height % 2:
            raise ValueError("compiled canvas dimensions must be even")
        return self


class RenderProfile(FrozenContract):
    """Quality-only choices which are deliberately outside the compiled plan."""

    schema_version: Literal["7G.render-profile.1"] = "7G.render-profile.1"
    profile_id: Literal["creative_preview", "final"]
    width: int = Field(ge=2, le=7680)
    height: int = Field(ge=2, le=7680)
    fps: Literal[30] = 30
    video_bitrate: str = Field(min_length=2, max_length=32)
    encoder: Literal["auto", "nvenc", "cpu"] = "auto"
    sampling_precision: Literal["full", "preview"] = "full"

    @model_validator(mode="after")
    def _valid_profile(self) -> "RenderProfile":
        if self.width % 2 or self.height % 2:
            raise ValueError("render profile dimensions must be even")
        if self.profile_id == "final" and self.sampling_precision != "full":
            raise ValueError("final render profile requires full sampling precision")
        return self


class AssetManifestEntry(FrozenContract):
    asset_id: str = Field(pattern=ID_PATTERN)
    asset_type: Literal["font", "image", "overlay"]
    checksum: str = Field(pattern=HASH_PATTERN)
    approved: Literal[True] = True


class BackendAssignment(FrozenContract):
    domain: Literal["base_video", "caption", "composition", "motion", "broll"]
    backend_id: Literal["ffmpeg", "libass", "none", "legacy_passthrough"]
    backend_version: str = Field(min_length=1, max_length=120)
    deterministic: Literal[True] = True


class RenderGraphNode(FrozenContract):
    node_id: str = Field(pattern=ID_PATTERN)
    node_kind: Literal[
        "captions", "composition", "broll", "motion",
        "base_visual", "composite", "encode", "qc",
    ]
    dependency_ids: tuple[str, ...] = ()
    cache_key: str = Field(pattern=HASH_PATTERN)
    backend_domain: Literal["base_video", "caption", "composition", "motion", "broll"]


class CompiledInputFingerprints(FrozenContract):
    production_plan_sha256: str = Field(pattern=HASH_PATTERN)
    creative_intent_sha256: str = Field(pattern=HASH_PATTERN)
    proposal_sha256: str = Field(pattern=HASH_PATTERN)
    evidence_sha256: str = Field(pattern=HASH_PATTERN)
    edit_mapping_sha256: str = Field(pattern=HASH_PATTERN)
    caption_plan_sha256: str = Field(pattern=HASH_PATTERN)
    composition_plan_sha256: str = Field(pattern=HASH_PATTERN)
    motion_plan_sha256: str = Field(pattern=HASH_PATTERN)
    source_broll_plan_sha256: str = Field(pattern=HASH_PATTERN)


class CompiledRenderPlan(FrozenContract):
    """Immutable Phase 7 hand-off; no raw proposal or executable strings."""

    schema_version: Literal["7G.compiled-render-plan.1"] = "7G.compiled-render-plan.1"
    production_plan: ImmutableProductionPlanLink
    intent_id: str = Field(pattern=ID_PATTERN)
    intent_hash: str = Field(pattern=HASH_PATTERN)
    input_fingerprints: CompiledInputFingerprints
    source_output_mapping: SourceOutputTimeMap
    canvas: CanvasPlan
    caption_plan: CaptionPlan
    composition_plan: CompositionPlan
    motion_plan: MotionPlan
    source_broll_plan: SourceBRollPlan
    assets: tuple[AssetManifestEntry, ...] = ()
    backends: tuple[BackendAssignment, ...] = ()
    render_graph_nodes: tuple[RenderGraphNode, ...] = Field(min_length=1)
    expected_quality_constraints: tuple[str, ...] = ()
    ordered_fallbacks: tuple[str, ...] = ()
    compatibility_mode: Literal["native", "legacy_passthrough"] = "native"
    plan_hash: str = Field(pattern=HASH_PATTERN)
    parity_signature: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _valid_compiled_plan(self) -> "CompiledRenderPlan":
        if any(
            plan.intent_id != self.intent_id
            for plan in (self.caption_plan, self.composition_plan, self.motion_plan, self.source_broll_plan)
        ):
            raise ValueError("all domain plans must reference the compiled intent")
        seen: set[str] = set()
        for node in self.render_graph_nodes:
            if node.node_id in seen:
                raise ValueError("render graph node ids must be unique")
            if any(dependency not in seen for dependency in node.dependency_ids):
                raise ValueError("render graph dependencies must precede their consumer")
            seen.add(node.node_id)
        expected_hash = _compiled_plan_hash(self.model_dump(mode="json"))
        if self.plan_hash != expected_hash:
            raise ValueError("COMPILED_PLAN_HASH_MISMATCH")
        expected_parity = _parity_signature(self.model_dump(mode="json"), expected_hash)
        if self.parity_signature != expected_parity:
            raise ValueError("PARITY_SIGNATURE_MISMATCH")
        return self


class RenderParityManifest(FrozenContract):
    """Auditable semantic/timing/layout identity emitted by every render profile."""

    schema_version: Literal["7G.parity-manifest.1"] = "7G.parity-manifest.1"
    profile_id: Literal["creative_preview", "final"]
    plan_hash: str = Field(pattern=HASH_PATTERN)
    parity_signature: str = Field(pattern=HASH_PATTERN)
    fps: Literal[30] = 30
    event_frames_hash: str = Field(pattern=HASH_PATTERN)
    resolved_lines_hash: str = Field(pattern=HASH_PATTERN)
    normalized_geometry_hash: str = Field(pattern=HASH_PATTERN)
    font_asset_hash: str = Field(pattern=HASH_PATTERN)
    motion_math_hash: str = Field(pattern=HASH_PATTERN)
    output_checksum: str | None = Field(default=None, pattern=HASH_PATTERN)


class ParityCheckResult(FrozenContract):
    status: Literal["matched", "mismatch"]
    mismatch_fields: tuple[str, ...] = ()


def build_render_parity_manifest(
    plan: CompiledRenderPlan,
    profile: RenderProfile,
    *,
    output_checksum: str | None = None,
) -> RenderParityManifest:
    """Bind a quality profile to one plan without changing creative identity."""

    if profile.fps != plan.canvas.fps:
        raise ValueError("RENDER_PROFILE_FPS_MISMATCH")
    if profile.width * plan.canvas.height != profile.height * plan.canvas.width:
        raise ValueError("RENDER_PROFILE_ASPECT_MISMATCH")
    if profile.profile_id == "final" and (
        profile.width != plan.canvas.width or profile.height != plan.canvas.height
    ):
        raise ValueError("FINAL_PROFILE_CANVAS_MISMATCH")
    parity = _parity_payload(plan.model_dump(mode="json"), plan.plan_hash)
    caption_events = parity["caption_events"]
    composition_events = parity["composition_events"]
    motion_events = parity["motion_events"]
    broll_events = parity["broll_events"]
    return RenderParityManifest(
        profile_id=profile.profile_id,
        plan_hash=plan.plan_hash,
        parity_signature=plan.parity_signature,
        event_frames_hash=canonical_hash({
            "timeline": parity["timeline"],
            "captions": [item["output"] for item in caption_events],
            "composition": [item["output"] for item in composition_events],
            "motion": [item["output"] for item in motion_events],
            "broll": [item["destination"] for item in broll_events],
        }),
        resolved_lines_hash=canonical_hash([item["lines"] for item in caption_events]),
        normalized_geometry_hash=canonical_hash({
            "captions": [
                {"lane": item["lane"], "bounds": item["bounds"]}
                for item in caption_events
            ],
            "composition": [
                {"layout": item["layout"], "target": item["target"], "crop": item["crop"]}
                for item in composition_events
            ],
        }),
        font_asset_hash=canonical_hash({
            "font": parity["caption_font"],
            "typography": parity["caption_typography"],
            "assets": parity["asset_hashes"],
        }),
        motion_math_hash=canonical_hash({
            "caption": [
                {
                    "primitive": item["primitive"],
                    "easing": item["easing"],
                    "duration": item["motion_duration_frames"],
                    "scale": item["scale_percent"],
                    "slide": item["slide_distance_ratio"],
                }
                for item in caption_events
            ],
            "motion": motion_events,
        }),
        output_checksum=output_checksum,
    )


def check_preview_final_parity(
    preview: RenderParityManifest,
    final: RenderParityManifest,
) -> ParityCheckResult:
    fields = (
        "plan_hash", "parity_signature", "fps", "event_frames_hash",
        "resolved_lines_hash", "normalized_geometry_hash", "font_asset_hash",
        "motion_math_hash",
    )
    mismatches = tuple(field for field in fields if getattr(preview, field) != getattr(final, field))
    if preview.profile_id != "creative_preview":
        mismatches = ("preview_profile_id", *mismatches)
    if final.profile_id != "final":
        mismatches = (*mismatches, "final_profile_id")
    return ParityCheckResult(
        status="mismatch" if mismatches else "matched",
        mismatch_fields=mismatches,
    )


def assert_preview_final_parity(
    preview: RenderParityManifest,
    final: RenderParityManifest,
) -> None:
    result = check_preview_final_parity(preview, final)
    if result.status == "mismatch":
        raise ValueError(
            "PREVIEW_FINAL_PARITY_MISMATCH: " + ", ".join(result.mismatch_fields)
        )


def _compiled_plan_hash(payload: Mapping[str, Any]) -> str:
    core = dict(payload)
    core.pop("plan_hash", None)
    core.pop("parity_signature", None)
    return canonical_hash(core)


def _parity_payload(payload: Mapping[str, Any], plan_hash: str) -> dict[str, Any]:
    caption = payload["caption_plan"]
    composition = payload["composition_plan"]
    motion = payload["motion_plan"]
    broll = payload["source_broll_plan"]
    return {
        "schema_version": PARITY_SIGNATURE_SCHEMA_VERSION,
        "plan_hash": plan_hash,
        "timeline": payload["source_output_mapping"],
        "fps": payload["canvas"]["fps"],
        "caption_events": [
            {
                "output": cue["output"],
                "lines": cue["resolved_lines"],
                "lane": cue["lane"],
                "bounds": cue["normalized_bounds"],
                "primitive": cue["primitive_id"],
                "easing": cue["easing_id"],
                "words": [
                    {
                        "output": word["output"],
                        "timing_source": word["timing_source"],
                    }
                    for word in cue.get("words", [])
                ],
                "emphasis": cue.get("emphasis"),
                "collision": cue.get("collision"),
                "resolved_font_size_ratio": cue.get("resolved_font_size_ratio"),
                "motion_duration_frames": cue.get("motion_duration_frames", 0),
                "scale_percent": cue.get("scale_percent", 100),
                "slide_distance_ratio": cue.get("slide_distance_ratio", 0),
            }
            for cue in caption["cues"]
        ],
        "caption_font": caption.get("font_manifest"),
        "caption_typography": caption.get("typography"),
        "composition_events": [
            {
                "output": segment["output"],
                "layout": segment["layout"],
                "target": segment["target"],
                "crop": segment["crop"],
                "easing": segment["easing_id"],
            }
            for segment in composition["segments"]
        ],
        "motion_events": [
            {
                "output": event["output"],
                "primitive": event["primitive_id"],
                "easing": event["easing_id"],
            }
            for event in motion["events"]
        ],
        "broll_events": [
            {"destination": segment["destination"], "source": segment["source_cutaway"]}
            for segment in broll["segments"]
        ],
        "asset_hashes": [item["checksum"] for item in payload["assets"]],
        "backends": [
            (item["domain"], item["backend_id"], item["backend_version"])
            for item in payload["backends"]
        ],
    }


def _parity_signature(payload: Mapping[str, Any], plan_hash: str) -> str:
    return canonical_hash(_parity_payload(payload, plan_hash))


def _cache_domain_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove lifecycle identity that is not an execution dependency.

    A caption-only ProductionPlan revision receives a new intent id, but that
    must not invalidate byte-identical composition/B-roll/motion work.
    """

    result = dict(value)
    result.pop("intent_id", None)
    return result


def _render_graph_nodes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    caption_key = canonical_hash({
        "captions": _cache_domain_payload(payload["caption_plan"]),
        "assets": [item for item in payload["assets"] if item["asset_type"] == "font"],
        "backend": [item for item in payload["backends"] if item["domain"] == "caption"],
    })
    composition_key = canonical_hash({
        "composition": _cache_domain_payload(payload["composition_plan"]),
        "mapping": payload["source_output_mapping"],
        "backend": [item for item in payload["backends"] if item["domain"] == "composition"],
    })
    broll_key = canonical_hash({
        "broll": _cache_domain_payload(payload["source_broll_plan"]),
        "mapping": payload["source_output_mapping"],
        "backend": [item for item in payload["backends"] if item["domain"] == "broll"],
    })
    motion_key = canonical_hash({
        "motion": _cache_domain_payload(payload["motion_plan"]),
        "backend": [item for item in payload["backends"] if item["domain"] == "motion"],
    })
    base_key = canonical_hash({
        "mapping": payload["source_output_mapping"],
        "canvas_aspect": [payload["canvas"]["width"], payload["canvas"]["height"]],
        "composition": composition_key,
        "broll": broll_key,
        "backend": [item for item in payload["backends"] if item["domain"] == "base_video"],
    })
    composite_key = canonical_hash({
        "base": base_key,
        "captions": caption_key,
        "motion": motion_key,
    })
    encode_key = canonical_hash({"composite": composite_key})
    qc_key = canonical_hash({
        "encode": encode_key,
        "constraints": payload["expected_quality_constraints"],
    })
    return [
        {
            "node_id": "captions",
            "node_kind": "captions",
            "dependency_ids": [],
            "cache_key": caption_key,
            "backend_domain": "caption",
        },
        {
            "node_id": "composition",
            "node_kind": "composition",
            "dependency_ids": [],
            "cache_key": composition_key,
            "backend_domain": "composition",
        },
        {
            "node_id": "broll",
            "node_kind": "broll",
            "dependency_ids": [],
            "cache_key": broll_key,
            "backend_domain": "broll",
        },
        {
            "node_id": "motion",
            "node_kind": "motion",
            "dependency_ids": [],
            "cache_key": motion_key,
            "backend_domain": "motion",
        },
        {
            "node_id": "base-visual",
            "node_kind": "base_visual",
            "dependency_ids": ["composition", "broll"],
            "cache_key": base_key,
            "backend_domain": "base_video",
        },
        {
            "node_id": "composite",
            "node_kind": "composite",
            "dependency_ids": ["captions", "motion", "base-visual"],
            "cache_key": composite_key,
            "backend_domain": "base_video",
        },
        {
            "node_id": "encode",
            "node_kind": "encode",
            "dependency_ids": ["composite"],
            "cache_key": encode_key,
            "backend_domain": "base_video",
        },
        {
            "node_id": "qc",
            "node_kind": "qc",
            "dependency_ids": ["encode"],
            "cache_key": qc_key,
            "backend_domain": "base_video",
        },
    ]


def compile_render_plan(
    intent: CreativeIntent,
    caption_plan: CaptionPlan,
    composition_plan: CompositionPlan,
    motion_plan: MotionPlan,
    source_broll_plan: SourceBRollPlan,
    canvas: CanvasPlan,
    *,
    assets: Iterable[AssetManifestEntry] = (),
    backends: Iterable[BackendAssignment] = (),
    compatibility_mode: Literal["native", "legacy_passthrough"] = "native",
) -> CompiledRenderPlan:
    _validate_domain_plans(
        intent,
        caption_plan,
        composition_plan,
        motion_plan,
        source_broll_plan,
    )
    payload: dict[str, Any] = {
        "schema_version": COMPILED_RENDER_PLAN_SCHEMA_VERSION,
        "production_plan": intent.production_plan.model_dump(mode="json"),
        "intent_id": intent.intent_id,
        "intent_hash": intent.canonical_hash(),
        "source_output_mapping": intent.source_output_mapping.model_dump(mode="json"),
        "canvas": canvas.model_dump(mode="json"),
        "caption_plan": caption_plan.model_dump(mode="json"),
        "composition_plan": composition_plan.model_dump(mode="json"),
        "motion_plan": motion_plan.model_dump(mode="json"),
        "source_broll_plan": source_broll_plan.model_dump(mode="json"),
        "assets": [item.model_dump(mode="json") for item in assets],
        "backends": [item.model_dump(mode="json") for item in backends],
        "expected_quality_constraints": [
            "no_raw_commands", "identity_match", "source_ranges_bounded",
            "source_broll_evidence_and_forbidden_checks", "a_roll_master_audio",
            "motion_capability_registry", "cross_domain_animation_budget",
            "motion_cooldown_and_concurrency", "reduced_motion_fallback",
            "preview_final_parity",
        ],
        "ordered_fallbacks": list(intent.ordered_fallbacks),
        "compatibility_mode": compatibility_mode,
    }
    payload["input_fingerprints"] = {
        "production_plan_sha256": intent.production_plan.plan_fingerprint,
        "creative_intent_sha256": payload["intent_hash"],
        "proposal_sha256": intent.proposal_hash,
        "evidence_sha256": intent.evidence_fingerprint,
        "edit_mapping_sha256": intent.source_output_mapping.fingerprint,
        "caption_plan_sha256": caption_plan.canonical_hash(),
        "composition_plan_sha256": composition_plan.canonical_hash(),
        "motion_plan_sha256": motion_plan.canonical_hash(),
        "source_broll_plan_sha256": source_broll_plan.canonical_hash(),
    }
    payload["render_graph_nodes"] = _render_graph_nodes(payload)
    plan_hash = _compiled_plan_hash(payload)
    payload["plan_hash"] = plan_hash
    payload["parity_signature"] = _parity_signature(payload, plan_hash)
    return CompiledRenderPlan.model_validate(payload)


def _validate_domain_plans(
    intent: CreativeIntent,
    captions: CaptionPlan,
    composition: CompositionPlan,
    motion: MotionPlan,
    broll: SourceBRollPlan,
) -> None:
    plans = (captions, composition, motion, broll)
    if any(plan.intent_id != intent.intent_id for plan in plans):
        raise ValueError("DOMAIN_PLAN_INTENT_MISMATCH")

    known_refs = {
        ref
        for decision in (
            *intent.beats,
            *intent.semantic_emphasis,
            *intent.composition_targets,
            *intent.motion_events,
            *intent.source_broll,
        )
        for ref in decision.evidence_refs
    }
    for cue in captions.cues:
        if cue.evidence_refs and not set(cue.evidence_refs).issubset(known_refs):
            raise ValueError("CAPTION_PLAN_EVIDENCE_MISMATCH")
        if cue.semantic_class is not None and not cue.evidence_refs:
            raise ValueError("semantic caption emphasis requires evidence")

    for segment in composition.segments:
        if segment.target == AttentionTarget.STABLE_SOURCE:
            target_valid = True
        else:
            target_valid = any(
                target.target == segment.target
                and target.output.contains(segment.output)
                and set(segment.evidence_refs).issubset(target.evidence_refs)
                and bool(segment.evidence_refs)
                for target in intent.composition_targets
            )
        if not target_valid:
            raise ValueError("COMPOSITION_PLAN_EVIDENCE_MISMATCH")
        if segment.punch_in is not None and not any(
            event.decision_id == segment.punch_in.event_id
            and event.domain == MotionDomain.COMPOSITION
            and event.output.contains(segment.punch_in.output)
            and set(segment.punch_in.evidence_refs).issubset(event.evidence_refs)
            and bool(segment.punch_in.evidence_refs)
            for event in intent.motion_events
        ):
            raise ValueError("COMPOSITION_PUNCH_IN_EVIDENCE_MISMATCH")

    for event in motion.events:
        if not any(
            request.decision_id == event.event_id
            and request.purpose == event.purpose
            and request.domain == event.domain
            and request.output.contains(event.output)
            and set(event.evidence_refs).issubset(request.evidence_refs)
            and bool(event.evidence_refs)
            for request in intent.motion_events
        ):
            raise ValueError("MOTION_PLAN_EVIDENCE_MISMATCH")

    if motion.schema_version == "7F.motion-plan.1":
        if motion.caption_plan_sha256 != captions.canonical_hash():
            raise ValueError("MOTION_CAPTION_PLAN_STALE")
        if motion.composition_plan_sha256 != composition.canonical_hash():
            raise ValueError("MOTION_COMPOSITION_PLAN_STALE")
        if motion.source_broll_plan_sha256 != broll.canonical_hash():
            raise ValueError("MOTION_SOURCE_BROLL_PLAN_STALE")
        if motion.intensity != intent.policy.intensity:
            raise ValueError("MOTION_INTENSITY_POLICY_MISMATCH")
        if motion.reduced_motion != intent.policy.reduced_motion:
            raise ValueError("MOTION_REDUCED_POLICY_MISMATCH")
        caption_ids = {item.cue_id for item in captions.cues}
        composition_ids = {item.segment_id for item in composition.segments}
        broll_ids = {item.segment_id for item in broll.segments}
        for event in motion.events:
            valid_ids = (
                caption_ids if event.domain == MotionDomain.CAPTION
                else composition_ids if event.domain == MotionDomain.COMPOSITION
                else broll_ids | composition_ids
            )
            if not event.target_plan_ids or not set(event.target_plan_ids).issubset(valid_ids):
                raise ValueError("MOTION_DOMAIN_TARGET_MISMATCH")

    if broll.segments and not intent.policy.source_broll_enabled:
        raise ValueError("SOURCE_BROLL_DISABLED")
    for broll_segment in broll.segments:
        if not any(
            request.decision_id == broll_segment.decision_id
            and request.output == broll_segment.destination
            and request.source_cutaway == broll_segment.source_cutaway
            and request.story_unit_id == broll_segment.story_unit_id
            and request.semantic_kind == broll_segment.semantic_kind
            and request.story_unit_evidence_ref in broll_segment.evidence_refs
            and set(broll_segment.scene_evidence_refs).issubset(request.source_cutaway_evidence_refs)
            for request in intent.source_broll
        ):
            raise ValueError("SOURCE_BROLL_PLAN_EVIDENCE_MISMATCH")
        if not set(broll_segment.evidence_refs).issubset({item.evidence_ref for item in intent.evidence_manifest}):
            raise ValueError("SOURCE_BROLL_PLAN_EVIDENCE_MISMATCH")
        if broll.schema_version == "7E.source-broll-plan.1" and broll_segment.safety_checks is None:
            raise ValueError("SOURCE_BROLL_PLAN_SAFETY_UNASSESSED")


def legacy_safe_intent(
    plan: ProductionPlan,
    mapping: SourceOutputTimeMap,
) -> CreativeIntent:
    """Describe an existing ProductionPlan without introducing new decisions."""

    reference = ImmutableProductionPlanLink.from_reference(plan.reference())
    payload = {
        "production_plan": reference.model_dump(mode="json"),
        "mapping": mapping.model_dump(mode="json"),
        "mode": "legacy_passthrough",
    }
    envelope = plan.envelope
    assert envelope is not None
    policy = CreativePolicy(
        preset_id=envelope.preset.preset_id,
        preset_version=envelope.preset.preset_version,
        platform=envelope.preset.platform,
        caption_style_family="clean",
        caption_density="balanced",
        intensity=Intensity.LOW,
        reduced_motion=True,
        source_broll_enabled=False,
    )
    digest = canonical_hash(payload)
    return CreativeIntent(
        intent_id=f"intent-legacy-{digest[:17]}",
        revision=1,
        production_plan=reference,
        source_output_mapping=mapping,
        evidence_fingerprint=envelope.input_fingerprints.analysis_sha256,
        evidence_manifest=(),
        proposal_hash=digest,
        policy=policy,
        confidence=0,
        provenance=("legacy_production_plan",),
        diagnostics=(CreativeDiagnostic(
            code="SAFE_FALLBACK_APPLIED",
            decision_id="legacy-compatibility",
            fallback="stable_source",
        ),),
    )


def caption_plan_from_legacy(
    intent: CreativeIntent,
    project: SubtitleProject,
    *,
    canvas_width: int,
    canvas_height: int,
) -> CaptionPlan:
    """Describe the fitted SubtitleProject without changing ASS generation."""

    bounds_by_cue = {item.cue_id: item for item in project.quality_decision.rendered_bounds}
    cues: list[CaptionCuePlan] = []
    diagnostics = [f"legacy_subtitle_project_sha256:{canonical_hash(project)}"]
    for cue in project.cues:
        raw_lines = tuple(line for line in cue.resolved_lines if line) or tuple(
            line for line in cue.text.splitlines() if line
        ) or (cue.text,)
        if len(raw_lines) > 2:
            diagnostics.append(f"{cue.cue_id}:legacy_line_count_exceeds_phase7_contract")
            raw_lines = raw_lines[:2]
        bounds = bounds_by_cue.get(cue.cue_id)
        normalized_bounds = None
        if bounds is not None:
            normalized_x = max(0.0, min(1.0 - (1 / canvas_width), bounds.x / canvas_width))
            normalized_y = max(0.0, min(1.0 - (1 / canvas_height), bounds.y / canvas_height))
            normalized_bounds = NormalizedRect(
                x=normalized_x,
                y=normalized_y,
                width=max(1 / canvas_width, min(1.0 - normalized_x, bounds.width / canvas_width)),
                height=max(1 / canvas_height, min(1.0 - normalized_y, bounds.height / canvas_height)),
            )
        cues.append(CaptionCuePlan(
            cue_id=cue.cue_id,
            output=OutputInterval.from_seconds(cue.start_seconds, cue.end_seconds),
            resolved_lines=raw_lines,
            lane="upper" if project.style.position == "top" else "lower",
            typography_token_id=f"legacy-{project.style.style_id}",
            primitive_id="legacy_passthrough",
            normalized_bounds=normalized_bounds,
        ))
    return CaptionPlan(
        intent_id=intent.intent_id,
        cues=tuple(cues),
        backend_id="legacy_passthrough",
        diagnostics=tuple(diagnostics),
    )


def composition_plan_from_legacy(
    intent: CreativeIntent,
    reframe: ReframePlan,
) -> CompositionPlan:
    """Describe persisted composition states while retaining their old owner."""

    segments: list[CompositionSegmentPlan] = []
    for item in reframe.composition_segments:
        crop = item.target_crop
        if (
            crop is not None
            and crop.crop_x is not None
            and crop.crop_y is not None
            and crop.crop_width is not None
            and crop.crop_height is not None
        ):
            rect = NormalizedRect(
                x=crop.crop_x / crop.source_width,
                y=crop.crop_y / crop.source_height,
                width=crop.crop_width / crop.source_width,
                height=crop.crop_height / crop.source_height,
            )
        else:
            rect = NormalizedRect(x=0, y=0, width=1, height=1)
        layout = {
            "fit_with_blur": LayoutFamily.FIT_BACKGROUND,
            "split_layout": LayoutFamily.SPLIT,
            "group_framing": LayoutFamily.WIDE_GROUP,
            "scene_wide": LayoutFamily.WIDE_GROUP,
        }.get(item.strategy, LayoutFamily.LEGACY_PASSTHROUGH)
        segments.append(CompositionSegmentPlan(
            segment_id=item.segment_id,
            output=OutputInterval.from_seconds(item.start_seconds, item.end_seconds),
            layout=layout,
            target=AttentionTarget.STABLE_SOURCE,
            crop=rect,
            easing_id="ease_in_out" if item.target_crop and item.target_crop.tracking_keyframes else "none",
            fallback="stable_source" if item.fallback_reason else "none",
        ))
    return CompositionPlan(
        intent_id=intent.intent_id,
        segments=tuple(segments),
        diagnostics=(f"legacy_reframe_plan_sha256:{canonical_hash(reframe)}",),
    )


def compile_legacy_render_plan(
    plan: ProductionPlan,
    mapping: SourceOutputTimeMap,
    *,
    caption_plan: CaptionPlan | None = None,
    composition_plan: CompositionPlan | None = None,
    subtitle_project: SubtitleProject | None = None,
    reframe_plan: ReframePlan | None = None,
) -> CompiledRenderPlan:
    """Compile a parity signature around the current render path.

    This adapter intentionally emits no motion and no B-roll insertion.  It
    does not call or replace ``VideoCompositionService``; current visuals remain
    owned by the established ProductionPlan/VideoProject renderer flow.
    """

    intent = legacy_safe_intent(plan, mapping)
    if caption_plan is not None and subtitle_project is not None:
        raise ValueError("provide either caption_plan or subtitle_project, not both")
    if composition_plan is not None and reframe_plan is not None:
        raise ValueError("provide either composition_plan or reframe_plan, not both")
    envelope = plan.envelope
    assert envelope is not None
    caption = caption_plan or (
        caption_plan_from_legacy(
            intent,
            subtitle_project,
            canvas_width=envelope.target.width,
            canvas_height=envelope.target.height,
        )
        if subtitle_project is not None
        else CaptionPlan(
            intent_id=intent.intent_id,
            backend_id="legacy_passthrough",
            diagnostics=("legacy subtitle behavior retained",),
        )
    )
    composition = composition_plan or (
        composition_plan_from_legacy(intent, reframe_plan)
        if reframe_plan is not None
        else CompositionPlan(
            intent_id=intent.intent_id,
            diagnostics=("legacy composition behavior retained",),
        )
    )
    motion = MotionPlan(
        intent_id=intent.intent_id,
        reduced_motion=True,
        diagnostics=("no Phase 7 motion enabled",),
    )
    broll = SourceBRollPlan(
        intent_id=intent.intent_id,
        diagnostics=("source B-roll insertion disabled",),
    )
    return compile_render_plan(
        intent,
        caption,
        composition,
        motion,
        broll,
        CanvasPlan(
            width=envelope.target.width,
            height=envelope.target.height,
            fps=30,
        ),
        backends=(
            BackendAssignment(
                domain="base_video", backend_id="legacy_passthrough",
                backend_version="video-project-3D.0", deterministic=True,
            ),
        ),
        compatibility_mode="legacy_passthrough",
    )
