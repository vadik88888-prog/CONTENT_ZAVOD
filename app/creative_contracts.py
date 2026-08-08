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

from app.production_models import ProductionPlan, ProductionPlanReference
from app.video_models import ReframePlan, SubtitleProject, VideoTimeline


CREATIVE_PROPOSAL_SCHEMA_VERSION = "7A.proposal.1"
CREATIVE_INTENT_SCHEMA_VERSION = "7A.intent.1"
TIME_MAPPING_SCHEMA_VERSION = "7A.time-map.1"
CAPTION_PLAN_SCHEMA_VERSION = "7A.caption-plan.1"
COMPOSITION_PLAN_SCHEMA_VERSION = "7A.composition-plan.1"
MOTION_PLAN_SCHEMA_VERSION = "7A.motion-plan.1"
SOURCE_BROLL_PLAN_SCHEMA_VERSION = "7A.source-broll-plan.1"
COMPILED_RENDER_PLAN_SCHEMA_VERSION = "7A.compiled-render-plan.1"
PARITY_SIGNATURE_SCHEMA_VERSION = "7A.parity.1"

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


def source_output_map_from_legacy_timeline(timeline: VideoTimeline) -> SourceOutputTimeMap:
    """Adapt the current VideoTimeline's persisted clip decisions to 30 fps."""

    segments: list[EditMapSegment] = []
    for clip in timeline.clips:
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
            output=OutputInterval.from_seconds(
                clip.timeline_start_seconds,
                clip.timeline_end_seconds,
            ),
        ))
    if not segments:
        raise ValueError("LEGACY_TIMELINE_HAS_NO_SOURCE_MAPPING")
    return SourceOutputTimeMap(segments=tuple(segments))


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


class CaptionCuePlan(FrozenContract):
    cue_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    resolved_lines: tuple[str, ...] = Field(min_length=1, max_length=2)
    lane: Literal["lower", "lower_mid", "upper_mid", "upper"]
    typography_token_id: str = Field(pattern=ID_PATTERN)
    semantic_class: SemanticClass | None = None
    evidence_refs: tuple[str, ...] = ()
    primitive_id: Literal["legacy_passthrough", "static", "fade", "karaoke"] = "static"
    easing_id: Literal["none", "linear", "ease_in_out"] = "none"
    normalized_bounds: NormalizedRect | None = None


class CaptionPlan(FrozenContract):
    schema_version: Literal["7A.caption-plan.1"] = "7A.caption-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    cues: tuple[CaptionCuePlan, ...] = ()
    backend_id: Literal["none", "libass", "legacy_passthrough"] = "none"
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_cues(self) -> "CaptionPlan":
        if any(right.output.start_frame < left.output.start_frame for left, right in zip(self.cues, self.cues[1:])):
            raise ValueError("caption cues must be ordered")
        return self


class CompositionSegmentPlan(FrozenContract):
    segment_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    layout: LayoutFamily
    target: AttentionTarget = AttentionTarget.STABLE_SOURCE
    target_ref: str | None = Field(default=None, pattern=ID_PATTERN)
    crop: NormalizedRect
    protected_regions: tuple[NormalizedRect, ...] = ()
    easing_id: Literal["none", "linear", "ease_in_out"] = "none"
    evidence_refs: tuple[str, ...] = ()
    fallback: Literal["none", "stable_source", "fit_background"] = "none"


class CompositionPlan(FrozenContract):
    schema_version: Literal["7A.composition-plan.1"] = "7A.composition-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    segments: tuple[CompositionSegmentPlan, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_segments(self) -> "CompositionPlan":
        if any(
            right.output.start_frame < left.output.end_frame
            for left, right in zip(self.segments, self.segments[1:])
        ):
            raise ValueError("composition segments must be ordered without overlap")
        return self


class MotionEventPlan(FrozenContract):
    event_id: str = Field(pattern=ID_PATTERN)
    output: OutputInterval
    purpose: MotionPurpose
    domain: MotionDomain
    primitive_id: Literal["static", "fade", "dissolve", "crop_translate", "punch_in"]
    easing_id: Literal["none", "linear", "ease_in_out"] = "none"
    intensity: Intensity
    evidence_refs: tuple[str, ...]
    fallback_primitive_id: Literal["static", "fade"] = "static"


class MotionPlan(FrozenContract):
    schema_version: Literal["7A.motion-plan.1"] = "7A.motion-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    events: tuple[MotionEventPlan, ...] = ()
    reduced_motion: bool = False
    diagnostics: tuple[str, ...] = ()


class SourceBRollSegmentPlan(FrozenContract):
    segment_id: str = Field(pattern=ID_PATTERN)
    destination: OutputInterval
    source_cutaway: SourceInterval
    story_unit_id: str = Field(pattern=ID_PATTERN)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    retain_source_audio: Literal[False] = False
    transition: Literal["cut", "short_dissolve"] = "cut"
    fallback: Literal["a_roll"] = "a_roll"


class SourceBRollPlan(FrozenContract):
    schema_version: Literal["7A.source-broll-plan.1"] = "7A.source-broll-plan.1"
    intent_id: str = Field(pattern=ID_PATTERN)
    segments: tuple[SourceBRollSegmentPlan, ...] = ()
    default_visual: Literal["a_roll"] = "a_roll"
    diagnostics: tuple[str, ...] = ()


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
    node_kind: Literal["base_visual", "caption_overlay", "composite", "quality_check"]
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

    schema_version: Literal["7A.compiled-render-plan.1"] = "7A.compiled-render-plan.1"
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
            }
            for cue in caption["cues"]
        ],
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


def _render_graph_nodes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_key = canonical_hash({
        "production_plan": payload["production_plan"],
        "mapping": payload["source_output_mapping"],
        "canvas": payload["canvas"],
        "composition": payload["composition_plan"],
        "broll": payload["source_broll_plan"],
        "backends": payload["backends"],
    })
    caption_key = canonical_hash({
        "captions": payload["caption_plan"],
        "motion": payload["motion_plan"],
        "assets": payload["assets"],
        "canvas": payload["canvas"],
        "backends": payload["backends"],
    })
    composite_key = canonical_hash({"base": base_key, "caption": caption_key})
    quality_key = canonical_hash({
        "composite": composite_key,
        "constraints": payload["expected_quality_constraints"],
    })
    return [
        {
            "node_id": "base-visual",
            "node_kind": "base_visual",
            "dependency_ids": [],
            "cache_key": base_key,
            "backend_domain": "base_video",
        },
        {
            "node_id": "caption-overlay",
            "node_kind": "caption_overlay",
            "dependency_ids": [],
            "cache_key": caption_key,
            "backend_domain": "caption",
        },
        {
            "node_id": "composite",
            "node_kind": "composite",
            "dependency_ids": ["base-visual", "caption-overlay"],
            "cache_key": composite_key,
            "backend_domain": "base_video",
        },
        {
            "node_id": "quality-check",
            "node_kind": "quality_check",
            "dependency_ids": ["composite"],
            "cache_key": quality_key,
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
            "no_raw_commands", "identity_match", "source_ranges_bounded", "preview_final_parity",
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
            continue
        if not any(
            target.target == segment.target
            and target.output.contains(segment.output)
            and set(segment.evidence_refs).issubset(target.evidence_refs)
            and bool(segment.evidence_refs)
            for target in intent.composition_targets
        ):
            raise ValueError("COMPOSITION_PLAN_EVIDENCE_MISMATCH")

    for event in motion.events:
        if not any(
            request.purpose == event.purpose
            and request.domain == event.domain
            and request.output.contains(event.output)
            and set(event.evidence_refs).issubset(request.evidence_refs)
            and bool(event.evidence_refs)
            for request in intent.motion_events
        ):
            raise ValueError("MOTION_PLAN_EVIDENCE_MISMATCH")

    if broll.segments and not intent.policy.source_broll_enabled:
        raise ValueError("SOURCE_BROLL_DISABLED")
    for segment in broll.segments:
        if not any(
            request.output == segment.destination
            and request.source_cutaway == segment.source_cutaway
            and request.story_unit_id == segment.story_unit_id
            and request.story_unit_evidence_ref in segment.evidence_refs
            and set(segment.evidence_refs).issubset(
                {*request.evidence_refs, *request.source_cutaway_evidence_refs}
            )
            for request in intent.source_broll
        ):
            raise ValueError("SOURCE_BROLL_PLAN_EVIDENCE_MISMATCH")


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
