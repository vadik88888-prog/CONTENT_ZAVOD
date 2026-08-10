from __future__ import annotations

"""Phase 7E evidence-linked cutaways from the current source only.

The planner never retrieves or synthesizes media.  It accepts persisted scene
evidence, proves every semantic and safety predicate, and otherwise leaves the
current A-roll visual/composition untouched.  A-roll audio is immutable here.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from app.creative_contracts import (
    AttentionTarget,
    BeatRole,
    CompositionPlan,
    CreativeIntent,
    EvidenceItem,
    FrozenContract,
    ID_PATTERN,
    SOURCE_BROLL_PLAN_SCHEMA_VERSION,
    SourceBRollPlan,
    SourceBRollQualityFinding,
    SourceBRollQualityMetrics,
    SourceBRollQualityProvenance,
    SourceBRollQualityReport,
    SourceBRollSafetyChecks,
    SourceBRollSegmentPlan,
    SourceBRollSemanticKind,
    SourceInterval,
    NormalizedRect,
    ResolvedBeat,
    ResolvedSourceBRoll,
)


SOURCE_BROLL_PLANNER_VERSION = "7E.source-broll-planner.1"

Verification = Literal["verified", "uncertain", "contradicted"]
Chronology = Literal["safe", "uncertain", "contradicted"]
Causality = Literal["not_claimed", "supported", "uncertain", "contradicted"]
PayoffSignal = Literal["none", "setup", "reveal", "result", "resolution", "unknown"]


class SourceSceneEvidence(FrozenContract):
    """A source-scoped semantic scene observation, never a render command."""

    scene_id: str = Field(pattern=ID_PATTERN)
    source_id: str = Field(pattern=ID_PATTERN)
    source: SourceInterval
    semantic_kinds: tuple[SourceBRollSemanticKind, ...] = Field(min_length=1)
    story_unit_ids: tuple[str, ...] = Field(min_length=1)
    beat_roles: tuple[BeatRole, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    source_crop: NormalizedRect | None = None
    source_target: AttentionTarget | None = None
    confidence: float = Field(ge=0, le=1)
    identity_status: Verification = "uncertain"
    attribution_status: Verification = "uncertain"
    chronology_status: Chronology = "uncertain"
    causality_status: Causality = "uncertain"
    rights_status: Verification = "uncertain"
    payoff_signal: PayoffSignal = "unknown"
    critical_screen_text: bool = False
    screen_text_readable: bool = False
    visible_speech_requires_sync: bool = False
    provenance: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_links(self) -> "SourceSceneEvidence":
        for values in (self.semantic_kinds, self.story_unit_ids, self.beat_roles, self.evidence_refs):
            if len(values) != len(set(values)):
                raise ValueError("source scene evidence links must be unique")
        return self


@dataclass(frozen=True, slots=True)
class SourceBRollPlannerConfig:
    minimum_relevance_confidence: float = 0.72
    dissolve_minimum_frames: int = 24

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_relevance_confidence <= 1:
            raise ValueError("source B-roll confidence threshold must be normalized")
        if self.dissolve_minimum_frames < 1:
            raise ValueError("source B-roll dissolve duration must be positive")


@dataclass(frozen=True, slots=True)
class _Rejection:
    code: str
    decision_id: str
    scene_id: str | None
    message: str


class SourceBRollPlanner:
    """Select only cutaways whose relevance and forbidden checks are proven."""

    def __init__(self, config: SourceBRollPlannerConfig | None = None) -> None:
        self.config = config or SourceBRollPlannerConfig()

    def plan(
        self,
        intent: CreativeIntent,
        scenes: Iterable[SourceSceneEvidence],
        composition_plan: CompositionPlan,
    ) -> SourceBRollPlan:
        if composition_plan.intent_id != intent.intent_id:
            raise ValueError("SOURCE_BROLL_COMPOSITION_INTENT_MISMATCH")
        ordered_scenes = tuple(sorted(scenes, key=lambda item: (item.source.start_tick, item.scene_id)))
        manifest = {item.evidence_ref: item for item in intent.evidence_manifest}
        selected: list[SourceBRollSegmentPlan] = []
        rejections: list[_Rejection] = []
        used_ranges: list[SourceInterval] = []

        for request in sorted(intent.source_broll, key=lambda item: (item.output.start_frame, item.decision_id)):
            beat = _linked_beat(intent.beats, request)
            candidates = [
                scene for scene in ordered_scenes
                if scene.source.contains(request.source_cutaway)
                and request.story_unit_id in scene.story_unit_ids
                and request.semantic_kind in scene.semantic_kinds
            ]
            candidates.sort(key=lambda item: (-item.confidence, item.source.start_tick, item.scene_id))
            scene: SourceSceneEvidence | None = None
            rejection: _Rejection | None = None
            for candidate in candidates or (None,):
                candidate_rejection = self._reject(
                    intent, request, beat, candidate, manifest, selected, used_ranges,
                )
                if candidate_rejection is None:
                    scene = candidate
                    break
                if rejection is None:
                    rejection = candidate_rejection
            if scene is None:
                assert rejection is not None
                rejections.append(rejection)
                continue
            assert beat is not None and scene is not None
            evidence_refs = tuple(dict.fromkeys((
                *request.evidence_refs,
                *request.source_cutaway_evidence_refs,
                request.story_unit_evidence_ref,
                *scene.evidence_refs,
                *beat.evidence_refs,
            )))
            duration = request.output.end_frame - request.output.start_frame
            composition_ids = tuple(
                item.segment_id for item in composition_plan.segments
                if item.output.overlaps(request.output)
            )
            selected.append(SourceBRollSegmentPlan(
                segment_id=f"source-broll-{len(selected) + 1:03d}",
                decision_id=request.decision_id,
                destination=request.output,
                source_cutaway=request.source_cutaway,
                source_crop=scene.source_crop,
                source_target=scene.source_target,
                source_scene_id=scene.scene_id,
                story_unit_id=request.story_unit_id,
                beat_role=beat.role,
                semantic_kind=request.semantic_kind,
                evidence_refs=evidence_refs,
                scene_evidence_refs=scene.evidence_refs,
                relevance_confidence=round(min(request.confidence, scene.confidence, beat.confidence), 7),
                retain_source_audio=False,
                audio_timeline="a_roll_master",
                transition="short_dissolve" if duration >= self.config.dissolve_minimum_frames else "cut",
                fallback="a_roll",
                fallback_composition_segment_ids=composition_ids,
                safety_checks=SourceBRollSafetyChecks(),
                provenance=scene.provenance,
            ))
            used_ranges.append(request.source_cutaway)

        report = _quality_report(intent, composition_plan, selected, rejections)
        return SourceBRollPlan(
            schema_version=SOURCE_BROLL_PLAN_SCHEMA_VERSION,
            intent_id=intent.intent_id,
            segments=tuple(selected),
            composition_plan_sha256=composition_plan.canonical_hash(),
            quality_report=report,
            diagnostics=tuple(
                f"{item.code}:{item.decision_id}:{item.scene_id or 'none'}" for item in rejections
            ),
        )

    def _reject(
        self,
        intent: CreativeIntent,
        request: ResolvedSourceBRoll,
        beat: ResolvedBeat | None,
        scene: SourceSceneEvidence | None,
        manifest: Mapping[str, EvidenceItem],
        selected: Sequence[SourceBRollSegmentPlan],
        used_ranges: Sequence[SourceInterval],
    ) -> _Rejection | None:
        def reject(code: str, message: str) -> _Rejection:
            return _Rejection(code, request.decision_id, scene.scene_id if scene else None, message)

        if not intent.policy.source_broll_enabled:
            return reject("SOURCE_BROLL_DISABLED", "Source B-roll is disabled; A-roll remains active.")
        if beat is None:
            return reject("SOURCE_BROLL_BEAT_UNLINKED", "No evidence-backed beat contains the destination.")
        if scene is None:
            return reject("SOURCE_BROLL_RELEVANCE_UNPROVEN", "No source scene proves StoryUnit and semantic relevance.")
        if scene.source_id != intent.production_plan.identity.source_id:
            return reject("SOURCE_BROLL_SOURCE_IDENTITY_MISMATCH", "The scene belongs to another source.")
        if beat.role not in scene.beat_roles:
            return reject("SOURCE_BROLL_BEAT_UNLINKED", "The scene is not linked to the current beat role.")
        refs = {*request.evidence_refs, *request.source_cutaway_evidence_refs, *scene.evidence_refs}
        if any(ref not in manifest for ref in refs):
            return reject("SOURCE_BROLL_EVIDENCE_MISSING", "The cutaway references evidence outside CreativeIntent.")
        if not set(scene.evidence_refs).issubset(request.source_cutaway_evidence_refs):
            return reject(
                "SOURCE_BROLL_SCENE_EVIDENCE_UNLINKED",
                "Scene evidence is not linked by the bounded B-roll proposal.",
            )
        if not any(
            manifest[ref].evidence_kind in {"scene", "visual"}
            and manifest[ref].source.overlaps(request.source_cutaway)
            for ref in scene.evidence_refs
        ):
            return reject("SOURCE_BROLL_RELEVANCE_UNPROVEN", "No scene/visual evidence overlaps the cutaway.")
        relevance = min(request.confidence, scene.confidence, beat.confidence)
        if relevance < self.config.minimum_relevance_confidence:
            return reject("SOURCE_BROLL_CONFIDENCE_LOW", "Evidence confidence is below the safe relevance threshold.")
        if scene.identity_status != "verified":
            return reject("SOURCE_BROLL_IDENTITY_UNCERTAIN", "Scene identity is not verified.")
        if scene.attribution_status != "verified":
            return reject("SOURCE_BROLL_ATTRIBUTION_UNCERTAIN", "Scene attribution is not verified.")
        if scene.chronology_status != "safe":
            return reject("SOURCE_BROLL_CHRONOLOGY_UNSAFE", "Scene chronology is uncertain or contradicted.")
        if scene.causality_status not in {"not_claimed", "supported"}:
            return reject("SOURCE_BROLL_CAUSALITY_UNSAFE", "The insertion could imply unsupported causality.")
        if scene.rights_status != "verified":
            return reject("SOURCE_BROLL_RIGHTS_UNCERTAIN", "Source usage rights are not verified.")
        if scene.visible_speech_requires_sync:
            return reject("SOURCE_BROLL_LIP_SYNC_REQUIRED", "Visible speech would conflict with A-roll master audio.")
        if request.semantic_kind == SourceBRollSemanticKind.SCREEN and scene.critical_screen_text and not scene.screen_text_readable:
            return reject("SOURCE_BROLL_SCREEN_TEXT_UNREADABLE", "Critical screen text is not reliably readable.")
        if _premature_payoff(intent.beats, beat, scene):
            return reject("SOURCE_BROLL_PREMATURE_REVEAL", "The scene reveals payoff before the payoff beat.")
        if any(item.overlaps(request.source_cutaway) for item in used_ranges):
            return reject("SOURCE_BROLL_FILLER_RANGE_REPEATED", "The same filler range was already selected.")
        if any(item.destination.overlaps(request.output) for item in selected):
            return reject("SOURCE_BROLL_DESTINATION_OVERLAP", "Another cutaway already owns the destination.")
        return None


def build_source_broll_plan(
    intent: CreativeIntent,
    scenes: Iterable[SourceSceneEvidence],
    composition_plan: CompositionPlan,
    *,
    config: SourceBRollPlannerConfig | None = None,
) -> SourceBRollPlan:
    return SourceBRollPlanner(config).plan(intent, scenes, composition_plan)


def _linked_beat(beats: Sequence[ResolvedBeat], request: ResolvedSourceBRoll) -> ResolvedBeat | None:
    matches = [beat for beat in beats if beat.output.contains(request.output)]
    return max(matches, key=lambda item: (item.importance, item.confidence), default=None)


def _premature_payoff(
    beats: Sequence[ResolvedBeat], current: ResolvedBeat, scene: SourceSceneEvidence,
) -> bool:
    if scene.payoff_signal not in {"reveal", "result", "resolution"}:
        return False
    if current.role == BeatRole.PAYOFF:
        return False
    payoff_starts = [item.output.start_frame for item in beats if item.role == BeatRole.PAYOFF]
    return not payoff_starts or current.output.start_frame < min(payoff_starts)


def _quality_report(
    intent: CreativeIntent,
    composition: CompositionPlan,
    selected: Sequence[SourceBRollSegmentPlan],
    rejections: Sequence[_Rejection],
) -> SourceBRollQualityReport:
    findings = tuple(SourceBRollQualityFinding(
        code=item.code,
        severity="warning",
        decision_id=item.decision_id,
        scene_id=item.scene_id,
        measured_value="rejected_to_a_roll",
        threshold="all semantic and safety predicates verified",
        message=item.message,
    ) for item in rejections)
    codes = Counter(item.code for item in rejections)
    kinds = Counter(item.semantic_kind for item in selected)
    return SourceBRollQualityReport(
        status="PASS_WITH_WARNINGS" if findings else "PASS",
        findings=findings,
        metrics=SourceBRollQualityMetrics(
            proposal_count=len(intent.source_broll),
            selected_count=len(selected),
            rejected_count=len(rejections),
            a_roll_fallback_count=len(rejections),
            repeated_range_count=codes["SOURCE_BROLL_FILLER_RANGE_REPEATED"],
            premature_reveal_count=codes["SOURCE_BROLL_PREMATURE_REVEAL"],
            attribution_violation_count=codes["SOURCE_BROLL_ATTRIBUTION_UNCERTAIN"],
            causality_violation_count=codes["SOURCE_BROLL_CAUSALITY_UNSAFE"],
            chronology_violation_count=codes["SOURCE_BROLL_CHRONOLOGY_UNSAFE"],
            selected_duration_frames=sum(item.destination.end_frame - item.destination.start_frame for item in selected),
            semantic_kind_counts=tuple(sorted(kinds.items(), key=lambda item: item[0].value)),
        ),
        provenance=SourceBRollQualityProvenance(
            producer="source_broll_planner",
            planner_version=SOURCE_BROLL_PLANNER_VERSION,
            intent_id=intent.intent_id,
            evidence_fingerprint=intent.evidence_fingerprint,
            composition_plan_sha256=composition.canonical_hash(),
        ),
    )
