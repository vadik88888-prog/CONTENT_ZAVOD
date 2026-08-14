from __future__ import annotations

"""Candidate-owned persistence and safe revision for native creative plans."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from app.composition_planning import TargetObservation
from app.config import AppConfig
from app.creative_contracts import (
    CaptionFeasibilityDecision,
    CaptionPlan,
    CompiledRenderPlan,
    CreativeIntent,
    CreativePolicy,
    FrozenContract,
    HASH_PATTERN,
    ImmutableProductionPlanLink,
    OUTPUT_FPS,
    OutputInterval,
    SourceInterval,
    canonical_hash,
)
from app.creative_policy import CREATIVE_POLICY_VERSION, creative_preset_definition
from app.production_models import ProductionPlan
from app.source_broll_planning import SourceSceneEvidence
from app.utils import read_json, stable_file_hash, utc_now, write_json


CREATIVE_HANDOFF_SCHEMA_VERSION = "7G.3.creative-handoff.1"
CREATIVE_EXECUTION_SCHEMA_VERSION = "7G.3.creative-execution.1"
CAPTION_FEASIBILITY_ARTIFACT_SCHEMA_VERSION = "7J.2A-3.caption-feasibility-artifact.1"
CAPTION_FEASIBILITY_REFERENCE_SCHEMA_VERSION = "7J.2A-3.caption-feasibility-reference.1"
CAPTION_FEASIBILITY_ARTIFACT_RELATIVE_PATH = Path(
    "caption-feasibility/caption-feasibility-decision.json"
)

ExecutionStatus = Literal["native_rich", "native_fallback", "legacy"]


class CreativeArtifactError(ValueError):
    pass


class CandidateCaptionFeasibilitySpan(FrozenContract):
    evidence_id: str = Field(min_length=1)
    cue_ids: tuple[str, ...] = Field(min_length=1)
    word_ids: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    source: SourceInterval
    immutable_word_output: OutputInterval
    mapping_segment_ids: tuple[str, ...] = Field(min_length=1)
    character_count: int = Field(gt=0)
    available_frames: int = Field(gt=0)
    required_frames: int = Field(gt=0)
    measured_cps: float = Field(gt=0)
    hard_cps_ceiling: float = Field(gt=0)


class CandidateCaptionFeasibilityArtifact(FrozenContract):
    """Candidate-owned durable copy of the validated pre-render decision."""

    schema_version: Literal["7J.2A-3.caption-feasibility-artifact.1"] = (
        CAPTION_FEASIBILITY_ARTIFACT_SCHEMA_VERSION
    )
    production_plan: ImmutableProductionPlanLink
    candidate_id: str = Field(min_length=1)
    caption_plan_schema_version: str = Field(min_length=1)
    caption_intent_id: str = Field(min_length=1)
    caption_plan_fingerprint: str = Field(pattern=HASH_PATTERN)
    compiled_render_plan_hash: str = Field(pattern=HASH_PATTERN)
    compiled_caption_feasibility_sha256: str = Field(pattern=HASH_PATTERN)
    decision_id: str = Field(min_length=1)
    status: Literal["FEASIBLE", "INFEASIBLE", "NOT_APPLICABLE"]
    blocker_code: Literal["CAPTION_CPS_INFEASIBLE"] | None = None
    decision: CaptionFeasibilityDecision
    evidence_spans: tuple[CandidateCaptionFeasibilitySpan, ...] = ()
    decision_fingerprint: str = Field(pattern=HASH_PATTERN)
    artifact_fingerprint: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _valid_identity_and_fingerprints(self) -> "CandidateCaptionFeasibilityArtifact":
        if self.candidate_id != self.production_plan.identity.candidate_id:
            raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_CANDIDATE_MISMATCH")
        if self.decision_id != self.decision.decision_id or self.status != self.decision.status:
            raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_DECISION_MISMATCH")
        expected_blocker = "CAPTION_CPS_INFEASIBLE" if self.decision.status == "INFEASIBLE" else None
        if self.blocker_code != expected_blocker or self.decision.reason_code != (
            expected_blocker or (
                "CAPTION_TEMPORALLY_FEASIBLE"
                if self.decision.status == "FEASIBLE" else "NO_MAPPED_WORDS"
            )
        ):
            raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_STATUS_MISMATCH")
        decision_fingerprint = self.decision.canonical_hash()
        if (
            self.decision_fingerprint != decision_fingerprint
            or self.compiled_caption_feasibility_sha256 != decision_fingerprint
        ):
            raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_DECISION_HASH_MISMATCH")
        evidence = {item.evidence_id: item for item in self.decision.evidence}
        spans = {item.evidence_id: item for item in self.evidence_spans}
        if len(spans) != len(self.evidence_spans) or set(spans) != set(evidence):
            raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_EVIDENCE_MISMATCH")
        for evidence_id, item in evidence.items():
            span = spans[evidence_id]
            if (
                span.word_ids != item.word_ids
                or span.text != item.text
                or span.source != item.source
                or span.immutable_word_output != item.immutable_word_output
                or span.mapping_segment_ids != item.mapping_segment_ids
                or span.character_count != item.character_count
                or span.available_frames != (
                    item.immutable_word_output.end_frame
                    - item.immutable_word_output.start_frame
                )
                or span.required_frames != item.required_frames
                or span.measured_cps != round(
                    item.character_count / (span.available_frames / OUTPUT_FPS), 6
                )
                or span.hard_cps_ceiling != item.hard_cps_ceiling
            ):
                raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_SPAN_MISMATCH")
        payload = self.model_dump(mode="json")
        payload.pop("artifact_fingerprint", None)
        if canonical_hash(payload) != self.artifact_fingerprint:
            raise ValueError("CAPTION_FEASIBILITY_ARTIFACT_HASH_MISMATCH")
        return self


class CandidateCaptionFeasibilityReference(FrozenContract):
    schema_version: Literal["7J.2A-3.caption-feasibility-reference.1"] = (
        CAPTION_FEASIBILITY_REFERENCE_SCHEMA_VERSION
    )
    candidate_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    status: Literal["FEASIBLE", "INFEASIBLE", "NOT_APPLICABLE"]
    blocker_code: Literal["CAPTION_CPS_INFEASIBLE"] | None = None
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=HASH_PATTERN)
    artifact_fingerprint: str = Field(pattern=HASH_PATTERN)
    decision_fingerprint: str = Field(pattern=HASH_PATTERN)
    compiled_render_plan_hash: str = Field(pattern=HASH_PATTERN)


class CandidateCreativeHandoff(FrozenContract):
    schema_version: Literal["7G.3.creative-handoff.1"] = "7G.3.creative-handoff.1"
    production_plan: ImmutableProductionPlanLink
    candidate_id: str = Field(min_length=1)
    creative_intent_hash: str = Field(pattern=HASH_PATTERN)
    evidence_fingerprint: str = Field(pattern=HASH_PATTERN)
    mapping_fingerprint: str = Field(pattern=HASH_PATTERN)
    target_observations: tuple[TargetObservation, ...] = ()
    source_scenes: tuple[SourceSceneEvidence, ...] = ()
    artifact_fingerprint: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def _valid_fingerprint(self) -> "CandidateCreativeHandoff":
        payload = self.model_dump(mode="json")
        payload.pop("artifact_fingerprint", None)
        if canonical_hash(payload) != self.artifact_fingerprint:
            raise ValueError("CREATIVE_HANDOFF_HASH_MISMATCH")
        if self.candidate_id != self.production_plan.identity.candidate_id:
            raise ValueError("CREATIVE_HANDOFF_CANDIDATE_MISMATCH")
        return self


class CandidateCreativeExecution(FrozenContract):
    schema_version: Literal["7G.3.creative-execution.1"] = "7G.3.creative-execution.1"
    production_plan: ImmutableProductionPlanLink
    candidate_id: str = Field(min_length=1)
    creative_intent_id: str = Field(min_length=1)
    creative_intent_hash: str = Field(pattern=HASH_PATTERN)
    creative_intent_revision: int = Field(ge=1)
    compiled_render_plan_hash: str = Field(pattern=HASH_PATTERN)
    parent_production_plan_hash: str = Field(pattern=HASH_PATTERN)
    parent_compiled_plan_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    parent_creative_intent_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    evidence_fingerprint: str = Field(pattern=HASH_PATTERN)
    mapping_fingerprint: str = Field(pattern=HASH_PATTERN)
    parity_signature: str = Field(pattern=HASH_PATTERN)
    execution_status: ExecutionStatus
    reason_codes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    style_revision: bool = False
    created_at: str

    @model_validator(mode="after")
    def _valid_status(self) -> "CandidateCreativeExecution":
        if self.candidate_id != self.production_plan.identity.candidate_id:
            raise ValueError("CREATIVE_EXECUTION_CANDIDATE_MISMATCH")
        if self.parent_production_plan_hash != self.production_plan.plan_fingerprint:
            raise ValueError("CREATIVE_EXECUTION_PARENT_PLAN_MISMATCH")
        if self.execution_status == "native_fallback" and not self.reason_codes:
            raise ValueError("native_fallback requires reason codes")
        if self.style_revision and (
            self.parent_compiled_plan_hash is None or self.parent_creative_intent_hash is None
        ):
            raise ValueError("creative style revision requires parent hashes")
        return self


def build_creative_handoff(
    intent: CreativeIntent,
    *,
    target_observations: tuple[TargetObservation, ...] = (),
    source_scenes: tuple[SourceSceneEvidence, ...] = (),
) -> CandidateCreativeHandoff:
    payload: dict[str, Any] = {
        "schema_version": CREATIVE_HANDOFF_SCHEMA_VERSION,
        "production_plan": intent.production_plan.model_dump(mode="json"),
        "candidate_id": intent.production_plan.identity.candidate_id,
        "creative_intent_hash": intent.canonical_hash(),
        "evidence_fingerprint": intent.evidence_fingerprint,
        "mapping_fingerprint": intent.source_output_mapping.fingerprint,
        "target_observations": [item.model_dump(mode="json") for item in target_observations],
        "source_scenes": [item.model_dump(mode="json") for item in source_scenes],
    }
    payload["artifact_fingerprint"] = canonical_hash(payload)
    return CandidateCreativeHandoff.model_validate(payload)


def build_creative_execution(
    intent: CreativeIntent,
    compiled_plan: CompiledRenderPlan,
    *,
    execution_status: ExecutionStatus,
    reason_codes: tuple[str, ...] = (),
    diagnostics: tuple[str, ...] = (),
    parent_compiled_plan_hash: str | None = None,
    parent_creative_intent_hash: str | None = None,
    style_revision: bool = False,
) -> CandidateCreativeExecution:
    return CandidateCreativeExecution(
        production_plan=intent.production_plan,
        candidate_id=intent.production_plan.identity.candidate_id,
        creative_intent_id=intent.intent_id,
        creative_intent_hash=intent.canonical_hash(),
        creative_intent_revision=intent.revision,
        compiled_render_plan_hash=compiled_plan.plan_hash,
        parent_production_plan_hash=intent.production_plan.plan_fingerprint,
        parent_compiled_plan_hash=parent_compiled_plan_hash,
        parent_creative_intent_hash=parent_creative_intent_hash,
        evidence_fingerprint=intent.evidence_fingerprint,
        mapping_fingerprint=intent.source_output_mapping.fingerprint,
        parity_signature=compiled_plan.parity_signature,
        execution_status=execution_status,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        diagnostics=diagnostics,
        style_revision=style_revision,
        created_at=utc_now(),
    )


def build_candidate_caption_feasibility_artifact(
    compiled_plan: CompiledRenderPlan,
) -> CandidateCaptionFeasibilityArtifact:
    caption_plan: CaptionPlan = compiled_plan.caption_plan
    decision = caption_plan.feasibility_decision
    if decision is None:
        raise CreativeArtifactError("CAPTION_FEASIBILITY_DECISION_MISSING")
    compiled_fingerprint = compiled_plan.input_fingerprints.caption_feasibility_sha256
    decision_fingerprint = decision.canonical_hash()
    if compiled_fingerprint != decision_fingerprint:
        raise CreativeArtifactError("CAPTION_FEASIBILITY_COMPILED_FINGERPRINT_MISMATCH")

    cue_ids_by_word: dict[str, list[str]] = {}
    for cue in caption_plan.cues:
        for word in cue.words:
            cue_ids_by_word.setdefault(word.word_id, []).append(cue.cue_id)
    spans: list[dict[str, Any]] = []
    for evidence in decision.evidence:
        cue_ids = tuple(dict.fromkeys(
            cue_id
            for word_id in evidence.word_ids
            for cue_id in cue_ids_by_word.get(word_id, ())
        ))
        if not cue_ids:
            raise CreativeArtifactError("CAPTION_FEASIBILITY_EVIDENCE_CUE_MISSING")
        spans.append({
            "evidence_id": evidence.evidence_id,
            "cue_ids": cue_ids,
            "word_ids": evidence.word_ids,
            "text": evidence.text,
            "source": evidence.source.model_dump(mode="json"),
            "immutable_word_output": evidence.immutable_word_output.model_dump(mode="json"),
            "mapping_segment_ids": evidence.mapping_segment_ids,
            "character_count": evidence.character_count,
            "available_frames": (
                evidence.immutable_word_output.end_frame
                - evidence.immutable_word_output.start_frame
            ),
            "required_frames": evidence.required_frames,
            "measured_cps": round(
                evidence.character_count
                / (
                    (
                        evidence.immutable_word_output.end_frame
                        - evidence.immutable_word_output.start_frame
                    ) / OUTPUT_FPS
                ),
                6,
            ),
            "hard_cps_ceiling": evidence.hard_cps_ceiling,
        })

    payload: dict[str, Any] = {
        "schema_version": CAPTION_FEASIBILITY_ARTIFACT_SCHEMA_VERSION,
        "production_plan": compiled_plan.production_plan.model_dump(mode="json"),
        "candidate_id": compiled_plan.production_plan.identity.candidate_id,
        "caption_plan_schema_version": caption_plan.schema_version,
        "caption_intent_id": caption_plan.intent_id,
        "caption_plan_fingerprint": caption_plan.canonical_hash(),
        "compiled_render_plan_hash": compiled_plan.plan_hash,
        "compiled_caption_feasibility_sha256": compiled_fingerprint,
        "decision_id": decision.decision_id,
        "status": decision.status,
        "blocker_code": (
            "CAPTION_CPS_INFEASIBLE" if decision.status == "INFEASIBLE" else None
        ),
        "decision": decision.model_dump(mode="json"),
        "evidence_spans": spans,
        "decision_fingerprint": decision_fingerprint,
    }
    payload["artifact_fingerprint"] = canonical_hash(payload)
    return CandidateCaptionFeasibilityArtifact.model_validate(payload)


def persist_candidate_caption_feasibility(
    root: Path,
    *,
    compiled_plan: CompiledRenderPlan,
) -> CandidateCaptionFeasibilityReference:
    """Persist and revalidate caption feasibility before the render gate runs."""

    artifact = build_candidate_caption_feasibility_artifact(compiled_plan)
    path = (root / CAPTION_FEASIBILITY_ARTIFACT_RELATIVE_PATH).resolve()
    write_json(path, artifact.model_dump(mode="json"))
    try:
        persisted = CandidateCaptionFeasibilityArtifact.model_validate(read_json(path, None))
    except Exception as error:
        raise CreativeArtifactError(
            f"CAPTION_FEASIBILITY_ARTIFACT_INVALID: {error}"
        ) from error
    if persisted != artifact:
        raise CreativeArtifactError("CAPTION_FEASIBILITY_ARTIFACT_WRITE_MISMATCH")
    reference = CandidateCaptionFeasibilityReference(
        candidate_id=artifact.candidate_id,
        decision_id=artifact.decision_id,
        status=artifact.status,
        blocker_code=artifact.blocker_code,
        artifact_path=str(path),
        artifact_sha256=stable_file_hash(path),
        artifact_fingerprint=artifact.artifact_fingerprint,
        decision_fingerprint=artifact.decision_fingerprint,
        compiled_render_plan_hash=artifact.compiled_render_plan_hash,
    )
    # Count the artifact as valid only after a full contract and SHA check.
    load_candidate_caption_feasibility(root, reference=reference, compiled_plan=compiled_plan)
    return reference


def load_candidate_caption_feasibility(
    root: Path,
    *,
    reference: CandidateCaptionFeasibilityReference,
    compiled_plan: CompiledRenderPlan,
) -> CandidateCaptionFeasibilityArtifact:
    expected_path = (root / CAPTION_FEASIBILITY_ARTIFACT_RELATIVE_PATH).resolve()
    if Path(reference.artifact_path).resolve() != expected_path:
        raise CreativeArtifactError("CAPTION_FEASIBILITY_ARTIFACT_PATH_MISMATCH")
    try:
        artifact = CandidateCaptionFeasibilityArtifact.model_validate(read_json(expected_path, None))
    except Exception as error:
        raise CreativeArtifactError(
            f"CAPTION_FEASIBILITY_ARTIFACT_INVALID: {error}"
        ) from error
    if stable_file_hash(expected_path) != reference.artifact_sha256:
        raise CreativeArtifactError("CAPTION_FEASIBILITY_ARTIFACT_SHA_MISMATCH")
    decision = compiled_plan.caption_plan.feasibility_decision
    if (
        artifact.production_plan != compiled_plan.production_plan
        or artifact.compiled_render_plan_hash != compiled_plan.plan_hash
        or artifact.caption_plan_fingerprint != compiled_plan.caption_plan.canonical_hash()
        or decision is None
        or artifact.decision != decision
        or artifact.artifact_fingerprint != reference.artifact_fingerprint
        or artifact.decision_fingerprint != reference.decision_fingerprint
        or artifact.candidate_id != reference.candidate_id
        or artifact.decision_id != reference.decision_id
        or artifact.status != reference.status
        or artifact.blocker_code != reference.blocker_code
    ):
        raise CreativeArtifactError("CAPTION_FEASIBILITY_ARTIFACT_IDENTITY_MISMATCH")
    return artifact


def persist_candidate_creative_identity(
    root: Path,
    *,
    intent: CreativeIntent,
    compiled_plan: CompiledRenderPlan,
    handoff: CandidateCreativeHandoff,
    execution: CandidateCreativeExecution,
) -> None:
    write_json(root / "creative-intent.json", intent.model_dump(mode="json"))
    write_json(root / "creative-handoff.json", handoff.model_dump(mode="json"))
    write_json(root / "creative-execution.json", execution.model_dump(mode="json"))
    write_json(root / "compiled-render-plan.json", compiled_plan.model_dump(mode="json"))


def load_candidate_creative_identity(
    root: Path,
    plan: ProductionPlan,
) -> tuple[CreativeIntent, CompiledRenderPlan, CandidateCreativeHandoff, CandidateCreativeExecution]:
    try:
        intent = CreativeIntent.model_validate(read_json(root / "creative-intent.json", None))
        compiled = CompiledRenderPlan.model_validate(read_json(root / "compiled-render-plan.json", None))
        handoff = CandidateCreativeHandoff.model_validate(read_json(root / "creative-handoff.json", None))
        execution = CandidateCreativeExecution.model_validate(read_json(root / "creative-execution.json", None))
    except Exception as error:
        raise CreativeArtifactError(f"CREATIVE_PARENT_ARTIFACT_INVALID: {error}") from error
    reference = ImmutableProductionPlanLink.from_reference(plan.reference())
    if any(item != reference for item in (
        intent.production_plan, compiled.production_plan, handoff.production_plan, execution.production_plan,
    )):
        raise CreativeArtifactError("CREATIVE_PARENT_IDENTITY_MISMATCH")
    intent_hash = intent.canonical_hash()
    expected = {
        "compiled_intent": compiled.intent_hash,
        "handoff_intent": handoff.creative_intent_hash,
        "execution_intent": execution.creative_intent_hash,
    }
    if any(value != intent_hash for value in expected.values()):
        raise CreativeArtifactError("CREATIVE_PARENT_INTENT_HASH_MISMATCH")
    if execution.compiled_render_plan_hash != compiled.plan_hash:
        raise CreativeArtifactError("CREATIVE_PARENT_COMPILED_PLAN_HASH_MISMATCH")
    if execution.parity_signature != compiled.parity_signature:
        raise CreativeArtifactError("CREATIVE_PARENT_PARITY_SIGNATURE_MISMATCH")
    mapping_fingerprint = intent.source_output_mapping.fingerprint
    if (
        handoff.mapping_fingerprint != mapping_fingerprint
        or execution.mapping_fingerprint != mapping_fingerprint
        or compiled.source_output_mapping != intent.source_output_mapping
        or compiled.input_fingerprints.edit_mapping_sha256 != mapping_fingerprint
    ):
        raise CreativeArtifactError("CREATIVE_PARENT_MAPPING_FINGERPRINT_MISMATCH")
    if (
        handoff.evidence_fingerprint != intent.evidence_fingerprint
        or execution.evidence_fingerprint != intent.evidence_fingerprint
        or compiled.input_fingerprints.evidence_sha256 != intent.evidence_fingerprint
    ):
        raise CreativeArtifactError("CREATIVE_PARENT_EVIDENCE_FINGERPRINT_MISMATCH")
    return intent, compiled, handoff, execution


def creative_policy_for_config(
    parent: CreativePolicy, config: AppConfig, *, parent_selection_mode: str = "explicit",
) -> CreativePolicy:
    parent_mode = parent_selection_mode
    requested_mode = config.product_flow.preset_selection_mode
    if parent_mode == requested_mode and (
        requested_mode == "auto"
        or config.product_flow.subtitle_preset == parent.preset_id
    ):
        # An approved policy is pinned. Automatic recommendations and later
        # policy-table changes may affect new drafts, never a rerender of this
        # identity. Platform is still an explicit render revision input.
        return parent.model_copy(update={"platform": config.product_flow.platform})
    family_policy = creative_preset_definition(
        config.product_flow.subtitle_preset,  # type: ignore[arg-type]
        config.product_flow.preset_version,
    )
    return parent.model_copy(update={
        "preset_id": config.product_flow.subtitle_preset,
        "preset_version": config.product_flow.preset_version,
        "platform": config.product_flow.platform,
        "caption_style_family": family_policy.caption_style_family,
        "caption_density": family_policy.caption_density,
        "intensity": family_policy.intensity_ceiling,
        # Rights and evidence do not become stronger during a style revision.
        "source_broll_enabled": parent.source_broll_enabled,
    })


def revise_creative_intent(parent: CreativeIntent, config: AppConfig) -> CreativeIntent:
    parent_mode = _intent_preset_selection_mode(parent)
    requested_mode = config.product_flow.preset_selection_mode
    policy = _creative_policy_for_intent(parent, config)
    if policy == parent.policy and requested_mode == parent_mode:
        return parent
    revision = parent.revision + 1
    identity = canonical_hash({
        "parent_intent_hash": parent.canonical_hash(),
        "revision": revision,
        "creative_policy_version": CREATIVE_POLICY_VERSION,
        "policy": policy.model_dump(mode="json"),
        "evidence_fingerprint": parent.evidence_fingerprint,
        "mapping_fingerprint": parent.source_output_mapping.fingerprint,
    })
    return parent.model_copy(update={
        "intent_id": f"intent-revision-{identity[:16]}",
        "revision": revision,
        "policy": policy,
        "provenance": (
            *tuple(
                item for item in parent.provenance
                if not item.startswith((
                    "creative_policy:",
                    "preset_selection:", "preset_provenance:",
                    "preset_effective:", "preset_recommendation:",
                ))
            ),
            f"creative_policy:{CREATIVE_POLICY_VERSION}",
            f"preset_selection:{requested_mode}",
            f"preset_provenance:{config.product_flow.preset_provenance}",
            f"preset_effective:{policy.preset_id}",
            f"preset_recommendation:{config.product_flow.recommended_subtitle_preset}",
            f"style_revision:{revision}",
        ),
    })


def creative_policy_changed(intent: CreativeIntent, config: AppConfig) -> bool:
    return (
        _creative_policy_for_intent(intent, config) != intent.policy
        or _intent_preset_selection_mode(intent) != config.product_flow.preset_selection_mode
    )


def _intent_preset_selection_mode(intent: CreativeIntent) -> str:
    marker = next(
        (item for item in reversed(intent.provenance) if item.startswith("preset_selection:")),
        None,
    )
    # Creative intents written before 7J.1 stored an effective pinned preset.
    return marker.partition(":")[2] if marker is not None else "explicit"


def _creative_policy_for_intent(parent: CreativeIntent, config: AppConfig) -> CreativePolicy:
    return creative_policy_for_config(
        parent.policy,
        config,
        parent_selection_mode=_intent_preset_selection_mode(parent),
    )
