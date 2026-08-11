from __future__ import annotations

"""Candidate-owned persistence and safe revision for native creative plans."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from app.composition_planning import TargetObservation
from app.config import AppConfig
from app.creative_contracts import (
    CompiledRenderPlan,
    CreativeIntent,
    CreativePolicy,
    FrozenContract,
    HASH_PATTERN,
    ImmutableProductionPlanLink,
    canonical_hash,
)
from app.creative_policy import preset_family_policy
from app.production_models import ProductionPlan
from app.source_broll_planning import SourceSceneEvidence
from app.utils import read_json, utc_now, write_json


CREATIVE_HANDOFF_SCHEMA_VERSION = "7G.3.creative-handoff.1"
CREATIVE_EXECUTION_SCHEMA_VERSION = "7G.3.creative-execution.1"

ExecutionStatus = Literal["native_rich", "native_fallback", "legacy"]


class CreativeArtifactError(ValueError):
    pass


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


def creative_policy_for_config(parent: CreativePolicy, config: AppConfig) -> CreativePolicy:
    family_policy = preset_family_policy(config.product_flow.subtitle_preset)  # type: ignore[arg-type]
    return parent.model_copy(update={
        "preset_id": config.product_flow.subtitle_preset,
        "preset_version": config.product_flow.preset_version,
        "platform": config.product_flow.platform,
        "caption_style_family": family_policy.caption_style_family,
        "intensity": family_policy.intensity_ceiling,
        # Rights and evidence do not become stronger during a style revision.
        "source_broll_enabled": parent.source_broll_enabled,
    })


def revise_creative_intent(parent: CreativeIntent, config: AppConfig) -> CreativeIntent:
    policy = creative_policy_for_config(parent.policy, config)
    if policy == parent.policy:
        return parent
    revision = parent.revision + 1
    identity = canonical_hash({
        "parent_intent_hash": parent.canonical_hash(),
        "revision": revision,
        "policy": policy.model_dump(mode="json"),
        "evidence_fingerprint": parent.evidence_fingerprint,
        "mapping_fingerprint": parent.source_output_mapping.fingerprint,
    })
    return parent.model_copy(update={
        "intent_id": f"intent-revision-{identity[:16]}",
        "revision": revision,
        "policy": policy,
        "provenance": (*parent.provenance, f"style_revision:{revision}"),
    })


def creative_policy_changed(intent: CreativeIntent, config: AppConfig) -> bool:
    return creative_policy_for_config(intent.policy, config) != intent.policy
