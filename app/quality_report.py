"""Persisted Final Quality Gate aggregation for canonical V2 artifacts.

This module deliberately consumes already-persisted decisions and validation
results.  It does not run ffprobe, inspect frames, or retry rendering: the
existing producers remain the owners of those low-level checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from app.candidate_quality import cached_hard_eligibility_reason_codes
from app.clip_results import ClipResult
from app.utils import stable_file_hash, stable_text_hash, utc_now


QUALITY_REPORT_SCHEMA_VERSION = "5G.0"
QUALITY_STATUSES = frozenset({"PASS", "PASS_WITH_WARNINGS", "BLOCKED"})
SEMANTIC_DIALOGUE_CONFIDENCE_THRESHOLD = 0.5
QualitySeverity = Literal["warning", "blocker"]


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """One normalized finding emitted from an existing quality producer."""

    code: str
    severity: QualitySeverity
    evidence: dict[str, Any]
    measured_value: Any = None
    threshold: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)
    interval: dict[str, float] | None = None
    config_version: str = QUALITY_REPORT_SCHEMA_VERSION
    auto_fix_available: bool = False
    user_message: str = ""
    technical_details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "evidence": self.evidence,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "provenance": self.provenance,
            "interval": self.interval,
            "config_version": self.config_version,
            "auto_fix_available": self.auto_fix_available,
            "user_message": self.user_message,
            "technical_details": self.technical_details,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Versioned, per-artifact final readiness source of truth."""

    report_id: str
    artifact_id: str
    artifact_path: str
    artifact_sha256: str
    run_id: str
    project_id: str | None
    source_id: str
    candidate_id: str
    edit_plan_id: str
    render_id: str
    status: str
    checks: tuple[dict[str, Any], ...]
    findings: tuple[QualityFinding, ...]
    metrics: dict[str, Any]
    fallbacks: list[Any]
    created_at: str
    provenance: dict[str, Any]
    schema_version: str = QUALITY_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        findings = [item.to_dict() for item in self.findings]
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "edit_plan_id": self.edit_plan_id,
            "render_id": self.render_id,
            "status": self.status,
            "checks": list(self.checks),
            "findings": findings,
            "metrics": self.metrics,
            "fallbacks": self.fallbacks,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }

    def reference(self, path: Path) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "path": str(path),
            "artifact_id": self.artifact_id,
            "candidate_id": self.candidate_id,
            "edit_plan_id": self.edit_plan_id,
            "status": self.status,
            "schema_version": self.schema_version,
        }


def aggregate_quality_status(reports: Iterable[QualityReport | dict[str, Any]]) -> str:
    """Apply the quality standard's blocker-first aggregation exactly once."""

    statuses = [
        item.status if isinstance(item, QualityReport) else str(item.get("status") or "")
        for item in reports
    ]
    if not statuses or any(status == "BLOCKED" for status in statuses):
        return "BLOCKED"
    if any(status == "PASS_WITH_WARNINGS" for status in statuses):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def quality_status_from_findings(findings: Iterable[QualityFinding]) -> str:
    severities = {item.severity for item in findings}
    if "blocker" in severities:
        return "BLOCKED"
    if "warning" in severities:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def artifact_id_for(result: ClipResult, artifact_sha256: str) -> str:
    """Stable identity binds the canonical bytes to their declared parents."""

    identity = {
        "run_id": result.run_id,
        "candidate_id": result.candidate_id,
        "production_plan_id": result.production_plan_id,
        "revision_id": result.revision_id,
        "sha256": artifact_sha256,
    }
    return f"artifact-{stable_text_hash(str(identity))[:24]}"


def exact_dialogue_semantic_blocker(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Return the shared blocker for exact dialogue mappings, if any.

    Candidate-level averages are intentionally absent from this policy: every
    dialogue/fact mapping that the plan would publish must independently meet
    the existing semantic confidence floor.
    """

    mappings = plan.get("dialogue_mappings") if isinstance(plan, dict) else None
    if not isinstance(mappings, list):
        return None
    low_confidence: list[dict[str, Any]] = []
    for item in mappings:
        if not isinstance(item, dict):
            continue
        evidence_mappings = item.get("evidence_mappings")
        exact_mappings = evidence_mappings if isinstance(evidence_mappings, list) and evidence_mappings else [item]
        for exact in exact_mappings:
            if not isinstance(exact, dict):
                continue
            try:
                confidence = float(exact.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < SEMANTIC_DIALOGUE_CONFIDENCE_THRESHOLD:
                low_confidence.append({
                    "segment_id": item.get("segment_id"),
                    "fact_id": exact.get("fact_id"),
                    "transcript_segment_id": exact.get("transcript_segment_id"),
                    "confidence": round(confidence, 6),
                    "source_start_seconds": exact.get("source_start_seconds"),
                    "source_end_seconds": exact.get("source_end_seconds"),
                })
    if not low_confidence:
        return None
    threshold = SEMANTIC_DIALOGUE_CONFIDENCE_THRESHOLD
    return {
        "code": "AUDIO_UNINTELLIGIBLE",
        "severity": "blocker",
        "evidence": {"low_confidence_dialogue": low_confidence},
        "measured_value": min(item["confidence"] for item in low_confidence),
        "threshold": f">={threshold}",
        "producer": "semantic_content_quality",
        "message": "Published dialogue contains low-confidence ASR/semantic content.",
        "details": (
            "At least one exact dialogue/fact mapping is below the semantic "
            "confidence floor; aggregate candidate confidence cannot override it."
        ),
    }


def build_quality_report(
    *,
    artifact_path: Path,
    result: ClipResult,
    run_id: str,
    project_id: str | None,
    source: dict[str, Any],
    plan: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    diversity_decision: dict[str, Any] | None,
    render_report: dict[str, Any] | None,
    audio_report: dict[str, Any] | None,
    all_results: Iterable[ClipResult],
    config_version: str = QUALITY_REPORT_SCHEMA_VERSION,
) -> QualityReport:
    """Normalize existing quality evidence for one already-rendered artifact."""

    resolved_path = artifact_path.resolve()
    source_id = str(source.get("id") or "")
    artifact_sha256 = stable_file_hash(resolved_path) if resolved_path.is_file() else ""
    artifact_id = result.artifact_id or artifact_id_for(result, artifact_sha256 or "missing")
    render = render_report if isinstance(render_report, dict) else {}
    audio = audio_report if isinstance(audio_report, dict) else {}
    plan_data = plan if isinstance(plan, dict) else {}
    candidate_data = candidate if isinstance(candidate, dict) else {}
    findings: list[QualityFinding] = []

    def finding(
        code: str,
        severity: QualitySeverity,
        evidence: dict[str, Any],
        *,
        measured_value: Any = None,
        threshold: Any = None,
        producer: str,
        message: str,
        details: str = "",
        interval: dict[str, float] | None = None,
        auto_fix_available: bool = False,
    ) -> None:
        findings.append(QualityFinding(
            code=code,
            severity=severity,
            evidence=evidence,
            measured_value=measured_value,
            threshold=threshold,
            provenance={"producer": producer, "config_version": config_version},
            interval=interval,
            config_version=config_version,
            auto_fix_available=auto_fix_available,
            user_message=message,
            technical_details=details or message,
        ))

    _collect_artifact_identity(
        finding, resolved_path, artifact_sha256, result, run_id, source_id, artifact_id, render,
        all_results,
    )
    _collect_eligibility(finding, candidate_data)
    _collect_diversity(finding, diversity_decision, result.candidate_id)
    _collect_plan_and_boundary(finding, plan_data, result, project_id, source_id)
    _collect_continuity(finding, plan_data, render)
    _collect_semantic_content(finding, plan_data)
    _collect_creative_execution(finding, render)
    _collect_composition_and_subtitles(finding, render)
    _collect_audio(finding, audio)
    _collect_ffprobe(finding, render)

    status = quality_status_from_findings(findings)
    checks = _check_catalog(
        findings,
        candidate=candidate_data,
        diversity_decision=diversity_decision,
        plan=plan_data,
        render=render,
        audio=audio,
        artifact={"path": str(resolved_path), "sha256": artifact_sha256, "artifact_id": artifact_id},
        config_version=config_version,
    )
    report_id = "quality-" + stable_text_hash(str({
        "artifact_id": artifact_id,
        "run_id": run_id,
        "status": status,
        "findings": [item.to_dict() for item in findings],
    }))[:24]
    metrics = {
        "artifact": {
            "byte_size": resolved_path.stat().st_size if resolved_path.is_file() else None,
            "sha256": artifact_sha256 or None,
        },
        "technical": {
            key: render.get(key)
            for key in ("duration", "audio_duration", "sync_difference_ms", "resolution", "fps", "validation")
            if key in render
        },
        "audio": dict(audio.get("validation") or {}) if isinstance(audio.get("validation"), dict) else {},
        "captions": _caption_quality(render) or {},
        "composition": _composition_quality(render) or {},
        "source_broll": _source_broll_quality(render) or {},
        "motion": _motion_quality(render) or {},
        "continuity": _continuity_metrics(plan_data, render),
    }
    composition_fallbacks = _composition_fallbacks(render)
    source_broll_fallbacks = _source_broll_fallbacks(render)
    motion_fallbacks = _motion_fallbacks(render)
    native_creative_qc = render.get("compatibility_mode") == "native"
    fallbacks = _unique([
        *(render.get("fallback_reasons", []) if isinstance(render.get("fallback_reasons"), list) else []),
        *(render.get("warnings", []) if isinstance(render.get("warnings"), list) else []),
        *composition_fallbacks,
        *source_broll_fallbacks,
        *motion_fallbacks,
    ])
    return QualityReport(
        report_id=report_id,
        artifact_id=artifact_id,
        artifact_path=str(resolved_path),
        artifact_sha256=artifact_sha256,
        run_id=run_id,
        project_id=project_id,
        source_id=source_id,
        candidate_id=result.candidate_id,
        edit_plan_id=result.production_plan_id,
        render_id=result.revision_id,
        status=status,
        checks=tuple(checks),
        findings=tuple(findings),
        metrics=metrics,
        fallbacks=fallbacks,
        created_at=utc_now(),
        provenance={
            "owner": "final_quality_gate",
            "quality_config_version": config_version,
            "low_level_checks_reused": [
                "eligibility", "diversity", "boundary_decision", "production_plan_envelope",
                "continuity_decision", "source_output_time_map",
                *(
                    [
                        "compiled_render_plan", "caption_quality_report",
                        "composition_quality_report", "source_broll_quality_report",
                        "motion_quality_report",
                    ]
                    if native_creative_qc
                    else ["composition_quality_decision", "subtitle_quality_decision"]
                ),
                "audio_validation",
                "render_validation", "artifact_identity",
            ],
        },
    )


def read_quality_report(value: Any) -> dict[str, Any] | None:
    """Validate the minimal persisted contract used by desktop completion."""

    if not isinstance(value, dict):
        return None
    required = ("schema_version", "report_id", "artifact_id", "artifact_path", "run_id", "status")
    if any(not value.get(key) for key in required):
        return None
    if value.get("schema_version") != QUALITY_REPORT_SCHEMA_VERSION:
        return None
    if str(value.get("status")) not in QUALITY_STATUSES:
        return None
    if not isinstance(value.get("checks"), list) or not isinstance(value.get("findings"), list):
        return None
    for check in value["checks"]:
        if not isinstance(check, dict) or not all(key in check for key in (
            "code", "severity", "evidence", "measured_value", "threshold", "provenance",
        )):
            return None
    for finding in value["findings"]:
        if not isinstance(finding, dict) or not all(key in finding for key in (
            "code", "severity", "evidence", "measured_value", "threshold", "provenance",
        )):
            return None
    return value


def _collect_artifact_identity(
    finding: Any,
    path: Path,
    checksum: str,
    result: ClipResult,
    run_id: str,
    source_id: str,
    artifact_id: str,
    render: dict[str, Any],
    all_results: Iterable[ClipResult],
) -> None:
    evidence = {
        "path": str(path), "artifact_id": artifact_id, "run_id": result.run_id,
        "candidate_id": result.candidate_id, "edit_plan_id": result.production_plan_id,
        "render_id": result.revision_id, "source_id": source_id,
    }
    missing = [
        name for name, value in {
            "artifact": path.is_file() and path.stat().st_size > 0,
            "checksum": bool(checksum),
            "candidate_id": bool(result.candidate_id),
            "edit_plan_id": bool(result.production_plan_id),
            "render_id": bool(result.revision_id),
            "run_id": result.run_id == run_id,
        }.items() if not value
    ]
    reported_path = str(render.get("output_file") or "")
    if reported_path:
        try:
            if Path(reported_path).resolve() != path:
                missing.append("render_output_file")
        except OSError:
            missing.append("render_output_file")
    if missing:
        finding(
            "WRONG_ARTIFACT_LINK", "blocker", {**evidence, "missing_or_mismatched": missing},
            measured_value=missing, threshold="all artifact parents and checksum must match",
            producer="artifact_identity", message="Final artifact identity does not match its declared parents.",
        )
    duplicates = [
        other.candidate_id for other in all_results
        if other is not result and (
            (other.artifact_id and other.artifact_id == artifact_id)
            or (other.content_fingerprint and other.content_fingerprint == result.content_fingerprint)
            or (other.production_plan_id and other.production_plan_id == result.production_plan_id)
        )
    ]
    if duplicates:
        finding(
            "DUPLICATE_OUTPUT", "blocker", {**evidence, "duplicate_candidate_ids": duplicates},
            measured_value=duplicates, threshold="one canonical artifact per candidate/plan/content",
            producer="clip_result_registry", message="Canonical output duplicates another final artifact.",
        )


def _collect_eligibility(finding: Any, candidate: dict[str, Any]) -> None:
    decision = candidate.get("eligibility_decision") if isinstance(candidate, dict) else None
    if not isinstance(decision, dict) or decision.get("state") == "legacy_unassessed":
        virality = candidate.get("virality") if isinstance(candidate, dict) else None
        cached = virality.get("eligibility") if isinstance(virality, dict) else None
        cached_codes, critical_failures = cached_hard_eligibility_reason_codes(cached)
        if critical_failures:
            codes = [code.value for code in cached_codes]
            code = "CONTEXT_DEBT_CRITICAL" if "CONTEXT_DEBT_CRITICAL" in codes else "SEMANTIC_INCOMPLETE"
            finding(
                code,
                "blocker",
                {
                    "eligibility_decision": decision or None,
                    "cached_eligibility": cached,
                },
                measured_value=critical_failures,
                threshold="no explicit cached critical failures",
                producer="eligibility",
                message="Legacy candidate has an explicit cached hard eligibility rejection.",
            )
            return
        finding(
            "QUALITY_CONFIDENCE_LOW", "warning", {"eligibility_decision": decision or None},
            measured_value="legacy_unassessed", threshold="assessed eligible candidate",
            producer="eligibility", message="Candidate eligibility is legacy or unavailable.",
        )
        return
    if decision.get("eligible") is not True:
        codes = [str(item) for item in decision.get("reason_codes", [])]
        code = "CONTEXT_DEBT_CRITICAL" if "CONTEXT_DEBT_CRITICAL" in codes else "SEMANTIC_INCOMPLETE"
        finding(
            code, "blocker", {"eligibility_decision": decision}, measured_value=codes,
            threshold="eligible=true", producer="eligibility",
            message="Candidate did not pass the persisted eligibility decision.",
        )


def _collect_diversity(finding: Any, decision: dict[str, Any] | None, candidate_id: str) -> None:
    if not isinstance(decision, dict) or str(decision.get("schema_version") or "") == "legacy":
        finding(
            "QUALITY_CONFIDENCE_LOW", "warning", {"diversity_decision": decision or None},
            measured_value="legacy_unassessed", threshold="versioned diversity decision",
            producer="diversity", message="Diversity decision is legacy or unavailable.",
        )
        return
    selected = {str(item) for item in decision.get("selected_candidate_ids", [])}
    if selected and candidate_id not in selected:
        finding(
            "WRONG_ARTIFACT_LINK", "blocker", {"diversity_decision": decision, "candidate_id": candidate_id},
            measured_value=candidate_id, threshold="candidate belongs to selected diversity set",
            producer="diversity", message="Rendered candidate is absent from the persisted diversity selection.",
        )


def _collect_semantic_content(finding: Any, plan: dict[str, Any]) -> None:
    """Block source dialogue whose semantic confidence is not publishable.

    Candidate-level averages can hide a corrupted sentence.  ProductionPlan
    carries the grounded per-fact confidence, so the final gate evaluates the
    exact dialogue that reached the MP4 instead of trusting the aggregate ASR
    score used during ranking.
    """

    blocker = exact_dialogue_semantic_blocker(plan)
    if blocker is not None:
        finding(
            blocker["code"], blocker["severity"], blocker["evidence"],
            measured_value=blocker["measured_value"],
            threshold=blocker["threshold"], producer=blocker["producer"],
            message=blocker["message"], details=blocker["details"],
        )


def _collect_plan_and_boundary(
    finding: Any,
    plan: dict[str, Any],
    result: ClipResult,
    project_id: str | None,
    source_id: str,
) -> None:
    envelope = plan.get("envelope") if isinstance(plan, dict) else None
    if not isinstance(envelope, dict) or envelope.get("compatibility_mode") == "legacy_adapter":
        legacy_gap = _legacy_source_plan_has_gap(plan)
        finding(
            "LEGACY_PLAN_REBUILD_REQUIRED" if legacy_gap else "QUALITY_CONFIDENCE_LOW",
            "blocker" if legacy_gap else "warning",
            {"plan_envelope": envelope or None, "physical_source_gap": legacy_gap},
            measured_value="legacy_adapter", threshold="native ProductionPlan envelope",
            producer="production_plan",
            message=(
                "Legacy source plan contains an unexplained physical cut and must be rebuilt from cached analysis."
                if legacy_gap else "Production plan uses an explicit legacy compatibility contract."
            ),
        )
        return
    raw_identity = envelope.get("identity")
    identity: dict[str, Any] = raw_identity if isinstance(raw_identity, dict) else {}
    expected = {
        "candidate_id": result.candidate_id,
        "source_id": source_id,
        "project_id": project_id,
    }
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items() if value is not None and identity.get(key) != value
    }
    if str(plan.get("plan_id") or "") != result.production_plan_id:
        mismatches["plan_id"] = {"expected": result.production_plan_id, "actual": plan.get("plan_id")}
    if mismatches:
        code = "SOURCE_IDENTITY_MISMATCH" if "source_id" in mismatches else "EDIT_PLAN_MISMATCH"
        finding(
            code, "blocker", {"identity": identity, "mismatches": mismatches}, measured_value=mismatches,
            threshold="plan identity matches canonical ClipResult", producer="production_plan_envelope",
            message="Production plan identity does not match the canonical output.",
        )
    boundary = plan.get("boundary_decision") if isinstance(plan.get("boundary_decision"), dict) else None
    if not boundary:
        finding(
            "QUALITY_CONFIDENCE_LOW", "warning", {"boundary_decision": None},
            measured_value="unavailable", threshold="persisted BoundaryDecision", producer="boundary_decision",
            message="Boundary decision is unavailable for this plan.",
        )
        return
    interval = _boundary_interval(boundary)
    if boundary.get("word_integrity") is not True:
        finding(
            "BOUNDARY_WORD_CUT", "blocker", {"boundary_decision": boundary}, measured_value=False,
            threshold=True, producer="boundary_decision", message="Boundary decision reports a word-integrity failure.",
            interval=interval,
        )
    if boundary.get("semantic_completion") is not True or boundary.get("payoff_preserved") is not True:
        finding(
            "SEMANTIC_INCOMPLETE", "blocker", {"boundary_decision": boundary},
            measured_value={"semantic_completion": boundary.get("semantic_completion"), "payoff_preserved": boundary.get("payoff_preserved")},
            threshold={"semantic_completion": True, "payoff_preserved": True}, producer="boundary_decision",
            message="Boundary decision reports an incomplete semantic ending.", interval=interval,
        )


def _legacy_source_plan_has_gap(plan: dict[str, Any]) -> bool:
    audio_mode = plan.get("audio_mode")
    if audio_mode is None:
        segments = plan.get("segments") if isinstance(plan.get("segments"), list) else []
        has_narration = any(
            isinstance(item, dict) and item.get("segment_type") == "narration"
            for item in segments
        )
        audio_mode = "voiceover" if has_narration else "original"
    if audio_mode not in {"original", "original_enhanced"}:
        return False
    segments = plan.get("segments")
    source_items = [
        item for item in segments
        if isinstance(item, dict) and item.get("segment_type") == "original_dialogue"
    ] if isinstance(segments, list) else []
    if not source_items:
        mappings = plan.get("dialogue_mappings")
        source_items = mappings if isinstance(mappings, list) else []
    if not source_items:
        return False
    ranges: list[tuple[float, float]] = []
    for item in source_items:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["source_start_seconds"])
            end = float(item["source_end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    approved: dict[str, Any] | None = None
    continuity = plan.get("continuity_decision")
    boundary = plan.get("boundary_decision")
    if isinstance(continuity, dict) and isinstance(continuity.get("approved_source_range"), dict):
        approved = continuity["approved_source_range"]
    elif isinstance(boundary, dict) and isinstance(boundary.get("refined_range"), dict):
        approved = boundary["refined_range"]
    try:
        approved_start = float(approved["start_seconds"]) if approved is not None else None
        approved_end = float(approved["end_seconds"]) if approved is not None else None
    except (KeyError, TypeError, ValueError):
        approved_start = approved_end = None
    cursor: float | None = approved_start
    for start, end in sorted(ranges):
        if approved_start is not None and end <= approved_start + 0.001:
            continue
        if approved_end is not None and start >= approved_end - 0.001:
            break
        if cursor is not None and start > cursor + 0.001:
            return True
        cursor = max(cursor or end, end)
    return bool(approved_end is not None and (cursor is None or cursor < approved_end - 0.001))


def _collect_continuity(finding: Any, plan: dict[str, Any], render: dict[str, Any]) -> None:
    """Block native delivery when a non-dialogue omission lacks provenance."""

    envelope = plan.get("envelope") if isinstance(plan.get("envelope"), dict) else {}
    native = envelope.get("compatibility_mode") == "native"
    decision = plan.get("continuity_decision") if isinstance(plan.get("continuity_decision"), dict) else None
    if decision is None:
        if native:
            finding(
                "CONTINUITY_DECISION_MISSING", "blocker", {"continuity_decision": None},
                measured_value="unavailable", threshold="persisted A-2 ContinuityDecision",
                producer="continuity_decision",
                message="Native production plan has no continuity provenance for non-dialogue spans.",
            )
        return
    try:
        expected_sha256 = _continuity_fingerprint(decision)
    except Exception as error:
        finding(
            "CONTINUITY_DECISION_INVALID", "blocker", {"continuity_decision": decision},
            measured_value=str(error), threshold="valid A-2 ContinuityDecision",
            producer="continuity_decision",
            message="Continuity decision cannot be validated as a versioned production artifact.",
        )
        return
    expected_id = decision.get("decision_id")
    expected_version = decision.get("schema_version")
    fingerprints = envelope.get("input_fingerprints") if isinstance(envelope.get("input_fingerprints"), dict) else {}
    if native and (
        envelope.get("continuity_decision_ref") != expected_id
        or fingerprints.get("continuity_decision_sha256") != expected_sha256
    ):
        finding(
            "CONTINUITY_DECISION_IDENTITY_MISMATCH", "blocker",
            {"continuity_decision": decision, "plan_envelope": envelope},
            measured_value={
                "ref": envelope.get("continuity_decision_ref"),
                "sha256": fingerprints.get("continuity_decision_sha256"),
            },
            threshold={"ref": expected_id, "sha256": expected_sha256},
            producer="continuity_decision",
            message="Production envelope/cache identity does not bind the exact continuity decision.",
        )
        return
    mapping = _source_output_time_map(render)
    if mapping is None:
        required = decision.get("required_spans") if isinstance(decision.get("required_spans"), list) else []
        omissions = decision.get("omitted_spans") if isinstance(decision.get("omitted_spans"), list) else []
        needs_omission_proof = bool(required) or any(
            isinstance(span, dict) and span.get("rationale_type") == "unexplained"
            for span in omissions
        )
        finding(
            "CONTINUITY_TIME_MAP_MISSING", "blocker" if needs_omission_proof else "warning",
            {"continuity_decision": decision},
            measured_value="unavailable", threshold="SourceOutputTimeMap bound to ContinuityDecision",
            producer="continuity_decision",
            message=(
                "Final quality cannot verify required or unexplained continuity spans without the source/output time map."
                if needs_omission_proof
                else "Continuity has no required or unexplained spans, but the source/output time map was not persisted."
            ),
        )
        return
    actual_identity = {
        "decision_id": mapping.get("continuity_decision_id"),
        "schema_version": mapping.get("continuity_decision_version"),
        "sha256": mapping.get("continuity_decision_sha256"),
    }
    if actual_identity != {
        "decision_id": expected_id,
        "schema_version": expected_version,
        "sha256": expected_sha256,
    }:
        finding(
            "CONTINUITY_TIME_MAP_MISMATCH", "blocker",
            {"continuity_decision": decision, "source_output_time_map": mapping},
            measured_value=actual_identity,
            threshold={"decision_id": expected_id, "schema_version": expected_version, "sha256": expected_sha256},
            producer="continuity_decision",
            message="Source/output time map is not bound to the exact continuity decision used by production.",
        )
        return
    ranges = _source_ranges_from_map(mapping)
    required = decision.get("required_spans") if isinstance(decision.get("required_spans"), list) else []
    for span in required:
        source_range = span.get("source_range") if isinstance(span, dict) else None
        if not _map_covers(source_range, ranges):
            finding(
                "CONTINUITY_REQUIRED_SPAN_LOST", "blocker",
                {"continuity_decision": decision, "required_span": span, "source_output_time_map": mapping},
                measured_value=source_range, threshold="required source span retained in output",
                producer="continuity_decision",
                message="Evidence-backed continuity span is absent from the rendered source map.",
                interval=source_range if isinstance(source_range, dict) else None,
            )
    omissions = decision.get("omitted_spans") if isinstance(decision.get("omitted_spans"), list) else []
    unresolved = [
        span for span in omissions
        if isinstance(span, dict)
        and span.get("rationale_type") == "unexplained"
        and not _map_covers(span.get("source_range"), ranges)
    ]
    if unresolved:
        finding(
            "CONTINUITY_UNEXPLAINED_OMISSION", "blocker",
            {"continuity_decision": decision, "unexplained_omissions": unresolved, "source_output_time_map": mapping},
            measured_value={"mode": decision.get("mode"), "unexplained_omission_count": len(unresolved)},
            threshold="no unexplained non-dialogue omissions remain in the rendered map",
            producer="continuity_decision",
            message="Continuity evidence is uncertain and production omitted non-dialogue spans without a proven rationale.",
        )


def _source_output_time_map(render: dict[str, Any]) -> dict[str, Any] | None:
    mapping = render.get("source_output_time_map")
    if isinstance(mapping, dict):
        return mapping
    compiled = render.get("compiled_render_plan")
    if isinstance(compiled, dict) and isinstance(compiled.get("source_output_mapping"), dict):
        return compiled["source_output_mapping"]
    return None


def _continuity_fingerprint(decision: dict[str, Any]) -> str:
    """Use the production contract's canonical decision serialization."""

    from app.production_models import ContinuityDecision

    return ContinuityDecision.model_validate(decision).fingerprint()


def _source_ranges_from_map(mapping: dict[str, Any]) -> list[tuple[float, float]]:
    raw_segments = mapping.get("segments") if isinstance(mapping.get("segments"), list) else []
    ranges: list[tuple[float, float]] = []
    ticks_per_second = float(mapping.get("source_ticks_per_second") or 1_000_000)
    for item in raw_segments:
        source = item.get("source") if isinstance(item, dict) else None
        if not isinstance(source, dict):
            continue
        try:
            start = float(source["start_tick"]) / ticks_per_second
            end = float(source["end_tick"]) / ticks_per_second
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if end > start:
            ranges.append((start, end))
    return sorted(ranges)


def _map_covers(source_range: Any, ranges: list[tuple[float, float]]) -> bool:
    if not isinstance(source_range, dict):
        return False
    try:
        cursor = float(source_range["start_seconds"])
        end = float(source_range["end_seconds"])
    except (KeyError, TypeError, ValueError):
        return False
    for start, stop in ranges:
        if stop < cursor - 0.001:
            continue
        if start > cursor + 0.001:
            break
        cursor = max(cursor, stop)
        if cursor >= end - 0.001:
            return True
    return False


def _continuity_metrics(plan: dict[str, Any], render: dict[str, Any]) -> dict[str, Any]:
    decision = plan.get("continuity_decision") if isinstance(plan.get("continuity_decision"), dict) else {}
    mapping = _source_output_time_map(render)
    return {
        "decision_id": decision.get("decision_id"),
        "mode": decision.get("mode"),
        "required_span_count": len(decision.get("required_spans", [])) if isinstance(decision.get("required_spans"), list) else 0,
        "omitted_span_count": len(decision.get("omitted_spans", [])) if isinstance(decision.get("omitted_spans"), list) else 0,
        "source_output_mapping_present": mapping is not None,
    }


def _native_caption_event_collisions(render: dict[str, Any]) -> list[dict[str, Any]]:
    caption_plan = render.get("caption_plan")
    if not isinstance(caption_plan, dict):
        compiled = render.get("compiled_render_plan")
        caption_plan = compiled.get("caption_plan") if isinstance(compiled, dict) else None
    raw_cues = caption_plan.get("cues") if isinstance(caption_plan, dict) else None
    if not isinstance(raw_cues, list):
        return []

    cues: list[dict[str, Any]] = []
    for cue in raw_cues:
        if not isinstance(cue, dict):
            continue
        output = cue.get("output")
        bounds = cue.get("normalized_bounds")
        if not isinstance(output, dict) or not isinstance(bounds, dict):
            continue
        try:
            cues.append({
                "cue_id": str(cue.get("cue_id") or "unknown-caption"),
                "start": int(output["start_frame"]),
                "end": int(output["end_frame"]),
                "x": float(bounds["x"]),
                "y": float(bounds["y"]),
                "width": float(bounds["width"]),
                "height": float(bounds["height"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    cues.sort(key=lambda item: (item["start"], item["end"], item["cue_id"]))

    collisions: list[dict[str, Any]] = []
    for left_index, left in enumerate(cues):
        for right in cues[left_index + 1:]:
            if right["start"] >= left["end"]:
                break
            overlap_frames = min(left["end"], right["end"]) - max(left["start"], right["start"])
            if overlap_frames <= 0:
                continue
            width = max(0.0, min(left["x"] + left["width"], right["x"] + right["width"]) - max(left["x"], right["x"]))
            height = max(0.0, min(left["y"] + left["height"], right["y"] + right["height"]) - max(left["y"], right["y"]))
            if width <= 0 or height <= 0:
                continue
            collisions.append({
                "left_cue_id": left["cue_id"],
                "right_cue_id": right["cue_id"],
                "overlap_frames": overlap_frames,
                "start_frame": max(left["start"], right["start"]),
                "end_frame": min(left["end"], right["end"]),
            })
    return collisions


def _collect_composition_and_subtitles(finding: Any, render: dict[str, Any]) -> None:
    native = render.get("compatibility_mode") == "native"
    compiled = render.get("compiled_render_plan")
    if native and not isinstance(compiled, dict):
        finding(
            "EDIT_PLAN_MISMATCH", "blocker", {"compiled_render_plan": compiled},
            measured_value="missing", threshold="validated native CompiledRenderPlan",
            producer="compiled_render_plan",
            message="Native creative QC is missing its immutable CompiledRenderPlan.",
        )
    raw_quality = render.get("quality")
    quality: dict[str, Any] = raw_quality if isinstance(raw_quality, dict) else {}
    raw_composition = render.get("composition")
    composition: dict[str, Any] = raw_composition if isinstance(raw_composition, dict) else {}
    raw_segments = composition.get("segments") if not native else None
    segments: list[Any] = raw_segments if isinstance(raw_segments, list) else []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        decision = segment.get("composition_quality_decision")
        decision = decision if isinstance(decision, dict) else {}
        status = str(decision.get("status") or segment.get("composition_quality_status") or "")
        if status in {"blocked", "failed", "failed_repairable"}:
            finding(
                "FACE_CROP_CRITICAL", "blocker", {"segment": segment, "decision": decision},
                measured_value=status, threshold="passed", producer="composition_quality_decision",
                message="Composition quality decision blocks the final crop.",
                interval=_segment_interval(segment),
            )
        elif status in {"fallback", "evidence_unavailable", "passed_with_warning"}:
            finding(
                "TARGET_CONFIDENCE_LOW", "warning", {"segment": segment, "decision": decision},
                measured_value=status, threshold="passed with valid evidence", producer="composition_quality_decision",
                message="Composition quality uses a declared fallback or limited evidence.",
                interval=_segment_interval(segment),
            )
    raw_subtitle_layout = render.get("subtitle_layout") if not native else None
    subtitle_layout: dict[str, Any] = raw_subtitle_layout if isinstance(raw_subtitle_layout, dict) else {}
    raw_subtitle = subtitle_layout.get("quality_decision")
    subtitle: dict[str, Any] | None = raw_subtitle if isinstance(raw_subtitle, dict) else None
    if subtitle is not None:
        subtitle_status = str(subtitle.get("status") or "legacy_unassessed")
        codes = [str(item) for item in subtitle.get("reason_codes", [])]
        if subtitle_status == "blocked":
            code = "SUBTITLE_OUT_OF_FRAME" if "SUBTITLE_OUT_OF_FRAME" in codes else "SUBTITLE_UNREADABLE"
            finding(
                code, "blocker", {"subtitle_quality_decision": subtitle}, measured_value=codes,
                threshold="subtitle layout passed", producer="subtitle_quality_decision",
                message="Subtitle quality decision blocks the final layout.",
            )
        elif subtitle_status in {"passed_with_warning", "legacy_unassessed"}:
            code = "SUBTITLE_CPS_HIGH" if "CPS_TOO_HIGH" in codes else "QUALITY_CONFIDENCE_LOW"
            finding(
                code, "warning", {"subtitle_quality_decision": subtitle}, measured_value=codes,
                threshold="subtitle layout passed", producer="subtitle_quality_decision",
                message="Subtitle quality decision requires attention.",
            )
    caption = _caption_quality(render)
    reported_overlap_cues: set[str] = set()
    if caption is not None:
        raw_caption_findings = caption.get("findings")
        caption_findings: list[Any] = raw_caption_findings if isinstance(raw_caption_findings, list) else []
        for item in caption_findings:
            if not isinstance(item, dict):
                continue
            severity = "blocker" if item.get("severity") == "blocker" else "warning"
            code = str(item.get("code") or "CAPTION_QUALITY_DEGRADED")
            if code == "CAPTION_SIMULTANEOUS_OVERLAP" and severity == "blocker":
                reported_overlap_cues.add(str(item.get("cue_id") or ""))
            finding(
                code, severity, {"caption_quality_report": caption, "caption_finding": item},
                measured_value=item.get("measured_value"), threshold=item.get("threshold"),
                producer="caption_quality_report",
                message=str(item.get("message") or "Semantic caption quality requires attention."),
            )
        if caption.get("status") == "BLOCKED" and not caption_findings:
            finding(
                "CAPTION_QUALITY_BLOCKED", "blocker", {"caption_quality_report": caption},
                measured_value="BLOCKED", threshold="PASS or PASS_WITH_WARNINGS",
                producer="caption_quality_report",
                message="Semantic caption quality blocks the final artifact.",
            )
    if native:
        for overlap in _native_caption_event_collisions(render):
            if overlap["right_cue_id"] in reported_overlap_cues:
                continue
            finding(
                "CAPTION_SIMULTANEOUS_OVERLAP",
                "blocker",
                {"caption_event_collision": overlap},
                measured_value=overlap["overlap_frames"],
                threshold=0,
                producer="caption_quality_report",
                message="Native caption events overlap in time and rendered screen area.",
            )
    composition_report = _composition_quality(render)
    if composition_report is not None:
        raw_findings = composition_report.get("findings")
        composition_findings: list[Any] = raw_findings if isinstance(raw_findings, list) else []
        for item in composition_findings:
            if not isinstance(item, dict):
                continue
            severity = "blocker" if item.get("severity") == "blocker" else "warning"
            code = str(item.get("code") or "COMPOSITION_QUALITY_DEGRADED")
            finding(
                code, severity,
                {"composition_quality_report": composition_report, "composition_finding": item},
                measured_value=item.get("measured_value"), threshold=item.get("threshold"),
                producer="composition_quality_report",
                message=str(item.get("message") or "Dynamic composition quality requires attention."),
            )
        if composition_report.get("status") == "BLOCKED" and not composition_findings:
            finding(
                "COMPOSITION_QUALITY_BLOCKED", "blocker",
                {"composition_quality_report": composition_report},
                measured_value="BLOCKED", threshold="PASS or PASS_WITH_WARNINGS",
                producer="composition_quality_report",
                message="Dynamic composition quality blocks the final artifact.",
            )
    source_broll_report = _source_broll_quality(render)
    if source_broll_report is not None:
        raw_findings = source_broll_report.get("findings")
        broll_findings: list[Any] = raw_findings if isinstance(raw_findings, list) else []
        for item in broll_findings:
            if not isinstance(item, dict):
                continue
            severity = "blocker" if item.get("severity") == "blocker" else "warning"
            finding(
                str(item.get("code") or "SOURCE_BROLL_QUALITY_DEGRADED"), severity,
                {"source_broll_quality_report": source_broll_report, "source_broll_finding": item},
                measured_value=item.get("measured_value"), threshold=item.get("threshold"),
                producer="source_broll_quality_report",
                message=str(item.get("message") or "Source B-roll safety requires attention."),
            )
        if source_broll_report.get("status") == "BLOCKED" and not broll_findings:
            finding(
                "SOURCE_BROLL_QUALITY_BLOCKED", "blocker",
                {"source_broll_quality_report": source_broll_report},
                measured_value="BLOCKED", threshold="PASS or PASS_WITH_WARNINGS",
                producer="source_broll_quality_report",
                message="Source B-roll safety blocks the final artifact.",
            )
    motion_report = _motion_quality(render)
    if motion_report is not None:
        raw_findings = motion_report.get("findings")
        motion_findings: list[Any] = raw_findings if isinstance(raw_findings, list) else []
        for item in motion_findings:
            if not isinstance(item, dict):
                continue
            severity = "blocker" if item.get("severity") == "blocker" else "warning"
            finding(
                str(item.get("code") or "MOTION_QUALITY_DEGRADED"), severity,
                {"motion_quality_report": motion_report, "motion_finding": item},
                measured_value=item.get("measured_value"), threshold=item.get("threshold"),
                producer="motion_quality_report",
                message=str(item.get("message") or "Editorial motion requires attention."),
            )
        if motion_report.get("status") == "BLOCKED" and not motion_findings:
            finding(
                "MOTION_QUALITY_BLOCKED", "blocker",
                {"motion_quality_report": motion_report},
                measured_value="BLOCKED", threshold="PASS or PASS_WITH_WARNINGS",
                producer="motion_quality_report",
                message="Editorial motion quality blocks the final artifact.",
            )
    if quality.get("status") == "failed" and not segments and subtitle is None:
        finding(
            "MEDIA_INVALID", "blocker", {"output_quality": quality}, measured_value=quality.get("errors", []),
            threshold="existing output quality validation passed", producer="output_quality",
            message="Existing final output quality validation failed.",
        )


def _collect_creative_execution(finding: Any, render: dict[str, Any]) -> None:
    native = render.get("compatibility_mode") == "native"
    status = str(render.get("execution_status") or "")
    reasons = [str(item) for item in render.get("execution_reason_codes", [])]
    if native and status not in {"native_rich", "native_fallback"}:
        finding(
            "CREATIVE_EXECUTION_STATUS_MISSING", "blocker",
            {"execution_status": status, "reason_codes": reasons},
            measured_value=status or "missing",
            threshold="native_rich or native_fallback",
            producer="creative_execution",
            message="Native render is missing its explicit creative execution status.",
        )
        return
    if status == "native_fallback":
        finding(
            "NATIVE_CREATIVE_FALLBACK", "warning",
            {"execution_status": status, "reason_codes": reasons},
            measured_value=reasons,
            threshold="explicit safe fallback diagnostics",
            producer="creative_execution",
            message="Native render used declared evidence-bounded fallbacks.",
        )
    if status == "native_rich":
        missing, layer_evidence = _missing_native_rich_layers(render)
        if missing:
            finding(
                "NATIVE_RICH_EVIDENCE_MISSING", "blocker",
                {
                    "execution_status": status,
                    "missing_required_layers": missing,
                    "required_layer_evidence": layer_evidence,
                },
                measured_value=missing,
                threshold="all required native creative layers executed",
                producer="creative_execution",
                message="native_rich cannot mask missing required creative layers.",
            )


def _missing_native_rich_layers(render: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Mirror the compiled-plan rich gate without requiring optional B-roll."""

    caption_plan = render.get("caption_plan")
    raw_cues = caption_plan.get("cues") if isinstance(caption_plan, dict) else None
    cues = [item for item in raw_cues if isinstance(item, dict)] if isinstance(raw_cues, list) else []

    motion_plan = render.get("motion_plan")
    raw_events = motion_plan.get("events") if isinstance(motion_plan, dict) else None
    events = [item for item in raw_events if isinstance(item, dict)] if isinstance(raw_events, list) else []
    animated_events = [
        item for item in events
        if str(item.get("primitive_id") or "") not in {"", "static"}
    ]
    presented_roles = {
        str(item.get("beat_role") or "") for item in cues
        if str(item.get("beat_role") or "")
    } | {
        str(item.get("purpose") or "") for item in animated_events
        if str(item.get("purpose") or "")
    }

    composition_plan = render.get("composition_plan")
    raw_segments = composition_plan.get("segments") if isinstance(composition_plan, dict) else None
    segments = [
        item for item in raw_segments if isinstance(item, dict)
    ] if isinstance(raw_segments, list) else []
    reframed_segments = [
        item for item in segments
        if str(item.get("target") or "") not in {"", "stable_source"}
    ]
    emphasis_count = sum(item.get("emphasis") is not None for item in cues)

    missing: list[str] = []
    if not cues:
        missing.append("CAPTION_LAYER_NOT_EXECUTED")
    if not emphasis_count:
        missing.append("SEMANTIC_EMPHASIS_NOT_EXECUTED")
    if "hook" not in presented_roles:
        missing.append("HOOK_PRESENTATION_NOT_EXECUTED")
    if "payoff" not in presented_roles:
        missing.append("PAYOFF_PRESENTATION_NOT_EXECUTED")
    if not reframed_segments:
        missing.append("COMPOSITION_REFRAME_NOT_EXECUTED")
    if not animated_events:
        missing.append("MOTION_LAYER_NOT_EXECUTED")

    return missing, {
        "caption_cue_count": len(cues),
        "semantic_emphasis_count": emphasis_count,
        "presented_roles": sorted(presented_roles),
        "reframed_composition_segment_count": len(reframed_segments),
        "animated_motion_event_count": len(animated_events),
    }


def _collect_audio(finding: Any, audio: dict[str, Any]) -> None:
    validation = audio.get("validation") if isinstance(audio.get("validation"), dict) else None
    if validation is None:
        finding(
            "QUALITY_CONFIDENCE_LOW", "warning", {"audio_validation": None}, measured_value="unavailable",
            threshold="persisted audio validation", producer="audio_validation",
            message="Audio validation is unavailable for this render.",
        )
        return
    status = str(validation.get("status") or "")
    if status == "invalid":
        finding(
            "AUDIO_SILENT_CRITICAL", "blocker", {"audio_validation": validation},
            measured_value=validation.get("messages", []), threshold="audio validation valid",
            producer="audio_validation", message="Existing audio validation failed.",
        )
    elif status == "warning":
        finding(
            "AUDIO_LOUDNESS_OUTSIDE_TARGET", "warning", {"audio_validation": validation},
            measured_value=validation.get("messages", []), threshold="audio validation valid",
            producer="audio_validation", message="Existing audio validation reported a warning.",
        )


def _collect_ffprobe(finding: Any, render: dict[str, Any]) -> None:
    status = str(render.get("validation") or "")
    if status == "invalid":
        finding(
            "MEDIA_INVALID", "blocker", {"render_validation": render}, measured_value=status,
            threshold="valid", producer="render_validation", message="Existing ffprobe validation failed.",
        )
    elif status == "warning":
        finding(
            "DURATION_VARIANCE_LOW", "warning", {"render_validation": render}, measured_value=status,
            threshold="valid", producer="render_validation", message="Existing ffprobe validation reported a warning.",
        )
    elif not status:
        finding(
            "QUALITY_CONFIDENCE_LOW", "warning", {"render_validation": None}, measured_value="unavailable",
            threshold="persisted ffprobe validation", producer="render_validation",
            message="ffprobe validation is unavailable for this render.",
        )


def _check_catalog(
    findings: list[QualityFinding],
    *,
    candidate: dict[str, Any],
    diversity_decision: dict[str, Any] | None,
    plan: dict[str, Any],
    render: dict[str, Any],
    audio: dict[str, Any],
    artifact: dict[str, Any],
    config_version: str,
) -> list[dict[str, Any]]:
    """Persist pass-state provenance too, without turning it into a finding."""

    composition_quality = _composition_quality(render)
    composition_evidence = render.get("composition_plan") or render.get("composition")
    composition_producer = (
        "composition_quality_report" if composition_quality is not None
        else "composition_quality_decision"
    )
    source_broll_quality = _source_broll_quality(render)
    motion_quality = _motion_quality(render)
    caption_quality = _caption_quality(render)
    native_captions = (
        render.get("compatibility_mode") == "native"
        and caption_quality is not None
    )
    subtitle_producer = (
        "caption_quality_report" if native_captions else "subtitle_quality_decision"
    )
    sources = [
        ("ELIGIBILITY", "eligibility", candidate.get("eligibility_decision"), "eligible=true"),
        ("DIVERSITY", "diversity", diversity_decision, "candidate selected by versioned diversity decision"),
        ("BOUNDARIES", "boundary_decision", plan.get("boundary_decision"), "safe complete boundary"),
        (
            "CONTINUITY", "continuity_decision", plan.get("continuity_decision"),
            "every omitted non-dialogue span has a typed rationale and required spans survive",
        ),
        (
            "SEMANTIC_CONTENT", "semantic_content_quality", plan.get("dialogue_mappings"),
            "every published dialogue mapping has grounded confidence >= 0.5",
        ),
        ("PLAN_IDENTITY", "production_plan_envelope", plan.get("envelope"), "plan parents match output"),
        (
            "COMPOSITION", composition_producer, composition_evidence,
            "composition decision passed",
        ),
        (
            "SOURCE_BROLL", "source_broll_quality_report", source_broll_quality,
            "evidence relevance and forbidden insertion checks passed",
        ),
        (
            "EDITORIAL_MOTION", "motion_quality_report", motion_quality,
            "registry, cooldown, concurrency, readability and animation budget passed",
        ),
        (
            "SUBTITLES", subtitle_producer,
            render.get("caption_plan") or render.get("subtitle_layout"),
            "subtitle and semantic caption decisions passed",
        ),
        (
            "SEMANTIC_CAPTIONS", "caption_quality_report", caption_quality,
            "readability, timing, safe zones and protected-region overlap passed",
        ),
        ("AUDIO", "audio_validation", audio.get("validation"), "audio validation valid"),
        ("FFPROBE", "render_validation", render.get("validation"), "ffprobe validation valid"),
        ("ARTIFACT_IDENTITY", "artifact_identity", artifact, "canonical bytes and parents match"),
    ]
    checks: list[dict[str, Any]] = []
    for code, producer, evidence, threshold in sources:
        related = [item for item in findings if item.provenance.get("producer") == producer]
        severity = "blocker" if any(item.severity == "blocker" for item in related) else (
            "warning" if related else "none"
        )
        checks.append({
            "code": code,
            "severity": severity,
            "status": "blocked" if severity == "blocker" else "warning" if severity == "warning" else "passed",
            "evidence": evidence,
            "measured_value": [item.code for item in related] if related else "passed",
            "threshold": threshold,
            "provenance": {"producer": producer, "config_version": config_version},
        })
    return checks


def _caption_quality(render: dict[str, Any]) -> dict[str, Any] | None:
    """Read the 7C report from either compiled-plan or render-report shape."""

    caption_plan = render.get("caption_plan")
    if isinstance(caption_plan, dict) and isinstance(caption_plan.get("quality_report"), dict):
        return dict(caption_plan["quality_report"])
    subtitle_layout = render.get("subtitle_layout")
    if isinstance(subtitle_layout, dict) and isinstance(subtitle_layout.get("caption_quality_report"), dict):
        return dict(subtitle_layout["caption_quality_report"])
    return None


def _composition_quality(render: dict[str, Any]) -> dict[str, Any] | None:
    """Read the 7D report from a compiled plan or render-report shape."""

    composition_plan = render.get("composition_plan")
    if isinstance(composition_plan, dict) and isinstance(composition_plan.get("quality_report"), dict):
        return dict(composition_plan["quality_report"])
    composition = render.get("composition")
    if isinstance(composition, dict) and isinstance(composition.get("quality_report"), dict):
        return dict(composition["quality_report"])
    return None


def _source_broll_quality(render: dict[str, Any]) -> dict[str, Any] | None:
    """Read the 7E report from a compiled-plan or render-report shape."""

    plan = render.get("source_broll_plan")
    if isinstance(plan, dict) and isinstance(plan.get("quality_report"), dict):
        return dict(plan["quality_report"])
    return None


def _motion_quality(render: dict[str, Any]) -> dict[str, Any] | None:
    """Read the 7F report from a compiled-plan or render-report shape."""

    plan = render.get("motion_plan")
    if isinstance(plan, dict) and isinstance(plan.get("quality_report"), dict):
        return dict(plan["quality_report"])
    return None


def _composition_fallbacks(render: dict[str, Any]) -> list[str]:
    composition_plan = render.get("composition_plan")
    if not isinstance(composition_plan, dict):
        return []
    raw_segments = composition_plan.get("segments")
    segments: list[Any] = raw_segments if isinstance(raw_segments, list) else []
    return [
        f"composition:{item.get('segment_id')}:{item.get('fallback')}"
        for item in segments
        if isinstance(item, dict) and item.get("fallback") not in {None, "none"}
    ]


def _source_broll_fallbacks(render: dict[str, Any]) -> list[str]:
    plan = render.get("source_broll_plan")
    if not isinstance(plan, dict):
        return []
    report = plan.get("quality_report")
    raw_findings = report.get("findings") if isinstance(report, dict) else None
    findings: list[Any] = raw_findings if isinstance(raw_findings, list) else []
    return [
        f"source_broll:{item.get('decision_id')}:a_roll_current_composition"
        for item in findings if isinstance(item, dict) and item.get("decision_id")
    ]


def _motion_fallbacks(render: dict[str, Any]) -> list[str]:
    plan = render.get("motion_plan")
    if not isinstance(plan, dict):
        return []
    report = plan.get("quality_report")
    raw_findings = report.get("findings") if isinstance(report, dict) else None
    findings: list[Any] = raw_findings if isinstance(raw_findings, list) else []
    return [
        f"motion:{item.get('event_id')}:calm_fallback"
        for item in findings if isinstance(item, dict) and item.get("event_id")
    ]


def _boundary_interval(boundary: dict[str, Any]) -> dict[str, float] | None:
    raw = boundary.get("allowed_source_range")
    if not isinstance(raw, dict):
        return None
    try:
        return {"start_seconds": float(raw["start_seconds"]), "end_seconds": float(raw["end_seconds"])}
    except (KeyError, TypeError, ValueError):
        return None


def _segment_interval(segment: dict[str, Any]) -> dict[str, float] | None:
    try:
        return {"start_seconds": float(segment["start_seconds"]), "end_seconds": float(segment["end_seconds"])}
    except (KeyError, TypeError, ValueError):
        return None


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
