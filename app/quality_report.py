"""Persisted Final Quality Gate aggregation for canonical V2 artifacts.

This module deliberately consumes already-persisted decisions and validation
results.  It does not run ffprobe, inspect frames, or retry rendering: the
existing producers remain the owners of those low-level checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from app.clip_results import ClipResult
from app.utils import stable_file_hash, stable_text_hash, utc_now


QUALITY_REPORT_SCHEMA_VERSION = "5G.0"
QUALITY_STATUSES = frozenset({"PASS", "PASS_WITH_WARNINGS", "BLOCKED"})
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
    }
    fallbacks = _unique([
        *(render.get("fallback_reasons", []) if isinstance(render.get("fallback_reasons"), list) else []),
        *(render.get("warnings", []) if isinstance(render.get("warnings"), list) else []),
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
                "composition_quality_decision", "subtitle_quality_decision", "audio_validation",
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


def _collect_plan_and_boundary(
    finding: Any,
    plan: dict[str, Any],
    result: ClipResult,
    project_id: str | None,
    source_id: str,
) -> None:
    envelope = plan.get("envelope") if isinstance(plan, dict) else None
    if not isinstance(envelope, dict) or envelope.get("compatibility_mode") == "legacy_adapter":
        finding(
            "QUALITY_CONFIDENCE_LOW", "warning", {"plan_envelope": envelope or None},
            measured_value="legacy_adapter", threshold="native ProductionPlan envelope",
            producer="production_plan", message="Production plan uses an explicit legacy compatibility contract.",
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


def _collect_composition_and_subtitles(finding: Any, render: dict[str, Any]) -> None:
    raw_quality = render.get("quality")
    quality: dict[str, Any] = raw_quality if isinstance(raw_quality, dict) else {}
    raw_composition = render.get("composition")
    composition: dict[str, Any] = raw_composition if isinstance(raw_composition, dict) else {}
    raw_segments = composition.get("segments")
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
    raw_subtitle_layout = render.get("subtitle_layout")
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
    if caption is not None:
        raw_caption_findings = caption.get("findings")
        caption_findings: list[Any] = raw_caption_findings if isinstance(raw_caption_findings, list) else []
        for item in caption_findings:
            if not isinstance(item, dict):
                continue
            severity = "blocker" if item.get("severity") == "blocker" else "warning"
            code = str(item.get("code") or "CAPTION_QUALITY_DEGRADED")
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
    if quality.get("status") == "failed" and not segments and subtitle is None:
        finding(
            "MEDIA_INVALID", "blocker", {"output_quality": quality}, measured_value=quality.get("errors", []),
            threshold="existing output quality validation passed", producer="output_quality",
            message="Existing final output quality validation failed.",
        )


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

    sources = [
        ("ELIGIBILITY", "eligibility", candidate.get("eligibility_decision"), "eligible=true"),
        ("DIVERSITY", "diversity", diversity_decision, "candidate selected by versioned diversity decision"),
        ("BOUNDARIES", "boundary_decision", plan.get("boundary_decision"), "safe complete boundary"),
        ("PLAN_IDENTITY", "production_plan_envelope", plan.get("envelope"), "plan parents match output"),
        ("COMPOSITION", "composition_quality_decision", render.get("composition"), "composition decision passed"),
        (
            "SUBTITLES", "subtitle_quality_decision",
            render.get("caption_plan") or render.get("subtitle_layout"),
            "subtitle and semantic caption decisions passed",
        ),
        (
            "SEMANTIC_CAPTIONS", "caption_quality_report", _caption_quality(render),
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
