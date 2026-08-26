from __future__ import annotations

"""Versioned hand-off contract between source analysis and selected rendering.

The artifact deliberately keeps large source-derived data in the existing work
cache.  It contains the immutable identifiers, a compact review payload and
references needed to load the full scored candidates again for rendering.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.candidate_quality import EligibilityDecision, legacy_eligibility_decision
from app.editorial_profile_policy import (
    CandidateEditorialDecision,
    evaluate_editorial_candidate,
)
from app.utils import read_json, stable_file_hash, utc_now, write_json


ANALYSIS_ARTIFACT_SCHEMA_VERSION = "1.1"
LEGACY_ANALYSIS_ARTIFACT_SCHEMA_VERSION = "1.0"
LEGACY_ANALYSIS_WARNING = (
    "LEGACY_ANALYSIS_ARTIFACT_1_0: references are mutable and do not carry producer-bound integrity metadata."
)
ANALYSIS_INTEGRITY_SCHEMA_VERSION = "analysis-integrity.1"


class AnalysisArtifactError(ValueError):
    """Raised when an analysis hand-off cannot be trusted."""


@dataclass(slots=True)
class AnalysisArtifact:
    analysis_id: str
    project_id: str | None
    created_at: str
    source: dict[str, Any]
    source_fingerprint: str
    analysis_fingerprint: str
    work_directory: str
    candidate_data_ref: str
    references: dict[str, str]
    candidates: list[dict[str, Any]]
    recommendation: dict[str, Any]
    summary: dict[str, Any]
    content_profile: dict[str, Any]
    duration_seconds: float | None
    analysis_run_id: str = ""
    snapshot_directory: str = ""
    reference_integrity: dict[str, dict[str, Any]] = field(default_factory=dict)
    producer: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    recommended_count: dict[str, int] = field(default_factory=dict)
    status: str = "analysis_ready"
    schema_version: str = ANALYSIS_ARTIFACT_SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)
    verified_sha256: str = field(default="", init=False, repr=False, compare=False)
    verified_byte_size: int = field(default=0, init=False, repr=False, compare=False)

    def validate(self) -> None:
        if self.schema_version not in {
            ANALYSIS_ARTIFACT_SCHEMA_VERSION, LEGACY_ANALYSIS_ARTIFACT_SCHEMA_VERSION,
        }:
            raise AnalysisArtifactError("Unsupported analysis artifact schema.")
        if self.status != "analysis_ready":
            raise AnalysisArtifactError("Analysis artifact is not ready for rendering.")
        required = (self.analysis_id, self.source_fingerprint, self.analysis_fingerprint, self.work_directory, self.candidate_data_ref)
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise AnalysisArtifactError("Analysis artifact is missing required identifiers.")
        if not isinstance(self.source, dict) or not str(self.source.get("id") or "").strip():
            raise AnalysisArtifactError("Analysis artifact does not identify its source.")
        if not isinstance(self.candidates, list):
            raise AnalysisArtifactError("Analysis artifact candidates are invalid.")
        if self.candidate_count < 0 or self.candidate_count != len(self.candidates):
            raise AnalysisArtifactError("Analysis artifact candidate count is invalid.")
        if any(
            key not in {"min", "max", "default"} or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.recommended_count.items()
        ):
            raise AnalysisArtifactError("Analysis artifact recommendation range is invalid.")
        if self.schema_version == ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            self._validate_v11()

    def _validate_v11(self) -> None:
        if not self.analysis_run_id.strip() or not self.snapshot_directory.strip():
            raise AnalysisArtifactError("Analysis artifact is missing immutable run lineage.")
        if not isinstance(self.producer, dict) or not all(
            str(self.producer.get(key) or "").strip() for key in ("name", "version", "analysis_run_id")
        ):
            raise AnalysisArtifactError("Analysis artifact producer metadata is invalid.")
        if str(self.producer.get("analysis_run_id")) != self.analysis_run_id:
            raise AnalysisArtifactError("Analysis artifact producer run identity mismatch.")
        if "final_selection" not in self.references or "candidate_data" not in self.references:
            raise AnalysisArtifactError("Analysis artifact snapshot is missing required final references.")
        if self.references.get("candidate_data") != self.candidate_data_ref:
            raise AnalysisArtifactError("Analysis artifact candidate reference is not snapshot-bound.")
        if set(self.reference_integrity) != set(self.references):
            raise AnalysisArtifactError("Analysis artifact integrity manifest does not cover every reference.")
        for name, descriptor in self.reference_integrity.items():
            if not isinstance(descriptor, dict):
                raise AnalysisArtifactError(f"Analysis reference integrity is invalid: {name}.")
            checksum = str(descriptor.get("sha256") or "")
            byte_size = descriptor.get("byte_size")
            producer = descriptor.get("producer")
            if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
                raise AnalysisArtifactError(f"Analysis reference checksum is invalid: {name}.")
            if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
                raise AnalysisArtifactError(f"Analysis reference byte size is invalid: {name}.")
            if not isinstance(producer, dict) or not all(
                str(producer.get(key) or "").strip() for key in ("name", "version", "analysis_run_id")
            ):
                raise AnalysisArtifactError(f"Analysis reference producer metadata is invalid: {name}.")
            if str(producer.get("analysis_run_id")) != self.analysis_run_id:
                raise AnalysisArtifactError(f"Analysis reference producer run mismatch: {name}.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data.pop("verified_sha256", None)
        data.pop("verified_byte_size", None)
        return data

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())

    def write_with_integrity(self, path: Path) -> str:
        """Write the hand-off plus a producer-bound checksum sidecar."""

        if self.schema_version != ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            raise AnalysisArtifactError("Only AnalysisArtifact 1.1 can emit an integrity sidecar.")
        data = self.to_dict()
        if path.exists():
            if read_json(path, None) != data:
                raise AnalysisArtifactError("Immutable analysis artifact already exists with different content.")
        else:
            write_json(path, data)
        checksum = stable_file_hash(path)
        byte_size = path.stat().st_size
        sidecar = analysis_integrity_path(path)
        payload = {
            "schema_version": ANALYSIS_INTEGRITY_SCHEMA_VERSION,
            "analysis_id": self.analysis_id,
            "analysis_run_id": self.analysis_run_id,
            "analysis_artifact_sha256": checksum,
            "byte_size": byte_size,
            "producer": dict(self.producer),
        }
        if sidecar.exists():
            if read_json(sidecar, None) != payload:
                raise AnalysisArtifactError("Immutable analysis integrity sidecar already exists with different content.")
        else:
            write_json(sidecar, payload)
        self.verified_sha256 = checksum
        self.verified_byte_size = byte_size
        return checksum

    @classmethod
    def read(cls, path: Path) -> "AnalysisArtifact":
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            raise AnalysisArtifactError("Analysis artifact file is missing or corrupted.")
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> "AnalysisArtifact":
        artifact = cls(
            analysis_id=str(raw.get("analysis_id") or ""),
            project_id=str(raw["project_id"]) if raw.get("project_id") else None,
            created_at=str(raw.get("created_at") or ""),
            source=dict(raw.get("source") or {}),
            source_fingerprint=str(raw.get("source_fingerprint") or ""),
            analysis_fingerprint=str(raw.get("analysis_fingerprint") or ""),
            work_directory=str(raw.get("work_directory") or ""),
            candidate_data_ref=str(raw.get("candidate_data_ref") or ""),
            references={str(key): str(value) for key, value in dict(raw.get("references") or {}).items()},
            candidates=[
                _moments_review_candidate(item)
                for item in raw.get("candidates", []) if isinstance(item, dict)
            ],
            recommendation=dict(raw.get("recommendation") or {}),
            summary=dict(raw.get("summary") or {}),
            content_profile=dict(raw.get("content_profile") or {}),
            duration_seconds=_optional_float(raw.get("duration_seconds")),
            analysis_run_id=str(raw.get("analysis_run_id") or ""),
            snapshot_directory=str(raw.get("snapshot_directory") or ""),
            reference_integrity={
                str(key): dict(value) for key, value in dict(raw.get("reference_integrity") or {}).items()
                if isinstance(value, dict)
            },
            producer=dict(raw.get("producer") or {}),
            candidate_count=_candidate_count(raw),
            recommended_count=_recommended_count(raw),
            status=str(raw.get("status") or ""),
            schema_version=str(raw.get("schema_version") or ""),
            warnings=[str(item) for item in raw.get("warnings", [])],
        )
        if artifact.schema_version == LEGACY_ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            _append_warning(artifact.warnings, LEGACY_ANALYSIS_WARNING)
        artifact.validate()
        return artifact

    @classmethod
    def read_verified(cls, path: Path, *, expected_sha256: str | None = None) -> "AnalysisArtifact":
        """Read only after checksum verification, then verify every v1.1 snapshot member."""

        if not path.is_file():
            raise AnalysisArtifactError("Analysis artifact file is missing or corrupted.")
        try:
            payload = path.read_bytes()
            raw = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AnalysisArtifactError("Analysis artifact file is missing or corrupted.") from error
        if not isinstance(raw, dict):
            raise AnalysisArtifactError("Analysis artifact file is missing or corrupted.")
        checksum = hashlib.sha256(payload).hexdigest()
        byte_size = len(payload)
        artifact = cls._from_raw(raw)
        if expected_sha256 is not None and checksum != expected_sha256:
            raise AnalysisArtifactError("ANALYSIS_INTEGRITY_MISMATCH: analysis artifact checksum mismatch.")
        if artifact.schema_version == ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            try:
                sidecar = read_json(analysis_integrity_path(path), None)
            except (OSError, json.JSONDecodeError) as error:
                raise AnalysisArtifactError(
                    "ANALYSIS_INTEGRITY_MISMATCH: analysis integrity sidecar is corrupted."
                ) from error
            if not isinstance(sidecar, dict):
                raise AnalysisArtifactError("ANALYSIS_INTEGRITY_MISMATCH: analysis integrity sidecar is missing.")
            if (
                sidecar.get("schema_version") != ANALYSIS_INTEGRITY_SCHEMA_VERSION
                or sidecar.get("analysis_id") != artifact.analysis_id
                or sidecar.get("analysis_run_id") != artifact.analysis_run_id
                or sidecar.get("analysis_artifact_sha256") != checksum
                or sidecar.get("byte_size") != byte_size
                or sidecar.get("producer") != artifact.producer
            ):
                raise AnalysisArtifactError("ANALYSIS_INTEGRITY_MISMATCH: analysis integrity sidecar is invalid.")
            artifact.verify_snapshot()
        else:
            _append_warning(
                artifact.warnings,
                "LEGACY_ANALYSIS_CHECKSUM_ONLY: checksum was established at read time; no immutable snapshot manifest exists.",
            )
        artifact.verified_sha256 = checksum
        artifact.verified_byte_size = byte_size
        return artifact

    def verify_snapshot(self) -> None:
        if self.schema_version != ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            return
        root = Path(self.snapshot_directory).resolve()
        if not root.is_dir():
            raise AnalysisArtifactError("ANALYSIS_INTEGRITY_MISMATCH: immutable snapshot directory is missing.")
        for name, raw_path in self.references.items():
            path = Path(raw_path).resolve()
            descriptor = self.reference_integrity[name]
            if not path.is_relative_to(root) or not path.is_file():
                raise AnalysisArtifactError(f"ANALYSIS_INTEGRITY_MISMATCH: unsafe or missing snapshot reference: {name}.")
            try:
                matches = path.stat().st_size == descriptor["byte_size"] and stable_file_hash(path) == descriptor["sha256"]
            except OSError as error:
                raise AnalysisArtifactError(
                    f"ANALYSIS_INTEGRITY_MISMATCH: snapshot reference is unreadable: {name}."
                ) from error
            if not matches:
                raise AnalysisArtifactError(f"ANALYSIS_INTEGRITY_MISMATCH: snapshot reference changed: {name}.")

    def load_reference(self, name: str) -> dict[str, Any]:
        raw_path = self.references.get(name)
        if not raw_path:
            raise AnalysisArtifactError(f"Analysis artifact is missing its {name} reference.")
        path = Path(raw_path).resolve()
        root = Path(
            self.snapshot_directory if self.schema_version == ANALYSIS_ARTIFACT_SCHEMA_VERSION else self.work_directory
        ).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise AnalysisArtifactError(f"Analysis reference is unavailable or unsafe: {name}.")
        if self.schema_version == ANALYSIS_ARTIFACT_SCHEMA_VERSION:
            descriptor = self.reference_integrity.get(name, {})
            try:
                matches = path.stat().st_size == descriptor.get("byte_size") and stable_file_hash(path) == descriptor.get("sha256")
            except OSError as error:
                raise AnalysisArtifactError(f"Analysis reference is unreadable: {name}.") from error
            if not matches:
                raise AnalysisArtifactError(f"ANALYSIS_INTEGRITY_MISMATCH: snapshot reference changed: {name}.")
        try:
            value = read_json(path, None)
        except (OSError, json.JSONDecodeError) as error:
            raise AnalysisArtifactError(f"Analysis reference is corrupted: {name}.") from error
        if not isinstance(value, dict):
            raise AnalysisArtifactError(f"Analysis reference is corrupted: {name}.")
        return value


def candidate_review_payload(
    candidate: dict[str, Any],
    selected_ids: set[str],
    content_profile: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose only review-relevant fields; full evidence remains in cache."""

    candidate_id = str(candidate.get("id") or "")
    viral = _dict_value(candidate.get("virality"))
    potential = _dict_value(viral.get("viral_potential"))
    eligibility = _dict_value(viral.get("eligibility"))
    feature_profile = _dict_value(viral.get("feature_profile"))
    feature_values = _dict_value(feature_profile.get("features"))
    confidence_data = _dict_value(potential.get("confidence"))
    confidence = _confidence(candidate, confidence_data, feature_values)
    potential_level = _potential_level(candidate, potential)
    reasons = _review_reasons(candidate, viral, potential)
    risks = _review_risks(candidate, viral, potential, confidence_data)
    payoff = _payoff_summary(candidate)
    compact_features = {
        name: {
            "score": _optional_float(value.get("score")),
            "confidence": _optional_float(value.get("confidence")),
            "explanation": str(value.get("explanation") or ""),
        }
        for name, value in feature_values.items()
        if isinstance(value, dict) and name in {
            "hook_strength", "curiosity_gap", "payoff_strength", "momentum",
            "standalone_strength", "context_dependency", "visual", "audio",
        }
    }
    boundary = _dict_value(candidate.get("boundary_diagnostics"))
    selection_diagnostics = _dict_value(candidate.get("selection_diagnostics"))
    assessed_production_feasibility = _dict_value(
        selection_diagnostics.get("production_feasibility")
    )
    production_feasibility = _moments_production_feasibility(
        assessed_production_feasibility
    )
    eligibility_decision = _eligibility_decision(candidate.get("eligibility_decision"))
    ranking_decision = evaluate_editorial_candidate(
        candidate,
        content_profile,
        score=_optional_float(candidate.get("score")),
        confidence=confidence,
        production_feasibility=assessed_production_feasibility,
        source=source,
    )
    brain_recommended = (
        candidate_id in selected_ids
        or ranking_decision.surfacing_state.value == "RECOMMENDED"
    )
    editorial_decision = evaluate_editorial_candidate(
        candidate,
        content_profile,
        score=_optional_float(candidate.get("score")),
        confidence=confidence,
        recommended=brain_recommended,
        production_feasibility=production_feasibility,
        source=source,
    )
    start = _optional_float(candidate.get("start"))
    end = _optional_float(candidate.get("end"))
    duration = _optional_float(candidate.get("duration"))
    retention_score = _percent(potential.get("retention_potential_score"))
    publishability_score = _percent(potential.get("publishability_score"))
    viral_score = _percent(potential.get("score") if "score" in potential else viral.get("ranking_sort_score"))
    return {
        "candidate_id": candidate_id,
        "story_unit_id": candidate.get("story_unit_id"),
        "chapter_id": candidate.get("chapter_id"),
        "start": start,
        "end": end,
        "duration": duration,
        # Stable long-form names are the public review contract.  The short
        # aliases above remain readable by persisted pre-review projects.
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": duration,
        "title": str(candidate.get("title") or candidate.get("core_idea") or "Фрагмент видео"),
        "core_idea": str(candidate.get("core_idea") or candidate.get("summary") or ""),
        "hook_summary": str(candidate.get("hook") or ""),
        "payoff_summary": payoff,
        "summary": str(candidate.get("summary") or ""),
        "transcript_excerpt": str(candidate.get("text") or ""),
        "text": candidate.get("text", ""),
        "reason": candidate.get("reason", ""),
        "score": candidate.get("score"),
        "retention_score": retention_score,
        "publishability_score": publishability_score,
        "viral_score": viral_score,
        "confidence": confidence,
        "potential": potential_level,
        "reasons": reasons,
        "risks": risks,
        "feature_profile": {
            "content_strategy": feature_profile.get("content_strategy"),
            "strongest_factors": list(viral.get("strongest_factors") or []),
            "weakest_factors": list(viral.get("weakest_factors") or []),
            "features": compact_features,
        },
        # Keep the production-facing decision separate from the Moments
        # projection.  The latter may surface a weak candidate for source
        # review, but it must never grant permission to start a Draft.
        "eligibility_decision": eligibility_decision.to_dict(),
        "production_editorial_decision": ranking_decision.to_dict(),
        "editorial_decision": editorial_decision.to_dict(),
        "surfacing_state": editorial_decision.surfacing_state.value,
        "selectable": editorial_decision.selectable,
        "boundary_evidence": {
            "overall_boundary_score": boundary.get("overall_boundary_score"),
            "semantic_completion": boundary.get("semantic_completion"),
            "context_independence": boundary.get("context_independence"),
            "payoff_preserved": boundary.get("payoff_preserved"),
            "continuation_risk": boundary.get("continuation_risk"),
            "start_reason": candidate.get("start_boundary_reason"),
            "end_reason": candidate.get("end_boundary_reason"),
            "warnings": list(boundary.get("fallback_reason") and [boundary["fallback_reason"]] or []),
        },
        "preview": {
            "kind": "source_range",
            "start_seconds": start,
            "end_seconds": end,
            "requires_production_render": False,
            "thumbnail": {
                "kind": "lazy_source_frame",
                "timestamp_seconds": round(start + min(1.0, max(0.0, (end - start) / 2)) if start is not None and end is not None else 0.0, 3),
                "requires_production_render": False,
            },
        },
        "state": "analyzed",
        "recommended": editorial_decision.surfacing_state.value == "RECOMMENDED",
        "selected_by_recommendation": brain_recommended,
        "recommendation_status": editorial_decision.surfacing_state.value.lower(),
        "production_feasibility": production_feasibility,
        "virality_level": potential.get("level") or potential_level,
        "publishability_status": eligibility.get("status"),
        "warnings": [
            *list(candidate.get("warnings") or []),
            *risks,
            *(
                [str(production_feasibility.get("reason"))]
                if assessed_production_feasibility.get("status") == "GUARANTEED_BLOCKED" else []
            ),
        ],
    }


def candidate_is_draftable(candidate: object) -> bool:
    """Return whether a review candidate can safely enter the Draft workflow."""

    if not isinstance(candidate, dict):
        return False
    candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "").strip()
    if not candidate_id:
        return False
    try:
        start = float(
            candidate.get("start_seconds")
            if candidate.get("start_seconds") is not None else candidate.get("start")
        )
        end = float(
            candidate.get("end_seconds")
            if candidate.get("end_seconds") is not None else candidate.get("end")
        )
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start):
        return False
    production_decision = _production_editorial_decision(candidate)
    return bool(
        production_decision
        and production_decision.selectable
        and not production_decision.hard_blockers
    )


def _production_editorial_decision(candidate: dict[str, Any]) -> CandidateEditorialDecision | None:
    """Read the persisted production decision, deriving it for legacy reviews."""

    persisted = candidate.get("production_editorial_decision")
    if isinstance(persisted, dict):
        try:
            return CandidateEditorialDecision.from_dict(persisted)
        except (TypeError, ValueError):
            return None
    return evaluate_editorial_candidate(candidate, None)


def _moments_review_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    projected = dict(candidate)
    if (
        projected.get("recommended") is True
        or projected.get("surfacing_state") == "RECOMMENDED"
    ):
        projected["selected_by_recommendation"] = True
    feasibility = projected.get("production_feasibility")
    if isinstance(feasibility, dict):
        projected["production_feasibility"] = _moments_production_feasibility(feasibility)
    return projected


def _moments_production_feasibility(value: object) -> dict[str, Any]:
    """Keep feasibility evidence while removing its Moments permission effect."""

    projected = _dict_value(value)
    assessed_status = str(projected.get("diagnostic_status") or projected.get("status") or "")
    if assessed_status == "GUARANTEED_BLOCKED":
        projected["diagnostic_status"] = assessed_status
        projected["status"] = "ADVISORY"
        projected["selectability_effect"] = "ranking_and_warning_only"
    return projected


def _eligibility_decision(value: object) -> EligibilityDecision:
    if not isinstance(value, dict):
        return legacy_eligibility_decision()
    try:
        return EligibilityDecision.from_dict(value)
    except (TypeError, ValueError):
        return legacy_eligibility_decision()


def _dict_value(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def potential_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Return a stable UI summary without changing any ranking formula."""

    counts = {"high": 0, "medium": 0, "low": 0}
    for candidate in candidates:
        level = str(candidate.get("potential") or "low")
        counts[level if level in counts else "low"] += 1
    return counts


def _potential_level(candidate: dict[str, Any], potential: dict[str, Any]) -> str:
    raw = str(potential.get("level") or "").lower()
    if raw in {"excellent", "strong", "high"}:
        return "high"
    if raw in {"moderate", "medium"}:
        return "medium"
    if raw in {"weak", "low"}:
        return "low"
    score = _optional_float(candidate.get("score")) or 0.0
    return "high" if score >= 70 else "medium" if score >= 50 else "low"


def _confidence(candidate: dict[str, Any], confidence: dict[str, Any], features: dict[str, Any]) -> float:
    vector = _dict_value(candidate.get("feature_vector"))
    values = [_optional_float(confidence.get("score")), _optional_float(vector.get("transcript_confidence"))]
    for value in features.values():
        if isinstance(value, dict):
            values.append(_optional_float(value.get("confidence")))
    usable = [value for value in values if value is not None]
    return round(sum(usable) / len(usable), 3) if usable else 0.5


def _review_reasons(candidate: dict[str, Any], viral: dict[str, Any], potential: dict[str, Any]) -> list[str]:
    factors = [str(item) for item in viral.get("strongest_factors", []) if str(item)]
    explanations = [str(item) for item in candidate.get("explanations", []) if str(item)]
    selection = str(candidate.get("selection_reason") or "")
    known = {
        "hook": "Сильное начало.", "curiosity": "Есть интрига в начале.",
        "emotion": "Эмоциональная динамика поддерживает внимание.",
        "payoff": "Есть самостоятельный payoff.", "retention": "Хорошая внутренняя динамика.",
        "publishability": "Фрагмент пригоден к публикации.", "quotability": "Есть запоминающаяся фраза.",
        "usefulness": "Есть практическая ценность.", "momentum": "Мысль заметно развивается.",
    }
    values = [known[item] for item in factors if item in known]
    values.extend(explanations[:2])
    if selection:
        values.append(selection)
    return _unique(values)[:4] or ["Оценка основана на границах, качестве речи и структуре фрагмента."]


def _review_risks(candidate: dict[str, Any], viral: dict[str, Any], potential: dict[str, Any], confidence: dict[str, Any]) -> list[str]:
    values = [str(item) for item in viral.get("weakest_factors", []) if str(item)]
    readable = {
        "context_dependency": "Может требоваться дополнительный контекст.",
        "missing_payoff": "Payoff выражен неявно.", "weak_ending": "Финал может быть недостаточно сильным.",
        "slow_start": "Начало может быть недостаточно быстрым.",
    }
    risks = [readable.get(value, value.replace("_", " ")) for value in values]
    risks.extend(str(item) for item in confidence.get("warnings", []) if str(item))
    diagnostics = _dict_value(candidate.get("boundary_diagnostics"))
    if diagnostics.get("fallback_reason"):
        risks.append(str(diagnostics["fallback_reason"]))
    vector = _dict_value(candidate.get("feature_vector"))
    if (_optional_float(vector.get("transcript_confidence")) or 1.0) < 0.7:
        risks.append("Уверенность распознавания речи ниже обычной.")
    return _unique(risks)[:4]


def _payoff_summary(candidate: dict[str, Any]) -> str:
    text = str(candidate.get("summary") or candidate.get("text") or "").strip()
    if not text:
        return ""
    sentences = [part.strip() for part in text.replace("!", ".").replace("?", ".").split(".") if part.strip()]
    return sentences[-1] if sentences else text


def _percent(value: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    return round(numeric * 100, 1) if 0 <= numeric <= 1 else round(numeric, 1)


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def _candidate_count(raw: dict[str, Any]) -> int:
    value = raw.get("candidate_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    candidates = raw.get("candidates")
    return len(candidates) if isinstance(candidates, list) else 0


def _recommended_count(raw: dict[str, Any]) -> dict[str, int]:
    value = raw.get("recommended_count")
    if isinstance(value, dict):
        result = {
            str(key): int(item)
            for key, item in value.items()
            if str(key) in {"min", "max", "default"} and isinstance(item, int) and not isinstance(item, bool) and item >= 0
        }
        if result:
            return result
    recommendation = raw.get("recommendation")
    clip_count = recommendation.get("clip_count") if isinstance(recommendation, dict) else None
    interval = clip_count.get("estimated_publishable_clip_range") if isinstance(clip_count, dict) else None
    selected = recommendation.get("selected_candidate_ids") if isinstance(recommendation, dict) else []
    default = len(selected) if isinstance(selected, list) else 0
    if isinstance(interval, dict):
        lower = interval.get("min")
        upper = interval.get("max")
        return {
            "min": int(lower) if isinstance(lower, int) and lower >= 0 else default,
            "max": int(upper) if isinstance(upper, int) and upper >= 0 else default,
            "default": default,
        }
    return {"min": default, "max": default, "default": default}


def new_analysis_artifact(**kwargs: Any) -> AnalysisArtifact:
    """Small constructor boundary that keeps the timestamp creation consistent."""

    if not kwargs.get("analysis_run_id"):
        kwargs.setdefault("schema_version", LEGACY_ANALYSIS_ARTIFACT_SCHEMA_VERSION)
    return AnalysisArtifact(created_at=utc_now(), **kwargs)


def analysis_integrity_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.integrity.json")


def _append_warning(warnings: list[str], value: str) -> None:
    if value not in warnings:
        warnings.append(value)
