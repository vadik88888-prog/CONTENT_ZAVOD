from __future__ import annotations

"""Versioned hand-off contract between source analysis and selected rendering.

The artifact deliberately keeps large source-derived data in the existing work
cache.  It contains the immutable identifiers, a compact review payload and
references needed to load the full scored candidates again for rendering.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils import read_json, utc_now, write_json


ANALYSIS_ARTIFACT_SCHEMA_VERSION = "1.0"


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
    status: str = "analysis_ready"
    schema_version: str = ANALYSIS_ARTIFACT_SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.schema_version != ANALYSIS_ARTIFACT_SCHEMA_VERSION:
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

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "AnalysisArtifact":
        raw = read_json(path, None)
        if not isinstance(raw, dict):
            raise AnalysisArtifactError("Analysis artifact file is missing or corrupted.")
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
            candidates=[dict(item) for item in raw.get("candidates", []) if isinstance(item, dict)],
            recommendation=dict(raw.get("recommendation") or {}),
            summary=dict(raw.get("summary") or {}),
            content_profile=dict(raw.get("content_profile") or {}),
            duration_seconds=_optional_float(raw.get("duration_seconds")),
            status=str(raw.get("status") or ""),
            schema_version=str(raw.get("schema_version") or ""),
            warnings=[str(item) for item in raw.get("warnings", [])],
        )
        artifact.validate()
        return artifact


def candidate_review_payload(candidate: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    """Expose only review-relevant fields; full evidence remains in cache."""

    candidate_id = str(candidate.get("id") or "")
    viral = candidate.get("virality") if isinstance(candidate.get("virality"), dict) else {}
    potential = viral.get("viral_potential") if isinstance(viral.get("viral_potential"), dict) else {}
    eligibility = viral.get("eligibility") if isinstance(viral.get("eligibility"), dict) else {}
    feature_profile = viral.get("feature_profile") if isinstance(viral.get("feature_profile"), dict) else {}
    feature_values = feature_profile.get("features") if isinstance(feature_profile.get("features"), dict) else {}
    confidence_data = potential.get("confidence") if isinstance(potential.get("confidence"), dict) else {}
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
    boundary = candidate.get("boundary_diagnostics") if isinstance(candidate.get("boundary_diagnostics"), dict) else {}
    selected = candidate_id in selected_ids
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
        },
        "state": "analyzed",
        "selected_by_recommendation": selected,
        "recommendation_status": "recommended" if selected else "not_recommended",
        "virality_level": potential.get("level") or potential_level,
        "publishability_status": eligibility.get("status"),
        "warnings": [*list(candidate.get("warnings") or []), *risks],
    }


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
    vector = candidate.get("feature_vector") if isinstance(candidate.get("feature_vector"), dict) else {}
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
    diagnostics = candidate.get("boundary_diagnostics") if isinstance(candidate.get("boundary_diagnostics"), dict) else {}
    if diagnostics.get("fallback_reason"):
        risks.append(str(diagnostics["fallback_reason"]))
    vector = candidate.get("feature_vector") if isinstance(candidate.get("feature_vector"), dict) else {}
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


def new_analysis_artifact(**kwargs: Any) -> AnalysisArtifact:
    """Small constructor boundary that keeps the timestamp creation consistent."""

    return AnalysisArtifact(created_at=utc_now(), **kwargs)
