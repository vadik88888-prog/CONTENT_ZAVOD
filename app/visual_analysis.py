"""Optional, bounded visual analysis used only to improve vertical reframing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import AppConfig


VISUAL_ANALYSIS_SCHEMA_VERSION = "5D.0"
_TRACKING_TARGETS = {
    "primary_face", "primary_person", "important_object", "screen_region",
    "subject_group", "scene_center", "none",
}
_SCENE_TYPES = {
    "TALKING_HEAD", "INTERVIEW_SINGLE", "INTERVIEW_MULTI", "PODCAST",
    "PRODUCT_DEMO", "HANDS_ON_DEMO", "PRESENTATION_SCREEN", "GAMEPLAY",
    "CINEMATIC_SCENE", "FULL_BODY_ACTION", "UNKNOWN",
}
_FRAMING_OBSERVATIONS = {
    "head_only", "head_shoulders", "chest_up", "upper_body", "full_body",
    "object", "screen", "unknown",
}


class VisualAnalysisSchemaError(ValueError):
    """The provider response did not satisfy the strict visual-analysis contract."""


def _analysis_result(
    *, status: str, evidence_status: str, reason: str | None = None,
    subject_keyframes: list[dict[str, Any]] | None = None, sample_count: int = 0,
    fallback_stage: str | None = None,
) -> dict[str, Any]:
    """Persist explicit evidence state for every optional visual-analysis outcome."""

    result: dict[str, Any] = {
        "schema_version": VISUAL_ANALYSIS_SCHEMA_VERSION,
        "enabled": True,
        "status": status,
        "evidence_status": evidence_status,
        "subject_keyframes": subject_keyframes or [],
        "sample_count": sample_count,
    }
    if reason:
        result["reason"] = reason
    if fallback_stage:
        result["fallback_provenance"] = {"stage": fallback_stage, "reason": reason or "unspecified"}
    return result


def analyse_video_subjects(source: Path, duration_seconds: float, config: AppConfig) -> dict[str, Any]:
    """Return the legacy local-evidence hand-off without making a paid call.

    Goal 6B centralizes every new frame request in ``VisionGateway`` after the
    6A.1 timeline has selected sparse keyframes.  Keeping this compatibility
    artifact avoids changing existing composition/render contracts while
    preventing the former pre-timeline provider path from bypassing budgets or
    charging for duplicate arbitrary samples.
    """

    if not config.optional_visual_features:
        result = _analysis_result(
            status="skipped", evidence_status="evidence_unavailable", reason="disabled",
            fallback_stage="configuration",
        )
        result["enabled"] = False
        return result
    return _analysis_result(
        status="fallback",
        evidence_status="fallback",
        reason="delegated_to_budgeted_vision_gateway",
        fallback_stage="vision_gateway",
    )


_SUBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "time_seconds": {"type": "number", "minimum": 0},
                    "normalized_x": {"type": "number", "minimum": 0, "maximum": 1},
                    "normalized_y": {"type": "number", "minimum": 0, "maximum": 1},
                    "normalized_width": {"type": "number", "minimum": 0.02, "maximum": 1},
                    "normalized_height": {"type": "number", "minimum": 0.02, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tracking_target": {
                        "type": "string",
                        "enum": [
                            "primary_face", "primary_person", "important_object", "screen_region",
                            "subject_group", "scene_center", "none",
                        ],
                    },
                    "visible_face_count": {"type": "integer", "minimum": 0, "maximum": 32},
                    "active_speaker_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "scene_id": {"type": "string", "maxLength": 160},
                    "scene_type": {"type": "string", "enum": sorted(_SCENE_TYPES)},
                    "framing_observation": {"type": "string", "enum": sorted(_FRAMING_OBSERVATIONS)},
                    "eye_line_y": {"type": "number", "minimum": 0, "maximum": 1},
                    "gesture_active": {"type": "boolean"},
                    "gesture_area_visible": {"type": "boolean"},
                },
                "required": [
                    "time_seconds", "normalized_x", "normalized_y", "normalized_width", "normalized_height",
                    "confidence", "tracking_target", "visible_face_count", "active_speaker_confidence", "scene_id",
                    "scene_type", "framing_observation", "eye_line_y", "gesture_active", "gesture_area_visible",
                ],
            },
        },
    },
    "required": ["subjects"],
}

_STRICT_SUBJECT_FIELDS = frozenset(_SUBJECT_SCHEMA["properties"]["subjects"]["items"]["properties"])


def _validate_subject_response(value: Any) -> list[dict[str, Any]]:
    """Validate the provider payload before a permissive legacy adapter sees it.

    Responses strict JSON schema needs every property of a strict object to be
    required.  The local check preserves that invariant even when a test double
    or a provider SDK returns parsed JSON without enforcing it.
    """

    if not isinstance(value, dict) or set(value) != {"subjects"} or not isinstance(value["subjects"], list):
        raise VisualAnalysisSchemaError("visual response must contain only a subjects array")
    keyframes: list[dict[str, Any]] = []
    for item in value["subjects"]:
        if not isinstance(item, dict) or set(item) != _STRICT_SUBJECT_FIELDS:
            raise VisualAnalysisSchemaError("visual subject does not satisfy the strict schema")
        if not _strict_subject_types(item):
            raise VisualAnalysisSchemaError("visual subject does not use strict JSON value types")
        keyframe = _keyframe(item, require_complete=True)
        if keyframe is None:
            raise VisualAnalysisSchemaError("visual subject contains invalid bounded evidence")
        keyframes.append(keyframe)
    return keyframes


def _strict_subject_types(item: dict[str, Any]) -> bool:
    """Mirror JSON-schema primitive types before legacy numeric coercion."""

    number_fields = {
        "time_seconds", "normalized_x", "normalized_y", "normalized_width", "normalized_height",
        "confidence", "active_speaker_confidence", "eye_line_y",
    }
    if any(isinstance(item[name], bool) or not isinstance(item[name], (int, float)) for name in number_fields):
        return False
    if isinstance(item["visible_face_count"], bool) or not isinstance(item["visible_face_count"], int):
        return False
    if not isinstance(item["scene_id"], str) or len(item["scene_id"]) > 160:
        return False
    if item["tracking_target"] not in _TRACKING_TARGETS or item["scene_type"] not in _SCENE_TYPES:
        return False
    if item["framing_observation"] not in _FRAMING_OBSERVATIONS:
        return False
    return isinstance(item["gesture_active"], bool) and isinstance(item["gesture_area_visible"], bool)


def _sample_times(duration: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    count = min(8, max(2, round(duration / 45)))
    return [round(duration * (index + 1) / (count + 1), 3) for index in range(count)]


def _keyframe(value: dict[str, Any], *, require_complete: bool = False) -> dict[str, Any] | None:
    try:
        result = {name: float(value[name]) for name in ("time_seconds", "normalized_x", "normalized_y", "confidence")}
    except (KeyError, TypeError, ValueError):
        return None
    if result["time_seconds"] < 0 or not 0 <= result["normalized_x"] <= 1 or not 0 <= result["normalized_y"] <= 1 or not 0 <= result["confidence"] <= 1:
        return None
    optional_fields = (
        "normalized_width", "normalized_height", "active_speaker_confidence", "tracking_target",
        "visible_face_count", "scene_id", "scene_type", "framing_observation", "eye_line_y",
        "gesture_active", "gesture_area_visible",
    )
    if require_complete and any(name not in value for name in optional_fields):
        return None
    for name in ("normalized_width", "normalized_height", "active_speaker_confidence", "eye_line_y"):
        if name not in value:
            continue
        try:
            parsed = float(value[name])
        except (TypeError, ValueError):
            if require_complete:
                return None
            continue
        if (
            (name in {"active_speaker_confidence", "eye_line_y"} and 0 <= parsed <= 1)
            or (name not in {"active_speaker_confidence", "eye_line_y"} and 0.02 <= parsed <= 1)
        ):
            result[name] = parsed
        elif require_complete:
            return None
    target = value.get("tracking_target")
    if target in _TRACKING_TARGETS:
        result["tracking_target"] = str(target)
    elif require_complete:
        return None
    try:
        face_count = int(value.get("visible_face_count", 1))
    except (TypeError, ValueError):
        if require_complete:
            return None
        face_count = 1
    if 0 <= face_count <= 32:
        result["visible_face_count"] = face_count
    elif require_complete:
        return None
    if isinstance(value.get("scene_id"), str) and value["scene_id"].strip():
        result["scene_id"] = value["scene_id"].strip()[:160]
    elif require_complete and not isinstance(value.get("scene_id"), str):
        return None
    scene_type = value.get("scene_type")
    if scene_type in _SCENE_TYPES:
        result["scene_type"] = str(scene_type)
    elif require_complete:
        return None
    framing = value.get("framing_observation")
    if framing in _FRAMING_OBSERVATIONS:
        result["framing_observation"] = str(framing)
    elif require_complete:
        return None
    for name in ("gesture_active", "gesture_area_visible"):
        if isinstance(value.get(name), bool):
            result[name] = value[name]
        elif require_complete:
            return None
    return result
