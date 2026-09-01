from __future__ import annotations

"""Strict, privacy-bounded contracts for local friend-beta feedback events."""

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


FEEDBACK_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_FORBIDDEN_IDENTIFIER_PARTS = ("api_key", "bearer", "transcript", "sk-")
_TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "event_id", "occurred_at", "session_id", "sequence",
    "domain", "name", "project_id", "analysis_id", "candidate_id",
    "draft_id", "run_id", "clip_result_id", "surface", "payload",
})


class FeedbackContractError(ValueError):
    """An event does not belong to the closed feedback schema."""


class FeedbackDomain(StrEnum):
    EDITORIAL = "editorial"
    CREATIVE = "creative"
    OUTCOME = "outcome"


class EditorialEventName(StrEnum):
    MOMENT_SHOWN = "moment_shown"
    MOMENT_PREVIEWED = "moment_previewed"
    MOMENT_SELECTED = "moment_selected"
    MOMENT_REJECTED = "moment_rejected"
    BOUNDARY_CHANGED = "boundary_changed"


class CreativeEventName(StrEnum):
    DRAFT_SHOWN = "draft_shown"
    DRAFT_PREVIEWED = "draft_previewed"
    DRAFT_APPROVED = "draft_approved"
    DRAFT_REJECTED = "draft_rejected"
    CREATIVE_OVERRIDE_CHANGED = "creative_override_changed"


class OutcomeEventName(StrEnum):
    FINAL_CREATED = "final_created"
    FINAL_SHOWN = "final_shown"
    FINAL_SELECTED = "final_selected"
    FINAL_OPEN_REQUESTED = "final_open_requested"
    FINAL_REVEAL_REQUESTED = "final_reveal_requested"
    FINAL_MARKED_USED = "final_marked_used"
    FINAL_EXPORTED = "final_exported"


class FeedbackSurface(StrEnum):
    SETUP = "setup"
    MOMENTS = "moments"
    DRAFTS = "drafts"
    FINAL = "final"


class EditorialRejectReason(StrEnum):
    WRONG_TOPIC = "wrong_topic"
    WEAK_HOOK = "weak_hook"
    NEEDS_CONTEXT = "needs_context"
    DUPLICATE = "duplicate"
    BAD_START = "bad_start"
    BAD_END = "bad_end"
    OTHER = "other"


class CreativeRejectReason(StrEnum):
    CAPTIONS = "captions"
    CROP = "crop"
    COMPOSITION = "composition"
    MOTION_PACING = "motion_pacing"
    BROLL = "broll"
    AUDIO = "audio"
    VISUAL_STYLE = "visual_style"
    QUALITY_WARNING = "quality_warning"
    OTHER = "other"


class BoundarySide(StrEnum):
    START = "start"
    END = "end"


class CreativeOverrideField(StrEnum):
    SUBTITLES_ENABLED = "subtitles_enabled"
    SUBTITLE_STYLE = "subtitle_style"
    SAME_SOURCE_BROLL_ALLOWED = "same_source_broll_allowed"
    AUDIO_MODE = "audio_mode"


class CreativeOverrideScope(StrEnum):
    PROJECT = "project"
    DRAFT = "draft"


class FinalExportMethod(StrEnum):
    COPY_TO = "copy_to"
    SYSTEM_SHARE = "system_share"


_EVENT_DOMAIN: dict[str, FeedbackDomain] = {
    **{item.value: FeedbackDomain.EDITORIAL for item in EditorialEventName},
    **{item.value: FeedbackDomain.CREATIVE for item in CreativeEventName},
    **{item.value: FeedbackDomain.OUTCOME for item in OutcomeEventName},
}
_EVENT_SURFACES: dict[str, frozenset[FeedbackSurface]] = {
    **{item.value: frozenset({FeedbackSurface.MOMENTS}) for item in EditorialEventName},
    **{item.value: frozenset({FeedbackSurface.DRAFTS}) for item in CreativeEventName},
    **{item.value: frozenset({FeedbackSurface.FINAL}) for item in OutcomeEventName},
}
_EVENT_SURFACES[CreativeEventName.CREATIVE_OVERRIDE_CHANGED.value] = frozenset({
    FeedbackSurface.SETUP, FeedbackSurface.DRAFTS,
})
_SHOWN_EVENTS = frozenset({
    EditorialEventName.MOMENT_SHOWN.value,
    CreativeEventName.DRAFT_SHOWN.value,
    OutcomeEventName.FINAL_SHOWN.value,
})
_EMPTY_PAYLOAD_EVENTS = frozenset({
    EditorialEventName.MOMENT_PREVIEWED.value,
    EditorialEventName.MOMENT_SELECTED.value,
    CreativeEventName.DRAFT_PREVIEWED.value,
    CreativeEventName.DRAFT_APPROVED.value,
    OutcomeEventName.FINAL_CREATED.value,
    OutcomeEventName.FINAL_SELECTED.value,
    OutcomeEventName.FINAL_OPEN_REQUESTED.value,
    OutcomeEventName.FINAL_REVEAL_REQUESTED.value,
    OutcomeEventName.FINAL_MARKED_USED.value,
})
_SUBTITLE_STYLES = frozenset({"documentary", "clean", "minimal", "dynamic"})
_AUDIO_MODES = frozenset({"original", "original_enhanced", "voiceover", "replace_voice", "mixed"})


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    schema_version: int
    event_id: str
    occurred_at: str
    session_id: str
    sequence: int
    domain: FeedbackDomain
    name: str
    project_id: str
    analysis_id: str | None = None
    candidate_id: str | None = None
    draft_id: str | None = None
    run_id: str | None = None
    clip_result_id: str | None = None
    surface: FeedbackSurface = FeedbackSurface.MOMENTS
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_feedback_event(self)
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "occurred_at": self.occurred_at,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "domain": self.domain.value,
            "name": self.name,
            "project_id": self.project_id,
            "analysis_id": self.analysis_id,
            "candidate_id": self.candidate_id,
            "draft_id": self.draft_id,
            "run_id": self.run_id,
            "clip_result_id": self.clip_result_id,
            "surface": self.surface.value,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeedbackEvent":
        if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_FIELDS:
            raise FeedbackContractError("Feedback event fields do not match schema v1.")
        try:
            domain = FeedbackDomain(value["domain"])
            surface = FeedbackSurface(value["surface"])
        except (TypeError, ValueError) as error:
            raise FeedbackContractError("Feedback domain or surface is unknown.") from error
        payload = value["payload"]
        if not isinstance(payload, Mapping):
            raise FeedbackContractError("Feedback payload must be an object.")
        return cls(
            schema_version=value["schema_version"],
            event_id=value["event_id"],
            occurred_at=value["occurred_at"],
            session_id=value["session_id"],
            sequence=value["sequence"],
            domain=domain,
            name=value["name"],
            project_id=value["project_id"],
            analysis_id=value["analysis_id"],
            candidate_id=value["candidate_id"],
            draft_id=value["draft_id"],
            run_id=value["run_id"],
            clip_result_id=value["clip_result_id"],
            surface=surface,
            payload=dict(payload),
        )

    def dedupe_key(self) -> tuple[str, ...] | None:
        if self.name == EditorialEventName.MOMENT_SHOWN.value:
            assert self.candidate_id is not None
            return ("shown", self.session_id, self.surface.value, self.name, self.candidate_id)
        if self.name == CreativeEventName.DRAFT_SHOWN.value:
            assert self.draft_id is not None and self.candidate_id is not None
            return (
                "shown", self.session_id, self.surface.value, self.name,
                self.draft_id, self.candidate_id,
            )
        if self.name == OutcomeEventName.FINAL_SHOWN.value:
            assert self.clip_result_id is not None
            return ("shown", self.session_id, self.surface.value, self.name, self.clip_result_id)
        if self.name == OutcomeEventName.FINAL_CREATED.value:
            assert self.clip_result_id is not None
            return ("final_created", self.project_id, self.clip_result_id)
        return None


def new_feedback_event(
    *,
    occurred_at: str,
    session_id: str,
    sequence: int,
    domain: FeedbackDomain | str,
    name: EditorialEventName | CreativeEventName | OutcomeEventName | str,
    project_id: str,
    analysis_id: str | None = None,
    candidate_id: str | None = None,
    draft_id: str | None = None,
    run_id: str | None = None,
    clip_result_id: str | None = None,
    surface: FeedbackSurface | str,
    payload: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> FeedbackEvent:
    try:
        resolved_domain = FeedbackDomain(domain)
        resolved_surface = FeedbackSurface(surface)
    except (TypeError, ValueError) as error:
        raise FeedbackContractError("Feedback domain or surface is unknown.") from error
    return FeedbackEvent(
        schema_version=FEEDBACK_SCHEMA_VERSION,
        event_id=event_id or str(uuid.uuid4()),
        occurred_at=occurred_at,
        session_id=session_id,
        sequence=sequence,
        domain=resolved_domain,
        name=str(name),
        project_id=project_id,
        analysis_id=analysis_id,
        candidate_id=candidate_id,
        draft_id=draft_id,
        run_id=run_id,
        clip_result_id=clip_result_id,
        surface=resolved_surface,
        payload=dict(payload or {}),
    )


def validate_feedback_event(event: FeedbackEvent) -> None:
    if event.schema_version != FEEDBACK_SCHEMA_VERSION:
        raise FeedbackContractError("Unsupported feedback schema version.")
    _validate_uuid(event.event_id, "event_id")
    _validate_uuid(event.session_id, "session_id")
    _validate_timestamp(event.occurred_at)
    if isinstance(event.sequence, bool) or not isinstance(event.sequence, int) or event.sequence < 1:
        raise FeedbackContractError("Feedback sequence must be a positive integer.")
    if not isinstance(event.domain, FeedbackDomain) or not isinstance(event.surface, FeedbackSurface):
        raise FeedbackContractError("Feedback domain and surface must use closed enums.")
    expected_domain = _EVENT_DOMAIN.get(event.name)
    if expected_domain is None or event.domain != expected_domain:
        raise FeedbackContractError("Feedback event name does not belong to its domain.")
    if event.surface not in _EVENT_SURFACES[event.name]:
        raise FeedbackContractError("Feedback event is assigned to the wrong surface.")
    if not isinstance(event.payload, Mapping):
        raise FeedbackContractError("Feedback payload must be an object.")
    _validate_identifier(event.project_id, "project_id", required=True)
    for field_name in ("analysis_id", "candidate_id", "draft_id", "run_id", "clip_result_id"):
        _validate_identifier(getattr(event, field_name), field_name, required=False)
    _validate_identity_requirements(event)
    _validate_payload(event.name, event.payload)


def _validate_identity_requirements(event: FeedbackEvent) -> None:
    if event.domain == FeedbackDomain.EDITORIAL:
        _require(event.analysis_id, "analysis_id", event.name)
        _require(event.candidate_id, "candidate_id", event.name)
        return
    if event.domain == FeedbackDomain.CREATIVE:
        if event.name == CreativeEventName.CREATIVE_OVERRIDE_CHANGED.value:
            if event.payload.get("scope") == CreativeOverrideScope.DRAFT.value:
                _require(event.analysis_id, "analysis_id", event.name)
                _require(event.candidate_id, "candidate_id", event.name)
                _require(event.draft_id, "draft_id", event.name)
        else:
            _require(event.analysis_id, "analysis_id", event.name)
            _require(event.candidate_id, "candidate_id", event.name)
            _require(event.draft_id, "draft_id", event.name)
        return
    _require(event.candidate_id, "candidate_id", event.name)
    _require(event.run_id, "run_id", event.name)
    _require(event.clip_result_id, "clip_result_id", event.name)


def _validate_payload(name: str, payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise FeedbackContractError("Feedback payload must be an object.")
    keys = set(payload)
    if name in _EMPTY_PAYLOAD_EVENTS:
        _require_keys(keys, frozenset(), frozenset(), name)
    elif name in _SHOWN_EVENTS:
        optional = (
            frozenset({"rank", "recommended"})
            if name == EditorialEventName.MOMENT_SHOWN.value
            else frozenset({"rank"})
        )
        _require_keys(keys, frozenset(), optional, name)
        if "rank" in payload and (
            isinstance(payload["rank"], bool) or not isinstance(payload["rank"], int)
            or not 0 <= payload["rank"] <= 9999
        ):
            raise FeedbackContractError("Shown rank must be an integer from 0 to 9999.")
        if "recommended" in payload and not isinstance(payload["recommended"], bool):
            raise FeedbackContractError("Shown recommended must be boolean.")
    elif name == EditorialEventName.MOMENT_REJECTED.value:
        _require_keys(keys, frozenset({"reason"}), frozenset(), name)
        _enum_value(EditorialRejectReason, payload["reason"], "editorial reject reason")
    elif name == CreativeEventName.DRAFT_REJECTED.value:
        _require_keys(keys, frozenset({"reason"}), frozenset(), name)
        _enum_value(CreativeRejectReason, payload["reason"], "creative reject reason")
    elif name == EditorialEventName.BOUNDARY_CHANGED.value:
        required = frozenset({
            "boundary", "old_start_seconds", "old_end_seconds",
            "new_start_seconds", "new_end_seconds", "delta_seconds",
        })
        _require_keys(keys, required, frozenset(), name)
        _enum_value(BoundarySide, payload["boundary"], "boundary side")
        old_start = _bounded_number(payload["old_start_seconds"], "old_start_seconds", minimum=0)
        old_end = _bounded_number(payload["old_end_seconds"], "old_end_seconds", minimum=0)
        new_start = _bounded_number(payload["new_start_seconds"], "new_start_seconds", minimum=0)
        new_end = _bounded_number(payload["new_end_seconds"], "new_end_seconds", minimum=0)
        _bounded_number(payload["delta_seconds"], "delta_seconds", minimum=-3600, maximum=3600)
        if old_end <= old_start or new_end <= new_start:
            raise FeedbackContractError("Boundary ranges must have positive duration.")
    elif name == CreativeEventName.CREATIVE_OVERRIDE_CHANGED.value:
        required = frozenset({"field", "scope", "old_value", "new_value"})
        _require_keys(keys, required, frozenset(), name)
        field_name = _enum_value(CreativeOverrideField, payload["field"], "creative override field")
        _enum_value(CreativeOverrideScope, payload["scope"], "creative override scope")
        _validate_override_value(field_name, payload["old_value"])
        _validate_override_value(field_name, payload["new_value"])
        if payload["old_value"] == payload["new_value"]:
            raise FeedbackContractError("Creative override must change a value.")
    elif name == OutcomeEventName.FINAL_EXPORTED.value:
        _require_keys(keys, frozenset({"method"}), frozenset(), name)
        _enum_value(FinalExportMethod, payload["method"], "final export method")
    else:
        raise FeedbackContractError("Feedback event has no payload contract.")


def _validate_override_value(field_name: str, value: Any) -> None:
    if field_name in {
        CreativeOverrideField.SUBTITLES_ENABLED.value,
        CreativeOverrideField.SAME_SOURCE_BROLL_ALLOWED.value,
    }:
        if not isinstance(value, bool):
            raise FeedbackContractError("Boolean creative override has a non-boolean value.")
        return
    allowed = {
        CreativeOverrideField.SUBTITLE_STYLE.value: _SUBTITLE_STYLES,
        CreativeOverrideField.AUDIO_MODE.value: _AUDIO_MODES,
    }[field_name]
    if not isinstance(value, str) or value not in allowed:
        raise FeedbackContractError("Creative override value is not in its closed value set.")


def _require_keys(keys: set[str], required: frozenset[str], optional: frozenset[str], name: str) -> None:
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise FeedbackContractError(f"Payload fields do not match the {name} contract.")


def _validate_uuid(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise FeedbackContractError(f"{field_name} must be a UUID string.")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise FeedbackContractError(f"{field_name} must be a UUID string.") from error
    if str(parsed) != value or parsed.version != 4:
        raise FeedbackContractError(f"{field_name} must use canonical UUIDv4 form.")


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise FeedbackContractError("occurred_at must be a UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise FeedbackContractError("occurred_at must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FeedbackContractError("occurred_at must include the UTC offset.")


def _validate_identifier(value: Any, field_name: str, *, required: bool) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FeedbackContractError(f"{field_name} is not a bounded stable identifier.")
    lowered = value.casefold()
    if any(token in lowered for token in _FORBIDDEN_IDENTIFIER_PARTS):
        raise FeedbackContractError(f"{field_name} resembles forbidden raw or secret data.")


def _require(value: str | None, field_name: str, name: str) -> None:
    if value is None:
        raise FeedbackContractError(f"{name} requires {field_name}.")


def _enum_value(enum_type: type[StrEnum], value: Any, label: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as error:
        raise FeedbackContractError(f"Unknown {label}.") from error


def _bounded_number(
    value: Any,
    label: str,
    *,
    minimum: float = 0,
    maximum: float = 604800,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackContractError(f"{label} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise FeedbackContractError(f"{label} is outside its safe range.")
    return number


__all__ = [
    "BoundarySide", "CreativeEventName", "CreativeOverrideField", "CreativeOverrideScope",
    "CreativeRejectReason", "EditorialEventName", "EditorialRejectReason", "FEEDBACK_SCHEMA_VERSION",
    "FeedbackContractError", "FeedbackDomain", "FeedbackEvent", "FeedbackSurface", "FinalExportMethod",
    "OutcomeEventName", "new_feedback_event", "validate_feedback_event",
]
