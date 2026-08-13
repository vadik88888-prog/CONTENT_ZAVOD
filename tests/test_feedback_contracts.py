from __future__ import annotations

import uuid

import pytest

from app.feedback_contracts import (
    CreativeEventName,
    EditorialEventName,
    FEEDBACK_SCHEMA_VERSION,
    FeedbackContractError,
    FeedbackDomain,
    FeedbackEvent,
    FeedbackSurface,
    OutcomeEventName,
    new_feedback_event,
)


SESSION_ID = "11111111-1111-4111-8111-111111111111"
EVENT_ID = "22222222-2222-4222-8222-222222222222"
NOW = "2026-08-13T12:00:00+00:00"


def _event(**overrides):
    values = {
        "occurred_at": NOW,
        "session_id": SESSION_ID,
        "sequence": 1,
        "domain": FeedbackDomain.EDITORIAL,
        "name": EditorialEventName.MOMENT_SHOWN,
        "project_id": "project-1",
        "analysis_id": "analysis-1",
        "candidate_id": "candidate-1",
        "surface": FeedbackSurface.MOMENTS,
        "payload": {"rank": 0, "recommended": True},
        "event_id": EVENT_ID,
    }
    values.update(overrides)
    return new_feedback_event(**values)


def test_schema_v1_round_trips_with_only_closed_top_level_fields() -> None:
    event = _event()

    restored = FeedbackEvent.from_dict(event.to_dict())

    assert restored == event
    assert restored.schema_version == FEEDBACK_SCHEMA_VERSION
    assert uuid.UUID(restored.event_id)
    assert uuid.UUID(restored.session_id)


def test_payload_is_copied_and_cannot_be_mutated_after_validation() -> None:
    payload = {"rank": 0}
    event = _event(payload=payload)

    payload["transcript_text"] = "must not leak"

    assert event.to_dict()["payload"] == {"rank": 0}
    with pytest.raises(TypeError):
        event.payload["rank"] = 1  # type: ignore[index]


def test_event_name_must_match_domain_and_surface() -> None:
    with pytest.raises(FeedbackContractError, match="domain"):
        _event(domain=FeedbackDomain.CREATIVE)
    with pytest.raises(FeedbackContractError, match="surface"):
        _event(surface=FeedbackSurface.DRAFTS)


@pytest.mark.parametrize("forbidden", [
    {"transcript_text": "raw speech"},
    {"media_path": "C:/private/video.mp4"},
    {"api_key": "sk-secret"},
])
def test_payload_allowlist_rejects_raw_transcript_media_path_and_api_data(forbidden) -> None:
    with pytest.raises(FeedbackContractError, match="Payload fields"):
        _event(payload=forbidden)


def test_identifier_rejects_paths_and_secret_shaped_values() -> None:
    with pytest.raises(FeedbackContractError, match="identifier"):
        _event(candidate_id="C:/private/video.mp4")
    with pytest.raises(FeedbackContractError, match="forbidden"):
        _event(candidate_id="sk-secret-token")


def test_reject_reasons_are_required_closed_enums() -> None:
    with pytest.raises(FeedbackContractError, match="Unknown editorial reject reason"):
        _event(name=EditorialEventName.MOMENT_REJECTED, payload={"reason": "free text"})

    accepted = _event(
        name=EditorialEventName.MOMENT_REJECTED,
        payload={"reason": "needs_context"},
    )
    assert accepted.payload == {"reason": "needs_context"}

    creative = _event(
        domain=FeedbackDomain.CREATIVE,
        name=CreativeEventName.DRAFT_REJECTED,
        surface=FeedbackSurface.DRAFTS,
        draft_id="draft-1",
        payload={"reason": "captions"},
    )
    assert creative.payload == {"reason": "captions"}


def test_boundary_change_accepts_only_safe_numeric_ranges() -> None:
    payload = {
        "boundary": "start",
        "old_start_seconds": 4.0,
        "old_end_seconds": 20.0,
        "new_start_seconds": 3.5,
        "new_end_seconds": 20.0,
        "delta_seconds": -0.5,
    }
    assert _event(name=EditorialEventName.BOUNDARY_CHANGED, payload=payload).payload == payload

    with pytest.raises(FeedbackContractError, match="positive duration"):
        _event(
            name=EditorialEventName.BOUNDARY_CHANGED,
            payload={**payload, "new_start_seconds": 21.0},
        )


def test_creative_override_values_are_field_specific_and_no_op_is_rejected() -> None:
    base = {
        "domain": FeedbackDomain.CREATIVE,
        "name": CreativeEventName.CREATIVE_OVERRIDE_CHANGED,
        "surface": FeedbackSurface.SETUP,
        "analysis_id": None,
        "candidate_id": None,
        "payload": {
            "field": "subtitle_style",
            "scope": "project",
            "old_value": "documentary",
            "new_value": "clean",
        },
    }
    assert _event(**base).payload["new_value"] == "clean"

    with pytest.raises(FeedbackContractError, match="closed value set"):
        _event(**{**base, "payload": {**base["payload"], "new_value": "C:/font.ttf"}})
    with pytest.raises(FeedbackContractError, match="must change"):
        _event(**{**base, "payload": {**base["payload"], "new_value": "documentary"}})

    with pytest.raises(FeedbackContractError, match="draft_id"):
        _event(**{
            **base,
            "surface": FeedbackSurface.DRAFTS,
            "payload": {**base["payload"], "scope": "draft"},
            "analysis_id": "analysis-1",
            "candidate_id": "candidate-1",
            "draft_id": None,
        })


def test_outcome_requires_stable_result_and_run_identity() -> None:
    with pytest.raises(FeedbackContractError, match="run_id"):
        _event(
            domain=FeedbackDomain.OUTCOME,
            name=OutcomeEventName.FINAL_CREATED,
            surface=FeedbackSurface.FINAL,
            analysis_id=None,
            clip_result_id="result-1",
            payload={},
        )

    event = _event(
        domain=FeedbackDomain.OUTCOME,
        name=OutcomeEventName.FINAL_CREATED,
        surface=FeedbackSurface.FINAL,
        analysis_id=None,
        run_id="run-1",
        clip_result_id="result-1",
        payload={},
    )
    assert event.dedupe_key() == ("final_created", "project-1", "result-1")


def test_deserialization_rejects_unknown_top_level_data() -> None:
    raw = _event().to_dict()
    raw["source_path"] = "C:/private/video.mp4"

    with pytest.raises(FeedbackContractError, match="fields"):
        FeedbackEvent.from_dict(raw)
