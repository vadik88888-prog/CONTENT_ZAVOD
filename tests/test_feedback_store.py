from __future__ import annotations

import json
from pathlib import Path

from app.feedback_contracts import FeedbackDomain, FeedbackSurface
from app.feedback_store import FEEDBACK_FILE_NAME, FeedbackStore


SESSION_ONE = "11111111-1111-4111-8111-111111111111"
SESSION_TWO = "22222222-2222-4222-8222-222222222222"
NOW = "2026-08-13T12:00:00+00:00"


def _moment(store: FeedbackStore, name: str = "moment_shown", **overrides):
    values = {
        "domain": FeedbackDomain.EDITORIAL,
        "name": name,
        "project_id": "project-1",
        "analysis_id": "analysis-1",
        "candidate_id": "candidate-1",
        "surface": FeedbackSurface.MOMENTS,
        "payload": {"rank": 0} if name == "moment_shown" else {},
    }
    values.update(overrides)
    return store.record(**values)


def _final_created(store: FeedbackStore):
    return store.record(
        domain=FeedbackDomain.OUTCOME,
        name="final_created",
        project_id="project-1",
        candidate_id="candidate-1",
        run_id="run-1",
        clip_result_id="result-1",
        surface=FeedbackSurface.FINAL,
    )


def _draft_shown(store: FeedbackStore, candidate_id: str):
    return store.record(
        domain=FeedbackDomain.CREATIVE,
        name="draft_shown",
        project_id="project-1",
        analysis_id="analysis-1",
        candidate_id=candidate_id,
        draft_id="draft-1",
        surface=FeedbackSurface.DRAFTS,
        payload={"rank": 0},
    )


def test_store_appends_jsonl_with_session_sequence_uuid_and_timestamp(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "project", session_id=SESSION_ONE, clock=lambda: NOW)

    first = _moment(store)
    second = _moment(store, "moment_selected")

    assert first.persisted and second.persisted
    assert first.event and first.event.sequence == 1 and first.event.occurred_at == NOW
    assert second.event and second.event.sequence == 2
    assert store.path.name == FEEDBACK_FILE_NAME
    assert store.read_events() == [first.event, second.event]
    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["session_id"] == SESSION_ONE for line in lines)


def test_shown_is_deduplicated_within_session_but_not_across_sessions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    first_store = FeedbackStore(project, session_id=SESSION_ONE, clock=lambda: NOW)

    assert _moment(first_store).persisted
    duplicate = _moment(first_store)
    assert duplicate.duplicate and not duplicate.persisted

    second_store = FeedbackStore(project, session_id=SESSION_TWO, clock=lambda: NOW)
    assert _moment(second_store).persisted
    assert len(second_store.read_events()) == 2


def test_draft_shown_dedupe_keeps_distinct_candidates_in_one_draft(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "project", session_id=SESSION_ONE, clock=lambda: NOW)

    assert _draft_shown(store, "candidate-1").persisted
    assert _draft_shown(store, "candidate-2").persisted
    assert _draft_shown(store, "candidate-1").duplicate
    assert len(store.read_events()) == 2


def test_final_created_is_deduplicated_across_sessions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    first = FeedbackStore(project, session_id=SESSION_ONE, clock=lambda: NOW)
    second = FeedbackStore(project, session_id=SESSION_TWO, clock=lambda: NOW)

    assert _final_created(first).persisted
    duplicate = _final_created(second)

    assert duplicate.duplicate and not duplicate.persisted
    assert len(second.read_events()) == 1


def test_corrupt_tail_preserves_previous_events_and_is_repaired_on_next_append(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = FeedbackStore(project, session_id=SESSION_ONE, clock=lambda: NOW)
    first = _moment(store)
    with store.path.open("ab") as file:
        file.write(b'{"partial":')

    reopened = FeedbackStore(project, session_id=SESSION_TWO, clock=lambda: NOW)
    assert reopened.read_events() == [first.event]
    assert reopened.last_error_code == "corrupt_tail"

    appended = _moment(reopened, "moment_selected")
    assert appended.persisted
    assert reopened.read_events() == [first.event, appended.event]
    assert reopened.last_error_code is None


def test_write_failure_is_best_effort_and_does_not_consume_sequence_or_dedupe(
    tmp_path: Path, monkeypatch,
) -> None:
    store = FeedbackStore(tmp_path / "project", session_id=SESSION_ONE, clock=lambda: NOW)
    original = store._append_line
    monkeypatch.setattr(store, "_append_line", lambda _event: (_ for _ in ()).throw(PermissionError()))

    failed = _moment(store)

    assert not failed.persisted and not failed.duplicate
    assert failed.error_code == "write_failed"
    monkeypatch.setattr(store, "_append_line", original)
    retried = _moment(store)
    assert retried.persisted and retried.event and retried.event.sequence == 1


def test_invalid_event_is_not_written_and_does_not_raise(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "project", session_id=SESSION_ONE, clock=lambda: NOW)

    result = _moment(store, payload={"transcript_text": "raw transcript"})

    assert not result.persisted and result.error_code == "invalid_event"
    assert not store.path.exists()


def test_store_never_serializes_raw_transcript_media_path_or_api_fields(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "project", session_id=SESSION_ONE, clock=lambda: NOW)
    assert _moment(store).persisted

    serialized = store.path.read_text(encoding="utf-8").casefold()
    assert "transcript" not in serialized
    assert "media_path" not in serialized
    assert "api_key" not in serialized
    assert "c:/" not in serialized
