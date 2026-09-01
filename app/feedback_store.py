from __future__ import annotations

"""Append-only, best-effort local storage for privacy-bounded feedback events."""

import errno
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from app.feedback_contracts import (
    CreativeEventName,
    EditorialEventName,
    FeedbackContractError,
    FeedbackDomain,
    FeedbackEvent,
    FeedbackSurface,
    OutcomeEventName,
    new_feedback_event,
)
from app.utils import utc_now


FEEDBACK_DIRECTORY_NAME = "feedback"
FEEDBACK_FILE_NAME = "events.v1.jsonl"
EDITORIAL_FEEDBACK_FILE_NAME = "editorial.v1.jsonl"
CREATIVE_FEEDBACK_FILE_NAME = "creative.v1.jsonl"
OUTCOME_FEEDBACK_FILE_NAME = "outcome.v1.jsonl"
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class FeedbackWriteResult:
    event: FeedbackEvent | None
    persisted: bool
    duplicate: bool = False
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class _ReadResult:
    events: tuple[FeedbackEvent, ...]
    valid_bytes: int
    corrupt_tail: bool
    error_code: str | None = None


class FeedbackStore:
    """Own one project's local JSONL feedback log.

    ``record`` is intentionally best-effort: invalid contracts and filesystem
    failures return a typed result and never escape into the product flow.
    """

    def __init__(
        self,
        project_directory: Path,
        *,
        session_id: str | None = None,
        clock: Callable[[], str] = utc_now,
        file_name: str = FEEDBACK_FILE_NAME,
    ) -> None:
        self.project_directory = Path(project_directory)
        self.path = self.project_directory / FEEDBACK_DIRECTORY_NAME / file_name
        self.session_id = session_id or str(uuid.uuid4())
        self._clock = clock
        self._lock = _lock_for(self.path)
        self._sequence = 0
        self._dedupe_keys: set[tuple[str, ...]] = set()
        self._valid_bytes = 0
        self._corrupt_tail = False
        self.last_error_code: str | None = None
        self._restore_state()

    def read_events(self) -> list[FeedbackEvent]:
        with self._lock:
            result = self._read_valid_prefix()
            self._apply_read_state(result)
            return list(result.events)

    def record(
        self,
        *,
        domain: FeedbackDomain | str,
        name: EditorialEventName | CreativeEventName | OutcomeEventName | str,
        project_id: str,
        surface: FeedbackSurface | str,
        analysis_id: str | None = None,
        candidate_id: str | None = None,
        draft_id: str | None = None,
        run_id: str | None = None,
        clip_result_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> FeedbackWriteResult:
        with self._lock:
            # Refresh inside the shared path lock so separately-constructed
            # stores in the same desktop process cannot bypass sequence or
            # dedupe state.  The desktop shell itself remains single-process.
            self._apply_read_state(self._read_valid_prefix())
            try:
                event = new_feedback_event(
                    occurred_at=self._clock(),
                    session_id=self.session_id,
                    sequence=self._sequence + 1,
                    domain=domain,
                    name=name,
                    project_id=project_id,
                    analysis_id=analysis_id,
                    candidate_id=candidate_id,
                    draft_id=draft_id,
                    run_id=run_id,
                    clip_result_id=clip_result_id,
                    surface=surface,
                    payload=payload,
                )
            except (FeedbackContractError, TypeError, ValueError):
                self.last_error_code = "invalid_event"
                return FeedbackWriteResult(None, False, error_code=self.last_error_code)

            dedupe_key = event.dedupe_key()
            if dedupe_key is not None and dedupe_key in self._dedupe_keys:
                self.last_error_code = None
                return FeedbackWriteResult(None, False, duplicate=True)

            try:
                if self._corrupt_tail:
                    self._truncate_corrupt_tail()
                self._append_line(event)
            except OSError:
                self.last_error_code = "write_failed"
                return FeedbackWriteResult(event, False, error_code=self.last_error_code)

            self._sequence = event.sequence
            if dedupe_key is not None:
                self._dedupe_keys.add(dedupe_key)
            self.last_error_code = None
            return FeedbackWriteResult(event, True)

    def _restore_state(self) -> None:
        with self._lock:
            self._apply_read_state(self._read_valid_prefix())

    def _apply_read_state(self, result: _ReadResult) -> None:
        self._valid_bytes = result.valid_bytes
        self._corrupt_tail = result.corrupt_tail
        self.last_error_code = result.error_code
        self._sequence = max(
            (event.sequence for event in result.events if event.session_id == self.session_id),
            default=0,
        )
        self._dedupe_keys = {
            key for event in result.events if (key := event.dedupe_key()) is not None
        }

    def _read_valid_prefix(self) -> _ReadResult:
        if not self.path.exists():
            return _ReadResult((), 0, False)
        events: list[FeedbackEvent] = []
        valid_bytes = 0
        try:
            with self.path.open("rb") as file:
                while True:
                    raw = file.readline()
                    if not raw:
                        break
                    next_offset = file.tell()
                    if not raw.strip():
                        valid_bytes = next_offset
                        continue
                    try:
                        decoded = raw.decode("utf-8")
                        value = json.loads(decoded)
                        if not isinstance(value, dict):
                            raise FeedbackContractError("Feedback line is not an object.")
                        event = FeedbackEvent.from_dict(value)
                    except (UnicodeDecodeError, json.JSONDecodeError, FeedbackContractError, TypeError, ValueError):
                        return _ReadResult(tuple(events), valid_bytes, True, "corrupt_tail")
                    events.append(event)
                    valid_bytes = next_offset
        except OSError:
            return _ReadResult((), 0, False, "read_failed")
        return _ReadResult(tuple(events), valid_bytes, False)

    def _truncate_corrupt_tail(self) -> None:
        with self.path.open("r+b") as file:
            file.truncate(self._valid_bytes)
            file.flush()
            _fsync_if_supported(file)
        self._corrupt_tail = False

    def _append_line(self, event: FeedbackEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            event.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        needs_separator = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                needs_separator = existing.read(1) != b"\n"
        with self.path.open("ab") as file:
            if needs_separator:
                file.write(b"\n")
            file.write(payload)
            file.flush()
            _fsync_if_supported(file)
            self._valid_bytes = file.tell()


def _fsync_if_supported(file: Any) -> None:
    try:
        os.fsync(file.fileno())
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise


def _lock_for(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.resolve(strict=False)))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


__all__ = [
    "CREATIVE_FEEDBACK_FILE_NAME", "EDITORIAL_FEEDBACK_FILE_NAME", "FEEDBACK_DIRECTORY_NAME",
    "FEEDBACK_FILE_NAME", "OUTCOME_FEEDBACK_FILE_NAME", "FeedbackStore", "FeedbackWriteResult",
]
