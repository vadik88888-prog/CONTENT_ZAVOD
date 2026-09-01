from __future__ import annotations

"""Privacy-bounded export for the local Friend Beta feedback journal."""

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.feedback_contracts import FeedbackDomain, FeedbackEvent
from app.feedback_store import (
    CREATIVE_FEEDBACK_FILE_NAME,
    EDITORIAL_FEEDBACK_FILE_NAME,
    OUTCOME_FEEDBACK_FILE_NAME,
    FeedbackStore,
)
from app.utils import utc_now


FEEDBACK_EXPORT_DIRECTORY_NAME = "feedback-exports"
FEEDBACK_EXPORT_SCHEMA_VERSION = 1
_DOMAIN_FILES = {
    FeedbackDomain.EDITORIAL: EDITORIAL_FEEDBACK_FILE_NAME,
    FeedbackDomain.CREATIVE: CREATIVE_FEEDBACK_FILE_NAME,
    FeedbackDomain.OUTCOME: OUTCOME_FEEDBACK_FILE_NAME,
}


@dataclass(frozen=True, slots=True)
class FeedbackExportResult:
    path: Path
    event_count: int
    project_count: int


def export_feedback_archive(
    project_directories: Iterable[Path],
    destination_root: Path,
) -> FeedbackExportResult:
    """Create one sendable archive from validated local feedback events only.

    The archive is deliberately rebuilt from parsed contracts rather than
    copying application directories. It therefore cannot contain media, API
    secrets, transcript references, project metadata, logs, or diagnostics.
    """

    events_by_domain: dict[FeedbackDomain, list[FeedbackEvent]] = {
        domain: [] for domain in FeedbackDomain
    }
    project_ids: set[str] = set()
    for directory in project_directories:
        for domain, file_name in _DOMAIN_FILES.items():
            events = FeedbackStore(Path(directory), file_name=file_name).read_events()
            events_by_domain[domain].extend(
                event for event in events if event.domain == domain
            )
            project_ids.update(event.project_id for event in events if event.domain == domain)

    now = datetime.now(timezone.utc)
    destination = Path(destination_root) / FEEDBACK_EXPORT_DIRECTORY_NAME
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"content-factory-feedback-{now.strftime('%Y%m%dT%H%M%SZ')}.zip"
    serial = 1
    while path.exists():
        path = destination / f"content-factory-feedback-{now.strftime('%Y%m%dT%H%M%SZ')}-{serial}.zip"
        serial += 1
    temporary = path.with_suffix(".tmp")
    total = sum(len(events) for events in events_by_domain.values())
    summary = {
        "schema_version": FEEDBACK_EXPORT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "event_count": total,
        "project_count": len(project_ids),
        "event_counts": {
            domain.value: len(events_by_domain[domain]) for domain in FeedbackDomain
        },
        "privacy": {
            "includes_api_keys": False,
            "includes_media": False,
            "includes_full_transcripts": False,
            "includes_project_paths": False,
        },
    }
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "summary.json",
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            for domain, file_name in _DOMAIN_FILES.items():
                serialized = "".join(
                    json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    for event in events_by_domain[domain]
                )
                archive.writestr(f"events/{file_name}", serialized)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return FeedbackExportResult(path=path, event_count=total, project_count=len(project_ids))


__all__ = [
    "FEEDBACK_EXPORT_DIRECTORY_NAME", "FEEDBACK_EXPORT_SCHEMA_VERSION", "FeedbackExportResult",
    "export_feedback_archive",
]
