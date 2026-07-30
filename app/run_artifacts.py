"""Engine-owned locations for a pipeline run and its review artifacts.

The desktop client must never rebuild a location from a source file name.  A
source display name is presentation data: it can contain a URL title, Unicode,
or a different slug than the one used by an older engine.  This module keeps a
small, run-id-addressable record of the *actual* locations selected by the
engine and provides a deliberately conservative reader for older runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.utils import read_json, stable_text_hash, utc_now, write_json


RUN_ARTIFACT_METADATA_VERSION = "1.0"


def normalized_path(path: Path | str) -> str:
    """Return an absolute, normalized filesystem path without requiring it exists."""

    return str(Path(path).expanduser().resolve())


def run_metadata_path(engine_root: Path, run_id: str) -> Path:
    """Stable lookup key that depends only on the run ID, never on a slug."""

    return engine_root.resolve() / "work" / "run-metadata" / f"{stable_text_hash(run_id)}.json"


def make_run_artifact_metadata(
    *,
    engine_root: Path,
    run_id: str,
    project_id: str | None,
    work_directory: Path,
    output_directory: Path,
    state_path: Path | None = None,
    report_path: Path | None = None,
    analysis_artifact_path: Path | None = None,
    draft_artifact_path: Path | None = None,
    manifest_path: Path | None = None,
    output_files: list[Path] | None = None,
    terminal_status: str | None = None,
) -> dict[str, Any]:
    """Build the public engine-to-desktop artifact contract for one run."""

    work = Path(work_directory).resolve()
    output = Path(output_directory).resolve()
    paths: dict[str, Any] = {
        "work_directory": normalized_path(work),
        "output_directory": normalized_path(output),
        "state_path": normalized_path(state_path or work / "state.json"),
        "heartbeat_path": normalized_path(work / "heartbeat.json"),
        "report_path": normalized_path(report_path or output / "report.json"),
        "manifest_path": normalized_path(manifest_path or output / "manifest.json"),
        "analysis_artifact_path": normalized_path(analysis_artifact_path) if analysis_artifact_path else None,
        "draft_artifact_path": normalized_path(draft_artifact_path) if draft_artifact_path else None,
        "output_files": [normalized_path(path) for path in output_files or []],
    }
    return {
        "schema_version": RUN_ARTIFACT_METADATA_VERSION,
        "run_id": run_id,
        "project_id": project_id,
        "metadata_path": normalized_path(run_metadata_path(engine_root, run_id)),
        "updated_at": utc_now(),
        "terminal_status": terminal_status,
        "paths": paths,
    }


def write_run_artifact_metadata(engine_root: Path, metadata: dict[str, Any]) -> Path:
    """Atomically publish the latest engine-owned locations for a run."""

    run_id = str(metadata.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Run metadata requires run_id.")
    path = run_metadata_path(engine_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, metadata)
    return path


def read_run_artifact_metadata(
    path: Path,
    *,
    run_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any] | None:
    """Read a metadata record only when its identity and mandatory paths match."""

    try:
        value = read_json(path, None)
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not _matches_identity(value, run_id, project_id):
        return None
    paths = value.get("paths")
    if not isinstance(paths, dict):
        return None
    required = ("state_path", "report_path", "work_directory", "output_directory")
    if any(not _absolute_string(paths.get(name)) for name in required):
        return None
    return value


def find_run_artifact_metadata(
    engine_root: Path,
    *,
    run_id: str,
    project_id: str | None,
    preferred_path: Path | None = None,
) -> dict[str, Any] | None:
    """Find one run by identity, without calculating any source-derived slug.

    The indexed metadata is the normal contract.  The scan is intentionally a
    fallback for versions that only wrote ``report.json``/``analysis.json``.
    It searches files by their embedded run/project identifiers, never by the
    source title or a guessed output directory.
    """

    root = Path(engine_root).resolve()
    candidates = [preferred_path, run_metadata_path(root, run_id)]
    for candidate in candidates:
        if candidate is None:
            continue
        value = read_run_artifact_metadata(candidate, run_id=run_id, project_id=project_id)
        if value is not None:
            return value

    for state_path in _iter_files(root / "work", "state.json"):
        state = _read_object(state_path)
        run = state.get("run") if isinstance(state, dict) else None
        if not isinstance(run, dict) or not _matches_identity(run, run_id, project_id):
            continue
        paths = run.get("paths")
        if isinstance(paths, dict):
            metadata = {
                "schema_version": "legacy-state",
                "run_id": run_id,
                "project_id": run.get("project_id"),
                "metadata_path": None,
                "updated_at": run.get("updated_at"),
                "terminal_status": run.get("terminal_status"),
                "paths": paths,
            }
            if read_run_artifact_metadata_value(metadata, run_id=run_id, project_id=project_id):
                return metadata

    for report_path in _iter_files(root / "output", "report.json"):
        report = _read_object(report_path)
        run = report.get("run") if isinstance(report, dict) else None
        if isinstance(run, dict) and _matches_identity(run, run_id, project_id):
            return _legacy_report_metadata(root, report_path, report, run_id, project_id)

    # Some legacy analysis-only jobs wrote analysis.json before the report
    # snapshot was successfully copied to desktop history.  The folder name is
    # only used to associate a discovered artifact with its own run ID; no
    # source slug is constructed or inferred.
    for analysis_path in _iter_files(root / "output", "analysis.json"):
        if analysis_path.parent.name != run_id:
            continue
        analysis = _read_object(analysis_path)
        if not isinstance(analysis, dict) or not _matches_project(analysis.get("project_id"), project_id):
            continue
        output = analysis_path.parent
        return make_run_artifact_metadata(
            engine_root=root,
            run_id=run_id,
            project_id=project_id,
            work_directory=_legacy_work_directory(root, run_id) or output,
            output_directory=output,
            analysis_artifact_path=analysis_path,
            terminal_status="analysis_ready",
        )
    return None


def read_run_artifact_metadata_value(
    value: dict[str, Any], *, run_id: str | None = None, project_id: str | None = None,
) -> bool:
    """Validation variant for metadata read from an already-open state file."""

    if not _matches_identity(value, run_id, project_id):
        return False
    paths = value.get("paths")
    return isinstance(paths, dict) and all(
        _absolute_string(paths.get(name))
        for name in ("state_path", "report_path", "work_directory", "output_directory")
    )


def _legacy_report_metadata(
    root: Path, report_path: Path, report: dict[str, Any], run_id: str, project_id: str | None,
) -> dict[str, Any]:
    run = report.get("run", {})
    output = Path(str(run.get("run_directory") or report_path.parent)).resolve()
    analysis = _existing_path(run.get("analysis_artifact_path")) or _existing_path(
        report.get("clip_intelligence", {}).get("analysis_artifact_ref")
        if isinstance(report.get("clip_intelligence"), dict) else None
    )
    draft = _existing_path(run.get("draft_artifact_path"))
    return make_run_artifact_metadata(
        engine_root=root,
        run_id=run_id,
        project_id=project_id or str(run.get("project_id") or "") or None,
        work_directory=_legacy_work_directory(root, run_id) or output,
        output_directory=output,
        report_path=report_path,
        analysis_artifact_path=analysis,
        draft_artifact_path=draft,
        manifest_path=_existing_path(run.get("manifest_path")) or output / "manifest.json",
        output_files=[Path(item) for item in report.get("output_files", []) if isinstance(item, str)],
        terminal_status=str(run.get("terminal_status") or report.get("terminal", {}).get("status") or "") or None,
    )


def _legacy_work_directory(root: Path, run_id: str) -> Path | None:
    for state_path in _iter_files(root / "work", "state.json"):
        if state_path.parent.name == run_id:
            return state_path.parent
    return None


def _iter_files(root: Path, name: str):
    if not root.is_dir():
        return ()
    try:
        return root.rglob(name)
    except OSError:
        return ()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path, {})
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _existing_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _matches_identity(value: dict[str, Any], run_id: str | None, project_id: str | None) -> bool:
    return (
        (run_id is None or str(value.get("run_id") or "") == run_id)
        and _matches_project(value.get("project_id"), project_id)
    )


def _matches_project(value: object, project_id: str | None) -> bool:
    return project_id is None or str(value or "") == project_id


def _absolute_string(value: object) -> bool:
    try:
        return isinstance(value, str) and bool(value.strip()) and Path(value).is_absolute()
    except (TypeError, ValueError):
        return False
