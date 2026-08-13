from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.clip_results import ClipResult, primary_clip_results
from app.gui.models import ProjectRun
from app.utils import read_json


@dataclass(frozen=True, slots=True)
class RunUiProjection:
    """Small cached UI view of a run-owned manifest or legacy report."""

    primary_results: tuple[ClipResult, ...] = ()
    content_summary_report: dict[str, Any] | None = None
    candidate_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class RunProjectionCache:
    """Manifest-first run projection with signature-keyed legacy fallback.

    Large legacy reports are parsed at most once for an unchanged
    ``(path, size, mtime)`` signature.  Only the reduced projection is retained;
    the full JSON object is released after projection.
    """

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[tuple[int, int], RunUiProjection]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def for_run(self, run: ProjectRun) -> RunUiProjection:
        manifest = self.manifest_for_run(run)
        if manifest is not None:
            return manifest
        if not run.report_path:
            return RunUiProjection(warnings=tuple(run.warnings))
        report = self._read_cached(Path(run.report_path), run, manifest=False)
        if report is None:
            return RunUiProjection(warnings=tuple(run.warnings))
        return report

    def manifest_for_run(self, run: ProjectRun) -> RunUiProjection | None:
        """Return only the small canonical manifest projection; never fall back."""

        manifest_path = self._manifest_path(run)
        if manifest_path is None:
            return None
        return self._read_cached(manifest_path, run, manifest=True)

    def _read_cached(
        self, path: Path, run: ProjectRun, *, manifest: bool,
    ) -> RunUiProjection | None:
        try:
            resolved = path.expanduser().resolve()
            stat = resolved.stat()
        except OSError:
            return None
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(resolved)
        if cached is not None and cached[0] == signature:
            return cached[1]
        raw = read_json(resolved, {})
        if not isinstance(raw, dict) or not self._matches_run(raw, run, manifest=manifest):
            return None
        projection = self._project(raw, run, manifest=manifest)
        self._cache[resolved] = (signature, projection)
        return projection

    @staticmethod
    def _manifest_path(run: ProjectRun) -> Path | None:
        execution = run.settings_snapshot.get("execution", {})
        if not isinstance(execution, dict):
            return None
        engine_paths = execution.get("engine_paths", {})
        value = engine_paths.get("manifest_path") if isinstance(engine_paths, dict) else None
        value = value or execution.get("manifest_path")
        return Path(str(value)) if value else None

    @staticmethod
    def _matches_run(raw: dict[str, Any], run: ProjectRun, *, manifest: bool) -> bool:
        if manifest:
            if str(raw.get("run_id") or "") != run.run_id:
                return False
            project_id = str(raw.get("project_id") or "")
            return not project_id or project_id == run.project_id
        report_run = raw.get("run")
        if not isinstance(report_run, dict):
            # Pre-run-identity reports remain eligible only through their
            # explicit ProjectRun-owned snapshot path.
            return True
        run_id = str(report_run.get("run_id") or "")
        project_id = str(report_run.get("project_id") or "")
        return (not run_id or run_id == run.run_id) and (not project_id or project_id == run.project_id)

    @classmethod
    def _project(
        cls, raw: dict[str, Any], run: ProjectRun, *, manifest: bool,
    ) -> RunUiProjection:
        registry = raw.get("primary_results")
        if isinstance(registry, list):
            results = tuple(
                result for value in registry
                if (result := ClipResult.from_dict(value)) is not None
            )
        elif manifest:
            results = ()
        else:
            results = tuple(primary_clip_results(raw.get("production_render")))

        understanding = raw.get("content_understanding")
        summary_report = cls._content_summary_report(raw, understanding)
        metadata = cls._candidate_metadata(raw)
        warnings = list(run.warnings)
        if not manifest:
            values = raw.get("warnings", [])
            if isinstance(values, list):
                warnings.extend(str(value) for value in values if str(value).strip())
            production = raw.get("production_render", {})
            values = production.get("warnings", []) if isinstance(production, dict) else []
            if isinstance(values, list):
                warnings.extend(str(value) for value in values if str(value).strip())
        return RunUiProjection(
            primary_results=results,
            content_summary_report=summary_report,
            candidate_metadata=metadata,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _content_summary_report(
        raw: dict[str, Any], understanding: object,
    ) -> dict[str, Any] | None:
        if not isinstance(understanding, dict) or not understanding.get("enabled"):
            return None
        profile = understanding.get("profile", {})
        content_map = understanding.get("content_map", {})
        recommendation = understanding.get("clip_count_recommendation", {})
        coverage = understanding.get("coverage_map", understanding.get("coverage", {}))
        if not all(isinstance(value, dict) for value in (profile, content_map, recommendation, coverage)):
            return None
        chapters = content_map.get("chapters", [])
        selected_chapters = coverage.get("selected_chapters", [])
        compact_understanding = {
            "enabled": True,
            "profile": {"detected_content_type": profile.get("detected_content_type")},
            "content_map": {"chapters": [None] * len(chapters) if isinstance(chapters, list) else []},
            "clip_count_recommendation": {
                "estimated_publishable_clip_range": recommendation.get("estimated_publishable_clip_range", {}),
                "estimated_story_count": recommendation.get(
                    "estimated_story_count", understanding.get("story_unit_count", 0),
                ),
            },
            "coverage_map": {
                "selected_chapters": list(selected_chapters) if isinstance(selected_chapters, list) else [],
            },
        }
        compact: dict[str, Any] = {"content_understanding": compact_understanding}
        virality = raw.get("virality")
        intelligence = raw.get("clip_intelligence")
        candidates = intelligence.get("candidates", []) if isinstance(intelligence, dict) else []
        chosen = None
        if isinstance(candidates, list):
            chosen = next(
                (
                    item for item in candidates
                    if isinstance(item, dict) and item.get("selected") and isinstance(item.get("virality"), dict)
                ),
                next(
                    (item for item in candidates if isinstance(item, dict) and isinstance(item.get("virality"), dict)),
                    None,
                ),
            )
        if isinstance(virality, dict):
            compact["virality"] = {"enabled": virality.get("enabled")}
        if isinstance(chosen, dict):
            compact["clip_intelligence"] = {"candidates": [{
                "selected": chosen.get("selected"),
                "virality": chosen.get("virality"),
            }]}
        return compact

    @staticmethod
    def _candidate_metadata(raw: dict[str, Any]) -> dict[str, dict[str, object]]:
        intelligence = raw.get("clip_intelligence", {})
        candidates = intelligence.get("candidates", []) if isinstance(intelligence, dict) else []
        if not isinstance(candidates, list):
            return {}
        metadata: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "")
            if not candidate_id:
                continue
            excerpt = str(
                candidate.get("title") or candidate.get("core_idea") or candidate.get("text") or ""
            ).strip()
            metadata[candidate_id] = {
                "title": (excerpt[:96].rstrip() + "…") if len(excerpt) > 96 else excerpt,
                "start": candidate.get("start_seconds", candidate.get("start")),
                "end": candidate.get("end_seconds", candidate.get("end")),
            }
        return metadata
