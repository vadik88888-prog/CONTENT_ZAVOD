from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from app.content_profile_taxonomy import AUTO_PROFILE_INPUT
from app.product_flow import ProcessingIntent
from app.source_models import SourceSpec
from app.utils import utc_now


class ProjectStatus:
    NEW = "new"
    SOURCE_READY = "source_ready"
    ANALYZING = "analyzing"
    ANALYSIS_READY = "analysis_ready"
    REVIEWING_CANDIDATES = "reviewing_candidates"
    RENDERING_SELECTED = "rendering_selected"
    PARTIALLY_RENDERED = "partially_rendered"
    # Legacy product-flow states remain readable and migrate gradually through
    # the desktop service instead of invalidating existing user projects.
    DRAFT = "draft"
    READY = "ready"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    ALL: ClassVar[frozenset[str]] = frozenset({
        NEW, SOURCE_READY, ANALYZING, ANALYSIS_READY, REVIEWING_CANDIDATES,
        RENDERING_SELECTED, PARTIALLY_RENDERED,
        DRAFT, READY, QUEUED, PROCESSING, COMPLETED, COMPLETED_WITH_WARNINGS,
        FAILED, CANCELLED, INTERRUPTED,
    })


@dataclass(slots=True)
class ProjectOptions:
    """Durable user choices for the simple product flow and advanced rendering."""

    processing_mode: str = "standard"
    deep_analysis: str = "auto"
    platform: str = "universal"
    clip_count: str = "3"
    subtitles_enabled: bool = True
    subtitle_style: str = "documentary"
    preset_selection_mode: str = "auto"
    audio_mode: str = "original"
    editorial_intent: str = ""
    profile_format_override: str = AUTO_PROFILE_INPUT
    profile_editorial_mode_override: str = AUTO_PROFILE_INPUT
    profile_domain_override: str = AUTO_PROFILE_INPUT
    profile_traits_override: list[str] = field(default_factory=list)
    composition_strategy: str = "safe_auto"
    same_source_broll_allowed: bool = False
    encoder: str = "auto"
    use_cache: bool = True
    recompute_all: bool = False

    def validate(self) -> None:
        self.processing_intent().validate()
        if self.encoder not in {"auto", "cpu", "nvenc"}:
            raise ValueError("Unsupported encoder.")
        if self.composition_strategy not in {"safe_auto", "center_crop", "fit_blur_background", "fit_solid_background", "top_crop"}:
            raise ValueError("Unsupported composition strategy.")
        if not all(isinstance(item, bool) for item in (
            self.subtitles_enabled, self.same_source_broll_allowed, self.use_cache, self.recompute_all,
        )):
            raise ValueError("Project options must contain booleans.")

    def processing_intent(self) -> ProcessingIntent:
        return ProcessingIntent(
            processing_mode=self.processing_mode,
            deep_analysis=self.deep_analysis,
            platform=self.platform,
            clip_count=str(self.clip_count),
            subtitle_preset=self.subtitle_style,
            preset_selection_mode=self.preset_selection_mode,
            audio_mode=self.audio_mode,
            editorial_intent=self.editorial_intent,
            profile_format_override=self.profile_format_override,
            profile_editorial_mode_override=self.profile_editorial_mode_override,
            profile_domain_override=self.profile_domain_override,
            profile_traits_override=tuple(self.profile_traits_override),
        )


@dataclass(slots=True)
class SetupState:
    """Persisted, user-facing setup context for a project.

    It deliberately contains only an estimate and an explanation of the next
    run.  Pipeline settings continue to live in :class:`ProjectOptions`; this
    state lets a reopened project show the same clear preflight summary without
    serialising runtime paths, secrets, or engine internals.
    """

    last_estimate: dict[str, Any] = field(default_factory=dict)
    estimated_at: str | None = None
    change_summary: str = ""
    needs_new_analysis: bool = False
    reused_stages: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.last_estimate, dict):
            raise ValueError("Setup estimate must be an object.")
        if self.estimated_at is not None and not isinstance(self.estimated_at, str):
            raise ValueError("Setup estimate timestamp is invalid.")
        if not isinstance(self.change_summary, str):
            raise ValueError("Setup change summary is invalid.")
        if not isinstance(self.needs_new_analysis, bool):
            raise ValueError("Setup analysis requirement is invalid.")
        if not isinstance(self.reused_stages, list) or not all(isinstance(item, str) and item for item in self.reused_stages):
            raise ValueError("Setup reused stages are invalid.")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SetupState":
        raw = value if isinstance(value, dict) else {}
        reused_raw = raw.get("reused_stages")
        state = cls(
            last_estimate=dict(raw.get("last_estimate") or {}),
            estimated_at=str(raw["estimated_at"]) if raw.get("estimated_at") else None,
            change_summary=str(raw.get("change_summary") or ""),
            needs_new_analysis=bool(raw.get("needs_new_analysis", False)),
            reused_stages=[str(item) for item in reused_raw if str(item)] if isinstance(reused_raw, list) else [],
        )
        state.validate()
        return state


@dataclass(slots=True)
class DesktopProject:
    """Durable user project.  The source is referenced, never copied."""

    project_id: str
    name: str
    created_at: str
    updated_at: str
    source_path: str
    project_directory: str
    status: str = ProjectStatus.DRAFT
    settings: ProjectOptions = field(default_factory=ProjectOptions)
    latest_run_id: str | None = None
    thumbnail_path: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    source_spec: SourceSpec = field(default_factory=SourceSpec)
    setup_state: SetupState = field(default_factory=SetupState)
    analysis_artifact_path: str | None = None
    analysis_id: str | None = None
    analysis_fingerprint: str | None = None
    draft_artifact_path: str | None = None
    draft_id: str | None = None
    # A draft run may prepare only a subset of moments.  Keep the immutable
    # artifact that owns each ready candidate so later selections can combine
    # drafts without re-running analysis or silently dropping earlier work.
    candidate_draft_artifacts: dict[str, str] = field(default_factory=dict)
    candidate_states: dict[str, str] = field(default_factory=dict)
    candidate_errors: dict[str, str] = field(default_factory=dict)
    # The compact ``candidate_states`` field above remains the compatibility
    # projection used by older screens and project files.  Keep the three
    # independently recoverable user-facing states as well: a draft can fail
    # without changing an approval decision, and an export can fail without
    # invalidating its ready draft.
    candidate_draft_statuses: dict[str, str] = field(default_factory=dict)
    candidate_approval_states: dict[str, str] = field(default_factory=dict)
    candidate_export_statuses: dict[str, str] = field(default_factory=dict)
    # Explicit pre-production choice.  It survives restart and is distinct
    # from selected_candidate_ids, which means a user has approved a ready
    # draft for the expensive production render.
    review_selected_candidate_ids: list[str] = field(default_factory=list)
    selected_candidate_ids: list[str] = field(default_factory=list)
    candidate_boundary_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Selection is part of the review workspace rather than a widget-local
    # detail.  Persist it so a reopened project restores the same preview.
    active_preview_candidate_id: str | None = None
    # The result viewer is part of the durable project workspace.  Store the
    # stable result identity rather than a list index or filename, so a
    # reopened project returns to the exact output the person last reviewed.
    last_final_result_id: str | None = None
    schema_version: int = 3

    def validate(self) -> None:
        if not self.project_id or not self.name.strip():
            raise ValueError("Project id and name are required.")
        if self.status not in ProjectStatus.ALL:
            raise ValueError("Unsupported project status.")
        if self.schema_version != 3:
            raise ValueError("Unsupported project schema version.")
        self.settings.validate()
        self.source_spec.validate()
        self.setup_state.validate()
        supported_candidate_states = {
            "analyzed", "draft_planning", "draft_ready", "draft_failed",
            "selected", "production_rendering", "rendered",
        }
        if any(not key or value not in supported_candidate_states for key, value in self.candidate_states.items()):
            raise ValueError("Unsupported candidate review state.")
        supported_draft_statuses = {"pending", "running", "ready", "failed"}
        supported_approval_states = {"pending", "approved", "rejected"}
        supported_export_statuses = {"pending", "running", "ready", "failed"}
        if any(not key or value not in supported_draft_statuses for key, value in self.candidate_draft_statuses.items()):
            raise ValueError("Unsupported candidate draft status.")
        if any(not key or value not in supported_approval_states for key, value in self.candidate_approval_states.items()):
            raise ValueError("Unsupported candidate approval status.")
        if any(not key or value not in supported_export_statuses for key, value in self.candidate_export_statuses.items()):
            raise ValueError("Unsupported candidate export status.")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError("Selected candidate ids must be unique.")
        if len(self.review_selected_candidate_ids) != len(set(self.review_selected_candidate_ids)) or any(
            not candidate_id for candidate_id in self.review_selected_candidate_ids
        ):
            raise ValueError("Review candidate ids must be unique and non-empty.")
        if any(not candidate_id or not isinstance(path, str) or not path.strip()
               for candidate_id, path in self.candidate_draft_artifacts.items()):
            raise ValueError("Candidate draft artifact reference is invalid.")
        if any(not candidate_id or not isinstance(message, str) or not message.strip()
               for candidate_id, message in self.candidate_errors.items()):
            raise ValueError("Candidate error is invalid.")
        if self.last_final_result_id is not None and not self.last_final_result_id.strip():
            raise ValueError("Last final result identity is invalid.")
        if self.active_preview_candidate_id is not None and not self.active_preview_candidate_id.strip():
            raise ValueError("Active preview candidate identity is invalid.")
        for candidate_id, override in self.candidate_boundary_overrides.items():
            if not candidate_id or not isinstance(override, dict):
                raise ValueError("Candidate boundary override is invalid.")
            try:
                start, end = float(override["start"]), float(override["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Candidate boundary override has no valid range.") from error
            if end <= start:
                raise ValueError("Candidate boundary override is reversed.")

    @property
    def source(self) -> Path:
        return Path(self.source_spec.downloaded_path or self.source_path)

    @property
    def directory(self) -> Path:
        return Path(self.project_directory)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DesktopProject":
        settings = value.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("Project settings are corrupted.")
        supported_settings = {
            "processing_mode", "deep_analysis", "platform", "clip_count",
            "subtitles_enabled", "subtitle_style", "preset_selection_mode", "audio_mode", "composition_strategy",
            "editorial_intent", "profile_format_override", "profile_editorial_mode_override",
            "profile_domain_override", "profile_traits_override",
            "same_source_broll_allowed", "encoder", "use_cache", "recompute_all",
        }
        migrated_settings = {key: item for key, item in settings.items() if key in supported_settings}
        if "preset_selection_mode" not in settings:
            # Before this contract every persisted subtitle style was the
            # effective pinned value. Never reinterpret it as automatic.
            migrated_settings["preset_selection_mode"] = "explicit"
        source_metadata = dict(value.get("source_metadata") or {})
        draft_artifact_path = str(value["draft_artifact_path"]) if value.get("draft_artifact_path") else None
        candidate_states = {str(key): str(item) for key, item in dict(value.get("candidate_states") or {}).items()}
        candidate_draft_artifacts = {
            str(key): str(item) for key, item in dict(value.get("candidate_draft_artifacts") or {}).items()
        }
        # Schema v3 originally retained only the latest draft path.  Existing
        # ready candidates all belonged to that artifact, so reconstruct the
        # per-candidate references during a non-destructive read migration.
        if not candidate_draft_artifacts and draft_artifact_path:
            candidate_draft_artifacts = {
                candidate_id: draft_artifact_path for candidate_id, state in candidate_states.items()
                if state in {"draft_ready", "selected"}
            }
        selected_candidate_ids = [str(item) for item in value.get("selected_candidate_ids", [])]
        review_selected_candidate_ids = [str(item) for item in value.get("review_selected_candidate_ids", [])]
        if not review_selected_candidate_ids:
            review_selected_candidate_ids = list(selected_candidate_ids)
        draft_statuses = {
            str(key): str(item)
            for key, item in dict(value.get("candidate_draft_statuses") or {}).items()
        }
        approval_states = {
            str(key): str(item)
            for key, item in dict(value.get("candidate_approval_states") or {}).items()
        }
        export_statuses = {
            str(key): str(item)
            for key, item in dict(value.get("candidate_export_statuses") or {}).items()
        }
        # Existing schema-v3 projects used one combined review state.  Infer
        # the independent axes without mutating their source representation.
        for candidate_id, state in candidate_states.items():
            draft_statuses.setdefault(
                candidate_id,
                "running" if state == "draft_planning" else
                "failed" if state == "draft_failed" else
                "ready" if state in {"draft_ready", "selected", "production_rendering", "rendered"} else
                "pending",
            )
            approval_states.setdefault(
                candidate_id,
                "approved" if state in {"selected", "production_rendering", "rendered"} else "pending",
            )
            export_statuses.setdefault(
                candidate_id,
                "running" if state == "production_rendering" else
                "ready" if state == "rendered" else "pending",
            )
        project = cls(
            project_id=str(value["project_id"]),
            name=str(value["name"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            source_path=str(value["source_path"]),
            project_directory=str(value["project_directory"]),
            status=str(value.get("status", ProjectStatus.DRAFT)),
            settings=ProjectOptions(**migrated_settings),
            latest_run_id=str(value["latest_run_id"]) if value.get("latest_run_id") else None,
            thumbnail_path=str(value["thumbnail_path"]) if value.get("thumbnail_path") else None,
            source_metadata=source_metadata,
            source_spec=SourceSpec.from_dict(
                value.get("source_spec"), fallback_path=str(value.get("source_path", "")), fallback_metadata=source_metadata,
            ),
            setup_state=SetupState.from_dict(value.get("setup_state")),
            analysis_artifact_path=str(value["analysis_artifact_path"]) if value.get("analysis_artifact_path") else None,
            analysis_id=str(value["analysis_id"]) if value.get("analysis_id") else None,
            analysis_fingerprint=str(value["analysis_fingerprint"]) if value.get("analysis_fingerprint") else None,
            draft_artifact_path=draft_artifact_path,
            draft_id=str(value["draft_id"]) if value.get("draft_id") else None,
            candidate_draft_artifacts=candidate_draft_artifacts,
            candidate_states=candidate_states,
            candidate_errors={str(key): str(item) for key, item in dict(value.get("candidate_errors") or {}).items()},
            candidate_draft_statuses=draft_statuses,
            candidate_approval_states=approval_states,
            candidate_export_statuses=export_statuses,
            review_selected_candidate_ids=review_selected_candidate_ids,
            selected_candidate_ids=selected_candidate_ids,
            candidate_boundary_overrides={
                str(key): dict(item) for key, item in dict(value.get("candidate_boundary_overrides") or {}).items()
                if isinstance(item, dict)
            },
            last_final_result_id=(
                str(value["last_final_result_id"]) if value.get("last_final_result_id") else None
            ),
            active_preview_candidate_id=(
                str(value["active_preview_candidate_id"])
                if value.get("active_preview_candidate_id") else None
            ),
            # Older projects had only ``source_path``.  They are migrated in memory
            # to an explicit local source and written as v3 on the next save.
            schema_version=3,
        )
        project.validate()
        return project
