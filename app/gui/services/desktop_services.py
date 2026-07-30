from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from app.candidate_review import validate_boundary_override
from app.gui.models import DesktopProject, DesktopSettings, ProjectRun, ProjectStatus, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore, InputValidationError
from app.gui.services.error_mapping import redact_secrets
from app.gui.services.pipeline_facade import (
    STATE_PERSISTENCE_WARNING,
    PipelineCompletion,
    PipelineFacade,
    PreparedPipelineRun,
)
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.product_flow import calibrate_processing_estimate
from app.source_download import validate_public_video_url
from app.utils import read_json, stable_text_hash, utc_now


@dataclass(slots=True)
class DesktopServices:
    """Application-service boundary shared by view-models and Qt widgets."""

    engine_root: Path
    settings_store: SettingsStore
    settings: DesktopSettings
    projects: DesktopProjectStore
    runs: RunHistoryStore
    pipeline: PipelineFacade
    system: SystemService

    @classmethod
    def create(cls, engine_root: Path) -> "DesktopServices":
        settings_store = SettingsStore()
        settings = settings_store.load()
        projects = DesktopProjectStore(Path(settings.data_directory))
        runs = RunHistoryStore(projects)
        services = cls(
            engine_root=engine_root.resolve(),
            settings_store=settings_store,
            settings=settings,
            projects=projects,
            runs=runs,
            pipeline=PipelineFacade(engine_root),
            system=SystemService(engine_root),
        )
        services.recover_interrupted_runs()
        return services

    def save_settings(self) -> None:
        self.settings_store.save(self.settings)

    def reconfigure_data_directory(self, directory: Path) -> None:
        self.settings.data_directory = str(directory.expanduser().resolve())
        self.settings.validate()
        self.projects = DesktopProjectStore(Path(self.settings.data_directory))
        self.runs = RunHistoryStore(self.projects)
        self.save_settings()
        self.recover_interrupted_runs()

    def list_projects(self) -> list[DesktopProject]:
        return self.projects.list()

    def create_project(self, source_path: str | Path) -> DesktopProject:
        metadata = self.pipeline.inspect_source(source_path)
        project = self.projects.create(source_path, source_metadata=metadata)
        self._refresh_setup_state(project, "Источник готов. Настройте обработку и запустите анализ, когда будете готовы.")
        self.projects.save(project)
        return project

    def create_url_project(self, url: str, metadata: dict) -> DesktopProject:
        project = self.projects.create_url(validate_public_video_url(url), metadata)
        self._refresh_setup_state(project, "Ссылка проверена. Настройте обработку; загрузка начнётся только после вашего запуска.")
        self.projects.save(project)
        return project

    def mark_url_download_started(self, project: DesktopProject) -> None:
        if project.source_spec.kind != "url":
            raise InputValidationError("Этот проект использует локальный файл.")
        project.source_spec.download_state = "downloading"
        project.source_spec.error_message = None
        self.projects.save(project)

    def complete_url_download(self, project: DesktopProject, path: str | Path) -> DesktopProject:
        source = validate_video_path(path)
        source_directory = (Path(project.project_directory) / "sources").resolve()
        if not source.is_relative_to(source_directory):
            raise InputValidationError("Загруженный файл должен находиться в папке проекта.")
        metadata = self.pipeline.inspect_source(source)
        project.source_path = str(source)
        project.source_metadata = metadata
        project.source_spec.downloaded_path = str(source)
        project.source_spec.metadata = metadata
        project.source_spec.download_state = "downloaded"
        project.source_spec.error_message = None
        self._refresh_setup_state(project, "Видео загружено. Настройки и оценка обновлены по фактическому файлу.")
        self.projects.save(project)
        return project

    def fail_url_download(self, project: DesktopProject, message: str, *, cancelled: bool = False) -> None:
        if project.source_spec.kind != "url":
            return
        project.source_spec.download_state = "cancelled" if cancelled else "failed"
        project.source_spec.error_message = redact_secrets(message)
        self.projects.save(project)

    def save_project(self, project: DesktopProject) -> None:
        project.settings.validate()
        self.projects.save(project)

    def update_project_options(self, project: DesktopProject, **values: object) -> DesktopProject:
        """Persist a setup change and tell the person what the next run will reuse.

        The method never launches or invalidates pipeline artifacts.  Existing
        candidates and drafts remain inspectable; only the setup explanation
        describes whether a later analysis is needed for a changed option.
        """

        changed: list[str] = []
        for name, value in values.items():
            if not hasattr(project.settings, name):
                continue
            if getattr(project.settings, name) != value:
                setattr(project.settings, name, value)
                changed.append(name)
        project.settings.validate()
        if not changed:
            return project
        has_analysis = bool(project.analysis_artifact_path)
        analysis_options = {"processing_mode", "deep_analysis", "clip_count"}
        needs_analysis = has_analysis and bool(analysis_options.intersection(changed))
        if needs_analysis:
            summary = (
                "Настройки сохранены для следующего анализа. Текущие найденные моменты и черновики не изменены; "
                "чтобы применить этот режим, выполните новый анализ видео."
            )
            reused = ["текущие моменты и черновики остаются доступными для просмотра"]
        elif has_analysis:
            summary = (
                "Настройки сохранены. Сохранённый анализ и найденные моменты можно использовать повторно; "
                "новый анализ не нужен."
            )
            reused = ["сохранённый анализ", "найденные моменты"]
        else:
            summary = "Настройки сохранены и будут применены к первому анализу."
            reused = []
        self._refresh_setup_state(project, summary, needs_new_analysis=needs_analysis, reused_stages=reused)
        self.projects.save(project)
        return project

    def _refresh_setup_state(
        self,
        project: DesktopProject,
        summary: str | None = None,
        *,
        needs_new_analysis: bool | None = None,
        reused_stages: list[str] | None = None,
    ) -> None:
        """Store a safe, serialisable estimate for the next time the project opens."""

        try:
            estimate = self.processing_estimate(project)
        except Exception:
            # A first-run configuration can be incomplete.  Keep the source and
            # choices durable and let the screen show its honest fallback state.
            return
        project.setup_state.last_estimate = estimate.to_dict()
        project.setup_state.estimated_at = utc_now()
        if summary is not None:
            project.setup_state.change_summary = summary
        if needs_new_analysis is not None:
            project.setup_state.needs_new_analysis = needs_new_analysis
        if reused_stages is not None:
            project.setup_state.reused_stages = list(reused_stages)

    def set_review_selection(self, project: DesktopProject, candidate_ids: list[str]) -> DesktopProject:
        """Persist the user's candidate choice before any draft or render starts."""

        unique = list(dict.fromkeys(str(item) for item in candidate_ids if str(item)))
        if len(unique) > 3:
            raise InputValidationError("Для одного прохода выберите от одного до трёх моментов.")
        unknown = [item for item in unique if item not in project.candidate_states]
        if unknown:
            raise InputValidationError("Один из выбранных моментов отсутствует в сохранённом анализе.")
        project.review_selected_candidate_ids = unique
        removed = set(project.selected_candidate_ids) - set(unique)
        for candidate_id in removed:
            if project.candidate_states.get(candidate_id) == "selected":
                project.candidate_states[candidate_id] = "draft_ready"
        project.selected_candidate_ids = [item for item in project.selected_candidate_ids if item in unique]
        if project.status not in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            project.status = ProjectStatus.REVIEWING_CANDIDATES if project.analysis_artifact_path else project.status
        self.projects.save(project)
        return project

    def delete_project(self, project_id: str) -> None:
        self.projects.delete(project_id)

    def runs_for(self, project: DesktopProject) -> list[ProjectRun]:
        return self.runs.list(project.project_id)

    def processing_estimate(self, project: DesktopProject):
        """A serialisable preflight estimate for the project screen."""

        return self.setup_preflight(project)[1]

    def setup_preflight(self, project: DesktopProject):
        """Resolve the existing pipeline's setup choices for a view-model."""

        _intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        return resolved, calibrate_processing_estimate(estimate, self.runs_for(project))

    def refresh_setup_estimate(self, project: DesktopProject) -> DesktopProject:
        """Refresh the durable preflight when a project is opened again."""

        previous = dict(project.setup_state.last_estimate)
        previous_at = project.setup_state.estimated_at
        self._refresh_setup_state(project)
        if project.setup_state.last_estimate != previous or (project.setup_state.last_estimate and previous_at is None):
            self.projects.save(project)
        return project

    def prepare_run(self, project: DesktopProject) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status == ProjectStatus.PROCESSING:
            raise RuntimeError("Этот проект уже обрабатывается.")
        source = project.source
        if not source.is_file():
            if project.source_spec.kind == "url":
                raise InputValidationError("Сначала загрузите видео по ссылке.")
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        estimate = calibrate_processing_estimate(estimate, self.runs_for(project))
        run = self.runs.create(
            project,
            settings_snapshot={
                "project_options": {
                    "processing_mode": project.settings.processing_mode,
                    "deep_analysis": project.settings.deep_analysis,
                    "platform": project.settings.platform,
                    "clip_count": project.settings.clip_count,
                    "subtitles_enabled": project.settings.subtitles_enabled,
                    "subtitle_style": project.settings.subtitle_style,
                    "audio_mode": project.settings.audio_mode,
                    "composition_strategy": project.settings.composition_strategy,
                    "encoder": project.settings.encoder,
                    "use_cache": project.settings.use_cache,
                    "recompute_all": project.settings.recompute_all,
                },
                "product_flow": {
                    "user_intent": intent.to_dict(),
                    "resolved_config": resolved.to_dict(),
                    "estimate": estimate.to_dict(),
                },
                "local_test_mode": self.settings.local_test_mode,
            },
            source_snapshot={
                "kind": project.source_spec.kind,
                "path": str(source),
                "name": source.name,
                "size_bytes": source.stat().st_size,
            },
            pipeline_version="0.1.0",
        )
        run.cost_estimate = estimate.estimated_ai_cost_max
        self.runs.save(run)
        try:
            prepared = self.pipeline.prepare(project, run, self.settings)
        except Exception:
            run.status = RunStatus.FAILED
            run.finished_at = utc_now()
            run.error_summary = "Не удалось подготовить запуск."
            self.runs.save(run)
            raise
        # This must happen before Qt starts the child process.  A silent CLI
        # must never leave a run without an immediately inspectable log.
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.PROCESSING
        project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def prepare_analysis(self, project: DesktopProject) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            raise RuntimeError("Этот проект уже обрабатывается.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        run = self.runs.create(
            project,
            settings_snapshot={
                "project_options": asdict(project.settings),
                "product_flow": {"user_intent": intent.to_dict(), "resolved_config": resolved.to_dict(), "estimate": estimate.to_dict()},
                "local_test_mode": self.settings.local_test_mode,
            },
            source_snapshot={"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            pipeline_version="0.1.0", run_kind=RunKind.ANALYSIS,
        )
        run.cost_estimate = estimate.estimated_ai_cost_max
        self.runs.save(run)
        try:
            prepared = self.pipeline.prepare_analysis(project, run, self.settings)
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить анализ."
            self.runs.save(run)
            raise
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.ANALYZING; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def prepare_draft(self, project: DesktopProject, candidate_ids: list[str]) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            raise RuntimeError("Этот проект уже обрабатывается.")
        if not project.analysis_artifact_path:
            raise InputValidationError("Сначала выполните анализ видео.")
        outside_selection = [item for item in candidate_ids if item not in project.review_selected_candidate_ids]
        if outside_selection:
            raise InputValidationError("Сначала выберите моменты для подготовки черновика.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        run = self.runs.create(
            project, {"analysis_id": project.analysis_id, "candidate_ids": list(candidate_ids), "local_test_mode": self.settings.local_test_mode},
            {"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            "0.1.0", run_kind=RunKind.DRAFT,
        )
        try:
            prepared = self.pipeline.prepare_draft(project, run, self.settings, candidate_ids)
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить черновик."
            self.runs.save(run)
            raise
        for candidate_id in candidate_ids:
            project.candidate_states[candidate_id] = "draft_planning"
            project.candidate_errors.pop(candidate_id, None)
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.PROCESSING; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def select_draft_candidates(self, project: DesktopProject, candidate_ids: list[str]) -> DesktopProject:
        if not project.candidate_draft_artifacts:
            raise InputValidationError("Сначала соберите Draft Preview.")
        if not candidate_ids:
            raise InputValidationError("Выберите хотя бы один готовый черновик.")
        unavailable = [item for item in candidate_ids if project.candidate_states.get(item) not in {"draft_ready", "selected"}]
        if unavailable:
            raise InputValidationError("В production можно отправлять только готовые черновики.")
        missing_artifacts = [item for item in candidate_ids if not Path(project.candidate_draft_artifacts.get(item, "")).is_file()]
        if missing_artifacts:
            raise InputValidationError("Черновик для выбранного момента больше не доступен.")
        for candidate_id, state in list(project.candidate_states.items()):
            if state == "selected" and candidate_id not in candidate_ids:
                project.candidate_states[candidate_id] = "draft_ready"
        for candidate_id in candidate_ids:
            project.candidate_states[candidate_id] = "selected"
        project.selected_candidate_ids = list(candidate_ids)
        project.review_selected_candidate_ids = list(candidate_ids)
        project.status = ProjectStatus.REVIEWING_CANDIDATES
        self.projects.save(project)
        return project

    def set_draft_approval(
        self, project: DesktopProject, candidate_id: str, approved: bool,
    ) -> DesktopProject:
        """Persist one explicit production decision without changing draft selection.

        ``review_selected_candidate_ids`` answers "which moments are being
        prepared or reviewed".  ``selected_candidate_ids`` remains the much
        narrower, deliberate hand-off to the expensive production render.
        Keeping those choices separate prevents an accidental render when a
        user is merely looking through ready drafts.
        """

        if candidate_id not in project.candidate_states:
            raise InputValidationError("Черновик отсутствует в сохранённом анализе.")
        if approved:
            if project.candidate_states.get(candidate_id) not in {"draft_ready", "selected"}:
                raise InputValidationError("Подтвердить можно только готовый черновик.")
            if not Path(project.candidate_draft_artifacts.get(candidate_id, "")).is_file():
                raise InputValidationError("Не удалось найти сохранённый черновик для этого момента.")
            if candidate_id not in project.selected_candidate_ids:
                project.selected_candidate_ids.append(candidate_id)
            if candidate_id not in project.review_selected_candidate_ids:
                project.review_selected_candidate_ids.append(candidate_id)
            project.candidate_states[candidate_id] = "selected"
        else:
            project.selected_candidate_ids = [
                item for item in project.selected_candidate_ids if item != candidate_id
            ]
            if project.candidate_states.get(candidate_id) == "selected":
                project.candidate_states[candidate_id] = "draft_ready"
        project.status = ProjectStatus.REVIEWING_CANDIDATES
        self.projects.save(project)
        return project

    def adjust_candidate_boundary(
        self, project: DesktopProject, candidate_id: str, boundary: str, delta_seconds: float,
    ) -> tuple[DesktopProject, dict]:
        """Persist one review edit after cached-only boundary revalidation."""

        if boundary not in {"start", "end"} or delta_seconds not in {-1.0, -0.5, 0.5, 1.0}:
            raise InputValidationError("Доступны только шаги границы ±0.5 или ±1.0 секунды.")
        artifact_path = Path(str(project.analysis_artifact_path or ""))
        artifact = read_json(artifact_path, {}) if artifact_path.is_file() else {}
        candidates = artifact.get("candidates", []) if isinstance(artifact, dict) else []
        candidate = next((item for item in candidates if isinstance(item, dict) and str(item.get("candidate_id")) == candidate_id), None)
        if not candidate:
            raise InputValidationError("Кандидат не найден в сохранённом анализе.")
        current = dict(project.candidate_boundary_overrides.get(candidate_id) or {})
        start = float(current.get("start", candidate.get("start", 0)))
        end = float(current.get("end", candidate.get("end", 0)))
        if boundary == "start":
            start += delta_seconds
        else:
            end += delta_seconds
        references = artifact.get("references", {}) if isinstance(artifact, dict) else {}
        transcript_features = read_json(Path(str(references.get("transcript_features") or "")), {}) if references.get("transcript_features") else {}
        scenes = read_json(Path(str(references.get("scene_boundaries") or "")), {}) if references.get("scene_boundaries") else {}
        validation = validate_boundary_override(
            start, end,
            source_duration=float(project.source_metadata.get("duration")) if project.source_metadata.get("duration") is not None else None,
            minimum_duration=15.0, maximum_duration=60.0,
            transcript_features=transcript_features if isinstance(transcript_features, dict) else {},
            scenes=scenes if isinstance(scenes, dict) else {},
        )
        if not validation["valid"]:
            raise InputValidationError(" ".join(validation["errors"]))
        project.candidate_boundary_overrides[candidate_id] = {
            "start": validation["start"], "end": validation["end"],
            "warnings": validation["warnings"], "revalidation": validation["revalidation"],
            "candidate_boundary_fingerprint": stable_text_hash(
                f"{project.analysis_fingerprint or ''}:{candidate_id}:{validation['start']:.3f}:{validation['end']:.3f}"
            ),
        }
        if project.candidate_states.get(candidate_id) in {"draft_ready", "selected", "draft_failed"}:
            project.candidate_states[candidate_id] = "analyzed"
        project.candidate_draft_artifacts.pop(candidate_id, None)
        project.candidate_errors.pop(candidate_id, None)
        project.selected_candidate_ids = [item for item in project.selected_candidate_ids if item != candidate_id]
        self.projects.save(project)
        return project, validation

    def prepare_selected_render(self, project: DesktopProject) -> tuple[ProjectRun, PreparedPipelineRun]:
        if project.status in {ProjectStatus.ANALYZING, ProjectStatus.PROCESSING, ProjectStatus.RENDERING_SELECTED}:
            raise RuntimeError("Этот проект уже обрабатывается.")
        candidate_ids = list(project.selected_candidate_ids)
        if not candidate_ids:
            raise InputValidationError("Сначала подтвердите черновики для production render.")
        missing_artifacts = [item for item in candidate_ids if not Path(project.candidate_draft_artifacts.get(item, "")).is_file()]
        if missing_artifacts:
            raise InputValidationError("Не удалось найти сохранённый черновик для выбранного момента.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        run = self.runs.create(
            project, {"draft_id": project.draft_id, "candidate_ids": candidate_ids, "local_test_mode": self.settings.local_test_mode},
            {"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            "0.1.0", run_kind=RunKind.SELECTED_RENDER,
        )
        try:
            prepared = self.pipeline.prepare_selected_render(project, run, self.settings, candidate_ids)
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить production render."
            self.runs.save(run)
            raise
        for candidate_id in candidate_ids:
            project.candidate_states[candidate_id] = "production_rendering"
            project.candidate_errors.pop(candidate_id, None)
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.RENDERING_SELECTED; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def prepare_render_revision(self, project: DesktopProject, parent_run: ProjectRun) -> tuple[ProjectRun, PreparedPipelineRun]:
        """Create an immutable export revision from existing production/audio artifacts."""

        if project.status == ProjectStatus.PROCESSING:
            raise RuntimeError("Этот проект уже обрабатывается.")
        if parent_run.status not in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}:
            raise InputValidationError("Повторный экспорт доступен только после успешного запуска.")
        source = project.source
        if not source.is_file():
            raise InputValidationError("Исходный видеофайл больше недоступен.")
        intent, resolved, estimate = self.pipeline.plan_processing(project, self.settings)
        previous = parent_run.settings_snapshot.get("project_options", {})
        previous_audio_mode = str(previous.get("audio_mode", "original"))
        if previous_audio_mode != project.settings.audio_mode:
            raise InputValidationError("Для изменения аудиорежима запустите полное создание ролика.")
        current = {
            "subtitle_style": project.settings.subtitle_style,
            "subtitles_enabled": project.settings.subtitles_enabled,
            "platform": project.settings.platform,
            "audio_mode": project.settings.audio_mode,
            "composition_strategy": project.settings.composition_strategy,
            "encoder": project.settings.encoder,
        }
        changed = {name: value for name, value in current.items() if previous.get(name) != value}
        run = self.runs.create(
            project,
            settings_snapshot={
                "project_options": {**current, "processing_mode": project.settings.processing_mode, "deep_analysis": project.settings.deep_analysis, "clip_count": project.settings.clip_count, "use_cache": project.settings.use_cache, "recompute_all": False},
                "product_flow": {"user_intent": intent.to_dict(), "resolved_config": resolved.to_dict(), "estimate": estimate.to_dict()},
                "local_test_mode": self.settings.local_test_mode,
            },
            source_snapshot={"kind": project.source_spec.kind, "path": str(source), "name": source.name, "size_bytes": source.stat().st_size},
            pipeline_version="0.1.0", run_kind=RunKind.RENDER_REVISION, parent_run_id=parent_run.run_id,
            changed_settings=changed, invalidated_stages=["production_render"],
        )
        run.cost_estimate = 0.0
        self.runs.save(run)
        try:
            prepared = self.pipeline.prepare_render_revision(project, run, self.settings, parent_run)
        except Exception:
            run.status = RunStatus.FAILED; run.finished_at = utc_now(); run.error_summary = "Не удалось подготовить повторный экспорт."
            self.runs.save(run)
            raise
        self.record_launch_context(run, prepared)
        project.status = ProjectStatus.PROCESSING; project.latest_run_id = run.run_id
        self.projects.save(project)
        return run, prepared

    def record_launch_context(self, run: ProjectRun, prepared: PreparedPipelineRun) -> None:
        """Persist the desktop launch contract without leaking environment secrets."""

        run.settings_snapshot["execution"] = {
            "state_path": str(prepared.state_path),
            "report_path": str(prepared.report_path),
            "output_directory": str(prepared.output_directory),
            "runtime_config_path": str(prepared.runtime_config_path),
            "run_id": prepared.run_id,
            "manifest_path": str(prepared.manifest_path) if prepared.manifest_path else None,
            "source_path": str(prepared.source_path) if prepared.source_path else None,
            "runtime_flags": dict(prepared.runtime_flags),
        }
        self.runs.save(run)
        flags = [argument for argument in prepared.arguments if argument.startswith("--")]
        self.append_log(run, "Desktop pipeline launch prepared.")
        self.append_log(run, f"command: {prepared.command_line()}")
        self.append_log(run, f"cwd: {prepared.working_directory}")
        self.append_log(run, f"source: {prepared.source_path or '<unknown>'}")
        self.append_log(run, f"runtime config: {prepared.runtime_config_path}")
        self.append_log(run, f"output directory: {prepared.output_directory}")
        self.append_log(run, f"flags: {' '.join(flags) if flags else '<none>'}")
        runtime_flags = "; ".join(
            f"{key}={value}" for key, value in prepared.runtime_flags.items()
        )
        self.append_log(run, f"runtime flags: {runtime_flags or '<unknown>'}")
        self.append_log(
            run,
            "environment: PYTHONUNBUFFERED=1; PYTHONIOENCODING=utf-8; "
            "inherited environment values are intentionally not logged.",
        )

    def append_log(self, run: ProjectRun, line: str) -> None:
        if not run.log_path:
            return
        path = Path(run.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        limit = 2 * 1024 * 1024
        if path.exists() and path.stat().st_size >= limit:
            rotated = path.with_suffix(".1.log")
            if rotated.exists():
                rotated.unlink()
            path.replace(rotated)
        with path.open("a", encoding="utf-8") as file:
            file.write(f"{utc_now()} {redact_secrets(line).rstrip()}\n")

    def finish_success(self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun) -> ProjectRun:
        completion = self.pipeline.completion(prepared)
        if completion.error_summary:
            return self.finish_failure(project, run, completion.error_summary, completion.technical_details)
        return self._finish_completion(project, run, completion)

    def recover_failed_process(
        self, project: DesktopProject, run: ProjectRun, prepared: PreparedPipelineRun,
    ) -> ProjectRun | None:
        """Keep verified outputs when only final service-state persistence failed."""

        completion = self.pipeline.recovery_completion(prepared, run.started_at)
        if completion is None:
            return None
        return self._finish_completion(project, run, completion, state_persistence_degraded=True)

    def _finish_completion(
        self,
        project: DesktopProject,
        run: ProjectRun,
        completion: PipelineCompletion,
        *,
        state_persistence_degraded: bool = False,
    ) -> ProjectRun:
        warnings = list(completion.warnings)
        if state_persistence_degraded and STATE_PERSISTENCE_WARNING not in warnings:
            warnings.append(STATE_PERSISTENCE_WARNING)
        report = read_json(completion.report_path, {})
        run.status = RunStatus.COMPLETED_WITH_WARNINGS if warnings else RunStatus.COMPLETED
        run.finished_at = utc_now()
        run.warnings = warnings
        run.cost_estimate = completion.cost_estimate
        run.actual_cost = None  # Local estimates are intentionally never treated as billed cost.
        self.runs.snapshot_report_and_outputs(run, completion.report_path, completion.output_files)
        self.runs.save(run)
        run_info = report.get("run", {}) if isinstance(report, dict) else {}
        if run.run_kind == RunKind.ANALYSIS:
            run.status = RunStatus.ANALYSIS_READY
            project.analysis_artifact_path = str(run_info.get("analysis_artifact_path") or "") or None
            project.analysis_id = str(run_info.get("analysis_id") or "") or None
            project.analysis_fingerprint = str(run_info.get("analysis_fingerprint") or "") or None
            candidates = report.get("clip_intelligence", {}).get("candidates", []) if isinstance(report.get("clip_intelligence"), dict) else []
            project.candidate_states = {
                str(item.get("id")): "analyzed" for item in candidates
                if isinstance(item, dict) and item.get("id")
            }
            project.selected_candidate_ids = []
            project.review_selected_candidate_ids = []
            project.candidate_draft_artifacts = {}
            project.candidate_errors = {}
            project.draft_artifact_path = None
            project.draft_id = None
            project.status = ProjectStatus.ANALYSIS_READY
        elif run.run_kind == RunKind.DRAFT:
            run.status = RunStatus.DRAFT_READY
            project.draft_artifact_path = str(run_info.get("draft_artifact_path") or "") or None
            project.draft_id = str(run_info.get("draft_id") or "") or None
            draft_candidates = report.get("candidate_flow", {}).get("draft_candidates", []) if isinstance(report.get("candidate_flow"), dict) else []
            for item in draft_candidates:
                if isinstance(item, dict) and item.get("candidate_id") and item.get("state"):
                    candidate_id = str(item["candidate_id"])
                    state = str(item["state"])
                    project.candidate_states[candidate_id] = state
                    if state == "draft_ready" and project.draft_artifact_path:
                        project.candidate_draft_artifacts[candidate_id] = project.draft_artifact_path
                        project.candidate_errors.pop(candidate_id, None)
                    elif state == "draft_failed":
                        project.candidate_draft_artifacts.pop(candidate_id, None)
                        message = str(item.get("error") or "Не удалось подготовить черновик.").strip()
                        if message:
                            project.candidate_errors[candidate_id] = message
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.SELECTED_RENDER:
            completed_ids = {
                str(item.get("candidate_id")) for item in report.get("production_render", {}).get("items", [])
                if isinstance(item, dict) and item.get("status") in {"completed", "warning"}
            } if isinstance(report.get("production_render"), dict) else set()
            flow_items = report.get("candidate_flow", {}).get("items", []) if isinstance(report.get("candidate_flow"), dict) else []
            failures = {
                str(item.get("candidate_id")): str(item.get("message") or item.get("reason") or "Не удалось создать ролик.")
                for item in flow_items if isinstance(item, dict) and item.get("outcome") == "failed" and item.get("candidate_id")
            }
            for candidate_id in project.selected_candidate_ids:
                if candidate_id in completed_ids:
                    project.candidate_states[candidate_id] = "rendered"
                    project.candidate_errors.pop(candidate_id, None)
                else:
                    project.candidate_states[candidate_id] = "draft_ready"
                    if candidate_id in failures:
                        project.candidate_errors[candidate_id] = failures[candidate_id]
            render_incomplete = len(completed_ids) < len(project.selected_candidate_ids)
            if render_incomplete:
                run.status = RunStatus.PARTIALLY_RENDERED
                project.status = ProjectStatus.PARTIALLY_RENDERED
                project.selected_candidate_ids = [
                    candidate_id for candidate_id in project.selected_candidate_ids if candidate_id not in completed_ids
                ]
            elif warnings:
                run.status = RunStatus.COMPLETED_WITH_WARNINGS
                project.status = ProjectStatus.COMPLETED_WITH_WARNINGS
                project.selected_candidate_ids = []
            else:
                run.status = RunStatus.COMPLETED
                project.status = ProjectStatus.COMPLETED
                project.selected_candidate_ids = []
        else:
            project.status = ProjectStatus.COMPLETED_WITH_WARNINGS if warnings else ProjectStatus.COMPLETED
        project.latest_run_id = run.run_id
        self.projects.save(project)
        return run

    def finish_failure(
        self, project: DesktopProject, run: ProjectRun, message: str, technical_details: str | None = None,
    ) -> ProjectRun:
        run.status = RunStatus.FAILED
        run.finished_at = utc_now()
        run.error_summary = redact_secrets(message)
        run.technical_details = redact_secrets(technical_details or message)
        self.append_log(run, run.technical_details)
        self.runs.save(run)
        if run.run_kind == RunKind.SELECTED_RENDER:
            for candidate_id in project.selected_candidate_ids:
                if project.candidate_states.get(candidate_id) == "production_rendering":
                    project.candidate_states[candidate_id] = "selected"
                    project.candidate_errors[candidate_id] = run.error_summary
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.DRAFT:
            for candidate_id, state in list(project.candidate_states.items()):
                if state == "draft_planning":
                    project.candidate_states[candidate_id] = "analyzed"
                    project.candidate_errors[candidate_id] = run.error_summary
            project.status = ProjectStatus.ANALYSIS_READY if project.analysis_artifact_path else ProjectStatus.SOURCE_READY
        elif run.run_kind == RunKind.ANALYSIS:
            project.status = ProjectStatus.SOURCE_READY
        else:
            project.status = ProjectStatus.FAILED
        self.projects.save(project)
        return run

    def finish_cancelled(self, project: DesktopProject, run: ProjectRun) -> ProjectRun:
        run.status = RunStatus.CANCELLED
        run.finished_at = utc_now()
        run.error_summary = "Создание ролика отменено пользователем."
        self.runs.save(run)
        if run.run_kind == RunKind.SELECTED_RENDER:
            for candidate_id in project.selected_candidate_ids:
                if project.candidate_states.get(candidate_id) == "production_rendering":
                    project.candidate_states[candidate_id] = "selected"
            project.status = ProjectStatus.REVIEWING_CANDIDATES
        elif run.run_kind == RunKind.DRAFT:
            for candidate_id, state in list(project.candidate_states.items()):
                if state == "draft_planning":
                    project.candidate_states[candidate_id] = "analyzed"
            project.status = ProjectStatus.ANALYSIS_READY if project.analysis_artifact_path else ProjectStatus.SOURCE_READY
        elif run.run_kind == RunKind.ANALYSIS:
            project.status = ProjectStatus.SOURCE_READY
        else:
            project.status = ProjectStatus.CANCELLED
        self.projects.save(project)
        return run

    def recover_interrupted_runs(self) -> int:
        recovered_or_interrupted = 0
        for project in self.projects.list():
            changed = False
            for run in self.runs.list(project.project_id):
                if run.status not in RunStatus.ACTIVE:
                    continue
                prepared = self.pipeline.prepared_from_execution(run)
                if prepared and self.recover_failed_process(project, run, prepared):
                    recovered_or_interrupted += 1
                    changed = True
                    continue
                run.status = RunStatus.INTERRUPTED
                run.finished_at = utc_now()
                run.error_summary = "Предыдущий запуск был прерван при закрытии приложения."
                self.runs.save(run)
                recovered_or_interrupted += 1
                changed = True
            if changed:
                latest = next(iter(self.runs.list(project.project_id)), None)
                if latest and latest.status == RunStatus.COMPLETED_WITH_WARNINGS:
                    project.status = ProjectStatus.COMPLETED_WITH_WARNINGS
                elif latest and latest.status == RunStatus.COMPLETED:
                    project.status = ProjectStatus.COMPLETED
                elif latest and latest.status == RunStatus.INTERRUPTED:
                    project.status = ProjectStatus.INTERRUPTED
                self.projects.save(project)
        return recovered_or_interrupted
