from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from app.gui.models.desktop_project import DesktopProject
from app.gui.models.processing_state import ProcessingSnapshot
from app.gui.models.project_run import ProjectRun, RunKind, RunStatus


FlowStep = Literal["download", "settings", "processing", "candidates", "drafts", "finished"]


@dataclass(frozen=True, slots=True)
class ProjectPresentation:
    """One deterministic projection used by project cards and the workspace."""

    flow_step: FlowStep
    status_label: str
    latest_run: ProjectRun | None
    active: bool = False


def derive_project_presentation(
    project: DesktopProject,
    runs: Iterable[ProjectRun],
    *,
    snapshot: ProcessingSnapshot | None = None,
    has_final_outputs: bool = False,
) -> ProjectPresentation:
    owned_runs = [run for run in runs if run.project_id == project.project_id]
    latest = _latest_run(project, owned_runs)
    active = bool(snapshot and snapshot.phase in {"preparing", "running", "cancelling"})
    if active:
        if snapshot and snapshot.stage == "download":
            return ProjectPresentation("download", "Загружаем видео", latest, True)
        return ProjectPresentation("processing", _active_label(latest), latest, True)

    # Project cards do not own the in-memory snapshot.  The persisted active
    # run is therefore their authoritative signal while its owner continues
    # in the background.  Startup recovery turns abandoned active runs into
    # ``interrupted`` before the workspace is shown.
    if latest and latest.status in RunStatus.ACTIVE:
        return ProjectPresentation("processing", _active_label(latest), latest, True)

    if not project.source_spec.is_ready:
        if project.source_spec.download_state == "downloading":
            return ProjectPresentation("download", "Загружаем видео", latest, True)
        labels = {
            "failed": "Не удалось загрузить",
            "cancelled": "Загрузка остановлена",
        }
        return ProjectPresentation("download", labels.get(project.source_spec.download_state, "Источник ждёт загрузки"), latest)

    if latest and latest.status in {
        RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.CANCELLED, RunStatus.PARTIALLY_RENDERED,
    }:
        if latest.run_kind == RunKind.DRAFT:
            return ProjectPresentation("drafts", _terminal_label(latest), latest)
        if latest.run_kind in {RunKind.SELECTED_RENDER, RunKind.RENDER_REVISION}:
            step: FlowStep = "drafts" if project.candidate_draft_artifacts else "processing"
            return ProjectPresentation(step, _terminal_label(latest), latest)
        if latest.run_kind in {RunKind.ANALYSIS, RunKind.FULL} and not project.analysis_artifact_path:
            return ProjectPresentation("processing", _terminal_label(latest), latest)

    review_ids = set(project.review_selected_candidate_ids) | set(project.selected_candidate_ids)
    if any(
        project.candidate_states.get(candidate_id)
        in {"draft_ready", "draft_failed", "selected", "production_rendering"}
        for candidate_id in review_ids
    ):
        return ProjectPresentation("drafts", _draft_label(project), latest)

    if has_final_outputs:
        label = "Готово с замечаниями" if project.status == "completed_with_warnings" else "Ролики готовы"
        return ProjectPresentation("finished", label, latest)

    states = tuple(project.candidate_states.values())
    if project.candidate_draft_artifacts or project.selected_candidate_ids or any(
        state in {"draft_planning", "draft_ready", "draft_failed", "selected", "production_rendering"}
        for state in states
    ):
        return ProjectPresentation("drafts", _draft_label(project), latest)
    if any(state == "rendered" for state in states):
        return ProjectPresentation("finished", "Ролики готовы", latest)
    if project.analysis_artifact_path:
        return ProjectPresentation("candidates", "Моменты готовы", latest)
    return ProjectPresentation("settings", "Готов к настройке", latest)


def _latest_run(project: DesktopProject, runs: list[ProjectRun]) -> ProjectRun | None:
    if project.latest_run_id:
        matched = next((run for run in runs if run.run_id == project.latest_run_id), None)
        if matched is not None:
            return matched
    return max(runs, key=lambda run: (run.started_at, run.run_id), default=None)


def _active_label(run: ProjectRun | None) -> str:
    if run and run.run_kind == RunKind.ANALYSIS:
        return "Ищем моменты"
    if run and run.run_kind == RunKind.DRAFT:
        return "Создаём черновики"
    if run and run.run_kind in {RunKind.SELECTED_RENDER, RunKind.RENDER_REVISION}:
        return "Создаём ролики"
    return "Идёт обработка"


def _terminal_label(run: ProjectRun) -> str:
    return {
        RunStatus.INTERRUPTED: "Работа прервана",
        RunStatus.CANCELLED: "Работа остановлена",
        RunStatus.PARTIALLY_RENDERED: "Готово частично",
    }.get(run.status, "Нужно внимание")


def _draft_label(project: DesktopProject) -> str:
    if any(value == "failed" for value in project.candidate_draft_statuses.values()):
        return "Черновики готовы частично"
    if any(value == "failed" for value in project.candidate_export_statuses.values()):
        return "Экспорт готов частично"
    if project.selected_candidate_ids:
        return "Готов к финальной сборке"
    if project.candidate_draft_artifacts:
        return "Черновики готовы"
    return "Выберите моменты"
