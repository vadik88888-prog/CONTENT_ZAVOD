from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.clip_results import ClipResult, result_paths, unique_primary_results
from app.analysis_artifact import AnalysisArtifact, AnalysisArtifactError
from app.run_manifest import is_run_scoped_path
from app.run_artifacts import find_run_artifact_metadata, run_metadata_path
from app.config import load_config
from app.draft_artifact import DraftArtifact, DraftArtifactError, new_draft_artifact
from app.gui.models import DesktopProject, DesktopSettings, ProjectRun
from app.gui.services.desktop_project_store import InputValidationError, validate_video_path
from app.media import probe_video
from app.product_flow import (
    CostPricing,
    ProcessingEstimate,
    ProcessingIntent,
    ResolvedProcessingConfig,
    apply_resolved_processing_config,
    estimate_processing,
    resolve_processing_intent,
)
from app.quality_report import QUALITY_REPORT_SCHEMA_VERSION, aggregate_quality_status, read_quality_report
from app.runtime import RuntimeLayout
from app.utils import read_json, safe_name, stable_file_hash


STATE_PERSISTENCE_WARNING = "Ролики созданы, но не удалось сохранить служебное состояние"


def _source_duration_seconds(project: DesktopProject) -> float | None:
    value = project.source_metadata.get("duration")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


@dataclass(frozen=True, slots=True)
class PreparedPipelineRun:
    program: str
    arguments: list[str]
    working_directory: Path
    state_path: Path
    report_path: Path
    output_directory: Path
    runtime_config_path: Path
    source_path: Path | None = None
    runtime_flags: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None
    manifest_path: Path | None = None
    # Desktop-only observability paths.  The engine owns heartbeat.json next to
    # state.json; the desktop runner owns the rotating pipeline log.
    heartbeat_path: Path | None = None
    log_path: Path | None = None
    source_duration_seconds: float | None = None
    # The only path desktop can know before the engine starts.  It is addressed
    # by run_id, never by a source filename or its slug.  Once the engine writes
    # it, state/report/output paths are replaced with the engine's real values.
    artifact_metadata_path: Path | None = None
    project_id: str | None = None
    # New launches require the engine's indexed metadata contract.  Legacy
    # records may opt into the expensive identity scan when that index was not
    # available in the engine version that created them.
    allow_legacy_artifact_scan: bool = True

    def command_line(self) -> str:
        """Return the Windows-safe command line used by the desktop runner."""

        return subprocess.list2cmdline([self.program, *self.arguments])


@dataclass(frozen=True, slots=True)
class PipelineCompletion:
    report_path: Path
    output_files: list[Path]
    warnings: list[str]
    error_summary: str | None
    technical_details: str | None
    cost_estimate: float | None
    canonical_results: bool = False
    quality_status: str | None = None
    quality_report_paths: tuple[Path, ...] = ()
    legacy_technical_completion: bool = True


@dataclass(frozen=True, slots=True)
class RecoveredDraftProgress:
    """Verified candidate-owned previews retained by an interrupted draft run."""

    artifact_path: Path
    artifact: DraftArtifact
    ready_candidate_ids: list[str]
    invalid_candidate_ids: list[str]


@dataclass(frozen=True, slots=True)
class ApprovedDraftSelection:
    """The independently usable subset of a user's approved draft choices."""

    candidate_ids: list[str]
    errors: dict[str, str]


@dataclass(frozen=True, slots=True)
class ReportedPipelineFailure:
    """A current engine report whose terminal state is a real item failure."""

    prepared: PreparedPipelineRun
    report: dict[str, Any]
    terminal: dict[str, Any]


class PipelineFacade:
    """The only desktop layer that knows how to invoke the existing engine."""

    def __init__(self, engine_root: Path | RuntimeLayout) -> None:
        self.runtime = (
            engine_root
            if isinstance(engine_root, RuntimeLayout)
            else RuntimeLayout.for_source(engine_root, data=engine_root)
        )
        self.engine_root = self.runtime.data
        self.resources_root = self.runtime.resources

    def inspect_source(self, source_path: str | Path) -> dict[str, Any]:
        path = validate_video_path(source_path)
        metadata = probe_video(path)
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "duration": metadata.get("duration"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "fps": metadata.get("fps"),
            "video_codec": metadata.get("video_codec"),
            "audio_streams": metadata.get("audio_streams"),
        }

    def plan_processing(
        self, project: DesktopProject, settings: DesktopSettings,
    ) -> tuple[ProcessingIntent, ResolvedProcessingConfig, ProcessingEstimate]:
        """Resolve the user choices and an estimate before a pipeline process starts."""

        base_config = self._base_config(settings)
        # Planning is also used by the GUI before a child process is prepared.
        # Test harnesses and a first-run shell may not yet have an engine config;
        # the built-in defaults still provide a useful, explicitly low-confidence
        # estimate without hiding an eventual prepare-time configuration error.
        config = load_config(base_config if base_config.is_file() else None)
        intent = project.settings.processing_intent()
        source_metadata = dict(project.source_metadata)
        analysis_path = Path(str(project.analysis_artifact_path or ""))
        if analysis_path.is_file():
            try:
                artifact = AnalysisArtifact.read(analysis_path)
            except (AnalysisArtifactError, OSError):
                artifact = None
            if artifact is not None and (
                not project.analysis_id or artifact.analysis_id == project.analysis_id
            ):
                # Draft/render planning happens after content understanding.
                # Feed that persisted evidence into auto selection instead of
                # relying only on a filename heuristic.
                source_metadata.update(artifact.content_profile)
        resolved = resolve_processing_intent(intent, source_metadata)
        ai_available = not settings.local_test_mode and config.ai.provider != "mock"
        tts_available = not settings.local_test_mode and config.tts.provider == "openai"
        pricing = CostPricing(
            input_token_price=config.ai.input_token_price,
            output_token_price=config.ai.output_token_price,
            tts_cost_per_1m_characters=config.tts.cost_per_1m_characters,
            ai_available=ai_available,
            tts_available=tts_available,
        )
        estimate = estimate_processing(
            resolved,
            project.source_metadata,
            paid_ai_available=ai_available,
            pricing=pricing,
        )
        return intent, resolved, estimate

    def prepare(self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings) -> PreparedPipelineRun:
        source_path = validate_video_path(project.source)
        config = load_config(self._base_config(settings))
        _intent, resolved, _estimate = self.plan_processing(project, settings)
        self._apply_project_options(config, project, settings, resolved)

        run_directory = Path(project.project_directory) / "runs" / run.run_id
        config_path = run_directory / "runtime-config.yaml"
        config_path.write_text(
            yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        # The engine is executed as a child of QProcess, where Python's normal
        # stdout buffering hides useful diagnostics until the process exits.
        arguments = ["process", "--input", str(source_path), "--config", str(config_path), "--run-id", run.run_id, "--project-id", project.project_id, "--transform-script"]
        if settings.local_test_mode:
            arguments.extend(["--mock-ai", "--no-ai-transformation"])
        if project.settings.recompute_all or not project.settings.use_cache:
            arguments.extend([
                "--recompute-intelligence", "--recompute-transformation", "--recompute-production-plan",
                "--recompute-tts", "--recompute-audio", "--recompute-production-render",
            ])
        return self._pending_prepared(
            arguments, source_path, config_path, run.run_id, project.project_id,
            {
                "device": str(config.device),
                "encoder": str(config.production_render.encoder),
                "cache": str(config.production_render.cache_enabled).lower(),
                "mock_ai": str(settings.local_test_mode).lower(),
                "ai_provider": str(config.ai.provider),
                "processing_mode": resolved.processing_mode,
                "platform": resolved.platform.platform,
            },
            source_duration_seconds=_source_duration_seconds(project),
        )

    def prepare_analysis(self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings) -> PreparedPipelineRun:
        """Prepare the isolated analysis contract; it cannot start delivery."""

        source_path, config, resolved, config_path = self._prepare_mode_paths(project, run, settings)
        arguments = [
            "analyze", "--input", str(source_path), "--config", str(config_path),
            "--run-id", run.run_id, "--project-id", project.project_id,
        ]
        if settings.local_test_mode:
            arguments.extend(["--mock-ai", "--no-ai-rerank"])
        if project.settings.recompute_all or not project.settings.use_cache:
            arguments.append("--recompute-intelligence")
        return self._prepared_mode(
            arguments, source_path, config_path, run.run_id, project.project_id,
            {"mode": "analysis", "device": str(config.device), "cache": str(project.settings.use_cache).lower(),
             "processing_mode": resolved.processing_mode, "platform": resolved.platform.platform},
            source_duration_seconds=_source_duration_seconds(project),
        )

    def prepare_draft(
        self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings, candidate_ids: list[str],
    ) -> PreparedPipelineRun:
        """Prepare draft plan + fast preview from one immutable analysis artifact."""

        analysis_path = Path(str(project.analysis_artifact_path or ""))
        if not analysis_path.is_file() or not project.analysis_id:
            raise InputValidationError("Сначала завершите анализ и сохраните analysis.json.")
        if not candidate_ids:
            raise InputValidationError("Выберите хотя бы один кандидат для чернового просмотра.")
        source_path, config, resolved, config_path = self._prepare_mode_paths(project, run, settings)
        arguments = [
            "draft", "--input", str(source_path), "--config", str(config_path),
            "--run-id", run.run_id, "--project-id", project.project_id, "--analysis", str(analysis_path),
            "--analysis-id", project.analysis_id,
        ]
        if project.analysis_fingerprint:
            arguments.extend(["--analysis-fingerprint", project.analysis_fingerprint])
        for candidate_id in candidate_ids:
            arguments.extend(["--candidate-id", candidate_id])
            override = project.candidate_boundary_overrides.get(candidate_id)
            if isinstance(override, dict) and "start" in override and "end" in override:
                arguments.extend(["--candidate-boundary", f"{candidate_id}:{override['start']}:{override['end']}"])
        if settings.local_test_mode:
            arguments.extend(["--mock-ai", "--no-ai-transformation"])
        return self._prepared_mode(
            arguments, source_path, config_path, run.run_id, project.project_id,
            {"mode": "draft", "analysis_id": project.analysis_id, "candidate_count": str(len(candidate_ids)),
             "processing_mode": resolved.processing_mode, "platform": resolved.platform.platform},
            source_duration_seconds=_source_duration_seconds(project),
        )

    def prepare_selected_render(
        self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings, candidate_ids: list[str],
    ) -> PreparedPipelineRun:
        """Prepare the expensive 1080x1920 job only from reviewed draft.json."""

        if not project.candidate_draft_artifacts:
            raise InputValidationError("Сначала подготовьте и проверьте предпросмотр черновика.")
        if not candidate_ids:
            raise InputValidationError("Выберите черновики, из которых нужно создать готовые ролики.")
        source_path, config, resolved, config_path = self._prepare_mode_paths(project, run, settings)
        draft_path = self._compose_approved_draft(project, candidate_ids, config_path.parent)
        arguments = [
            "render", "--input", str(source_path), "--config", str(config_path),
            "--run-id", run.run_id, "--project-id", project.project_id, "--draft", str(draft_path),
            "--confirm-production",
        ]
        for candidate_id in candidate_ids:
            arguments.extend(["--candidate-id", candidate_id])
        if project.settings.subtitles_enabled is False:
            arguments.append("--disable-subtitles")
        return self._prepared_mode(
            arguments, source_path, config_path, run.run_id, project.project_id,
            {"mode": "selected_render", "draft_id": project.draft_id, "candidate_count": str(len(candidate_ids)),
             "encoder": str(config.production_render.encoder), "processing_mode": resolved.processing_mode,
             "platform": resolved.platform.platform},
            source_duration_seconds=_source_duration_seconds(project),
        )

    @staticmethod
    def inspect_approved_drafts(
        project: DesktopProject, candidate_ids: list[str],
    ) -> ApprovedDraftSelection:
        """Return every independently composable approved draft.

        This is deliberately a non-writing preflight.  A stale or corrupted
        artifact is an item failure, not a reason to throw away another
        candidate's immutable draft.  Detailed ProductionPlan validation stays
        in the engine hand-off, where it is reported per candidate with the
        relevant validation stage.
        """

        artifacts: dict[Path, DraftArtifact] = {}
        analyses: dict[Path, AnalysisArtifact] = {}
        baseline: DraftArtifact | None = None
        valid: list[str] = []
        errors: dict[str, str] = {}
        for candidate_id in dict.fromkeys(str(item) for item in candidate_ids if str(item)):
            raw_path = project.candidate_draft_artifacts.get(candidate_id)
            if not raw_path:
                errors[candidate_id] = "Сохранённый предпросмотр черновика не найден."
                continue
            try:
                path = Path(str(raw_path)).resolve()
            except OSError:
                errors[candidate_id] = "Не удалось открыть сохранённый предпросмотр черновика."
                continue
            if not path.is_file():
                errors[candidate_id] = "Сохранённый предпросмотр черновика больше недоступен."
                continue
            try:
                artifact = artifacts.setdefault(path, DraftArtifact.read(path))
            except (DraftArtifactError, OSError, ValueError):
                errors[candidate_id] = "Сохранённый предпросмотр черновика повреждён; создайте его заново."
                continue
            if (
                artifact.project_id and artifact.project_id != project.project_id
            ) or (
                project.analysis_id and artifact.analysis_id != project.analysis_id
            ) or (
                project.analysis_fingerprint and artifact.analysis_fingerprint != project.analysis_fingerprint
            ):
                errors[candidate_id] = "Этот предпросмотр создан для другой версии проекта; обновите его."
                continue
            try:
                analysis_path = Path(artifact.analysis_artifact_path).resolve()
            except OSError:
                errors[candidate_id] = "Не удалось открыть сохранённый анализ для этого черновика."
                continue
            if not analysis_path.is_file():
                errors[candidate_id] = "Для этого черновика исходный анализ больше недоступен."
                continue
            try:
                analysis = analyses.setdefault(analysis_path, AnalysisArtifact.read(analysis_path))
            except (AnalysisArtifactError, OSError, ValueError):
                errors[candidate_id] = "Сохранённый анализ повреждён; повторите анализ видео."
                continue
            if (
                analysis.analysis_id != artifact.analysis_id
                or analysis.analysis_fingerprint != artifact.analysis_fingerprint
                or analysis.source_fingerprint != artifact.source_fingerprint
                or (analysis.project_id and analysis.project_id != project.project_id)
            ):
                errors[candidate_id] = "Этот предпросмотр не совпадает с текущей версией проекта; обновите его."
                continue
            record = next(
                (item for item in artifact.candidates if str(item.get("candidate_id") or "") == candidate_id),
                None,
            )
            if not isinstance(record, dict) or record.get("state") not in {"draft_ready", "selected"}:
                errors[candidate_id] = "Выбранный черновик ещё не готов к созданию финального ролика."
                continue
            if baseline is None:
                baseline = artifact
            elif not PipelineFacade._same_draft_context(baseline, artifact):
                errors[candidate_id] = "Черновик создан для другой версии проекта; обновите его."
                continue
            valid.append(candidate_id)
        return ApprovedDraftSelection(valid, errors)

    @staticmethod
    def _same_draft_context(left: DraftArtifact, right: DraftArtifact) -> bool:
        return (
            left.analysis_id == right.analysis_id
            and left.analysis_fingerprint == right.analysis_fingerprint
            and left.analysis_artifact_path == right.analysis_artifact_path
            and left.source_fingerprint == right.source_fingerprint
            and left.project_id == right.project_id
        )

    @staticmethod
    def _compose_approved_draft(project: DesktopProject, candidate_ids: list[str], run_directory: Path) -> Path:
        """Combine ready candidate drafts in the user's explicit output order.

        Each fast-preview run is immutable and may cover only a subset of
        candidates.  The production command still receives one trusted draft
        contract, assembled here without touching analysis or re-planning.
        """

        artifacts: dict[Path, DraftArtifact] = {}
        records: list[dict[str, Any]] = []
        baseline: DraftArtifact | None = None
        for candidate_id in candidate_ids:
            raw_path = project.candidate_draft_artifacts.get(candidate_id)
            path = Path(str(raw_path or "")).resolve()
            if not raw_path or not path.is_file():
                raise InputValidationError("Черновик для выбранного момента больше не доступен.")
            try:
                artifact = artifacts.setdefault(path, DraftArtifact.read(path))
            except DraftArtifactError as error:
                raise InputValidationError("Сохранённый предпросмотр черновика повреждён.") from error
            if baseline is None:
                baseline = artifact
            elif (
                artifact.analysis_id != baseline.analysis_id
                or artifact.analysis_fingerprint != baseline.analysis_fingerprint
                or artifact.analysis_artifact_path != baseline.analysis_artifact_path
                or artifact.source_fingerprint != baseline.source_fingerprint
                or artifact.project_id != baseline.project_id
            ):
                raise InputValidationError("Выбранные черновики относятся к разным анализам и не могут быть объединены.")
            record = next(
                (item for item in artifact.candidates if str(item.get("candidate_id") or "") == candidate_id),
                None,
            )
            if not isinstance(record, dict) or record.get("state") not in {"draft_ready", "selected"}:
                raise InputValidationError("Выбранный черновик ещё не готов к созданию финального ролика.")
            records.append(dict(record))
        if baseline is None:  # Defensive: caller already rejected an empty selection.
            raise InputValidationError("Выберите хотя бы один готовый черновик.")
        approved = new_draft_artifact(
            draft_id=f"approved-{run_directory.name}",
            analysis_id=baseline.analysis_id,
            analysis_fingerprint=baseline.analysis_fingerprint,
            analysis_artifact_path=baseline.analysis_artifact_path,
            project_id=baseline.project_id or project.project_id,
            source_fingerprint=baseline.source_fingerprint,
            candidates=records,
            warnings=[warning for artifact in artifacts.values() for warning in artifact.warnings],
        )
        path = run_directory / "approved-draft.json"
        approved.write(path)
        return path

    def _prepare_mode_paths(
        self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings,
    ) -> tuple[Path, Any, ResolvedProcessingConfig, Path]:
        source_path = validate_video_path(project.source)
        config = load_config(self._base_config(settings))
        _intent, resolved, _estimate = self.plan_processing(project, settings)
        self._apply_project_options(config, project, settings, resolved)
        run_directory = Path(project.project_directory) / "runs" / run.run_id
        config_path = run_directory / "runtime-config.yaml"
        config_path.write_text(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return source_path, config, resolved, config_path

    def _prepared_mode(
        self, arguments: list[str], source_path: Path, config_path: Path, run_id: str, project_id: str,
        runtime_flags: dict[str, str], *,
        source_duration_seconds: float | None = None,
    ) -> PreparedPipelineRun:
        return self._pending_prepared(
            arguments, source_path, config_path, run_id, project_id, runtime_flags,
            source_duration_seconds=source_duration_seconds,
        )

    def _pending_prepared(
        self, arguments: list[str], source_path: Path, config_path: Path, run_id: str, project_id: str,
        runtime_flags: dict[str, str], *, source_duration_seconds: float | None = None,
    ) -> PreparedPipelineRun:
        """Prepare a launch without deriving engine output locations in GUI code."""

        metadata_path = run_metadata_path(self.engine_root, run_id)
        command = self.runtime.internal_cli_command(arguments)
        return PreparedPipelineRun(
            program=str(command.program), arguments=list(command.arguments),
            working_directory=command.working_directory,
            # These are a harmless pre-engine sentinel.  No completion logic
            # reads them: it resolves the engine metadata first.
            state_path=metadata_path, report_path=metadata_path,
            output_directory=metadata_path.parent, runtime_config_path=config_path, source_path=source_path,
            run_id=run_id, runtime_flags=runtime_flags,
            source_duration_seconds=source_duration_seconds,
            artifact_metadata_path=metadata_path, project_id=project_id,
            allow_legacy_artifact_scan=False,
        )

    def prepare_render_revision(
        self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings, parent_run: ProjectRun,
    ) -> PreparedPipelineRun:
        """Prepare the dependency-bounded CLI path: render only, never analysis or audio."""

        source_path = validate_video_path(project.source)
        config = load_config(self._base_config(settings))
        _intent, resolved, _estimate = self.plan_processing(project, settings)
        self._apply_project_options(config, project, settings, resolved)
        run_directory = Path(project.project_directory) / "runs" / run.run_id
        config_path = run_directory / "runtime-config.yaml"
        config_path.write_text(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        parent_prepared = self.prepared_from_execution(parent_run)
        parent_resolved = self.resolve_engine_paths(parent_prepared) if parent_prepared else None
        parent_output = parent_resolved.output_directory if parent_resolved else None
        # Very old desktop records retained only this engine-returned output
        # directory.  It is still an explicit artifact path, not a slug we
        # reconstruct from the source title.
        if parent_output is None:
            parent_execution = parent_run.settings_snapshot.get("execution", {})
            parent_value = parent_execution.get("output_directory") if isinstance(parent_execution, dict) else None
            parent_output = Path(str(parent_value)) if isinstance(parent_value, str) and parent_value.strip() else None
        if parent_output is None or not parent_output.is_dir():
            raise InputValidationError("Не найдена run directory исходного успешного запуска для повторного экспорта.")
        candidate_ids: list[str] = []
        parent_report = read_json(parent_output / "report.json", {})
        primary_results = parent_report.get("primary_results", []) if isinstance(parent_report, dict) else []
        if project.last_final_result_id and isinstance(primary_results, list):
            selected = next((
                item for item in primary_results
                if isinstance(item, dict)
                and str(item.get("clip_result_id") or "") == project.last_final_result_id
            ), None)
            if isinstance(selected, dict) and selected.get("candidate_id"):
                candidate_ids = [str(selected["candidate_id"])]
        if not candidate_ids:
            candidate_ids = [
                str(item) for item in parent_run.settings_snapshot.get("candidate_ids", [])
                if str(item)
            ]
        arguments = [
            "process", "--input", str(source_path), "--config", str(config_path), "--run-id", run.run_id, "--project-id", project.project_id,
            "--upstream-run-directory", str(parent_output),
            "--production-render-only", "--recompute-production-render",
        ]
        for candidate_id in candidate_ids:
            arguments.extend(["--candidate-id", candidate_id])
        return self._pending_prepared(
            arguments, source_path, config_path, run.run_id, project.project_id,
            {
                "render_only": "true", "device": str(config.device), "encoder": str(config.production_render.encoder),
                "cache": str(config.production_render.cache_enabled).lower(), "processing_mode": resolved.processing_mode,
                "platform": resolved.platform.platform, "candidate_count": str(len(candidate_ids)),
            },
            source_duration_seconds=_source_duration_seconds(project),
        )

    def completion(self, prepared: PreparedPipelineRun) -> PipelineCompletion:
        resolved = self.resolve_engine_paths(prepared)
        if resolved is None:
            return self._failed_completion(
                prepared,
                "Итоговый отчёт обработки не найден.",
                "Pipeline launch metadata could not be resolved.",
            )
        prepared = resolved
        if not prepared.report_path.is_file():
            return self._failed_completion(
                prepared, "Итоговый отчёт обработки не найден.",
                f"Expected report is missing: {prepared.report_path}",
            )
        try:
            raw = read_json(prepared.report_path, {})
        except (OSError, ValueError) as error:
            return self._failed_completion(
                prepared, "Не удалось прочитать итоговый отчёт обработки.",
                f"Report cannot be parsed: {prepared.report_path}; {error}",
            )
        if not isinstance(raw, dict):
            return self._failed_completion(
                prepared, "Не удалось прочитать итоговый отчёт обработки.",
                f"Report root is not a JSON object: {prepared.report_path}",
            )
        terminal = raw.get("terminal", {})
        if isinstance(terminal, dict) and terminal.get("status") == "failed":
            code = str(terminal.get("error_code") or "PIPELINE_FAILED")
            message = str(terminal.get("message") or "Pipeline завершился без финального ролика.")
            return self._failed_completion(prepared, message, f"{code}: {message}")
        if isinstance(terminal, dict) and terminal.get("status") in {"analysis_ready", "draft_ready"}:
            outputs = [
                Path(str(value)) for value in raw.get("output_files", [])
                if isinstance(value, str) and Path(value).is_file()
            ]
            return PipelineCompletion(
                report_path=prepared.report_path,
                output_files=outputs,
                warnings=[str(value) for value in raw.get("warnings", [])],
                error_summary=None,
                technical_details=None,
                cost_estimate=None,
                canonical_results=False,
            )
        manifest: dict[str, Any] | None = None
        if prepared.run_id:
            manifest_path = prepared.manifest_path or prepared.output_directory / "manifest.json"
            try:
                manifest_value = read_json(manifest_path, {})
            except (OSError, ValueError) as error:
                return self._failed_completion(prepared, "Не удалось проверить результат текущей сборки.", str(error))
            if not isinstance(manifest_value, dict) or manifest_value.get("run_id") != prepared.run_id:
                return self._failed_completion(
                    prepared, "Не удалось проверить результат текущей сборки.",
                    f"Expected run_id={prepared.run_id} in {manifest_path}",
                )
            manifest = manifest_value
        production = raw.get("production_render", {})
        if not isinstance(production, dict):
            return self._failed_completion(
                prepared, "Не удалось создать итоговый видеофайл.",
                "report.production_render is missing or invalid; final MP4 contract is unavailable.",
            )
        status = str(production.get("status", ""))
        if status not in {"completed", "warning", "partial"}:
            return self._failed_completion(
                prepared, "Не удалось создать итоговый видеофайл.",
                f"production_render.status={status or 'missing'}; expected completed or warning.",
            )
        final_path = self._final_output_path(prepared, production)
        validation_error = self._validate_final_mp4(final_path)
        if validation_error and not isinstance(raw.get("primary_results"), list):
            return self._failed_completion(prepared, "Не удалось создать итоговый видеофайл.", validation_error)
        warnings = [str(value) for value in raw.get("warnings", [])]
        if status == "partial":
            warnings.append("Некоторые выбранные черновики не удалось собрать; их можно повторить отдельно.")
        production_warnings = production.get("warnings", [])
        if not isinstance(production_warnings, list):
            production_warnings = [production_warnings]
        warnings.extend(str(value) for value in production_warnings if str(value) not in warnings)
        persistence = raw.get("state_persistence", {})
        if isinstance(persistence, dict) and persistence.get("status") == "degraded":
            if STATE_PERSISTENCE_WARNING not in warnings:
                warnings.append(STATE_PERSISTENCE_WARNING)
        ai = raw.get("ai", {}) if isinstance(raw.get("ai"), dict) else {}
        tts = raw.get("tts", {}) if isinstance(raw.get("tts"), dict) else {}
        values = [ai.get("estimated_cost"), tts.get("estimated_cost")]
        estimate = sum(float(item) for item in values if isinstance(item, (int, float)))
        registry_value = manifest.get("primary_results") if manifest is not None else raw.get("primary_results")
        if isinstance(registry_value, list):
            registry = [
                item for value in registry_value
                if (item := ClipResult.from_dict(value)) is not None and item.primary
            ]
            distinct_registry = unique_primary_results(registry)
            if len(distinct_registry) != len(registry):
                warnings.append("Повторяющиеся итоговые ролики скрыты; доступны только уникальные результаты.")
            registry = distinct_registry
            if not registry:
                return self._failed_completion(
                    prepared,
                    "Не удалось создать итоговый видеофайл.",
                    "report.primary_results is present but contains no successful primary ClipResult entries.",
                )
            if manifest is not None:
                if any(result.run_id != prepared.run_id for result in registry):
                    return self._failed_completion(prepared, "Результат относится к другой сборке проекта.", "ClipResult.run_id mismatch.")
                if any(not result.clip_result_id or not result.revision_id for result in registry):
                    return self._failed_completion(
                        prepared,
                        "Не удалось проверить готовый ролик.",
                        "Canonical ClipResult must provide clip_result_id and revision_id.",
                    )
                if any(not is_run_scoped_path(Path(result.output_file), prepared.output_directory) for result in registry):
                    return self._failed_completion(prepared, "Готовый ролик сохранён в неожиданном месте.", "Canonical result path escapes run directory.")
            output_files = result_paths(registry, prepared.output_directory)
            for candidate in output_files:
                artifact_error = self._validate_final_mp4(candidate)
                if artifact_error:
                    return self._failed_completion(prepared, "Не удалось создать итоговый видеофайл.", artifact_error)
            quality_status, quality_paths, quality_warnings, quality_error = self._validate_quality_gate(
                raw, manifest, registry, prepared,
            )
            if quality_error:
                return self._failed_completion(
                    prepared,
                    "Final Quality Gate не подтвердил готовность результата.",
                    quality_error,
                )
            warnings.extend(item for item in quality_warnings if item not in warnings)
            cost = 0.0 if prepared.runtime_flags.get("render_only") == "true" else estimate or None
            return PipelineCompletion(
                prepared.report_path, output_files, warnings, None, None, cost, True,
                quality_status, tuple(quality_paths), quality_status is None,
            )
        output_files = [final_path]
        for value in raw.get("output_files", []) if isinstance(raw.get("output_files"), list) else []:
            path = Path(str(value))
            candidate = path if path.is_absolute() else prepared.output_directory / path
            if candidate == final_path or candidate in output_files:
                continue
            # The legacy renderer can leave report entries for a previous cache
            # run.  Missing paths are not current partial failures and must not
            # downgrade a valid production result.
            if not candidate.is_file():
                continue
            artifact_error = self._validate_final_mp4(candidate)
            if artifact_error:
                warnings.append("Один из дополнительных роликов не прошёл проверку и не был добавлен в результаты.")
                continue
            output_files.append(candidate)
        quality_status, quality_paths, quality_warnings, quality_error = self._validate_quality_gate(
            raw, manifest, [], prepared,
        )
        if quality_error:
            return self._failed_completion(
                prepared,
                "Final Quality Gate не подтвердил готовность результата.",
                quality_error,
            )
        warnings.extend(item for item in quality_warnings if item not in warnings)
        cost = 0.0 if prepared.runtime_flags.get("render_only") == "true" else estimate or None
        return PipelineCompletion(
            prepared.report_path, output_files, warnings, None, None, cost, False,
            quality_status, tuple(quality_paths), quality_status is None,
        )

    def recovery_completion(self, prepared: PreparedPipelineRun, started_at: str) -> PipelineCompletion | None:
        """Return a verified canonical completion suitable for a failed process.

        A non-zero process exit is not enough to discard outputs: the report
        must be newer than the run and its canonical ClipResult registry must
        still validate every primary MP4.
        """

        resolved = self.resolve_engine_paths(prepared)
        if resolved is None:
            return None
        prepared = resolved
        if not prepared.run_id and not self._report_is_current(prepared.report_path, started_at):
            return None
        completion = self.completion(prepared)
        recoverable_mode = prepared.runtime_flags.get("mode") in {"analysis", "draft"}
        if completion.error_summary or (not completion.canonical_results and not recoverable_mode):
            return None
        return completion

    def reported_failure(
        self, prepared: PreparedPipelineRun, started_at: str,
    ) -> ReportedPipelineFailure | None:
        """Read a current terminal failure without treating it as a generic crash.

        Draft and selected-render modes deliberately write a terminal report
        when every requested item is invalid.  The report owns the actionable
        candidate states; losing it because Qt reports exit code 2 would turn a
        recoverable per-item failure into a misleading batch failure.
        """

        resolved = self.resolve_engine_paths(prepared)
        if resolved is None or not resolved.report_path.is_file():
            return None
        if not resolved.run_id and not self._report_is_current(resolved.report_path, started_at):
            return None
        try:
            report = read_json(resolved.report_path, {})
        except (OSError, ValueError):
            return None
        if not isinstance(report, dict):
            return None
        terminal = report.get("terminal")
        if not isinstance(terminal, dict) or terminal.get("status") != "failed":
            return None
        mode = resolved.runtime_flags.get("mode")
        code = str(terminal.get("error_code") or "")
        # Only the two known item-level hand-off terminals are safe to
        # rehydrate as retryable review state.  Quality-gate failures can own
        # canonical output files and must continue through normal completion
        # validation instead of being mistaken for an empty batch.
        if (mode == "draft" and code != "NO_DRAFT_PREVIEWS") or (
            mode == "selected_render" and code != "NO_RENDERABLE_CLIPS"
        ):
            return None
        if mode not in {"draft", "selected_render"}:
            return None
        run_info = report.get("run")
        if not isinstance(run_info, dict):
            if resolved.run_id or resolved.project_id:
                return None
            run_info = {}
        report_run_id = str(run_info.get("run_id") or "")
        report_project_id = str(run_info.get("project_id") or "")
        legacy_current = resolved.allow_legacy_artifact_scan and self._report_is_current(
            resolved.report_path, started_at,
        )
        if resolved.run_id and report_run_id != resolved.run_id:
            if report_run_id or not legacy_current:
                return None
        if resolved.project_id and report_project_id != resolved.project_id:
            if report_project_id or not legacy_current:
                return None
        return ReportedPipelineFailure(resolved, report, terminal)

    def recover_draft_progress(
        self,
        prepared: PreparedPipelineRun,
        expected_candidate_ids: list[str],
    ) -> RecoveredDraftProgress | None:
        """Read only the explicit progress contract of this isolated draft run.

        The method intentionally does not search for MP4 files.  A file is
        eligible only when the atomic progress artifact binds its candidate ID,
        reviewed source range, deterministic preview path, and completed state.
        """

        resolved = self.resolve_engine_paths(prepared)
        if (
            resolved is None
            or not resolved.run_id
            or resolved.runtime_flags.get("mode") != "draft"
            or not expected_candidate_ids
            or len(expected_candidate_ids) != len(set(expected_candidate_ids))
        ):
            return None
        prepared = resolved
        progress_path = prepared.output_directory / "draft-progress.json"
        try:
            artifact = DraftArtifact.read(progress_path)
        except (DraftArtifactError, OSError, ValueError):
            return None
        if (
            artifact.run_id != prepared.run_id
            or artifact.status != "draft_partial"
            or (prepared.project_id and artifact.project_id != prepared.project_id)
            or artifact.analysis_id != prepared.runtime_flags.get("analysis_id")
        ):
            return None
        records = artifact.candidates
        record_ids = [str(record.get("candidate_id") or "") for record in records]
        if record_ids != expected_candidate_ids:
            return None

        ready: list[str] = []
        invalid: list[str] = []
        for index, record in enumerate(records, start=1):
            candidate_id = expected_candidate_ids[index - 1]
            if not self._valid_draft_binding(record, candidate_id, index, prepared.output_directory):
                return None
            if record.get("state") != "draft_ready":
                continue
            if self._valid_draft_preview(record, candidate_id, index, prepared.output_directory):
                ready.append(candidate_id)
            else:
                invalid.append(candidate_id)
        return RecoveredDraftProgress(progress_path, artifact, ready, invalid)

    @staticmethod
    def _valid_draft_binding(
        record: dict[str, Any], candidate_id: str, index: int, output_directory: Path,
    ) -> bool:
        if not isinstance(record, dict) or str(record.get("candidate_id") or "") != candidate_id:
            return False
        try:
            start = float(record["source_start_seconds"])
            end = float(record["source_end_seconds"])
            requested_index = int(record["requested_index"])
        except (KeyError, TypeError, ValueError):
            return False
        if end <= start or requested_index != index:
            return False
        # A non-ready record must not smuggle an output into recovery.
        if record.get("state") != "draft_ready":
            return not str(record.get("output_file") or "").strip()
        return True

    def _valid_draft_preview(
        self,
        record: dict[str, Any],
        candidate_id: str,
        index: int,
        output_directory: Path,
    ) -> bool:
        preview = record.get("preview")
        output_value = str(record.get("output_file") or "").strip()
        final_script = record.get("draft_final_script")
        plan = record.get("draft_production_plan")
        plan_metadata = plan.get("metadata") if isinstance(plan, dict) else None
        if (
            not isinstance(preview, dict)
            or not isinstance(final_script, dict)
            or not isinstance(plan_metadata, dict)
            or str(final_script.get("candidate_id") or "") != candidate_id
            or str(plan_metadata.get("candidate_id") or "") != candidate_id
            or preview.get("status") != "draft_ready"
            or not output_value
        ):
            return False
        preview_value = str(preview.get("output_file") or "").strip()
        if not preview_value or Path(preview_value) != Path(output_value):
            return False
        path = Path(output_value)
        if not path.is_absolute():
            path = output_directory / path
        if preview.get("kind") == "creative":
            candidate_output = (
                output_directory
                if index == 1
                else output_directory / "candidates" / safe_name(candidate_id, f"clip-{index:02d}")
            )
            expected = candidate_output / "creative-preview" / "creative-preview.mp4"
            try:
                same_path = path.resolve() == expected.resolve()
            except OSError:
                return False
            if not same_path or not is_run_scoped_path(path, output_directory):
                return False
            manifest_path = expected.parent / "parity-manifest.json"
            compiled_path = expected.parent / "compiled-render-plan.json"
            manifest = read_json(manifest_path, {}) if manifest_path.is_file() else {}
            compiled = read_json(compiled_path, {}) if compiled_path.is_file() else {}
            if (
                preview.get("render_profile") != "creative_preview"
                or not isinstance(manifest, dict)
                or not isinstance(compiled, dict)
                or str(preview.get("compiled_plan_hash") or "") != str(compiled.get("plan_hash") or "")
                or str(preview.get("parity_signature") or "") != str(compiled.get("parity_signature") or "")
                or str(manifest.get("plan_hash") or "") != str(compiled.get("plan_hash") or "")
                or str(manifest.get("parity_signature") or "") != str(compiled.get("parity_signature") or "")
                or str(manifest.get("profile_id") or "") != "creative_preview"
                or str(manifest.get("output_checksum") or "") != stable_file_hash(path)
            ):
                return False
            return self._validate_final_mp4(path) is None
        expected = output_directory / "drafts" / f"{index:02d}-{safe_name(candidate_id, f'candidate-{index:02d}')}" / "draft-preview.mp4"
        try:
            same_path = path.resolve() == expected.resolve()
        except OSError:
            return False
        if not same_path or not is_run_scoped_path(path, output_directory):
            return False
        try:
            bound_start = float(record["source_start_seconds"])
            bound_end = float(record["source_end_seconds"])
        except (KeyError, TypeError, ValueError):
            return False
        segments = preview.get("segments")
        if not isinstance(segments, list) or not segments:
            return False
        for segment in segments:
            try:
                start = float(segment["source_start_seconds"])
                end = float(segment["source_end_seconds"])
            except (KeyError, TypeError, ValueError):
                return False
            if end <= start or start < bound_start - 0.25 or end > bound_end + 0.25:
                return False
        return self._validate_final_mp4(path) is None

    def prepared_from_execution(self, run: ProjectRun) -> PreparedPipelineRun | None:
        """Reconstruct a read-only completion contract stored at launch time."""

        execution = run.settings_snapshot.get("execution", {})
        if not isinstance(execution, dict):
            return None
        runtime_config = execution.get("runtime_config_path", "")
        # A deliberately absent run_id is a legacy non-canonical completion
        # contract.  Do not promote it to the desktop history UUID: that would
        # incorrectly require a manifest that the legacy engine never wrote.
        stored_run_id = execution.get("run_id") if "run_id" in execution else run.run_id
        run_id = str(stored_run_id).strip() if stored_run_id else None
        project_id = str(execution.get("project_id") or run.project_id).strip() or None
        runtime_flags = execution.get("runtime_flags", {})
        source = execution.get("source_path")
        metadata_value = str(execution.get("artifact_metadata_path") or "").strip()
        allow_legacy_artifact_scan = execution.get("allow_legacy_artifact_scan")
        if not isinstance(allow_legacy_artifact_scan, bool):
            # Records written before the indexed-launch contract retain their
            # conservative compatibility scan.
            allow_legacy_artifact_scan = True
        if run_id and metadata_value:
            metadata_path = Path(metadata_value)
            return PreparedPipelineRun(
                program="", arguments=[], working_directory=self.engine_root,
                state_path=metadata_path, report_path=metadata_path, output_directory=metadata_path.parent,
                runtime_config_path=Path(str(runtime_config or metadata_path.with_name("runtime-config.yaml"))),
                source_path=Path(str(source)) if source else None,
                runtime_flags={str(key): str(value) for key, value in runtime_flags.items()} if isinstance(runtime_flags, dict) else {},
                run_id=run_id, artifact_metadata_path=metadata_path, project_id=project_id,
                allow_legacy_artifact_scan=allow_legacy_artifact_scan,
            )
        engine_paths = execution.get("engine_paths")
        if isinstance(engine_paths, dict) and all(
            isinstance(engine_paths.get(name), str) and engine_paths[name].strip()
            for name in ("state_path", "report_path", "output_directory")
        ):
            return PreparedPipelineRun(
                program="", arguments=[], working_directory=self.engine_root,
                state_path=Path(str(engine_paths["state_path"])),
                report_path=Path(str(engine_paths["report_path"])),
                output_directory=Path(str(engine_paths["output_directory"])),
                runtime_config_path=Path(str(runtime_config or engine_paths["report_path"])).with_name("runtime-config.yaml"),
                source_path=Path(str(source)) if source else None,
                runtime_flags={str(key): str(value) for key, value in runtime_flags.items()} if isinstance(runtime_flags, dict) else {},
                run_id=run_id,
                manifest_path=Path(str(engine_paths["manifest_path"])) if engine_paths.get("manifest_path") else None,
                heartbeat_path=Path(str(engine_paths["heartbeat_path"])) if engine_paths.get("heartbeat_path") else None,
                project_id=project_id,
            )
        required_paths = ("state_path", "report_path", "output_directory")
        if not all(isinstance(execution.get(name), str) and execution[name].strip() for name in required_paths):
            return None
        try:
            report_path = Path(str(execution["report_path"]))
            output_directory = Path(str(execution["output_directory"]))
            state_path = Path(str(execution["state_path"]))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(str(path) for path in (report_path, output_directory, state_path)):
            return None
        manifest_value = str(execution.get("manifest_path") or "").strip()
        manifest_path = Path(manifest_value) if manifest_value else None
        return PreparedPipelineRun(
            program="", arguments=[], working_directory=self.engine_root,
            state_path=state_path, report_path=report_path, output_directory=output_directory,
            runtime_config_path=Path(str(runtime_config or report_path.with_name("runtime-config.yaml"))),
            source_path=Path(str(source)) if source else None,
            runtime_flags={str(key): str(value) for key, value in runtime_flags.items()} if isinstance(runtime_flags, dict) else {},
            run_id=run_id,
            manifest_path=manifest_path,
            project_id=project_id,
        )

    def resolve_engine_paths(self, prepared: PreparedPipelineRun | None) -> PreparedPipelineRun | None:
        """Replace desktop placeholders with paths published by the engine.

        A missing indexed record intentionally triggers an identity scan for
        legacy reports/analysis artifacts.  It never reconstructs a source
        slug, so differences in URL titles or Unicode normalization are benign.
        """

        if prepared is None or not prepared.run_id:
            return prepared
        metadata = find_run_artifact_metadata(
            self.engine_root,
            run_id=prepared.run_id,
            project_id=prepared.project_id,
            preferred_path=prepared.artifact_metadata_path,
            allow_legacy_scan=prepared.allow_legacy_artifact_scan,
        )
        if metadata is None:
            return prepared
        paths = metadata["paths"]
        try:
            state_path = Path(str(paths["state_path"]))
            report_path = Path(str(paths["report_path"]))
            output_directory = Path(str(paths["output_directory"]))
        except (KeyError, TypeError, ValueError):
            return prepared
        heartbeat_value = paths.get("heartbeat_path")
        manifest_value = paths.get("manifest_path")
        return replace(
            prepared,
            state_path=state_path,
            report_path=report_path,
            output_directory=output_directory,
            heartbeat_path=Path(str(heartbeat_value)) if heartbeat_value else None,
            manifest_path=Path(str(manifest_value)) if manifest_value else None,
            artifact_metadata_path=Path(str(metadata["metadata_path"])) if metadata.get("metadata_path") else prepared.artifact_metadata_path,
            project_id=str(metadata.get("project_id") or prepared.project_id or "") or None,
            # runtime config remains a desktop launch setting; work_directory is
            # only used for observability, so keep the engine root as QProcess cwd.
        )

    @staticmethod
    def _failed_completion(prepared: PreparedPipelineRun, summary: str, details: str) -> PipelineCompletion:
        return PipelineCompletion(prepared.report_path, [], [], summary, details, None)

    @staticmethod
    def _validate_quality_gate(
        raw: dict[str, Any],
        manifest: dict[str, Any] | None,
        registry: list[ClipResult],
        prepared: PreparedPipelineRun,
    ) -> tuple[str | None, list[Path], list[str], str | None]:
        """Trust only persisted V2 reports; older runs remain legacy technical completion."""

        report_gate = raw.get("quality_gate")
        manifest_gate = manifest.get("quality_gate") if isinstance(manifest, dict) else None
        if report_gate is None and manifest_gate is None:
            return None, [], [], None
        if not isinstance(report_gate, dict) or not isinstance(manifest_gate, dict):
            return None, [], [], "V2 Quality Gate must be present in both report.json and manifest.json."
        if (
            report_gate.get("schema_version") != QUALITY_REPORT_SCHEMA_VERSION
            or manifest_gate.get("schema_version") != QUALITY_REPORT_SCHEMA_VERSION
            or report_gate.get("status") != manifest_gate.get("status")
        ):
            return None, [], [], "report.json and manifest.json disagree on the V2 Quality Gate contract."
        status = str(report_gate.get("status") or "")
        references = report_gate.get("reports")
        manifest_references = manifest_gate.get("reports")
        if status not in {"PASS", "PASS_WITH_WARNINGS", "BLOCKED"} or not isinstance(references, list) or not references:
            return None, [], [], "V2 Quality Gate is missing a valid persisted QualityReport reference."
        if not isinstance(manifest_references, list) or {
            str(item.get("report_id")) for item in references if isinstance(item, dict)
        } != {
            str(item.get("report_id")) for item in manifest_references if isinstance(item, dict)
        }:
            return None, [], [], "report.json and manifest.json reference different QualityReports."
        by_report_id = {item.quality_report_id: item for item in registry if item.quality_report_id}
        paths: list[Path] = []
        report_statuses: list[dict[str, Any]] = []
        warnings: list[str] = []
        for reference in references:
            if not isinstance(reference, dict):
                return None, [], [], "QualityReport reference is not an object."
            report_path_value = reference.get("path")
            if not isinstance(report_path_value, str) or not report_path_value.strip():
                return None, [], [], "QualityReport reference is missing its path."
            report_path = Path(report_path_value)
            if not is_run_scoped_path(report_path, prepared.output_directory) or not report_path.is_file():
                return None, [], [], "QualityReport path is missing or outside the current run."
            report = read_quality_report(read_json(report_path, {}))
            if report is None:
                return None, [], [], f"QualityReport is invalid: {report_path}"
            if report.get("report_id") != reference.get("report_id") or report.get("status") != reference.get("status"):
                return None, [], [], "QualityReport reference does not match persisted report contents."
            result = by_report_id.get(str(report.get("report_id") or ""))
            if registry and result is None:
                return None, [], [], "Canonical ClipResult is missing its QualityReport reference."
            if result is not None:
                if (
                    report.get("artifact_id") != result.artifact_id
                    or report.get("candidate_id") != result.candidate_id
                    or report.get("edit_plan_id") != result.production_plan_id
                    or report.get("run_id") != result.run_id
                    or Path(str(report.get("artifact_path") or "")).resolve() != Path(result.output_file).resolve()
                ):
                    return None, [], [], "QualityReport artifact identity does not match canonical ClipResult."
                if report.get("artifact_sha256") != stable_file_hash(Path(result.output_file)):
                    return None, [], [], "QualityReport checksum does not match the final MP4."
            for finding in report.get("findings", []):
                if isinstance(finding, dict) and finding.get("severity") == "warning":
                    message = str(finding.get("user_message") or finding.get("code") or "Quality warning")
                    if message not in warnings:
                        warnings.append(message)
            paths.append(report_path)
            report_statuses.append(report)
        if aggregate_quality_status(report_statuses) != status:
            return None, [], [], "Quality Gate aggregate status does not match persisted QualityReports."
        if status == "BLOCKED":
            return None, [], [], "Final Quality Gate is BLOCKED; inspect persisted QualityReport findings."
        return status, paths, warnings, None

    @staticmethod
    def _report_is_current(report_path: Path, started_at: str) -> bool:
        try:
            started = datetime.fromisoformat(started_at)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            # File timestamps have lower precision on some Windows volumes.
            return report_path.is_file() and report_path.stat().st_mtime >= started.timestamp() - 2
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _final_output_path(prepared: PreparedPipelineRun, production: dict[str, Any]) -> Path:
        """Use the engine report first; final-short.mp4 is the documented fallback contract."""

        reported = production.get("output_file")
        if isinstance(reported, str) and reported.strip():
            candidate = Path(reported)
            return candidate if candidate.is_absolute() else prepared.output_directory / candidate
        return prepared.output_directory / "production-render" / "final-short.mp4"

    @staticmethod
    def _validate_final_mp4(path: Path) -> str | None:
        if not path.exists():
            return f"Expected final MP4 does not exist: {path}"
        if not path.is_file():
            return f"Expected final MP4 is not a file: {path}"
        if path.stat().st_size <= 0:
            return f"Expected final MP4 is empty: {path}"
        try:
            metadata = probe_video(path)
        except Exception as error:
            return f"Expected final MP4 is not readable by ffprobe: {path}; {error}"
        if not metadata.get("width") or not metadata.get("height"):
            return f"Expected final MP4 has no readable video stream: {path}"
        return None

    def _base_config(self, settings: DesktopSettings) -> Path:
        if settings.config_path:
            configured = Path(settings.config_path).expanduser()
            if configured.is_file():
                return configured
        example = self.resources_root / "config.example.yaml"
        if example.is_file():
            return example
        # Keep a locally renamed user configuration usable without changing the
        # normal repository contract.  Desktop settings still take precedence.
        renamed = self.resources_root / "config.yaml.yaml"
        return renamed if renamed.is_file() else example

    @staticmethod
    def _apply_project_options(
        config: Any, project: DesktopProject, settings: DesktopSettings,
        resolved: ResolvedProcessingConfig,
    ) -> None:
        """Turn user intent into established pipeline settings and production stages."""

        options = project.settings
        options.validate()
        config.transformation.enabled = True
        config.production.enabled = True
        config.tts.enabled = True
        config.audio_composition.enabled = True
        config.production_render.enabled = True
        config.production_render.subtitles_enabled = options.subtitles_enabled
        config.production_render.subtitle_style = options.subtitle_style
        config.production_render.crop_strategy = options.composition_strategy
        config.production_render.same_source_broll_allowed = options.same_source_broll_allowed
        config.production_render.encoder = options.encoder
        config.production_render.cache_enabled = options.use_cache
        config.device = settings.device_preference
        apply_resolved_processing_config(config, resolved)
        # A user can always choose not to reuse existing artifacts.  The product
        # preset controls normal cache policy; this explicit advanced switch wins.
        config.production_render.cache_enabled = options.use_cache and config.production_render.cache_enabled
        if settings.local_test_mode:
            config.ai.provider = "mock"
            config.tts.provider = "mock"
        config.validate()
