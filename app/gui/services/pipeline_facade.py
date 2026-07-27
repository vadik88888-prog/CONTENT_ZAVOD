from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.clip_results import ClipResult, result_paths, unique_primary_results
from app.run_manifest import is_run_scoped_path
from app.config import load_config
from app.gui.models import DesktopProject, DesktopSettings, ProjectRun
from app.gui.services.desktop_project_store import InputValidationError, validate_video_path
from app.media import probe_video
from app.product_flow import (
    ProcessingEstimate,
    ProcessingIntent,
    ResolvedProcessingConfig,
    apply_resolved_processing_config,
    estimate_processing,
    resolve_processing_intent,
)
from app.sources import local_source
from app.utils import read_json


STATE_PERSISTENCE_WARNING = "Ролики созданы, но не удалось сохранить служебное состояние"


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


class PipelineFacade:
    """The only desktop layer that knows how to invoke the existing engine."""

    def __init__(self, engine_root: Path) -> None:
        self.engine_root = engine_root.resolve()

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
        resolved = resolve_processing_intent(intent, project.source_metadata)
        paid_ai_available = not settings.local_test_mode and (
            config.ai.provider != "mock" or config.tts.provider != "mock"
        )
        estimate = estimate_processing(
            resolved,
            project.source_metadata,
            paid_ai_available=paid_ai_available,
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

        source = local_source(str(source_path))
        run_key = f"{source.display_name}-{source.id[:12]}"
        work_directory = self.engine_root / "work" / run_key / "runs" / run.run_id
        output_directory = self.engine_root / "output" / run_key / "runs" / run.run_id
        # The engine is executed as a child of QProcess, where Python's normal
        # stdout buffering hides useful diagnostics until the process exits.
        arguments = ["-u", "-m", "app", "process", "--input", str(source_path), "--config", str(config_path), "--run-id", run.run_id, "--transform-script"]
        if settings.local_test_mode:
            arguments.extend(["--mock-ai", "--no-ai-transformation"])
        if project.settings.recompute_all or not project.settings.use_cache:
            arguments.extend([
                "--recompute-intelligence", "--recompute-transformation", "--recompute-production-plan",
                "--recompute-tts", "--recompute-audio", "--recompute-production-render",
            ])
        return PreparedPipelineRun(
            program=sys.executable,
            arguments=arguments,
            working_directory=self.engine_root,
            state_path=work_directory / "state.json",
            report_path=output_directory / "report.json",
            output_directory=output_directory,
            runtime_config_path=config_path,
            source_path=source_path,
            run_id=run.run_id,
            manifest_path=output_directory / "manifest.json",
            runtime_flags={
                "device": str(config.device),
                "encoder": str(config.production_render.encoder),
                "cache": str(config.production_render.cache_enabled).lower(),
                "mock_ai": str(settings.local_test_mode).lower(),
                "ai_provider": str(config.ai.provider),
                "processing_mode": resolved.processing_mode,
                "platform": resolved.platform.platform,
            },
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
        source = local_source(str(source_path))
        parent_execution = parent_run.settings_snapshot.get("execution", {})
        parent_value = parent_execution.get("output_directory") if isinstance(parent_execution, dict) else None
        parent_output = Path(str(parent_value)) if isinstance(parent_value, str) and parent_value.strip() else None
        if parent_output is None or not parent_output.is_dir():
            raise InputValidationError("Не найдена run directory исходного успешного запуска для повторного экспорта.")
        run_key = f"{source.display_name}-{source.id[:12]}"
        work_directory = self.engine_root / "work" / run_key / "runs" / run.run_id
        output_directory = self.engine_root / "output" / run_key / "runs" / run.run_id
        arguments = [
            "-u", "-m", "app", "process", "--input", str(source_path), "--config", str(config_path), "--run-id", run.run_id,
            "--upstream-run-directory", str(parent_output),
            "--production-render-only", "--recompute-production-render",
        ]
        return PreparedPipelineRun(
            program=sys.executable, arguments=arguments, working_directory=self.engine_root,
            state_path=work_directory / "state.json", report_path=output_directory / "report.json",
            output_directory=output_directory, runtime_config_path=config_path, source_path=source_path,
            run_id=run.run_id, manifest_path=output_directory / "manifest.json",
            runtime_flags={
                "render_only": "true", "device": str(config.device), "encoder": str(config.production_render.encoder),
                "cache": str(config.production_render.cache_enabled).lower(), "processing_mode": resolved.processing_mode,
                "platform": resolved.platform.platform,
            },
        )

    def completion(self, prepared: PreparedPipelineRun) -> PipelineCompletion:
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
        manifest: dict[str, Any] | None = None
        if prepared.run_id:
            manifest_path = prepared.manifest_path or prepared.output_directory / "manifest.json"
            try:
                manifest_value = read_json(manifest_path, {})
            except (OSError, ValueError) as error:
                return self._failed_completion(prepared, "Не удалось прочитать manifest текущего запуска.", str(error))
            if not isinstance(manifest_value, dict) or manifest_value.get("run_id") != prepared.run_id:
                return self._failed_completion(
                    prepared, "Manifest текущего запуска отсутствует или повреждён.",
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
        if status not in {"completed", "warning"}:
            return self._failed_completion(
                prepared, "Не удалось создать итоговый видеофайл.",
                f"production_render.status={status or 'missing'}; expected completed or warning.",
            )
        final_path = self._final_output_path(prepared, production)
        validation_error = self._validate_final_mp4(final_path)
        if validation_error and not isinstance(raw.get("primary_results"), list):
            return self._failed_completion(prepared, "Не удалось создать итоговый видеофайл.", validation_error)
        warnings = [str(value) for value in raw.get("warnings", [])]
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
                    return self._failed_completion(prepared, "Manifest содержит результат другого запуска.", "ClipResult.run_id mismatch.")
                if any(not is_run_scoped_path(Path(result.output_file), prepared.output_directory) for result in registry):
                    return self._failed_completion(prepared, "Manifest содержит путь вне текущего запуска.", "Canonical result path escapes run directory.")
            output_files = result_paths(registry, prepared.output_directory)
            for candidate in output_files:
                artifact_error = self._validate_final_mp4(candidate)
                if artifact_error:
                    return self._failed_completion(prepared, "Не удалось создать итоговый видеофайл.", artifact_error)
            cost = 0.0 if prepared.runtime_flags.get("render_only") == "true" else estimate or None
            return PipelineCompletion(prepared.report_path, output_files, warnings, None, None, cost, True)
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
        cost = 0.0 if prepared.runtime_flags.get("render_only") == "true" else estimate or None
        return PipelineCompletion(prepared.report_path, output_files, warnings, None, None, cost)

    def recovery_completion(self, prepared: PreparedPipelineRun, started_at: str) -> PipelineCompletion | None:
        """Return a verified canonical completion suitable for a failed process.

        A non-zero process exit is not enough to discard outputs: the report
        must be newer than the run and its canonical ClipResult registry must
        still validate every primary MP4.
        """

        if not prepared.run_id and not self._report_is_current(prepared.report_path, started_at):
            return None
        completion = self.completion(prepared)
        if completion.error_summary or not completion.canonical_results:
            return None
        return completion

    def prepared_from_execution(self, run: ProjectRun) -> PreparedPipelineRun | None:
        """Reconstruct a read-only completion contract stored at launch time."""

        execution = run.settings_snapshot.get("execution", {})
        if not isinstance(execution, dict):
            return None
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
        runtime_config = execution.get("runtime_config_path", "")
        run_id = str(execution.get("run_id") or "").strip() or None
        manifest_value = str(execution.get("manifest_path") or "").strip()
        if run_id and not manifest_value:
            return None
        manifest_path = Path(manifest_value) if manifest_value else None
        source = execution.get("source_path")
        runtime_flags = execution.get("runtime_flags", {})
        return PreparedPipelineRun(
            program="", arguments=[], working_directory=self.engine_root,
            state_path=state_path, report_path=report_path, output_directory=output_directory,
            runtime_config_path=Path(str(runtime_config or report_path.with_name("runtime-config.yaml"))),
            source_path=Path(str(source)) if source else None,
            runtime_flags={str(key): str(value) for key, value in runtime_flags.items()} if isinstance(runtime_flags, dict) else {},
            run_id=run_id,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _failed_completion(prepared: PreparedPipelineRun, summary: str, details: str) -> PipelineCompletion:
        return PipelineCompletion(prepared.report_path, [], [], summary, details, None)

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
        example = self.engine_root / "config.example.yaml"
        if example.is_file():
            return example
        # Keep a locally renamed user configuration usable without changing the
        # normal repository contract.  Desktop settings still take precedence.
        renamed = self.engine_root / "config.yaml.yaml"
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
