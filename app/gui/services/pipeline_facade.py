from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.config import load_config
from app.gui.models import DesktopProject, DesktopSettings, ProjectRun
from app.gui.services.desktop_project_store import validate_video_path
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
        source_path = validate_video_path(project.source_path)
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
        work_directory = self.engine_root / "work" / run_key
        output_directory = self.engine_root / "output" / run_key
        # The engine is executed as a child of QProcess, where Python's normal
        # stdout buffering hides useful diagnostics until the process exits.
        arguments = ["-u", "-m", "app", "process", "--input", str(source_path), "--config", str(config_path), "--transform-script"]
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
        if validation_error:
            return self._failed_completion(prepared, "Не удалось создать итоговый видеофайл.", validation_error)
        warnings = [str(value) for value in raw.get("warnings", [])]
        production_warnings = production.get("warnings", [])
        if not isinstance(production_warnings, list):
            production_warnings = [production_warnings]
        warnings.extend(str(value) for value in production_warnings if str(value) not in warnings)
        ai = raw.get("ai", {}) if isinstance(raw.get("ai"), dict) else {}
        tts = raw.get("tts", {}) if isinstance(raw.get("tts"), dict) else {}
        values = [ai.get("estimated_cost"), tts.get("estimated_cost")]
        estimate = sum(float(item) for item in values if isinstance(item, (int, float)))
        return PipelineCompletion(prepared.report_path, [final_path], warnings, None, None, estimate or None)

    @staticmethod
    def _failed_completion(prepared: PreparedPipelineRun, summary: str, details: str) -> PipelineCompletion:
        return PipelineCompletion(prepared.report_path, [], [], summary, details, None)

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
