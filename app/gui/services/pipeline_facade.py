from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import load_config
from app.gui.models import DesktopProject, DesktopSettings, ProjectRun
from app.gui.services.desktop_project_store import validate_video_path
from app.media import probe_video
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


@dataclass(frozen=True, slots=True)
class PipelineCompletion:
    report_path: Path
    output_files: list[Path]
    warnings: list[str]
    error_summary: str | None
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

    def prepare(self, project: DesktopProject, run: ProjectRun, settings: DesktopSettings) -> PreparedPipelineRun:
        source_path = validate_video_path(project.source_path)
        config = load_config(self._base_config(settings))
        self._apply_project_options(config, project, settings)

        run_directory = Path(project.project_directory) / "runs" / run.run_id
        config_path = run_directory / "runtime-config.yaml"
        config_path.write_text(
            yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        source = local_source(str(source_path))
        run_key = f"{source.display_name}-{source.id[:12]}"
        work_directory = self.engine_root / "work" / run_key
        output_directory = self.engine_root / "output" / run_key
        arguments = ["-m", "app", "process", "--input", str(source_path), "--config", str(config_path), "--transform-script"]
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
        )

    def completion(self, prepared: PreparedPipelineRun) -> PipelineCompletion:
        if not prepared.report_path.is_file():
            return PipelineCompletion(prepared.report_path, [], [], "Итоговый отчёт обработки не найден.", None)
        raw = read_json(prepared.report_path, {})
        if not isinstance(raw, dict):
            return PipelineCompletion(prepared.report_path, [], [], "Итоговый отчёт имеет неверный формат.", None)
        paths = [Path(str(value)) for value in raw.get("output_files", [])]
        production = raw.get("production_render", {})
        if isinstance(production, dict) and production.get("output_file"):
            paths.append(Path(str(production["output_file"])))
        output_files = list(dict.fromkeys(path for path in paths if path.is_file()))
        warnings = [str(value) for value in raw.get("warnings", [])]
        ai = raw.get("ai", {}) if isinstance(raw.get("ai"), dict) else {}
        tts = raw.get("tts", {}) if isinstance(raw.get("tts"), dict) else {}
        values = [ai.get("estimated_cost"), tts.get("estimated_cost")]
        estimate = sum(float(item) for item in values if isinstance(item, (int, float)))
        return PipelineCompletion(prepared.report_path, output_files, warnings, None, estimate or None)

    def _base_config(self, settings: DesktopSettings) -> Path:
        if settings.config_path:
            configured = Path(settings.config_path).expanduser()
            if configured.is_file():
                return configured
        return self.engine_root / "config.example.yaml"

    @staticmethod
    def _apply_project_options(config: Any, project: DesktopProject, settings: DesktopSettings) -> None:
        """Turn on existing production stages; do not introduce desktop-only logic."""

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
        if settings.local_test_mode:
            config.ai.provider = "mock"
            config.tts.provider = "mock"
        config.validate()
