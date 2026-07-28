from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from app.config import load_config
from app.doctor import collect_checks, format_report
from app.errors import ClipEngineError
from app.pipeline import Pipeline
from app.utils import read_json


def _load_dotenv(root: Path) -> None:
    path = root / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Content Factory — Clip Engine: локальный конвейер коротких видео.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Проверить программы и папки, нужные для работы.")
    doctor.add_argument("--config", type=Path, help="Путь к YAML-конфигурации.")
    process = commands.add_parser("process", help="Создать короткие вертикальные клипы из одного видео.")
    source = process.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Путь к локальному видеофайлу.")
    source.add_argument("--url", help="Публичная поддерживаемая ссылка для yt-dlp.")
    process.add_argument("--config", type=Path, help="Путь к YAML-конфигурации.")
    process.add_argument("--run-id", help="Обязательный идентификатор desktop run; без него CLI создаст новый isolated run.")
    process.add_argument("--upstream-run-directory", type=Path, help=argparse.SUPPRESS)
    process.add_argument("--project-id", help=argparse.SUPPRESS)
    process.add_argument(
        "--mock-ai", action="store_true",
        help="Детерминированная локальная оценка без внешнего AI API.",
    )
    process.add_argument(
        "--no-ai-rerank", action="store_true",
        help="Отключить LLM и выбрать клипы только по локальному ranking.",
    )
    process.add_argument(
        "--recompute-intelligence", action="store_true",
        help="Пересчитать признаки, candidates и ranking без удаления исходного файла.",
    )
    transformation_switch = process.add_mutually_exclusive_group()
    transformation_switch.add_argument(
        "--transform-script", dest="transform_script", action="store_true",
        help="Создать отдельный grounded transformed-script artifact после выбора клипа.",
    )
    transformation_switch.add_argument(
        "--no-transform-script", dest="transform_script", action="store_false",
        help="Не создавать transformed-script даже если transformation.enabled=true.",
    )
    process.set_defaults(transform_script=None)
    process.add_argument(
        "--transformation-mode",
        choices=["faithful_compression", "hook_first", "educational", "story", "listicle", "provocative", "calm_expert", "direct_response", "auto"],
        help="Режим narrative plan для transformed script.",
    )
    process.add_argument(
        "--transformation-ai-strategy", choices=["staged", "compact", "local_only"],
        help="Стратегия AI transformation: staged, compact или local_only.",
    )
    process.add_argument("--target-duration", type=float, help="Целевая длительность transformed script в секундах.")
    process.add_argument("--output-language", choices=["auto", "ru", "en"], help="Язык transformed script; перевод пока не реализован.")
    process.add_argument("--allow-translation", action="store_true", help="Явно разрешить будущую отдельную translation stage.")
    process.add_argument("--allow-cta", action="store_true", help="Разрешить CTA только при подтверждении source evidence.")
    process.add_argument(
        "--no-ai-transformation", action="store_true",
        help="Создать только conservative local fallback; AI reranking остаётся независимым.",
    )
    process.add_argument(
        "--recompute-transformation", action="store_true",
        help="Пересчитать только transformation stages и report, не запуская транскрибацию заново.",
    )
    process.add_argument(
        "--strict-grounding", action="store_true",
        help="Явно включить строгую детерминированную проверку grounding.",
    )
    process.add_argument(
        "--print-transformed-script", action="store_true",
        help="После process вывести text final transformed script в консоль.",
    )
    process.add_argument(
        "--production-plan-only", action="store_true",
        help="Построить FinalScript и Production Plan, но не запускать существующий render/ASS pipeline.",
    )
    process.add_argument(
        "--recompute-production-plan", action="store_true",
        help="Пересчитать только Production Plan и report, сохранив FinalScript cache.",
    )
    process.add_argument(
        "--tts-only", action="store_true",
        help="Синтезировать TTS только из уже существующего production-plan.json; render и план не запускаются.",
    )
    process.add_argument(
        "--recompute-tts", action="store_true",
        help="Игнорировать TTS cache и заново сгенерировать narration artifacts.",
    )
    process.add_argument(
        "--disable-tts", action="store_true",
        help="Отключить TTS для этого запуска без изменения Production Plan.",
    )
    process.add_argument("--tts-provider", choices=["openai", "mock", "local"], help="TTS provider для этого запуска.")
    process.add_argument("--tts-voice", help="Явно выбрать provider voice; по умолчанию auto mapping из VoiceProfile.")
    process.add_argument("--tts-model", help="Модель TTS provider для этого запуска.")
    process.add_argument("--tts-budget-limit", type=float, help="Максимальная оценочная стоимость TTS в USD для этого запуска.")
    process.add_argument(
        "--audio-only", action="store_true",
        help="Собрать только Audio Project из существующих ProductionPlan/TTS artifacts без TTS, Whisper, ASS или video render.",
    )
    process.add_argument(
        "--recompute-audio", action="store_true",
        help="Игнорировать cache извлечённого dialogue/normalised narration для Audio Project.",
    )
    process.add_argument(
        "--production-render-only", action="store_true",
        help="Собрать только финальный Goal 3D MP4 из существующих ProductionPlan и AudioProject без AI/TTS/audio mix/legacy render.",
    )
    process.add_argument(
        "--recompute-production-render", action="store_true",
        help="Игнорировать production render cache и заново собрать финальный MP4.",
    )
    process.add_argument(
        "--disable-production-render", action="store_true",
        help="Отключить production render для этого запуска, не меняя legacy render.",
    )
    process.add_argument("--output-width", type=int, help="Ширина Goal 3D холста 9:16 (чётное число).")
    process.add_argument("--output-height", type=int, help="Высота Goal 3D холста 9:16 (чётное число).")
    process.add_argument("--output-fps", type=float, help="FPS Goal 3D production render.")
    process.add_argument(
        "--crop-strategy", choices=["safe_auto", "center_crop", "fit_blur_background", "fit_solid_background", "top_crop", "manual_normalized_crop"],
        help="Стратегия vertical composition для production render.",
    )
    process.add_argument("--subtitle-style", choices=["minimal", "documentary", "dynamic", "clean"], help="Стиль production subtitles.")
    process.add_argument("--disable-subtitles", action="store_true", help="Не burn-in production ASS в финальный MP4.")
    process.add_argument("--video-encoder", choices=["auto", "nvenc", "cpu"], help="Encoder Goal 3D: auto, nvenc или cpu.")
    analyze = commands.add_parser("analyze", help="Analyse a source without starting render delivery.")
    analyze_source = analyze.add_mutually_exclusive_group(required=True)
    analyze_source.add_argument("--input", help="Path to a local video file.")
    analyze_source.add_argument("--url", help="Public source URL for yt-dlp.")
    analyze.add_argument("--config", type=Path, help="Path to a YAML configuration file.")
    analyze.add_argument("--run-id", help="Identifier for the isolated analysis run.")
    analyze.add_argument("--project-id", help=argparse.SUPPRESS)
    analyze.add_argument("--mock-ai", action="store_true", help="Use deterministic local scoring without an AI API.")
    analyze.add_argument("--no-ai-rerank", action="store_true", help="Disable LLM ranking for this analysis run.")
    analyze.add_argument("--recompute-intelligence", action="store_true", help="Recompute cached source intelligence.")
    draft = commands.add_parser("draft", help="Build Fast Draft Preview from explicit analysis candidates.")
    draft_source = draft.add_mutually_exclusive_group(required=True)
    draft_source.add_argument("--input", help="Path to the same local source video.")
    draft_source.add_argument("--url", help="Original public source URL.")
    draft.add_argument("--analysis", type=Path, required=True, help="Path to ready analysis.json.")
    draft.add_argument("--analysis-id", help="Expected analysis ID for the hand-off check.")
    draft.add_argument("--analysis-fingerprint", help="Expected analysis fingerprint for the hand-off check.")
    draft.add_argument("--candidate-id", action="append", required=True, help="Candidate ID to assemble; repeat in desired review order.")
    draft.add_argument("--candidate-boundary", action="append", default=[], help="Override one candidate boundary as candidate_id:start:end.")
    draft.add_argument("--project-id", required=True, help=argparse.SUPPRESS)
    draft.add_argument("--config", type=Path, help="Path to a YAML configuration file.")
    draft.add_argument("--run-id", help="Identifier for the isolated draft run.")
    draft.add_argument("--mock-ai", action="store_true", help="Use deterministic local transformation provider.")
    draft.add_argument("--no-ai-transformation", action="store_true", help="Use conservative local transformation only.")
    render = commands.add_parser("render", help="Production-render only user-approved draft candidates.")
    render_source = render.add_mutually_exclusive_group(required=True)
    render_source.add_argument("--input", help="Path to the same local source video.")
    render_source.add_argument("--url", help="Original public source URL.")
    render.add_argument("--draft", type=Path, required=True, help="Path to reviewed draft.json.")
    render.add_argument("--candidate-id", action="append", required=True, help="Approved draft candidate ID; repeat in output order.")
    render.add_argument("--confirm-production", action="store_true", required=True, help="Explicit user confirmation for the expensive 1080x1920 render.")
    render.add_argument("--project-id", required=True, help=argparse.SUPPRESS)
    render.add_argument("--config", type=Path, help="Path to a YAML configuration file with current render settings.")
    render.add_argument("--run-id", help="Identifier for the isolated production-render run.")
    render.add_argument("--disable-tts", action="store_true", help="Do not synthesize TTS for this render run.")
    render.add_argument("--output-width", type=int, help="Current production render width.")
    render.add_argument("--output-height", type=int, help="Current production render height.")
    render.add_argument("--output-fps", type=float, help="Current production output FPS.")
    render.add_argument("--crop-strategy", choices=["safe_auto", "center_crop", "fit_blur_background", "fit_solid_background", "top_crop", "manual_normalized_crop"])
    render.add_argument("--subtitle-style", choices=["minimal", "documentary", "dynamic", "clean"])
    render.add_argument("--disable-subtitles", action="store_true")
    render.add_argument("--video-encoder", choices=["auto", "nvenc", "cpu"])
    return parser


def main(argv: list[str] | None = None) -> int:
    root = Path.cwd()
    _load_dotenv(root)
    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        try:
            config = load_config(arguments.config)
        except ClipEngineError as error:
            print(f"Ошибка: {error}", file=sys.stderr)
            return 2
        print(format_report(collect_checks(root, config)))
        return 0
    if arguments.command == "analyze":
        try:
            config = load_config(arguments.config)
            result = Pipeline(
                root, config, mock_ai=arguments.mock_ai,
                no_ai_rerank=arguments.no_ai_rerank,
                recompute_intelligence=arguments.recompute_intelligence,
                analysis_only=True,
                run_id=arguments.run_id,
                project_id=arguments.project_id,
            ).run(input_path=arguments.input, url=arguments.url)
        except ClipEngineError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        print("Analysis is ready; no render was started.")
        print(f"Recommended candidates: {result.selected_clips}")
        print(f"Analysis ID: {result.analysis_id}")
        print(f"Analysis: {result.analysis_path}")
        print(f"Report: {result.report_path}")
        for warning in result.warnings:
            print(f"Warning: {warning}")
        return 0
    if arguments.command == "draft":
        try:
            config = load_config(arguments.config)
            _apply_draft_command_arguments(config)
            result = Pipeline(
                root, config,
                mock_ai=arguments.mock_ai,
                no_ai_transformation=arguments.no_ai_transformation,
                run_id=arguments.run_id,
                project_id=arguments.project_id,
                analysis_artifact_path=arguments.analysis,
                selected_candidate_ids=list(arguments.candidate_id),
                expected_analysis_id=arguments.analysis_id,
                expected_analysis_fingerprint=arguments.analysis_fingerprint,
                draft_only=True,
                candidate_boundary_overrides=_parse_boundary_overrides(arguments.candidate_boundary),
            ).run(input_path=arguments.input, url=arguments.url)
        except ClipEngineError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        report = read_json(result.report_path, {})
        terminal = report.get("terminal", {}) if isinstance(report, dict) else {}
        if isinstance(terminal, dict) and terminal.get("status") == "failed":
            print(f"Error: {terminal.get('message') or 'Draft preview failed.'}", file=sys.stderr)
            return 2
        print("Draft previews are ready; production render was not started.")
        print(f"Draft-ready candidates: {result.selected_clips}; previews: {len(result.output_files)}")
        print(f"Draft: {result.draft_path}")
        print(f"Report: {result.report_path}")
        for warning in result.warnings:
            print(f"Warning: {warning}")
        return 0
    if arguments.command == "render":
        try:
            config = load_config(arguments.config)
            _apply_render_command_arguments(config, arguments)
            result = Pipeline(
                root, config, disable_tts=arguments.disable_tts,
                run_id=arguments.run_id, project_id=arguments.project_id,
                draft_artifact_path=arguments.draft,
                selected_candidate_ids=list(arguments.candidate_id),
            ).run(input_path=arguments.input, url=arguments.url)
        except ClipEngineError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        report = read_json(result.report_path, {})
        terminal = report.get("terminal", {}) if isinstance(report, dict) else {}
        if isinstance(terminal, dict) and terminal.get("status") == "failed":
            print(f"Error: {terminal.get('message') or 'Production render failed.'}", file=sys.stderr)
            return 2
        print("Approved production render finished.")
        print(f"Candidates: {result.selected_clips}; outputs: {len(result.output_files)}")
        print(f"Results: {result.output_directory}")
        print(f"Report: {result.report_path}")
        for warning in result.warnings:
            print(f"Warning: {warning}")
        return 0
    try:
        config = load_config(arguments.config)
        only_modes = [arguments.tts_only, arguments.audio_only, arguments.production_render_only]
        if sum(bool(value) for value in only_modes) > 1:
            raise ClipEngineError("--tts-only, --audio-only и --production-render-only — отдельные изолированные этапы; выберите один.")
        if arguments.production_plan_only and (arguments.audio_only or arguments.production_render_only):
            raise ClipEngineError("--production-plan-only нельзя использовать вместе с --audio-only или --production-render-only.")
        if arguments.production_render_only and arguments.disable_production_render:
            raise ClipEngineError("--production-render-only несовместим с --disable-production-render.")
        _apply_transformation_arguments(config, arguments)
        _apply_tts_arguments(config, arguments)
        _apply_audio_arguments(config, arguments)
        _apply_production_render_arguments(config, arguments)
        result = Pipeline(
            root, config, mock_ai=arguments.mock_ai,
            no_ai_rerank=arguments.no_ai_rerank,
            recompute_intelligence=arguments.recompute_intelligence,
            transform_script=arguments.transform_script,
            no_ai_transformation=arguments.no_ai_transformation,
            recompute_transformation=arguments.recompute_transformation,
            production_plan_only=arguments.production_plan_only,
            recompute_production_plan=arguments.recompute_production_plan,
            tts_only=arguments.tts_only,
            recompute_tts=arguments.recompute_tts,
            disable_tts=arguments.disable_tts,
            audio_only=arguments.audio_only,
            recompute_audio=arguments.recompute_audio,
            production_render_only=arguments.production_render_only,
            recompute_production_render=arguments.recompute_production_render,
            disable_production_render=arguments.disable_production_render,
            run_id=arguments.run_id,
            upstream_run_directory=arguments.upstream_run_directory,
            project_id=arguments.project_id,
        ).run(
            input_path=arguments.input, url=arguments.url
        )
    except ClipEngineError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    report = read_json(result.report_path, {})
    terminal = report.get("terminal", {}) if isinstance(report, dict) else {}
    if isinstance(terminal, dict) and terminal.get("status") == "failed":
        code = str(terminal.get("error_code") or "PIPELINE_FAILED")
        message = str(terminal.get("message") or "Pipeline завершился без финального ролика.")
        print(f"Ошибка [{code}]: {message}", file=sys.stderr)
        print(f"Отчёт: {result.report_path}", file=sys.stderr)
        return 2
    print("Готово.")
    print(f"Клипов: {len(result.output_files)} (выбрано: {result.selected_clips})")
    print(f"Результаты: {result.output_directory}")
    print(f"Отчёт: {result.report_path}")
    production = report.get("production_plan", {}) if isinstance(report, dict) else {}
    if isinstance(production, dict) and production.get("enabled"):
        print(
            "Production Plan: "
            f"{production.get('status')} · narration={production.get('narration_count', 0)} "
            f"dialogue={production.get('dialogue_count', 0)} · "
            f"~{float(production.get('estimated_duration', 0) or 0):.1f} с"
        )
    tts = report.get("tts", {}) if isinstance(report, dict) else {}
    if isinstance(tts, dict) and tts.get("enabled"):
        print(
            "TTS: "
            f"{tts.get('status')} · provider={tts.get('provider')} · "
            f"generated={tts.get('generated_count', 0)} cache={tts.get('cache_hit_count', 0)} "
            f"fallback={tts.get('fallback_count', 0)}"
        )
    audio = report.get("audio", {}) if isinstance(report, dict) else {}
    if isinstance(audio, dict) and audio.get("enabled"):
        print(
            "Audio: "
            f"{audio.get('status')} · narration={audio.get('narration_count', 0)} "
            f"dialogue={audio.get('dialogue_count', 0)} · "
            f"{float(audio.get('mix_duration', 0) or 0):.1f} с"
        )
    production_render = report.get("production_render", {}) if isinstance(report, dict) else {}
    if isinstance(production_render, dict) and production_render.get("enabled"):
        print(
            "Production render: "
            f"{production_render.get('status')} · {production_render.get('resolution')} · "
            f"{float(production_render.get('duration', 0) or 0):.1f} с · "
            f"encoder={production_render.get('encoder')} cache={production_render.get('cache_hit')}"
        )
    if arguments.print_transformed_script:
        transformation = report.get("content_transformation", {}) if isinstance(report, dict) else {}
        final = transformation.get("final_script", {}) if isinstance(transformation, dict) else {}
        text = final.get("full_text") if isinstance(final, dict) else None
        if text:
            print("\nTransformed script:")
            print(text)
        else:
            print("\nTransformed script не создан (см. content_transformation в report.json).")
    for warning in result.warnings:
        print(f"Предупреждение: {warning}")
    return 0


def _apply_transformation_arguments(config, arguments: argparse.Namespace) -> None:
    """Apply process-only overrides before provider construction and validate them once."""

    transformation = config.transformation
    if arguments.transformation_mode:
        transformation.mode = arguments.transformation_mode
    if arguments.transformation_ai_strategy:
        transformation.ai_strategy = arguments.transformation_ai_strategy
    if arguments.target_duration is not None:
        transformation.target_duration_seconds = arguments.target_duration
    if arguments.output_language:
        transformation.output_language = arguments.output_language
    if arguments.allow_translation:
        transformation.allow_translation = True
    if arguments.allow_cta:
        transformation.allow_cta = True
    if arguments.strict_grounding:
        transformation.strict_grounding = True
    config.validate()


def _apply_tts_arguments(config, arguments: argparse.Namespace) -> None:
    """Apply ephemeral TTS overrides without changing the checked-in configuration."""

    tts = config.tts
    if arguments.tts_provider:
        tts.provider = arguments.tts_provider
    if arguments.tts_voice:
        tts.voice = arguments.tts_voice
    if arguments.tts_model:
        tts.model = arguments.tts_model
    if arguments.tts_budget_limit is not None:
        tts.budget_limit = arguments.tts_budget_limit
    if arguments.tts_only and not arguments.disable_tts:
        # `--tts-only` is an explicit request to execute the TTS service from an existing plan.
        tts.enabled = True
    config.validate()


def _apply_audio_arguments(config, arguments: argparse.Namespace) -> None:
    """Enable the isolated Audio Project flow only when the user requests it."""

    if arguments.audio_only:
        config.audio_composition.enabled = True
    config.validate()


def _apply_production_render_arguments(config, arguments: argparse.Namespace) -> None:
    """Apply Goal 3D-only overrides after all old config paths are settled."""

    render = config.production_render
    if arguments.production_render_only:
        render.enabled = True
    if arguments.output_width is not None:
        render.output_width = arguments.output_width
    if arguments.output_height is not None:
        render.output_height = arguments.output_height
    if arguments.output_fps is not None:
        render.output_fps = arguments.output_fps
    if arguments.crop_strategy:
        render.crop_strategy = arguments.crop_strategy
    if arguments.subtitle_style:
        render.subtitle_style = arguments.subtitle_style
    if arguments.disable_subtitles:
        render.subtitles_enabled = False
    if arguments.video_encoder:
        render.encoder = arguments.video_encoder
    config.validate()


def _apply_render_command_arguments(config, arguments: argparse.Namespace) -> None:
    """Apply the small, visual-only override surface of the selected renderer."""

    config.production.enabled = True
    config.production_render.enabled = True
    render = config.production_render
    if arguments.output_width is not None:
        render.output_width = arguments.output_width
    if arguments.output_height is not None:
        render.output_height = arguments.output_height
    if arguments.output_fps is not None:
        render.output_fps = arguments.output_fps
    if arguments.crop_strategy:
        render.crop_strategy = arguments.crop_strategy
    if arguments.subtitle_style:
        render.subtitle_style = arguments.subtitle_style
    if arguments.disable_subtitles:
        render.subtitles_enabled = False
    if arguments.video_encoder:
        render.encoder = arguments.video_encoder
    config.validate()


def _apply_draft_command_arguments(config) -> None:
    """A draft always needs its FinalScript and ProductionPlan, never delivery services."""

    config.transformation.enabled = True
    config.production.enabled = True
    config.validate()


def _parse_boundary_overrides(values: list[str]) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    for raw in values:
        try:
            candidate_id, start, end = raw.rsplit(":", 2)
            start_value, end_value = float(start), float(end)
        except (AttributeError, TypeError, ValueError) as error:
            raise ClipEngineError("--candidate-boundary must be candidate_id:start:end.") from error
        if not candidate_id.strip() or end_value <= start_value:
            raise ClipEngineError("--candidate-boundary has an invalid range.")
        parsed[candidate_id] = {"start": start_value, "end": end_value}
    return parsed
