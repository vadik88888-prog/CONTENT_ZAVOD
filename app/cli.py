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
    try:
        config = load_config(arguments.config)
        if arguments.tts_only and arguments.audio_only:
            raise ClipEngineError("--tts-only и --audio-only нельзя использовать вместе: это отдельные изолированные этапы.")
        if arguments.production_plan_only and arguments.audio_only:
            raise ClipEngineError("--production-plan-only и --audio-only нельзя использовать вместе.")
        _apply_transformation_arguments(config, arguments)
        _apply_tts_arguments(config, arguments)
        _apply_audio_arguments(config, arguments)
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
        ).run(
            input_path=arguments.input, url=arguments.url
        )
    except ClipEngineError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    print("Готово.")
    print(f"Клипов: {len(result.output_files)} (выбрано: {result.selected_clips})")
    print(f"Результаты: {result.output_directory}")
    print(f"Отчёт: {result.report_path}")
    report = read_json(result.report_path, {})
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
