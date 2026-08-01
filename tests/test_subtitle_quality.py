from __future__ import annotations

from types import SimpleNamespace

from app.config import AppConfig
from app.output_quality import validate_output_quality
from app.production_subtitles import _retime_cues_for_readability, assess_subtitle_quality, resolve_subtitle_style
from app.video_models import SubtitleCue, SubtitleProject, SubtitleWordTiming


def _config() -> AppConfig:
    config = AppConfig()
    config.production_render.output_width = 360
    config.production_render.output_height = 640
    config.production_render.subtitle_style = "documentary"
    return config


def _cue(
    text: str, *, lines: list[str] | None = None, duration: float = 2.0,
    fallback: bool = False, timing_source: str = "source_word_timings", font_size: int = 19,
) -> SubtitleCue:
    words = text.split()
    timings = [
        SubtitleWordTiming(
            text=word, start_seconds=index * duration / len(words),
            end_seconds=(index + 1) * duration / len(words),
        )
        for index, word in enumerate(words)
    ]
    return SubtitleCue(
        cue_id="cue-quality-001", segment_id="segment-quality", speaker="speaker", text=text,
        start_seconds=0, end_seconds=duration, word_count=len(words),
        line_count=len(lines or [text]), style_id="documentary", source_type="dialogue",
        word_timings=timings, timing_source=timing_source, original_text=text,
        resolved_lines=lines or [text], resolved_font_size=font_size,
        split_reason="safe_fallback" if fallback else "punctuation",
        layout_state="fallback_fitted" if fallback else "fitted", fallback_used=fallback,
    )


def test_quality_decision_selects_versioned_ru_and_en_profiles() -> None:
    config = _config()
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)

    ru = assess_subtitle_quality([_cue("Это безопасная русская фраза.")], style, config.production_render)
    en = assess_subtitle_quality([_cue("This is a readable English phrase.")], style, config.production_render)

    assert ru.schema_version == "5E.0" and ru.language_profile == "ru"
    assert en.schema_version == "5E.0" and en.language_profile == "en"
    assert ru.metrics["rendered_bounds_count"] == 1
    assert ru.provenance["renderer"] == "production_subtitles.write_production_ass"


def test_quality_decision_records_readability_fallback_and_unknown_language_reasons() -> None:
    config = _config()
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)
    long_cyrillic = _cue(
        "сверхдлинноесловобезпробелов", lines=["сверхдлинноесловобезпробелов"], fallback=True, font_size=8,
    )
    stressed = _cue(
        "and a very long line that is deliberately far beyond the configured subtitle line limit",
        lines=["and", "this continuation is deliberately far beyond the safe subtitle line limit", "third line"], duration=0.5,
    )
    unknown = _cue("12345 !!!", duration=2.0)

    readable = assess_subtitle_quality([long_cyrillic], style, config.production_render)
    stressed_decision = assess_subtitle_quality([stressed], style, config.production_render)
    unknown_decision = assess_subtitle_quality([unknown], style, config.production_render)

    assert {"FALLBACK_FITTING_USED", "WORD_SPLIT_RISK"} <= set(readable.reason_codes)
    assert readable.status == "passed_with_warning"
    assert {"CPS_TOO_HIGH", "LINE_TOO_LONG", "TOO_MANY_LINES", "WEAK_SEMANTIC_BREAK"} <= set(stressed_decision.reason_codes)
    assert unknown_decision.language_profile == "unknown"
    assert "LANGUAGE_PROFILE_UNKNOWN" in unknown_decision.reason_codes


def test_trusted_high_cps_and_geometry_collisions_block_before_final_ready() -> None:
    config = _config()
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)
    fast = _cue("Слишком быстрая фраза для чтения", duration=0.25)
    decision = assess_subtitle_quality([fast], style, config.production_render)

    assert decision.status == "blocked"
    assert "CPS_TOO_HIGH" in decision.reason_codes

    face = SimpleNamespace(
        center_x=0.5, center_y=0.86, width=0.34, height=0.20, confidence=0.95, target="primary_face",
    )
    segment = SimpleNamespace(start_seconds=0, end_seconds=2, subject_bounds=[face], target_crop=None)
    overlap = assess_subtitle_quality([_cue("Safe words here.")], style, config.production_render, composition_segments=[segment])
    assert overlap.status == "blocked"
    assert {"SUBTITLE_OVERLAPS_FACE", "SUBTITLE_OVERLAPS_TARGET"} <= set(overlap.reason_codes)

    low_margin_style = style.model_copy(update={"bottom_margin": 0})
    platform = assess_subtitle_quality([_cue("Safe words here.")], low_margin_style, config.production_render, platform="tiktok")
    assert platform.status == "blocked"
    assert "PLATFORM_SAFE_ZONE_VIOLATION" in platform.reason_codes

    overflow = assess_subtitle_quality([_cue("X" * 80, duration=4.0)], style, config.production_render)
    assert overflow.status == "blocked"
    assert "SUBTITLE_OUT_OF_FRAME" in overflow.reason_codes

    project = SubtitleProject(
        project_id="subtitle-quality-blocked", audio_project_id="audio-quality", duration_seconds=2,
        style=style, cues=[fast], quality_decision=decision,
    )
    report = validate_output_quality(
        SimpleNamespace(timeline=SimpleNamespace(clips=[object()]), subtitle_project=project, target_duration_seconds=2),
        subtitles_enabled=True,
    )
    assert report["status"] == "failed"
    assert report["subtitle_quality"]["severity"] == "blocker"


def test_source_word_caption_retiming_uses_available_dialogue_time_without_relaxing_cps_ceiling() -> None:
    config = _config()
    config.production_render.subtitle_language = "ru"
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)
    first = _cue("fast phrase one two now", lines=["fast phrase", "one two now"], duration=0.8)
    source_second = _cue("slow cue", duration=2.2)
    second = source_second.model_copy(update={
        "cue_id": "cue-quality-002",
        "start_seconds": 0.8,
        "end_seconds": 3.0,
        "word_timings": [
            item.model_copy(update={
                "start_seconds": round(item.start_seconds + 0.8, 3),
                "end_seconds": round(item.end_seconds + 0.8, 3),
            })
            for item in source_second.word_timings
        ],
    })

    retimed = _retime_cues_for_readability(
        [first, second], clip_start=0.0, clip_end=3.0, maximum_cps=20.0,
    )
    decision = assess_subtitle_quality(retimed, style, config.production_render)

    assert retimed[0].start_seconds == 0
    assert retimed[0].end_seconds == 0.95
    assert retimed[1].start_seconds == 0.95
    assert all(left.end_seconds <= right.start_seconds for left, right in zip(retimed, retimed[1:]))
    assert retimed[1].start_seconds - second.start_seconds <= 0.75
    assert decision.status == "passed_with_warning"
    assert "CPS_TOO_HIGH" in decision.reason_codes


def test_legacy_subtitle_project_remains_readable_as_unassessed_warning() -> None:
    config = _config()
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)
    project = SubtitleProject(
        project_id="legacy-subtitles", audio_project_id="audio-quality", duration_seconds=2,
        style=style, cues=[_cue("Legacy cue.")],
    )

    report = validate_output_quality(
        SimpleNamespace(timeline=SimpleNamespace(clips=[object()]), subtitle_project=project, target_duration_seconds=2),
        subtitles_enabled=True,
    )
    assert project.quality_decision.status == "legacy_unassessed"
    assert report["status"] == "warning"
