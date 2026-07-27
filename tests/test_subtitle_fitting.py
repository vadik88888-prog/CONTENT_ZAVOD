from __future__ import annotations

from types import SimpleNamespace

from app.config import AppConfig
from app.output_quality import validate_output_quality
from app.production_subtitles import (
    _Token,
    _ass_font_size,
    _fit_tokens,
    _semantic_groups,
    _resolved_cues,
    resolve_subtitle_style,
    write_production_ass,
)
from app.video_models import SubtitleProject


def _config() -> AppConfig:
    config = AppConfig()
    config.production_render.output_width = 360
    config.production_render.output_height = 640
    config.production_render.subtitle_style = "documentary"
    return config


def _tokens(words: list[str]) -> list[_Token]:
    return [_Token(word, index * 0.42, (index + 1) * 0.42) for index, word in enumerate(words)]


def _fitted_cues(words: list[str]):
    config = _config()
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)
    groups = _fit_tokens(_tokens(words), style, config.production_render)
    cues = _resolved_cues(
        groups, " ".join(words), "dialogue-006", "speaker", "dialogue", style, config.production_render,
    )
    return config, style, groups, cues


def test_long_russian_phrase_is_split_without_word_loss_or_timestamp_regression() -> None:
    words = "или чуть позже и у тебя не вышло потому что между жизнью и смертью есть один шаг".split()
    config, style, _groups, cues = _fitted_cues(words)

    assert len(cues) > 1
    assert [word.text for cue in cues for word in cue.word_timings] == words
    assert all(cue.line_count <= style.max_lines for cue in cues)
    assert all(cue.layout_state in {"fitted", "fallback_fitted"} for cue in cues)
    assert all(cue.start_seconds >= 0 and cue.end_seconds > cue.start_seconds for cue in cues)
    assert all(cue.end_seconds - cue.start_seconds <= config.production_render.subtitle_max_duration for cue in cues)
    assert all(left.end_seconds <= right.start_seconds for left, right in zip(cues, cues[1:]))


def test_russian_preposition_and_single_short_word_are_not_stranded_when_a_valid_wrap_exists() -> None:
    words = "посмотрите на парня который сделал этот важный шаг".split()
    _config_value, _style, _groups, cues = _fitted_cues(words)

    lines = [line.casefold() for cue in cues for line in cue.resolved_lines]
    assert not any(line.split()[-1] in {"в", "во", "на", "и", "у", "к", "с", "не"} for line in lines if line.split())
    assert not any(len(line.split()) == 1 and len(line) <= 3 for line in lines[1:])


def test_font_reduction_stays_in_approved_range_and_safe_fallback_preserves_text() -> None:
    config, style, groups, cues = _fitted_cues(["сверхдлинноесловобезпробелов" * 8])
    base = _ass_font_size(style, config.production_render.output_width, config.production_render.output_height)

    assert all(group.font_size >= round(base * config.production_render.subtitle_min_font_scale) for group in groups)
    assert all(group.font_size <= base for group in groups)
    assert any(group.fallback for group in groups)
    assert "".join(cue.text for cue in cues) == "сверхдлинноесловобезпробелов" * 8


def test_dynamic_ass_uses_the_same_resolved_lines_as_quality_validation(tmp_path) -> None:
    config = _config()
    config.production_render.subtitle_style = "dynamic"
    style, _fallback, _warning = resolve_subtitle_style(config.production_render)
    words = "это длинная русская фраза которая должна получить две визуальные строки".split()
    groups = _fit_tokens(_tokens(words), style, config.production_render)
    cues = _resolved_cues(groups, " ".join(words), "dialogue-006", "speaker", "dialogue", style, config.production_render)
    project = SubtitleProject(
        project_id="subtitle-test", audio_project_id="audio-test", duration_seconds=10,
        style=style, cues=cues,
    )
    ass = tmp_path / "subtitles.ass"
    write_production_ass(project, ass, 360, 640)
    rendered = ass.read_text(encoding="utf-8-sig")

    assert all(cue.line_count <= 2 for cue in cues)
    assert "\\N" in rendered
    quality = validate_output_quality(
        SimpleNamespace(timeline=SimpleNamespace(clips=[object()]), subtitle_project=project, target_duration_seconds=10),
        subtitles_enabled=True,
    )
    assert quality["status"] in {"passed", "warning"}
    invalid = project.model_copy(update={"cues": [cues[0].model_copy(update={"layout_state": "raw"})]})
    invalid_quality = validate_output_quality(
        SimpleNamespace(timeline=SimpleNamespace(clips=[object()]), subtitle_project=invalid, target_duration_seconds=10),
        subtitles_enabled=True,
    )
    assert invalid_quality["status"] == "failed"


def test_cue_dialogue_006_regression_fixture_fits_without_losing_words_or_exceeding_duration() -> None:
    words = (
        "и вот когда кажется что уже поздно потому что между жизнью и смертью "
        "остаётся всего один последний решающий шаг"
    ).split()
    config, _style, _groups, cues = _fitted_cues(words)
    project = SubtitleProject(
        project_id="subtitle-cue-dialogue-006", audio_project_id="audio-test",
        duration_seconds=max(cue.end_seconds for cue in cues), style=resolve_subtitle_style(config.production_render)[0], cues=cues,
    )

    assert all(cue.cue_id.startswith("cue-dialogue-006-") for cue in project.cues)
    assert [word.text for cue in project.cues for word in cue.word_timings] == words
    assert all(cue.line_count <= 2 for cue in project.cues)
    assert all(0 <= cue.start_seconds < cue.end_seconds <= project.duration_seconds for cue in project.cues)
    assert all(left.end_seconds <= right.start_seconds for left, right in zip(project.cues, project.cues[1:]))


def test_semantic_grouping_prefers_minimum_duration_when_source_timing_allows_it() -> None:
    config = _config()
    config.production_render.subtitle_min_duration = 0.45
    tokens = [_Token(f"слово{index}", index * 0.1, index * 0.1 + 0.09) for index in range(9)]

    groups = _semantic_groups(tokens, config.production_render)

    first, _reason = groups[0]
    assert first[-1].end - first[0].start >= config.production_render.subtitle_min_duration
