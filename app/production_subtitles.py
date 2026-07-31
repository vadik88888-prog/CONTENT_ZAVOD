from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import ProductionRenderConfig
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.utils import stable_text_hash, write_bytes_atomic
from app.video_models import (
    SubtitleCue,
    SubtitleProject,
    SubtitleQualityDecision,
    SubtitleQualityFinding,
    SubtitleRenderedBounds,
    SubtitleStyle,
    SubtitleWordTiming,
)


_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "minimal": {"font_size": 56, "font_weight": "normal", "text_color": "#FFFFFF", "highlight_color": "#FFFFFF", "outline_color": "#202020", "outline_width": 2, "shadow": 1, "background": "transparent", "position": "bottom", "bottom_margin": 170, "alignment": "center", "uppercase": False},
    "documentary": {"font_size": 58, "font_weight": "bold", "text_color": "#FFFFFF", "highlight_color": "#FFFFFF", "outline_color": "#101010", "outline_width": 1.5, "shadow": 0, "background": "transparent", "position": "bottom", "bottom_margin": 220, "alignment": "center", "uppercase": False},
    "dynamic": {"font_size": 68, "font_weight": "bold", "text_color": "#FFFFFF", "highlight_color": "#FFD54A", "outline_color": "#111111", "outline_width": 4, "shadow": 2, "background": "transparent", "position": "bottom", "bottom_margin": 190, "alignment": "center", "uppercase": True},
    "clean": {"font_size": 58, "font_weight": "bold", "text_color": "#FFFFFF", "highlight_color": "#FFFFFF", "outline_color": "#242424", "outline_width": 2, "shadow": 1, "background": "transparent", "position": "bottom", "bottom_margin": 165, "alignment": "center", "uppercase": False},
}

_NO_BREAK_AFTER = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with",
    "в", "во", "и", "к", "ко", "на", "о", "об", "от", "по", "с", "со", "у", "за", "из", "не",
})
_PUNCTUATION = frozenset(".!?;:,")
_METRICS_APPLICATION: Any | None = None
_PLATFORM_SAFE_INSETS: dict[str, tuple[float, float, float, float]] = {
    # left, top, right, bottom. The bottom inset deliberately stays below the
    # established documentary margin so current safe layouts remain valid.
    "universal": (0.03, 0.03, 0.03, 0.03),
    "tiktok": (0.06, 0.05, 0.12, 0.08),
    "reels": (0.06, 0.05, 0.08, 0.08),
    "shorts": (0.05, 0.05, 0.05, 0.07),
}


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _ResolvedGroup:
    tokens: tuple[_Token, ...]
    lines: tuple[tuple[_Token, ...], ...]
    font_size: int
    fallback: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SubtitleLanguageProfile:
    language: str
    target_cps: float
    maximum_cps: float
    max_chars_per_line: int
    max_lines: int
    min_timing_confidence: float
    long_word_characters: int


_LANGUAGE_PROFILES: dict[str, SubtitleLanguageProfile] = {
    "ru": SubtitleLanguageProfile(
        language="ru", target_cps=17.0, maximum_cps=20.0, max_chars_per_line=28,
        max_lines=2, min_timing_confidence=0.70, long_word_characters=20,
    ),
    "en": SubtitleLanguageProfile(
        language="en", target_cps=19.0, maximum_cps=22.0, max_chars_per_line=32,
        max_lines=2, min_timing_confidence=0.70, long_word_characters=24,
    ),
}


def build_subtitle_project(
    plan: ProductionPlan, audio_project: Any, config: ProductionRenderConfig, transcript: dict[str, Any] | None = None,
    *, composition_segments: list[Any] | None = None, platform: str = "universal",
) -> SubtitleProject:
    """Build a final, fitted subtitle plan from actual AudioProject timing.

    The resolved lines are the same lines emitted to ASS, so the quality
    validator never evaluates the pre-layout transcript representation.
    """

    style, font_fallback, font_warning = resolve_subtitle_style(config)
    plan_segments = {segment.segment_id: segment for segment in plan.segments}
    transcript_words = _transcript_words(transcript)
    cues: list[SubtitleCue] = []
    warnings = [font_warning] if font_warning else []
    for clip in audio_project.timeline.clips:
        segment = plan_segments.get(clip.production_segment_id)
        if isinstance(segment, NarrationSegment):
            text, source_type, speaker, words = segment.text, "narration", "narrator", []
        elif isinstance(segment, DialogueSegment):
            words = _dialogue_word_timings(segment, clip, transcript_words)
            text = " ".join(item.text for item in words) if words else segment.source_text
            source_type, speaker = "dialogue", segment.speaker
        else:
            continue
        tokens = _tokens(text, words, float(clip.timeline_start_seconds), float(clip.timeline_end_seconds))
        resolved = _fit_tokens(tokens, style, config)
        if any(item.fallback for item in resolved):
            warnings.append(
                f"Subtitle fallback fit used for {clip.production_segment_id or clip.clip_id}; text was preserved."
            )
        cues.extend(_resolved_cues(
            resolved, text, clip.production_segment_id or clip.clip_id, speaker, source_type, style, config,
            timing_source="source_word_timings" if isinstance(segment, DialogueSegment) and words else "estimated",
        ))
    duration = float(audio_project.timeline.duration_seconds)
    project_id = f"subtitles-{audio_project.project_id}-{stable_text_hash(style.model_dump_json())[:12]}"
    decision = assess_subtitle_quality(
        cues, style, config, composition_segments=composition_segments, platform=platform,
        provenance={
            "stage": "pre_render_ass_layout",
            "fitter": "production_subtitles._fit_tokens",
            "renderer": "production_subtitles.write_production_ass",
            "quality_config_version": config.subtitle_quality_version,
            "platform": platform,
        },
    )
    if decision.status == "blocked":
        warnings.append(
            "Subtitle Quality V2 blocked the resolved ASS layout: " + ", ".join(decision.reason_codes)
        )
    return SubtitleProject(
        project_id=project_id, audio_project_id=audio_project.project_id, duration_seconds=duration,
        style=style, cues=cues, font_fallback_used=font_fallback, warnings=_unique(warnings),
        quality_decision=decision,
    )


def resolve_subtitle_style(config: ProductionRenderConfig) -> tuple[SubtitleStyle, bool, str | None]:
    requested = config.subtitle_font_family.strip()
    font, fallback, warning = _resolve_font(requested)
    values = dict(_STYLE_PRESETS[config.subtitle_style])
    values.update({
        "style_id": config.subtitle_style,
        "font_family": font,
        "max_chars_per_line": config.subtitle_max_chars_per_line,
        "max_lines": config.subtitle_max_lines,
    })
    return SubtitleStyle(**values), fallback, warning


def _resolve_font(requested: str) -> tuple[str, bool, str | None]:
    if not requested or any(character in requested for character in "\\/\x00"):
        return "Arial", True, "Requested subtitle font is unsafe or empty; using Arial."
    matcher = shutil.which("fc-match")
    if matcher:
        try:
            match = subprocess.run(
                [matcher, "-f", "%{family}", requested], capture_output=True, timeout=10, check=True,
                **UTF8_REPLACE_TEXT,
            ).stdout.strip()
            if match and requested.casefold() in match.casefold():
                return requested, False, None
            return "Arial", True, f"Subtitle font '{requested}' is unavailable; using Arial."
        except (OSError, subprocess.SubprocessError):
            pass
    if requested.casefold().startswith("missing") or requested.casefold().startswith("__"):
        return "Arial", True, f"Subtitle font '{requested}' is unavailable; using Arial."
    return requested, False, None


def split_subtitle_text(text: str, config: ProductionRenderConfig) -> list[str]:
    """Compatibility helper returning deterministic semantic/text chunks."""

    clean = " ".join(text.split())
    if not clean:
        return []
    limit_chars = config.subtitle_max_chars_per_line * config.subtitle_max_lines
    words = [
        piece
        for word in clean.split()
        for piece in (word[index:index + limit_chars] for index in range(0, len(word), limit_chars))
    ]
    chunks: list[list[str]] = []
    current: list[str] = []
    limit = max(config.subtitle_min_words_per_cue, config.subtitle_max_words_per_cue)
    for word in words:
        current.append(word)
        if (
            len(current) >= limit
            or len(" ".join(current)) > limit_chars
            or (word.endswith(tuple(_PUNCTUATION)) and len(current) >= config.subtitle_min_words_per_cue)
        ):
            chunks.append(current); current = []
    if current:
        chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


def _tokens(text: str, timings: list[SubtitleWordTiming], start: float, end: float) -> list[_Token]:
    if timings:
        return [_Token(item.text, item.start_seconds, item.end_seconds) for item in timings]
    words = text.split()
    if not words or end <= start:
        return []
    duration = (end - start) / len(words)
    return [_Token(word, start + index * duration, start + (index + 1) * duration) for index, word in enumerate(words)]


def _fit_tokens(tokens: list[_Token], style: SubtitleStyle, config: ProductionRenderConfig) -> list[_ResolvedGroup]:
    if not tokens:
        return []
    groups = _semantic_groups(tokens, config)
    base_size = _ass_font_size(style, config.output_width, config.output_height)
    result: list[_ResolvedGroup] = []
    for group, reason in groups:
        result.extend(_fit_group(tuple(group), reason, style, config, base_size))
    return result


def _semantic_groups(tokens: list[_Token], config: ProductionRenderConfig) -> list[tuple[list[_Token], str]]:
    result: list[tuple[list[_Token], str]] = []
    cursor = 0
    while cursor < len(tokens):
        maximum = min(len(tokens), cursor + config.subtitle_max_words_per_cue)
        duration_limit = tokens[cursor].start + config.subtitle_max_duration
        candidates = [index for index in range(cursor + config.subtitle_min_words_per_cue, maximum + 1)]
        within_duration = [index for index in candidates if tokens[index - 1].end <= duration_limit]
        if within_duration:
            candidates = within_duration
        # Prefer a readable minimum cue duration whenever the source timings
        # offer one.  If even the longest permitted group is shorter, retain
        # the words rather than inventing an artificial pause or moving their
        # timestamps beyond the speech.
        readable = [
            index for index in candidates
            if tokens[index - 1].end - tokens[cursor].start >= config.subtitle_min_duration
        ]
        if readable:
            candidates = readable
        if not candidates:
            candidates = [max(cursor + 1, maximum)]
        end = max(candidates, key=lambda index: _boundary_score(tokens, index, cursor, len(tokens)))
        reason = "punctuation" if tokens[end - 1].text.rstrip().endswith(tuple(_PUNCTUATION)) else "semantic_group"
        result.append((tokens[cursor:end], reason))
        cursor = end
    return result


def _boundary_score(tokens: list[_Token], index: int, start: int, total: int) -> float:
    if index >= total:
        return 1000.0
    previous, following = tokens[index - 1], tokens[index]
    score = 0.0
    if previous.text.rstrip().endswith((".", "!", "?")):
        score += 80.0
    elif previous.text.rstrip().endswith((",", ";", ":", "—", "-")):
        score += 45.0
    gap = max(0.0, following.start - previous.end)
    score += min(gap, 1.0) * 30.0
    score -= abs((index - start) - 5) * 0.5
    return score


def _fit_group(
    tokens: tuple[_Token, ...], reason: str, style: SubtitleStyle, config: ProductionRenderConfig, base_size: int,
) -> list[_ResolvedGroup]:
    if not tokens:
        return []
    widths = _font_sizes(base_size, config.subtitle_min_font_scale)
    for font_size in widths:
        lines = _wrapped_token_lines(tokens, style, config, font_size)
        if len(lines) <= style.max_lines and _lines_fit(lines, style, config, font_size):
            return [_ResolvedGroup(tokens, tuple(tuple(line) for line in lines), font_size, font_size != base_size, reason)]
    if len(tokens) > 1:
        split = _best_fit_split(tokens)
        if split:
            return [
                *_fit_group(tokens[:split], "width_fit", style, config, base_size),
                *_fit_group(tokens[split:], "width_fit", style, config, base_size),
            ]
    # A single unbreakable word cannot be silently removed. Keep it with the
    # approved minimum font size and mark the explicit safe fallback.
    font_size = widths[-1]
    return [_ResolvedGroup(tokens, (tokens,), font_size, True, "safe_fallback")]


def _best_fit_split(tokens: tuple[_Token, ...]) -> int | None:
    if len(tokens) < 2:
        return None
    center = len(tokens) / 2
    choices = range(1, len(tokens))
    return max(choices, key=lambda index: _boundary_score(list(tokens), index, 0, len(tokens)) - abs(index - center))


def _font_sizes(base: int, minimum_scale: float) -> list[int]:
    minimum = max(8, round(base * minimum_scale))
    values = [base]
    for scale in (0.95, 0.90, minimum_scale):
        value = max(minimum, round(base * scale))
        if value not in values:
            values.append(value)
    return values


def _wrapped_token_lines(
    tokens: tuple[_Token, ...], style: SubtitleStyle, config: ProductionRenderConfig, font_size: int,
) -> list[list[_Token]]:
    maximum_width = _maximum_width(style, config)
    lines: list[list[_Token]] = []
    current: list[_Token] = []
    for token in tokens:
        proposed = [*current, token]
        text = _line_text(proposed, style)
        too_wide = _measure_width(text, style, font_size) > maximum_width
        too_many_chars = len(text) > style.max_chars_per_line * 1.35
        if current and (too_wide or too_many_chars):
            if len(current) > 1 and _normalise_word(current[-1].text) in _NO_BREAK_AFTER:
                carried = current.pop()
                lines.append(current)
                current = [carried, token]
            else:
                lines.append(current)
                current = [token]
        else:
            current.append(token)
    if current:
        lines.append(current)
    if len(lines) == 2 and len(lines[1]) == 1 and _is_short_word(lines[1][0].text) and len(lines[0]) > 1:
        moved = lines[0][-1]
        replacement = [moved, *lines[1]]
        if _measure_width(_line_text(replacement, style), style, font_size) <= maximum_width:
            lines[0] = lines[0][:-1]
            lines[1] = replacement
    return lines


def _lines_fit(lines: list[list[_Token]], style: SubtitleStyle, config: ProductionRenderConfig, font_size: int) -> bool:
    maximum_width = _maximum_width(style, config)
    return all(_measure_width(_line_text(line, style), style, font_size) <= maximum_width for line in lines)


def _maximum_width(style: SubtitleStyle, config: ProductionRenderConfig) -> float:
    return max(24.0, config.output_width * config.subtitle_max_rendered_width_ratio - (style.outline_width * 4))


def _line_text(tokens: list[_Token], style: SubtitleStyle) -> str:
    value = " ".join(item.text for item in tokens)
    return value.upper() if style.uppercase else value


def _measure_width(text: str, style: SubtitleStyle, font_size: int) -> float:
    """Prefer Qt's real font metrics; retain a deterministic CLI fallback."""

    global _METRICS_APPLICATION
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtGui import QFont, QFontMetricsF, QGuiApplication

        application = QGuiApplication.instance()
        if application is None:
            if QCoreApplication.instance() is not None:
                raise RuntimeError("A non-GUI Qt application is already active.")
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            _METRICS_APPLICATION = QGuiApplication([])
            application = _METRICS_APPLICATION
        if not isinstance(application, QGuiApplication):
            raise RuntimeError("Qt GUI font metrics are unavailable in this process.")
        font = QFont(style.font_family)
        font.setPixelSize(font_size)
        font.setBold(style.font_weight == "bold")
        return float(QFontMetricsF(font).horizontalAdvance(text)) + style.outline_width * 2
    except Exception:
        # This path is for environments where a GUI application cannot exist;
        # it is deliberately conservative, so it splits earlier rather than
        # allowing a possible overflow.
        return sum((1.05 if character.isupper() else 0.92 if ord(character) > 127 else 0.78) * font_size for character in text) + style.outline_width * 2


def _resolved_cues(
    groups: list[_ResolvedGroup], original_text: str, segment_id: str, speaker: str, source_type: str,
    style: SubtitleStyle, config: ProductionRenderConfig, timing_source: str = "estimated",
) -> list[SubtitleCue]:
    base_size = _ass_font_size(style, config.output_width, config.output_height)
    original_tokens = _tokens(original_text, [], 0.0, max(1.0, len(original_text.split()) / 2.0))
    original_line_count = max(1, len(_wrapped_token_lines(tuple(original_tokens), style, config, base_size)))
    cues: list[SubtitleCue] = []
    for index, group in enumerate(groups, start=1):
        start, end = group.tokens[0].start, group.tokens[-1].end
        lines = [" ".join(token.text for token in line) for line in group.lines]
        text = " ".join(token.text for token in group.tokens)
        rendered = text.upper() if style.uppercase else text
        rendered_lines = [line.upper() if style.uppercase else line for line in lines]
        timings = [SubtitleWordTiming(text=item.text, start_seconds=round(item.start, 3), end_seconds=round(item.end, 3)) for item in group.tokens]
        state = "fallback_fitted" if group.fallback else "fitted"
        cues.append(SubtitleCue(
            cue_id=f"cue-{segment_id}-{index:03d}", segment_id=segment_id, speaker=speaker,
            text=rendered, start_seconds=round(start, 3), end_seconds=round(end, 3),
            word_count=len(group.tokens), line_count=len(rendered_lines), style_id=style.style_id,
            source_type=source_type, word_timings=timings, timing_source=timing_source, original_text=original_text,
            original_line_count=original_line_count, resolved_lines=rendered_lines,
            resolved_font_size=group.font_size, split_reason=group.reason,
            layout_state=state, fallback_used=group.fallback,
        ))
    return cues


def assess_subtitle_quality(
    cues: list[SubtitleCue], style: SubtitleStyle, config: ProductionRenderConfig,
    *, composition_segments: list[Any] | None = None, platform: str = "universal",
    provenance: dict[str, str] | None = None,
) -> SubtitleQualityDecision:
    """Assess the fitted ASS payload without creating a parallel subtitle writer.

    Bounds are calculated from the same style metrics and resolved lines that
    ``write_production_ass`` emits. This gives the renderer a deterministic
    layout blocker before an expensive media render starts.
    """

    language = _resolve_subtitle_language(cues, config)
    profile = _LANGUAGE_PROFILES.get(language)
    line_limit = min(style.max_chars_per_line, profile.max_chars_per_line) if profile else style.max_chars_per_line
    max_lines = min(style.max_lines, profile.max_lines) if profile else style.max_lines
    maximum_cps = profile.maximum_cps if profile else 20.0
    target_cps = profile.target_cps if profile else min(maximum_cps, config.subtitle_reading_speed_cps)
    min_timing_confidence = profile.min_timing_confidence if profile else 0.70
    long_word_limit = profile.long_word_characters if profile else 20
    findings: list[SubtitleQualityFinding] = []
    bounds: list[SubtitleRenderedBounds] = []
    cps_values: list[float] = []
    timing_confidences: list[float] = []
    max_line_length = 0
    composition_segments = list(composition_segments or [])

    def add(
        reason_code: str, severity: str, message: str, *, cue_id: str | None = None,
        metrics: dict[str, float | int | bool] | None = None,
    ) -> None:
        findings.append(SubtitleQualityFinding(
            reason_code=reason_code, severity=severity, cue_id=cue_id,
            metrics=metrics or {}, message=message,
        ))

    if profile is None:
        add(
            "LANGUAGE_PROFILE_UNKNOWN", "warning",
            "Subtitle language could not be matched to a supported RU or EN profile.",
        )

    for cue in cues:
        duration = max(0.001, cue.end_seconds - cue.start_seconds)
        rendered_lines = list(cue.resolved_lines or [cue.text])
        cps = _subtitle_cps(cue.text, duration)
        cps_values.append(cps)
        timing_confidence = _timing_confidence(cue)
        timing_confidences.append(timing_confidence)
        if cps > target_cps:
            add(
                "CPS_TOO_HIGH", "blocker" if cps > maximum_cps and timing_confidence >= min_timing_confidence else "warning",
                f"Cue reading speed is {cps:.1f} CPS; profile target is {target_cps:.1f} and maximum is {maximum_cps:.1f}.",
                cue_id=cue.cue_id, metrics={"cps": round(cps, 3), "target_cps": target_cps, "maximum_cps": maximum_cps},
            )
        if len(rendered_lines) > max_lines or cue.line_count > max_lines:
            add(
                "TOO_MANY_LINES", "blocker",
                f"Cue resolves to {len(rendered_lines)} lines; the {language} profile permits {max_lines}.",
                cue_id=cue.cue_id, metrics={"line_count": len(rendered_lines), "maximum_lines": max_lines},
            )
        for line in rendered_lines:
            max_line_length = max(max_line_length, len(line))
            if len(line) > line_limit:
                width_excess = _measure_width(line, style, cue.resolved_font_size or _ass_font_size(style, config.output_width, config.output_height)) > _maximum_width(style, config)
                add(
                    "LINE_TOO_LONG", "blocker" if width_excess or len(line) > line_limit * 1.30 else "warning",
                    f"Resolved line contains {len(line)} characters; the {language} profile limit is {line_limit}.",
                    cue_id=cue.cue_id,
                    metrics={"line_length": len(line), "maximum_line_length": line_limit, "rendered_width_excess": width_excess},
                )
            for word in line.split():
                if len(_normalise_word(word)) > long_word_limit:
                    add(
                        "WORD_SPLIT_RISK", "warning",
                        f"Unbreakable word '{word[:80]}' exceeds the {language} profile's safe word length.",
                        cue_id=cue.cue_id,
                        metrics={"word_length": len(_normalise_word(word)), "safe_word_length": long_word_limit},
                    )
        if _has_weak_line_break(rendered_lines) or cue.split_reason == "width_fit":
            add(
                "WEAK_SEMANTIC_BREAK", "warning",
                "Resolved layout required a width-driven or syntactically weak semantic break.",
                cue_id=cue.cue_id,
            )
        if cue.fallback_used or cue.layout_state == "fallback_fitted":
            add(
                "FALLBACK_FITTING_USED", "warning",
                "Safe fallback fitting was used; the subtitle layout is not a clean pass.",
                cue_id=cue.cue_id,
            )
        if timing_confidence < min_timing_confidence:
            add(
                "TIMING_CONFIDENCE_LOW", "warning",
                f"Cue timing confidence is {timing_confidence:.2f}; profile minimum is {min_timing_confidence:.2f}.",
                cue_id=cue.cue_id,
                metrics={"timing_confidence": timing_confidence, "minimum_timing_confidence": min_timing_confidence},
            )
        rendered_bounds = _rendered_bounds(cue, style, config)
        bounds.append(rendered_bounds)
        if _outside_canvas(rendered_bounds, config.output_width, config.output_height):
            add(
                "SUBTITLE_OUT_OF_FRAME", "blocker",
                "Resolved ASS bounds extend outside the output frame.", cue_id=cue.cue_id,
                metrics=_bounds_metrics(rendered_bounds, config.output_width, config.output_height),
            )
        if _violates_platform_safe_zone(rendered_bounds, config.output_width, config.output_height, platform):
            add(
                "PLATFORM_SAFE_ZONE_VIOLATION", "blocker",
                f"Resolved ASS bounds enter the {platform} platform exclusion zone.", cue_id=cue.cue_id,
                metrics=_bounds_metrics(rendered_bounds, config.output_width, config.output_height),
            )
        for target_kind, target_rect, target_id in _overlapping_targets(cue, composition_segments, config):
            overlap = _overlap_ratio(rendered_bounds, target_rect)
            if overlap <= 0:
                continue
            if target_kind == "face":
                add(
                    "SUBTITLE_OVERLAPS_FACE", "blocker",
                    "Resolved subtitle bounds overlap an observed face.", cue_id=cue.cue_id,
                    metrics={"overlap_ratio": overlap, "target_id": target_id},
                )
            else:
                add(
                    "SUBTITLE_OVERLAPS_TARGET", "blocker",
                    "Resolved subtitle bounds overlap an observed composition target.", cue_id=cue.cue_id,
                    metrics={"overlap_ratio": overlap, "target_id": target_id},
                )

    for left, right in zip(cues, cues[1:]):
        if left.segment_id == right.segment_id and _weak_cue_boundary(left, right):
            add(
                "WEAK_SEMANTIC_BREAK", "warning",
                "Adjacent fitted cues split a continuous phrase without a strong semantic boundary.",
                cue_id=right.cue_id,
            )

    reason_codes = list(dict.fromkeys(item.reason_code for item in findings))
    severity = "blocker" if any(item.severity == "blocker" for item in findings) else "warning" if findings else "none"
    status = "blocked" if severity == "blocker" else "passed_with_warning" if severity == "warning" else "passed"
    return SubtitleQualityDecision(
        status=status,
        language_profile=language,
        severity=severity,
        metrics={
            "cue_count": len(cues),
            "max_cps": round(max(cps_values, default=0.0), 3),
            "mean_cps": round(sum(cps_values) / len(cps_values), 3) if cps_values else 0.0,
            "max_line_length": max_line_length,
            "max_line_count": max((cue.line_count for cue in cues), default=0),
            "fallback_cue_count": sum(cue.fallback_used for cue in cues),
            "minimum_timing_confidence": round(min(timing_confidences, default=1.0), 3),
            "mean_timing_confidence": round(sum(timing_confidences) / len(timing_confidences), 3) if timing_confidences else 1.0,
            "rendered_bounds_count": len(bounds),
            "face_overlap_count": sum(item.reason_code == "SUBTITLE_OVERLAPS_FACE" for item in findings),
            "target_overlap_count": sum(item.reason_code == "SUBTITLE_OVERLAPS_TARGET" for item in findings),
            "platform_safe_zone_violation_count": sum(item.reason_code == "PLATFORM_SAFE_ZONE_VIOLATION" for item in findings),
            "out_of_frame_count": sum(item.reason_code == "SUBTITLE_OUT_OF_FRAME" for item in findings),
            "language_profile_recognized": profile is not None,
        },
        reason_codes=reason_codes,
        findings=findings,
        rendered_bounds=bounds,
        provenance={
            "stage": "pre_render_ass_layout",
            "fitter": "production_subtitles._fit_tokens",
            "renderer": "production_subtitles.write_production_ass",
            "geometry": "ass_style_metrics_v1",
            "language_detection": "subtitle_language_override_or_unicode_script",
            "quality_config_version": config.subtitle_quality_version,
            "platform": platform,
            **(provenance or {}),
        },
    )


def _resolve_subtitle_language(cues: list[SubtitleCue], config: ProductionRenderConfig) -> str:
    if config.subtitle_language in _LANGUAGE_PROFILES:
        return config.subtitle_language
    text = " ".join(cue.text for cue in cues)
    cyrillic = sum("\u0400" <= character <= "\u052f" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    if cyrillic and cyrillic >= latin:
        return "ru"
    if latin:
        return "en"
    return "unknown"


def _subtitle_cps(text: str, duration: float) -> float:
    return len("".join(text.split())) / max(duration, 0.001)


def _timing_confidence(cue: SubtitleCue) -> float:
    timings = list(cue.word_timings)
    if not timings:
        return 0.45
    coverage = min(1.0, len(timings) / max(1, cue.word_count))
    within_cue = all(
        timing.start_seconds >= cue.start_seconds - 0.02 and timing.end_seconds <= cue.end_seconds + 0.02
        for timing in timings
    )
    ordered = within_cue and all(
        left.end_seconds <= right.start_seconds + 0.02
        for left, right in zip(timings, timings[1:])
    )
    if cue.timing_source == "source_word_timings":
        return round(min(1.0, 0.50 + 0.25 * coverage + (0.25 if ordered else 0.0)), 3)
    # Estimated narration timings retain useful cue-level placement evidence,
    # but must not turn a CPS warning into a false hard block.
    return round(min(0.68, 0.35 + 0.20 * coverage + (0.13 if ordered else 0.0)), 3)


def _has_weak_line_break(lines: list[str]) -> bool:
    for left, right in zip(lines, lines[1:]):
        left_words, right_words = left.split(), right.split()
        if not left_words or not right_words:
            continue
        if _normalise_word(left_words[-1]) in _NO_BREAK_AFTER:
            return True
        if right_words[0][:1] in _PUNCTUATION:
            return True
    return False


def _weak_cue_boundary(left: SubtitleCue, right: SubtitleCue) -> bool:
    left_words, right_words = left.text.split(), right.text.split()
    if not left_words or not right_words:
        return False
    gap = max(0.0, right.start_seconds - left.end_seconds)
    return (
        gap < 0.12
        and not left_words[-1].rstrip().endswith(tuple(_PUNCTUATION))
        and _normalise_word(left_words[-1]) not in _NO_BREAK_AFTER
    )


def _rendered_bounds(cue: SubtitleCue, style: SubtitleStyle, config: ProductionRenderConfig) -> SubtitleRenderedBounds:
    font_size, outline, shadow, margin_horizontal, margin_vertical = _ass_metrics(
        style, config.output_width, config.output_height,
    )
    resolved_font_size = cue.resolved_font_size or font_size
    lines = list(cue.resolved_lines or [cue.text])
    width = max((_measure_width(line, style, resolved_font_size) for line in lines), default=float(resolved_font_size)) + 2 * (outline + shadow)
    line_height = max(float(resolved_font_size) * 1.22, float(resolved_font_size + 2 * outline + shadow))
    height = line_height * len(lines) + 2 * (outline + shadow)
    if style.alignment == "left":
        x = float(margin_horizontal)
    else:
        x = (config.output_width - width) / 2
    y = float(margin_vertical) if style.position == "top" else config.output_height - margin_vertical - height
    return SubtitleRenderedBounds(
        cue_id=cue.cue_id, x=round(x, 3), y=round(y, 3), width=round(width, 3), height=round(height, 3),
        start_seconds=cue.start_seconds, end_seconds=cue.end_seconds,
    )


def _outside_canvas(bounds: SubtitleRenderedBounds, width: int, height: int) -> bool:
    return bounds.x < 0 or bounds.y < 0 or bounds.x + bounds.width > width or bounds.y + bounds.height > height


def _violates_platform_safe_zone(bounds: SubtitleRenderedBounds, width: int, height: int, platform: str) -> bool:
    left, top, right, bottom = _PLATFORM_SAFE_INSETS.get(platform, _PLATFORM_SAFE_INSETS["universal"])
    return (
        bounds.x < width * left
        or bounds.y < height * top
        or bounds.x + bounds.width > width * (1 - right)
        or bounds.y + bounds.height > height * (1 - bottom)
    )


def _bounds_metrics(bounds: SubtitleRenderedBounds, width: int, height: int) -> dict[str, float | int | bool]:
    return {
        "x": round(bounds.x, 3), "y": round(bounds.y, 3),
        "right": round(bounds.x + bounds.width, 3), "bottom": round(bounds.y + bounds.height, 3),
        "canvas_width": width, "canvas_height": height,
    }


def _overlapping_targets(
    cue: SubtitleCue, segments: list[Any], config: ProductionRenderConfig,
) -> list[tuple[str, SubtitleRenderedBounds, int]]:
    result: list[tuple[str, SubtitleRenderedBounds, int]] = []
    for segment in segments:
        if cue.end_seconds <= float(getattr(segment, "start_seconds", 0)) or cue.start_seconds >= float(getattr(segment, "end_seconds", 0)):
            continue
        for index, bound in enumerate(list(getattr(segment, "subject_bounds", []) or [])):
            if float(getattr(bound, "confidence", 0)) < 0.20:
                continue
            projected = _project_subject_bounds(segment, bound, config)
            target_id = index + 1
            target = str(getattr(bound, "target", "none"))
            if target == "primary_face":
                result.append(("face", projected, target_id))
            if target != "none":
                result.append(("target", projected, target_id))
    return result


def _project_subject_bounds(segment: Any, bound: Any, config: ProductionRenderConfig) -> SubtitleRenderedBounds:
    center_x, center_y = float(getattr(bound, "center_x", 0.5)), float(getattr(bound, "center_y", 0.5))
    width, height = float(getattr(bound, "width", 0.16)), float(getattr(bound, "height", 0.20))
    crop = getattr(segment, "target_crop", None)
    if crop is None or not getattr(crop, "crop_width", None) or not getattr(crop, "crop_height", None):
        return SubtitleRenderedBounds(
            cue_id="target", x=center_x * config.output_width - width * config.output_width / 2,
            y=center_y * config.output_height - height * config.output_height / 2,
            width=width * config.output_width, height=height * config.output_height,
            start_seconds=0, end_seconds=1,
        )
    crop_width, crop_height = float(crop.crop_width), float(crop.crop_height)
    source_width, source_height = float(crop.source_width), float(crop.source_height)
    crop_x, crop_y = _crop_origin_for_bound(segment, bound)
    x = ((center_x - width / 2) * source_width - crop_x) / crop_width * config.output_width
    y = ((center_y - height / 2) * source_height - crop_y) / crop_height * config.output_height
    return SubtitleRenderedBounds(
        cue_id="target", x=x, y=y, width=width * source_width / crop_width * config.output_width,
        height=height * source_height / crop_height * config.output_height, start_seconds=0, end_seconds=1,
    )


def _crop_origin_for_bound(segment: Any, bound: Any) -> tuple[float, float]:
    crop = segment.target_crop
    assert crop is not None and crop.crop_width and crop.crop_height
    frames = list(getattr(crop, "tracking_keyframes", []) or [])
    if frames:
        source_start = float(getattr(segment, "source_start_seconds", 0) or 0)
        local_time = max(0.0, float(getattr(bound, "time_seconds", 0)) - source_start)
        center_x, center_y = _tracking_center_at(frames, local_time)
        x = center_x * crop.source_width - crop.crop_width / 2
        y = center_y * crop.source_height - crop.crop_height / 2
    elif crop.crop_x is not None and crop.crop_y is not None:
        x, y = float(crop.crop_x), float(crop.crop_y)
    else:
        x = float(crop.normalized_x) * crop.source_width - crop.crop_width / 2
        y = float(crop.normalized_y) * crop.source_height - crop.crop_height / 2
    return (
        min(max(0.0, x), crop.source_width - crop.crop_width),
        min(max(0.0, y), crop.source_height - crop.crop_height),
    )


def _tracking_center_at(frames: list[Any], time_seconds: float) -> tuple[float, float]:
    if time_seconds <= float(frames[0].time_seconds):
        return float(frames[0].normalized_x), float(frames[0].normalized_y)
    for left, right in zip(frames, frames[1:]):
        if time_seconds <= float(right.time_seconds):
            progress = (time_seconds - float(left.time_seconds)) / max(0.001, float(right.time_seconds) - float(left.time_seconds))
            return (
                float(left.normalized_x) + (float(right.normalized_x) - float(left.normalized_x)) * progress,
                float(left.normalized_y) + (float(right.normalized_y) - float(left.normalized_y)) * progress,
            )
    return float(frames[-1].normalized_x), float(frames[-1].normalized_y)


def _overlap_ratio(left: SubtitleRenderedBounds, right: SubtitleRenderedBounds) -> float:
    overlap_width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    overlap_height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    return round((overlap_width * overlap_height) / max(1.0, left.width * left.height), 4)


def _transcript_words(transcript: dict[str, Any] | None) -> list[tuple[str, float, float]]:
    if not isinstance(transcript, dict):
        return []
    raw = transcript.get("words", [])
    if not isinstance(raw, list):
        return []
    result: list[tuple[str, float, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and end > start:
            result.append((text, start, end))
    return result


def _dialogue_word_timings(
    segment: DialogueSegment, clip: Any, transcript_words: list[tuple[str, float, float]],
) -> list[SubtitleWordTiming]:
    source_start, source_end = segment.source_start_seconds, segment.source_end_seconds
    source_duration = source_end - source_start
    timeline_start = float(clip.timeline_start_seconds)
    timeline_end = float(clip.timeline_end_seconds)
    if source_duration <= 0 or timeline_end <= timeline_start:
        return []
    scale = (timeline_end - timeline_start) / source_duration
    result: list[SubtitleWordTiming] = []
    for text, start, end in transcript_words:
        if end <= source_start or start >= source_end:
            continue
        clipped_start, clipped_end = max(start, source_start), min(end, source_end)
        local_start = timeline_start + (clipped_start - source_start) * scale
        local_end = timeline_start + (clipped_end - source_start) * scale
        if local_end > local_start:
            result.append(SubtitleWordTiming(text=text, start_seconds=round(local_start, 3), end_seconds=round(local_end, 3)))
    return result


def write_production_ass(project: SubtitleProject, path: Path, width: int, height: int) -> Path:
    style = project.style
    alignment = 2 if style.position == "bottom" and style.alignment == "center" else 1 if style.position == "bottom" else 8
    bold = -1 if style.font_weight == "bold" else 0
    font_size, outline_width, shadow, margin_horizontal, margin_vertical = _ass_metrics(style, width, height)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Production,{_escape_style(style.font_family)},{font_size},{_ass_color(style.text_color)},{_ass_color(style.highlight_color)},{_ass_color(style.outline_color)},{_ass_color(style.background)}, {bold},0,0,0,100,100,0,0,1,{outline_width},{shadow},{alignment},{margin_horizontal},{margin_horizontal},{margin_vertical},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events = [
        f"Dialogue: 0,{_ass_time(cue.start_seconds)},{_ass_time(cue.end_seconds)},Production,{_escape(cue.speaker)},0,0,0,,{_format_cue(cue, style, font_size)}"
        for cue in project.cues
    ]
    write_bytes_atomic(path, (header + "\n".join(events) + "\n").encode("utf-8-sig"))
    return path


def _ass_metrics(style: SubtitleStyle, width: int, height: int) -> tuple[int, int, int, int, int]:
    scale = min(width / 1080, height / 1920)
    return (
        _ass_font_size(style, width, height), max(1, round(style.outline_width * scale)),
        max(0, round(style.shadow * scale)), max(12, round(70 * scale)), max(32, round(style.bottom_margin * scale)),
    )


def _ass_font_size(style: SubtitleStyle, width: int, height: int) -> int:
    return max(12, round(style.font_size * min(width / 1080, height / 1920)))


def _format_cue(cue: SubtitleCue, style: SubtitleStyle, default_font_size: int) -> str:
    prefix = f"{{\\fs{cue.resolved_font_size}}}" if cue.resolved_font_size and cue.resolved_font_size != default_font_size else ""
    if style.style_id == "dynamic":
        return prefix + _format_dynamic(cue, style)
    lines = cue.resolved_lines or _wrapped_lines(cue.text, style.max_chars_per_line)
    return prefix + r"\N".join(_escape(line) for line in lines)


def _format_dynamic(cue: SubtitleCue, style: SubtitleStyle) -> str:
    words = cue.text.split()
    if not words:
        return ""
    line_counts = [len(line.split()) for line in (cue.resolved_lines or [cue.text])]
    boundaries: set[int] = set()
    cursor = 0
    for count in line_counts[:-1]:
        cursor += count
        boundaries.add(cursor)
    if len(cue.word_timings) == len(words):
        payload = [
            f"{{\\k{max(1, round((timing.end_seconds - timing.start_seconds) * 100))}}}{_escape(timing.text.upper() if style.uppercase else timing.text)}"
            for timing in cue.word_timings
        ]
    else:
        total_centiseconds = max(len(words), round((cue.end_seconds - cue.start_seconds) * 100))
        base, remainder = divmod(total_centiseconds, len(words))
        payload = [f"{{\\k{base + (1 if index < remainder else 0)}}}{_escape(word)}" for index, word in enumerate(words)]
    return " ".join(item + (r"\N" if index + 1 in boundaries else "") for index, item in enumerate(payload))


def _wrapped_lines(text: str, maximum: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[list[str]] = []
    current: list[str] = []
    for word in words:
        proposed = " ".join([*current, word])
        if current and len(proposed) > maximum:
            if len(current) > 1 and _normalise_word(current[-1]) in _NO_BREAK_AFTER:
                word = current.pop() + " " + word
            lines.append(current); current = [word]
        else:
            current.append(word)
    if current:
        lines.append(current)
    return [" ".join(line) for line in lines]


def _normalise_word(value: str) -> str:
    return value.casefold().rstrip(".,!?;:—-")


def _is_short_word(value: str) -> bool:
    return len(_normalise_word(value)) <= 3


def _escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _escape_style(value: str) -> str:
    return re.sub(r"[,{}\\\n\r]", "", value)


def _ass_color(value: str) -> str:
    if value == "transparent":
        return "&HFF000000"
    raw = value.lstrip("#")
    return f"&H00{raw[4:6]}{raw[2:4]}{raw[:2]}"


def _ass_time(seconds: float) -> str:
    total = max(0, round(seconds * 100))
    hours, remainder = divmod(total, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds_part, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds_part:02d}.{centiseconds:02d}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
