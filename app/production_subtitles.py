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
from app.video_models import SubtitleCue, SubtitleProject, SubtitleStyle, SubtitleWordTiming


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


def build_subtitle_project(
    plan: ProductionPlan, audio_project: Any, config: ProductionRenderConfig, transcript: dict[str, Any] | None = None,
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
        ))
    duration = float(audio_project.timeline.duration_seconds)
    project_id = f"subtitles-{audio_project.project_id}-{stable_text_hash(style.model_dump_json())[:12]}"
    return SubtitleProject(
        project_id=project_id, audio_project_id=audio_project.project_id, duration_seconds=duration,
        style=style, cues=cues, font_fallback_used=font_fallback, warnings=_unique(warnings),
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
    style: SubtitleStyle, config: ProductionRenderConfig,
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
            source_type=source_type, word_timings=timings, original_text=original_text,
            original_line_count=original_line_count, resolved_lines=rendered_lines,
            resolved_font_size=group.font_size, split_reason=group.reason,
            layout_state=state, fallback_used=group.fallback,
        ))
    return cues


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
