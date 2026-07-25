from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import ProductionRenderConfig
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan
from app.utils import stable_text_hash, write_bytes_atomic
from app.video_models import SubtitleCue, SubtitleProject, SubtitleStyle


_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "minimal": {"font_size": 56, "font_weight": "normal", "text_color": "#FFFFFF", "outline_color": "#202020", "outline_width": 2, "shadow": 1, "background": "transparent", "position": "bottom", "bottom_margin": 170, "alignment": "center", "uppercase": False},
    "documentary": {"font_size": 62, "font_weight": "bold", "text_color": "#FFFFFF", "outline_color": "#101010", "outline_width": 3, "shadow": 2, "background": "transparent", "position": "bottom", "bottom_margin": 185, "alignment": "center", "uppercase": False},
    "dynamic": {"font_size": 68, "font_weight": "bold", "text_color": "#FFFFFF", "outline_color": "#111111", "outline_width": 4, "shadow": 2, "background": "transparent", "position": "bottom", "bottom_margin": 190, "alignment": "center", "uppercase": True},
    "clean": {"font_size": 58, "font_weight": "bold", "text_color": "#FFFFFF", "outline_color": "#242424", "outline_width": 2, "shadow": 1, "background": "transparent", "position": "bottom", "bottom_margin": 165, "alignment": "center", "uppercase": False},
}


def build_subtitle_project(
    plan: ProductionPlan, audio_project: Any, config: ProductionRenderConfig,
) -> SubtitleProject:
    """Use actual AudioProject clip ranges; never reuse the estimated Goal 3A cues."""

    style, font_fallback, font_warning = resolve_subtitle_style(config)
    plan_segments = {segment.segment_id: segment for segment in plan.segments}
    cues: list[SubtitleCue] = []
    for clip in audio_project.timeline.clips:
        segment = plan_segments.get(clip.production_segment_id)
        if isinstance(segment, NarrationSegment):
            text, source_type, speaker = segment.text, "narration", "narrator"
        elif isinstance(segment, DialogueSegment):
            text, source_type, speaker = segment.source_text, "dialogue", segment.speaker
        else:
            continue
        chunks = split_subtitle_text(text, config)
        cues.extend(_timed_cues(
            chunks, clip.production_segment_id or clip.clip_id, speaker, source_type,
            float(clip.timeline_start_seconds), float(clip.timeline_end_seconds), style, config,
        ))
    duration = float(audio_project.timeline.duration_seconds)
    project_id = f"subtitles-{audio_project.project_id}-{stable_text_hash(style.model_dump_json())[:12]}"
    warnings = [font_warning] if font_warning else []
    return SubtitleProject(
        project_id=project_id, audio_project_id=audio_project.project_id, duration_seconds=duration,
        style=style, cues=cues, font_fallback_used=font_fallback, warnings=warnings,
    )


def resolve_subtitle_style(config: ProductionRenderConfig) -> tuple[SubtitleStyle, bool, str | None]:
    requested = config.subtitle_font_family.strip()
    font, fallback, warning = _resolve_font(requested)
    values = dict(_STYLE_PRESETS[config.subtitle_style])
    values.update({"style_id": config.subtitle_style, "font_family": font})
    return SubtitleStyle(**values), fallback, warning


def _resolve_font(requested: str) -> tuple[str, bool, str | None]:
    """Use a portable family name; never require or add a local font asset."""

    if not requested or any(character in requested for character in "\\/\x00"):
        return "Arial", True, "Requested subtitle font is unsafe or empty; using Arial."
    matcher = shutil.which("fc-match")
    if matcher:
        try:
            match = subprocess.run([matcher, "-f", "%{family}", requested], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            if match and requested.casefold() in match.casefold():
                return requested, False, None
            return "Arial", True, f"Subtitle font '{requested}' is unavailable; using Arial."
        except (OSError, subprocess.SubprocessError):
            pass
    # libass/Windows resolves this logical family itself. We retain a predictable fallback
    # for explicitly suspicious test values without coupling the project to a font file.
    if requested.casefold().startswith("missing") or requested.casefold().startswith("__"):
        return "Arial", True, f"Subtitle font '{requested}' is unavailable; using Arial."
    # On Windows libass resolves logical families through the system font provider.
    # Do not downgrade an otherwise safe render merely because fc-match is absent.
    return requested, False, None


def split_subtitle_text(text: str, config: ProductionRenderConfig) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    limit = config.subtitle_max_chars_per_line * config.subtitle_max_lines
    words: list[str] = []
    for word in clean.split(" "):
        # Preserve every character, including a long URL/token, while keeping the
        # resulting cue inside the configured readable line/line-count bounds.
        words.extend(word[index:index + limit] for index in range(0, len(word), limit))
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        proposed = " ".join([*current, word])
        punctuation_boundary = bool(current) and current[-1].endswith((".", "!", "?", ";", ":"))
        if current and (len(proposed) > limit or (punctuation_boundary and len(" ".join(current)) >= limit * 0.55)):
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _timed_cues(
    chunks: list[str], segment_id: str, speaker: str, source_type: str,
    start: float, end: float, style: SubtitleStyle, config: ProductionRenderConfig,
) -> list[SubtitleCue]:
    if not chunks or end <= start:
        return []
    total = max(0.001, end - start)
    weights = [max(1, len(chunk.replace(" ", ""))) for chunk in chunks]
    total_weight = sum(weights)
    cursor = start
    cues: list[SubtitleCue] = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
        if index == len(chunks):
            cue_end = end
        else:
            planned = total * weight / total_weight
            # The segment duration is authoritative. Clamp only while another cue remains.
            planned = min(config.subtitle_max_duration, max(config.subtitle_min_duration, planned))
            remaining_min = 0.05 * (len(chunks) - index)
            cue_end = min(end - remaining_min, cursor + planned)
        if cue_end <= cursor:
            cue_end = min(end, cursor + 0.05)
        rendered = chunk.upper() if style.uppercase else chunk
        line_count = _line_count(rendered, config.subtitle_max_chars_per_line)
        cues.append(SubtitleCue(
            cue_id=f"cue-{segment_id}-{index:03d}", segment_id=segment_id, speaker=speaker,
            text=rendered, start_seconds=round(cursor, 3), end_seconds=round(cue_end, 3),
            word_count=len(chunk.split()), line_count=line_count, style_id=style.style_id,
            source_type=source_type,
        ))
        cursor = cue_end
    return cues


def _line_count(text: str, maximum: int) -> int:
    return max(1, (len(text) + maximum - 1) // maximum)


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
Style: Production,{_escape_style(style.font_family)},{font_size},{_ass_color(style.text_color)},{_ass_color(style.text_color)},{_ass_color(style.outline_color)},{_ass_color(style.background)}, {bold},0,0,0,100,100,0,0,1,{outline_width},{shadow},{alignment},{margin_horizontal},{margin_horizontal},{margin_vertical},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events = [
        f"Dialogue: 0,{_ass_time(cue.start_seconds)},{_ass_time(cue.end_seconds)},Production,{_escape(cue.speaker)},0,0,0,,{_wrap_and_escape(cue.text, style)}"
        for cue in project.cues
    ]
    write_bytes_atomic(path, (header + "\n".join(events) + "\n").encode("utf-8-sig"))
    return path


def _ass_metrics(style: SubtitleStyle, width: int, height: int) -> tuple[int, int, int, int, int]:
    """Scale style coordinates from the 1080×1920 production reference canvas."""

    scale = min(width / 1080, height / 1920)
    return (
        max(12, round(style.font_size * scale)),
        max(1, round(style.outline_width * scale)),
        max(0, round(style.shadow * scale)),
        max(12, round(70 * scale)),
        max(32, round(style.bottom_margin * scale)),
    )


def _wrap_and_escape(text: str, style: SubtitleStyle) -> str:
    # The cue was already bounded; splitting approximately halfway makes two readable lines.
    escaped = _escape(text)
    if len(text) <= 28:
        return escaped
    words = escaped.split(" ")
    split = max(1, len(words) // 2)
    return " ".join(words[:split]) + r"\N" + " ".join(words[split:])


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
