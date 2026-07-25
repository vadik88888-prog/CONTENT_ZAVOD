from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models import Candidate


def create_ass(
    transcript: dict[str, Any],
    clip: Candidate,
    path: Path,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    words = [
        item for item in transcript.get("words", [])
        if float(item["end"]) >= clip.start and float(item["start"]) <= clip.end
    ]
    header = _header(width, height)
    events: list[str] = []
    for group in _groups(words):
        start = max(0.0, float(group[0]["start"]) - clip.start)
        end = max(start + 0.05, float(group[-1]["end"]) - clip.start)
        text = _two_line_text([str(item["text"]).strip() for item in group])
        if text:
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{_escape(text)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return path


def _groups(words: list[dict[str, Any]], maximum_words: int = 6) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current and (
            len(current) >= maximum_words
            or float(word["end"]) - float(current[0]["start"]) > 2.6
        ):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _two_line_text(words: list[str]) -> str:
    if len(words) <= 3:
        return " ".join(words)
    split = (len(words) + 1) // 2
    return " ".join(words[:split]) + r"\N" + " ".join(words[split:])


def _header(width: int, height: int) -> str:
    font_size = max(42, round(width * 0.056))
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,70,70,270,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def _ass_time(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    hours, remaining = divmod(total_centiseconds, 360000)
    minutes, remaining = divmod(remaining, 6000)
    seconds_part, centiseconds = divmod(remaining, 100)
    return f"{hours}:{minutes:02d}:{seconds_part:02d}.{centiseconds:02d}"


def _escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
