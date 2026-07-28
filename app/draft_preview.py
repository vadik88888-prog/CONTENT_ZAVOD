from __future__ import annotations

"""Fast, intentionally lightweight assembled preview for candidate review."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import ProductionRenderError
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan
from app.sources import Source
from app.utils import write_json


@dataclass(slots=True)
class DraftPreviewResult:
    output_file: Path
    subtitle_file: Path
    segments: list[dict[str, Any]]
    estimated_duration_seconds: float
    actual_duration_seconds: float | None
    composition: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "draft_ready",
            "output_file": str(self.output_file),
            "subtitle_file": str(self.subtitle_file),
            "segments": self.segments,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "actual_duration_seconds": self.actual_duration_seconds,
            "composition": self.composition,
        }


class DraftPreviewService:
    """Builds a reviewable sequence, never a production-quality export."""

    width = 540
    height = 960
    fps = 24

    def render(self, plan: ProductionPlan, source: Source, output_directory: Path) -> DraftPreviewResult:
        if not source.path.is_file():
            raise ProductionRenderError("Source video for draft preview is unavailable.")
        segments = _source_segments(plan)
        if not segments:
            raise ProductionRenderError("Draft ProductionPlan has no source segments for a preview.")
        output_directory.mkdir(parents=True, exist_ok=True)
        subtitle_file = output_directory / "draft-subtitles.ass"
        output_file = output_directory / "draft-preview.mp4"
        _write_draft_ass(subtitle_file, segments)
        command = _preview_command(source.path, segments, subtitle_file, output_file, self.width, self.height, self.fps)
        executable = shutil.which("ffmpeg")
        if not executable:
            raise ProductionRenderError("FFmpeg is required for a draft preview.")
        command[0] = executable
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0 or not output_file.is_file() or output_file.stat().st_size == 0:
            detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()[-1000:]
            raise ProductionRenderError(f"Draft preview render failed: {detail}")
        result = DraftPreviewResult(
            output_file=output_file,
            subtitle_file=subtitle_file,
            segments=segments,
            estimated_duration_seconds=float(plan.timeline.estimated_duration_seconds),
            actual_duration_seconds=round(sum(float(item["duration_seconds"]) for item in segments), 3),
            composition={
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "strategy": "draft_fit_center",
                "encoder_preset": "ultrafast",
                "subtitles": "draft_ass",
            },
        )
        write_json(output_directory / "draft-preview.json", result.to_dict())
        return result


def _source_segments(plan: ProductionPlan) -> list[dict[str, Any]]:
    narrations = {
        segment.segment_id: segment
        for segment in plan.segments if isinstance(segment, NarrationSegment)
    }
    source: list[dict[str, Any]] = []
    ordered = sorted(plan.dialogue_mappings, key=lambda item: item.order)
    for index, segment in enumerate(ordered):
        assert isinstance(segment, DialogueSegment)
        narration = next((narrations[item] for item in segment.linked_segment_ids if item in narrations), None)
        role = narration.narration_role if narration is not None else "body"
        if not segment.linked_segment_ids:
            role = "hook" if index == 0 else "payoff" if index == len(ordered) - 1 else "development"
        source.append({
            "segment_id": segment.segment_id,
            "order": index + 1,
            "role": role,
            "source_start_seconds": segment.source_start_seconds,
            "source_end_seconds": segment.source_end_seconds,
            "duration_seconds": round(segment.source_end_seconds - segment.source_start_seconds, 3),
            "source_text": segment.source_text,
            "draft_subtitle_text": narration.text if narration is not None else segment.source_text,
        })
    return source


def _preview_command(
    source: Path, segments: list[dict[str, Any]], subtitle_file: Path,
    output_file: Path, width: int, height: int, fps: int,
) -> list[str]:
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for segment in segments:
        command.extend([
            "-ss", f"{float(segment['source_start_seconds']):.3f}",
            "-t", f"{float(segment['duration_seconds']):.3f}", "-i", str(source),
        ])
    visual = []
    for index in range(len(segments)):
        visual.append(
            f"[{index}:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x111111,setsar=1[v{index}]"
        )
    audio = [f"[{index}:a]aresample=44100,asetpts=N/SR/TB[a{index}]" for index in range(len(segments))]
    joined_streams = "".join(f"[v{index}][a{index}]" for index in range(len(segments)))
    escaped_ass = subtitle_file.resolve().as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    filters = [
        *visual, *audio,
        f"{joined_streams}concat=n={len(segments)}:v=1:a=1[vcat][aout]",
        f"[vcat]ass='{escaped_ass}'[vout]",
    ]
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", str(output_file),
    ])
    return command


def _write_draft_ass(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 540", "PlayResY: 960", "",
        "[V4+ Styles]", "Format: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Draft,Arial,34,&H00FFFFFF,&H90000000,1,2,0,2,28,28,84,1", "",
        "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    cursor = 0.0
    for segment in segments:
        end = cursor + float(segment["duration_seconds"])
        text = str(segment.get("draft_subtitle_text") or segment.get("source_text") or "").replace("{", "(").replace("}", ")").replace("\n", " ")
        if text:
            lines.append(f"Dialogue: 0,{_ass_time(cursor)},{_ass_time(end)},Draft,,0,0,0,,{text}")
        cursor = end
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ass_time(value: float) -> str:
    total = max(0, int(round(value * 100)))
    hours, total = divmod(total, 360000)
    minutes, total = divmod(total, 6000)
    seconds, centiseconds = divmod(total, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
