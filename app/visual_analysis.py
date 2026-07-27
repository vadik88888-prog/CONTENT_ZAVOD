"""Optional, bounded visual analysis used only to improve vertical reframing."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.ai import sanitize_api_error
from app.config import AppConfig


def analyse_video_subjects(source: Path, duration_seconds: float, config: AppConfig) -> dict[str, Any]:
    """Return sparse, cacheable subject positions or a safe non-fatal fallback.

    Only a handful of JPEG samples are sent when the user enabled deep analysis.
    The result intentionally stores positions, not source frames, so project
    storage remains local and compact.
    """

    if not config.optional_visual_features:
        return {"enabled": False, "status": "skipped", "reason": "disabled", "subject_keyframes": []}
    if config.ai.provider != "openai" or not os.getenv("OPENAI_API_KEY"):
        return {"enabled": True, "status": "fallback", "reason": "visual_provider_unavailable", "subject_keyframes": []}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"enabled": True, "status": "fallback", "reason": "ffmpeg_unavailable", "subject_keyframes": []}
    times = _sample_times(duration_seconds)
    images: list[tuple[float, str]] = []
    for time_seconds in times:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{time_seconds:.3f}", "-i", str(source), "-frames:v", "1", "-vf", "scale=512:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                capture_output=True, timeout=45, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.stdout:
            images.append((time_seconds, base64.b64encode(result.stdout).decode("ascii")))
    if not images:
        return {"enabled": True, "status": "fallback", "reason": "no_sample_frames", "subject_keyframes": []}
    try:
        from openai import OpenAI

        content: list[dict[str, Any]] = [{
            "type": "input_text",
            "text": (
                "For each frame, return only non-identifying composition signals. Locate the most important visual target "
                "with normalized center x/y (0..1), approximate normalized width/height, and confidence. Select a "
                "tracking_target: primary_face, primary_person, important_object, screen_region, subject_group, "
                "scene_center, or none. Count visible relevant faces. Set active_speaker_confidence only when visual "
                "speaking evidence is clear; otherwise use 0. scene_id may describe a stable shot/setup but must not "
                "identify people. Do not infer names, identities, or unobserved facts."
            ),
        }]
        for time_seconds, image in images:
            content.append({"type": "input_text", "text": f"time_seconds={time_seconds:.3f}"})
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}", "detail": "low"})
        response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).responses.create(
            model=config.ai.model, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "subject_positions", "strict": True, "schema": _SUBJECT_SCHEMA}},
        )
        parsed = json.loads(response.output_text)
        items = parsed.get("subjects", []) if isinstance(parsed, dict) else []
        keyframes = [_keyframe(item) for item in items if isinstance(item, dict)]
        keyframes = [item for item in keyframes if item is not None]
        return {"enabled": True, "status": "completed", "subject_keyframes": keyframes, "sample_count": len(images)}
    except Exception as error:
        return {"enabled": True, "status": "fallback", "reason": sanitize_api_error(error, os.getenv("OPENAI_API_KEY")), "subject_keyframes": []}


_SUBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "time_seconds": {"type": "number"},
                    "normalized_x": {"type": "number", "minimum": 0, "maximum": 1},
                    "normalized_y": {"type": "number", "minimum": 0, "maximum": 1},
                    "normalized_width": {"type": "number", "minimum": 0.02, "maximum": 1},
                    "normalized_height": {"type": "number", "minimum": 0.02, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tracking_target": {
                        "type": "string",
                        "enum": [
                            "primary_face", "primary_person", "important_object", "screen_region",
                            "subject_group", "scene_center", "none",
                        ],
                    },
                    "visible_face_count": {"type": "integer", "minimum": 0, "maximum": 32},
                    "active_speaker_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "scene_id": {"type": "string", "maxLength": 160},
                },
                "required": ["time_seconds", "normalized_x", "normalized_y", "confidence"],
            },
        },
    },
    "required": ["subjects"],
}


def _sample_times(duration: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    count = min(8, max(2, round(duration / 45)))
    return [round(duration * (index + 1) / (count + 1), 3) for index in range(count)]


def _keyframe(value: dict[str, Any]) -> dict[str, Any] | None:
    try:
        result = {name: float(value[name]) for name in ("time_seconds", "normalized_x", "normalized_y", "confidence")}
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= result["normalized_x"] <= 1 or not 0 <= result["normalized_y"] <= 1 or not 0 <= result["confidence"] <= 1:
        return None
    for name in ("normalized_width", "normalized_height", "active_speaker_confidence"):
        if name not in value:
            continue
        try:
            parsed = float(value[name])
        except (TypeError, ValueError):
            continue
        if (name == "active_speaker_confidence" and 0 <= parsed <= 1) or (name != "active_speaker_confidence" and 0.02 <= parsed <= 1):
            result[name] = parsed
    target = value.get("tracking_target")
    if target in {"primary_face", "primary_person", "important_object", "screen_region", "subject_group", "scene_center", "none"}:
        result["tracking_target"] = str(target)
    try:
        face_count = int(value.get("visible_face_count", 1))
    except (TypeError, ValueError):
        face_count = 1
    if 0 <= face_count <= 32:
        result["visible_face_count"] = face_count
    if isinstance(value.get("scene_id"), str) and value["scene_id"].strip():
        result["scene_id"] = value["scene_id"].strip()[:160]
    return result
