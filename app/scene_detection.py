from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.config import SceneDetectionConfig
from app.errors import ClipEngineError
from app.media import _tool, run_checked


PTS_RE = re.compile(r"pts_time:([0-9.]+)")
SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def detect_scene_boundaries(
    video_path: Path, duration: float, config: SceneDetectionConfig
) -> dict[str, Any]:
    if not config.enabled:
        return {"enabled": False, "boundaries": [], "scene_boundary_count": 0}
    try:
        result = run_checked([
            _tool("ffmpeg"), "-hide_banner", "-loglevel", "info", "-i", str(video_path),
            "-filter:v", f"select='gt(scene,{config.threshold})',metadata=print", "-an", "-f", "null", "-",
        ], timeout=max(120, int(duration * 3)))
    except ClipEngineError as error:
        return {"enabled": True, "threshold": config.threshold, "boundaries": [], "scene_boundary_count": 0, "warning": str(error)}
    return {
        "enabled": True,
        "threshold": config.threshold,
        "boundaries": parse_scene_output(result.stderr or result.stdout, duration),
        "scene_boundary_count": 0,
    } | _boundary_count(result.stderr or result.stdout, duration)


def parse_scene_output(output: str, duration: float) -> list[dict[str, float]]:
    pending_time: float | None = None
    values: list[tuple[float, float]] = []
    for line in output.splitlines():
        match = PTS_RE.search(line)
        if match:
            pending_time = float(match.group(1))
        score = SCORE_RE.search(line)
        if score and pending_time is not None:
            values.append((pending_time, float(score.group(1))))
            pending_time = None
    result: list[dict[str, float]] = []
    for index, (timestamp, score) in enumerate(values):
        previous = values[index - 1][0] if index else 0.0
        following = values[index + 1][0] if index + 1 < len(values) else duration
        result.append({
            "timestamp": round(timestamp, 3),
            "scene_change_score": round(score, 5),
            "distance_from_previous_scene": round(max(0.0, timestamp - previous), 3),
            "distance_to_next_scene": round(max(0.0, following - timestamp), 3),
        })
    return result


def window_scene_features(start: float, end: float, scenes: dict[str, Any]) -> dict[str, float]:
    boundaries = scenes.get("boundaries", [])
    inside = [item for item in boundaries if start < float(item["timestamp"]) < end]
    near_start = any(abs(float(item["timestamp"]) - start) <= 1.0 for item in boundaries)
    near_end = any(abs(float(item["timestamp"]) - end) <= 1.0 for item in boundaries)
    duration = max(0.1, end - start)
    return {
        "scene_changes_inside": len(inside),
        "scene_change_near_start": float(near_start),
        "scene_change_near_end": float(near_end),
        "visual_activity": round(min(1.0, len(inside) / max(1.0, duration / 8)), 3),
        "static_scene_ratio": round(max(0.0, 1.0 - min(1.0, len(inside) / max(1.0, duration / 8))), 3),
    }


def _boundary_count(output: str, duration: float) -> dict[str, int]:
    return {"scene_boundary_count": len(parse_scene_output(output, duration))}
