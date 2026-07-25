from __future__ import annotations

from typing import Any

from app.config import AppConfig
from app.models import Candidate
from app.utils import write_json


def build_candidates(transcript: dict[str, Any], config: AppConfig) -> list[Candidate]:
    segments = transcript.get("segments", [])
    candidates: list[Candidate] = []
    current: list[dict[str, Any]] = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        if end <= start:
            continue
        if current and end - float(current[0]["start"]) > config.max_clip_duration:
            _add_candidate(candidates, current, config, transcript["duration"])
            current = []
        current.append(segment)
        duration = end - float(current[0]["start"])
        terminal = _terminal_text(str(segment.get("text", "")))
        if duration >= config.target_clip_duration or (duration >= config.min_clip_duration and terminal):
            _add_candidate(candidates, current, config, transcript["duration"])
            current = []
    if current:
        _add_candidate(candidates, current, config, transcript["duration"])
    return _dedupe(candidates, config.overlap_threshold)


def _add_candidate(
    destination: list[Candidate], segments: list[dict[str, Any]], config: AppConfig, total_duration: float
) -> None:
    if not segments:
        return
    raw_start = float(segments[0]["start"])
    raw_end = float(segments[-1]["end"])
    if raw_end - raw_start < config.min_clip_duration:
        return
    padding_budget = max(0.0, config.max_clip_duration - (raw_end - raw_start))
    pre_roll = min(config.pre_roll_seconds, padding_budget / 2)
    post_roll = min(config.post_roll_seconds, padding_budget - pre_roll)
    start = max(0.0, raw_start - pre_roll)
    end = min(float(total_duration), raw_end + post_roll)
    text = " ".join(str(item.get("text", "")).strip() for item in segments).strip()
    destination.append(Candidate(
        id=f"candidate-{len(destination) + 1:03d}",
        start=start,
        end=end,
        text=text,
        reason="Локальное смысловое окно: границы установлены по сегментам речи.",
    ))


def _terminal_text(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…"))


def _dedupe(candidates: list[Candidate], threshold: float) -> list[Candidate]:
    result: list[Candidate] = []
    for candidate in candidates:
        if not any(_overlap_ratio(candidate, previous) >= threshold for previous in result):
            result.append(candidate)
    return result


def _overlap_ratio(first: Candidate, second: Candidate) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shortest = min(first.duration, second.duration)
    return overlap / shortest if shortest else 0.0


def save_candidates(path, candidates: list[Candidate]) -> None:
    write_json(path, {"candidates": [candidate.to_dict() for candidate in candidates]})
