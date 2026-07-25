from __future__ import annotations

from typing import Any

from app.config import CandidateGenerationConfig
from app.models import Candidate
from app.transcript_features import candidate_transcript_features


def generate_candidates(
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scene_boundaries: dict[str, Any],
    config: CandidateGenerationConfig,
) -> list[Candidate]:
    return generate_candidates_with_stats(transcript, transcript_features, audio_features, scene_boundaries, config)[0]


def generate_candidates_with_stats(
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scene_boundaries: dict[str, Any],
    config: CandidateGenerationConfig,
) -> tuple[list[Candidate], int]:
    segments = transcript_features.get("segments", [])
    total_duration = float(transcript.get("duration", segments[-1]["end"] if segments else 0))
    if not segments:
        return [], 0
    candidates: list[Candidate] = []
    for start_index, start_segment in enumerate(segments):
        if len(candidates) >= config.max_candidates:
            break
        start = float(start_segment["start"])
        endings = [
            (index, segment)
            for index, segment in enumerate(segments[start_index:], start_index)
            if config.min_duration_seconds <= float(segment["end"]) - start <= config.max_duration_seconds
        ]
        if not endings:
            continue
        end_index, end_segment = min(
            endings,
            key=lambda item: _end_cost(item[1], start, config.target_duration_seconds),
        )
        end = float(end_segment["end"])
        if end - start < config.min_duration_seconds:
            continue
        ids = [int(segment["id"]) for segment in segments[start_index:end_index + 1]]
        text = " ".join(str(segment["text"]).strip() for segment in segments[start_index:end_index + 1]).strip()
        start_reason = _start_reason(start_segment, audio_features, scene_boundaries, config)
        end_reason = _end_reason(end_segment, audio_features, scene_boundaries, config)
        features = candidate_transcript_features(start, end, transcript_features)
        candidates.append(Candidate(
            id=f"candidate-{len(candidates) + 1:03d}",
            start=max(0.0, start),
            end=min(total_duration, end),
            text=text,
            reason="Естественные границы речи, пауз, сцен и целевой длительности.",
            transcript_segment_ids=ids,
            start_boundary_reason=start_reason,
            end_boundary_reason=end_reason,
            feature_vector=features,
        ))
    return _dedupe(candidates, config.overlap_limit), len(candidates)


def _end_cost(segment: dict[str, Any], start: float, target: float) -> float:
    duration_cost = abs((float(segment["end"]) - start) - target)
    boundary_bonus = 4 if segment.get("sentence_end") else 0
    pause_bonus = min(3.0, float(segment.get("pause_after_seconds", 0)) * 3)
    return duration_cost - boundary_bonus - pause_bonus


def _start_reason(
    segment: dict[str, Any], audio: dict[str, Any], scenes: dict[str, Any], config: CandidateGenerationConfig
) -> str:
    start = float(segment["start"])
    if float(segment.get("pause_before_seconds", 0)) >= 0.3:
        return "Пауза перед началом фразы."
    if segment.get("sentence_start"):
        return "Начало предложения."
    if _scene_near(start, scenes, config.boundary_search_radius_seconds):
        return "Смена сцены рядом с границей."
    return "Граница сегмента речи."


def _end_reason(
    segment: dict[str, Any], audio: dict[str, Any], scenes: dict[str, Any], config: CandidateGenerationConfig
) -> str:
    end = float(segment["end"])
    if segment.get("sentence_end"):
        return "Конец предложения."
    if float(segment.get("pause_after_seconds", 0)) >= 0.3:
        return "Пауза после фразы."
    if _scene_near(end, scenes, config.boundary_search_radius_seconds):
        return "Смена сцены рядом с границей."
    return "Граница сегмента речи."


def _scene_near(timestamp: float, scenes: dict[str, Any], radius: float) -> bool:
    return any(abs(float(item["timestamp"]) - timestamp) <= radius for item in scenes.get("boundaries", []))


def _dedupe(candidates: list[Candidate], limit: float) -> list[Candidate]:
    result: list[Candidate] = []
    for candidate in candidates:
        if any(_overlap(candidate, previous) >= limit for previous in result):
            continue
        result.append(candidate)
    return result


def _overlap(first: Candidate, second: Candidate) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    smallest = min(first.duration, second.duration)
    return overlap / smallest if smallest else 0.0
