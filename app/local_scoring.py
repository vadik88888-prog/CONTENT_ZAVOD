from __future__ import annotations

from typing import Any

from app.audio_features import window_audio_features
from app.candidate_quality import build_composition_intent, build_eligibility_decision, build_score_v2
from app.config import ScoringConfig
from app.models import Candidate
from app.scene_detection import window_scene_features


def score_candidates(
    candidates: list[Candidate], audio_features: dict[str, Any], scene_boundaries: dict[str, Any], config: ScoringConfig,
    *, min_duration_seconds: float | None = None, max_duration_seconds: float | None = None,
    visual_analysis: dict[str, Any] | None = None,
    transcript_features: dict[str, Any] | None = None,
) -> list[Candidate]:
    for candidate in candidates:
        transcript = dict(candidate.feature_vector)
        if transcript_features is not None:
            transcript["speech_clarity_segments"] = _speech_clarity_segments(candidate, transcript_features)
        audio = window_audio_features(candidate.start, candidate.end, audio_features)
        visual = window_scene_features(candidate.start, candidate.end, scene_boundaries)
        features = {**transcript, **audio, **visual}
        scores = {
            "hook": _bounded(float(features.get("hook_phrase_score", 0))),
            "completeness": _bounded(float(features.get("completeness_score", 0))),
            "clarity": _clarity(features),
            "speech_density": _density(float(features.get("speech_density", 0))),
            "pacing": _pacing(float(features.get("words_per_second", 0))),
            "audio_energy": _bounded(float(features.get("audio_energy", 0)) * 100),
            "scene_structure": _scene_structure(features),
            "context_independence": _bounded(100 - float(features.get("context_dependency_score", 0))),
            "boundary_quality": _boundary_quality(features),
        }
        weighted = sum(scores[name] * config.weights[name] for name in config.weights)
        repetition_penalty = _bounded(float(features.get("repetition_score", 0)) * config.repetition_penalty_weight)
        filler_penalty = _bounded(float(features.get("filler_word_ratio", 0)) * config.filler_penalty_weight)
        candidate.feature_vector = features
        candidate.local_scores = {
            **{key: round(value, 3) for key, value in scores.items()},
            "repetition_penalty": round(repetition_penalty, 3),
            "filler_penalty": round(filler_penalty, 3),
            "weighted_score": round(weighted, 3),
        }
        candidate.eligibility_decision = build_eligibility_decision(
            candidate, features, config_version=config.candidate_quality_config_version,
            min_duration_seconds=min_duration_seconds, max_duration_seconds=max_duration_seconds,
            visual_analysis=visual_analysis,
        )
        candidate.candidate_score_v2 = build_score_v2(
            candidate, scores, features, candidate.eligibility_decision,
            config_version=config.candidate_quality_config_version,
            min_duration_seconds=min_duration_seconds, max_duration_seconds=max_duration_seconds,
            visual_analysis=visual_analysis,
        )
        candidate.local_quality_score = round(candidate.candidate_score_v2.final_score, 3)
        candidate.composition_intent = build_composition_intent(candidate)
        candidate.explanations = _explanations(candidate, scores, repetition_penalty, filler_penalty)
    return candidates


def _speech_clarity_segments(candidate: Candidate, transcript_features: dict[str, Any]) -> list[dict[str, Any]]:
    """Reuse persisted ASR segments for A-1 without invoking any provider."""

    expected_ids = set(candidate.transcript_segment_ids)
    segments: list[dict[str, Any]] = []
    for item in transcript_features.get("segments", []):
        if not isinstance(item, dict):
            continue
        try:
            segment_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if segment_id in expected_ids:
            segments.append({
                "id": segment_id, "start": item.get("start"), "end": item.get("end"),
                "transcript_confidence": item.get("transcript_confidence"),
            })
    return segments


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, value))


def _clarity(features: dict[str, Any]) -> float:
    confidence = features.get("transcript_confidence")
    base = 65.0 if confidence is None else float(confidence) * 100
    return _bounded(base - float(features.get("filler_word_ratio", 0)) * 35)


def _density(value: float) -> float:
    return _bounded(value * 100)


def _pacing(words_per_second: float) -> float:
    return _bounded(100 - min(100, abs(words_per_second - 2.5) * 35))


def _scene_structure(features: dict[str, Any]) -> float:
    activity = float(features.get("visual_activity", 0)) * 60
    boundaries = 20 * (float(features.get("scene_change_near_start", 0)) + float(features.get("scene_change_near_end", 0)))
    return _bounded(30 + activity + boundaries)


def _boundary_quality(features: dict[str, Any]) -> float:
    score = 30.0
    score += 25 if features.get("sentence_start") else 0
    score += 25 if features.get("sentence_end") else 0
    score += 10 * float(features.get("silence_before", 0))
    score += 10 * float(features.get("silence_after", 0))
    return _bounded(score)


def _explanations(candidate: Candidate, scores: dict[str, float], repetition: float, filler: float) -> list[str]:
    reasons = [candidate.start_boundary_reason, candidate.end_boundary_reason]
    if scores["hook"] >= 40:
        reasons.append("Есть детерминированный hook-паттерн или выразительная пунктуация.")
    if scores["completeness"] >= 80:
        reasons.append("Фрагмент начинается и заканчивается на естественных границах мысли.")
    if scores["audio_energy"] >= 55:
        reasons.append("Энергия аудио выше среднего уровня фрагмента.")
    if repetition:
        reasons.append("Применён штраф за повторяемость текста.")
    if filler:
        reasons.append("Применён штраф за слова-паразиты.")
    return [reason for reason in reasons if reason]
