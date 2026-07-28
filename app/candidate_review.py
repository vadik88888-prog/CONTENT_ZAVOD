from __future__ import annotations

"""Lightweight, non-destructive candidate boundary review helpers."""

from typing import Any


def validate_boundary_override(
    start: float,
    end: float,
    *,
    source_duration: float | None,
    minimum_duration: float,
    maximum_duration: float,
    transcript_features: dict[str, Any] | None = None,
    scenes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an edited range without recomputing source intelligence.

    The review is intentionally bounded to cached transcript/scene data.  It
    returns warnings for a human to judge rather than silently replacing the
    user's boundary with a new ranking decision.
    """

    errors: list[str] = []
    warnings: list[str] = []
    if start < 0:
        errors.append("Начало не может быть меньше 0 секунд.")
    if source_duration is not None and end > source_duration:
        errors.append("Конец выходит за длительность исходного видео.")
    if end <= start:
        errors.append("Конец должен быть позже начала.")
    duration = round(end - start, 3)
    if duration < minimum_duration:
        errors.append(f"Черновик короче безопасного минимума {minimum_duration:.1f} с.")
    if duration > maximum_duration:
        errors.append(f"Черновик длиннее безопасного максимума {maximum_duration:.1f} с.")
    feature_segments = transcript_features.get("segments", []) if isinstance(transcript_features, dict) else []
    if isinstance(feature_segments, list) and feature_segments:
        near_start = any(abs(float(item.get("start", -999)) - start) <= 0.65 for item in feature_segments if isinstance(item, dict))
        near_end = any(abs(float(item.get("end", -999)) - end) <= 0.65 for item in feature_segments if isinstance(item, dict))
        if not near_start:
            warnings.append("Начало не рядом с известной границей речи: проверьте, что фраза не обрезана.")
        if not near_end:
            warnings.append("Конец не рядом с известной границей речи: проверьте payoff и завершение мысли.")
    boundaries = scenes.get("boundaries", []) if isinstance(scenes, dict) else []
    if isinstance(boundaries, list):
        cuts = [float(item.get("timestamp")) for item in boundaries if isinstance(item, dict) and item.get("timestamp") is not None]
        if any(abs(cut - start) <= 0.12 or abs(cut - end) <= 0.12 for cut in cuts):
            warnings.append("Граница совпадает с монтажным cut: в preview проверьте естественность перехода.")
    return {
        "valid": not errors,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": duration,
        "errors": errors,
        "warnings": warnings,
        "revalidation": "cached_transcript_and_scene_only",
    }
