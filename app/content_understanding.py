"""Deterministic, source-scoped content understanding for Goal 5A.

The module deliberately keeps language interpretation separate from pipeline
decisions.  It produces validated, grounded artifacts from transcript and
existing media signals; later stages may enrich them with structured AI
proposals, but can always fall back to this local implementation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


VIDEO_CONTENT_PROFILE_SCHEMA_VERSION = "5A.1"
CONTENT_STRATEGY_VERSION = "5A.1"

CONTENT_TYPES = frozenset({
    "podcast", "interview", "lecture", "educational", "motivational",
    "movie_or_series", "gameplay", "commentary", "documentary",
    "news_or_analysis", "vlog", "tutorial", "mixed", "unknown",
})
DOMINANT_FORMATS = frozenset({
    "single_speaker_monologue", "multi_speaker_dialogue", "host_guest",
    "narrated_visual", "scene_driven", "gameplay_commentary",
    "screen_recording", "mixed", "unknown",
})

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", re.UNICODE)
_MOTIVATIONAL_TERMS = (
    "не сдавай", "побед", "вер", "мечт", "шанс", "успех", "сильн",
    "never give up", "believe", "win", "success", "fight", "dream",
)
_EDUCATIONAL_TERMS = (
    "как ", "почему", "объясн", "урок", "метод", "шаг", "learn",
    "how ", "why ", "lesson", "because", "example",
)
_DIALOGUE_TERMS = ("вопрос", "ответ", "спросил", "интервью", "question", "answer")


@dataclass(slots=True)
class VideoContentProfile:
    schema_version: str
    source_id: str
    source_duration_seconds: float
    language: str
    detected_content_type: str
    content_type_confidence: float
    secondary_content_types: list[str]
    dominant_format: str
    speaker_count_estimate: int
    dialogue_style: str
    narrative_style: str
    pacing_profile: str
    emotional_curve_summary: str
    visual_density: float
    speech_density: float
    useful_content_density: float
    repetition_level: float
    recommended_short_strategy: str
    recommended_clip_duration_range: dict[str, float]
    estimated_story_count: int
    estimated_publishable_clip_range: dict[str, int]
    analysis_confidence: float
    warnings: list[str] = field(default_factory=list)
    strategy_id: str = "generic_fallback"
    fallback_used: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != VIDEO_CONTENT_PROFILE_SCHEMA_VERSION:
            raise ValueError("Unsupported VideoContentProfile schema version.")
        if not self.source_id or self.source_duration_seconds < 0:
            raise ValueError("VideoContentProfile requires a source id and non-negative duration.")
        if self.detected_content_type not in CONTENT_TYPES:
            raise ValueError("Unsupported VideoContentProfile content type.")
        if self.dominant_format not in DOMINANT_FORMATS:
            raise ValueError("Unsupported VideoContentProfile dominant format.")
        if not 0 <= self.content_type_confidence <= 1 or not 0 <= self.analysis_confidence <= 1:
            raise ValueError("VideoContentProfile confidence must be between zero and one.")
        if self.speaker_count_estimate < 0:
            raise ValueError("VideoContentProfile speaker_count_estimate cannot be negative.")
        if not all(0 <= value <= 1 for value in (
            self.visual_density, self.speech_density, self.useful_content_density, self.repetition_level,
        )):
            raise ValueError("VideoContentProfile density values must be between zero and one.")
        minimum = self.recommended_clip_duration_range.get("min_seconds")
        maximum = self.recommended_clip_duration_range.get("max_seconds")
        if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)) or not 0 < minimum <= maximum:
            raise ValueError("VideoContentProfile has an invalid recommended clip duration range.")
        lower = self.estimated_publishable_clip_range.get("min")
        upper = self.estimated_publishable_clip_range.get("max")
        if not isinstance(lower, int) or not isinstance(upper, int) or not 0 <= lower <= upper:
            raise ValueError("VideoContentProfile has an invalid publishable clip range.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoContentProfile":
        profile = cls(
            schema_version=str(data.get("schema_version", "")),
            source_id=str(data.get("source_id", "")),
            source_duration_seconds=float(data.get("source_duration_seconds", 0)),
            language=str(data.get("language") or "unknown"),
            detected_content_type=str(data.get("detected_content_type") or "unknown"),
            content_type_confidence=float(data.get("content_type_confidence", 0)),
            secondary_content_types=[str(item) for item in data.get("secondary_content_types", [])],
            dominant_format=str(data.get("dominant_format") or "unknown"),
            speaker_count_estimate=int(data.get("speaker_count_estimate", 0)),
            dialogue_style=str(data.get("dialogue_style") or "unknown"),
            narrative_style=str(data.get("narrative_style") or "unknown"),
            pacing_profile=str(data.get("pacing_profile") or "unknown"),
            emotional_curve_summary=str(data.get("emotional_curve_summary") or "unknown"),
            visual_density=float(data.get("visual_density", 0)),
            speech_density=float(data.get("speech_density", 0)),
            useful_content_density=float(data.get("useful_content_density", 0)),
            repetition_level=float(data.get("repetition_level", 0)),
            recommended_short_strategy=str(data.get("recommended_short_strategy") or "generic_fallback"),
            recommended_clip_duration_range=dict(data.get("recommended_clip_duration_range", {})),
            estimated_story_count=int(data.get("estimated_story_count", 0)),
            estimated_publishable_clip_range=dict(data.get("estimated_publishable_clip_range", {})),
            analysis_confidence=float(data.get("analysis_confidence", 0)),
            warnings=[str(item) for item in data.get("warnings", [])],
            strategy_id=str(data.get("strategy_id") or "generic_fallback"),
            fallback_used=bool(data.get("fallback_used", True)),
            evidence=dict(data.get("evidence", {})),
        )
        profile.validate()
        return profile


class ContentStrategy(Protocol):
    """Small extension point for Goal 5A strategies, not a scoring engine."""

    strategy_id: str

    def build_profile(
        self,
        source: dict[str, Any],
        metadata: dict[str, Any],
        transcript: dict[str, Any],
        transcript_features: dict[str, Any],
        audio_features: dict[str, Any],
        scenes: dict[str, Any],
        visual_analysis: dict[str, Any],
        config: Any,
    ) -> VideoContentProfile:
        ...


@dataclass(slots=True)
class DeterministicContentStrategy:
    """Grounded fallback used for every source until an optional AI proposal is valid."""

    strategy_id: str = "generic_fallback"

    def build_profile(
        self,
        source: dict[str, Any],
        metadata: dict[str, Any],
        transcript: dict[str, Any],
        transcript_features: dict[str, Any],
        audio_features: dict[str, Any],
        scenes: dict[str, Any],
        visual_analysis: dict[str, Any],
        config: Any,
    ) -> VideoContentProfile:
        raw_segments = [item for item in transcript.get("segments", []) if isinstance(item, dict)]
        feature_segments = [item for item in transcript_features.get("segments", []) if isinstance(item, dict)]
        text = " ".join(str(item.get("text", "")).strip() for item in raw_segments).strip()
        tokens = _tokens(text)
        duration = max(0.0, float(metadata.get("duration") or transcript.get("duration") or 0.0))
        speaker_count = _speaker_count(raw_segments)
        content_type, confidence, secondary = _content_type(text, str(source.get("display_name") or ""), speaker_count)
        dominant_format = _dominant_format(content_type, speaker_count, text, scenes)
        speech_density = _speech_density(tokens, duration, feature_segments)
        repetition = _average(feature_segments, "repetition_score")
        filler = _average(feature_segments, "filler_word_ratio")
        visual_density = _visual_density(duration, scenes, visual_analysis)
        useful_density = _bounded(speech_density * (1 - filler) * (1 - repetition * 0.45))
        emotion = _emotional_summary(text, audio_features)
        pacing = _pacing(tokens, duration)
        strategy = _strategy_id(content_type, dominant_format)
        estimated_stories = _preliminary_story_count(duration, useful_density, repetition, raw_segments)
        clip_range = _clip_range(estimated_stories, useful_density, repetition)
        warnings: list[str] = []
        if not raw_segments:
            warnings.append("Транскрипт пуст: применён безопасный общий fallback.")
        if speaker_count == 0:
            warnings.append("Не удалось оценить число говорящих по исходному транскрипту.")
        if duration <= 0:
            warnings.append("Длительность источника недоступна; профиль имеет пониженную уверенность.")
        analysis_confidence = _bounded(
            (0.45 if raw_segments else 0.05)
            + min(0.25, len(tokens) / 800)
            + (0.15 if feature_segments else 0.0)
            + (0.10 if duration > 0 else 0.0)
            + (0.05 if visual_analysis else 0.0)
        )
        profile = VideoContentProfile(
            schema_version=VIDEO_CONTENT_PROFILE_SCHEMA_VERSION,
            source_id=str(source.get("id") or transcript.get("source_id") or "unknown"),
            source_duration_seconds=round(duration, 3),
            language=str(transcript.get("language") or transcript_features.get("language") or "unknown"),
            detected_content_type=content_type,
            content_type_confidence=round(confidence, 3),
            secondary_content_types=secondary,
            dominant_format=dominant_format,
            speaker_count_estimate=speaker_count,
            dialogue_style=("dialogue" if speaker_count >= 2 else "monologue" if speaker_count == 1 else "unknown"),
            narrative_style=_narrative_style(content_type, text),
            pacing_profile=pacing,
            emotional_curve_summary=emotion,
            visual_density=round(visual_density, 3),
            speech_density=round(speech_density, 3),
            useful_content_density=round(useful_density, 3),
            repetition_level=round(repetition, 3),
            recommended_short_strategy=_recommended_short_strategy(strategy),
            recommended_clip_duration_range={
                "min_seconds": float(getattr(config.candidate_generation, "min_duration_seconds", 15.0)),
                "max_seconds": float(getattr(config.candidate_generation, "max_duration_seconds", 60.0)),
            },
            estimated_story_count=estimated_stories,
            estimated_publishable_clip_range=clip_range,
            analysis_confidence=round(analysis_confidence, 3),
            warnings=warnings,
            strategy_id=strategy,
            fallback_used=True,
            evidence={
                "transcript_segment_count": len(raw_segments),
                "word_count": len(tokens),
                "speaker_ids": _speaker_ids(raw_segments),
                "scene_boundary_count": len(scenes.get("boundaries", [])),
                "filename_signal_used": False,
            },
        )
        profile.validate()
        return profile


def build_video_content_profile(
    source: dict[str, Any],
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    visual_analysis: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Build a validated profile without treating filename data as primary evidence."""

    profile = DeterministicContentStrategy().build_profile(
        source, metadata, transcript, transcript_features, audio_features, scenes, visual_analysis, config,
    )
    return profile.to_dict()


def _tokens(text: str) -> list[str]:
    return [item.casefold() for item in _WORD_RE.findall(text)]


def _speaker_ids(segments: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("speaker_id") or item.get("speaker") or "").strip() for item in segments if str(item.get("speaker_id") or item.get("speaker") or "").strip()})


def _speaker_count(segments: list[dict[str, Any]]) -> int:
    identifiers = _speaker_ids(segments)
    return len(identifiers) if identifiers else (1 if segments else 0)


def _content_type(text: str, filename: str, speaker_count: int) -> tuple[str, float, list[str]]:
    lowered = text.casefold()
    motivational = sum(term in lowered for term in _MOTIVATIONAL_TERMS)
    educational = sum(term in lowered for term in _EDUCATIONAL_TERMS)
    dialogue = sum(term in lowered for term in _DIALOGUE_TERMS) + (2 if speaker_count >= 2 else 0)
    # Filename is intentionally only a weak, tie-breaking signal.
    filename_hint = "мотив" in filename.casefold() or "motivat" in filename.casefold()
    scores = {"motivational": motivational, "educational": educational, "interview": dialogue}
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "unknown", 0.35 if text else 0.1, []
    if best_type == "interview" and speaker_count < 2:
        best_type = "commentary"
    confidence = min(0.92, 0.45 + best_score * 0.10 + (0.08 if speaker_count >= 2 and best_type == "interview" else 0.0))
    if filename_hint and best_type == "motivational":
        confidence = min(0.94, confidence + 0.02)
    secondary = [name for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if name != best_type and score > 0]
    return best_type, confidence, secondary[:2]


def _dominant_format(content_type: str, speaker_count: int, text: str, scenes: dict[str, Any]) -> str:
    if speaker_count >= 2:
        return "multi_speaker_dialogue"
    if speaker_count == 1 and text.strip():
        return "single_speaker_monologue"
    if len(scenes.get("boundaries", [])) >= 8:
        return "scene_driven"
    return "unknown"


def _strategy_id(content_type: str, dominant_format: str) -> str:
    if content_type == "motivational" and dominant_format == "single_speaker_monologue":
        return "motivational_monologue"
    if dominant_format in {"multi_speaker_dialogue", "host_guest"}:
        return "generic_dialogue"
    if content_type in {"educational", "lecture", "tutorial"}:
        return "generic_educational"
    if dominant_format in {"scene_driven", "narrated_visual", "gameplay_commentary", "screen_recording"}:
        return "generic_scene_driven"
    if dominant_format == "single_speaker_monologue":
        return "generic_monologue"
    return "generic_fallback"


def _recommended_short_strategy(strategy: str) -> str:
    return {
        "motivational_monologue": "Выбирать завершённые мотивирующие аргументы и эмоциональные payoffs.",
        "generic_monologue": "Выбирать автономные тезисы, примеры и выводы на границах предложений.",
        "generic_dialogue": "Сохранять пары вопрос–ответ и не переключать смысл посреди реплики.",
        "generic_educational": "Сохранять объяснение вместе с примером или выводом.",
        "generic_scene_driven": "Приоритизировать завершённые визуально-смысловые эпизоды.",
    }.get(strategy, "Использовать безопасные законченные фрагменты с подтверждёнными границами.")


def _speech_density(tokens: list[str], duration: float, features: list[dict[str, Any]]) -> float:
    if features:
        return _bounded(_average(features, "speech_density"))
    return _bounded((len(tokens) / max(duration, 0.01)) / 3.5)


def _visual_density(duration: float, scenes: dict[str, Any], visual_analysis: dict[str, Any]) -> float:
    per_minute = len(scenes.get("boundaries", [])) / max(duration / 60.0, 1.0)
    sampled = len(visual_analysis.get("samples", [])) if isinstance(visual_analysis, dict) else 0
    return _bounded(min(1.0, per_minute / 12.0 + min(0.2, sampled / 100.0)))


def _emotional_summary(text: str, audio_features: dict[str, Any]) -> str:
    punctuation = text.count("!") + text.count("?")
    energy = _average([item for item in audio_features.get("windows", []) if isinstance(item, dict)], "energy")
    if punctuation >= 4 or energy >= 0.65:
        return "выраженные эмоциональные пики"
    if punctuation >= 1 or energy >= 0.35:
        return "умеренная эмоциональная динамика"
    return "ровная эмоциональная подача"


def _pacing(tokens: list[str], duration: float) -> str:
    wps = len(tokens) / max(duration, 0.01)
    if wps < 1.4:
        return "slow"
    if wps > 3.4:
        return "fast"
    return "balanced"


def _narrative_style(content_type: str, text: str) -> str:
    lowered = text.casefold()
    if content_type == "motivational":
        return "argumentative_motivational"
    if content_type in {"educational", "lecture", "tutorial"}:
        return "explanatory"
    if any(token in lowered for token in ("сначала", "потом", "затем", "first", "then")):
        return "chronological"
    return "monologic" if text.strip() else "unknown"


def _preliminary_story_count(duration: float, useful_density: float, repetition: float, segments: list[dict[str, Any]]) -> int:
    if not segments:
        return 0
    base = max(1, round(duration / 75.0))
    density_bonus = 1 if useful_density >= 0.55 else 0
    repetition_penalty = 1 if repetition >= 0.45 else 0
    return max(1, base + density_bonus - repetition_penalty)


def _clip_range(stories: int, useful_density: float, repetition: float) -> dict[str, int]:
    if stories <= 0:
        return {"min": 0, "max": 0}
    minimum = max(1, stories - (1 if repetition > 0.45 else 0))
    maximum = max(minimum, stories + (1 if useful_density >= 0.6 else 0))
    return {"min": minimum, "max": maximum}


def _average(items: list[dict[str, Any]], field_name: str) -> float:
    values = [float(item[field_name]) for item in items if item.get(field_name) is not None]
    return sum(values) / len(values) if values else 0.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))
