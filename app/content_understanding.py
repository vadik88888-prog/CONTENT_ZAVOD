"""Deterministic, source-scoped content understanding for Goal 5A.

The module deliberately keeps language interpretation separate from pipeline
decisions.  It produces validated, grounded artifacts from transcript and
existing media signals; later stages may enrich them with structured AI
proposals, but can always fall back to this local implementation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from app.models import Candidate
from app.transcript_features import candidate_transcript_features
from app.utils import stable_text_hash


VIDEO_CONTENT_PROFILE_SCHEMA_VERSION = "5A.1"
CONTENT_STRATEGY_VERSION = "5A.1"
GLOBAL_CONTENT_MAP_SCHEMA_VERSION = "5A.1"
STORY_UNIT_SCHEMA_VERSION = "5A.1"

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
_TOPIC_MARKERS = ("теперь", "другая тема", "важно понять", "первое", "второе", "finally", "next")
_PAYOFF_MARKERS = ("поэтому", "значит", "вывод", "итог", "вот почему", "therefore", "that is why", "the point")
_SETUP_MARKERS = ("если", "когда", "проблем", "вопрос", "почему", "if ", "when ", "question", "problem")
_STOP_WORDS = frozenset({
    "и", "а", "но", "что", "это", "как", "в", "на", "с", "по", "к", "за", "из", "у", "не", "мы", "вы",
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "is", "it", "that", "this", "for", "with",
})


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


@dataclass(slots=True)
class ContentChapter:
    chapter_id: str
    start: float
    end: float
    duration: float
    title: str
    summary: str
    main_topic: str
    subtopics: list[str]
    speaker_ids: list[str]
    transcript_segment_ids: list[int]
    opening_function: str
    narrative_function: str
    emotional_tone: str
    emotional_intensity: float
    information_density: float
    visual_activity: float
    dependency_on_previous: float
    dependency_on_next: float
    standalone_potential: float
    candidate_story_count: int
    confidence: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentChapter":
        return cls(
            chapter_id=str(data.get("chapter_id") or ""),
            start=float(data.get("start", 0)), end=float(data.get("end", 0)),
            duration=float(data.get("duration", 0)), title=str(data.get("title") or ""),
            summary=str(data.get("summary") or ""), main_topic=str(data.get("main_topic") or ""),
            subtopics=[str(item) for item in data.get("subtopics", [])],
            speaker_ids=[str(item) for item in data.get("speaker_ids", [])],
            transcript_segment_ids=[int(item) for item in data.get("transcript_segment_ids", [])],
            opening_function=str(data.get("opening_function") or "unknown"),
            narrative_function=str(data.get("narrative_function") or "unknown"),
            emotional_tone=str(data.get("emotional_tone") or "neutral"),
            emotional_intensity=float(data.get("emotional_intensity", 0)),
            information_density=float(data.get("information_density", 0)),
            visual_activity=float(data.get("visual_activity", 0)),
            dependency_on_previous=float(data.get("dependency_on_previous", 0)),
            dependency_on_next=float(data.get("dependency_on_next", 0)),
            standalone_potential=float(data.get("standalone_potential", 0)),
            candidate_story_count=int(data.get("candidate_story_count", 0)),
            confidence=float(data.get("confidence", 0)), evidence=dict(data.get("evidence", {})),
        )


@dataclass(slots=True)
class ContentSignature:
    normalized_core_idea: str
    topic_ids: list[str]
    chapter_id: str
    narrative_function: str
    emotional_signature: str
    key_entities: list[str]
    key_claims: list[str]
    keyword_set: list[str]
    lexical_signature: str
    semantic_embedding_ref: str | None
    source_range: dict[str, float]
    transcript_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StoryUnit:
    story_unit_id: str
    chapter_id: str
    start: float
    end: float
    duration: float
    transcript_segment_ids: list[int]
    title: str
    core_idea: str
    hook_seed: str
    setup: str
    development: str
    payoff: str
    ending: str
    emotional_arc: str
    dominant_emotion: str
    speaker_context: str
    required_previous_context: str
    required_next_context: str
    standalone_score: float
    completeness_score: float
    clarity_score: float
    context_dependency_score: float
    information_density: float
    repetition_score: float
    transformation_potential: float
    publishability_precheck: bool
    content_signature: dict[str, Any]
    confidence: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryUnit":
        return cls(
            story_unit_id=str(data.get("story_unit_id") or ""), chapter_id=str(data.get("chapter_id") or ""),
            start=float(data.get("start", 0)), end=float(data.get("end", 0)), duration=float(data.get("duration", 0)),
            transcript_segment_ids=[int(item) for item in data.get("transcript_segment_ids", [])],
            title=str(data.get("title") or ""), core_idea=str(data.get("core_idea") or ""),
            hook_seed=str(data.get("hook_seed") or ""), setup=str(data.get("setup") or ""),
            development=str(data.get("development") or ""), payoff=str(data.get("payoff") or ""),
            ending=str(data.get("ending") or ""), emotional_arc=str(data.get("emotional_arc") or "neutral"),
            dominant_emotion=str(data.get("dominant_emotion") or "neutral"),
            speaker_context=str(data.get("speaker_context") or "unknown"),
            required_previous_context=str(data.get("required_previous_context") or ""),
            required_next_context=str(data.get("required_next_context") or ""),
            standalone_score=float(data.get("standalone_score", 0)),
            completeness_score=float(data.get("completeness_score", 0)),
            clarity_score=float(data.get("clarity_score", 0)),
            context_dependency_score=float(data.get("context_dependency_score", 0)),
            information_density=float(data.get("information_density", 0)),
            repetition_score=float(data.get("repetition_score", 0)),
            transformation_potential=float(data.get("transformation_potential", 0)),
            publishability_precheck=bool(data.get("publishability_precheck", False)),
            content_signature=dict(data.get("content_signature", {})),
            confidence=float(data.get("confidence", 0)), evidence=dict(data.get("evidence", {})),
        )


@dataclass(slots=True)
class GlobalContentMap:
    schema_version: str
    source_id: str
    source_duration_seconds: float
    chapters: list[ContentChapter]
    story_units: list[StoryUnit]
    analysis_confidence: float
    fallback_used: bool
    warnings: list[str]
    evidence: dict[str, Any]

    def validate(self, transcript: dict[str, Any] | None = None) -> None:
        if self.schema_version != GLOBAL_CONTENT_MAP_SCHEMA_VERSION:
            raise ValueError("Unsupported GlobalContentMap schema version.")
        previous_end = -1.0
        chapter_ids: set[str] = set()
        covered_ids: list[int] = []
        for chapter in self.chapters:
            if not chapter.chapter_id or chapter.chapter_id in chapter_ids:
                raise ValueError("ContentMap chapters must have unique ids.")
            if not chapter.start < chapter.end or abs(chapter.duration - (chapter.end - chapter.start)) > 0.02:
                raise ValueError("ContentMap chapter duration is invalid.")
            if chapter.start < previous_end - 0.01:
                raise ValueError("ContentMap chapters must be chronological without uncontrolled overlap.")
            if not chapter.transcript_segment_ids or not chapter.evidence.get("evidence_text"):
                raise ValueError("ContentMap chapters require grounded transcript evidence.")
            previous_end = chapter.end
            chapter_ids.add(chapter.chapter_id)
            covered_ids.extend(chapter.transcript_segment_ids)
        if transcript is not None:
            expected = [
                item["id"] for index, raw in enumerate(transcript.get("segments", []))
                if (item := _valid_segment(raw, index)) is not None
            ]
            if sorted(covered_ids) != expected:
                raise ValueError("ContentMap must cover every valid transcript segment exactly once.")
        for unit in self.story_units:
            chapter = next((item for item in self.chapters if item.chapter_id == unit.chapter_id), None)
            if chapter is None or not unit.start < unit.end or not unit.transcript_segment_ids:
                raise ValueError("StoryUnit must have a valid containing chapter and range.")
            if unit.start < chapter.start - 0.01 or unit.end > chapter.end + 0.01:
                raise ValueError("StoryUnit must stay within its ContentChapter.")
            if not set(unit.transcript_segment_ids).issubset(chapter.transcript_segment_ids):
                raise ValueError("StoryUnit must reference only its chapter transcript segments.")
            if not unit.evidence.get("evidence_text") or not unit.content_signature.get("transcript_fingerprint"):
                raise ValueError("StoryUnit requires grounded evidence and a content signature.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_duration_seconds": self.source_duration_seconds,
            "chapters": [item.to_dict() for item in self.chapters],
            "story_units": [item.to_dict() for item in self.story_units],
            "analysis_confidence": self.analysis_confidence,
            "fallback_used": self.fallback_used,
            "warnings": self.warnings,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], transcript: dict[str, Any] | None = None) -> "GlobalContentMap":
        result = cls(
            schema_version=str(data.get("schema_version") or ""), source_id=str(data.get("source_id") or ""),
            source_duration_seconds=float(data.get("source_duration_seconds", 0)),
            chapters=[ContentChapter.from_dict(item) for item in data.get("chapters", []) if isinstance(item, dict)],
            story_units=[StoryUnit.from_dict(item) for item in data.get("story_units", []) if isinstance(item, dict)],
            analysis_confidence=float(data.get("analysis_confidence", 0)), fallback_used=bool(data.get("fallback_used", True)),
            warnings=[str(item) for item in data.get("warnings", [])], evidence=dict(data.get("evidence", {})),
        )
        result.validate(transcript)
        return result


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


def build_global_content_map(
    source: dict[str, Any],
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    visual_analysis: dict[str, Any],
    profile_data: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Create a fully grounded fallback ContentMap from ordered transcript evidence.

    The algorithm intentionally favours safe continuity over speculative topic
    labels: every valid transcript segment belongs to one chapter, and every
    StoryUnit is a contiguous subset of its chapter.
    """

    profile = VideoContentProfile.from_dict(profile_data)
    segments = [item for index, raw in enumerate(transcript.get("segments", [])) if (item := _valid_segment(raw, index))]
    if not segments:
        empty = GlobalContentMap(
            schema_version=GLOBAL_CONTENT_MAP_SCHEMA_VERSION,
            source_id=profile.source_id,
            source_duration_seconds=float(metadata.get("duration") or profile.source_duration_seconds),
            chapters=[], story_units=[], analysis_confidence=0.0, fallback_used=True,
            warnings=["ContentMap не построен: в транскрипте нет валидных сегментов."],
            evidence={"transcript_segment_count": 0, "strategy_id": profile.strategy_id},
        )
        return empty.to_dict()
    feature_rows = [item for item in transcript_features.get("segments", []) if isinstance(item, dict)]
    # TranscriptFeatures historically indexes rows by position while a future
    # transcript may expose stable explicit ids.  Bind both sources by their
    # ordered evidence position, so all downstream references stay grounded in
    # the ids exposed by the original transcript.
    features = {
        int(segment["id"]): feature_rows[index]
        for index, segment in enumerate(segments) if index < len(feature_rows)
    }
    chapter_groups = _chapter_groups(segments, features, config.content_understanding)
    chapters = [
        _make_chapter(index, group, features, scenes, visual_analysis, profile)
        for index, group in enumerate(chapter_groups, start=1)
    ]
    story_units: list[StoryUnit] = []
    for chapter, group in zip(chapters, chapter_groups, strict=True):
        story_units.extend(_make_story_units(chapter, group, features, config.content_understanding))
    result = GlobalContentMap(
        schema_version=GLOBAL_CONTENT_MAP_SCHEMA_VERSION,
        source_id=profile.source_id,
        source_duration_seconds=float(metadata.get("duration") or profile.source_duration_seconds),
        chapters=chapters,
        story_units=story_units,
        analysis_confidence=round(min(profile.analysis_confidence, 0.82), 3),
        fallback_used=True,
        warnings=list(profile.warnings),
        evidence={
            "strategy_id": profile.strategy_id,
            "transcript_segment_count": len(segments),
            "chaptering": "deterministic: pauses, speaker turns, topic markers, bounded chapter duration",
            "story_extraction": "deterministic: sentence closure, question-answer continuity, setup-payoff continuity",
            "ai_grounding": "not_requested; local fallback is grounded in transcript segment ids",
        },
    )
    result.validate(transcript)
    return result.to_dict()


def story_units_artifact(content_map_data: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    """Write a small independently cacheable reference artifact without duplicating inference."""

    content_map = GlobalContentMap.from_dict(content_map_data, transcript)
    return {
        "schema_version": STORY_UNIT_SCHEMA_VERSION,
        "source_id": content_map.source_id,
        "content_map_schema_version": content_map.schema_version,
        "story_units": [item.to_dict() for item in content_map.story_units],
        "fallback_used": content_map.fallback_used,
    }


def validate_global_content_map(data: dict[str, Any], transcript: dict[str, Any]) -> GlobalContentMap:
    """Public validation point for local or future structured AI content maps."""

    return GlobalContentMap.from_dict(data, transcript)


def _valid_segment(raw: Any, fallback_id: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        start = float(raw.get("start", 0))
        end = float(raw.get("end", start))
    except (TypeError, ValueError):
        return None
    text = str(raw.get("text", "")).strip()
    if not text or not start < end:
        return None
    return {
        "id": int(raw.get("id", fallback_id)), "start": round(start, 3), "end": round(end, 3),
        "text": text, "speaker_id": str(raw.get("speaker_id") or raw.get("speaker") or "").strip(),
    }


def _chapter_groups(
    segments: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if current and _start_new_chapter(current, segment, segments[index - 1], features, settings):
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def _start_new_chapter(
    current: list[dict[str, Any]], candidate: dict[str, Any], previous: dict[str, Any],
    features: dict[int, dict[str, Any]], settings: Any,
) -> bool:
    previous_feature = features.get(int(previous["id"]), {})
    pause = max(0.0, float(candidate["start"]) - float(previous["end"]))
    current_duration = float(previous["end"]) - float(current[0]["start"])
    speaker_changed = bool(candidate.get("speaker_id") and previous.get("speaker_id") and candidate["speaker_id"] != previous["speaker_id"])
    text = str(candidate["text"]).casefold()
    marker = any(text.startswith(item) for item in _TOPIC_MARKERS)
    terminal = bool(previous_feature.get("sentence_end", _sentence_end(str(previous["text"]))))
    return (
        current_duration >= float(settings.max_chapter_seconds)
        or speaker_changed and terminal
        or pause >= float(settings.chapter_pause_seconds) and terminal
        or marker and terminal and current_duration >= 18.0
    )


def _make_chapter(
    index: int,
    group: list[dict[str, Any]],
    features: dict[int, dict[str, Any]],
    scenes: dict[str, Any],
    visual_analysis: dict[str, Any],
    profile: VideoContentProfile,
) -> ContentChapter:
    start, end = float(group[0]["start"]), float(group[-1]["end"])
    text = _group_text(group)
    keywords = _keywords(text)
    first_feature = features.get(int(group[0]["id"]), {})
    last_feature = features.get(int(group[-1]["id"]), {})
    information = _bounded(_average([features.get(int(item["id"]), {}) for item in group], "speech_density") * (1 - _average([features.get(int(item["id"]), {}) for item in group], "filler_word_ratio")))
    visual = _visual_activity(start, end, scenes, visual_analysis)
    dependency_previous = _bounded(float(first_feature.get("context_dependency_score", 25)) / 100)
    dependency_next = _continuation_risk(group[-1], last_feature)
    standalone = _bounded((information * 0.45) + ((1 - dependency_previous) * 0.25) + ((1 - dependency_next) * 0.30))
    narrative = _narrative_function(text, index == 1, False)
    return ContentChapter(
        chapter_id=f"chapter-{index:03d}", start=start, end=end, duration=round(end - start, 3),
        title=_title_from_text(text), summary=_summary_from_text(text), main_topic=(keywords[0] if keywords else "общая тема"),
        subtopics=keywords[1:5], speaker_ids=_speaker_ids(group), transcript_segment_ids=[int(item["id"]) for item in group],
        opening_function="hook" if index == 1 and ("?" in text or "!" in text) else "introduction" if index == 1 else "transition",
        narrative_function=narrative, emotional_tone=_dominant_emotion(text),
        emotional_intensity=round(_emotional_intensity(text), 3), information_density=round(information, 3),
        visual_activity=round(visual, 3), dependency_on_previous=round(dependency_previous, 3),
        dependency_on_next=round(dependency_next, 3), standalone_potential=round(standalone, 3),
        candidate_story_count=0, confidence=round(min(0.85, profile.analysis_confidence), 3),
        evidence={"segment_ids": [int(item["id"]) for item in group], "evidence_text": text[:1200]},
    )


def _make_story_units(
    chapter: ContentChapter, group: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any,
) -> list[StoryUnit]:
    units: list[StoryUnit] = []
    current: list[dict[str, Any]] = []
    for position, segment in enumerate(group):
        current.append(segment)
        following = group[position + 1] if position + 1 < len(group) else None
        if _close_story_unit(current, following, features, settings):
            units.append(_make_story_unit(f"story-{len(units) + 1:03d}", chapter, current, features))
            current = []
    if current:
        units.append(_make_story_unit(f"story-{len(units) + 1:03d}", chapter, current, features))
    chapter.candidate_story_count = len(units)
    return units


def _close_story_unit(
    current: list[dict[str, Any]], following: dict[str, Any] | None,
    features: dict[int, dict[str, Any]], settings: Any,
) -> bool:
    last = current[-1]
    duration = float(last["end"]) - float(current[0]["start"])
    feature = features.get(int(last["id"]), {})
    terminal = bool(feature.get("sentence_end", _sentence_end(str(last["text"]))))
    question = "?" in str(last["text"])
    text = _group_text(current).casefold()
    setup_waiting = any(marker in text for marker in _SETUP_MARKERS) and not any(marker in text for marker in _PAYOFF_MARKERS)
    if following is None:
        return True
    if duration >= float(settings.max_story_unit_seconds):
        return terminal and not question and not setup_waiting
    if duration < float(settings.min_story_unit_seconds) or not terminal:
        return False
    if question:
        return False
    pause = max(0.0, float(following["start"]) - float(last["end"]))
    if setup_waiting and duration < float(settings.target_story_unit_seconds) * 1.5:
        return False
    return duration >= float(settings.target_story_unit_seconds) or pause >= 0.35


def _make_story_unit(
    unit_id: str, chapter: ContentChapter, group: list[dict[str, Any]], features: dict[int, dict[str, Any]],
) -> StoryUnit:
    start, end = float(group[0]["start"]), float(group[-1]["end"])
    text = _group_text(group)
    first_feature = features.get(int(group[0]["id"]), {})
    last_feature = features.get(int(group[-1]["id"]), {})
    information = _bounded(_average([features.get(int(item["id"]), {}) for item in group], "speech_density") * (1 - _average([features.get(int(item["id"]), {}) for item in group], "filler_word_ratio")))
    repetition = _bounded(_average([features.get(int(item["id"]), {}) for item in group], "repetition_score"))
    context_dependency = _bounded(float(first_feature.get("context_dependency_score", 25)) / 100)
    complete = _bounded(0.45 + (0.3 if bool(last_feature.get("sentence_end", _sentence_end(str(group[-1]["text"])))) else 0) + (0.15 if not _continuation_risk(group[-1], last_feature) else 0))
    clarity = _bounded(0.7 - repetition * 0.25 - _average([features.get(int(item["id"]), {}) for item in group], "filler_word_ratio") * 0.25)
    standalone = _bounded(complete * 0.42 + clarity * 0.25 + (1 - context_dependency) * 0.23 + information * 0.10)
    narrative = _narrative_function(text, False, False)
    signature = _content_signature(chapter.chapter_id, start, end, text, narrative, _dominant_emotion(text))
    payoff = _last_sentence(text) if any(marker in text.casefold() for marker in _PAYOFF_MARKERS) else ""
    setup = _first_sentence(text) if any(marker in text.casefold() for marker in _SETUP_MARKERS) else ""
    publishable = standalone >= 0.55 and complete >= 0.6 and context_dependency <= 0.6 and bool(text.strip())
    return StoryUnit(
        story_unit_id=f"{chapter.chapter_id}-{unit_id}", chapter_id=chapter.chapter_id,
        start=start, end=end, duration=round(end - start, 3), transcript_segment_ids=[int(item["id"]) for item in group],
        title=_title_from_text(text), core_idea=_first_sentence(text), hook_seed=_first_sentence(text)[:220],
        setup=setup, development=text, payoff=payoff, ending=_last_sentence(text),
        emotional_arc="rising_to_payoff" if payoff else "contained_statement", dominant_emotion=_dominant_emotion(text),
        speaker_context=", ".join(_speaker_ids(group)) or "primary_speaker",
        required_previous_context="" if context_dependency <= 0.45 else "Начало может зависеть от предыдущей фразы.",
        required_next_context="" if complete >= 0.6 else "Нужно следующее предложение для завершения мысли.",
        standalone_score=round(standalone, 3), completeness_score=round(complete, 3), clarity_score=round(clarity, 3),
        context_dependency_score=round(context_dependency, 3), information_density=round(information, 3),
        repetition_score=round(repetition, 3), transformation_potential=round(_bounded(standalone * 0.7 + information * 0.3), 3),
        publishability_precheck=publishable, content_signature=signature.to_dict(),
        confidence=round(_bounded(0.45 + complete * 0.35 + clarity * 0.20), 3),
        evidence={"segment_ids": [int(item["id"]) for item in group], "evidence_text": text[:1200]},
    )


def _content_signature(
    chapter_id: str, start: float, end: float, text: str, narrative: str, emotion: str,
) -> ContentSignature:
    keywords = _keywords(text)
    core = _normalise_idea(_first_sentence(text))
    entities = _entities(text)
    return ContentSignature(
        normalized_core_idea=core, topic_ids=[f"topic:{item}" for item in keywords[:5]], chapter_id=chapter_id,
        narrative_function=narrative, emotional_signature=emotion, key_entities=entities,
        key_claims=[_first_sentence(text)] if text else [], keyword_set=keywords,
        lexical_signature="|".join(keywords[:12]), semantic_embedding_ref=None,
        source_range={"start": round(start, 3), "end": round(end, 3)},
        transcript_fingerprint=stable_text_hash(_normalise_idea(text)),
    )


def _group_text(group: list[dict[str, Any]]) -> str:
    return " ".join(str(item["text"]).strip() for item in group).strip()


def _keywords(text: str) -> list[str]:
    counts = Counter(token for token in _tokens(text) if len(token) > 2 and token not in _STOP_WORDS)
    return [token for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]]


def _summary_from_text(text: str) -> str:
    return " ".join(_sentences(text)[:2])[:500] or text[:500]


def _title_from_text(text: str) -> str:
    return _first_sentence(text)[:120].rstrip(".,!?…") or "Смысловая часть"


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?…])\s+", text) if item.strip()]


def _first_sentence(text: str) -> str:
    return (_sentences(text) or [text.strip()])[0]


def _last_sentence(text: str) -> str:
    return (_sentences(text) or [text.strip()])[-1]


def _sentence_end(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…"))


def _narrative_function(text: str, is_first: bool, is_last: bool) -> str:
    lowered = text.casefold()
    if is_first and ("?" in text or "!" in text):
        return "hook"
    if any(term in lowered for term in ("итог", "вывод", "в конце", "conclusion")) or is_last:
        return "conclusion"
    if "?" in text:
        return "problem"
    if any(term in lowered for term in _PAYOFF_MARKERS):
        return "payoff"
    if any(term in lowered for term in ("например", "example", "к примеру")):
        return "example"
    if any(term in lowered for term in ("потому", "объясн", "because", "how ", "why ")):
        return "explanation"
    if any(term in lowered for term in _SETUP_MARKERS):
        return "setup"
    return "context"


def _dominant_emotion(text: str) -> str:
    lowered = text.casefold()
    if any(term in lowered for term in ("страх", "боюсь", "fear", "afraid")):
        return "tension"
    if "!" in text:
        return "emphatic"
    if "?" in text:
        return "curious"
    return "neutral"


def _emotional_intensity(text: str) -> float:
    return _bounded((text.count("!") * 0.18) + (text.count("?") * 0.12))


def _continuation_risk(segment: dict[str, Any], feature: dict[str, Any]) -> float:
    text = str(segment["text"]).strip().casefold()
    if not bool(feature.get("sentence_end", _sentence_end(text))):
        return 1.0
    if text.endswith(("и", "а", "но", "or", "and", "but")) or any(text.endswith(marker) for marker in ("если", "когда", "because", "if")):
        return 0.85
    return 0.15 if not any(marker in text for marker in _SETUP_MARKERS) else 0.45


def _visual_activity(start: float, end: float, scenes: dict[str, Any], visual_analysis: dict[str, Any]) -> float:
    span = max(1.0, end - start)
    boundary_count = sum(start <= float(item.get("timestamp", -1)) <= end for item in scenes.get("boundaries", []) if isinstance(item, dict))
    sample_count = sum(start <= float(item.get("timestamp", -1)) <= end for item in visual_analysis.get("samples", []) if isinstance(item, dict))
    return _bounded(boundary_count / max(1.0, span / 8.0) * 0.5 + sample_count / max(1.0, span / 4.0) * 0.1)


def _normalise_idea(text: str) -> str:
    return " ".join(_tokens(text))[:500]


def _entities(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b(?:[A-ZА-ЯЁ][a-zа-яё]{2,}|[A-Z]{2,})\b", text)))[:12]


@dataclass(slots=True)
class BoundaryPoint:
    timestamp: float
    boundary_type: str
    confidence: float
    supporting_signals: list[str]
    penalties: list[str]
    transcript_segment_id: int
    word_index: int | None
    scene_boundary_distance: float | None
    silence_before: float
    silence_after: float
    speaker_change: bool
    sentence_completion: bool
    semantic_completion_score: float
    continuation_probability: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SemanticBoundaryResolution:
    start: float
    end: float
    transcript_segment_ids: list[int]
    text: str
    diagnostics: dict[str, Any]


class SemanticBoundaryEngine:
    """Resolve safe source ranges; duration is deliberately a soft preference."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def resolve(
        self,
        story_unit: StoryUnit,
        transcript: dict[str, Any],
        transcript_features: dict[str, Any],
        scenes: dict[str, Any],
    ) -> SemanticBoundaryResolution:
        segments = [item for index, raw in enumerate(transcript.get("segments", [])) if (item := _valid_segment(raw, index))]
        segment_by_id = {int(item["id"]): item for item in segments}
        relevant = [segment_by_id[item] for item in story_unit.transcript_segment_ids if item in segment_by_id]
        if not relevant:
            return SemanticBoundaryResolution(
                story_unit.start, story_unit.end, list(story_unit.transcript_segment_ids), story_unit.development,
                _invalid_boundary_diagnostics("StoryUnit не имеет существующих transcript segment IDs."),
            )
        feature_rows = [item for item in transcript_features.get("segments", []) if isinstance(item, dict)]
        features = {int(segment["id"]): feature_rows[index] for index, segment in enumerate(segments) if index < len(feature_rows)}
        words = _timestamped_words(transcript, segments)
        source_duration = max(float(transcript.get("duration") or 0), float(segments[-1]["end"]))
        start_index = min(segments.index(item) for item in relevant)
        end_index = max(segments.index(item) for item in relevant)
        final_index, extension_reason = self._resolve_semantic_end(segments, end_index, features)
        final_segment = segments[final_index]
        start_segment = segments[start_index]
        start_word_position = _first_word_position(words, float(start_segment["start"]), float(start_segment["end"]))
        end_word_position = _last_word_position(words, float(final_segment["start"]), float(final_segment["end"]))
        start_word = words[start_word_position] if start_word_position is not None else None
        end_word = words[end_word_position] if end_word_position is not None else None
        start_time, head_padding, silence_before = self._resolve_head(start_word, words, start_word_position, start_segment)
        end_time, tail_padding, silence_after = self._resolve_tail(end_word, words, end_word_position, final_segment, source_duration)
        start_feature = features.get(int(start_segment["id"]), {})
        end_feature = features.get(int(final_segment["id"]), {})
        start_point = self._start_point(start_segment, start_feature, start_word, start_word_position, silence_before, scenes)
        end_point = self._end_point(final_segment, end_feature, end_word, end_word_position, silence_after, scenes, extension_reason)
        word_integrity = (
            (start_word is None or start_time <= float(start_word["start"]) + 0.001)
            and (end_word is None or end_time >= float(end_word["end"]) - 0.001)
        )
        sentence_integrity = start_point.sentence_completion and end_point.sentence_completion
        continuation_risk = end_point.continuation_probability
        payoff_preserved = not bool(story_unit.setup) or bool(story_unit.payoff) or end_point.sentence_completion
        valid = (
            start_point.boundary_type != "forbidden_start"
            and end_point.boundary_type != "forbidden_end"
            and word_integrity
            and sentence_integrity
            and continuation_risk <= float(self.settings.continuation_risk_threshold)
            and payoff_preserved
            and start_time < end_time
        )
        overall = _bounded(
            0.20 * start_point.confidence + 0.25 * end_point.confidence
            + 0.15 * float(word_integrity) + 0.15 * float(sentence_integrity)
            + 0.15 * (1 - continuation_risk) + 0.10 * float(payoff_preserved)
        )
        final_segments = segments[start_index:final_index + 1]
        diagnostics = {
            "schema_version": str(getattr(self.settings, "boundary_schema_version", "5A.1")),
            "requested_range": {"start": round(story_unit.start, 3), "end": round(story_unit.end, 3)},
            "resolved_range": {"start": round(start_time, 3), "end": round(end_time, 3)},
            "start_boundary": start_point.to_dict(), "end_boundary": end_point.to_dict(),
            "head_extension_seconds": round(max(0.0, story_unit.start - start_time), 3),
            "tail_extension_seconds": round(max(0.0, end_time - story_unit.end), 3),
            "head_padding_seconds": round(head_padding, 3), "tail_padding_seconds": round(tail_padding, 3),
            "word_integrity": word_integrity, "sentence_integrity": sentence_integrity,
            "semantic_completion": round(end_point.semantic_completion_score, 3),
            "context_independence": round(1 - story_unit.context_dependency_score, 3),
            "tail_naturalness": round(_tail_naturalness(tail_padding, silence_after, self.settings), 3),
            "head_naturalness": round(_head_naturalness(head_padding, silence_before, self.settings), 3),
            "payoff_preserved": payoff_preserved, "continuation_risk": round(continuation_risk, 3),
            "overall_boundary_score": round(overall, 3), "eligible": valid,
            "fallback_reason": "" if valid else _boundary_failure_reason(start_point, end_point, word_integrity, sentence_integrity, payoff_preserved),
            "semantic_extension_reason": extension_reason,
        }
        return SemanticBoundaryResolution(start_time, end_time, [int(item["id"]) for item in final_segments], _group_text(final_segments), diagnostics)

    def _resolve_semantic_end(
        self, segments: list[dict[str, Any]], end_index: int, features: dict[int, dict[str, Any]],
    ) -> tuple[int, str]:
        initial = segments[end_index]
        feature = features.get(int(initial["id"]), {})
        if _sentence_complete(initial, feature) and _continuation_risk(initial, feature) <= float(self.settings.continuation_risk_threshold):
            return end_index, "story_unit_complete"
        limit = float(initial["end"]) + float(self.settings.max_semantic_extension_seconds)
        for index in range(end_index + 1, len(segments)):
            candidate = segments[index]
            if float(candidate["end"]) > limit:
                break
            feature = features.get(int(candidate["id"]), {})
            if _sentence_complete(candidate, feature) and _continuation_risk(candidate, feature) <= float(self.settings.continuation_risk_threshold):
                return index, "extended_to_sentence_completion"
        if end_index == len(segments) - 1:
            # Legacy/fallback transcripts can omit punctuation even though no
            # later speech evidence exists.  Preserve the whole final word and
            # report the lower-confidence terminal fallback instead of silently
            # reverting to an arbitrary duration window.
            return end_index, "transcript_terminal_fallback_without_punctuation"
        return end_index, "completion_not_found_within_safe_extension"

    def _resolve_head(
        self, word: dict[str, Any] | None, words: list[dict[str, Any]], word_index: int | None,
        segment: dict[str, Any],
    ) -> tuple[float, float, float]:
        word_start = float(word["start"]) if word else float(segment["start"])
        previous_end = float(words[word_index - 1]["end"]) if word_index is not None and word_index > 0 else 0.0
        silence = max(0.0, word_start - previous_end)
        padding = min(float(self.settings.target_head_padding_seconds), float(self.settings.max_head_padding_seconds), silence)
        return round(word_start - padding, 3), padding, silence

    def _resolve_tail(
        self, word: dict[str, Any] | None, words: list[dict[str, Any]], word_index: int | None,
        segment: dict[str, Any], source_duration: float,
    ) -> tuple[float, float, float]:
        word_end = float(word["end"]) if word else float(segment["end"])
        next_start = float(words[word_index + 1]["start"]) if word_index is not None and word_index + 1 < len(words) else source_duration
        silence = max(0.0, min(source_duration, next_start) - word_end)
        padding = min(float(self.settings.target_tail_padding_seconds), float(self.settings.max_tail_padding_seconds), silence)
        return round(min(source_duration, word_end + padding), 3), padding, silence

    def _start_point(
        self, segment: dict[str, Any], feature: dict[str, Any], word: dict[str, Any] | None,
        word_index: int | None, silence_before: float, scenes: dict[str, Any],
    ) -> BoundaryPoint:
        complete = bool(feature.get("sentence_start", _sentence_start(segment)))
        dependent = _starts_dependent(str(segment["text"]))
        boundary_type = "strong_start" if complete and silence_before >= 0.1 else "acceptable_start" if complete else "forbidden_start" if dependent else "weak_start"
        signals = (["sentence_start"] if complete else []) + (["pause_before"] if silence_before >= 0.1 else [])
        penalties = ["dependent_clause"] if dependent else ([] if complete else ["mid_sentence"])
        return BoundaryPoint(
            timestamp=float(word["start"] if word else segment["start"]), boundary_type=boundary_type,
            confidence=0.95 if boundary_type == "strong_start" else 0.75 if boundary_type == "acceptable_start" else 0.25,
            supporting_signals=signals, penalties=penalties, transcript_segment_id=int(segment["id"]), word_index=word_index,
            scene_boundary_distance=_scene_distance(float(segment["start"]), scenes), silence_before=round(silence_before, 3),
            silence_after=0.0, speaker_change=False, sentence_completion=complete,
            semantic_completion_score=1.0 if complete else 0.25, continuation_probability=0.0 if complete else 0.7,
            reason="Начало полного первого слова на границе предложения." if complete else "Начало зависит от предыдущей фразы.",
        )

    def _end_point(
        self, segment: dict[str, Any], feature: dict[str, Any], word: dict[str, Any] | None,
        word_index: int | None, silence_after: float, scenes: dict[str, Any], extension_reason: str,
    ) -> BoundaryPoint:
        transcript_terminal_fallback = extension_reason == "transcript_terminal_fallback_without_punctuation"
        complete = _sentence_complete(segment, feature) or transcript_terminal_fallback
        continuation = 0.45 if transcript_terminal_fallback else _continuation_risk(segment, feature)
        boundary_type = "strong_end" if not transcript_terminal_fallback and complete and silence_after >= float(self.settings.min_tail_padding_seconds) else "acceptable_end" if complete else "forbidden_end"
        signals = (["sentence_completion"] if _sentence_complete(segment, feature) else []) + (["transcript_terminal"] if transcript_terminal_fallback else []) + (["silence_after"] if silence_after >= float(self.settings.min_tail_padding_seconds) else [])
        penalties = ["transcript_missing_terminal_punctuation"] if transcript_terminal_fallback else ([] if complete else ["unfinished_grammar_or_required_continuation"])
        return BoundaryPoint(
            timestamp=float(word["end"] if word else segment["end"]), boundary_type=boundary_type,
            confidence=0.96 if boundary_type == "strong_end" else 0.78 if boundary_type == "acceptable_end" else 0.15,
            supporting_signals=signals, penalties=penalties, transcript_segment_id=int(segment["id"]), word_index=word_index,
            scene_boundary_distance=_scene_distance(float(segment["end"]), scenes), silence_before=0.0,
            silence_after=round(silence_after, 3), speaker_change=False, sentence_completion=complete,
            semantic_completion_score=0.62 if transcript_terminal_fallback else 0.95 if complete else 0.10, continuation_probability=round(continuation, 3),
            reason="Конец полного последнего слова и завершённого предложения." if not transcript_terminal_fallback and complete else "Транскрипт заканчивается после полного последнего слова; применён безопасный fallback." if transcript_terminal_fallback else extension_reason,
        )


def generate_semantic_candidates(
    content_map_data: dict[str, Any], transcript: dict[str, Any], transcript_features: dict[str, Any],
    scenes: dict[str, Any], config: Any,
) -> tuple[list[Candidate], int]:
    """Turn StoryUnits into traceable candidate ranges without duration truncation."""

    content_map = GlobalContentMap.from_dict(content_map_data, transcript)
    engine = SemanticBoundaryEngine(config.content_understanding)
    candidates: list[Candidate] = []
    for unit in content_map.story_units:
        resolution = engine.resolve(unit, transcript, transcript_features, scenes)
        diagnostics = resolution.diagnostics
        candidate = Candidate(
            id=f"candidate-{unit.story_unit_id}", start=resolution.start, end=resolution.end,
            text=resolution.text, reason="SemanticBoundaryEngine: естественные границы StoryUnit.",
            transcript_segment_ids=resolution.transcript_segment_ids,
            start_boundary_reason=str(diagnostics.get("start_boundary", {}).get("reason", "")),
            end_boundary_reason=str(diagnostics.get("end_boundary", {}).get("reason", "")),
            feature_vector=candidate_transcript_features(resolution.start, resolution.end, transcript_features),
            explanations=["Кандидат построен из самостоятельной StoryUnit с проверенными границами."],
            chapter_id=unit.chapter_id, story_unit_id=unit.story_unit_id, core_idea=unit.core_idea,
            content_signature=dict(unit.content_signature), boundary_diagnostics=diagnostics,
        )
        candidates.append(candidate)
    return candidates, len(candidates)


def _timestamped_words(transcript: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(transcript.get("segments", [])):
        segment = _valid_segment(raw, index)
        if segment is None:
            continue
        words = raw.get("words", []) if isinstance(raw, dict) else []
        for word in words if isinstance(words, list) else []:
            if not isinstance(word, dict):
                continue
            try:
                start, end = float(word.get("start")), float(word.get("end"))
            except (TypeError, ValueError):
                continue
            if start < end:
                rows.append({"start": start, "end": end, "text": str(word.get("text") or word.get("word") or "").strip(), "segment_id": int(segment["id"])})
    if not rows:
        for segment in segments:
            rows.append({"start": float(segment["start"]), "end": float(segment["end"]), "text": str(segment["text"]), "segment_id": int(segment["id"])})
    return sorted(rows, key=lambda item: (float(item["start"]), float(item["end"])))


def _first_word_position(words: list[dict[str, Any]], start: float, end: float) -> int | None:
    return next((index for index, word in enumerate(words) if float(word["end"]) > start - 0.001 and float(word["start"]) < end + 0.001), None)


def _last_word_position(words: list[dict[str, Any]], start: float, end: float) -> int | None:
    positions = [index for index, word in enumerate(words) if float(word["end"]) > start - 0.001 and float(word["start"]) < end + 0.001]
    return positions[-1] if positions else None


def _sentence_complete(segment: dict[str, Any], feature: dict[str, Any]) -> bool:
    return bool(feature.get("sentence_end", _sentence_end(str(segment["text"]))))


def _sentence_start(segment: dict[str, Any]) -> bool:
    text = str(segment["text"]).strip()
    return bool(text and text[0].isupper() and not _starts_dependent(text))


def _starts_dependent(text: str) -> bool:
    return text.strip().casefold().startswith(("и ", "а ", "но ", "потому что ", "and ", "but ", "so ", "because "))


def _scene_distance(timestamp: float, scenes: dict[str, Any]) -> float | None:
    points = [abs(float(item.get("timestamp", timestamp)) - timestamp) for item in scenes.get("boundaries", []) if isinstance(item, dict)]
    return round(min(points), 3) if points else None


def _tail_naturalness(padding: float, silence: float, settings: Any) -> float:
    if silence <= 0:
        return 0.25
    return _bounded(0.55 + min(0.45, padding / max(float(settings.min_tail_padding_seconds), 0.01) * 0.2))


def _head_naturalness(padding: float, silence: float, settings: Any) -> float:
    if silence <= 0:
        return 0.55
    return _bounded(0.65 + min(0.35, padding / max(float(settings.target_head_padding_seconds), 0.01) * 0.2))


def _invalid_boundary_diagnostics(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "5A.1", "eligible": False, "fallback_reason": reason,
        "word_integrity": False, "sentence_integrity": False, "overall_boundary_score": 0.0,
    }


def _boundary_failure_reason(
    start: BoundaryPoint, end: BoundaryPoint, word_integrity: bool, sentence_integrity: bool, payoff_preserved: bool,
) -> str:
    if not word_integrity:
        return "Boundary нарушает целостность первого или последнего слова."
    if start.boundary_type == "forbidden_start":
        return "Начало зависит от предыдущей незавершённой фразы."
    if end.boundary_type == "forbidden_end" or not sentence_integrity:
        return "Конец не завершает предложение или требует продолжения."
    if not payoff_preserved:
        return "Граница отбрасывает обязательный payoff StoryUnit."
    return "Boundary не прошла безопасный semantic contract."
