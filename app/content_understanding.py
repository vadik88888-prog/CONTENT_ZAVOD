"""Deterministic, source-scoped content understanding for Goal 5A.

The module deliberately keeps language interpretation separate from pipeline
decisions.  It produces validated, grounded artifacts from transcript and
existing media signals; later stages may enrich them with structured AI
proposals, but can always fall back to this local implementation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from app.content_profile_taxonomy import (
    CONTENT_PROFILE_SCHEMA_VERSION,
    LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS,
    PROFILE_AXIS_ORDER,
    order_profile_ids,
    profile_value_ids,
    unknown_fallback,
    user_override_ids,
)
from app.diversity import (
    DIVERSITY_DECISION_SCHEMA_VERSION,
    DiversityDecision,
    DiversityExclusion,
    DiversitySelection,
    DiversitySimilarity,
    interval_metrics,
    is_temporal_duplicate,
    transcript_similarity,
)
from app.models import Candidate, ScoredCandidate
from app.multimodal_evidence import evidence_for_range, validate_multimodal_timeline
from app.production_models import BoundaryDecision
from app.transcript_features import candidate_transcript_features
from app.utils import stable_text_hash


VIDEO_CONTENT_PROFILE_SCHEMA_VERSION = CONTENT_PROFILE_SCHEMA_VERSION
LEGACY_VIDEO_CONTENT_PROFILE_SCHEMA_VERSION = LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS[0]
CONTENT_STRATEGY_VERSION = "5A.4"
GLOBAL_CONTENT_MAP_SCHEMA_VERSION = "5A.1"
STORY_UNIT_SCHEMA_VERSION = "5A.1"
BOUNDARY_DECISION_SCHEMA_VERSION = "5C.1"

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
# Compatibility aliases for callers that imported the pre-registry names.  The
# values are projections of the canonical registry, never independent lists.
PROFILE_FORMATS = frozenset(profile_value_ids("format"))
PROFILE_EDITORIAL_MODES = frozenset(profile_value_ids("editorial_mode"))
PROFILE_DOMAINS = frozenset(profile_value_ids("domain"))
PROFILE_TRAITS = frozenset(profile_value_ids("traits"))

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
    detected_profile: dict[str, Any] = field(default_factory=dict)
    effective_profile: dict[str, Any] = field(default_factory=dict)
    manual_override: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != VIDEO_CONTENT_PROFILE_SCHEMA_VERSION:
            raise ValueError("Unsupported VideoContentProfile schema version.")
        if not self.source_id or self.source_duration_seconds < 0:
            raise ValueError("VideoContentProfile requires a source id and non-negative duration.")
        if self.detected_content_type not in CONTENT_TYPES:
            raise ValueError("Unsupported VideoContentProfile content type.")
        if self.dominant_format not in DOMINANT_FORMATS:
            raise ValueError("Unsupported VideoContentProfile dominant format.")
        _validate_detected_profile(self.detected_profile)
        _validate_effective_profile(self.effective_profile)
        _validate_manual_override(self.manual_override)
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
        migrated = _migrate_legacy_profile(data)
        profile = cls(
            schema_version=str(migrated.get("schema_version", "")),
            source_id=str(migrated.get("source_id", "")),
            source_duration_seconds=float(migrated.get("source_duration_seconds", 0)),
            language=str(migrated.get("language") or "unknown"),
            detected_content_type=str(migrated.get("detected_content_type") or "unknown"),
            content_type_confidence=float(migrated.get("content_type_confidence", 0)),
            secondary_content_types=[str(item) for item in migrated.get("secondary_content_types", [])],
            dominant_format=str(migrated.get("dominant_format") or "unknown"),
            speaker_count_estimate=int(migrated.get("speaker_count_estimate", 0)),
            dialogue_style=str(migrated.get("dialogue_style") or "unknown"),
            narrative_style=str(migrated.get("narrative_style") or "unknown"),
            pacing_profile=str(migrated.get("pacing_profile") or "unknown"),
            emotional_curve_summary=str(migrated.get("emotional_curve_summary") or "unknown"),
            visual_density=float(migrated.get("visual_density", 0)),
            speech_density=float(migrated.get("speech_density", 0)),
            useful_content_density=float(migrated.get("useful_content_density", 0)),
            repetition_level=float(migrated.get("repetition_level", 0)),
            recommended_short_strategy=str(migrated.get("recommended_short_strategy") or "generic_fallback"),
            recommended_clip_duration_range=dict(migrated.get("recommended_clip_duration_range", {})),
            estimated_story_count=int(migrated.get("estimated_story_count", 0)),
            estimated_publishable_clip_range=dict(migrated.get("estimated_publishable_clip_range", {})),
            analysis_confidence=float(migrated.get("analysis_confidence", 0)),
            warnings=[str(item) for item in migrated.get("warnings", [])],
            strategy_id=str(migrated.get("strategy_id") or "generic_fallback"),
            fallback_used=bool(migrated.get("fallback_used", True)),
            evidence=dict(migrated.get("evidence", {})),
            detected_profile=dict(migrated.get("detected_profile", {})),
            effective_profile=dict(migrated.get("effective_profile", {})),
            manual_override=dict(migrated.get("manual_override", {})),
        )
        profile.validate()
        return profile


def validate_video_content_profile(data: dict[str, Any], *, expected_source_id: str | None = None) -> None:
    profile = VideoContentProfile.from_dict(data)
    if expected_source_id is not None and profile.source_id != expected_source_id:
        raise ValueError("VideoContentProfile source_id does not match the active source.")


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
    multimodal_evidence: dict[str, Any] = field(default_factory=dict)

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
            multimodal_evidence=dict(data.get("multimodal_evidence", {})),
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
            if unit.multimodal_evidence:
                interval = unit.multimodal_evidence.get("interval", {})
                if (
                    unit.multimodal_evidence.get("source_id") != self.source_id
                    or abs(float(interval.get("start_seconds", -1)) - unit.start) > 0.01
                    or abs(float(interval.get("end_seconds", -1)) - unit.end) > 0.01
                ):
                    raise ValueError("StoryUnit multimodal evidence must match its source range.")

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

    def detect_chapters(self, segments: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any) -> list[list[dict[str, Any]]]:
        ...

    def build_story_units(self, chapter: ContentChapter, segments: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any) -> list[StoryUnit]:
        ...

    def recommend_duration(self, profile: VideoContentProfile) -> dict[str, float]:
        ...

    def evaluate_standalone(self, story_unit: StoryUnit) -> float:
        ...

    def resolve_boundaries(self, story_unit: StoryUnit, transcript: dict[str, Any], transcript_features: dict[str, Any], scenes: dict[str, Any], settings: Any) -> "SemanticBoundaryResolution":
        ...

    def estimate_clip_count(self, content_map: "GlobalContentMap", profile: VideoContentProfile, requested_count: int) -> dict[str, Any]:
        ...

    def coverage_dimensions(self) -> tuple[str, ...]:
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
        speech_density = _speech_density(tokens, duration, feature_segments)
        repetition = _average(feature_segments, "repetition_score")
        filler = _average(feature_segments, "filler_word_ratio")
        visual_density = _visual_density(duration, scenes, visual_analysis)
        useful_density = _bounded(speech_density * (1 - filler) * (1 - repetition * 0.45))
        emotion = _emotional_summary(text, audio_features)
        pacing = _pacing(tokens, duration)
        filename = str(source.get("display_name") or source.get("path") or "")
        detected_profile = _detect_profile_axes(
            text=text,
            filename=filename,
            speaker_count=speaker_count,
            speech_density=speech_density,
            visual_density=visual_density,
            repetition=repetition,
            pacing=pacing,
            emotional_summary=emotion,
            scenes=scenes,
            visual_analysis=visual_analysis,
        )
        settings = config.content_understanding
        manual_override = _normalise_manual_override(getattr(settings, "manual_override", {}))
        effective_profile = _resolve_effective_profile(
            detected_profile,
            manual_override,
            min_confidence=float(getattr(settings, "profile_detection_min_confidence", 0.45)),
        )
        content_type, confidence, secondary = _legacy_content_type_projection(detected_profile, effective_profile)
        dominant_format = _legacy_format_projection(effective_profile["format"])
        strategy = _strategy_id(content_type, dominant_format)
        profile_fallback_used = any(
            value == "safe_fallback" for value in effective_profile["resolution"].values()
        )
        estimated_stories = _preliminary_story_count(duration, useful_density, repetition, raw_segments)
        clip_range = _clip_range(estimated_stories, useful_density, repetition)
        warnings: list[str] = []
        if not raw_segments:
            warnings.append("Транскрипт пуст: применён безопасный общий fallback.")
        if speaker_count == 0:
            warnings.append("Не удалось оценить число говорящих по исходному транскрипту.")
        if duration <= 0:
            warnings.append("Длительность источника недоступна; профиль имеет пониженную уверенность.")
        if str(getattr(settings, "profile_schema_version", VIDEO_CONTENT_PROFILE_SCHEMA_VERSION)) != VIDEO_CONTENT_PROFILE_SCHEMA_VERSION:
            warnings.append("Конфигурация профиля 5A.1 совместимо обновлена до схемы 5A.2.")
        analysis_confidence = _bounded(
            (0.45 if raw_segments else 0.05)
            + min(0.25, len(tokens) / 800)
            + (0.15 if feature_segments else 0.0)
            + (0.10 if duration > 0 else 0.0)
            + (0.05 if _visual_evidence_available(visual_analysis) else 0.0)
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
            fallback_used=profile_fallback_used,
            evidence={
                "transcript_segment_count": len(raw_segments),
                "word_count": len(tokens),
                "speaker_ids": _speaker_ids(raw_segments),
                "scene_boundary_count": len(scenes.get("boundaries", [])),
                "filename_signal_used": bool(detected_profile.get("provenance", {}).get("filename_signal_used")),
                "profile_axes": list(PROFILE_AXIS_ORDER),
            },
            detected_profile=detected_profile,
            effective_profile=effective_profile,
            manual_override=manual_override,
        )
        profile.validate()
        return profile

    def detect_chapters(self, segments: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any) -> list[list[dict[str, Any]]]:
        return _chapter_groups(segments, features, settings)

    def build_story_units(self, chapter: ContentChapter, segments: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any) -> list[StoryUnit]:
        return _make_story_units(chapter, segments, features, settings)

    def recommend_duration(self, profile: VideoContentProfile) -> dict[str, float]:
        return dict(profile.recommended_clip_duration_range)

    def evaluate_standalone(self, story_unit: StoryUnit) -> float:
        return story_unit.standalone_score

    def resolve_boundaries(self, story_unit: StoryUnit, transcript: dict[str, Any], transcript_features: dict[str, Any], scenes: dict[str, Any], settings: Any) -> "SemanticBoundaryResolution":
        return SemanticBoundaryEngine(settings).resolve(story_unit, transcript, transcript_features, scenes)

    def estimate_clip_count(self, content_map: "GlobalContentMap", profile: VideoContentProfile, requested_count: int) -> dict[str, Any]:
        return recommend_clip_count(content_map.to_dict(), profile.to_dict(), requested_count)

    def coverage_dimensions(self) -> tuple[str, ...]:
        return ("temporal", "chapter", "topic", "story_unit", "emotional", "speaker", "narrative_function")


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

    enabled = bool(getattr(config.content_understanding, "enabled", True))
    profile = DeterministicContentStrategy().build_profile(
        source if enabled else {"id": source.get("id")},
        metadata,
        transcript if enabled else {"source_id": transcript.get("source_id"), "language": transcript.get("language"), "segments": []},
        transcript_features if enabled else {"segments": []},
        audio_features if enabled else {},
        scenes if enabled else {},
        visual_analysis if enabled else {},
        config,
    )
    profile.evidence["detection_enabled"] = enabled
    if not enabled:
        profile.warnings.append("Автоопределение профиля отключено; применён manual override или безопасный fallback.")
    return profile.to_dict()


def _tokens(text: str) -> list[str]:
    return [item.casefold() for item in _WORD_RE.findall(text)]


def _speaker_ids(segments: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("speaker_id") or item.get("speaker") or "").strip() for item in segments if str(item.get("speaker_id") or item.get("speaker") or "").strip()})


def _speaker_count(segments: list[dict[str, Any]]) -> int:
    identifiers = _speaker_ids(segments)
    return len(identifiers) if identifiers else (1 if segments else 0)


def _axis(value: str, confidence: float, evidence: list[str]) -> dict[str, Any]:
    return {"value": value, "confidence": round(_bounded(confidence), 3), "evidence": evidence}


def _keyword_score(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def _detect_profile_axes(
    *, text: str, filename: str, speaker_count: int, speech_density: float,
    visual_density: float, repetition: float, pacing: str, emotional_summary: str,
    scenes: dict[str, Any], visual_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Return evidence-bearing proposals; executable consumers use only the validated effective profile."""

    lowered = text.casefold()
    filename_lower = filename.casefold()
    visual_blob = json.dumps(visual_analysis, ensure_ascii=False).casefold() if isinstance(visual_analysis, dict) else ""
    scene_count = len(scenes.get("boundaries", []))
    filename_signal_used = False

    gameplay_terms = ("gameplay", "геймплей", "матч", "катка", "pubg", "minecraft", "fortnite")
    screen_terms = ("экран", "интерфейс", "нажмите", "click", "screen", "dashboard", "код")
    mixed_terms = (
        "стрим", "stream", "livestream", "реакц", "reaction", "reacts",
        "обзор", "product review", "tech review", "unboxing", "распаковк",
    )
    scene_driven_terms = (
        "влог", "vlog", "путешеств", "travel", "trip", "туризм",
        "рецепт", "recipe", "cooking", "фильм", "сериал", "movie", "series",
        "спорт", "фитнес", "workout", "fitness",
    )
    gameplay_text = _keyword_score(lowered, gameplay_terms)
    gameplay_file = _keyword_score(filename_lower, gameplay_terms)
    screen_text = _keyword_score(lowered, screen_terms)
    screen_file = _keyword_score(filename_lower, screen_terms)
    mixed_text = _keyword_score(lowered, mixed_terms)
    mixed_file = _keyword_score(filename_lower, mixed_terms)
    scene_driven_text = _keyword_score(lowered, scene_driven_terms)
    scene_driven_file = _keyword_score(filename_lower, scene_driven_terms)
    gameplay_visual = any(term in visual_blob for term in ("gameplay", "game ui", "video game"))
    screen_visual = any(term in visual_blob for term in ("screen recording", "software ui", "computer screen"))
    mixed_visual = any(term in visual_blob for term in ("reaction video", "livestream", "product review"))
    scene_driven_visual = any(term in visual_blob for term in ("vlog", "travel", "cooking", "movie", "sports", "workout"))
    if gameplay_text or gameplay_file or gameplay_visual:
        filename_signal_used = bool(gameplay_file)
        format_axis = _axis(
            "gameplay",
            (0.62 + min(0.25, gameplay_text * 0.08) if gameplay_text or gameplay_visual else 0.3)
            + (0.04 if gameplay_file else 0),
            (["transcript:gameplay_terms"] if gameplay_text else [])
            + (["filename:weak_gameplay_hint"] if gameplay_file else [])
            + (["visual:gameplay_label"] if gameplay_visual else []),
        )
    elif screen_text or screen_file or screen_visual:
        filename_signal_used = bool(screen_file)
        format_axis = _axis(
            "screen_demo",
            (0.58 + min(0.28, screen_text * 0.08) if screen_text or screen_visual else 0.3)
            + (0.04 if screen_file else 0),
            (["transcript:screen_terms"] if screen_text else [])
            + (["filename:weak_screen_hint"] if screen_file else [])
            + (["visual:screen_label"] if screen_visual else []),
        )
    elif mixed_text or mixed_file or mixed_visual:
        filename_signal_used = bool(mixed_file)
        format_axis = _axis(
            "mixed",
            (0.58 + min(0.24, mixed_text * 0.08) if mixed_text or mixed_visual else 0.3)
            + (0.04 if mixed_file else 0),
            (["transcript:mixed_format_terms"] if mixed_text else [])
            + (["filename:weak_mixed_format_hint"] if mixed_file else [])
            + (["visual:mixed_format_label"] if mixed_visual else []),
        )
    elif scene_driven_text or scene_driven_file or scene_driven_visual:
        filename_signal_used = bool(scene_driven_file)
        format_axis = _axis(
            "scene_driven",
            (0.58 + min(0.24, scene_driven_text * 0.08) if scene_driven_text or scene_driven_visual else 0.3)
            + (0.04 if scene_driven_file else 0),
            (["transcript:scene_driven_terms"] if scene_driven_text else [])
            + (["filename:weak_scene_driven_hint"] if scene_driven_file else [])
            + (["visual:scene_driven_label"] if scene_driven_visual else []),
        )
    elif speaker_count >= 2:
        format_axis = _axis("dialogue", 0.82, ["transcript:multiple_speakers"])
    elif speaker_count == 1 and text.strip():
        format_axis = _axis("talking_head", 0.68, ["transcript:single_speaker", "transcript:speech_present"])
    elif scene_count >= 8 or visual_density >= 0.55:
        format_axis = _axis("scene_driven", 0.62, ["scenes:high_visual_activity"])
    else:
        format_axis = _axis(unknown_fallback("format"), 0.2, ["fallback:insufficient_format_evidence"])

    editorial_terms = {
        "interview": _DIALOGUE_TERMS,
        "motivational": _MOTIVATIONAL_TERMS,
        "explanatory": _EDUCATIONAL_TERMS,
        "commentary": ("обзор", "реакц", "комментар", "review", "reaction", "commentary"),
        "demonstration": ("покажу", "смотрите", "делаем", "нажмите", "рецепт", "тренировк", "demo", "demonstrat", "tutorial", "workout", "step by step"),
        "news_analysis": ("новост", "сегодня", "событи", "аналитик", "news", "breaking", "report"),
        "narrative": ("однажды", "история", "случилось", "сначала", "потом", "влог", "путешеств", "story", "once", "then", "vlog", "travel"),
        "entertainment": ("смешн", "шут", "прикол", "развлеч", "фильм", "сериал", "funny", "joke", "entertain", "movie", "series"),
    }
    editorial_scores = {name: _keyword_score(lowered, terms) for name, terms in editorial_terms.items()}
    if speaker_count >= 2:
        editorial_scores["interview"] += 2
    editorial_mode, editorial_score = max(editorial_scores.items(), key=lambda item: (item[1], item[0]))
    if editorial_score <= 0:
        enough_speech = len(_tokens(text)) >= 6
        editorial_mode = "commentary" if enough_speech else unknown_fallback("editorial_mode")
        editorial_confidence = 0.52 if enough_speech else 0.2
        editorial_evidence = ["transcript:speech_present"] if enough_speech else ["fallback:insufficient_editorial_evidence"]
    else:
        editorial_confidence = min(0.93, 0.48 + editorial_score * 0.11)
        editorial_evidence = [f"transcript:{editorial_mode}_terms"]
        if editorial_mode == "interview" and speaker_count >= 2:
            editorial_evidence.append("transcript:multiple_speakers")
    editorial_axis = _axis(editorial_mode, editorial_confidence, editorial_evidence)

    domain_terms = {
        "gaming": gameplay_terms,
        "technology": ("технолог", "программ", "компьют", "нейросет", "код", "software", "tech", " ai "),
        "education": ("обуч", "урок", "учеб", "экзамен", "learn", "lesson", "course"),
        "business": ("бизнес", "компан", "продаж", "клиент", "market", "business", "startup"),
        "finance": ("деньг", "инвест", "кредит", "банк", "акци", "finance", "invest", "stock"),
        "food": ("еда", "рецепт", "готов", "кухн", "блюд", "food", "recipe", "cook"),
        "health": ("здоров", "врач", "лечен", "трениров", "спорт", "фитнес", "health", "doctor", "fitness", "sport", "workout"),
        "news": ("новост", "репортаж", "событи", "news", "breaking"),
        "lifestyle": ("влог", "путешеств", "туризм", "дом", "семь", "vlog", "travel", "trip", "journey", "lifestyle"),
        "entertainment": ("фильм", "сериал", "музык", "шоу", "реакц", "movie", "series", "music", "show", "reaction"),
    }
    domain_scores = {name: _keyword_score(lowered, terms) for name, terms in domain_terms.items()}
    filename_domain_scores = {name: _keyword_score(filename_lower, terms) for name, terms in domain_terms.items()}
    domain, domain_score = max(domain_scores.items(), key=lambda item: (item[1], item[0]))
    filename_domain = max(filename_domain_scores.items(), key=lambda item: (item[1], item[0]))
    if domain_score <= 0 and filename_domain[1] > 0:
        domain, domain_score = filename_domain
        filename_signal_used = True
        domain_axis = _axis(domain, 0.3 + min(0.08, domain_score * 0.03), [f"filename:weak_{domain}_hint"])
    elif domain_score > 0:
        domain_axis = _axis(domain, min(0.9, 0.5 + domain_score * 0.1), [f"transcript:{domain}_terms"])
    else:
        domain_axis = _axis("general" if text.strip() else unknown_fallback("domain"), 0.45 if text.strip() else 0.2,
                            ["fallback:general_spoken_content"] if text.strip() else ["fallback:insufficient_domain_evidence"])

    traits: list[dict[str, Any]] = []
    def add_trait(value: str, confidence: float, evidence: str) -> None:
        traits.append(_axis(value, confidence, [evidence]))
    if speech_density >= 0.3:
        add_trait("speech_led", 0.75, "transcript:speech_density")
    if visual_density >= 0.5:
        add_trait("visual_led", 0.68, "scenes:visual_density")
    if speaker_count == 1:
        add_trait("single_speaker", 0.9, "transcript:speaker_ids")
    elif speaker_count >= 2:
        add_trait("multi_speaker", 0.9, "transcript:speaker_ids")
    if editorial_mode == "interview":
        add_trait("question_answer", editorial_confidence, "transcript:question_answer_terms")
    if pacing == "fast":
        add_trait("high_pacing", 0.72, "transcript:words_per_second")
    elif pacing == "slow":
        add_trait("low_pacing", 0.72, "transcript:words_per_second")
    if "пики" in emotional_summary:
        add_trait("high_emotion", 0.66, "audio:energy_frames_or_punctuation")
    if speech_density >= 0.58:
        add_trait("dense_information", 0.65, "transcript:speech_density")
    if repetition >= 0.45:
        add_trait("repetitive", 0.7, "transcript:repetition_score")
    if format_axis["value"] == "screen_demo":
        add_trait("screen_content", format_axis["confidence"], "profile:format")
    if format_axis["value"] in {"scene_driven", "gameplay"}:
        add_trait("scene_driven", format_axis["confidence"], "profile:format")
    if editorial_mode in {"explanatory", "demonstration"}:
        add_trait("instructional", editorial_confidence, "profile:editorial_mode")
    return {
        "format": format_axis,
        "editorial_mode": editorial_axis,
        "domain": domain_axis,
        "traits": traits,
        "provenance": {"filename_signal_used": filename_signal_used, "detector": CONTENT_STRATEGY_VERSION},
    }


def _normalise_manual_override(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    override = {
        "format": raw.get("format") or None,
        "editorial_mode": raw.get("editorial_mode") or None,
        "domain": raw.get("domain") or None,
        "traits": list(order_profile_ids("traits", [str(item) for item in raw.get("traits", [])]))
        if isinstance(raw.get("traits", []), list) else [],
    }
    active = any(override[name] for name in PROFILE_AXIS_ORDER[:-1]) or bool(override["traits"])
    override["provenance"] = "user" if active else "none"
    override["revision_id"] = stable_text_hash(json.dumps(override, ensure_ascii=False, sort_keys=True))[:16] if active else None
    _validate_manual_override(override)
    return override


def _resolve_effective_profile(
    detected: dict[str, Any], manual_override: dict[str, Any], *, min_confidence: float,
) -> dict[str, Any]:
    effective: dict[str, Any] = {"resolution": {}}
    for name in PROFILE_AXIS_ORDER[:-1]:
        override_value = manual_override.get(name)
        proposal = detected[name]
        if override_value:
            effective[name] = override_value
            effective["resolution"][name] = "manual_override"
        elif float(proposal["confidence"]) >= min_confidence:
            effective[name] = proposal["value"]
            effective["resolution"][name] = "detected"
        else:
            effective[name] = unknown_fallback(name)
            effective["resolution"][name] = "safe_fallback"
    if manual_override.get("traits"):
        effective["traits"] = list(manual_override["traits"])
        effective["resolution"]["traits"] = "manual_override"
    else:
        effective["traits"] = list(order_profile_ids("traits", {
            str(item["value"]) for item in detected.get("traits", [])
            if float(item.get("confidence", 0)) >= min_confidence
        }))
        effective["resolution"]["traits"] = "detected"
    _validate_effective_profile(effective)
    return effective


def _legacy_content_type_projection(
    detected: dict[str, Any], effective: dict[str, Any],
) -> tuple[str, float, list[str]]:
    editorial = effective["editorial_mode"]
    format_value = effective["format"]
    domain = effective["domain"]
    content_type = {
        "interview": "interview", "motivational": "motivational", "explanatory": "educational",
        "demonstration": "tutorial", "news_analysis": "news_or_analysis", "narrative": "movie_or_series",
        "entertainment": "commentary", "commentary": "commentary",
    }.get(editorial, "unknown")
    if format_value == "gameplay" or domain == "gaming":
        content_type = "gameplay"
    elif domain == "entertainment" and editorial == "narrative":
        content_type = "movie_or_series"
    confidences = [
        float(detected[name]["confidence"]) for name in PROFILE_AXIS_ORDER[:-1]
        if effective["resolution"][name] == "detected"
    ]
    confidence = sum(confidences) / len(confidences) if confidences else (1.0 if "manual_override" in effective["resolution"].values() else 0.2)
    return content_type, _bounded(confidence), []


def _legacy_format_projection(value: str) -> str:
    return {
        "talking_head": "single_speaker_monologue", "dialogue": "multi_speaker_dialogue",
        "screen_demo": "screen_recording", "gameplay": "gameplay_commentary",
        "scene_driven": "scene_driven", "mixed": "mixed", "unknown": "unknown",
    }[value]


def _validate_detected_profile(profile: dict[str, Any]) -> None:
    if set(profile) != {*PROFILE_AXIS_ORDER, "provenance"}:
        raise ValueError("VideoContentProfile detected_profile has an invalid shape.")
    for name in PROFILE_AXIS_ORDER[:-1]:
        proposal = profile.get(name)
        if not isinstance(proposal, dict) or proposal.get("value") not in profile_value_ids(name):
            raise ValueError(f"VideoContentProfile detected {name} is invalid.")
        if not 0 <= float(proposal.get("confidence", -1)) <= 1 or not isinstance(proposal.get("evidence"), list):
            raise ValueError(f"VideoContentProfile detected {name} evidence is invalid.")
    traits = profile.get("traits")
    if not isinstance(traits, list) or any(
        not isinstance(item, dict) or item.get("value") not in profile_value_ids("traits")
        or not 0 <= float(item.get("confidence", -1)) <= 1 or not isinstance(item.get("evidence"), list)
        for item in traits
    ):
        raise ValueError("VideoContentProfile detected traits are invalid.")
    if not isinstance(profile.get("provenance"), dict):
        raise ValueError("VideoContentProfile detected provenance is invalid.")


def _validate_effective_profile(profile: dict[str, Any]) -> None:
    if any(profile.get(axis_id) not in profile_value_ids(axis_id) for axis_id in PROFILE_AXIS_ORDER[:-1]):
        raise ValueError("VideoContentProfile effective axes are invalid.")
    if not isinstance(profile.get("traits"), list) or any(item not in profile_value_ids("traits") for item in profile["traits"]):
        raise ValueError("VideoContentProfile effective traits are invalid.")
    resolution = profile.get("resolution")
    if not isinstance(resolution, dict) or set(resolution) != set(PROFILE_AXIS_ORDER):
        raise ValueError("VideoContentProfile effective resolution is invalid.")
    if any(value not in {"detected", "manual_override", "safe_fallback", "legacy_migration"} for value in resolution.values()):
        raise ValueError("VideoContentProfile effective resolution provenance is invalid.")


def _validate_manual_override(override: dict[str, Any]) -> None:
    if override.get("format") not in frozenset(user_override_ids("format")) | {None}:
        raise ValueError("Unsupported manual profile format override.")
    if override.get("editorial_mode") not in frozenset(user_override_ids("editorial_mode")) | {None}:
        raise ValueError("Unsupported manual profile editorial_mode override.")
    if override.get("domain") not in frozenset(user_override_ids("domain")) | {None}:
        raise ValueError("Unsupported manual profile domain override.")
    if not isinstance(override.get("traits"), list) or any(item not in user_override_ids("traits") for item in override["traits"]):
        raise ValueError("Unsupported manual profile traits override.")
    if override.get("provenance") not in {"none", "user"}:
        raise ValueError("Unsupported manual profile override provenance.")


def _migrate_legacy_profile(data: dict[str, Any]) -> dict[str, Any]:
    if str(data.get("schema_version")) != LEGACY_VIDEO_CONTENT_PROFILE_SCHEMA_VERSION:
        return dict(data)
    migrated = dict(data)
    format_value = {
        "single_speaker_monologue": "talking_head", "multi_speaker_dialogue": "dialogue",
        "host_guest": "dialogue", "screen_recording": "screen_demo",
        "gameplay_commentary": "gameplay", "narrated_visual": "scene_driven",
        "scene_driven": "scene_driven", "mixed": "mixed",
    }.get(str(data.get("dominant_format")), unknown_fallback("format"))
    editorial_mode = {
        "interview": "interview", "motivational": "motivational", "educational": "explanatory",
        "lecture": "explanatory", "tutorial": "demonstration", "commentary": "commentary",
        "news_or_analysis": "news_analysis", "movie_or_series": "narrative",
    }.get(str(data.get("detected_content_type")), unknown_fallback("editorial_mode"))
    confidence = _bounded(float(data.get("content_type_confidence", 0)))
    migrated["schema_version"] = VIDEO_CONTENT_PROFILE_SCHEMA_VERSION
    migrated["detected_profile"] = {
        "format": _axis(format_value, confidence, ["migration:legacy_dominant_format"]),
        "editorial_mode": _axis(editorial_mode, confidence, ["migration:legacy_content_type"]),
        "domain": _axis("gaming" if data.get("detected_content_type") == "gameplay" else unknown_fallback("domain"), confidence, ["migration:legacy_content_type"]),
        "traits": [],
        "provenance": {
            "filename_signal_used": bool(
                (data.get("evidence") if isinstance(data.get("evidence"), dict) else {}).get("filename_signal_used")
            ),
            "detector": "legacy_5A.1",
        },
    }
    migrated["effective_profile"] = {
        "format": format_value, "editorial_mode": editorial_mode,
        "domain": "gaming" if data.get("detected_content_type") == "gameplay" else unknown_fallback("domain"),
        "traits": [],
        "resolution": {name: "legacy_migration" for name in PROFILE_AXIS_ORDER},
    }
    migrated["manual_override"] = _normalise_manual_override({})
    return migrated


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
    if not isinstance(visual_analysis, dict):
        sampled = 0
    else:
        sampled = int(visual_analysis.get("sample_count") or len(visual_analysis.get("subject_keyframes", [])))
    return _bounded(min(1.0, per_minute / 12.0 + min(0.2, sampled / 100.0)))


def _visual_evidence_available(visual_analysis: dict[str, Any]) -> bool:
    return bool(
        isinstance(visual_analysis, dict)
        and str(visual_analysis.get("evidence_status") or visual_analysis.get("status") or "").lower()
        not in {"", "skipped", "unavailable", "failed"}
        and (visual_analysis.get("sample_count") or visual_analysis.get("subject_keyframes"))
    )


def _emotional_summary(text: str, audio_features: dict[str, Any]) -> str:
    punctuation = text.count("!") + text.count("?")
    energy = _average(
        [item for item in audio_features.get("energy_frames", []) if isinstance(item, dict)],
        "normalized_loudness",
    )
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
    multimodal_timeline: dict[str, Any] | None = None,
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
            evidence={"transcript_segment_count": 0, "strategy_id": "evidence_driven"},
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
    if multimodal_timeline is not None:
        validate_multimodal_timeline(multimodal_timeline, expected_source_id=profile.source_id)
        for unit in story_units:
            unit.multimodal_evidence = evidence_for_range(multimodal_timeline, unit.start, unit.end)
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
            "strategy_id": "evidence_driven",
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
        # A pause is meaningful only after the current chapter has enough
        # material to form a publishable StoryUnit.  Otherwise a sparse
        # transcript turns every sentence into a chapter and leaves the final
        # selector with sub-minimum "clips".
        or (
            pause >= float(settings.chapter_pause_seconds)
            and terminal
            and current_duration >= float(settings.min_story_unit_seconds)
        )
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
            units.append(_make_story_unit(f"story-{len(units) + 1:03d}", chapter, current, features, settings))
            current = []
    if current:
        units.append(_make_story_unit(f"story-{len(units) + 1:03d}", chapter, current, features, settings))
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
    unit_id: str, chapter: ContentChapter, group: list[dict[str, Any]], features: dict[int, dict[str, Any]], settings: Any,
) -> StoryUnit:
    start, end = float(group[0]["start"]), float(group[-1]["end"])
    duration = end - start
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
    publishable = (
        duration >= float(settings.min_story_unit_seconds)
        and standalone >= 0.55
        and complete >= 0.6
        and context_dependency <= 0.6
        and bool(text.strip())
    )
    return StoryUnit(
        story_unit_id=f"{chapter.chapter_id}-{unit_id}", chapter_id=chapter.chapter_id,
        start=start, end=end, duration=round(duration, 3), transcript_segment_ids=[int(item["id"]) for item in group],
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
    # A question must normally carry its answer/payoff forward.  The boundary
    # resolver will seek the next complete segment instead of ending on it.
    if text.endswith("?"):
        return 0.75
    if text.endswith(("и", "а", "но", "or", "and", "but")) or any(text.endswith(marker) for marker in ("если", "когда", "because", "if")):
        return 0.85
    return 0.15 if not any(marker in text for marker in _SETUP_MARKERS) else 0.45


def _speaker_changed(first: dict[str, Any] | None, second: dict[str, Any] | None) -> bool:
    if first is None or second is None:
        return False
    first_speaker = str(first.get("speaker_id") or "").strip()
    second_speaker = str(second.get("speaker_id") or "").strip()
    return bool(first_speaker and second_speaker and first_speaker != second_speaker)


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
        previous_segment = segments[start_index - 1] if start_index > 0 else None
        next_segment = segments[final_index + 1] if final_index + 1 < len(segments) else None
        start_word_position = _first_word_position(words, float(start_segment["start"]), float(start_segment["end"]))
        end_word_position = _last_word_position(words, float(final_segment["start"]), float(final_segment["end"]))
        start_word = words[start_word_position] if start_word_position is not None else None
        end_word = words[end_word_position] if end_word_position is not None else None
        start_time, head_padding, silence_before = self._resolve_head(start_word, words, start_word_position, start_segment)
        end_time, tail_padding, silence_after = self._resolve_tail(end_word, words, end_word_position, final_segment, source_duration)
        start_feature = features.get(int(start_segment["id"]), {})
        end_feature = features.get(int(final_segment["id"]), {})
        start_point = self._start_point(
            start_segment, start_feature, start_word, start_word_position, silence_before, scenes, previous_segment,
        )
        end_point = self._end_point(
            final_segment, end_feature, end_word, end_word_position, silence_after, scenes, extension_reason, next_segment,
        )
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
        allowed_range = {"start_seconds": round(start_time, 3), "end_seconds": round(end_time, 3)}
        hook_range = {
            "start_seconds": round(float(start_word["start"] if start_word else start_segment["start"]), 3),
            "end_seconds": round(min(float(start_segment["end"]), end_time), 3),
        }
        completion_range = {
            "start_seconds": round(max(float(final_segment["start"]), start_time), 3),
            "end_seconds": round(float(end_word["end"] if end_word else final_segment["end"]), 3),
        }
        safe_start_points = sorted({
            round(point, 3)
            for point in [start_time, *[float(item["start"]) for item in final_segments], *[float(item["start"]) for item in words]]
            if start_time - 0.001 <= point <= end_time + 0.001
        })
        safe_end_points = sorted({
            round(point, 3)
            for point in [end_time, *[float(item["end"]) for item in final_segments], *[float(item["end"]) for item in words]]
            if start_time - 0.001 <= point <= end_time + 0.001
        })
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
            "continuation_risk_threshold": round(float(self.settings.continuation_risk_threshold), 3),
            "overall_boundary_score": round(overall, 3), "eligible": valid,
            "fallback_reason": "" if valid else _boundary_failure_reason(start_point, end_point, word_integrity, sentence_integrity, payoff_preserved),
            "semantic_extension_reason": extension_reason,
            "pause_evidence": {
                "head_silence_seconds": round(silence_before, 3),
                "tail_silence_seconds": round(silence_after, 3),
                "pre_roll_seconds": round(head_padding, 3),
                "post_roll_seconds": round(tail_padding, 3),
                "intentional_pause_preserved": bool(head_padding > 0 or tail_padding > 0),
            },
            "question_context": {
                "start_is_question": str(start_segment["text"]).strip().endswith("?"),
                "end_is_question": str(final_segment["text"]).strip().endswith("?"),
                "answer_or_completion_included": not str(final_segment["text"]).strip().endswith("?"),
            },
            "allowed_source_range": allowed_range,
            "required_evidence": [
                {
                    "requirement_type": "hook", "required": bool(story_unit.hook_seed.strip()),
                    "source_range": hook_range, "transcript_segment_id": int(start_segment["id"]),
                    "reason": "Hook must remain in the final source dialogue.",
                    "evidence": {"text": story_unit.hook_seed, "boundary_reason": start_point.reason},
                },
                {
                    "requirement_type": "completion", "required": True,
                    "source_range": completion_range, "transcript_segment_id": int(final_segment["id"]),
                    "reason": "The final complete thought must remain in the final source dialogue.",
                    "evidence": {"text": story_unit.ending, "boundary_reason": end_point.reason},
                },
                {
                    "requirement_type": "payoff", "required": bool(story_unit.payoff.strip()),
                    "source_range": completion_range, "transcript_segment_id": int(final_segment["id"]),
                    "reason": "Detected payoff must remain in the final source dialogue.",
                    "evidence": {"text": story_unit.payoff, "boundary_reason": end_point.reason},
                },
            ],
            "safe_start_points": safe_start_points,
            "safe_end_points": safe_end_points,
        }
        # Candidate ids are deterministically derived from StoryUnit ids in the
        # existing pipeline, so the engine can emit the durable decision at the
        # same moment it resolves the source boundary.
        diagnostics["boundary_decision"] = _boundary_decision_payload(
            f"candidate-{story_unit.story_unit_id}", diagnostics,
        )
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
        if end_index == len(segments) - 1 and not _sentence_complete(initial, feature):
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
        word_index: int | None, silence_before: float, scenes: dict[str, Any], previous_segment: dict[str, Any] | None,
    ) -> BoundaryPoint:
        complete = bool(feature.get("sentence_start", _sentence_start(segment)))
        dependent = _starts_dependent(str(segment["text"]))
        speaker_change = _speaker_changed(previous_segment, segment)
        scene_distance = _scene_distance(float(segment["start"]), scenes)
        scene_aligned = scene_distance is not None and scene_distance <= 0.25
        boundary_type = "strong_start" if complete and (silence_before >= 0.1 or speaker_change) else "acceptable_start" if complete else "forbidden_start" if dependent else "weak_start"
        signals = (["sentence_start"] if complete else []) + (["pause_before"] if silence_before >= 0.1 else []) + (["speaker_change"] if speaker_change else []) + (["scene_boundary_nearby"] if scene_aligned else [])
        penalties = ["dependent_clause"] if dependent else ([] if complete else ["mid_sentence"])
        base_confidence = 0.95 if boundary_type == "strong_start" else 0.75 if boundary_type == "acceptable_start" else 0.25
        return BoundaryPoint(
            timestamp=float(word["start"] if word else segment["start"]), boundary_type=boundary_type,
            confidence=_bounded(base_confidence + (0.025 if speaker_change else 0.0) + (0.025 if scene_aligned else 0.0)),
            supporting_signals=signals, penalties=penalties, transcript_segment_id=int(segment["id"]), word_index=word_index,
            scene_boundary_distance=scene_distance, silence_before=round(silence_before, 3),
            silence_after=0.0, speaker_change=speaker_change, sentence_completion=complete,
            semantic_completion_score=1.0 if complete else 0.25, continuation_probability=0.0 if complete else 0.7,
            reason="Начало полного первого слова на границе предложения." if complete else "Начало зависит от предыдущей фразы.",
        )

    def _end_point(
        self, segment: dict[str, Any], feature: dict[str, Any], word: dict[str, Any] | None,
        word_index: int | None, silence_after: float, scenes: dict[str, Any], extension_reason: str,
        next_segment: dict[str, Any] | None,
    ) -> BoundaryPoint:
        transcript_terminal_fallback = extension_reason == "transcript_terminal_fallback_without_punctuation"
        complete = _sentence_complete(segment, feature) or transcript_terminal_fallback
        continuation = 0.45 if transcript_terminal_fallback else _continuation_risk(segment, feature)
        boundary_type = "strong_end" if not transcript_terminal_fallback and complete and silence_after >= float(self.settings.min_tail_padding_seconds) else "acceptable_end" if complete else "forbidden_end"
        speaker_change = _speaker_changed(segment, next_segment)
        scene_distance = _scene_distance(float(segment["end"]), scenes)
        scene_aligned = scene_distance is not None and scene_distance <= 0.25
        signals = (["sentence_completion"] if _sentence_complete(segment, feature) else []) + (["transcript_terminal"] if transcript_terminal_fallback else []) + (["silence_after"] if silence_after >= float(self.settings.min_tail_padding_seconds) else []) + (["speaker_change_after"] if speaker_change else []) + (["scene_boundary_nearby"] if scene_aligned else [])
        penalties = ["transcript_missing_terminal_punctuation"] if transcript_terminal_fallback else ([] if complete else ["unfinished_grammar_or_required_continuation"])
        base_confidence = 0.96 if boundary_type == "strong_end" else 0.78 if boundary_type == "acceptable_end" else 0.15
        return BoundaryPoint(
            timestamp=float(word["end"] if word else segment["end"]), boundary_type=boundary_type,
            confidence=_bounded(base_confidence + (0.025 if speaker_change else 0.0) + (0.025 if scene_aligned else 0.0)),
            supporting_signals=signals, penalties=penalties, transcript_segment_id=int(segment["id"]), word_index=word_index,
            scene_boundary_distance=scene_distance, silence_before=0.0,
            silence_after=round(silence_after, 3), speaker_change=speaker_change, sentence_completion=complete,
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
        candidates.append(build_semantic_candidate(
            [unit], transcript, transcript_features, scenes, engine,
            candidate_id=f"candidate-{unit.story_unit_id}",
        ))
    return candidates, len(candidates)


def build_semantic_candidate(
    units: list[StoryUnit],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    scenes: dict[str, Any],
    engine: SemanticBoundaryEngine,
    *,
    candidate_id: str,
) -> Candidate:
    """Resolve one or more adjacent StoryUnits through the existing boundary engine."""

    if not units or any(unit.chapter_id != units[0].chapter_id for unit in units):
        raise ValueError("A semantic candidate requires adjacent StoryUnits from one chapter.")
    ordered = sorted(units, key=lambda item: (item.start, item.end, item.story_unit_id))
    primary = max(ordered, key=lambda item: (item.standalone_score, item.completeness_score, item.story_unit_id))
    boundary_unit = ordered[0] if len(ordered) == 1 else _combined_story_unit(ordered, primary)
    resolution = engine.resolve(boundary_unit, transcript, transcript_features, scenes)
    diagnostics = resolution.diagnostics
    boundary_decision = diagnostics.get("boundary_decision")
    if _has_complete_boundary_evidence(diagnostics) and (
        not isinstance(boundary_decision, dict) or boundary_decision.get("candidate_id") != candidate_id
    ):
        diagnostics = {
            **diagnostics,
            "boundary_decision": _boundary_decision_payload(candidate_id, diagnostics),
        }
    story_unit_ids = [unit.story_unit_id for unit in ordered]
    return Candidate(
        id=candidate_id, start=resolution.start, end=resolution.end,
        text=resolution.text, reason="SemanticBoundaryEngine: естественные границы StoryUnit.",
        transcript_segment_ids=resolution.transcript_segment_ids,
        start_boundary_reason=str(diagnostics.get("start_boundary", {}).get("reason", "")),
        end_boundary_reason=str(diagnostics.get("end_boundary", {}).get("reason", "")),
        feature_vector=candidate_transcript_features(resolution.start, resolution.end, transcript_features),
        explanations=[
            "Кандидат построен из самостоятельной StoryUnit с проверенными границами."
            if len(ordered) == 1 else
            "Кандидат построен из связанных StoryUnits с проверенными семантическими границами."
        ],
        chapter_id=primary.chapter_id, story_unit_id=primary.story_unit_id, story_unit_ids=story_unit_ids,
        core_idea=primary.core_idea, content_signature=dict(primary.content_signature),
        boundary_diagnostics=diagnostics,
        semantic_evidence={
            "story_unit_id": primary.story_unit_id,
            "story_unit_ids": story_unit_ids,
            "hook": ordered[0].hook_seed,
            "payoff": ordered[-1].payoff,
            "setup": ordered[0].setup,
            "ending": ordered[-1].ending,
            "completeness_score": max(unit.completeness_score for unit in ordered),
            "context_dependency_score": min(unit.context_dependency_score for unit in ordered),
            "information_density": max(unit.information_density for unit in ordered),
            "evidence": (
                ordered[0].evidence if len(ordered) == 1
                else {unit.story_unit_id: unit.evidence for unit in ordered}
            ),
        },
    )


def _combined_story_unit(units: list[StoryUnit], primary: StoryUnit) -> StoryUnit:
    first, last = units[0], units[-1]
    segment_ids = [segment_id for unit in units for segment_id in unit.transcript_segment_ids]
    development = " ".join(unit.development.strip() for unit in units if unit.development.strip())
    signature = dict(primary.content_signature)
    signature["source_range"] = {"start": first.start, "end": last.end}
    signature["transcript_fingerprint"] = stable_text_hash("|".join(str(value) for value in segment_ids))
    count = float(len(units))
    return StoryUnit(
        story_unit_id="+".join(unit.story_unit_id for unit in units),
        chapter_id=first.chapter_id,
        start=first.start,
        end=last.end,
        duration=round(last.end - first.start, 3),
        transcript_segment_ids=segment_ids,
        title=primary.title,
        core_idea=primary.core_idea,
        hook_seed=first.hook_seed,
        setup=first.setup,
        development=development,
        payoff=last.payoff or last.ending,
        ending=last.ending,
        emotional_arc="multimodal_linked_sequence",
        dominant_emotion=primary.dominant_emotion,
        speaker_context=", ".join(dict.fromkeys(unit.speaker_context for unit in units)),
        required_previous_context=first.required_previous_context,
        required_next_context=last.required_next_context,
        standalone_score=sum(unit.standalone_score for unit in units) / count,
        completeness_score=max(unit.completeness_score for unit in units),
        clarity_score=sum(unit.clarity_score for unit in units) / count,
        context_dependency_score=min(unit.context_dependency_score for unit in units),
        information_density=max(unit.information_density for unit in units),
        repetition_score=sum(unit.repetition_score for unit in units) / count,
        transformation_potential=max(unit.transformation_potential for unit in units),
        publishability_precheck=any(unit.publishability_precheck for unit in units),
        content_signature=signature,
        confidence=min(unit.confidence for unit in units),
        evidence={"story_unit_ids": [unit.story_unit_id for unit in units], "evidence_text": development[:1200]},
        multimodal_evidence={},
    )


def _has_complete_boundary_evidence(diagnostics: dict[str, Any]) -> bool:
    return all(
        isinstance(diagnostics.get(name), dict)
        for name in ("requested_range", "resolved_range", "allowed_source_range", "start_boundary", "end_boundary")
    )


def ensure_candidate_boundary_decision(candidate: Candidate) -> dict[str, Any] | None:
    """Validate or deterministically promote complete legacy 5A diagnostics.

    Old AnalysisArtifacts persisted the same word/sentence boundary evidence
    used by 5C, but not the typed BoundaryDecision wrapper.  A Draft may reuse
    that evidence only when the legacy diagnostics are complete and internally
    valid; otherwise production must stop before transformation.
    """

    diagnostics = dict(candidate.boundary_diagnostics or {})
    existing = diagnostics.get("boundary_decision")
    if isinstance(existing, dict):
        try:
            decision = BoundaryDecision.model_validate(existing)
        except ValueError:
            return None
        if decision.candidate_id != candidate.id:
            return None
        if (
            abs(decision.refined_range.start_seconds - candidate.start) > 0.001
            or abs(decision.refined_range.end_seconds - candidate.end) > 0.001
        ):
            return None
        return decision.model_dump(mode="json")

    required = ("requested_range", "resolved_range", "start_boundary", "end_boundary")
    if not all(isinstance(diagnostics.get(name), dict) for name in required):
        return None
    try:
        resolved_start = float(diagnostics["resolved_range"]["start"])
        resolved_end = float(diagnostics["resolved_range"]["end"])
        start_boundary = diagnostics["start_boundary"]
        end_boundary = diagnostics["end_boundary"]
        start_timestamp = max(resolved_start, min(resolved_end, float(start_boundary["timestamp"])))
        end_timestamp = max(resolved_start, min(resolved_end, float(end_boundary["timestamp"])))
        start_segment_id = int(start_boundary["transcript_segment_id"])
        end_segment_id = int(end_boundary["transcript_segment_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if resolved_end <= resolved_start:
        return None

    # Padding belongs to the allowed candidate range, not to the required
    # spoken evidence.  Bind the migrated hook/completion requirements to a
    # tiny interval at the proven first/last transcript boundary; otherwise
    # pre-roll or post-roll silence would be impossible for dialogue mappings
    # to cover.
    hook_start = start_timestamp
    hook_end = min(resolved_end, start_timestamp + 0.001)
    completion_start = max(resolved_start, end_timestamp - 0.001)
    completion_end = end_timestamp
    diagnostics.update({
        "allowed_source_range": {
            "start_seconds": resolved_start,
            "end_seconds": resolved_end,
        },
        "pause_evidence": {
            "head_silence_seconds": float(start_boundary.get("silence_before", 0) or 0),
            "tail_silence_seconds": float(end_boundary.get("silence_after", 0) or 0),
            "pre_roll_seconds": float(diagnostics.get("head_padding_seconds", 0) or 0),
            "post_roll_seconds": float(diagnostics.get("tail_padding_seconds", 0) or 0),
            "intentional_pause_preserved": bool(
                float(diagnostics.get("head_padding_seconds", 0) or 0)
                or float(diagnostics.get("tail_padding_seconds", 0) or 0)
            ),
        },
        "question_context": {
            "start_is_question": False,
            "end_is_question": False,
            "answer_or_completion_included": bool(diagnostics.get("sentence_integrity", False)),
        },
        "required_evidence": [
            {
                "requirement_type": "hook", "required": True,
                "source_range": {"start_seconds": hook_start, "end_seconds": hook_end},
                "transcript_segment_id": start_segment_id,
                "reason": "Legacy 5A opening boundary evidence must survive production.",
                "evidence": dict(start_boundary),
            },
            {
                "requirement_type": "completion", "required": True,
                "source_range": {"start_seconds": completion_start, "end_seconds": completion_end},
                "transcript_segment_id": end_segment_id,
                "reason": "Legacy 5A completion boundary evidence must survive production.",
                "evidence": dict(end_boundary),
            },
            {
                "requirement_type": "payoff", "required": bool(diagnostics.get("payoff_preserved", False)),
                "source_range": {"start_seconds": completion_start, "end_seconds": completion_end},
                "transcript_segment_id": end_segment_id,
                "reason": "Legacy 5A payoff evidence must survive production.",
                "evidence": dict(end_boundary),
            },
        ],
        "safe_start_points": [resolved_start],
        "safe_end_points": [resolved_end],
    })
    try:
        decision = BoundaryDecision.model_validate(
            _boundary_decision_payload(candidate.id, diagnostics)
        )
    except ValueError:
        return None
    payload = decision.model_dump(mode="json")
    candidate.boundary_diagnostics = {
        **diagnostics,
        "boundary_decision": payload,
        "boundary_decision_migration": "legacy_5A_evidence_promoted_at_draft_preflight",
    }
    return payload


def _boundary_decision_payload(candidate_id: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Promote existing 5A diagnostics into the persisted 5C decision shape."""

    requested = diagnostics["requested_range"]
    resolved = diagnostics["resolved_range"]
    allowed = diagnostics["allowed_source_range"]
    fingerprint = stable_text_hash(json.dumps(
        {"candidate_id": candidate_id, "decision_inputs": diagnostics},
        ensure_ascii=False,
        sort_keys=True,
    ))[:12]
    fallback_reason = str(diagnostics.get("fallback_reason") or "").strip() or None
    extension_reason = str(diagnostics.get("semantic_extension_reason") or "")
    return {
        "schema_version": BOUNDARY_DECISION_SCHEMA_VERSION,
        "decision_id": f"boundary-{candidate_id}-{fingerprint}",
        "candidate_id": candidate_id,
        "rough_range": {
            "start_seconds": float(requested["start"]),
            "end_seconds": float(requested["end"]),
        },
        "refined_range": {
            "start_seconds": float(resolved["start"]),
            "end_seconds": float(resolved["end"]),
        },
        "allowed_source_range": dict(allowed),
        "start_reason": str(diagnostics["start_boundary"].get("reason") or "resolved_start_boundary"),
        "end_reason": str(diagnostics["end_boundary"].get("reason") or "resolved_end_boundary"),
        "word_integrity": bool(diagnostics.get("word_integrity", False)),
        "sentence_integrity": bool(diagnostics.get("sentence_integrity", False)),
        "semantic_completion": float(diagnostics.get("semantic_completion", 0)) >= 0.6,
        "payoff_preserved": bool(diagnostics.get("payoff_preserved", False)),
        "continuation_risk": float(diagnostics.get("continuation_risk", 1)),
        "continuation_risk_threshold": float(diagnostics.get("continuation_risk_threshold", 0.65)),
        "pre_roll_seconds": float(diagnostics.get("head_padding_seconds", 0)),
        "post_roll_seconds": float(diagnostics.get("tail_padding_seconds", 0)),
        "confidence": float(diagnostics.get("overall_boundary_score", 0)),
        "start_evidence": dict(diagnostics["start_boundary"]),
        "end_evidence": dict(diagnostics["end_boundary"]),
        "pause_evidence": dict(diagnostics.get("pause_evidence", {})),
        "question_context": dict(diagnostics.get("question_context", {})),
        "required_evidence": [dict(item) for item in diagnostics.get("required_evidence", []) if isinstance(item, dict)],
        "safe_start_points": [float(item) for item in diagnostics.get("safe_start_points", [])],
        "safe_end_points": [float(item) for item in diagnostics.get("safe_end_points", [])],
        "fallback_used": extension_reason not in {"", "story_unit_complete"},
        "fallback_reason": fallback_reason,
    }


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


def editorial_intent_affinity(candidate: Candidate, story: StoryUnit | None, intent: str) -> dict[str, Any]:
    """Bounded lexical relevance used only after eligibility and boundary gates have passed."""

    intent_tokens = {item for item in _tokens(intent) if len(item) > 2 and item not in _STOP_WORDS}
    evidence_text = " ".join(filter(None, (
        candidate.text,
        story.title if story else "",
        story.core_idea if story else "",
        story.hook_seed if story else "",
        story.payoff if story else "",
    )))
    evidence_tokens = set(_tokens(evidence_text))
    matched = sorted(intent_tokens & evidence_tokens)
    affinity = len(matched) / len(intent_tokens) if intent_tokens else 0.0
    return {
        "intent_present": bool(intent_tokens),
        "affinity": round(_bounded(affinity), 6),
        "matched_terms": matched[:12],
        "policy": "post_eligibility_ranking_only",
    }


def select_with_coverage(
    scored: list[ScoredCandidate], config: Any, content_map_data: dict[str, Any],
    *, production_feasibility: dict[str, Any] | None = None,
    content_profile: dict[str, Any] | None = None,
) -> tuple[list[ScoredCandidate], dict[str, Any]]:
    """Select strong StoryUnits with existing coverage plus deterministic MMR diversity."""

    content_map = GlobalContentMap.from_dict(content_map_data)
    settings = config.content_understanding
    weights = settings.coverage_weights
    stories = {item.story_unit_id: item for item in content_map.story_units}
    selected: list[ScoredCandidate] = []
    rejected: dict[str, DiversityExclusion] = {}
    selections: list[DiversitySelection] = []
    requested = min(config.max_clips, config.ai_reranking.final_clip_count)
    diversity_lambda = float(settings.diversity_lambda)
    from app.production_feasibility import production_feasibility_index
    from app.editorial_profile_policy import editorial_decision_from_candidate, evaluate_editorial_candidate

    feasibility_by_id = production_feasibility_index(production_feasibility)
    allow_ranked_replacements = bool(
        isinstance(production_feasibility, dict)
        and production_feasibility.get("allow_ranked_replacements")
    )
    editorial_by_id = {}
    for item in scored:
        decision = (
            evaluate_editorial_candidate(
                item.candidate,
                content_profile,
                score=float(item.score),
                production_feasibility=feasibility_by_id.get(item.candidate.id),
            )
            if content_profile is not None
            else editorial_decision_from_candidate(item.candidate)
            or evaluate_editorial_candidate(
                item.candidate,
                None,
                score=float(item.score),
                production_feasibility=feasibility_by_id.get(item.candidate.id),
            )
        )
        item.candidate.editorial_decision = decision
        editorial_by_id[item.candidate.id] = decision
    eligible_for_similarity = [
        item for item in scored
        if editorial_by_id[item.candidate.id].selectable
        and feasibility_by_id.get(item.candidate.id, {}).get("status") != "GUARANTEED_BLOCKED"
    ]
    similarities, similarity_index = _eligible_diversity_similarities(eligible_for_similarity, stories)

    def reject(item: ScoredCandidate, reason_code: str, reason: str, *, against: ScoredCandidate | None = None,
               similarity: DiversitySimilarity | None = None) -> None:
        rejected[item.candidate.id] = DiversityExclusion(
            candidate_id=item.candidate.id,
            reason_code=reason_code,
            reason=reason,
            against_candidate_id=against.candidate.id if against is not None else None,
            max_similarity=similarity.composite_similarity if similarity is not None else None,
            similarity=similarity,
        )

    for item in scored:
        boundary = item.candidate.boundary_diagnostics
        decision = editorial_by_id[item.candidate.id]
        story = stories.get(str(item.candidate.story_unit_id or ""))
        strong_story = bool(
            story
            and story.publishability_precheck
            and story.standalone_score >= settings.strong_story_unit_threshold
        )
        virality_ready = bool(
            getattr(getattr(config, "virality", None), "enabled", False)
            and item.virality.get("selection_eligible", False)
            and item.score >= float(getattr(config.virality, "minimum_quality_score", 0.52)) * 100
        )
        if not decision.selectable:
            codes = list(decision.hard_blockers)
            reject(
                item,
                "EDITORIAL_INTEGRITY_NOT_PASSED",
                f"Structural/technical policy rejected candidate: codes={','.join(codes)}.",
            )
        elif feasibility_by_id.get(item.candidate.id, {}).get("status") == "GUARANTEED_BLOCKED":
            feasibility = feasibility_by_id[item.candidate.id]
            reject(
                item,
                "PRODUCTION_FEASIBILITY_BLOCKED",
                str(
                    feasibility.get("reason")
                    or "Guaranteed blocked by provider-free production feasibility."
                ),
            )
        elif boundary and not bool(boundary.get("eligible", False)):
            reject(
                item,
                "BOUNDARY_REJECTED",
                str(boundary.get("fallback_reason") or "Semantic boundary не прошла validation."),
            )
        elif not item.selected and not allow_ranked_replacements:
            reject(item, "BASE_SELECTION_REJECTED", item.rejection_reason or "Не прошёл базовый quality ranking.")
        elif item.candidate.duration < float(config.min_clip_duration):
            reject(
                item,
                "DURATION_BELOW_MINIMUM",
                (
                    f"Длительность {item.candidate.duration:.2f} с меньше минимальных "
                    f"{float(config.min_clip_duration):.2f} с."
                ),
            )
        elif item.score < config.score_threshold and not virality_ready and (
            not strong_story or item.score < settings.coverage_min_quality_score
        ):
            reject(item, "QUALITY_BELOW_THRESHOLD", "Оценка ниже порога.")
    while len(selected) < requested:
        ranked: list[tuple[float, float, str, ScoredCandidate, dict[str, Any], ScoredCandidate | None, DiversitySimilarity | None]] = []
        for item in scored:
            if item.candidate.id in rejected or item in selected:
                continue
            temporal_duplicate = _coverage_temporal_duplicate(item, selected, config)
            if temporal_duplicate is not None:
                chosen, reason = temporal_duplicate
                similarity = _diversity_similarity_for(item, chosen, similarity_index)
                reject(item, "TEMPORAL_DUPLICATE", reason, against=chosen, similarity=similarity)
                continue
            max_similarity, similar_to, similarity = _max_selected_diversity_similarity(item, selected, similarity_index)
            if similarity is not None and max_similarity >= float(settings.semantic_duplicate_threshold):
                reject(
                    item,
                    "SEMANTIC_DUPLICATE",
                    (
                        f"Семантически повторяет кандидата {similar_to.candidate.id} "
                        f"(similarity={max_similarity:.3f})."
                    ),
                    against=similar_to,
                    similarity=similarity,
                )
                continue
            details = _coverage_increment(item, selected, stories, config)
            story = stories.get(str(item.candidate.story_unit_id or ""))
            standalone = story.standalone_score if story else 0.0
            completeness = story.completeness_score if story else item.completeness_score / 100
            context_dependency = story.context_dependency_score if story else item.context_dependency_score / 100
            repetition = story.repetition_score if story else float(item.candidate.feature_vector.get("repetition_score", 0))
            boundary_score = float(item.candidate.boundary_diagnostics.get("overall_boundary_score", 0.65))
            intent_details = editorial_intent_affinity(
                item.candidate,
                story,
                str(getattr(settings, "editorial_intent", "") or ""),
            )
            score = (
                float(item.virality.get("ranking_sort_score", item.score / 100)) * weights["base_quality"]
                + standalone * weights["standalone"]
                + completeness * weights["completeness"]
                + boundary_score * weights["boundary"]
                + details["incremental_coverage_score"] * weights["incremental_coverage"]
                + details["new_chapter"] * weights["chapter_diversity"]
                + details["new_topic_ratio"] * weights["topic_diversity"]
                + details["new_emotion"] * weights["emotional_diversity"]
                + details["new_temporal_region"] * weights["temporal_diversity"]
                - details["semantic_duplicate_similarity"] * weights["semantic_duplicate_penalty"]
                - context_dependency * weights["context_dependency_penalty"]
                - repetition * weights["repetition_penalty"]
                + intent_details["affinity"] * float(getattr(settings, "editorial_intent_weight", 0.08))
            )
            details["editorial_intent"] = intent_details
            details["coverage_selection_score"] = round(score, 6)
            mmr_score = diversity_lambda * score - (1 - diversity_lambda) * max_similarity
            details["diversity"] = {
                "reason_code": "SELECTED_MMR",
                "lambda": round(diversity_lambda, 6),
                "max_similarity": round(max_similarity, 6),
                "against_candidate_id": similar_to.candidate.id if similar_to is not None else None,
                "mmr_score": round(mmr_score, 6),
                "similarity": similarity.to_dict() if similarity is not None else None,
            }
            ranked.append((mmr_score, score, item.candidate.id, item, details, similar_to, similarity))
        if not ranked:
            break
        mmr_score, coverage_score, _candidate_id, best, details, similar_to, similarity = max(
            ranked, key=lambda item: (item[0], item[1], item[2]),
        )
        best.candidate.incremental_coverage_score = float(details["incremental_coverage_score"])
        best.selection_reason = "Выбран: качество, coverage и semantic diversity подтверждены MMR-политикой."
        best.selection_diagnostics = {
            "decision": "accepted_coverage",
            **details,
            "production_feasibility": feasibility_by_id.get(best.candidate.id),
        }
        selected.append(best)
        selections.append(DiversitySelection(
            candidate_id=best.candidate.id,
            coverage_quality_score=coverage_score,
            max_similarity=float(details["diversity"]["max_similarity"]),
            against_candidate_id=similar_to.candidate.id if similar_to is not None else None,
            mmr_score=mmr_score,
            similarity=similarity,
        ))
    selected_ids = {item.candidate.id for item in selected}
    for item in scored:
        if item.candidate.id in selected_ids:
            item.selected = True
            continue
        item.selected = False
        exclusion = rejected.get(item.candidate.id)
        if exclusion is None:
            exclusion = DiversityExclusion(
                candidate_id=item.candidate.id,
                reason_code="LIMIT_REACHED",
                reason="Не вошёл в лимит после coverage-aware selection.",
            )
            rejected[item.candidate.id] = exclusion
        item.selection_reason = exclusion.reason
        item.selection_diagnostics = {
            "decision": "rejected_coverage",
            "reason": exclusion.reason,
            "reason_code": exclusion.reason_code,
            "diversity": exclusion.to_dict(),
            "production_feasibility": feasibility_by_id.get(item.candidate.id),
        }
    score_order = {
        item.candidate.id: position
        for position, item in enumerate(
            sorted(scored, key=lambda value: (-value.score, value.candidate.id)), start=1,
        )
    }
    selection_order = {item.candidate.id: position for position, item in enumerate(selected, start=1)}
    for item in scored:
        quality = item.candidate.candidate_score_v2
        score_diagnostics = quality.diagnostics if quality is not None else {}
        contributions = dict(score_diagnostics.get("positive_contributions") or {})
        strongest = sorted(contributions.items(), key=lambda value: (-float(value[1]), value[0]))[:4]
        item.selection_diagnostics["ranking"] = {
            "code_owned_final_score": item.score,
            "quality_rank": score_order[item.candidate.id],
            "selection_rank": selection_order.get(item.candidate.id),
            "strongest_factor_contributions": [
                {"factor": name, "points": round(float(points), 3)} for name, points in strongest
            ],
            "context_debt_deduction": score_diagnostics.get("context_debt_deduction", 0),
            "penalty_total": score_diagnostics.get("penalty_total", 0),
            "placement_reason": (
                "selected_by_existing_mmr_after_code_owned_quality_and_eligibility"
                if item.candidate.id in selection_order
                else str(item.selection_diagnostics.get("reason_code") or "not_selected")
            ),
            "ai_final_selection_used": False,
        }
    coverage = build_coverage_map(content_map_data, scored, selected, config)
    result_reason_code = "REQUEST_SATISFIED"
    if len(selected) < requested:
        result_reason_code = (
            "INSUFFICIENT_UNIQUE_CANDIDATES"
            if any(item.reason_code in {"SEMANTIC_DUPLICATE", "TEMPORAL_DUPLICATE"} for item in rejected.values())
            else "INSUFFICIENT_ELIGIBLE_CANDIDATES"
        )
    diversity_decision = DiversityDecision(
        schema_version=str(getattr(settings, "diversity_schema_version", DIVERSITY_DECISION_SCHEMA_VERSION)),
        config_version=str(getattr(settings, "diversity_config_version", "unknown")),
        requested_count=requested,
        lambda_value=diversity_lambda,
        eligible_candidate_ids=sorted(item.candidate.id for item in eligible_for_similarity),
        selected_candidate_ids=[item.candidate.id for item in selected],
        selections=selections,
        exclusions=[rejected[item.candidate.id] for item in scored if item.candidate.id in rejected],
        similarities=similarities,
        result_reason_code=result_reason_code,
    )
    coverage["diversity_decision"] = diversity_decision.to_dict()
    return selected, coverage


def _eligible_diversity_similarities(
    candidates: list[ScoredCandidate], stories: dict[str, StoryUnit],
) -> tuple[list[DiversitySimilarity], dict[tuple[str, str], DiversitySimilarity]]:
    """Calculate the complete pair matrix once, excluding ineligible candidates."""

    records: list[DiversitySimilarity] = []
    index: dict[tuple[str, str], DiversitySimilarity] = {}
    ordered = sorted(candidates, key=lambda item: item.candidate.id)
    for position, first in enumerate(ordered):
        for second in ordered[position + 1:]:
            record = _candidate_diversity_similarity(first, second, stories)
            records.append(record)
            index[_diversity_pair_key(first.candidate.id, second.candidate.id)] = record
    return records, index


def _candidate_diversity_similarity(
    first: ScoredCandidate, second: ScoredCandidate, stories: dict[str, StoryUnit],
) -> DiversitySimilarity:
    """Combine only persisted source evidence; missing evidence never triggers inference."""

    first_candidate, second_candidate = first.candidate, second.candidate
    first_story = stories.get(str(first_candidate.story_unit_id or ""))
    second_story = stories.get(str(second_candidate.story_unit_id or ""))
    first_signature = first_candidate.content_signature
    second_signature = second_candidate.content_signature
    components: dict[str, float] = {}

    semantic = _max_text_similarity(
        [first_candidate.text, first_candidate.core_idea, str(first_signature.get("normalized_core_idea") or "")],
        [second_candidate.text, second_candidate.core_idea, str(second_signature.get("normalized_core_idea") or "")],
    )
    if semantic is not None:
        components["semantic"] = semantic

    key_claim = _max_text_similarity(
        _candidate_key_claims(first_candidate, first_story),
        _candidate_key_claims(second_candidate, second_story),
    )
    if key_claim is not None:
        components["key_claim"] = key_claim

    payoff = _max_text_similarity(
        _candidate_payoffs(first_candidate, first_story),
        _candidate_payoffs(second_candidate, second_story),
    )
    if payoff is not None:
        components["payoff"] = payoff

    metrics = interval_metrics(
        first_candidate.start, first_candidate.end, second_candidate.start, second_candidate.end,
    )
    components["source_time"] = max(metrics.iou, metrics.containment)

    first_topics = {str(item) for item in first_signature.get("topic_ids", []) if str(item)}
    second_topics = {str(item) for item in second_signature.get("topic_ids", []) if str(item)}
    first_chapter = str(first_candidate.chapter_id or (first_story.chapter_id if first_story else ""))
    second_chapter = str(second_candidate.chapter_id or (second_story.chapter_id if second_story else ""))
    if first_topics or second_topics or (first_chapter and second_chapter):
        topic_overlap = _set_similarity(first_topics, second_topics)
        components["chapter_topic"] = max(topic_overlap, float(bool(first_chapter and first_chapter == second_chapter)))

    first_pattern = _rhetorical_pattern(first_candidate, first_story)
    second_pattern = _rhetorical_pattern(second_candidate, second_story)
    if first_pattern and second_pattern:
        components["rhetorical_pattern"] = float(first_pattern == second_pattern)

    weights = {
        "semantic": 0.45,
        "key_claim": 0.20,
        "payoff": 0.10,
        "source_time": 0.10,
        "chapter_topic": 0.08,
        "rhetorical_pattern": 0.07,
    }
    denominator = sum(weights[name] for name in components)
    composite = sum(weights[name] * value for name, value in components.items()) / denominator if denominator else 0.0
    return DiversitySimilarity(
        candidate_id=first_candidate.id,
        other_candidate_id=second_candidate.id,
        composite_similarity=round(composite, 6),
        components={name: round(value, 6) for name, value in components.items()},
        available_components=list(components),
    )


def _candidate_key_claims(candidate: Candidate, story: StoryUnit | None) -> list[str]:
    signature = candidate.content_signature
    values: list[Any] = [signature.get("key_claims"), candidate.core_idea]
    if story is not None:
        values.extend([story.core_idea, story.content_signature.get("key_claims")])
    return _text_values(values)


def _candidate_payoffs(candidate: Candidate, story: StoryUnit | None) -> list[str]:
    values: list[Any] = [candidate.semantic_evidence.get("payoff")]
    if story is not None:
        values.append(story.payoff)
    return _text_values(values)


def _rhetorical_pattern(candidate: Candidate, story: StoryUnit | None) -> str:
    signature = candidate.content_signature
    values = (
        candidate.semantic_evidence.get("rhetorical_pattern"),
        signature.get("rhetorical_pattern"),
        signature.get("narrative_function"),
        story.emotional_arc if story is not None else None,
    )
    return next((str(value).strip() for value in values if str(value or "").strip()), "")


def _text_values(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            result.append(value.strip())
        elif isinstance(value, (list, tuple, set)):
            result.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(result))


def _max_text_similarity(first: list[str], second: list[str]) -> float | None:
    if not first or not second:
        return None
    return max(transcript_similarity(left, right) for left in first for right in second)


def _set_similarity(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _diversity_pair_key(first_id: str, second_id: str) -> tuple[str, str]:
    return tuple(sorted((first_id, second_id)))


def _diversity_similarity_for(
    first: ScoredCandidate, second: ScoredCandidate,
    index: dict[tuple[str, str], DiversitySimilarity],
) -> DiversitySimilarity | None:
    return index.get(_diversity_pair_key(first.candidate.id, second.candidate.id))


def _max_selected_diversity_similarity(
    item: ScoredCandidate, selected: list[ScoredCandidate],
    index: dict[tuple[str, str], DiversitySimilarity],
) -> tuple[float, ScoredCandidate | None, DiversitySimilarity | None]:
    comparisons = [
        (similarity.composite_similarity, chosen.candidate.id, chosen, similarity)
        for chosen in selected
        if (similarity := _diversity_similarity_for(item, chosen, index)) is not None
    ]
    if not comparisons:
        return 0.0, None, None
    similarity, _candidate_id, chosen, record = max(comparisons, key=lambda value: (value[0], value[1]))
    return similarity, chosen, record


def _coverage_temporal_duplicate(
    item: ScoredCandidate, selected: list[ScoredCandidate], config: Any,
) -> tuple[ScoredCandidate, str] | None:
    for chosen in selected:
        metrics = interval_metrics(item.candidate.start, item.candidate.end, chosen.candidate.start, chosen.candidate.end)
        if is_temporal_duplicate(
            metrics,
            overlap_threshold=config.overlap_threshold,
            minimum_distance_seconds=config.min_selected_clip_distance_seconds,
        ):
            return chosen, f"Дублирует временной диапазон {chosen.candidate.id}."
    return None


def build_coverage_map(
    content_map_data: dict[str, Any], candidates: list[ScoredCandidate], selected: list[ScoredCandidate], config: Any,
) -> dict[str, Any]:
    content_map = GlobalContentMap.from_dict(content_map_data)
    stories = {item.story_unit_id: item for item in content_map.story_units}
    strong = [item for item in content_map.story_units if item.publishability_precheck and item.standalone_score >= config.content_understanding.strong_story_unit_threshold]
    selected_story_ids = [str(item.candidate.story_unit_id) for item in selected if item.candidate.story_unit_id]
    selected_chapters = list(dict.fromkeys(str(item.candidate.chapter_id) for item in selected if item.candidate.chapter_id))
    all_topics = _all_topics(content_map.story_units)
    covered_topics = sorted({topic for item in selected for topic in item.candidate.content_signature.get("topic_ids", [])})
    covered_ranges = _merge_ranges([(item.candidate.start, item.candidate.end) for item in selected])
    covered_emotions = {str(item.candidate.content_signature.get("emotional_signature") or "neutral") for item in selected}
    all_emotions = {str(item.content_signature.get("emotional_signature") or "neutral") for item in content_map.story_units}
    covered_narrative = {str(item.candidate.content_signature.get("narrative_function") or "unknown") for item in selected}
    all_narrative = {str(item.content_signature.get("narrative_function") or "unknown") for item in content_map.story_units}
    covered_speakers = {stories[story_id].speaker_context for story_id in selected_story_ids if story_id in stories}
    all_speakers = {item.speaker_context for item in content_map.story_units}
    selected_story_set = set(selected_story_ids)
    clusters = _duplicate_clusters(content_map.story_units)
    explanations = [
        {
            "candidate_id": item.candidate.id, "chapter_id": item.candidate.chapter_id,
            "story_unit_id": item.candidate.story_unit_id, "core_idea": item.candidate.core_idea,
            "incremental_coverage_score": item.candidate.incremental_coverage_score,
            "reason": item.selection_reason, "diagnostics": item.selection_diagnostics,
        }
        for item in selected
    ]
    return {
        "schema_version": str(config.content_understanding.coverage_schema_version),
        "available_chapters": [item.chapter_id for item in content_map.chapters],
        "available_story_units": [item.story_unit_id for item in content_map.story_units],
        "strong_story_units": [item.story_unit_id for item in strong],
        "selected_story_units": selected_story_ids,
        "uncovered_strong_story_units": [item.story_unit_id for item in strong if item.story_unit_id not in selected_story_set],
        "covered_topics": covered_topics,
        "uncovered_topics": sorted(set(all_topics) - set(covered_topics)),
        "covered_temporal_ranges": [{"start": round(start, 3), "end": round(end, 3)} for start, end in covered_ranges],
        "coverage_ratio_by_dimension": {
            "temporal": round(sum(end - start for start, end in covered_ranges) / max(1.0, content_map.source_duration_seconds), 3),
            "chapter": _ratio(len(selected_chapters), len(content_map.chapters)),
            "topic": _ratio(len(covered_topics), len(all_topics)),
            "story_unit": _ratio(len(selected_story_set), len(strong) or len(content_map.story_units)),
            "emotional": _ratio(len(covered_emotions), len(all_emotions)),
            "speaker": _ratio(len(covered_speakers), len(all_speakers)),
            "narrative_function": _ratio(len(covered_narrative), len(all_narrative)),
        },
        "duplicate_content_clusters": clusters,
        "selection_explanations": explanations,
        "selected_chapters": selected_chapters,
        "selected_candidate_count": len(selected),
    }


def recommend_clip_count(content_map_data: dict[str, Any], profile_data: dict[str, Any], requested_count: int) -> dict[str, Any]:
    """Post-analysis recommendation based on distinct publishable StoryUnits."""

    content_map = GlobalContentMap.from_dict(content_map_data)
    profile = VideoContentProfile.from_dict(profile_data)
    strong = [item for item in content_map.story_units if item.publishability_precheck and item.standalone_score >= 0.55]
    distinct: dict[str, StoryUnit] = {}
    for item in strong:
        key = str(item.content_signature.get("transcript_fingerprint") or item.story_unit_id)
        distinct.setdefault(key, item)
    count = len(distinct)
    minimum = 0 if count == 0 else max(1, count - (2 if count >= 4 else 1))
    maximum = count
    if count == 0:
        explanation = "После анализа не найдено самостоятельных publishable StoryUnits; клипы не рекомендуются без ручной проверки."
    else:
        explanation = (
            f"Найдено {count} самостоятельных сильных фрагмента(ов) из {len(content_map.story_units)} StoryUnits; "
            f"рекомендуем создать {minimum}–{maximum} ролика(ов), не дублируя одну историю."
        )
    return {
        "schema_version": "5A.1", "post_analysis": True, "estimated_story_count": len(content_map.story_units),
        "strong_story_unit_count": count, "estimated_publishable_clip_range": {"min": minimum, "max": maximum},
        "requested_clip_count": requested_count, "recommended_clip_duration_range": profile.recommended_clip_duration_range,
        "explanation": explanation, "strategy_id": profile.strategy_id,
    }


def _coverage_increment(
    item: ScoredCandidate, selected: list[ScoredCandidate], stories: dict[str, StoryUnit], config: Any,
) -> dict[str, Any]:
    candidate = item.candidate
    chosen_chapters = {other.candidate.chapter_id for other in selected}
    chosen_stories = {other.candidate.story_unit_id for other in selected}
    chosen_topics = {topic for other in selected for topic in other.candidate.content_signature.get("topic_ids", [])}
    candidate_topics = set(candidate.content_signature.get("topic_ids", []))
    new_topic_ratio = _ratio(len(candidate_topics - chosen_topics), len(candidate_topics)) if candidate_topics else 0.0
    emotion = str(candidate.content_signature.get("emotional_signature") or "neutral")
    chosen_emotions = {str(other.candidate.content_signature.get("emotional_signature") or "neutral") for other in selected}
    narrative = str(candidate.content_signature.get("narrative_function") or "unknown")
    chosen_narrative = {str(other.candidate.content_signature.get("narrative_function") or "unknown") for other in selected}
    temporal_new = 1.0 if not selected or all(abs(candidate.start - other.candidate.start) >= config.min_selected_clip_distance_seconds for other in selected) else 0.0
    semantic_similarity = max((
        max(
            transcript_similarity(candidate.text, other.candidate.text),
            _signature_similarity(candidate.content_signature, other.candidate.content_signature),
        )
        for other in selected
    ), default=0.0)
    signals = [
        1.0 if candidate.story_unit_id not in chosen_stories else 0.0,
        1.0 if candidate.chapter_id not in chosen_chapters else 0.0,
        new_topic_ratio,
        1.0 if emotion not in chosen_emotions else 0.0,
        1.0 if narrative not in chosen_narrative else 0.0,
        temporal_new,
    ]
    return {
        "incremental_coverage_score": round(sum(signals) / len(signals), 3),
        "new_story_unit": signals[0], "new_chapter": signals[1], "new_topic_ratio": round(new_topic_ratio, 3),
        "new_emotion": signals[3], "new_narrative_function": signals[4], "new_temporal_region": temporal_new,
        "semantic_duplicate_similarity": round(semantic_similarity, 3),
    }


def _coverage_duplicate(item: ScoredCandidate, selected: list[ScoredCandidate], config: Any, semantic_threshold: float) -> str | None:
    for chosen in selected:
        metrics = interval_metrics(item.candidate.start, item.candidate.end, chosen.candidate.start, chosen.candidate.end)
        if is_temporal_duplicate(metrics, overlap_threshold=config.overlap_threshold, minimum_distance_seconds=config.min_selected_clip_distance_seconds):
            return f"Дублирует временной диапазон {chosen.candidate.id}."
        similarity = max(
            transcript_similarity(item.candidate.text, chosen.candidate.text),
            _signature_similarity(item.candidate.content_signature, chosen.candidate.content_signature),
        )
        if similarity >= semantic_threshold:
            return f"Семантически повторяет StoryUnit {chosen.candidate.story_unit_id or chosen.candidate.id}."
    return None


def _signature_similarity(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_tokens = set(_tokens(str(first.get("normalized_core_idea") or ""))) | set(first.get("keyword_set", []))
    second_tokens = set(_tokens(str(second.get("normalized_core_idea") or ""))) | set(second.get("keyword_set", []))
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def _all_topics(stories: list[StoryUnit]) -> list[str]:
    return sorted({topic for item in stories for topic in item.content_signature.get("topic_ids", [])})


def _duplicate_clusters(stories: list[StoryUnit]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for story in stories:
        fingerprint = str(story.content_signature.get("transcript_fingerprint") or story.story_unit_id)
        grouped.setdefault(fingerprint, []).append(story.story_unit_id)
    return [
        {"fingerprint": fingerprint, "story_unit_ids": identifiers}
        for fingerprint, identifiers in sorted(grouped.items()) if len(identifiers) > 1
    ]


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((start, end) for start, end in ranges if start < end):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0
