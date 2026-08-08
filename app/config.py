from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from app.errors import ClipEngineError


DEFAULT_COVERAGE_SELECTION_WEIGHTS = {
    "base_quality": 0.36,
    "standalone": 0.12,
    "completeness": 0.10,
    "boundary": 0.11,
    "incremental_coverage": 0.12,
    "chapter_diversity": 0.05,
    "topic_diversity": 0.06,
    "emotional_diversity": 0.02,
    "temporal_diversity": 0.02,
    "semantic_duplicate_penalty": 0.25,
    "context_dependency_penalty": 0.10,
    "repetition_penalty": 0.08,
}


@dataclass(slots=True)
class AIConfig:
    """Configuration shared by every AI provider."""

    provider: str = "openai"
    model: str = "gpt-5-mini"
    max_retries: int = 2
    # USD per token. Kept in configuration because prices can change.
    input_token_price: float | None = 0.00000025
    output_token_price: float | None = 0.000002

    def validate(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in {"openai", "gemini", "mock"}:
            raise ClipEngineError("ai.provider: openai, gemini или mock.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ClipEngineError("ai.model не должен быть пустым.")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 5:
            raise ClipEngineError("ai.max_retries должен быть числом от 0 до 5.")
        for name, value in (
            ("ai.input_token_price", self.input_token_price),
            ("ai.output_token_price", self.output_token_price),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise ClipEngineError(f"{name} не может быть отрицательным.")


@dataclass(slots=True)
class VisionConfig:
    """Paid vision limits shared by PASS 1 and the callable PASS 2 contract."""

    enabled: bool = True
    cache_enabled: bool = True
    prompt_version: str = "6B.pass1.1"
    pass2_prompt_version: str = "6B.pass2.1"
    schema_version: str = "6B.1"
    pass1_batch_size: int = 3
    pass2_min_frames: int = 3
    pass2_max_frames: int = 7
    standard_max_frames: int = 12
    standard_max_calls: int = 4
    standard_max_tokens: int = 12000
    standard_max_estimated_cost: float = 0.05
    maximum_max_frames: int = 32
    maximum_max_calls: int = 10
    maximum_max_tokens: int = 32000
    maximum_max_estimated_cost: float = 0.15
    prompt_input_tokens: int = 700
    low_detail_input_tokens_per_frame: int = 300
    high_detail_input_tokens_per_frame: int = 900
    max_output_tokens_per_call: int = 2400
    frame_width: int = 512

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.cache_enabled, bool):
            raise ClipEngineError("vision.enabled и vision.cache_enabled должны быть true или false.")
        for name, value in (
            ("vision.prompt_version", self.prompt_version),
            ("vision.pass2_prompt_version", self.pass2_prompt_version),
            ("vision.schema_version", self.schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ClipEngineError(f"{name} не должен быть пустым.")
        integers: tuple[tuple[str, int, int, int], ...] = (
            ("vision.pass1_batch_size", self.pass1_batch_size, 2, 4),
            ("vision.pass2_min_frames", self.pass2_min_frames, 3, 7),
            ("vision.pass2_max_frames", self.pass2_max_frames, 3, 7),
            ("vision.standard_max_frames", self.standard_max_frames, 0, 256),
            ("vision.standard_max_calls", self.standard_max_calls, 0, 128),
            ("vision.standard_max_tokens", self.standard_max_tokens, 0, 1_000_000),
            ("vision.maximum_max_frames", self.maximum_max_frames, 0, 256),
            ("vision.maximum_max_calls", self.maximum_max_calls, 0, 128),
            ("vision.maximum_max_tokens", self.maximum_max_tokens, 0, 1_000_000),
            ("vision.prompt_input_tokens", self.prompt_input_tokens, 1, 100_000),
            ("vision.low_detail_input_tokens_per_frame", self.low_detail_input_tokens_per_frame, 1, 100_000),
            ("vision.high_detail_input_tokens_per_frame", self.high_detail_input_tokens_per_frame, 1, 100_000),
            ("vision.max_output_tokens_per_call", self.max_output_tokens_per_call, 1, 100_000),
            ("vision.frame_width", self.frame_width, 128, 2048),
        )
        for integer_name, integer_value, minimum, maximum in integers:
            if isinstance(integer_value, bool) or not isinstance(integer_value, int) or not minimum <= integer_value <= maximum:
                raise ClipEngineError(f"{integer_name} должен быть целым числом от {minimum} до {maximum}.")
        if self.pass2_min_frames > self.pass2_max_frames:
            raise ClipEngineError("vision.pass2_min_frames не может быть больше pass2_max_frames.")
        for cost_name, cost_value in (
            ("vision.standard_max_estimated_cost", self.standard_max_estimated_cost),
            ("vision.maximum_max_estimated_cost", self.maximum_max_estimated_cost),
        ):
            if isinstance(cost_value, bool) or not isinstance(cost_value, (int, float)) or cost_value < 0:
                raise ClipEngineError(f"{cost_name} не может быть отрицательным.")


@dataclass(slots=True)
class TranscriptFeatureConfig:
    hook_patterns: list[str] = field(default_factory=lambda: [
        "почему", "как", "никогда", "главная ошибка", "вот что произошло",
        "самое важное", "how", "why", "never", "the main mistake", "what happened",
    ])
    filler_words: list[str] = field(default_factory=lambda: [
        "ээ", "эм", "ну", "как бы", "типа", "короче", "uh", "um", "like",
    ])

    def validate(self) -> None:
        for name, values in (("transcript_features.hook_patterns", self.hook_patterns), ("transcript_features.filler_words", self.filler_words)):
            if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
                raise ClipEngineError(f"{name} должен быть непустым списком строк.")


@dataclass(slots=True)
class AudioAnalysisConfig:
    window_seconds: float = 0.1
    silence_threshold: float = 0.08
    min_silence_seconds: float = 0.3

    def validate(self) -> None:
        if not 0.02 <= self.window_seconds <= 2:
            raise ClipEngineError("audio_analysis.window_seconds должен быть от 0.02 до 2.")
        if not 0 <= self.silence_threshold <= 1:
            raise ClipEngineError("audio_analysis.silence_threshold должен быть от 0 до 1.")
        if not 0 <= self.min_silence_seconds <= 10:
            raise ClipEngineError("audio_analysis.min_silence_seconds должен быть от 0 до 10.")


@dataclass(slots=True)
class SceneDetectionConfig:
    enabled: bool = True
    threshold: float = 0.35

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ClipEngineError("scene_detection.enabled должен быть true или false.")
        if not 0.01 <= self.threshold <= 1:
            raise ClipEngineError("scene_detection.threshold должен быть от 0.01 до 1.")


@dataclass(slots=True)
class CandidateGenerationConfig:
    min_duration_seconds: float = 15.0
    target_duration_seconds: float = 30.0
    max_duration_seconds: float = 60.0
    boundary_search_radius_seconds: float = 4.0
    max_candidates: int = 100
    overlap_limit: float = 0.75

    def validate(self) -> None:
        if not (0 < self.min_duration_seconds <= self.target_duration_seconds <= self.max_duration_seconds <= 180):
            raise ClipEngineError("candidate_generation: minimum ≤ target ≤ maximum ≤ 180.")
        if not 0 <= self.boundary_search_radius_seconds <= 20:
            raise ClipEngineError("candidate_generation.boundary_search_radius_seconds должен быть от 0 до 20.")
        if not 1 <= self.max_candidates <= 500:
            raise ClipEngineError("candidate_generation.max_candidates должен быть от 1 до 500.")
        if not 0 <= self.overlap_limit <= 1:
            raise ClipEngineError("candidate_generation.overlap_limit должен быть от 0 до 1.")


@dataclass(slots=True)
class ContentUnderstandingConfig:
    """Versioned semantic-analysis settings, isolated from render-only options."""

    enabled: bool = True
    strategy_version: str = "5A.1"
    profile_schema_version: str = "5A.1"
    content_map_schema_version: str = "5A.1"
    story_unit_schema_version: str = "5A.1"
    chapter_pause_seconds: float = 1.25
    max_chapter_seconds: float = 110.0
    min_story_unit_seconds: float = 12.0
    target_story_unit_seconds: float = 38.0
    max_story_unit_seconds: float = 90.0
    boundary_schema_version: str = "5C.1"
    max_head_padding_seconds: float = 0.5
    target_head_padding_seconds: float = 0.25
    min_tail_padding_seconds: float = 0.25
    target_tail_padding_seconds: float = 0.65
    max_tail_padding_seconds: float = 1.5
    max_semantic_extension_seconds: float = 12.0
    continuation_risk_threshold: float = 0.65
    coverage_schema_version: str = "5A.1"
    coverage_selection_version: str = "5A.1"
    coverage_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COVERAGE_SELECTION_WEIGHTS))
    strong_story_unit_threshold: float = 0.55
    semantic_duplicate_threshold: float = 0.78
    coverage_min_quality_score: float = 55.0
    diversity_schema_version: str = "5B.2"
    diversity_config_version: str = "5B.2"
    diversity_lambda: float = 0.76

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ClipEngineError("content_understanding.enabled должен быть true или false.")
        for name, value in (
            ("content_understanding.strategy_version", self.strategy_version),
            ("content_understanding.profile_schema_version", self.profile_schema_version),
            ("content_understanding.content_map_schema_version", self.content_map_schema_version),
            ("content_understanding.story_unit_schema_version", self.story_unit_schema_version),
            ("content_understanding.boundary_schema_version", self.boundary_schema_version),
            ("content_understanding.coverage_schema_version", self.coverage_schema_version),
            ("content_understanding.coverage_selection_version", self.coverage_selection_version),
            ("content_understanding.diversity_schema_version", self.diversity_schema_version),
            ("content_understanding.diversity_config_version", self.diversity_config_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ClipEngineError(f"{name} не должен быть пустым.")
        if not 0 <= self.chapter_pause_seconds <= 20:
            raise ClipEngineError("content_understanding.chapter_pause_seconds должен быть от 0 до 20.")
        if not 10 <= self.max_chapter_seconds <= 900:
            raise ClipEngineError("content_understanding.max_chapter_seconds должен быть от 10 до 900.")
        if not (
            1 <= self.min_story_unit_seconds <= self.target_story_unit_seconds
            <= self.max_story_unit_seconds <= 180
        ):
            raise ClipEngineError(
                "content_understanding: min StoryUnit ≤ target StoryUnit ≤ max StoryUnit ≤ 180."
            )
        if not 0 <= self.target_head_padding_seconds <= self.max_head_padding_seconds <= 2:
            raise ClipEngineError("content_understanding head padding должен быть от 0 до 2 секунд.")
        if not 0 <= self.min_tail_padding_seconds <= self.target_tail_padding_seconds <= self.max_tail_padding_seconds <= 3:
            raise ClipEngineError("content_understanding tail padding должен быть от 0 до 3 секунд.")
        if not 0 <= self.max_semantic_extension_seconds <= 30:
            raise ClipEngineError("content_understanding.max_semantic_extension_seconds должен быть от 0 до 30.")
        if not 0 <= self.continuation_risk_threshold <= 1:
            raise ClipEngineError("content_understanding.continuation_risk_threshold должен быть от 0 до 1.")
        if set(self.coverage_weights) != set(DEFAULT_COVERAGE_SELECTION_WEIGHTS):
            raise ClipEngineError("content_understanding.coverage_weights должен содержать все заданные компоненты coverage selection.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in self.coverage_weights.values()):
            raise ClipEngineError("content_understanding.coverage_weights должен содержать неотрицательные числа.")
        if not 0 <= self.strong_story_unit_threshold <= 1 or not 0 <= self.semantic_duplicate_threshold <= 1:
            raise ClipEngineError("content_understanding coverage thresholds должны быть от 0 до 1.")
        if not 0 <= self.coverage_min_quality_score <= 100:
            raise ClipEngineError("content_understanding.coverage_min_quality_score должен быть от 0 до 100.")
        if not 0 <= self.diversity_lambda <= 1:
            raise ClipEngineError("content_understanding.diversity_lambda должен быть от 0 до 1.")


DEFAULT_VIRALITY_WEIGHTS = {
    "hook": 0.13,
    "curiosity": 0.07,
    "emotion": 0.09,
    "conflict": 0.07,
    "specificity": 0.08,
    "novelty": 0.06,
    "usefulness": 0.08,
    "quotability": 0.08,
    "momentum": 0.09,
    "payoff": 0.12,
    "retention": 0.08,
    "publishability": 0.05,
}
VIRALITY_STRATEGY_IDS = (
    "motivational_monologue", "generic_monologue", "generic_dialogue", "generic_educational",
    "generic_scene_driven", "generic_fallback",
)
DEFAULT_VIRALITY_STRATEGY_WEIGHTS = {
    "motivational_monologue": {
        "hook": 0.14, "curiosity": 0.06, "emotion": 0.13, "conflict": 0.06, "specificity": 0.06,
        "novelty": 0.05, "usefulness": 0.05, "quotability": 0.11, "momentum": 0.09, "payoff": 0.14,
        "retention": 0.07, "publishability": 0.04,
    },
    "generic_monologue": dict(DEFAULT_VIRALITY_WEIGHTS),
    "generic_dialogue": {
        "hook": 0.11, "curiosity": 0.11, "emotion": 0.08, "conflict": 0.13, "specificity": 0.05,
        "novelty": 0.05, "usefulness": 0.06, "quotability": 0.06, "momentum": 0.10, "payoff": 0.13,
        "retention": 0.08, "publishability": 0.04,
    },
    "generic_educational": {
        "hook": 0.09, "curiosity": 0.05, "emotion": 0.06, "conflict": 0.07, "specificity": 0.15,
        "novelty": 0.04, "usefulness": 0.17, "quotability": 0.05, "momentum": 0.08, "payoff": 0.13,
        "retention": 0.07, "publishability": 0.04,
    },
    "generic_scene_driven": {
        "hook": 0.12, "curiosity": 0.07, "emotion": 0.12, "conflict": 0.12, "specificity": 0.04,
        "novelty": 0.09, "usefulness": 0.04, "quotability": 0.05, "momentum": 0.14, "payoff": 0.12,
        "retention": 0.06, "publishability": 0.03,
    },
    "generic_fallback": dict(DEFAULT_VIRALITY_WEIGHTS),
}


@dataclass(slots=True)
class ViralityScoringConfig:
    """Goal 5B policy. Scores are comparative content signals, never view predictions."""

    enabled: bool = False
    schema_version: str = "5B.2"
    scoring_config_version: str = "5B.2"
    strategy_version: str = "5B.2"
    semantic_ai_mode: str = "auto"
    max_ai_batch_candidates: int = 20
    minimum_quality_score: float = 0.45
    minimum_publishability_score: float = 0.55
    strong_story_unit_threshold: float = 0.55
    uncertainty_tiebreak_weight: float = 0.08
    dead_zone_minimum_seconds: float = 1.4
    dead_zone_penalty_weight: float = 0.10
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_VIRALITY_WEIGHTS))
    strategy_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {name: dict(values) for name, values in DEFAULT_VIRALITY_STRATEGY_WEIGHTS.items()}
    )

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ClipEngineError("virality.enabled должен быть true или false.")
        for name, value in (
            ("virality.schema_version", self.schema_version),
            ("virality.scoring_config_version", self.scoring_config_version),
            ("virality.strategy_version", self.strategy_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ClipEngineError(f"{name} не должен быть пустым.")
        if self.semantic_ai_mode not in {"off", "auto", "on"}:
            raise ClipEngineError("virality.semantic_ai_mode: off, auto или on.")
        if not 1 <= self.max_ai_batch_candidates <= 100:
            raise ClipEngineError("virality.max_ai_batch_candidates должен быть от 1 до 100.")
        for name, value in (
            ("minimum_quality_score", self.minimum_quality_score),
            ("minimum_publishability_score", self.minimum_publishability_score),
            ("strong_story_unit_threshold", self.strong_story_unit_threshold),
            ("uncertainty_tiebreak_weight", self.uncertainty_tiebreak_weight),
            ("dead_zone_penalty_weight", self.dead_zone_penalty_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ClipEngineError(f"virality.{name} должен быть от 0 до 1.")
        if not 0.2 <= self.dead_zone_minimum_seconds <= 20:
            raise ClipEngineError("virality.dead_zone_minimum_seconds должен быть от 0.2 до 20.")
        if set(self.weights) != set(DEFAULT_VIRALITY_WEIGHTS):
            raise ClipEngineError("virality.weights должен содержать все компоненты ViralPotentialScore.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in self.weights.values()):
            raise ClipEngineError("virality.weights должен содержать неотрицательные числа.")
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ClipEngineError("Сумма virality.weights должна быть равна 1.0.")
        if set(self.strategy_weights) != set(VIRALITY_STRATEGY_IDS):
            raise ClipEngineError("virality.strategy_weights должен содержать все поддерживаемые content strategies.")
        for strategy, weights in self.strategy_weights.items():
            if not isinstance(weights, dict) or set(weights) != set(DEFAULT_VIRALITY_WEIGHTS):
                raise ClipEngineError(f"virality.strategy_weights.{strategy} должен содержать все компоненты ViralPotentialScore.")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
                raise ClipEngineError(f"virality.strategy_weights.{strategy} должен содержать неотрицательные числа.")
            if abs(sum(weights.values()) - 1.0) > 0.001:
                raise ClipEngineError(f"Сумма virality.strategy_weights.{strategy} должна быть равна 1.0.")


DEFAULT_SCORING_WEIGHTS = {
    "hook": 0.18,
    "completeness": 0.18,
    "clarity": 0.12,
    "speech_density": 0.08,
    "pacing": 0.08,
    "audio_energy": 0.08,
    "scene_structure": 0.10,
    "context_independence": 0.10,
    "boundary_quality": 0.08,
}


@dataclass(slots=True)
class ScoringConfig:
    candidate_quality_config_version: str = "5B.1"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORING_WEIGHTS))
    repetition_penalty_weight: float = 12.0
    filler_penalty_weight: float = 18.0

    def validate(self) -> None:
        if not isinstance(self.candidate_quality_config_version, str) or not self.candidate_quality_config_version.strip():
            raise ClipEngineError("scoring.candidate_quality_config_version не должен быть пустым.")
        expected = set(DEFAULT_SCORING_WEIGHTS)
        if set(self.weights) != expected:
            raise ClipEngineError("scoring.weights должен содержать все заданные компоненты local scoring.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in self.weights.values()):
            raise ClipEngineError("scoring.weights должен содержать неотрицательные числа.")
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ClipEngineError("Сумма scoring.weights должна быть равна 1.0.")
        if not 0 <= self.repetition_penalty_weight <= 100 or not 0 <= self.filler_penalty_weight <= 100:
            raise ClipEngineError("Веса штрафов scoring должны быть от 0 до 100.")


@dataclass(slots=True)
class AIRerankingConfig:
    enabled: bool = True
    shortlist_size: int = 15
    final_clip_count: int = 3

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ClipEngineError("ai_reranking.enabled должен быть true или false.")
        if not 1 <= self.shortlist_size <= 100:
            raise ClipEngineError("ai_reranking.shortlist_size должен быть от 1 до 100.")
        if not 1 <= self.final_clip_count <= self.shortlist_size:
            raise ClipEngineError("ai_reranking.final_clip_count должен быть от 1 до shortlist_size.")


DEFAULT_TRANSFORMATION_WEIGHTS = {
    "hook_strength": 0.10,
    "clarity": 0.10,
    "completeness": 0.10,
    "brevity": 0.08,
    "information_density": 0.08,
    "context_independence": 0.09,
    "naturalness": 0.08,
    "pacing": 0.07,
    "ending_strength": 0.07,
    "factual_grounding": 0.14,
    "source_coverage": 0.09,
}


@dataclass(slots=True)
class TransformationConfig:
    """Typed, backwards-compatible settings for the Goal 2 script artifact."""

    # Disabled by default: an existing process command must not unexpectedly make an
    # external request or create a new product artifact until the user asks for it.
    enabled: bool = False
    mode: str = "auto"
    ai_strategy: str = "compact"
    target_duration_seconds: float = 35.0
    min_duration_seconds: float = 20.0
    max_duration_seconds: float = 60.0
    target_words_per_second: float = 2.4
    preserve_language: bool = True
    output_language: str = "auto"
    allow_translation: bool = False
    context_before_seconds: float = 10.0
    context_after_seconds: float = 10.0
    allow_cta: bool = False
    max_repair_attempts: int = 1
    fallback_enabled: bool = True
    strict_grounding: bool = True
    minimum_grounding_score: float = 0.90
    minimum_quality_score: float = 0.55
    semantic_validation_enabled: bool = True
    cache_enabled: bool = True
    # The mock modes are intentional test fixtures; normal CLI use stays "valid".
    mock_mode: str = "valid"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TRANSFORMATION_WEIGHTS))

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ClipEngineError("transformation.enabled должен быть true или false.")
        if self.mode not in {
            "faithful_compression", "hook_first", "educational", "story", "listicle",
            "provocative", "calm_expert", "direct_response", "auto",
        }:
            raise ClipEngineError("transformation.mode содержит неподдерживаемый режим.")
        if self.ai_strategy not in {"staged", "compact", "local_only"}:
            raise ClipEngineError("transformation.ai_strategy: staged, compact или local_only.")
        if not (0 < self.min_duration_seconds <= self.target_duration_seconds <= self.max_duration_seconds <= 180):
            raise ClipEngineError("transformation: min_duration_seconds ≤ target_duration_seconds ≤ max_duration_seconds ≤ 180.")
        if not 0.5 <= self.target_words_per_second <= 5.0:
            raise ClipEngineError("transformation.target_words_per_second должен быть от 0.5 до 5.")
        if self.output_language not in {"auto", "ru", "en"}:
            raise ClipEngineError("transformation.output_language: auto, ru или en.")
        if not 0 <= self.context_before_seconds <= 60 or not 0 <= self.context_after_seconds <= 60:
            raise ClipEngineError("transformation context_before_seconds/context_after_seconds должны быть от 0 до 60.")
        if not 0 <= self.max_repair_attempts <= 2:
            raise ClipEngineError("transformation.max_repair_attempts должен быть от 0 до 2.")
        for name, value in (
            ("minimum_grounding_score", self.minimum_grounding_score),
            ("minimum_quality_score", self.minimum_quality_score),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ClipEngineError(f"transformation.{name} должен быть от 0 до 1.")
        if not all(isinstance(value, bool) for value in (
            self.preserve_language, self.allow_translation, self.allow_cta,
            self.fallback_enabled, self.strict_grounding, self.semantic_validation_enabled,
            self.cache_enabled,
        )):
            raise ClipEngineError("Флаги transformation должны быть true или false.")
        if self.mock_mode not in {
            "valid", "invalid_fact_id", "unsupported_number", "changed_negation",
            "malformed_json", "provider_error", "empty_script", "repair_success", "repair_failure",
        }:
            raise ClipEngineError("transformation.mock_mode содержит неподдерживаемый тестовый режим.")
        if set(self.weights) != set(DEFAULT_TRANSFORMATION_WEIGHTS):
            raise ClipEngineError("transformation.weights должен содержать все компоненты ScriptQualityScore.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in self.weights.values()):
            raise ClipEngineError("transformation.weights должен содержать неотрицательные числа.")
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ClipEngineError("Сумма transformation.weights должна быть равна 1.0.")


@dataclass(slots=True)
class ProductionConfig:
    """Local-only Goal 3A planning settings; no TTS or media tool settings exist here."""

    enabled: bool = True
    cache_enabled: bool = True
    narration_words_per_second: float = 2.4
    pause_after_narration_seconds: float = 0.25
    voice_profile_id: str = "default-documentary"
    voice_gender: str = "neutral"
    voice_style: str = "documentary"
    original_dialogue_speaker: str = "original_speaker_unknown"
    # Source dialogue is the product default; synthetic narration is opt-in.
    audio_mode: str = "original"

    def validate(self) -> None:
        if self.audio_mode not in {"original", "original_enhanced", "voiceover", "replace_voice", "mixed"}:
            raise ClipEngineError("production.audio_mode must be a supported audio mode.")
        if not isinstance(self.enabled, bool) or not isinstance(self.cache_enabled, bool):
            raise ClipEngineError("production.enabled и production.cache_enabled должны быть true или false.")
        if not 0.5 <= self.narration_words_per_second <= 5.0:
            raise ClipEngineError("production.narration_words_per_second должен быть от 0.5 до 5.")
        if not 0 <= self.pause_after_narration_seconds <= 3:
            raise ClipEngineError("production.pause_after_narration_seconds должен быть от 0 до 3.")
        if not self.voice_profile_id.strip() or not self.original_dialogue_speaker.strip():
            raise ClipEngineError("production voice_profile_id и original_dialogue_speaker не должны быть пустыми.")
        if self.voice_gender not in {"male", "female", "neutral"}:
            raise ClipEngineError("production.voice_gender: male, female или neutral.")
        if self.voice_style not in {"calm", "energetic", "documentary", "conversational"}:
            raise ClipEngineError("production.voice_style: calm, energetic, documentary или conversational.")


@dataclass(slots=True)
class TTSConfig:
    """Goal 3B settings. Disabled by default so existing process runs stay unchanged."""

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini-tts"
    voice: str = "auto"
    language: str = "auto"
    speed: float = 1.0
    budget_limit: float = 1.0
    cost_per_1m_characters: float = 15.0
    timeout_seconds: float = 45.0
    max_retries: int = 2
    cache_enabled: bool = True
    output_format: str = "wav"
    sample_rate: int = 48000
    # ProductionPlan timing is a local WPS heuristic. A difference is informative,
    # not automatically broken audio; only an extreme difference is an error.
    duration_warning_ratio: float = 0.50
    duration_error_ratio: float = 6.0
    minimum_audio_duration: float = 0.10
    maximum_segment_duration: float = 120.0
    provider_config_version: str = "3B.0"
    # Test-only deterministic provider behaviours; production config should remain valid.
    mock_mode: str = "valid"

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.cache_enabled, bool):
            raise ClipEngineError("tts.enabled и tts.cache_enabled должны быть true или false.")
        if self.provider not in {"openai", "mock", "local"}:
            raise ClipEngineError("tts.provider: openai, mock или local.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ClipEngineError("tts.model не должен быть пустым.")
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise ClipEngineError("tts.voice не должен быть пустым.")
        if self.language not in {"auto", "ru", "en"}:
            raise ClipEngineError("tts.language: auto, ru или en.")
        if not 0.25 <= self.speed <= 4:
            raise ClipEngineError("tts.speed должен быть от 0.25 до 4.")
        for name, value in (
            ("tts.budget_limit", self.budget_limit),
            ("tts.cost_per_1m_characters", self.cost_per_1m_characters),
            ("tts.duration_warning_ratio", self.duration_warning_ratio),
            ("tts.duration_error_ratio", self.duration_error_ratio),
            ("tts.minimum_audio_duration", self.minimum_audio_duration),
            ("tts.maximum_segment_duration", self.maximum_segment_duration),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ClipEngineError(f"{name} должен быть неотрицательным числом.")
        if self.duration_error_ratio < self.duration_warning_ratio:
            raise ClipEngineError("tts.duration_error_ratio должен быть не меньше tts.duration_warning_ratio.")
        if self.maximum_segment_duration <= 0 or self.minimum_audio_duration > self.maximum_segment_duration:
            raise ClipEngineError("tts.maximum_segment_duration должен быть положительным и не меньше minimum_audio_duration.")
        if not 1 <= self.sample_rate <= 192000:
            raise ClipEngineError("tts.sample_rate должен быть от 1 до 192000.")
        if self.output_format != "wav":
            raise ClipEngineError("Goal 3B поддерживает только tts.output_format: wav.")
        if not 1 <= self.timeout_seconds <= 300:
            raise ClipEngineError("tts.timeout_seconds должен быть от 1 до 300.")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 5:
            raise ClipEngineError("tts.max_retries должен быть числом от 0 до 5.")
        if not self.provider_config_version.strip():
            raise ClipEngineError("tts.provider_config_version не должен быть пустым.")
        if self.mock_mode not in {"valid", "provider_error", "timeout", "empty_audio", "malformed_response"}:
            raise ClipEngineError("tts.mock_mode содержит неподдерживаемый тестовый режим.")


@dataclass(slots=True)
class AudioCompositionConfig:
    """Goal 3C audio-only composition settings; disabled to preserve legacy process runs."""

    enabled: bool = False
    cache_enabled: bool = True
    sample_rate: int = 48000
    output_format: str = "wav"
    ducking_enabled: bool = True
    duck_level: float = 0.35
    duck_attack_seconds: float = 0.08
    duck_release_seconds: float = 0.20
    preserve_original_events: bool = True
    narration_target_lufs: float = -16.0
    narration_true_peak_db: float = -1.5
    narration_lra: float = 11.0
    engine_version: str = "3C.0"

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.cache_enabled, bool):
            raise ClipEngineError("audio_composition.enabled и cache_enabled должны быть true или false.")
        if not isinstance(self.ducking_enabled, bool) or not isinstance(self.preserve_original_events, bool):
            raise ClipEngineError("audio_composition флаги должны быть true или false.")
        if self.output_format != "wav":
            raise ClipEngineError("Goal 3C поддерживает только audio_composition.output_format: wav.")
        if not 8000 <= self.sample_rate <= 192000:
            raise ClipEngineError("audio_composition.sample_rate должен быть от 8000 до 192000.")
        if not 0.10 <= self.duck_level <= 1:
            raise ClipEngineError("audio_composition.duck_level должен быть от 0.10 до 1.")
        if not 0 <= self.duck_attack_seconds <= 5 or not 0 <= self.duck_release_seconds <= 5:
            raise ClipEngineError("audio_composition duck attack/release должны быть от 0 до 5 секунд.")
        if not -40 <= self.narration_target_lufs <= -5:
            raise ClipEngineError("audio_composition.narration_target_lufs должен быть от -40 до -5.")
        if not -12 <= self.narration_true_peak_db <= -0.1:
            raise ClipEngineError("audio_composition.narration_true_peak_db должен быть от -12 до -0.1.")
        if not 1 <= self.narration_lra <= 20:
            raise ClipEngineError("audio_composition.narration_lra должен быть от 1 до 20.")
        if not self.engine_version.strip():
            raise ClipEngineError("audio_composition.engine_version не должен быть пустым.")


@dataclass(slots=True)
class ProductionRenderConfig:
    """Goal 3D final video composition settings; disabled by default for legacy safety."""

    enabled: bool = False
    cache_enabled: bool = True
    output_width: int = 1080
    output_height: int = 1920
    output_fps: float = 30.0
    video_codec: str = "h264"
    video_bitrate: str = "6M"
    pixel_format: str = "yuv420p"
    encoder: str = "auto"
    crop_strategy: str = "fit_blur_background"
    manual_crop_x: float = 0.5
    manual_crop_y: float = 0.5
    transitions: str = "cut"
    subtitles_enabled: bool = True
    subtitle_style: str = "documentary"
    subtitle_font_family: str = "Arial"
    subtitle_language: str = "auto"
    subtitle_max_chars_per_line: int = 28
    subtitle_max_lines: int = 2
    subtitle_min_duration: float = 0.45
    subtitle_max_duration: float = 3.5
    subtitle_reading_speed_cps: float = 15.0
    subtitle_min_words_per_cue: int = 2
    subtitle_max_words_per_cue: int = 9
    subtitle_max_rendered_width_ratio: float = 0.95
    subtitle_min_font_scale: float = 0.80
    subtitle_quality_version: str = "5E.0"
    maximum_freeze_duration: float = 1.5
    maximum_loop_duration: float = 0.0
    minimum_clip_duration: float = 0.10
    maximum_speed_adjustment: float = 1.0
    av_sync_warning_ms: float = 100.0
    av_sync_error_ms: float = 350.0
    maximum_duration_difference: float = 0.35
    render_config_version: str = "3D.0"

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.cache_enabled, bool) or not isinstance(self.subtitles_enabled, bool):
            raise ClipEngineError("production_render.enabled, cache_enabled и subtitles_enabled должны быть true или false.")
        if self.output_width < 2 or self.output_height < 2 or self.output_width % 2 or self.output_height % 2:
            raise ClipEngineError("production_render output_width/output_height должны быть положительными чётными числами.")
        if abs((self.output_width / self.output_height) - (9 / 16)) > 0.002:
            raise ClipEngineError("production_render поддерживает только холст 9:16.")
        if not 1 < self.output_fps <= 120:
            raise ClipEngineError("production_render.output_fps должен быть от 1 до 120.")
        if self.video_codec != "h264" or self.pixel_format != "yuv420p":
            raise ClipEngineError("Goal 3D поддерживает только H.264 и yuv420p.")
        if not isinstance(self.video_bitrate, str) or not self.video_bitrate.strip():
            raise ClipEngineError("production_render.video_bitrate не должен быть пустым.")
        if self.encoder not in {"auto", "nvenc", "cpu"}:
            raise ClipEngineError("production_render.encoder: auto, nvenc или cpu.")
        if self.crop_strategy not in {"safe_auto", "center_crop", "fit_blur_background", "fit_solid_background", "top_crop", "manual_normalized_crop"}:
            raise ClipEngineError("production_render.crop_strategy содержит неподдерживаемую стратегию.")
        if self.transitions not in {"cut", "short_crossfade", "fade_from_black", "fade_to_black"}:
            raise ClipEngineError("production_render.transitions содержит неподдерживаемый переход.")
        if self.subtitle_style not in {"minimal", "documentary", "dynamic", "clean"}:
            raise ClipEngineError("production_render.subtitle_style: minimal, documentary, dynamic или clean.")
        if not isinstance(self.subtitle_font_family, str) or not self.subtitle_font_family.strip() or len(self.subtitle_font_family) > 160:
            raise ClipEngineError("production_render.subtitle_font_family должен быть непустой безопасной строкой.")
        if self.subtitle_language not in {"auto", "ru", "en"}:
            raise ClipEngineError("production_render.subtitle_language: auto, ru или en.")
        if not 8 <= self.subtitle_max_chars_per_line <= 80 or not 1 <= self.subtitle_max_lines <= 2:
            raise ClipEngineError("production_render subtitle_max_chars_per_line/max_lines вне допустимого диапазона.")
        if not 0.05 <= self.subtitle_min_duration <= self.subtitle_max_duration <= 12:
            raise ClipEngineError("production_render subtitle duration limits некорректны.")
        if not 4 <= self.subtitle_reading_speed_cps <= 40:
            raise ClipEngineError("production_render.subtitle_reading_speed_cps должен быть от 4 до 40.")
        if not 1 <= self.subtitle_min_words_per_cue <= self.subtitle_max_words_per_cue <= 30:
            raise ClipEngineError("production_render subtitle word-per-cue limits некорректны.")
        if not 0.60 <= self.subtitle_max_rendered_width_ratio <= 0.95:
            raise ClipEngineError("production_render.subtitle_max_rendered_width_ratio должен быть от 0.45 до 0.95.")
        if not 0.80 <= self.subtitle_min_font_scale <= 1.0:
            raise ClipEngineError("production_render.subtitle_min_font_scale должен быть от 0.80 до 1.0.")
        if not isinstance(self.subtitle_quality_version, str) or not self.subtitle_quality_version.strip():
            raise ClipEngineError("production_render.subtitle_quality_version не должен быть пустым.")
        if not 0 <= self.maximum_freeze_duration <= 5 or not 0 <= self.maximum_loop_duration <= 5:
            raise ClipEngineError("production_render maximum freeze/loop duration должны быть от 0 до 5.")
        if not 0.04 <= self.minimum_clip_duration <= 10:
            raise ClipEngineError("production_render.minimum_clip_duration должен быть от 0.04 до 10.")
        if not 1 <= self.maximum_speed_adjustment <= 1.25:
            raise ClipEngineError("production_render.maximum_speed_adjustment должен быть от 1 до 1.25.")
        if not 0 <= self.manual_crop_x <= 1 or not 0 <= self.manual_crop_y <= 1:
            raise ClipEngineError("production_render manual crop coordinates должны быть от 0 до 1.")
        if not 0 <= self.av_sync_warning_ms <= self.av_sync_error_ms <= 5000:
            raise ClipEngineError("production_render AV sync thresholds некорректны.")
        if not 0.01 <= self.maximum_duration_difference <= 10:
            raise ClipEngineError("production_render.maximum_duration_difference должен быть от 0.01 до 10.")
        if not self.render_config_version.strip():
            raise ClipEngineError("production_render.render_config_version не должен быть пустым.")


@dataclass(slots=True)
class ProductFlowConfig:
    """Resolved user intent recorded with the runtime configuration and report."""

    processing_mode: str = "standard"
    deep_analysis_requested: str = "auto"
    deep_analysis_resolved: bool = False
    deep_analysis_reason: str = "Не запрашивался."
    platform: str = "universal"
    clip_count: int = 3
    subtitle_preset: str = "documentary"
    audio_mode: str = "original"
    preset_version: str = "4B.1"

    def validate(self) -> None:
        if self.audio_mode not in {"original", "original_enhanced", "voiceover", "replace_voice", "mixed"}:
            raise ClipEngineError("product_flow.audio_mode must be a supported audio mode.")
        if self.processing_mode not in {"fast", "standard", "maximum"}:
            raise ClipEngineError("product_flow.processing_mode: fast, standard или maximum.")
        if self.deep_analysis_requested not in {"auto", "on", "off"}:
            raise ClipEngineError("product_flow.deep_analysis_requested: auto, on или off.")
        if not isinstance(self.deep_analysis_resolved, bool):
            raise ClipEngineError("product_flow.deep_analysis_resolved должен быть true или false.")
        if not isinstance(self.deep_analysis_reason, str) or not self.deep_analysis_reason.strip():
            raise ClipEngineError("product_flow.deep_analysis_reason не должен быть пустым.")
        if self.platform not in {"tiktok", "reels", "shorts", "universal"}:
            raise ClipEngineError("product_flow.platform содержит неподдерживаемую платформу.")
        if not 1 <= self.clip_count <= 5:
            raise ClipEngineError("product_flow.clip_count должен быть от 1 до 5.")
        if self.subtitle_preset not in {"minimal", "documentary", "dynamic", "clean"}:
            raise ClipEngineError("product_flow.subtitle_preset содержит неподдерживаемый стиль.")
        if not isinstance(self.preset_version, str) or not self.preset_version.strip():
            raise ClipEngineError("product_flow.preset_version не должен быть пустым.")


@dataclass(slots=True)
class AppConfig:
    whisper_model: str = "small"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    min_clip_duration: float = 15.0
    target_clip_duration: float = 35.0
    max_clip_duration: float = 60.0
    max_clips: int = 5
    score_threshold: int = 60
    overlap_threshold: float = 0.55
    pre_roll_seconds: float = 0.35
    post_roll_seconds: float = 0.35
    render_mode: str = "blur-background"
    subtitles_enabled: bool = True
    output_width: int = 1080
    output_height: int = 1920
    encoder_preference: str = "auto"
    delete_downloaded_source: bool = False
    ai: AIConfig = field(default_factory=AIConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    transcript_features: TranscriptFeatureConfig = field(default_factory=TranscriptFeatureConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    candidate_generation: CandidateGenerationConfig = field(default_factory=CandidateGenerationConfig)
    content_understanding: ContentUnderstandingConfig = field(default_factory=ContentUnderstandingConfig)
    virality: ViralityScoringConfig = field(default_factory=ViralityScoringConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ai_reranking: AIRerankingConfig = field(default_factory=AIRerankingConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio_composition: AudioCompositionConfig = field(default_factory=AudioCompositionConfig)
    production_render: ProductionRenderConfig = field(default_factory=ProductionRenderConfig)
    product_flow: ProductFlowConfig = field(default_factory=ProductFlowConfig)
    min_selected_clip_distance_seconds: float = 8.0
    optional_visual_features: bool = False
    # Compatibility flag for older local configurations; --mock-ai has priority.
    mock_ai: bool = False

    def validate(self) -> None:
        if not (0 < self.min_clip_duration <= self.target_clip_duration <= self.max_clip_duration):
            raise ClipEngineError(
                "Длительности клипа должны удовлетворять: minimum ≤ target ≤ maximum."
            )
        if self.max_clip_duration > 180:
            raise ClipEngineError("Максимальная длительность клипа не должна превышать 180 секунд.")
        if self.max_clips < 1:
            raise ClipEngineError("Количество клипов должно быть не меньше одного.")
        if not 0 <= self.score_threshold <= 100:
            raise ClipEngineError("Порог оценки должен быть от 0 до 100.")
        if not 0 <= self.overlap_threshold <= 1:
            raise ClipEngineError("Порог пересечения должен быть от 0 до 1.")
        if self.render_mode not in {"blur-background", "center-crop"}:
            raise ClipEngineError("Режим рендера: blur-background или center-crop.")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ClipEngineError("device: auto, cuda или cpu.")
        if self.encoder_preference not in {"auto", "nvenc", "cpu"}:
            raise ClipEngineError("encoder_preference: auto, nvenc или cpu.")
        self.ai.validate()
        self.vision.validate()
        self.transcript_features.validate()
        self.audio_analysis.validate()
        self.scene_detection.validate()
        self.candidate_generation.validate()
        self.content_understanding.validate()
        self.virality.validate()
        self.scoring.validate()
        self.ai_reranking.validate()
        self.transformation.validate()
        self.production.validate()
        self.tts.validate()
        self.audio_composition.validate()
        self.production_render.validate()
        self.product_flow.validate()
        if not 0 <= self.min_selected_clip_distance_seconds <= 600:
            raise ClipEngineError("min_selected_clip_distance_seconds должен быть от 0 до 600.")
        if not isinstance(self.optional_visual_features, bool):
            raise ClipEngineError("optional_visual_features должен быть true или false.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | None = None) -> AppConfig:
    values: dict[str, Any] = {}
    if path is not None:
        if not path.exists():
            raise ClipEngineError(f"Файл конфигурации не найден: {path}")
        try:
            import yaml
        except ImportError as error:
            raise ClipEngineError(
                "Для чтения config.yaml установите зависимости: pip install -r requirements.txt"
            ) from error
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ClipEngineError("Конфигурация должна быть YAML-объектом key: value.")
        allowed = {item.name for item in fields(AppConfig)}
        unknown = sorted(set(loaded) - allowed)
        if unknown:
            raise ClipEngineError(f"Неизвестные параметры config.yaml: {', '.join(unknown)}")
        values = dict(loaded)
        nested = {
            "ai": AIConfig,
            "vision": VisionConfig,
            "transcript_features": TranscriptFeatureConfig,
            "audio_analysis": AudioAnalysisConfig,
            "scene_detection": SceneDetectionConfig,
            "candidate_generation": CandidateGenerationConfig,
            "content_understanding": ContentUnderstandingConfig,
            "virality": ViralityScoringConfig,
            "scoring": ScoringConfig,
            "ai_reranking": AIRerankingConfig,
            "transformation": TransformationConfig,
            "production": ProductionConfig,
            "tts": TTSConfig,
            "audio_composition": AudioCompositionConfig,
            "production_render": ProductionRenderConfig,
            "product_flow": ProductFlowConfig,
        }
        for name, config_type in nested.items():
            nested_values = values.get(name)
            if nested_values is None:
                continue
            if not isinstance(nested_values, dict):
                raise ClipEngineError(f"{name} в config.yaml должен быть YAML-объектом key: value.")
            nested_allowed = {item.name for item in fields(config_type)}
            nested_unknown = sorted(set(nested_values) - nested_allowed)
            if nested_unknown:
                raise ClipEngineError(
                    f"Неизвестные параметры {name} в config.yaml: {', '.join(nested_unknown)}"
                )
            values[name] = config_type(**nested_values)
    config = AppConfig(**values)
    config.validate()
    return config
