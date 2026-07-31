import pytest

from app.config import AppConfig, ViralityScoringConfig
from app.errors import ClipEngineError


def test_default_configuration_is_valid() -> None:
    AppConfig().validate()


def test_invalid_clip_duration_order_is_rejected() -> None:
    with pytest.raises(ClipEngineError, match="Длительности"):
        AppConfig(min_clip_duration=40, target_clip_duration=30).validate()


def test_unknown_render_mode_is_rejected() -> None:
    with pytest.raises(ClipEngineError, match="Режим рендера"):
        AppConfig(render_mode="face-tracking").validate()


def test_invalid_tts_provider_and_duration_limits_are_rejected() -> None:
    config = AppConfig()
    config.tts.provider = "unknown"
    with pytest.raises(ClipEngineError, match="tts.provider"):
        config.validate()
    config = AppConfig()
    config.tts.maximum_segment_duration = 0
    with pytest.raises(ClipEngineError, match="maximum_segment_duration"):
        config.validate()


def test_content_understanding_versions_are_validated() -> None:
    config = AppConfig()
    config.content_understanding.strategy_version = ""
    with pytest.raises(ClipEngineError, match="strategy_version"):
        config.validate()


def test_diversity_policy_requires_versions_and_normalized_lambda() -> None:
    config = AppConfig()
    config.content_understanding.diversity_config_version = ""
    with pytest.raises(ClipEngineError, match="diversity_config_version"):
        config.validate()

    config = AppConfig()
    config.content_understanding.diversity_lambda = 1.01
    with pytest.raises(ClipEngineError, match="diversity_lambda"):
        config.validate()


def test_virality_policy_validates_complete_normalized_weights() -> None:
    config = AppConfig()
    config.virality.weights["hook"] = 0.2
    with pytest.raises(ClipEngineError, match="Сумма virality.weights"):
        config.validate()

    policy = ViralityScoringConfig(semantic_ai_mode="invalid")
    with pytest.raises(ClipEngineError, match="semantic_ai_mode"):
        policy.validate()

    policy = ViralityScoringConfig()
    policy.strategy_weights.pop("generic_dialogue")
    with pytest.raises(ClipEngineError, match="strategy_weights"):
        policy.validate()
