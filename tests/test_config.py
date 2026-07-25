import pytest

from app.config import AppConfig
from app.errors import ClipEngineError


def test_default_configuration_is_valid() -> None:
    AppConfig().validate()


def test_invalid_clip_duration_order_is_rejected() -> None:
    with pytest.raises(ClipEngineError, match="Длительности"):
        AppConfig(min_clip_duration=40, target_clip_duration=30).validate()


def test_unknown_render_mode_is_rejected() -> None:
    with pytest.raises(ClipEngineError, match="Режим рендера"):
        AppConfig(render_mode="face-tracking").validate()
