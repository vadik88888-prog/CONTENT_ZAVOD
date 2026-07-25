from app.config import AppConfig
from app.rendering import _filters


def test_blurred_render_filter_has_vertical_canvas_and_subtitles(tmp_path) -> None:
    subtitles = tmp_path / "clip.ass"
    result = _filters(AppConfig(), subtitles)

    assert "scale=1080:1920" in result
    assert "boxblur" in result
    assert "ass=" in result
    assert result.endswith("[vout]")


def test_center_crop_filter_does_not_blur() -> None:
    result = _filters(AppConfig(render_mode="center-crop"), None)

    assert "crop=1080:1920" in result
    assert "boxblur" not in result
    assert result.endswith("[vout]")
