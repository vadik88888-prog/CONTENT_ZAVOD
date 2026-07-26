from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.audio_service import AudioCompositionService
from app.cli import build_parser
from app.errors import ProductionRenderError
from app.pipeline import Pipeline, StageTracker
from app.production_subtitles import build_subtitle_project, resolve_subtitle_style, split_subtitle_text, write_production_ass
from app.sources import local_source
from app.video_composition import (
    VideoCompositionService,
    build_reframe_plan,
    build_video_timeline,
    make_crop_plan,
    probe_media,
    production_render_report_section,
    _ffmpeg,
    _timeline_filter,
)
from app.video_models import CanvasConfig, CropPlan, RenderValidation, SourceVideoClip, VideoTimeline
from tests.test_audio_composition import _audio_config, _plan, _tts_result


def _source_video(path: Path, *, width: int = 320, height: int = 180) -> Path:
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("ffmpeg is required for production video composition tests")
    subprocess.run([
        executable, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"testsrc2=size={width}x{height}:rate=30", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ], check=True)
    return path


def _upstream(tmp_path: Path):
    config = _audio_config()
    config.production_render.enabled = True
    config.production_render.output_width = 180
    config.production_render.output_height = 320
    config.production_render.output_fps = 30
    config.production_render.video_bitrate = "500k"
    config.production_render.encoder = "cpu"
    config.validate()
    plan = _plan()
    source_path = _source_video(tmp_path / "source.mp4")
    source = local_source(str(source_path))
    transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Source dialogue remains audible."}]}
    audio = AudioCompositionService(tmp_path, config).compose(
        plan, source, transcript, _tts_result(tmp_path, config, plan), tmp_path / "work", tmp_path / "out",
    )
    return config, plan, source, transcript, audio


def test_video_models_validate_canvas_crop_and_timeline_ranges() -> None:
    canvas = CanvasConfig(width=180, height=320, fps=30)
    crop = CropPlan(strategy="center_crop", source_width=320, source_height=180, crop_width=100, crop_height=180, crop_x=110, crop_y=0)
    clip = SourceVideoClip(
        clip_id="clip-001", order=1, timeline_start_seconds=0, timeline_end_seconds=1,
        duration_seconds=1, source_path="source.mp4", source_start_seconds=0, source_end_seconds=1,
        visual_strategy="mapped_source", crop_plan=crop, status="ready",
    )
    assert VideoTimeline(clips=[clip], duration_seconds=1).duration_seconds == 1
    with pytest.raises(ValidationError):
        CanvasConfig(width=181, height=320, fps=30)
    with pytest.raises(ValidationError):
        CropPlan(strategy="center_crop", source_width=100, source_height=100, crop_width=80, crop_height=80, crop_x=30, crop_y=0)
    with pytest.raises(ValidationError):
        SourceVideoClip(**{**clip.model_dump(), "timeline_end_seconds": 0.5})


def test_missing_ffmpeg_and_ffprobe_fail_with_clear_render_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.video_composition.shutil.which", lambda _name: None)
    with pytest.raises(ProductionRenderError, match="ffprobe"):
        probe_media(Path("source.mp4"), require_video=True)
    with pytest.raises(ProductionRenderError, match="ffmpeg"):
        _ffmpeg()


def test_timeline_maps_actual_audio_dialogue_and_uses_deterministic_pause_fallback(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    timeline, fallbacks = build_video_timeline(
        plan, audio, transcript, source.path, probe_media(source.path, require_video=True),
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    dialogue = next(clip for clip in timeline.clips if clip.production_segment_id == "dialogue-001")
    assert dialogue.source_start_seconds == 1 and dialogue.source_end_seconds == 2
    assert timeline.duration_seconds == audio.timeline.duration_seconds
    assert fallbacks and all("silence" in reason for reason in fallbacks)
    assert [clip.order for clip in timeline.clips] == list(range(1, len(timeline.clips) + 1))


def test_prepared_source_clip_keeps_freeze_padding_duration(tmp_path: Path) -> None:
    config, _plan_value, source, _transcript, _audio = _upstream(tmp_path)
    canvas = CanvasConfig(width=180, height=320, fps=30)
    clip = SourceVideoClip(
        clip_id="visual-freeze", order=1, timeline_start_seconds=0, timeline_end_seconds=2,
        duration_seconds=2, source_path=str(source.path), source_start_seconds=0,
        source_end_seconds=1.4, visual_strategy="mapped_source",
        crop_plan=make_crop_plan(probe_media(source.path, require_video=True), canvas, config.production_render),
        freeze_duration_seconds=0.6, status="fallback", fallback_reason="source_clip_short_freeze",
    )
    destination = tmp_path / "freeze-padded.mp4"

    VideoCompositionService(tmp_path, config)._prepare_visual_clip(clip, canvas, destination)

    actual = float(probe_media(destination, require_video=True)["video_duration"])
    assert abs(actual - clip.duration_seconds) <= 0.15


def test_missing_or_invalid_mapping_is_reported_without_random_clip_selection(tmp_path: Path) -> None:
    config, plan, source, _transcript, audio = _upstream(tmp_path)
    raw = plan.model_dump(mode="json")
    raw["segments"][1]["source_end_seconds"] = 99
    raw["dialogue_mappings"][0]["source_end_seconds"] = 99
    broken = type(plan).model_validate(raw)
    first, reasons_a = build_video_timeline(
        broken, audio, {}, source.path, probe_media(source.path, require_video=True),
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    second, reasons_b = build_video_timeline(
        broken, audio, {}, source.path, probe_media(source.path, require_video=True),
        CanvasConfig(width=180, height=320, fps=30), config.production_render,
    )
    assert reasons_a == reasons_b and reasons_a
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert all(clip.status in {"ready", "fallback", "placeholder"} for clip in first.clips)


@pytest.mark.parametrize(("width", "height", "strategy"), [
    (320, 180, "center_crop"), (180, 320, "top_crop"), (240, 240, "fit_blur_background"),
])
def test_crop_plans_stay_inside_horizontal_vertical_and_square_sources(width: int, height: int, strategy: str) -> None:
    from app.config import ProductionRenderConfig

    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy=strategy)
    config.validate()
    plan = make_crop_plan(
        {"display_width": width, "display_height": height, "rotation": 0}, CanvasConfig(width=180, height=320, fps=30), config,
    )
    if plan.crop_width is not None:
        assert plan.crop_x + plan.crop_width <= width
        assert plan.crop_y + plan.crop_height <= height


def test_reframe_plan_uses_safe_fallback_and_smooths_subject_keyframes() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="center_crop")
    fallback = build_reframe_plan(
        {"display_width": 320, "display_height": 180}, canvas, config, VideoTimeline(clips=[], duration_seconds=0),
    )
    assert fallback.strategy == "center_crop"
    assert fallback.subject_detection_used is False
    tracked = build_reframe_plan(
        {
            "display_width": 320, "display_height": 180,
            "subject_keyframes": [
                {"time_seconds": 0, "normalized_x": 0.15, "normalized_y": 0.3, "confidence": 0.9},
                {"time_seconds": 1, "normalized_x": 0.95, "normalized_y": 0.9, "confidence": 0.9},
            ],
        }, canvas, config, VideoTimeline(clips=[], duration_seconds=0),
    )
    assert tracked.strategy == "subject_crop" and tracked.subject_detection_used
    assert tracked.keyframes[1].normalized_x - tracked.keyframes[0].normalized_x <= 0.16
    assert tracked.keyframes[1].normalized_y - tracked.keyframes[0].normalized_y <= 0.12


def test_safe_auto_reframe_preserves_unknown_landscape_and_uses_subject_when_confident() -> None:
    from app.config import ProductionRenderConfig

    canvas = CanvasConfig(width=180, height=320, fps=30)
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="safe_auto")
    source = {"display_width": 1920, "display_height": 1080, "rotation": 0}
    crop = make_crop_plan(source, canvas, config)
    reframe = build_reframe_plan(source, canvas, config, VideoTimeline(clips=[], duration_seconds=0))

    assert crop.strategy == "fit_blur_background"
    assert reframe.strategy == "blur_fallback"
    assert reframe.fallback_reason and "unsafe crop" in reframe.fallback_reason

    tracked_source = {
        **source,
        "subject_keyframes": [{"time_seconds": 0, "normalized_x": 0.8, "normalized_y": 0.4, "confidence": 0.9}],
    }
    tracked_crop = make_crop_plan(tracked_source, canvas, config)
    tracked_reframe = build_reframe_plan(tracked_source, canvas, config, VideoTimeline(clips=[], duration_seconds=0))
    assert tracked_crop.strategy == "manual_normalized_crop"
    assert tracked_reframe.strategy == "subject_crop" and tracked_reframe.subject_detection_used


def test_subject_anchor_changes_the_actual_horizontal_crop() -> None:
    from app.config import ProductionRenderConfig

    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="center_crop")
    canvas = CanvasConfig(width=180, height=320, fps=30)
    crop = make_crop_plan(
        {"display_width": 1280, "display_height": 720, "rotation": 0, "subject_keyframes": [{"normalized_x": 0.8, "normalized_y": 0.4, "confidence": 0.9}]},
        canvas, config,
    )
    assert crop.strategy == "manual_normalized_crop"
    assert crop.crop_x is not None and crop.crop_x > (1280 - (720 * 9 / 16)) / 2


def test_rotation_metadata_is_reflected_in_safe_display_crop(tmp_path: Path) -> None:
    executable = shutil.which("ffmpeg")
    if not executable:
        pytest.skip("ffmpeg is required for rotation metadata test")
    source = _source_video(tmp_path / "source.mp4", width=320, height=180)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run([
        executable, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-c", "copy",
        "-metadata:s:v:0", "rotate=90", str(rotated),
    ], check=True)
    info = probe_media(rotated, require_video=True)
    assert info["rotation"] in {0, 90, 270}
    if info["rotation"] in {90, 270}:
        assert info["display_width"] == info["height"] and info["display_height"] == info["width"]
    from app.config import ProductionRenderConfig
    config = ProductionRenderConfig(output_width=180, output_height=320, crop_strategy="center_crop")
    config.validate()
    crop = make_crop_plan(info, CanvasConfig(width=180, height=320, fps=30), config)
    assert crop.crop_width is None or crop.crop_x + crop.crop_width <= crop.source_width


def test_subtitles_use_audio_timeline_split_long_text_and_escape_ass(tmp_path: Path) -> None:
    config, plan, _source, _transcript, audio = _upstream(tmp_path)
    raw = plan.model_dump(mode="json")
    raw["segments"][0]["text"] = "{Long} narration text is split deterministically into readable subtitle groups without rewriting the original words. " * 2
    raw["segments"][0]["word_count"] = len(raw["segments"][0]["text"].split())
    long_plan = type(plan).model_validate(raw)
    project = build_subtitle_project(long_plan, audio, config.production_render)
    assert len(split_subtitle_text(raw["segments"][0]["text"], config.production_render)) > 1
    assert len(split_subtitle_text("x" * 500, config.production_render)) > 1
    assert all(left.end_seconds <= right.start_seconds for left, right in zip(project.cues, project.cues[1:]))
    ass = tmp_path / "production-subtitles.ass"
    write_production_ass(project, ass, 180, 320)
    content = ass.read_text(encoding="utf-8-sig")
    assert r"\{" in content and "Dialogue:" in content
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Production,"))
    metrics = style_line.split(",")
    assert metrics[2] == "12" and metrics[16:20] == ["1", "0", "2", "12"]
    config.production_render.subtitle_font_family = "__missing_font__"
    _style, fallback, warning = resolve_subtitle_style(config.production_render)
    assert fallback and warning and "Arial" in warning


def test_dynamic_subtitles_use_word_level_highlighting_and_keep_cyrillic(tmp_path: Path) -> None:
    config, plan, _source, _transcript, audio = _upstream(tmp_path)
    config.production_render.subtitle_style = "dynamic"
    raw = plan.model_dump(mode="json")
    raw["segments"][0]["text"] = "Это заметный динамический заголовок"
    raw["segments"][0]["word_count"] = 4
    dynamic_plan = type(plan).model_validate(raw)
    project = build_subtitle_project(dynamic_plan, audio, config.production_render)
    ass = tmp_path / "dynamic.ass"
    write_production_ass(project, ass, 180, 320)
    content = ass.read_text(encoding="utf-8-sig")
    assert "&H004AD5FF" in content  # #FFD54A in ASS BGR order
    assert r"{\k" in content
    assert "ДИНАМИЧЕСКИЙ" in content


def test_dialogue_subtitles_use_source_word_timestamps_when_available(tmp_path: Path) -> None:
    config, plan, _source, _transcript, audio = _upstream(tmp_path)
    config.production_render.subtitle_style = "dynamic"
    transcript = {
        "words": [
            {"start": 1.0, "end": 1.2, "text": "Source"},
            {"start": 1.2, "end": 1.45, "text": "dialogue"},
            {"start": 1.45, "end": 1.7, "text": "remains"},
            {"start": 1.7, "end": 2.0, "text": "audible."},
        ]
    }

    project = build_subtitle_project(plan, audio, config.production_render, transcript)
    cue = next(item for item in project.cues if item.source_type == "dialogue")
    dialogue_clip = next(item for item in audio.timeline.clips if item.clip_type == "dialogue")
    assert [item.text for item in cue.word_timings] == ["Source", "dialogue", "remains", "audible."]
    assert cue.word_timings[0].start_seconds == pytest.approx(dialogue_clip.timeline_start_seconds)
    ass = tmp_path / "dialogue-timing.ass"
    write_production_ass(project, ass, 180, 320)
    assert r"{\k20}SOURCE" in ass.read_text(encoding="utf-8-sig")


def test_transition_filters_preserve_audio_timeline_duration() -> None:
    crossfade, label = _timeline_filter([1.0, 2.0, 1.0], "short_crossfade")
    assert "xfade" in crossfade and "tpad" in crossfade and label == "[vconcat]"
    fade, label = _timeline_filter([1.0], "fade_from_black")
    assert "fade=t=in" in fade and label == "[vfaded]"


def test_short_crossfade_renders_without_changing_final_duration(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.transitions = "short_crossfade"
    result = VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert result.result and result.result.validation.status == "valid"
    assert abs(result.actual_duration_seconds - audio.mix.duration_seconds) < 0.15


def test_render_cpu_mux_subtitles_cache_and_secret_free_report(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    first = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert first.status == "completed"
    assert first.result and first.result.validation.status == "valid"
    assert Path(first.result.output_file or "").is_file()
    assert first.metadata.ai_called is False and first.metadata.tts_regenerated is False and first.metadata.audio_remixed is False
    second = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert second.result and second.result.cache_hit
    cache = tmp_path / "work" / "production-render-cache" / f"{second.metadata.cache_key}.json"
    cache.write_text("{corrupt", encoding="utf-8")
    rebuilt = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert rebuilt.result and rebuilt.result.cache_hit is False
    report = production_render_report_section(second)
    assert report["cache_hit"] and "sk-" not in json.dumps(report)
    assert report["quality"]["status"] == "passed"


def test_auto_encoder_falls_back_to_cpu_when_nvenc_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.encoder = "auto"
    monkeypatch.setattr("app.video_composition.nvenc_available", lambda: False)
    result = VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert result.result and result.result.encoder == "libx264" and result.result.hardware_fallback


def test_explicit_unavailable_nvenc_fails_without_creating_false_final(tmp_path: Path, monkeypatch) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.encoder = "nvenc"
    monkeypatch.setattr("app.video_composition.nvenc_available", lambda: False)
    with pytest.raises(ProductionRenderError, match="nvenc"):
        VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert not (tmp_path / "out" / "production-render" / "final-short.mp4").exists()


def test_subtitle_style_change_rerenders_without_changing_audio(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    first = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    audio_checksum = Path(audio.mix.mixed_audio_path or "").read_bytes()
    config.production_render.subtitle_style = "clean"
    second = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert first.result and second.result and second.result.cache_hit is False
    assert Path(audio.mix.mixed_audio_path or "").read_bytes() == audio_checksum


def test_subtitle_disabled_still_exports_ass_artifact_without_burning_it_into_mp4(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    config.production_render.subtitles_enabled = False
    result = VideoCompositionService(tmp_path, config).compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    assert result.result and result.result.validation.status == "valid"
    root = tmp_path / "out" / "production-render"
    assert (root / "production-subtitles.ass").is_file()
    assert result.render_request.subtitles_enabled is False


def test_atomic_invalid_final_keeps_previous_mp4(tmp_path: Path, monkeypatch) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    service = VideoCompositionService(tmp_path, config)
    first = service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out")
    final = Path(first.result.output_file or "")
    checksum = final.read_bytes()
    monkeypatch.setattr(
        "app.video_composition.validate_final_video",
        lambda *args, **kwargs: RenderValidation(status="invalid", messages=["forced invalid"]),
    )
    with pytest.raises(ProductionRenderError, match="validation"):
        service.compose(plan, audio, source, transcript, tmp_path / "work", tmp_path / "out", force_recompute=True)
    assert final.read_bytes() == checksum


def test_invalid_mixed_audio_and_cli_production_flags_are_clear(tmp_path: Path) -> None:
    config, plan, source, transcript, audio = _upstream(tmp_path)
    broken = audio.model_copy(update={"mix": audio.mix.model_copy(update={"mixed_audio_path": str(tmp_path / "missing.wav")})})
    with pytest.raises(ProductionRenderError, match="mixed_audio"):
        VideoCompositionService(tmp_path, config).compose(plan, broken, source, transcript, tmp_path / "work", tmp_path / "out")
    arguments = build_parser().parse_args([
        "process", "--input", "source.mp4", "--production-render-only", "--recompute-production-render",
        "--disable-subtitles", "--crop-strategy", "center_crop", "--subtitle-style", "clean", "--video-encoder", "cpu",
    ])
    assert arguments.production_render_only and arguments.recompute_production_render and arguments.disable_subtitles
    disabled = build_parser().parse_args(["process", "--input", "source.mp4", "--disable-production-render"])
    assert disabled.disable_production_render


def test_production_render_only_preserves_existing_legacy_mp4(tmp_path: Path) -> None:
    config, plan, source, transcript, _audio = _upstream(tmp_path)
    pipeline = Pipeline(tmp_path, config, production_render_only=True)
    prepared_source, work_directory, output_directory = pipeline._prepare_source(str(source.path), None)
    tts = _tts_result(tmp_path, config, plan)
    audio = AudioCompositionService(tmp_path, config).compose(
        plan, prepared_source, transcript, tts, work_directory, output_directory,
    )
    assert audio.export.path
    from app.utils import write_json
    write_json(output_directory / "production-plan.json", plan.model_dump(mode="json"))
    write_json(work_directory / "transcript.json", transcript)
    old_mp4 = output_directory / "legacy.mp4"
    old_mp4.write_bytes(b"legacy-video")
    write_json(output_directory / "report.json", {"output_files": [str(old_mp4)], "selected_clips_count": 1})
    result = pipeline._run_production_render_only(StageTracker(work_directory / "state.json"), prepared_source, work_directory, output_directory)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert old_mp4.read_bytes() == b"legacy-video"
    assert result.output_files and result.output_files[0].name == "final-short.mp4"
    assert report["production_render"]["ai_called"] is False
