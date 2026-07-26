from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.audio_models import AudioProject
from app.config import AppConfig, ProductionRenderConfig
from app.errors import ProductionRenderError
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan
from app.production_subtitles import build_subtitle_project, write_production_ass
from app.rendering import nvenc_available
from app.sources import Source
from app.utils import read_json, stable_file_hash, stable_text_hash, utc_now, write_bytes_atomic, write_json
from app.video_models import (
    CanvasConfig,
    CropPlan,
    FillClip,
    FreezeFrameClip,
    RenderArtifact,
    RenderError,
    RenderMetadata,
    RenderRequest,
    RenderResult,
    RenderValidation,
    ReframeKeyframe,
    ReframePlan,
    SourceVideoClip,
    SubtitleProject,
    VideoClipModel,
    VideoProject,
    VideoTimeline,
    VideoTrack,
    VideoTransition,
)


PRODUCTION_RENDER_ENGINE_VERSION = "3D.1"


class VideoCompositionService:
    """Goal 3D executor. It only consumes existing production artifacts and source media."""

    def __init__(self, root: Path, config: AppConfig) -> None:
        self.root = root.resolve()
        self.config = config

    def compose(
        self, plan: ProductionPlan, audio_project: AudioProject, source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path,
        force_recompute: bool = False, visual_analysis: dict[str, Any] | None = None,
    ) -> VideoProject:
        render_config = self.config.production_render
        if not source.path.is_file():
            raise ProductionRenderError("Исходный video file для production render не найден.")
        source_info = probe_media(source.path, require_video=True)
        if visual_analysis and isinstance(visual_analysis.get("subject_keyframes"), list):
            source_info["subject_keyframes"] = visual_analysis["subject_keyframes"]
        mixed_path = Path(audio_project.mix.mixed_audio_path or "")
        if not mixed_path.is_file():
            raise ProductionRenderError("mixed_audio.wav не найден для production render.")
        mixed_info = probe_media(mixed_path, require_audio=True)
        source_checksum = stable_file_hash(source.path)
        mixed_checksum = stable_file_hash(mixed_path)
        canvas = CanvasConfig(
            width=render_config.output_width, height=render_config.output_height,
            fps=render_config.output_fps, pixel_format=render_config.pixel_format,
        )
        timeline, fallback_reasons = build_video_timeline(
            plan, audio_project, transcript, source.path, source_info, canvas, render_config,
        )
        reframe_plan = build_reframe_plan(source_info, canvas, render_config, timeline)
        actual_audio_duration = float(mixed_info["audio_duration"])
        if abs(timeline.duration_seconds - actual_audio_duration) > render_config.maximum_duration_difference:
            raise ProductionRenderError(
                "AudioProject timeline и mixed_audio.wav имеют несовместимую длительность: "
                f"{timeline.duration_seconds:.3f}s vs {actual_audio_duration:.3f}s."
            )
        subtitle_project = build_subtitle_project(plan, audio_project, render_config)
        cache_key = _render_cache_key(
            source_checksum, mixed_checksum, audio_project, timeline, subtitle_project, canvas, render_config,
        )
        project_id = f"video-{plan.plan_id}-{audio_project.project_id}-{cache_key[:12]}"
        request = RenderRequest(
            video_project_id=project_id, source_path=str(source.path), mixed_audio_path=str(mixed_path),
            canvas=canvas, encoder_preference=render_config.encoder, video_bitrate=render_config.video_bitrate,
            subtitles_enabled=render_config.subtitles_enabled,
        )
        metadata = RenderMetadata(
            production_plan_id=plan.plan_id, audio_project_id=audio_project.project_id,
            source_checksum=source_checksum, mixed_audio_checksum=mixed_checksum,
            render_config_version=render_config.render_config_version, cache_key=cache_key,
            created_at=utc_now(), updated_at=utc_now(),
        )
        track = VideoTrack(
            track_id="track-visual", clips=timeline.clips,
            status="fallback" if fallback_reasons else "ready",
        )
        project = VideoProject(
            project_id=project_id, status="skipped", source_video_path=str(source.path), source_checksum=source_checksum,
            production_plan_id=plan.plan_id, audio_project_id=audio_project.project_id, mixed_audio_path=str(mixed_path),
            canvas=canvas, target_duration_seconds=timeline.duration_seconds, actual_duration_seconds=0,
            timeline=timeline, reframe_plan=reframe_plan, tracks=[track], subtitle_project=subtitle_project, render_request=request,
            metadata=metadata, warnings=list(subtitle_project.warnings), fallback_reasons=fallback_reasons,
        )
        render_root = output_directory / "production-render"
        cache_path = self.root / "work" / "production-render-cache" / f"{cache_key}.json"
        cached = self._try_cache(project, render_root, cache_path, force_recompute)
        if cached is not None:
            return cached
        return self._render(project, source_info, mixed_info, render_root, cache_path)

    def _try_cache(
        self, project: VideoProject, render_root: Path, cache_path: Path, force_recompute: bool,
    ) -> VideoProject | None:
        if not self.config.production_render.cache_enabled or force_recompute:
            return None
        try:
            cached = read_json(cache_path, {})
        except (OSError, json.JSONDecodeError):
            return None
        final_path = render_root / "final-short.mp4"
        if not isinstance(cached, dict) or cached.get("cache_key") != project.metadata.cache_key:
            return None
        if not final_path.is_file() or cached.get("checksum") != stable_file_hash(final_path):
            return None
        try:
            validation = validate_final_video(final_path, project.canvas, Path(project.mixed_audio_path), self.config.production_render)
        except ProductionRenderError:
            return None
        if validation.status == "invalid":
            return None
        artifacts = _artifacts(render_root, final_path)
        result = RenderResult(
            status="warning" if validation.status == "warning" else "completed", output_file=str(final_path),
            encoder=str(cached.get("encoder") or "cache"), hardware_fallback=bool(cached.get("hardware_fallback", False)),
            cache_hit=True, validation=validation, artifacts=artifacts,
            warnings=[*project.warnings, *validation.messages],
        )
        complete = project.model_copy(update={
            "status": result.status, "actual_duration_seconds": validation.video_duration_seconds or 0,
            "result": result, "artifact_paths": [item.path for item in artifacts],
            "metadata": project.metadata.model_copy(update={"updated_at": utc_now()}),
        })
        _write_project_artifacts(complete, render_root)
        return complete

    def _render(
        self, project: VideoProject, source_info: dict[str, Any], mixed_info: dict[str, Any],
        render_root: Path, cache_path: Path,
    ) -> VideoProject:
        clips_root = render_root / "clips"
        temp_root = render_root / "temp"
        render_root.mkdir(parents=True, exist_ok=True)
        clips_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        ass_path = render_root / "production-subtitles.ass"
        if project.subtitle_project is not None:
            write_production_ass(project.subtitle_project, ass_path, project.canvas.width, project.canvas.height)
        prepared: list[Path] = []
        temporary: Path | None = None
        try:
            for clip in project.timeline.clips:
                destination = clips_root / f"{clip.order:03d}-{clip.clip_id}.mp4"
                self._prepare_visual_clip(clip, project.canvas, destination)
                prepared.append(destination)
            temporary = _temporary_path(temp_root, ".mp4")
            encoder, hardware_fallback, encoder_warning = self._mux_final(
                prepared, Path(project.mixed_audio_path), ass_path if self.config.production_render.subtitles_enabled else None,
                temporary, project.canvas, [clip.duration_seconds for clip in project.timeline.clips],
            )
            validation = validate_final_video(temporary, project.canvas, Path(project.mixed_audio_path), self.config.production_render)
            if validation.status == "invalid":
                temporary.unlink(missing_ok=True)
                raise ProductionRenderError("Final MP4 не прошёл обязательную ffprobe validation.")
            final_path = render_root / "final-short.mp4"
            temporary.replace(final_path)
        except ProductionRenderError:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        except Exception as error:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise ProductionRenderError(f"Production render failed: {_safe_error(error)}") from error
        warnings = [*project.warnings, *validation.messages]
        if encoder_warning:
            warnings.append(encoder_warning)
        result = RenderResult(
            status="warning" if warnings or validation.status == "warning" else "completed",
            output_file=str(final_path), encoder=encoder, hardware_fallback=hardware_fallback,
            cache_hit=False, validation=validation, artifacts=[], warnings=warnings,
        )
        complete = project.model_copy(update={
            "status": result.status, "actual_duration_seconds": validation.video_duration_seconds or 0,
            "result": result, "warnings": warnings,
            "metadata": project.metadata.model_copy(update={"updated_at": utc_now()}),
        })
        _write_project_artifacts(complete, render_root)
        artifacts = _artifacts(render_root, final_path, prepared)
        result = result.model_copy(update={"artifacts": artifacts})
        complete = complete.model_copy(update={
            "result": result, "artifact_paths": [item.path for item in artifacts],
            "metadata": complete.metadata.model_copy(update={"updated_at": utc_now()}),
        })
        _write_project_artifacts(complete, render_root)
        write_json(cache_path, {
            "schema_version": PRODUCTION_RENDER_ENGINE_VERSION, "cache_key": complete.metadata.cache_key,
            "checksum": stable_file_hash(final_path), "encoder": encoder,
            "hardware_fallback": hardware_fallback, "created_at": utc_now(),
        })
        return complete

    def _prepare_visual_clip(self, clip: VideoClipModel, canvas: CanvasConfig, destination: Path) -> None:
        ffmpeg = _ffmpeg()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(clip, FillClip):
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", f"color=c=0x161616:s={canvas.width}x{canvas.height}:r={canvas.fps}",
                "-t", f"{clip.duration_seconds:.6f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", canvas.pixel_format, "-movflags", "+faststart", str(destination),
            ]
            _run_ffmpeg(command, "fill clip")
            return
        source = Path(clip.source_path)
        if not source.is_file() or clip.source_start_seconds is None or clip.source_end_seconds is None:
            raise ProductionRenderError(f"Visual clip {clip.clip_id} has no usable source media.")
        available = max(0.04, clip.source_end_seconds - clip.source_start_seconds)
        filter_graph = _visual_filter(clip, canvas)
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{clip.source_start_seconds:.6f}",
            # -t must be an input option.  As an output option it truncates the
            # filtered stream and removes a legitimate tpad freeze extension.
            "-t", f"{available:.6f}", "-i", str(source), "-filter_complex", filter_graph,
            "-map", "[vout]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", canvas.pixel_format, "-movflags", "+faststart", str(destination),
        ]
        _run_ffmpeg(command, f"visual clip {clip.clip_id}")
        rendered = probe_media(destination, require_video=True)
        if abs(float(rendered["video_duration"]) - clip.duration_seconds) > 0.15:
            raise ProductionRenderError(f"Prepared visual clip {clip.clip_id} has an invalid duration.")

    def _mux_final(
        self, clips: list[Path], mixed_audio: Path, ass_path: Path | None,
        destination: Path, canvas: CanvasConfig, durations: list[float],
    ) -> tuple[str, bool, str | None]:
        if not clips:
            raise ProductionRenderError("Video timeline has no renderable visual clips.")
        ffmpeg = _ffmpeg()
        inputs: list[str] = []
        for clip in clips:
            inputs.extend(["-i", str(clip)])
        inputs.extend(["-i", str(mixed_audio)])
        graph, video_label = _timeline_filter(durations, self.config.production_render.transitions)
        if ass_path is not None:
            graph += f";{video_label}ass='{_filter_path(ass_path)}'[vout]"
        else:
            graph += f";{video_label}null[vout]"
        requested = self.config.production_render.encoder
        if requested == "cpu":
            _run_ffmpeg(_final_command(ffmpeg, inputs, graph, len(clips), destination, canvas, "libx264", self.config.production_render.video_bitrate), "final production render")
            return "libx264", False, None
        if requested == "nvenc" and not nvenc_available():
            raise ProductionRenderError("Запрошен production_render.encoder=nvenc, но h264_nvenc недоступен.")
        if requested == "auto" and not nvenc_available():
            _run_ffmpeg(_final_command(ffmpeg, inputs, graph, len(clips), destination, canvas, "libx264", self.config.production_render.video_bitrate), "final production render")
            return "libx264", True, "NVENC недоступен: production render безопасно выполнен на CPU."
        try:
            _run_ffmpeg(_final_command(ffmpeg, inputs, graph, len(clips), destination, canvas, "h264_nvenc", self.config.production_render.video_bitrate), "final production render")
            return "h264_nvenc", False, None
        except ProductionRenderError as error:
            if requested == "nvenc":
                raise
            _run_ffmpeg(_final_command(ffmpeg, inputs, graph, len(clips), destination, canvas, "libx264", self.config.production_render.video_bitrate), "final production render CPU fallback")
            return "libx264", True, f"NVENC render failed; CPU fallback used: {_safe_error(error)}"


def build_video_timeline(
    plan: ProductionPlan, audio_project: AudioProject, transcript: dict[str, Any], source_path: Path,
    source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig,
) -> tuple[VideoTimeline, list[str]]:
    """Map actual audio clips to source visuals with deterministic, reported fallbacks."""

    source_duration = float(source_info["video_duration"])
    crop_plan = make_crop_plan(source_info, canvas, config)
    plan_segments = {segment.segment_id: segment for segment in plan.segments}
    transcript_ranges = _transcript_ranges(transcript)
    candidates: list[tuple[float, float, str | None, str | None, str | None]] = []
    for audio_clip in audio_project.timeline.clips:
        segment = plan_segments.get(audio_clip.production_segment_id)
        candidates.append(_source_range_for_audio(audio_clip, segment, transcript_ranges))
    clips: list[VideoClipModel] = []
    fallback_reasons: list[str] = []
    previous: SourceVideoClip | FreezeFrameClip | None = None
    cursor = 0.0
    for index, (audio_clip, candidate) in enumerate(zip(audio_project.timeline.clips, candidates), start=1):
        start, end = cursor, float(audio_clip.timeline_end_seconds)
        duration = round(end - start, 3)
        raw_start, raw_end, fact_id, speaker, mapping_kind = candidate
        visual: VideoClipModel | None = None
        if raw_start >= 0 and raw_end > raw_start and raw_end <= source_duration + 0.02:
            visual = _source_clip(
                index, start, end, source_path, raw_start, raw_end, audio_clip.production_segment_id,
                fact_id, speaker, "mapped_source", crop_plan, duration, config, None,
            )
        if visual is None:
            reason = f"{audio_clip.production_segment_id or audio_clip.clip_id}: missing_or_invalid_{mapping_kind}_mapping"
            fallback_reasons.append(reason)
            visual = _fallback_visual(
                index, index - 1, start, end, source_path, candidates, previous, crop_plan, duration,
                audio_clip.production_segment_id, fact_id, speaker, source_duration, config, reason,
            )
        clips.append(visual)
        if isinstance(visual, (SourceVideoClip, FreezeFrameClip)):
            previous = visual
        cursor = end
    transitions = _transitions(clips, config)
    timeline = VideoTimeline(
        clips=clips, transitions=transitions, duration_seconds=round(float(audio_project.timeline.duration_seconds), 3),
    )
    return timeline, fallback_reasons


def _source_range_for_audio(
    audio_clip: Any, segment: Any, transcript_ranges: dict[int, tuple[float, float]],
) -> tuple[float, float, str | None, str | None, str]:
    if isinstance(segment, DialogueSegment):
        return segment.source_start_seconds, segment.source_end_seconds, segment.fact_id, segment.speaker, "dialogue"
    if isinstance(segment, NarrationSegment):
        for identifier in segment.source_segment_ids:
            value = transcript_ranges.get(identifier)
            if value is not None:
                return value[0], value[1], segment.fact_ids[0] if segment.fact_ids else None, "narrator", "narration"
        return -1, -1, segment.fact_ids[0] if segment.fact_ids else None, "narrator", "narration"
    return -1, -1, None, None, "silence"


def _source_clip(
    order: int, timeline_start: float, timeline_end: float, source_path: Path, source_start: float, source_end: float,
    production_segment_id: str | None, fact_id: str | None, speaker: str | None, strategy: str,
    crop_plan: CropPlan, duration: float, config: ProductionRenderConfig, fallback_reason: str | None,
) -> SourceVideoClip | None:
    available = round(source_end - source_start, 3)
    if available < config.minimum_clip_duration:
        return None
    end = min(source_end, source_start + duration)
    freeze = max(0.0, round(duration - (end - source_start), 3))
    if freeze > config.maximum_freeze_duration + 0.001:
        return None
    fallback = fallback_reason or ("source_clip_short_freeze" if freeze else None)
    return SourceVideoClip(
        clip_id=f"visual-{order:03d}", order=order, timeline_start_seconds=timeline_start,
        timeline_end_seconds=timeline_end, duration_seconds=duration, source_path=str(source_path),
        source_start_seconds=round(source_start, 3), source_end_seconds=round(end, 3),
        production_segment_id=production_segment_id, fact_id=fact_id, speaker=speaker,
        visual_strategy=strategy, crop_plan=crop_plan, freeze_duration_seconds=freeze,
        status="fallback" if fallback else "ready", fallback_reason=fallback,
    )


def _fallback_visual(
    order: int, candidate_index: int, start: float, end: float, source_path: Path,
    candidates: list[tuple[float, float, str | None, str | None, str]], previous: SourceVideoClip | FreezeFrameClip | None,
    crop_plan: CropPlan, duration: float, production_segment_id: str | None, fact_id: str | None,
    speaker: str | None, source_duration: float, config: ProductionRenderConfig, reason: str,
) -> VideoClipModel:
    same_fact = next((
        item for index, item in enumerate(candidates)
        if index != candidate_index and fact_id and item[2] == fact_id and item[1] > item[0] and item[1] <= source_duration
    ), None)
    if same_fact:
        mapped = _source_clip(order, start, end, source_path, same_fact[0], same_fact[1], production_segment_id, fact_id, speaker, "candidate_excerpt", crop_plan, duration, config, reason + ":same_fact")
        if mapped:
            return mapped
    if previous is not None:
        mapped = _source_clip(order, start, end, source_path, previous.source_start_seconds, previous.source_end_seconds, production_segment_id, fact_id, speaker, "previous_visual", crop_plan, duration, config, reason + ":previous_visual")
        if mapped:
            return mapped
    next_range = next((
        item for item in candidates[candidate_index + 1:]
        if item[1] > item[0] and item[1] <= source_duration
    ), None)
    if next_range:
        mapped = _source_clip(order, start, end, source_path, next_range[0], next_range[1], production_segment_id, fact_id, speaker, "next_visual", crop_plan, duration, config, reason + ":next_visual")
        if mapped:
            return mapped
    # A short silence is safest as a held prior frame only after mapped ranges fail.
    if previous is not None and duration <= config.maximum_freeze_duration:
        point = max(0.0, min(source_duration, previous.source_end_seconds))
        return FreezeFrameClip(
            clip_id=f"visual-{order:03d}", order=order, timeline_start_seconds=start, timeline_end_seconds=end,
            duration_seconds=duration, source_path=str(source_path), source_start_seconds=max(0.0, point - 0.04),
            source_end_seconds=point, production_segment_id=production_segment_id, fact_id=fact_id, speaker=speaker,
            crop_plan=crop_plan, freeze_duration_seconds=duration, status="fallback",
            fallback_reason=reason + ":freeze_previous_frame",
        )
    # Deterministic safe excerpt from the selected source evidence is the final media fallback.
    candidate_excerpt = next((item for item in candidates if item[1] > item[0] and item[1] <= source_duration), None)
    if candidate_excerpt:
        mapped = _source_clip(order, start, end, source_path, candidate_excerpt[0], candidate_excerpt[1], production_segment_id, fact_id, speaker, "candidate_excerpt", crop_plan, duration, config, reason + ":candidate_excerpt")
        if mapped:
            return mapped
    return FillClip(
        clip_id=f"visual-{order:03d}", order=order, timeline_start_seconds=start, timeline_end_seconds=end,
        duration_seconds=duration, production_segment_id=production_segment_id, fact_id=fact_id, speaker=speaker,
        fallback_reason=reason + ":neutral_fill",
    )


def _transitions(clips: list[VideoClipModel], config: ProductionRenderConfig) -> list[VideoTransition]:
    if len(clips) < 2:
        return []
    duration = 0.15 if config.transitions == "short_crossfade" else 0.0
    return [
        VideoTransition(
            transition_type=config.transitions, from_clip_id=left.clip_id, to_clip_id=right.clip_id,
            duration_seconds=duration,
        )
        for left, right in zip(clips, clips[1:])
    ]


def make_crop_plan(source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig) -> CropPlan:
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    rotation = int(source_info.get("rotation", 0))
    if config.crop_strategy in {"fit_blur_background", "fit_solid_background"}:
        return CropPlan(strategy=config.crop_strategy, source_width=width, source_height=height, display_rotation_degrees=rotation)
    target = canvas.width / canvas.height
    if width / height >= target:
        crop_height = height
        crop_width = _even_down(height * target)
    else:
        crop_width = width
        crop_height = _even_down(width / target)
    if config.crop_strategy == "top_crop":
        x, y = (width - crop_width) // 2, 0
    else:
        x = _even_down((width - crop_width) * config.manual_crop_x)
        y = _even_down((height - crop_height) * config.manual_crop_y)
    x = max(0, min(x, width - crop_width))
    y = max(0, min(y, height - crop_height))
    return CropPlan(
        strategy=config.crop_strategy, source_width=width, source_height=height, display_rotation_degrees=rotation,
        normalized_x=config.manual_crop_x, normalized_y=config.manual_crop_y,
        crop_width=crop_width, crop_height=crop_height, crop_x=x, crop_y=y,
    )


def build_reframe_plan(
    source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig, timeline: VideoTimeline,
) -> ReframePlan:
    """Persist the composition decision independently of FFmpeg filter generation.

    ``subject_keyframes`` is an optional, cacheable input from a visual-analysis
    provider.  Until that provider has high-confidence detections, the plan uses
    a deterministic centred crop or a documented contain/blur fallback rather
    than pretending subject tracking happened.
    """

    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    target = canvas.width / canvas.height
    ratio = width / height
    raw_keyframes = source_info.get("subject_keyframes", [])
    keyframes: list[ReframeKeyframe] = []
    if isinstance(raw_keyframes, list):
        for item in raw_keyframes:
            if not isinstance(item, dict):
                continue
            try:
                keyframes.append(ReframeKeyframe(
                    time_seconds=float(item["time_seconds"]), normalized_x=float(item["normalized_x"]),
                    normalized_y=float(item["normalized_y"]), confidence=float(item.get("confidence", 0)),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    confident = [item for item in keyframes if item.confidence >= 0.55]
    if ratio <= target * 1.03:
        return ReframePlan(
            strategy="original_vertical", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
        )
    if config.crop_strategy == "center_crop" and confident:
        return ReframePlan(
            strategy="subject_crop", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
            keyframes=_smooth_reframe_keyframes(confident), subject_detection_used=True,
        )
    if config.crop_strategy == "center_crop":
        return ReframePlan(
            strategy="center_crop", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
            fallback_reason="No high-confidence subject observations are available.",
        )
    return ReframePlan(
        strategy="blur_fallback" if config.crop_strategy == "fit_blur_background" else "contain",
        source_width=width, source_height=height, canvas_width=canvas.width, canvas_height=canvas.height,
        subtitle_reserved_bottom_ratio=0.16, fallback_reason="Crop strategy preserves the full source frame.",
    )


def _smooth_reframe_keyframes(keyframes: list[ReframeKeyframe]) -> list[ReframeKeyframe]:
    ordered = sorted(keyframes, key=lambda item: item.time_seconds)
    result: list[ReframeKeyframe] = []
    previous: ReframeKeyframe | None = None
    for item in ordered:
        if previous is None:
            smoothed = item
        else:
            # Bound per-sample motion to keep a tracker from making abrupt jumps.
            smoothed = ReframeKeyframe(
                time_seconds=item.time_seconds,
                normalized_x=max(previous.normalized_x - 0.16, min(previous.normalized_x + 0.16, item.normalized_x)),
                normalized_y=max(previous.normalized_y - 0.12, min(previous.normalized_y + 0.12, item.normalized_y)),
                confidence=item.confidence,
            )
        result.append(smoothed)
        previous = smoothed
    return result


def _visual_filter(clip: SourceVideoClip | FreezeFrameClip, canvas: CanvasConfig) -> str:
    crop = clip.crop_plan
    assert crop is not None
    tail = (
        f",fps={canvas.fps},tpad=stop_mode=clone:stop_duration={clip.freeze_duration_seconds:.6f},"
        f"trim=duration={clip.duration_seconds:.6f},setpts=PTS-STARTPTS,format={canvas.pixel_format}[vout]"
    )
    if crop.strategy == "fit_blur_background":
        return (
            f"[0:v]split=2[bg][fg];[bg]scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=increase,"
            f"crop={canvas.width}:{canvas.height},boxblur=20:10[blur];[fg]scale={canvas.width}:{canvas.height}:"
            f"force_original_aspect_ratio=decrease[fit];[blur][fit]overlay=(W-w)/2:(H-h),setsar=1" + tail
        )
    if crop.strategy == "fit_solid_background":
        return (
            f"color=c=0x161616:s={canvas.width}x{canvas.height}:r={canvas.fps}[bg];"
            f"[0:v]scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=decrease[fit];"
            f"[bg][fit]overlay=(W-w)/2:(H-h),setsar=1" + tail
        )
    assert crop.crop_width and crop.crop_height and crop.crop_x is not None and crop.crop_y is not None
    return (
        f"[0:v]crop={crop.crop_width}:{crop.crop_height}:{crop.crop_x}:{crop.crop_y},"
        f"scale={canvas.width}:{canvas.height},setsar=1" + tail
    )


def probe_media(path: Path, require_video: bool = False, require_audio: bool = False) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ProductionRenderError("ffprobe не найден для production render.")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=90, check=True,
        )
        raw = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ProductionRenderError(f"ffprobe не смог прочитать artifact: {path.name}") from error
    streams = raw.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if require_video and not isinstance(video, dict):
        raise ProductionRenderError("Source media не содержит video stream.")
    if require_audio and not isinstance(audio, dict):
        raise ProductionRenderError("Audio artifact не содержит audio stream.")
    result_data: dict[str, Any] = {"format_duration": _duration(raw.get("format", {}).get("duration"))}
    if isinstance(video, dict):
        rotation = _rotation(video)
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
        result_data.update({
            "video_duration": _duration(video.get("duration"), result_data["format_duration"]),
            "video_codec": str(video.get("codec_name") or ""), "width": width, "height": height,
            "display_width": display_width, "display_height": display_height, "rotation": rotation,
            "fps": _fps(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")),
            "pixel_format": str(video.get("pix_fmt") or ""),
        })
    if isinstance(audio, dict):
        result_data.update({
            "audio_duration": _duration(audio.get("duration"), result_data["format_duration"]),
            "audio_codec": str(audio.get("codec_name") or ""),
        })
    return result_data


def validate_final_video(path: Path, canvas: CanvasConfig, mixed_audio: Path, config: ProductionRenderConfig) -> RenderValidation:
    if not path.is_file() or path.stat().st_size < 1024:
        return RenderValidation(status="invalid", messages=["Final MP4 is missing or empty."])
    try:
        media = probe_media(path, require_video=True, require_audio=True)
        audio_reference = probe_media(mixed_audio, require_audio=True)
    except ProductionRenderError as error:
        return RenderValidation(status="invalid", messages=[_safe_error(error)])
    messages: list[str] = []
    if media.get("video_codec") != "h264" or media.get("audio_codec") not in {"aac", "mp4a"}:
        messages.append("Final MP4 has an unsupported video or audio codec.")
    if media.get("width") != canvas.width or media.get("height") != canvas.height:
        messages.append("Final MP4 resolution does not match the production canvas.")
    if media.get("pixel_format") != canvas.pixel_format:
        messages.append("Final MP4 pixel format is not yuv420p.")
    if abs(float(media.get("fps", 0)) - canvas.fps) > 0.25:
        messages.append("Final MP4 FPS does not match the production canvas.")
    video_duration = float(media.get("video_duration") or 0)
    audio_duration = float(media.get("audio_duration") or 0)
    reference_duration = float(audio_reference.get("audio_duration") or 0)
    difference_ms = round(max(abs(video_duration - reference_duration), abs(audio_duration - reference_duration)) * 1000, 3)
    if difference_ms > config.av_sync_error_ms:
        messages.append("Final MP4 audio/video duration exceeds the configured error tolerance.")
        status = "invalid"
    elif messages or difference_ms > config.av_sync_warning_ms:
        if difference_ms > config.av_sync_warning_ms:
            messages.append("Final MP4 audio/video duration exceeds the configured warning tolerance.")
        status = "warning"
    else:
        status = "valid"
    return RenderValidation(
        status=status, video_duration_seconds=video_duration, audio_duration_seconds=audio_duration,
        sync_difference_ms=difference_ms, width=media.get("width"), height=media.get("height"),
        fps=media.get("fps"), video_codec=media.get("video_codec"), audio_codec=media.get("audio_codec"),
        pixel_format=media.get("pixel_format"), messages=messages,
    )


def _render_cache_key(
    source_checksum: str, mixed_checksum: str, audio_project: AudioProject, timeline: VideoTimeline,
    subtitles: SubtitleProject, canvas: CanvasConfig, config: ProductionRenderConfig,
) -> str:
    return stable_text_hash(json.dumps({
        "source_checksum": source_checksum, "mixed_audio_checksum": mixed_checksum,
        "audio_project_checksum": stable_text_hash(audio_project.model_dump_json()),
        "timeline": timeline.model_dump(mode="json"), "subtitle_project": subtitles.model_dump(mode="json"),
        "canvas": canvas.model_dump(mode="json"), "crop": config.crop_strategy,
        "encoder": config.encoder, "codec": config.video_codec, "bitrate": config.video_bitrate,
        "subtitles_enabled": config.subtitles_enabled, "version": config.render_config_version,
        "engine_version": PRODUCTION_RENDER_ENGINE_VERSION,
    }, sort_keys=True, ensure_ascii=False))


def _write_project_artifacts(project: VideoProject, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    timeline_path = root / "video-timeline.json"
    reframe_path = root / "reframe-plan.json"
    subtitle_path = root / "subtitle-project.json"
    project_path = root / "video-project.json"
    result_path = root / "render-result.json"
    summary_path = root / "render-summary.txt"
    artifacts = [str(project_path), str(timeline_path), str(reframe_path), str(subtitle_path), str(result_path), str(summary_path)]
    if (root / "production-subtitles.ass").is_file():
        artifacts.append(str(root / "production-subtitles.ass"))
    if project.result and project.result.output_file:
        artifacts.append(project.result.output_file)
    complete = project.model_copy(update={"artifact_paths": artifacts})
    write_json(timeline_path, complete.timeline.model_dump(mode="json"))
    write_json(reframe_path, complete.reframe_plan.model_dump(mode="json"))
    if complete.subtitle_project is not None:
        write_json(subtitle_path, complete.subtitle_project.model_dump(mode="json"))
    write_json(project_path, complete.model_dump(mode="json"))
    write_json(result_path, complete.result.model_dump(mode="json") if complete.result else {"status": complete.status})
    write_bytes_atomic(summary_path, _summary(complete).encode("utf-8"))


def _artifacts(root: Path, final_path: Path, clips: list[Path] | None = None) -> list[RenderArtifact]:
    result: list[RenderArtifact] = []
    pairs = [
        ("final_mp4", final_path), ("video_project", root / "video-project.json"),
        ("video_timeline", root / "video-timeline.json"), ("reframe_plan", root / "reframe-plan.json"), ("subtitle_project", root / "subtitle-project.json"),
        ("production_ass", root / "production-subtitles.ass"), ("render_result", root / "render-result.json"),
        ("summary", root / "render-summary.txt"),
    ]
    pairs.extend(("clip", path) for path in clips or [])
    for kind, path in pairs:
        if path.is_file():
            # project/result/summary contain the artifact list itself; a self-checksum would
            # become stale as soon as it is serialized. Their paths and sizes stay auditable.
            checksum = None if kind in {"video_project", "render_result", "summary"} else stable_file_hash(path)
            result.append(RenderArtifact(artifact_type=kind, path=str(path), checksum=checksum, byte_size=path.stat().st_size))
    return result


def _summary(project: VideoProject) -> str:
    result = project.result
    return "\n".join([
        f"Video Project: {project.project_id}", f"Status: {project.status}",
        f"Duration: {project.actual_duration_seconds:.3f} s", f"Visual clips: {len(project.timeline.clips)}",
        f"Subtitle cues: {len(project.subtitle_project.cues) if project.subtitle_project else 0}",
        f"Encoder: {result.encoder if result else 'not-run'}", f"Cache hit: {result.cache_hit if result else False}",
        "No AI call, TTS regeneration, audio remix, legacy render mutation, or source media mutation was performed.", "",
    ])


def _final_command(
    ffmpeg: str, inputs: list[str], graph: str, clip_count: int, destination: Path,
    canvas: CanvasConfig, encoder: str, bitrate: str,
) -> list[str]:
    return [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex", graph,
        "-map", "[vout]", "-map", f"{clip_count}:a", "-c:v", encoder,
        "-preset", "p4" if encoder == "h264_nvenc" else "medium", "-b:v", bitrate,
        "-pix_fmt", canvas.pixel_format, "-r", f"{canvas.fps}", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(destination),
    ]


def _timeline_filter(durations: list[float], transition: str) -> tuple[str, str]:
    if not durations:
        raise ProductionRenderError("No visual durations were supplied for final mux.")
    if transition == "short_crossfade" and len(durations) > 1:
        fade = 0.15
        current = "[0:v]"
        elapsed = durations[0]
        for index, duration in enumerate(durations[1:], start=1):
            label = f"[xf{index}]"
            offset = max(0.0, elapsed - fade)
            graph = f"{current}[{index}:v]xfade=transition=fade:duration={fade:.3f}:offset={offset:.6f}{label}"
            current = label
            elapsed += duration - fade
            if index == 1:
                parts = [graph]
            else:
                parts.append(graph)
        loss = fade * (len(durations) - 1)
        parts.append(f"{current}tpad=stop_mode=clone:stop_duration={loss:.6f},trim=duration={sum(durations):.6f}[vconcat]")
        return ";".join(parts), "[vconcat]"
    labels = "".join(f"[{index}:v]" for index in range(len(durations)))
    graph = f"{labels}concat=n={len(durations)}:v=1:a=0[vconcat]"
    fade = min(0.15, durations[0])
    if transition == "fade_from_black":
        return graph + f";[vconcat]fade=t=in:st=0:d={fade:.3f}[vfaded]", "[vfaded]"
    if transition == "fade_to_black":
        start = max(0.0, sum(durations) - min(0.15, durations[-1]))
        return graph + f";[vconcat]fade=t=out:st={start:.6f}:d={fade:.3f}[vfaded]", "[vfaded]"
    return graph, "[vconcat]"


def _run_ffmpeg(command: list[str], context: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired as error:
        raise ProductionRenderError(f"FFmpeg timed out during {context}.") from error
    except (OSError, subprocess.CalledProcessError) as error:
        details = getattr(error, "stderr", "") or getattr(error, "stdout", "") or ""
        raise ProductionRenderError(f"FFmpeg failed during {context}: {details[-1200:]}") from error


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise ProductionRenderError("ffmpeg не найден для production render.")
    return executable


def _filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def _transcript_ranges(transcript: dict[str, Any]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for index, item in enumerate(transcript.get("segments", []) if isinstance(transcript, dict) else []):
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
            identifier = int(item.get("id", index))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            result[identifier] = (start, end)
    return result


def _duration(value: Any, fallback: float = 0.0) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return fallback


def _fps(value: str) -> float:
    try:
        parsed = float(Fraction(value))
        return round(parsed, 3) if parsed > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict[str, Any]) -> int:
    value = stream.get("tags", {}).get("rotate") if isinstance(stream.get("tags"), dict) else None
    for side in stream.get("side_data_list", []) if isinstance(stream.get("side_data_list"), list) else []:
        if isinstance(side, dict) and side.get("rotation") is not None:
            value = side["rotation"]
    try:
        return int(round(float(value))) % 360 if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _even_down(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _temporary_path(directory: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=suffix) as temporary:
        return Path(temporary.name)


def _safe_error(error: BaseException) -> str:
    return str(error).replace("\n", " ")[:1200]


def production_render_report_section(project: VideoProject) -> dict[str, Any]:
    result = project.result
    validation = result.validation if result else RenderValidation(status="invalid", messages=["Render result is unavailable."])
    return {
        "enabled": True,
        "status": project.status,
        "video_project_id": project.project_id,
        "output_file": result.output_file if result else None,
        "resolution": f"{project.canvas.width}x{project.canvas.height}",
        "aspect_ratio": project.canvas.aspect_ratio,
        "fps": project.canvas.fps,
        "video_codec": project.render_request.video_codec,
        "encoder": result.encoder if result else None,
        "hardware_fallback": bool(result.hardware_fallback) if result else False,
        "duration": project.actual_duration_seconds,
        "audio_duration": validation.audio_duration_seconds,
        "sync_difference_ms": validation.sync_difference_ms,
        "clip_count": len(project.timeline.clips),
        "subtitle_cue_count": len(project.subtitle_project.cues) if project.subtitle_project else 0,
        "cache_hit": bool(result.cache_hit) if result else False,
        "validation": validation.status,
        "warnings": project.warnings,
        "fallback_reasons": project.fallback_reasons,
        "errors": [item.model_dump(mode="json") for item in result.errors] if result else [],
        "artifacts": project.artifact_paths,
        "ai_called": False,
        "tts_regenerated": False,
        "audio_remixed": False,
    }
