from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from app.audio_models import AudioProject
from app.config import AppConfig, ProductionRenderConfig
from app.creative_contracts import (
    CompiledRenderPlan,
    RenderParityManifest,
    RenderProfile,
    assert_preview_final_parity,
    build_render_parity_manifest,
    compile_legacy_render_plan,
    source_output_map_from_legacy_timeline,
)
from app.errors import ProductionRenderError
from app.output_quality import validate_output_quality
from app.production_models import DialogueSegment, NarrationSegment, ProductionPlan, validate_renderer_handoff
from app.production_subtitles import build_subtitle_project, write_production_ass
from app.rendering import nvenc_available
from app.render_cache import CacheArtifact, GranularRenderCache, runtime_cache_key
from app.sources import Source
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.utils import read_json, stable_file_hash, stable_text_hash, utc_now, write_bytes_atomic, write_json
from app.video_models import (
    CanvasConfig,
    CompositionSegment,
    CompositionQualityDecision,
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
    SubjectBounds,
    VideoClipModel,
    VideoProject,
    VideoTimeline,
    VideoTrack,
    VideoTransition,
)


PRODUCTION_RENDER_ENGINE_VERSION = "7G.0"


class VideoCompositionService:
    """Goal 3D executor. It only consumes existing production artifacts and source media."""

    def __init__(self, root: Path, config: AppConfig) -> None:
        self.root = root.resolve()
        self.config = config

    def compose(
        self, plan: ProductionPlan, audio_project: AudioProject, source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path,
        force_recompute: bool = False, visual_analysis: dict[str, Any] | None = None,
        render_profile: RenderProfile | Literal["creative_preview", "final"] = "final",
        compiled_plan: CompiledRenderPlan | None = None,
    ) -> VideoProject:
        render_config = self.config.production_render
        if not source.path.is_file():
            raise ProductionRenderError("Исходный video file для production render не найден.")
        handoff_failure = validate_renderer_handoff(
            plan,
            audio_project,
            source_id=source.id,
            source_sha256=stable_file_hash(source.path),
            transcript=transcript,
            expected_preset_id=self.config.product_flow.subtitle_preset,
            expected_preset_version=self.config.product_flow.preset_version,
            expected_platform=self.config.product_flow.platform,
            expected_target=(render_config.output_width, render_config.output_height, render_config.output_fps),
        )
        if handoff_failure is not None:
            raise ProductionRenderError(
                f"{handoff_failure.code}: ProductionPlan render handoff rejected: "
                f"{json.dumps(handoff_failure.evidence, ensure_ascii=False, sort_keys=True)}"
            )
        source_info = probe_media(source.path, require_video=True)
        if visual_analysis and isinstance(visual_analysis.get("subject_keyframes"), list):
            source_info["subject_keyframes"] = visual_analysis["subject_keyframes"]
        _attach_visual_evidence_context(source_info, visual_analysis)
        source_info["composition_intent"] = dict(plan.composition_intent)
        scene_analysis = read_json(work_directory / "scene_boundaries.json", {})
        if isinstance(scene_analysis, dict) and isinstance(scene_analysis.get("boundaries"), list):
            source_info["scene_boundaries"] = scene_analysis["boundaries"]
        mixed_path = Path(audio_project.mix.mixed_audio_path or "")
        if not mixed_path.is_file():
            raise ProductionRenderError("mixed_audio.wav не найден для production render.")
        mixed_info = probe_media(mixed_path, require_audio=True)
        source_checksum = stable_file_hash(source.path)
        mixed_checksum = stable_file_hash(mixed_path)
        planning_canvas = CanvasConfig(
            width=render_config.output_width, height=render_config.output_height,
            fps=render_config.output_fps, pixel_format="yuv420p",
        )
        timeline, fallback_reasons = build_video_timeline(
            plan, audio_project, transcript, source.path, source_info, planning_canvas, render_config,
        )
        plan_reference = plan.reference()
        reframe_plan = build_reframe_plan(source_info, planning_canvas, render_config, timeline).model_copy(update={
            "plan_reference": plan_reference,
        })
        timeline = apply_composition_segments(timeline, reframe_plan.composition_segments)
        composition_fallbacks = [
            f"{segment.segment_id}: {segment.fallback_reason}"
            for segment in reframe_plan.composition_segments
            if segment.fallback_reason
        ]
        actual_audio_duration = float(mixed_info["audio_duration"])
        if abs(timeline.duration_seconds - actual_audio_duration) > render_config.maximum_duration_difference:
            raise ProductionRenderError(
                "AudioProject timeline и mixed_audio.wav имеют несовместимую длительность: "
                f"{timeline.duration_seconds:.3f}s vs {actual_audio_duration:.3f}s."
            )
        subtitle_project = build_subtitle_project(
            plan, audio_project, render_config, transcript,
            composition_segments=reframe_plan.composition_segments,
            platform=self.config.product_flow.platform,
        ).model_copy(update={
            "plan_reference": plan_reference,
        })
        generated_plan = compile_legacy_render_plan(
            plan,
            source_output_map_from_legacy_timeline(timeline),
            subtitle_project=subtitle_project,
            reframe_plan=reframe_plan,
        )
        if compiled_plan is not None and (
            compiled_plan.plan_hash != generated_plan.plan_hash
            or compiled_plan.parity_signature != generated_plan.parity_signature
        ):
            raise ProductionRenderError(
                "PREVIEW_FINAL_PARITY_MISMATCH: supplied CompiledRenderPlan does not match resolved render inputs."
            )
        compiled_plan = compiled_plan or generated_plan
        profile = _resolve_render_profile(compiled_plan, render_profile, render_config)
        canvas = CanvasConfig(
            width=profile.width, height=profile.height, fps=profile.fps,
            pixel_format="yuv420p",
        )
        cache_key = _render_cache_key(
            plan, source_checksum, mixed_checksum, audio_project, timeline, subtitle_project, canvas, render_config,
            platform=self.config.product_flow.platform,
            product_flow_revision=self.config.product_flow.preset_version,
            compiled_plan=compiled_plan,
            render_profile=profile,
        )
        project_id = f"video-{plan.plan_id}-{audio_project.project_id}-{cache_key[:12]}"
        request = RenderRequest(
            video_project_id=project_id, source_path=str(source.path), mixed_audio_path=str(mixed_path),
            canvas=canvas, encoder_preference=profile.encoder, video_bitrate=profile.video_bitrate,
            subtitles_enabled=render_config.subtitles_enabled,
        )
        metadata = RenderMetadata(
            production_plan_id=plan.plan_id, audio_project_id=audio_project.project_id,
            source_checksum=source_checksum, mixed_audio_checksum=mixed_checksum,
            render_config_version=render_config.render_config_version, cache_key=cache_key,
            compiled_plan_hash=compiled_plan.plan_hash,
            parity_signature=compiled_plan.parity_signature,
            render_profile_id=profile.profile_id,
            created_at=utc_now(), updated_at=utc_now(),
        )
        track = VideoTrack(
            track_id="track-visual", clips=timeline.clips,
            status="fallback" if fallback_reasons or composition_fallbacks else "ready",
        )
        project = VideoProject(
            project_id=project_id, status="skipped", source_video_path=str(source.path), source_checksum=source_checksum,
            production_plan_id=plan.plan_id, plan_reference=plan_reference,
            audio_project_id=audio_project.project_id, mixed_audio_path=str(mixed_path),
            canvas=canvas, target_duration_seconds=timeline.duration_seconds, actual_duration_seconds=0,
            timeline=timeline, reframe_plan=reframe_plan, tracks=[track], subtitle_project=subtitle_project, render_request=request,
            metadata=metadata,
            warnings=list(subtitle_project.warnings),
            fallback_reasons=[
                *fallback_reasons,
                *composition_fallbacks,
                *([reframe_plan.fallback_reason] if reframe_plan.fallback_reason else []),
            ],
        )
        quality = validate_output_quality(project, project.render_request.subtitles_enabled)
        if quality["status"] == "failed":
            raise ProductionRenderError(
                "Resolved Subtitle Quality V2 validation failed before final-ready state: "
                + "; ".join(quality["errors"])
            )
        render_root = output_directory / (
            "creative-preview" if profile.profile_id == "creative_preview" else "production-render"
        )
        if profile.profile_id == "final":
            preview_manifest_path = output_directory / "creative-preview" / "parity-manifest.json"
            if preview_manifest_path.is_file():
                try:
                    preview_manifest = RenderParityManifest.model_validate(read_json(preview_manifest_path, {}))
                    preview_output = output_directory / "creative-preview" / "creative-preview.mp4"
                    if (
                        preview_manifest.output_checksum is None
                        or not preview_output.is_file()
                        or stable_file_hash(preview_output) != preview_manifest.output_checksum
                    ):
                        raise ValueError("CREATIVE_PREVIEW_ARTIFACT_CHECKSUM_MISMATCH")
                    final_manifest = build_render_parity_manifest(compiled_plan, profile)
                    assert_preview_final_parity(preview_manifest, final_manifest)
                except (OSError, ValueError) as error:
                    raise ProductionRenderError(
                        f"PREVIEW_FINAL_PARITY_MISMATCH: Final render blocked: {_safe_error(error)}"
                    ) from error
        cache_path = self.root / "work" / "production-render-cache" / f"{cache_key}.json"
        return self._render(
            project, source_info, mixed_info, render_root, cache_path,
            compiled_plan=compiled_plan, render_profile=profile,
            force_recompute=force_recompute,
        )

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
        quality = validate_output_quality(project, project.render_request.subtitles_enabled)
        if quality["status"] == "failed":
            return None
        artifacts = _artifacts(render_root, final_path)
        warnings = [*project.warnings, *validation.messages, *quality["warnings"]]
        result = RenderResult(
            status="warning" if warnings or validation.status == "warning" else "completed", output_file=str(final_path),
            encoder=str(cached.get("encoder") or "cache"), hardware_fallback=bool(cached.get("hardware_fallback", False)),
            cache_hit=True, validation=validation, artifacts=artifacts,
            warnings=warnings,
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
        *, compiled_plan: CompiledRenderPlan, render_profile: RenderProfile,
        force_recompute: bool,
    ) -> VideoProject:
        quality = validate_output_quality(project, project.render_request.subtitles_enabled)
        if quality["status"] == "failed":
            raise ProductionRenderError("Resolved subtitle quality validation failed: " + "; ".join(quality["errors"]))
        temp_root = render_root / "temp"
        render_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        cache = GranularRenderCache(
            self.root / "work" / "creative-render-cache",
            producer_version=PRODUCTION_RENDER_ENGINE_VERSION,
        )
        nodes = {node.node_id: node for node in compiled_plan.render_graph_nodes}
        cache_hits: dict[str, bool] = {}
        bypass_cache = force_recompute or not self.config.production_render.cache_enabled

        captions_key = runtime_cache_key(nodes["captions"].cache_key)
        captions = None if bypass_cache else cache.load("captions", captions_key, suffix=".ass")
        cache_hits["captions"] = captions is not None
        if captions is None:
            generated_ass = _temporary_path(temp_root, ".ass")
            try:
                if project.subtitle_project is not None:
                    # Keep the plan canvas as ASS PlayRes. libass scales the same
                    # resolved lines/font geometry into either quality profile.
                    write_production_ass(
                        project.subtitle_project, generated_ass,
                        compiled_plan.canvas.width, compiled_plan.canvas.height,
                    )
                else:
                    write_bytes_atomic(generated_ass, b"")
                if generated_ass.stat().st_size == 0:
                    write_bytes_atomic(generated_ass, b"; captions disabled\n")
                captions = cache.store_file("captions", captions_key, generated_ass, suffix=".ass")
            finally:
                generated_ass.unlink(missing_ok=True)

        plan_artifacts: dict[str, CacheArtifact] = {}
        for node_id, payload in (
            ("composition", compiled_plan.composition_plan.model_dump(mode="json")),
            ("broll", compiled_plan.source_broll_plan.model_dump(mode="json")),
            ("motion", compiled_plan.motion_plan.model_dump(mode="json")),
        ):
            # Cache execution inputs, not the parent lifecycle id. The complete
            # identity remains in compiled-render-plan.json; excluding intent_id
            # here is what permits caption-only revisions to reuse visual work.
            payload.pop("intent_id", None)
            node_key = runtime_cache_key(nodes[node_id].cache_key)
            artifact = None if bypass_cache else cache.load(node_id, node_key, suffix=".json")
            cache_hits[node_id] = artifact is not None
            if artifact is None:
                artifact = cache.store_json(node_id, node_key, payload)
            plan_artifacts[node_id] = artifact

        profile_payload = render_profile.model_dump(mode="json")
        visual_profile_payload = {
            "width": render_profile.width,
            "height": render_profile.height,
            "fps": render_profile.fps,
            "sampling_precision": render_profile.sampling_precision,
        }
        base_key = runtime_cache_key(
            nodes["base-visual"].cache_key,
            profile=visual_profile_payload,
            inputs={"source_checksum": project.source_checksum},
        )
        base_dependencies = {
            "composition": plan_artifacts["composition"].checksum,
            "broll": plan_artifacts["broll"].checksum,
        }
        base = None if bypass_cache else cache.load(
            "base-visual", base_key, suffix=".mp4", dependency_checksums=base_dependencies,
        )
        cache_hits["base-visual"] = base is not None
        if base is None:
            generated_base = _temporary_path(temp_root, ".mp4")
            try:
                self._render_base_visual(
                    project.timeline.clips, project.canvas, generated_base,
                    project.timeline.transitions,
                )
                rendered = probe_media(generated_base, require_video=True)
                if abs(float(rendered["video_duration"]) - project.timeline.duration_seconds) > 0.15:
                    raise ProductionRenderError("Base visual cache node has an invalid duration.")
                base = cache.store_file(
                    "base-visual", base_key, generated_base, suffix=".mp4",
                    dependency_checksums=base_dependencies,
                )
            finally:
                generated_base.unlink(missing_ok=True)

        assert captions is not None and base is not None
        composite_key = runtime_cache_key(
            nodes["composite"].cache_key,
            profile=visual_profile_payload,
            inputs={
                "mixed_audio_checksum": project.metadata.mixed_audio_checksum,
                "subtitles_enabled": project.render_request.subtitles_enabled,
            },
        )
        composite_dependencies = {
            "base-visual": base.checksum,
            "captions": captions.checksum,
            "motion": plan_artifacts["motion"].checksum,
        }
        composite = None if bypass_cache else cache.load(
            "composite", composite_key, suffix=".json",
            dependency_checksums=composite_dependencies,
        )
        cache_hits["composite"] = composite is not None
        if composite is None:
            composite = cache.store_json(
                "composite", composite_key,
                {
                    "base_visual_checksum": base.checksum,
                    "caption_checksum": captions.checksum,
                    "motion_checksum": plan_artifacts["motion"].checksum,
                    "mixed_audio_checksum": project.metadata.mixed_audio_checksum,
                    "subtitles_enabled": project.render_request.subtitles_enabled,
                },
                dependency_checksums=composite_dependencies,
            )

        encode_key = runtime_cache_key(
            nodes["encode"].cache_key,
            profile=profile_payload,
            inputs={
                "composite_checksum": composite.checksum,
                "pixel_format": project.canvas.pixel_format,
            },
        )
        encode_dependencies = {"composite": composite.checksum}
        encoded = None if bypass_cache else cache.load(
            "encode", encode_key, suffix=".mp4", dependency_checksums=encode_dependencies,
        )
        cache_hits["encode"] = encoded is not None
        temporary: Path | None = None
        encoder = "cache"
        hardware_fallback = False
        encoder_warning: str | None = None
        try:
            if encoded is None:
                temporary = _temporary_path(temp_root, ".mp4")
                encoder, hardware_fallback, encoder_warning = self._mux_base_visual(
                    base.path, Path(project.mixed_audio_path),
                    captions.path if project.render_request.subtitles_enabled else None,
                    temporary, project.canvas, render_profile.video_bitrate, render_profile.encoder,
                )
                validation = validate_final_video(
                    temporary, project.canvas, Path(project.mixed_audio_path), self.config.production_render,
                )
                if validation.status == "invalid":
                    raise ProductionRenderError("Final MP4 не прошёл обязательную ffprobe validation.")
                encoded = cache.store_file(
                    "encode", encode_key, temporary, suffix=".mp4",
                    dependency_checksums=encode_dependencies,
                )
            final_path = render_root / (
                "creative-preview.mp4" if render_profile.profile_id == "creative_preview" else "final-short.mp4"
            )
        except ProductionRenderError:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        except Exception as error:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise ProductionRenderError(f"Production render failed: {_safe_error(error)}") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        qc_key = runtime_cache_key(
            nodes["qc"].cache_key,
            profile=profile_payload,
            inputs={
                "output_checksum": encoded.checksum,
                "quality_policy": {
                    "av_sync_warning_ms": self.config.production_render.av_sync_warning_ms,
                    "av_sync_error_ms": self.config.production_render.av_sync_error_ms,
                    "maximum_duration_difference": self.config.production_render.maximum_duration_difference,
                    "render_config_version": self.config.production_render.render_config_version,
                },
            },
        )
        qc_dependencies = {"encode": encoded.checksum}
        qc = None if bypass_cache else cache.load(
            "qc", qc_key, suffix=".json", dependency_checksums=qc_dependencies,
        )
        cache_hits["qc"] = qc is not None
        if qc is not None:
            qc_payload = read_json(qc.path, {})
            try:
                validation = RenderValidation.model_validate(qc_payload.get("validation", {}))
            except (AttributeError, ValueError):
                qc = None
        if qc is None:
            validation = validate_final_video(
                encoded.path, project.canvas, Path(project.mixed_audio_path), self.config.production_render,
            )
            if validation.status == "invalid":
                raise ProductionRenderError("Cached/rendered MP4 failed QC validation.")
            qc = cache.store_json(
                "qc", qc_key,
                {"validation": validation.model_dump(mode="json"), "output_checksum": encoded.checksum},
                dependency_checksums=qc_dependencies,
            )
            cache_hits["qc"] = False

        # Publish only after the content-addressed encode and QC nodes have both
        # validated. A failed/corrupt fallback therefore cannot replace the last
        # known-good preview or final artifact.
        cache.materialize(encoded, final_path)

        parity_manifest = build_render_parity_manifest(
            compiled_plan, render_profile, output_checksum=encoded.checksum,
        )
        write_json(render_root / "compiled-render-plan.json", compiled_plan.model_dump(mode="json"))
        write_json(render_root / "parity-manifest.json", parity_manifest.model_dump(mode="json"))
        ass_path = render_root / "production-subtitles.ass"
        cache.materialize(captions, ass_path)
        warnings = [*project.warnings, *validation.messages, *quality["warnings"]]
        if encoder_warning:
            warnings.append(encoder_warning)
        result = RenderResult(
            status="warning" if warnings or validation.status == "warning" else "completed",
            output_file=str(final_path), encoder=encoder, hardware_fallback=hardware_fallback,
            cache_hit=cache_hits["encode"], validation=validation, artifacts=[], warnings=warnings,
        )
        complete = project.model_copy(update={
            "status": result.status, "actual_duration_seconds": validation.video_duration_seconds or 0,
            "result": result, "warnings": warnings,
            "metadata": project.metadata.model_copy(update={
                "updated_at": utc_now(),
                "cache_node_hits": cache_hits,
                "single_pass_encode": not project.render_request.subtitles_enabled,
            }),
        })
        _write_project_artifacts(complete, render_root)
        artifacts = _artifacts(render_root, final_path)
        result = result.model_copy(update={"artifacts": artifacts})
        complete = complete.model_copy(update={
            "result": result, "artifact_paths": [item.path for item in artifacts],
            "metadata": complete.metadata.model_copy(update={"updated_at": utc_now()}),
        })
        _write_project_artifacts(complete, render_root)
        write_json(cache_path, {
            "schema_version": PRODUCTION_RENDER_ENGINE_VERSION, "cache_key": complete.metadata.cache_key,
            "checksum": stable_file_hash(final_path), "encoder": encoder,
            "hardware_fallback": hardware_fallback,
            "compiled_plan_hash": compiled_plan.plan_hash,
            "parity_signature": compiled_plan.parity_signature,
            "cache_node_hits": cache_hits,
            "created_at": utc_now(),
        })
        return complete

    def _render_base_visual(
        self,
        clips: list[VideoClipModel],
        canvas: CanvasConfig,
        destination: Path,
        transitions: list[VideoTransition],
    ) -> None:
        """Compose all source ranges into one reusable visual mezzanine.

        The former path encoded every clip separately and encoded the joined
        result again.  This graph performs one base encode regardless of clip
        count; caption revisions reuse it without repeating decode/crop work.
        """

        if not clips:
            raise ProductionRenderError("Video timeline has no renderable visual clips.")
        inputs: list[str] = []
        filters: list[str] = []
        labels: list[str] = []
        for index, clip in enumerate(clips):
            label = f"[base{index}]"
            labels.append(label)
            if isinstance(clip, FillClip):
                inputs.extend([
                    "-f", "lavfi", "-t", f"{clip.duration_seconds:.6f}",
                    "-i", f"color=c=0x161616:s={canvas.width}x{canvas.height}:r={canvas.fps}",
                ])
                filters.append(
                    f"[{index}:v]fps={canvas.fps},trim=duration={clip.duration_seconds:.6f},"
                    f"setpts=PTS-STARTPTS,format={canvas.pixel_format}{label}"
                )
                continue
            source = Path(clip.source_path)
            if not source.is_file() or clip.source_start_seconds is None or clip.source_end_seconds is None:
                raise ProductionRenderError(f"Visual clip {clip.clip_id} has no usable source media.")
            available = max(0.04, clip.source_end_seconds - clip.source_start_seconds)
            inputs.extend([
                "-ss", f"{clip.source_start_seconds:.6f}", "-t", f"{available:.6f}",
                "-i", str(source),
            ])
            filters.append(_visual_filter(clip, canvas, input_label=f"[{index}:v]", output_label=label))
        transition_plan: str | list[VideoTransition] = self.config.production_render.transitions
        if self.config.production_render.transitions == "short_crossfade" and any(
            item.transition_type == "cut" for item in transitions
        ):
            transition_plan = transitions
        timeline_graph, video_label = _timeline_filter(
            [clip.duration_seconds for clip in clips], transition_plan, input_labels=labels,
        )
        filters.append(timeline_graph)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex", ";".join(filters), "-map", video_label,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", canvas.pixel_format, "-r", str(canvas.fps),
            "-movflags", "+faststart", str(destination),
        ]
        _run_ffmpeg(command, "base visual composition")

    def _mux_base_visual(
        self,
        base_visual: Path,
        mixed_audio: Path,
        ass_path: Path | None,
        destination: Path,
        canvas: CanvasConfig,
        video_bitrate: str,
        encoder_preference: Literal["auto", "nvenc", "cpu"],
    ) -> tuple[str, bool, str | None]:
        """Composite captions when needed; otherwise mux without a second video encode."""

        ffmpeg = _ffmpeg()
        if ass_path is None:
            _run_ffmpeg([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(base_visual), "-i", str(mixed_audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(destination),
            ], "single-pass production mux")
            return "copy", False, None

        graph = f"[0:v]ass='{_filter_path(ass_path)}'[vout]"
        requested = encoder_preference

        def command(encoder: str) -> list[str]:
            return [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(base_visual), "-i", str(mixed_audio),
                "-filter_complex", graph, "-map", "[vout]", "-map", "1:a:0",
                "-c:v", encoder, "-preset", "p4" if encoder == "h264_nvenc" else "medium",
                "-b:v", video_bitrate, "-pix_fmt", canvas.pixel_format, "-r", str(canvas.fps),
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination),
            ]

        if requested == "cpu":
            _run_ffmpeg(command("libx264"), "caption composite encode")
            return "libx264", False, None
        if requested == "nvenc" and not nvenc_available():
            raise ProductionRenderError("Запрошен production_render.encoder=nvenc, но h264_nvenc недоступен.")
        if requested == "auto" and not nvenc_available():
            _run_ffmpeg(command("libx264"), "caption composite encode")
            return "libx264", True, "NVENC недоступен: production render безопасно выполнен на CPU."
        try:
            _run_ffmpeg(command("h264_nvenc"), "caption composite encode")
            return "h264_nvenc", False, None
        except ProductionRenderError as error:
            if requested == "nvenc":
                raise
            _run_ffmpeg(command("libx264"), "caption composite encode CPU fallback")
            return "libx264", True, f"NVENC render failed; CPU fallback used: {_safe_error(error)}"

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
        destination: Path, canvas: CanvasConfig, durations: list[float], transitions: list[VideoTransition],
    ) -> tuple[str, bool, str | None]:
        if not clips:
            raise ProductionRenderError("Video timeline has no renderable visual clips.")
        ffmpeg = _ffmpeg()
        inputs: list[str] = []
        for clip in clips:
            inputs.extend(["-i", str(clip)])
        inputs.extend(["-i", str(mixed_audio)])
        transition_plan: str | list[VideoTransition] = self.config.production_render.transitions
        if self.config.production_render.transitions == "short_crossfade" and any(
            item.transition_type == "cut" for item in transitions
        ):
            transition_plan = transitions
        graph, video_label = _timeline_filter(durations, transition_plan)
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
    return _split_timeline_at_scene_boundaries(timeline, source_info, config), fallback_reasons


def _source_range_for_audio(
    audio_clip: Any, segment: Any, transcript_ranges: dict[int, tuple[float, float]],
) -> tuple[float, float, str | None, str | None, str]:
    if isinstance(segment, DialogueSegment):
        return segment.source_start_seconds, segment.source_end_seconds, segment.fact_id, segment.speaker, "dialogue"
    if isinstance(segment, NarrationSegment):
        if segment.source_ranges:
            source = segment.source_ranges[0]
            return (
                source.source_start_seconds,
                source.source_end_seconds,
                segment.fact_ids[0] if segment.fact_ids else None,
                "narrator",
                "narration",
            )
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


def _split_timeline_at_scene_boundaries(
    timeline: VideoTimeline, source_info: dict[str, Any], config: ProductionRenderConfig,
) -> VideoTimeline:
    """Turn reliable source shot boundaries into renderable visual intervals.

    Source and output time are one-to-one for ordinary mapped clips.  Short
    freeze-pad clips keep their established fallback intact rather than being
    split into potentially invalid source ranges.
    """

    boundaries = _scene_boundary_times(source_info)
    if not boundaries:
        return timeline
    expanded: list[VideoClipModel] = []
    source_parents: list[str | None] = []
    for clip in timeline.clips:
        if not isinstance(clip, SourceVideoClip) or clip.freeze_duration_seconds > 0:
            expanded.append(clip)
            source_parents.append(None)
            continue
        assert clip.source_start_seconds is not None and clip.source_end_seconds is not None
        source_duration = clip.source_end_seconds - clip.source_start_seconds
        if abs(source_duration - clip.duration_seconds) > 0.02:
            expanded.append(clip)
            source_parents.append(None)
            continue
        cuts = [
            item for item in boundaries
            if clip.source_start_seconds + 0.04 < item < clip.source_end_seconds - 0.04
        ]
        if not cuts:
            expanded.append(clip)
            source_parents.append(None)
            continue
        points = [clip.source_start_seconds, *cuts, clip.source_end_seconds]
        for index, (left, right) in enumerate(zip(points, points[1:]), start=1):
            duration = round(right - left, 3)
            if duration <= 0.02:
                continue
            expanded.append(clip.model_copy(update={
                "clip_id": f"{clip.clip_id}-scene-{index:02d}",
                "duration_seconds": duration, "source_start_seconds": round(left, 3),
                "source_end_seconds": round(right, 3), "freeze_duration_seconds": 0.0,
            }))
            source_parents.append(clip.clip_id)
    if len(expanded) == len(timeline.clips):
        return timeline
    normalized: list[VideoClipModel] = []
    cursor = 0.0
    for order, clip in enumerate(expanded, start=1):
        # Preserve the exact total timeline duration after decimal rounding.
        duration = clip.duration_seconds if order < len(expanded) else round(timeline.duration_seconds - cursor, 3)
        normalized.append(clip.model_copy(update={
            "order": order, "timeline_start_seconds": round(cursor, 3),
            "timeline_end_seconds": round(cursor + duration, 3), "duration_seconds": duration,
        }))
        cursor += duration
    transitions: list[VideoTransition] = []
    for index, (left, right) in enumerate(zip(normalized, normalized[1:])):
        same_source_shot_parent = source_parents[index] is not None and source_parents[index] == source_parents[index + 1]
        transition_type = "cut" if same_source_shot_parent else config.transitions
        transitions.append(VideoTransition(
            transition_type=transition_type, from_clip_id=left.clip_id, to_clip_id=right.clip_id,
            duration_seconds=0 if transition_type == "cut" else 0.15,
        ))
    return VideoTimeline(clips=normalized, transitions=transitions, duration_seconds=timeline.duration_seconds)


def make_crop_plan(source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig) -> CropPlan:
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    rotation = int(source_info.get("rotation", 0))
    target = canvas.width / canvas.height
    strategy = config.crop_strategy
    subject = _subject_anchor(source_info)
    automatic_subject_crop = strategy == "safe_auto"
    # Auto mode must never make a destructive landscape crop without reliable
    # subject evidence. The full frame with a background is safer than guessing.
    if strategy == "safe_auto":
        if width / height > target * 1.03 and subject is None:
            return CropPlan(strategy="fit_blur_background", source_width=width, source_height=height, display_rotation_degrees=rotation)
        strategy = "center_crop"
    if strategy in {"fit_blur_background", "fit_solid_background"}:
        return CropPlan(strategy=strategy, source_width=width, source_height=height, display_rotation_degrees=rotation)
    if width / height >= target:
        crop_height = height
        crop_width = _even_down(height * target)
    else:
        crop_width = width
        crop_height = _even_down(width / target)
    if strategy == "top_crop":
        x, y = (width - crop_width) // 2, 0
        normalized_x, normalized_y, strategy = 0.5, 0.0, "top_crop"
    elif automatic_subject_crop and subject is not None and strategy == "center_crop":
        normalized_x, normalized_y = subject
        x = _even_down((width - crop_width) * normalized_x)
        y = _even_down((height - crop_height) * normalized_y)
        strategy = "manual_normalized_crop"
    else:
        x = _even_down((width - crop_width) * config.manual_crop_x)
        y = _even_down((height - crop_height) * config.manual_crop_y)
        normalized_x, normalized_y, strategy = config.manual_crop_x, config.manual_crop_y, strategy
    x = max(0, min(x, width - crop_width))
    y = max(0, min(y, height - crop_height))
    return CropPlan(
        strategy=strategy, source_width=width, source_height=height, display_rotation_degrees=rotation,
        normalized_x=normalized_x, normalized_y=normalized_y,
        crop_width=crop_width, crop_height=crop_height, crop_x=x, crop_y=y,
    )


def _subject_anchor(source_info: dict[str, Any]) -> tuple[float, float] | None:
    raw = source_info.get("subject_keyframes")
    if not isinstance(raw, list):
        return None
    observations: list[tuple[float, float, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x, y, confidence = float(item["normalized_x"]), float(item["normalized_y"]), float(item["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and confidence >= 0.55:
            observations.append((x, y, confidence))
    if not observations:
        return None
    weight = sum(item[2] for item in observations)
    return (
        sum(item[0] * item[2] for item in observations) / weight,
        sum(item[1] * item[2] for item in observations) / weight,
    )


_STATIC_SUBJECT_CONFIDENCE = 0.55
_TRACKING_CONFIDENCE = 0.70
_MINIMUM_FOCUS_HOLD_SECONDS = 1.25
_ACTIVE_SPEAKER_HYSTERESIS = 0.08
_MAX_SAFE_TRACKING_SPEED = 0.28
_TRACKING_MODES = {"face_tracking", "person_tracking", "active_speaker_tracking", "object_tracking"}
_TRACKING_TARGETS = {
    "primary_face", "primary_person", "active_speaker", "important_object",
    "screen_region", "subject_group", "scene_center", "none",
}
_SCENE_TYPES = {
    "TALKING_HEAD", "INTERVIEW_SINGLE", "INTERVIEW_MULTI", "PODCAST",
    "PRODUCT_DEMO", "HANDS_ON_DEMO", "PRESENTATION_SCREEN", "GAMEPLAY",
    "CINEMATIC_SCENE", "FULL_BODY_ACTION", "UNKNOWN",
}
_PERSON_SCENES = {"TALKING_HEAD", "INTERVIEW_SINGLE", "PODCAST"}
_ACTION_SCENES = {"CINEMATIC_SCENE", "FULL_BODY_ACTION"}
_SCREEN_SCENES = {"PRESENTATION_SCREEN", "GAMEPLAY"}
_PRODUCT_SCENES = {"PRODUCT_DEMO", "HANDS_ON_DEMO"}


def _attach_visual_evidence_context(source_info: dict[str, Any], visual_analysis: dict[str, Any] | None) -> None:
    """Carry the persisted visual-analysis evidence state into composition.

    Older callers may pass subject keyframes directly.  They remain supported
    as valid evidence when observations are present; an absent analysis is
    explicitly unavailable rather than silently treated as a safe crop.
    """

    analysis = visual_analysis if isinstance(visual_analysis, dict) else {}
    raw_status = analysis.get("evidence_status")
    if raw_status in {"valid", "fallback", "evidence_unavailable"}:
        status = str(raw_status)
    elif analysis.get("status") == "completed" and isinstance(analysis.get("subject_keyframes"), list):
        status = "valid"
    elif analysis.get("status") == "fallback":
        status = "fallback"
    elif isinstance(source_info.get("subject_keyframes"), list) and source_info["subject_keyframes"]:
        status = "valid"
    else:
        status = "evidence_unavailable"
    source_info["visual_evidence_status"] = status
    source_info["visual_evidence"] = {
        "analysis_schema_version": analysis.get("schema_version"),
        "analysis_status": analysis.get("status") or ("legacy_keyframes" if status == "valid" else "unavailable"),
        "sample_count": analysis.get("sample_count", 0),
        "keyframe_count": len(source_info.get("subject_keyframes") or []),
    }
    provenance = analysis.get("fallback_provenance")
    if isinstance(provenance, dict):
        source_info["visual_fallback_provenance"] = {
            "stage": str(provenance.get("stage") or "visual_analysis"),
            "reason": str(provenance.get("reason") or analysis.get("reason") or "unspecified"),
        }
    elif status != "valid":
        source_info["visual_fallback_provenance"] = {
            "stage": "visual_analysis",
            "reason": str(analysis.get("reason") or "visual_evidence_unavailable"),
        }


def _visual_evidence_context(source_info: dict[str, Any], observations: list[SubjectBounds]) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Normalize current and legacy visual-analysis records for one segment."""

    raw = source_info.get("visual_evidence_status")
    if raw in {"valid", "fallback", "evidence_unavailable"}:
        status = str(raw)
    elif observations:
        status = "valid"
    else:
        status = "evidence_unavailable"
    evidence = dict(source_info.get("visual_evidence") or {})
    evidence.update({
        "segment_observation_count": len(observations),
        "source_scene_type": str(source_info.get("scene_type") or ""),
    })
    raw_provenance = source_info.get("visual_fallback_provenance")
    provenance = (
        {"stage": str(raw_provenance.get("stage") or "visual_analysis"), "reason": str(raw_provenance.get("reason") or "unspecified")}
        if isinstance(raw_provenance, dict)
        else {"stage": "composition", "reason": "No valid visual-analysis evidence was supplied for this segment."}
    )
    return status, evidence, provenance


def _scene_type_for_segment(source_info: dict[str, Any], observations: list[SubjectBounds]) -> str:
    """Choose a bounded visual scene type from evidence, never transcript guesswork."""

    explicit = str(source_info.get("scene_type") or "").upper()
    if explicit in _SCENE_TYPES:
        return explicit
    weights: dict[str, float] = {}
    for item in observations:
        if item.scene_type in _SCENE_TYPES and item.scene_type != "UNKNOWN":
            weights[item.scene_type] = weights.get(item.scene_type, 0) + item.confidence
    if weights:
        return max(weights, key=weights.get)
    targets = {item.target for item in observations}
    face_count = max((item.visible_face_count for item in observations), default=0)
    if "screen_region" in targets:
        return "PRESENTATION_SCREEN"
    if "important_object" in targets:
        return "PRODUCT_DEMO"
    if "subject_group" in targets or face_count > 1:
        return "INTERVIEW_MULTI"
    if targets & {"primary_face", "primary_person", "active_speaker"}:
        return "TALKING_HEAD"
    return "UNKNOWN"


def _framing_intent(scene_type: str) -> str:
    if scene_type in _PERSON_SCENES:
        return "CHEST_UP_PERSON"
    if scene_type == "INTERVIEW_MULTI":
        return "GROUP_CONVERSATION"
    if scene_type in _PRODUCT_SCENES:
        return "PRODUCT_OR_HANDS"
    if scene_type in _SCREEN_SCENES:
        return "SCREEN_FIRST"
    if scene_type in _ACTION_SCENES:
        return "PRESERVE_WIDE_ACTION"
    return "CONSERVATIVE_WIDE"


def build_reframe_plan(
    source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig, timeline: VideoTimeline,
) -> ReframePlan:
    """Build explainable scene-level composition decisions before selecting a crop.

    A crop is static unless it demonstrably cannot retain the relevant subject.
    Dynamic tracking is only emitted for a confident, single target with enough
    time to move smoothly.  The resulting decisions are persisted even when a
    safe wide composition is selected, so a report can explain why tracking was
    deliberately avoided.
    """

    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    if not timeline.clips:
        return _legacy_reframe_plan(source_info, canvas, config)
    segments = build_composition_segments(source_info, canvas, config, timeline)
    segments = _apply_active_speaker_hysteresis(segments, canvas)
    segments = _validate_tracking_decisions(segments, canvas)
    segments = _apply_composition_quality_diagnostics(segments, canvas, source_info)
    observations = [
        ReframeKeyframe(
            time_seconds=bound.time_seconds, normalized_x=bound.center_x,
            normalized_y=bound.center_y, confidence=bound.confidence,
        )
        for segment in segments for bound in segment.subject_bounds
        if bound.confidence >= _STATIC_SUBJECT_CONFIDENCE
    ]
    subject_strategies = {"subject_crop", "face_crop"}
    if any(segment.strategy in subject_strategies for segment in segments):
        strategy = "subject_crop"
        fallback_reason = None
    elif all(segment.strategy == "original_vertical" for segment in segments):
        strategy = "original_vertical"
        fallback_reason = None
    elif any(segment.strategy == "center_crop" for segment in segments):
        strategy = "center_crop"
        fallback_reason = next((segment.fallback_reason for segment in segments if segment.fallback_reason), None)
    elif any(segment.strategy == "fit_with_blur" for segment in segments):
        strategy = "blur_fallback"
        fallback_reason = next((segment.fallback_reason for segment in segments if segment.fallback_reason), None)
    else:
        strategy = "contain"
        fallback_reason = next((segment.fallback_reason for segment in segments if segment.fallback_reason), None)
    return ReframePlan(
        strategy=strategy, source_width=width, source_height=height,
        canvas_width=canvas.width, canvas_height=canvas.height,
        subtitle_reserved_bottom_ratio=0.16 if strategy != "blur_fallback" else 0.20,
        keyframes=_smooth_reframe_keyframes(observations) if observations else [],
        composition_segments=segments, subject_detection_used=bool(observations),
        fallback_reason=fallback_reason,
    )


def build_composition_segments(
    source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig, timeline: VideoTimeline,
) -> list[CompositionSegment]:
    """Return one deterministic decision per visual timeline interval.

    Input observations are intentionally sparse and non-identifying.  Missing
    evidence therefore selects a wider composition instead of inventing a face
    target or silently using the last subject seen elsewhere in the source.
    """

    result: list[CompositionSegment] = []
    outgoing_transitions = {item.from_clip_id: item.transition_type for item in timeline.transitions}
    for index, clip in enumerate(timeline.clips, start=1):
        transition = outgoing_transitions.get(clip.clip_id, "cut")
        segment_id = f"composition-{index:03d}"
        if not isinstance(clip, (SourceVideoClip, FreezeFrameClip)):
            result.append(CompositionSegment(
                segment_id=segment_id, visual_clip_id=clip.clip_id,
                start_seconds=clip.timeline_start_seconds, end_seconds=clip.timeline_end_seconds,
                strategy="safe_fallback", confidence=0, fallback_reason=clip.fallback_reason or "No source visual is available.",
                transition_type=transition, tracking_mode="safe_fallback", tracking_target="none",
                tracking_required=False, tracking_confidence=0,
                tracking_reason="A fill visual has no trackable source subject.", static_crop_sufficient=False,
                tracking_risk="unsafe", fallback_strategy="safe_fallback", wide_safe_layout_required=True,
                tracking_validation_status="not_applicable",
            ))
            continue
        assert clip.source_start_seconds is not None and clip.source_end_seconds is not None
        observations = _subject_observations(
            source_info, clip.source_start_seconds, clip.source_end_seconds,
        )
        result.append(_decide_composition_segment(
            segment_id, clip, observations, source_info, canvas, config, transition,
        ))
    return _apply_composition_quality_diagnostics(result, canvas, source_info)


def apply_composition_segments(timeline: VideoTimeline, segments: list[CompositionSegment]) -> VideoTimeline:
    """Apply the resolved segment crop to its corresponding renderable visual clip."""

    by_clip = {segment.visual_clip_id: segment for segment in segments if segment.visual_clip_id}
    clips: list[VideoClipModel] = []
    for clip in timeline.clips:
        segment = by_clip.get(clip.clip_id)
        if not isinstance(clip, (SourceVideoClip, FreezeFrameClip)) or segment is None or segment.target_crop is None:
            clips.append(clip)
            continue
        updates: dict[str, Any] = {"crop_plan": segment.target_crop}
        if segment.fallback_reason:
            updates.update({"status": "fallback", "fallback_reason": segment.fallback_reason})
        clips.append(clip.model_copy(update=updates))
    return timeline.model_copy(update={"clips": clips})


def _legacy_reframe_plan(source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig) -> ReframePlan:
    """Keep the public no-timeline helper compatible with the former contract."""

    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    target = canvas.width / canvas.height
    ratio = width / height
    observations = _subject_observations(source_info, 0, float(source_info.get("video_duration") or float("inf")))
    keyframes = [
        ReframeKeyframe(
            time_seconds=item.time_seconds, normalized_x=item.center_x,
            normalized_y=item.center_y, confidence=item.confidence,
        )
        for item in observations
    ]
    confident = [item for item in keyframes if item.confidence >= _STATIC_SUBJECT_CONFIDENCE]
    if ratio <= target * 1.03:
        return ReframePlan(
            strategy="original_vertical", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
        )
    if config.crop_strategy == "safe_auto" and not confident:
        return ReframePlan(
            strategy="blur_fallback", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.20,
            fallback_reason="Subject confidence is insufficient; full-frame blur fallback avoids an unsafe crop.",
        )
    if config.crop_strategy == "safe_auto" and confident:
        return ReframePlan(
            strategy="subject_crop", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
            keyframes=_smooth_reframe_keyframes(confident), subject_detection_used=True,
        )
    if config.crop_strategy == "safe_auto":
        return ReframePlan(
            strategy="center_crop", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
            fallback_reason="No high-confidence subject observations are available.",
        )
    if config.crop_strategy == "center_crop":
        return ReframePlan(
            strategy="center_crop", source_width=width, source_height=height,
            canvas_width=canvas.width, canvas_height=canvas.height, subtitle_reserved_bottom_ratio=0.16,
            fallback_reason="Manual centre crop takes priority over automatic subject tracking.",
        )
    return ReframePlan(
        strategy="blur_fallback" if config.crop_strategy == "fit_blur_background" else "contain",
        source_width=width, source_height=height, canvas_width=canvas.width, canvas_height=canvas.height,
        subtitle_reserved_bottom_ratio=0.16, fallback_reason="Crop strategy preserves the full source frame.",
    )


def _decide_composition_segment(
    segment_id: str, clip: SourceVideoClip | FreezeFrameClip, observations: list[SubjectBounds],
    source_info: dict[str, Any], canvas: CanvasConfig, config: ProductionRenderConfig, transition: str,
) -> CompositionSegment:
    assert clip.source_start_seconds is not None and clip.source_end_seconds is not None
    scene_change_count = _scene_change_count(source_info, clip.source_start_seconds, clip.source_end_seconds)
    common = {
        "segment_id": segment_id, "visual_clip_id": clip.clip_id,
        "start_seconds": clip.timeline_start_seconds, "end_seconds": clip.timeline_end_seconds,
        "source_start_seconds": clip.source_start_seconds, "source_end_seconds": clip.source_end_seconds,
        "speaker_id": clip.speaker,
        "scene_change_count": scene_change_count,
        "subject_bounds": observations, "transition_type": transition,
        "editorial_intent": dict(source_info.get("composition_intent") or {}),
    }
    if scene_change_count:
        common["transition_type"] = "cut"
    duration = max(0.01, clip.source_end_seconds - clip.source_start_seconds)
    scene_type = _scene_type_for_segment(source_info, observations)
    framing_intent = _framing_intent(scene_type)
    if config.crop_strategy in {"center_crop", "fit_blur_background", "fit_solid_background", "top_crop", "manual_normalized_crop"}:
        crop = make_crop_plan(source_info, canvas, config)
        strategy = "fit_with_blur" if crop.strategy == "fit_blur_background" else "center_crop"
        return CompositionSegment(
            **common, strategy=strategy, target_crop=crop, confidence=_mean_confidence(observations),
            target_center_x=crop.normalized_x, target_center_y=crop.normalized_y,
            target_scale=_crop_scale(crop), tracking_mode="none", tracking_target="none",
            tracking_required=False, tracking_confidence=0,
            tracking_reason=f"Manual crop strategy '{config.crop_strategy}' takes priority over automatic tracking.",
            static_crop_sufficient=True, tracking_risk="none", fallback_strategy="none",
        )
    if _source_is_vertical(source_info, canvas):
        return CompositionSegment(
            **common, strategy="original_vertical", target_crop=_crop_plan_for_center(source_info, canvas, 0.5, 0.5),
            confidence=_mean_confidence(observations), tracking_mode="scene_wide", tracking_target="scene_center",
            tracking_required=False, tracking_confidence=_mean_confidence(observations),
            tracking_reason="The source already fits the vertical canvas; a wider stable composition is safer.",
            static_crop_sufficient=True, tracking_risk="none", fallback_strategy="none",
        )
    if not observations:
        if config.crop_strategy == "safe_auto":
            crop = _wide_crop(source_info, canvas)
            return CompositionSegment(
                **common, strategy="fit_with_blur", target_crop=crop, confidence=0,
                target_scale=_crop_scale(crop), tracking_mode="safe_fallback", tracking_target="none",
                tracking_required=False, tracking_confidence=0,
                tracking_reason="No segment-local subject detection is reliable enough to justify a crop.",
                static_crop_sufficient=False, tracking_risk="unsafe", fallback_strategy="fit_with_blur",
                wide_safe_layout_required=True,
                fallback_reason="No reliable local subject observations; full-frame layout avoids an unsafe crop.",
            )
        crop = _crop_plan_for_center(source_info, canvas, 0.5, 0.5)
        return CompositionSegment(
            **common, strategy="center_crop", target_crop=crop, confidence=0,
            target_center_x=0.5, target_center_y=0.5, target_scale=_crop_scale(crop),
            tracking_mode="none", tracking_target="scene_center", tracking_required=False, tracking_confidence=0,
            tracking_reason="No reliable local subject observations; use the configured static centre crop.",
            static_crop_sufficient=True, tracking_risk="low", fallback_strategy="scene_wide",
            fallback_reason="No high-confidence subject observations are available.",
        )

    confidence = _mean_confidence(observations)
    target = _dominant_target(observations)
    center_x, center_y = _weighted_center(observations)
    static_sufficient = _static_crop_sufficient(observations, source_info, canvas)
    risk = _tracking_risk(observations, scene_change_count)
    face_too_small_for_tracking = target == "primary_face" and max(
        (min(item.width, item.height) for item in observations), default=0,
    ) < 0.07
    face_count = max((item.visible_face_count for item in observations), default=0)
    active_speaker_confidence = sum(item.active_speaker_confidence * item.confidence for item in observations) / max(
        sum(item.confidence for item in observations), 0.001,
    )

    if scene_type == "UNKNOWN":
        crop = _wide_crop(source_info, canvas)
        reason = "Scene type is unknown; use a conservative full-scene layout instead of a face-centric crop."
        return CompositionSegment(
            **common, strategy="fit_with_blur" if crop.strategy == "fit_blur_background" else "scene_wide",
            target_crop=crop, confidence=confidence, target_center_x=0.5, target_center_y=0.5,
            target_scale=_crop_scale(crop), tracking_mode="scene_wide", tracking_target="scene_center",
            tracking_required=False, tracking_confidence=confidence, tracking_reason=reason,
            static_crop_sufficient=False, tracking_risk="medium", fallback_strategy="scene_wide",
            wide_safe_layout_required=True, fallback_reason=reason,
        )

    if scene_type in _ACTION_SCENES:
        crop = _wide_crop(source_info, canvas)
        reason = "The scene contains action or cinematic composition; preserve the wide source composition."
        return CompositionSegment(
            **common, strategy="fit_with_blur" if crop.strategy == "fit_blur_background" else "scene_wide",
            target_crop=crop, confidence=confidence, target_center_x=0.5, target_center_y=0.5,
            target_scale=_crop_scale(crop), tracking_mode="scene_wide", tracking_target="scene_center",
            tracking_required=False, tracking_confidence=confidence, tracking_reason=reason,
            static_crop_sufficient=False, tracking_risk="low", fallback_strategy="scene_wide",
            wide_safe_layout_required=True,
        )

    if scene_type in _SCREEN_SCENES:
        crop = _wide_crop(source_info, canvas)
        return CompositionSegment(
            **common, strategy="fit_with_blur" if crop.strategy == "fit_blur_background" else "scene_wide",
            target_crop=crop, confidence=confidence, target_center_x=center_x, target_center_y=center_y,
            target_scale=_crop_scale(crop), tracking_mode="scene_wide", tracking_target="screen_region",
            tracking_required=False, tracking_confidence=confidence,
            tracking_reason="Screen or gameplay content is scene-primary; preserve a wide readable layout.",
            static_crop_sufficient=False, tracking_risk="none", fallback_strategy="none", wide_safe_layout_required=True,
        )

    if scene_change_count:
        crop = _wide_crop(source_info, canvas)
        reason = "Shot changes occur inside this interval; a controlled cut/wide composition is safer than tracking across unrelated frames."
        return CompositionSegment(
            **common, strategy="fit_with_blur" if crop.strategy == "fit_blur_background" else "scene_wide",
            target_crop=crop, confidence=confidence, target_center_x=0.5, target_center_y=0.5,
            target_scale=_crop_scale(crop), tracking_mode="scene_wide", tracking_target="scene_center",
            tracking_required=False, tracking_confidence=confidence, tracking_reason=reason,
            static_crop_sufficient=False, tracking_risk="unsafe", fallback_strategy="scene_wide",
            wide_safe_layout_required=True, fallback_reason=reason,
        )

    if target == "screen_region":
        crop = _wide_crop(source_info, canvas)
        return CompositionSegment(
            **common, strategy="fit_with_blur", target_crop=crop, confidence=confidence,
            target_center_x=center_x, target_center_y=center_y, target_scale=_crop_scale(crop),
            tracking_mode="scene_wide", tracking_target="screen_region", tracking_required=False,
            tracking_confidence=confidence,
            tracking_reason="Screen content is the primary visual; a wide stable layout is more readable than face tracking.",
            static_crop_sufficient=False, tracking_risk="none", fallback_strategy="none", wide_safe_layout_required=True,
        )

    if face_count > 1 or target == "subject_group":
        group_fits = _group_crop_sufficient(observations, source_info, canvas)
        crop = _crop_plan_for_center(source_info, canvas, center_x, center_y) if group_fits else _wide_crop(source_info, canvas)
        if active_speaker_confidence >= 0.80 and clip.speaker and duration >= _MINIMUM_FOCUS_HOLD_SECONDS and not static_sufficient and risk in {"none", "low", "medium"}:
            dynamic_crop = _tracking_crop(source_info, canvas, observations, clip.source_start_seconds, duration)
            return CompositionSegment(
                **common, strategy="face_crop", target_crop=dynamic_crop, confidence=confidence,
                target_center_x=center_x, target_center_y=center_y, target_scale=_crop_scale(dynamic_crop),
                tracking_mode="active_speaker_tracking", tracking_target="active_speaker", tracking_required=True,
                tracking_confidence=active_speaker_confidence,
                tracking_reason="A confident active-speaker signal persists beyond the minimum focus hold duration.",
                static_crop_sufficient=False, tracking_risk=risk, fallback_strategy="group_framing",
                minimum_focus_hold_seconds=_MINIMUM_FOCUS_HOLD_SECONDS,
            )
        return CompositionSegment(
            **common, strategy="group_framing" if group_fits else "fit_with_blur", target_crop=crop, confidence=confidence,
            target_center_x=center_x, target_center_y=center_y, target_scale=_crop_scale(crop),
            tracking_mode="group_framing", tracking_target="subject_group", tracking_required=False,
            tracking_confidence=confidence,
            tracking_reason="Multiple relevant faces are present; hold the group rather than switching on short or uncertain turns.",
            static_crop_sufficient=group_fits, tracking_risk="low" if group_fits else "medium",
            fallback_strategy="none" if group_fits else "fit_with_blur", wide_safe_layout_required=not group_fits,
            fallback_reason=None if group_fits else "Multiple important people do not fit safely in a narrow crop.",
        )

    if static_sufficient:
        crop_center_x, crop_center_y = _framing_crop_center(
            observations, source_info, canvas, framing_intent,
        )
        crop = _crop_plan_for_center(source_info, canvas, crop_center_x, crop_center_y)
        return CompositionSegment(
            **common, strategy="subject_crop" if scene_type in _PERSON_SCENES else ("face_crop" if target == "primary_face" else "subject_crop"), target_crop=crop,
            confidence=confidence, target_center_x=crop_center_x, target_center_y=crop_center_y, target_scale=_crop_scale(crop),
            tracking_mode="static_subject_crop", tracking_target=target, tracking_required=False,
            tracking_confidence=confidence,
            tracking_reason=(
                "A chest-up static crop keeps the person, shoulders and headroom inside the safe area."
                if scene_type in _PERSON_SCENES else
                "A single static crop keeps the important subject inside the safe area for this segment."
            ),
            static_crop_sufficient=True, tracking_risk="none" if confidence >= _TRACKING_CONFIDENCE else "low",
            fallback_strategy="none",
        )

    dynamic_target = "important_object" if target == "important_object" else ("primary_face" if target == "primary_face" else "primary_person")
    if (
        len(observations) >= 2 and confidence >= _TRACKING_CONFIDENCE
        and risk in {"none", "low", "medium"} and not face_too_small_for_tracking
    ):
        crop = _tracking_crop(
            source_info, canvas, observations, clip.source_start_seconds, duration, framing_intent=framing_intent,
        )
        mode = "object_tracking" if dynamic_target == "important_object" else (
            "face_tracking" if dynamic_target == "primary_face" else "person_tracking"
        )
        return CompositionSegment(
            **common, strategy="subject_crop" if scene_type in _PERSON_SCENES else ("face_crop" if mode == "face_tracking" else "subject_crop"), target_crop=crop,
            confidence=confidence, target_center_x=center_x, target_center_y=center_y, target_scale=_crop_scale(crop),
            tracking_mode=mode, tracking_target=dynamic_target, tracking_required=True,
            tracking_confidence=confidence,
            tracking_reason="The subject moves beyond a safe static crop and confident observations support smooth tracking.",
            static_crop_sufficient=False, tracking_risk=risk, fallback_strategy="fit_with_blur",
        )

    crop = _wide_crop(source_info, canvas)
    reason = (
        "Face is too small for stable tracking; preserve the wider scene instead."
        if face_too_small_for_tracking
        else "Subject movement exceeds a static crop, but tracking confidence or stability is below the safe threshold."
    )
    return CompositionSegment(
        **common, strategy="fit_with_blur", target_crop=crop, confidence=confidence,
        target_center_x=center_x, target_center_y=center_y, target_scale=_crop_scale(crop),
        tracking_mode="safe_fallback", tracking_target="none", tracking_required=False,
        tracking_confidence=confidence, tracking_reason=reason, static_crop_sufficient=False,
        tracking_risk="unsafe" if risk in {"high", "unsafe"} else "high", fallback_strategy="fit_with_blur",
        wide_safe_layout_required=True, fallback_reason=reason,
    )


def _subject_observations(source_info: dict[str, Any], start: float, end: float) -> list[SubjectBounds]:
    raw = source_info.get("subject_keyframes")
    if not isinstance(raw, list):
        return []
    result: list[SubjectBounds] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            time_seconds = float(item.get("time_seconds", start))
            if time_seconds < start - 0.02 or time_seconds > end + 0.02:
                continue
            target = str(item.get("tracking_target", item.get("target", "primary_person")))
            if target not in _TRACKING_TARGETS:
                target = "primary_person"
            result.append(SubjectBounds(
                time_seconds=max(start, min(end, time_seconds)),
                center_x=float(item["normalized_x"]), center_y=float(item["normalized_y"]),
                width=max(0.02, min(1.0, float(item.get("normalized_width", item.get("width", 0.16))))),
                height=max(0.02, min(1.0, float(item.get("normalized_height", item.get("height", 0.20))))),
                confidence=float(item.get("confidence", 0)), target=target,
                visible_face_count=max(0, int(item.get("visible_face_count", item.get("face_count", 1)))),
                active_speaker_confidence=float(item.get("active_speaker_confidence", 0)),
                scene_id=str(item["scene_id"])[:160] if item.get("scene_id") is not None else None,
                scene_type=str(item.get("scene_type", "UNKNOWN")).upper() if str(item.get("scene_type", "UNKNOWN")).upper() in _SCENE_TYPES else "UNKNOWN",
                framing_observation=(
                    str(item.get("framing_observation", "unknown"))
                    if str(item.get("framing_observation", "unknown")) in {
                        "head_only", "head_shoulders", "chest_up", "upper_body", "full_body", "object", "screen", "unknown",
                    } else "unknown"
                ),
                eye_line_y=float(item["eye_line_y"]) if item.get("eye_line_y") is not None else None,
                gesture_active=bool(item.get("gesture_active", False)),
                gesture_area_visible=bool(item.get("gesture_area_visible", False)),
            ))
        except (TypeError, ValueError, KeyError):
            continue
    return sorted(result, key=lambda value: value.time_seconds)


def _mean_confidence(observations: list[SubjectBounds]) -> float:
    if not observations:
        return 0.0
    return round(sum(item.confidence for item in observations) / len(observations), 3)


def _weighted_center(observations: list[SubjectBounds]) -> tuple[float, float]:
    weight = sum(item.confidence for item in observations)
    if weight <= 0:
        return 0.5, 0.5
    return (
        sum(item.center_x * item.confidence for item in observations) / weight,
        sum(item.center_y * item.confidence for item in observations) / weight,
    )


def _dominant_target(observations: list[SubjectBounds]) -> str:
    if any(item.target == "screen_region" for item in observations):
        return "screen_region"
    if any(item.target == "subject_group" for item in observations):
        return "subject_group"
    weights: dict[str, float] = {}
    for item in observations:
        weights[item.target] = weights.get(item.target, 0) + item.confidence
    return max(weights, key=weights.get, default="primary_person")


def _source_is_vertical(source_info: dict[str, Any], canvas: CanvasConfig) -> bool:
    return int(source_info["display_width"]) / int(source_info["display_height"]) <= (canvas.width / canvas.height) * 1.03


def _crop_dimensions(source_info: dict[str, Any], canvas: CanvasConfig) -> tuple[int, int]:
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    target = canvas.width / canvas.height
    if width / height >= target:
        return _even_down(height * target), height
    return width, _even_down(width / target)


def _static_crop_sufficient(observations: list[SubjectBounds], source_info: dict[str, Any], canvas: CanvasConfig) -> bool:
    if not observations or _mean_confidence(observations) < _STATIC_SUBJECT_CONFIDENCE:
        return False
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    crop_width, crop_height = _crop_dimensions(source_info, canvas)
    span_x = max(item.center_x + item.width / 2 for item in observations) - min(item.center_x - item.width / 2 for item in observations)
    span_y = max(item.center_y + item.height / 2 for item in observations) - min(item.center_y - item.height / 2 for item in observations)
    return span_x <= (crop_width / width) * 0.78 and span_y <= (crop_height / height) * 0.56


def _group_crop_sufficient(observations: list[SubjectBounds], source_info: dict[str, Any], canvas: CanvasConfig) -> bool:
    if not observations:
        return False
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    crop_width, crop_height = _crop_dimensions(source_info, canvas)
    group_width = max(item.center_x + item.width / 2 for item in observations) - min(item.center_x - item.width / 2 for item in observations)
    group_height = max(item.center_y + item.height / 2 for item in observations) - min(item.center_y - item.height / 2 for item in observations)
    return group_width <= (crop_width / width) * 0.84 and group_height <= (crop_height / height) * 0.70


def _tracking_risk(observations: list[SubjectBounds], scene_change_count: int = 0) -> str:
    if scene_change_count or len({item.scene_id for item in observations if item.scene_id}) > 1:
        return "unsafe"
    confidence = _mean_confidence(observations)
    if confidence < _STATIC_SUBJECT_CONFIDENCE:
        return "unsafe"
    speeds: list[float] = []
    for left, right in zip(observations, observations[1:]):
        elapsed = right.time_seconds - left.time_seconds
        if elapsed <= 0:
            return "high"
        speeds.append(max(abs(right.center_x - left.center_x), abs(right.center_y - left.center_y)) / elapsed)
    max_speed = max(speeds, default=0.0)
    if max_speed > _MAX_SAFE_TRACKING_SPEED:
        return "high"
    if confidence < _TRACKING_CONFIDENCE or len(observations) < 3 or max_speed > 0.12:
        return "medium"
    return "low" if max_speed else "none"


def _scene_change_count(source_info: dict[str, Any], start: float, end: float) -> int:
    return sum(start + 0.02 < timestamp < end - 0.02 for timestamp in _scene_boundary_times(source_info))


def _scene_boundary_times(source_info: dict[str, Any]) -> list[float]:
    raw = source_info.get("scene_boundaries", [])
    if not isinstance(raw, list):
        return []
    timestamps: list[float] = []
    for item in raw:
        try:
            timestamp = float(item["timestamp"] if isinstance(item, dict) else item)
        except (TypeError, KeyError, ValueError):
            continue
        if timestamp >= 0:
            timestamps.append(round(timestamp, 3))
    return sorted(set(timestamps))


def _crop_plan_for_center(source_info: dict[str, Any], canvas: CanvasConfig, center_x: float, center_y: float) -> CropPlan:
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    crop_width, crop_height = _crop_dimensions(source_info, canvas)
    x = max(0, min(_even_down(center_x * width - crop_width / 2), width - crop_width))
    y = max(0, min(_even_down(center_y * height - crop_height / 2), height - crop_height))
    return CropPlan(
        strategy="manual_normalized_crop", source_width=width, source_height=height,
        display_rotation_degrees=int(source_info.get("rotation", 0)), normalized_x=center_x, normalized_y=center_y,
        crop_width=crop_width, crop_height=crop_height, crop_x=x, crop_y=y,
    )


def _framing_crop_center(
    observations: list[SubjectBounds], source_info: dict[str, Any], canvas: CanvasConfig, framing_intent: str,
) -> tuple[float, float]:
    """Place a person chest-up: preserve headroom and put eyes near the upper third.

    The operation only shifts an already aspect-correct crop and clamps it to
    source bounds.  It never introduces a new zoom rule or a separate tracker.
    """

    center_x, center_y = _weighted_center(observations)
    if framing_intent != "CHEST_UP_PERSON" or not observations:
        return center_x, center_y
    width, height = int(source_info["display_width"]), int(source_info["display_height"])
    _crop_width, crop_height = _crop_dimensions(source_info, canvas)
    crop_fraction_y = min(1.0, crop_height / max(height, 1))
    weight = sum(item.confidence for item in observations) or 1.0
    eye_line = sum(
        (item.eye_line_y if item.eye_line_y is not None else item.center_y - item.height * 0.18) * item.confidence
        for item in observations
    ) / weight
    # A target at 0.34 of the crop creates modest headroom without turning a
    # speaking person into a head-only close-up.
    desired_crop_center_y = eye_line + 0.16 * crop_fraction_y
    return center_x, max(crop_fraction_y / 2, min(1 - crop_fraction_y / 2, desired_crop_center_y))


def _wide_crop(source_info: dict[str, Any], canvas: CanvasConfig) -> CropPlan:
    if not _source_is_vertical(source_info, canvas):
        return CropPlan(
            strategy="fit_blur_background", source_width=int(source_info["display_width"]),
            source_height=int(source_info["display_height"]), display_rotation_degrees=int(source_info.get("rotation", 0)),
        )
    return _crop_plan_for_center(source_info, canvas, 0.5, 0.5)


def _tracking_crop(
    source_info: dict[str, Any], canvas: CanvasConfig, observations: list[SubjectBounds], source_start: float, duration: float,
    *, framing_intent: str = "CONSERVATIVE_WIDE",
) -> CropPlan:
    center_x, center_y = _framing_crop_center(observations, source_info, canvas, framing_intent)
    crop = _crop_plan_for_center(source_info, canvas, center_x, center_y)
    raw: list[ReframeKeyframe] = []
    for item in observations:
        relative_time = max(0.0, min(duration, item.time_seconds - source_start))
        item_x, item_y = _framing_crop_center([item], source_info, canvas, framing_intent)
        if raw and relative_time <= raw[-1].time_seconds:
            raw[-1] = ReframeKeyframe(
                time_seconds=raw[-1].time_seconds, normalized_x=item_x,
                normalized_y=item_y, confidence=item.confidence,
            )
        else:
            raw.append(ReframeKeyframe(
                time_seconds=relative_time, normalized_x=item_x,
                normalized_y=item_y, confidence=item.confidence,
            ))
    if raw and raw[0].time_seconds > 0:
        raw.insert(0, raw[0].model_copy(update={"time_seconds": 0.0}))
    if raw and raw[-1].time_seconds < duration:
        raw.append(raw[-1].model_copy(update={"time_seconds": duration}))
    return crop.model_copy(update={"tracking_keyframes": _smooth_reframe_keyframes(raw)})


def _crop_scale(crop: CropPlan) -> float | None:
    if not crop.crop_width or not crop.crop_height:
        return None
    return round(max(crop.source_width / crop.crop_width, crop.source_height / crop.crop_height), 3)


def _apply_active_speaker_hysteresis(
    segments: list[CompositionSegment], canvas: CanvasConfig,
) -> list[CompositionSegment]:
    """Avoid ping-pong speaker cuts unless the new focus is clearly stronger.

    The sparse visual analyzer cannot safely attribute every short utterance to
    a face.  A new speaker therefore needs both the minimum hold interval and a
    confidence advantage over the held focus.  Otherwise the composition holds
    the group in a wide, stable frame.
    """

    result: list[CompositionSegment] = []
    held: CompositionSegment | None = None
    for segment in segments:
        if segment.tracking_mode != "active_speaker_tracking":
            result.append(segment)
            continue
        duration = segment.end_seconds - segment.start_seconds
        switch_is_uncertain = (
            held is not None and segment.speaker_id != held.speaker_id
            and segment.tracking_confidence < held.tracking_confidence + _ACTIVE_SPEAKER_HYSTERESIS
        )
        if duration >= segment.minimum_focus_hold_seconds and not switch_is_uncertain:
            result.append(segment)
            held = segment
            continue
        crop = _wide_crop_from_segment(segment, canvas)
        reason = (
            "Active-speaker switch did not clear the confidence hysteresis; keep both participants visible."
            if switch_is_uncertain
            else "Active-speaker turn is shorter than the minimum focus hold duration; keep both participants visible."
        )
        result.append(segment.model_copy(update={
            "strategy": "fit_with_blur" if crop.strategy == "fit_blur_background" else "group_framing",
            "target_crop": crop, "target_center_x": 0.5, "target_center_y": 0.5,
            "target_scale": _crop_scale(crop), "tracking_mode": "group_framing",
            "tracking_target": "subject_group", "tracking_required": False,
            "tracking_reason": reason, "tracking_risk": "medium",
            "fallback_strategy": "group_framing", "wide_safe_layout_required": True,
            "tracking_validation_status": "not_applicable",
        }))
    return result


def _validate_tracking_decisions(segments: list[CompositionSegment], canvas: CanvasConfig) -> list[CompositionSegment]:
    """Repair dynamic plans that fail the persisted tracking-quality contract."""

    result: list[CompositionSegment] = []
    for segment in segments:
        if segment.tracking_mode not in _TRACKING_MODES:
            result.append(segment)
            continue
        reasons: list[str] = []
        crop = segment.target_crop
        frames = crop.tracking_keyframes if crop is not None else []
        diagnostics = _tracking_quality_metrics(segment)
        if segment.static_crop_sufficient:
            reasons.append("A static crop is already sufficient.")
        if segment.tracking_risk in {"high", "unsafe"}:
            reasons.append("Crop movement is too abrupt or detection is unstable.")
        if len(frames) < 2:
            reasons.append("Tracking does not have enough independent keyframes.")
        if not diagnostics["target_visible"]:
            reasons.append("The selected crop does not keep the target fully visible.")
        if not diagnostics["target_in_safe_zone"]:
            reasons.append("The target would leave the crop safe zone or lose required headroom.")
        if not diagnostics["subtitle_safe"]:
            reasons.append("The tracked target risks persistent overlap with the subtitle zone.")
        if float(diagnostics["max_crop_speed"]) > _MAX_SAFE_TRACKING_SPEED:
            reasons.append("The crop would need a visibly abrupt correction.")
        if int(diagnostics["minor_correction_count"]) > 2:
            reasons.append("The tracker would make repeated small corrective moves instead of holding a stable crop.")
        if not diagnostics["movement_justified"]:
            reasons.append("Crop movement is not justified by meaningful subject movement.")
        if (
            segment.tracking_mode == "active_speaker_tracking"
            and max((item.visible_face_count for item in segment.subject_bounds), default=0) > 1
            and segment.tracking_confidence < 0.80
        ):
            reasons.append("The active-speaker signal is too weak to switch focus between visible faces.")
        if not reasons:
            result.append(segment.model_copy(update={
                "tracking_validation_status": "passed", "tracking_diagnostics": diagnostics,
                "composition_quality_status": "passed",
            }))
            continue
        message = "Tracking disabled: " + " ".join(reasons)
        if segment.static_crop_sufficient and segment.subject_bounds:
            center_x, center_y = _weighted_center(segment.subject_bounds)
            assert crop is not None
            static_crop = _crop_plan_for_center({
                "display_width": crop.source_width, "display_height": crop.source_height,
                "rotation": crop.display_rotation_degrees,
            }, canvas, center_x, center_y)
            result.append(segment.model_copy(update={
                "strategy": "face_crop" if segment.tracking_target == "primary_face" else "subject_crop",
                "target_crop": static_crop, "target_center_x": center_x, "target_center_y": center_y,
                "target_scale": _crop_scale(static_crop), "tracking_mode": "static_subject_crop",
                "tracking_required": False, "tracking_reason": message,
                "tracking_risk": "low", "fallback_strategy": "static_subject_crop",
                "fallback_reason": message, "tracking_validation_status": "failed_repaired",
                "tracking_validation_reasons": reasons, "tracking_diagnostics": diagnostics,
                "composition_quality_status": "passed_with_warning", "composition_quality_reasons": reasons,
            }))
            continue
        fallback_crop = _wide_crop_from_segment(segment, canvas)
        result.append(segment.model_copy(update={
            "strategy": "fit_with_blur" if fallback_crop.strategy == "fit_blur_background" else "scene_wide",
            "target_crop": fallback_crop, "target_center_x": 0.5, "target_center_y": 0.5,
            "target_scale": _crop_scale(fallback_crop), "tracking_mode": "safe_fallback",
            "tracking_target": "scene_center", "tracking_required": False,
            "tracking_reason": message, "tracking_risk": "unsafe", "fallback_strategy": "fit_with_blur",
            "wide_safe_layout_required": True, "fallback_reason": message,
            "tracking_validation_status": "failed_repaired", "tracking_validation_reasons": reasons,
            "tracking_diagnostics": diagnostics, "composition_quality_status": "passed_with_warning",
            "composition_quality_reasons": reasons,
        }))
    return result


def _tracking_quality_metrics(segment: CompositionSegment) -> dict[str, float | int | bool]:
    """Measure actual tracked crop coverage against the observed target bounds."""

    crop = segment.target_crop
    if crop is None or not crop.crop_width or not crop.crop_height:
        return {
            "target_visible": False, "target_in_safe_zone": False, "subtitle_safe": False,
            "max_crop_speed": 0.0, "minor_correction_count": 0,
            "subject_motion": 0.0, "crop_motion": 0.0, "movement_justified": False,
            "subject_screen_ratio": 0.0,
        }
    frames = crop.tracking_keyframes
    if not frames or not segment.subject_bounds:
        return {
            "target_visible": False, "target_in_safe_zone": False, "subtitle_safe": False,
            "max_crop_speed": 0.0, "minor_correction_count": 0,
            "subject_motion": 0.0, "crop_motion": 0.0, "movement_justified": False,
            "subject_screen_ratio": 0.0,
        }
    source_start = segment.source_start_seconds or 0.0
    target_visible = True
    target_in_safe_zone = True
    subtitle_safe = True
    subject_ratios: list[float] = []
    for bound in segment.subject_bounds:
        center_x, center_y = _tracking_center_at(frames, max(0.0, bound.time_seconds - source_start))
        crop_x = _crop_origin_for_center(center_x, crop.source_width, crop.crop_width)
        crop_y = _crop_origin_for_center(center_y, crop.source_height, crop.crop_height)
        left = (bound.center_x - bound.width / 2) * crop.source_width
        right = (bound.center_x + bound.width / 2) * crop.source_width
        top = (bound.center_y - bound.height / 2) * crop.source_height
        bottom = (bound.center_y + bound.height / 2) * crop.source_height
        target_visible = target_visible and (
            left >= crop_x and right <= crop_x + crop.crop_width
            and top >= crop_y and bottom <= crop_y + crop.crop_height
        )
        safe_left = crop_x + crop.crop_width * 0.05
        safe_right = crop_x + crop.crop_width * 0.95
        safe_top = crop_y + crop.crop_height * 0.06
        safe_bottom = crop_y + crop.crop_height * 0.84
        target_in_safe_zone = target_in_safe_zone and (
            left >= safe_left and right <= safe_right and top >= safe_top and bottom <= safe_bottom
        )
        subtitle_safe = subtitle_safe and bottom <= safe_bottom
        subject_ratios.append(min(1.0, (bound.width * bound.height) / max(
            (crop.crop_width / crop.source_width) * (crop.crop_height / crop.source_height), 0.001,
        )))
    crop_centers = [
        (
            (_crop_origin_for_center(item.normalized_x, crop.source_width, crop.crop_width) + crop.crop_width / 2) / crop.source_width,
            (_crop_origin_for_center(item.normalized_y, crop.source_height, crop.crop_height) + crop.crop_height / 2) / crop.source_height,
        )
        for item in frames
    ]
    crop_steps = [
        max(abs(right[0] - left[0]), abs(right[1] - left[1]))
        for left, right in zip(crop_centers, crop_centers[1:])
    ]
    crop_speeds = [
        step / max(0.001, right.time_seconds - left.time_seconds)
        for step, left, right in zip(crop_steps, frames, frames[1:])
    ]
    subject_steps = [
        max(abs(right.center_x - left.center_x), abs(right.center_y - left.center_y))
        for left, right in zip(segment.subject_bounds, segment.subject_bounds[1:])
    ]
    subject_motion = max(subject_steps, default=0.0)
    crop_motion = max(crop_steps, default=0.0)
    return {
        "target_visible": target_visible,
        "target_in_safe_zone": target_in_safe_zone,
        "subtitle_safe": subtitle_safe,
        "max_crop_speed": round(max(crop_speeds, default=0.0), 4),
        "minor_correction_count": sum(0 < step <= 0.012 for step in crop_steps),
        "subject_motion": round(subject_motion, 4),
        "crop_motion": round(crop_motion, 4),
        "movement_justified": subject_motion >= 0.035 and crop_motion >= 0.01,
        "subject_screen_ratio": round(sum(subject_ratios) / len(subject_ratios), 4),
    }


def _tracking_center_at(keyframes: list[ReframeKeyframe], time_seconds: float) -> tuple[float, float]:
    if time_seconds <= keyframes[0].time_seconds:
        return keyframes[0].normalized_x, keyframes[0].normalized_y
    for left, right in zip(keyframes, keyframes[1:]):
        if time_seconds <= right.time_seconds:
            progress = (time_seconds - left.time_seconds) / max(0.001, right.time_seconds - left.time_seconds)
            return (
                left.normalized_x + (right.normalized_x - left.normalized_x) * progress,
                left.normalized_y + (right.normalized_y - left.normalized_y) * progress,
            )
    return keyframes[-1].normalized_x, keyframes[-1].normalized_y


def _crop_origin_for_center(center: float, source_size: int, crop_size: int) -> int:
    return max(0, min(_even_down(center * source_size - crop_size / 2), source_size - crop_size))


def _apply_composition_quality_diagnostics(
    segments: list[CompositionSegment], canvas: CanvasConfig, source_info: dict[str, Any],
) -> list[CompositionSegment]:
    """Persist 5D scene-aware checks without taking ownership of Goal 5G readiness."""

    result: list[CompositionSegment] = []
    for segment in segments:
        diagnostics = _composition_diagnostics(segment, canvas)
        decision = _composition_quality_decision(segment, canvas, source_info, diagnostics)
        reasons = list(segment.composition_quality_reasons)
        if diagnostics["blur_coverage_ratio"] > 0.75:
            reasons.append("Blur background occupies too much of the vertical canvas.")
        if segment.subject_bounds and diagnostics["subject_screen_ratio"] < 0.025:
            reasons.append("The detected subject would be too small to read reliably on a mobile screen.")
        if segment.tracking_validation_status == "failed_repaired":
            reasons.append("The composition was recalculated after tracking quality validation failed.")
        reasons.extend(decision.reason_codes)
        quality_status = "failed" if decision.status == "blocked" else (
            "passed_with_warning" if decision.status != "passed" or reasons else "passed"
        )
        result.append(segment.model_copy(update={
            "composition_diagnostics": diagnostics,
            "composition_quality_status": quality_status,
            "composition_quality_reasons": list(dict.fromkeys(reasons)),
            "composition_quality_decision": decision,
        }))
    return result


def _composition_quality_decision(
    segment: CompositionSegment, canvas: CanvasConfig, source_info: dict[str, Any], diagnostics: dict[str, float],
) -> CompositionQualityDecision:
    """Evaluate crop safety against the declared content meaning and evidence state."""

    evidence_status, evidence, fallback_provenance = _visual_evidence_context(source_info, segment.subject_bounds)
    scene_type = _scene_type_for_segment(source_info, segment.subject_bounds)
    framing_intent = _framing_intent(scene_type)
    selected_target = segment.tracking_target
    metrics = _composition_quality_metrics(segment, canvas, diagnostics, scene_type, framing_intent)
    codes: list[str] = []
    hard_codes: set[str] = set()
    has_person = scene_type in _PERSON_SCENES | {"INTERVIEW_MULTI"}
    valid_observations = bool(segment.subject_bounds)

    if evidence_status == "evidence_unavailable":
        codes.append("VISUAL_EVIDENCE_UNAVAILABLE")
    if scene_type == "UNKNOWN":
        codes.append("UNKNOWN_SCENE_FALLBACK")
    if has_person and scene_type in _PERSON_SCENES:
        if metrics["chest_shoulder_framing"] < 0.5:
            codes.append("CHEST_FRAMING_MISSING")
        if metrics["head_only_ratio"] > 0.5:
            codes.extend(["HEAD_ONLY_CROP", "SHOULDERS_CROPPED"])
            hard_codes.add("HEAD_ONLY_CROP")
        if metrics["headroom_ratio"] < 0.035:
            codes.append("INSUFFICIENT_HEADROOM")
        if metrics["face_edge_margin"] < 0.025:
            codes.append("FACE_TOO_CLOSE_TO_EDGE")
            hard_codes.add("FACE_TOO_CLOSE_TO_EDGE")
        if metrics["gesture_active_ratio"] > 0 and metrics["gesture_area_visibility"] < 0.95:
            codes.append("GESTURE_AREA_CROPPED")
            hard_codes.add("GESTURE_AREA_CROPPED")
        if segment.strategy == "face_crop":
            codes.append("WRONG_FRAMING_FOR_CONTENT_TYPE")
            hard_codes.add("WRONG_FRAMING_FOR_CONTENT_TYPE")
    if scene_type == "INTERVIEW_MULTI":
        if selected_target == "active_speaker" and metrics["active_speaker_presence"] < 0.80:
            codes.append("ACTIVE_SPEAKER_MISSING")
            hard_codes.add("ACTIVE_SPEAKER_MISSING")
        if selected_target not in {"active_speaker", "subject_group", "scene_center"}:
            codes.append("WRONG_FRAMING_FOR_CONTENT_TYPE")
            hard_codes.add("WRONG_FRAMING_FOR_CONTENT_TYPE")
    if evidence_status == "valid" and scene_type != "UNKNOWN" and not metrics["scene_framing_match"]:
        codes.append("WRONG_FRAMING_FOR_CONTENT_TYPE")
        hard_codes.add("WRONG_FRAMING_FOR_CONTENT_TYPE")
    if scene_type in _PRODUCT_SCENES and valid_observations and metrics["product_screen_visibility"] < 0.98:
        codes.append("PRODUCT_TARGET_MISSING")
        hard_codes.add("PRODUCT_TARGET_MISSING")
    if scene_type in _SCREEN_SCENES:
        if selected_target != "screen_region" or metrics["product_screen_visibility"] < 0.98:
            codes.append("SCREEN_CONTENT_CROPPED")
            hard_codes.add("SCREEN_CONTENT_CROPPED")
    if scene_type == "FULL_BODY_ACTION":
        if segment.tracking_mode not in {"scene_wide", "safe_fallback"} or metrics["full_body_visibility"] < 0.98:
            codes.append("FULL_BODY_ACTION_CROPPED")
            hard_codes.add("FULL_BODY_ACTION_CROPPED")
    if scene_type == "CINEMATIC_SCENE" and segment.tracking_mode not in {"scene_wide", "safe_fallback"}:
        codes.append("CINEMATIC_COMPOSITION_BROKEN")
        hard_codes.add("CINEMATIC_COMPOSITION_BROKEN")
    if valid_observations and metrics["target_presence"] < 0.98:
        # A detected target that lies outside its resolved crop is unsafe even
        # when another metric looks visually plausible.
        if scene_type in _PRODUCT_SCENES:
            codes.append("PRODUCT_TARGET_MISSING")
            hard_codes.add("PRODUCT_TARGET_MISSING")
        elif scene_type in _SCREEN_SCENES:
            codes.append("SCREEN_CONTENT_CROPPED")
            hard_codes.add("SCREEN_CONTENT_CROPPED")
        elif has_person:
            codes.append("FACE_TOO_CLOSE_TO_EDGE")
            hard_codes.add("FACE_TOO_CLOSE_TO_EDGE")
    if valid_observations and metrics["empty_frame_risk"] >= 0.90:
        codes.append("EMPTY_FRAME_DOMINANT")
        hard_codes.add("EMPTY_FRAME_DOMINANT")
    # A 16:9 source needs roughly 3.16x horizontal scaling to fill 9:16;
    # flag only materially tighter crops instead of blocking normal reframes.
    if metrics["digital_zoom_scale"] > 3.60:
        codes.append("EXCESSIVE_DIGITAL_ZOOM")
        hard_codes.add("EXCESSIVE_DIGITAL_ZOOM")

    codes = list(dict.fromkeys(codes))
    if hard_codes:
        status = "blocked"
    elif evidence_status == "evidence_unavailable":
        status = "evidence_unavailable"
    elif scene_type == "UNKNOWN" or evidence_status == "fallback" or segment.fallback_reason:
        status = "fallback"
    elif codes or segment.tracking_validation_status in {"passed_with_warning", "failed_repaired"}:
        status = "passed_with_warning"
    else:
        status = "passed"
    if status in {"fallback", "evidence_unavailable"} or segment.fallback_reason:
        fallback_provenance = {
            "stage": "composition" if segment.fallback_reason else fallback_provenance["stage"],
            "reason": segment.fallback_reason or fallback_provenance["reason"],
        }
    else:
        fallback_provenance = {}
    return CompositionQualityDecision(
        status=status, scene_type=scene_type, framing_intent=framing_intent,
        selected_target=selected_target, evidence_status=evidence_status, evidence=evidence,
        confidence=segment.confidence, metrics=metrics, reason_codes=codes,
        fallback_provenance=fallback_provenance,
    )


def _composition_quality_metrics(
    segment: CompositionSegment, canvas: CanvasConfig, diagnostics: dict[str, float], scene_type: str, framing_intent: str,
) -> dict[str, float | int | bool]:
    """Measure target retention, framing, empty-frame risk and crop motion."""

    crop = segment.target_crop
    bounds = segment.subject_bounds
    fully_visible: list[bool] = []
    headroom: list[float] = []
    edge_margins: list[float] = []
    full_body_visible: list[bool] = []
    if crop is not None and crop.crop_width and crop.crop_height:
        for bound in bounds:
            crop_x, crop_y, crop_width, crop_height = _crop_window_at(segment, bound.time_seconds)
            left = (bound.center_x - bound.width / 2) * crop.source_width
            right = (bound.center_x + bound.width / 2) * crop.source_width
            top = (bound.center_y - bound.height / 2) * crop.source_height
            bottom = (bound.center_y + bound.height / 2) * crop.source_height
            fully_visible.append(
                left >= crop_x and right <= crop_x + crop_width and top >= crop_y and bottom <= crop_y + crop_height
            )
            headroom.append(max(0.0, (top - crop_y) / max(crop_height, 1)))
            edge_margins.append(max(0.0, min(
                (left - crop_x) / max(crop_width, 1), (crop_x + crop_width - right) / max(crop_width, 1),
                (top - crop_y) / max(crop_height, 1), (crop_y + crop_height - bottom) / max(crop_height, 1),
            )))
            if bound.framing_observation == "full_body":
                full_body_visible.append(fully_visible[-1])
    elif bounds:
        # fit/contain layouts preserve the complete source frame.
        fully_visible = [True] * len(bounds)
        headroom = [max(0.0, item.center_y - item.height / 2) for item in bounds]
        edge_margins = [max(0.0, min(
            item.center_x - item.width / 2, 1 - (item.center_x + item.width / 2),
            item.center_y - item.height / 2, 1 - (item.center_y + item.height / 2),
        )) for item in bounds]
        full_body_visible = [True for item in bounds if item.framing_observation == "full_body"]
    tracking = segment.tracking_diagnostics or _tracking_quality_metrics(segment)
    chest_observations = [item for item in bounds if item.framing_observation in {"chest_up", "upper_body", "full_body"}]
    head_only = [item for item in bounds if item.framing_observation == "head_only"]
    gesture_active = [item for item in bounds if item.gesture_active]
    important = [item for item in bounds if item.target in {"important_object", "screen_region"}]
    if scene_type in _SCREEN_SCENES:
        important = [item for item in bounds if item.target == "screen_region"] or important
    if scene_type in _PRODUCT_SCENES:
        important = [item for item in bounds if item.target == "important_object"] or important
    visible_ratio = sum(fully_visible) / len(fully_visible) if fully_visible else 0.0
    subject_area = diagnostics["subject_screen_ratio"]
    crop_scale = _crop_scale(crop) if crop is not None else 1.0
    return {
        "face_visibility": round(visible_ratio, 4),
        "chest_shoulder_framing": round(len(chest_observations) / len(bounds), 4) if bounds else 0.0,
        "head_only_ratio": round(len(head_only) / len(bounds), 4) if bounds else 0.0,
        "headroom_ratio": round(min(headroom, default=0.0), 4),
        "face_edge_margin": round(min(edge_margins, default=0.0), 4),
        "target_presence": round(visible_ratio, 4),
        "active_speaker_presence": round(max((item.active_speaker_confidence for item in bounds), default=0.0), 4),
        "gesture_active_ratio": round(len(gesture_active) / len(bounds), 4) if bounds else 0.0,
        "gesture_area_visibility": round(
            sum(item.gesture_area_visible for item in gesture_active) / len(gesture_active), 4,
        ) if gesture_active else 1.0,
        "product_screen_visibility": round(
            sum(fully_visible[bounds.index(item)] for item in important) / len(important), 4,
        ) if important else 0.0,
        "full_body_visibility": round(sum(full_body_visible) / len(full_body_visible), 4) if full_body_visible else 1.0,
        "empty_frame_risk": round(max(0.0, diagnostics["unused_visual_area_ratio"] + (0.0 if subject_area >= 0.025 else 0.25)), 4),
        "digital_zoom_scale": round(float(crop_scale or 1.0), 4),
        "crop_stability": bool(float(tracking.get("max_crop_speed", 0.0)) <= _MAX_SAFE_TRACKING_SPEED),
        "crop_velocity": round(float(tracking.get("max_crop_speed", 0.0)), 4),
        "crop_movement": round(float(tracking.get("crop_motion", 0.0)), 4),
        "crop_switch_frequency": round(1 / max(segment.minimum_focus_hold_seconds, 0.001) if segment.tracking_mode == "active_speaker_tracking" else 0.0, 4),
        "scene_framing_match": bool(
            (framing_intent == "CHEST_UP_PERSON" and segment.strategy == "subject_crop")
            or (framing_intent == "GROUP_CONVERSATION" and segment.tracking_target in {"subject_group", "active_speaker"})
            or (framing_intent == "PRODUCT_OR_HANDS" and segment.tracking_target == "important_object")
            or (framing_intent == "SCREEN_FIRST" and segment.tracking_target == "screen_region")
            or (framing_intent in {"PRESERVE_WIDE_ACTION", "CONSERVATIVE_WIDE"} and segment.tracking_mode in {"scene_wide", "safe_fallback"})
        ),
    }


def _crop_window_at(segment: CompositionSegment, time_seconds: float) -> tuple[int, int, int, int]:
    crop = segment.target_crop
    assert crop is not None and crop.crop_width and crop.crop_height
    if crop.tracking_keyframes:
        center_x, center_y = _tracking_center_at(
            crop.tracking_keyframes, max(0.0, time_seconds - (segment.source_start_seconds or 0.0)),
        )
        return (
            _crop_origin_for_center(center_x, crop.source_width, crop.crop_width),
            _crop_origin_for_center(center_y, crop.source_height, crop.crop_height),
            crop.crop_width, crop.crop_height,
        )
    return crop.crop_x or 0, crop.crop_y or 0, crop.crop_width, crop.crop_height


def _composition_diagnostics(segment: CompositionSegment, canvas: CanvasConfig) -> dict[str, float]:
    crop = segment.target_crop
    if crop is None:
        return {
            "foreground_coverage_ratio": 0.0,
            "blur_coverage_ratio": 0.0,
            "subject_screen_ratio": 0.0,
            "unused_visual_area_ratio": 1.0,
        }
    if crop.strategy == "fit_blur_background":
        source_aspect = crop.source_width / crop.source_height
        canvas_aspect = canvas.width / canvas.height
        foreground = min(1.0, canvas_aspect / source_aspect) if source_aspect >= canvas_aspect else min(1.0, source_aspect / canvas_aspect)
        blur = 1.0 - foreground
    else:
        foreground, blur = 1.0, 0.0
    if crop.crop_width and crop.crop_height:
        visible_area = (crop.crop_width / crop.source_width) * (crop.crop_height / crop.source_height)
        subject_ratio = max((item.width * item.height / max(visible_area, 0.001) for item in segment.subject_bounds), default=0.0)
    else:
        subject_ratio = max((item.width * item.height * foreground for item in segment.subject_bounds), default=0.0)
    return {
        "foreground_coverage_ratio": round(foreground, 4),
        "blur_coverage_ratio": round(blur, 4),
        "subject_screen_ratio": round(min(1.0, subject_ratio), 4),
        "unused_visual_area_ratio": round(blur if not segment.subject_bounds else max(0.0, blur - subject_ratio), 4),
    }


def _wide_crop_from_segment(segment: CompositionSegment, canvas: CanvasConfig) -> CropPlan:
    crop = segment.target_crop
    if crop is None:
        raise ProductionRenderError("Tracking fallback cannot resolve a source crop.")
    if crop.source_width / crop.source_height > (canvas.width / canvas.height) * 1.03:
        return CropPlan(
            strategy="fit_blur_background", source_width=crop.source_width, source_height=crop.source_height,
            display_rotation_degrees=crop.display_rotation_degrees,
        )
    return _crop_plan_for_center({
        "display_width": crop.source_width, "display_height": crop.source_height,
        "rotation": crop.display_rotation_degrees,
    }, canvas, 0.5, 0.5)


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


def _visual_filter(
    clip: SourceVideoClip | FreezeFrameClip,
    canvas: CanvasConfig,
    *,
    input_label: str = "[0:v]",
    output_label: str = "[vout]",
) -> str:
    crop = clip.crop_plan
    assert crop is not None
    label_prefix = output_label.strip("[]").replace("-", "_")
    bg_label = f"[{label_prefix}_bg]"
    fg_label = f"[{label_prefix}_fg]"
    blur_label = f"[{label_prefix}_blur]"
    fit_label = f"[{label_prefix}_fit]"
    tail = (
        f",fps={canvas.fps},tpad=stop_mode=clone:stop_duration={clip.freeze_duration_seconds:.6f},"
        f"trim=duration={clip.duration_seconds:.6f},setpts=PTS-STARTPTS,format={canvas.pixel_format}{output_label}"
    )
    if crop.strategy == "fit_blur_background":
        return (
            f"{input_label}split=2{bg_label}{fg_label};{bg_label}scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=increase,"
            f"crop={canvas.width}:{canvas.height},boxblur=20:10{blur_label};{fg_label}scale={canvas.width}:{canvas.height}:"
            f"force_original_aspect_ratio=decrease{fit_label};{blur_label}{fit_label}overlay=(W-w)/2:(H-h)/2,setsar=1" + tail
        )
    if crop.strategy == "fit_solid_background":
        return (
            f"color=c=0x161616:s={canvas.width}x{canvas.height}:r={canvas.fps}{bg_label};"
            f"{input_label}scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=decrease{fit_label};"
            f"{bg_label}{fit_label}overlay=(W-w)/2:(H-h)/2,setsar=1" + tail
        )
    assert crop.crop_width and crop.crop_height and crop.crop_x is not None and crop.crop_y is not None
    if crop.tracking_keyframes:
        x = _tracking_crop_expression(crop.tracking_keyframes, crop.source_width, crop.crop_width, "x")
        y = _tracking_crop_expression(crop.tracking_keyframes, crop.source_height, crop.crop_height, "y")
        return (
            f"{input_label}crop={crop.crop_width}:{crop.crop_height}:x='{x}':y='{y}',"
            f"scale={canvas.width}:{canvas.height},setsar=1" + tail
        )
    return (
        f"{input_label}crop={crop.crop_width}:{crop.crop_height}:{crop.crop_x}:{crop.crop_y},"
        f"scale={canvas.width}:{canvas.height},setsar=1" + tail
    )


def _tracking_crop_expression(
    keyframes: list[ReframeKeyframe], source_size: int, crop_size: int, axis: str,
) -> str:
    """Linearly interpolate a bounded crop origin using FFmpeg's frame time ``t``."""

    values: list[tuple[float, int]] = []
    for keyframe in keyframes:
        center = keyframe.normalized_x if axis == "x" else keyframe.normalized_y
        origin = max(0, min(_even_down(center * source_size - crop_size / 2), source_size - crop_size))
        values.append((keyframe.time_seconds, origin))
    if len(values) == 1:
        return str(values[0][1])
    expression = str(values[-1][1])
    for index in range(len(values) - 2, -1, -1):
        start_time, start_value = values[index]
        end_time, end_value = values[index + 1]
        delta = max(0.001, end_time - start_time)
        linear = f"{start_value}+({end_value}-{start_value})*(t-{start_time:.6f})/{delta:.6f}"
        expression = f"if(lt(t\\,{end_time:.6f})\\,{linear}\\,{expression})"
    first_time, first_value = values[0]
    if first_time > 0:
        expression = f"if(lt(t\\,{first_time:.6f})\\,{first_value}\\,{expression})"
    return expression


def probe_media(path: Path, require_video: bool = False, require_audio: bool = False) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ProductionRenderError("ffprobe не найден для production render.")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, timeout=90, check=True, **UTF8_REPLACE_TEXT,
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
    plan: ProductionPlan, source_checksum: str, mixed_checksum: str, audio_project: AudioProject, timeline: VideoTimeline,
    subtitles: SubtitleProject, canvas: CanvasConfig, config: ProductionRenderConfig,
    *, platform: str, product_flow_revision: str,
    compiled_plan: CompiledRenderPlan,
    render_profile: RenderProfile,
) -> str:
    return stable_text_hash(json.dumps({
        "source_checksum": source_checksum, "mixed_audio_checksum": mixed_checksum,
        "production_plan": {
            "plan_id": plan.plan_id,
            "candidate_id": plan.metadata.candidate_id,
            "source_id": plan.metadata.source_id,
            "source_range": [
                min((item.source_start_seconds for item in plan.dialogue_mappings), default=None),
                max((item.source_end_seconds for item in plan.dialogue_mappings), default=None),
            ],
            "final_script_hash": plan.metadata.final_script_hash,
            "audio_mode": plan.audio_mode,
            "plan_fingerprint": plan.plan_fingerprint(),
            "plan_envelope": plan.envelope.model_dump(mode="json", exclude={"created_at"}) if plan.envelope else None,
        },
        "audio_project_checksum": stable_text_hash(audio_project.model_dump_json()),
        "timeline": timeline.model_dump(mode="json"), "subtitle_project": subtitles.model_dump(mode="json"),
        "canvas": canvas.model_dump(mode="json"), "crop": config.crop_strategy,
        "encoder": config.encoder, "codec": config.video_codec, "bitrate": config.video_bitrate,
        "subtitles_enabled": config.subtitles_enabled, "render_config": asdict(config),
        "platform": platform, "product_flow_revision": product_flow_revision,
        "compiled_plan_hash": compiled_plan.plan_hash,
        "parity_signature": compiled_plan.parity_signature,
        "render_profile": render_profile.model_dump(mode="json"),
        "version": config.render_config_version,
        "engine_version": PRODUCTION_RENDER_ENGINE_VERSION,
    }, sort_keys=True, ensure_ascii=False))


def _resolve_render_profile(
    plan: CompiledRenderPlan,
    requested: RenderProfile | Literal["creative_preview", "final"],
    config: ProductionRenderConfig,
) -> RenderProfile:
    if isinstance(requested, RenderProfile):
        profile = requested
    elif requested == "final":
        profile = RenderProfile(
            profile_id="final",
            width=plan.canvas.width,
            height=plan.canvas.height,
            video_bitrate=config.video_bitrate,
            encoder=config.encoder,
            sampling_precision="full",
        )
    else:
        scale = min(1.0, 540 / plan.canvas.width, 960 / plan.canvas.height)
        width = max(2, _even_down(plan.canvas.width * scale))
        height = max(2, _even_down(plan.canvas.height * scale))
        profile = RenderProfile(
            profile_id="creative_preview",
            width=width,
            height=height,
            video_bitrate="1800k",
            encoder=config.encoder,
            sampling_precision="preview",
        )
    # This validates fps, aspect ratio and final-canvas identity up front.
    build_render_parity_manifest(plan, profile)
    return profile


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
    for extra in (root / "compiled-render-plan.json", root / "parity-manifest.json"):
        if extra.is_file():
            artifacts.append(str(extra))
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
        ("compiled_render_plan", root / "compiled-render-plan.json"),
        ("parity_manifest", root / "parity-manifest.json"),
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
        f"Render profile: {project.metadata.render_profile_id}",
        f"Compiled plan: {project.metadata.compiled_plan_hash or 'missing'}",
        f"Parity signature: {project.metadata.parity_signature or 'missing'}",
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


def _timeline_filter(
    durations: list[float],
    transition: str | list[VideoTransition],
    *,
    input_labels: list[str] | None = None,
) -> tuple[str, str]:
    if not durations:
        raise ProductionRenderError("No visual durations were supplied for final mux.")
    labels = input_labels or [f"[{index}:v]" for index in range(len(durations))]
    if len(labels) != len(durations):
        raise ProductionRenderError("Timeline input labels do not match visual durations.")
    if isinstance(transition, list):
        return _mixed_timeline_filter(durations, transition, input_labels=labels)
    if transition == "short_crossfade" and len(durations) > 1:
        fade = 0.15
        current = labels[0]
        elapsed = durations[0]
        for index, duration in enumerate(durations[1:], start=1):
            label = f"[xf{index}]"
            offset = max(0.0, elapsed - fade)
            graph = f"{current}{labels[index]}xfade=transition=fade:duration={fade:.3f}:offset={offset:.6f}{label}"
            current = label
            elapsed += duration - fade
            if index == 1:
                parts = [graph]
            else:
                parts.append(graph)
        loss = fade * (len(durations) - 1)
        parts.append(f"{current}tpad=stop_mode=clone:stop_duration={loss:.6f},trim=duration={sum(durations):.6f}[vconcat]")
        return ";".join(parts), "[vconcat]"
    joined_labels = "".join(labels)
    graph = f"{joined_labels}concat=n={len(durations)}:v=1:a=0[vconcat]"
    fade = min(0.15, durations[0])
    if transition == "fade_from_black":
        return graph + f";[vconcat]fade=t=in:st=0:d={fade:.3f}[vfaded]", "[vfaded]"
    if transition == "fade_to_black":
        start = max(0.0, sum(durations) - min(0.15, durations[-1]))
        return graph + f";[vconcat]fade=t=out:st={start:.6f}:d={fade:.3f}[vfaded]", "[vfaded]"
    return graph, "[vconcat]"


def _mixed_timeline_filter(
    durations: list[float],
    transitions: list[VideoTransition],
    *,
    input_labels: list[str] | None = None,
) -> tuple[str, str]:
    """Honor controlled scene cuts while retaining requested crossfades elsewhere."""

    if len(durations) == 1:
        return _timeline_filter(durations, "cut", input_labels=input_labels)
    labels = input_labels or [f"[{index}:v]" for index in range(len(durations))]
    if len(labels) != len(durations):
        raise ProductionRenderError("Timeline input labels do not match visual durations.")
    by_pair = [item.transition_type for item in transitions]
    # concat outputs AVTB while decoded MP4 clips commonly use a stream-specific
    # timebase. Normalize every branch before a later xfade can join it.
    parts = [f"{label}settb=AVTB,setpts=PTS-STARTPTS[v{index}]" for index, label in enumerate(labels)]
    current = "[v0]"
    elapsed = durations[0]
    crossfade_loss = 0.0
    for index, duration in enumerate(durations[1:], start=1):
        kind = by_pair[index - 1] if index - 1 < len(by_pair) else "cut"
        label = f"[mix{index}]"
        incoming = f"[v{index}]"
        if kind == "short_crossfade":
            fade = min(0.15, duration, elapsed)
            offset = max(0.0, elapsed - fade)
            parts.append(
                f"{current}{incoming}xfade=transition=fade:duration={fade:.3f}:offset={offset:.6f}{label}"
            )
            elapsed += duration - fade
            crossfade_loss += fade
        else:
            parts.append(f"{current}{incoming}concat=n=2:v=1:a=0{label}")
            elapsed += duration
        current = label
    if crossfade_loss:
        parts.append(
            f"{current}tpad=stop_mode=clone:stop_duration={crossfade_loss:.6f},"
            f"trim=duration={sum(durations):.6f}[vconcat]"
        )
        return ";".join(parts), "[vconcat]"
    return ";".join(parts), current


def _run_ffmpeg(command: list[str], context: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=7200, **UTF8_REPLACE_TEXT)
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
    quality = validate_output_quality(project, project.render_request.subtitles_enabled)
    subtitles = project.subtitle_project
    composition_segments = project.reframe_plan.composition_segments
    strategy_counts: dict[str, int] = {}
    tracking_mode_counts: dict[str, int] = {}
    for segment in composition_segments:
        strategy_counts[segment.strategy] = strategy_counts.get(segment.strategy, 0) + 1
        tracking_mode_counts[segment.tracking_mode] = tracking_mode_counts.get(segment.tracking_mode, 0) + 1
    diagnostics = [segment.composition_diagnostics for segment in composition_segments]
    mean = lambda name: round(sum(float(item.get(name, 0)) for item in diagnostics) / len(diagnostics), 4) if diagnostics else 0.0
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
        "subtitles_enabled": project.render_request.subtitles_enabled,
        "subtitle_cue_count": len(subtitles.cues) if subtitles else 0,
        "subtitle_layout": {
            "contract_version": subtitles.layout_contract_version if subtitles else None,
            "quality_decision": subtitles.quality_decision.model_dump(mode="json") if subtitles else None,
            "resolved_cue_count": len(subtitles.cues) if subtitles else 0,
            "cues": [
                {
                    "cue_id": cue.cue_id,
                    "segment_id": cue.segment_id,
                    "original_text": cue.original_text or cue.text,
                    "original_line_count": cue.original_line_count,
                    "resolved_lines": list(cue.resolved_lines),
                    "resolved_font_size": cue.resolved_font_size,
                    "split_reason": cue.split_reason,
                    "fallback_used": cue.fallback_used,
                    "layout_state": cue.layout_state,
                }
                for cue in subtitles.cues
            ] if subtitles else [],
            "final_validation": quality,
        },
        "cache_hit": bool(result.cache_hit) if result else False,
        "render_profile": project.metadata.render_profile_id,
        "compiled_plan_hash": project.metadata.compiled_plan_hash,
        "parity_signature": project.metadata.parity_signature,
        "cache_nodes": dict(project.metadata.cache_node_hits),
        "single_pass_encode": project.metadata.single_pass_encode,
        "validation": validation.status,
        "warnings": project.warnings,
        "fallback_reasons": project.fallback_reasons,
        "composition": {
            "strategy": project.reframe_plan.strategy,
            "subject_detection_used": project.reframe_plan.subject_detection_used,
            "summary": {
                "strategy_counts": strategy_counts,
                "tracking_mode_counts": tracking_mode_counts,
                "mean_subject_detection_confidence": round(
                    sum(segment.confidence for segment in composition_segments) / len(composition_segments), 4,
                ) if composition_segments else 0.0,
                "foreground_coverage_ratio": mean("foreground_coverage_ratio"),
                "blur_coverage_ratio": mean("blur_coverage_ratio"),
                "subject_screen_ratio": mean("subject_screen_ratio"),
                "unused_visual_area_ratio": mean("unused_visual_area_ratio"),
                "scene_transition_count": sum(
                    transition.transition_type == "cut"
                    and "-scene-" in (transition.from_clip_id or "")
                    and "-scene-" in (transition.to_clip_id or "")
                    for transition in project.timeline.transitions
                ),
                "fallback_reasons": [
                    {"segment_id": segment.segment_id, "reason": segment.fallback_reason}
                    for segment in composition_segments if segment.fallback_reason
                ],
            },
            "segments": [
                segment.model_dump(mode="json")
                for segment in project.reframe_plan.composition_segments
            ],
            "tracking_validation": quality.get("tracking", {}),
        },
        "quality": quality,
        "errors": [item.model_dump(mode="json") for item in result.errors] if result else [],
        "artifacts": project.artifact_paths,
        "ai_called": False,
        "tts_regenerated": False,
        "audio_remixed": False,
    }
