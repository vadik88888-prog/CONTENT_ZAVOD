from __future__ import annotations

"""Native Phase 7 creative-plan assembly and typed renderer adaptation."""

from typing import Any, Iterable, Literal, Sequence

from app.caption_planning import build_caption_plan
from app.caption_presets import caption_preset_definition, with_caption_preset_override
from app.composition_planning import TargetObservation, build_composition_plan
from app.config import AppConfig
from app.creative_contracts import (
    AssetManifestEntry,
    BackendAssignment,
    CanvasPlan,
    CaptionPlan,
    CompiledRenderPlan,
    CompositionCropKeyframe,
    CreativeIntent,
    CreativePolicy,
    ImmutableProductionPlanLink,
    LayoutFamily,
    MotionDomain,
    MotionPlan,
    NormalizedRect,
    OutputInterval,
    SourceOutputTimeMap,
    canonical_hash,
    compile_render_plan,
    seconds_to_output_frame,
)
from app.creative_policy import (
    CREATIVE_POLICY_VERSION,
    creative_preset_definition,
    preset_family_policy,
)
from app.motion_planning import build_motion_plan
from app.production_models import ProductionPlan
from app.source_broll_planning import SourceSceneEvidence, build_source_broll_plan
from app.video_models import (
    CropPlan,
    ReframeKeyframe,
    SourceVideoClip,
    VideoClipModel,
    VideoTimeline,
    VideoTransition,
)


NATIVE_CREATIVE_EXECUTION_VERSION = "7G.1"


def default_native_creative_intent(
    plan: ProductionPlan,
    mapping: SourceOutputTimeMap,
    config: AppConfig,
) -> CreativeIntent:
    """Create the conservative native intent used when no revised intent was supplied.

    This does not infer editorial events. It carries the approved production
    identity and existing product policy into the native planners; evidence-free
    optional decisions safely resolve to A-roll/static composition.
    """

    envelope = plan.envelope
    assert envelope is not None
    try:
        family_policy = creative_preset_definition(
            envelope.preset.preset_id,  # type: ignore[arg-type]
            envelope.preset.preset_version,
        )
    except KeyError:
        # Compatibility-only envelopes (for example legacy/3A) predate the
        # creative registry. Preserve their pinned envelope identity while
        # retaining the established configured safe-family adaptation.
        family_policy = preset_family_policy(
            config.product_flow.subtitle_preset,  # type: ignore[arg-type]
        )
    policy = CreativePolicy(
        preset_id=envelope.preset.preset_id,
        preset_version=envelope.preset.preset_version,
        platform=envelope.preset.platform,
        caption_style_family=caption_preset_definition(
            config.product_flow.caption_preset_id  # type: ignore[arg-type]
        ).style_family,
        caption_density={
            "minimal": "low", "clean": "balanced",
            "editorial": "balanced", "emphasis": "high",
        }[caption_preset_definition(
            config.product_flow.caption_preset_id  # type: ignore[arg-type]
        ).style_family],
        intensity=family_policy.intensity_ceiling,
        reduced_motion=config.product_flow.reduced_motion,
        source_broll_enabled=config.production_render.same_source_broll_allowed,
        user_override_ids=with_caption_preset_override(
            (), config.product_flow.caption_preset_id,  # type: ignore[arg-type]
        ),
    )
    identity = {
        "production_plan": plan.reference().model_dump(mode="json"),
        "mapping": mapping.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "creative_policy_version": CREATIVE_POLICY_VERSION,
        "preset_selection": {
            "mode": config.product_flow.preset_selection_mode,
            "provenance": config.product_flow.preset_provenance,
            "configured": config.product_flow.configured_subtitle_preset,
            "recommended": config.product_flow.recommended_subtitle_preset,
            "effective": config.product_flow.subtitle_preset,
            "caption_preset": config.product_flow.caption_preset_id,
            "reduced_motion": config.product_flow.reduced_motion,
        },
        "version": NATIVE_CREATIVE_EXECUTION_VERSION,
    }
    digest = canonical_hash(identity)
    return CreativeIntent(
        intent_id=f"intent-native-{digest[:18]}",
        revision=1,
        production_plan=ImmutableProductionPlanLink.from_reference(plan.reference()),
        source_output_mapping=mapping,
        evidence_fingerprint=envelope.input_fingerprints.analysis_sha256,
        proposal_hash=digest,
        policy=policy,
        confidence=0,
        provenance=(
            "native_production_default",
            f"creative_policy:{CREATIVE_POLICY_VERSION}",
            f"preset_selection:{config.product_flow.preset_selection_mode}",
            f"preset_provenance:{config.product_flow.preset_provenance}",
            f"preset_effective:{config.product_flow.subtitle_preset}",
            f"preset_recommendation:{config.product_flow.recommended_subtitle_preset}",
            f"caption_preset:{config.product_flow.caption_preset_id}",
        ),
    )


def compile_native_creative_plan(
    intent: CreativeIntent,
    transcript: dict[str, Any],
    config: AppConfig,
    *,
    source_width: int,
    source_height: int,
    target_observations: Iterable[TargetObservation] = (),
    source_scenes: Iterable[SourceSceneEvidence] = (),
) -> CompiledRenderPlan:
    """Run the production 7C -> 7D -> 7E -> 7F -> 7G compiler chain."""

    captions = build_caption_plan(intent, transcript, config.production_render)
    composition = build_composition_plan(
        intent,
        target_observations,
        source_width=source_width,
        source_height=source_height,
    )
    broll = build_source_broll_plan(intent, source_scenes, composition)
    motion = build_motion_plan(intent, captions, composition, broll)
    assets: tuple[AssetManifestEntry, ...] = ()
    if captions.font_manifest is not None and captions.font_manifest.file_sha256 is not None:
        assets = (
            AssetManifestEntry(
                asset_id=captions.font_manifest.font_id,
                asset_type="font",
                checksum=captions.font_manifest.file_sha256,
            ),
            *(AssetManifestEntry(
                asset_id=face.font_id,
                asset_type="font",
                checksum=face.file_sha256,
            ) for face in captions.font_manifest.companion_faces),
        )
    return compile_render_plan(
        intent,
        captions,
        composition,
        motion,
        broll,
        CanvasPlan(
            width=config.production_render.output_width,
            height=config.production_render.output_height,
            fps=30,
        ),
        assets=assets,
        backends=(
            BackendAssignment(
                domain="base_video", backend_id="ffmpeg",
                backend_version=NATIVE_CREATIVE_EXECUTION_VERSION,
            ),
            BackendAssignment(
                domain="caption", backend_id="libass",
                backend_version="7C.libass-tier1.1",
            ),
            BackendAssignment(
                domain="composition", backend_id="ffmpeg",
                backend_version="7D.ffmpeg-composition.1",
            ),
            BackendAssignment(
                domain="broll", backend_id="ffmpeg",
                backend_version="7E.ffmpeg-source-broll.1",
            ),
            BackendAssignment(
                domain="motion", backend_id="ffmpeg",
                backend_version="7F.ffmpeg-motion.1",
            ),
        ),
    )


def validate_native_handoff(
    plan: ProductionPlan,
    mapping: SourceOutputTimeMap,
    compiled_plan: CompiledRenderPlan,
) -> None:
    reference = plan.reference()
    if compiled_plan.production_plan.model_dump(mode="json") != reference.model_dump(mode="json"):
        raise ValueError("NATIVE_COMPILED_PLAN_PRODUCTION_IDENTITY_MISMATCH")
    if compiled_plan.source_output_mapping != mapping:
        raise ValueError("NATIVE_COMPILED_PLAN_EDIT_MAPPING_MISMATCH")


def caption_plan_with_motion(captions: CaptionPlan, motion: MotionPlan) -> CaptionPlan:
    """Resolve assessed 7F caption events into the Tier 1 ASS primitives."""

    events = {
        target_id: event
        for event in motion.events
        if event.domain == MotionDomain.CAPTION
        for target_id in event.target_plan_ids
    }
    if not events:
        return captions
    cues = []
    for cue in captions.cues:
        event = events.get(cue.cue_id)
        if event is None:
            cues.append(cue)
            continue
        primitive = event.primitive_id if event.primitive_id in {"static", "fade", "scale", "slide"} else "static"
        cues.append(cue.model_copy(update={
            "primitive_id": primitive,
            "easing_id": event.easing_id if primitive != "static" else "none",
            "motion_duration_frames": event.duration_frames if primitive != "static" else 0,
            "scale_percent": max(94, min(108, round(event.scale_from * 100))) if primitive == "scale" else 100,
            "slide_distance_ratio": min(0.05, abs(event.translate_y_ratio)) if primitive == "slide" else 0,
        }))
    return captions.model_copy(update={"cues": tuple(cues)})


def apply_native_visual_plan(
    timeline: VideoTimeline,
    compiled_plan: CompiledRenderPlan,
    *,
    source_width: int,
    source_height: int,
    rotation: Literal[0, 90, 180, 270] = 0,
) -> VideoTimeline:
    """Execute normalized 7D geometry and 7E source cutaways on the visual track."""

    fps = compiled_plan.source_output_mapping.output_fps
    boundaries = {
        frame
        for segment in (
            *compiled_plan.composition_plan.segments,
            *compiled_plan.source_broll_plan.segments,
        )
        for frame in (
            segment.output.start_frame if hasattr(segment, "output") else segment.destination.start_frame,
            segment.output.end_frame if hasattr(segment, "output") else segment.destination.end_frame,
        )
    }
    mapped_outputs = {
        segment.map_id: segment.output
        for segment in compiled_plan.source_output_mapping.segments
    }
    expanded: list[tuple[VideoClipModel, OutputInterval]] = []
    destination_cursor = 0
    for clip in timeline.clips:
        output_start = max(
            destination_cursor,
            seconds_to_output_frame(clip.timeline_start_seconds),
        )
        output_end = max(
            output_start + 1,
            seconds_to_output_frame(clip.timeline_end_seconds, end=True),
        )
        destination_cursor = output_end
        output = OutputInterval(start_frame=output_start, end_frame=output_end)
        if (
            clip.source_start_seconds is not None
            and clip.source_end_seconds is not None
            and clip.source_end_seconds > clip.source_start_seconds
        ):
            authoritative = mapped_outputs.get(f"legacy-{clip.clip_id}")
            if authoritative is None or authoritative != output:
                raise ValueError("NATIVE_TIMELINE_MAPPING_PARTITION_MISMATCH")

        if not isinstance(clip, SourceVideoClip):
            expanded.append((_slice_clip_frames(clip, output, output, None, fps), output))
            continue
        cuts = sorted(
            frame for frame in boundaries
            if output.start_frame < frame < output.end_frame
        )
        if not cuts:
            expanded.append((_slice_clip_frames(clip, output, output, None, fps), output))
            continue
        points = [output.start_frame, *cuts, output.end_frame]
        for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
            interval = OutputInterval(start_frame=start, end_frame=end)
            expanded.append((_slice_clip_frames(clip, output, interval, index, fps), interval))

    rendered: list[tuple[VideoClipModel, OutputInterval]] = []
    for clip, output in expanded:
        if not isinstance(clip, SourceVideoClip):
            rendered.append((clip, output))
            continue
        start_frame = output.start_frame
        end_frame = output.end_frame
        composition = next((
            item for item in compiled_plan.composition_plan.segments
            if item.output.start_frame <= start_frame and end_frame <= item.output.end_frame
        ), None)
        broll = next((
            item for item in compiled_plan.source_broll_plan.segments
            if item.destination.start_frame <= start_frame and end_frame <= item.destination.end_frame
        ), None)
        crop = clip.crop_plan
        if composition is not None:
            crop = _native_crop_plan(
                composition.crop,
                composition.layout,
                source_width,
                source_height,
                rotation,
                composition.crop_keyframes,
                clip.timeline_start_seconds,
                clip.timeline_end_seconds,
            )
        updates: dict[str, object] = {"crop_plan": crop}
        if broll is not None:
            destination_frames = broll.destination.end_frame - broll.destination.start_frame
            source_seconds = (broll.source_cutaway.end_tick - broll.source_cutaway.start_tick) / 1_000_000
            offset = (start_frame - broll.destination.start_frame) / destination_frames
            span = (end_frame - start_frame) / destination_frames
            source_start = broll.source_cutaway.start_tick / 1_000_000 + source_seconds * offset
            source_end = source_start + source_seconds * span
            broll_crop = (
                _broll_crop_rect(broll.source_crop, source_width, source_height)
                if broll.source_crop is not None else None
            )
            crop = (
                _native_crop_plan(
                    broll_crop,
                    LayoutFamily.SINGLE_SUBJECT,
                    source_width,
                    source_height,
                    rotation,
                    (),
                    clip.timeline_start_seconds,
                    clip.timeline_end_seconds,
                )
                if broll_crop is not None else CropPlan(
                    strategy="fit_blur_background",
                    source_width=source_width,
                    source_height=source_height,
                    display_rotation_degrees=rotation,
                )
            )
            updates.update({
                "source_start_seconds": round(source_start, 6),
                "source_end_seconds": round(source_end, 6),
                "freeze_duration_seconds": 0.0,
                "visual_strategy": "candidate_excerpt",
                "status": "ready",
                "fallback_reason": None,
                "crop_plan": crop,
            })
        rendered.append((clip.model_copy(update=updates), output))

    normalized_with_outputs = [
        (clip.model_copy(update={"order": order}), output)
        for order, (clip, output) in enumerate(rendered, start=1)
    ]
    normalized = [clip for clip, _ in normalized_with_outputs]
    transitions = []
    for (left, _), (right, right_output) in zip(
        normalized_with_outputs, normalized_with_outputs[1:],
    ):
        boundary_frame = right_output.start_frame
        broll = next((
            item for item in compiled_plan.source_broll_plan.segments
            if boundary_frame in {item.destination.start_frame, item.destination.end_frame}
        ), None)
        transition_type = "short_crossfade" if broll is not None and broll.transition == "short_dissolve" else "cut"
        transitions.append(VideoTransition(
            transition_type=transition_type,
            from_clip_id=left.clip_id,
            to_clip_id=right.clip_id,
            duration_seconds=0.15 if transition_type == "short_crossfade" else 0,
        ))
    return timeline.model_copy(update={
        "clips": normalized,
        "transitions": transitions,
        "duration_seconds": round(destination_cursor / fps, 9),
    })


def _slice_clip_frames(
    clip: VideoClipModel,
    source_output: OutputInterval,
    output: OutputInterval,
    index: int | None,
    fps: int,
) -> VideoClipModel:
    duration_frames = output.end_frame - output.start_frame
    updates: dict[str, object] = {
        "timeline_start_seconds": round(output.start_frame / fps, 9),
        "timeline_end_seconds": round(output.end_frame / fps, 9),
        "duration_seconds": round(duration_frames / fps, 9),
    }
    if index is not None:
        updates["clip_id"] = f"{clip.clip_id}-native-{index:02d}"
    if isinstance(clip, SourceVideoClip):
        source_span = clip.source_end_seconds - clip.source_start_seconds
        output_span = source_output.end_frame - source_output.start_frame
        left_ratio = (output.start_frame - source_output.start_frame) / output_span
        right_ratio = (output.end_frame - source_output.start_frame) / output_span
        source_start = clip.source_start_seconds + source_span * left_ratio
        source_end = clip.source_start_seconds + source_span * right_ratio
        updates.update({
            "source_start_seconds": round(source_start, 6),
            "source_end_seconds": round(source_end, 6),
            "freeze_duration_seconds": 0.0,
        })
    return clip.model_copy(update=updates)


def _native_crop_plan(
    rect: NormalizedRect,
    layout: LayoutFamily,
    source_width: int,
    source_height: int,
    rotation: Literal[0, 90, 180, 270],
    keyframes: Sequence[CompositionCropKeyframe],
    clip_start: float,
    clip_end: float,
) -> CropPlan:
    if layout == LayoutFamily.FIT_BACKGROUND:
        return CropPlan(
            strategy="fit_blur_background",
            source_width=source_width,
            source_height=source_height,
            display_rotation_degrees=rotation,
        )
    width = _even_dimension(rect.width * source_width, source_width)
    height = _even_dimension(rect.height * source_height, source_height)
    x = _even_origin(rect.x * source_width, source_width - width)
    y = _even_origin(rect.y * source_height, source_height - height)
    tracking = []
    seen_times: set[float] = set()
    for keyframe in keyframes:
        time = round(keyframe.frame / 30 - clip_start, 6)
        if time < 0 or time >= clip_end - clip_start or time in seen_times:
            continue
        seen_times.add(time)
        tracking.append(ReframeKeyframe(
            time_seconds=time,
            normalized_x=keyframe.crop.x + keyframe.crop.width / 2,
            normalized_y=keyframe.crop.y + keyframe.crop.height / 2,
            confidence=1,
        ))
    return CropPlan(
        strategy="manual_normalized_crop",
        source_width=source_width,
        source_height=source_height,
        display_rotation_degrees=rotation,
        normalized_x=rect.x + rect.width / 2,
        normalized_y=rect.y + rect.height / 2,
        crop_width=width,
        crop_height=height,
        crop_x=x,
        crop_y=y,
        tracking_keyframes=tracking,
    )


def _broll_crop_rect(
    target: NormalizedRect,
    source_width: int,
    source_height: int,
) -> NormalizedRect:
    """Derive cutaway geometry only from its own scene evidence."""

    output_aspect = 9 / 16
    source_aspect = source_width / source_height
    normalized_width = output_aspect / max(source_aspect, 1e-9)
    margin = 0.1
    height = min(1.0, max(0.58, target.height / (1 - 2 * margin)))
    width = min(1.0, max(normalized_width * height, target.width / (1 - 2 * margin)))
    height = min(1.0, max(height, width / max(normalized_width, 1e-9)))
    width = min(1.0, normalized_width * height)
    center_x = target.x + target.width / 2
    center_y = target.y + target.height / 2
    x = min(1 - width, max(0.0, center_x - width / 2))
    y = min(1 - height, max(0.0, center_y - height / 2))
    return NormalizedRect(x=x, y=y, width=width, height=height)


def _even_dimension(value: float, maximum: int) -> int:
    result = max(2, min(maximum, int(round(value)) // 2 * 2))
    return result if result <= maximum else maximum - maximum % 2


def _even_origin(value: float, maximum: int) -> int:
    return max(0, min(maximum, int(round(value)) // 2 * 2))
