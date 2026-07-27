from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


VIDEO_PROJECT_SCHEMA_VERSION = "3D.0"
VIDEO_TIMELINE_VERSION = "3D.0"
SUBTITLE_PROJECT_SCHEMA_VERSION = "3D.0"


class CanvasConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=2, le=7680)
    height: int = Field(ge=2, le=7680)
    fps: float = Field(gt=1, le=120)
    pixel_format: Literal["yuv420p"] = "yuv420p"
    aspect_ratio: Literal["9:16"] = "9:16"

    @model_validator(mode="after")
    def _valid_canvas(self) -> "CanvasConfig":
        if self.width % 2 or self.height % 2:
            raise ValueError("canvas dimensions must be even for yuv420p")
        if abs((self.width / self.height) - (9 / 16)) > 0.002:
            raise ValueError("production canvas must use a 9:16 aspect ratio")
        return self


class ReframeKeyframe(BaseModel):
    """A normalized subject position used by a static or moving crop."""

    model_config = ConfigDict(extra="forbid")

    time_seconds: float = Field(ge=0)
    normalized_x: float = Field(ge=0, le=1)
    normalized_y: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class CropPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["center_crop", "fit_blur_background", "fit_solid_background", "top_crop", "manual_normalized_crop"]
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    display_rotation_degrees: Literal[0, 90, 180, 270] = 0
    normalized_x: float = Field(default=0.5, ge=0, le=1)
    normalized_y: float = Field(default=0.5, ge=0, le=1)
    crop_width: int | None = Field(default=None, gt=0)
    crop_height: int | None = Field(default=None, gt=0)
    crop_x: int | None = Field(default=None, ge=0)
    crop_y: int | None = Field(default=None, ge=0)
    tracking_keyframes: list[ReframeKeyframe] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _valid_crop(self) -> "CropPlan":
        values = (self.crop_width, self.crop_height, self.crop_x, self.crop_y)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError("manual crop fields must be provided together")
        if self.crop_width is not None:
            assert self.crop_height is not None and self.crop_x is not None and self.crop_y is not None
            if self.crop_width > self.source_width or self.crop_height > self.source_height:
                raise ValueError("crop cannot exceed source dimensions")
            if self.crop_x + self.crop_width > self.source_width or self.crop_y + self.crop_height > self.source_height:
                raise ValueError("crop offsets must stay within source dimensions")
        if self.tracking_keyframes and self.crop_width is None:
            raise ValueError("tracking keyframes require a concrete crop window")
        if any(
            right.time_seconds <= left.time_seconds
            for left, right in zip(self.tracking_keyframes, self.tracking_keyframes[1:])
        ):
            raise ValueError("tracking keyframes must be strictly ordered")
        return self


class SubjectBounds(BaseModel):
    """A conservative normalized subject observation; it never identifies a person."""

    model_config = ConfigDict(extra="forbid")

    time_seconds: float = Field(ge=0)
    center_x: float = Field(ge=0, le=1)
    center_y: float = Field(ge=0, le=1)
    width: float = Field(default=0.16, gt=0, le=1)
    height: float = Field(default=0.20, gt=0, le=1)
    confidence: float = Field(ge=0, le=1)
    target: Literal[
        "primary_face", "primary_person", "active_speaker", "important_object",
        "screen_region", "subject_group", "scene_center", "none",
    ] = "primary_person"
    visible_face_count: int = Field(default=1, ge=0, le=32)
    active_speaker_confidence: float = Field(default=0, ge=0, le=1)
    scene_id: str | None = Field(default=None, max_length=160)


class CompositionSegment(BaseModel):
    """Scene-level vertical-composition and tracking decision for one visual interval."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
    visual_clip_id: str | None = None
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, ge=0)
    speaker_id: str | None = Field(default=None, max_length=160)
    scene_change_count: int = Field(default=0, ge=0)
    strategy: Literal[
        "subject_crop", "face_crop", "center_crop", "fit_with_blur", "split_layout",
        "safe_fallback", "group_framing", "scene_wide", "original_vertical",
    ]
    subject_bounds: list[SubjectBounds] = Field(default_factory=list)
    target_crop: CropPlan | None = None
    target_scale: float | None = Field(default=None, gt=0, le=16)
    target_center_x: float = Field(default=0.5, ge=0, le=1)
    target_center_y: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    fallback_reason: str | None = None
    transition_type: Literal["cut", "short_crossfade", "hold"] = "cut"

    tracking_mode: Literal[
        "none", "static_subject_crop", "face_tracking", "person_tracking",
        "active_speaker_tracking", "object_tracking", "group_framing", "scene_wide", "safe_fallback",
    ]
    tracking_target: Literal[
        "primary_face", "primary_person", "active_speaker", "important_object",
        "screen_region", "subject_group", "scene_center", "none",
    ]
    tracking_required: bool
    tracking_confidence: float = Field(ge=0, le=1)
    tracking_reason: str = Field(min_length=1, max_length=1200)
    static_crop_sufficient: bool
    tracking_risk: Literal["none", "low", "medium", "high", "unsafe"]
    fallback_strategy: Literal[
        "none", "static_subject_crop", "group_framing", "scene_wide", "fit_with_blur", "safe_fallback",
    ] = "none"
    wide_safe_layout_required: bool = False
    minimum_focus_hold_seconds: float = Field(default=1.25, ge=0, le=30)
    tracking_validation_status: Literal["not_applicable", "passed", "passed_with_warning", "failed_repaired"] = "not_applicable"
    tracking_validation_reasons: list[str] = Field(default_factory=list)
    tracking_diagnostics: dict[str, float | int | bool] = Field(default_factory=dict)
    composition_quality_status: Literal["passed", "passed_with_warning", "failed_repairable", "failed"] = "passed"
    composition_quality_reasons: list[str] = Field(default_factory=list)
    composition_diagnostics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_composition_segment(self) -> "CompositionSegment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("composition segment must have positive duration")
        if (self.source_start_seconds is None) != (self.source_end_seconds is None):
            raise ValueError("composition segment source timestamps must be provided together")
        if self.source_start_seconds is not None and self.source_end_seconds is not None:
            if self.source_end_seconds <= self.source_start_seconds:
                raise ValueError("composition segment source range must be positive")
        dynamic = {"face_tracking", "person_tracking", "active_speaker_tracking", "object_tracking"}
        if self.tracking_mode in dynamic and not self.tracking_required:
            raise ValueError("dynamic tracking mode must be marked as required")
        if self.static_crop_sufficient and self.tracking_required:
            raise ValueError("tracking cannot be required when a static crop is sufficient")
        if self.tracking_mode in dynamic and self.tracking_target == "none":
            raise ValueError("dynamic tracking needs a concrete target")
        return self


class ReframePlan(BaseModel):
    """Persisted visual-layout decision, independent from FFmpeg filter syntax."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["subject_crop", "center_crop", "contain", "blur_fallback", "original_vertical"]
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    canvas_width: int = Field(gt=0)
    canvas_height: int = Field(gt=0)
    subtitle_reserved_bottom_ratio: float = Field(ge=0, le=0.5)
    keyframes: list[ReframeKeyframe] = Field(default_factory=list)
    composition_segments: list[CompositionSegment] = Field(default_factory=list)
    subject_detection_used: bool = False
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def _valid_plan(self) -> "ReframePlan":
        if self.strategy == "subject_crop" and not self.keyframes and not self.composition_segments:
            raise ValueError("subject crop needs at least one keyframe")
        if self.subject_detection_used and self.strategy != "subject_crop" and not self.composition_segments:
            raise ValueError("subject detection must resolve to a documented composition segment")
        if self.fallback_reason and self.strategy == "subject_crop":
            raise ValueError("subject crop cannot carry a fallback reason")
        if any(
            right.start_seconds < left.end_seconds - 0.02
            for left, right in zip(self.composition_segments, self.composition_segments[1:])
        ):
            raise ValueError("composition segments must be ordered without overlap")
        return self


class VideoClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
    clip_type: str
    order: int = Field(ge=1)
    timeline_start_seconds: float = Field(ge=0)
    timeline_end_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    source_path: str | None = None
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, ge=0)
    production_segment_id: str | None = None
    fact_id: str | None = None
    speaker: str | None = None
    visual_strategy: Literal["mapped_source", "previous_visual", "next_visual", "freeze_frame", "candidate_excerpt", "fill"]
    crop_plan: CropPlan | None = None
    freeze_duration_seconds: float = Field(default=0, ge=0)
    status: Literal["ready", "fallback", "placeholder"]
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def _valid_duration(self) -> "VideoClip":
        if abs((self.timeline_end_seconds - self.timeline_start_seconds) - self.duration_seconds) > 0.02:
            raise ValueError("video clip timeline range must match duration")
        if (self.source_start_seconds is None) != (self.source_end_seconds is None):
            raise ValueError("source timestamps must be provided together")
        if self.source_start_seconds is not None and self.source_end_seconds is not None and self.source_end_seconds < self.source_start_seconds:
            raise ValueError("source end must not precede source start")
        if self.status == "fallback" and not self.fallback_reason:
            raise ValueError("fallback video clips must record a fallback reason")
        return self


class SourceVideoClip(VideoClip):
    clip_type: Literal["source_video"] = "source_video"
    visual_strategy: Literal["mapped_source", "previous_visual", "next_visual", "candidate_excerpt"]
    source_path: str = Field(min_length=1)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    crop_plan: CropPlan


class FreezeFrameClip(VideoClip):
    clip_type: Literal["freeze_frame"] = "freeze_frame"
    visual_strategy: Literal["freeze_frame"] = "freeze_frame"
    source_path: str = Field(min_length=1)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    crop_plan: CropPlan
    freeze_duration_seconds: float = Field(gt=0)


class FillClip(VideoClip):
    clip_type: Literal["fill"] = "fill"
    visual_strategy: Literal["fill"] = "fill"
    status: Literal["fallback", "placeholder"] = "fallback"
    fallback_reason: str = Field(min_length=1)


VideoClipModel = Annotated[SourceVideoClip | FreezeFrameClip | FillClip, Field(discriminator="clip_type")]


class VideoTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transition_type: Literal["cut", "short_crossfade", "fade_from_black", "fade_to_black"] = "cut"
    from_clip_id: str | None = None
    to_clip_id: str | None = None
    duration_seconds: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def _cut_has_no_duration(self) -> "VideoTransition":
        if self.transition_type == "cut" and self.duration_seconds != 0:
            raise ValueError("cut transition must not change duration")
        return self


class VideoTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: str
    track_type: Literal["visual"] = "visual"
    clips: list[VideoClipModel] = Field(default_factory=list)
    status: Literal["ready", "fallback", "empty"]


class VideoTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_version: str = VIDEO_TIMELINE_VERSION
    clips: list[VideoClipModel]
    transitions: list[VideoTransition] = Field(default_factory=list)
    duration_seconds: float = Field(ge=0)
    source_of_truth: Literal["audio_project_mixed_audio"] = "audio_project_mixed_audio"

    @model_validator(mode="after")
    def _sequential(self) -> "VideoTimeline":
        if [clip.order for clip in self.clips] != sorted(clip.order for clip in self.clips):
            raise ValueError("video timeline clips must be ordered")
        previous = 0.0
        for clip in self.clips:
            if abs(clip.timeline_start_seconds - previous) > 0.02:
                raise ValueError("video timeline cannot contain implicit gaps")
            previous = clip.timeline_end_seconds
        if abs(previous - self.duration_seconds) > 0.02:
            raise ValueError("video timeline duration must match final clip")
        return self


class SubtitleCue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cue_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
    segment_id: str
    speaker: str
    text: str = Field(min_length=1, max_length=4000)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    word_count: int = Field(ge=1)
    line_count: int = Field(ge=1, le=8)
    style_id: str
    source_type: Literal["narration", "dialogue"]
    word_timings: list["SubtitleWordTiming"] = Field(default_factory=list)
    original_text: str = ""
    original_line_count: int = Field(default=1, ge=1, le=32)
    resolved_lines: list[str] = Field(default_factory=list)
    resolved_font_size: int | None = Field(default=None, ge=8, le=240)
    split_reason: str | None = None
    layout_state: Literal["raw", "segmented", "wrapped", "fitted", "fallback_fitted", "invalid"] = "fitted"
    fallback_used: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "SubtitleCue":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("subtitle cue must have positive duration")
        return self


class SubtitleWordTiming(BaseModel):
    """Actual word timing in the final mixed-audio timeline when available."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=400)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "SubtitleWordTiming":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("subtitle word timing must have positive duration")
        return self


class SubtitleStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style_id: Literal["minimal", "documentary", "dynamic", "clean"]
    font_family: str = Field(min_length=1, max_length=160)
    font_size: int = Field(ge=12, le=240)
    font_weight: Literal["normal", "bold"] = "bold"
    text_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    highlight_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_width: float = Field(ge=0, le=16)
    shadow: float = Field(ge=0, le=16)
    background: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$|^transparent$")
    position: Literal["bottom", "top"] = "bottom"
    bottom_margin: int = Field(ge=0, le=1000)
    alignment: Literal["center", "left"] = "center"
    uppercase: bool = False
    max_chars_per_line: int = Field(default=28, ge=8, le=80)
    max_lines: int = Field(default=2, ge=1, le=4)


class SubtitleProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    schema_version: str = SUBTITLE_PROJECT_SCHEMA_VERSION
    audio_project_id: str
    duration_seconds: float = Field(ge=0)
    style: SubtitleStyle
    cues: list[SubtitleCue] = Field(default_factory=list)
    font_fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    layout_contract_version: str = "4D.0"

    @model_validator(mode="after")
    def _valid_cues(self) -> "SubtitleProject":
        previous = 0.0
        for cue in self.cues:
            if cue.start_seconds < previous - 0.02 or cue.end_seconds > self.duration_seconds + 0.02:
                raise ValueError("subtitle cues must be ordered and within duration")
            previous = cue.end_seconds
        return self


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_project_id: str
    source_path: str
    mixed_audio_path: str
    canvas: CanvasConfig
    encoder_preference: Literal["auto", "nvenc", "cpu"]
    video_codec: Literal["h264"] = "h264"
    video_bitrate: str = Field(min_length=2, max_length=32)
    subtitles_enabled: bool
    render_schema_version: str = VIDEO_PROJECT_SCHEMA_VERSION


class RenderArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["final_mp4", "video_project", "video_timeline", "reframe_plan", "subtitle_project", "production_ass", "render_result", "summary", "clip"]
    path: str
    checksum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    byte_size: int = Field(ge=0)


class RenderValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["valid", "warning", "invalid"]
    video_duration_seconds: float | None = Field(default=None, ge=0)
    audio_duration_seconds: float | None = Field(default=None, ge=0)
    sync_difference_ms: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0)
    video_codec: str | None = None
    audio_codec: str | None = None
    pixel_format: str | None = None
    messages: list[str] = Field(default_factory=list)


class RenderError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str
    message: str
    recoverable: bool = False


class RenderMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = VIDEO_PROJECT_SCHEMA_VERSION
    production_plan_id: str
    audio_project_id: str
    source_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    mixed_audio_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    render_config_version: str
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: str
    updated_at: str
    ai_called: Literal[False] = False
    tts_regenerated: Literal[False] = False
    audio_remixed: Literal[False] = False
    legacy_render_mutated: Literal[False] = False


class RenderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "warning", "failed", "skipped"]
    output_file: str | None = None
    encoder: str | None = None
    hardware_fallback: bool = False
    cache_hit: bool = False
    validation: RenderValidation
    artifacts: list[RenderArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[RenderError] = Field(default_factory=list)


class VideoProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    status: Literal["completed", "warning", "failed", "skipped"]
    source_video_path: str
    source_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    production_plan_id: str
    audio_project_id: str
    mixed_audio_path: str
    canvas: CanvasConfig
    target_duration_seconds: float = Field(ge=0)
    actual_duration_seconds: float = Field(ge=0)
    timeline: VideoTimeline
    reframe_plan: ReframePlan
    tracks: list[VideoTrack]
    subtitle_project: SubtitleProject | None = None
    render_request: RenderRequest
    metadata: RenderMetadata
    result: RenderResult | None = None
    warnings: list[str] = Field(default_factory=list)
    fallback_reasons: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _duration_matches_timeline(self) -> "VideoProject":
        if abs(self.timeline.duration_seconds - self.target_duration_seconds) > 0.02:
            raise ValueError("VideoProject target duration must match video timeline")
        if {track.track_type for track in self.tracks} != {"visual"}:
            raise ValueError("VideoProject must contain one visual track")
        return self
