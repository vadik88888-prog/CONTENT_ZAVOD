"""User-facing processing presets and estimates for the desktop product flow.

This module is deliberately independent from Qt and the command-line parser.  The
desktop supplies an :class:`ProcessingIntent`; this module resolves it once into
runtime values that the existing pipeline already understands.  Keeping that
translation here prevents product presets from being copied into every screen,
service and pipeline stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, cast

from app.content_profile_taxonomy import (
    AUTO_PROFILE_INPUT,
    order_profile_ids,
    profile_input_ids,
    user_override_ids,
)
from app.creative_policy import (
    PresetFamily,
    creative_preset_definition,
    recommend_preset_family,
    resolve_preset_family,
)


PROCESSING_MODES = frozenset({"fast", "standard", "maximum"})
DEEP_ANALYSIS_MODES = frozenset({"auto", "on", "off"})
PLATFORMS = frozenset({"tiktok", "reels", "shorts", "universal"})
SUBTITLE_PRESETS = frozenset({"minimal", "documentary", "dynamic", "clean"})
PRESET_SELECTION_MODES = frozenset({"auto", "explicit"})
CLIP_COUNTS = frozenset({"auto", "1", "3", "5"})
AUDIO_MODES = frozenset({"original", "original_enhanced", "voiceover", "replace_voice", "mixed"})
# Compatibility aliases derived from the canonical registry.
PROFILE_FORMAT_OVERRIDES = frozenset(profile_input_ids("format"))
PROFILE_EDITORIAL_MODE_OVERRIDES = frozenset(profile_input_ids("editorial_mode"))
PROFILE_DOMAIN_OVERRIDES = frozenset(profile_input_ids("domain"))
PROFILE_TRAIT_OVERRIDES = frozenset(user_override_ids("traits"))
PRESET_RESOLVER_VERSION = "4B.1"


@dataclass(frozen=True, slots=True)
class PlatformPreset:
    platform: str
    label: str
    width: int
    height: int
    safe_top_ratio: float
    safe_bottom_ratio: float
    subtitle_bottom_ratio: float
    target_duration_seconds: float
    maximum_duration_seconds: float
    video_bitrate: str
    output_fps: float
    filename_suffix: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PLATFORM_PRESETS: dict[str, PlatformPreset] = {
    "tiktok": PlatformPreset("tiktok", "TikTok", 1080, 1920, 0.10, 0.19, 0.18, 35.0, 60.0, "6M", 30.0, "tiktok"),
    "reels": PlatformPreset("reels", "Instagram Reels", 1080, 1920, 0.13, 0.18, 0.17, 35.0, 90.0, "6M", 30.0, "reels"),
    "shorts": PlatformPreset("shorts", "YouTube Shorts", 1080, 1920, 0.08, 0.14, 0.14, 40.0, 60.0, "7M", 30.0, "shorts"),
    "universal": PlatformPreset("universal", "Универсальный вертикальный", 1080, 1920, 0.10, 0.16, 0.16, 35.0, 60.0, "6M", 30.0, "vertical"),
}


@dataclass(frozen=True, slots=True)
class ProcessingIntent:
    """The small set of choices exposed in the main desktop experience."""

    processing_mode: str = "standard"
    deep_analysis: str = "auto"
    platform: str = "universal"
    clip_count: str = "3"
    subtitle_preset: str = "documentary"
    preset_selection_mode: str = "auto"
    audio_mode: str = "original"
    editorial_intent: str = ""
    profile_format_override: str = AUTO_PROFILE_INPUT
    profile_editorial_mode_override: str = AUTO_PROFILE_INPUT
    profile_domain_override: str = AUTO_PROFILE_INPUT
    profile_traits_override: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.processing_mode not in PROCESSING_MODES:
            raise ValueError("Unsupported processing mode.")
        if self.deep_analysis not in DEEP_ANALYSIS_MODES:
            raise ValueError("Unsupported deep analysis mode.")
        if self.platform not in PLATFORMS:
            raise ValueError("Unsupported platform.")
        if str(self.clip_count) not in CLIP_COUNTS:
            raise ValueError("Unsupported clip count.")
        if self.subtitle_preset not in SUBTITLE_PRESETS:
            raise ValueError("Unsupported subtitle preset.")
        if self.preset_selection_mode not in PRESET_SELECTION_MODES:
            raise ValueError("Unsupported preset selection mode.")
        if self.audio_mode not in AUDIO_MODES:
            raise ValueError("Unsupported audio mode.")
        if not isinstance(self.editorial_intent, str) or len(self.editorial_intent.strip()) > 500:
            raise ValueError("Editorial intent must be a string up to 500 characters.")
        if self.profile_format_override not in PROFILE_FORMAT_OVERRIDES:
            raise ValueError("Unsupported profile format override.")
        if self.profile_editorial_mode_override not in PROFILE_EDITORIAL_MODE_OVERRIDES:
            raise ValueError("Unsupported profile editorial mode override.")
        if self.profile_domain_override not in PROFILE_DOMAIN_OVERRIDES:
            raise ValueError("Unsupported profile domain override.")
        if any(item not in PROFILE_TRAIT_OVERRIDES for item in self.profile_traits_override):
            raise ValueError("Unsupported profile traits override.")

    @property
    def requested_clip_count(self) -> int | None:
        return None if str(self.clip_count) == "auto" else int(self.clip_count)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "processing_mode": self.processing_mode,
            "deep_analysis": self.deep_analysis,
            "platform": self.platform,
            "clip_count": str(self.clip_count),
            "subtitle_preset": self.subtitle_preset,
            "preset_selection_mode": self.preset_selection_mode,
            "audio_mode": self.audio_mode,
            "editorial_intent": self.editorial_intent.strip(),
            "profile_format_override": self.profile_format_override,
            "profile_editorial_mode_override": self.profile_editorial_mode_override,
            "profile_domain_override": self.profile_domain_override,
            "profile_traits_override": list(order_profile_ids("traits", list(self.profile_traits_override))),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProcessingIntent":
        intent = cls(
            processing_mode=str(value.get("processing_mode", "standard")),
            deep_analysis=str(value.get("deep_analysis", "auto")),
            platform=str(value.get("platform", "universal")),
            clip_count=str(value.get("clip_count", "3")),
            subtitle_preset=str(value.get("subtitle_preset", "documentary")),
            # Payloads without provenance predate automatic selection. Their
            # stored preset is user-owned/pinned and must remain effective.
            preset_selection_mode=str(value.get("preset_selection_mode", "explicit")),
            audio_mode=str(value.get("audio_mode", "original")),
            editorial_intent=str(value.get("editorial_intent", "")),
            profile_format_override=str(value.get("profile_format_override", AUTO_PROFILE_INPUT)),
            profile_editorial_mode_override=str(value.get("profile_editorial_mode_override", AUTO_PROFILE_INPUT)),
            profile_domain_override=str(value.get("profile_domain_override", AUTO_PROFILE_INPUT)),
            profile_traits_override=tuple(
                str(item) for item in (
                    value.get("profile_traits_override")
                    if isinstance(value.get("profile_traits_override"), (list, tuple))
                    else []
                )
            ),
        )
        intent.validate()
        return intent


@dataclass(frozen=True, slots=True)
class DeepAnalysisDecision:
    requested: str
    resolved: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    estimated_benefit: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolvedProcessingConfig:
    """Concrete, serialisable runtime choices produced from an intent."""

    processing_mode: str
    deep_analysis: DeepAnalysisDecision
    platform: PlatformPreset
    clip_count: int
    configured_subtitle_preset: str
    subtitle_preset: str
    recommended_subtitle_preset: str
    preset_selection_mode: str
    preset_provenance: str
    preset_version: str
    audio_mode: str
    editorial_intent: str
    manual_profile_override: dict[str, Any]
    candidate_limit: int
    shortlist_size: int
    ai_reranking_enabled: bool
    transformation_strategy: str
    video_bitrate: str
    crop_strategy: str
    cache_policy: str
    resolver_version: str = PRESET_RESOLVER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "processing_mode": self.processing_mode,
            "deep_analysis": self.deep_analysis.to_dict(),
            "platform": self.platform.to_dict(),
            "clip_count": self.clip_count,
            "configured_subtitle_preset": self.configured_subtitle_preset,
            "subtitle_preset": self.subtitle_preset,
            "effective_subtitle_preset": self.subtitle_preset,
            "recommended_subtitle_preset": self.recommended_subtitle_preset,
            "preset_selection_mode": self.preset_selection_mode,
            "preset_provenance": self.preset_provenance,
            "preset_version": self.preset_version,
            "audio_mode": self.audio_mode,
            "editorial_intent": self.editorial_intent,
            "manual_profile_override": self.manual_profile_override,
            "candidate_limit": self.candidate_limit,
            "shortlist_size": self.shortlist_size,
            "ai_reranking_enabled": self.ai_reranking_enabled,
            "transformation_strategy": self.transformation_strategy,
            "video_bitrate": self.video_bitrate,
            "crop_strategy": self.crop_strategy,
            "cache_policy": self.cache_policy,
            "resolver_version": self.resolver_version,
        }


@dataclass(frozen=True, slots=True)
class ProcessingEstimate:
    estimated_seconds_min: int
    estimated_seconds_max: int
    estimated_ai_cost_min: float | None
    estimated_ai_cost_max: float | None
    estimated_clips_min: int
    estimated_clips_max: int
    deep_analysis_resolved: bool
    cached_stages: tuple[str, ...]
    confidence: str
    assumptions: tuple[str, ...]
    cost_drivers: tuple[str, ...] = ()
    cost_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_seconds_min": self.estimated_seconds_min,
            "estimated_seconds_max": self.estimated_seconds_max,
            "estimated_ai_cost_min": self.estimated_ai_cost_min,
            "estimated_ai_cost_max": self.estimated_ai_cost_max,
            "estimated_clips_min": self.estimated_clips_min,
            "estimated_clips_max": self.estimated_clips_max,
            "deep_analysis_resolved": self.deep_analysis_resolved,
            "cached_stages": list(self.cached_stages),
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "cost_drivers": list(self.cost_drivers),
            "cost_note": self.cost_note,
        }


@dataclass(frozen=True, slots=True)
class CostPricing:
    """Configured provider tariffs used by the preflight estimate.

    This is deliberately a small value object instead of a UI constant.  The
    desktop obtains it from the active engine configuration, so changing a
    provider tariff changes the estimate before the next run.
    """

    input_token_price: float | None
    output_token_price: float | None
    tts_cost_per_1m_characters: float | None = None
    ai_available: bool = True
    tts_available: bool = False

    @property
    def can_estimate_ai(self) -> bool:
        return self.ai_available and self.input_token_price is not None and self.output_token_price is not None

    @property
    def can_estimate_tts(self) -> bool:
        return self.tts_available and self.tts_cost_per_1m_characters is not None


def resolve_deep_analysis(requested: str, source_metadata: dict[str, Any] | None = None) -> DeepAnalysisDecision:
    """Resolve automatic analysis conservatively from local, non-AI evidence.

    A future visual sampler can add ``visual_activity_score`` and ``content_kind``
    to the source metadata without changing the user-facing model.  Unknown
    sources default to speech-first processing rather than silently making a
    paid visual request.
    """

    if requested not in DEEP_ANALYSIS_MODES:
        raise ValueError("Unsupported deep analysis mode.")
    metadata = source_metadata or {}
    evidence: dict[str, Any] = {}
    for key in (
        "duration", "width", "height", "fps", "visual_activity_score", "scene_density",
        "speech_ratio", "silence_ratio", "content_kind", "genre", "title", "filename",
    ):
        if key in metadata:
            evidence[key] = metadata[key]
    if requested == "on":
        return DeepAnalysisDecision(requested, True, "Включено по вашему выбору.", evidence, "high")
    if requested == "off":
        return DeepAnalysisDecision(requested, False, "Выключено по вашему выбору.", evidence, "none")

    kind = " ".join(
        str(metadata.get(key, "")) for key in ("content_kind", "genre", "title", "filename")
    ).lower()
    activity = _number(metadata.get("visual_activity_score"))
    scene_density = _number(metadata.get("scene_density"))
    speech_ratio = _number(metadata.get("speech_ratio"))
    static_markers = ("podcast", "lecture", "interview", "talking", "talking_head", "webinar", "screen_recording")
    dynamic_markers = ("gameplay", "game", "pubg", "movie", "film", "series", "trailer", "review", "travel", "sport", "concert")
    if any(marker in kind for marker in static_markers):
        return DeepAnalysisDecision("auto", False, "Похоже на разговорный или статичный материал: сначала достаточно анализа речи.", evidence, "low")
    if any(marker in kind for marker in dynamic_markers):
        return DeepAnalysisDecision("auto", True, "Похоже на динамичное видео, где важны события в кадре.", evidence, "high")
    if activity is not None and activity >= 0.55:
        return DeepAnalysisDecision("auto", True, "В видео заметна высокая визуальная активность.", evidence, "high")
    if scene_density is not None and scene_density >= 0.55:
        return DeepAnalysisDecision("auto", True, "В видео часто меняются сцены; дополнительный разбор кадра может помочь.", evidence, "high")
    if speech_ratio is not None and speech_ratio >= 0.65 and (activity is None or activity < 0.45):
        return DeepAnalysisDecision("auto", False, "Преобладает речь, а заметных динамичных сцен пока нет.", evidence, "low")
    if activity is not None or scene_density is not None:
        return DeepAnalysisDecision("auto", False, "Визуальная активность невысокая; достаточно анализа речи.", evidence, "low")
    return DeepAnalysisDecision(
        "auto", False,
        "Недостаточно признаков до первого анализа: выбран бережный вариант без дополнительного разбора кадра.",
        evidence, "unknown",
    )


def resolve_processing_intent(intent: ProcessingIntent, source_metadata: dict[str, Any] | None = None) -> ResolvedProcessingConfig:
    intent.validate()
    platform = PLATFORM_PRESETS[intent.platform]
    deep_analysis = resolve_deep_analysis(intent.deep_analysis, source_metadata)
    requested_count = intent.requested_clip_count
    defaults = {
        "fast": {"clips": 1, "candidates": 40, "shortlist": 8, "reranking": False, "strategy": "local_only", "bitrate": "4M", "crop": "fit_blur_background"},
        "standard": {"clips": 3, "candidates": 100, "shortlist": 15, "reranking": True, "strategy": "compact", "bitrate": platform.video_bitrate, "crop": "center_crop"},
        "maximum": {"clips": 5, "candidates": 160, "shortlist": 30, "reranking": True, "strategy": "staged", "bitrate": "8M", "crop": "center_crop"},
    }[intent.processing_mode]
    clip_count = requested_count or int(defaults["clips"])
    content_profile = source_metadata
    recommended_preset = recommend_preset_family(content_profile)
    explicit_choice = (
        cast(PresetFamily, intent.subtitle_preset)
        if intent.preset_selection_mode == "explicit"
        else None
    )
    effective_preset = resolve_preset_family(
        user_choice=explicit_choice,
        content_type=content_profile,
    )
    preset_version = creative_preset_definition(effective_preset).preset_version
    return ResolvedProcessingConfig(
        processing_mode=intent.processing_mode,
        deep_analysis=deep_analysis,
        platform=platform,
        clip_count=clip_count,
        configured_subtitle_preset=intent.subtitle_preset,
        subtitle_preset=effective_preset,
        recommended_subtitle_preset=recommended_preset,
        preset_selection_mode=intent.preset_selection_mode,
        preset_provenance=(
            "explicit_selection"
            if intent.preset_selection_mode == "explicit"
            else "content_recommendation"
        ),
        preset_version=preset_version,
        audio_mode=intent.audio_mode,
        editorial_intent=intent.editorial_intent.strip(),
        manual_profile_override={
            **({"format": intent.profile_format_override} if intent.profile_format_override != AUTO_PROFILE_INPUT else {}),
            **({"editorial_mode": intent.profile_editorial_mode_override} if intent.profile_editorial_mode_override != AUTO_PROFILE_INPUT else {}),
            **({"domain": intent.profile_domain_override} if intent.profile_domain_override != AUTO_PROFILE_INPUT else {}),
            **({"traits": list(order_profile_ids("traits", list(intent.profile_traits_override)))} if intent.profile_traits_override else {}),
        },
        candidate_limit=int(defaults["candidates"]),
        shortlist_size=max(clip_count, int(defaults["shortlist"])),
        ai_reranking_enabled=bool(defaults["reranking"]),
        transformation_strategy=str(defaults["strategy"]),
        video_bitrate=str(defaults["bitrate"]),
        crop_strategy=str(defaults["crop"]),
        cache_policy="reuse" if intent.processing_mode != "maximum" else "reuse-with-quality-refresh",
    )


def estimate_processing(
    resolved: ResolvedProcessingConfig,
    source_metadata: dict[str, Any] | None = None,
    *,
    paid_ai_available: bool,
    cached_stages: tuple[str, ...] = (),
    pricing: CostPricing | None = None,
) -> ProcessingEstimate:
    """Return a bounded estimate, never a misleading single-point promise."""

    metadata = source_metadata or {}
    duration = max(0.0, _number(metadata.get("duration")) or 0.0)
    minutes = max(1.0, duration / 60.0)
    mode_multiplier = {"fast": 0.75, "standard": 1.35, "maximum": 2.25}[resolved.processing_mode]
    base_seconds = 35 + minutes * 42 * mode_multiplier + resolved.clip_count * 9
    if resolved.deep_analysis.resolved:
        base_seconds += minutes * 38
    if cached_stages:
        base_seconds *= 0.62
    estimate_min = max(20, int(round(base_seconds * 0.75)))
    estimate_max = max(estimate_min + 15, int(round(base_seconds * 1.35)))
    # The UI receives configured provider tariffs through ``pricing``.  The
    # fallback keeps the public pure function useful for existing callers while
    # retaining the same defaults as AIConfig.  It is never used by the desktop
    # preflight, which always passes the active configuration explicitly.
    active_pricing = pricing or CostPricing(
        input_token_price=0.00000025,
        output_token_price=0.000002,
        ai_available=paid_ai_available,
    )
    cost_drivers = [
        "длительность исходного видео",
        {"fast": "быстрый режим", "standard": "стандартный режим", "maximum": "максимальный режим"}[resolved.processing_mode],
        f"подготовка до {resolved.clip_count} ролика(ов)",
    ]
    if resolved.deep_analysis.resolved:
        cost_drivers.append("дополнительный разбор кадров")

    cost_min = cost_max = None
    if active_pricing.can_estimate_ai:
        # These are bounded request-volume assumptions, not prices.  Tariffs
        # themselves are always taken from AppConfig.  They track the existing
        # candidate limits and shortlist sizes resolved above, so the estimate
        # changes when a product mode actually changes pipeline work.
        input_per_minute = {"fast": 220, "standard": 420, "maximum": 720}[resolved.processing_mode]
        output_per_shortlist = {"fast": 42, "standard": 78, "maximum": 120}[resolved.processing_mode]
        output_per_clip = {"fast": 120, "standard": 230, "maximum": 360}[resolved.processing_mode]
        input_tokens = minutes * input_per_minute + resolved.shortlist_size * 55
        output_tokens = resolved.shortlist_size * output_per_shortlist + resolved.clip_count * output_per_clip
        if resolved.deep_analysis.resolved:
            input_tokens += minutes * 260
            output_tokens += resolved.clip_count * 75
        base_cost = input_tokens * float(active_pricing.input_token_price) + output_tokens * float(active_pricing.output_token_price)
        cost_min, cost_max = base_cost * 0.72, base_cost * 1.36
        if resolved.audio_mode in {"voiceover", "replace_voice", "mixed"} and active_pricing.can_estimate_tts:
            narration_chars_min = resolved.clip_count * resolved.platform.target_duration_seconds * 9
            narration_chars_max = resolved.clip_count * resolved.platform.target_duration_seconds * 18
            tts_rate = float(active_pricing.tts_cost_per_1m_characters)
            cost_min += narration_chars_min * tts_rate / 1_000_000
            cost_max += narration_chars_max * tts_rate / 1_000_000
            cost_drivers.append("выбранный способ озвучки")
        cost_min, cost_max = round(cost_min, 4), round(max(cost_min, cost_max), 4)
        cost_note = "Диапазон рассчитан по тарифам, указанным в текущих настройках, и объёму работы до запуска."
    elif paid_ai_available:
        cost_note = "Тарифы для текущего сервиса не заданы, поэтому стоимость до запуска не показываем."
    else:
        cost_note = "Платные обращения в этом запуске не используются."
    requested = resolved.clip_count
    return ProcessingEstimate(
        estimated_seconds_min=estimate_min,
        estimated_seconds_max=estimate_max,
        estimated_ai_cost_min=cost_min,
        estimated_ai_cost_max=cost_max,
        estimated_clips_min=max(1, requested - 1) if requested > 1 else 1,
        estimated_clips_max=requested,
        deep_analysis_resolved=resolved.deep_analysis.resolved,
        cached_stages=tuple(cached_stages),
        confidence="medium" if duration else "low",
        assumptions=(
            "Оценка зависит от длительности, выбранного режима и мощности компьютера.",
            "Итоговое число роликов зависит от найденных подходящих фрагментов.",
        ),
        cost_drivers=tuple(cost_drivers),
        cost_note=cost_note,
    )


def calibrate_processing_estimate(estimate: ProcessingEstimate, runs: list[Any]) -> ProcessingEstimate:
    """Adjust a new estimate only from persisted comparable completed runs."""

    ratios: list[float] = []
    for run in runs:
        if str(getattr(run, "status", "")) not in {"completed", "completed_with_warnings"}:
            continue
        snapshot = getattr(run, "settings_snapshot", {})
        if not isinstance(snapshot, dict):
            continue
        previous = snapshot.get("product_flow", {})
        if not isinstance(previous, dict):
            continue
        previous_estimate = previous.get("estimate", {})
        if not isinstance(previous_estimate, dict):
            continue
        try:
            started = datetime.fromisoformat(str(getattr(run, "started_at")))
            finished = datetime.fromisoformat(str(getattr(run, "finished_at")))
            actual = (finished - started).total_seconds()
            midpoint = (float(previous_estimate["estimated_seconds_min"]) + float(previous_estimate["estimated_seconds_max"])) / 2
        except (KeyError, TypeError, ValueError):
            continue
        if 5 <= actual <= 24 * 3600 and midpoint > 0:
            ratios.append(max(0.55, min(1.9, actual / midpoint)))
    if len(ratios) < 2:
        return estimate
    factor = sum(ratios[-8:]) / min(8, len(ratios))
    low = max(15, int(round(estimate.estimated_seconds_min * factor * 0.9)))
    high = max(low + 15, int(round(estimate.estimated_seconds_max * factor * 1.1)))
    return ProcessingEstimate(
        estimated_seconds_min=low, estimated_seconds_max=high,
        estimated_ai_cost_min=estimate.estimated_ai_cost_min, estimated_ai_cost_max=estimate.estimated_ai_cost_max,
        estimated_clips_min=estimate.estimated_clips_min, estimated_clips_max=estimate.estimated_clips_max,
        deep_analysis_resolved=estimate.deep_analysis_resolved, cached_stages=estimate.cached_stages,
        confidence="calibrated",
        assumptions=(*estimate.assumptions, f"Оценка откалибрована по {len(ratios)} завершённым запускам этого приложения."),
        cost_drivers=estimate.cost_drivers,
        cost_note=estimate.cost_note,
    )


def apply_resolved_processing_config(config: Any, resolved: ResolvedProcessingConfig) -> None:
    """Apply only established pipeline settings; no desktop-only execution path."""

    config.max_clips = resolved.clip_count
    config.ai_reranking.final_clip_count = resolved.clip_count
    config.ai_reranking.shortlist_size = resolved.shortlist_size
    config.ai_reranking.enabled = resolved.ai_reranking_enabled
    config.candidate_generation.max_candidates = resolved.candidate_limit
    config.transformation.ai_strategy = resolved.transformation_strategy
    config.production_render.output_width = resolved.platform.width
    config.production_render.output_height = resolved.platform.height
    config.production_render.output_fps = resolved.platform.output_fps
    config.production_render.video_bitrate = resolved.video_bitrate
    config.production_render.crop_strategy = resolved.crop_strategy
    config.production_render.subtitle_style = resolved.subtitle_preset
    config.production.audio_mode = resolved.audio_mode
    config.production_render.cache_enabled = resolved.cache_policy.startswith("reuse")
    config.optional_visual_features = resolved.deep_analysis.resolved
    config.content_understanding.editorial_intent = resolved.editorial_intent
    config.content_understanding.manual_override = dict(resolved.manual_profile_override)
    # Goal 5B is active for the product flow while AppConfig remains backward
    # compatible for external programmatic callers that did not opt in.
    config.virality.enabled = True
    config.virality.semantic_ai_mode = "off" if resolved.processing_mode == "fast" or resolved.deep_analysis.requested == "off" else "auto"
    flow = config.product_flow
    flow.processing_mode = resolved.processing_mode
    flow.deep_analysis_requested = resolved.deep_analysis.requested
    flow.deep_analysis_resolved = resolved.deep_analysis.resolved
    flow.deep_analysis_reason = resolved.deep_analysis.reason
    flow.platform = resolved.platform.platform
    flow.clip_count = resolved.clip_count
    flow.configured_subtitle_preset = resolved.configured_subtitle_preset
    flow.subtitle_preset = resolved.subtitle_preset
    flow.recommended_subtitle_preset = resolved.recommended_subtitle_preset
    flow.preset_selection_mode = resolved.preset_selection_mode
    flow.preset_provenance = resolved.preset_provenance
    flow.audio_mode = resolved.audio_mode
    flow.preset_version = resolved.preset_version

def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
