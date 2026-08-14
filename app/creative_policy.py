from __future__ import annotations

"""Versioned friend-beta definitions for the four existing preset IDs."""

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping

from app.caption_presets import CaptionPresetId, CaptionStyleFamily
from app.content_profile_taxonomy import PROFILE_AXIS_ORDER
from app.creative_contracts import Intensity


CREATIVE_POLICY_VERSION = "7K.1"
CREATIVE_PRESET_VERSION = "1.0.0"

PresetFamily = Literal["minimal", "documentary", "dynamic", "clean"]


@dataclass(frozen=True, slots=True)
class CreativePresetDefinition:
    preset_id: PresetFamily
    preset_version: str
    label: str
    family: PresetFamily
    policy_version: str
    caption_preset_id: CaptionPresetId
    caption_style_family: CaptionStyleFamily
    caption_density: Literal["low", "balanced", "high"]
    intensity_ceiling: Intensity
    composition_profile_id: Literal["stable_speaker", "evidence_dynamic", "safe_editorial", "source_first"]
    motion_profile_id: Literal["minimal", "balanced", "expressive"]
    audio_profile_id: Literal["source_dialogue"] = "source_dialogue"
    ordered_fallbacks: tuple[str, ...] = (
        "drop_emphasis", "static_state", "stable_source", "approved_font", "block",
    )
    source_extra_shots_default: Literal[False] = False
    stock_broll_allowed: Literal[False] = False
    generative_broll_allowed: Literal[False] = False


CREATIVE_PRESET_DEFINITIONS: dict[PresetFamily, CreativePresetDefinition] = {
    "minimal": CreativePresetDefinition(
        preset_id="minimal", preset_version=CREATIVE_PRESET_VERSION,
        label="Minimal Premium", family="minimal", policy_version=CREATIVE_POLICY_VERSION,
        caption_preset_id="minimal_light", caption_style_family="minimal", caption_density="low",
        intensity_ceiling=Intensity.LOW, composition_profile_id="source_first",
        motion_profile_id="minimal",
    ),
    "documentary": CreativePresetDefinition(
        preset_id="documentary", preset_version=CREATIVE_PRESET_VERSION,
        label="Educational", family="documentary", policy_version=CREATIVE_POLICY_VERSION,
        caption_preset_id="editorial_narrow", caption_style_family="editorial",
        caption_density="balanced", intensity_ceiling=Intensity.BALANCED,
        composition_profile_id="safe_editorial", motion_profile_id="balanced",
    ),
    "dynamic": CreativePresetDefinition(
        preset_id="dynamic", preset_version=CREATIVE_PRESET_VERSION,
        label="Expert Dynamic", family="dynamic", policy_version=CREATIVE_POLICY_VERSION,
        caption_preset_id="accent_yellow", caption_style_family="emphasis", caption_density="high",
        intensity_ceiling=Intensity.HIGH, composition_profile_id="evidence_dynamic",
        motion_profile_id="expressive",
    ),
    "clean": CreativePresetDefinition(
        preset_id="clean", preset_version=CREATIVE_PRESET_VERSION,
        label="Clean Podcast", family="clean", policy_version=CREATIVE_POLICY_VERSION,
        caption_preset_id="clean_white", caption_style_family="clean", caption_density="balanced",
        intensity_ceiling=Intensity.LOW, composition_profile_id="stable_speaker",
        motion_profile_id="minimal",
    ),
}

CREATIVE_PRESET_VERSIONS: dict[tuple[PresetFamily, str], CreativePresetDefinition] = {
    (item.preset_id, item.preset_version): item
    for item in CREATIVE_PRESET_DEFINITIONS.values()
}
CURRENT_CREATIVE_PRESET_VERSIONS: dict[PresetFamily, str] = {
    item.preset_id: item.preset_version for item in CREATIVE_PRESET_DEFINITIONS.values()
}

# Compatibility names remain importable for existing callers and persisted-flow
# tests. They reference the same immutable definitions, not a second registry.
PresetFamilyPolicy = CreativePresetDefinition
PRESET_FAMILY_POLICIES = CREATIVE_PRESET_DEFINITIONS


def creative_profile_signal(content_profile: str | Mapping[str, Any] | None) -> str:
    """Adapt structured v2 or legacy profile data to the preset policy signal.

    The structured effective profile is authoritative when present.  Reading
    detected and effective values separately here prevents a detection from
    silently overriding an explicit user choice.
    """

    if isinstance(content_profile, str):
        return content_profile
    if not isinstance(content_profile, Mapping):
        return ""
    nested = content_profile.get("content_profile")
    profile = nested if isinstance(nested, Mapping) else content_profile
    effective = profile.get("effective_profile")
    if isinstance(effective, Mapping):
        signals: list[str] = []
        for axis_id in PROFILE_AXIS_ORDER:
            value = effective.get(axis_id)
            if isinstance(value, (list, tuple)):
                signals.extend(str(item) for item in value if item)
            elif value:
                signals.append(str(value))
        if signals:
            return " ".join(signals)
    legacy_structured = " ".join(
        str(profile.get(key) or "")
        for key in ("detected_content_type", "content_type", "content_kind", "dominant_format")
    ).strip()
    if legacy_structured:
        return legacy_structured
    return " ".join(
        str(profile.get(key) or "")
        for key in ("genre", "title", "filename")
    ).strip()


def recommend_preset_family(content_type: str | Mapping[str, Any] | None) -> PresetFamily:
    """Return a conservative recommendation; this never represents a user choice."""

    normalized = re.sub(r"[^a-z0-9]+", "_", creative_profile_signal(content_type).casefold()).strip("_")
    tokens = frozenset(filter(None, normalized.split("_")))
    if tokens & {"game", "gameplay", "gaming", "stream"}:
        return "minimal"
    if tokens & {"vlog", "travel", "food", "lifestyle"}:
        return "dynamic"
    if tokens & {"interview", "qa", "conversation", "dialogue"}:
        return "documentary"
    if tokens & {"podcast", "talking", "lecture", "webinar"}:
        return "clean"
    return "documentary"


def resolve_preset_family(
    *, user_choice: PresetFamily | None, content_type: str | Mapping[str, Any] | None,
) -> PresetFamily:
    """Resolve policy with the explicit user choice above automatic recommendation."""

    return user_choice if user_choice is not None else recommend_preset_family(content_type)


def preset_family_policy(family: PresetFamily) -> CreativePresetDefinition:
    return CREATIVE_PRESET_DEFINITIONS[family]


def creative_preset_definition(
    preset_id: PresetFamily,
    preset_version: str | None = None,
) -> CreativePresetDefinition:
    version = preset_version or CURRENT_CREATIVE_PRESET_VERSIONS[preset_id]
    return CREATIVE_PRESET_VERSIONS[(preset_id, version)]
