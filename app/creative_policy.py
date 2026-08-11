from __future__ import annotations

"""Versioned calibration data for the four existing creative preset families."""

from dataclasses import dataclass
import re
from typing import Literal

from app.creative_contracts import Intensity


CREATIVE_POLICY_VERSION = "7J.1"

PresetFamily = Literal["minimal", "documentary", "dynamic", "clean"]
CaptionStyleFamily = Literal["clean", "emphasis", "minimal", "editorial"]


@dataclass(frozen=True, slots=True)
class PresetFamilyPolicy:
    family: PresetFamily
    policy_version: Literal["7J.1"]
    caption_style_family: CaptionStyleFamily
    intensity_ceiling: Intensity
    source_extra_shots_default: Literal[False] = False


PRESET_FAMILY_POLICIES: dict[PresetFamily, PresetFamilyPolicy] = {
    "minimal": PresetFamilyPolicy(
        family="minimal", policy_version=CREATIVE_POLICY_VERSION,
        caption_style_family="minimal", intensity_ceiling=Intensity.LOW,
    ),
    "documentary": PresetFamilyPolicy(
        family="documentary", policy_version=CREATIVE_POLICY_VERSION,
        caption_style_family="editorial", intensity_ceiling=Intensity.BALANCED,
    ),
    "dynamic": PresetFamilyPolicy(
        family="dynamic", policy_version=CREATIVE_POLICY_VERSION,
        caption_style_family="emphasis", intensity_ceiling=Intensity.HIGH,
    ),
    "clean": PresetFamilyPolicy(
        family="clean", policy_version=CREATIVE_POLICY_VERSION,
        caption_style_family="clean", intensity_ceiling=Intensity.LOW,
    ),
}


def recommend_preset_family(content_type: str | None) -> PresetFamily:
    """Return a conservative recommendation; this never represents a user choice."""

    normalized = re.sub(r"[^a-z0-9]+", "_", (content_type or "").casefold()).strip("_")
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
    *, user_choice: PresetFamily | None, content_type: str | None,
) -> PresetFamily:
    """Resolve policy with the explicit user choice above automatic recommendation."""

    return user_choice if user_choice is not None else recommend_preset_family(content_type)


def preset_family_policy(family: PresetFamily) -> PresetFamilyPolicy:
    return PRESET_FAMILY_POLICIES[family]
