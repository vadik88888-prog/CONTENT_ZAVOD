from __future__ import annotations

"""Versioned caption visual policies for the friend-beta Creative Engine."""

from dataclasses import dataclass
from typing import Literal


CAPTION_PRESET_REGISTRY_VERSION = "friend-beta.captions.1"
CAPTION_PRESET_VERSION = "1.0.0"

CaptionPresetId = Literal[
    "clean_white",
    "minimal_light",
    "accent_yellow",
    "karaoke_yellow",
    "editorial_narrow",
    "contrast_box",
]
CaptionStyleFamily = Literal["clean", "emphasis", "minimal", "editorial"]


@dataclass(frozen=True, slots=True)
class CaptionPresetDefinition:
    preset_id: CaptionPresetId
    preset_version: str
    style_family: CaptionStyleFamily
    legacy_style_id: Literal["clean", "dynamic", "minimal", "documentary"]
    preferred_font_asset_id: str
    font_weight: Literal["normal", "bold"]
    motion_profile_id: Literal[
        "static_safe", "semantic_fade", "semantic_dynamic", "semantic_karaoke",
    ]
    allowed_primitives: tuple[Literal["static", "fade", "scale", "slide", "karaoke"], ...]
    background_mode: Literal["transparent", "opaque_box"] = "transparent"
    background_color: str = "#000000"
    background_opacity: float = 0.0
    box_padding_ratio: float = 0.0
    reduced_motion_fallback: Literal["static", "fade"] = "static"

    @property
    def token_id(self) -> str:
        return f"caption-preset:{self.preset_id}:{self.preset_version}"


CAPTION_PRESET_DEFINITIONS: dict[CaptionPresetId, CaptionPresetDefinition] = {
    "clean_white": CaptionPresetDefinition(
        preset_id="clean_white", preset_version=CAPTION_PRESET_VERSION,
        style_family="clean", legacy_style_id="clean",
        preferred_font_asset_id="font.golos-text.bold", font_weight="bold",
        motion_profile_id="semantic_fade", allowed_primitives=("static", "fade"),
    ),
    "minimal_light": CaptionPresetDefinition(
        preset_id="minimal_light", preset_version=CAPTION_PRESET_VERSION,
        style_family="minimal", legacy_style_id="minimal",
        preferred_font_asset_id="font.inter.regular", font_weight="normal",
        motion_profile_id="static_safe", allowed_primitives=("static", "fade"),
    ),
    "accent_yellow": CaptionPresetDefinition(
        preset_id="accent_yellow", preset_version=CAPTION_PRESET_VERSION,
        style_family="emphasis", legacy_style_id="dynamic",
        preferred_font_asset_id="font.inter.bold", font_weight="bold",
        motion_profile_id="semantic_dynamic",
        allowed_primitives=("static", "fade", "scale", "slide", "karaoke"),
        reduced_motion_fallback="fade",
    ),
    "karaoke_yellow": CaptionPresetDefinition(
        preset_id="karaoke_yellow", preset_version=CAPTION_PRESET_VERSION,
        style_family="emphasis", legacy_style_id="dynamic",
        preferred_font_asset_id="font.inter.bold", font_weight="bold",
        motion_profile_id="semantic_karaoke", allowed_primitives=("static", "karaoke"),
    ),
    "editorial_narrow": CaptionPresetDefinition(
        preset_id="editorial_narrow", preset_version=CAPTION_PRESET_VERSION,
        style_family="editorial", legacy_style_id="documentary",
        preferred_font_asset_id="font.pt-sans-narrow.bold", font_weight="bold",
        motion_profile_id="semantic_fade",
        allowed_primitives=("static", "fade", "scale", "karaoke"),
    ),
    "contrast_box": CaptionPresetDefinition(
        preset_id="contrast_box", preset_version=CAPTION_PRESET_VERSION,
        style_family="clean", legacy_style_id="clean",
        preferred_font_asset_id="font.golos-text.bold", font_weight="bold",
        motion_profile_id="static_safe", allowed_primitives=("static", "fade"),
        background_mode="opaque_box", background_color="#000000", background_opacity=0.68,
        box_padding_ratio=0.006,
    ),
}

CAPTION_PRESET_VERSIONS: dict[tuple[CaptionPresetId, str], CaptionPresetDefinition] = {
    (item.preset_id, item.preset_version): item
    for item in CAPTION_PRESET_DEFINITIONS.values()
}
CURRENT_CAPTION_PRESET_VERSIONS: dict[CaptionPresetId, str] = {
    item.preset_id: item.preset_version for item in CAPTION_PRESET_DEFINITIONS.values()
}
_TOKEN_INDEX = {item.token_id: item for item in CAPTION_PRESET_VERSIONS.values()}
_DEFAULT_BY_STYLE: dict[CaptionStyleFamily, CaptionPresetId] = {
    "clean": "clean_white",
    "emphasis": "accent_yellow",
    "minimal": "minimal_light",
    "editorial": "editorial_narrow",
}


def caption_preset_definition(
    preset_id: CaptionPresetId,
    preset_version: str | None = None,
) -> CaptionPresetDefinition:
    version = preset_version or CURRENT_CAPTION_PRESET_VERSIONS[preset_id]
    return CAPTION_PRESET_VERSIONS[(preset_id, version)]


def default_caption_preset_for_style(style_family: CaptionStyleFamily) -> CaptionPresetDefinition:
    return caption_preset_definition(_DEFAULT_BY_STYLE[style_family])


def caption_preset_from_token_id(token_id: str) -> CaptionPresetDefinition | None:
    return _TOKEN_INDEX.get(token_id)
