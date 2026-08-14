from __future__ import annotations

"""Versioned Tier 1 caption policies backed by bundled static font faces."""

from dataclasses import dataclass
from typing import Literal


CAPTION_PRESET_REGISTRY_VERSION = "production.captions.2"
CAPTION_PRESET_VERSION = "2.0.0"
CAPTION_CALIBRATED_PRESET_VERSION = "2.1.0"

CaptionPresetId = Literal[
    "clean_white",
    "minimal_light",
    "accent_yellow",
    "editorial_narrow",
    "karaoke_yellow",
    "contrast_box",
    "word_pop",
]
CaptionStyleFamily = Literal["clean", "emphasis", "minimal", "editorial"]
CaptionPrimitive = Literal["static", "fade", "scale", "slide", "karaoke", "word_pop"]


@dataclass(frozen=True, slots=True)
class CaptionPresetDefinition:
    preset_id: CaptionPresetId
    preset_version: str
    label: str
    style_family: CaptionStyleFamily
    legacy_style_id: Literal["clean", "dynamic", "minimal", "documentary"]
    preferred_font_asset_id: str
    semantic_font_asset_id: str | None
    font_weight: Literal["normal", "bold"]
    font_size_ratio: float
    minimum_font_scale: float
    line_height: float
    text_color: str
    highlight_color: str
    outline_color: str
    outline_width_ratio: float
    shadow_ratio: float
    max_width_ratio: float
    uppercase_emphasis: bool
    motion_profile_id: Literal[
        "static_safe", "semantic_fade", "semantic_dynamic", "semantic_karaoke", "spoken_word_pop",
    ]
    allowed_primitives: tuple[CaptionPrimitive, ...]
    display_mode: Literal["phrase", "single_spoken_word"] = "phrase"
    semantic_bold: bool = False
    background_mode: Literal["transparent", "opaque_box"] = "transparent"
    background_color: str = "#000000"
    background_opacity: float = 0.0
    box_padding_ratio: float = 0.0
    reduced_motion_fallback: Literal["static", "fade"] = "static"
    pop_minimum_frames: int = 7
    pop_scale_keyframes: tuple[int, int, int] = (100, 100, 100)
    semantic_pop_scale_keyframes: tuple[int, int, int] = (100, 100, 100)
    reduced_motion_semantic_scale: int = 100

    @property
    def token_id(self) -> str:
        return f"caption-preset:{self.preset_id}:{self.preset_version}"

    @property
    def font_asset_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(filter(None, (
            self.preferred_font_asset_id,
            self.semantic_font_asset_id,
        ))))


CAPTION_PRESET_DEFINITIONS: dict[CaptionPresetId, CaptionPresetDefinition] = {
    "clean_white": CaptionPresetDefinition(
        preset_id="clean_white", preset_version=CAPTION_PRESET_VERSION, label="Clean",
        style_family="clean", legacy_style_id="clean",
        preferred_font_asset_id="font.manrope.bold", semantic_font_asset_id=None,
        font_weight="bold", font_size_ratio=0.032, minimum_font_scale=0.76,
        line_height=1.18, text_color="#FFFFFF", highlight_color="#FFFFFF",
        outline_color="#242424", outline_width_ratio=0.00105, shadow_ratio=0.00052,
        max_width_ratio=0.86, uppercase_emphasis=False,
        motion_profile_id="semantic_fade", allowed_primitives=("static", "fade"),
    ),
    "minimal_light": CaptionPresetDefinition(
        preset_id="minimal_light", preset_version=CAPTION_CALIBRATED_PRESET_VERSION, label="Minimal Premium",
        style_family="minimal", legacy_style_id="minimal",
        preferred_font_asset_id="font.commissioner.light", semantic_font_asset_id=None,
        font_weight="normal", font_size_ratio=0.036, minimum_font_scale=0.78,
        line_height=1.20, text_color="#FFFFFF", highlight_color="#FFFFFF",
        outline_color="#202020", outline_width_ratio=0.00105, shadow_ratio=0.00039,
        max_width_ratio=0.80, uppercase_emphasis=False,
        motion_profile_id="static_safe", allowed_primitives=("static", "fade"),
    ),
    "accent_yellow": CaptionPresetDefinition(
        preset_id="accent_yellow", preset_version=CAPTION_CALIBRATED_PRESET_VERSION, label="Impact",
        style_family="emphasis", legacy_style_id="dynamic",
        preferred_font_asset_id="font.oswald.bold", semantic_font_asset_id=None,
        font_weight="bold", font_size_ratio=0.043, minimum_font_scale=0.72,
        line_height=1.08, text_color="#FFFFFF", highlight_color="#FFD54A",
        outline_color="#111111", outline_width_ratio=0.00208, shadow_ratio=0.00104,
        max_width_ratio=0.84, uppercase_emphasis=True,
        motion_profile_id="semantic_dynamic",
        allowed_primitives=("static", "fade", "scale", "slide", "karaoke"),
        reduced_motion_fallback="fade",
    ),
    "editorial_narrow": CaptionPresetDefinition(
        preset_id="editorial_narrow", preset_version=CAPTION_CALIBRATED_PRESET_VERSION, label="Editorial",
        style_family="editorial", legacy_style_id="documentary",
        preferred_font_asset_id="font.pt-sans-narrow.regular",
        semantic_font_asset_id="font.pt-sans-narrow.bold",
        font_weight="normal", font_size_ratio=0.040, minimum_font_scale=0.76,
        line_height=1.14, text_color="#FFFFFF", highlight_color="#FFFFFF",
        outline_color="#101010", outline_width_ratio=0.00105, shadow_ratio=0.0,
        max_width_ratio=0.84, uppercase_emphasis=False, semantic_bold=True,
        motion_profile_id="semantic_fade",
        allowed_primitives=("static", "fade", "scale", "karaoke"),
    ),
    "karaoke_yellow": CaptionPresetDefinition(
        preset_id="karaoke_yellow", preset_version=CAPTION_PRESET_VERSION, label="Active / Karaoke",
        style_family="emphasis", legacy_style_id="dynamic",
        preferred_font_asset_id="font.golos-text.bold", semantic_font_asset_id=None,
        font_weight="bold", font_size_ratio=0.036, minimum_font_scale=0.74,
        line_height=1.14, text_color="#FFFFFF", highlight_color="#FFD54A",
        outline_color="#111111", outline_width_ratio=0.00182, shadow_ratio=0.00078,
        max_width_ratio=0.86, uppercase_emphasis=False,
        motion_profile_id="semantic_karaoke", allowed_primitives=("static", "karaoke"),
    ),
    "contrast_box": CaptionPresetDefinition(
        preset_id="contrast_box", preset_version=CAPTION_PRESET_VERSION, label="Contrast Box 2.0",
        style_family="clean", legacy_style_id="clean",
        preferred_font_asset_id="font.rubik.semibold", semantic_font_asset_id=None,
        font_weight="bold", font_size_ratio=0.034, minimum_font_scale=0.76,
        line_height=1.18, text_color="#FFFFFF", highlight_color="#FFFFFF",
        outline_color="#000000", outline_width_ratio=0.0, shadow_ratio=0.0,
        max_width_ratio=0.84, uppercase_emphasis=False,
        motion_profile_id="static_safe", allowed_primitives=("static", "fade"),
        background_mode="opaque_box", background_color="#000000", background_opacity=0.72,
        box_padding_ratio=0.008,
    ),
    "word_pop": CaptionPresetDefinition(
        preset_id="word_pop", preset_version=CAPTION_CALIBRATED_PRESET_VERSION, label="Word Pop",
        style_family="emphasis", legacy_style_id="dynamic",
        preferred_font_asset_id="font.unbounded.bold", semantic_font_asset_id=None,
        font_weight="bold", font_size_ratio=0.049, minimum_font_scale=0.68,
        line_height=1.05, text_color="#FFFFFF", highlight_color="#C6FF00",
        outline_color="#111111", outline_width_ratio=0.00182, shadow_ratio=0.00052,
        max_width_ratio=0.78, uppercase_emphasis=False,
        motion_profile_id="spoken_word_pop", allowed_primitives=("static", "word_pop"),
        display_mode="single_spoken_word", pop_minimum_frames=7,
        pop_scale_keyframes=(88, 112, 100),
        semantic_pop_scale_keyframes=(84, 118, 100),
        reduced_motion_semantic_scale=106,
    ),
}


# Old tokens remain readable for persisted CaptionPlans. New compilation always
# resolves through the current definitions above.
_LEGACY_PRESETS = {
    preset_id: CaptionPresetDefinition(
        preset_id=preset_id,
        preset_version="1.0.0",
        label=item.label,
        style_family=item.style_family,
        legacy_style_id=item.legacy_style_id,
        preferred_font_asset_id=item.preferred_font_asset_id,
        semantic_font_asset_id=item.semantic_font_asset_id,
        font_weight=item.font_weight,
        font_size_ratio=item.font_size_ratio,
        minimum_font_scale=item.minimum_font_scale,
        line_height=item.line_height,
        text_color=item.text_color,
        highlight_color=item.highlight_color,
        outline_color=item.outline_color,
        outline_width_ratio=item.outline_width_ratio,
        shadow_ratio=item.shadow_ratio,
        max_width_ratio=item.max_width_ratio,
        uppercase_emphasis=item.uppercase_emphasis,
        motion_profile_id=item.motion_profile_id,
        allowed_primitives=item.allowed_primitives,
        display_mode=item.display_mode,
        semantic_bold=item.semantic_bold,
        background_mode=item.background_mode,
        background_color=item.background_color,
        background_opacity=0.68 if preset_id == "contrast_box" else item.background_opacity,
        box_padding_ratio=0.006 if preset_id == "contrast_box" else item.box_padding_ratio,
        reduced_motion_fallback=item.reduced_motion_fallback,
        pop_minimum_frames=item.pop_minimum_frames,
        pop_scale_keyframes=item.pop_scale_keyframes,
        semantic_pop_scale_keyframes=item.semantic_pop_scale_keyframes,
        reduced_motion_semantic_scale=item.reduced_motion_semantic_scale,
    )
    for preset_id, item in CAPTION_PRESET_DEFINITIONS.items()
    if preset_id != "word_pop"
}

CAPTION_PRESET_VERSIONS: dict[tuple[CaptionPresetId, str], CaptionPresetDefinition] = {
    (item.preset_id, item.preset_version): item
    for item in (*_LEGACY_PRESETS.values(), *CAPTION_PRESET_DEFINITIONS.values())
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


def caption_preset_from_policy_id(policy_id: str) -> CaptionPresetDefinition | None:
    return CAPTION_PRESET_DEFINITIONS.get(policy_id)  # type: ignore[arg-type]
