from __future__ import annotations

"""Stable friend-beta font asset identities and license metadata.

The registry is descriptive: renderer identity remains the registry asset ID plus
the exact file SHA-256 recorded in ``CaptionFontManifest``.  A host font never
becomes redistributable merely because it can be resolved locally.
"""

from dataclasses import dataclass
import re
from typing import Literal


FONT_ASSET_REGISTRY_VERSION = "friend-beta.fonts.1"

FontWeight = Literal["normal", "bold"]
FontStyle = Literal["normal", "italic"]


@dataclass(frozen=True, slots=True)
class FontAssetDefinition:
    asset_id: str
    asset_version: str
    family: str
    style: FontStyle
    weight: FontWeight
    license_id: str
    license_notice_required: bool
    source_url: str
    supported_scripts: tuple[Literal["latin", "cyrillic"], ...]
    redistribution_allowed: bool


def _curated(
    asset_id: str,
    family: str,
    weight: FontWeight,
    source_url: str,
) -> FontAssetDefinition:
    return FontAssetDefinition(
        asset_id=asset_id,
        asset_version="friend-beta.1",
        family=family,
        style="normal",
        weight=weight,
        license_id="OFL-1.1",
        license_notice_required=True,
        source_url=source_url,
        supported_scripts=("latin", "cyrillic"),
        redistribution_allowed=True,
    )


FONT_ASSET_DEFINITIONS: dict[str, FontAssetDefinition] = {
    item.asset_id: item
    for item in (
        _curated(
            "font.golos-text.regular", "Golos Text", "normal",
            "https://github.com/google/fonts/tree/main/ofl/golostext",
        ),
        _curated(
            "font.golos-text.bold", "Golos Text", "bold",
            "https://github.com/google/fonts/tree/main/ofl/golostext",
        ),
        _curated(
            "font.inter.regular", "Inter", "normal",
            "https://github.com/google/fonts/tree/main/ofl/inter",
        ),
        _curated(
            "font.inter.bold", "Inter", "bold",
            "https://github.com/google/fonts/tree/main/ofl/inter",
        ),
        _curated(
            "font.pt-sans-narrow.regular", "PT Sans Narrow", "normal",
            "https://github.com/google/fonts/tree/main/ofl/ptsansnarrow",
        ),
        _curated(
            "font.pt-sans-narrow.bold", "PT Sans Narrow", "bold",
            "https://github.com/google/fonts/tree/main/ofl/ptsansnarrow",
        ),
    )
}

FONT_ASSET_VERSIONS: dict[tuple[str, str], FontAssetDefinition] = {
    (item.asset_id, item.asset_version): item for item in FONT_ASSET_DEFINITIONS.values()
}
CURRENT_FONT_ASSET_VERSIONS: dict[str, str] = {
    item.asset_id: item.asset_version for item in FONT_ASSET_DEFINITIONS.values()
}

_FACE_INDEX = {
    (item.family.casefold(), item.style, item.weight): item
    for item in FONT_ASSET_DEFINITIONS.values()
}


def font_asset_definition(
    asset_id: str,
    asset_version: str | None = None,
) -> FontAssetDefinition:
    version = asset_version or CURRENT_FONT_ASSET_VERSIONS[asset_id]
    return FONT_ASSET_VERSIONS[(asset_id, version)]


def curated_font_asset_for_face(
    family: str,
    *,
    style: FontStyle = "normal",
    weight: FontWeight = "bold",
) -> FontAssetDefinition | None:
    return _FACE_INDEX.get((family.casefold(), style, weight))


def resolved_font_asset_id(
    family: str,
    *,
    style: FontStyle = "normal",
    weight: FontWeight = "bold",
) -> str:
    """Return a stable face ID; file SHA-256 supplies the exact version."""

    curated = curated_font_asset_for_face(family, style=style, weight=weight)
    if curated is not None:
        return curated.asset_id
    slug = re.sub(r"[^a-z0-9]+", "-", family.casefold()).strip("-") or "unknown"
    return f"font.system.{slug}.{style}.{weight}"


def system_font_license_metadata(asset_id: str, family: str) -> FontAssetDefinition:
    """Describe a host-provided font without granting redistribution rights."""

    return FontAssetDefinition(
        asset_id=asset_id,
        asset_version="system-file-checksum",
        family=family,
        style="normal",
        weight="bold" if asset_id.endswith(".bold") else "normal",
        license_id="LicenseRef-SystemFont",
        license_notice_required=False,
        source_url="system-font-registry",
        supported_scripts=("latin", "cyrillic"),
        redistribution_allowed=False,
    )
