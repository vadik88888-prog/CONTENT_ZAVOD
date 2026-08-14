from __future__ import annotations

"""Exact, redistributable caption-font assets used by Preview and Final.

The registry is the existing Caption System font authority.  Every production
entry names one bundled static face and pins its local bytes by SHA-256.  Host
fonts remain representable for legacy manifests, but approved caption presets
never resolve through them.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal


FONT_ASSET_REGISTRY_VERSION = "production.caption-fonts.2"

FontWeight = Literal["normal", "bold"]
FontStyle = Literal["normal", "italic"]


@dataclass(frozen=True, slots=True)
class FontAssetDefinition:
    asset_id: str
    asset_version: str
    family: str
    render_family: str
    subfamily: str
    postscript_name: str
    style: FontStyle
    weight: FontWeight
    weight_class: int
    file_name: str
    file_sha256: str
    license_id: str
    license_file: str
    license_sha256: str
    license_notice_required: bool
    source_url: str
    supported_scripts: tuple[Literal["latin", "cyrillic"], ...]
    redistribution_allowed: bool


def _bundled(
    asset_id: str,
    family: str,
    subfamily: str,
    postscript_name: str,
    weight: FontWeight,
    weight_class: int,
    file_name: str,
    file_sha256: str,
    license_file: str,
    license_sha256: str,
    source_url: str,
    render_family: str | None = None,
) -> FontAssetDefinition:
    return FontAssetDefinition(
        asset_id=asset_id,
        asset_version="production.1",
        family=family,
        render_family=render_family or family,
        subfamily=subfamily,
        postscript_name=postscript_name,
        style="normal",
        weight=weight,
        weight_class=weight_class,
        file_name=file_name,
        file_sha256=file_sha256,
        license_id="OFL-1.1",
        license_file=f"licenses/{license_file}",
        license_sha256=license_sha256,
        license_notice_required=True,
        source_url=source_url,
        supported_scripts=("latin", "cyrillic"),
        redistribution_allowed=True,
    )


FONT_ASSET_DEFINITIONS: dict[str, FontAssetDefinition] = {
    item.asset_id: item
    for item in (
        _bundled(
            "font.manrope.bold", "Manrope", "Bold", "Manrope-Bold", "bold", 700,
            "Manrope-Bold.ttf",
            "4aed5d180a4f41ed21f07e678486f889bb40eb0ddf5f473769b6302f507d1e36",
            "Manrope-OFL-1.1.txt",
            "58d49f25b2cacdfe83739d557ac9319c48bf3ed3e9e33b6678ddb972b475ce7c",
            "https://github.com/google/fonts/tree/main/ofl/manrope",
        ),
        _bundled(
            "font.commissioner.light", "Commissioner", "Light", "Commissioner-Light", "normal", 300,
            "Commissioner-Light.ttf",
            "fadc0f71d5e6a89af6200d5ff46b81d5b5c7bf6663da3031093d4a19e50124ee",
            "Commissioner-OFL-1.1.txt",
            "d1dceb754629c9cf264e80560014c1de31ef2b6540c682f3b247d4b9914b4ef9",
            "https://github.com/google/fonts/tree/main/ofl/commissioner",
            render_family="Commissioner Light",
        ),
        _bundled(
            "font.oswald.bold", "Oswald", "Bold", "Oswald-Bold", "bold", 700,
            "Oswald-Bold.ttf",
            "21843a0ccca2e97030da1330961646510ed6e9b8bb39620974c1195b9803034a",
            "Oswald-OFL-1.1.txt",
            "f55dcf1905ca45acef56f9e41183c20e66c12feba0ef137727358fe58f0dddc2",
            "https://github.com/google/fonts/tree/main/ofl/oswald",
        ),
        _bundled(
            "font.pt-sans-narrow.regular", "PT Sans Narrow", "Regular", "PTSans-Narrow", "normal", 400,
            "PTSansNarrow-Regular.ttf",
            "d4882d4b26690f2951c476fd2b3bb4abfb20f3763a2cc4de04f30466e923baeb",
            "PTSansNarrow-OFL-1.1.txt",
            "0f13c2ec3dcb2abf44bbc913526d47bd0dfc7d3e711b9bb79103716fcb89619b",
            "https://github.com/google/fonts/tree/main/ofl/ptsansnarrow",
        ),
        _bundled(
            "font.pt-sans-narrow.bold", "PT Sans Narrow", "Bold", "PTSans-NarrowBold", "bold", 700,
            "PTSansNarrow-Bold.ttf",
            "d94bed3e84c75c97c0d477840896f7bf63c3b341540228be98edd1a6e98a6a0e",
            "PTSansNarrow-OFL-1.1.txt",
            "0f13c2ec3dcb2abf44bbc913526d47bd0dfc7d3e711b9bb79103716fcb89619b",
            "https://github.com/google/fonts/tree/main/ofl/ptsansnarrow",
        ),
        _bundled(
            "font.golos-text.bold", "Golos Text", "Bold", "GolosText-Bold", "bold", 700,
            "GolosText-Bold.ttf",
            "cac7fc9fd8c577f0f2ed0119e01a4c1169c3142158b8bfb0b6ecf6cc6eb97cad",
            "GolosText-OFL-1.1.txt",
            "fb10e75607c8f3835ca3566d160112d7092d21cc00f3f9521b1f092e8548270b",
            "https://github.com/google/fonts/tree/main/ofl/golostext",
        ),
        _bundled(
            "font.rubik.semibold", "Rubik", "SemiBold", "Rubik-SemiBold", "bold", 600,
            "Rubik-SemiBold.ttf",
            "3645e8b65efcdd3bccde6afeb1e482fbdc34c1a810ee2f19fb9e6e8837744938",
            "Rubik-OFL-1.1.txt",
            "fbf425c49bf998fb9cc1b52f8ca27d5f84dec079b60c40f654f242b52474d536",
            "https://github.com/google/fonts/tree/main/ofl/rubik",
            render_family="Rubik SemiBold",
        ),
        _bundled(
            "font.unbounded.bold", "Unbounded", "Bold", "Unbounded-Bold", "bold", 700,
            "Unbounded-Bold.ttf",
            "a9f663beb8f7e99e94e61a775c742486a09f53fe7f90e1af7997ee317b199f66",
            "Unbounded-OFL-1.1.txt",
            "8f34e6e20f569b9ca4cd645e30726cf373c0c937f5cab5bd437eae7da23adda6",
            "https://github.com/google/fonts/tree/main/ofl/unbounded",
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


def bundled_font_asset_path(definition: FontAssetDefinition) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "fonts" / definition.file_name


def bundled_font_license_path(definition: FontAssetDefinition) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "fonts" / definition.license_file


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
        render_family=family,
        subfamily="Bold" if asset_id.endswith(".bold") else "Regular",
        postscript_name=family.replace(" ", ""),
        style="normal",
        weight="bold" if asset_id.endswith(".bold") else "normal",
        weight_class=700 if asset_id.endswith(".bold") else 400,
        file_name="system-font",
        file_sha256="0" * 64,
        license_id="LicenseRef-SystemFont",
        license_file="system-font-registry",
        license_sha256="0" * 64,
        license_notice_required=False,
        source_url="system-font-registry",
        supported_scripts=("latin", "cyrillic"),
        redistribution_allowed=False,
    )
