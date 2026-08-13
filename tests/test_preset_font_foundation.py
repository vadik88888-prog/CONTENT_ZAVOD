from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

from app.caption_planning import (
    _font_registry_rank,
    materialize_caption_font_directory,
    resolve_caption_font_manifest,
)
from app.caption_presets import CAPTION_PRESET_DEFINITIONS, CAPTION_PRESET_VERSIONS
from app.creative_policy import (
    CREATIVE_PRESET_DEFINITIONS,
    CREATIVE_PRESET_VERSIONS,
    PRESET_FAMILY_POLICIES,
)
from app.font_assets import FONT_ASSET_DEFINITIONS, FONT_ASSET_VERSIONS, font_asset_definition
from app.video_composition import _ass_filter


def test_creative_presets_are_versioned_complete_legacy_id_definitions() -> None:
    assert CREATIVE_PRESET_DEFINITIONS is PRESET_FAMILY_POLICIES
    assert set(CREATIVE_PRESET_DEFINITIONS) == {"minimal", "documentary", "dynamic", "clean"}
    assert {item.preset_version for item in CREATIVE_PRESET_DEFINITIONS.values()} == {"1.0.0"}
    assert len(CREATIVE_PRESET_VERSIONS) == len(CREATIVE_PRESET_DEFINITIONS)
    assert {
        item.caption_preset_id for item in CREATIVE_PRESET_DEFINITIONS.values()
    } == {"minimal_light", "editorial_narrow", "accent_yellow", "clean_white"}
    assert all(
        item.source_extra_shots_default is False
        and item.stock_broll_allowed is False
        and item.generative_broll_allowed is False
        for item in CREATIVE_PRESET_DEFINITIONS.values()
    )


def test_caption_presets_have_stable_versioned_tokens_and_valid_font_assets() -> None:
    assert set(CAPTION_PRESET_DEFINITIONS) == {
        "clean_white", "minimal_light", "accent_yellow", "karaoke_yellow",
        "editorial_narrow", "contrast_box",
    }
    tokens = {item.token_id for item in CAPTION_PRESET_DEFINITIONS.values()}
    assert len(tokens) == len(CAPTION_PRESET_DEFINITIONS)
    assert len(CAPTION_PRESET_VERSIONS) == len(CAPTION_PRESET_DEFINITIONS)
    assert all(token.endswith(":1.0.0") for token in tokens)
    assert all(
        item.preferred_font_asset_id in FONT_ASSET_DEFINITIONS
        for item in CAPTION_PRESET_DEFINITIONS.values()
    )


def test_curated_font_registry_separates_regular_bold_and_records_license() -> None:
    assert len(FONT_ASSET_DEFINITIONS) == 6
    assert len(FONT_ASSET_VERSIONS) == len(FONT_ASSET_DEFINITIONS)
    for family in ("Golos Text", "Inter", "PT Sans Narrow"):
        faces = [item for item in FONT_ASSET_DEFINITIONS.values() if item.family == family]
        assert {item.weight for item in faces} == {"normal", "bold"}
        assert len({item.asset_id for item in faces}) == 2
        assert all(item.license_id == "OFL-1.1" for item in faces)
        assert all(item.license_notice_required and item.redistribution_allowed for item in faces)
        assert all(item.supported_scripts == ("latin", "cyrillic") for item in faces)
        assert all(font_asset_definition(item.asset_id) is item for item in faces)


def test_windows_registry_rank_selects_the_requested_exact_weight() -> None:
    regular = "Arial (TrueType)"
    bold = "Arial Bold (TrueType)"
    assert _font_registry_rank(bold, "arial", "bold") < _font_registry_rank(regular, "arial", "bold")
    assert _font_registry_rank(regular, "arial", "normal") < _font_registry_rank(bold, "arial", "normal")


def test_manifest_and_controlled_fontsdir_preserve_exact_face_bytes(tmp_path: Path) -> None:
    regular = resolve_caption_font_manifest("Arial", weight="normal")
    bold = resolve_caption_font_manifest("Arial", weight="bold")
    assert regular.font_id != bold.font_id
    if os.name == "nt":
        assert regular.file_sha256 and bold.file_sha256
        assert regular.file_name != bold.file_name
        assert regular.file_sha256 != bold.file_sha256
    if bold.file_sha256 is None:
        return

    controlled = materialize_caption_font_directory(bold, tmp_path / "controlled-fonts")
    files = list(controlled.iterdir())
    assert len(files) == 1
    assert sha256(files[0].read_bytes()).hexdigest() == bold.file_sha256
    assert bold.font_id in files[0].name


def test_native_ass_filter_uses_explicit_controlled_fontsdir(tmp_path: Path) -> None:
    value = _ass_filter(tmp_path / "captions.ass", tmp_path / "fonts")
    assert value.startswith("ass='")
    assert ":fontsdir='" in value
    assert "captions.ass" in value and "fonts" in value
