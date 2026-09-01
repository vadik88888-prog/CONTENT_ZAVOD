from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import struct

import pytest

from app.caption_planning import (
    _resolve_font_manifest,
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
from app.font_assets import (
    FONT_ASSET_DEFINITIONS,
    FONT_ASSET_VERSIONS,
    bundled_font_asset_path,
    bundled_font_license_path,
    font_asset_definition,
)
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
        "editorial_narrow", "contrast_box", "word_pop",
    }
    tokens = {item.token_id for item in CAPTION_PRESET_DEFINITIONS.values()}
    assert len(tokens) == len(CAPTION_PRESET_DEFINITIONS)
    assert len(CAPTION_PRESET_VERSIONS) == 19
    assert {
        item.preset_id: item.preset_version
        for item in CAPTION_PRESET_DEFINITIONS.values()
    } == {
        "clean_white": "2.2.0",
        "minimal_light": "2.2.0",
        "accent_yellow": "2.2.0",
        "editorial_narrow": "2.2.0",
        "karaoke_yellow": "2.2.0",
        "contrast_box": "2.2.0",
        "word_pop": "2.1.0",
    }
    assert all(
        set(item.font_asset_ids).issubset(FONT_ASSET_DEFINITIONS)
        for item in CAPTION_PRESET_DEFINITIONS.values()
    )
    assert CAPTION_PRESET_DEFINITIONS["word_pop"].highlight_color == "#C6FF00"
    assert CAPTION_PRESET_DEFINITIONS["word_pop"].pop_scale_keyframes == (88, 112, 100)
    assert CAPTION_PRESET_DEFINITIONS["word_pop"].semantic_pop_scale_keyframes == (84, 118, 100)


def test_final_visual_calibration_changes_only_the_four_approved_policies() -> None:
    minimal = CAPTION_PRESET_DEFINITIONS["minimal_light"]
    impact = CAPTION_PRESET_DEFINITIONS["accent_yellow"]
    editorial = CAPTION_PRESET_DEFINITIONS["editorial_narrow"]
    word_pop = CAPTION_PRESET_DEFINITIONS["word_pop"]

    assert (minimal.font_size_ratio, minimal.outline_width_ratio, minimal.shadow_ratio) == (
        0.029, 0.0, 0.0,
    )
    assert minimal.preferred_font_asset_id == "font.commissioner.light"
    assert minimal.font_weight == "normal"
    assert impact.font_size_ratio == 0.043
    assert impact.preferred_font_asset_id == "font.oswald.bold"
    assert (impact.text_color, impact.highlight_color) == ("#FFFFFF", "#FFD54A")
    assert (editorial.font_size_ratio, editorial.outline_width_ratio) == (0.036, 0.0)
    assert editorial.font_asset_ids == (
        "font.pt-sans-narrow.regular", "font.pt-sans-narrow.bold",
    )
    assert editorial.semantic_bold
    assert word_pop.font_size_ratio == 0.049
    assert word_pop.preferred_font_asset_id == "font.unbounded.bold"
    assert (word_pop.text_color, word_pop.highlight_color) == ("#FFFFFF", "#C6FF00")
    assert word_pop.display_mode == "single_spoken_word"
    assert word_pop.pop_scale_keyframes == (88, 112, 100)
    assert word_pop.semantic_pop_scale_keyframes == (84, 118, 100)


def test_current_caption_presets_keep_seven_visually_distinct_identities() -> None:
    presets = CAPTION_PRESET_DEFINITIONS

    assert len({item.preferred_font_asset_id for item in presets.values()}) == 7
    assert presets["clean_white"].motion_profile_id == "semantic_fade"
    assert presets["minimal_light"].allowed_primitives == ("static",)
    assert presets["minimal_light"].font_size_ratio < presets["clean_white"].font_size_ratio
    assert presets["accent_yellow"].uppercase_emphasis is True
    assert presets["accent_yellow"].highlight_color == "#FFD54A"
    assert presets["editorial_narrow"].semantic_bold is True
    assert presets["editorial_narrow"].background_color == "#17212B"
    assert presets["karaoke_yellow"].allowed_primitives == ("static", "karaoke")
    assert presets["contrast_box"].background_mode == "opaque_box"
    assert presets["word_pop"].display_mode == "single_spoken_word"


def test_locked_caption_policies_remain_byte_stable() -> None:
    expected = {
        "clean_white": "c09a3ec772466b4ed7023dd1672ae68b7b474a21baca75cfc6fb0f30925366ee",
        "karaoke_yellow": "87ac65be1c63610b8af8a0b04b1b9bf716fbd30aa2a757bf57dd2e8b8d96595a",
        "contrast_box": "39bec9e5e3eb6d87e28e29f2d45dacbc7d75b3148a2b16d08afbff1346d9dc0d",
    }
    for preset_id, digest in expected.items():
        payload = json.dumps(
            asdict(CAPTION_PRESET_DEFINITIONS[preset_id]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        assert sha256(payload).hexdigest() == digest


def test_curated_font_registry_separates_regular_bold_and_records_license() -> None:
    assert len(FONT_ASSET_DEFINITIONS) == 8
    assert len(FONT_ASSET_VERSIONS) == len(FONT_ASSET_DEFINITIONS)
    assert {item.weight_class for item in FONT_ASSET_DEFINITIONS.values()} == {300, 400, 600, 700}
    for item in FONT_ASSET_DEFINITIONS.values():
        assert item.license_id == "OFL-1.1"
        assert item.license_notice_required and item.redistribution_allowed
        assert item.supported_scripts == ("latin", "cyrillic")
        assert font_asset_definition(item.asset_id) is item
        font_path = bundled_font_asset_path(item)
        license_path = bundled_font_license_path(item)
        assert font_path.name == item.file_name and font_path.is_file()
        assert sha256(font_path.read_bytes()).hexdigest() == item.file_sha256
        assert license_path.is_file()
        assert sha256(license_path.read_bytes()).hexdigest() == item.license_sha256
        assert "SIL OPEN FONT LICENSE Version 1.1" in license_path.read_text(encoding="utf-8")


def _font_tables(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    count = struct.unpack_from(">H", data, 4)[0]
    return {
        tag.decode("ascii"): data[offset:offset + length]
        for tag, _checksum, offset, length in (
            struct.unpack_from(">4sIII", data, 12 + index * 16)
            for index in range(count)
        )
    }


def _font_names(table: bytes, *name_ids: int) -> set[str]:
    _format, count, string_offset = struct.unpack_from(">HHH", table, 0)
    result: set[str] = set()
    for index in range(count):
        platform, _encoding, _language, name_id, length, offset = struct.unpack_from(
            ">HHHHHH", table, 6 + index * 12
        )
        if name_id not in name_ids:
            continue
        raw = table[string_offset + offset:string_offset + offset + length]
        result.add(raw.decode("utf-16-be" if platform in {0, 3} else "mac_roman").strip())
    return result


def _has_codepoint(cmap: bytes, codepoint: int) -> bool:
    count = struct.unpack_from(">H", cmap, 2)[0]
    offsets = {struct.unpack_from(">I", cmap, 8 + index * 8)[0] for index in range(count)}
    for offset in offsets:
        table = cmap[offset:]
        fmt = struct.unpack_from(">H", table, 0)[0]
        if fmt == 4 and codepoint <= 0xFFFF:
            segments = struct.unpack_from(">H", table, 6)[0] // 2
            end_base = 14
            start_base = end_base + segments * 2 + 2
            delta_base = start_base + segments * 2
            range_base = delta_base + segments * 2
            for index in range(segments):
                start = struct.unpack_from(">H", table, start_base + index * 2)[0]
                end = struct.unpack_from(">H", table, end_base + index * 2)[0]
                if start <= codepoint <= end:
                    delta = struct.unpack_from(">h", table, delta_base + index * 2)[0]
                    range_offset = struct.unpack_from(">H", table, range_base + index * 2)[0]
                    if range_offset == 0:
                        return ((codepoint + delta) & 0xFFFF) != 0
                    address = range_base + index * 2 + range_offset + (codepoint - start) * 2
                    glyph = struct.unpack_from(">H", table, address)[0]
                    return glyph != 0 and ((glyph + delta) & 0xFFFF) != 0
        if fmt == 12:
            groups = struct.unpack_from(">I", table, 12)[0]
            for index in range(groups):
                start, end, glyph = struct.unpack_from(">III", table, 16 + index * 12)
                if start <= codepoint <= end:
                    return glyph + codepoint - start != 0
    return False


def test_bundled_font_name_weight_and_required_glyph_tables_match_registry() -> None:
    for item in FONT_ASSET_DEFINITIONS.values():
        tables = _font_tables(bundled_font_asset_path(item))
        assert item.family in _font_names(tables["name"], 1, 16)
        assert item.subfamily in _font_names(tables["name"], 2, 17)
        assert item.postscript_name in _font_names(tables["name"], 6)
        assert struct.unpack_from(">H", tables["OS/2"], 4)[0] == item.weight_class
        assert all(_has_codepoint(tables["cmap"], ord(character)) for character in "AzАяЁё")


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


def test_approved_bundled_font_checksum_mismatch_fails_closed(tmp_path: Path, monkeypatch) -> None:
    damaged = tmp_path / "Manrope-Bold.ttf"
    damaged.write_bytes(b"not-a-font")
    monkeypatch.setattr("app.caption_planning.bundled_font_asset_path", lambda _item: damaged)

    with pytest.raises(ValueError, match="CAPTION_BUNDLED_FONT_CHECKSUM_MISMATCH"):
        _resolve_font_manifest(
            "Manrope",
            weight="bold",
            supplied=None,
            preferred_asset_id="font.manrope.bold",
            companion_asset_ids=(),
        )
