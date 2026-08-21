from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from app.config import AppConfig
from app.content_profile_taxonomy import (
    AUTO_PROFILE_INPUT,
    CONTENT_PROFILE_PRESETS,
    CONTENT_PROFILE_SCHEMA_VERSION,
    LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS,
    PROFILE_AXIS_ORDER,
    PROFILE_TAXONOMY,
    SUPPORTED_CONTENT_PROFILE_SCHEMA_VERSIONS,
    UNKNOWN_PROFILE_ID,
    content_profile_preset_ids,
    content_profile_preset_mapping,
    order_profile_ids,
    profile_input_ids,
    profile_value_ids,
    unknown_fallback,
    user_override_ids,
)
from app.content_understanding import (
    PROFILE_DOMAINS,
    PROFILE_EDITORIAL_MODES,
    PROFILE_FORMATS,
    PROFILE_TRAITS,
    VIDEO_CONTENT_PROFILE_SCHEMA_VERSION,
)
from app.errors import ClipEngineError
from app.product_flow import (
    PROFILE_DOMAIN_OVERRIDES,
    PROFILE_EDITORIAL_MODE_OVERRIDES,
    PROFILE_FORMAT_OVERRIDES,
    PROFILE_TRAIT_OVERRIDES,
    ProcessingIntent,
)


def test_taxonomy_is_dependency_free_and_owns_schema_axis_order_and_fallbacks() -> None:
    source_path = Path(__file__).resolve().parents[1] / "app" / "content_profile_taxonomy.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module != "__future__"
    }

    assert imported_roots <= {"dataclasses", "types", "typing"}
    assert CONTENT_PROFILE_SCHEMA_VERSION == VIDEO_CONTENT_PROFILE_SCHEMA_VERSION == "5A.3"
    assert LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS == ("5A.1", "5A.2")
    assert SUPPORTED_CONTENT_PROFILE_SCHEMA_VERSIONS == {"5A.1", "5A.2", "5A.3"}
    assert tuple(PROFILE_TAXONOMY) == PROFILE_AXIS_ORDER
    assert unknown_fallback("format") == UNKNOWN_PROFILE_ID
    assert unknown_fallback("editorial_mode") == UNKNOWN_PROFILE_ID
    assert unknown_fallback("domain") == UNKNOWN_PROFILE_ID
    assert unknown_fallback("traits") == ()


def test_auto_is_input_only_and_unknown_is_not_user_overridable() -> None:
    for axis_id in PROFILE_AXIS_ORDER:
        assert AUTO_PROFILE_INPUT not in profile_value_ids(axis_id)
        assert profile_input_ids(axis_id)[0] == AUTO_PROFILE_INPUT
    for axis_id in PROFILE_AXIS_ORDER[:-1]:
        assert UNKNOWN_PROFILE_ID in profile_value_ids(axis_id)
        assert UNKNOWN_PROFILE_ID not in user_override_ids(axis_id)

    with pytest.raises(ValueError, match="format override"):
        ProcessingIntent(profile_format_override=UNKNOWN_PROFILE_ID).validate()

    reversed_traits = list(reversed(user_override_ids("traits")))
    assert order_profile_ids("traits", reversed_traits) == user_override_ids("traits")
    assert ProcessingIntent(profile_traits_override=tuple(reversed_traits)).to_dict()["profile_traits_override"] == list(
        user_override_ids("traits")
    )


EXPECTED_CONTENT_PRESET_MAPPINGS = {
    "podcast": {"format": "dialogue", "editorial_mode": "commentary", "domain": "general", "traits": ["speech_led", "multi_speaker", "low_pacing"]},
    "interview": {"format": "dialogue", "editorial_mode": "interview", "domain": "general", "traits": ["speech_led", "multi_speaker", "question_answer"]},
    "talking_head_expert": {"format": "talking_head", "editorial_mode": "explanatory", "domain": "education", "traits": ["speech_led", "single_speaker", "dense_information"]},
    "gameplay": {"format": "gameplay", "editorial_mode": "commentary", "domain": "gaming", "traits": ["visual_led", "high_pacing", "scene_driven"]},
    "stream": {"format": "mixed", "editorial_mode": "commentary", "domain": "entertainment", "traits": ["speech_led", "visual_led", "high_pacing"]},
    "vlog_lifestyle": {"format": "scene_driven", "editorial_mode": "narrative", "domain": "lifestyle", "traits": ["visual_led", "scene_driven"]},
    "food": {"format": "scene_driven", "editorial_mode": "demonstration", "domain": "food", "traits": ["visual_led", "scene_driven", "instructional"]},
    "travel": {"format": "scene_driven", "editorial_mode": "narrative", "domain": "lifestyle", "traits": ["visual_led", "scene_driven"]},
    "tutorial_education": {"format": "screen_demo", "editorial_mode": "demonstration", "domain": "education", "traits": ["speech_led", "visual_led", "dense_information", "screen_content", "instructional"]},
    "review": {"format": "mixed", "editorial_mode": "commentary", "domain": "general", "traits": ["speech_led", "visual_led", "dense_information"]},
    "reaction": {"format": "mixed", "editorial_mode": "commentary", "domain": "entertainment", "traits": ["speech_led", "visual_led", "high_emotion"]},
    "story_entertainment": {"format": "scene_driven", "editorial_mode": "narrative", "domain": "entertainment", "traits": ["visual_led", "high_emotion", "scene_driven"]},
    "movie_series": {"format": "scene_driven", "editorial_mode": "entertainment", "domain": "entertainment", "traits": ["visual_led", "scene_driven"]},
    "sports_fitness": {"format": "scene_driven", "editorial_mode": "demonstration", "domain": "health", "traits": ["visual_led", "high_pacing", "scene_driven", "instructional"]},
    "news_commentary": {"format": "talking_head", "editorial_mode": "news_analysis", "domain": "news", "traits": ["speech_led", "single_speaker", "dense_information"]},
}


def test_all_15_user_facing_presets_have_stable_valid_deterministic_mappings() -> None:
    assert content_profile_preset_ids(include_auto=True) == (AUTO_PROFILE_INPUT, *EXPECTED_CONTENT_PRESET_MAPPINGS)
    assert tuple(CONTENT_PROFILE_PRESETS) == tuple(EXPECTED_CONTENT_PRESET_MAPPINGS)
    assert len({preset.label for preset in CONTENT_PROFILE_PRESETS.values()}) == 15

    for preset_id, expected in EXPECTED_CONTENT_PRESET_MAPPINGS.items():
        assert content_profile_preset_mapping(preset_id) == expected
        assert expected["format"] in profile_value_ids("format")
        assert expected["editorial_mode"] in profile_value_ids("editorial_mode")
        assert expected["domain"] in profile_value_ids("domain")
        assert all(trait in profile_value_ids("traits") for trait in expected["traits"])


def test_schema_config_product_and_profile_validation_cannot_drift_from_registry() -> None:
    assert PROFILE_FORMATS == frozenset(profile_value_ids("format"))
    assert PROFILE_EDITORIAL_MODES == frozenset(profile_value_ids("editorial_mode"))
    assert PROFILE_DOMAINS == frozenset(profile_value_ids("domain"))
    assert PROFILE_TRAITS == frozenset(profile_value_ids("traits"))
    assert PROFILE_FORMAT_OVERRIDES == frozenset(profile_input_ids("format"))
    assert PROFILE_EDITORIAL_MODE_OVERRIDES == frozenset(profile_input_ids("editorial_mode"))
    assert PROFILE_DOMAIN_OVERRIDES == frozenset(profile_input_ids("domain"))
    assert PROFILE_TRAIT_OVERRIDES == frozenset(user_override_ids("traits"))

    for axis_id in PROFILE_AXIS_ORDER[:-1]:
        for value_id in user_override_ids(axis_id):
            intent_kwargs = {f"profile_{axis_id}_override": value_id}
            ProcessingIntent(**intent_kwargs).validate()
            config = AppConfig()
            config.content_understanding.manual_override = {axis_id: value_id}
            config.validate()
    for value_id in user_override_ids("traits"):
        ProcessingIntent(profile_traits_override=(value_id,)).validate()
        config = AppConfig()
        config.content_understanding.manual_override = {"traits": [value_id]}
        config.validate()


def test_profile_consumers_do_not_redeclare_axis_order_or_auto_sentinel() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    consumer_paths = (
        repository_root / "app" / "config.py",
        repository_root / "app" / "content_understanding.py",
        repository_root / "app" / "product_flow.py",
        repository_root / "app" / "gui" / "screens" / "project_screen.py",
    )

    repeated_axis_orders: list[str] = []
    for source_path in consumer_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            values = tuple(
                item.value for item in node.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
            if values == PROFILE_AXIS_ORDER:
                repeated_axis_orders.append(str(source_path.relative_to(repository_root)))

    assert repeated_axis_orders == []

def test_config_preserves_legacy_schema_compatibility() -> None:
    current = AppConfig()
    assert current.content_understanding.profile_schema_version == CONTENT_PROFILE_SCHEMA_VERSION
    current.validate()

    legacy = AppConfig()
    legacy.content_understanding.profile_schema_version = LEGACY_CONTENT_PROFILE_SCHEMA_VERSIONS[0]
    legacy.validate()

    invalid = AppConfig()
    invalid.content_understanding.profile_schema_version = "5A.0"
    with pytest.raises(ClipEngineError, match="profile_schema_version"):
        invalid.validate()


def test_real_qt_profile_controls_follow_registry_order() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QComboBox

    from app.gui.screens.project_screen import _populate_content_profile_preset, _populate_profile_override

    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires a QApplication process")
    app = QApplication.instance() or QApplication([])

    controls: list[QComboBox] = []
    try:
        for axis_id in PROFILE_AXIS_ORDER:
            combo = QComboBox()
            _populate_profile_override(combo, axis_id)
            combo.show()
            app.processEvents()
            controls.append(combo)
            assert combo.isVisible()
            assert tuple(combo.itemData(index) for index in range(combo.count())) == profile_input_ids(axis_id)
        preset_combo = QComboBox()
        _populate_content_profile_preset(preset_combo)
        preset_combo.show()
        app.processEvents()
        controls.append(preset_combo)
        assert preset_combo.isVisible()
        assert tuple(preset_combo.itemData(index) for index in range(preset_combo.count())) == content_profile_preset_ids(include_auto=True)
    finally:
        for combo in controls:
            combo.close()
        app.processEvents()
