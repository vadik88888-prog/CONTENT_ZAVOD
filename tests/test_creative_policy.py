from __future__ import annotations

from app.creative_contracts import Intensity
from app.creative_policy import (
    CREATIVE_POLICY_VERSION,
    PRESET_FAMILY_POLICIES,
    creative_profile_signal,
    recommend_preset_family,
    resolve_preset_family,
)


def test_existing_four_preset_families_have_versioned_internal_ceiling_policies() -> None:
    assert set(PRESET_FAMILY_POLICIES) == {"minimal", "documentary", "dynamic", "clean"}
    assert {policy.policy_version for policy in PRESET_FAMILY_POLICIES.values()} == {
        CREATIVE_POLICY_VERSION,
    }
    assert PRESET_FAMILY_POLICIES["minimal"].intensity_ceiling == Intensity.LOW
    assert PRESET_FAMILY_POLICIES["documentary"].intensity_ceiling == Intensity.BALANCED
    assert PRESET_FAMILY_POLICIES["dynamic"].intensity_ceiling == Intensity.HIGH
    assert PRESET_FAMILY_POLICIES["clean"].intensity_ceiling == Intensity.LOW
    assert all(
        policy.source_extra_shots_default is False
        for policy in PRESET_FAMILY_POLICIES.values()
    )


def test_content_recommendations_cover_calibration_fixtures() -> None:
    assert recommend_preset_family("podcast / talking head") == "clean"
    assert recommend_preset_family("interview") == "documentary"
    assert recommend_preset_family("vlog / food / travel") == "dynamic"
    assert recommend_preset_family("visual-heavy / gameplay") == "minimal"


def test_creative_profile_adapter_prefers_structured_effective_profile() -> None:
    profile = {
        "detected_content_type": "podcast",
        "dominant_format": "single_speaker_monologue",
        "detected_profile": {
            "format": {"value": "talking_head"},
            "editorial_mode": {"value": "explanatory"},
            "domain": {"value": "education"},
        },
        "effective_profile": {
            "format": "gameplay",
            "editorial_mode": "commentary",
            "domain": "gaming",
            "traits": ["visual_led", "high_pacing"],
        },
    }

    assert creative_profile_signal(profile) == "gameplay commentary gaming"
    assert recommend_preset_family(profile) == "minimal"


def test_creative_auto_mapping_ignores_traits_until_an_approved_mapping_exists() -> None:
    base = {
        "effective_profile": {
            "format": "talking_head",
            "editorial_mode": "commentary",
            "domain": "general",
            "traits": [],
        },
    }
    trait_only_change = {
        "effective_profile": {
            **base["effective_profile"],
            "traits": ["scene_driven", "high_pacing", "visual_led"],
        },
    }

    assert creative_profile_signal(base) == creative_profile_signal(trait_only_change)
    assert recommend_preset_family(base) == recommend_preset_family(trait_only_change) == "clean"


def test_explicit_user_preset_always_wins_over_content_recommendation() -> None:
    assert resolve_preset_family(
        user_choice="minimal", content_type="podcast / talking head",
    ) == "minimal"
