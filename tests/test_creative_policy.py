from __future__ import annotations

from app.creative_contracts import Intensity
from app.creative_policy import (
    CREATIVE_POLICY_VERSION,
    PRESET_FAMILY_POLICIES,
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


def test_explicit_user_preset_always_wins_over_content_recommendation() -> None:
    assert resolve_preset_family(
        user_choice="minimal", content_type="podcast / talking head",
    ) == "minimal"
