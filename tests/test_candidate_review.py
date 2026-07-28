from __future__ import annotations

from app.candidate_review import validate_boundary_override


def test_boundary_override_is_checked_against_cached_duration_and_speech_context() -> None:
    result = validate_boundary_override(
        10.5, 31.0, source_duration=60.0, minimum_duration=15.0, maximum_duration=60.0,
        transcript_features={"segments": [{"start": 10.0, "end": 31.0}]},
        scenes={"boundaries": [{"timestamp": 10.5}]},
    )

    assert result["valid"] is True
    assert result["revalidation"] == "cached_transcript_and_scene_only"
    assert any("монтажным cut" in warning for warning in result["warnings"])


def test_boundary_override_rejects_invalid_short_or_reversed_ranges() -> None:
    result = validate_boundary_override(
        20.0, 18.0, source_duration=60.0, minimum_duration=15.0, maximum_duration=60.0,
    )

    assert result["valid"] is False
    assert result["errors"]
