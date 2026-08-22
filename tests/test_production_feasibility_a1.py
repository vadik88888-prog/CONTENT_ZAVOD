from __future__ import annotations

from app.production_feasibility import (
    A1_POLICY_VERSION,
    PRODUCTION_FEASIBILITY_POLICY_VERSION,
    _a1_gate_result,
)


def _plan(*, confidence: float, duration: float) -> dict:
    return {
        "dialogue_mappings": [{
            "segment_id": "dialogue-1",
            "confidence": 0.99,
            "source_start_seconds": 0.0,
            "source_end_seconds": 10.0,
            "evidence_mappings": [{
                "fact_id": "fact-1",
                "transcript_segment_id": 1,
                "confidence": confidence,
                "source_start_seconds": 3.0,
                "source_end_seconds": 3.0 + duration,
            }],
        }],
    }


def test_a1_records_isolated_low_confidence_speech_without_guaranteed_block() -> None:
    a1 = _a1_gate_result(_plan(confidence=0.024, duration=0.40))

    assert a1["status"] == "PASS"
    assert a1["reason_code"] == "A1_DIALOGUE_CONFIDENCE_RISK"
    assert a1["evidence"]["severity"] == "warning"
    assert a1["policy_version"] == A1_POLICY_VERSION
    assert PRODUCTION_FEASIBILITY_POLICY_VERSION.endswith(".2")


def test_a1_blocks_materially_corrupted_dialogue() -> None:
    a1 = _a1_gate_result(_plan(confidence=0.494, duration=1.40))

    assert a1["status"] == "BLOCKED"
    assert a1["reason_code"] == "AUDIO_UNINTELLIGIBLE"
    assert a1["evidence"]["severity"] == "blocker"


def test_a1_blocks_low_confidence_speech_at_material_candidate_coverage() -> None:
    plan = _plan(confidence=0.30, duration=0.40)
    plan["dialogue_mappings"][0]["source_start_seconds"] = 3.0
    plan["dialogue_mappings"][0]["source_end_seconds"] = 5.0

    a1 = _a1_gate_result(plan)

    assert a1["status"] == "BLOCKED"
    reasons = a1["evidence"]["evidence"]["materiality"]["materiality_reasons"]
    assert "LOW_CONFIDENCE_COVERAGE_MATERIAL" in reasons
