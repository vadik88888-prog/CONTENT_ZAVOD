from __future__ import annotations

import json
from pathlib import Path

from app.clip_results import ClipResult
from app.continuity import build_continuity_decision
from app.quality_report import build_quality_report


FIXTURE = Path(__file__).with_name("fixtures") / "a2_gameplay_cached_analysis.json"


def _cached_gameplay() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _report_for(
    tmp_path: Path,
    *,
    candidate_id: str,
    boundary: dict,
    decision: object,
    mapped_ranges: list[tuple[float, float]],
):
    artifact = tmp_path / "final.mp4"
    artifact.write_bytes(b"a2 continuity fixture")
    decision_data = decision.model_dump(mode="json")  # type: ignore[attr-defined]
    mapping = {
        "schema_version": "7A.time-map.1",
        "source_ticks_per_second": 1_000_000,
        "output_fps": 30,
        "continuity_decision_id": decision.decision_id,  # type: ignore[attr-defined]
        "continuity_decision_version": decision.schema_version,  # type: ignore[attr-defined]
        "continuity_decision_sha256": decision.fingerprint(),  # type: ignore[attr-defined]
        "segments": [
            {
                "map_id": f"map-{index:03d}",
                "source": {
                    "start_tick": round(start * 1_000_000),
                    "end_tick": round(end * 1_000_000),
                },
                "output": {"start_frame": index * 30, "end_frame": (index + 1) * 30},
            }
            for index, (start, end) in enumerate(mapped_ranges)
        ],
    }
    result = ClipResult(
        candidate_id=candidate_id,
        output_file=str(artifact),
        clip_result_id=f"{candidate_id}:plan-a2",
        production_plan_id="plan-a2",
        content_fingerprint="a2-fixture",
        run_id="run-a2",
        revision_id="run-a2:render-1",
    )
    plan = {
        "plan_id": "plan-a2",
        "envelope": {
            "compatibility_mode": "native",
            "identity": {
                "candidate_id": candidate_id,
                "source_id": "source-a2",
                "run_id": "run-a2",
                "project_id": "project-a2",
            },
            "continuity_decision_ref": decision.decision_id,  # type: ignore[attr-defined]
            "input_fingerprints": {"continuity_decision_sha256": decision.fingerprint()},  # type: ignore[attr-defined]
        },
        "boundary_decision": boundary,
        "continuity_decision": decision_data,
        "dialogue_mappings": [],
    }
    report = build_quality_report(
        artifact_path=artifact,
        result=result,
        run_id="run-a2",
        project_id="project-a2",
        source={"id": "source-a2"},
        plan=plan,
        candidate={"id": candidate_id, "eligibility_decision": {"state": "assessed", "eligible": True, "reason_codes": []}},
        diversity_decision={"schema_version": "5B.2", "selected_candidate_ids": [candidate_id]},
        render_report={
            "validation": "valid",
            "quality": {"status": "passed"},
            "composition": {"segments": []},
            "subtitle_layout": {"quality_decision": {"status": "passed", "reason_codes": []}},
            "source_output_time_map": mapping,
        },
        audio_report={"validation": {"status": "valid", "messages": []}},
        all_results=[result],
    )
    return report


def test_cached_gameplay_weak_evidence_exposes_seven_unexplained_gaps_and_blocks(tmp_path: Path) -> None:
    """Cached analysis only: this test intentionally performs no Brain/Vision call."""

    fixture = _cached_gameplay()
    decision = build_continuity_decision(
        candidate_id=fixture["candidate_id"],
        boundary_decision=fixture["boundary_decision"],
        primary_evidence=fixture["primary_evidence"],
        multimodal_context=fixture["multimodal_context"],
    )

    assert decision is not None
    assert decision.mode == "uncertain"
    assert len(decision.omitted_spans) == 7
    assert round(sum(
        item.source_range.end_seconds - item.source_range.start_seconds
        for item in decision.omitted_spans
    ), 2) == 5.28
    assert {item.rationale_type for item in decision.omitted_spans} == {"unexplained"}

    report = _report_for(
        tmp_path,
        candidate_id=fixture["candidate_id"],
        boundary=fixture["boundary_decision"],
        decision=decision,
        mapped_ranges=[(item["start"], item["end"]) for item in fixture["primary_evidence"]],
    )
    assert report.status == "BLOCKED"
    finding = next(item for item in report.findings if item.code == "CONTINUITY_UNEXPLAINED_OMISSION")
    assert finding.measured_value == {"mode": "uncertain", "unexplained_omission_count": 7}
    assert next(item for item in report.checks if item["code"] == "CONTINUITY")["status"] == "blocked"


def test_evidence_backed_required_bridge_survives_without_unexplained_omissions(tmp_path: Path) -> None:
    fixture = _cached_gameplay()
    boundary = dict(fixture["boundary_decision"])
    boundary["candidate_id"] = "candidate-controlled-a2"
    boundary["decision_id"] = "boundary-controlled-a2"
    boundary["rough_range"] = {"start_seconds": 4.0, "end_seconds": 20.0}
    boundary["refined_range"] = {"start_seconds": 4.0, "end_seconds": 20.0}
    boundary["allowed_source_range"] = {"start_seconds": 4.0, "end_seconds": 20.0}
    decision = build_continuity_decision(
        candidate_id="candidate-controlled-a2",
        boundary_decision=boundary,
        primary_evidence=[
            {"segment_id": 1, "start": 4.0, "end": 6.0},
            {"segment_id": 2, "start": 18.0, "end": 20.0},
        ],
        multimodal_context={
            "continuity_required_spans": [{
                "requirement_type": "semantic_bridge",
                "source_range": {"start_seconds": 6.0, "end_seconds": 18.0},
                "rationale": "Observed action resolves the setup before the spoken payoff.",
                "evidence": {"source": "controlled_fixture", "observation_ids": ["action-1"]},
            }],
        },
    )

    assert decision is not None
    assert decision.mode == "preserve_required_spans"
    assert len(decision.required_spans) == 1
    assert decision.omitted_spans == []
    report = _report_for(
        tmp_path,
        candidate_id="candidate-controlled-a2",
        boundary=boundary,
        decision=decision,
        mapped_ranges=[(4.0, 18.0), (18.0, 20.0)],
    )
    assert report.status == "PASS"


def test_evidence_backed_compaction_does_not_retain_the_whole_boundary(tmp_path: Path) -> None:
    fixture = _cached_gameplay()
    decision = build_continuity_decision(
        candidate_id=fixture["candidate_id"],
        boundary_decision=fixture["boundary_decision"],
        primary_evidence=fixture["primary_evidence"][:2],
        multimodal_context={
            "continuity_omissions": [{
                "rationale_type": "silence",
                "source_range": {"start_seconds": 268.56, "end_seconds": 269.28},
                "rationale": "Measured pause has no visual action or semantic bridge evidence.",
                "evidence": {"source": "controlled_fixture", "silence_seconds": 0.72},
            }],
        },
    )

    assert decision is not None
    assert decision.mode == "compact_dialogue"
    assert decision.required_spans == []
    assert [item.rationale_type for item in decision.omitted_spans] == ["silence"]
    report = _report_for(
        tmp_path,
        candidate_id=fixture["candidate_id"],
        boundary=fixture["boundary_decision"],
        decision=decision,
        mapped_ranges=[(267.92, 268.56), (269.28, 275.69)],
    )
    assert report.status == "PASS"


def test_content_labels_cannot_turn_interview_or_story_into_full_boundary_retention() -> None:
    fixture = _cached_gameplay()
    for label in ("story", "interview_answer", "gameplay"):
        decision = build_continuity_decision(
            candidate_id=fixture["candidate_id"],
            boundary_decision=fixture["boundary_decision"],
            primary_evidence=[
                {"segment_id": 1, "start": 267.92, "end": 268.56},
                {"segment_id": 2, "start": 269.28, "end": 275.69},
            ],
            # ContentProfile labels are intentionally absent from the A-2
            # decision API.  Supplying one as unrelated evidence changes nothing.
            multimodal_context={"content_type": label},
        )
        assert decision is not None
        assert decision.mode == "uncertain"
        assert len(decision.required_spans) == 0
        assert len(decision.omitted_spans) == 1
