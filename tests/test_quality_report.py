from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.clip_results import ClipResult
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.pipeline import build_terminal_state
from app.quality_report import (
    SEMANTIC_DIALOGUE_CONFIDENCE_THRESHOLD,
    build_quality_report,
    exact_dialogue_semantic_blocker,
)
from app.production_models import ContinuityDecision
from app.utils import write_json


def _inputs(tmp_path: Path, *, validation: str = "valid", word_integrity: bool = True):
    artifact = tmp_path / "runs" / "run-1" / "results" / "final.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"synthetic final artifact")
    result = ClipResult(
        candidate_id="candidate-1",
        output_file=str(artifact),
        clip_result_id="candidate-1:plan-1",
        production_plan_id="plan-1",
        content_fingerprint="content-1",
        run_id="run-1",
        revision_id="run-1:render-01",
    )
    continuity = ContinuityDecision.model_validate({
        "schema_version": "A-2.continuity.1",
        "decision_id": "continuity-candidate-1-safe",
        "candidate_id": "candidate-1",
        "boundary_decision_id": "boundary-candidate-1",
        "boundary_decision_sha256": "b" * 64,
        "approved_source_range": {"start_seconds": 1.0, "end_seconds": 5.0},
        "mode": "compact_dialogue",
        "required_spans": [],
        "omitted_spans": [],
    })
    plan = {
        "plan_id": "plan-1",
        "envelope": {
            "compatibility_mode": "native",
            "identity": {
                "candidate_id": "candidate-1", "source_id": "source-1",
                "run_id": "run-1", "project_id": "project-1",
            },
            "continuity_decision_ref": continuity.decision_id,
            "input_fingerprints": {"continuity_decision_sha256": continuity.fingerprint()},
        },
        "boundary_decision": {
            "word_integrity": word_integrity,
            "semantic_completion": True,
            "payoff_preserved": True,
            "allowed_source_range": {"start_seconds": 1.0, "end_seconds": 5.0},
        },
        "continuity_decision": continuity.model_dump(mode="json"),
    }
    candidate = {
        "id": "candidate-1",
        "eligibility_decision": {
            "state": "assessed", "eligible": True, "reason_codes": [],
        },
    }
    render = {
        "output_file": str(artifact),
        "validation": validation,
        "source_output_time_map": {
            "schema_version": "7A.time-map.1",
            "source_ticks_per_second": 1_000_000,
            "output_fps": 30,
            "continuity_decision_id": continuity.decision_id,
            "continuity_decision_version": continuity.schema_version,
            "continuity_decision_sha256": continuity.fingerprint(),
            "segments": [{
                "map_id": "map-source-1",
                "source": {"start_tick": 1_000_000, "end_tick": 5_000_000},
                "output": {"start_frame": 0, "end_frame": 120},
            }],
        },
        "quality": {"status": "passed"},
        "composition": {"segments": []},
        "subtitle_layout": {"quality_decision": {"status": "passed", "reason_codes": []}},
    }
    audio = {"validation": {"status": "valid", "messages": []}}
    diversity = {"schema_version": "5B.2", "selected_candidate_ids": ["candidate-1"]}
    return artifact, result, plan, candidate, render, audio, diversity


def _report(tmp_path: Path, *, validation: str = "valid", word_integrity: bool = True):
    artifact, result, plan, candidate, render, audio, diversity = _inputs(
        tmp_path, validation=validation, word_integrity=word_integrity,
    )
    report = build_quality_report(
        artifact_path=artifact,
        result=result,
        run_id="run-1",
        project_id="project-1",
        source={"id": "source-1"},
        plan=plan,
        candidate=candidate,
        diversity_decision=diversity,
        render_report=render,
        audio_report=audio,
        all_results=[result],
    )
    return artifact, result, report


def _set_native_rich_render(render: dict) -> None:
    def passed_quality() -> dict:
        return {"status": "PASS", "findings": [], "metrics": {}}

    render.update({
        "compatibility_mode": "native",
        "execution_status": "native_rich",
        "execution_reason_codes": ["SOURCE_BROLL_USAGE_NOT_AUTHORIZED"],
        "creative_qc_source": "compiled_render_plan",
        "compiled_render_plan": {"schema_version": "7G.compiled-render-plan.1", "plan_hash": "a" * 64},
        "caption_plan": {
            "cues": [{
                "cue_id": "caption-001",
                "beat_role": None,
                "emphasis": {"emphasis_id": "emphasis-001"},
            }],
            "quality_report": passed_quality(),
        },
        "composition_plan": {
            "segments": [{"segment_id": "composition-001", "target": "person"}],
            "quality_report": passed_quality(),
        },
        "source_broll_plan": {
            "segments": [],
            "quality_report": {
                **passed_quality(),
                "metrics": {"proposal_count": 0, "selected_count": 0},
            },
        },
        "motion_plan": {
            "events": [
                {"event_id": "motion-hook", "purpose": "hook", "primitive_id": "slide"},
                {"event_id": "motion-payoff", "purpose": "payoff", "primitive_id": "scale"},
            ],
            "quality_report": passed_quality(),
        },
    })


def _set_continuity_case(
    plan: dict, render: dict, continuity: ContinuityDecision, mapped_ranges: list[tuple[float, float]],
) -> None:
    plan["continuity_decision"] = continuity.model_dump(mode="json")
    plan["envelope"]["continuity_decision_ref"] = continuity.decision_id
    plan["envelope"]["input_fingerprints"]["continuity_decision_sha256"] = continuity.fingerprint()
    render["source_output_time_map"].update({
        "continuity_decision_id": continuity.decision_id,
        "continuity_decision_version": continuity.schema_version,
        "continuity_decision_sha256": continuity.fingerprint(),
        "segments": [
            {
                "map_id": f"map-{index}",
                "source": {"start_tick": int(start * 1_000_000), "end_tick": int(end * 1_000_000)},
                "output": {"start_frame": index * 30, "end_frame": (index + 1) * 30},
            }
            for index, (start, end) in enumerate(mapped_ranges)
        ],
    })


def _unexplained_large_omission() -> ContinuityDecision:
    return ContinuityDecision.model_validate({
        "schema_version": "A-2.continuity.1",
        "decision_id": "continuity-candidate-1-large-gap",
        "candidate_id": "candidate-1",
        "boundary_decision_id": "boundary-candidate-1",
        "boundary_decision_sha256": "b" * 64,
        "approved_source_range": {"start_seconds": 1.0, "end_seconds": 20.0},
        "mode": "uncertain",
        "required_spans": [],
        "omitted_spans": [{
            "source_range": {"start_seconds": 8.0, "end_seconds": 14.0},
            "rationale_type": "unexplained",
            "rationale": "No persisted rationale explains this six-second source gap.",
            "evidence": {"source": "regression_fixture"},
        }],
    })


def test_continuous_source_map_passes_a2_despite_asr_evidence_gap(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    continuity = _unexplained_large_omission()
    _set_continuity_case(plan, render, continuity, [(1.0, 20.0)])

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio, all_results=[result],
    )

    assert report.status == "PASS"
    assert not any(item.code == "CONTINUITY_UNEXPLAINED_OMISSION" for item in report.findings)


def test_real_large_unexplained_source_omission_remains_blocked_by_a2(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    continuity = _unexplained_large_omission()
    _set_continuity_case(plan, render, continuity, [(1.0, 8.0), (14.0, 20.0)])

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio, all_results=[result],
    )

    assert report.status == "BLOCKED"
    finding = next(item for item in report.findings if item.code == "CONTINUITY_UNEXPLAINED_OMISSION")
    assert finding.measured_value == {"mode": "uncertain", "unexplained_omission_count": 1}


def test_fake_silence_without_evidence_is_blocked_as_invalid_continuity(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    plan["continuity_decision"]["mode"] = "compact_dialogue"
    plan["continuity_decision"]["omitted_spans"] = [{
        "source_range": {"start_seconds": 2.0, "end_seconds": 3.0},
        "rationale_type": "silence",
        "rationale": "Claimed silence without persisted measurement.",
        "evidence": {},
    }]

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio, all_results=[result],
    )

    assert report.status == "BLOCKED"
    assert any(item.code == "CONTINUITY_DECISION_INVALID" for item in report.findings)


def test_quality_report_clean_v2_artifact_passes(tmp_path: Path) -> None:
    _artifact, _result, report = _report(tmp_path)

    assert report.status == "PASS"
    data = report.to_dict()
    assert data["schema_version"] == "5G.0"
    assert data["findings"] == []
    assert {item["code"] for item in data["checks"]} == {
        "ELIGIBILITY", "DIVERSITY", "BOUNDARIES", "CONTINUITY", "PLAN_IDENTITY", "COMPOSITION",
        "SOURCE_BROLL", "EDITORIAL_MOTION", "SUBTITLES", "SEMANTIC_CAPTIONS", "AUDIO", "FFPROBE",
        "ARTIFACT_IDENTITY", "SEMANTIC_CONTENT",
    }
    assert data["artifact_id"] and data["artifact_sha256"]


def test_compact_dialogue_without_omissions_warns_when_render_map_is_unavailable(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    render.pop("source_output_time_map")

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio, all_results=[result],
    )

    finding = next(item for item in report.findings if item.code == "CONTINUITY_TIME_MAP_MISSING")
    assert finding.severity == "warning"
    assert report.status == "PASS_WITH_WARNINGS"


def test_quality_report_warning_preserves_machine_readable_evidence(tmp_path: Path) -> None:
    _artifact, _result, report = _report(tmp_path, validation="warning")

    assert report.status == "PASS_WITH_WARNINGS"
    finding = report.findings[0].to_dict()
    assert finding["code"] == "DURATION_VARIANCE_LOW"
    assert finding["severity"] == "warning"
    assert {"evidence", "measured_value", "threshold", "provenance"} <= finding.keys()


def test_corrupt_food_low_confidence_dialogue_remains_a_quality_blocker(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    plan["dialogue_mappings"] = [{
        "segment_id": "dialogue-001", "fact_id": "fact-001",
        "transcript_segment_id": 227, "confidence": 0.463,
        "source_start_seconds": 682.0, "source_end_seconds": 684.0,
    }]

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "BLOCKED"
    blocker = next(item for item in report.findings if item.code == "AUDIO_UNINTELLIGIBLE")
    assert blocker.provenance["producer"] == "semantic_content_quality"
    assert next(item for item in report.checks if item["code"] == "SEMANTIC_CONTENT")["status"] == "blocked"


def test_preview_and_final_share_exact_dialogue_semantic_blocker_policy(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    plan["dialogue_mappings"] = [
        {
            "segment_id": "dialogue-safe", "fact_id": "fact-safe",
            "transcript_segment_id": 1, "confidence": SEMANTIC_DIALOGUE_CONFIDENCE_THRESHOLD,
            "source_start_seconds": 1.0, "source_end_seconds": 2.0,
        },
        {
            "segment_id": "dialogue-unsafe", "fact_id": "fact-unsafe",
            "transcript_segment_id": 2, "confidence": 0.499,
            "source_start_seconds": 2.0, "source_end_seconds": 3.0,
        },
    ]

    preview_blocker = exact_dialogue_semantic_blocker(plan)
    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )
    final_blocker = next(item for item in report.findings if item.code == "AUDIO_UNINTELLIGIBLE")

    assert preview_blocker is not None
    assert preview_blocker["code"] == final_blocker.code
    assert preview_blocker["threshold"] == final_blocker.threshold == ">=0.5"
    assert preview_blocker["evidence"] == final_blocker.evidence
    assert exact_dialogue_semantic_blocker({
        **plan,
        "dialogue_mappings": [plan["dialogue_mappings"][0]],
    }) is None


def test_semantic_quality_reads_exact_evidence_inside_continuous_media_segment() -> None:
    blocker = exact_dialogue_semantic_blocker({
        "dialogue_mappings": [{
            "segment_id": "dialogue-continuous",
            "confidence": 0.99,
            "source_start_seconds": 1.0,
            "source_end_seconds": 20.0,
            "evidence_mappings": [{
                "fact_id": "fact-low-confidence",
                "transcript_segment_id": 7,
                "confidence": 0.42,
                "source_start_seconds": 8.0,
                "source_end_seconds": 9.0,
            }],
        }],
    })

    assert blocker is not None
    assert blocker["evidence"]["low_confidence_dialogue"][0]["fact_id"] == "fact-low-confidence"


def test_semantic_caption_readability_overlap_and_timing_flow_into_quality_report(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    render["caption_plan"] = {
        "quality_report": {
            "schema_version": "7C.caption-quality.1",
            "status": "BLOCKED",
            "metrics": {"max_cps": 21.8, "protected_overlap_count": 1, "weak_timing_cue_count": 1},
            "findings": [{
                "code": "CAPTION_PROTECTED_REGION_OVERLAP", "severity": "blocker",
                "cue_id": "caption-003", "measured_value": 0.24, "threshold": 0.01,
                "message": "No caption lane avoids an important face/object/screen region.",
            }],
        },
    }

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "BLOCKED"
    assert report.metrics["captions"]["metrics"]["weak_timing_cue_count"] == 1
    assert any(item.code == "CAPTION_PROTECTED_REGION_OVERLAP" for item in report.findings)
    assert next(item for item in report.checks if item["code"] == "SEMANTIC_CAPTIONS")["status"] == "blocked"


def test_dynamic_composition_jitter_and_unsafe_crop_flow_into_quality_report(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    render["composition_plan"] = {
        "segments": [{"segment_id": "composition-001", "fallback": "fit_background"}],
        "quality_report": {
            "schema_version": "7D.composition-quality.1",
            "status": "BLOCKED",
            "metrics": {"jitter_event_count": 2, "unsafe_crop_count": 1},
            "findings": [{
                "code": "COMPOSITION_JITTER", "severity": "blocker",
                "segment_id": "composition-001", "measured_value": 2, "threshold": 0,
                "message": "The crop track contains unintended direction reversals.",
            }],
        },
    }

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "BLOCKED"
    assert report.metrics["composition"]["metrics"]["unsafe_crop_count"] == 1
    assert any(item.code == "COMPOSITION_JITTER" for item in report.findings)
    assert next(item for item in report.checks if item["code"] == "COMPOSITION")["status"] == "blocked"
    assert "composition:composition-001:fit_background" in report.fallbacks


def test_source_broll_rejection_flows_into_quality_report_as_safe_fallback(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    render["source_broll_plan"] = {
        "fallback_policy": "a_roll_current_composition",
        "quality_report": {
            "schema_version": "7E.source-broll-quality.1",
            "status": "PASS_WITH_WARNINGS",
            "metrics": {"selected_count": 0, "a_roll_fallback_count": 1},
            "findings": [{
                "code": "SOURCE_BROLL_PREMATURE_REVEAL", "severity": "warning",
                "decision_id": "broll-1", "measured_value": "rejected_to_a_roll",
                "threshold": "payoff timing safe",
                "message": "The scene reveals payoff before the payoff beat.",
            }],
        },
    }

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "PASS_WITH_WARNINGS"
    assert report.metrics["source_broll"]["metrics"]["a_roll_fallback_count"] == 1
    assert next(item for item in report.checks if item["code"] == "SOURCE_BROLL")["status"] == "warning"
    assert "source_broll:broll-1:a_roll_current_composition" in report.fallbacks


def test_motion_budget_suppression_flows_into_quality_report_as_safe_fallback(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    render["motion_plan"] = {
        "quality_report": {
            "schema_version": "7F.motion-quality.1",
            "status": "PASS_WITH_WARNINGS",
            "metrics": {"requested_event_count": 5, "emitted_event_count": 3, "budget_suppression_count": 2},
            "findings": [{
                "code": "MOTION_BUDGET_SUPPRESSED", "severity": "warning",
                "event_id": "motion-4", "measured_value": "points=10,frames=58",
                "threshold": "points<=8,frames<=42",
                "message": "The lower-priority animation exceeded the global animation budget.",
            }],
        },
    }

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "PASS_WITH_WARNINGS"
    assert report.metrics["motion"]["metrics"]["budget_suppression_count"] == 2
    assert next(item for item in report.checks if item["code"] == "EDITORIAL_MOTION")["status"] == "warning"
    assert "motion:motion-4:calm_fallback" in report.fallbacks


def test_native_creative_qc_ignores_legacy_subtitle_and_reframe_decisions(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    render.update({
        "compatibility_mode": "native",
        "execution_status": "native_fallback",
        "execution_reason_codes": ["TEST_EVIDENCE_UNAVAILABLE"],
        "creative_qc_source": "compiled_render_plan",
        "compiled_render_plan": {"schema_version": "7G.compiled-render-plan.1", "plan_hash": "a" * 64},
        "composition": {"segments": [{
            "segment_id": "legacy-crop",
            "composition_quality_status": "failed",
            "composition_quality_decision": {"status": "blocked"},
        }]},
        "subtitle_layout": {"quality_decision": {"status": "blocked", "reason_codes": ["LEGACY_ONLY"]}},
        "caption_plan": {"quality_report": {"status": "PASS", "findings": [], "metrics": {}}},
        "composition_plan": {"quality_report": {"status": "PASS", "findings": [], "metrics": {}}},
        "source_broll_plan": {"quality_report": {"status": "PASS", "findings": [], "metrics": {}}},
        "motion_plan": {"quality_report": {"status": "PASS", "findings": [], "metrics": {}}},
    })

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "PASS_WITH_WARNINGS"
    assert not any(item.provenance.get("producer") in {
        "composition_quality_decision", "subtitle_quality_decision",
    } for item in report.findings)


def test_native_rich_accepts_optional_broll_off_when_required_layers_executed(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    _set_native_rich_render(render)

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "PASS"
    assert report.metrics["source_broll"]["metrics"]["selected_count"] == 0
    assert not any(item.code == "NATIVE_RICH_EVIDENCE_MISSING" for item in report.findings)


def test_native_rich_blocks_truly_empty_required_layers(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    _set_native_rich_render(render)
    render["caption_plan"]["cues"] = []
    render["composition_plan"]["segments"] = []
    render["motion_plan"]["events"] = []

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    finding = next(item for item in report.findings if item.code == "NATIVE_RICH_EVIDENCE_MISSING")
    assert report.status == "BLOCKED"
    assert set(finding.measured_value) == {
        "CAPTION_LAYER_NOT_EXECUTED",
        "SEMANTIC_EMPHASIS_NOT_EXECUTED",
        "HOOK_PRESENTATION_NOT_EXECUTED",
        "PAYOFF_PRESENTATION_NOT_EXECUTED",
        "COMPOSITION_REFRAME_NOT_EXECUTED",
        "MOTION_LAYER_NOT_EXECUTED",
    }
    assert "source_broll_plan" not in finding.evidence


def test_legacy_adapter_keeps_legacy_subtitle_and_reframe_qc(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    plan["envelope"]["compatibility_mode"] = "legacy_adapter"
    render.update({
        "compatibility_mode": "legacy_adapter",
        "composition": {"segments": [{
            "segment_id": "legacy-crop",
            "composition_quality_status": "failed",
            "composition_quality_decision": {"status": "blocked"},
        }]},
        "subtitle_layout": {"quality_decision": {"status": "blocked", "reason_codes": ["LEGACY_ONLY"]}},
    })

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio,
        all_results=[result],
    )

    assert report.status == "BLOCKED"
    assert any(item.provenance.get("producer") == "composition_quality_decision" for item in report.findings)
    assert any(item.provenance.get("producer") == "subtitle_quality_decision" for item in report.findings)


def test_legacy_source_gap_requires_native_plan_rebuild(tmp_path: Path) -> None:
    artifact, result, plan, candidate, render, audio, diversity = _inputs(tmp_path)
    plan["envelope"]["compatibility_mode"] = "legacy_adapter"
    plan.pop("audio_mode", None)
    plan["dialogue_mappings"] = [
        {"source_start_seconds": 1.0, "source_end_seconds": 2.0},
        {"source_start_seconds": 3.0, "source_end_seconds": 5.0},
    ]

    report = build_quality_report(
        artifact_path=artifact, result=result, run_id="run-1", project_id="project-1",
        source={"id": "source-1"}, plan=plan, candidate=candidate,
        diversity_decision=diversity, render_report=render, audio_report=audio, all_results=[result],
    )

    assert report.status == "BLOCKED"
    assert any(item.code == "LEGACY_PLAN_REBUILD_REQUIRED" for item in report.findings)


def test_quality_blocker_cannot_be_hidden_by_ready_mp4_count(tmp_path: Path) -> None:
    artifact, _result, report = _report(tmp_path, word_integrity=False)
    reference = report.reference(artifact.with_name("quality-report-01.json"))

    assert report.status == "BLOCKED"
    assert any(item.code == "BOUNDARY_WORD_CUT" and item.severity == "blocker" for item in report.findings)
    terminal = build_terminal_state(
        1, [artifact], {"failed": 0}, delivery_required=True, quality_reports=[reference],
    )
    assert terminal["status"] == "failed"
    assert terminal["error_code"] == "QUALITY_GATE_BLOCKED"
    assert terminal["quality_gate"]["status"] == "BLOCKED"


def test_desktop_reads_the_same_persisted_quality_status_and_recovers(tmp_path: Path, monkeypatch) -> None:
    artifact, result, report = _report(tmp_path, validation="warning")
    report_path = artifact.with_name("quality-report-01.json")
    write_json(report_path, report.to_dict())
    result = replace(
        result,
        artifact_id=report.artifact_id,
        artifact_checksum=report.artifact_sha256,
        quality_report_id=report.report_id,
        quality_report_path=str(report_path),
        quality_status=report.status,
    )
    reference = report.reference(report_path)
    gate = {"schema_version": "5G.0", "status": "PASS_WITH_WARNINGS", "reports": [reference]}
    run_directory = artifact.parents[1]
    report_json = run_directory / "report.json"
    manifest = run_directory / "manifest.json"
    write_json(report_json, {
        "production_render": {"status": "completed", "output_file": str(artifact)},
        "primary_results": [result.to_dict()], "quality_gate": gate, "warnings": [], "ai": {}, "tts": {},
    })
    write_json(manifest, {"run_id": "run-1", "primary_results": [result.to_dict()], "quality_gate": gate})
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path, state_path=run_directory / "state.json",
        report_path=report_json, output_directory=run_directory, runtime_config_path=run_directory / "runtime.yaml",
        run_id="run-1", manifest_path=manifest,
    )
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    facade = PipelineFacade(tmp_path)
    completion = facade.completion(prepared)
    recovered = facade.recovery_completion(prepared, "2026-01-01T00:00:00+00:00")

    assert completion.error_summary is None
    assert completion.quality_status == "PASS_WITH_WARNINGS"
    assert completion.quality_report_paths == (report_path,)
    assert completion.legacy_technical_completion is False
    assert recovered is not None and recovered.quality_status == "PASS_WITH_WARNINGS"


def test_desktop_rejects_artifact_changed_after_quality_report(tmp_path: Path, monkeypatch) -> None:
    artifact, result, report = _report(tmp_path)
    report_path = artifact.with_name("quality-report-01.json")
    write_json(report_path, report.to_dict())
    result = replace(
        result,
        artifact_id=report.artifact_id,
        artifact_checksum=report.artifact_sha256,
        quality_report_id=report.report_id,
        quality_report_path=str(report_path),
        quality_status=report.status,
    )
    gate = {"schema_version": "5G.0", "status": "PASS", "reports": [report.reference(report_path)]}
    run_directory = artifact.parents[1]
    report_json = run_directory / "report.json"
    manifest = run_directory / "manifest.json"
    write_json(report_json, {
        "production_render": {"status": "completed", "output_file": str(artifact)},
        "primary_results": [result.to_dict()], "quality_gate": gate,
    })
    write_json(manifest, {"run_id": "run-1", "primary_results": [result.to_dict()], "quality_gate": gate})
    artifact.write_bytes(b"artifact was replaced after quality validation")
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path, state_path=run_directory / "state.json",
        report_path=report_json, output_directory=run_directory, runtime_config_path=run_directory / "runtime.yaml",
        run_id="run-1", manifest_path=manifest,
    )
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    completion = PipelineFacade(tmp_path).completion(prepared)

    assert completion.error_summary == "Final Quality Gate не подтвердил готовность результата."
    assert "checksum" in (completion.technical_details or "")


def test_legacy_completion_is_not_promoted_to_v2_pass(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "final.mp4"
    artifact.write_bytes(b"legacy final artifact")
    report_path = tmp_path / "report.json"
    write_json(report_path, {"production_render": {"status": "completed", "output_file": str(artifact)}})
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=tmp_path, state_path=tmp_path / "state.json",
        report_path=report_path, output_directory=tmp_path, runtime_config_path=tmp_path / "runtime.yaml",
    )
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    completion = PipelineFacade(tmp_path).completion(prepared)

    assert completion.error_summary is None
    assert completion.quality_status is None
    assert completion.legacy_technical_completion is True
