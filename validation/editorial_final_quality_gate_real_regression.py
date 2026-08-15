from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from app.analysis_artifact import AnalysisArtifact
from app.clip_results import ClipResult
from app.draft_artifact import DraftArtifact
from app.editorial_profile_policy import resolve_editorial_profile
from app.media import probe_video
from app.pipeline import build_terminal_state
from app.quality_report import build_editorial_final_handoff, build_quality_report
from app.utils import read_json, stable_text_hash, utc_now, write_json


RUN_ID = "670f4563b56c4ae6acd1f00f2e499a7d"
CANDIDATE_ID = "candidate-chapter-027-story-001"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    return stable_text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _one(items: Any, candidate_id: str) -> dict[str, Any]:
    matches = [
        item for item in items if isinstance(item, dict)
        and str(item.get("candidate_id") or item.get("id") or "") == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {candidate_id} record, got {len(matches)}.")
    return matches[0]


def _run_directory(repository_root: Path, run_id: str) -> Path:
    matches = list((repository_root / "output").glob(f"*/runs/{run_id}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one real run {run_id}, got {len(matches)}.")
    return matches[0].resolve()


def run(output_directory: Path, *, run_id: str = RUN_ID) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    run_directory = _run_directory(repository_root, run_id)
    report_data = read_json(run_directory / "report.json", {})
    manifest = read_json(run_directory / "manifest.json", {})
    draft_path = Path(str(manifest["content_understanding"]["draft_artifact_ref"])).resolve()
    draft = DraftArtifact.read(draft_path)
    analysis_path = Path(draft.analysis_artifact_path).resolve()
    analysis = AnalysisArtifact.read_verified(
        analysis_path, expected_sha256=draft.analysis_artifact_sha256,
    )
    source = analysis.load_reference("source")
    content_profile = analysis.load_reference("content_profile")
    resolved_profile = resolve_editorial_profile(content_profile, source=source)
    draft_candidate = _one(draft.candidates, CANDIDATE_ID)
    scored_candidate = _one(report_data["clip_intelligence"]["candidates"], CANDIDATE_ID)
    raw_editorial_value = draft_candidate.get("editorial_decision")
    if not isinstance(raw_editorial_value, dict):
        raise RuntimeError("Real Draft has no persisted CandidateEditorialDecision.")
    raw_editorial: dict[str, Any] = deepcopy(raw_editorial_value)
    editorial, handoff = build_editorial_final_handoff(
        raw_editorial,
        candidate_id=CANDIDATE_ID,
        record_candidate_id=str(draft_candidate.get("candidate_id") or ""),
        expected_profile=resolved_profile.to_dict(),
        draft_id=draft.draft_id,
        analysis_id=analysis.analysis_id,
        analysis_run_id=analysis.analysis_run_id or analysis.analysis_id,
        analysis_sha256=analysis.verified_sha256,
    )
    if editorial is None or handoff["status"] != "passed":
        raise AssertionError(f"Real AVAILABLE editorial handoff failed: {handoff!r}")
    candidate = {
        **scored_candidate,
        "candidate_id": CANDIDATE_ID,
        "eligibility_decision": draft_candidate.get("eligibility_decision"),
        "editorial_decision": raw_editorial,
        "editorial_final_handoff": handoff,
    }
    primary_raw = _one(manifest["primary_results"], CANDIDATE_ID)
    result = ClipResult.from_dict(primary_raw)
    if result is None:
        raise RuntimeError("Real canonical ClipResult is invalid.")
    final_mp4 = Path(result.output_file).resolve()
    production_item = _one(report_data["production_plan"]["items"], CANDIDATE_ID)
    audio_item = _one(report_data["audio"]["items"], CANDIDATE_ID)
    render_item = _one(report_data["production_render"]["items"], CANDIDATE_ID)
    plan = production_item["plan"]
    audio_report = audio_item["report"]
    render_report = render_item["report"]
    creative_root = Path(str(draft_candidate["creative_identity_root"])).resolve()
    preview_manifest_path = creative_root / "parity-manifest.json"
    final_manifest_path = run_directory / "production-render" / "parity-manifest.json"
    original_quality_path = run_directory / "results" / "quality-report-01.json"
    source_path = Path(str(source["path"])).resolve()
    protected_paths = {
        "source": source_path,
        "analysis": analysis_path,
        "approved_draft": draft_path,
        "production_plan": Path(str(draft_candidate["production_plan_ref"])).resolve(),
        "creative_preview": Path(str(draft_candidate["preview"]["output_file"])).resolve(),
        "preview_parity_manifest": preview_manifest_path,
        "final_mp4": final_mp4,
        "final_parity_manifest": final_manifest_path,
        "original_quality_report": original_quality_path,
    }
    hashes_before = {name: _sha256(path) for name, path in protected_paths.items()}
    boundary_hash_before = _json_hash(plan["boundary_decision"])
    continuity_hash_before = _json_hash(plan["continuity_decision"])

    available_report = build_quality_report(
        artifact_path=final_mp4,
        result=result,
        run_id=run_id,
        project_id=str(manifest.get("project_id") or ""),
        source=source,
        plan=plan,
        candidate=candidate,
        diversity_decision=None,
        render_report=render_report,
        audio_report=audio_report,
        all_results=[result],
    )
    blocked_editorial = deepcopy(raw_editorial)
    blocked_editorial.update({
        "surfacing_state": "BLOCKED",
        "selectable": False,
        "hard_blockers": ["SEMANTIC_INCOMPLETE"],
        "primary_reason": "SEMANTIC_INCOMPLETE",
    })
    _blocked_decision, blocked_handoff = build_editorial_final_handoff(
        blocked_editorial,
        candidate_id=CANDIDATE_ID,
        record_candidate_id=CANDIDATE_ID,
        expected_profile=resolved_profile.to_dict(),
        draft_id=draft.draft_id,
        analysis_id=analysis.analysis_id,
        analysis_run_id=analysis.analysis_run_id or analysis.analysis_id,
        analysis_sha256=analysis.verified_sha256,
    )
    blocked_report = build_quality_report(
        artifact_path=final_mp4,
        result=result,
        run_id=run_id,
        project_id=str(manifest.get("project_id") or ""),
        source=source,
        plan=plan,
        candidate={
            **candidate,
            "editorial_decision": blocked_editorial,
            "editorial_final_handoff": blocked_handoff,
        },
        diversity_decision=None,
        render_report=render_report,
        audio_report=audio_report,
        all_results=[result],
    )
    available_terminal = build_terminal_state(
        1,
        [final_mp4],
        {
            "found": 95, "selected": 1, "transformed": 1, "production_plans": 1,
            "render_attempts": 1, "rendered": 1, "rejected": 94, "failed": 0,
        },
        delivery_required=True,
        quality_reports=[{"status": available_report.status}],
    )
    preview_parity = read_json(preview_manifest_path, {})
    final_parity = read_json(final_manifest_path, {})
    media_probe = probe_video(final_mp4)
    hashes_after = {name: _sha256(path) for name, path in protected_paths.items()}
    checks = {
        "real_candidate_available": editorial.surfacing_state.value == "AVAILABLE" and editorial.selectable,
        "legacy_incomplete_story_preserved": (
            "incomplete_story"
            in candidate["virality"]["eligibility"]["critical_failures"]
        ),
        "available_final_passes_with_warnings": available_report.status == "PASS_WITH_WARNINGS",
        "available_final_has_no_blockers": not any(
            finding.severity == "blocker" for finding in available_report.findings
        ),
        "available_terminal_accepted": available_terminal["status"] != "failed",
        "blocked_editorial_final_blocked": blocked_report.status == "BLOCKED",
        "blocked_editorial_finding_present": any(
            finding.code == "SEMANTIC_INCOMPLETE"
            and finding.provenance.get("producer") == "editorial_profile_policy"
            for finding in blocked_report.findings
        ),
        "final_mp4_probe_valid": (
            media_probe.get("duration", 0) > 0
            and media_probe.get("width") == 1080
            and media_probe.get("height") == 1920
            and media_probe.get("audio_streams", 0) > 0
        ),
        "final_mp4_hash_matches_canonical_result": _sha256(final_mp4) == str(primary_raw["artifact_checksum"]),
        "preview_final_plan_hash_match": preview_parity.get("plan_hash") == final_parity.get("plan_hash"),
        "preview_final_parity_signature_match": (
            preview_parity.get("parity_signature") == final_parity.get("parity_signature")
        ),
        "analysis_snapshot_verified": analysis.verified_sha256 == draft.analysis_artifact_sha256,
        "protected_artifacts_unchanged": hashes_before == hashes_after,
        "boundary_decision_unchanged": boundary_hash_before == _json_hash(plan["boundary_decision"]),
        "continuity_decision_unchanged": continuity_hash_before == _json_hash(plan["continuity_decision"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"Real Final Quality Gate regression failed: {checks!r}")
    output_directory.mkdir(parents=True, exist_ok=True)
    available_path = output_directory / "quality-report-available.json"
    write_json(available_path, available_report.to_dict())
    write_json(output_directory / "runtime-evidence.json", {
        "schema_version": "editorial-final-quality-gate-regression.1",
        "captured_at": utc_now(),
        "run_id": run_id,
        "candidate_id": CANDIDATE_ID,
        "source_quality_report_status": read_json(original_quality_path, {}).get("status"),
        "new_quality_report": {
            "status": available_report.status,
            "report_id": available_report.report_id,
            "path": available_path.name,
            "sha256": _sha256(available_path),
        },
        "negative_control_quality_report": {
            "status": blocked_report.status,
            "report_id": blocked_report.report_id,
            "blocker_codes": [
                finding.code for finding in blocked_report.findings
                if finding.severity == "blocker"
            ],
        },
        "editorial_permission": raw_editorial,
        "editorial_final_handoff": handoff,
        "legacy_eligibility_decision": draft_candidate.get("eligibility_decision"),
        "final_mp4": {
            "path": str(final_mp4),
            "sha256": _sha256(final_mp4),
            "probe": media_probe,
        },
        "lineage": {
            "analysis_id": analysis.analysis_id,
            "analysis_run_id": analysis.analysis_run_id,
            "analysis_sha256": analysis.verified_sha256,
            "draft_id": draft.draft_id,
            "boundary_decision_sha256": boundary_hash_before,
            "continuity_decision_sha256": continuity_hash_before,
            "preview_plan_hash": preview_parity.get("plan_hash"),
            "final_plan_hash": final_parity.get("plan_hash"),
            "preview_parity_signature": preview_parity.get("parity_signature"),
            "final_parity_signature": final_parity.get("parity_signature"),
        },
        "protected_artifact_hashes": hashes_after,
        "checks": checks,
        "result": "PASS",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the real AVAILABLE Draft through Final Quality Gate.")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    arguments = parser.parse_args()
    run(arguments.output_directory, run_id=arguments.run_id)
