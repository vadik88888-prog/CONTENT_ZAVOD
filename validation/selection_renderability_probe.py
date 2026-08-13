"""Provider-free cached holdout probe for recommendation renderability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.analysis_artifact import AnalysisArtifact
from app.candidate_quality import build_eligibility_decision
from app.config import load_config
from app.content_understanding import ensure_candidate_boundary_decision, select_with_coverage
from app.models import scored_from_dict
from app.production_feasibility import (
    production_feasibility_index,
    resolve_recommendation_production_feasibility,
)
from app.production_plan import ProductionPlanEnvelopeContext
from app.utils import read_json, stable_text_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation" / "metrics" / "goal7j3-selection-renderability.json"


def _holdouts() -> dict[str, Path]:
    values = {
        name: ROOT / "validation" / "artifacts" / "goal7i" / name / "analysis-handoff.json"
        for name in ("podcast", "interview", "food", "gameplay")
    }
    story = list((ROOT / "output").glob(
        "**/d0c9bad6342b4e849eed61877b9fa16d/analysis.json"
    ))
    if len(story) != 1:
        raise FileNotFoundError(
            "Expected the cached story holdout analysis run d0c9bad6342b4e849eed61877b9fa16d."
        )
    values["story"] = story[0]
    return values


def _reference(analysis: AnalysisArtifact, name: str) -> dict[str, Any]:
    path = Path(analysis.references[name])
    value = read_json(path, None)
    if not isinstance(value, dict):
        raise ValueError(f"Cached analysis reference is unavailable: {name} ({path}).")
    return value


def _envelope(analysis: AnalysisArtifact, transcript: dict[str, Any], config: Any) -> ProductionPlanEnvelopeContext:
    return ProductionPlanEnvelopeContext(
        project_id=analysis.project_id or "project-holdout-probe",
        run_id="run-selection-renderability-probe",
        analysis_id=analysis.analysis_id,
        analysis_fingerprint=analysis.analysis_fingerprint,
        # Identity hashes do not affect A-1/A-3 decisions. Keep the probe fully
        # cached and avoid rereading multi-gigabyte media merely for an envelope.
        source_sha256=stable_text_hash(analysis.source_fingerprint),
        transcript_sha256=stable_text_hash(str(transcript)),
        preset_id=config.product_flow.subtitle_preset,
        preset_version=config.product_flow.preset_version,
        platform=config.product_flow.platform,
        target_width=config.production_render.output_width,
        target_height=config.production_render.output_height,
        target_fps=config.production_render.output_fps,
    )


def probe_holdout(name: str, path: Path, config: Any) -> dict[str, Any]:
    analysis = AnalysisArtifact.read(path)
    ranked_path = Path(analysis.work_directory) / "virality_ranking.json"
    ranked = read_json(ranked_path, None)
    if not isinstance(ranked, dict):
        raise ValueError(f"Cached virality ranking is unavailable: {ranked_path}.")
    ranked_items = {
        str(item.get("id")): item
        for item in ranked.get("candidates", [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    final_candidates = read_json(Path(analysis.candidate_data_ref), None)
    if not isinstance(final_candidates, dict):
        raise ValueError(f"Cached final candidate data is unavailable: {analysis.candidate_data_ref}.")
    scored = []
    for item in final_candidates.get("candidates", []):
        if not isinstance(item, dict) or str(item.get("id") or "") not in ranked_items:
            continue
        ranked_item = ranked_items[str(item["id"])]
        scored.append(scored_from_dict({
            **item,
            # The final artifact owns refreshed eligibility/boundary evidence;
            # the pre-final ranking owns the pool eligible for MMR replacement.
            "selected": bool(ranked_item.get("selected")),
            "virality": dict(ranked_item.get("virality") or item.get("virality") or {}),
        }))
    transcript = _reference(analysis, "transcript")
    content_map = _reference(analysis, "content_map")
    visual_analysis = _reference(analysis, "visual_analysis")
    for item in scored:
        candidate = item.candidate
        candidate.eligibility_decision = build_eligibility_decision(
            candidate,
            candidate.feature_vector,
            config_version=config.scoring.candidate_quality_config_version,
            min_duration_seconds=config.candidate_generation.min_duration_seconds,
            max_duration_seconds=config.candidate_generation.max_duration_seconds,
            visual_analysis=visual_analysis,
        )
        ensure_candidate_boundary_decision(candidate)
    feasibility = resolve_recommendation_production_feasibility(
        scored,
        content_map=content_map,
        source=_reference(analysis, "source"),
        metadata=_reference(analysis, "metadata"),
        transcript=transcript,
        transcript_features=_reference(analysis, "transcript_features"),
        audio_features=_reference(analysis, "audio_features"),
        scenes=_reference(analysis, "scene_boundaries"),
        multimodal_timeline=_reference(analysis, "multimodal_timeline"),
        story_units=_reference(analysis, "story_units"),
        config=config,
        envelope_context=_envelope(analysis, transcript, config),
    )
    selection_input = [scored_from_dict(item.to_dict()) for item in scored]
    selected, coverage = select_with_coverage(
        selection_input,
        config,
        content_map,
        production_feasibility=feasibility,
    )
    feasibility_by_id = production_feasibility_index(feasibility)
    before = [str(item) for item in analysis.recommendation.get("selected_candidate_ids", [])]
    after = [item.candidate.id for item in selected]
    exact_recommendations = [{
        "candidate_id": item.candidate.id,
        "selection_reason": item.selection_reason,
        "production_status": feasibility_by_id[item.candidate.id]["status"],
        "production_reason_code": feasibility_by_id[item.candidate.id]["reason_code"],
    } for item in selected]
    blocked_before = [{
        "candidate_id": candidate_id,
        "reason_code": feasibility_by_id[candidate_id]["reason_code"],
        "reason": feasibility_by_id[candidate_id]["reason"],
    } for candidate_id in before if (
        candidate_id in feasibility_by_id
        and feasibility_by_id[candidate_id]["status"] == "GUARANTEED_BLOCKED"
    )]
    viable_after = bool(after) and all(
        feasibility_by_id[candidate_id]["status"] == "VIABLE"
        for candidate_id in after
    )
    return {
        "holdout": name,
        "analysis_artifact": str(path),
        "ranked_candidate_count": len(scored),
        "recommended_before": before,
        "recommended_after": after,
        "blocked_before": blocked_before,
        "exact_recommendations": exact_recommendations,
        "selection_result_reason_code": (
            coverage.get("diversity_decision", {}).get("result_reason_code")
        ),
        "feasibility_summary": feasibility["summary"],
        "selection_iterations": feasibility["selection_iterations"],
        "provider_calls": feasibility["provider_calls"],
        "viable_recommendation": viable_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    # The cached acceptance analyses were intentionally collected with the
    # permissive selection score floor. This is probe input, not a threshold change.
    config.score_threshold = 0
    holdouts = [probe_holdout(name, path, config) for name, path in _holdouts().items()]
    payload = {
        "schema_version": "7J.3.selection-renderability-probe.1",
        "source": "cached AnalysisArtifact + ranked candidates + persisted Phase-6 evidence",
        "provider_calls": {"brain": 0, "vision": 0, "transformation": 0},
        "holdout_count": len(holdouts),
        "passing_holdout_count": sum(item["viable_recommendation"] for item in holdouts),
        "status": "PASS" if all(item["viable_recommendation"] for item in holdouts) else "FAILED",
        "holdouts": holdouts,
    }
    write_json(args.output, payload)
    for item in holdouts:
        print(
            f"{item['holdout']}: {','.join(item['recommended_after']) or '<none>'} | "
            + "; ".join(
                f"{value['candidate_id']}={value['production_reason_code']}"
                for value in item["exact_recommendations"]
            )
        )
    print(f"status={payload['status']} provider_calls={payload['provider_calls']} output={args.output}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
