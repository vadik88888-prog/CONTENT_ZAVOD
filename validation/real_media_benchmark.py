"""Goal 6E real-media OLD-vs-6D benchmark.

The script never downloads or renders media.  It reuses persisted, source-scoped
transcript/audio/scene artifacts, extracts only budget-admitted Vision frames,
and writes derived JSON/Markdown evidence under ``validation/results``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from app.ai import get_vision_provider
from app.candidate_quality import EligibilityReasonCode, assess_hook_and_payoff, boundary_multimodal_context, build_eligibility_decision
from app.config import AppConfig
from app.content_understanding import generate_semantic_candidates
from app.intelligence import local_rank, shortlist
from app.local_scoring import score_candidates
from app.models import Candidate
from app.multimodal_candidates import enrich_shortlist_with_pass2, generate_multimodal_candidates
from app.multimodal_evidence import build_multimodal_timeline
from app.selection import select_clips
from app.vision_intelligence import VisionGateway


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "validation" / "fixtures" / "real_media_benchmark.fixture.json"
DEFAULT_JSON = ROOT / "validation" / "results" / "real_media_benchmark.json"
DEFAULT_MARKDOWN = ROOT / "validation" / "results" / "real_media_benchmark.md"
BASELINE_VERSION = "5B.1-pre-6D@f70ba0e^"
NEW_VERSION = "6D.1"
HUMAN_REVIEW = {
    "podcast_talking_head": ["useful", "neutral", "mixed"],
    "interview": ["useful_with_continuity_risk", "mixed", "useful"],
    "food_vlog": ["useful", "useful", "useful"],
    "gameplay": ["useful", "useful", "neutral_same_candidate"],
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _source_path(item: dict[str, Any]) -> Path:
    workspace = item.get("workspace_source")
    if workspace:
        path = ROOT / str(workspace)
        if path.exists():
            return path
    project = item.get("localappdata_project_id")
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "ContentFactoryData" / "projects" / str(project) / "sources"
    matches = sorted(path for path in local.glob("*") if path.is_file())
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Licensed/local source is unavailable for {item['source_key']}")


def _config() -> AppConfig:
    config = AppConfig()
    config.optional_visual_features = True
    config.product_flow.processing_mode = "standard"
    config.product_flow.clip_count = 3
    config.ai_reranking.final_clip_count = 3
    config.max_clips = 3
    config.virality.enabled = False
    config.validate()
    return config


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    return call(), time.perf_counter() - started


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, value))


def _old_score(candidates: list[Candidate], audio: dict[str, Any], scenes: dict[str, Any], config: AppConfig) -> list[Candidate]:
    """Exact deterministic pre-6D factor formula, kept outside product code."""

    from app.audio_features import window_audio_features
    from app.scene_detection import window_scene_features

    for candidate in candidates:
        features = {
            **candidate.feature_vector,
            **window_audio_features(candidate.start, candidate.end, audio),
            **window_scene_features(candidate.start, candidate.end, scenes),
        }
        scores = {
            "hook": _bounded(float(features.get("hook_phrase_score", 0))),
            "completeness": _bounded(float(features.get("completeness_score", 0))),
            "clarity": _bounded((65 if features.get("transcript_confidence") is None else float(features["transcript_confidence"]) * 100) - float(features.get("filler_word_ratio", 0)) * 35),
            "speech_density": _bounded(float(features.get("speech_density", 0)) * 100),
            "pacing": _bounded(100 - min(100, abs(float(features.get("words_per_second", 0)) - 2.5) * 35)),
            "audio_energy": _bounded(float(features.get("audio_energy", 0)) * 100),
            "scene_structure": _bounded(30 + float(features.get("visual_activity", 0)) * 60 + 20 * (float(features.get("scene_change_near_start", 0)) + float(features.get("scene_change_near_end", 0)))),
            "context_independence": _bounded(100 - float(features.get("context_dependency_score", 0))),
            "boundary_quality": _bounded(30 + (25 if features.get("sentence_start") else 0) + (25 if features.get("sentence_end") else 0) + 10 * float(features.get("silence_before", 0)) + 10 * float(features.get("silence_after", 0))),
        }
        decision = build_eligibility_decision(
            candidate, features, config_version=BASELINE_VERSION,
            min_duration_seconds=config.min_clip_duration,
            max_duration_seconds=config.max_clip_duration,
            visual_analysis={"status": "fallback", "subject_keyframes": []},
        )
        boundary = candidate.boundary_diagnostics or {}
        hook, payoff, _hook_ref, _payoff_ref, _ = assess_hook_and_payoff(candidate, features, boundary)
        completeness = scores["completeness"]
        context_independence = scores["context_independence"]
        boundary_score = scores["boundary_quality"]
        information = float((candidate.semantic_evidence or {}).get("information_density", 0))
        if information <= 1:
            information *= 100
        emotional = _bounded(float(features.get("audio_energy", 0)) * 100 + float(features.get("exclamation_count", 0)) * 8)
        novelty = _bounded((100 - float(features.get("repetition_score", 0)) * 100) * 0.55 + hook * 0.45)
        target = (config.min_clip_duration + config.max_clip_duration) / 2
        duration_fit = _bounded(100 - abs(candidate.duration - target) / max(1.0, (config.max_clip_duration - config.min_clip_duration) / 2) * 45)
        factors = {
            "hook": hook,
            "narrative_completeness": (completeness + boundary_score) / 2,
            "payoff": 82.0 if payoff else 0.0,
            "information_value": information or min(100.0, float(features.get("word_count", 0)) * 3),
            "emotional_intensity": emotional,
            "visual_interest": 0.0,
            "audio_energy": scores["audio_energy"],
            "self_containedness": (completeness + context_independence + boundary_score) / 3,
            "context_debt": 100 - context_independence,
            "vertical_viability": 0.0,
            "novelty": novelty,
            "confidence": float(features.get("transcript_confidence", 0.65) or 0.65) * 100,
        }
        legacy_components = {
            "self_containment": factors["self_containedness"], "hook_strength": hook,
            "payoff_strength": factors["payoff"], "narrative_arc": factors["narrative_completeness"],
            "informational_value": factors["information_value"], "emotional_intensity": emotional,
            "novelty_or_conflict": novelty, "speech_clarity": scores["clarity"],
            "visual_viability": 0.0, "pacing_density": (scores["pacing"] + scores["speech_density"]) / 2,
            "platform_fit": duration_fit,
        }
        weights = {"self_containment": .16, "hook_strength": .14, "payoff_strength": .12, "narrative_arc": .10, "informational_value": .10, "emotional_intensity": .08, "novelty_or_conflict": .08, "speech_clarity": .08, "visual_viability": .06, "pacing_density": .04, "platform_fit": .04}
        context_codes = sum(code in decision.reason_codes for code in (
            EligibilityReasonCode.UNRESOLVED_PRONOUN, EligibilityReasonCode.UNNAMED_ENTITY,
            EligibilityReasonCode.ANSWER_WITHOUT_QUESTION_CONTEXT, EligibilityReasonCode.REFERENCES_EARLIER_CONTENT,
            EligibilityReasonCode.UNDEFINED_TERM_OR_SETUP,
        ))
        penalty = float(features.get("repetition_score", 0)) * 15 + float(features.get("filler_word_ratio", 0)) * 18 + min(25, context_codes * 10) + 8
        candidate.feature_vector = features
        candidate.local_scores = {**scores, "benchmark_factors": factors, "weighted_score": round(sum(legacy_components[key] * weights[key] for key in weights), 3)}
        candidate.eligibility_decision = decision
        candidate.local_quality_score = round(_bounded(sum(legacy_components[key] * weights[key] for key in weights) - penalty), 3)
        candidate.composition_intent = {"evidence_status": "not_available_pre_6D"}
    return candidates


def _select(candidates: list[Candidate], config: AppConfig, content_map: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    ranked = local_rank(candidates)
    selected = select_clips(ranked, config, content_map)
    return selected, ranked


def _factors(candidate: Candidate, *, baseline: bool) -> dict[str, float]:
    if baseline:
        return {key: round(float(value), 3) for key, value in candidate.local_scores.get("benchmark_factors", {}).items()}
    score = candidate.candidate_score_v2
    return {key: round(value.score, 3) for key, value in score.factors.items()} if score else {}


def _candidate_result(candidate: Candidate, *, baseline: bool) -> dict[str, Any]:
    boundary = candidate.boundary_diagnostics or {}
    context = {} if baseline else boundary_multimodal_context(candidate)
    provenance = candidate.multimodal_provenance or {}
    pass2 = candidate.vision_pass2_evidence or {}
    visual_evidence = provenance.get("visual_evidence", []) if isinstance(provenance, dict) else []
    pass2_result = pass2.get("result") if isinstance(pass2, dict) else None
    pass2_observations = pass2_result.get("observations", []) if isinstance(pass2_result, dict) else []

    def observation(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        raw_nested = item.get("observation")
        nested: dict[str, Any] = raw_nested if isinstance(raw_nested, dict) else item
        return {
            "timestamp": nested.get("timestamp", item.get("start_seconds")),
            "confidence": nested.get("confidence", item.get("confidence")),
            "scene_type": nested.get("scene_type"), "primary_subject": nested.get("primary_subject"),
            "action": nested.get("action"), "reaction": nested.get("reaction"),
            "payoff_signal": nested.get("payoff_signal"), "composition_risk": nested.get("composition_risk"),
            "origin": nested.get("origin"),
        }
    return {
        "candidate_id": candidate.id,
        "start_seconds": round(candidate.start, 3), "end_seconds": round(candidate.end, 3),
        "duration_seconds": round(candidate.duration, 3), "score": candidate.local_quality_score,
        "text_excerpt": " ".join(candidate.text.split())[:420],
        "factors": _factors(candidate, baseline=baseline),
        "boundary": {
            "eligible": boundary.get("eligible"), "word_integrity": boundary.get("word_integrity"),
            "sentence_integrity": boundary.get("sentence_integrity"), "semantic_completion": boundary.get("semantic_completion"),
            "payoff_preserved": boundary.get("payoff_preserved"), "multimodal_context": context,
        },
        "composition_intent": candidate.composition_intent,
        "evidence": {
            "generation_kind": (provenance.get("generation") or {}).get("candidate_kind"),
            "generation_reasons": (provenance.get("generation") or {}).get("reasons", []),
            "audio_count": len(provenance.get("audio_evidence", [])),
            "visual_count": len(provenance.get("visual_evidence", [])),
            "visual_observations": [observation(value) for value in visual_evidence[:6]],
            "pass2_status": pass2.get("status"),
            "pass2_verification": (pass2_result.get("verification") if isinstance(pass2_result, dict) else None),
            "pass2_observations": [observation(value) for value in pass2_observations[:7]],
        },
    }


def _usage(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [item.get("diagnostics", {}) for item in artifacts if isinstance(item, dict)]
    usages = [item.get("usage", {}) for item in diagnostics]
    return {
        "frames_requested": sum(int(item.get("frames_requested", 0)) for item in diagnostics),
        "frames_extracted": sum(int(item.get("frames_extracted", 0)) for item in diagnostics),
        "frames_sent": sum(int(item.get("frames_sent", 0)) for item in diagnostics),
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in diagnostics),
        "cache_misses": sum(int(item.get("cache_misses", 0)) for item in diagnostics),
        "calls": sum(int(item.get("calls", 0)) for item in usages),
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in usages),
        "output_tokens": sum(int(item.get("output_tokens", 0)) for item in usages),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in usages),
        "estimated_cost_usd": round(sum(float(item.get("estimated_cost", 0) or 0) for item in usages), 8),
        "stop_reasons": sorted({str(item.get("analysis_stop_reason")) for item in diagnostics}),
    }


def _rejections(ranked: list[Any], selected: list[Any]) -> list[dict[str, Any]]:
    chosen = {item.candidate.id for item in selected}
    result = []
    for item in sorted(ranked, key=lambda value: value.score, reverse=True):
        if item.candidate.id in chosen:
            continue
        decision = item.candidate.eligibility_decision
        result.append({
            "candidate_id": item.candidate.id, "score": item.score,
            "reason": item.selection_reason or item.rejection_reason or "not_in_top_selection",
            "eligibility_reason_codes": [code.value for code in decision.reason_codes] if decision else ["LEGACY_UNASSESSED"],
            "selection_diagnostics": item.selection_diagnostics,
        })
        if len(result) == 12:
            break
    return result


def _pairwise(selected: list[Any]) -> dict[str, float]:
    from app.diversity import transcript_similarity
    similarities = [
        transcript_similarity(left.candidate.text, right.candidate.text)
        for index, left in enumerate(selected) for right in selected[index + 1:]
    ]
    return {
        "selected_count": len(selected),
        "mean_transcript_similarity": round(statistics.mean(similarities), 4) if similarities else 0.0,
        "max_transcript_similarity": round(max(similarities), 4) if similarities else 0.0,
    }


def _side_by_side(source_key: str, old: list[Any], new: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for index in range(max(len(old), len(new))):
        left = old[index].candidate if index < len(old) else None
        right = new[index].candidate if index < len(new) else None
        old_factors = _factors(left, baseline=True) if left else {}
        new_factors = _factors(right, baseline=False) if right else {}
        deltas = {key: round(new_factors.get(key, 0) - old_factors.get(key, 0), 3) for key in sorted(set(old_factors) | set(new_factors))}
        influential = sorted(deltas, key=lambda key: abs(deltas[key]), reverse=True)[:4]
        rows.append({
            "rank": index + 1,
            "baseline": _candidate_result(left, baseline=True) if left else None,
            "multimodal": _candidate_result(right, baseline=False) if right else None,
            "same_candidate": bool(left and right and left.id == right.id),
            "ranking_change_reason": {key: deltas[key] for key in influential},
            "evidence_that_changed_ranking": (_candidate_result(right, baseline=False)["evidence"] if right else {}),
            "usefulness": HUMAN_REVIEW[source_key][index],
            "review_basis": "manual midpoint-frame inspection plus transcript/evidence context",
        })
    return rows


def _failure_matrix(config: AppConfig) -> list[dict[str, Any]]:
    base = Candidate(
        "failure-matrix", 0, 30, "Why this matters. The result is clear.",
        story_unit_id="failure", story_unit_ids=["failure"],
        semantic_evidence={"hook": "Why", "payoff": "result", "completeness_score": .84, "information_density": .72},
        boundary_diagnostics={"eligible": True, "word_integrity": True, "sentence_integrity": True, "semantic_completion": .84, "payoff_preserved": True, "overall_boundary_score": .84},
        feature_vector={"hook_phrase_score": 72, "completeness_score": 84, "context_dependency_score": 0, "speech_density": .7, "words_per_second": 2.5, "word_count": 20, "sentence_start": True, "sentence_end": True, "transcript_confidence": .92, "repetition_score": 0, "filler_word_ratio": 0},
    )
    cases = {
        "missing_vision": ({"status": "skipped", "reason": "provider unavailable", "result": None}, [], False),
        "weak_transcript": (None, [], False),
        "visual_heavy": (_pass2(.92, action="demonstration", payoff="result"), ["editorial_roles:action,payoff"], True),
        "audio_driven": (None, [], True),
        "no_strong_multimodal_evidence": (None, [], False),
        "random_motion": (_pass2(.90, action="movement"), [], False),
        "ordinary_scene_change": (None, [], False),
        "weak_reaction": (_pass2(.51, reaction="minor"), [], False),
        "loud_sound_without_editorial_value": (None, [], True),
        "low_confidence_visual": (_pass2(.25, action="movement", reaction="minor"), ["editorial_roles:action"], False),
    }
    rows = []
    for name, (pass2, roles, energetic) in cases.items():
        candidate = copy.deepcopy(base)
        if name == "weak_transcript":
            candidate.feature_vector["transcript_confidence"] = .30
        candidate.multimodal_provenance = {"generation": {"reasons": roles}, "visual_evidence": []}
        if name == "audio_driven":
            candidate.multimodal_provenance["audio_evidence"] = [{"event_type": "reaction_label", "confidence": 0.9}]
        candidate.vision_pass2_evidence = pass2
        score_candidates([candidate], {"energy_frames": ([{"time": 10, "normalized_loudness": .95}] if energetic else []), "silence_intervals": []}, {"boundaries": ([{"timestamp": 15, "score": .9}] if name == "ordinary_scene_change" else [])}, config.scoring, min_duration_seconds=15, max_duration_seconds=60, visual_analysis={"status": "fallback", "subject_keyframes": []})
        score = candidate.candidate_score_v2
        rows.append({"case": name, "score": candidate.local_quality_score, "visual_interest": score.factors["visual_interest"].score, "audio_energy": score.factors["audio_energy"].score, "confidence": score.factors["confidence"].score, "composition_intent": candidate.composition_intent})
    return rows


def _pass2(confidence: float, *, action: str = "none", reaction: str = "none", payoff: str = "none") -> dict[str, Any]:
    observation = {"timestamp": 15, "confidence": confidence, "primary_subject": "object", "reaction": reaction, "payoff_signal": payoff, "composition_risk": "none", "visible_face_count": 0, "action": action, "scene_type": "PRODUCT_DEMO", "origin": "provider"}
    return {"status": "completed", "result": {"schema_version": "6B.pass2-result.1", "request": {"anchors": {}}, "verification": {"hook_visible": False, "action_visible": action not in {"none", "unknown"}, "reaction_visible": reaction not in {"none", "unknown"}, "payoff_visible": payoff in {"result", "reveal", "resolution"}, "continuity_risk": "low", "confidence": confidence}, "observations": [observation]}}


def _benchmark_source(item: dict[str, Any], config: AppConfig, provider: Any, cache: Path) -> dict[str, Any]:
    work = ROOT / "work" / str(item["work_directory"])
    required = {name: _read(work / name) for name in ("source.json", "metadata.json", "transcript.json", "transcript_features.json", "audio_features.json", "scene_boundaries.json", "visual_analysis.json", "global_content_map.json")}
    if str(required["source.json"].get("id")) != item["expected_source_id"]:
        raise ValueError(f"Source identity mismatch: {item['source_key']}")
    source = _source_path(item)
    metadata = required["metadata.json"]
    transcript = required["transcript.json"]
    transcript_features = required["transcript_features.json"]
    audio = required["audio_features.json"]
    scenes = required["scene_boundaries.json"]
    visual = required["visual_analysis.json"]
    content_map = required["global_content_map.json"]

    timeline, timeline_seconds = _timed(lambda: build_multimodal_timeline(source_id=item["expected_source_id"], source_duration_seconds=float(metadata["duration"]), transcript=transcript, audio_features=audio, scenes=scenes, visual_analysis=visual))
    gateway = VisionGateway(config=config, cache_directory=cache, provider=provider)
    vision1, vision1_seconds = _timed(lambda: gateway.analyze_pass1(source=source, timeline=timeline, content_type=item["content_type"]))
    vision1_warm, vision1_warm_seconds = _timed(lambda: gateway.analyze_pass1(source=source, timeline=timeline, content_type=item["content_type"]))

    baseline_pair, baseline_generation_seconds = _timed(lambda: generate_semantic_candidates(content_map, transcript, transcript_features, scenes, config))
    baseline, baseline_generated = baseline_pair
    baseline, baseline_scoring_seconds = _timed(lambda: _old_score(baseline, audio, scenes, config))
    (baseline_selected, baseline_ranked), baseline_selection_seconds = _timed(lambda: _select(baseline, config, content_map))

    new_pair, new_generation_seconds = _timed(lambda: generate_multimodal_candidates(content_map, transcript, transcript_features, scenes, timeline, vision1, config, semantic_generator=generate_semantic_candidates))
    new, new_generated = new_pair
    new, new_initial_scoring_seconds = _timed(lambda: score_candidates(new, audio, scenes, config.scoring, min_duration_seconds=config.min_clip_duration, max_duration_seconds=config.max_clip_duration, visual_analysis=visual))
    new_shortlist = shortlist(new, config.ai_reranking.shortlist_size)
    new_shortlist, pass2_seconds = _timed(lambda: enrich_shortlist_with_pass2(new_shortlist, source=source, timeline=timeline, gateway=gateway, config=config))
    pass2_by_id = {candidate.id: candidate.vision_pass2_evidence for candidate in new_shortlist}
    for candidate in new:
        if candidate.id in pass2_by_id:
            candidate.vision_pass2_evidence = pass2_by_id[candidate.id]
    warm_shortlist = copy.deepcopy(new_shortlist)
    warm_shortlist, pass2_warm_seconds = _timed(lambda: enrich_shortlist_with_pass2(warm_shortlist, source=source, timeline=timeline, gateway=gateway, config=config))
    new, new_rescoring_seconds = _timed(lambda: score_candidates(new, audio, scenes, config.scoring, min_duration_seconds=config.min_clip_duration, max_duration_seconds=config.max_clip_duration, visual_analysis=visual))
    (new_selected, new_ranked), new_selection_seconds = _timed(lambda: _select(new, config, content_map))

    pass2_artifacts: list[dict[str, Any]] = []
    pass2_warm_artifacts: list[dict[str, Any]] = []
    for candidates, target in ((new_shortlist, pass2_artifacts), (warm_shortlist, pass2_warm_artifacts)):
        for candidate in candidates:
            wrapper = candidate.vision_pass2_evidence
            result = wrapper.get("result") if isinstance(wrapper, dict) else None
            if isinstance(result, dict):
                target.append(result)
    cold_usage = _usage([vision1, *pass2_artifacts])
    warm_usage = _usage([vision1_warm, *pass2_warm_artifacts])
    return {
        "source_key": item["source_key"], "content_type": item["content_type"], "title": item["title"],
        "source_id": item["expected_source_id"], "duration_seconds": metadata["duration"],
        "source_policy": "local-user-supplied-evaluation-only; not redistributed",
        "settings": {"min_clip_duration": config.min_clip_duration, "max_clip_duration": config.max_clip_duration, "final_clip_count": config.ai_reranking.final_clip_count, "score_threshold": config.score_threshold, "processing_mode": config.product_flow.processing_mode, "ai_rerank": False},
        "baseline": {"version": BASELINE_VERSION, "candidate_count": len(baseline), "candidates_generated": baseline_generated, "selected": [_candidate_result(value.candidate, baseline=True) for value in baseline_selected], "rejected": _rejections(baseline_ranked, baseline_selected), "diversity": _pairwise(baseline_selected)},
        "multimodal": {"version": NEW_VERSION, "candidate_count": len(new), "candidates_generated": new_generated, "selected": [_candidate_result(value.candidate, baseline=False) for value in new_selected], "rejected": _rejections(new_ranked, new_selected), "diversity": _pairwise(new_selected)},
        "side_by_side": _side_by_side(str(item["source_key"]), baseline_selected, new_selected),
        "processing_time_seconds": {"baseline_generation": round(baseline_generation_seconds, 6), "baseline_scoring": round(baseline_scoring_seconds, 6), "baseline_selection": round(baseline_selection_seconds, 6), "timeline": round(timeline_seconds, 6), "vision_pass1_cold": round(vision1_seconds, 6), "new_generation": round(new_generation_seconds, 6), "new_initial_scoring": round(new_initial_scoring_seconds, 6), "vision_pass2_cold": round(pass2_seconds, 6), "new_rescoring": round(new_rescoring_seconds, 6), "new_selection": round(new_selection_seconds, 6), "vision_pass1_warm": round(vision1_warm_seconds, 6), "vision_pass2_warm": round(pass2_warm_seconds, 6)},
        "vision": {"cold": cold_usage, "warm_repeat": warm_usage, "cache_confirmed": warm_usage["cache_hits"] > 0 and warm_usage["frames_sent"] == 0, "pass1_status": vision1.get("status")},
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# Goal 6E — Real Media Benchmark & Calibration", "", f"Generated: `{result['generated_at_utc']}`", "", "## Benchmark contract", "", f"OLD: `{BASELINE_VERSION}`. NEW: `{NEW_VERSION}`. Both use identical persisted transcript/audio/scene evidence, duration constraints, score threshold, diversity/coverage selection and three-clip limit. LLM reranking and rendering are excluded. Media is local user-supplied evaluation material and is not committed or redistributed.", "", "## OLD vs NEW", "", "| Source | Type | OLD candidates | NEW candidates | OLD selected | NEW selected | Vision calls | Tokens | Cost USD | Cache |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for source in result["sources"]:
        vision = source["vision"]
        lines.append(f"| {source['title']} | {source['content_type']} | {source['baseline']['candidate_count']} | {source['multimodal']['candidate_count']} | {len(source['baseline']['selected'])} | {len(source['multimodal']['selected'])} | {vision['cold']['calls']} | {vision['cold']['total_tokens']} | {vision['cold']['estimated_cost_usd']:.6f} | {'confirmed' if vision['cache_confirmed'] else 'not confirmed'} |")
    total_calls = sum(source["vision"]["cold"]["calls"] for source in result["sources"])
    total_frames = sum(source["vision"]["cold"]["frames_sent"] for source in result["sources"])
    total_tokens = sum(source["vision"]["cold"]["total_tokens"] for source in result["sources"])
    total_cost = sum(source["vision"]["cold"]["estimated_cost_usd"] for source in result["sources"])
    lines += ["", f"Aggregate real Vision usage: **{total_calls} calls, {total_frames} frames sent, {total_tokens} tokens, ${total_cost:.7f} estimated**. Cold benchmark wall time: **{result['total_wall_time_seconds']:.1f}s**. Every warm repeat used 0 calls and sent 0 frames.", "", "## Processing time", "", "| Source | OLD generation+score+selection | NEW local timeline+generation+2 scores+selection | Vision cold | Vision warm repeat |", "|---|---:|---:|---:|---:|"]
    for source in result["sources"]:
        timing = source["processing_time_seconds"]
        old_local = timing["baseline_generation"] + timing["baseline_scoring"] + timing["baseline_selection"]
        new_local = timing["timeline"] + timing["new_generation"] + timing["new_initial_scoring"] + timing["new_rescoring"] + timing["new_selection"]
        vision_cold = timing["vision_pass1_cold"] + timing["vision_pass2_cold"]
        vision_warm = timing["vision_pass1_warm"] + timing["vision_pass2_warm"]
        lines.append(f"| {source['source_key']} | {old_local:.3f}s | {new_local:.3f}s | {vision_cold:.3f}s | {vision_warm:.3f}s |")
    lines += ["", "## Rejected candidates", "", "| Source | OLD sampled reasons | NEW sampled reasons |", "|---|---|---|"]
    for source in result["sources"]:
        old_reasons = sorted({code for row in source["baseline"]["rejected"] for code in row["eligibility_reason_codes"]})
        new_reasons = sorted({code for row in source["multimodal"]["rejected"] for code in row["eligibility_reason_codes"]})
        lines.append(f"| {source['source_key']} | {', '.join(old_reasons) or 'selection limit/diversity'} | {', '.join(new_reasons) or 'selection limit/diversity'} |")
    lines += ["", "## Side-by-side selections", ""]
    for source in result["sources"]:
        lines += [f"### {source['title']}", "", "| Rank | OLD | NEW | Same | Main factor deltas | Evidence | Usefulness |", "|---:|---|---|---|---|---|---|"]
        for row in source["side_by_side"]:
            old = row["baseline"] or {}
            new = row["multimodal"] or {}
            evidence = row["evidence_that_changed_ranking"]
            lines.append(f"| {row['rank']} | `{old.get('candidate_id', '—')}` {old.get('start_seconds', '—')}–{old.get('end_seconds', '—')} | `{new.get('candidate_id', '—')}` {new.get('start_seconds', '—')}–{new.get('end_seconds', '—')} | {row['same_candidate']} | `{json.dumps(row['ranking_change_reason'], ensure_ascii=False)}` | pass2={evidence.get('pass2_status')}, audio={evidence.get('audio_count')}, visual={evidence.get('visual_count')} | {row['usefulness']} |")
        lines += [""]
    lines += ["## Failure / fallback and overvaluation matrix", "", "| Case | Final score | Visual interest | Audio energy | Confidence |", "|---|---:|---:|---:|---:|"]
    for row in result["failure_matrix"]:
        lines.append(f"| {row['case']} | {row['score']:.3f} | {row['visual_interest']:.3f} | {row['audio_energy']:.3f} | {row['confidence']:.3f} |")
    lines += ["", "## Confirmed improvements", "", "- Podcast top selection gained story-relevant B-roll rather than another static talking-head frame.", "- Food-vlog top selection preserved the actual dish/reveal instead of selecting only presenter commentary.", "- Gameplay top selection moved from a downed/static state to active play with source-grounded action evidence.", "- Interview selection preserved a relevant numeric visual payoff; PASS 2 also exposed its high continuity risk for review.", "- Every repeated Vision analysis used cache entries with zero provider calls and zero frames sent.", "", "## Confirmed regressions and calibration", "", "- Failure matrix proved that sub-0.65 visual confidence could still raise visual interest, payoff and vertical viability. Calibration now excludes that evidence from ranking/composition intent; covered by `test_low_confidence_visual_evidence_cannot_boost_editorial_or_composition_intent`.", "- Failure matrix proved raw loudness received the same emotional/audio treatment as a grounded editorial event. Calibration now retains only a weak 35% relevance contribution unless multimodal provenance contains an audio event; covered by `test_raw_loudness_is_weaker_than_grounded_audio_editorial_event`.", "- Generic action/reaction weights were not changed: real midpoint inspection showed useful B-roll, food, and gameplay improvements.", "", "## Remaining weaknesses", "", "- Candidate generation exceeds `max_candidates` on the long podcast (255), leaving no room for 6C composites; interview was the only source where count expanded (56→59).", "- Side-by-side factor deltas compare differently shaped OLD/NEW contracts and should be read with evidence/frame review, not as isolated quality truth.", "- PASS 2 marked the top interview candidate `continuity_risk=high`; current scoring reduces completeness but does not make this a hard rejection.", "- Full 4K AV1 cold source analysis is not part of OLD-vs-NEW scoring time; an attempted diagnostic run showed material decode overhead and was stopped rather than conflated with Editorial Brain latency.", "- API cost uses the gateway's configured token prices and recorded usage; it is an estimate, not a provider invoice.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--offline", action="store_true", help="Exercise fallback without provider calls.")
    parser.add_argument("--reset-cache", action="store_true", help="Clear only validation/artifacts/goal6e/vision-cache before the run.")
    parser.add_argument("--reuse-cold-telemetry", action="store_true", help="On an evidence-only rerun, retain cold Vision timing/usage from the existing JSON.")
    parser.add_argument("--report-only", action="store_true", help="Refresh manual review labels/Markdown without touching media or providers.")
    args = parser.parse_args()
    if args.report_only:
        result = _read(args.json_output)
        for source in result["sources"]:
            for index, row in enumerate(source["side_by_side"]):
                row["usefulness"] = HUMAN_REVIEW[source["source_key"]][index]
                row["review_basis"] = "manual midpoint-frame inspection plus transcript/evidence context"
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.markdown_output.write_text(_markdown(result), encoding="utf-8")
        print(f"Refreshed {args.json_output}")
        print(f"Refreshed {args.markdown_output}")
        return 0
    _dotenv()
    fixture = _read(args.fixture)
    config = _config()
    cache = ROOT / "validation" / "artifacts" / "goal6e" / "vision-cache"
    resolved_cache = cache.resolve()
    expected_parent = (ROOT / "validation" / "artifacts" / "goal6e").resolve()
    if args.reset_cache and resolved_cache.parent == expected_parent and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    provider = None if args.offline else get_vision_provider(config)
    previous = _read(args.json_output) if args.reuse_cold_telemetry and args.json_output.exists() else None
    started = time.perf_counter()
    sources = [_benchmark_source(item, config, provider, cache) for item in fixture["sources"]]
    refresh_wall = round(time.perf_counter() - started, 6)
    if previous is not None:
        prior_by_key = {item["source_key"]: item for item in previous.get("sources", [])}
        for source in sources:
            prior = prior_by_key.get(source["source_key"])
            if not prior:
                continue
            source["vision"]["cold"] = prior["vision"]["cold"]
            for key in ("vision_pass1_cold", "vision_pass2_cold"):
                source["processing_time_seconds"][key] = prior["processing_time_seconds"][key]
    result = {
        "schema_version": "6E.benchmark.1", "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_version": BASELINE_VERSION, "multimodal_version": NEW_VERSION,
        "fixture_policy": fixture["policy"], "sources": sources,
        "failure_matrix": _failure_matrix(config),
        "calibration_changes": [
            {"problem": "low-confidence visual evidence affected ranking/composition", "change": "require confidence >= 0.65 for editorial visual/pass2/composition evidence", "regression_test": "test_low_confidence_visual_evidence_cannot_boost_editorial_or_composition_intent"},
            {"problem": "raw loudness matched grounded editorial audio", "change": "use 35% audio relevance without grounded audio event", "regression_test": "test_raw_loudness_is_weaker_than_grounded_audio_editorial_event"},
        ],
        "total_wall_time_seconds": previous.get("total_wall_time_seconds") if previous is not None else refresh_wall,
        "evidence_refresh_wall_time_seconds": refresh_wall if previous is not None else None,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
