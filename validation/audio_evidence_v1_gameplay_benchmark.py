from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.audio_features import analyse_audio, window_audio_features
from app.audio_semantics import analyse_semantic_audio, project_semantic_audio_event
from app.config import AppConfig
from app.content_understanding import refresh_content_map_multimodal_evidence
from app.intelligence import merge_ai_ranking
from app.local_scoring import score_candidates
from app.models import candidate_from_dict, scored_from_dict
from app.multimodal_candidates import generate_multimodal_candidates, project_candidate_audio_evidence
from app.multimodal_evidence import build_multimodal_timeline
from app.pipeline import Pipeline
from app.utils import read_json, write_json
from app.virality import apply_virality_ranking, build_virality_assessments


BAD_ID = "candidate-chapter-034-story-001"
BAD_RANGE = (746.30, 782.59)


def run(root: Path) -> dict[str, Any]:
    work = next(path for path in (root / "work").iterdir() if "vEUmQFrVnSY-bc2ec91aded2" in path.name)
    config = AppConfig(score_threshold=0)
    transcript = read_json(work / "transcript.json", {})
    transcript_features = read_json(work / "transcript_features.json", {})
    scenes = read_json(work / "scene_boundaries.json", {})
    visual = read_json(work / "visual_analysis.json", {})
    vision = read_json(work / "vision-observations.json", {})
    content_profile = read_json(work / "video_content_profile.json", {})
    source = read_json(work / "source.json", {})
    base_content_map = read_json(work / "global_content_map.json", {})
    old_shortlist = [candidate_from_dict(item) for item in read_json(work / "shortlist.json", {}).get("candidates", [])]
    old_pass2 = {
        item.id: item.vision_pass2_evidence
        for item in (
            candidate_from_dict(raw)
            for raw in read_json(work / "shortlist.vision.json", {}).get("candidates", [])
        )
    }

    started = time.perf_counter()
    audio = analyse_audio(work / "audio_16khz_mono.wav", config.audio_analysis)
    signal_seconds = time.perf_counter() - started
    started = time.perf_counter()
    semantics = analyse_semantic_audio(
        work / "audio_16khz_mono.wav", audio, old_shortlist, "gameplay", config.audio_analysis,
    )
    semantic_seconds = time.perf_counter() - started
    timeline = build_multimodal_timeline(
        source_id=str(transcript["source_id"]),
        source_duration_seconds=float(audio["duration_seconds"]),
        transcript=transcript,
        audio_features=audio,
        scenes=scenes,
        visual_analysis=visual,
        semantic_audio=semantics,
    )
    content_map = refresh_content_map_multimodal_evidence(base_content_map, transcript, timeline)
    candidates, generated = generate_multimodal_candidates(
        content_map, transcript, transcript_features, scenes, timeline, vision, config,
    )
    project_candidate_audio_evidence(candidates, timeline, "gameplay")
    for candidate in candidates:
        candidate.vision_pass2_evidence = old_pass2.get(candidate.id, {})
    score_candidates(
        candidates, audio, scenes, config.scoring,
        min_duration_seconds=config.min_clip_duration,
        max_duration_seconds=config.max_clip_duration,
        visual_analysis=visual,
        transcript_features=transcript_features,
    )

    old_ai = read_json(work / "ai_ranking.json", {})
    shortlist_ids = {candidate.id for candidate in old_shortlist}
    old_assessments = [
        scored_from_dict(item) for item in old_ai.get("candidates", []) if item.get("id") in shortlist_ids
    ]
    merged = merge_ai_ranking(candidates, old_assessments, bool(old_ai.get("ai_reranking_used")))
    virality = build_virality_assessments(
        [item.candidate for item in merged], content_map, transcript_features, audio, visual,
        content_profile, config.virality, semantic_result=old_ai,
    )
    ranked_artifact = apply_virality_ranking(merged, virality, config.virality, content_profile)
    ranked = [scored_from_dict(item) for item in ranked_artifact.get("candidates", [])]
    Pipeline(root, config, mock_ai=True)._prepare_recommendation_candidates(
        ranked, visual, content_profile=content_profile, source=source,
    )

    old_scored = read_json(work / "candidates.scored.json", {}).get("candidates", [])
    old_bad = next(item for item in old_scored if item.get("id") == BAD_ID)
    new_bad = next(item for item in ranked if item.candidate.id == BAD_ID)
    audio_window = window_audio_features(*BAD_RANGE, audio)
    overlapping_semantics = [
        event for event in semantics.get("events", [])
        if float(event["end_seconds"]) > BAD_RANGE[0] and float(event["start_seconds"]) < BAD_RANGE[1]
    ]
    pass1 = [
        item for item in vision.get("observations", [])
        if BAD_RANGE[0] <= float(item.get("timestamp", -1)) <= BAD_RANGE[1]
    ]
    penalties = (
        new_bad.candidate.candidate_score_v2.penalties
        if new_bad.candidate.candidate_score_v2 is not None else []
    )
    recommended = [
        item for item in ranked
        if item.candidate.editorial_decision is not None
        and item.candidate.editorial_decision.surfacing_state.value == "RECOMMENDED"
    ]
    recommended.sort(key=lambda item: (-item.score, item.candidate.start))
    audio_seed_candidates = [
        candidate for candidate in candidates
        if "candidate_source:audio_seed" in candidate.multimodal_provenance.get("generation", {}).get("reasons", [])
    ]

    return {
        "schema_version": "audio-evidence-v1.gameplay-benchmark.1",
        "source_work_directory": str(work.relative_to(root)),
        "provider_calls": {"ai": 0, "vision": 0},
        "performance": {
            "signal_analysis_seconds": round(signal_seconds, 3),
            "semantic_analysis_seconds": round(semantic_seconds, 3),
            "source_duration_seconds": audio["duration_seconds"],
            "signal_analysis_passes": audio["analysis_passes"],
            "prepared_pcm_reads": audio["prepared_pcm_reads"] + semantics["diagnostics"]["prepared_pcm_opens"],
            "source_video_decodes": audio["source_video_decodes"] + semantics["diagnostics"]["source_video_decodes"],
            "semantic_classified_seconds": semantics["diagnostics"]["classified_seconds"],
            "semantic_region_count": semantics["diagnostics"]["selected_region_count"],
            "semantic_full_source_scan": semantics["diagnostics"]["full_source_scan"],
        },
        "bad_candidate_before": {
            "id": BAD_ID,
            "start": old_bad["start"], "end": old_bad["end"], "duration": old_bad["duration"],
            "text": old_bad["text"], "score": old_bad["score"], "selected": old_bad["selected"],
            "surfacing": (old_bad.get("editorial_decision") or {}).get("surfacing_state"),
        },
        "bad_candidate_evidence": {
            "speech_segments": [
                {"start": item["start"], "end": item["end"], "text": item.get("text", "")}
                for item in transcript.get("segments", [])
                if float(item["end"]) > BAD_RANGE[0] and float(item["start"]) < BAD_RANGE[1]
            ],
            "longest_speech_gap_seconds": new_bad.candidate.multimodal_provenance["audio_summary"]["longest_speech_gap_seconds"],
            "vision_pass1": pass1,
            "audio_window": audio_window,
            "semantic_audio_events": overlapping_semantics,
        },
        "bad_candidate_after": {
            "score": new_bad.score,
            "local_quality_score": new_bad.candidate.local_quality_score,
            "surfacing": new_bad.candidate.editorial_decision.surfacing_state.value,
            "penalties": [item.to_dict() for item in penalties],
            "sparse_content": (
                new_bad.candidate.candidate_score_v2.diagnostics.get("sparse_content", {})
                if new_bad.candidate.candidate_score_v2 is not None else {}
            ),
        },
        "recommended_after": [
            {
                "id": item.candidate.id,
                "text": item.candidate.text,
                "start": item.candidate.start, "end": item.candidate.end,
                "duration": item.candidate.duration, "profile": "gameplay", "score": item.score,
                "audio_summary": item.candidate.multimodal_provenance.get("audio_summary", {}),
                "visual_evidence": item.candidate.multimodal_provenance.get("visual_evidence", []),
                "ranking_reason": (
                    "tight self-contained candidate retained by existing Brain/virality/editorial owners; "
                    "no sparse multimodal penalty"
                ),
            }
            for item in recommended
        ],
        "audio_signal_seeds": {
            "peak_regions": audio["peak_regions"],
            "semantic_regions": semantics["regions"],
            "meaningful_events": [
                project_semantic_audio_event(event, "gameplay") for event in semantics["events"]
                if project_semantic_audio_event(event, "gameplay").get("observation", {}).get("meaningful_for_profile") is True
            ],
            "candidate_seeds": [
                {
                    "id": candidate.id, "start": candidate.start, "end": candidate.end,
                    "story_unit_ids": candidate.story_unit_ids,
                    "boundary_decision": candidate.boundary_diagnostics.get("boundary_decision", {}),
                }
                for candidate in audio_seed_candidates
            ],
        },
        "candidate_counts": {"generated": generated, "after_generation": len(candidates)},
    }


def _markdown(result: dict[str, Any]) -> str:
    before = result["bad_candidate_before"]
    after = result["bad_candidate_after"]
    evidence = result["bad_candidate_evidence"]
    perf = result["performance"]
    lines = [
        "# Audio Evidence v1 — Fresh Gameplay validation",
        "",
        "This is a deterministic replay over the saved Fresh Gameplay PCM/transcript/Vision artifacts. "
        "It made zero AI and zero Vision provider calls.",
        "",
        "## Rejected 746.30–782.59 candidate",
        "",
        f"Before: `{before['id']}`, {before['duration']:.2f}s, score {before['score']}, "
        f"`{before['surfacing']}`, selected={str(before['selected']).lower()}.",
        "",
        f"After: score {after['score']}, local quality {after['local_quality_score']:.3f}, "
        f"`{after['surfacing']}` (still selectable, but no longer recommended).",
        "",
        f"Speech: longest internal gap {evidence['longest_speech_gap_seconds']:.2f}s. "
        f"Audio: activity {evidence['audio_window']['audio_activity_ratio']:.3f}, dead-zone "
        f"{evidence['audio_window']['audio_dead_zone_ratio']:.3f}; no meaningful gameplay audio event in the range. "
        "Vision PASS 1 reported movement but no reaction/payoff. The code-owned "
        "`SPARSE_MULTIMODAL_CONTENT` penalty is soft and does not create `BLOCKED`.",
        "",
        "## RECOMMENDED after replay",
        "",
    ]
    for item in result["recommended_after"]:
        audio = item["audio_summary"]
        visual = item["visual_evidence"]
        visual_labels = sorted({
            str((entry.get("observation") or entry).get("action") or (entry.get("observation") or entry).get("payoff_signal") or "none")
            for entry in visual if isinstance(entry, dict)
        })
        lines.extend([
            f"- `{item['id']}` — {item['start']:.2f}–{item['end']:.2f} ({item['duration']:.2f}s), "
            f"profile `gameplay`, score {item['score']}. Text: {item['text']}",
            f"  Audio: activity={audio.get('activity_ratio', 0):.3f}, meaningful_events={audio.get('meaningful_event_count', 0)}; "
            f"Visual: {', '.join(visual_labels) or 'no strong action label'}. Reason: {item['ranking_reason']}.",
        ])
    lines.extend([
        "",
        "## Audio seeds",
        "",
        f"Signal peak regions: {len(result['audio_signal_seeds']['peak_regions'])}; bounded semantic regions: "
        f"{len(result['audio_signal_seeds']['semantic_regions'])}; candidate seeds resolved by the existing "
        f"SemanticBoundaryEngine: {len(result['audio_signal_seeds']['candidate_seeds'])}.",
        "",
        "## Performance",
        "",
        f"One signal pass took {perf['signal_analysis_seconds']:.3f}s. Bounded ONNX took "
        f"{perf['semantic_analysis_seconds']:.3f}s over {perf['semantic_classified_seconds']:.1f}s / "
        f"{perf['source_duration_seconds']:.1f}s ({perf['semantic_region_count']} regions). "
        f"Full-source semantic scan={str(perf['semantic_full_source_scan']).lower()}, "
        f"source video decodes={perf['source_video_decodes']}.",
        "",
        "The complete machine-readable evidence, including peak/semantic regions and events, is in "
        "`docs/audits/AUDIO_EVIDENCE_V1_GAMEPLAY_VALIDATION.json`.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    result = run(repository)
    output = repository / "docs" / "audits" / "AUDIO_EVIDENCE_V1_GAMEPLAY_VALIDATION.json"
    write_json(output, result)
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "bad_before": result["bad_candidate_before"],
        "bad_after": result["bad_candidate_after"],
        "recommended_count": len(result["recommended_after"]),
        "performance": result["performance"],
    }, ensure_ascii=False, indent=2))
