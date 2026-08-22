"""Multimodal candidate generation and shortlist-only Vision PASS 2 enrichment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Protocol

from app.ai import sanitize_api_error
from app.audio_semantics import project_semantic_audio_event
from app.content_understanding import (
    GlobalContentMap,
    SemanticBoundaryEngine,
    StoryUnit,
    build_semantic_candidate,
    generate_semantic_candidates,
)
from app.models import Candidate
from app.multimodal_evidence import audio_summary_for_range, validate_multimodal_timeline
from app.utils import stable_text_hash
from app.vision_intelligence import build_pass2_request


CANDIDATE_PROVENANCE_SCHEMA_VERSION = "6C.2"
PASS2_EVIDENCE_SCHEMA_VERSION = "6C.pass2-evidence.1"


class Pass2Gateway(Protocol):
    def analyze_pass2(
        self, *, source: Path, timeline: dict[str, Any], request: dict[str, Any],
    ) -> dict[str, Any]: ...


def generate_multimodal_candidates(
    content_map_data: dict[str, Any],
    transcript: dict[str, Any],
    transcript_features: dict[str, Any],
    scenes: dict[str, Any],
    timeline: dict[str, Any],
    vision_pass1: dict[str, Any],
    config: Any,
    semantic_generator: Callable[..., tuple[list[Candidate], int]] = generate_semantic_candidates,
    audio_profile_id: str | None = None,
) -> tuple[list[Candidate], int]:
    """Extend semantic candidates with grounded audio/visual seeds and linked ranges."""

    validate_multimodal_timeline(timeline)
    content_map = GlobalContentMap.from_dict(content_map_data, transcript)
    baseline, baseline_count = semantic_generator(
        content_map_data, transcript, transcript_features, scenes, config,
    )
    stories = {item.story_unit_id: item for item in content_map.story_units}
    signals = _strong_signals(timeline, vision_pass1, config.candidate_generation.multimodal_min_confidence)

    for candidate in baseline:
        if not candidate.story_unit_ids and candidate.story_unit_id:
            candidate.story_unit_ids = [candidate.story_unit_id]
        candidate_signals = _signals_for_range(signals, candidate.start, candidate.end)
        unit = stories.get(str(candidate.story_unit_id or ""))
        transcript_strength = _transcript_strength([unit] if unit is not None else [])
        kind = _candidate_kind(transcript_strength, candidate_signals)
        score = _initial_score(transcript_strength, candidate_signals)
        reasons = _generation_reasons(kind, candidate_signals, expanded=False)
        _attach_provenance(
            candidate, timeline, vision_pass1, candidate_signals,
            kind=kind, initial_score=score, reasons=reasons,
            audio_profile_id=audio_profile_id,
            original_ranges=[{"story_unit_id": value, "start": stories[value].start, "end": stories[value].end}
                             for value in candidate.story_unit_ids if value in stories],
        )

    audio_seeds = _audio_seed_candidates(
        content_map.story_units, signals, transcript, transcript_features, scenes, timeline,
        vision_pass1, config,
        audio_profile_id,
    )
    composites = _composite_candidates(
        content_map.story_units, signals, transcript, transcript_features, scenes, timeline,
        vision_pass1, config,
        audio_profile_id,
    )
    available = max(0, int(config.candidate_generation.max_candidates) - len(baseline))
    audio_seeds = audio_seeds[:min(available, int(config.audio_analysis.semantic_max_peak_regions))]
    available -= len(audio_seeds)
    composites = composites[:min(available, int(config.candidate_generation.multimodal_max_composite_candidates))]
    return baseline + audio_seeds + composites, baseline_count + len(audio_seeds) + len(composites)


def enrich_shortlist_with_pass2(
    candidates: list[Candidate],
    *,
    source: Path,
    timeline: dict[str, Any],
    gateway: Pass2Gateway,
    config: Any,
) -> list[Candidate]:
    """Run deep vision only for a budget-bounded prefix of the existing shortlist."""

    validate_multimodal_timeline(timeline)
    limit = _pass2_candidate_limit(config)
    admitted = candidates[:limit]
    for candidate in admitted:
        try:
            request = build_pass2_request(
                candidate_id=candidate.id,
                window_start=candidate.start,
                window_end=candidate.end,
                anchors=_pass2_anchors(candidate),
                timeline=timeline,
                max_frames=int(config.vision.pass2_max_frames),
            )
            result = gateway.analyze_pass2(source=source, timeline=timeline, request=request)
            candidate.vision_pass2_evidence = {
                "schema_version": PASS2_EVIDENCE_SCHEMA_VERSION,
                "status": str(result.get("status") or "completed"),
                "reason": None,
                "result": result,
            }
        except Exception as error:  # PASS 2 is optional evidence, never a candidate gate.
            candidate.vision_pass2_evidence = {
                "schema_version": PASS2_EVIDENCE_SCHEMA_VERSION,
                "status": "skipped",
                "reason": sanitize_api_error(error),
                "result": None,
            }
    for candidate in candidates[limit:]:
        candidate.vision_pass2_evidence = {
            "schema_version": PASS2_EVIDENCE_SCHEMA_VERSION,
            "status": "not_requested",
            "reason": "pass2_shortlist_budget_limit",
            "result": None,
        }
    return candidates


def project_candidate_audio_evidence(
    candidates: list[Candidate], timeline: dict[str, Any], profile_id: str,
) -> list[Candidate]:
    """Attach profile-aware audio interpretation without changing candidate ranges."""

    validate_multimodal_timeline(timeline)
    for candidate in candidates:
        provenance = candidate.multimodal_provenance
        if not isinstance(provenance, dict):
            provenance = {}
            candidate.multimodal_provenance = provenance
        provenance["audio_evidence"] = _candidate_audio_evidence(
            timeline, candidate.start, candidate.end, profile_id,
        )
        provenance["audio_summary"] = audio_summary_for_range(
            timeline, candidate.start, candidate.end, profile_id,
        )
        provenance["audio_profile_id"] = profile_id
    return candidates


def refresh_candidate_timeline_evidence(
    candidates: list[Candidate], timeline: dict[str, Any],
) -> list[Candidate]:
    """Rebind unchanged candidate ranges to the enriched source timeline."""

    validate_multimodal_timeline(timeline)
    for candidate in candidates:
        provenance = candidate.multimodal_provenance
        if not isinstance(provenance, dict):
            provenance = {}
            candidate.multimodal_provenance = provenance
        provenance["analysis_run_id"] = timeline["analysis_run_id"]
        provenance["transcript_evidence"] = [
            item for item in timeline.get("transcript_events", [])
            if _overlaps(item, candidate.start, candidate.end)
        ]
        provenance["audio_evidence"] = _candidate_audio_evidence(
            timeline, candidate.start, candidate.end, None,
        )
        provenance["audio_summary"] = audio_summary_for_range(
            timeline, candidate.start, candidate.end,
        )
        provenance["keyframe_evidence"] = [
            item for item in timeline.get("keyframes", [])
            if candidate.start <= float(item["time_seconds"]) <= candidate.end
        ]
    return candidates


def _strong_signals(
    timeline: dict[str, Any], vision_pass1: dict[str, Any], minimum: float,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for event in timeline.get("audio_event_map", []):
        event_type = str(event.get("event_type") or "")
        confidence = float(event.get("confidence", 0))
        observation = event.get("observation", {}) if isinstance(event.get("observation"), dict) else {}
        if confidence < minimum:
            continue
        if event_type == "background_music":
            continue
        if event_type == "semantic_audio_event":
            # Profile-neutral classifier output is Brain evidence. Candidate
            # boundaries remain owned by source-stable peaks/signals.
            continue
        elif event_type == "reaction_label":
            roles = ["reaction"]
        elif event_type == "peak_region" and observation.get("candidate_seed") is True:
            roles = ["attention"]
        elif event_type == "emphasis":
            roles = ["attention"]
        else:
            continue
        for role in roles:
            signals.append(_signal(
                "audio", role,
                float(event["start_seconds"]), float(event["end_seconds"]), confidence,
                f"timeline:{event['event_id']}", event,
            ))
    for event in timeline.get("visual_event_map", []):
        confidence = float(event.get("confidence", 0))
        if confidence < minimum or event.get("event_type") != "subject_observation":
            continue
        observation = event.get("observation", {})
        motion = observation.get("motion_action", {}) if isinstance(observation, dict) else {}
        objects = observation.get("objects_persons", {}) if isinstance(observation, dict) else {}
        roles: list[str] = []
        if motion.get("motion_evidence_between_samples") is True or motion.get("gesture_observed") is True:
            roles.append("action")
        if objects.get("object_is_active_target") is True:
            roles.append("payoff")
        for role in roles:
            signals.append(_signal(
                "visual", role, float(event["start_seconds"]), float(event["end_seconds"]),
                confidence, f"timeline:{event['event_id']}", event,
            ))
    for observation in vision_pass1.get("observations", []):
        if not isinstance(observation, dict) or observation.get("origin") not in {"provider", "cache"}:
            continue
        confidence = float(observation.get("confidence", 0))
        if confidence < minimum:
            continue
        roles = []
        if observation.get("action") in {"gesture", "movement", "demonstration", "interaction"}:
            roles.append("action")
        if observation.get("reaction") not in {None, "none", "unknown"}:
            roles.append("reaction")
        if observation.get("payoff_signal") in {"reveal", "result", "resolution"}:
            roles.append("payoff")
        if observation.get("primary_subject") in {"object", "screen"} and not roles:
            roles.append("action")
        for role in roles:
            timestamp = float(observation.get("timestamp", 0))
            signals.append(_signal(
                "visual", role, timestamp, timestamp, confidence,
                f"vision_pass1:{observation.get('keyframe_id')}", observation,
            ))
    return sorted(signals, key=lambda item: (item["time"], item["modality"], item["role"], item["source_ref"]))


def _audio_seed_candidates(
    stories: list[StoryUnit], signals: list[dict[str, Any]], transcript: dict[str, Any],
    transcript_features: dict[str, Any], scenes: dict[str, Any], timeline: dict[str, Any],
    vision_pass1: dict[str, Any], config: Any,
    audio_profile_id: str | None,
) -> list[Candidate]:
    """Resolve bounded audio proposals through the existing SemanticBoundaryEngine."""

    audio = [
        item for item in signals
        if item["modality"] == "audio"
        and item["evidence"].get("event_type") in {"peak_region", "semantic_audio_event"}
    ]
    engine = SemanticBoundaryEngine(config.content_understanding)
    ordered_stories = sorted(stories, key=lambda item: (item.start, item.end, item.story_unit_id))
    result: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    link_gap = float(config.candidate_generation.multimodal_link_gap_seconds)
    for signal in sorted(audio, key=lambda item: (-item["confidence"], item["time"])):
        center = (signal["start"] + signal["end"]) / 2
        nearby = [
            unit for unit in ordered_stories
            if unit.start <= signal["end"] + link_gap and unit.end >= signal["start"] - link_gap
        ]
        if not nearby:
            continue
        unit = min(nearby, key=lambda item: (0.0 if item.start <= center <= item.end else min(abs(center - item.start), abs(center - item.end)), -item.standalone_score, item.story_unit_id))
        seed_id = str((signal["evidence"].get("observation") or {}).get("region_id") or signal["evidence"].get("event_id"))
        key = (unit.story_unit_id, seed_id)
        if key in seen:
            continue
        seen.add(key)
        candidate_id = f"candidate-audio-{stable_text_hash('|'.join(key))[:12]}"
        candidate = build_semantic_candidate(
            [unit], transcript, transcript_features, scenes, engine, candidate_id=candidate_id,
        )
        _attach_provenance(
            candidate, timeline, vision_pass1, _signals_for_range(signals, candidate.start, candidate.end),
            kind="audio", initial_score=_initial_score(_transcript_strength([unit]), [signal]),
            reasons=[
                "candidate_source:audio_seed", "bounded_audio_peak_seed",
                f"audio_seed_event:{signal['evidence'].get('event_type')}",
                "resolved_by_existing_semantic_boundary_engine",
            ],
            audio_profile_id=audio_profile_id,
            original_ranges=[{"story_unit_id": unit.story_unit_id, "start": unit.start, "end": unit.end}],
        )
        candidate.reason = "Bounded audio seed resolved by the existing SemanticBoundaryEngine."
        result.append(candidate)
    return result


def _composite_candidates(
    stories: list[StoryUnit], signals: list[dict[str, Any]], transcript: dict[str, Any],
    transcript_features: dict[str, Any], scenes: dict[str, Any], timeline: dict[str, Any],
    vision_pass1: dict[str, Any], config: Any,
    audio_profile_id: str | None,
) -> list[Candidate]:
    proposals: list[tuple[float, list[StoryUnit], list[dict[str, Any]]]] = []
    ordered = sorted(stories, key=lambda item: (item.start, item.end, item.story_unit_id))
    maximum_duration = float(config.candidate_generation.max_duration_seconds)
    link_gap = float(config.candidate_generation.multimodal_link_gap_seconds)
    for first, second in zip(ordered, ordered[1:]):
        if first.chapter_id != second.chapter_id or second.start - first.end > link_gap:
            continue
        if second.end - first.start > maximum_duration:
            continue
        first_signals = _signals_for_range(signals, first.start, first.end)
        second_signals = _signals_for_range(signals, second.start, second.end)
        linked = first_signals + second_signals
        roles = {item["role"] for item in linked}
        modalities = {item["modality"] for item in linked}
        has_sequence = bool(first_signals and second_signals and (len(roles) >= 2 or len(modalities) >= 2))
        speech_plus_visible_payoff = bool(
            _transcript_strength([first]) >= 0.55
            and any(item["role"] in {"action", "reaction", "payoff"} for item in second_signals)
            and any(item["role"] in {"reaction", "payoff"} for item in linked)
        )
        if not (has_sequence or speech_plus_visible_payoff):
            continue
        score = _initial_score(_transcript_strength([first, second]), linked) + 0.08
        proposals.append((min(1.0, score), [first, second], linked))
    proposals.sort(key=lambda item: (-item[0], item[1][0].start, item[1][-1].end))

    engine = SemanticBoundaryEngine(config.content_understanding)
    result: list[Candidate] = []
    seen_units: set[tuple[str, ...]] = set()
    for score, units, linked in proposals:
        unit_ids = tuple(unit.story_unit_id for unit in units)
        if unit_ids in seen_units:
            continue
        seen_units.add(unit_ids)
        digest = stable_text_hash("|".join(unit_ids))[:12]
        candidate = build_semantic_candidate(
            units, transcript, transcript_features, scenes, engine,
            candidate_id=f"candidate-mm-{digest}",
        )
        reasons = _generation_reasons("multimodal", linked, expanded=True)
        _attach_provenance(
            candidate, timeline, vision_pass1, linked, kind="multimodal",
            initial_score=score, reasons=reasons,
            audio_profile_id=audio_profile_id,
            original_ranges=[{"story_unit_id": unit.story_unit_id, "start": unit.start, "end": unit.end} for unit in units],
        )
        candidate.reason = "Multimodal evidence linked adjacent StoryUnits; boundaries remain SemanticBoundaryEngine decisions."
        result.append(candidate)
    return result


def _attach_provenance(
    candidate: Candidate, timeline: dict[str, Any], vision_pass1: dict[str, Any],
    signals: list[dict[str, Any]], *, kind: str, initial_score: float,
    reasons: list[str], original_ranges: list[dict[str, Any]], audio_profile_id: str | None,
) -> None:
    start, end = candidate.start, candidate.end
    candidate.candidate_kind = kind
    candidate.multimodal_provenance = {
        "schema_version": CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        "analysis_run_id": timeline["analysis_run_id"],
        "story_unit_ids": list(candidate.story_unit_ids),
        "transcript_evidence": [
            item for item in timeline.get("transcript_events", []) if _overlaps(item, start, end)
        ],
        "audio_evidence": _candidate_audio_evidence(timeline, start, end, audio_profile_id),
        "audio_summary": audio_summary_for_range(timeline, start, end, audio_profile_id),
        "visual_evidence": [
            item["evidence"] for item in signals if item["modality"] == "visual"
        ],
        "keyframe_evidence": [
            item for item in timeline.get("keyframes", []) if start <= float(item["time_seconds"]) <= end
        ],
        "vision_pass1_status": str(vision_pass1.get("status") or "unavailable"),
        "generation": {
            "candidate_kind": kind,
            "initial_filter_score": round(initial_score, 6),
            "initial_filter_passed": True,
            "reasons": reasons,
            "anchors": _anchors_from_signals(signals, start, end),
            "range_expanded": len(candidate.story_unit_ids) > 1,
            "original_story_unit_ranges": original_ranges,
            "resolved_candidate_range": {"start": start, "end": end},
        },
    }


def _candidate_audio_evidence(
    timeline: dict[str, Any], start: float, end: float, profile_id: str | None,
) -> list[dict[str, Any]]:
    allowed = {
        "reaction_label", "emphasis", "relative_spike", "burst", "peak_region",
        "semantic_audio_event", "background_music",
    }
    result: list[dict[str, Any]] = []
    for item in timeline.get("audio_event_map", []):
        if item.get("event_type") not in allowed or not _overlaps(item, start, end):
            continue
        if item.get("event_type") in {"semantic_audio_event", "background_music"} and profile_id:
            result.append(project_semantic_audio_event(item, profile_id))
        else:
            result.append(item)
    return result


def _candidate_kind(transcript_strength: float, signals: list[dict[str, Any]]) -> str:
    modalities = {item["modality"] for item in signals}
    if len(modalities) >= 2 or (transcript_strength >= 0.55 and modalities):
        return "multimodal"
    if "visual" in modalities:
        return "visual"
    if "audio" in modalities:
        return "audio"
    return "transcript"


def _transcript_strength(units: list[StoryUnit]) -> float:
    if not units:
        return 0.0
    return max(
        min(1.0, unit.standalone_score * 0.55 + unit.completeness_score * 0.30 + unit.information_density * 0.15)
        for unit in units
    )


def _initial_score(transcript_strength: float, signals: list[dict[str, Any]]) -> float:
    audio = max((item["confidence"] for item in signals if item["modality"] == "audio"), default=0.0)
    visual = max((item["confidence"] for item in signals if item["modality"] == "visual"), default=0.0)
    role_bonus = min(0.18, 0.06 * len({item["role"] for item in signals}))
    return min(1.0, max(transcript_strength, audio * 0.78, visual * 0.82) + role_bonus)


def _generation_reasons(kind: str, signals: list[dict[str, Any]], *, expanded: bool) -> list[str]:
    reasons = [f"candidate_source:{kind}"]
    roles = sorted({item["role"] for item in signals})
    modalities = sorted({item["modality"] for item in signals})
    if modalities:
        reasons.append(f"strong_modalities:{','.join(modalities)}")
    if roles:
        reasons.append(f"editorial_roles:{','.join(roles)}")
    if expanded:
        reasons.append("range_expanded_to_preserve_linked_action_reaction_or_payoff")
    if kind == "transcript":
        reasons.append("grounded_story_unit_transcript_window")
    return reasons


def _pass2_anchors(candidate: Candidate) -> dict[str, float | None]:
    generation = candidate.multimodal_provenance.get("generation", {})
    anchors = generation.get("anchors", {}) if isinstance(generation, dict) else {}
    start, end = candidate.start, candidate.end
    duration = end - start
    return {
        "hook": _bounded_anchor(anchors.get("hook"), start, start),
        "action": _bounded_anchor(anchors.get("action"), start, start + duration * 0.35),
        "reaction": _bounded_anchor(anchors.get("reaction"), start, start + duration * 0.65),
        "payoff": _bounded_anchor(anchors.get("payoff"), start, end),
    }


def _anchors_from_signals(signals: list[dict[str, Any]], start: float, end: float) -> dict[str, float]:
    by_role = {role: [item["time"] for item in signals if item["role"] == role] for role in ("hook", "action", "reaction", "payoff")}
    return {
        "hook": min(by_role["hook"], default=start),
        "action": min(by_role["action"], default=start + (end - start) * 0.35),
        "reaction": min(by_role["reaction"], default=start + (end - start) * 0.65),
        "payoff": max(by_role["payoff"], default=end),
    }


def _pass2_candidate_limit(config: Any) -> int:
    requested = int(config.vision.pass2_max_candidates)
    mode = str(config.product_flow.processing_mode)
    if (
        mode == "fast" or requested <= 0 or not config.vision.enabled
        or not bool(getattr(config, "optional_visual_features", False))
    ):
        return 0
    if mode == "maximum":
        frames = int(config.vision.maximum_max_frames)
        calls = int(config.vision.maximum_max_calls)
        tokens = int(config.vision.maximum_max_tokens)
        cost = float(config.vision.maximum_max_estimated_cost)
    else:
        frames = int(config.vision.standard_max_frames)
        calls = int(config.vision.standard_max_calls)
        tokens = int(config.vision.standard_max_tokens)
        cost = float(config.vision.standard_max_estimated_cost)
    per_candidate_frames = int(config.vision.pass2_max_frames)
    per_candidate_calls = math.ceil(per_candidate_frames / 3)
    per_candidate_tokens = (
        per_candidate_calls * (int(config.vision.prompt_input_tokens) + int(config.vision.max_output_tokens_per_call))
        + per_candidate_frames * int(config.vision.high_detail_input_tokens_per_frame)
    )
    price_in = config.ai.input_token_price
    price_out = config.ai.output_token_price
    per_candidate_cost = 0.0
    if price_in is not None and price_out is not None:
        per_candidate_cost = (
            (per_candidate_calls * int(config.vision.prompt_input_tokens)
             + per_candidate_frames * int(config.vision.high_detail_input_tokens_per_frame)) * float(price_in)
            + per_candidate_calls * int(config.vision.max_output_tokens_per_call) * float(price_out)
        )
    limits = [requested, frames // max(1, per_candidate_frames), calls // max(1, per_candidate_calls)]
    if per_candidate_tokens > 0:
        limits.append(max(1, tokens // per_candidate_tokens) if tokens > 0 else 0)
    if per_candidate_cost > 0:
        limits.append(max(1, int(cost // per_candidate_cost)) if cost > 0 else 0)
    return max(0, min(limits))


def _signal(
    modality: str, role: str, start: float, end: float, confidence: float,
    source_ref: str, evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "modality": modality, "role": role, "time": start, "start": start, "end": end,
        "confidence": confidence, "source_ref": source_ref, "evidence": evidence,
    }


def _signals_for_range(signals: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [item for item in signals if item["start"] <= end and item["end"] >= start]


def _overlaps(item: dict[str, Any], start: float, end: float) -> bool:
    item_start = float(item.get("start_seconds", 0))
    item_end = float(item.get("end_seconds", item_start))
    return item_start <= end and item_end >= start


def _bounded_anchor(value: Any, start: float, fallback: float) -> float:
    try:
        return max(start, float(value))
    except (TypeError, ValueError):
        return fallback
