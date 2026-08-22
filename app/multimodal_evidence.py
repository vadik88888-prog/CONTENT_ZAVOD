"""Grounded, source-scoped multimodal evidence for editorial intelligence.

This module is deliberately provider-free.  It normalises evidence already
produced by transcript, PCM audio, scene detection and optional visual subject
analysis into one persisted source timeline.  Sparse keyframes are a selection
plan for possible future vision analysis; building the plan never extracts or
sends a frame.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from app.audio_semantics import project_semantic_audio_event
from app.utils import stable_text_hash


MULTIMODAL_TIMELINE_SCHEMA_VERSION = "audio-evidence.1"
MULTIMODAL_ANALYSIS_VERSION = "multimodal.audio-v1.1"
STORY_UNIT_EVIDENCE_SCHEMA_VERSION = "6A.story-range.1"
TIME_BASE = {
    "unit": "seconds",
    "origin": "source_media_start",
    "interval_convention": "start_inclusive_end_exclusive",
}

_TIMELINE_FIELDS = frozenset({
    "schema_version", "analysis_version", "source_id", "analysis_run_id",
    "source_duration_seconds", "time_base", "input_fingerprints", "transcript_events",
    "audio_event_map", "visual_event_map", "scenes", "keyframes", "diagnostics", "provenance",
})
_EVENT_FIELDS = frozenset({
    "event_id", "event_type", "start_seconds", "end_seconds", "confidence",
    "observation", "provenance",
})
_REACTION_LABEL = re.compile(
    r"[\[(](laughter|laughs?|applause|reaction|смех|сме[её]тся|аплодисменты|реакция)[\])]",
    re.IGNORECASE,
)


class MultimodalEvidenceError(ValueError):
    """A persisted multimodal artifact is invalid or identity-mismatched."""


def multimodal_analysis_run_id(
    source_id: str,
    transcript: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    visual_analysis: dict[str, Any],
    semantic_audio: dict[str, Any] | None = None,
) -> str:
    """Return the immutable analysis identity shared by cache reusers."""

    fingerprint = _fingerprints(transcript, audio_features, scenes, visual_analysis, semantic_audio)
    payload = {
        "schema_version": MULTIMODAL_TIMELINE_SCHEMA_VERSION,
        "analysis_version": MULTIMODAL_ANALYSIS_VERSION,
        "source_id": source_id,
        "input_fingerprints": fingerprint,
    }
    return f"multimodal-{stable_text_hash(_canonical(payload))[:20]}"


def build_multimodal_timeline(
    *,
    source_id: str,
    source_duration_seconds: float,
    transcript: dict[str, Any],
    audio_features: dict[str, Any],
    scenes: dict[str, Any],
    visual_analysis: dict[str, Any],
    semantic_audio: dict[str, Any] | None = None,
    analysis_run_id: str | None = None,
) -> dict[str, Any]:
    """Build one validated timeline solely from existing local evidence."""

    duration = max(0.0, _number(source_duration_seconds) or 0.0)
    fingerprints = _fingerprints(transcript, audio_features, scenes, visual_analysis, semantic_audio)
    expected_run_id = multimodal_analysis_run_id(
        source_id, transcript, audio_features, scenes, visual_analysis, semantic_audio,
    )
    if analysis_run_id is not None and analysis_run_id != expected_run_id:
        raise MultimodalEvidenceError("Multimodal analysis_run_id does not match its input evidence.")

    transcript_events = _transcript_events(transcript, duration)
    audio_events = _audio_events(transcript_events, audio_features, semantic_audio or {}, duration)
    scene_intervals, scene_events, scene_status = _scene_evidence(scenes, duration)
    subject_events, subject_status = _subject_events(visual_analysis, duration)
    visual_events = sorted(
        [*scene_events, *subject_events], key=lambda item: (item["start_seconds"], item["event_id"]),
    )
    keyframes, keyframe_limit = _select_keyframes(
        duration, transcript_events, audio_events, visual_events, scene_intervals,
    )
    evidence_diagnostics = _evidence_diagnostics(
        transcript, transcript_events, audio_features, audio_events, visual_analysis,
        subject_events, scenes, scene_intervals, scene_status, subject_status,
    )
    available = [name for name, value in evidence_diagnostics.items() if value["status"] in {"available", "partial"}]
    missing = [
        {"modality": name, "reason": value.get("reason") or "evidence_unavailable"}
        for name, value in evidence_diagnostics.items() if value["status"] == "missing"
    ]
    result = {
        "schema_version": MULTIMODAL_TIMELINE_SCHEMA_VERSION,
        "analysis_version": MULTIMODAL_ANALYSIS_VERSION,
        "source_id": source_id,
        "analysis_run_id": expected_run_id,
        "source_duration_seconds": round(duration, 3),
        "time_base": dict(TIME_BASE),
        "input_fingerprints": fingerprints,
        "transcript_events": transcript_events,
        "audio_event_map": audio_events,
        "visual_event_map": visual_events,
        "scenes": scene_intervals,
        "keyframes": keyframes,
        "diagnostics": {
            "scenes": {
                "status": scene_status,
                "count": len(scene_intervals),
                "boundary_count": len(scene_events),
            },
            "keyframes": {
                "status": "planned" if keyframes else "unavailable",
                "count": len(keyframes),
                "limit": keyframe_limit,
                "strategy": "scene_boundaries+measured_motion+grounded_relevance",
                "analyzed_count": sum(item["analysis_status"] == "existing_visual_evidence" for item in keyframes),
                "future_vision_api_eligible_count": sum(bool(item["future_vision_api_eligible"]) for item in keyframes),
            },
            "evidence": evidence_diagnostics,
            "available_evidence": available,
            "missing_evidence": missing,
            "external_vision_api_calls": 0,
        },
        "provenance": [
            _provenance("transcript.json", "segments/words", "direct_timestamped_observation"),
            _provenance("audio_features.json", "energy_frames/silence_intervals", "local_pcm_measurement"),
            _provenance("audio_semantic_events.json", "bounded_regions/events", "bounded_yamnet_onnx_inference"),
            _provenance("scene_boundaries.json", "boundaries", "local_ffmpeg_scene_detection"),
            _provenance("visual_analysis.json", "subject_keyframes", "existing_analysis_only"),
        ],
    }
    return validate_multimodal_timeline(
        result, expected_source_id=source_id, expected_analysis_run_id=expected_run_id,
    )


def validate_multimodal_timeline(
    data: dict[str, Any],
    *,
    expected_source_id: str | None = None,
    expected_analysis_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate schema, identities and source-time bounds on a cache read/write."""

    if not isinstance(data, dict) or set(data) != _TIMELINE_FIELDS:
        raise MultimodalEvidenceError("Multimodal timeline has an invalid top-level schema.")
    if data.get("schema_version") != MULTIMODAL_TIMELINE_SCHEMA_VERSION:
        raise MultimodalEvidenceError("Unsupported multimodal timeline schema version.")
    if data.get("analysis_version") != MULTIMODAL_ANALYSIS_VERSION:
        raise MultimodalEvidenceError("Unsupported multimodal analysis version.")
    source_id = str(data.get("source_id") or "")
    analysis_run_id = str(data.get("analysis_run_id") or "")
    if not source_id or not analysis_run_id:
        raise MultimodalEvidenceError("Multimodal timeline requires source and analysis identities.")
    if expected_source_id is not None and source_id != expected_source_id:
        raise MultimodalEvidenceError("Multimodal timeline belongs to a different source.")
    if expected_analysis_run_id is not None and analysis_run_id != expected_analysis_run_id:
        raise MultimodalEvidenceError("Multimodal timeline belongs to a different analysis run.")
    duration = _number(data.get("source_duration_seconds"))
    if duration is None or duration < 0:
        raise MultimodalEvidenceError("Multimodal timeline duration is invalid.")
    if data.get("time_base") != TIME_BASE:
        raise MultimodalEvidenceError("Multimodal timeline uses an unsupported time base.")
    if not isinstance(data.get("input_fingerprints"), dict) or set(data["input_fingerprints"]) != {
        "transcript", "audio", "scenes", "visual", "semantic_audio",
    }:
        raise MultimodalEvidenceError("Multimodal timeline input fingerprints are incomplete.")
    if any(not isinstance(value, str) or len(value) != 64 for value in data["input_fingerprints"].values()):
        raise MultimodalEvidenceError("Multimodal timeline input fingerprints are invalid.")

    all_ids: set[str] = set()
    for collection_name in ("transcript_events", "audio_event_map", "visual_event_map"):
        collection = data.get(collection_name)
        if not isinstance(collection, list):
            raise MultimodalEvidenceError(f"{collection_name} must be an array.")
        previous = -1.0
        for event in collection:
            _validate_event(event, duration)
            event_id = str(event["event_id"])
            if event_id in all_ids:
                raise MultimodalEvidenceError("Multimodal event ids must be globally unique.")
            all_ids.add(event_id)
            if float(event["start_seconds"]) < previous:
                raise MultimodalEvidenceError(f"{collection_name} must be chronological.")
            previous = float(event["start_seconds"])

    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise MultimodalEvidenceError("Multimodal scenes must be an array.")
    previous_end = 0.0
    scene_ids: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise MultimodalEvidenceError("Multimodal scene must be an object.")
        scene_id = str(scene.get("scene_id") or "")
        start = _number(scene.get("start_seconds"))
        end = _number(scene.get("end_seconds"))
        confidence = _number(scene.get("confidence"))
        if (
            not scene_id or scene_id in scene_ids or start is None or end is None
            or confidence is None or not 0 <= confidence <= 1 or start < 0
            or end < start or end > duration + 0.001 or start < previous_end - 0.001
            or not _valid_provenance(scene.get("provenance"))
        ):
            raise MultimodalEvidenceError("Multimodal scene is invalid.")
        scene_ids.add(scene_id)
        previous_end = end

    keyframes = data.get("keyframes")
    if not isinstance(keyframes, list):
        raise MultimodalEvidenceError("Multimodal keyframes must be an array.")
    keyframe_ids: set[str] = set()
    previous_time = -1.0
    for keyframe in keyframes:
        if not isinstance(keyframe, dict):
            raise MultimodalEvidenceError("Multimodal keyframe must be an object.")
        keyframe_id = str(keyframe.get("keyframe_id") or "")
        timestamp = _number(keyframe.get("time_seconds"))
        confidence = _number(keyframe.get("confidence"))
        reasons = keyframe.get("selection_reasons")
        if (
            not keyframe_id or keyframe_id in keyframe_ids or timestamp is None
            or not 0 <= timestamp <= duration + 0.001 or timestamp < previous_time
            or confidence is None or not 0 <= confidence <= 1
            or not isinstance(reasons, list) or not reasons
            or not isinstance(keyframe.get("evidence_refs"), list)
            or not _valid_provenance(keyframe.get("provenance"))
        ):
            raise MultimodalEvidenceError("Multimodal keyframe is invalid.")
        keyframe_ids.add(keyframe_id)
        previous_time = timestamp

    diagnostics = data.get("diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or not isinstance(diagnostics.get("scenes"), dict)
        or not isinstance(diagnostics.get("keyframes"), dict)
        or not isinstance(diagnostics.get("evidence"), dict)
        or diagnostics.get("external_vision_api_calls") != 0
    ):
        raise MultimodalEvidenceError("Multimodal diagnostics are invalid.")
    if (
        diagnostics["scenes"].get("count") != len(scenes)
        or diagnostics["keyframes"].get("count") != len(keyframes)
        or diagnostics["keyframes"].get("analyzed_count")
        != sum(item.get("analysis_status") == "existing_visual_evidence" for item in keyframes)
        or not isinstance(diagnostics.get("available_evidence"), list)
        or not isinstance(diagnostics.get("missing_evidence"), list)
    ):
        raise MultimodalEvidenceError("Multimodal diagnostic counts do not match the artifact.")
    if not _valid_provenance(data.get("provenance")):
        raise MultimodalEvidenceError("Multimodal timeline provenance is invalid.")
    return data


def evidence_for_range(
    timeline: dict[str, Any], start_seconds: float, end_seconds: float,
) -> dict[str, Any]:
    """Return stable timeline references overlapping one StoryUnit range."""

    validate_multimodal_timeline(timeline)
    start = _number(start_seconds)
    end = _number(end_seconds)
    duration = float(timeline["source_duration_seconds"])
    if start is None or end is None or start < 0 or not start < end or end > duration + 0.001:
        raise MultimodalEvidenceError("StoryUnit evidence range is outside the multimodal timeline.")

    sections = {
        "transcript": "transcript_events",
        "audio": "audio_event_map",
        "visual": "visual_event_map",
    }
    event_refs = {
        name: [_event_ref(item) for item in timeline[field] if _overlaps(item, start, end)]
        for name, field in sections.items()
    }
    scene_refs = [
        {
            "scene_id": item["scene_id"],
            "start_seconds": item["start_seconds"],
            "end_seconds": item["end_seconds"],
            "confidence": item["confidence"],
        }
        for item in timeline["scenes"] if _overlaps(item, start, end)
    ]
    keyframe_refs = [
        {
            "keyframe_id": item["keyframe_id"],
            "time_seconds": item["time_seconds"],
            "selection_reasons": item["selection_reasons"],
            "analysis_status": item["analysis_status"],
        }
        for item in timeline["keyframes"] if start <= float(item["time_seconds"]) < end
    ]
    evidence_status = timeline["diagnostics"]["evidence"]
    return {
        "schema_version": STORY_UNIT_EVIDENCE_SCHEMA_VERSION,
        "timeline_schema_version": timeline["schema_version"],
        "analysis_run_id": timeline["analysis_run_id"],
        "source_id": timeline["source_id"],
        "interval": {"start_seconds": round(start, 3), "end_seconds": round(end, 3)},
        "event_refs": event_refs,
        "scene_refs": scene_refs,
        "keyframe_refs": keyframe_refs,
        "evidence_status": {name: dict(value) for name, value in evidence_status.items()},
        "available_evidence": list(timeline["diagnostics"]["available_evidence"]),
        "missing_evidence": list(timeline["diagnostics"]["missing_evidence"]),
    }


def audio_summary_for_range(
    timeline: dict[str, Any], start_seconds: float, end_seconds: float,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Compact code-owned activity/semantic evidence for one existing candidate."""

    validate_multimodal_timeline(timeline)
    start = float(start_seconds)
    end = float(end_seconds)
    duration = max(0.001, end - start)
    overlapping = [item for item in timeline.get("audio_event_map", []) if _overlaps(item, start, end)]
    speech = _merged_ranges(overlapping, start, end, {"speech"})
    activity = _merged_ranges(overlapping, start, end, {"activity"})
    dead = _merged_ranges(overlapping, start, end, {"dead_zone"})
    projected = [
        project_semantic_audio_event(item, profile_id)
        if profile_id and item.get("event_type") in {"semantic_audio_event", "background_music"}
        else item
        for item in overlapping
    ]
    semantic = [
        item for item in projected
        if item.get("event_type") == "semantic_audio_event"
        and isinstance(item.get("observation"), dict)
        and item["observation"].get("meaningful_for_profile") is True
    ]
    music = [item for item in projected if item.get("event_type") == "background_music"]
    spikes = [item for item in overlapping if item.get("event_type") in {"relative_spike", "burst", "peak_region"}]
    speech_gap = _longest_gap(start, end, speech)
    longest_dead = max((upper - lower for lower, upper in dead), default=0.0)
    meaningful = sorted(semantic, key=lambda item: (-float(item.get("confidence", 0)), item["start_seconds"]))[:6]
    return {
        "schema_version": "audio-range-summary.1",
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "speech_coverage_ratio": round(_ranges_duration(speech) / duration, 6),
        "longest_speech_gap_seconds": round(speech_gap, 3),
        "activity_ratio": round(_ranges_duration(activity) / duration, 6),
        "dead_zone_ratio": round(_ranges_duration(dead) / duration, 6),
        "longest_audio_dead_zone_seconds": round(longest_dead, 3),
        "spike_count": len(spikes),
        "spike_peak": round(max((float(item.get("confidence", 0)) for item in spikes), default=0.0), 6),
        "meaningful_event_count": len(semantic),
        "meaningful_events": [{
            "event_id": item["event_id"],
            "label": item["observation"].get("label"),
            "event_group": item["observation"].get("event_group"),
            "start_seconds": item["start_seconds"],
            "end_seconds": item["end_seconds"],
            "confidence": item["confidence"],
            "signal_salience": item["observation"].get("signal_salience"),
            "editorial_roles": list(item["observation"].get("editorial_roles") or []),
            "payoff_claim": False,
        } for item in meaningful],
        "background_music_present": bool(music),
        "background_music_only": bool(music and not semantic),
        "semantic_classifier_status": (
            "available" if any(item.get("event_type") in {"semantic_audio_event", "background_music"} for item in overlapping)
            else "no_event_in_range"
        ),
        "payoff_claim": False,
        "analysis_run_id": timeline["analysis_run_id"],
    }


def _merged_ranges(
    events: list[dict[str, Any]], start: float, end: float, event_types: set[str],
) -> list[tuple[float, float]]:
    ranges = sorted((
        (max(start, float(item["start_seconds"])), min(end, float(item["end_seconds"])))
        for item in events if item.get("event_type") in event_types
    ), key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for lower, upper in ranges:
        if upper <= lower:
            continue
        if merged and lower <= merged[-1][1] + 0.001:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return merged


def _ranges_duration(ranges: list[tuple[float, float]]) -> float:
    return sum(upper - lower for lower, upper in ranges)


def _longest_gap(start: float, end: float, ranges: list[tuple[float, float]]) -> float:
    cursor = start
    longest = 0.0
    for lower, upper in ranges:
        longest = max(longest, lower - cursor)
        cursor = max(cursor, upper)
    return max(longest, end - cursor)


def _transcript_events(transcript: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    raw_global_words = transcript.get("words")
    global_words: list[Any] = raw_global_words if isinstance(raw_global_words, list) else []
    raw_segments = transcript.get("segments")
    segments: list[Any] = raw_segments if isinstance(raw_segments, list) else []
    for index, raw in enumerate(segments):
        if not isinstance(raw, dict):
            continue
        start, end = _bounded_interval(raw.get("start"), raw.get("end"), duration)
        text = str(raw.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        segment_words = raw.get("words")
        raw_words: list[Any] = segment_words if isinstance(segment_words, list) else [
            item for item in global_words if isinstance(item, dict) and _word_overlaps(item, start, end)
        ]
        words: list[dict[str, Any]] = []
        for item in raw_words:
            if isinstance(item, dict) and (word := _word_observation(item, duration)) is not None:
                words.append(word)
        speaker = str(raw.get("speaker_id") or raw.get("speaker") or "").strip() or None
        confidence, confidence_available = _observed_confidence(raw, words)
        segment_id = raw.get("id", index)
        events.append(_event(
            "transcript", "transcript_segment", start, end, confidence,
            {
                "transcript_segment_id": segment_id,
                "text": text,
                "speaker_id": speaker,
                "word_timestamps": words,
                "confidence_available": confidence_available,
            },
            [_provenance("transcript.json", f"segments[{index}]", "direct_timestamped_observation")],
            stable_hint=str(segment_id),
        ))
    return sorted(events, key=lambda item: (item["start_seconds"], item["event_id"]))


def _audio_events(
    transcript_events: list[dict[str, Any]], audio: dict[str, Any],
    semantic_audio: dict[str, Any], duration: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_speaker: str | None = None
    for transcript_event in transcript_events:
        observation = transcript_event["observation"]
        events.append(_event(
            "audio", "speech", transcript_event["start_seconds"], transcript_event["end_seconds"],
            transcript_event["confidence"],
            {
                "speaker_id": observation.get("speaker_id"),
                "transcript_event_id": transcript_event["event_id"],
                "confidence_available": observation.get("confidence_available", False),
            },
            [_provenance("transcript.json", f"event:{transcript_event['event_id']}", "speech_range_from_transcript")],
        ))
        speaker = observation.get("speaker_id")
        if isinstance(speaker, str) and speaker:
            if previous_speaker is not None and speaker != previous_speaker:
                events.append(_event(
                    "audio", "speaker_change", transcript_event["start_seconds"], transcript_event["start_seconds"],
                    transcript_event["confidence"],
                    {"from_speaker_id": previous_speaker, "to_speaker_id": speaker},
                    [_provenance("transcript.json", f"event:{transcript_event['event_id']}", "adjacent_speaker_labels")],
                ))
            previous_speaker = speaker
        for match in _REACTION_LABEL.finditer(str(observation.get("text") or "")):
            events.append(_event(
                "audio", "reaction_label", transcript_event["start_seconds"], transcript_event["end_seconds"],
                transcript_event["confidence"],
                {"label": match.group(1).casefold(), "claim_scope": "explicit_transcript_label_only"},
                [_provenance("transcript.json", f"event:{transcript_event['event_id']}", "explicit_reaction_label")],
                stable_hint=match.group(1).casefold(),
            ))

    window = max(0.001, _number(audio.get("window_seconds")) or 0.5)
    energy_rows: list[tuple[float, float, float | None, dict[str, Any]]] = []
    for item in audio.get("energy_frames", []):
        if not isinstance(item, dict):
            continue
        timestamp = _number(item.get("time"))
        loudness = _number(item.get("normalized_loudness"))
        raw_energy = _number(item.get("audio_energy"))
        if timestamp is None or loudness is None or not 0 <= timestamp <= duration or not 0 <= loudness <= 1:
            continue
        energy_rows.append((timestamp, loudness, raw_energy, item))
    threshold = _emphasis_threshold([item[1] for item in energy_rows])
    for index, (timestamp, loudness, raw_energy, signal) in enumerate(energy_rows):
        end = min(duration, timestamp + window)
        events.append(_event(
            "audio", "energy", timestamp, end, 1.0,
            {
                "normalized_loudness": round(loudness, 6), "audio_energy": raw_energy,
                "relative_loudness": _number(signal.get("relative_loudness")),
                "spike_score": _number(signal.get("spike_score")),
                "onset_score": _number(signal.get("onset_score")),
                "burst_score": _number(signal.get("burst_score")),
                "activity_score": _number(signal.get("activity_score")),
                "noisiness": _number(signal.get("noisiness")),
                "measurement_window_seconds": window,
            },
            [_provenance("audio_features.json", f"energy_frames[{index}]", "local_pcm_rms_measurement")],
        ))
        if loudness >= threshold and loudness > 0:
            events.append(_event(
                "audio", "emphasis", timestamp, end, min(1.0, loudness),
                {"normalized_loudness": round(loudness, 6), "threshold": threshold, "method": "relative_pcm_energy"},
                [_provenance("audio_features.json", f"energy_frames[{index}]", "deterministic_energy_threshold")],
            ))
        spike = _number(signal.get("spike_score")) or 0.0
        onset = _number(signal.get("onset_score")) or 0.0
        burst = _number(signal.get("burst_score")) or 0.0
        if spike >= 0.62 or onset >= 0.58:
            events.append(_event(
                "audio", "relative_spike", timestamp, end, max(spike, onset),
                {"spike_score": round(spike, 6), "onset_score": round(onset, 6), "method": "source_relative_pcm"},
                [_provenance("audio_features.json", f"energy_frames[{index}]", "deterministic_relative_spike")],
            ))
        if burst >= 0.58:
            events.append(_event(
                "audio", "burst", timestamp, end, burst,
                {"burst_score": round(burst, 6), "method": "bounded_onset_density"},
                [_provenance("audio_features.json", f"energy_frames[{index}]", "deterministic_onset_density")],
            ))
    for index, item in enumerate(audio.get("silence_intervals", [])):
        if not isinstance(item, dict):
            continue
        silence_start, silence_end = _bounded_interval(item.get("start"), item.get("end"), duration)
        if silence_start is None or silence_end is None:
            continue
        provenance = [_provenance("audio_features.json", f"silence_intervals[{index}]", "local_pcm_silence_threshold")]
        events.append(_event("audio", "silence", silence_start, silence_end, 1.0, {"duration_seconds": round(silence_end - silence_start, 3)}, provenance))
        if silence_end - silence_start >= 0.5:
            events.append(_event(
                "audio", "pause", silence_start, silence_end, 1.0,
                {"duration_seconds": round(silence_end - silence_start, 3), "minimum_pause_seconds": 0.5}, provenance,
            ))
    for event_type, field in (("activity", "activity_intervals"), ("dead_zone", "dead_zones")):
        for index, item in enumerate(audio.get(field, [])):
            if not isinstance(item, dict):
                continue
            start, end = _bounded_interval(item.get("start"), item.get("end"), duration)
            if start is None or end is None:
                continue
            events.append(_event(
                "audio", event_type, start, end, 1.0,
                {"duration_seconds": round(end - start, 3), "source": "audio_signal_analysis"},
                [_provenance("audio_features.json", f"{field}[{index}]", "deterministic_activity_segmentation")],
            ))
    for index, item in enumerate(audio.get("peak_regions", [])):
        if not isinstance(item, dict):
            continue
        start, end = _bounded_interval(item.get("start"), item.get("end"), duration)
        confidence = _number(item.get("score"))
        if start is None or end is None or confidence is None:
            continue
        events.append(_event(
            "audio", "peak_region", start, end, confidence,
            {**{key: item.get(key) for key in (
                "region_id", "peak_time", "score", "relative_loudness", "spike_score", "onset_score", "burst_score",
            )}, "candidate_seed": True, "payoff_claim": False},
            [_provenance("audio_features.json", f"peak_regions[{index}]", "bounded_audio_peak_seed")],
            stable_hint=str(item.get("region_id") or index),
        ))
    if semantic_audio.get("status") in {"completed", "partial"}:
        for item in semantic_audio.get("events", []):
            if not isinstance(item, dict):
                continue
            try:
                _validate_event(item, duration)
            except MultimodalEvidenceError:
                continue
            events.append(dict(item))
    return sorted(events, key=lambda item: (item["start_seconds"], item["event_id"]))


def _scene_evidence(
    scenes: dict[str, Any], duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    enabled = scenes.get("enabled") is True
    warning = str(scenes.get("warning") or "").strip()
    boundaries: list[tuple[float, float, int]] = []
    for index, raw in enumerate(scenes.get("boundaries", [])):
        if not isinstance(raw, dict):
            continue
        timestamp = _number(raw.get("timestamp"))
        score = _number(raw.get("scene_change_score"))
        if timestamp is None or score is None or not 0 < timestamp < duration or not 0 <= score <= 1:
            continue
        boundaries.append((timestamp, score, index))
    boundaries.sort(key=lambda item: item[0])
    if not enabled or warning:
        return [], [], "missing"

    scene_events = [
        _event(
            "visual", "scene_change", timestamp, timestamp, score,
            {"scene_change_score": round(score, 5)},
            [_provenance("scene_boundaries.json", f"boundaries[{index}]", "local_ffmpeg_scene_detection")],
        )
        for timestamp, score, index in boundaries
    ]
    points = [0.0, *[item[0] for item in boundaries], duration]
    intervals: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
        if end < start:
            continue
        boundary_score = boundaries[index - 2][1] if index > 1 else 1.0
        locator = f"boundaries[{boundaries[index - 2][2]}]" if index > 1 else "source_start"
        intervals.append({
            "scene_id": f"scene-{index:04d}",
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "confidence": round(_clamp(boundary_score), 6),
            "provenance": [_provenance("scene_boundaries.json", locator, "derived_scene_interval")],
        })
    return intervals, scene_events, "available"


def _subject_events(
    visual: dict[str, Any], duration: float,
) -> tuple[list[dict[str, Any]], str]:
    if visual.get("evidence_status") != "valid":
        return [], "missing"
    rows: list[tuple[float, dict[str, Any], int]] = []
    for index, raw in enumerate(visual.get("subject_keyframes", [])):
        if not isinstance(raw, dict):
            continue
        timestamp = _number(raw.get("time_seconds"))
        if timestamp is not None and 0 <= timestamp <= duration:
            rows.append((timestamp, raw, index))
    rows.sort(key=lambda item: item[0])
    events: list[dict[str, Any]] = []
    previous: tuple[float, dict[str, Any], int] | None = None
    for timestamp, raw, index in rows:
        confidence = _clamp(_number(raw.get("confidence")) or 0.0)
        target = str(raw.get("tracking_target") or "none")
        displacement: float | None = None
        if previous is not None:
            old_x, old_y = _number(previous[1].get("normalized_x")), _number(previous[1].get("normalized_y"))
            new_x, new_y = _number(raw.get("normalized_x")), _number(raw.get("normalized_y"))
            previous_scene = str(previous[1].get("scene_id") or "").strip()
            current_scene = str(raw.get("scene_id") or "").strip()
            sample_gap = timestamp - previous[0]
            if (
                previous_scene and current_scene == previous_scene and 0 < sample_gap <= 5.0
                and None not in {old_x, old_y, new_x, new_y}
            ):
                assert old_x is not None and old_y is not None and new_x is not None and new_y is not None
                displacement = math.hypot(new_x - old_x, new_y - old_y)
        bbox = {
            name: _number(raw.get(name))
            for name in ("normalized_x", "normalized_y", "normalized_width", "normalized_height")
        }
        observation = {
            "faces": {
                "visible_count": int(raw["visible_face_count"]) if isinstance(raw.get("visible_face_count"), int) else None,
                "active_speaker_confidence": _number(raw.get("active_speaker_confidence")),
            },
            "active_subject": {"target_type": target, "normalized_bbox": bbox},
            "objects_persons": {
                "observed_target_type": target,
                "person_is_active_target": target in {"primary_face", "primary_person", "subject_group"},
                "object_is_active_target": target == "important_object",
            },
            "screen_text_product": {
                "screen_region_is_active_target": target == "screen_region",
                "text_evidence": "missing",
                "product_evidence": "missing",
            },
            "motion_action": {
                "normalized_subject_displacement": round(displacement, 6) if displacement is not None else None,
                "motion_evidence_between_samples": displacement is not None and displacement >= 0.035,
                "gesture_observed": raw.get("gesture_active") if isinstance(raw.get("gesture_active"), bool) else None,
                "gesture_area_visible": raw.get("gesture_area_visible") if isinstance(raw.get("gesture_area_visible"), bool) else None,
            },
            "framing_relevance": {
                "scene_id": str(raw.get("scene_id") or ""),
                "scene_type": str(raw.get("scene_type") or "UNKNOWN"),
                "framing_observation": str(raw.get("framing_observation") or "unknown"),
                "active_target_relevance": round(confidence, 6) if target != "none" else 0.0,
            },
        }
        events.append(_event(
            "visual", "subject_observation", timestamp, timestamp, confidence, observation,
            [_provenance("visual_analysis.json", f"subject_keyframes[{index}]", "existing_visual_observation")],
        ))
        previous = (timestamp, raw, index)
    return events, "available" if events else "missing"


def _select_keyframes(
    duration: float,
    transcript_events: list[dict[str, Any]],
    audio_events: list[dict[str, Any]],
    visual_events: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    if scenes:
        candidates.append(_keyframe_candidate(0.0, "scene_start", 0.55, 1.0, None, None))
    for event in visual_events:
        if event["event_type"] == "scene_change":
            candidates.append(_keyframe_candidate(
                event["start_seconds"], "scene_boundary", 0.9 + 0.1 * event["confidence"],
                event["confidence"], event["event_id"], "existing_visual_evidence",
            ))
        elif event["event_type"] == "subject_observation":
            observation = event["observation"]
            relevance = float(observation["framing_relevance"]["active_target_relevance"])
            motion = observation["motion_action"]["normalized_subject_displacement"]
            if relevance > 0:
                candidates.append(_keyframe_candidate(
                    event["start_seconds"], "framing_relevance", 0.55 + 0.35 * relevance,
                    event["confidence"], event["event_id"], "existing_visual_evidence",
                ))
            if isinstance(motion, (int, float)) and motion >= 0.035:
                candidates.append(_keyframe_candidate(
                    event["start_seconds"], "measured_motion", min(1.0, 0.65 + float(motion)),
                    event["confidence"], event["event_id"], "existing_visual_evidence",
                ))
            if observation["motion_action"].get("gesture_observed") is True:
                candidates.append(_keyframe_candidate(
                    event["start_seconds"], "observed_action", 0.78,
                    event["confidence"], event["event_id"], "existing_visual_evidence",
                ))
    for event in audio_events:
        if event["event_type"] == "speaker_change":
            candidates.append(_keyframe_candidate(
                event["start_seconds"], "speaker_change", 0.82, event["confidence"], event["event_id"], None,
            ))
        elif event["event_type"] == "emphasis":
            candidates.append(_keyframe_candidate(
                event["start_seconds"], "audio_emphasis", 0.58 + 0.3 * event["confidence"],
                event["confidence"], event["event_id"], None,
            ))
    for event in transcript_events:
        candidates.append(_keyframe_candidate(
            event["start_seconds"], "speech_onset", 0.45 + 0.2 * event["confidence"],
            event["confidence"], event["event_id"], None,
        ))

    merged: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: (value["time_seconds"], -value["score"])):
        if merged and abs(float(item["time_seconds"]) - float(merged[-1]["time_seconds"])) <= 0.3:
            target = merged[-1]
            if item["score"] > target["score"]:
                target["time_seconds"] = item["time_seconds"]
                target["score"] = item["score"]
            target["reasons"].update(item["reasons"])
            target["evidence_refs"].update(item["evidence_refs"])
            target["confidence"] = max(target["confidence"], item["confidence"])
            if item["analysis_status"] == "existing_visual_evidence":
                target["analysis_status"] = "existing_visual_evidence"
            continue
        merged.append(item)

    limit = min(64, max(4, int(math.ceil(max(duration, 1.0) / 15.0)) * 3))
    chosen: list[dict[str, Any]] = []
    for item in sorted(merged, key=lambda value: (-value["score"], value["time_seconds"])):
        if any(abs(float(item["time_seconds"]) - float(other["time_seconds"])) < 0.75 for other in chosen):
            continue
        chosen.append(item)
        if len(chosen) >= limit:
            break
    chosen.sort(key=lambda value: value["time_seconds"])
    result: list[dict[str, Any]] = []
    for index, item in enumerate(chosen, start=1):
        result.append({
            "keyframe_id": f"keyframe-{index:04d}-{stable_text_hash(str(item['time_seconds']))[:8]}",
            "time_seconds": round(float(item["time_seconds"]), 3),
            "selection_reasons": sorted(item["reasons"]),
            "relevance_score": round(_clamp(float(item["score"])), 6),
            "confidence": round(_clamp(float(item["confidence"])), 6),
            "analysis_status": item["analysis_status"],
            "future_vision_api_eligible": True,
            "evidence_refs": sorted(item["evidence_refs"]),
            "provenance": [_provenance(
                "multimodal_timeline.json", "keyframe_selection",
                "deterministic_sparse_selection_no_frame_analysis",
            )],
        })
    return result, limit


def _evidence_diagnostics(
    transcript: dict[str, Any],
    transcript_events: list[dict[str, Any]],
    audio: dict[str, Any],
    audio_events: list[dict[str, Any]],
    visual: dict[str, Any],
    subject_events: list[dict[str, Any]],
    scenes: dict[str, Any],
    scene_intervals: list[dict[str, Any]],
    scene_status: str,
    subject_status: str,
) -> dict[str, dict[str, Any]]:
    transcript_status = "available" if transcript_events else "missing"
    speakers_available = any(item["observation"].get("speaker_id") for item in transcript_events)
    words_available = any(item["observation"].get("word_timestamps") for item in transcript_events)
    pcm_available = bool(audio.get("energy_frames") or audio.get("silence_intervals")) and not audio.get("warning")
    speech_available = any(item["event_type"] == "speech" for item in audio_events)
    audio_status = "available" if pcm_available and speech_available else "partial" if pcm_available or speech_available else "missing"
    visual_status = "available" if subject_status == "available" else "partial" if scene_status == "available" else "missing"
    scene_reason = (
        str(scenes.get("warning") or "scene_detection_failed")
        if scenes.get("enabled") is True else "scene_detection_disabled"
    )
    visual_reason = str(visual.get("reason") or visual.get("evidence_status") or "visual_evidence_unavailable")
    return {
        "transcript": {
            "status": transcript_status,
            "event_count": len(transcript_events),
            "available_fields": (["text", "timestamps"] if transcript_events else [])
            + (["speakers"] if speakers_available else [])
            + (["word_timestamps"] if words_available else []),
            "missing_fields": ([] if transcript_events else ["text", "timestamps"])
            + ([] if speakers_available else ["speakers"])
            + ([] if words_available else ["word_timestamps"]),
            "reason": None if transcript_events else ("empty_transcript" if transcript.get("empty_transcript") else "transcript_unavailable"),
        },
        "audio": {
            "status": audio_status,
            "event_count": len(audio_events),
            "available_fields": [
                name for name, present in {
                    "speech": speech_available,
                    "silence": any(item["event_type"] == "silence" for item in audio_events),
                    "pauses": any(item["event_type"] == "pause" for item in audio_events),
                    "energy_emphasis": any(item["event_type"] == "energy" for item in audio_events),
                    "speaker_changes": any(item["event_type"] == "speaker_change" for item in audio_events),
                    "reaction_labels": any(item["event_type"] == "reaction_label" for item in audio_events),
                    "relative_spikes": any(item["event_type"] == "relative_spike" for item in audio_events),
                    "bursts": any(item["event_type"] == "burst" for item in audio_events),
                    "activity_dead_zones": any(item["event_type"] in {"activity", "dead_zone"} for item in audio_events),
                    "semantic_events": any(item["event_type"] in {"semantic_audio_event", "background_music"} for item in audio_events),
                }.items() if present
            ],
            "missing_fields": ["pcm_energy", "silence"] if not pcm_available else [],
            "reason": str(audio.get("warning") or "") or (None if pcm_available or speech_available else "audio_evidence_unavailable"),
        },
        "visual": {
            "status": visual_status,
            "event_count": len(subject_events),
            "available_fields": [
                name for name, present in {
                    "faces_active_subject": bool(subject_events),
                    "objects_persons": bool(subject_events),
                    "screen_target_classification": bool(subject_events),
                    "scene_changes": bool(scene_intervals),
                    "motion_action": bool(subject_events),
                    "framing_relevance": bool(subject_events),
                }.items() if present
            ],
            "missing_fields": ([] if subject_events else [
                "faces_active_subject", "objects_persons", "screen_text_product",
                "motion_action", "framing_relevance",
            ]) + ["text", "product"],
            "reason": None if subject_events else visual_reason,
        },
        "scenes": {
            "status": scene_status,
            "event_count": len(scene_intervals),
            "available_fields": ["boundaries", "scene_intervals"] if scene_intervals else [],
            "missing_fields": [] if scene_intervals else ["boundaries", "scene_intervals"],
            "reason": None if scene_intervals else scene_reason,
        },
    }


def _event(
    modality: str,
    event_type: str,
    start: float,
    end: float,
    confidence: float,
    observation: dict[str, Any],
    provenance: list[dict[str, str]],
    *,
    stable_hint: str = "",
) -> dict[str, Any]:
    identity = f"{modality}|{event_type}|{start:.6f}|{end:.6f}|{stable_hint}|{_canonical(observation)}"
    return {
        "event_id": f"{modality}-{event_type}-{stable_text_hash(identity)[:16]}",
        "event_type": event_type,
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "confidence": round(_clamp(confidence), 6),
        "observation": observation,
        "provenance": provenance,
    }


def _validate_event(event: Any, duration: float) -> None:
    if not isinstance(event, dict) or set(event) != _EVENT_FIELDS:
        raise MultimodalEvidenceError("Multimodal event has an invalid schema.")
    start = _number(event.get("start_seconds"))
    end = _number(event.get("end_seconds"))
    confidence = _number(event.get("confidence"))
    if (
        not str(event.get("event_id") or "") or not str(event.get("event_type") or "")
        or start is None or end is None or start < 0 or end < start or end > duration + 0.001
        or confidence is None or not 0 <= confidence <= 1
        or not isinstance(event.get("observation"), dict)
        or not _valid_provenance(event.get("provenance"))
    ):
        raise MultimodalEvidenceError("Multimodal event is invalid.")


def _keyframe_candidate(
    timestamp: float,
    reason: str,
    score: float,
    confidence: float,
    evidence_ref: str | None,
    analysis_status: str | None,
) -> dict[str, Any]:
    return {
        "time_seconds": timestamp,
        "reasons": {reason},
        "score": _clamp(score),
        "confidence": _clamp(confidence),
        "evidence_refs": {evidence_ref} if evidence_ref else set(),
        "analysis_status": analysis_status or "not_analyzed",
    }


def _event_ref(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "start_seconds": event["start_seconds"],
        "end_seconds": event["end_seconds"],
        "confidence": event["confidence"],
    }


def _overlaps(item: dict[str, Any], start: float, end: float) -> bool:
    item_start = float(item["start_seconds"])
    item_end = float(item["end_seconds"])
    if item_start == item_end:
        return start <= item_start < end
    return item_start < end and item_end > start


def _word_overlaps(word: dict[str, Any], start: float, end: float) -> bool:
    word_start = _number(word.get("start"))
    word_end = _number(word.get("end"))
    return word_start is not None and word_end is not None and word_start < end and word_end > start


def _word_observation(word: dict[str, Any], duration: float) -> dict[str, Any] | None:
    start, end = _bounded_interval(word.get("start"), word.get("end"), duration)
    text = str(word.get("text") or word.get("word") or "").strip()
    if start is None or end is None or not text:
        return None
    confidence = _number(word.get("probability"))
    return {
        "start_seconds": start,
        "end_seconds": end,
        "text": text,
        "confidence": round(_clamp(confidence), 6) if confidence is not None else None,
    }


def _observed_confidence(raw: dict[str, Any], words: list[dict[str, Any]]) -> tuple[float, bool]:
    for name in ("confidence", "transcript_confidence"):
        value = _number(raw.get(name))
        if value is not None and 0 <= value <= 1:
            return value, True
    values = [float(item["confidence"]) for item in words if item.get("confidence") is not None]
    if values:
        return sum(values) / len(values), True
    return 0.0, False


def _emphasis_threshold(values: list[float]) -> float:
    if not values:
        return 1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return round(_clamp(max(0.72, mean + math.sqrt(variance) * 0.5)), 6)


def _bounded_interval(start_value: Any, end_value: Any, duration: float) -> tuple[float | None, float | None]:
    start = _number(start_value)
    end = _number(end_value)
    if start is None or end is None or start < 0 or not start < end or start >= duration:
        return None, None
    bounded_end = min(end, duration)
    if bounded_end <= start:
        return None, None
    return round(start, 3), round(bounded_end, 3)


def _fingerprints(
    transcript: dict[str, Any], audio: dict[str, Any], scenes: dict[str, Any], visual: dict[str, Any],
    semantic_audio: dict[str, Any] | None = None,
) -> dict[str, str]:
    return {
        "transcript": stable_text_hash(_canonical(transcript)),
        "audio": stable_text_hash(_canonical(audio)),
        "scenes": stable_text_hash(_canonical(scenes)),
        "visual": stable_text_hash(_canonical(visual)),
        "semantic_audio": stable_text_hash(_canonical(semantic_audio or {})),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _provenance(artifact: str, locator: str, method: str) -> dict[str, str]:
    return {"artifact": artifact, "locator": locator, "method": method}


def _valid_provenance(value: Any) -> bool:
    return (
        isinstance(value, list) and bool(value)
        and all(
            isinstance(item, dict)
            and set(item) == {"artifact", "locator", "method"}
            and all(isinstance(item[name], str) and bool(item[name]) for name in ("artifact", "locator", "method"))
            for item in value
        )
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp(value: float | None) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))
