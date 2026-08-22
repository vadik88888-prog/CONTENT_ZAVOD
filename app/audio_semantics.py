"""Bounded YAMNet-compatible semantic audio evidence for the existing pipeline."""

from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Any, Callable, Iterable

from app.audio_features import window_audio_features
from app.config import AudioAnalysisConfig
from app.models import Candidate
from app.utils import stable_file_hash, stable_text_hash


AUDIO_SEMANTIC_ANALYSIS_VERSION = "audio-semantic.onnx.1"
YAMNET_FRAME_SECONDS = 0.96
YAMNET_HOP_SECONDS = 0.48

_MUSIC = ("music", "musical", "song", "singing", "instrument", "orchestra", "choir")
_SPEECH = ("speech", "conversation", "narration", "monologue", "whispering")
_REACTION = ("laughter", "giggle", "applause", "cheering", "gasp", "scream", "whoop", "crowd")
_IMPACT = (
    "gunshot", "gunfire", "fusillade", "machine gun", "explosion", "artillery", "boom",
    "crash", "smash", "impact", "thump", "thud", "thunk", "punch", "slap",
)
_COOKING = ("sizzle", "frying", "chopping", "cutlery", "dishes", "blender", "food")
_SPORTS = ("basketball", "football", "soccer", "tennis", "volleyball", "skateboard", "whistle")
_TOOL = ("tool", "hammer", "drill", "sawing", "mechanism", "typing", "keyboard", "click")
_ALERT = ("alarm", "siren", "beep", "bleep", "bell", "buzzer")
_GAME = ("video game", "computer game")
_ENVIRONMENT = ("vehicle", "traffic", "water", "wind", "rain", "thunder", "animal", "bird", "door", "footstep")

_PROFILE_GROUPS = {
    "podcast": {"reaction"},
    "interview": {"reaction"},
    "talking_head_expert": {"reaction", "alert"},
    "gameplay": {"reaction", "impact", "alert", "game"},
    "stream": {"reaction", "impact", "alert", "game"},
    "vlog_lifestyle": {"reaction", "impact", "alert", "environment"},
    "food": {"reaction", "cooking", "tool"},
    "travel": {"reaction", "impact", "alert", "environment"},
    "tutorial_education": {"reaction", "tool", "alert"},
    "review": {"reaction", "tool", "alert"},
    "reaction": {"reaction", "impact", "alert"},
    "story_entertainment": {"reaction", "impact", "alert", "environment"},
    "movie_series": {"reaction", "impact", "alert", "environment"},
    "sports_fitness": {"reaction", "impact", "sports", "alert"},
    "news_commentary": {"reaction", "alert"},
}
AUDIO_PROFILE_IDS = frozenset(_PROFILE_GROUPS)

if len(AUDIO_PROFILE_IDS) != 15:
    raise RuntimeError("Semantic audio interpretation must cover all 15 content profiles.")


class AudioSemanticError(ValueError):
    pass


def select_semantic_audio_regions(
    audio_features: dict[str, Any],
    shortlisted_candidates: Iterable[Candidate],
    config: AudioAnalysisConfig,
) -> list[dict[str, Any]]:
    """Select a deduplicated bounded peak + existing-shortlist inference budget."""

    duration = max(0.0, float(audio_features.get("duration_seconds") or 0.0))
    proposals: list[dict[str, Any]] = []
    peaks = sorted(
        [item for item in audio_features.get("peak_regions", []) if isinstance(item, dict)],
        key=lambda item: (-float(item.get("score", 0)), float(item.get("start", 0))),
    )[:config.semantic_max_peak_regions]
    for item in peaks:
        region = _bounded_region(
            float(item.get("start", 0)), float(item.get("end", 0)), duration,
            config.semantic_max_region_seconds,
        )
        if region is None:
            continue
        proposals.append({
            "region_id": str(item.get("region_id") or f"peak-{len(proposals):03d}"),
            "start": region[0], "end": region[1], "sources": ["audio_peak"],
            "candidate_ids": [], "signal_score": round(float(item.get("score", 0)), 6),
            "peak_time": float(item.get("peak_time", (region[0] + region[1]) / 2)),
        })

    shortlist = list(shortlisted_candidates)[:config.semantic_max_shortlist_regions]
    for shortlist_rank, candidate in enumerate(shortlist):
        anchor = _strongest_frame_time(audio_features, candidate.start, candidate.end)
        half = config.semantic_max_region_seconds / 2
        region = _bounded_region(anchor - half, anchor + half, duration, config.semantic_max_region_seconds)
        if region is None:
            continue
        proposals.append({
            "region_id": f"shortlist-{stable_text_hash(candidate.id)[:12]}",
            "start": region[0], "end": region[1], "sources": ["shortlist"],
            "candidate_ids": [candidate.id],
            "signal_score": round(_window_signal_peak(audio_features, region[0], region[1]), 6),
            "peak_time": round(anchor, 3),
            "shortlist_rank": shortlist_rank,
        })

    deduplicated: list[dict[str, Any]] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (
            0 if "audio_peak" in item["sources"] else 1,
            -item["signal_score"] if "audio_peak" in item["sources"] else item.get("shortlist_rank", 0),
            item["start"],
        ),
    ):
        duplicate = next((item for item in deduplicated if _region_overlap_ratio(item, proposal) >= 0.65), None)
        if duplicate is not None:
            duplicate["sources"] = sorted(set(duplicate["sources"] + proposal["sources"]))
            duplicate["candidate_ids"] = sorted(set(duplicate["candidate_ids"] + proposal["candidate_ids"]))
            duplicate["signal_score"] = max(duplicate["signal_score"], proposal["signal_score"])
            continue
        if sum(item["end"] - item["start"] for item in deduplicated) + proposal["end"] - proposal["start"] > config.semantic_max_total_seconds:
            continue
        deduplicated.append(proposal)

    for item in deduplicated:
        summary = window_audio_features(item["start"], item["end"], audio_features)
        item["activity_ratio"] = summary["audio_activity_ratio"]
        item["dead_zone_ratio"] = summary["audio_dead_zone_ratio"]
        item["relative_loudness"] = summary["relative_loudness"]
    return sorted(deduplicated, key=lambda item: (item["start"], item["region_id"]))


def analyse_semantic_audio(
    path: Path,
    audio_features: dict[str, Any],
    shortlisted_candidates: Iterable[Candidate],
    _profile_id: str | None,
    config: AudioAnalysisConfig,
    *,
    inference: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    regions = select_semantic_audio_regions(audio_features, shortlisted_candidates, config)
    base = {
        "schema_version": config.semantic_schema_version,
        "analysis_version": AUDIO_SEMANTIC_ANALYSIS_VERSION,
        "profile_id": "profile_neutral",
        "regions": regions,
        "events": [],
        "runtime": {
            "backend": "onnxruntime-cpu", "provider": "CPUExecutionProvider",
            "tensorflow_required": False, "tensorflow_loaded": False,
            "model_path": config.semantic_model_path,
            "class_map_path": config.semantic_class_map_path,
            "model_sha256": None, "class_map_sha256": None,
        },
        "diagnostics": {
            "selected_region_count": len(regions),
            "peak_region_count": sum("audio_peak" in item["sources"] for item in regions),
            "shortlist_region_count": sum("shortlist" in item["sources"] for item in regions),
            "classified_seconds": round(sum(item["end"] - item["start"] for item in regions), 3),
            "full_source_scan": False, "prepared_pcm_opens": 0, "source_video_decodes": 0,
            "inference_calls": 0, "warnings": [],
        },
    }
    if not config.semantic_enabled:
        return {**base, "status": "disabled", "reason": "semantic_audio_disabled"}
    if not regions:
        return {**base, "status": "completed", "reason": "no_bounded_regions_selected"}

    try:
        model_path = _asset_path(config.semantic_model_path)
        class_map_path = _asset_path(config.semantic_class_map_path)
        model_hash = stable_file_hash(model_path)
        class_map_hash = stable_file_hash(class_map_path)
        _verify_hash("model", model_hash, config.semantic_model_sha256)
        _verify_hash("class map", class_map_hash, config.semantic_class_map_sha256)
        labels = _read_labels(class_map_path)
        base["runtime"].update({"model_sha256": model_hash, "class_map_sha256": class_map_hash})
        runner = inference or _onnx_runner(model_path)
        events: list[dict[str, Any]] = []
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
                raise AudioSemanticError("Semantic audio expects prepared 16 kHz PCM WAV 16-bit mono.")
            base["diagnostics"]["prepared_pcm_opens"] = 1
            for region in regions:
                source.setpos(min(source.getnframes(), round(region["start"] * 16000)))
                raw = source.readframes(round((region["end"] - region["start"]) * 16000))
                waveform = _float_waveform(raw)
                scores = runner(waveform)
                base["diagnostics"]["inference_calls"] += 1
                events.extend(_events_from_scores(scores, labels, region, audio_features, config))
        base["events"] = _merge_semantic_events(events)
        return {**base, "status": "completed", "reason": None}
    except Exception as error:
        base["diagnostics"]["warnings"].append(str(error))
        return {**base, "status": "unavailable", "reason": f"semantic_audio_unavailable:{error}"}


def validate_semantic_audio(data: dict[str, Any], config: AudioAnalysisConfig) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema_version") != config.semantic_schema_version:
        raise AudioSemanticError("Unsupported semantic audio artifact schema.")
    if data.get("analysis_version") != AUDIO_SEMANTIC_ANALYSIS_VERSION:
        raise AudioSemanticError("Unsupported semantic audio analysis version.")
    if data.get("status") not in {"completed", "disabled", "unavailable", "partial"}:
        raise AudioSemanticError("Semantic audio artifact status is invalid.")
    if not isinstance(data.get("regions"), list) or not isinstance(data.get("events"), list):
        raise AudioSemanticError("Semantic audio regions/events must be arrays.")
    if not isinstance(data.get("diagnostics"), dict) or data["diagnostics"].get("full_source_scan") is not False:
        raise AudioSemanticError("Semantic audio diagnostics must prove bounded inference.")
    for event in data["events"]:
        if not isinstance(event, dict) or not {
            "event_id", "event_type", "start_seconds", "end_seconds", "confidence", "observation", "provenance",
        }.issubset(event):
            raise AudioSemanticError("Semantic audio event is invalid.")
    return data


def project_semantic_audio_event(event: dict[str, Any], profile_id: str) -> dict[str, Any]:
    """Project a source-stable classification into profile-aware Brain evidence."""

    projected = {**event, "observation": dict(event.get("observation") or {})}
    observation = projected["observation"]
    group = str(observation.get("event_group") or "other")
    meaningful = profile_id in set(observation.get("meaningful_profile_ids") or [])
    observation.update({
        "profile_id": profile_id,
        "meaningful_for_profile": meaningful,
        "editorial_roles": _editorial_roles(group, meaningful),
        "payoff_claim": False,
    })
    return projected


def _onnx_runner(model_path: Path) -> Callable[[Any], Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return lambda waveform: session.run([output_name], {input_name: waveform})[0]


def _float_waveform(raw: bytes) -> Any:
    import numpy as np

    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _events_from_scores(
    scores: Any, labels: list[dict[str, str]], region: dict[str, Any], audio: dict[str, Any],
    config: AudioAnalysisConfig,
) -> list[dict[str, Any]]:
    import numpy as np

    matrix = np.asarray(scores)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise AudioSemanticError("YAMNet score tensor does not match the class map.")
    events: list[dict[str, Any]] = []
    for frame_index, row in enumerate(matrix):
        start = float(region["start"]) + frame_index * YAMNET_HOP_SECONDS
        end = min(float(region["end"]), start + YAMNET_FRAME_SECONDS)
        if start >= float(region["end"]):
            break
        top = np.argsort(row)[-config.semantic_top_classes:][::-1]
        for class_index in top:
            confidence = float(row[int(class_index)])
            if confidence < config.semantic_min_confidence:
                continue
            label = labels[int(class_index)]
            group = _event_group(label["display_name"])
            activity = _window_signal_peak(audio, start, end)
            meaningful_profiles = sorted(
                profile for profile, groups in _PROFILE_GROUPS.items()
                if group in groups and activity >= config.activity_threshold
            )
            event_type = "background_music" if group == "music" else "semantic_audio_event"
            digest = stable_text_hash(f"{region['region_id']}|{frame_index}|{label['index']}|{start:.3f}")[:16]
            events.append({
                "event_id": f"audio-semantic-{digest}",
                "event_type": event_type,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "confidence": round(confidence, 6),
                "observation": {
                    "class_index": int(label["index"]), "class_mid": label["mid"],
                    "label": label["display_name"], "event_group": group,
                    "meaningful_profile_ids": meaningful_profiles,
                    "payoff_claim": False, "background_music_only": group == "music",
                    "signal_salience": round(activity, 6), "region_sources": list(region["sources"]),
                    "candidate_ids": list(region["candidate_ids"]),
                },
                "provenance": [{
                    "artifact": "audio_semantic_events.json",
                    "locator": f"regions:{region['region_id']}:frame:{frame_index}:class:{int(label['index'])}",
                    "method": "bounded_yamnet_onnx_inference",
                }],
            })
    return events


def _merge_semantic_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item["start_seconds"], -item["confidence"], item["event_id"])):
        duplicate = next((
            item for item in reversed(merged[-20:])
            if item["observation"]["class_index"] == event["observation"]["class_index"]
            and item["end_seconds"] >= event["start_seconds"] - 0.001
        ), None)
        if duplicate is None:
            merged.append(event)
            continue
        duplicate["end_seconds"] = max(duplicate["end_seconds"], event["end_seconds"])
        duplicate["confidence"] = max(duplicate["confidence"], event["confidence"])
        duplicate["observation"]["signal_salience"] = max(
            duplicate["observation"]["signal_salience"], event["observation"]["signal_salience"],
        )
        duplicate["observation"]["meaningful_profile_ids"] = sorted(set(
            duplicate["observation"]["meaningful_profile_ids"] + event["observation"]["meaningful_profile_ids"]
        ))
        duplicate["observation"]["candidate_ids"] = sorted(set(
            duplicate["observation"]["candidate_ids"] + event["observation"]["candidate_ids"]
        ))
    return merged


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(item) for item in csv.DictReader(stream)]
    if len(rows) != 521 or any(int(item["index"]) != index for index, item in enumerate(rows)):
        raise AudioSemanticError("YAMNet class map must contain ordered indices 0..520.")
    return rows


def _asset_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _verify_hash(name: str, actual: str, expected: str) -> None:
    if expected and actual.casefold() != expected.casefold():
        raise AudioSemanticError(f"Semantic audio {name} SHA-256 mismatch.")


def _event_group(label: str) -> str:
    value = label.casefold()
    for group, tokens in (
        ("music", _MUSIC), ("speech", _SPEECH), ("reaction", _REACTION),
        ("impact", _IMPACT), ("cooking", _COOKING), ("sports", _SPORTS),
        ("tool", _TOOL), ("alert", _ALERT), ("game", _GAME),
        ("environment", _ENVIRONMENT),
    ):
        if any(token in value for token in tokens):
            return group
    return "other"


def _editorial_roles(group: str, meaningful: bool) -> list[str]:
    if not meaningful:
        return []
    if group == "reaction":
        return ["reaction"]
    if group in {"impact", "sports", "cooking", "tool", "alert", "game", "environment"}:
        return ["action"]
    return []


def _bounded_region(start: float, end: float, duration: float, maximum: float) -> tuple[float, float] | None:
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end <= start:
        return None
    if end - start > maximum:
        center = (start + end) / 2
        start = max(0.0, min(center - maximum / 2, max(0.0, duration - maximum)))
        end = min(duration, start + maximum)
    return round(start, 3), round(end, 3)


def _strongest_frame_time(audio: dict[str, Any], start: float, end: float) -> float:
    frames = [
        item for item in audio.get("energy_frames", [])
        if isinstance(item, dict) and start <= float(item.get("time", -1)) <= end
    ]
    if not frames:
        return (start + end) / 2
    strongest = max(
        frames,
        key=lambda item: (
            float(item.get("activity_score", item.get("normalized_loudness", 0))),
            -abs(float(item.get("time", 0)) - (start + end) / 2),
        ),
    )
    return float(strongest["time"])


def _window_signal_peak(audio: dict[str, Any], start: float, end: float) -> float:
    return max((
        float(item.get("activity_score", item.get("normalized_loudness", 0)))
        for item in audio.get("energy_frames", [])
        if isinstance(item, dict) and start <= float(item.get("time", -1)) < end
    ), default=0.0)


def _region_overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    overlap = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    return overlap / max(0.001, min(first["end"] - first["start"], second["end"] - second["start"]))
