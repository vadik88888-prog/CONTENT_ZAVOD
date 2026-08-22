from __future__ import annotations

import math
import statistics
import struct
import wave
from pathlib import Path
from typing import Any

from app.config import AudioAnalysisConfig
from app.errors import StageError


AUDIO_SIGNAL_ANALYSIS_VERSION = "audio-signal.local.1"


def analyse_audio(path: Path, config: AudioAnalysisConfig) -> dict[str, Any]:
    """Stream the prepared mono PCM once and persist compact source-relative evidence."""

    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            if channels != 1 or sample_width != 2:
                raise StageError("Audio analysis expects prepared PCM WAV 16-bit mono.")
            window_frames = max(1, round(sample_rate * config.window_seconds))
            values: list[dict[str, float]] = []
            frame_index = 0
            while True:
                raw = source.readframes(window_frames)
                if not raw:
                    break
                samples = struct.unpack(f"<{len(raw) // 2}h", raw)
                rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples))) / 32768.0
                peak_amplitude = max((abs(sample) for sample in samples), default=0) / 32768.0
                crossings = sum(
                    (samples[index] < 0 <= samples[index + 1]) or (samples[index] >= 0 > samples[index + 1])
                    for index in range(len(samples) - 1)
                )
                values.append({
                    "time": round(frame_index / sample_rate, 3),
                    "audio_energy": round(rms, 6),
                    "peak_amplitude": round(peak_amplitude, 6),
                    "zero_crossing_rate": round(crossings / max(1, len(samples) - 1), 6),
                })
                frame_index += len(samples)
    except (wave.Error, EOFError, OSError) as error:
        return _unavailable(config, error)

    duration = frames / sample_rate
    _enrich_signal_frames(values, config)
    silences = _silences(values, config)
    activity = _state_intervals(
        values, duration, config.window_seconds, "activity_score", config.activity_threshold,
        config.min_activity_seconds, active=True,
    )
    dead_zones = _state_intervals(
        values, duration, config.window_seconds, "activity_score", config.activity_threshold,
        config.dead_zone_seconds, active=False,
    )
    peak_regions = _peak_regions(values, duration, config)
    changes = [
        abs(values[index]["relative_loudness"] - values[index - 1]["relative_loudness"])
        for index in range(1, len(values))
    ]
    active_seconds = sum(float(item["end"]) - float(item["start"]) for item in activity)
    dead_seconds = sum(float(item["end"]) - float(item["start"]) for item in dead_zones)
    return {
        "schema_version": config.signal_schema_version,
        "analysis_version": AUDIO_SIGNAL_ANALYSIS_VERSION,
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 3),
        "window_seconds": config.window_seconds,
        "analysis_passes": 1,
        "prepared_pcm_reads": 1,
        "source_video_decodes": 0,
        "energy_frames": values,
        "silence_intervals": silences,
        "activity_intervals": activity,
        "dead_zones": dead_zones,
        "peak_regions": peak_regions,
        "energy_peak": round(max((item["audio_energy"] for item in values), default=0.0), 6),
        "audio_energy_change_peak": round(max(changes, default=0.0), 6),
        "signal_summary": {
            "activity_ratio": round(active_seconds / duration, 6) if duration else 0.0,
            "dead_zone_ratio": round(dead_seconds / duration, 6) if duration else 0.0,
            "relative_loudness_mean": round(_mean(values, "relative_loudness"), 6),
            "spike_peak": round(max((item["spike_score"] for item in values), default=0.0), 6),
            "onset_count": sum(item["onset_score"] >= config.onset_threshold for item in values),
            "burst_peak": round(max((item["burst_score"] for item in values), default=0.0), 6),
            "noisiness_mean": round(_mean(values, "noisiness"), 6),
            "peak_region_count": len(peak_regions),
        },
    }


def window_audio_features(start: float, end: float, audio: dict[str, Any]) -> dict[str, float]:
    frames = [item for item in audio.get("energy_frames", []) if start <= float(item["time"]) <= end]
    values = [float(item.get("normalized_loudness", 0)) for item in frames]
    relative = [float(item.get("relative_loudness", item.get("normalized_loudness", 0))) for item in frames]
    changes = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
    duration = max(0.001, end - start)
    dead_seconds = _overlap_seconds(start, end, audio.get("dead_zones", []))
    active_seconds = _overlap_seconds(start, end, audio.get("activity_intervals", []))
    return {
        "audio_energy": round(sum(values) / len(values), 3) if values else 0.0,
        "audio_energy_change": round(sum(changes) / len(changes), 3) if changes else 0.0,
        "relative_loudness": round(sum(relative) / len(relative), 3) if relative else 0.0,
        "audio_activity_ratio": round(active_seconds / duration, 3),
        "audio_dead_zone_ratio": round(dead_seconds / duration, 3),
        "audio_spike_peak": round(max((float(item.get("spike_score", 0)) for item in frames), default=0.0), 3),
        "audio_onset_count": float(sum(float(item.get("onset_detected", 0)) >= 1.0 for item in frames)),
        "audio_burst_peak": round(max((float(item.get("burst_score", 0)) for item in frames), default=0.0), 3),
        "audio_noisiness": round(sum(float(item.get("noisiness", 0)) for item in frames) / len(frames), 3) if frames else 0.0,
        "silence_before": _near_silence(start, audio.get("silence_intervals", [])),
        "silence_after": _near_silence(end, audio.get("silence_intervals", [])),
    }


def _unavailable(config: AudioAnalysisConfig, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": config.signal_schema_version,
        "analysis_version": AUDIO_SIGNAL_ANALYSIS_VERSION,
        "sample_rate": 0,
        "duration_seconds": 0.0,
        "window_seconds": config.window_seconds,
        "analysis_passes": 0,
        "prepared_pcm_reads": 0,
        "source_video_decodes": 0,
        "energy_frames": [],
        "silence_intervals": [],
        "activity_intervals": [],
        "dead_zones": [],
        "peak_regions": [],
        "energy_peak": 0.0,
        "audio_energy_change_peak": 0.0,
        "signal_summary": {},
        "warning": f"Audio features unavailable: {error}",
    }


def _enrich_signal_frames(values: list[dict[str, float]], config: AudioAnalysisConfig) -> None:
    energies = [float(item["audio_energy"]) for item in values]
    peak = max(energies, default=0.0)
    median = statistics.median(energies) if energies else 0.0
    p95 = _quantile(energies, 0.95)
    scale = max(p95 - median, p95 * 0.35, 1e-9)
    onset_flags: list[bool] = []
    recent: list[float] = []
    recent_limit = max(2, round(1.0 / max(config.window_seconds, 0.001)))
    for index, item in enumerate(values):
        energy = energies[index]
        normalized = energy / peak if peak else 0.0
        # Salience is relative to the source baseline, not merely to digital
        # silence. A constant low bed therefore does not masquerade as action.
        relative = max(0.0, min(1.0, (energy - median) / scale))
        local_baseline = statistics.median(recent) if recent else median
        spike = max(0.0, min(1.0, (energy - local_baseline) / scale))
        previous = energies[index - 1] if index else energy
        onset = max(0.0, min(1.0, (energy - previous) / scale))
        noisiness = max(0.0, min(1.0, float(item.get("zero_crossing_rate", 0)) / 0.35))
        onset_flags.append(onset >= config.onset_threshold)
        recent.append(energy)
        if len(recent) > recent_limit:
            recent.pop(0)
        item.update({
            "normalized_loudness": round(normalized, 6),
            "relative_loudness": round(relative, 6),
            "spike_score": round(spike, 6),
            "onset_score": round(onset, 6),
            "onset_detected": float(onset >= config.onset_threshold),
            "noisiness": round(noisiness, 6),
        })
    burst_radius = max(1, round(0.5 / max(config.window_seconds, 0.001)))
    for index, item in enumerate(values):
        lower = max(0, index - burst_radius)
        upper = min(len(values), index + burst_radius + 1)
        burst = sum(onset_flags[lower:upper]) / max(1, upper - lower)
        item["burst_score"] = round(min(1.0, burst * 3.0), 6)
        item["activity_score"] = round(min(1.0, max(
            float(item["relative_loudness"]),
            float(item["spike_score"]) * 0.9,
            float(item["burst_score"]) * 0.72,
        )), 6)


def _peak_regions(values: list[dict[str, float]], duration: float, config: AudioAnalysisConfig) -> list[dict[str, float | str]]:
    ranked: list[tuple[float, float, dict[str, float]]] = []
    for index, item in enumerate(values):
        relative = float(item.get("relative_loudness", 0))
        spike = float(item.get("spike_score", 0))
        onset = float(item.get("onset_score", 0))
        burst = float(item.get("burst_score", 0))
        score = max(relative * 0.72 + spike * 0.18 + burst * 0.10, spike * 0.72 + onset * 0.28)
        previous = float(values[index - 1].get("relative_loudness", 0)) if index else -1.0
        following = float(values[index + 1].get("relative_loudness", 0)) if index + 1 < len(values) else -1.0
        if score < config.spike_threshold or relative < previous or relative < following:
            continue
        ranked.append((score, float(item["time"]), item))
    selected: list[tuple[float, float, dict[str, float]]] = []
    for row in sorted(ranked, key=lambda value: (-value[0], value[1])):
        if any(abs(row[1] - existing[1]) < config.peak_min_separation_seconds for existing in selected):
            continue
        selected.append(row)
        if len(selected) >= config.peak_region_count:
            break
    half = config.peak_region_seconds / 2
    result = []
    for rank, (score, timestamp, item) in enumerate(selected, start=1):
        start = max(0.0, min(timestamp - half, max(0.0, duration - config.peak_region_seconds)))
        end = min(duration, start + config.peak_region_seconds)
        result.append({
            "region_id": f"audio-peak-{rank:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "peak_time": round(timestamp, 3),
            "score": round(score, 6),
            "relative_loudness": round(float(item.get("relative_loudness", 0)), 6),
            "spike_score": round(float(item.get("spike_score", 0)), 6),
            "onset_score": round(float(item.get("onset_score", 0)), 6),
            "burst_score": round(float(item.get("burst_score", 0)), 6),
            "source": "audio_signal_peak",
        })
    return sorted(result, key=lambda item: (float(item["start"]), str(item["region_id"])))


def _silences(values: list[dict[str, float]], config: AudioAnalysisConfig) -> list[dict[str, float]]:
    return _state_intervals(
        values,
        (float(values[-1]["time"]) + config.window_seconds) if values else 0.0,
        config.window_seconds,
        "normalized_loudness",
        config.silence_threshold,
        config.min_silence_seconds,
        active=False,
    )


def _state_intervals(
    values: list[dict[str, float]], duration: float, window_seconds: float,
    field: str, threshold: float, minimum_seconds: float, *, active: bool,
) -> list[dict[str, float]]:
    intervals: list[dict[str, float]] = []
    start: float | None = None
    for item in values:
        timestamp = float(item["time"])
        matches = float(item.get(field, 0)) >= threshold if active else float(item.get(field, 0)) < threshold
        if matches and start is None:
            start = timestamp
        elif not matches and start is not None:
            if timestamp - start >= minimum_seconds:
                intervals.append({"start": round(start, 3), "end": round(timestamp, 3)})
            start = None
    if start is not None:
        end = min(duration, float(values[-1]["time"]) + window_seconds) if values else duration
        if end - start >= minimum_seconds:
            intervals.append({"start": round(start, 3), "end": round(end, 3)})
    return intervals


def _overlap_seconds(start: float, end: float, intervals: list[dict[str, Any]]) -> float:
    total = 0.0
    for interval in intervals:
        try:
            lower = max(start, float(interval["start"]))
            upper = min(end, float(interval["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        total += max(0.0, upper - lower)
    return total


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[position]


def _mean(values: list[dict[str, float]], field: str) -> float:
    return sum(float(item.get(field, 0)) for item in values) / len(values) if values else 0.0


def _near_silence(timestamp: float, intervals: list[dict[str, float]], radius: float = 0.45) -> float:
    for interval in intervals:
        if float(interval["start"]) - radius <= timestamp <= float(interval["end"]) + radius:
            return 1.0
    return 0.0
