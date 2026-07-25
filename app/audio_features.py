from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any

from app.config import AudioAnalysisConfig
from app.errors import StageError


def analyse_audio(path: Path, config: AudioAnalysisConfig) -> dict[str, Any]:
    """Stream a mono PCM WAV and retain a compact energy time series only."""

    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            if channels != 1 or sample_width != 2:
                raise StageError("Audio analysis ожидает PCM WAV 16-bit mono из этапа preparation.")
            window_frames = max(1, round(sample_rate * config.window_seconds))
            values: list[dict[str, float]] = []
            frame_index = 0
            while True:
                raw = source.readframes(window_frames)
                if not raw:
                    break
                samples = struct.unpack(f"<{len(raw) // 2}h", raw)
                rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples))) / 32768.0
                values.append({"time": round(frame_index / sample_rate, 3), "audio_energy": round(rms, 6)})
                frame_index += len(samples)
    except (wave.Error, EOFError, OSError) as error:
        # Audio extraction has already succeeded; a non-PCM or damaged cache must
        # not prevent transcript/local fallback from finishing the pipeline.
        return {
            "sample_rate": 0,
            "duration_seconds": 0.0,
            "window_seconds": config.window_seconds,
            "energy_frames": [],
            "silence_intervals": [],
            "energy_peak": 0.0,
            "audio_energy_change_peak": 0.0,
            "warning": f"Audio features недоступны: {error}",
        }
    peak = max((item["audio_energy"] for item in values), default=0.0)
    for item in values:
        item["normalized_loudness"] = round(item["audio_energy"] / peak, 6) if peak else 0.0
    silences = _silences(values, config)
    changes = [
        abs(values[index]["normalized_loudness"] - values[index - 1]["normalized_loudness"])
        for index in range(1, len(values))
    ]
    return {
        "sample_rate": sample_rate,
        "duration_seconds": round(frames / sample_rate, 3),
        "window_seconds": config.window_seconds,
        "energy_frames": values,
        "silence_intervals": silences,
        "energy_peak": round(peak, 6),
        "audio_energy_change_peak": round(max(changes, default=0.0), 6),
    }


def window_audio_features(start: float, end: float, audio: dict[str, Any]) -> dict[str, float]:
    frames = [
        item for item in audio.get("energy_frames", [])
        if start <= float(item["time"]) <= end
    ]
    values = [float(item.get("normalized_loudness", 0)) for item in frames]
    changes = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
    return {
        "audio_energy": round(sum(values) / len(values), 3) if values else 0.0,
        "audio_energy_change": round(sum(changes) / len(changes), 3) if changes else 0.0,
        "silence_before": _near_silence(start, audio.get("silence_intervals", [])),
        "silence_after": _near_silence(end, audio.get("silence_intervals", [])),
    }


def _silences(values: list[dict[str, float]], config: AudioAnalysisConfig) -> list[dict[str, float]]:
    intervals: list[dict[str, float]] = []
    start: float | None = None
    last_time = 0.0
    for item in values:
        time = float(item["time"])
        quiet = float(item.get("normalized_loudness", 0)) <= config.silence_threshold
        if quiet and start is None:
            start = time
        if not quiet and start is not None:
            if time - start >= config.min_silence_seconds:
                intervals.append({"start": round(start, 3), "end": round(time, 3)})
            start = None
        last_time = time
    if start is not None and last_time - start + config.window_seconds >= config.min_silence_seconds:
        intervals.append({"start": round(start, 3), "end": round(last_time + config.window_seconds, 3)})
    return intervals


def _near_silence(timestamp: float, intervals: list[dict[str, float]], radius: float = 0.45) -> float:
    for interval in intervals:
        if float(interval["start"]) - radius <= timestamp <= float(interval["end"]) + radius:
            return 1.0
    return 0.0
