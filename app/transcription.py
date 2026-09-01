from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.cuda_runtime import probe_cuda_runtime
from app.errors import DependencyError, StageError
from app.models import Segment, Word
from app.utils import write_json


@dataclass(frozen=True, slots=True)
class TranscriptionRuntimeSettings:
    device: str
    compute_type: str
    fallback_reason: str | None = None


def resolve_transcription_runtime(config: AppConfig) -> TranscriptionRuntimeSettings:
    """Choose an ASR device only after checking the usable CUDA runtime."""

    if config.device == "cpu":
        return TranscriptionRuntimeSettings("cpu", "int8")

    cuda = probe_cuda_runtime()
    if cuda.usable:
        compute_type = config.compute_type if config.compute_type != "auto" else "float16"
        return TranscriptionRuntimeSettings("cuda", compute_type)

    # This also applies to an explicit CUDA preference: a partial runtime is
    # not an executable CUDA path, and CPU int8 is the defined safe fallback.
    return TranscriptionRuntimeSettings("cpu", "int8", cuda.fallback_reason)


def transcription_settings(config: AppConfig) -> tuple[str, str]:
    runtime = resolve_transcription_runtime(config)
    return runtime.device, runtime.compute_type


def transcribe(
    audio_path: Path,
    source_id: str,
    source_duration: float,
    config: AppConfig,
    destination: Path,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise DependencyError(
            "faster-whisper не установлен. Выполните: pip install -r requirements.txt"
        ) from error
    started = time.perf_counter()
    selected_runtime = resolve_transcription_runtime(config)
    device, compute_type = selected_runtime.device, selected_runtime.compute_type
    attempts = [(device, compute_type)]
    if config.device == "auto" and device == "cuda":
        attempts.append(("cpu", "int8"))
    errors: list[str] = []
    fallback_reason = selected_runtime.fallback_reason
    segments: list[Segment] = []
    words: list[Word] = []
    info: Any = None
    for current_device, current_compute_type in attempts:
        try:
            model = WhisperModel(
                config.whisper_model, device=current_device, compute_type=current_compute_type
            )
            segments_iterator, info = model.transcribe(
                str(audio_path),
                language=config.language,
                word_timestamps=True,
                vad_filter=True,
            )
            current_segments: list[Segment] = []
            current_words: list[Word] = []
            for result in segments_iterator:
                segment_words = [
                    Word(
                        start=float(word.start),
                        end=float(word.end),
                        text=str(word.word).strip(),
                        probability=float(word.probability) if word.probability is not None else None,
                    )
                    for word in (result.words or [])
                    if word.start is not None and word.end is not None and str(word.word).strip()
                ]
                current_words.extend(segment_words)
                current_segments.append(Segment(
                    start=float(result.start), end=float(result.end),
                    text=result.text.strip(), words=segment_words,
                ))
            segments, words = current_segments, current_words
            device, compute_type = current_device, current_compute_type
            break
        except Exception as error:
            errors.append(str(error))
            if current_device == "cuda" and len(attempts) > 1:
                fallback_reason = str(error)
                continue
            advice = (
                "Проверьте модель, память GPU и CUDA. Для CPU задайте device: cpu в config.yaml."
            )
            raise StageError(f"Не удалось распознать речь: {error}. {advice}") from error
    runtime = time.perf_counter() - started
    transcription_duration_seconds = round(runtime, 3)
    data = {
        "source_id": source_id,
        "language": getattr(info, "language", config.language),
        "language_probability": getattr(info, "language_probability", None),
        "duration": source_duration,
        "segments": [segment.to_dict() for segment in segments],
        "words": [word.to_dict() for word in words],
        "model": config.whisper_model,
        "runtime": {
            "device": device,
            "selected_device": device,
            "compute_type": compute_type,
            "platform": platform.platform(),
            "fallback_reason": fallback_reason,
            "transcription_duration_seconds": transcription_duration_seconds,
        },
        "processing_duration_seconds": transcription_duration_seconds,
        "empty_transcript": not bool(segments),
    }
    write_json(destination, data)
    destination.with_suffix(".txt").write_text(
        "\n".join(segment.text for segment in segments), encoding="utf-8"
    )
    return data
