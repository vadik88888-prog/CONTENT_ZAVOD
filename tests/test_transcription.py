from __future__ import annotations

import sys
from types import SimpleNamespace

from app.config import AppConfig
from app.cuda_runtime import CudaRuntimeProbe
from app.transcription import transcribe


def test_auto_device_retries_on_cpu_when_cuda_initialization_fails(tmp_path, monkeypatch) -> None:
    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            if device == "cuda":
                raise RuntimeError("cublas64_12.dll is not found")

        def transcribe(self, audio_path, language, word_timestamps, vad_filter):
            word = SimpleNamespace(start=0.0, end=1.0, word="hello", probability=0.9)
            segment = SimpleNamespace(start=0.0, end=1.0, text="hello", words=[word])
            return iter([segment]), SimpleNamespace(language="en", language_probability=0.99)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    transcript = transcribe(
        audio, "source", 1, AppConfig(device="auto"), tmp_path / "transcript.json"
    )

    assert transcript["runtime"]["device"] == "cpu"
    assert "cublas64_12.dll" in transcript["runtime"]["fallback_reason"]


def test_incomplete_cuda_runtime_uses_cpu_before_creating_whisper_model(tmp_path, monkeypatch) -> None:
    created_devices: list[str] = []

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            created_devices.append(device)
            assert device == "cpu"
            assert compute_type == "int8"

        def transcribe(self, audio_path, language, word_timestamps, vad_filter):
            word = SimpleNamespace(start=0.0, end=1.0, word="hello", probability=0.9)
            segment = SimpleNamespace(start=0.0, end=1.0, text="hello", words=[word])
            return iter([segment]), SimpleNamespace(language="en", language_probability=0.99)

    monkeypatch.setattr(
        "app.transcription.probe_cuda_runtime",
        lambda: CudaRuntimeProbe(
            device_count=1,
            usable=False,
            fallback_reason="CUDA runtime incomplete: required cublas64_12.dll is unavailable",
            required_libraries=("cublas64_12.dll",),
        ),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    transcript = transcribe(
        audio, "source", 1, AppConfig(device="auto"), tmp_path / "transcript.json"
    )

    assert created_devices == ["cpu"]
    assert transcript["runtime"]["selected_device"] == "cpu"
    assert transcript["runtime"]["fallback_reason"].startswith("CUDA runtime incomplete")
    assert transcript["runtime"]["transcription_duration_seconds"] == transcript["processing_duration_seconds"]


def test_explicit_cuda_also_uses_cpu_when_its_runtime_is_incomplete(tmp_path, monkeypatch) -> None:
    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            assert (device, compute_type) == ("cpu", "int8")

        def transcribe(self, audio_path, language, word_timestamps, vad_filter):
            return iter([]), SimpleNamespace(language="en", language_probability=0.99)

    monkeypatch.setattr(
        "app.transcription.probe_cuda_runtime",
        lambda: CudaRuntimeProbe(1, False, "CUDA runtime incomplete: required cublas64_12.dll is unavailable"),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    transcript = transcribe(
        audio, "source", 1, AppConfig(device="cuda"), tmp_path / "transcript.json"
    )

    assert transcript["runtime"]["device"] == "cpu"
    assert transcript["runtime"]["fallback_reason"].startswith("CUDA runtime incomplete")
