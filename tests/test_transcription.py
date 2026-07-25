from __future__ import annotations

import sys
from types import SimpleNamespace

from app.config import AppConfig
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
