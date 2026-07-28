from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import AppConfig
from app.content_transformation import run_content_transformation
from app.draft_preview import DraftPreviewService
from app.models import Candidate
from app.production_plan import build_production_plan
from app.semantic_extraction import build_source_context
from app.sources import local_source


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for the draft-preview smoke test")
def test_fast_draft_preview_assembles_source_segments_with_subtitles(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    subprocess.run([
        shutil.which("ffmpeg") or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "2.4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source_path),
    ], check=True)
    config = AppConfig()
    config.transformation.ai_strategy = "local_only"
    text = "First show the useful source fact. Then finish the complete standalone thought."
    candidate = Candidate("draft-preview-001", 0.2, 2.0, text, transcript_segment_ids=[0])
    transcript = {"language": "en", "segments": [{"start": 0.2, "end": 2.0, "text": text}]}
    features = {"segments": [{
        "id": 0, "start": 0.2, "end": 2.0, "sentence_start": True, "sentence_end": True,
        "speech_density": 0.7, "pause_before_seconds": 0.1, "pause_after_seconds": 0.1,
        "filler_word_ratio": 0.0, "repetition_score": 0.0,
    }]}
    context = build_source_context(
        {"id": "draft-source", "path": str(source_path)}, {}, candidate, transcript, features, {},
        {"boundaries": []}, config.transformation,
    )
    outcome = run_content_transformation(context, config.transformation, None, force_local=True)
    plan = build_production_plan(outcome, config.production)

    result = DraftPreviewService().render(plan, local_source(str(source_path)), tmp_path / "draft")

    assert result.output_file.is_file() and result.output_file.stat().st_size > 0
    assert result.subtitle_file.is_file()
    assert result.segments and all("source_start_seconds" in item for item in result.segments)
    assert result.composition["width"] == 540 and result.composition["height"] == 960
    assert result.estimated_duration_seconds > 0
    assert result.actual_duration_seconds and result.actual_duration_seconds > 0
