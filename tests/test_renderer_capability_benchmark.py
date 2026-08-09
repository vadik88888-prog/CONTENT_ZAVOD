from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from validation.renderer_capability_benchmark import (
    REGISTRY,
    ease_in_out,
    geometry_delta,
    normalized_bounds,
    resolve_primitive,
    summarize_samples,
)


def test_registry_has_bounded_safe_fallbacks_for_every_backend() -> None:
    for backend in ("libass", "qt_rgba"):
        for primitive in REGISTRY:
            resolution = resolve_primitive(backend, primitive)
            assert resolution["effective"] in REGISTRY
            assert resolution["backend_mapping"] is not None

    assert resolve_primitive("libass", "per_glyph_motion") == {
        "requested": "per_glyph_motion",
        "effective": "karaoke",
        "backend_mapping": "karaoke_fill",
        "degraded": True,
    }


def test_unknown_backend_and_primitive_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark backend"):
        resolve_primitive("shell", "fade")
    with pytest.raises(ValueError, match="Unknown benchmark primitive"):
        resolve_primitive("libass", "raw_expression")


def test_easing_and_normalized_geometry_are_deterministic() -> None:
    assert ease_in_out(-1) == 0
    assert ease_in_out(0.5) == 0.5
    assert ease_in_out(2) == 1
    preview = normalized_bounds((54, 96, 486, 864), 540, 960)
    final = normalized_bounds((108, 192, 972, 1728), 1080, 1920)
    assert preview == final
    assert geometry_delta(preview, final) == 0


def test_summary_reports_median_p95_rtf_and_variability() -> None:
    samples = [
        {"wall_seconds": value, "cpu_utilization_percent": 10 + value, "gpu_utilization_percent": None, "peak_rss_mb": 100 + value, "peak_vram_delta_mb": None, "output_bytes": 1000, "output_sha256": str(value)}
        for value in (1.0, 1.2, 2.0)
    ]
    summary = summarize_samples(samples, duration_seconds=2.0)
    assert summary["repeat_count"] == 3
    assert summary["wall_seconds"]["median"] == 1.2
    assert summary["wall_seconds"]["p95"] == 2.0
    assert summary["rtf_median"] == 0.6
    assert summary["peak_rss_mb_max"] == 102.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg/Qt renderer environment is required")
def test_qt_rgba_ru_en_shaping_bounds_and_frame_hash_are_deterministic() -> None:
    # The real benchmark is a standalone process. Keep its QGuiApplication and
    # application-font lifecycle isolated from the repository's GUI tests too.
    code = """
import hashlib, json, os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from validation.renderer_capability_benchmark import _qt_image
rows = []
for text in ('Съешь ещё этих мягких булок', 'AV office 123.45'):
    first, first_meta = _qt_image(540, 960, 30, 'static', text, transparent=False)
    second, second_meta = _qt_image(540, 960, 30, 'static', text, transparent=False)
    rows.append({
        'first_hash': hashlib.sha256(first).hexdigest(),
        'second_hash': hashlib.sha256(second).hexdigest(),
        'glyph_count': first_meta['glyph_count'],
        'nonempty_glyph_bounds': first_meta['nonempty_glyph_bounds'],
        'first_glyph_hash': first_meta['glyph_indexes_sha256'],
        'second_glyph_hash': second_meta['glyph_indexes_sha256'],
        'first_bounds': first_meta['observed']['bounds'],
        'second_bounds': second_meta['observed']['bounds'],
    })
print(json.dumps(rows))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=60,
    )
    rows = json.loads(result.stdout)
    for row in rows:
        assert row["first_hash"] == row["second_hash"]
        assert row["glyph_count"] > 0
        assert row["nonempty_glyph_bounds"] > 0
        assert row["first_glyph_hash"] == row["second_glyph_hash"]
        assert row["first_bounds"] == row["second_bounds"]
    assert rows[0]["first_hash"] != rows[1]["first_hash"]
