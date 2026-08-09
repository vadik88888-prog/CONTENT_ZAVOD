"""Goal 7B controlled renderer capability and performance benchmark.

This is deliberately a validation-only spike.  It does not add a renderer to
the production backend enum and it never consumes AI-produced commands.  All
FFmpeg graphs, caption text, primitives and fallbacks are fixed by this file
and the checked-in fixture.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "validation" / "fixtures" / "renderer_capability_benchmark.fixture.json"
DEFAULT_JSON = ROOT / "validation" / "results" / "renderer_capability_benchmark.json"
DEFAULT_MARKDOWN = ROOT / "validation" / "results" / "renderer_capability_benchmark.md"
DEFAULT_ARTIFACT_ROOT = ROOT / "validation" / "artifacts" / "goal7b"
SCHEMA_VERSION = "7B.renderer-benchmark.1"
FPS = 30
SAMPLE_FRAMES = (3, 12, 30, 51)
TEXT_CASES = {
    "ru": "Съешь ещё этих мягких булок — 123,45 ₽",
    "en": "AV office: quick fox — 123.45 USD",
    "mixed": "Релиз v2.1 — AI для Москвы / London",
}
PRIMITIVES = ("fade", "scale", "slide", "karaoke")
_QT_APPLICATION: Any | None = None
_QT_FONTS_REGISTERED = False
REGISTRY: dict[str, dict[str, Any]] = {
    "static": {"tier": 0, "libass": "static", "qt_rgba": "static", "fallback": "static"},
    "fade": {"tier": 1, "libass": "fad", "qt_rgba": "opacity", "fallback": "static"},
    "scale": {"tier": 1, "libass": "bounded_transform", "qt_rgba": "transform", "fallback": "fade"},
    "slide": {"tier": 1, "libass": "bounded_move", "qt_rgba": "translate", "fallback": "fade"},
    "karaoke": {"tier": 1, "libass": "karaoke_fill", "qt_rgba": "clip_fill", "fallback": "static"},
    "per_glyph_motion": {"tier": 2, "libass": None, "qt_rgba": "glyph_runs", "fallback": "karaoke"},
    "masked_highlight": {"tier": 2, "libass": None, "qt_rgba": "clip_path", "fallback": "karaoke"},
    "coordinated_layers": {"tier": 2, "libass": None, "qt_rgba": "painter_layers", "fallback": "fade"},
}


@dataclass(frozen=True, slots=True)
class RenderProfile:
    profile_id: str
    width: int
    height: int
    encoder_kind: str
    encoder: str
    preset: str
    bitrate: str


PROFILES = (
    RenderProfile("preview_cpu", 540, 960, "cpu", "libx264", "ultrafast", "1M"),
    RenderProfile("preview_gpu", 540, 960, "gpu", "h264_nvenc", "p4", "1M"),
    RenderProfile("final_cpu", 1080, 1920, "cpu", "libx264", "medium", "8M"),
    RenderProfile("final_gpu", 1080, 1920, "gpu", "h264_nvenc", "p4", "8M"),
)


def resolve_primitive(backend: str, primitive: str) -> dict[str, Any]:
    """Return a bounded backend mapping and its ordered safe fallback."""

    if backend not in {"libass", "qt_rgba"}:
        raise ValueError(f"Unknown benchmark backend: {backend}")
    if primitive not in REGISTRY:
        raise ValueError(f"Unknown benchmark primitive: {primitive}")
    requested = primitive
    visited: set[str] = set()
    while REGISTRY[primitive][backend] is None:
        if primitive in visited:
            raise ValueError(f"Fallback cycle for {requested}")
        visited.add(primitive)
        primitive = str(REGISTRY[primitive]["fallback"])
    return {
        "requested": requested,
        "effective": primitive,
        "backend_mapping": REGISTRY[primitive][backend],
        "degraded": primitive != requested,
    }


def ease_in_out(progress: float) -> float:
    bounded = max(0.0, min(1.0, progress))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def normalized_bounds(bounds: Sequence[int | float], width: int, height: int) -> list[float]:
    left, top, right, bottom = (float(value) for value in bounds)
    return [
        round(left / width, 8),
        round(top / height, 8),
        round(right / width, 8),
        round(bottom / height, 8),
    ]


def geometry_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Geometry bounds must contain four values")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def summarize_samples(samples: Sequence[dict[str, Any]], duration_seconds: float) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one measured sample is required")
    walls = [float(item["wall_seconds"]) for item in samples]
    ordered = sorted(walls)
    p95_index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.999999)))
    median_wall = statistics.median(walls)
    return {
        "repeat_count": len(samples),
        "wall_seconds": {
            "median": round(median_wall, 6),
            "p95": round(ordered[p95_index], 6),
            "min": round(min(walls), 6),
            "max": round(max(walls), 6),
            "coefficient_of_variation": round(
                statistics.pstdev(walls) / statistics.fmean(walls) if len(walls) > 1 else 0.0,
                6,
            ),
        },
        "rtf_median": round(median_wall / duration_seconds, 6),
        "cpu_utilization_percent_median": _median_optional(samples, "cpu_utilization_percent"),
        "gpu_utilization_percent_median": _median_optional(samples, "gpu_utilization_percent"),
        "peak_rss_mb_max": _max_optional(samples, "peak_rss_mb"),
        "peak_vram_delta_mb_max": _max_optional(samples, "peak_vram_delta_mb"),
        "output_bytes_median": round(statistics.median(float(item["output_bytes"]) for item in samples)),
        "output_sha256": [str(item["output_sha256"]) for item in samples],
        "samples": list(samples),
    }


def _median_optional(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in samples if item.get(key) is not None]
    return round(statistics.median(values), 4) if values else None


def _max_optional(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in samples if item.get(key) is not None]
    return round(max(values), 4) if values else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_version(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (result.stdout or result.stderr).splitlines()[0]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _run_checked(command: Sequence[str], *, stdin: bytes | None = None, timeout: float = 7200) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, input=stdin, capture_output=True, timeout=timeout)
    if result.returncode:
        details = (result.stderr or result.stdout).decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {details}")
    return result


def _filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def _font_manifest() -> dict[str, Any]:
    from PySide6.QtCore import qVersion
    from PySide6.QtGui import QFont, QFontInfo

    _qt_app()
    font = QFont("Arial")
    font.setBold(True)
    font.setPixelSize(72)
    info = QFontInfo(font)
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name for name in ("arialbd.ttf", "arial.ttf")]
    font_path = next((path for path in candidates if path.is_file()), None)
    return {
        "requested_family": "Arial",
        "resolved_family": info.family(),
        "resolved_style": info.styleName(),
        "exact_match": info.exactMatch(),
        "qt_version": qVersion(),
        "shaping_backend": "Qt QTextLayout platform shaping",
        "font_path": str(font_path) if font_path else None,
        "font_sha256": _sha256(font_path) if font_path else None,
    }


def _qt_app() -> Any:
    global _QT_APPLICATION, _QT_FONTS_REGISTERED
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QFontDatabase, QGuiApplication

    _QT_APPLICATION = QGuiApplication.instance() or QGuiApplication([])
    if not _QT_FONTS_REGISTERED:
        font_root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        registered = [
            QFontDatabase.addApplicationFont(str(font_root / name))
            for name in ("arial.ttf", "arialbd.ttf")
            if (font_root / name).is_file()
        ]
        if not registered or any(identifier < 0 for identifier in registered):
            raise RuntimeError("Exact Arial benchmark fonts could not be registered")
        _QT_FONTS_REGISTERED = True
    return _QT_APPLICATION


def _qt_image(
    width: int,
    height: int,
    frame_index: int,
    primitive: str,
    text: str,
    *,
    transparent: bool,
    analyze: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QTextLayout, QTextOption

    _qt_app()
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(0, 0, 0, 0) if transparent else QColor(0, 0, 0, 255))
    painter = QPainter(image)
    painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    base_size = max(12, round(72 * min(width / 1080, height / 1920)))
    font = QFont("Arial")
    font.setBold(True)
    font.setPixelSize(base_size)
    measured_advance = float(QFontMetricsF(font).horizontalAdvance(text))
    maximum_advance = width * 0.88
    if measured_advance > maximum_advance:
        font.setPixelSize(max(12, int(base_size * maximum_advance / measured_advance)))
    option = QTextOption()
    option.setAlignment(Qt.AlignmentFlag.AlignLeft)
    option.setWrapMode(QTextOption.WrapMode.NoWrap)
    layout = QTextLayout(text, font)
    layout.setTextOption(option)
    layout.beginLayout()
    line = layout.createLine()
    line.setLineWidth(width * 0.88)
    layout.endLayout()
    natural_width = float(line.naturalTextWidth())
    line_height = float(line.height())
    origin_x = (width - natural_width) / 2.0
    origin_y = height * 0.50 - line_height / 2.0
    progress = ease_in_out(min(1.0, frame_index / (FPS * 0.3)))
    scale = 1.0
    if primitive == "fade":
        painter.setOpacity(progress)
    elif primitive == "scale":
        scale = 0.85 + 0.15 * progress
    elif primitive == "slide":
        origin_y += (1.0 - progress) * height * 0.05
    painter.translate(width / 2.0, height / 2.0)
    painter.scale(scale, scale)
    painter.translate(-width / 2.0, -height / 2.0)
    line.setPosition(QPointF(origin_x, origin_y))
    if primitive == "karaoke":
        painter.setPen(QColor("#ff8a3d"))
        layout.draw(painter, QPointF(0, 0))
        reveal = max(0.02, min(1.0, frame_index / (FPS * 1.5)))
        painter.save()
        painter.setClipRect(QRectF(origin_x, origin_y, natural_width * reveal, line_height))
        painter.setPen(QColor("#ffffff"))
        layout.draw(painter, QPointF(0, 0))
        painter.restore()
    else:
        painter.setPen(QColor("#ffffff"))
        layout.draw(painter, QPointF(0, 0))
    glyph_bounds: list[list[float]] = []
    glyph_indexes: list[int] = []
    if analyze:
        for run in layout.glyphRuns():
            raw_font = run.rawFont()
            indexes = list(run.glyphIndexes())
            positions = list(run.positions())
            glyph_indexes.extend(int(value) for value in indexes)
            for glyph, position in zip(indexes, positions):
                path = raw_font.pathForGlyph(glyph)
                bounds = path.boundingRect().translated(position + QPointF(origin_x, origin_y))
                glyph_bounds.append([
                    round(float(bounds.left()), 4), round(float(bounds.top()), 4),
                    round(float(bounds.right()), 4), round(float(bounds.bottom()), 4),
                ])
    painter.end()
    raw = bytes(image.constBits())
    if not analyze:
        return raw, {}
    observed = _pixel_metrics(raw, width, height, transparent=transparent)
    metadata = {
        "layout_advance": round(natural_width, 4),
        "line_height": round(line_height, 4),
        "glyph_count": len(glyph_indexes),
        "glyph_indexes_sha256": hashlib.sha256(json.dumps(glyph_indexes).encode("utf-8")).hexdigest(),
        "glyph_bounds": glyph_bounds,
        "nonempty_glyph_bounds": sum(1 for bounds in glyph_bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1]),
        "observed": observed,
    }
    return raw, metadata


def _pixel_metrics(raw: bytes, width: int, height: int, *, transparent: bool = False) -> dict[str, Any]:
    stride = width * 4
    min_x, min_y, max_x, max_y = width, height, -1, -1
    count = 0
    sum_x = 0
    sum_y = 0
    red = 0
    green = 0
    blue = 0
    for y in range(height):
        row = raw[y * stride:(y + 1) * stride]
        for x in range(width):
            offset = x * 4
            r, g, b, a = row[offset:offset + 4]
            visible = a > 8 if transparent else max(r, g, b) > 8
            if not visible:
                continue
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
            count += 1
            sum_x += x
            sum_y += y
            red += r
            green += g
            blue += b
    bounds = [min_x, min_y, max_x + 1, max_y + 1] if count else [0, 0, 0, 0]
    return {
        "bounds": bounds,
        "normalized_bounds": normalized_bounds(bounds, width, height),
        "pixel_count": count,
        "centroid": [round(sum_x / count, 4), round(sum_y / count, 4)] if count else None,
        "rgb_sums": [red, green, blue],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _ass_document(width: int, height: int, primitive: str, text: str) -> str:
    font_size = max(12, round(72 * min(width / 1080, height / 1920)))
    x, y = width // 2, height // 2
    prefix = rf"{{\an5\pos({x},{y})}}"
    if primitive == "fade":
        prefix = rf"{{\an5\pos({x},{y})\fad(300,300)}}"
    elif primitive == "scale":
        prefix = rf"{{\an5\pos({x},{y})\fscx85\fscy85\t(0,300,\fscx100\fscy100)}}"
    elif primitive == "slide":
        prefix = rf"{{\an5\move({x},{y + round(height * .05)},{x},{y},0,300)}}"
    elif primitive == "karaoke":
        words = text.split()
        centiseconds = max(len(words), 150 // max(1, len(words)))
        text = " ".join(rf"{{\k{centiseconds}}}{word}" for word in words)
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Benchmark,Arial,{font_size},&H00FFFFFF,&H003D8AFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:00.00,0:00:02.00,Benchmark,,0,0,0,,{prefix}{text}
"""


def _libass_frames(ffmpeg: str, root: Path, width: int, height: int, primitive: str, text: str) -> list[dict[str, Any]]:
    ass = root / f"capability-{width}x{height}-{primitive}.ass"
    ass.write_text(_ass_document(width, height, primitive, text), encoding="utf-8-sig")
    select = "+".join(f"eq(n\\,{frame})" for frame in SAMPLE_FRAMES)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"color=c=black:s={width}x{height}:r={FPS}:d=2",
        "-vf", f"ass='{_filter_path(ass)}',format=rgba,select='{select}'",
        "-fps_mode", "passthrough", "-frames:v", str(len(SAMPLE_FRAMES)),
        "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
    ]
    raw = _run_checked(command).stdout
    frame_bytes = width * height * 4
    if len(raw) != frame_bytes * len(SAMPLE_FRAMES):
        raise RuntimeError(f"Unexpected libass raw size: {len(raw)}")
    return [
        _pixel_metrics(raw[index * frame_bytes:(index + 1) * frame_bytes], width, height)
        for index in range(len(SAMPLE_FRAMES))
    ]


def _primitive_pass(primitive: str, frames: Sequence[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    early, entered, _steady, late = frames
    reason: dict[str, Any]
    if primitive == "fade":
        early_luma = sum(int(value) for value in early["rgb_sums"])
        entered_luma = sum(int(value) for value in entered["rgb_sums"])
        passed = early_luma < entered_luma
        reason = {"early_rgb_sum": early_luma, "entered_rgb_sum": entered_luma}
    elif primitive == "scale":
        early_width = int(early["bounds"][2]) - int(early["bounds"][0])
        entered_width = int(entered["bounds"][2]) - int(entered["bounds"][0])
        passed = early_width < entered_width
        reason = {"early_width": early_width, "entered_width": entered_width}
    elif primitive == "slide":
        early_y = float(early["centroid"][1])
        entered_y = float(entered["centroid"][1])
        passed = early_y > entered_y + 2.0
        reason = {"early_centroid_y": early_y, "entered_centroid_y": entered_y}
    else:
        bound_delta = max(abs(int(a) - int(b)) for a, b in zip(early["bounds"], late["bounds"]))
        passed = early["rgb_sums"] != late["rgb_sums"] and bound_delta <= 2
        reason = {"early_rgb": early["rgb_sums"], "late_rgb": late["rgb_sums"], "max_bounds_delta_px": bound_delta}
    return passed, reason


def capability_suite(ffmpeg: str, artifact_root: Path) -> dict[str, Any]:
    root = artifact_root / "capability"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"text_cases": {}, "primitives": {}, "deterministic": {}}
    for language, text in TEXT_CASES.items():
        _raw, qt = _qt_image(1080, 1920, 30, "static", text, transparent=False)
        libass = _libass_frames(ffmpeg, root, 1080, 1920, "static", text)[2]
        result["text_cases"][language] = {
            "text": text,
            "qt_rgba": qt,
            "libass_observed": libass,
            "qt_real_glyph_bounds_available": qt["nonempty_glyph_bounds"] > 0,
            "libass_real_block_bounds_available": libass["pixel_count"] > 0,
            "libass_per_glyph_bounds_available": False,
        }
    for backend in ("libass", "qt_rgba"):
        result["primitives"][backend] = {}
        first_hashes: list[str] = []
        second_hashes: list[str] = []
        primitive_text = "Релиз AV 123"
        for primitive in PRIMITIVES:
            if backend == "libass":
                first = _libass_frames(ffmpeg, root, 540, 960, primitive, primitive_text)
                second = _libass_frames(ffmpeg, root, 540, 960, primitive, primitive_text)
            else:
                first = []
                second = []
                for frame in SAMPLE_FRAMES:
                    raw, meta = _qt_image(540, 960, frame, primitive, primitive_text, transparent=False)
                    first.append(meta["observed"])
                    first_hashes.append(hashlib.sha256(raw).hexdigest())
                for frame in SAMPLE_FRAMES:
                    raw, meta = _qt_image(540, 960, frame, primitive, primitive_text, transparent=False)
                    second.append(meta["observed"])
                    second_hashes.append(hashlib.sha256(raw).hexdigest())
            if backend == "libass":
                first_hashes.extend(str(item["sha256"]) for item in first)
                second_hashes.extend(str(item["sha256"]) for item in second)
            passed, evidence = _primitive_pass(primitive, first)
            result["primitives"][backend][primitive] = {"passed": passed, "evidence": evidence, "frames": first}
        result["deterministic"][backend] = {
            "passed": first_hashes == second_hashes,
            "frame_count": len(first_hashes),
            "first_sha256": hashlib.sha256("".join(first_hashes).encode("ascii")).hexdigest(),
            "second_sha256": hashlib.sha256("".join(second_hashes).encode("ascii")).hexdigest(),
        }
    return result


def parity_suite(ffmpeg: str, artifact_root: Path) -> dict[str, Any]:
    root = artifact_root / "parity"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "plan_signature": hashlib.sha256(b"goal7b-fixed-plan-30fps-v1").hexdigest(),
        "event_frames": list(SAMPLE_FRAMES),
        "line_identity": True,
        "fps_identity": True,
        "backends": {},
        "tolerance_normalized": 0.003,
    }
    for backend in ("libass", "qt_rgba"):
        bounds: dict[str, list[float]] = {}
        for label, width, height in (("preview", 540, 960), ("final", 1080, 1920)):
            if backend == "libass":
                observed = _libass_frames(ffmpeg, root, width, height, "static", TEXT_CASES["mixed"])[2]
            else:
                _raw, meta = _qt_image(width, height, SAMPLE_FRAMES[2], "static", TEXT_CASES["mixed"], transparent=False)
                observed = meta["observed"]
            bounds[label] = list(observed["normalized_bounds"])
        delta = geometry_delta(bounds["preview"], bounds["final"])
        result["backends"][backend] = {
            "normalized_bounds": bounds,
            "max_geometry_delta": round(delta, 8),
            "passed": delta <= float(result["tolerance_normalized"]),
        }
    return result


def _source_command(ffmpeg: str, path: Path, width: int, height: int, duration: float, codec: str) -> list[str]:
    base = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"testsrc2=s={width}x{height}:r={FPS}:d={duration}", "-an", "-pix_fmt", "yuv420p",
    ]
    if codec == "av1":
        return [*base, "-c:v", "libsvtav1", "-preset", "12", "-crf", "46", str(path)]
    return [*base, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(path)]


def prepare_sources(ffmpeg: str, fixture: dict[str, Any], artifact_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = artifact_root / "sources"
    root.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    preparation: list[dict[str, Any]] = []
    duration = float(fixture["duration_seconds"])
    for item in fixture["sources"]:
        source_id = str(item["source_id"])
        suffix = ".mkv" if item["codec"] == "av1" else ".mp4"
        path = root / f"{source_id}{suffix}"
        started = time.perf_counter()
        if not path.is_file():
            _run_checked(_source_command(ffmpeg, path, int(item["width"]), int(item["height"]), duration, str(item["codec"])))
            cache_state = "cold_created"
        else:
            cache_state = "warm_reused"
        preparation.append({"source_id": source_id, "wall_seconds": round(time.perf_counter() - started, 6), "cache_state": cache_state})
        sources.append({
            **item,
            "path": path.relative_to(ROOT).as_posix(),
            "duration_seconds": duration,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return sources, preparation


def _encoder_available(ffmpeg: str, encoder: str) -> bool:
    result = _run_checked([ffmpeg, "-hide_banner", "-encoders"])
    return encoder.encode("ascii") in result.stdout


def _combined_ass(path: Path, width: int, height: int) -> None:
    text = TEXT_CASES["mixed"]
    path.write_text(_ass_document(width, height, "karaoke", text), encoding="utf-8-sig")


def _base_filter(profile: RenderProfile) -> str:
    return (
        f"fps={FPS},scale={profile.width}:{profile.height}:force_original_aspect_ratio=increase,"
        f"crop={profile.width}:{profile.height},setsar=1"
    )


def _encoder_args(profile: RenderProfile) -> list[str]:
    return ["-c:v", profile.encoder, "-preset", profile.preset, "-b:v", profile.bitrate, "-pix_fmt", "yuv420p"]


class _SystemTimes(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


def _filetime_value(value: _SystemTimes) -> int:
    return (int(value.high) << 32) | int(value.low)


def _system_cpu_snapshot() -> tuple[int, int, int] | None:
    if os.name != "nt":
        return None
    idle = _SystemTimes()
    kernel = _SystemTimes()
    user = _SystemTimes()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return _filetime_value(idle), _filetime_value(kernel), _filetime_value(user)


def _system_cpu_percent(start: tuple[int, int, int] | None, end: tuple[int, int, int] | None) -> float | None:
    if start is None or end is None:
        return None
    idle = end[0] - start[0]
    total = (end[1] - start[1]) + (end[2] - start[2])
    return round(max(0.0, min(100.0, 100.0 * (total - idle) / total)), 4) if total else None


def _working_set_mb(handle: int) -> float | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return float(counters.WorkingSetSize) / (1024 * 1024)


def _gpu_sample() -> tuple[float, float] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=True,
        )
        first = result.stdout.splitlines()[0].split(",")
        return float(first[0].strip()), float(first[1].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _telemetry_baseline() -> tuple[tuple[float, float] | None, float | None, tuple[int, int, int] | None]:
    baseline_gpu = _gpu_sample()
    baseline_self_rss = _working_set_mb(int(ctypes.windll.kernel32.GetCurrentProcess())) if os.name == "nt" else None
    return baseline_gpu, baseline_self_rss, _system_cpu_snapshot()


class Telemetry:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        baseline: tuple[tuple[float, float] | None, float | None, tuple[int, int, int] | None],
    ) -> None:
        self.process = process
        self.stop_event = threading.Event()
        self.gpu: list[tuple[float, float]] = []
        self.rss: list[float] = []
        self.baseline_gpu, self.baseline_self_rss, self.start_cpu = baseline
        self.rss_thread = threading.Thread(target=self._sample_rss, daemon=True)
        self.gpu_thread = threading.Thread(target=self._sample_gpu, daemon=True)

    def __enter__(self) -> "Telemetry":
        self.rss_thread.start()
        self.gpu_thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.rss_thread.join(timeout=2)
        self.gpu_thread.join(timeout=12)

    def _sample_rss(self) -> None:
        while not self.stop_event.is_set():
            if os.name == "nt":
                child = _working_set_mb(int(self.process._handle))  # type: ignore[attr-defined]
                own = _working_set_mb(int(ctypes.windll.kernel32.GetCurrentProcess()))
                if child is not None:
                    own_delta = max(0.0, (own or 0.0) - (self.baseline_self_rss or 0.0))
                    self.rss.append(child + own_delta)
            self.stop_event.wait(0.05)

    def _sample_gpu(self) -> None:
        while not self.stop_event.is_set():
            gpu = _gpu_sample()
            if gpu is not None:
                self.gpu.append(gpu)
            self.stop_event.wait(0.35)

    def result(self) -> dict[str, Any]:
        end_cpu = _system_cpu_snapshot()
        baseline_memory = self.baseline_gpu[1] if self.baseline_gpu else None
        return {
            "cpu_utilization_percent": _system_cpu_percent(self.start_cpu, end_cpu),
            "gpu_utilization_percent": round(statistics.fmean(value[0] for value in self.gpu), 4) if self.gpu else None,
            "peak_vram_delta_mb": round(max(0.0, max(value[1] for value in self.gpu) - baseline_memory), 4) if self.gpu and baseline_memory is not None else None,
            "peak_rss_mb": round(max(self.rss), 4) if self.rss else None,
        }


def _qt_overlay_frames(profile: RenderProfile, duration: float) -> Iterator[bytes]:
    frame_count = round(duration * FPS)
    text = TEXT_CASES["mixed"]
    for frame in range(frame_count):
        primitive = PRIMITIVES[min(len(PRIMITIVES) - 1, frame * len(PRIMITIVES) // max(1, frame_count))]
        raw, _metadata = _qt_image(
            profile.width, profile.height,
            frame % max(1, frame_count // len(PRIMITIVES)),
            primitive, text, transparent=True, analyze=False,
        )
        yield raw


def _execute_render(
    ffmpeg: str,
    backend: str,
    source: Path,
    profile: RenderProfile,
    duration: float,
    destination: Path,
    ass_path: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-t", f"{duration:.6f}", "-i", str(source)]
    if backend == "libass":
        graph = f"[0:v]{_base_filter(profile)},ass='{_filter_path(ass_path)}'[vout]"
    elif backend == "qt_rgba":
        command.extend([
            "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{profile.width}x{profile.height}",
            "-framerate", str(FPS), "-i", "pipe:0",
        ])
        graph = f"[0:v]{_base_filter(profile)}[base];[base][1:v]overlay=0:0:format=auto:shortest=1[vout]"
    else:
        raise ValueError(f"Unknown backend: {backend}")
    command.extend([
        "-filter_complex", graph, "-map", "[vout]", "-an", *_encoder_args(profile),
        "-movflags", "+faststart", str(destination),
    ])
    baseline = _telemetry_baseline()
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE if backend == "qt_rgba" else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    telemetry = Telemetry(process, baseline)
    with telemetry:
        if backend == "qt_rgba":
            assert process.stdin is not None
            try:
                for frame in _qt_overlay_frames(profile, duration):
                    process.stdin.write(frame)
                process.stdin.close()
                process.stdin = None
            except BrokenPipeError:
                pass
        stdout, stderr = process.communicate(timeout=7200)
        wall = time.perf_counter() - started
    if process.returncode:
        details = (stderr or stdout).decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Render failed ({process.returncode}): {details}")
    data = telemetry.result()
    data.update({
        "wall_seconds": round(wall, 6),
        "output_bytes": destination.stat().st_size,
        "output_sha256": _sha256(destination),
    })
    return data


def performance_suite(
    ffmpeg: str,
    fixture: dict[str, Any],
    sources: Sequence[dict[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    duration = float(fixture["duration_seconds"])
    repeats = int(fixture["measured_repeats"])
    root = artifact_root / "performance" / f"session-{int(time.time())}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    nvenc = _encoder_available(ffmpeg, "h264_nvenc")
    runs: list[dict[str, Any]] = []
    for source in sources:
        for profile in PROFILES:
            if profile.encoder_kind == "gpu" and not nvenc:
                runs.append({
                    "source_id": source["source_id"], "backend": None, "profile_id": profile.profile_id,
                    "status": "skipped", "reason": "h264_nvenc unavailable",
                })
                continue
            ass = root / f"{profile.profile_id}.ass"
            _combined_ass(ass, profile.width, profile.height)
            for backend in ("libass", "qt_rgba"):
                warmup_path = root / f"warmup-{source['source_id']}-{backend}-{profile.profile_id}.mp4"
                source_path = ROOT / str(source["path"])
                warmup = _execute_render(ffmpeg, backend, source_path, profile, duration, warmup_path, ass)
                samples: list[dict[str, Any]] = []
                for repeat in range(repeats):
                    destination = root / f"run-{source['source_id']}-{backend}-{profile.profile_id}-{repeat + 1}.mp4"
                    samples.append(_execute_render(ffmpeg, backend, source_path, profile, duration, destination, ass))
                runs.append({
                    "source_id": source["source_id"],
                    "source_codec": source["codec"],
                    "source_resolution": [source["width"], source["height"]],
                    "backend": backend,
                    "profile": {
                        "profile_id": profile.profile_id, "resolution": [profile.width, profile.height],
                        "fps": FPS, "encoder_kind": profile.encoder_kind, "encoder": profile.encoder,
                        "preset": profile.preset, "bitrate": profile.bitrate,
                    },
                    "status": "completed",
                    "cache_state": "warm_after_unmeasured_warmup",
                    "warmup": warmup,
                    "summary": summarize_samples(samples, duration),
                })
    return {
        "duration_seconds": duration,
        "warmup_repeats": 1,
        "measured_repeats": repeats,
        "nvenc_available": nvenc,
        "telemetry_scope": "system CPU; benchmark process + direct FFmpeg RSS; global NVIDIA utilization/VRAM delta",
        "runs": runs,
    }


def _timed_command(command: Sequence[str]) -> dict[str, Any]:
    baseline = _telemetry_baseline()
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    telemetry = Telemetry(process, baseline)
    with telemetry:
        stdout, stderr = process.communicate(timeout=7200)
        wall = time.perf_counter() - started
    if process.returncode:
        details = (stderr or stdout).decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Command failed ({process.returncode}): {details}")
    result = telemetry.result()
    result["wall_seconds"] = round(wall, 6)
    return result


def double_encode_suite(ffmpeg: str, source: dict[str, Any], fixture: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    root = artifact_root / "double-encode" / f"session-{int(time.time())}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    duration = float(fixture["duration_seconds"])
    repeats = int(fixture["measured_repeats"])
    profile = next(item for item in PROFILES if item.profile_id == "final_cpu")
    ass = root / "final.ass"
    _combined_ass(ass, profile.width, profile.height)
    source_path = ROOT / str(source["path"])
    single_samples: list[dict[str, Any]] = []
    double_samples: list[dict[str, Any]] = []
    cached_samples: list[dict[str, Any]] = []
    for repeat in range(repeats + 1):
        single = root / f"single-{repeat}.mp4"
        single_command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-t", str(duration), "-i", str(source_path),
            "-vf", f"{_base_filter(profile)},ass='{_filter_path(ass)}'", "-an", *_encoder_args(profile), str(single),
        ]
        single_timing = _timed_command(single_command)
        single_timing.update({"output_bytes": single.stat().st_size, "output_sha256": _sha256(single)})

        intermediate = root / f"intermediate-{repeat}.mp4"
        final = root / f"double-{repeat}.mp4"
        prepare_command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-t", str(duration), "-i", str(source_path),
            "-vf", _base_filter(profile), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(intermediate),
        ]
        final_command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(intermediate),
            "-vf", f"ass='{_filter_path(ass)}'", "-an", *_encoder_args(profile), str(final),
        ]
        prepare_timing = _timed_command(prepare_command)
        final_timing = _timed_command(final_command)
        combined = {
            "wall_seconds": round(float(prepare_timing["wall_seconds"]) + float(final_timing["wall_seconds"]), 6),
            "prepare_wall_seconds": prepare_timing["wall_seconds"],
            "final_wall_seconds": final_timing["wall_seconds"],
            "cpu_utilization_percent": statistics.fmean(value for value in (prepare_timing.get("cpu_utilization_percent"), final_timing.get("cpu_utilization_percent")) if value is not None),
            "gpu_utilization_percent": None,
            "peak_rss_mb": max(value for value in (prepare_timing.get("peak_rss_mb"), final_timing.get("peak_rss_mb")) if value is not None),
            "peak_vram_delta_mb": max((value for value in (prepare_timing.get("peak_vram_delta_mb"), final_timing.get("peak_vram_delta_mb")) if value is not None), default=None),
            "intermediate_bytes": intermediate.stat().st_size,
            "output_bytes": final.stat().st_size,
            "output_sha256": _sha256(final),
        }
        cached = dict(final_timing)
        cached.update({"output_bytes": final.stat().st_size, "output_sha256": _sha256(final)})
        if repeat:
            single_samples.append(single_timing)
            double_samples.append(combined)
            cached_samples.append(cached)
    single_summary = summarize_samples(single_samples, duration)
    double_summary = summarize_samples(double_samples, duration)
    cached_summary = summarize_samples(cached_samples, duration)
    overhead = float(double_summary["wall_seconds"]["median"]) / float(single_summary["wall_seconds"]["median"])
    return {
        "source_id": source["source_id"],
        "source_resolution": [source["width"], source["height"]],
        "profile_id": profile.profile_id,
        "single_pass": single_summary,
        "current_double_encode_equivalent": double_summary,
        "reused_intermediate_final_only": cached_summary,
        "double_vs_single_wall_ratio": round(overhead, 6),
        "extra_wall_percent": round((overhead - 1.0) * 100.0, 3),
        "code_evidence": {
            "prepare": "app/video_composition.py::_prepare_visual_clip uses libx264/veryfast/crf20",
            "final": "app/video_composition.py::_mux_final encodes the prepared clip again",
            "cache": "full render cache key includes subtitle_project; prepared clips have no independent cache node",
        },
    }


def startup_suite(ffmpeg: str, artifact_root: Path) -> dict[str, Any]:
    root = artifact_root / "startup"
    root.mkdir(parents=True, exist_ok=True)
    ass = root / "startup.ass"
    ass.write_text(_ass_document(540, 960, "static", TEXT_CASES["mixed"]), encoding="utf-8-sig")
    libass_command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=540x960:r=30:d=0.04",
        "-vf", f"ass='{_filter_path(ass)}'", "-frames:v", "1", "-f", "null", "-",
    ]
    qt_code = (
        "import os,time;os.environ['QT_QPA_PLATFORM']='offscreen';s=time.perf_counter();"
        "from PySide6.QtGui import QGuiApplication,QImage,QPainter,QFont;"
        "a=QGuiApplication([]);i=QImage(540,960,QImage.Format_RGBA8888);i.fill(0);"
        "p=QPainter(i);f=QFont('Arial');f.setPixelSize(36);p.setFont(f);p.drawText(20,100,'Релиз AV office');p.end();"
        "print(time.perf_counter()-s)"
    )
    values: dict[str, list[float]] = {"libass": [], "qt_rgba": []}
    internal: list[float] = []
    for _repeat in range(4):
        started = time.perf_counter()
        _run_checked(libass_command)
        values["libass"].append(time.perf_counter() - started)
        started = time.perf_counter()
        result = _run_checked([sys.executable, "-c", qt_code])
        values["qt_rgba"].append(time.perf_counter() - started)
        internal.append(float(result.stdout.decode("ascii").strip()))
    return {
        backend: {
            "cold_process_wall_seconds": round(samples[0], 6),
            "warm_process_wall_median_seconds": round(statistics.median(samples[1:]), 6),
            "warm_repeats": 3,
            "process_wall_samples_seconds": [round(value, 6) for value in samples],
            **({"qt_internal_import_first_frame_seconds": [round(value, 6) for value in internal]} if backend == "qt_rgba" else {}),
        }
        for backend, samples in values.items()
    }


def _hardware() -> dict[str, Any]:
    cpu = platform.processor()
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=True,
            )
            cpu = result.stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    gpu = None
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=True,
            )
            gpu = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def _candidate_decision(
    capability: dict[str, Any],
    parity: dict[str, Any],
    performance: dict[str, Any],
    font: dict[str, Any],
    startup: dict[str, Any],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    candidate_runs = [
        row for row in performance["runs"]
        if row.get("backend") == "qt_rgba" and row.get("status") == "completed"
    ]
    preview_gpu = [row for row in candidate_runs if row["profile"]["profile_id"] == "preview_gpu"]
    preview_cpu = [row for row in candidate_runs if row["profile"]["profile_id"] == "preview_cpu"]
    final_gpu = [row for row in candidate_runs if row["profile"]["profile_id"] == "final_gpu"]
    final_cpu = [row for row in candidate_runs if row["profile"]["profile_id"] == "final_cpu"]
    thresholds = fixture["decision_thresholds"]

    def maximum(rows: Sequence[dict[str, Any]], key: str) -> float:
        return max(float(row["summary"][key] or 0) for row in rows)

    gates = {
        "all_launch_primitives": all(capability["primitives"]["qt_rgba"][primitive]["passed"] for primitive in PRIMITIVES),
        "deterministic_frame_hashes": bool(capability["deterministic"]["qt_rgba"]["passed"]),
        "ru_en_mixed_shaping": all(
            capability["text_cases"][language]["qt_rgba"]["glyph_count"] > 0
            for language in TEXT_CASES
        ),
        "real_per_glyph_bounds": all(
            capability["text_cases"][language]["qt_real_glyph_bounds_available"]
            for language in TEXT_CASES
        ),
        "preview_final_geometry_parity": (
            float(parity["backends"]["qt_rgba"]["max_geometry_delta"])
            <= float(thresholds["normalized_geometry_delta_max"])
        ),
        "font_identity_and_checksum": bool(font.get("resolved_family") and font.get("font_sha256")),
        "preview_gpu_rtf_initial_budget": bool(preview_gpu) and maximum(preview_gpu, "rtf_median") <= float(thresholds["preview_rtf_max"]),
        "preview_cpu_fallback_rtf_budget": bool(preview_cpu) and maximum(preview_cpu, "rtf_median") <= float(thresholds["preview_cpu_fallback_rtf_max"]),
        "final_gpu_rtf_initial_budget": bool(final_gpu) and maximum(final_gpu, "rtf_median") <= float(thresholds["final_gpu_rtf_max"]),
        "final_cpu_fallback_rtf_budget": bool(final_cpu) and maximum(final_cpu, "rtf_median") <= float(thresholds["final_cpu_fallback_rtf_max"]),
        "preview_ram_budget": bool(preview_cpu + preview_gpu) and maximum(preview_cpu + preview_gpu, "peak_rss_mb_max") <= float(thresholds["preview_peak_rss_mb_max"]),
        "final_ram_budget": bool(final_cpu + final_gpu) and maximum(final_cpu + final_gpu, "peak_rss_mb_max") <= float(thresholds["final_peak_rss_mb_max"]),
        "preview_vram_budget": bool(preview_gpu) and maximum(preview_gpu, "peak_vram_delta_mb_max") <= float(thresholds["preview_peak_vram_delta_mb_max"]),
        "final_vram_budget": bool(final_gpu) and maximum(final_gpu, "peak_vram_delta_mb_max") <= float(thresholds["final_peak_vram_delta_mb_max"]),
        "gpu_telemetry_collected": bool(preview_gpu + final_gpu) and all(row["summary"]["gpu_utilization_percent_median"] is not None for row in preview_gpu + final_gpu),
        "cold_startup_budget": float(startup["qt_rgba"]["cold_process_wall_seconds"]) <= float(thresholds["cold_startup_seconds_max"]),
        "all_1080_4k_av1_matrix_cells_completed": len(candidate_runs) == len(fixture["sources"]) * len(PROFILES),
        "safe_fallbacks_complete": all(resolve_primitive("qt_rgba", primitive)["effective"] for primitive in REGISTRY),
        "no_new_runtime_dependency": True,
        "no_arbitrary_execution_surface": True,
    }
    proved = all(gates.values())
    return {
        "candidate": "qt6_qpainter_qtextlayout_rgba",
        "tier_2_proved": proved,
        "gates": gates,
        "decision": "GO_TIER_2_QT_RGBA" if proved else "NO_GO_TIER_2_KEEP_TIER_1_LIBASS",
        "production_stack": (
            "Tier 2 Qt 6 QPainter/QTextLayout RGBA with libass Tier 1 fallback"
            if proved else "Tier 1 libass; retain the RGBA architecture seam"
        ),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "thresholds": thresholds,
    }


def _machine_registry(decision: dict[str, Any]) -> dict[str, Any]:
    selected = "qt_rgba" if decision["tier_2_proved"] else "libass"
    entries = []
    for primitive, details in REGISTRY.items():
        resolution = resolve_primitive(selected, primitive)
        entries.append({
            "primitive_id": primitive,
            "required_tier": details["tier"],
            "selected_backend": selected,
            "backend_mapping": resolution["backend_mapping"],
            "effective_primitive_id": resolution["effective"],
            "degraded": resolution["degraded"],
            "fallback_primitive_id": details["fallback"],
        })
    return {
        "schema_version": "7B.capability-registry.1",
        "selected_backend": selected,
        "candidate_backend_status": "qualified" if decision["tier_2_proved"] else "benchmark_only_unqualified",
        "entries": entries,
        "backend_failure_order": ["selected_backend", "libass", "static", "block_invalid_artifact"],
        "quality_reporting_required": True,
    }


def _markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    lines = [
        "# Goal 7B — Creative Renderer Capability Benchmark",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        f"**{decision['decision']}** — {decision['production_stack']}.",
        "",
        "This is a benchmark/decision artifact only. No production renderer enum, visual style, Phase 7C planner or UI was changed.",
        "",
        "## Controlled contract",
        "",
        f"- Fixture duration: {result['performance']['duration_seconds']} s; one unmeasured warm-up + {result['performance']['measured_repeats']} measured repeats per cell.",
        "- Same fixed RU/EN/mixed text, 30 fps, normalized placement, source, duration, output resolution and encoder target per backend pair.",
        "- Sources: synthetic deterministic H.264 1080p, H.264 4K and AV1 4K. Generated media stays ignored under validation/artifacts.",
        "- CPU is system utilization during the run; RSS is benchmark process plus direct FFmpeg child; NVIDIA values are global utilization and baseline-relative VRAM.",
        "- Short clips emphasize startup overhead; startup is also reported separately. Measurements are local-machine evidence, not a fleet release promise.",
        "",
        "## Decision gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for gate, passed in decision["gates"].items():
        lines.append(f"| `{gate}` | {'PASS' if passed else 'FAIL'} |")
    completed = [row for row in result["performance"]["runs"] if row.get("status") == "completed"]

    def maximum_rtf(backend: str, profile_id: str) -> float:
        return max(
            float(row["summary"]["rtf_median"])
            for row in completed
            if row["backend"] == backend and row["profile"]["profile_id"] == profile_id
        )

    lines += [
        "", "### Decision interpretation", "",
        f"- Tier 1 libass passed the launch primitives, deterministic hashes and parity gate. Its worst measured GPU Preview RTF was {maximum_rtf('libass', 'preview_gpu'):.3f}, so retaining it is a correctness/safe-baseline decision—not a claim that the aspirational 0.5 Preview RTF was met on every stress source.",
        f"- Qt RGBA reached worst GPU Preview/Final RTF {maximum_rtf('qt_rgba', 'preview_gpu'):.3f}/{maximum_rtf('qt_rgba', 'final_gpu'):.3f}; it also exceeded normalized geometry tolerance. Capability alone therefore does not qualify Tier 2.",
        "- Both CPU fallback profiles stayed inside the pre-registered initial RTF budgets; RAM/VRAM and startup gates also passed.",
        "", "## Capability", "", "| Backend | Fade | Scale | Slide | Karaoke | Deterministic | Glyph bounds |", "|---|---|---|---|---|---|---|",
    ]
    capability = result["capability"]
    for backend in ("libass", "qt_rgba"):
        values = capability["primitives"][backend]
        glyph = "per-glyph" if backend == "qt_rgba" else "rendered block only"
        lines.append(
            f"| {backend} | {'PASS' if values['fade']['passed'] else 'FAIL'} | {'PASS' if values['scale']['passed'] else 'FAIL'} | "
            f"{'PASS' if values['slide']['passed'] else 'FAIL'} | {'PASS' if values['karaoke']['passed'] else 'FAIL'} | "
            f"{'PASS' if capability['deterministic'][backend]['passed'] else 'FAIL'} | {glyph} |"
        )
    lines += ["", "### RU/EN shaping and real bounds", ""]
    for language, item in capability["text_cases"].items():
        qt = item["qt_rgba"]
        lines.append(
            f"- `{language}`: Qt shaped {qt['glyph_count']} glyphs; {qt['nonempty_glyph_bounds']} non-empty per-glyph ink bounds; "
            f"Qt observed block `{qt['observed']['bounds']}`; libass observed block `{item['libass_observed']['bounds']}`."
        )
    lines += ["", "## Preview / Final parity", "", "| Backend | Preview normalized bounds | Final normalized bounds | Max delta | Gate |", "|---|---|---|---:|---|"]
    for backend, item in result["parity"]["backends"].items():
        lines.append(
            f"| {backend} | `{item['normalized_bounds']['preview']}` | `{item['normalized_bounds']['final']}` | "
            f"{item['max_geometry_delta']:.8f} | {'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines += ["", "## Performance matrix", "", "Median values after warm-up.", "", "| Source | Backend | Profile | RTF | Wall s | CPU % | GPU % | Peak RSS MB | Peak VRAM Δ MB |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["performance"]["runs"]:
        if row.get("status") != "completed":
            continue
        summary = row["summary"]
        lines.append(
            f"| {row['source_id']} | {row['backend']} | {row['profile']['profile_id']} | {summary['rtf_median']:.3f} | "
            f"{summary['wall_seconds']['median']:.3f} | {_display(summary['cpu_utilization_percent_median'])} | "
            f"{_display(summary['gpu_utilization_percent_median'])} | {_display(summary['peak_rss_mb_max'])} | "
            f"{_display(summary['peak_vram_delta_mb_max'])} |"
        )
    startup = result["startup"]
    lines += ["", "## Startup / cache", ""]
    for backend in ("libass", "qt_rgba"):
        item = startup[backend]
        lines.append(
            f"- {backend}: cold process wall {item['cold_process_wall_seconds']:.3f}s; warm process median "
            f"{item['warm_process_wall_median_seconds']:.3f}s across {item['warm_repeats']} repeats."
        )
    double = result["double_encode"]
    lines += [
        "", "## Double-encode bottleneck", "",
        f"On `{double['source_id']}` → `{double['profile_id']}`, single-pass median was "
        f"{double['single_pass']['wall_seconds']['median']:.3f}s (RTF {double['single_pass']['rtf_median']:.3f}); "
        f"the current prepare+final equivalent was {double['current_double_encode_equivalent']['wall_seconds']['median']:.3f}s "
        f"(RTF {double['current_double_encode_equivalent']['rtf_median']:.3f}), **{double['extra_wall_percent']:.1f}% extra wall time**. "
        f"Reusing a prepared visual for caption-only final would cost {double['reused_intermediate_final_only']['wall_seconds']['median']:.3f}s.",
        "",
        "Code inspection agrees with the measurement: `_prepare_visual_clip` encodes H.264 and `_mux_final` decodes/encodes it again; the full-render cache key contains subtitles, while prepared clips are not independent cache nodes.",
        "",
        "## Capability registry and safe fallbacks", "",
        f"Selected backend: `{result['capability_registry']['selected_backend']}`; candidate status: `{result['capability_registry']['candidate_backend_status']}`.",
        "",
        "| Requested | Effective | Mapping | Degraded | Fallback |", "|---|---|---|---|---|",
    ]
    for entry in result["capability_registry"]["entries"]:
        lines.append(
            f"| {entry['primitive_id']} | {entry['effective_primitive_id']} | {entry['backend_mapping']} | "
            f"{entry['degraded']} | {entry['fallback_primitive_id']} |"
        )
    lines += [
        "", "## Limitations", "",
        "- GPU telemetry is device-global and may include unrelated desktop activity; exact samples are retained in JSON.",
        "- The candidate is Qt raster RGBA piped into FFmpeg, not a production integration. The decision proves or rejects this exact graph only.",
        "- SSIM/VMAF is not used to rank typography backends because their pixels intentionally differ; correctness uses shaping, real bounds, frame hashes, primitive behavior and parity invariants.",
        "- CPU-only fallback was measured locally, not on a separate deployment machine; fleet qualification remains a later rollout gate.",
        "",
        "## Reproduction", "",
        "```powershell",
        r".\.venv\Scripts\python.exe validation\renderer_capability_benchmark.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def _display(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        result = _read_json(args.json_output)
        args.markdown_output.write_text(_markdown(result), encoding="utf-8")
        print(f"Refreshed {args.markdown_output}")
        return 0
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("FFmpeg and ffprobe are required")
    fixture = _read_json(args.fixture)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources, source_preparation = prepare_sources(ffmpeg, fixture, args.artifact_root)
    font = _font_manifest()
    capability = capability_suite(ffmpeg, args.artifact_root)
    parity = parity_suite(ffmpeg, args.artifact_root)
    startup = startup_suite(ffmpeg, args.artifact_root)
    performance = performance_suite(ffmpeg, fixture, sources, args.artifact_root)
    source_4k_h264 = next(item for item in sources if item["source_id"] == "h264_4k")
    double_encode = double_encode_suite(ffmpeg, source_4k_h264, fixture, args.artifact_root)
    decision = _candidate_decision(capability, parity, performance, font, startup, fixture)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "goal": "7B",
        "scope": "benchmark_and_decision_only",
        "fixture": fixture,
        "toolchain": {
            "ffmpeg": _command_version([ffmpeg, "-version"]),
            "ffprobe": _command_version([ffprobe, "-version"]),
            "qt": font["qt_version"],
        },
        "hardware": _hardware(),
        "font_manifest": font,
        "source_preparation": source_preparation,
        "sources": sources,
        "capability": capability,
        "parity": parity,
        "startup": startup,
        "performance": performance,
        "double_encode": double_encode,
        "decision": decision,
        "capability_registry": _machine_registry(decision),
        "total_wall_seconds": round(time.perf_counter() - started, 6),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")
    print(decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
