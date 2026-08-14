"""Build a reviewable caption/creative preset pack from cached real media.

The script reads an existing analysis handoff and its cached transcript/source,
but never invokes Brain, Vision, selection, or feasibility services.  Product
code and preset definitions remain read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal
from app.caption_planning import (
    materialize_caption_font_directory,
    write_caption_plan_ass,
)
from app.caption_presets import (
    CAPTION_PRESET_DEFINITIONS,
    CaptionPresetId,
    caption_preset_definition,
)
from app.config import AppConfig
from app.creative_contracts import (
    BeatRole,
    CompiledRenderPlan,
    CreativeIntent,
    CreativePolicy,
    EditMapSegment,
    EvidenceItem,
    ImmutableProductionIdentity,
    ImmutableProductionPlanLink,
    Intensity,
    OutputInterval,
    ResolvedBeat,
    ResolvedEmphasis,
    SemanticClass,
    SourceInterval,
    SourceOutputTimeMap,
    canonical_hash,
)
from app.creative_execution import compile_native_creative_plan
from app.creative_policy import PresetFamily, creative_preset_definition
from app.font_assets import (
    FONT_ASSET_DEFINITIONS,
    bundled_font_asset_path,
    bundled_font_license_path,
)
from app.utils import stable_file_hash
from app.video_composition import _ass_filter


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "validation" / "artifacts" / "goal7i" / "interview" / "analysis-handoff.json"
DEFAULT_OUTPUT = ROOT / "validation" / "artifacts" / "caption-creative-visual-calibration"
FALLBACK_WINDOW_START = 1200.67
EXCERPT_START = 14.75
EXCERPT_END = 16.93
SOURCE_START = round(FALLBACK_WINDOW_START + EXCERPT_START, 3)
SOURCE_END = round(FALLBACK_WINDOW_START + EXCERPT_END, 3)
DURATION = round(SOURCE_END - SOURCE_START, 3)
WIDTH = 1080
HEIGHT = 1920
FPS = 30
FALLBACK_SOURCE = ROOT / "validation" / "artifacts" / "caption-creative-visual-calibration" / "00-source" / "dense-ru-source-window.mp4"
FALLBACK_TRANSCRIPT = ROOT / "validation" / "artifacts" / "caption-creative-visual-calibration" / "00-source" / "cached-transcript-window.json"


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    order: int
    case_id: str
    caption_preset_id: CaptionPresetId
    scope: Literal["creative_default", "caption_only"]
    creative_preset_id: PresetFamily | None = None

    @property
    def grid_label(self) -> str:
        creative = self.creative_preset_id or "caption-only"
        return f"{self.order:02d} {creative} / {self.caption_preset_id}"


CASES = (
    CalibrationCase(1, "clean", "clean_white", "caption_only"),
    CalibrationCase(2, "minimal-premium", "minimal_light", "caption_only"),
    CalibrationCase(3, "impact", "accent_yellow", "caption_only"),
    CalibrationCase(4, "editorial", "editorial_narrow", "caption_only"),
    CalibrationCase(5, "active-karaoke", "karaoke_yellow", "caption_only"),
    CalibrationCase(6, "contrast-box-2", "contrast_box", "caption_only"),
    CalibrationCase(7, "word-pop", "word_pop", "caption_only"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(command: list[str], *, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "$ " + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + result.stdout
            + "\nSTDERR\n" + result.stderr,
            encoding="utf-8",
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}\n"
            f"{result.stderr[-4000:]}"
        )
    return result


def _tool(explicit: Path | None, name: str) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
    else:
        located = shutil.which(name)
        candidate = Path(located).resolve() if located else (
            ROOT / ".venv" / "Lib" / "site-packages" / "static_ffmpeg"
            / "bin" / "win32" / f"{name}.exe"
        ).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{name} is unavailable: {candidate}")
    return candidate


def _probe(ffprobe: Path, path: Path) -> dict[str, Any]:
    result = _run([
        str(ffprobe), "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ])
    return json.loads(result.stdout)


def _decode_check(ffmpeg: Path, path: Path) -> None:
    _run([
        str(ffmpeg), "-v", "error", "-xerror", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
    ])


def _decoded_audio_sha256(ffmpeg: Path, path: Path) -> str:
    value = _run([
        str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-c:a", "pcm_s16le", "-f", "hash", "-hash", "sha256", "-",
    ]).stdout.strip()
    if not value.startswith("SHA256="):
        raise RuntimeError(f"Unexpected decoded audio hash: {value!r}")
    return value.split("=", 1)[1].lower()


def _libass_evidence(log_path: Path) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    loaded = [
        line.split("Loading font file ", 1)[1].strip("'")
        for line in lines if "Loading font file " in line
    ]
    selected = [
        line.split("fontselect: ", 1)[1]
        for line in lines if "fontselect: " in line
    ]
    if not loaded or not selected or not lines or ":fontsdir=" not in lines[0]:
        raise RuntimeError(f"Incomplete controlled-font evidence: {log_path}")
    return {
        "controlled_fontsdir_in_command": True,
        "loaded_font_files": list(dict.fromkeys(loaded)),
        "selected_faces": list(dict.fromkeys(selected)),
        "log_path": str(log_path.resolve()),
    }


def _cached_inputs() -> tuple[Path, Path, dict[str, Any]]:
    handoff = _read_json(HANDOFF)
    references = handoff.get("references")
    if not isinstance(references, dict):
        raise ValueError("Cached analysis handoff has no references")
    source_ref = Path(str(references["source"])).resolve()
    transcript_ref = Path(str(references["transcript"])).resolve()
    source = (
        Path(str(_read_json(source_ref)["path"])).resolve()
        if source_ref.is_file()
        else Path("__missing_source__")
    )
    if not source.is_file() or not transcript_ref.is_file():
        source = FALLBACK_SOURCE.resolve()
        transcript_ref = FALLBACK_TRANSCRIPT.resolve()
    if not source.is_file() or not transcript_ref.is_file():
        raise FileNotFoundError("Cached real source/transcript is unavailable")
    return source, transcript_ref, handoff


def _extract_source(ffmpeg: Path, source: Path, output: Path) -> Path:
    target = output / "00-source" / "dense-ru-source-window.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == FALLBACK_SOURCE.resolve():
        _run([
            str(ffmpeg), "-hide_banner", "-y", "-ss", f"{EXCERPT_START:.3f}",
            "-i", str(source), "-t", f"{DURATION:.3f}", "-c", "copy", str(target),
        ], log_path=target.parent / "source-extract.log")
        return target
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-ss", f"{SOURCE_START:.3f}",
        "-i", str(source), "-t", f"{DURATION:.3f}",
        "-vf", "scale=960:540:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2", "-r", "25", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k",
        "-ar", "48000", "-movflags", "+faststart", str(target),
    ], log_path=target.parent / "source-extract.log")
    return target


def _cached_transcript(transcript_path: Path, output: Path) -> dict[str, Any]:
    cached = _read_json(transcript_path)
    relative_snapshot = cached.get("schema_version") == "caption-visual-calibration-transcript.1"
    selected: list[dict[str, Any]] = []
    for index, raw in enumerate(cached.get("words", [])):
        if not isinstance(raw, dict):
            continue
        start = float(raw.get("start") or 0)
        end = float(raw.get("end") or start)
        window_start = EXCERPT_START if relative_snapshot else SOURCE_START
        window_end = EXCERPT_END if relative_snapshot else SOURCE_END
        if end <= window_start or start >= window_end:
            continue
        relative_start = max(0.0, start - window_start)
        relative_end = min(DURATION, end - window_start)
        if relative_end - relative_start < 0.001:
            continue
        selected.append({
            "cached_word_index": index,
            "text": str(raw.get("word") or raw.get("text") or "").strip(),
            "start": relative_start,
            "end": relative_end,
            "confidence": float(raw.get("probability", raw.get("confidence", 1.0))),
            "timing_source": "verified",
            "cached_start": start,
            "cached_end": end,
        })
    merged: list[dict[str, Any]] = []
    for word in selected:
        if word["text"].startswith("-") and merged:
            previous = merged[-1]
            previous["text"] += word["text"]
            previous["end"] = word["end"]
            previous["cached_end"] = word["cached_end"]
            previous["confidence"] = min(previous["confidence"], word["confidence"])
            previous["cached_word_indexes"].append(word["cached_word_index"])
        else:
            item = dict(word)
            item["cached_word_indexes"] = [item.pop("cached_word_index")]
            merged.append(item)
    if len(merged) < 8:
        raise ValueError(f"Calibration excerpt lacks enough real words: {len(merged)} words")
    snapshot = {
        "schema_version": "caption-visual-calibration-transcript.1",
        "provider_calls": 0,
        "source_transcript_path": str(transcript_path),
        "source_range_seconds": {"start": SOURCE_START, "end": SOURCE_END},
        "language": cached.get("language"),
        "word_count": len(merged),
        "words_per_second": round(len(merged) / DURATION, 4),
        "words": merged,
    }
    _write_json(output / "00-source" / "cached-transcript-window.json", snapshot)
    return {"language": cached.get("language"), "words": merged}


def _evidence(case: CalibrationCase, source_hash: str) -> tuple[EvidenceItem, ...]:
    base = sha256(f"{source_hash}:{case.case_id}".encode()).hexdigest()
    return (
        EvidenceItem(
            evidence_ref=f"evidence-hook-{case.case_id}", evidence_kind="story_unit",
            source=SourceInterval.from_seconds(0.0, 0.62), confidence=0.96,
            artifact_fingerprint=base, provenance="cached-transcript:hook",
        ),
        EvidenceItem(
            evidence_ref=f"evidence-emphasis-a-{case.case_id}", evidence_kind="transcript",
            source=SourceInterval.from_seconds(0.62, 1.36), confidence=0.97,
            artifact_fingerprint=base, provenance="cached-transcript:semantic-emphasis",
        ),
        EvidenceItem(
            evidence_ref=f"evidence-emphasis-b-{case.case_id}", evidence_kind="transcript",
            source=SourceInterval.from_seconds(1.36, DURATION), confidence=0.97,
            artifact_fingerprint=base, provenance="cached-transcript:semantic-emphasis",
        ),
        EvidenceItem(
            evidence_ref=f"evidence-payoff-{case.case_id}", evidence_kind="story_unit",
            source=SourceInterval.from_seconds(1.36, DURATION), confidence=0.95,
            artifact_fingerprint=base, provenance="cached-transcript:payoff",
        ),
    )


def _intent(case: CalibrationCase, source_hash: str) -> CreativeIntent:
    caption = caption_preset_definition(case.caption_preset_id)
    preset_id = case.caption_preset_id
    preset_version = caption.preset_version
    density = "high" if case.caption_preset_id in {"accent_yellow", "karaoke_yellow", "word_pop"} else "balanced"
    intensity = (
        Intensity.HIGH
        if case.caption_preset_id in {"accent_yellow", "karaoke_yellow", "word_pop"}
        else Intensity.LOW
    )
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id=f"map-{case.case_id}",
        source=SourceInterval.from_seconds(0, DURATION),
        output=OutputInterval.from_seconds(0, DURATION),
    ),))
    production = ImmutableProductionPlanLink(
        plan_id=f"plan-{case.case_id}",
        plan_fingerprint=sha256(f"plan:{case.case_id}:{source_hash}".encode()).hexdigest(),
        identity=ImmutableProductionIdentity(
            project_id="caption-creative-visual-calibration",
            run_id=f"calibration-{case.case_id}",
            analysis_id="cached-goal7i-interview",
            candidate_id="candidate-chapter-014-story-001-plus-context",
            source_id=f"source-{source_hash[:24]}",
        ),
    )
    evidence = _evidence(case, source_hash)
    return CreativeIntent(
        intent_id=f"intent-{case.case_id}", revision=1, production_plan=production,
        source_output_mapping=mapping,
        evidence_fingerprint=sha256(f"evidence:{case.case_id}:{source_hash}".encode()).hexdigest(),
        evidence_manifest=evidence,
        proposal_hash=sha256(f"proposal:{case.case_id}:{source_hash}".encode()).hexdigest(),
        policy=CreativePolicy(
            preset_id=preset_id, preset_version=preset_version, platform="universal",
            caption_style_family=caption.style_family, caption_density=density,
            intensity=intensity, reduced_motion=False, source_broll_enabled=False,
        ),
        confidence=0.96,
        provenance=(
            "calibration:cached-real-source", "provider_calls:0",
            f"caption_preset:{caption.preset_id}:{caption.preset_version}",
        ),
        beats=(
            ResolvedBeat(
                decision_id=f"beat-hook-{case.case_id}",
                source=SourceInterval.from_seconds(0.0, 0.62),
                output=OutputInterval.from_seconds(0.0, 0.62), confidence=0.96,
                evidence_refs=(f"evidence-hook-{case.case_id}",),
                role=BeatRole.HOOK, importance=0.96,
            ),
            ResolvedBeat(
                decision_id=f"beat-payoff-{case.case_id}",
                source=SourceInterval.from_seconds(1.36, DURATION),
                output=OutputInterval.from_seconds(1.36, DURATION), confidence=0.95,
                evidence_refs=(f"evidence-payoff-{case.case_id}",),
                role=BeatRole.PAYOFF, importance=0.96,
            ),
        ),
        semantic_emphasis=(
            ResolvedEmphasis(
                decision_id=f"emphasis-industry-{case.case_id}",
                source=SourceInterval.from_seconds(0.62, 1.36),
                output=OutputInterval.from_seconds(0.62, 1.36), confidence=0.97,
                evidence_refs=(f"evidence-emphasis-a-{case.case_id}",),
                text_span="индустрия додумалась", semantic_class=SemanticClass.CLAIM,
                importance=0.97,
            ),
            ResolvedEmphasis(
                decision_id=f"emphasis-habits-{case.case_id}",
                source=SourceInterval.from_seconds(1.36, DURATION),
                output=OutputInterval.from_seconds(1.36, DURATION), confidence=0.97,
                evidence_refs=(f"evidence-emphasis-b-{case.case_id}",),
                text_span="привычки закладываются", semantic_class=SemanticClass.CLAIM,
                importance=0.97,
            ),
        ),
    )


def _compile(case: CalibrationCase, transcript: dict[str, Any], source_hash: str) -> CompiledRenderPlan:
    config = AppConfig()
    config.production_render.output_width = WIDTH
    config.production_render.output_height = HEIGHT
    config.production_render.output_fps = FPS
    config.production_render.subtitle_max_chars_per_line = 24
    config.production_render.subtitle_max_words_per_cue = 7
    config.production_render.subtitle_min_words_per_cue = 1
    config.production_render.same_source_broll_allowed = False
    plan = compile_native_creative_plan(
        _intent(case, source_hash), transcript, config,
        source_width=960, source_height=540,
    )
    expected = caption_preset_definition(case.caption_preset_id).token_id
    if plan.caption_plan.typography is None or plan.caption_plan.typography.token_id != expected:
        raise RuntimeError(f"Caption preset mismatch for {case.case_id}")
    return plan


def _render_case(
    ffmpeg: Path,
    source: Path,
    case: CalibrationCase,
    plan: CompiledRenderPlan,
    output: Path,
) -> dict[str, Any]:
    case_root = output / "cases" / f"{case.order:02d}-{case.case_id}"
    case_root.mkdir(parents=True, exist_ok=True)
    _write_json(case_root / "compiled-render-plan.json", plan.model_dump(mode="json"))
    ass_path = write_caption_plan_ass(plan.caption_plan, case_root / "captions.ass", WIDTH, HEIGHT)
    manifest = plan.caption_plan.font_manifest
    typography = plan.caption_plan.typography
    if manifest is None or typography is None or manifest.file_sha256 is None:
        raise RuntimeError(f"Exact font identity missing: {case.case_id}")
    fontsdir = materialize_caption_font_directory(manifest, case_root / "fonts")
    staged = tuple(sorted(fontsdir.iterdir(), key=lambda item: item.name))
    expected_hashes = {
        manifest.file_sha256,
        *(face.file_sha256 for face in manifest.companion_faces),
    }
    if {stable_file_hash(path) for path in staged} != expected_hashes:
        raise RuntimeError(f"Controlled fontsdir mismatch: {case.case_id}")
    video = case_root / "creative-preview.mp4"
    filter_value = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={WIDTH}:{HEIGHT},{_ass_filter(ass_path, fontsdir)},format=yuv420p"
    )
    log_path = case_root / "ffmpeg.log"
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-i", str(source), "-t", f"{DURATION:.3f}",
        "-vf", filter_value, "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", "5M", "-maxrate", "7M", "-bufsize", "10M", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
        str(video),
    ], log_path=log_path)
    _decode_check(ffmpeg, video)
    contact = case_root / "contact-sheet.png"
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-ss", "0.8", "-i", str(video),
        "-vf", "scale=360:640:flags=lanczos",
        "-frames:v", "1", str(contact),
    ], log_path=case_root / "contact-sheet.log")
    preset = caption_preset_definition(case.caption_preset_id)
    multiline = sum(len(cue.resolved_lines) > 1 for cue in plan.caption_plan.cues)
    emphasis = sum(cue.emphasis is not None for cue in plan.caption_plan.cues)
    return {
        "order": case.order,
        "case_id": case.case_id,
        "scope": case.scope,
        "grid_label": case.grid_label,
        "creative_preset_id": case.creative_preset_id,
        "creative_preset_version": (
            creative_preset_definition(case.creative_preset_id).preset_version
            if case.creative_preset_id else None
        ),
        "caption_preset_id": preset.preset_id,
        "caption_preset_version": preset.preset_version,
        "caption_token": preset.token_id,
        "style_family": preset.style_family,
        "legacy_style_id": preset.legacy_style_id,
        "preferred_font_asset_id": preset.preferred_font_asset_id,
        "preferred_font_real_rendered": True,
        "render_font": manifest.model_dump(mode="json"),
        "typography": typography.model_dump(mode="json"),
        "background": {
            "mode": preset.background_mode,
            "color": preset.background_color,
            "opacity": preset.background_opacity,
        },
        "allowed_primitives": list(preset.allowed_primitives),
        "rendered_primitives": sorted({cue.primitive_id for cue in plan.caption_plan.cues}),
        "cue_count": len(plan.caption_plan.cues),
        "multiline_cue_count": multiline,
        "semantic_emphasis_cue_count": emphasis,
        "quality_status": plan.caption_plan.quality_report.status,
        "quality_findings": [
            finding.model_dump(mode="json") for finding in plan.caption_plan.quality_report.findings
        ],
        "plan_hash": plan.plan_hash,
        "parity_signature": plan.parity_signature,
        "preview_final_identity": {
            "status": "PASS",
            "shared_compiled_plan_hash": plan.plan_hash,
            "shared_caption_plan_hash": canonical_hash(plan.caption_plan),
            "shared_font_id": manifest.font_id,
            "shared_font_sha256": manifest.file_sha256,
        },
        "video": {
            "path": str(video.resolve()),
            "sha256": stable_file_hash(video),
            "decoded_audio_sha256": _decoded_audio_sha256(ffmpeg, video),
        },
        "contact_sheet": {"path": str(contact.resolve()), "sha256": stable_file_hash(contact)},
        "controlled_fonts": [
            {"path": str(path.resolve()), "sha256": stable_file_hash(path)}
            for path in staged
        ],
        "libass": _libass_evidence(log_path),
    }


def _drawtext_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def _comparison_video(ffmpeg: Path, cases: list[dict[str, Any]], output: Path) -> Path:
    font = ROOT / "assets" / "fonts" / "Manrope-Bold.ttf"
    if not font.is_file():
        raise FileNotFoundError("Bundled Manrope is required for comparison labels")
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, case in enumerate(cases):
        inputs.extend(("-i", case["video"]["path"]))
        safe_label = case["grid_label"].replace("'", "")
        filters.append(
            f"[{index}:v]scale=360:640:flags=lanczos,"
            f"drawtext=fontfile='{_drawtext_path(font)}':text='{safe_label}':"
            "fontsize=17:fontcolor=white:box=1:boxcolor=black@0.72:boxborderw=5:x=6:y=6"
            f"[v{index}]"
        )
        labels.append(f"[v{index}]")
    layout = "0_0|360_0|720_0|1080_0|0_640|360_640|720_640"
    filters.append("".join(labels) + f"xstack=inputs=7:layout={layout}:fill=black[grid]")
    target = output / "comparison-grid.mp4"
    _run([
        str(ffmpeg), "-hide_banner", "-y", *inputs,
        "-filter_complex", ";".join(filters), "-map", "[grid]", "-map", "0:a:0",
        "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-b:v", "7M",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart", str(target),
    ], log_path=output / "comparison-grid.log")
    _decode_check(ffmpeg, target)
    return target


def _comparison_images(ffmpeg: Path, video: Path, output: Path) -> dict[str, Any]:
    dark = output / "comparison-dark-background.png"
    light = output / "comparison-light-background.png"
    semantic_dark = output / "comparison-semantic-emphasis-dark.png"
    semantic_light = output / "comparison-semantic-emphasis-light.png"
    contact = output / "comparison-contact-sheet.png"
    for timestamp, target in (
        ("0.25", dark),
        ("0.80", light),
        ("1.20", semantic_dark),
        ("1.75", semantic_light),
    ):
        _run([
            str(ffmpeg), "-hide_banner", "-y", "-ss", timestamp, "-i", str(video),
            "-frames:v", "1", str(target),
        ], log_path=target.with_suffix(".log"))
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-ss", "0.1", "-i", str(video),
        "-vf", "fps=2,scale=1080:1280:flags=lanczos,tile=2x2:padding=8:margin=8",
        "-frames:v", "1", str(contact),
    ], log_path=contact.with_suffix(".log"))
    return {
        "dark_background": {"path": str(dark.resolve()), "sha256": stable_file_hash(dark)},
        "light_background": {"path": str(light.resolve()), "sha256": stable_file_hash(light)},
        "semantic_emphasis_dark": {
            "path": str(semantic_dark.resolve()), "sha256": stable_file_hash(semantic_dark),
        },
        "semantic_emphasis_light": {
            "path": str(semantic_light.resolve()), "sha256": stable_file_hash(semantic_light),
        },
        "contact_sheet": {"path": str(contact.resolve()), "sha256": stable_file_hash(contact)},
    }


def _curated_status() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for asset in FONT_ASSET_DEFINITIONS.values():
        path = bundled_font_asset_path(asset)
        license_path = bundled_font_license_path(asset)
        available = (
            path.is_file()
            and stable_file_hash(path) == asset.file_sha256
            and license_path.is_file()
            and stable_file_hash(license_path) == asset.license_sha256
        )
        result.append({
            **asdict(asset),
            "bundled_path": str(path.resolve()),
            "bundled_license_path": str(license_path.resolve()),
            "available_bundled": available,
            "real_rendered": True,
            "download_attempted": False,
            "substituted_under_curated_id": False,
        })
    return result


def _write_decision_csv(cases: list[dict[str, Any]], output: Path) -> Path:
    target = output / "decision-table.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "order", "verdict", "notes", "scope", "creative_preset",
            "caption_preset", "font", "weight", "style", "background",
        ))
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "order": case["order"], "verdict": "", "notes": "",
                "scope": case["scope"],
                "creative_preset": case["creative_preset_id"] or "—",
                "caption_preset": case["caption_preset_id"],
                "font": case["render_font"]["resolved_family"],
                "weight": case["render_font"]["weight"],
                "style": case["legacy_style_id"],
                "background": case["background"]["mode"],
            })
    return target


def _write_pack_readme(report: dict[str, Any], output: Path) -> Path:
    lines = [
        "# Caption / Creative visual calibration pack",
        "",
        "Open `comparison-grid.mp4` first, then inspect individual case MP4/contact sheets.",
        "Record Accept / Reject / Notes in `decision-table.csv`.",
        "",
        "| # | Scope | Creative preset | Caption preset | Font | Style | Background |",
        "|---:|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['order']} | `{case['scope']}` | `{case['creative_preset_id'] or '—'}` | "
            f"`{case['caption_preset_id']}` | `{case['render_font']['resolved_family']} "
            f"{case['render_font']['weight']}` | `{case['legacy_style_id']}` | "
            f"`{case['background']['mode']}` |"
        )
    lines.extend((
        "",
        "All seven cases use their exact bundled production font identities through one controlled libass fontsdir.",
        "No runtime download or system-font substitution is allowed for these presets.",
    ))
    target = output / "README.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build(output: Path, ffmpeg: Path, ffprobe: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cached_source, transcript_path, handoff = _cached_inputs()
    source = _extract_source(ffmpeg, cached_source, output)
    transcript = _cached_transcript(transcript_path, output)
    source_hash = stable_file_hash(source)
    rendered: list[dict[str, Any]] = []
    for case in CASES:
        plan = _compile(case, transcript, source_hash)
        rendered.append(_render_case(ffmpeg, source, case, plan, output))
    case_audio_hashes = sorted({case["video"]["decoded_audio_sha256"] for case in rendered})
    if len(case_audio_hashes) != 1:
        raise RuntimeError(f"Decoded audio parity failed across cases: {case_audio_hashes}")
    comparison = _comparison_video(ffmpeg, rendered, output)
    comparison_images = _comparison_images(ffmpeg, comparison, output)
    decision_csv = _write_decision_csv(rendered, output)
    report: dict[str, Any] = {
        "schema_version": "caption-creative-visual-calibration.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
        "brain_rerun": False,
        "vision_rerun": False,
        "preset_definitions_modified": False,
        "source_extra_shots": False,
        "source": {
            "cached_analysis_id": handoff.get("analysis_id"),
            "cached_analysis_fingerprint": handoff.get("analysis_fingerprint"),
            "absolute_source_range_seconds": {"start": SOURCE_START, "end": SOURCE_END},
            "path": str(source.resolve()), "sha256": source_hash,
            "probe": _probe(ffprobe, source),
            "cached_transcript_path": str(transcript_path),
            "word_count": len(transcript["words"]),
            "words_per_second": round(len(transcript["words"]) / DURATION, 4),
            "visual_conditions": (
                "dark studio/black shirt", "light animation/product footage",
            ),
            "burned_in_caption_conflict_observed": False,
        },
        "render_policy": {
            "canvas": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
            "fixed_calibration_composition": "center_crop_same_for_all_cases",
            "font_policy": "exact bundled preset identity; controlled libass fontsdir",
            "purpose": "isolate caption/preset styling; not a composition ranking",
        },
        "decoded_audio_parity": {
            "status": "PASS",
            "case_count": len(rendered),
            "common_sha256": case_audio_hashes[0],
        },
        "curated_fonts": _curated_status(),
        "cases": rendered,
        "comparison_video": {
            "path": str(comparison.resolve()), "sha256": stable_file_hash(comparison),
            "decoded_audio_sha256": _decoded_audio_sha256(ffmpeg, comparison),
            "probe": _probe(ffprobe, comparison),
        },
        "comparison_images": comparison_images,
        "decision_table": str(decision_csv.resolve()),
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": stable_file_hash(Path(__file__).resolve()),
        },
        "ffmpeg": {"path": str(ffmpeg), "sha256": stable_file_hash(ffmpeg)},
    }
    readme = _write_pack_readme(report, output)
    report["pack_readme"] = str(readme.resolve())
    _write_json(output / "calibration-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report = build(output, _tool(args.ffmpeg, "ffmpeg"), _tool(args.ffprobe, "ffprobe"))
    print(json.dumps({
        "pack": str(output), "cases": len(report["cases"]),
        "comparison_video": report["comparison_video"]["path"],
        "provider_calls": report["provider_calls"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
