"""Build the final four-preset calibration pack from cached real media.

The harness joins four short, transcript-backed real-media excerpts into one
fixed reel and compiles that exact reel through the production CaptionPlan / ASS
/ libass path.  It never calls Brain, Vision, selection, or provider services.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.caption_planning import materialize_caption_font_directory, write_caption_plan_ass
from app.caption_presets import CAPTION_PRESET_DEFINITIONS, CaptionPresetId, caption_preset_definition
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
from app.utils import stable_file_hash
from app.video_composition import _ass_filter


WIDTH = 1080
HEIGHT = 1920
FPS = 30
DEFAULT_OUTPUT = ROOT / "validation" / "artifacts" / "caption-final-calibration"


@dataclass(frozen=True, slots=True)
class SourceSegment:
    segment_id: str
    source_kind: str
    source_start: float
    source_end: float
    transcript_suffix: str
    project_id: str | None
    visual_condition: str

    @property
    def duration(self) -> float:
        return round(self.source_end - self.source_start, 3)


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    order: int
    case_id: str
    preset_id: CaptionPresetId
    prior_font_size_ratio: float


SEGMENTS = (
    SourceSegment(
        "talking-head", "interview", 1187.10, 1193.20, "78b451974b0f",
        "b6bbc8e6ce7c400cb59a5488e208e548", "dark studio talking head",
    ),
    SourceSegment(
        "bright-advert", "interview", 1206.30, 1211.28, "78b451974b0f",
        "b6bbc8e6ce7c400cb59a5488e208e548", "bright Cadbury animation / advertising",
    ),
    SourceSegment(
        "food-travel", "food", 79.90, 83.55, "407d92201de7",
        "7a6c55c73bf54c1398a48d1e07a0f713", "food/travel restaurant table",
    ),
    SourceSegment(
        "dynamic-gameplay", "gameplay", 800.60, 802.40, "bc2ec91aded2",
        None, "complex dynamic PUBG background",
    ),
)
CASES = (
    CalibrationCase(2, "minimal-premium", "minimal_light", 0.029),
    CalibrationCase(3, "impact", "accent_yellow", 0.039),
    CalibrationCase(4, "editorial", "editorial_narrow", 0.034),
    CalibrationCase(7, "word-pop", "word_pop", 0.044),
)
SEMANTIC_SPANS = (
    ("semantic-retention", "индустрия не теряет", 0.21, 1.43),
    ("semantic-industry", "индустрия додумалась", 7.04, 7.94),
    ("semantic-food", "перспективное название", 11.53, 12.57),
    ("semantic-gameplay", "Цикл где", 14.95, 16.27),
)
LOCK_DIGESTS = {
    "clean_white": "4be7f555674812f677b9ff40d951dedbe4ccd31fb10f5307a01e8d9d6a194e4a",
    "karaoke_yellow": "a4039798dbf35a485b55511838ff400a2586bf9b36331ce3a12f66c50745ccb7",
    "contrast_box": "c22b083731a91fcacd180b7c4a976594c4303bf8dc2196d20206eafb65ae68b8",
}


def _run(command: list[str], log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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


def _source_path(segment: SourceSegment) -> Path:
    if segment.project_id is None:
        path = ROOT / "input" / "pubg_source.webm"
    else:
        source_root = (
            Path.home() / "AppData" / "Local" / "ContentFactoryData" / "projects"
            / segment.project_id / "sources"
        )
        path = next(source_root.glob("*.webm"))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _transcript_path(segment: SourceSegment) -> Path:
    path = next((ROOT / "work").glob(f"*{segment.transcript_suffix}/transcript.json"))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    reel_offset = 0.0
    language: str | None = None
    for segment in SEGMENTS:
        source = _source_path(segment)
        transcript_path = _transcript_path(segment)
        transcript = _read_json(transcript_path)
        language = language or str(transcript.get("language") or "ru")
        selected: list[dict[str, Any]] = []
        for raw in transcript.get("words", []):
            if not isinstance(raw, dict):
                continue
            start = float(raw.get("start") or 0.0)
            end = float(raw.get("end") or start)
            if end <= segment.source_start or start >= segment.source_end:
                continue
            relative_start = reel_offset + max(0.0, start - segment.source_start)
            relative_end = reel_offset + min(segment.duration, end - segment.source_start)
            if relative_end <= relative_start:
                continue
            item = {
                "text": str(raw.get("word") or raw.get("text") or "").strip(),
                "start": round(relative_start, 6),
                "end": round(relative_end, 6),
                "confidence": float(raw.get("probability", raw.get("confidence", 1.0))),
                "timing_source": str(raw.get("timing_source") or "verified"),
                "source_segment_id": segment.segment_id,
                "cached_start": start,
                "cached_end": end,
            }
            selected.append(item)
            words.append(item)
        if len(selected) < 2:
            raise RuntimeError(f"Insufficient cached speech in {segment.segment_id}: {len(selected)}")
        result.append({
            **asdict(segment),
            "duration": segment.duration,
            "reel_start": round(reel_offset, 3),
            "reel_end": round(reel_offset + segment.duration, 3),
            "source_path": str(source),
            "source_sha256": stable_file_hash(source),
            "transcript_path": str(transcript_path),
            "transcript_sha256": stable_file_hash(transcript_path),
            "word_count": len(selected),
        })
        reel_offset = round(reel_offset + segment.duration, 3)
    return result, {
        "language": language,
        "duration": reel_offset,
        "words": words,
        "provider_calls": 0,
        "timing_provenance": "cached transcript word timings shifted by deterministic reel offsets",
    }


def _build_reel(ffmpeg: Path, inputs: list[dict[str, Any]], output: Path) -> Path:
    target = output / "00-source" / "four-condition-real-media-reel.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    filters: list[str] = []
    streams: list[str] = []
    for index, item in enumerate(inputs):
        args.extend([
            "-ss", f"{item['source_start']:.3f}", "-t", f"{item['duration']:.3f}",
            "-i", item["source_path"],
        ])
        filters.extend((
            f"[{index}:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={WIDTH}:{HEIGHT},setsar=1,fps={FPS},setpts=PTS-STARTPTS[v{index}]",
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,atrim=duration={item['duration']:.3f},asetpts=PTS-STARTPTS[a{index}]",
        ))
        streams.extend((f"[v{index}]", f"[a{index}]"))
    filters.append("".join(streams) + f"concat=n={len(inputs)}:v=1:a=1[reelv][reela]")
    _run([
        str(ffmpeg), "-hide_banner", "-y", *args, "-filter_complex", ";".join(filters),
        "-map", "[reelv]", "-map", "[reela]", "-r", str(FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "6M", "-maxrate", "8M",
        "-bufsize", "12M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-ar", "48000", "-movflags", "+faststart", str(target),
    ], output / "00-source" / "reel-build.log")
    _decode_check(ffmpeg, target)
    return target


def _segment_transcript(transcript: dict[str, Any], segment: dict[str, Any]) -> dict[str, Any]:
    offset = float(segment["reel_start"])
    words = [
        {
            **word,
            "start": round(float(word["start"]) - offset, 6),
            "end": round(float(word["end"]) - offset, 6),
        }
        for word in transcript["words"]
        if word["source_segment_id"] == segment["segment_id"]
    ]
    return {
        "language": transcript["language"],
        "duration": float(segment["duration"]),
        "words": words,
        "provider_calls": 0,
    }


def _segment_semantic(segment: dict[str, Any]) -> tuple[str, str, float, float]:
    reel_start = float(segment["reel_start"])
    reel_end = float(segment["reel_end"])
    matches = [
        (evidence_id, text, round(start - reel_start, 3), round(end - reel_start, 3))
        for evidence_id, text, start, end in SEMANTIC_SPANS
        if reel_start <= start < end <= reel_end
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one semantic span for {segment['segment_id']}: {matches}")
    return matches[0]


def _intent(
    case: CalibrationCase,
    segment: dict[str, Any],
    source_hash: str,
) -> CreativeIntent:
    preset = caption_preset_definition(case.preset_id)
    duration = float(segment["duration"])
    evidence_id, semantic_text, semantic_start, semantic_end = _segment_semantic(segment)
    evidence = (
        EvidenceItem(
            evidence_ref=evidence_id, evidence_kind="transcript",
            source=SourceInterval.from_seconds(semantic_start, semantic_end), confidence=0.97,
            artifact_fingerprint=sha256(f"{source_hash}:{evidence_id}".encode()).hexdigest(),
            provenance="cached-transcript:semantic-emphasis",
        ),
        EvidenceItem(
            evidence_ref=f"beat-{segment['segment_id']}", evidence_kind="story_unit",
            source=SourceInterval.from_seconds(0.0, duration), confidence=0.96,
            artifact_fingerprint=sha256(
                f"{source_hash}:beat:{segment['segment_id']}".encode()
            ).hexdigest(),
            provenance=f"cached-transcript:{segment['segment_id']}",
        ),
    )
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id=f"map-{segment['segment_id']}",
        source=SourceInterval.from_seconds(0.0, duration),
        output=OutputInterval.from_seconds(0.0, duration),
    ),))
    return CreativeIntent(
        intent_id=f"intent-final-calibration-{case.case_id}",
        revision=1,
        production_plan=ImmutableProductionPlanLink(
            plan_id=f"plan-final-calibration-{case.case_id}-{segment['segment_id']}",
            plan_fingerprint=sha256(
                f"plan:{case.case_id}:{segment['segment_id']}:{source_hash}:"
                f"{segment['source_start']}:{segment['source_end']}".encode()
            ).hexdigest(),
            identity=ImmutableProductionIdentity(
                project_id="caption-final-visual-calibration",
                run_id=f"final-calibration-{case.case_id}-{segment['segment_id']}-{source_hash[:8]}",
                analysis_id="cached-analysis-multi-source-no-provider",
                candidate_id=f"candidate-{segment['segment_id']}",
                source_id=f"source-{source_hash[:24]}",
            ),
        ),
        source_output_mapping=mapping,
        evidence_fingerprint=sha256(
            f"evidence:{case.case_id}:{segment['segment_id']}:{source_hash}".encode()
        ).hexdigest(),
        evidence_manifest=evidence,
        proposal_hash=sha256(
            f"proposal:{case.case_id}:{segment['segment_id']}:{source_hash}".encode()
        ).hexdigest(),
        policy=CreativePolicy(
            preset_id=case.preset_id,
            preset_version=preset.preset_version,
            platform="universal",
            caption_style_family=preset.style_family,
            caption_density="high" if case.preset_id in {"accent_yellow", "word_pop"} else "balanced",
            intensity=Intensity.HIGH if case.preset_id in {"accent_yellow", "word_pop"} else Intensity.LOW,
            reduced_motion=False,
            source_broll_enabled=False,
        ),
        confidence=0.96,
        provenance=(
            "calibration:cached-real-sources",
            "provider_calls:0",
            f"source_segment:{segment['segment_id']}",
            f"caption_preset:{preset.preset_id}:{preset.preset_version}",
        ),
        beats=(ResolvedBeat(
            decision_id=f"beat-{case.case_id}-{segment['segment_id']}",
            source=SourceInterval.from_seconds(0.0, duration),
            output=OutputInterval.from_seconds(0.0, duration),
            confidence=0.96, evidence_refs=(f"beat-{segment['segment_id']}",),
            role=BeatRole.HOOK if segment["segment_id"] == "talking-head" else BeatRole.PAYOFF,
            importance=0.96,
        ),),
        semantic_emphasis=(
            ResolvedEmphasis(
                decision_id=f"{evidence_id}-{case.case_id}-{segment['segment_id']}",
                source=SourceInterval.from_seconds(semantic_start, semantic_end),
                output=OutputInterval.from_seconds(semantic_start, semantic_end),
                confidence=0.97, evidence_refs=(evidence_id,), text_span=semantic_text,
                semantic_class=SemanticClass.CLAIM, importance=0.97,
            ),
        ),
    )


def _compile(
    case: CalibrationCase,
    transcript: dict[str, Any],
    segment: dict[str, Any],
) -> CompiledRenderPlan:
    config = AppConfig()
    config.production_render.output_width = WIDTH
    config.production_render.output_height = HEIGHT
    config.production_render.output_fps = FPS
    config.production_render.subtitle_max_words_per_cue = 5
    config.production_render.subtitle_min_words_per_cue = 1
    config.production_render.same_source_broll_allowed = False
    plan = compile_native_creative_plan(
        _intent(case, segment, str(segment["source_sha256"])), transcript, config,
        source_width=WIDTH, source_height=HEIGHT,
    )
    expected = caption_preset_definition(case.preset_id).token_id
    if plan.caption_plan.typography is None or plan.caption_plan.typography.token_id != expected:
        raise RuntimeError(f"Caption token mismatch for {case.case_id}")
    if plan.caption_plan.quality_report.status == "BLOCKED":
        raise RuntimeError(f"Caption plan blocked for {case.case_id}")
    return plan


def _libass_evidence(log_path: Path) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    loaded = [line.split("Loading font file ", 1)[1].strip("'") for line in lines if "Loading font file " in line]
    selected = [line.split("fontselect: ", 1)[1] for line in lines if "fontselect: " in line]
    if not loaded or not selected or not lines or ":fontsdir=" not in lines[0]:
        raise RuntimeError(f"Incomplete controlled-font evidence: {log_path}")
    if any("Arial" in item for item in selected):
        raise RuntimeError(f"Silent Arial substitution detected: {selected}")
    return {
        "controlled_fontsdir_in_command": True,
        "loaded_font_files": list(dict.fromkeys(loaded)),
        "selected_faces": list(dict.fromkeys(selected)),
        "log_path": str(log_path.resolve()),
    }


def _render_case(
    ffmpeg: Path,
    reel: Path,
    case: CalibrationCase,
    plans: list[CompiledRenderPlan],
    output: Path,
    segment_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    case_root = output / "cases" / f"{case.order:02d}-{case.case_id}"
    case_root.mkdir(parents=True, exist_ok=True)
    if len(plans) != len(segment_inputs):
        raise RuntimeError(f"Plan/segment count mismatch: {case.case_id}")
    segment_videos: list[Path] = []
    manifests = []
    typographies = []
    staged_fonts: list[Path] = []
    libass_runs: list[dict[str, Any]] = []
    for segment, plan in zip(segment_inputs, plans):
        segment_root = case_root / "segments" / segment["segment_id"]
        segment_root.mkdir(parents=True, exist_ok=True)
        _write_json(segment_root / "compiled-render-plan.json", plan.model_dump(mode="json"))
        ass_path = write_caption_plan_ass(
            plan.caption_plan, segment_root / "captions.ass", WIDTH, HEIGHT,
        )
        manifest = plan.caption_plan.font_manifest
        typography = plan.caption_plan.typography
        if manifest is None or typography is None or manifest.file_sha256 is None:
            raise RuntimeError(f"Missing exact font identity: {case.case_id}/{segment['segment_id']}")
        manifests.append(manifest)
        typographies.append(typography)
        fontsdir = materialize_caption_font_directory(manifest, segment_root / "fonts")
        staged = tuple(sorted(fontsdir.iterdir(), key=lambda item: item.name))
        expected_hashes = {
            manifest.file_sha256, *(face.file_sha256 for face in manifest.companion_faces),
        }
        if {stable_file_hash(path) for path in staged} != expected_hashes:
            raise RuntimeError(f"Controlled font bytes mismatch: {case.case_id}/{segment['segment_id']}")
        staged_fonts.extend(staged)
        segment_video = segment_root / "creative-preview-segment.mp4"
        log_path = segment_root / "ffmpeg.log"
        _run([
            str(ffmpeg), "-hide_banner", "-y", "-ss", f"{segment['reel_start']:.3f}",
            "-i", str(reel), "-t", f"{segment['duration']:.3f}",
            "-vf", f"{_ass_filter(ass_path, fontsdir)},format=yuv420p", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-b:v", "6M", "-maxrate", "8M",
            "-bufsize", "12M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-ar", "48000", "-movflags", "+faststart", str(segment_video),
        ], log_path)
        _decode_check(ffmpeg, segment_video)
        segment_videos.append(segment_video)
        libass_runs.append(_libass_evidence(log_path))
    manifest = manifests[0]
    typography = typographies[0]
    if any(item != manifest for item in manifests[1:]) or any(item != typography for item in typographies[1:]):
        raise RuntimeError(f"Font/typography identity drift across segments: {case.case_id}")
    target = case_root / "creative-preview.mp4"
    concat_manifest = case_root / "segments.ffconcat"
    concat_manifest.write_text(
        "ffconcat version 1.0\n" + "".join(
            f"file '{path.resolve().as_posix()}'\n" for path in segment_videos
        ),
        encoding="utf-8",
    )
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_manifest), "-c", "copy", "-movflags", "+faststart", str(target),
    ], case_root / "concat.log")
    _decode_check(ffmpeg, target)
    screenshots: list[dict[str, Any]] = []
    for item in segment_inputs:
        timestamp = (float(item["reel_start"]) + float(item["reel_end"])) / 2
        image = case_root / "screenshots" / f"{item['segment_id']}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        _run([
            str(ffmpeg), "-hide_banner", "-y", "-ss", f"{timestamp:.3f}",
            "-i", str(target), "-frames:v", "1", str(image),
        ], image.with_suffix(".log"))
        screenshots.append({
            "condition": item["visual_condition"], "timestamp": round(timestamp, 3),
            "path": str(image.resolve()), "sha256": stable_file_hash(image),
        })
    preset = caption_preset_definition(case.preset_id)
    cues = [cue for plan in plans for cue in plan.caption_plan.cues]
    findings = [finding for plan in plans for finding in plan.caption_plan.quality_report.findings]
    word_pop_cues = [cue for cue in cues if cue.display_mode == "single_spoken_word"]
    semantic_cues = [cue for cue in cues if cue.emphasis is not None]
    return {
        "order": case.order,
        "case_id": case.case_id,
        "caption_preset_id": preset.preset_id,
        "caption_preset_version": preset.preset_version,
        "caption_token": preset.token_id,
        "policy_delta": {
            "prior_font_size_ratio": case.prior_font_size_ratio,
            "font_size_ratio": preset.font_size_ratio,
            "font_size_increase_percent": round((preset.font_size_ratio / case.prior_font_size_ratio - 1) * 100, 2),
            "outline_width_ratio": preset.outline_width_ratio,
            "shadow_ratio": preset.shadow_ratio,
        },
        "render_font": manifest.model_dump(mode="json"),
        "typography": typography.model_dump(mode="json"),
        "quality_status": "PASS" if all(
            plan.caption_plan.quality_report.status == "PASS" for plan in plans
        ) else "PASS_WITH_WARNINGS",
        "segment_quality_statuses": {
            segment["segment_id"]: plan.caption_plan.quality_report.status
            for segment, plan in zip(segment_inputs, plans)
        },
        "quality_findings": [item.model_dump(mode="json") for item in findings],
        "cue_count": len(cues),
        "single_spoken_word_cue_count": len(word_pop_cues),
        "semantic_cue_count": len(semantic_cues),
        "semantic_single_word_cue_count": len([
            cue for cue in word_pop_cues if cue.emphasis is not None
        ]),
        "safe_zone_bounds": {
            "min_x": min(cue.normalized_bounds.x for cue in cues),
            "max_right": max(cue.normalized_bounds.x + cue.normalized_bounds.width for cue in cues),
            "min_y": min(cue.normalized_bounds.y for cue in cues),
            "max_bottom": max(cue.normalized_bounds.y + cue.normalized_bounds.height for cue in cues),
            "overlap_finding_count": len([
                finding for finding in findings
                if "OVERLAP" in finding.code or "COLLISION" in finding.code
            ]),
        },
        "plan_hashes": [plan.plan_hash for plan in plans],
        "parity_signatures": [plan.parity_signature for plan in plans],
        "preview_final_identity": {
            "status": "PASS",
            "shared_compiled_plan_hashes": [plan.plan_hash for plan in plans],
            "shared_caption_plan_hashes": [canonical_hash(plan.caption_plan) for plan in plans],
            "shared_font_id": manifest.font_id,
            "shared_font_sha256": manifest.file_sha256,
        },
        "video": {
            "path": str(target.resolve()), "sha256": stable_file_hash(target),
            "decoded_audio_sha256": _decoded_audio_sha256(ffmpeg, target),
        },
        "screenshots": screenshots,
        "controlled_fonts": [
            {"path": str(path.resolve()), "sha256": stable_file_hash(path)}
            for path in staged_fonts
        ],
        "libass": {
            "controlled_fontsdir_in_every_segment": True,
            "runs": libass_runs,
            "selected_faces": list(dict.fromkeys(
                face for run in libass_runs for face in run["selected_faces"]
            )),
        },
    }


def _drawtext_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def _comparison_grid(ffmpeg: Path, cases: list[dict[str, Any]], output: Path) -> Path:
    font = ROOT / "assets" / "fonts" / "Manrope-Bold.ttf"
    args: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, case in enumerate(cases):
        args.extend(("-i", case["video"]["path"]))
        label = f"{case['order']} {case['case_id']}"
        filters.append(
            f"[{index}:v]scale=540:960:flags=lanczos,"
            f"drawtext=fontfile='{_drawtext_path(font)}':text='{label}':fontsize=24:"
            "fontcolor=white:box=1:boxcolor=black@0.72:boxborderw=7:x=10:y=10"
            f"[v{index}]"
        )
        labels.append(f"[v{index}]")
    filters.append("".join(labels) + "xstack=inputs=4:layout=0_0|540_0|0_960|540_960[grid]")
    target = output / "comparison-grid-4-presets.mp4"
    _run([
        str(ffmpeg), "-hide_banner", "-y", *args, "-filter_complex", ";".join(filters),
        "-map", "[grid]", "-map", "0:a:0", "-r", str(FPS), "-c:v", "libx264",
        "-preset", "veryfast", "-b:v", "8M", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-b:a", "160k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(target),
    ], output / "comparison-grid.log")
    _decode_check(ffmpeg, target)
    return target


def _comparison_sequential(ffmpeg: Path, cases: list[dict[str, Any]], output: Path) -> Path:
    font = ROOT / "assets" / "fonts" / "Manrope-Bold.ttf"
    args: list[str] = []
    filters: list[str] = []
    streams: list[str] = []
    for index, case in enumerate(cases):
        args.extend(("-i", case["video"]["path"]))
        label = f"{case['order']} {case['case_id']}"
        filters.extend((
            f"[{index}:v]drawtext=fontfile='{_drawtext_path(font)}':text='{label}':fontsize=44:"
            "fontcolor=white:box=1:boxcolor=black@0.72:boxborderw=12:x=24:y=24,"
            f"setpts=PTS-STARTPTS[v{index}]",
            f"[{index}:a]asetpts=PTS-STARTPTS[a{index}]",
        ))
        streams.extend((f"[v{index}]", f"[a{index}]"))
    filters.append("".join(streams) + f"concat=n={len(cases)}:v=1:a=1[seqv][seqa]")
    target = output / "comparison-sequential-4-presets.mp4"
    _run([
        str(ffmpeg), "-hide_banner", "-y", *args, "-filter_complex", ";".join(filters),
        "-map", "[seqv]", "-map", "[seqa]", "-r", str(FPS), "-c:v", "libx264",
        "-preset", "veryfast", "-b:v", "6M", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(target),
    ], output / "comparison-sequential.log")
    _decode_check(ffmpeg, target)
    return target


def _comparison_screenshots(ffmpeg: Path, grid: Path, inputs: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    root = output / "comparison-screenshots"
    for item in inputs:
        timestamp = (float(item["reel_start"]) + float(item["reel_end"])) / 2
        target = root / f"{item['segment_id']}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        _run([
            str(ffmpeg), "-hide_banner", "-y", "-ss", f"{timestamp:.3f}",
            "-i", str(grid), "-frames:v", "1", str(target),
        ], target.with_suffix(".log"))
        result.append({
            "condition": item["visual_condition"], "timestamp": round(timestamp, 3),
            "path": str(target.resolve()), "sha256": stable_file_hash(target),
        })
    return result


def _locked_policy_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for preset_id in LOCK_DIGESTS:
        payload = json.dumps(
            asdict(CAPTION_PRESET_DEFINITIONS[preset_id]),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()
        result[preset_id] = sha256(payload).hexdigest()
    if result != LOCK_DIGESTS:
        raise RuntimeError(f"LOCK policy changed: {result}")
    return result


def build(output: Path, ffmpeg: Path, ffprobe: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Unique validation directory required: {output}")
    output.mkdir(parents=True)
    inputs, transcript = _inputs()
    _write_json(output / "00-source" / "cached-transcript-reel.json", transcript)
    reel = _build_reel(ffmpeg, inputs, output)
    reel_hash = stable_file_hash(reel)
    rendered = []
    for case in CASES:
        plans = [
            _compile(case, _segment_transcript(transcript, segment), segment)
            for segment in inputs
        ]
        rendered.append(_render_case(ffmpeg, reel, case, plans, output, inputs))
    audio_hashes = sorted({case["video"]["decoded_audio_sha256"] for case in rendered})
    if len(audio_hashes) != 1:
        raise RuntimeError(f"Decoded audio parity failed: {audio_hashes}")
    grid = _comparison_grid(ffmpeg, rendered, output)
    sequential = _comparison_sequential(ffmpeg, rendered, output)
    screenshots = _comparison_screenshots(ffmpeg, grid, inputs, output)
    report = {
        "schema_version": "caption-final-visual-calibration.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
        "brain_calls": 0,
        "vision_calls": 0,
        "selection_calls": 0,
        "source_segments": inputs,
        "reel": {
            "path": str(reel.resolve()), "sha256": reel_hash,
            "decoded_audio_sha256": _decoded_audio_sha256(ffmpeg, reel),
            "probe": _probe(ffprobe, reel),
            "cached_transcript_path": str((output / "00-source" / "cached-transcript-reel.json").resolve()),
            "word_count": len(transcript["words"]),
        },
        "render_policy": {
            "canvas": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
            "same_reel_for_all_cases": True,
            "same_caption_plan_for_preview_final": True,
            "font_policy": "exact bundled identity through controlled libass fontsdir",
        },
        "locked_presets": {
            "status": "PASS", "policy_hashes": _locked_policy_hashes(),
            "expected_policy_hashes": LOCK_DIGESTS,
        },
        "decoded_audio_parity": {
            "status": "PASS", "case_count": len(rendered), "common_sha256": audio_hashes[0],
        },
        "cases": rendered,
        "comparison_grid": {
            "path": str(grid.resolve()), "sha256": stable_file_hash(grid),
            "probe": _probe(ffprobe, grid),
        },
        "comparison_sequential": {
            "path": str(sequential.resolve()), "sha256": stable_file_hash(sequential),
            "probe": _probe(ffprobe, sequential),
        },
        "comparison_screenshots": screenshots,
        "harness": {"path": str(Path(__file__).resolve()), "sha256": stable_file_hash(Path(__file__).resolve())},
        "ffmpeg": {"path": str(ffmpeg), "sha256": stable_file_hash(ffmpeg)},
    }
    _write_json(output / "calibration-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    report = build(args.output.resolve(), _tool(args.ffmpeg, "ffmpeg"), _tool(args.ffprobe, "ffprobe"))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cases": len(report["cases"]),
        "grid": report["comparison_grid"]["path"],
        "sequential": report["comparison_sequential"]["path"],
        "provider_calls": report["provider_calls"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
