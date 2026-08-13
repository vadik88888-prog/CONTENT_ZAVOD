"""Provider-free real-MP4 acceptance for friend-beta captions and fonts.

This harness deliberately starts from a persisted real source window.  It
compiles caption plans locally, stages the exact checksummed font face into an
isolated libass ``fontsdir``, and renders Preview/Final quality profiles from
the same immutable compiled plan.  It never invokes Brain, Vision, selection,
or production feasibility.
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
from typing import Any, Literal

from app.caption_planning import materialize_caption_font_directory, write_caption_plan_ass
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
    OutputInterval,
    RenderParityManifest,
    RenderProfile,
    ResolvedBeat,
    ResolvedEmphasis,
    SemanticClass,
    SourceInterval,
    SourceOutputTimeMap,
    assert_preview_final_parity,
    build_render_parity_manifest,
)
from app.creative_execution import compile_native_creative_plan
from app.creative_policy import PresetFamily, creative_preset_definition
from app.utils import stable_file_hash
from app.video_composition import _ass_filter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "validation" / "artifacts" / "goal7i" / "food" / "fallback-source-window.mp4"
DEFAULT_OUTPUT = ROOT / "validation" / "artifacts" / "preset-font-real-mp4-de4d605"
FPS = 30
PREVIEW = RenderProfile(
    profile_id="creative_preview", width=540, height=960, fps=FPS,
    video_bitrate="2M", encoder="cpu", sampling_precision="preview",
)
FINAL = RenderProfile(
    profile_id="final", width=1080, height=1920, fps=FPS,
    video_bitrate="6M", encoder="cpu", sampling_precision="full",
)


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    language: Literal["ru", "en"]
    creative_preset_id: PresetFamily
    requested_font_family: str
    words: tuple[str, ...]


CASES = (
    Case(
        "ru-minimal-arial-regular", "ru", "minimal", "Arial",
        ("Проверяем", "точный", "обычный", "шрифт", "на", "реальном", "видео"),
    ),
    Case(
        "ru-dynamic-arial-bold", "ru", "dynamic", "Arial",
        ("Точный", "жирный", "шрифт", "работает", "одинаково", "везде"),
    ),
    Case(
        "ru-documentary-segoe-bold", "ru", "documentary", "Segoe UI",
        ("Кириллица", "сохраняет", "форму", "и", "читаемость", "кадра"),
    ),
    Case(
        "en-clean-segoe-bold", "en", "clean", "Segoe UI",
        ("Exact", "bold", "face", "stays", "stable", "in", "both", "renders"),
    ),
    Case(
        "en-minimal-segoe-regular", "en", "minimal", "Segoe UI",
        ("Regular", "face", "identity", "is", "checksum", "controlled"),
    ),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(command: list[str], *, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log_path is not None:
        log_path.write_text(
            "$ " + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + result.stdout
            + "\nSTDERR\n" + result.stderr,
            encoding="utf-8",
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}\n{result.stderr[-4000:]}"
        )
    return result


def _tool(explicit: Path | None, name: str) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
    else:
        located = shutil.which(name)
        if located:
            candidate = Path(located).resolve()
        else:
            candidate = (
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
    result = _run([
        str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-c:a", "pcm_s16le", "-f", "hash", "-hash", "sha256", "-",
    ])
    prefix = "SHA256="
    value = result.stdout.strip()
    if not value.startswith(prefix) or len(value) != len(prefix) + 64:
        raise RuntimeError(f"Unexpected decoded audio hash: {value!r}")
    return value[len(prefix):].lower()


def _libass_evidence(log_path: Path) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    loaded = [line.split("Loading font file ", 1)[1].strip("'") for line in lines if "Loading font file " in line]
    selected = [line.split("fontselect: ", 1)[1] for line in lines if "fontselect: " in line]
    fontsdir_command = lines[0] if lines and ":fontsdir=" in lines[0] else None
    if not loaded or not selected or fontsdir_command is None:
        raise RuntimeError(f"Incomplete libass font evidence: {log_path}")
    return {
        "loaded_font_files": list(dict.fromkeys(loaded)),
        "selected_faces": list(dict.fromkeys(selected)),
        "controlled_fontsdir_in_command": True,
        "log_path": str(log_path.resolve()),
    }


def _case_intent(case: Case, source_sha256: str) -> CreativeIntent:
    preset = creative_preset_definition(case.creative_preset_id)
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id=f"map-{case.case_id}",
        source=SourceInterval.from_seconds(0, 8),
        output=OutputInterval(start_frame=0, end_frame=8 * FPS),
    ),))
    identity = ImmutableProductionIdentity(
        project_id="preset-font-real-mp4-de4d605",
        run_id=f"accept-{case.case_id}",
        analysis_id="cached-goal7i-food",
        candidate_id=case.case_id,
        source_id=f"source-{source_sha256[:24]}",
    )
    production = ImmutableProductionPlanLink(
        plan_id=f"plan-{case.case_id}",
        plan_fingerprint=sha256(f"{case.case_id}:{source_sha256}".encode()).hexdigest(),
        identity=identity,
    )
    evidence_hash = sha256(f"real-source:{source_sha256}".encode()).hexdigest()
    evidence = (
        EvidenceItem(
            evidence_ref=f"evidence-hook-{case.case_id}", evidence_kind="story_unit",
            source=SourceInterval.from_seconds(0, 2.1), confidence=0.99,
            artifact_fingerprint=evidence_hash,
            provenance="acceptance:cached-real-source-window",
        ),
        EvidenceItem(
            evidence_ref=f"evidence-emphasis-{case.case_id}", evidence_kind="transcript",
            source=SourceInterval.from_seconds(2.1, 4.8), confidence=0.99,
            artifact_fingerprint=evidence_hash,
            provenance=f"acceptance:{case.language}-caption-script",
        ),
    )
    return CreativeIntent(
        intent_id=f"intent-{case.case_id}", revision=1, production_plan=production,
        source_output_mapping=mapping, evidence_fingerprint=evidence_hash,
        evidence_manifest=evidence,
        proposal_hash=sha256(f"proposal:{case.case_id}".encode()).hexdigest(),
        policy=CreativePolicy(
            preset_id=preset.preset_id, preset_version=preset.preset_version,
            platform="universal", caption_style_family=preset.caption_style_family,
            caption_density=preset.caption_density, intensity=preset.intensity_ceiling,
            reduced_motion=False, source_broll_enabled=False,
        ),
        confidence=0.99,
        provenance=("acceptance:cached-real-source", "creative_policy:7K.1", "provider_calls:0"),
        beats=(ResolvedBeat(
            decision_id=f"beat-hook-{case.case_id}",
            source=SourceInterval.from_seconds(0, 2.1),
            output=OutputInterval(start_frame=0, end_frame=63), confidence=0.99,
            evidence_refs=(f"evidence-hook-{case.case_id}",),
            role=BeatRole.HOOK, importance=0.98,
        ),),
        semantic_emphasis=(ResolvedEmphasis(
            decision_id=f"emphasis-{case.case_id}",
            source=SourceInterval.from_seconds(2.1, 4.8),
            output=OutputInterval(start_frame=63, end_frame=144), confidence=0.99,
            evidence_refs=(f"evidence-emphasis-{case.case_id}",),
            text_span=" ".join(case.words[2:4]), semantic_class=SemanticClass.CLAIM,
            importance=0.98,
        ),),
    )


def _transcript(case: Case) -> dict[str, Any]:
    words = []
    for index, word in enumerate(case.words):
        start = 0.35 + index * 0.78
        words.append({
            "text": word, "start": start, "end": start + 0.58,
            "confidence": 0.99, "timing_source": "verified",
        })
    return {"language": case.language, "words": words}


def _compile(
    case: Case,
    source_sha256: str,
    *,
    source_width: int,
    source_height: int,
) -> CompiledRenderPlan:
    config = AppConfig()
    config.production_render.output_width = FINAL.width
    config.production_render.output_height = FINAL.height
    config.production_render.output_fps = FPS
    config.production_render.subtitle_font_family = case.requested_font_family
    config.production_render.subtitle_max_words_per_cue = 4
    config.production_render.subtitle_min_words_per_cue = 1
    config.production_render.same_source_broll_allowed = False
    return compile_native_creative_plan(
        _case_intent(case, source_sha256), _transcript(case), config,
        source_width=source_width, source_height=source_height,
    )


def _render(
    ffmpeg: Path, source: Path, case_root: Path, plan: CompiledRenderPlan,
    profile: RenderProfile,
) -> tuple[Path, Path, dict[str, Any]]:
    profile_root = case_root / profile.profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    ass_path = write_caption_plan_ass(
        plan.caption_plan, profile_root / "captions.ass", profile.width, profile.height,
    )
    fontsdir = materialize_caption_font_directory(
        plan.caption_plan.font_manifest, profile_root / "fonts",
    )
    output = profile_root / ("creative-preview.mp4" if profile.profile_id == "creative_preview" else "final-short.mp4")
    scale = (
        f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={profile.width}:{profile.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"{_ass_filter(ass_path, fontsdir)},format=yuv420p"
    )
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-i", str(source), "-t", "8",
        "-vf", scale, "-r", str(FPS), "-c:v", "libx264", "-preset", "medium",
        "-b:v", profile.video_bitrate, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
        str(output),
    ], log_path=profile_root / "ffmpeg.log")
    checksum = stable_file_hash(output)
    manifest = build_render_parity_manifest(plan, profile, output_checksum=checksum)
    manifest_path = profile_root / "parity-manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    return output, manifest_path, manifest.model_dump(mode="json")


def _ssim(ffmpeg: Path, preview: Path, final: Path, log_path: Path) -> float:
    result = _run([
        str(ffmpeg), "-hide_banner", "-i", str(preview), "-i", str(final),
        "-filter_complex", "[1:v]scale=540:960:flags=lanczos[final];[0:v][final]ssim",
        "-an", "-f", "null", "-",
    ], log_path=log_path)
    marker = "All:"
    for line in reversed(result.stderr.splitlines()):
        if marker in line:
            return float(line.split(marker, 1)[1].split()[0])
    raise RuntimeError("FFmpeg did not report SSIM")


def _contact_sheet(ffmpeg: Path, video: Path, output: Path) -> None:
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-i", str(video),
        "-vf", "fps=1/2,scale=270:480:flags=lanczos,tile=4x1:padding=8:margin=8",
        "-frames:v", "1", str(output),
    ], log_path=output.with_suffix(".log"))


def _paired_sheet(ffmpeg: Path, preview: Path, final: Path, output: Path) -> None:
    _run([
        str(ffmpeg), "-hide_banner", "-y", "-ss", "3.2", "-i", str(preview),
        "-ss", "3.2", "-i", str(final), "-filter_complex",
        "[0:v]scale=270:480:flags=lanczos[p];[1:v]scale=270:480:flags=lanczos[f];"
        "[p][f]hstack=inputs=2", "-frames:v", "1", str(output),
    ], log_path=output.with_suffix(".log"))


def run(source: Path, output: Path, ffmpeg: Path, ffprobe: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    source_sha256 = stable_file_hash(source)
    source_probe = _probe(ffprobe, source)
    video_stream = next(
        stream for stream in source_probe["streams"] if stream.get("codec_type") == "video"
    )
    source_width = int(video_stream["width"])
    source_height = int(video_stream["height"])
    results: list[dict[str, Any]] = []
    paired: list[Path] = []
    for case in CASES:
        case_root = output / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        plan = _compile(
            case, source_sha256,
            source_width=source_width, source_height=source_height,
        )
        _write_json(case_root / "compiled-render-plan.json", plan.model_dump(mode="json"))
        preview_path, preview_manifest_path, preview_manifest = _render(
            ffmpeg, source, case_root, plan, PREVIEW,
        )
        final_path, final_manifest_path, final_manifest = _render(
            ffmpeg, source, case_root, plan, FINAL,
        )
        assert_preview_final_parity(
            RenderParityManifest.model_validate(preview_manifest),
            RenderParityManifest.model_validate(final_manifest),
        )
        _decode_check(ffmpeg, preview_path)
        _decode_check(ffmpeg, final_path)
        preview_audio_sha256 = _decoded_audio_sha256(ffmpeg, preview_path)
        final_audio_sha256 = _decoded_audio_sha256(ffmpeg, final_path)
        if preview_audio_sha256 != final_audio_sha256:
            raise RuntimeError(f"Preview/Final decoded audio mismatch: {case.case_id}")
        preview_sheet = case_root / "creative-preview-contact-sheet.png"
        final_sheet = case_root / "final-contact-sheet.png"
        paired_sheet = case_root / "preview-final-paired-frame.png"
        _contact_sheet(ffmpeg, preview_path, preview_sheet)
        _contact_sheet(ffmpeg, final_path, final_sheet)
        _paired_sheet(ffmpeg, preview_path, final_path, paired_sheet)
        paired.append(paired_sheet)
        ssim = _ssim(ffmpeg, preview_path, final_path, case_root / "preview-final-ssim.log")
        manifest = plan.caption_plan.font_manifest
        assert manifest is not None and manifest.file_sha256 is not None
        staged_preview_files = tuple((case_root / PREVIEW.profile_id / "fonts").iterdir())
        staged_final_files = tuple((case_root / FINAL.profile_id / "fonts").iterdir())
        if len(staged_preview_files) != 1 or len(staged_final_files) != 1:
            raise RuntimeError(f"Controlled fontsdir must contain one exact face: {case.case_id}")
        staged_preview = staged_preview_files[0]
        staged_final = staged_final_files[0]
        results.append({
            **asdict(case),
            "caption_preset_token": plan.caption_plan.typography.token_id,
            "plan_hash": plan.plan_hash,
            "parity_signature": plan.parity_signature,
            "caption_quality_status": plan.caption_plan.quality_report.status,
            "caption_diagnostics": list(plan.caption_plan.diagnostics),
            "font": manifest.model_dump(mode="json"),
            "controlled_fontsdir": {
                "preview": str(staged_preview.resolve()),
                "preview_sha256": stable_file_hash(staged_preview),
                "final": str(staged_final.resolve()),
                "final_sha256": stable_file_hash(staged_final),
            },
            "preview": {
                "path": str(preview_path.resolve()), "sha256": stable_file_hash(preview_path),
                "probe": _probe(ffprobe, preview_path), "parity_manifest": str(preview_manifest_path.resolve()),
                "decode_check": "passed", "decoded_audio_sha256": preview_audio_sha256,
                "libass": _libass_evidence(case_root / PREVIEW.profile_id / "ffmpeg.log"),
            },
            "final": {
                "path": str(final_path.resolve()), "sha256": stable_file_hash(final_path),
                "probe": _probe(ffprobe, final_path), "parity_manifest": str(final_manifest_path.resolve()),
                "decode_check": "passed", "decoded_audio_sha256": final_audio_sha256,
                "libass": _libass_evidence(case_root / FINAL.profile_id / "ffmpeg.log"),
            },
            "preview_final_parity": "matched",
            "preview_final_decoded_audio": "matched",
            "preview_final_downscaled_ssim": ssim,
            "visual_evidence": {
                "preview_contact_sheet": str(preview_sheet.resolve()),
                "final_contact_sheet": str(final_sheet.resolve()),
                "paired_frame": str(paired_sheet.resolve()),
            },
        })
    overview = output / "comparison-contact-sheet.png"
    _run([
        str(ffmpeg), "-hide_banner", "-y",
        *sum((["-i", str(path)] for path in paired), []),
        "-filter_complex", "".join(f"[{index}:v]scale=432:384[p{index}];" for index in range(len(paired)))
        + "".join(f"[p{index}]" for index in range(len(paired)))
        + f"vstack=inputs={len(paired)}", "-frames:v", "1", str(overview),
    ], log_path=overview.with_suffix(".log"))
    report = {
        "schema_version": "preset-font-real-mp4-acceptance.1",
        "foundation_commit": "de4d605c84b9152b60389afaa88a8cbacd6e342a",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
        "brain_rerun": False,
        "vision_rerun": False,
        "source": {"path": str(source), "sha256": source_sha256, "probe": source_probe},
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": stable_file_hash(Path(__file__).resolve()),
        },
        "ffmpeg": {"path": str(ffmpeg), "sha256": stable_file_hash(ffmpeg)},
        "ffprobe": {"path": str(ffprobe), "sha256": stable_file_hash(ffprobe)},
        "comparison_contact_sheet": str(overview.resolve()),
        "cases": results,
    }
    _write_json(output / "acceptance-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    report = run(
        args.source, args.output.resolve(), _tool(args.ffmpeg, "ffmpeg"), _tool(args.ffprobe, "ffprobe"),
    )
    print(json.dumps({
        "report": str((args.output.resolve() / "acceptance-report.json")),
        "cases": len(report["cases"]), "provider_calls": report["provider_calls"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
