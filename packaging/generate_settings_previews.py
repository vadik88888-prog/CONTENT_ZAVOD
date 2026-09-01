"""Build canonical Friend Beta Settings preview MP4s.

This is a packaging-time tool.  It uses the production native creative
compiler, caption planner, versioned caption presets and bundled font
materializer.  The desktop app only plays the resulting immutable assets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from app.caption_planning import materialize_caption_font_directory, write_caption_plan_ass
from app.caption_presets import CAPTION_PRESET_DEFINITIONS, with_caption_preset_override
from app.config import AppConfig
from app.creative_contracts import (
    BeatRole,
    CreativeIntent,
    CreativePolicy,
    EditMapSegment,
    EvidenceItem,
    ImmutableProductionIdentity,
    ImmutableProductionPlanLink,
    OutputInterval,
    ResolvedBeat,
    ResolvedEmphasis,
    SemanticClass,
    SourceInterval,
    SourceOutputTimeMap,
)
from app.creative_execution import compile_native_creative_plan
from app.creative_policy import CREATIVE_PRESET_DEFINITIONS, creative_preset_definition
from app.settings_preview_assets import SETTINGS_PREVIEW_SCHEMA_VERSION
from app.utils import stable_file_hash
from app.video_composition import _ass_filter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "settings-previews"
FPS = 30
DURATION_SECONDS = 10
WIDTH = 1080
HEIGHT = 1920
SOURCE_VIDEO = OUTPUT / "source" / "talking-head-v2.mp4"
SOURCE_LICENSE = {
    "provider": "Mixkit",
    "asset_url": "https://mixkit.co/free-stock-video/portrait-of-an-influencer-talking-to-the-camera-42323/",
    "license_url": "https://mixkit.co/license/#videoFree",
    "license": "Mixkit Stock Video Free License",
    "source_asset_id": "42323",
    "source_title": "Portrait of an influencer talking to the camera",
}
WORDS = (
    "СИЛЬНАЯ", "МЫСЛЬ", "СТАНОВИТСЯ", "ЯСНОЙ", "КОГДА",
    "ЕЁ", "МОЖНО", "СРАЗУ", "ПОКАЗАТЬ", "В КАДРЕ",
)


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _intent(style_id: str, caption_id: str) -> CreativeIntent:
    style = creative_preset_definition(style_id)  # type: ignore[arg-type]
    caption = CAPTION_PRESET_DEFINITIONS[caption_id]  # type: ignore[index]
    mapping = SourceOutputTimeMap(segments=(EditMapSegment(
        map_id=f"settings-map-{style_id}-{caption_id}",
        source=SourceInterval.from_seconds(0, DURATION_SECONDS),
        output=OutputInterval(start_frame=0, end_frame=DURATION_SECONDS * FPS),
    ),))
    identity = ImmutableProductionIdentity(
        project_id="friend-beta-settings-demo",
        run_id=f"settings-{style_id}-{caption_id}",
        analysis_id="canonical-settings-demo-v1",
        candidate_id=f"settings-{style_id}-{caption_id}",
        source_id="canonical-settings-demo-source",
    )
    production = ImmutableProductionPlanLink(
        plan_id=f"settings-plan-{style_id}-{caption_id}",
        plan_fingerprint=sha256(f"settings:{style_id}:{caption_id}".encode()).hexdigest(),
        identity=identity,
    )
    evidence_hash = sha256(b"friend-beta-settings-demo-evidence-v1").hexdigest()
    evidence = (
        EvidenceItem(
            evidence_ref="settings-hook", evidence_kind="story_unit",
            source=SourceInterval.from_seconds(0.2, 2.0), confidence=0.99,
            artifact_fingerprint=evidence_hash, provenance="packaging:settings-demo",
        ),
        EvidenceItem(
            evidence_ref="settings-emphasis", evidence_kind="transcript",
            source=SourceInterval.from_seconds(2.0, 4.6), confidence=0.99,
            artifact_fingerprint=evidence_hash, provenance="packaging:settings-demo",
        ),
    )
    return CreativeIntent(
        intent_id=f"settings-intent-{style_id}-{caption_id}",
        revision=1,
        production_plan=production,
        source_output_mapping=mapping,
        evidence_fingerprint=evidence_hash,
        evidence_manifest=evidence,
        proposal_hash=sha256(f"proposal:{style_id}:{caption_id}".encode()).hexdigest(),
        policy=CreativePolicy(
            preset_id=style.preset_id,
            preset_version=style.preset_version,
            platform="universal",
            caption_style_family=caption.style_family,
            caption_density={
                "minimal": "low", "clean": "balanced",
                "editorial": "balanced", "emphasis": "high",
            }[caption.style_family],
            intensity=style.intensity_ceiling,
            reduced_motion=False,
            source_broll_enabled=False,
            user_override_ids=with_caption_preset_override((), caption.preset_id),
        ),
        confidence=0.99,
        provenance=(
            "packaging:canonical-settings-demo",
            f"creative_style:{style.preset_id}:{style.preset_version}",
            f"caption_preset:{caption.preset_id}:{caption.preset_version}",
            "provider_calls:0",
        ),
        beats=(ResolvedBeat(
            decision_id="settings-beat-hook",
            source=SourceInterval.from_seconds(0.2, 2.0),
            output=OutputInterval(start_frame=6, end_frame=60),
            confidence=0.99, evidence_refs=("settings-hook",),
            role=BeatRole.HOOK, importance=0.98,
        ),),
        semantic_emphasis=(ResolvedEmphasis(
            decision_id="settings-emphasis-claim",
            source=SourceInterval.from_seconds(2.0, 4.6),
            output=OutputInterval(start_frame=60, end_frame=138),
            confidence=0.99, evidence_refs=("settings-emphasis",),
            text_span="МЫСЛЬ СТАНОВИТСЯ ЯСНОЙ", semantic_class=SemanticClass.CLAIM,
            importance=0.98,
        ),),
    )


def _transcript() -> dict[str, Any]:
    words = []
    for index, word in enumerate(WORDS):
        start = 0.35 + index * 0.88
        words.append({
            "text": word, "start": start, "end": start + 0.62,
            "confidence": 0.99, "timing_source": "verified",
        })
    return {"language": "ru", "words": words}


def build() -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    if not SOURCE_VIDEO.is_file():
        raise FileNotFoundError(f"Missing canonical Settings demo source video: {SOURCE_VIDEO}")
    with tempfile.TemporaryDirectory(prefix="friend-beta-settings-preview-") as raw_temp:
        temp = Path(raw_temp)
        source_sha256 = stable_file_hash(SOURCE_VIDEO)
        for style_id in CREATIVE_PRESET_DEFINITIONS:
            style = creative_preset_definition(style_id)
            for caption_id, caption in CAPTION_PRESET_DEFINITIONS.items():
                config = AppConfig()
                config.product_flow.subtitle_preset = style_id
                config.product_flow.preset_version = style.preset_version
                config.product_flow.caption_preset_id = caption_id
                config.product_flow.caption_preset_version = caption.preset_version
                config.production_render.output_width = WIDTH
                config.production_render.output_height = HEIGHT
                config.production_render.output_fps = FPS
                config.production_render.same_source_broll_allowed = False
                compiled = compile_native_creative_plan(
                    _intent(style_id, caption_id), _transcript(), config,
                    source_width=WIDTH, source_height=HEIGHT,
                )
                case_temp = temp / style_id / caption_id
                ass = write_caption_plan_ass(
                    compiled.caption_plan, case_temp / "captions.ass", WIDTH, HEIGHT,
                )
                fonts = materialize_caption_font_directory(
                    compiled.caption_plan.font_manifest, case_temp / "fonts",
                )
                destination = OUTPUT / style_id / f"{caption_id}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                _run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(SOURCE_VIDEO), "-vf", _ass_filter(ass, fonts),
                    "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-maxrate", "3200k", "-bufsize", "6400k",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    str(destination),
                ])
                font_manifest = compiled.caption_plan.font_manifest
                items.append({
                    "creative_style_id": style_id,
                    "creative_style_version": style.preset_version,
                    "caption_preset_id": caption_id,
                    "caption_preset_version": caption.preset_version,
                    "caption_token_id": compiled.caption_plan.typography.token_id,
                    "font_asset_ids": list(caption.font_asset_ids),
                    "font_manifest": (
                        font_manifest.model_dump(mode="json") if font_manifest else None
                    ),
                    "compiled_plan_hash": compiled.plan_hash,
                    "parity_signature": compiled.parity_signature,
                    "path": destination.relative_to(OUTPUT).as_posix(),
                    "sha256": stable_file_hash(destination),
                    "size_bytes": destination.stat().st_size,
                    "demo_source_sha256": source_sha256,
                    "duration_seconds": DURATION_SECONDS,
                })
    report = {
        "schema_version": SETTINGS_PREVIEW_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "packaging/generate_settings_previews.py",
        "provider_calls": 0,
        "brain_rerun": False,
        "vision_rerun": False,
        "render_owner": "compile_native_creative_plan + write_caption_plan_ass + bundled fonts + libass",
        "demo_source": {
            "path": SOURCE_VIDEO.relative_to(OUTPUT).as_posix(),
            "sha256": source_sha256,
            "content": "vertical talking head with visible speech and natural hand motion",
            "license": SOURCE_LICENSE,
            "duration_seconds": DURATION_SECONDS,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
        },
        "items": items,
    }
    _write_json(OUTPUT / "manifest.json", report)
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps({"items": len(result["items"]), "manifest": str(OUTPUT / "manifest.json")}, ensure_ascii=False))
