"""Run the bounded Friend Beta Desktop acceptance path on persisted real media.

The runner deliberately works in a new Desktop data directory while keeping
the original immutable source and AnalysisArtifact references.  It drives the
real Qt actions for Moments -> Draft -> Creative Preview -> Approve -> Final,
then a separate ``restart`` invocation verifies process-restart persistence.
No Analysis stage is launched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from app.caption_presets import caption_preset_definition
from app.creative_execution import CAPTION_RENDER_BACKEND_VERSION
from app.clip_results import ClipResult
from app.creative_contracts import RenderParityManifest, assert_preview_final_parity
from app.draft_artifact import DraftArtifact
from app.gui.main_window import MainWindow
from app.gui.models import DesktopProject, DesktopSettings, ProjectStatus, RunKind, RunStatus
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.settings_store import SettingsStore, default_data_directory
from app.gui.styles import load_theme
from app.quality_report import aggregate_quality_status, read_quality_report
from app.run_manifest import is_run_scoped_path
from app.runtime import RuntimeLayout
from app.utils import read_json, write_json


DEFAULT_PROJECT_ID = "aa71f33b6e564c09a1edaf3f63312e03"
DEFAULT_CANDIDATE_ID = "candidate-chapter-009-story-001"
TERMINAL_STATUSES = RunStatus.ALL - RunStatus.ACTIVE
PARITY_FIELDS = (
    "plan_hash",
    "parity_signature",
    "fps",
    "event_frames_hash",
    "resolved_lines_hash",
    "normalized_geometry_hash",
    "font_asset_hash",
    "motion_math_hash",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _settle(app: QApplication, turns: int = 12) -> None:
    for _ in range(turns):
        app.processEvents()


def _wait_until(
    app: QApplication,
    predicate: Callable[[], Any],
    *,
    timeout_seconds: float,
    description: str,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        app.processEvents()
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {description}.")


def _capture(window: MainWindow, directory: Path, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.png"
    _settle(QApplication.instance())  # type: ignore[arg-type]
    if not window.grab().save(str(target)):
        raise RuntimeError(f"Could not save native screenshot to {target}.")
    return str(target.resolve())


def _runtime(engine_root: Path, data_directory: Path) -> RuntimeLayout:
    tools = engine_root / "packaging" / "windows" / "tools"
    ffmpeg = tools / "ffmpeg.exe"
    ffprobe = tools / "ffprobe.exe"
    if not ffmpeg.is_file() or not ffprobe.is_file():
        raise FileNotFoundError("Source-runtime FFmpeg tools are unavailable.")
    runtime = RuntimeLayout.for_source(
        engine_root,
        data=data_directory,
        program=Path(sys.executable),
        tools=tools,
    )
    runtime.activate()
    return runtime


def _copy_optional_cache(source_project_directory: Path, target_project_directory: Path) -> None:
    # These are review caches, not Analysis or renderer state.  Reusing them
    # keeps the AV1 source-player preload from competing with the acceptance
    # render while still exercising the normal media-preview path.
    for name in ("preview-proxies", "candidate-thumbnails", "thumbnails"):
        source = source_project_directory / name
        target = target_project_directory / name
        if source.is_dir():
            shutil.copytree(source, target)


def _prepare_isolated_data(
    source_data: Path,
    data_directory: Path,
    project_id: str,
    candidate_id: str,
) -> DesktopProject:
    if data_directory.exists():
        raise FileExistsError(
            f"Acceptance data directory already exists: {data_directory}. "
            "Choose a new directory; the runner never deletes persisted evidence."
        )
    data_directory.mkdir(parents=True)
    source_store = DesktopProjectStore(source_data)
    source_project = source_store.load(project_id)
    analysis_path = Path(source_project.analysis_artifact_path or "")
    if not analysis_path.is_file() or not source_project.analysis_id:
        raise RuntimeError("The positive acceptance project has no reusable AnalysisArtifact.")
    if candidate_id not in source_project.candidate_states:
        raise RuntimeError(f"Candidate {candidate_id!r} is absent from the persisted project.")

    target_project_directory = data_directory / "projects" / project_id
    target_project_directory.mkdir(parents=True)
    _copy_optional_cache(Path(source_project.project_directory), target_project_directory)

    project_payload = source_project.to_dict()
    project_payload.update(
        {
            "project_directory": str(target_project_directory.resolve()),
            "status": ProjectStatus.ANALYSIS_READY,
            "latest_run_id": None,
            "draft_artifact_path": None,
            "draft_id": None,
            "candidate_draft_artifacts": {},
            "candidate_errors": {},
            "review_selected_candidate_ids": [],
            "selected_candidate_ids": [],
            "candidate_boundary_overrides": {},
            "candidate_creative_overrides": {},
            "active_preview_candidate_id": None,
            "last_final_result_id": None,
        }
    )
    candidate_ids = list(project_payload.get("candidate_states", {}))
    project_payload["candidate_states"] = {value: "analyzed" for value in candidate_ids}
    project_payload["candidate_draft_statuses"] = {value: "pending" for value in candidate_ids}
    project_payload["candidate_approval_states"] = {value: "pending" for value in candidate_ids}
    project_payload["candidate_export_statuses"] = {value: "pending" for value in candidate_ids}
    setup_state = dict(project_payload.get("setup_state") or {})
    setup_state.update(
        {
            "needs_new_analysis": False,
            "change_summary": "Acceptance: сохранённый анализ переиспользуется без повторного запуска.",
            "reused_stages": ["сохранённый анализ", "Brain/Vision evidence"],
        }
    )
    project_payload["setup_state"] = setup_state
    project = DesktopProject.from_dict(project_payload)
    DesktopProjectStore(data_directory).save(project)

    source_settings = SettingsStore(source_data).load().to_dict()
    source_settings.update(
        {
            "data_directory": str(data_directory.resolve()),
            # Draft uses the existing deterministic local transformation in
            # acceptance so the run proves Desktop/artifact/renderer behavior
            # without a new network AI decision.  Final remains the real
            # production renderer and strict quality gate.
            "local_test_mode": True,
            "onboarding_completed": True,
            "window_geometry": None,
            "last_open_project_id": project_id,
            "last_screen": "project",
        }
    )
    SettingsStore(data_directory).save(DesktopSettings.from_dict(source_settings))
    return project


def _new_runs(services: DesktopServices, project_id: str, known_ids: set[str]) -> list[Any]:
    project = services.projects.load(project_id)
    return [run for run in services.runs_for(project) if run.run_id not in known_ids]


def _wait_for_run(
    app: QApplication,
    services: DesktopServices,
    project_id: str,
    known_ids: set[str],
    run_kind: str,
    timeout_seconds: float,
) -> Any:
    def terminal_run() -> Any:
        matches = [run for run in _new_runs(services, project_id, known_ids) if run.run_kind == run_kind]
        if not matches:
            return None
        run = max(matches, key=lambda value: (value.started_at, value.run_id))
        return run if run.status in TERMINAL_STATUSES else None

    return _wait_until(
        app,
        terminal_run,
        timeout_seconds=timeout_seconds,
        description=f"terminal {run_kind} run",
    )


def _draft_candidate_record(draft_path: Path, candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = read_json(draft_path, {})
    if not isinstance(draft, dict):
        raise AssertionError("DraftArtifact is not a JSON object.")
    record = next(
        (
            item
            for item in draft.get("candidates", [])
            if isinstance(item, dict) and str(item.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if not isinstance(record, dict):
        raise AssertionError("DraftArtifact lost the requested candidate identity.")
    return draft, record


def _ffprobe(ffprobe: Path, media: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(media),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise AssertionError("ffprobe returned an invalid payload.")
    return value


def _caption_identity(plan_path: Path, ass_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path, {})
    caption = plan.get("caption_plan", {}) if isinstance(plan, dict) else {}
    if not isinstance(caption, dict):
        caption = {}
    typography = caption.get("typography", {})
    font = caption.get("font_manifest", {})
    caption_backends = [
        item for item in plan.get("backends", [])
        if isinstance(item, dict) and item.get("domain") == "caption"
    ]
    cues = [cue for cue in caption.get("cues", []) if isinstance(cue, dict)]
    if not plan_path.is_file() or not ass_path.is_file() or not isinstance(typography, dict) or not isinstance(font, dict):
        raise AssertionError("Caption plan/ASS identity artifacts are missing.")
    token = str(typography.get("token_id") or "")
    cue_tokens = sorted({str(cue.get("typography_token_id") or "") for cue in cues})
    if not token or cue_tokens != [token]:
        raise AssertionError("Caption cues lost the compiled preset typography token.")
    if len(caption_backends) != 1 or caption_backends[0].get("backend_version") != CAPTION_RENDER_BACKEND_VERSION:
        raise AssertionError("Compiled caption backend is stale for the current ASS treatment.")
    ass = ass_path.read_text(encoding="utf-8-sig")
    dialogue_lines = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    style_line = next(
        (line for line in ass.splitlines() if line.startswith("Style: CaptionPlan,")),
        "",
    )
    style_parts = style_line.split(",", 2)
    ass_font_family = style_parts[1] if len(style_parts) >= 2 else ""
    resolved_family = str(font.get("resolved_family") or "")
    if not dialogue_lines or len(dialogue_lines) != len(cues):
        raise AssertionError("Rendered ASS dialogue count differs from the compiled CaptionPlan.")
    if not resolved_family or ass_font_family != resolved_family:
        raise AssertionError("Rendered ASS font family differs from the compiled font identity.")
    highlight = str(typography.get("highlight_color") or "").lstrip("#")
    if len(highlight) != 6:
        raise AssertionError("Caption highlight color is not a resolved RGB token.")
    ass_highlight = f"&H00{highlight[4:6]}{highlight[2:4]}{highlight[0:2]}&".upper()
    ass_transform_tag_count = sum(line.count("\\t(") for line in dialogue_lines)
    ass_word_pop_curve_count = sum(
        line.count("\\t(") == 2 for line in dialogue_lines
    )
    ass_highlight_count = sum(
        line.upper().count(f"\\1C{ass_highlight}") for line in dialogue_lines
    )
    treatment_counts: dict[str, int] = {}
    for cue in cues:
        key = "|".join(str(cue.get(name) or "") for name in (
            "display_mode", "timing_mode", "primitive_id", "fallback_reason",
        ))
        treatment_counts[key] = treatment_counts.get(key, 0) + 1
    expected_word_pop_curves = sum(
        cue.get("primitive_id") == "word_pop" for cue in cues
    )
    if ass_word_pop_curve_count != expected_word_pop_curves:
        raise AssertionError(
            "Physical ASS two-stage curves differ from Word Pop cue treatments."
        )
    weak_pairs = [
        (cue, line) for cue, line in zip(cues, dialogue_lines, strict=True)
        if token.startswith("caption-preset:word_pop:")
        and cue.get("fallback_reason") == "weak_timing"
    ]
    weak_static_highlight_count = sum(
        "\\t(" not in line
        and "\\move(" not in line
        and "\\fad(" not in line
        and f"\\1C{ass_highlight}" in line.upper()
        for _cue, line in weak_pairs
    )
    if weak_pairs and weak_static_highlight_count != len(weak_pairs):
        raise AssertionError("Weak-timing Word Pop ASS is not static and accent-preserving.")
    return {
        "compiled_plan_path": str(plan_path.resolve()),
        "compiled_plan_sha256": _sha256(plan_path),
        "ass_path": str(ass_path.resolve()),
        "ass_sha256": _sha256(ass_path),
        "token_id": token,
        "font_id": font.get("font_id"),
        "font_file": font.get("file_name"),
        "font_sha256": font.get("file_sha256"),
        "text_color": typography.get("text_color"),
        "highlight_color": typography.get("highlight_color"),
        "backend_version": caption_backends[0]["backend_version"],
        "cue_count": len(cues),
        "ass_dialogue_count": len(dialogue_lines),
        "ass_font_family": ass_font_family,
        "ass_transform_tag_count": ass_transform_tag_count,
        "ass_word_pop_curve_count": ass_word_pop_curve_count,
        "ass_weak_static_highlight_count": weak_static_highlight_count,
        "ass_highlight_tag_count": ass_highlight_count,
        "treatment_counts": treatment_counts,
        "diagnostics": list(caption.get("diagnostics", [])),
    }


def _validate_final(
    services: DesktopServices,
    runtime: RuntimeLayout,
    project_id: str,
    candidate_id: str,
    final_run: Any,
    draft_path: Path,
    draft_run_id: str,
    analysis_sha256: str,
) -> dict[str, Any]:
    execution = final_run.settings_snapshot.get("execution", {})
    engine_paths = execution.get("engine_paths", {}) if isinstance(execution, dict) else {}
    run_output = Path(str(engine_paths.get("output_directory") or ""))
    manifest_path = Path(str(engine_paths.get("manifest_path") or ""))
    manifest = read_json(manifest_path, {})
    if not isinstance(manifest, dict):
        raise AssertionError("Final manifest is missing or invalid.")
    if not is_run_scoped_path(manifest_path, run_output):
        raise AssertionError("Final manifest escaped its persisted run directory.")
    if manifest.get("project_id") != project_id or manifest.get("run_id") != final_run.run_id:
        raise AssertionError("Final manifest identity does not match its persisted run.")
    results = [item for item in manifest.get("primary_results", []) if isinstance(item, dict)]
    if len(results) != 1 or str(results[0].get("candidate_id") or "") != candidate_id:
        raise AssertionError("Final manifest contains a sibling or lost the requested candidate.")
    result = results[0]
    canonical = ClipResult.from_dict(result)
    if canonical is None:
        raise AssertionError("Final manifest primary result is not a valid ClipResult.")
    if (
        canonical.run_id != final_run.run_id
        or not canonical.clip_result_id
        or not canonical.production_plan_id
        or not canonical.revision_id
        or not canonical.artifact_id
    ):
        raise AssertionError("Final ClipResult is missing exact current-run identity.")
    if manifest.get("candidate_ids") != [candidate_id]:
        raise AssertionError("Final manifest candidate allow-list differs from the clicked candidate.")
    media = Path(canonical.output_file)
    if not is_run_scoped_path(media, run_output):
        raise AssertionError("Final MP4 escaped its persisted run directory.")
    if not media.is_file() or media.stat().st_size <= 0:
        raise AssertionError("Validated Final MP4 is absent or empty.")
    media_sha256 = _sha256(media)
    if media_sha256 != canonical.artifact_checksum:
        raise AssertionError("Final MP4 checksum differs from the canonical manifest.")

    quality_path = Path(canonical.quality_report_path)
    if not is_run_scoped_path(quality_path, run_output):
        raise AssertionError("QualityReport escaped its persisted run directory.")
    quality = read_quality_report(read_json(quality_path, {}))
    if quality is None:
        raise AssertionError("QualityReport is missing or invalid.")
    if (
        quality.get("project_id") != project_id
        or quality.get("run_id") != final_run.run_id
        or quality.get("candidate_id") != candidate_id
        or quality.get("report_id") != canonical.quality_report_id
        or quality.get("artifact_id") != canonical.artifact_id
        or Path(str(quality.get("artifact_path") or "")).resolve() != media.resolve()
        or quality.get("artifact_sha256") != media_sha256
        or quality.get("edit_plan_id") != canonical.production_plan_id
        or quality.get("render_id") != canonical.revision_id
        or quality.get("source_id") != manifest.get("source_id")
        or canonical.quality_status != quality.get("status")
    ):
        raise AssertionError("QualityReport lineage does not match the Final MP4.")
    if quality.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise AssertionError(f"Positive Final did not pass the strict quality gate: {quality.get('status')!r}")
    gate = manifest.get("quality_gate")
    if not isinstance(gate, dict) or aggregate_quality_status([quality]) != gate.get("status"):
        raise AssertionError("Canonical manifest and QualityReport disagree on quality status.")

    draft, draft_record = _draft_candidate_record(draft_path, candidate_id)
    draft_contract = DraftArtifact.read(draft_path)
    persisted_project = services.projects.load(project_id)
    if (
        draft_contract.run_id != draft_run_id
        or draft_contract.project_id != project_id
        or draft_contract.analysis_id != persisted_project.analysis_id
        or draft_contract.analysis_fingerprint != persisted_project.analysis_fingerprint
        or Path(draft_contract.analysis_artifact_path).resolve()
        != Path(persisted_project.analysis_artifact_path or "").resolve()
        or draft_contract.analysis_artifact_sha256 != analysis_sha256
        or [str(item.get("candidate_id") or "") for item in draft_contract.candidates] != [candidate_id]
    ):
        raise AssertionError("DraftArtifact exact run/Analysis/candidate lineage is invalid.")
    if draft.get("analysis_artifact_sha256") != analysis_sha256:
        raise AssertionError("DraftArtifact did not bind to the unchanged AnalysisArtifact.")
    preview = draft_record.get("preview", {})
    preview_path = Path(str(preview.get("output_file") or "")) if isinstance(preview, dict) else Path()
    if not preview_path.is_file():
        raise AssertionError("Creative Preview is missing.")

    draft_parity_path = Path(str(preview.get("parity_manifest_ref") or ""))
    copied_preview_path = run_output / "creative-preview" / "creative-preview.mp4"
    copied_parity_path = copied_preview_path.with_name("parity-manifest.json")
    final_parity_path = run_output / "production-render" / "parity-manifest.json"
    if not all(path.is_file() for path in (draft_parity_path, copied_preview_path, copied_parity_path, final_parity_path)):
        raise AssertionError("Preview/Final parity artifacts are missing.")
    if not all(is_run_scoped_path(path, run_output) for path in (
        copied_preview_path, copied_parity_path, final_parity_path,
    )):
        raise AssertionError("Copied Preview/Final parity artifacts escaped the Final run.")
    draft_parity = RenderParityManifest.model_validate(read_json(draft_parity_path, {}))
    copied_preview_parity = RenderParityManifest.model_validate(read_json(copied_parity_path, {}))
    final_parity = RenderParityManifest.model_validate(read_json(final_parity_path, {}))
    assert_preview_final_parity(draft_parity, final_parity)
    assert_preview_final_parity(copied_preview_parity, final_parity)
    preview_sha256 = _sha256(preview_path)
    copied_preview_sha256 = _sha256(copied_preview_path)
    if (
        draft_parity.output_checksum != preview_sha256
        or copied_preview_parity.output_checksum != copied_preview_sha256
        or final_parity.output_checksum != media_sha256
        or copied_preview_sha256 != preview_sha256
    ):
        raise AssertionError("Preview/Final parity manifests are not bound to their exact media bytes.")

    expected_preset = caption_preset_definition(
        persisted_project.settings.caption_preset_id  # type: ignore[arg-type]
    )
    preview_caption = _caption_identity(
        preview_path.with_name("compiled-render-plan.json"),
        preview_path.with_name("production-subtitles.ass"),
    )
    final_caption = _caption_identity(
        run_output / "production-render" / "compiled-render-plan.json",
        run_output / "production-render" / "production-subtitles.ass",
    )
    identity_fields = (
        "compiled_plan_sha256", "ass_sha256", "token_id", "font_id", "font_file",
        "font_sha256", "text_color", "highlight_color", "cue_count",
        "backend_version", "ass_dialogue_count", "ass_font_family",
        "ass_transform_tag_count", "ass_word_pop_curve_count",
        "ass_weak_static_highlight_count", "ass_highlight_tag_count",
        "treatment_counts", "diagnostics",
    )
    expected_identity = {
        "token_id": expected_preset.token_id,
        "font_id": expected_preset.preferred_font_asset_id,
        "text_color": expected_preset.text_color,
        "highlight_color": expected_preset.highlight_color,
    }
    if any(preview_caption[field] != value for field, value in expected_identity.items()):
        raise AssertionError(
            "Creative Preview caption preset differs from the persisted Settings choice."
        )
    if expected_preset.preset_id == "word_pop" and not any(
        key.startswith("single_spoken_word|word|word_pop|")
        and count > 0
        for key, count in preview_caption["treatment_counts"].items()
    ):
        raise AssertionError("Word Pop lost its spoken-word production treatment.")
    if expected_preset.preset_id == "word_pop" and (
        preview_caption["ass_word_pop_curve_count"] <= 0
        or preview_caption["ass_highlight_tag_count"] <= 0
    ):
        raise AssertionError("Physical ASS output lost Word Pop motion or accent treatment.")
    if any(preview_caption[field] != final_caption[field] for field in identity_fields):
        raise AssertionError("Creative Preview and Final caption identities differ.")

    probe = _ffprobe(runtime.tools / "ffprobe.exe", media)
    preview_probe = _ffprobe(runtime.tools / "ffprobe.exe", preview_path)
    streams = [item for item in probe.get("streams", []) if isinstance(item, dict)]
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise AssertionError("Final MP4 does not contain both video and audio streams.")
    if (
        video.get("codec_name") != "h264"
        or video.get("width") != 1080
        or video.get("height") != 1920
        or video.get("pix_fmt") != "yuv420p"
        or video.get("r_frame_rate") != "30/1"
        or audio.get("codec_name") != "aac"
    ):
        raise AssertionError(f"Final MP4 media contract mismatch: {probe!r}")
    preview_streams = [item for item in preview_probe.get("streams", []) if isinstance(item, dict)]
    preview_video = next((item for item in preview_streams if item.get("codec_type") == "video"), None)
    preview_audio = next((item for item in preview_streams if item.get("codec_type") == "audio"), None)
    if (
        not isinstance(preview_video, dict)
        or not isinstance(preview_audio, dict)
        or preview_video.get("codec_name") != "h264"
        or preview_video.get("width") != 540
        or preview_video.get("height") != 960
        or preview_video.get("pix_fmt") != "yuv420p"
        or preview_video.get("r_frame_rate") != "30/1"
        or preview_audio.get("codec_name") != "aac"
    ):
        raise AssertionError(f"Creative Preview media contract mismatch: {preview_probe!r}")

    return {
        "run_id": final_run.run_id,
        "status": final_run.status,
        "manifest_path": str(manifest_path.resolve()),
        "clip_result_id": result.get("clip_result_id"),
        "candidate_id": result.get("candidate_id"),
        "production_plan_id": result.get("production_plan_id"),
        "artifact_id": result.get("artifact_id"),
        "artifact_checksum": result.get("artifact_checksum"),
        "output_path": str(media.resolve()),
        "output_bytes": media.stat().st_size,
        "quality_report_path": str(quality_path.resolve()),
        "quality_status": quality.get("status"),
        "ffprobe": probe,
        "parity": {field: getattr(final_parity, field) for field in PARITY_FIELDS},
        "creative_preview_path": str(preview_path.resolve()),
        "creative_preview_sha256": preview_sha256,
        "creative_preview_ffprobe": preview_probe,
        "caption_preset": {
            "preset_id": expected_preset.preset_id,
            "preset_version": expected_preset.preset_version,
            "preview": preview_caption,
            "final": final_caption,
        },
    }


def _shutdown(window: MainWindow, app: QApplication) -> None:
    window.project_screen.preview.suspend()
    window.project_screen.setup_demo_preview.suspend()
    window.project_screen.final_results.preview.suspend()
    window.close()
    window.deleteLater()
    _settle(app)


def _run_acceptance(args: argparse.Namespace) -> int:
    engine_root = Path(__file__).resolve().parents[1]
    source_data = args.source_data.expanduser().resolve()
    work_directory = args.work_directory.expanduser().resolve()
    data_directory = work_directory / "desktop-data"
    screenshots = work_directory / "screenshots"
    work_directory.mkdir(parents=True, exist_ok=False)
    # _prepare_isolated_data owns the data-directory create guard; the parent
    # work directory is already the immutable root for all evidence.
    project = _prepare_isolated_data(source_data, data_directory, args.project_id, args.candidate_id)
    analysis_path = Path(project.analysis_artifact_path or "")
    analysis_sha256_before = _sha256(analysis_path)
    runtime = _runtime(engine_root, data_directory)
    services = DesktopServices.create(runtime)

    app = QApplication.instance() or QApplication([])
    if app.platformName().casefold() != "windows":
        raise AssertionError(f"Real Desktop acceptance requires the Windows Qt platform, got {app.platformName()!r}.")
    app.setStyleSheet(load_theme())
    window = MainWindow(services)
    window.resize(args.width, args.height)
    window.show()
    window.show_project(services.projects.load(args.project_id))
    _settle(app, 30)
    if not window.isVisible():
        raise AssertionError("Source-runtime Desktop window is not visible.")
    screen = window.project_screen
    if screen._flow_step != "candidates":
        raise AssertionError(f"Expected Moments, got {screen._flow_step!r}.")
    evidence: dict[str, Any] = {
        "scenario": "Fresh real Desktop Moments -> Draft -> Creative Preview -> Approve -> Final",
        "source_runtime": str(Path(sys.executable).resolve()),
        "draft_ai_mode": "local deterministic (--mock-ai --no-ai-transformation)",
        "final_render_mode": "real production renderer and strict quality gate",
        "source_data_read_only": str(source_data),
        "isolated_desktop_data": str(data_directory),
        "project_id": args.project_id,
        "candidate_id": args.candidate_id,
        "analysis": {
            "analysis_id": project.analysis_id,
            "analysis_fingerprint": project.analysis_fingerprint,
            "artifact_path": str(analysis_path.resolve()),
            "sha256_before": analysis_sha256_before,
        },
        "screenshots": {},
    }
    evidence["screenshots"]["moments"] = _capture(window, screenshots, "01-moments")

    select = screen._candidate_selection_buttons.get(args.candidate_id)
    if select is None:
        if not screen.view_all_button.isVisible() or not screen.view_all_button.isEnabled():
            raise AssertionError("The real View All Moments CTA is unavailable for the target candidate.")
        QTest.mouseClick(screen.view_all_button, Qt.MouseButton.LeftButton)
        _settle(app, 20)
        select = screen._candidate_selection_buttons.get(args.candidate_id)
    if select is None or not select.isEnabled():
        raise AssertionError("The positive AVAILABLE candidate has no selectable Moments action.")
    QTest.mouseClick(select, Qt.MouseButton.LeftButton)
    _settle(app, 20)
    selected = services.projects.load(args.project_id)
    if selected.review_selected_candidate_ids != [args.candidate_id]:
        raise AssertionError(f"Moments action routed {selected.review_selected_candidate_ids!r}.")
    if not screen.draft_button.isVisible() or not screen.draft_button.isEnabled():
        raise AssertionError("Selected Moment did not expose the Draft CTA.")

    known_runs = {run.run_id for run in services.runs_for(selected)}
    QTest.mouseClick(screen.draft_button, Qt.MouseButton.LeftButton)
    _wait_until(
        app,
        lambda: screen._flow_step == "processing",
        timeout_seconds=20.0,
        description="Draft processing screen",
    )
    evidence["screenshots"]["draft_processing"] = _capture(window, screenshots, "02-draft-processing")
    draft_run = _wait_for_run(
        app,
        services,
        args.project_id,
        known_runs,
        RunKind.DRAFT,
        args.timeout_seconds,
    )
    if draft_run.status != RunStatus.DRAFT_READY:
        raise AssertionError(f"Positive Draft ended as {draft_run.status}: {draft_run.error_summary}")
    draft_delta = _new_runs(services, args.project_id, known_runs)
    if len(draft_delta) != 1 or draft_delta[0].run_id != draft_run.run_id:
        raise AssertionError("One Draft click created duplicate or unexpected runs.")
    draft_execution = draft_run.settings_snapshot.get("execution", {})
    if (
        list(draft_run.settings_snapshot.get("candidate_ids") or []) != [args.candidate_id]
        or not isinstance(draft_execution, dict)
        or list(draft_execution.get("expected_candidate_ids") or []) != [args.candidate_id]
        or draft_execution.get("run_id") != draft_run.run_id
        or draft_execution.get("project_id") != args.project_id
    ):
        raise AssertionError("Draft dispatch allow-list differs from the clicked candidate.")
    _wait_until(
        app,
        lambda: not screen.viewmodel.active and screen._flow_step == "drafts",
        timeout_seconds=30.0,
        description="Drafts screen",
    )
    persisted = services.projects.load(args.project_id)
    if list(persisted.candidate_draft_artifacts) != [args.candidate_id]:
        raise AssertionError("Draft completion published a sibling candidate.")
    draft_path = Path(persisted.candidate_draft_artifacts[args.candidate_id])
    draft, draft_record = _draft_candidate_record(draft_path, args.candidate_id)
    if draft.get("project_id") != args.project_id or draft.get("analysis_id") != project.analysis_id:
        raise AssertionError("DraftArtifact project/Analysis lineage is invalid.")
    if list(draft_run.settings_snapshot.get("candidate_ids") or []) != [args.candidate_id]:
        raise AssertionError("Persisted Draft run allow-list differs from the clicked candidate.")
    preview = draft_record.get("preview", {})
    preview_path = Path(str(preview.get("output_file") or "")) if isinstance(preview, dict) else Path()
    preview_button = screen.findChild(QPushButton, f"draft-preview-candidate-{args.candidate_id}")
    if preview_button is None or not preview_button.isEnabled():
        raise AssertionError("Ready Draft did not expose its Creative Preview action.")
    QTest.mouseClick(preview_button, Qt.MouseButton.LeftButton)
    _settle(app, 30)
    if screen.preview.active_media_path is None or Path(screen.preview.active_media_path).resolve() != preview_path.resolve():
        raise AssertionError("Creative Preview action opened a different candidate artifact.")
    evidence["screenshots"]["creative_preview"] = _capture(window, screenshots, "03-creative-preview")

    approve = screen.findChild(QPushButton, f"approve-candidate-{args.candidate_id}")
    if approve is None or not approve.isEnabled():
        raise AssertionError("Ready Draft did not expose its exact approval action.")
    QTest.mouseClick(approve, Qt.MouseButton.LeftButton)
    _settle(app, 20)
    approved = services.projects.load(args.project_id)
    if approved.selected_candidate_ids != [args.candidate_id]:
        raise AssertionError(f"Draft approval routed {approved.selected_candidate_ids!r}.")
    if not screen.production_button.isVisible() or not screen.production_button.isEnabled():
        raise AssertionError("Approved Draft did not expose the Final CTA.")
    evidence["screenshots"]["approved_draft"] = _capture(window, screenshots, "04-approved-draft")

    known_runs = {run.run_id for run in services.runs_for(approved)}
    original_question = QMessageBox.question
    try:
        QMessageBox.question = staticmethod(  # type: ignore[method-assign]
            lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
        )
        QTest.mouseClick(screen.production_button, Qt.MouseButton.LeftButton)
    finally:
        QMessageBox.question = original_question  # type: ignore[method-assign]
    _wait_until(
        app,
        lambda: screen._flow_step == "processing",
        timeout_seconds=20.0,
        description="Final processing screen",
    )
    evidence["screenshots"]["final_processing"] = _capture(window, screenshots, "05-final-processing")
    final_run = _wait_for_run(
        app,
        services,
        args.project_id,
        known_runs,
        RunKind.SELECTED_RENDER,
        args.timeout_seconds,
    )
    if final_run.status not in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}:
        raise AssertionError(f"Positive Final ended as {final_run.status}: {final_run.error_summary}")
    final_delta = _new_runs(services, args.project_id, known_runs)
    if len(final_delta) != 1 or final_delta[0].run_id != final_run.run_id:
        raise AssertionError("One Final click created duplicate or unexpected runs.")
    final_execution = final_run.settings_snapshot.get("execution", {})
    if (
        list(final_run.settings_snapshot.get("candidate_ids") or []) != [args.candidate_id]
        or not isinstance(final_execution, dict)
        or list(final_execution.get("expected_candidate_ids") or []) != [args.candidate_id]
        or final_execution.get("run_id") != final_run.run_id
        or final_execution.get("project_id") != args.project_id
    ):
        raise AssertionError("Final dispatch allow-list differs from the approved Draft.")
    _wait_until(
        app,
        lambda: not screen.viewmodel.active and screen._flow_step == "finished",
        timeout_seconds=30.0,
        description="Results screen",
    )
    evidence["screenshots"]["results"] = _capture(window, screenshots, "06-results")

    analysis_sha256_after = _sha256(analysis_path)
    all_runs = services.runs_for(services.projects.load(args.project_id))
    run_kinds = {run.run_id: run.run_kind for run in all_runs}
    expected_run_kinds = {
        draft_run.run_id: RunKind.DRAFT,
        final_run.run_id: RunKind.SELECTED_RENDER,
    }
    persisted_analysis = services.projects.load(args.project_id)
    if (
        analysis_sha256_after != analysis_sha256_before
        or run_kinds != expected_run_kinds
        or persisted_analysis.analysis_artifact_path != project.analysis_artifact_path
        or persisted_analysis.analysis_id != project.analysis_id
        or persisted_analysis.analysis_fingerprint != project.analysis_fingerprint
    ):
        raise AssertionError("Acceptance flow changed or re-ran Analysis.")
    evidence["analysis"].update(
        {
            "sha256_after": analysis_sha256_after,
            "unchanged": True,
            "analysis_runs_created": [],
            "created_run_kinds": run_kinds,
        }
    )
    evidence["draft"] = {
        "run_id": draft_run.run_id,
        "status": draft_run.status,
        "candidate_ids": list(draft_run.settings_snapshot.get("candidate_ids") or []),
        "draft_id": draft.get("draft_id"),
        "artifact_path": str(draft_path.resolve()),
        "artifact_sha256": _sha256(draft_path),
        "creative_preview_path": str(preview_path.resolve()),
        "creative_preview_sha256": _sha256(preview_path),
        "invalidated_stages": draft_run.invalidated_stages,
    }
    evidence["final"] = _validate_final(
        services,
        runtime,
        args.project_id,
        args.candidate_id,
        final_run,
        draft_path,
        draft_run.run_id,
        analysis_sha256_before,
    )
    evidence["final"]["candidate_ids"] = list(final_run.settings_snapshot.get("candidate_ids") or [])
    evidence["final"]["invalidated_stages"] = final_run.invalidated_stages
    persisted = services.projects.load(args.project_id)
    if persisted.selected_candidate_ids:
        raise AssertionError("Successful Final left a ready candidate in the pending export queue.")
    if persisted.last_final_result_id != evidence["final"]["clip_result_id"]:
        raise AssertionError("Final completion did not persist its exact canonical result identity.")
    if screen.final_results.active_output_id != evidence["final"]["clip_result_id"]:
        raise AssertionError("Results viewer did not bind its actions to the persisted Final identity.")
    if persisted.latest_run_id != final_run.run_id:
        raise AssertionError("Project latest_run_id does not identify the completed Final run.")
    evidence["persisted_before_restart"] = {
        "status": persisted.status,
        "latest_run_id": persisted.latest_run_id,
        "selected_candidate_ids": persisted.selected_candidate_ids,
        "review_selected_candidate_ids": persisted.review_selected_candidate_ids,
        "candidate_draft_status": persisted.candidate_draft_statuses.get(args.candidate_id),
        "candidate_approval_state": persisted.candidate_approval_states.get(args.candidate_id),
        "candidate_export_status": persisted.candidate_export_statuses.get(args.candidate_id),
        "last_final_result_id": persisted.last_final_result_id,
    }
    write_json(work_directory / "acceptance-evidence.json", evidence)
    _shutdown(window, app)
    print(json.dumps(evidence, ensure_ascii=True))
    return 0


def _run_restart(args: argparse.Namespace) -> int:
    engine_root = Path(__file__).resolve().parents[1]
    work_directory = args.work_directory.expanduser().resolve()
    data_directory = work_directory / "desktop-data"
    if not (data_directory / "projects" / args.project_id / "project.json").is_file():
        raise FileNotFoundError("Acceptance project is absent; run the e2e phase first.")
    baseline = read_json(work_directory / "acceptance-evidence.json", {})
    if not isinstance(baseline, dict):
        raise AssertionError("Acceptance evidence is missing before restart.")
    runtime = _runtime(engine_root, data_directory)
    services = DesktopServices.create(runtime)
    app = QApplication.instance() or QApplication([])
    if app.platformName().casefold() != "windows":
        raise AssertionError(f"Real Desktop restart requires the Windows Qt platform, got {app.platformName()!r}.")
    app.setStyleSheet(load_theme())
    window = MainWindow(services)
    window.resize(args.width, args.height)
    window.show()
    _settle(app, 30)
    if not window.isVisible():
        raise AssertionError("Restarted source-runtime Desktop window is not visible.")
    if window.stack.currentIndex() != window.project_index:
        raise AssertionError("Process restart did not restore the persisted project route.")
    screen = window.project_screen
    project = services.projects.load(args.project_id)
    if screen.project is None or screen.project.project_id != args.project_id or screen._flow_step != "finished":
        raise AssertionError("Process restart did not restore the exact Results workspace.")
    expected_final = baseline.get("final", {})
    expected_result_id = str(expected_final.get("clip_result_id") or "") if isinstance(expected_final, dict) else ""
    expected_final_run_id = str(expected_final.get("run_id") or "") if isinstance(expected_final, dict) else ""
    expected_final_status = str(expected_final.get("status") or "") if isinstance(expected_final, dict) else ""
    draft_evidence = baseline.get("draft", {})
    draft_run_id = str(draft_evidence.get("run_id") or "") if isinstance(draft_evidence, dict) else ""
    draft_path = Path(str(draft_evidence.get("artifact_path") or "")) if isinstance(draft_evidence, dict) else Path()
    analysis_evidence = baseline.get("analysis", {})
    analysis_path = Path(str(analysis_evidence.get("artifact_path") or "")) if isinstance(analysis_evidence, dict) else Path()
    analysis_sha256 = str(analysis_evidence.get("sha256_after") or "") if isinstance(analysis_evidence, dict) else ""
    if not all((expected_result_id, expected_final_run_id, draft_run_id, analysis_sha256)):
        raise AssertionError("Restart baseline is missing exact run/artifact identities.")
    if (
        project.selected_candidate_ids != []
        or project.review_selected_candidate_ids != [args.candidate_id]
        or project.candidate_draft_statuses.get(args.candidate_id) != "ready"
        or project.candidate_approval_states.get(args.candidate_id) != "approved"
        or project.candidate_export_statuses.get(args.candidate_id) != "ready"
        or project.last_final_result_id != expected_result_id
    ):
        raise AssertionError("Candidate lifecycle/Results identity did not survive process restart.")
    if screen.final_results.active_output_id != expected_result_id:
        raise AssertionError("Restarted Results viewer did not bind to the persisted result identity.")
    latest = services.runs.load(args.project_id, str(project.latest_run_id))
    if (
        project.latest_run_id != expected_final_run_id
        or latest.run_id != expected_final_run_id
        or latest.run_kind != RunKind.SELECTED_RENDER
        or latest.status != expected_final_status
    ):
        raise AssertionError("Restart did not restore the exact completed Final run/status.")
    restart_runs = {run.run_id: run.run_kind for run in services.runs_for(project)}
    if restart_runs != {
        draft_run_id: RunKind.DRAFT,
        expected_final_run_id: RunKind.SELECTED_RENDER,
    }:
        raise AssertionError("Restart added, removed, or reclassified a persisted run.")
    if (
        not analysis_path.is_file()
        or _sha256(analysis_path) != analysis_sha256
        or Path(project.analysis_artifact_path or "").resolve() != analysis_path.resolve()
        or project.analysis_id != analysis_evidence.get("analysis_id")
        or project.analysis_fingerprint != analysis_evidence.get("analysis_fingerprint")
    ):
        raise AssertionError("Restart changed the reused Analysis identity or bytes.")
    projection = services.run_projection(latest)
    if len(projection.primary_results) != 1:
        raise AssertionError("Restart restored an ambiguous Final result set.")
    result = projection.primary_results[0]
    if (
        result.clip_result_id != expected_result_id
        or result.candidate_id != args.candidate_id
        or result.run_id != expected_final_run_id
    ):
        raise AssertionError("Restart opened a sibling or stale Final identity.")
    output = Path(result.output_file)
    if not output.is_file() or _sha256(output) != str(expected_final.get("artifact_checksum") or ""):
        raise AssertionError("Restarted Results lost the validated Final MP4.")
    revalidated = _validate_final(
        services,
        runtime,
        args.project_id,
        args.candidate_id,
        latest,
        draft_path,
        draft_run_id,
        analysis_sha256,
    )
    if (
        revalidated["clip_result_id"] != expected_result_id
        or revalidated["artifact_checksum"] != expected_final.get("artifact_checksum")
        or revalidated["parity"] != expected_final.get("parity")
    ):
        raise AssertionError("Restart revalidation differs from the accepted Final evidence.")
    screenshot = _capture(window, work_directory / "screenshots", "07-results-after-process-restart")
    evidence = {
        "process_restart": True,
        "project_id": project.project_id,
        "flow_step": screen._flow_step,
        "latest_run_id": latest.run_id,
        "latest_run_status": latest.status,
        "candidate_id": result.candidate_id,
        "clip_result_id": result.clip_result_id,
        "selected_candidate_ids": project.selected_candidate_ids,
        "review_selected_candidate_ids": project.review_selected_candidate_ids,
        "candidate_draft_status": project.candidate_draft_statuses.get(args.candidate_id),
        "candidate_approval_state": project.candidate_approval_states.get(args.candidate_id),
        "candidate_export_status": project.candidate_export_statuses.get(args.candidate_id),
        "final_path": str(output.resolve()),
        "final_sha256": _sha256(output),
        "screenshot": screenshot,
    }
    write_json(work_directory / "restart-evidence.json", evidence)
    _shutdown(window, app)
    print(json.dumps(evidence, ensure_ascii=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("e2e", "restart"))
    parser.add_argument("--source-data", type=Path, default=default_data_directory())
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    return _run_acceptance(args) if args.phase == "e2e" else _run_restart(args)


if __name__ == "__main__":
    raise SystemExit(main())
