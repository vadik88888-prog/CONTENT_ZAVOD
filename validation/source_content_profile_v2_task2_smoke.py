from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from static_ffmpeg import run as static_ffmpeg_run

import app.pipeline as pipeline_module
from app.analysis_artifact import AnalysisArtifact
from app.config import AppConfig
from app.continuity import build_continuity_decision
from app.draft_artifact import DraftArtifact
from app.pipeline import Pipeline
from app.utils import read_json, stable_file_hash, stable_text_hash, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation" / "evidence" / "source-content-profile-v2-task2" / "runtime-lineage-smoke.json"


def _prepare_media(_source_path: Path, work_directory: Path) -> dict[str, Any]:
    audio = work_directory / "audio.wav"
    audio.write_bytes(b"runtime-smoke-audio")
    metadata = {
        "duration": 42.0,
        "width": 320,
        "height": 180,
        "fps": 30.0,
        "audio_streams": 1,
        "audio_path": str(audio),
    }
    write_json(work_directory / "metadata.json", metadata)
    return metadata


def _transcribe(
    _audio_path: Path, source_id: str, source_duration: float, _config: AppConfig, destination: Path,
) -> dict[str, Any]:
    sentences = (
        "Why do projects fail before launch?",
        "They fail when teams skip the smallest validation step.",
        "The practical fix is to test one assumption before scaling.",
    )
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    cursor = 1.0
    for segment_id, sentence in enumerate(sentences):
        start = cursor
        for token in sentence.split():
            words.append({"start": cursor, "end": cursor + 0.45, "text": token, "confidence": 0.99})
            cursor += 0.55
        segments.append({
            "id": segment_id, "start": start, "end": cursor, "text": sentence, "speaker": "speaker-1",
        })
        cursor += 0.8
    transcript = {
        "source_id": source_id,
        "language": "en",
        "duration": source_duration,
        "segments": segments,
        "words": words,
        "model": "task2-runtime-smoke",
        "runtime": {"device": "cpu"},
        "processing_duration_seconds": 0.01,
    }
    write_json(destination, transcript)
    destination.with_suffix(".txt").write_text(" ".join(sentences), encoding="utf-8")
    return transcript


def _config(profile: dict[str, Any], intent: str) -> AppConfig:
    config = AppConfig(score_threshold=0)
    config.content_understanding.manual_override = profile
    config.content_understanding.editorial_intent = intent
    config.validate()
    return config


def _run_analysis(root: Path, source: Path, config: AppConfig, run_id: str) -> tuple[Any, AnalysisArtifact]:
    result = Pipeline(
        root, config, mock_ai=True, analysis_only=True, run_id=run_id, project_id="project-task2-smoke",
    ).run(input_path=str(source))
    if result.analysis_path is None:
        raise RuntimeError("analysis-only runtime did not publish AnalysisArtifact")
    return result, AnalysisArtifact.read_verified(result.analysis_path)


def _primary_evidence(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "start": item["start_seconds"],
            "end": item["end_seconds"],
            "segment_id": item.get("observation", {}).get("segment_id"),
        }
        for item in candidate["multimodal_provenance"]["transcript_evidence"]
    ]


def _fingerprint(value: Any) -> str:
    return stable_text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def run_smoke(output: Path) -> dict[str, Any]:
    ffmpeg, ffprobe = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
    with tempfile.TemporaryDirectory(prefix="content-factory-task2-") as temporary:
        root = Path(temporary)
        source = root / "real-source.mp4"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=blue:s=320x180:r=30:d=3", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(source),
        ], check=True, capture_output=True)
        probe = json.loads(subprocess.run([
            ffprobe, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate", "-of", "json", str(source),
        ], check=True, capture_output=True, text=True, encoding="utf-8").stdout)

        pipeline_module.prepare_media = _prepare_media
        pipeline_module.transcribe = _transcribe
        config_a = _config(
            {"format": "dialogue", "editorial_mode": "interview", "domain": "business", "traits": []},
            "Analysis A intent",
        )
        config_b = _config(
            {"format": "gameplay", "editorial_mode": "commentary", "domain": "gaming", "traits": ["visual_led"]},
            "Analysis B intent",
        )
        result_a, analysis_a = _run_analysis(root, source, config_a, "analysis-a")
        result_b, analysis_b = _run_analysis(root, source, config_b, "analysis-b")
        semantic_a = analysis_a.load_reference("semantic_boundaries")
        semantic_b = analysis_b.load_reference("semantic_boundaries")
        candidate_a = semantic_a["candidates"][0]
        candidate_b = semantic_b["candidates"][0]
        boundary_a = candidate_a["boundary_diagnostics"]["boundary_decision"]
        boundary_b = candidate_b["boundary_diagnostics"]["boundary_decision"]
        continuity_a = build_continuity_decision(
            candidate_id=candidate_a["id"], boundary_decision=boundary_a,
            primary_evidence=_primary_evidence(candidate_a), multimodal_context={},
        )
        continuity_b = build_continuity_decision(
            candidate_id=candidate_b["id"], boundary_decision=boundary_b,
            primary_evidence=_primary_evidence(candidate_b), multimodal_context={},
        )
        if continuity_a is None or continuity_b is None:
            raise RuntimeError("A-2 continuity decision was not produced")

        candidate_id = analysis_a.load_reference("candidate_data")["candidates"][0]["id"]
        draft_result = Pipeline(
            root, config_b, mock_ai=True, analysis_artifact_path=result_a.analysis_path,
            selected_candidate_ids=[candidate_id], expected_analysis_id=analysis_a.analysis_id,
            expected_analysis_fingerprint=analysis_a.analysis_fingerprint, draft_only=True,
            run_id="draft-from-analysis-a", project_id="project-task2-smoke",
        ).run(input_path=str(source))
        if draft_result.draft_path is None:
            raise RuntimeError("Draft(A) did not publish DraftArtifact")
        draft = DraftArtifact.read(draft_result.draft_path)
        draft_report = read_json(draft_result.report_path, {})
        report_b = read_json(result_b.report_path, {})
        checks = {
            "same_candidate_id": candidate_a["id"] == candidate_b["id"],
            "same_content_map": analysis_a.load_reference("content_map") == analysis_b.load_reference("content_map"),
            "same_boundary_decision": boundary_a == boundary_b,
            "same_a2_continuity_decision": continuity_a.model_dump(mode="json") == continuity_b.model_dump(mode="json"),
            "draft_reads_analysis_a_profile": (
                draft_report["content_understanding"]["profile"]["manual_override"]["format"] == "dialogue"
            ),
            "draft_binds_analysis_a_run": draft.analysis_run_id == "analysis-a",
            "draft_binds_analysis_a_sha256": draft.analysis_artifact_sha256 == stable_file_hash(result_a.analysis_path),
            "analysis_a_snapshot_survives_b": (
                analysis_a.load_reference("content_profile")["manual_override"]["format"] == "dialogue"
                and analysis_b.load_reference("content_profile")["manual_override"]["format"] == "gameplay"
            ),
            "all_references_integrity_bound": set(analysis_a.references) == set(analysis_a.reference_integrity),
            "final_selection_snapshotted": "final_selection" in analysis_a.references,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Task 2 runtime invariant failed: {checks}")
        evidence = {
            "schema_version": "source-content-profile-v2-task2-smoke.1",
            "created_at": utc_now(),
            "source": {
                "kind": "generated_real_mp4",
                "sha256": stable_file_hash(source),
                "byte_size": source.stat().st_size,
                "probe": probe,
            },
            "analysis_a": {
                "analysis_id": analysis_a.analysis_id,
                "analysis_run_id": analysis_a.analysis_run_id,
                "analysis_artifact_sha256": analysis_a.verified_sha256,
                "analysis_artifact_byte_size": analysis_a.verified_byte_size,
                "reference_count": len(analysis_a.references),
                "reference_integrity": analysis_a.reference_integrity,
                "profile_format": analysis_a.load_reference("content_profile")["manual_override"]["format"],
            },
            "analysis_b": {
                "analysis_id": analysis_b.analysis_id,
                "analysis_run_id": analysis_b.analysis_run_id,
                "profile_format": analysis_b.load_reference("content_profile")["manual_override"]["format"],
                "boundary_cache_hits": {
                    name: report_b["stages"][name]["cache_hit"]
                    for name in ("vision_pass1", "global_content_map", "semantic_boundaries")
                },
            },
            "candidate_id": candidate_a["id"],
            "boundary_decision_sha256": _fingerprint(boundary_a),
            "a2_continuity_decision_sha256": _fingerprint(continuity_a.model_dump(mode="json")),
            "draft_a": {
                "draft_id": draft.draft_id,
                "schema_version": draft.schema_version,
                "analysis_run_id": draft.analysis_run_id,
                "analysis_artifact_sha256": draft.analysis_artifact_sha256,
                "profile_format_read_by_draft": draft_report["content_understanding"]["profile"]["manual_override"]["format"],
            },
            "checks": checks,
            "result": "PASS",
        }
    write_json(output, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evidence = run_smoke(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "result": evidence["result"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
