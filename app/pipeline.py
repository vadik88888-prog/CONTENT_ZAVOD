from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, cast

from app.ai import (
    SEMANTIC_AI_PAYLOAD_VERSION,
    get_scorer,
    get_transformer,
    get_vision_provider,
    sanitize_api_error,
)
from app.ai_cost import collect_vision_usage
from app.analysis_artifact import AnalysisArtifact, AnalysisArtifactError, candidate_review_payload, new_analysis_artifact, potential_counts
from app.candidate_review import validate_boundary_override
from app.draft_artifact import DraftArtifact, DraftArtifactError, new_draft_artifact
from app.audio_modes import tts_eligibility
from app.clip_results import ClipResult, primary_clip_results, result_paths
from app.diversity import interval_metrics, transcript_similarity
from app.audio_features import analyse_audio
from app.audio_semantics import (
    AUDIO_SEMANTIC_ANALYSIS_VERSION,
    analyse_semantic_audio,
    validate_semantic_audio,
)
from app.audio_models import AudioProject
from app.audio_service import AudioCompositionService, audio_report_section
from app.config import AppConfig
from app.creative_contracts import CompiledRenderPlan, CreativeIntent
from app.creative_evidence import has_usable_composition_evidence
from app.creative_lifecycle import (
    CreativeArtifactError,
    creative_policy_changed,
    CandidateCreativeHandoff,
    load_candidate_creative_identity,
    revise_creative_intent,
)
from app.content_understanding import (
    CONTENT_PROFILE_CONTRACT_VERSION,
    CONTENT_PROFILE_DETECTOR_VERSION,
    PUBLISHABLE_STORY_EXPANSION_VERSION,
    CONTENT_STRATEGY_VERSION,
    SEMANTIC_CANDIDATE_GENERATION_VERSION,
    build_coverage_map,
    build_global_content_map,
    build_video_content_profile,
    generate_semantic_candidates,
    recommend_clip_count,
    refresh_content_map_multimodal_evidence,
    select_with_coverage,
    story_units_artifact,
    ensure_candidate_boundary_decision,
    expand_publishable_story_candidates,
    validate_video_content_profile,
)
from app.candidate_quality import (
    CANDIDATE_QUALITY_SCHEMA_VERSION,
    EligibilityDecision,
    resolve_eligibility_decision,
)
from app.editorial_profile_policy import evaluate_editorial_candidate, resolve_editorial_profile
from app.content_transformation import (
    TRANSFORMATION_ENGINE_VERSION,
    run_content_transformation,
    validate_transformation_outcome,
)
from app.errors import (
    NO_RENDERABLE_CLIPS,
    NO_RENDERABLE_CLIPS_MESSAGE,
    AudioCompositionError,
    ClipEngineError,
    ProductionPlanError,
    ProductionRenderError,
    SemanticCredentialError,
    StageError,
    TTSError,
    TransformationProviderError,
)
from app.intelligence import intelligence_summary, local_rank, merge_ai_ranking, shortlist
from app.local_scoring import score_candidates
from app.media import prepare_media
from app.models import Candidate, ScoredCandidate, candidate_from_dict, scored_from_dict
from app.multimodal_evidence import (
    MULTIMODAL_ANALYSIS_VERSION,
    build_multimodal_timeline,
    multimodal_analysis_run_id,
    validate_multimodal_timeline,
)
from app.multimodal_candidates import (
    CANDIDATE_PROVENANCE_SCHEMA_VERSION,
    PASS2_EVIDENCE_SCHEMA_VERSION,
    candidate_pass2_anchors,
    enrich_shortlist_with_pass2,
    generate_multimodal_candidates,
    project_candidate_audio_evidence,
    refresh_candidate_timeline_evidence,
)
from app.rendering import render_clip
from app.reporting import make_report
from app.run_artifacts import make_run_artifact_metadata, write_run_artifact_metadata
from app.run_manifest import is_run_scoped_path, write_run_manifest
from app.production_models import ProductionPlan
from app.speech_clarity_policy import SPEECH_CLARITY_POLICY_VERSION
from app.quality_report import (
    build_editorial_final_handoff,
    build_quality_report,
    exact_dialogue_semantic_blocker,
)
from app.production_plan import (
    PRODUCTION_PLAN_VERSION,
    ProductionPlanEnvelopeContext,
    build_production_plan,
    production_summary,
)
from app.production_feasibility import (
    PRODUCTION_FEASIBILITY_POLICY_VERSION,
    resolve_recommendation_production_feasibility,
    validate_production_feasibility_artifact,
)
from app.product_flow import DeepAnalysisDecision, resolve_deep_analysis
from app.scene_detection import detect_scene_boundaries
from app.secure_secrets import validate_api_key
from app.selection import select_clips
from app.sources import Source, local_source, url_source, validate_source_arguments
from app.subtitles import create_ass
from app.transcript_features import analyse_transcript
from app.transcription import transcribe
from app.semantic_extraction import build_source_context
from app.transformation_prompts import PROMPT_VERSIONS
from app.transformation_models import FINAL_SCRIPT_CONTRACT_VERSION, validate_final_script
from app.tts_service import TTSService, tts_report_section
from app.utils import AtomicWriteError, read_json, safe_name, stable_file_hash, stable_text_hash, utc_now, write_json
from app.video_composition import VideoCompositionService, production_render_report_section
from app.visual_analysis import analyse_video_subjects
from app.vision_intelligence import (
    VisionGateway,
    build_candidate_bounded_pass2_timeline,
    build_pass2_request,
    validate_pass2_request,
    validate_pass2_result,
    validate_vision_artifact,
)
from app.virality import (
    apply_profile_weighting_after_hard_gates,
    apply_virality_ranking,
    build_virality_assessments,
)


INTELLIGENCE_STAGES = (
    "transcript_features", "audio_features", "scene_detection", "candidates_v2",
    "local_scoring", "shortlist", "multimodal_scoring", "ai_ranking", "final_selection", "visual_analysis", "multimodal_seed_timeline", "audio_semantics", "multimodal_timeline", "pre_vision_content_profile", "video_content_profile", "vision_pass1",
    "global_content_map_base", "candidate_seed_basis", "global_content_map", "story_units", "semantic_boundaries", "vision_pass2", "virality_profiles", "virality_ranking",
    "production_feasibility", "coverage_map", "clip_count_recommendation", "render", "report",
)
INTELLIGENCE_ENGINE_VERSION = "1.10.0"
TRANSFORMATION_STAGES = (
    "transformation_source_context", "transformation_semantic_representation",
    "transformation_narrative_plan", "transformation_script_draft",
    "transformation_script_validation", "transformation_final_script", "transformation_result",
)
PRODUCTION_PLAN_STAGES = ("production_plan",)
TTS_STAGES = ("tts_generation",)
AUDIO_COMPOSITION_STAGES = ("audio_composition",)
PRODUCTION_RENDER_STAGES = ("production_render",)
DRAFT_COMPOSITION_PASS2_MODEL = "gpt-5.6-terra"
DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION = "6B.candidate-composition-pass2.1"


@dataclass(slots=True)
class PipelineResult:
    work_directory: Path
    output_directory: Path
    report_path: Path
    selected_clips: int
    output_files: list[Path]
    warnings: list[str]
    terminal_status: str = "completed"
    error_code: str | None = None
    analysis_path: Path | None = None
    analysis_id: str | None = None
    draft_path: Path | None = None
    draft_id: str | None = None


class RunHeartbeat:
    """Persist lightweight liveness while an expensive stage has no stdout."""

    INTERVAL_SECONDS = 10.0

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = {"stage": "preparing"}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.beat(str(self._state["stage"]))
        self._thread = threading.Thread(
            target=_heartbeat_loop,
            args=(self.path, self._stop_event, self._state_lock, self._state),
            name="pipeline-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def beat(self, stage: str) -> None:
        with self._state_lock:
            self._state["stage"] = stage
        _write_heartbeat(self.path, self._state_lock, self._state)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def __del__(self) -> None:
        self.stop()


def _heartbeat_loop(
    path: Path,
    stop_event: threading.Event,
    state_lock: threading.Lock,
    state: dict[str, str],
) -> None:
    while not stop_event.wait(RunHeartbeat.INTERVAL_SECONDS):
        _write_heartbeat(path, state_lock, state)


def _write_heartbeat(path: Path, state_lock: threading.Lock, state: dict[str, str]) -> None:
    try:
        with state_lock:
            stage = state["stage"]
        write_json(path, {"updated_at": utc_now(), "stage": stage, "pid": os.getpid()})
    except OSError:
        # A heartbeat must never make a media job fail merely because a scanner
        # or Explorer has a transient lock on the file.
        pass


class StageTracker:
    def __init__(self, state_path: Path, *, heartbeat: RunHeartbeat | None = None) -> None:
        self.path = state_path
        self.heartbeat = heartbeat
        self.data = read_json(state_path, {"created_at": utc_now(), "stages": {}})
        self.data.setdefault("stages", {})
        self.persistence_error: AtomicWriteError | None = None

    def completed(self, name: str, artifact: Path, cache_key: str | None = None) -> bool:
        stage = self.data["stages"].get(name, {})
        return (
            stage.get("status") == "completed"
            and artifact.exists()
            and (cache_key is None or stage.get("cache_key") == cache_key)
        )

    def start(self, name: str, cache_key: str | None = None, cache_hit: bool | None = None) -> None:
        if self.heartbeat is not None:
            self.heartbeat.beat(name)
        self.data["stages"][name] = {
            "status": "running", "started_at": utc_now(), "_started": time.perf_counter(),
            "cache_key": cache_key,
        }
        if cache_hit is not None:
            self.data["stages"][name]["cache_hit"] = cache_hit
        self._save()

    def finish(self, name: str, status: str = "completed", error: str | None = None) -> None:
        if self.heartbeat is not None:
            self.heartbeat.beat(name)
        stage = self.data["stages"].setdefault(name, {})
        started = stage.pop("_started", None)
        stage["status"] = status
        stage["finished_at"] = utc_now()
        stage["duration_seconds"] = round(time.perf_counter() - started, 3) if started else 0
        if error:
            stage["error"] = error
        self._save()

    def skip(self, name: str, reason: str) -> None:
        self.data["stages"][name] = {
            "status": "skipped", "reason": reason, "finished_at": utc_now(), "duration_seconds": 0,
        }
        self._save()

    def invalidate(self, reason: str, names: tuple[str, ...] | None = None) -> None:
        for name in names or tuple(self.data["stages"]):
            if name in self.data["stages"]:
                self.data["stages"][name] = {"status": "pending", "reason": reason}
        self._save()

    # Kept for compatibility with earlier state tests and cached work directories.
    def set_config_signature(self, value: str) -> None:
        self.data["config_signature"] = value
        self._save()

    def set_run_metadata(self, value: dict[str, Any]) -> None:
        """Persist engine-owned artifact locations alongside stage progress."""

        self.data["run"] = value
        self._save()

    def _save(self) -> None:
        # A later successful save supersedes an earlier transient failure.  Do
        # not persist a stale degraded marker into the canonical state file.
        self.data.pop("state_persistence", None)
        safe = json.loads(json.dumps(self.data))
        for stage in safe.get("stages", {}).values():
            stage.pop("_started", None)
        try:
            write_json(self.path, safe)
        except AtomicWriteError as error:
            self.persistence_error = error
            self.data["state_persistence"] = {
                "status": "degraded",
                "error": str(error.cause),
                "error_type": type(error.cause).__name__,
                "winerror": getattr(error.cause, "winerror", None),
                "fallback_state_path": str(error.fallback_path) if error.fallback_path else None,
            }
        else:
            self.persistence_error = None


class Pipeline:
    def __init__(
        self, root: Path, config: AppConfig, mock_ai: bool = False,
        no_ai_rerank: bool = False, recompute_intelligence: bool = False,
        transform_script: bool | None = None, no_ai_transformation: bool = False,
        recompute_transformation: bool = False,
        production_plan_only: bool = False, recompute_production_plan: bool = False,
        tts_only: bool = False, recompute_tts: bool = False, disable_tts: bool = False,
        audio_only: bool = False, recompute_audio: bool = False,
        production_render_only: bool = False, recompute_production_render: bool = False,
        disable_production_render: bool = False,
        analysis_only: bool = False,
        analysis_artifact_path: Path | None = None,
        selected_candidate_ids: list[str] | None = None,
        expected_analysis_id: str | None = None,
        expected_analysis_fingerprint: str | None = None,
        draft_only: bool = False,
        draft_artifact_path: Path | None = None,
        candidate_boundary_overrides: dict[str, dict[str, Any]] | None = None,
        run_id: str | None = None,
        upstream_run_directory: Path | None = None,
        project_id: str | None = None,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.mock_ai = mock_ai
        self.no_ai_rerank = no_ai_rerank
        self.recompute_intelligence = recompute_intelligence
        self.transform_script = transform_script
        self.no_ai_transformation = no_ai_transformation
        self.recompute_transformation = recompute_transformation
        self.production_plan_only = production_plan_only
        self.recompute_production_plan = recompute_production_plan
        self.tts_only = tts_only
        self.recompute_tts = recompute_tts
        self.disable_tts = disable_tts
        self.audio_only = audio_only
        self.recompute_audio = recompute_audio
        self.production_render_only = production_render_only
        self.recompute_production_render = recompute_production_render
        self.disable_production_render = disable_production_render
        self.analysis_only = analysis_only
        self.analysis_artifact_path = analysis_artifact_path.resolve() if analysis_artifact_path else None
        self.selected_candidate_ids = list(selected_candidate_ids or [])
        self.expected_analysis_id = expected_analysis_id
        self.expected_analysis_fingerprint = expected_analysis_fingerprint
        self.draft_only = draft_only
        self.draft_artifact_path = draft_artifact_path.resolve() if draft_artifact_path else None
        self.candidate_boundary_overrides = dict(candidate_boundary_overrides or {})
        self.run_id = safe_name(run_id or f"run-{uuid.uuid4().hex}", "run")
        self.upstream_run_directory = upstream_run_directory
        self.project_id = project_id
        self.started_at = ""
        self.run_work_directory: Path | None = None
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self._heartbeat: RunHeartbeat | None = None
        self._native_evidence_context: tuple[list[Any], dict[str, Any], dict[str, Any]] | None = None

    def __del__(self) -> None:
        heartbeat = getattr(self, "_heartbeat", None)
        if heartbeat is not None:
            heartbeat.stop()

    def run(self, input_path: str | None = None, url: str | None = None) -> PipelineResult:
        validate_source_arguments(input_path, url)
        self.started_at = utc_now()
        source, work_directory, output_directory = self._prepare_source(input_path, url)
        assert self.run_work_directory is not None
        self._heartbeat = RunHeartbeat(self.run_work_directory / "heartbeat.json")
        self._heartbeat.start()
        tracker = StageTracker(self.run_work_directory / "state.json", heartbeat=self._heartbeat)
        self._publish_run_paths(tracker, work_directory, output_directory)
        # Result lifecycle state belongs to exactly one run.  Source analysis
        # artifacts remain reusable across runs, with a separately locked cache
        # index.  This keeps a new render from inheriting an old final-output
        # state while retaining the expensive media/transcript cache.
        source_cache = StageTracker(work_directory / "cache-state.json")
        if self.analysis_artifact_path is not None:
            if not self.draft_only:
                raise ClipEngineError(
                    "Direct production render from analysis is disabled. Build and review a draft preview first."
                )
            return self._complete_run(self._run_draft_preview(
                tracker=tracker,
                source=source,
                work_directory=work_directory,
                output_directory=output_directory,
            ))
        if self.draft_artifact_path is not None:
            return self._complete_run(self._run_production_from_draft(
                tracker=tracker,
                source=source,
                work_directory=work_directory,
                output_directory=output_directory,
            ))
        if self.recompute_tts:
            tts_names = tuple(
                name for name in tracker.data.get("stages", {})
                if name == "report" or any(name.startswith(base) for base in TTS_STAGES)
            )
            tracker.invalidate("Запрошен --recompute-tts.", tts_names)
        if self.recompute_audio:
            audio_names = tuple(
                name for name in tracker.data.get("stages", {})
                if name == "report" or any(name.startswith(base) for base in AUDIO_COMPOSITION_STAGES)
            )
            tracker.invalidate("Запрошен --recompute-audio.", audio_names)
        if self.recompute_production_render:
            production_render_names = tuple(
                name for name in tracker.data.get("stages", {})
                if name == "report" or any(name.startswith(base) for base in PRODUCTION_RENDER_STAGES)
            )
            tracker.invalidate("Запрошен --recompute-production-render.", production_render_names)
        if self.tts_only:
            return self._complete_run(self._run_tts_only(tracker, source, work_directory, output_directory))
        if self.audio_only:
            return self._complete_run(self._run_audio_only(tracker, source, work_directory, output_directory))
        if self.production_render_only:
            return self._complete_run(self._run_production_render_only(tracker, source, work_directory, output_directory))
        if self.recompute_intelligence:
            tracker.invalidate("Запрошен --recompute-intelligence.", INTELLIGENCE_STAGES)
            source_cache.invalidate("Запрошен --recompute-intelligence.", INTELLIGENCE_STAGES)
        if self.recompute_transformation:
            transformation_names = tuple(
                name for name in tracker.data.get("stages", {})
                if name in TRANSFORMATION_STAGES
                or name == "report"
                or any(name.startswith(f"{base}:") for base in TRANSFORMATION_STAGES)
            )
            tracker.invalidate("Запрошен --recompute-transformation.", transformation_names)
        if self.recompute_production_plan:
            production_names = tuple(
                name for name in tracker.data.get("stages", {})
                if name == "report" or any(name.startswith(f"{base}:") for base in PRODUCTION_PLAN_STAGES)
            )
            tracker.invalidate("Запрошен --recompute-production-plan.", production_names)
        source_data = self._source_stage(tracker, source_cache, work_directory / "source.json", source)
        metadata = self._cached(
            tracker, "metadata", work_directory / "metadata.json", {"source": source.id},
            lambda: prepare_media(source.path, work_directory), cache_tracker=source_cache,
        )
        if not metadata.get("audio_path"):
            empty_transcript: dict[str, Any] = {"source_id": source.id, "segments": [], "words": [], "empty_transcript": True}
            empty_audio: dict[str, Any] = {"energy_frames": [], "silence_intervals": [], "warning": "audio_track_unavailable"}
            empty_scenes: dict[str, Any] = {"enabled": False, "boundaries": [], "scene_boundary_count": 0}
            empty_visual: dict[str, Any] = {
                "enabled": False, "status": "skipped", "evidence_status": "evidence_unavailable",
                "reason": "audio_less_pipeline_short_circuit", "subject_keyframes": [],
            }
            empty_analysis_run_id = multimodal_analysis_run_id(
                source.id, empty_transcript, empty_audio, empty_scenes, empty_visual,
            )
            self._cached(
                tracker, "multimodal_timeline", work_directory / "multimodal_timeline.json",
                {
                    "source": source.id, "duration": metadata.get("duration"),
                    "analysis_run_id": empty_analysis_run_id, "analysis_version": MULTIMODAL_ANALYSIS_VERSION,
                },
                lambda: _write(
                    work_directory / "multimodal_timeline.json",
                    build_multimodal_timeline(
                        source_id=source.id,
                        source_duration_seconds=float(metadata.get("duration") or 0),
                        transcript=empty_transcript,
                        audio_features=empty_audio,
                        scenes=empty_scenes,
                        visual_analysis=empty_visual,
                        analysis_run_id=empty_analysis_run_id,
                    ),
                ),
                cache_tracker=source_cache,
                validator=lambda data: validate_multimodal_timeline(
                    data, expected_source_id=source.id, expected_analysis_run_id=empty_analysis_run_id,
                ),
            )
            return self._complete_run(self._finish_without_audio(tracker, source_data, metadata, work_directory, output_directory))
        transcript = self._cached(
            tracker, "transcription", work_directory / "transcript.json",
            {"source": source.id, "whisper": self.config.whisper_model, "language": self.config.language, "device": self.config.device},
            lambda: transcribe(Path(str(metadata["audio_path"])), source.id, float(metadata["duration"]), self.config, work_directory / "transcript.json"),
            cache_tracker=source_cache,
        )
        transcript_features = self._cached(
            tracker, "transcript_features", work_directory / "transcript_features.json",
            {"transcript": _hash(transcript), "settings": self.config.transcript_features},
            lambda: _write(work_directory / "transcript_features.json", analyse_transcript(transcript, self.config.transcript_features)),
            cache_tracker=source_cache,
        )
        audio_features = self._cached(
            tracker, "audio_features", work_directory / "audio_features.json",
            {"audio": str(metadata["audio_path"]), "settings": self.config.audio_analysis},
            lambda: _write(work_directory / "audio_features.json", analyse_audio(Path(str(metadata["audio_path"])), self.config.audio_analysis)),
            cache_tracker=source_cache,
        )
        scenes = self._cached(
            tracker, "scene_detection", work_directory / "scene_boundaries.json",
            {"source": source.id, "settings": self.config.scene_detection},
            lambda: _write(work_directory / "scene_boundaries.json", detect_scene_boundaries(source.path, float(metadata["duration"]), self.config.scene_detection)),
            cache_tracker=source_cache,
        )
        visual_analysis = self._cached(
            tracker, "visual_analysis", work_directory / "visual_analysis.json",
            {"source": source.id, "duration": metadata.get("duration"), "enabled": self.config.optional_visual_features, "model": self.config.ai.model},
            lambda: _write(work_directory / "visual_analysis.json", analyse_video_subjects(source.path, float(metadata.get("duration") or 0), self.config)),
            cache_tracker=source_cache,
        )
        multimodal_seed_analysis_id = multimodal_analysis_run_id(
            source.id, transcript, audio_features, scenes, visual_analysis,
        )
        multimodal_seed_timeline = self._cached(
            tracker, "multimodal_seed_timeline", work_directory / "multimodal_seed_timeline.json",
            {
                "source": source.id,
                "analysis_run_id": multimodal_seed_analysis_id,
                "analysis_version": MULTIMODAL_ANALYSIS_VERSION,
                "transcript": _hash(transcript),
                "audio_features": _hash(audio_features),
                "scenes": _hash(scenes),
                "visual_analysis": _hash(visual_analysis),
            },
            lambda: _write(
                work_directory / "multimodal_seed_timeline.json",
                build_multimodal_timeline(
                    source_id=source.id,
                    source_duration_seconds=float(metadata.get("duration") or 0),
                    transcript=transcript,
                    audio_features=audio_features,
                    scenes=scenes,
                    visual_analysis=visual_analysis,
                    analysis_run_id=multimodal_seed_analysis_id,
                ),
            ),
            cache_tracker=source_cache,
            validator=lambda data: validate_multimodal_timeline(
                data, expected_source_id=source.id, expected_analysis_run_id=multimodal_seed_analysis_id,
            ),
        )
        pre_vision_content_profile = self._cached(
            tracker, "pre_vision_content_profile", work_directory / "pre_vision_content_profile.json",
            {
                "source": source.id,
                "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features),
                "audio_features": _hash(audio_features),
                "scenes": _hash(scenes),
                "visual_analysis": _hash(visual_analysis),
                "strategy_version": self.config.content_understanding.strategy_version,
                "profile_schema_version": self.config.content_understanding.profile_schema_version,
                "enabled": self.config.content_understanding.enabled,
                "manual_override": self.config.content_understanding.manual_override,
                "content_profile_preset": self.config.product_flow.content_profile_preset,
                "profile_detection_min_confidence": self.config.content_understanding.profile_detection_min_confidence,
                "profile_contract_version": CONTENT_PROFILE_CONTRACT_VERSION,
                "profile_detector_version": CONTENT_PROFILE_DETECTOR_VERSION,
                "implementation_version": CONTENT_STRATEGY_VERSION,
            },
            lambda: _write(
                work_directory / "pre_vision_content_profile.json",
                {
                    **build_video_content_profile(
                        source_data, metadata, transcript, transcript_features, audio_features, scenes,
                        visual_analysis, self.config,
                    ),
                    "content_profile_preset": self.config.product_flow.content_profile_preset,
                },
            ),
            cache_tracker=source_cache,
            validator=lambda data: validate_video_content_profile(data, expected_source_id=source.id),
        )
        # Admission must remain acyclic: the local profile decides whether
        # PASS 1 is worth running, and cannot depend on its own output.
        deep_analysis = self._finalize_deep_analysis(pre_vision_content_profile, metadata)
        vision_provider = None
        if (
            self.config.vision.enabled
            and self.config.optional_visual_features
            and self.config.product_flow.processing_mode != "fast"
        ):
            try:
                vision_provider = get_vision_provider(self.config, self.mock_ai)
            except ClipEngineError as error:
                self.warnings.append(f"Vision Gateway uses local evidence fallback: {sanitize_api_error(error)}")
        reusable_vision = _rebind_vision_artifact(
            work_directory / "vision-observations.json",
            multimodal_seed_timeline,
            provider="mock" if self.mock_ai else self.config.ai.provider,
            model=self.config.ai.model,
            processing_mode=self.config.product_flow.processing_mode,
            prompt_version=self.config.vision.prompt_version,
            schema_version=self.config.vision.schema_version,
        )
        vision_analysis = self._cached(
            tracker, "vision_pass1", work_directory / "vision-observations.json",
            {
                "source": source.id,
                "timeline_analysis_run_id": multimodal_seed_timeline["analysis_run_id"],
                "timeline": _hash(multimodal_seed_timeline),
                "boundary_evidence_profile": "genre_neutral",
                "processing_mode": self.config.product_flow.processing_mode,
                "deep_analysis": deep_analysis.to_dict(),
                "effective_profile": _hash(pre_vision_content_profile.get("effective_profile", {})),
                "profile_detector_version": pre_vision_content_profile.get("detector_version"),
                "vision": self.config.vision,
                "provider": "mock" if self.mock_ai else self.config.ai.provider,
                "model": self.config.ai.model,
            },
            lambda: _write(
                work_directory / "vision-observations.json",
                reusable_vision if reusable_vision is not None else VisionGateway(
                    config=self.config,
                    cache_directory=self.root / "work" / "vision-cache",
                    provider=vision_provider,
                ).analyze_pass1(
                    source=source.path,
                    timeline=multimodal_seed_timeline,
                    content_type=str(
                        (pre_vision_content_profile.get("effective_profile") or {}).get("profile_id")
                        or (pre_vision_content_profile.get("effective_profile") or {}).get("format")
                        or "unknown"
                    ),
                ),
            ),
            cache_tracker=source_cache,
            validator=lambda data: validate_vision_artifact(data, multimodal_seed_timeline),
        )
        content_profile = self._cached(
            tracker, "video_content_profile", work_directory / "video_content_profile.json",
            {
                "pre_vision_profile": _hash(pre_vision_content_profile),
                "vision_pass1": _hash(vision_analysis),
                "profile_detector_version": CONTENT_PROFILE_DETECTOR_VERSION,
                "implementation_version": CONTENT_STRATEGY_VERSION,
            },
            lambda: _write(
                work_directory / "video_content_profile.json",
                {
                    **build_video_content_profile(
                        source_data, metadata, transcript, transcript_features, audio_features, scenes,
                        visual_analysis, self.config, vision_pass1=vision_analysis,
                    ),
                    "content_profile_preset": self.config.product_flow.content_profile_preset,
                },
            ),
            cache_tracker=source_cache,
            validator=lambda data: validate_video_content_profile(data, expected_source_id=source.id),
        )
        base_content_map = self._cached(
            tracker, "global_content_map_base", work_directory / "global_content_map.base.json",
            {
                "source": source.id,
                "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features),
                "audio_features": _hash(audio_features),
                "scenes": _hash(scenes),
                "visual_analysis": _hash(visual_analysis),
                "multimodal_timeline": _hash(multimodal_seed_timeline),
                # ContentMap and semantic boundaries are evidence-driven. A
                # manual profile override must not invalidate them.
                "profile_evidence": _hash({
                    "source_id": content_profile.get("source_id"),
                    "source_duration_seconds": content_profile.get("source_duration_seconds"),
                    "analysis_confidence": content_profile.get("analysis_confidence"),
                    "warnings": content_profile.get("warnings"),
                }),
                "content_map_settings": {
                    "strategy_version": self.config.content_understanding.strategy_version,
                    "content_map_schema_version": self.config.content_understanding.content_map_schema_version,
                    "story_unit_schema_version": self.config.content_understanding.story_unit_schema_version,
                    "chapter_pause_seconds": self.config.content_understanding.chapter_pause_seconds,
                    "max_chapter_seconds": self.config.content_understanding.max_chapter_seconds,
                    "min_story_unit_seconds": self.config.content_understanding.min_story_unit_seconds,
                    "target_story_unit_seconds": self.config.content_understanding.target_story_unit_seconds,
                    "max_story_unit_seconds": self.config.content_understanding.max_story_unit_seconds,
                },
                "implementation_version": CONTENT_STRATEGY_VERSION,
            },
            lambda: _write(
                work_directory / "global_content_map.base.json",
                build_global_content_map(
                    source_data, metadata, transcript, transcript_features, audio_features, scenes,
                    visual_analysis, content_profile, self.config, multimodal_seed_timeline,
                ),
            ),
            cache_tracker=source_cache,
        )
        candidate_seed_basis = self._cached(
            tracker, "candidate_seed_basis", work_directory / "candidate_seed_basis.json",
            {
                "content_map": _hash(base_content_map), "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features), "scenes": _hash(scenes),
                "multimodal_seed_timeline": _hash(multimodal_seed_timeline),
                "vision_pass1": _hash(vision_analysis),
                "candidate_generation": self.config.candidate_generation,
                "semantic_candidate_generation_version": SEMANTIC_CANDIDATE_GENERATION_VERSION,
                "multimodal_candidate_version": CANDIDATE_PROVENANCE_SCHEMA_VERSION,
                "boundary_settings": {
                    "schema_version": self.config.content_understanding.boundary_schema_version,
                    "max_head_padding_seconds": self.config.content_understanding.max_head_padding_seconds,
                    "target_head_padding_seconds": self.config.content_understanding.target_head_padding_seconds,
                    "min_tail_padding_seconds": self.config.content_understanding.min_tail_padding_seconds,
                    "target_tail_padding_seconds": self.config.content_understanding.target_tail_padding_seconds,
                    "max_tail_padding_seconds": self.config.content_understanding.max_tail_padding_seconds,
                    "max_semantic_extension_seconds": self.config.content_understanding.max_semantic_extension_seconds,
                    "continuation_risk_threshold": self.config.content_understanding.continuation_risk_threshold,
                },
            },
            lambda: _write_generated_candidates(
                work_directory / "candidate_seed_basis.json",
                generate_multimodal_candidates(
                    base_content_map, transcript, transcript_features, scenes,
                    multimodal_seed_timeline, vision_analysis, self.config,
                    semantic_generator=generate_semantic_candidates,
                ),
            ),
            cache_tracker=source_cache,
        )
        # Semantic audio is deliberately bounded by source-relative peaks and
        # the existing local shortlist.  This provisional pass invokes no AI
        # provider and uses the same candidate/boundary owner as the final pass.
        audio_profile_id = resolve_editorial_profile(content_profile).profile_id

        def build_semantic_audio_artifact() -> dict[str, Any]:
            provisional_candidates = [
                candidate_from_dict(item) for item in candidate_seed_basis.get("candidates", [])
            ]
            score_candidates(
                provisional_candidates, audio_features, scenes, self.config.scoring,
                min_duration_seconds=self.config.min_clip_duration,
                max_duration_seconds=self.config.max_clip_duration,
                visual_analysis=vision_analysis,
                transcript_features=transcript_features,
            )
            semantic_shortlist = shortlist(provisional_candidates, self.config.ai_reranking.shortlist_size)
            return analyse_semantic_audio(
                Path(str(metadata["audio_path"])), audio_features, semantic_shortlist,
                None, self.config.audio_analysis,
            )

        semantic_audio = self._cached(
            tracker, "audio_semantics", work_directory / "audio_semantic_events.json",
            {
                "audio_features": _hash(audio_features),
                "prepared_audio": str(metadata["audio_path"]),
                "candidate_seed_basis": _hash(candidate_seed_basis),
                "seed_timeline": _hash(multimodal_seed_timeline),
                "vision_pass1": _hash(vision_analysis),
                "candidate_generation": self.config.candidate_generation,
                "shortlist_size": self.config.ai_reranking.shortlist_size,
                "settings": self.config.audio_analysis,
                "analysis_version": AUDIO_SEMANTIC_ANALYSIS_VERSION,
            },
            lambda: _write(
                work_directory / "audio_semantic_events.json",
                build_semantic_audio_artifact(),
            ),
            cache_tracker=source_cache,
            validator=lambda data: validate_semantic_audio(data, self.config.audio_analysis),
        )
        multimodal_analysis_id = multimodal_analysis_run_id(
            source.id, transcript, audio_features, scenes, visual_analysis, semantic_audio,
        )
        multimodal_timeline = self._cached(
            tracker, "multimodal_timeline", work_directory / "multimodal_timeline.json",
            {
                "source": source.id,
                "analysis_run_id": multimodal_analysis_id,
                "analysis_version": MULTIMODAL_ANALYSIS_VERSION,
                "seed_timeline": _hash(multimodal_seed_timeline),
                "semantic_audio": _hash(semantic_audio),
            },
            lambda: _write(
                work_directory / "multimodal_timeline.json",
                build_multimodal_timeline(
                    source_id=source.id,
                    source_duration_seconds=float(metadata.get("duration") or 0),
                    transcript=transcript,
                    audio_features=audio_features,
                    scenes=scenes,
                    visual_analysis=visual_analysis,
                    semantic_audio=semantic_audio,
                    analysis_run_id=multimodal_analysis_id,
                ),
            ),
            cache_tracker=source_cache,
            validator=lambda data: validate_multimodal_timeline(
                data, expected_source_id=source.id, expected_analysis_run_id=multimodal_analysis_id,
            ),
        )
        content_map = self._cached(
            tracker, "global_content_map", work_directory / "global_content_map.json",
            {"base_content_map": _hash(base_content_map), "multimodal_timeline": _hash(multimodal_timeline)},
            lambda: _write(
                work_directory / "global_content_map.json",
                refresh_content_map_multimodal_evidence(base_content_map, transcript, multimodal_timeline),
            ),
            cache_tracker=source_cache,
        )
        story_units = self._cached(
            tracker, "story_units", work_directory / "story_units.json",
            {"content_map": _hash(content_map), "schema_version": self.config.content_understanding.story_unit_schema_version},
            lambda: _write(work_directory / "story_units.json", story_units_artifact(content_map, transcript)),
            cache_tracker=source_cache,
        )
        semantic_boundaries = self._cached(
            tracker, "semantic_boundaries", work_directory / "semantic_boundaries.json",
            {
                "candidate_seed_basis": _hash(candidate_seed_basis),
                "multimodal_timeline": _hash(multimodal_timeline),
            },
            lambda: _write_generated_candidates(
                work_directory / "semantic_boundaries.json",
                (
                    refresh_candidate_timeline_evidence(
                        [candidate_from_dict(item) for item in candidate_seed_basis.get("candidates", [])],
                        multimodal_timeline,
                    ),
                    int(candidate_seed_basis.get("candidates_before_deduplication") or 0),
                ),
            ),
            cache_tracker=source_cache,
        )
        raw_candidates = self._cached(
            tracker, "candidates_v2", work_directory / "candidates_v2.json",
            {
                "semantic_boundaries": _hash(semantic_boundaries),
                "audio_profile_id": audio_profile_id,
            },
            lambda: _write(
                work_directory / "candidates_v2.json",
                {
                    **semantic_boundaries,
                    "candidates": [
                        item.to_dict() for item in project_candidate_audio_evidence(
                            [candidate_from_dict(raw) for raw in semantic_boundaries.get("candidates", [])],
                            multimodal_timeline,
                            audio_profile_id,
                        )
                    ],
                },
            ),
            cache_tracker=source_cache,
        )
        # Compatibility artifact retained for existing users of the pre-1.6 cache layout.
        write_json(work_directory / "candidates.raw.json", raw_candidates)
        candidates = [candidate_from_dict(item) for item in raw_candidates.get("candidates", [])]
        local_data = self._cached(
            tracker, "local_scoring", work_directory / "candidates.local.json",
            {
                "candidates": _hash(raw_candidates), "settings": self.config.scoring,
                "duration_constraints": [self.config.min_clip_duration, self.config.max_clip_duration],
                "visual_analysis": _hash(visual_analysis), "speech_clarity_policy": SPEECH_CLARITY_POLICY_VERSION,
            },
            lambda: _write_candidates(
                work_directory / "candidates.local.json",
                score_candidates(
                    candidates, audio_features, scenes, self.config.scoring,
                    min_duration_seconds=self.config.min_clip_duration,
                    max_duration_seconds=self.config.max_clip_duration,
                    visual_analysis=visual_analysis,
                    transcript_features=transcript_features,
                ),
            ),
            cache_tracker=source_cache,
        )
        candidates = [candidate_from_dict(item) for item in local_data.get("candidates", [])]
        shortlist_data = self._cached(
            tracker, "shortlist", work_directory / "shortlist.json",
            {"candidates": _hash(local_data), "size": self.config.ai_reranking.shortlist_size},
            lambda: _write_candidates(work_directory / "shortlist.json", shortlist(candidates, self.config.ai_reranking.shortlist_size)),
            cache_tracker=source_cache,
        )
        short_candidates = [candidate_from_dict(item) for item in shortlist_data.get("candidates", [])]
        pass2_data = self._cached(
            tracker, "vision_pass2", work_directory / "shortlist.vision.json",
            {
                "shortlist": _hash(shortlist_data),
                "timeline": _hash(multimodal_timeline),
                "deep_analysis": deep_analysis.to_dict(),
                "vision": self.config.vision,
                "processing_mode": self.config.product_flow.processing_mode,
                "provider": "mock" if self.mock_ai else self.config.ai.provider,
                "model": self.config.ai.model,
            },
            lambda: _write_candidates(
                work_directory / "shortlist.vision.json",
                enrich_shortlist_with_pass2(
                    short_candidates,
                    source=source.path,
                    timeline=multimodal_timeline,
                    gateway=VisionGateway(
                        config=self.config,
                        cache_directory=self.root / "work" / "vision-cache",
                        provider=vision_provider,
                    ),
                    config=self.config,
                ),
            ),
            cache_tracker=source_cache,
        )
        short_candidates = [candidate_from_dict(item) for item in pass2_data.get("candidates", [])]
        pass2_by_id = {candidate.id: candidate.vision_pass2_evidence for candidate in short_candidates}
        for candidate in candidates:
            if candidate.id in pass2_by_id:
                candidate.vision_pass2_evidence = pass2_by_id[candidate.id]
        multimodal_scoring_data = self._cached(
            tracker, "multimodal_scoring", work_directory / "candidates.multimodal.json",
            {
                "local_scoring": _hash(local_data), "pass2": _hash(pass2_data),
                "scoring_contract": CANDIDATE_QUALITY_SCHEMA_VERSION, "settings": self.config.scoring,
                "visual_analysis": _hash(visual_analysis),
            },
            lambda: _write_candidates(
                work_directory / "candidates.multimodal.json",
                score_candidates(
                    candidates, audio_features, scenes, self.config.scoring,
                    min_duration_seconds=self.config.min_clip_duration,
                    max_duration_seconds=self.config.max_clip_duration,
                    visual_analysis=visual_analysis,
                    transcript_features=transcript_features,
                ),
            ),
            cache_tracker=source_cache,
        )
        candidates = [candidate_from_dict(item) for item in multimodal_scoring_data.get("candidates", [])]
        rescored_by_id = {candidate.id: candidate for candidate in candidates}
        short_candidates = [rescored_by_id[item.id] for item in short_candidates if item.id in rescored_by_id]
        ai_data = self._cached(
            tracker, "ai_ranking", work_directory / "ai_ranking.json",
            {
                "shortlist": _hash(pass2_data), "multimodal_scoring": _hash(multimodal_scoring_data),
                "ai": self.config.ai, "reranking": self.config.ai_reranking,
                "semantic_ai": {
                    "virality_enabled": self.config.virality.enabled,
                    "mode": self.config.virality.semantic_ai_mode,
                    "admission_version": "semantic-auto.2",
                    "payload_version": SEMANTIC_AI_PAYLOAD_VERSION,
                },
                "mock": self.mock_ai, "disabled": self.no_ai_rerank,
            },
            lambda: self._ai_rerank(candidates, short_candidates, transcript, work_directory / "ai_ranking.json"),
            cache_tracker=source_cache,
        )
        virality_profiles: dict[str, Any] = {}
        virality_ranking: dict[str, Any] = {}
        ranked_data = ai_data
        if self.config.virality.enabled:
            virality_profiles = self._cached(
                tracker, "virality_profiles", work_directory / "virality_profiles.json",
                {
                    "source": source.id, "candidates": _hash(multimodal_scoring_data), "content_map": _hash(content_map),
                    "transcript_features": _hash(transcript_features), "audio_features": _hash(audio_features),
                    "visual_features": _hash(visual_analysis), "content_profile": _hash(content_profile),
                    "semantic_result": _hash({
                        "ai": ai_data.get("ai", {}),
                        "ai_reranking_used": ai_data.get("ai_reranking_used", False),
                        "ai_fallback_used": ai_data.get("ai_fallback_used", False),
                    }),
                    "virality": {
                        "schema_version": self.config.virality.schema_version,
                        "strategy_version": self.config.virality.strategy_version,
                        "semantic_ai_mode": self.config.virality.semantic_ai_mode,
                        "provider": self.config.ai.provider,
                        "model": self.config.ai.model,
                        "prompt_version": "5B.deterministic.1",
                        "dead_zone_minimum_seconds": self.config.virality.dead_zone_minimum_seconds,
                    },
                },
                lambda: _write(
                    work_directory / "virality_profiles.json",
                    build_virality_assessments(
                        [candidate_from_dict(item) for item in multimodal_scoring_data.get("candidates", [])], content_map,
                        transcript_features, audio_features, visual_analysis, content_profile, self.config.virality,
                        semantic_result=ai_data,
                    ),
                ),
                cache_tracker=source_cache,
            )
            virality_ranking = self._cached(
                tracker, "virality_ranking", work_directory / "virality_ranking.json",
                {
                    "legacy_candidates": _hash(ai_data), "assessments": _hash(virality_profiles),
                    "content_profile": _hash(content_profile),
                    "virality": {
                        "schema_version": self.config.virality.schema_version,
                        "scoring_config_version": self.config.virality.scoring_config_version,
                        "strategy_version": self.config.virality.strategy_version,
                        "weights": self.config.virality.weights,
                        "strategy_weights": self.config.virality.strategy_weights,
                        "minimum_quality_score": self.config.virality.minimum_quality_score,
                        "minimum_publishability_score": self.config.virality.minimum_publishability_score,
                        "dead_zone_penalty_weight": self.config.virality.dead_zone_penalty_weight,
                    },
                },
                lambda: _write(
                    work_directory / "virality_ranking.json",
                    apply_virality_ranking(
                        [scored_from_dict(item) for item in ai_data.get("candidates", [])], virality_profiles,
                        self.config.virality, content_profile,
                    ),
                ),
                cache_tracker=source_cache,
            )
            ranked_data = virality_ranking
        scored = [scored_from_dict(item) for item in ranked_data.get("candidates", [])]
        self._prepare_recommendation_candidates(
            scored, visual_analysis, content_profile=content_profile, source=source_data,
        )
        production_feasibility = self._cached(
            tracker,
            "production_feasibility",
            work_directory / "production_feasibility.json",
            {
                "policy_version": PRODUCTION_FEASIBILITY_POLICY_VERSION,
                "source": source.id,
                "scored": _hash(ranked_data),
                "prepared_candidates": _hash([item.to_dict() for item in scored]),
                "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features),
                "audio_features": _hash(audio_features),
                "scenes": _hash(scenes),
                "multimodal_timeline": _hash(multimodal_timeline),
                "story_units": _hash(story_units),
                "transformation": self.config.transformation,
                "production": self.config.production,
                "production_render": self.config.production_render,
                "product_flow": self.config.product_flow,
            },
            lambda: self._production_feasibility(
                scored,
                source=source,
                source_data=source_data,
                metadata=metadata,
                transcript=transcript,
                transcript_features=transcript_features,
                audio_features=audio_features,
                scenes=scenes,
                multimodal_timeline=multimodal_timeline,
                story_units=story_units,
                content_map=content_map,
                path=work_directory / "production_feasibility.json",
            ),
            cache_tracker=source_cache,
            validator=validate_production_feasibility_artifact,
        )
        if self.config.virality.enabled:
            # The pre-feasibility virality artifact is intentionally neutral.
            # Apply structured profile weights only to the in-memory selection
            # pool after Phase-6 eligibility, boundary, and feasibility facts
            # are all available; final_selection persists the resulting rank.
            ranked_data = apply_profile_weighting_after_hard_gates(
                scored,
                virality_profiles,
                self.config.virality,
                content_profile,
                production_feasibility,
            )
            scored = [scored_from_dict(item) for item in ranked_data.get("candidates", [])]
        final_data = self._cached(
            tracker, "final_selection", work_directory / "final_selection.json",
            {
                "policy_version": "coverage-diversity-mmr-5B.2",
                "scored": _hash(ranked_data), "content_map": _hash(content_map),
                "prepared_candidates": _hash([item.to_dict() for item in scored]),
                "production_feasibility": _hash(production_feasibility),
                "threshold": self.config.score_threshold, "overlap": self.config.overlap_threshold,
                "distance": self.config.min_selected_clip_distance_seconds, "limit": self.config.ai_reranking.final_clip_count,
                "coverage": {
                    "version": self.config.content_understanding.coverage_selection_version,
                    "weights": self.config.content_understanding.coverage_weights,
                    "strong_story_unit_threshold": self.config.content_understanding.strong_story_unit_threshold,
                    "semantic_duplicate_threshold": self.config.content_understanding.semantic_duplicate_threshold,
                    "coverage_min_quality_score": self.config.content_understanding.coverage_min_quality_score,
                },
                "diversity": {
                    "schema_version": self.config.content_understanding.diversity_schema_version,
                    "config_version": self.config.content_understanding.diversity_config_version,
                    "lambda": self.config.content_understanding.diversity_lambda,
                    "semantic_duplicate_threshold": self.config.content_understanding.semantic_duplicate_threshold,
                },
                "virality": {
                    "enabled": self.config.virality.enabled,
                    "schema_version": self.config.virality.schema_version,
                    "minimum_quality_score": self.config.virality.minimum_quality_score,
                    "minimum_publishability_score": self.config.virality.minimum_publishability_score,
                },
                "editorial_intent": {
                    "value": self.config.content_understanding.editorial_intent,
                    "weight": self.config.content_understanding.editorial_intent_weight,
                },
                "publishable_story_expansion_version": PUBLISHABLE_STORY_EXPANSION_VERSION,
            },
            lambda: self._final_selection(
                scored,
                work_directory / "final_selection.json",
                content_map,
                production_feasibility,
                transcript=transcript,
                transcript_features=transcript_features,
                scenes=scenes,
            ),
            cache_tracker=source_cache,
        )
        coverage_map = self._cached(
            tracker, "coverage_map", work_directory / "coverage_map.json",
            {"final_selection": _hash(final_data), "schema_version": self.config.content_understanding.coverage_schema_version},
            lambda: _write(
                work_directory / "coverage_map.json",
                dict(final_data.get("coverage") or build_coverage_map(content_map, scored, [], self.config)),
            ),
            cache_tracker=source_cache,
        )
        clip_count_recommendation = self._cached(
            tracker, "clip_count_recommendation", work_directory / "clip_count_recommendation.json",
            {
                "profile": _hash(content_profile), "content_map": _hash(content_map),
                "requested_count": self.config.ai_reranking.final_clip_count,
                "strategy_version": self.config.content_understanding.strategy_version,
            },
            lambda: _write(
                work_directory / "clip_count_recommendation.json",
                recommend_clip_count(content_map, content_profile, self.config.ai_reranking.final_clip_count),
            ),
            cache_tracker=source_cache,
        )
        self.warnings.extend(str(value) for value in final_data.get("warnings", []) if str(value))
        selected_ids = set(final_data.get("selected_ids", []))
        final_scored = [scored_from_dict(item) for item in final_data.get("candidates", [])]
        write_json(
            work_directory / "candidates.scored.json",
            {
                "candidates": [item.to_dict() for item in final_scored], "ai": ai_data.get("ai", {}),
                "virality": {key: value for key, value in virality_ranking.items() if key != "candidates"},
            },
        )
        selected = [item for item in final_scored if item.candidate.id in selected_ids]
        if self.analysis_only:
            return self._complete_run(self._finish_analysis_only(
                tracker=tracker,
                source=source,
                source_data=source_data,
                metadata=metadata,
                transcript=transcript,
                transcript_features=transcript_features,
                audio_features=audio_features,
                scenes=scenes,
                visual_analysis=visual_analysis,
                multimodal_timeline=multimodal_timeline,
                vision_analysis=vision_analysis,
                vision_pass2=pass2_data,
                deep_analysis=deep_analysis,
                content_profile=content_profile,
                content_map=content_map,
                story_units=story_units,
                raw_candidates=raw_candidates,
                candidates=candidates,
                short_candidates=short_candidates,
                ai_data=ai_data,
                virality_profiles=virality_profiles,
                virality_ranking=virality_ranking,
                final_data=final_data,
                production_feasibility=production_feasibility,
                coverage_map=coverage_map,
                clip_count_recommendation=clip_count_recommendation,
                final_scored=final_scored,
                selected_ids=selected_ids,
                work_directory=work_directory,
                output_directory=output_directory,
            ))
        transformation = self._transform_selected(
            tracker, source_data, metadata, selected, transcript, transcript_features,
            audio_features, scenes, work_directory, output_directory,
        )
        self.warnings.extend(transformation.get("warnings", []))
        plan_analysis_fingerprint = _hash({
            "source": source.id,
            "profile": content_profile,
            "content_map": content_map,
            "coverage": coverage_map,
        })
        production = self._build_production_plans(
            tracker, transformation, work_directory, output_directory,
            self._production_plan_envelope_context(
                source,
                transcript,
                analysis_id=f"analysis-{plan_analysis_fingerprint[:16]}",
                analysis_fingerprint=plan_analysis_fingerprint,
            ),
        )
        tts = self._run_tts(tracker, production, work_directory, output_directory)
        audio = self._run_audio(
            tracker, production, tts, source, transcript, work_directory, output_directory,
            Path(str(metadata["audio_path"])) if metadata.get("audio_path") else None,
        )
        self._native_evidence_context = (final_scored, multimodal_timeline, story_units)
        production_render = self._run_production_render(
            tracker, production, audio, source, transcript, work_directory, output_directory, visual_analysis,
        )
        self.warnings.extend(production_render.get("warnings", []))
        production_is_primary = bool(
            self.config.production_render.enabled and not self.disable_production_render and not self.production_plan_only
        )
        if production_is_primary:
            tracker.skip("render", "Legacy render skipped; production render owns final results.")
            render_data = {"output_files": [], "warnings": [], "errors": [], "skipped": "production_render_primary"}
        else:
            render_data = (
                self._skip_render_for_production_plan(tracker, work_directory / "render.json")
                if self.production_plan_only
                else self._cached(
                    tracker, "render", work_directory / "render.json",
                    {"selected": [(item.candidate.id, item.score) for item in selected], "render": self.config.render_mode, "dimensions": [self.config.output_width, self.config.output_height], "encoder": self.config.encoder_preference},
                    lambda: self._render(source, transcript, selected, output_directory, work_directory / "render.json"),
                )
            )
        registry = primary_clip_results(production_render)
        quality_reports: list[dict[str, Any]] | None = None
        if production_is_primary:
            registry, quality_reports = self._persist_quality_reports(
                output_directory=output_directory,
                registry=registry,
                source_data=source_data,
                production=production,
                audio=audio,
                production_render=production_render,
                final_scored=final_scored,
                diversity_decision=final_data.get("diversity_decision") if isinstance(final_data, dict) else None,
            )
            production_render["quality_reports"] = quality_reports
        self._assert_current_run_results(registry, output_directory)
        outputs = result_paths(registry, output_directory) if production_is_primary else [
            Path(value) for value in render_data.get("output_files", []) if Path(value).is_file()
        ]
        candidate_flow = build_candidate_flow(
            final_scored, selected_ids, transformation, production, production_render,
        )
        terminal = build_terminal_state(
            self.config.ai_reranking.final_clip_count,
            outputs,
            candidate_flow,
            delivery_required=not self.production_plan_only and not self.tts_only and not self.audio_only,
            quality_reports=quality_reports,
        )
        tracker.start("terminal")
        tracker.finish(
            "terminal", terminal["status"],
            terminal.get("message") if terminal.get("status") == "failed" else None,
        )
        if terminal["status"] == "failed":
            self.errors.append(f"{terminal['error_code']}: {terminal['message']}")
        elif terminal["status"] == "completed_with_warnings":
            self.warnings.append("Часть отобранных кандидатов не дошла до финального рендера; см. candidate_flow в report.json.")
        self.warnings.extend(render_data.get("warnings", []))
        self.errors.extend(render_data.get("errors", []))
        if source.downloaded and self.config.delete_downloaded_source and outputs:
            try:
                source.path.unlink(missing_ok=True)
                self.warnings.append("Загруженный исходник удалён по настройке delete_downloaded_source.")
            except OSError as error:
                self.warnings.append(f"Не удалось удалить загруженный исходник: {error}")
        ai_usage = ai_data.get("ai", {})
        if ai_usage.get("api_errors"):
            self.errors.extend([f"{ai_usage.get('provider', 'ai')}: {message}" for message in ai_usage["api_errors"]])
        summary = intelligence_summary(
            transcript_features, audio_features, scenes, candidates, short_candidates,
            bool(ai_data.get("ai_reranking_used")), bool(ai_data.get("ai_fallback_used")),
            str(ai_data.get("selection_mode", "local")), int(raw_candidates.get("candidates_generated", len(candidates))),
        )
        report_path = output_directory / "report.json"
        tracker.start("report", _hash({"final": final_data, "coverage": coverage_map, "recommendation": clip_count_recommendation, "vision": vision_analysis, "render": render_data, "ai": ai_usage, "transformation": transformation, "production": production, "tts": tts, "audio": audio, "production_render": production_render}))
        tracker.finish("report")
        virality_report = (
            {
                "enabled": True,
                "schema_version": self.config.virality.schema_version,
                "scoring_config_version": self.config.virality.scoring_config_version,
                "strategy_version": self.config.virality.strategy_version,
                "strategy_id": virality_ranking.get("strategy_id", virality_profiles.get("strategy_id", "generic_fallback")),
                "profiles_ref": str(work_directory / "virality_profiles.json"),
                "ranking_ref": str(work_directory / "virality_ranking.json"),
                "cost": dict(virality_profiles.get("cost", {})),
                "semantic_ai": dict(virality_profiles.get("semantic_ai", {})),
                "cache": {
                    "profiles_hit": bool(tracker.data.get("stages", {}).get("virality_profiles", {}).get("cache_hit", False)),
                    "ranking_hit": bool(tracker.data.get("stages", {}).get("virality_ranking", {}).get("cache_hit", False)),
                },
                "candidate_count": len(virality_ranking.get("candidates", [])),
            }
            if self.config.virality.enabled else {"enabled": False, "status": "disabled"}
        )
        report = make_report(
            report_path, source_data, metadata, self.config, tracker.data, len(selected), len(candidates),
            [str(item) for item in outputs], self.warnings, self.errors, ai_usage,
            gpu_used=transcript.get("runtime", {}).get("device") == "cuda",
            nvenc_used=bool(render_data.get("nvenc_used", False)),
            clip_intelligence={
                **summary,
                "processing_times": {
                    "preparation" if name == "metadata" else name: stage.get("duration_seconds", 0)
                    for name, stage in tracker.data.get("stages", {}).items()
                    if name in INTELLIGENCE_STAGES or name in {"metadata", "transcription"}
                },
                "candidates": [item.to_dict() for item in final_scored],
                "virality": {key: value for key, value in virality_ranking.items() if key != "candidates"},
            },
            content_transformation=transformation,
            production_plan=production,
            tts=tts,
            audio=audio,
            production_render=production_render,
            candidate_flow=candidate_flow,
            terminal=terminal,
            content_understanding={
                "enabled": self.config.content_understanding.enabled,
                "profile": content_profile,
                "content_map": content_map,
                "multimodal_timeline_ref": str(work_directory / "multimodal_timeline.json"),
                "multimodal_diagnostics": multimodal_timeline.get("diagnostics", {}),
                "vision_observations_ref": str(work_directory / "vision-observations.json"),
                "vision": vision_analysis,
                "deep_analysis": deep_analysis.to_dict(),
                "story_units_ref": str(work_directory / "story_units.json"),
                "semantic_boundaries_ref": str(work_directory / "semantic_boundaries.json"),
                "production_feasibility_ref": str(work_directory / "production_feasibility.json"),
                "production_feasibility": final_data.get("production_feasibility", {}),
                "coverage_map_ref": str(work_directory / "coverage_map.json"),
                "clip_count_recommendation_ref": str(work_directory / "clip_count_recommendation.json"),
                "story_unit_count": len(story_units.get("story_units", [])),
                "coverage": coverage_map,
                "coverage_map": coverage_map,
                "clip_count_recommendation": clip_count_recommendation,
                "strategy_version": self.config.content_understanding.strategy_version,
                "fallback_used": bool(content_profile.get("fallback_used", True)),
            },
            virality=virality_report,
            primary_results=[item.to_dict() for item in registry],
            quality_gate=_quality_gate_summary(quality_reports),
            vision_ai_usage=collect_vision_usage(vision_analysis, pass2_data),
            run={
                "run_id": self.run_id,
                "source_id": source.id,
                "run_directory": str(output_directory),
                "started_at": self.started_at,
                "requested_clip_count": self.config.ai_reranking.final_clip_count,
                "project_id": self.project_id,
                "terminal_status": terminal["status"],
                "error_code": terminal.get("error_code"),
            },
        )
        manifest = write_run_manifest(
            output_directory / "manifest.json", run_id=self.run_id, source=source_data,
            started_at=self.started_at, requested_clip_count=self.config.ai_reranking.final_clip_count,
            production_render=production_render, results=registry, run_directory=output_directory, project_id=self.project_id,
            content_understanding={
                "content_profile_ref": str(work_directory / "video_content_profile.json"),
                "content_map_ref": str(work_directory / "global_content_map.json"),
                "multimodal_timeline_ref": str(work_directory / "multimodal_timeline.json"),
                "multimodal_analysis_run_id": multimodal_timeline.get("analysis_run_id"),
                "vision_observations_ref": str(work_directory / "vision-observations.json"),
                "story_units_ref": str(work_directory / "story_units.json"),
                "semantic_boundary_ref": str(work_directory / "semantic_boundaries.json"),
                "production_feasibility_ref": str(work_directory / "production_feasibility.json"),
                "coverage_map_ref": str(work_directory / "coverage_map.json"),
                "clip_count_recommendation_ref": str(work_directory / "clip_count_recommendation.json"),
                "strategy_version": self.config.content_understanding.strategy_version,
                "analysis_fingerprint": _hash({
                    "profile": content_profile, "map": content_map,
                    "multimodal_timeline": multimodal_timeline, "vision": vision_analysis, "coverage": coverage_map,
                }),
            },
            virality={
                **virality_report,
                "analysis_fingerprint": _hash({"profiles": virality_profiles, "ranking": virality_ranking}) if self.config.virality.enabled else None,
            },
            terminal=terminal,
            quality_gate=_quality_gate_summary(quality_reports),
        )
        report["run"]["manifest_path"] = str(output_directory / "manifest.json")
        report["run"]["finished_at"] = manifest["finished_at"]
        write_json(report_path, report)
        return self._complete_run(PipelineResult(
            work_directory, output_directory, report_path, len(outputs), outputs, self.warnings,
            terminal_status=terminal["status"], error_code=terminal.get("error_code"),
        ))

    def _complete_run(self, result: PipelineResult) -> PipelineResult:
        self._publish_completed_run_paths(result)
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        return result

    def _publish_run_paths(self, tracker: StageTracker, work_directory: Path, output_directory: Path) -> None:
        """Publish the real paths selected by the engine before any work starts.

        GUI callers use this identity-keyed record instead of mirroring the
        source-name slug algorithm.  Keeping it in ``state.json`` as well makes
        a run recoverable when the separate lookup record was not retained.
        """

        metadata = make_run_artifact_metadata(
            engine_root=self.root,
            run_id=self.run_id,
            project_id=self.project_id,
            work_directory=work_directory,
            output_directory=output_directory,
        )
        tracker.set_run_metadata(metadata)
        write_run_artifact_metadata(self.root, metadata)

    def _publish_completed_run_paths(self, result: PipelineResult) -> None:
        """Update the engine metadata with terminal artifacts and report paths."""

        metadata = make_run_artifact_metadata(
            engine_root=self.root,
            run_id=self.run_id,
            project_id=self.project_id,
            work_directory=result.work_directory,
            output_directory=result.output_directory,
            report_path=result.report_path,
            analysis_artifact_path=result.analysis_path,
            draft_artifact_path=result.draft_path,
            manifest_path=(result.output_directory / "manifest.json") if (result.output_directory / "manifest.json").is_file() else None,
            output_files=result.output_files,
            terminal_status=result.terminal_status,
        )
        write_run_artifact_metadata(self.root, metadata)
        state_path = result.work_directory / "state.json"
        state = read_json(state_path, {})
        if isinstance(state, dict):
            state["run"] = metadata
            write_json(state_path, state)
        report = read_json(result.report_path, {})
        if isinstance(report, dict):
            run = report.setdefault("run", {})
            if isinstance(run, dict):
                run["artifact_metadata_path"] = metadata["metadata_path"]
                run["artifact_paths"] = dict(metadata["paths"])
                write_json(result.report_path, report)

    def _finish_analysis_only(
        self,
        *,
        tracker: StageTracker,
        source: Source,
        source_data: dict[str, Any],
        metadata: dict[str, Any],
        transcript: dict[str, Any],
        transcript_features: dict[str, Any],
        audio_features: dict[str, Any],
        scenes: dict[str, Any],
        visual_analysis: dict[str, Any],
        multimodal_timeline: dict[str, Any],
        vision_analysis: dict[str, Any],
        vision_pass2: dict[str, Any],
        deep_analysis: DeepAnalysisDecision,
        content_profile: dict[str, Any],
        content_map: dict[str, Any],
        story_units: dict[str, Any],
        raw_candidates: dict[str, Any],
        candidates: list[Candidate],
        short_candidates: list[Candidate],
        ai_data: dict[str, Any],
        virality_profiles: dict[str, Any],
        virality_ranking: dict[str, Any],
        final_data: dict[str, Any],
        production_feasibility: dict[str, Any],
        coverage_map: dict[str, Any],
        clip_count_recommendation: dict[str, Any],
        final_scored: list[Any],
        selected_ids: set[str],
        work_directory: Path,
        output_directory: Path,
    ) -> PipelineResult:
        """Persist a reusable intelligence result without starting delivery work."""

        scored_records = [item.to_dict() for item in final_scored]
        candidate_data = {
            "candidates": scored_records,
            "ai": dict(ai_data.get("ai") or {}),
            "virality": {key: value for key, value in virality_ranking.items() if key != "candidates"},
        }
        analysis_fingerprint = _hash({
            "engine": INTELLIGENCE_ENGINE_VERSION,
            "source": source.id,
            "profile": content_profile,
            "content_map": content_map,
            "multimodal_timeline": multimodal_timeline,
            "vision": vision_analysis,
            "ranking": final_data,
            "coverage": coverage_map,
            "recommendation": clip_count_recommendation,
        })
        analysis_id = f"analysis-{analysis_fingerprint[:16]}"
        artifact_path = output_directory / "analysis.json"
        snapshot_directory = output_directory / "analysis-snapshot"
        producer = {
            "name": "Pipeline._finish_analysis_only",
            "version": INTELLIGENCE_ENGINE_VERSION,
            "analysis_run_id": self.run_id,
        }
        snapshot_objects: dict[str, dict[str, Any]] = {
            "source": source_data,
            "metadata": metadata,
            "transcript": transcript,
            "transcript_features": transcript_features,
            "audio_features": audio_features,
            "scene_boundaries": scenes,
            "visual_analysis": visual_analysis,
            "multimodal_timeline": multimodal_timeline,
            "vision_observations": vision_analysis,
            "vision_pass2": vision_pass2,
            "content_profile": content_profile,
            "content_map": content_map,
            "story_units": story_units,
            "semantic_boundaries": raw_candidates,
            "production_feasibility": production_feasibility,
            "coverage_map": coverage_map,
            "clip_count_recommendation": clip_count_recommendation,
            "candidate_data": candidate_data,
            "final_selection": final_data,
        }
        if self.config.virality.enabled:
            snapshot_objects["virality_profiles"] = virality_profiles
            snapshot_objects["virality_ranking"] = virality_ranking
        references, reference_integrity = _write_analysis_snapshot(
            snapshot_directory, snapshot_objects, producer,
        )
        candidate_data_path = Path(references["candidate_data"])
        review_candidates = [
            candidate_review_payload(record, selected_ids, content_profile, source_data)
            for record in scored_records
        ]
        artifact = new_analysis_artifact(
            analysis_id=analysis_id,
            project_id=self.project_id,
            source=dict(source_data),
            source_fingerprint=source.id,
            analysis_fingerprint=analysis_fingerprint,
            work_directory=str(work_directory),
            candidate_data_ref=str(candidate_data_path),
            references=references,
            candidates=review_candidates,
            recommendation={
                "selected_candidate_ids": [item.candidate.id for item in final_scored if item.candidate.id in selected_ids],
                "clip_count": clip_count_recommendation,
                "coverage": coverage_map,
                "production_feasibility": final_data.get("production_feasibility", {}),
            },
            summary={
                "candidate_count": len(final_scored),
                "recommended_count": len(selected_ids),
                "source_duration_seconds": metadata.get("duration"),
                "content_type": (content_profile.get("effective_profile") or {}).get("profile_id"),
                "potential_counts": potential_counts(review_candidates),
            },
            content_profile={
                "detected_content_type": content_profile.get("detected_content_type"),
                "content_type_confidence": content_profile.get("content_type_confidence"),
                "strategy_id": content_profile.get("strategy_id"),
                "detected_profile": content_profile.get("detected_profile"),
                "effective_profile": content_profile.get("effective_profile"),
                "manual_override": content_profile.get("manual_override"),
                "content_profile_preset": content_profile.get("content_profile_preset", "auto"),
                "contract_version": content_profile.get("contract_version"),
                "detector_version": content_profile.get("detector_version"),
                "requested_mode": content_profile.get("requested_mode"),
                "requested_profile_id": content_profile.get("requested_profile_id"),
                "effective_profile_reason": content_profile.get("effective_profile_reason"),
            },
            duration_seconds=float(metadata["duration"]) if metadata.get("duration") is not None else None,
            analysis_run_id=self.run_id,
            snapshot_directory=str(snapshot_directory),
            reference_integrity=reference_integrity,
            producer=producer,
            candidate_count=len(review_candidates),
            recommended_count={
                "min": int(clip_count_recommendation.get("estimated_publishable_clip_range", {}).get("min", len(selected_ids)) or 0),
                "max": int(clip_count_recommendation.get("estimated_publishable_clip_range", {}).get("max", len(selected_ids)) or 0),
                "default": len(selected_ids),
            },
            warnings=list(self.warnings),
        )
        tracker.start("analysis_artifact", analysis_fingerprint)
        artifact.write_with_integrity(artifact_path)
        tracker.finish("analysis_artifact")
        for stage in (
            *TRANSFORMATION_STAGES, *PRODUCTION_PLAN_STAGES, *TTS_STAGES,
            *AUDIO_COMPOSITION_STAGES, *PRODUCTION_RENDER_STAGES, "render",
        ):
            tracker.skip(stage, "Analysis-only run: delivery stage was not started.")
        terminal: dict[str, Any] = {
            "status": "analysis_ready",
            "error_code": None,
            "message": "Analysis completed. Select candidates before rendering.",
            "analysis_id": analysis_id,
        }
        tracker.start("terminal")
        tracker.finish("terminal", "analysis_ready")
        summary = intelligence_summary(
            transcript_features, audio_features, scenes, candidates, short_candidates,
            bool(ai_data.get("ai_reranking_used")), bool(ai_data.get("ai_fallback_used")),
            str(ai_data.get("selection_mode", "local")), int(raw_candidates.get("candidates_generated", len(candidates))),
        )
        ai_usage = ai_data.get("ai", {})
        virality_report = {
            "enabled": self.config.virality.enabled,
            "profiles_ref": str(work_directory / "virality_profiles.json") if self.config.virality.enabled else None,
            "ranking_ref": str(work_directory / "virality_ranking.json") if self.config.virality.enabled else None,
            "cost": dict(virality_profiles.get("cost", {})),
            "semantic_ai": dict(virality_profiles.get("semantic_ai", {})),
            "candidate_count": len(virality_ranking.get("candidates", [])),
        }
        report_path = output_directory / "report.json"
        tracker.start("report", _hash({"analysis": analysis_fingerprint, "terminal": terminal}))
        tracker.finish("report")
        make_report(
            report_path, source_data, metadata, self.config, tracker.data, len(selected_ids), len(candidates),
            [], self.warnings, self.errors, ai_usage,
            gpu_used=transcript.get("runtime", {}).get("device") == "cuda",
            nvenc_used=False,
            clip_intelligence={
                **summary,
                "candidates": scored_records,
                "analysis_artifact_ref": str(artifact_path),
                "analysis_id": analysis_id,
            },
            content_understanding={
                "enabled": self.config.content_understanding.enabled,
                "profile": content_profile,
                "content_map": content_map,
                "multimodal_timeline_ref": str(work_directory / "multimodal_timeline.json"),
                "multimodal_diagnostics": multimodal_timeline.get("diagnostics", {}),
                "vision_observations_ref": str(work_directory / "vision-observations.json"),
                "vision_pass2_ref": str(work_directory / "shortlist.vision.json"),
                "vision": vision_analysis,
                "deep_analysis": deep_analysis.to_dict(),
                "story_units_ref": str(work_directory / "story_units.json"),
                "semantic_boundaries_ref": str(work_directory / "semantic_boundaries.json"),
                "production_feasibility_ref": str(work_directory / "production_feasibility.json"),
                "production_feasibility": final_data.get("production_feasibility", {}),
                "coverage_map_ref": str(work_directory / "coverage_map.json"),
                "clip_count_recommendation_ref": str(work_directory / "clip_count_recommendation.json"),
                "story_unit_count": len(story_units.get("story_units", [])),
                "coverage_map": coverage_map,
                "clip_count_recommendation": clip_count_recommendation,
                "strategy_version": self.config.content_understanding.strategy_version,
            },
            virality=virality_report,
            terminal=terminal,
            vision_ai_usage=collect_vision_usage(vision_analysis, vision_pass2),
            run={
                "run_id": self.run_id,
                "project_id": self.project_id,
                "source_id": source.id,
                "run_directory": str(output_directory),
                "started_at": self.started_at,
                "analysis_id": analysis_id,
                "analysis_artifact_path": str(artifact_path),
                "analysis_fingerprint": analysis_fingerprint,
                "terminal_status": "analysis_ready",
            },
        )
        return PipelineResult(
            work_directory=work_directory,
            output_directory=output_directory,
            report_path=report_path,
            selected_clips=len(selected_ids),
            output_files=[],
            warnings=self.warnings,
            terminal_status="analysis_ready",
            analysis_path=artifact_path,
            analysis_id=analysis_id,
        )

    def _run_draft_preview(
        self,
        *,
        tracker: StageTracker,
        source: Source,
        work_directory: Path,
        output_directory: Path,
    ) -> PipelineResult:
        """Deliver exactly the review selection without re-running intelligence."""

        assert self.analysis_artifact_path is not None
        try:
            analysis = AnalysisArtifact.read_verified(self.analysis_artifact_path)
        except AnalysisArtifactError as error:
            raise ClipEngineError(f"Analysis artifact cannot be used: {error}") from error
        self.warnings.extend(warning for warning in analysis.warnings if warning not in self.warnings)
        if analysis.project_id and self.project_id and analysis.project_id != self.project_id:
            raise ClipEngineError("Analysis artifact belongs to a different project.")
        if self.expected_analysis_id and analysis.analysis_id != self.expected_analysis_id:
            raise ClipEngineError("Analysis ID does not match the supplied artifact.")
        if self.expected_analysis_fingerprint and analysis.analysis_fingerprint != self.expected_analysis_fingerprint:
            raise ClipEngineError("Analysis fingerprint does not match the supplied artifact.")
        if analysis.source_fingerprint != source.id:
            raise ClipEngineError("The selected analysis belongs to a different source file.")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ClipEngineError("Selected candidate IDs must not contain duplicates.")
        if not self.selected_candidate_ids:
            raise ClipEngineError("Render requires at least one explicit selected candidate ID.")
        if analysis.schema_version == "1.0" and not Path(analysis.work_directory).resolve().is_relative_to(
            (self.root / "work").resolve()
        ):
            raise ClipEngineError("Legacy analysis work reference is outside this engine workspace.")

        def load_reference(name: str) -> dict[str, Any]:
            try:
                return analysis.load_reference(name)
            except AnalysisArtifactError as error:
                raise ClipEngineError(f"Analysis artifact cannot be used: {error}") from error

        candidate_data = load_reference("candidate_data")
        final_scored = [
            scored_from_dict(item) for item in candidate_data.get("candidates", []) if isinstance(item, dict)
        ]
        candidates_by_id = {item.candidate.id: item for item in final_scored}
        unknown = [candidate_id for candidate_id in self.selected_candidate_ids if candidate_id not in candidates_by_id]
        if unknown:
            raise ClipEngineError(f"Selected candidates are not present in this analysis: {', '.join(unknown)}.")
        selected = [candidates_by_id[candidate_id] for candidate_id in self.selected_candidate_ids]
        source_data = load_reference("source")
        metadata = load_reference("metadata")
        transcript = load_reference("transcript")
        transcript_features = load_reference("transcript_features")
        audio_features = load_reference("audio_features")
        scenes = load_reference("scene_boundaries")
        visual_analysis = load_reference("visual_analysis")
        content_profile = load_reference("content_profile")
        validate_video_content_profile(content_profile, expected_source_id=str(source_data.get("id") or ""))
        content_map = load_reference("content_map")
        coverage_map = load_reference("coverage_map")
        clip_count_recommendation = load_reference("clip_count_recommendation")
        story_units = load_reference("story_units")
        multimodal_timeline = load_reference("multimodal_timeline")
        # A saved review override belongs to one candidate.  Do not let one
        # stale/invalid edit discard the otherwise independent draft batch.
        # Keep the original range for a rejected edit so the durable progress
        # contract remains well-formed, then run transformation only for the
        # candidates whose override is usable.
        selected, boundary_failures = self._apply_boundary_overrides(
            selected, metadata, transcript_features, scenes, tracker,
        )
        selected, preflight_failures = self._preflight_selected_candidates(
            selected, visual_analysis, tracker, content_profile=content_profile, source=source_data,
        )
        candidate_failures = {**boundary_failures, **preflight_failures}
        composition_vision = self._ensure_draft_composition_evidence_isolated(
            [item for item in selected if item.candidate.id not in candidate_failures],
            source=source,
            timeline=multimodal_timeline,
            analysis=analysis,
            work_directory=work_directory,
            tracker=tracker,
        )
        self._write_draft_progress(
            output_directory=output_directory,
            analysis=analysis,
            source=source,
            selected=selected,
        )

        tracker.start("analysis_handoff", analysis.analysis_fingerprint, cache_hit=True)
        tracker.finish("analysis_handoff")
        transformable = [
            item for item in selected if item.candidate.id not in candidate_failures
        ]
        transformation = self._transform_selected(
            tracker, source_data, metadata, transformable, transcript, transcript_features,
            audio_features, scenes, work_directory, output_directory,
        )
        if candidate_failures:
            # ``_finish_draft_preview`` owns the item-level draft report.  Give
            # it an ordinary failed transformation outcome for every rejected
            # boundary so it writes the same retryable record/progress shape as
            # any other candidate failure.
            by_candidate = {
                str(item.get("candidate_id") or ""): item
                for item in transformation.get("items", []) if isinstance(item, dict)
            }
            transformed_items: list[dict[str, Any]] = []
            for item in selected:
                candidate_id = item.candidate.id
                if candidate_id in candidate_failures:
                    transformed_items.append({
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "error": candidate_failures[candidate_id],
                        "stage": (
                            f"boundary_override:{candidate_id}"
                            if candidate_id in boundary_failures
                            else f"candidate_preflight:{candidate_id}"
                        ),
                    })
                elif candidate_id in by_candidate:
                    transformed_items.append(by_candidate[candidate_id])
            transformation = dict(transformation)
            transformation["items"] = transformed_items
        self.warnings.extend(transformation.get("warnings", []))
        production = self._build_production_plans(
            tracker,
            transformation,
            work_directory,
            output_directory,
            self._production_plan_envelope_context(
                source,
                transcript,
                analysis_id=analysis.analysis_id,
                analysis_fingerprint=analysis.analysis_fingerprint,
            ),
        )
        return self._finish_draft_preview(
            tracker=tracker,
            analysis=analysis,
            source=source,
            source_data=source_data,
            metadata=metadata,
            transcript=transcript,
            content_profile=content_profile,
            content_map=content_map,
            story_units=story_units,
            multimodal_timeline=multimodal_timeline,
            visual_analysis=visual_analysis,
            coverage_map=coverage_map,
            clip_count_recommendation=clip_count_recommendation,
            final_scored=final_scored,
            selected=selected,
            composition_vision=composition_vision,
            transformation=transformation,
            production=production,
            work_directory=work_directory,
            output_directory=output_directory,
        )

    def _ensure_draft_composition_evidence_isolated(
        self,
        selected: list[Any],
        *,
        source: Source,
        timeline: dict[str, Any],
        analysis: AnalysisArtifact,
        work_directory: Path,
        tracker: StageTracker,
    ) -> dict[str, dict[str, Any]]:
        """Keep a candidate-only Vision failure local to that candidate."""

        outcomes: dict[str, dict[str, Any]] = {}
        for scored in selected:
            candidate = scored.candidate
            stage_name = f"draft_composition_vision:{candidate.id}"
            try:
                outcomes.update(self._ensure_draft_composition_evidence(
                    [scored],
                    source=source,
                    timeline=timeline,
                    analysis=analysis,
                    work_directory=work_directory,
                    tracker=tracker,
                ))
            except Exception as error:
                safe = sanitize_api_error(error)
                tracker.finish(stage_name, "warning", safe)
                self.warnings.append(
                    f"Candidate composition Vision unavailable for {candidate.id}: {safe}"
                )
                outcomes[candidate.id] = {
                    "schema_version": DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION,
                    "status": "skipped",
                    "reason": safe,
                    "cache_hit": False,
                    "usable_composition_evidence": False,
                    "artifact_ref": None,
                    "model": DRAFT_COMPOSITION_PASS2_MODEL,
                    "source_range": {
                        "start_seconds": candidate.start,
                        "end_seconds": candidate.end,
                    },
                }
        return outcomes

    def _ensure_draft_composition_evidence(
        self,
        selected: list[Any],
        *,
        source: Source,
        timeline: dict[str, Any],
        analysis: AnalysisArtifact,
        work_directory: Path,
        tracker: StageTracker,
    ) -> dict[str, dict[str, Any]]:
        """Fill a selected candidate's composition gap without changing analysis.

        The durable candidate artifact is keyed only by source/analysis/range and
        Vision contract inputs. Caption, style, crop, Preview, and Final settings
        are absent from the key, so their rerenders cannot trigger another call.
        """

        outcomes: dict[str, dict[str, Any]] = {}
        candidate_pass2_admitted = (
            self.config.vision.enabled
            and self.config.product_flow.processing_mode != "fast"
            and self.config.product_flow.deep_analysis_requested != "off"
        )
        target_config = replace(
            self.config,
            ai=replace(self.config.ai, model=DRAFT_COMPOSITION_PASS2_MODEL),
            # AUTO may intentionally skip full-source PASS 1, while a selected
            # Draft still admits one candidate-bounded PASS 2.  This copied
            # config is private to the candidate gateway and never enables the
            # source-level Vision switch on ``self.config``.
            optional_visual_features=candidate_pass2_admitted,
        )
        gateway: VisionGateway | None = None
        provider_label = "mock" if self.mock_ai else target_config.ai.provider
        cache_directory = work_directory / "candidate-vision-pass2"
        for scored in selected:
            candidate = scored.candidate
            candidate_data = candidate.to_dict()
            stage_name = f"draft_composition_vision:{candidate.id}"
            if has_usable_composition_evidence(candidate_data, timeline):
                tracker.skip(stage_name, "Selected candidate already has usable composition evidence.")
                outcomes[candidate.id] = {
                    "schema_version": DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION,
                    "status": "not_required",
                    "cache_hit": True,
                    "usable_composition_evidence": True,
                    "artifact_ref": (
                        candidate.vision_pass2_evidence.get("artifact_ref")
                        if isinstance(candidate.vision_pass2_evidence, dict) else None
                    ),
                    "model": DRAFT_COMPOSITION_PASS2_MODEL,
                    "source_range": {"start_seconds": candidate.start, "end_seconds": candidate.end},
                }
                continue

            if not candidate_pass2_admitted:
                reason = (
                    "vision_explicitly_off"
                    if target_config.product_flow.deep_analysis_requested == "off"
                    else "candidate_vision_not_admitted"
                )
                tracker.skip(stage_name, reason)
                outcomes[candidate.id] = {
                    "schema_version": DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION,
                    "status": "skipped",
                    "reason": reason,
                    "cache_hit": False,
                    "usable_composition_evidence": False,
                    "artifact_ref": None,
                    "model": DRAFT_COMPOSITION_PASS2_MODEL,
                    "source_range": {
                        "start_seconds": candidate.start,
                        "end_seconds": candidate.end,
                    },
                }
                continue

            anchors = candidate_pass2_anchors(candidate)
            bounded_timeline = build_candidate_bounded_pass2_timeline(
                timeline,
                candidate_id=candidate.id,
                window_start=candidate.start,
                window_end=candidate.end,
                anchors=anchors,
                max_frames=int(target_config.vision.pass2_max_frames),
            )
            request = build_pass2_request(
                candidate_id=candidate.id,
                window_start=candidate.start,
                window_end=candidate.end,
                anchors=anchors,
                timeline=bounded_timeline,
                max_frames=int(target_config.vision.pass2_max_frames),
            )
            cache_key = _hash({
                "schema_version": DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION,
                "source_id": source.id,
                "analysis_id": analysis.analysis_id,
                "analysis_run_id": timeline["analysis_run_id"],
                "analysis_artifact_sha256": analysis.verified_sha256,
                "candidate_id": candidate.id,
                "source_range": [round(candidate.start, 3), round(candidate.end, 3)],
                "request": request,
                "timeline_inputs": timeline["input_fingerprints"],
                "provider": provider_label,
                "model": DRAFT_COMPOSITION_PASS2_MODEL,
                "prompt_version": target_config.vision.pass2_prompt_version,
                "vision_schema_version": target_config.vision.schema_version,
                "frame_width": target_config.vision.frame_width,
            })
            artifact_path = cache_directory / (
                f"{safe_name(candidate.id, 'candidate')}-{cache_key[:20]}.json"
            )
            expected = {
                "cache_key": cache_key,
                "source_id": source.id,
                "analysis_id": analysis.analysis_id,
                "analysis_run_id": str(timeline["analysis_run_id"]),
                "analysis_artifact_sha256": analysis.verified_sha256,
                "candidate_id": candidate.id,
                "source_range": {
                    "start_seconds": round(candidate.start, 3),
                    "end_seconds": round(candidate.end, 3),
                },
                "provider": provider_label,
                "model": DRAFT_COMPOSITION_PASS2_MODEL,
                "prompt_version": target_config.vision.pass2_prompt_version,
                "vision_schema_version": target_config.vision.schema_version,
                "request": request,
            }
            cached = _read_candidate_composition_pass2(
                artifact_path, expected=expected, timeline=bounded_timeline,
            )
            if cached is not None:
                tracker.start(stage_name, cache_key, cache_hit=True)
                candidate.vision_pass2_evidence = dict(cached["vision_pass2_evidence"])
                usable = has_usable_composition_evidence(candidate.to_dict(), bounded_timeline)
                tracker.finish(stage_name, "completed" if usable else "warning")
                outcomes[candidate.id] = _candidate_composition_pass2_summary(
                    cached, artifact_path=artifact_path, cache_hit=True, usable=usable,
                )
                continue

            tracker.start(stage_name, cache_key, cache_hit=False)
            if gateway is None:
                provider = None
                if candidate_pass2_admitted:
                    try:
                        provider = get_vision_provider(target_config, self.mock_ai)
                    except ClipEngineError as error:
                        self.warnings.append(
                            "Candidate composition Vision uses local fallback: "
                            f"{sanitize_api_error(error)}"
                        )
                gateway = VisionGateway(
                    config=target_config,
                    cache_directory=self.root / "work" / "vision-cache",
                    provider=provider,
                )
            try:
                result = gateway.analyze_pass2(
                    source=source.path, timeline=bounded_timeline, request=request,
                )
                evidence = {
                    "schema_version": PASS2_EVIDENCE_SCHEMA_VERSION,
                    "status": str(result.get("status") or "completed"),
                    "reason": None,
                    "result": result,
                }
            except Exception as error:
                evidence = {
                    "schema_version": PASS2_EVIDENCE_SCHEMA_VERSION,
                    "status": "skipped",
                    "reason": sanitize_api_error(error),
                    "result": None,
                }
            lineage_id = f"candidate-composition-pass2-{cache_key[:24]}"
            evidence["lineage_ref"] = lineage_id
            evidence["artifact_ref"] = str(artifact_path)
            artifact = {
                "schema_version": DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION,
                "cache_key": cache_key,
                "lineage": {
                    **expected,
                    "lineage_id": lineage_id,
                    "trigger": "draft_candidate_composition_evidence_gap",
                    "analysis_artifact_ref": str(self.analysis_artifact_path),
                    "created_at": utc_now(),
                    "bounded_frame_count": len(request["frames"]),
                    "analysis_snapshot_mutated": False,
                },
                "vision_pass2_evidence": evidence,
            }
            write_json(artifact_path, artifact)
            candidate.vision_pass2_evidence = evidence
            usable = has_usable_composition_evidence(candidate.to_dict(), bounded_timeline)
            tracker.finish(stage_name, "completed" if usable else "warning")
            outcomes[candidate.id] = _candidate_composition_pass2_summary(
                artifact, artifact_path=artifact_path, cache_hit=False, usable=usable,
            )
        return outcomes

    def _preflight_selected_candidates(
        self,
        selected: list[Any],
        visual_analysis: dict[str, Any],
        tracker: StageTracker,
        *,
        content_profile: dict[str, Any],
        source: dict[str, Any],
    ) -> tuple[list[Any], dict[str, str]]:
        """Recheck production eligibility and BoundaryDecision before transform."""

        failures: dict[str, str] = {}
        for scored in selected:
            candidate = scored.candidate
            stage_name = f"candidate_preflight:{candidate.id}"
            fingerprint = _hash({
                "candidate": candidate.to_dict(),
                "minimum_duration": self.config.candidate_generation.min_duration_seconds,
                "maximum_duration": self.config.candidate_generation.max_duration_seconds,
                "quality_config": self.config.scoring.candidate_quality_config_version,
            })
            tracker.start(stage_name, fingerprint)
            decision = resolve_eligibility_decision(
                candidate,
                candidate.feature_vector,
                config_version=self.config.scoring.candidate_quality_config_version,
                min_duration_seconds=self.config.candidate_generation.min_duration_seconds,
                max_duration_seconds=self.config.candidate_generation.max_duration_seconds,
                visual_analysis=visual_analysis,
                cached_eligibility=dict(scored.virality.get("eligibility") or {}),
            )
            candidate.eligibility_decision = decision
            editorial = evaluate_editorial_candidate(
                candidate,
                content_profile,
                score=float(scored.score),
                production_feasibility=dict(scored.selection_diagnostics.get("production_feasibility") or {}),
                source=source,
            )
            candidate.editorial_decision = editorial
            if not editorial.selectable:
                codes = ",".join(editorial.hard_blockers) or "UNKNOWN"
                message = f"CANDIDATE_NOT_PRODUCTION_ELIGIBLE: {codes}."
                failures[candidate.id] = message
                tracker.finish(stage_name, "failed", message)
                continue
            boundary = ensure_candidate_boundary_decision(candidate)
            if boundary is None:
                message = "BOUNDARY_DECISION_REQUIRED: complete validated boundary evidence is unavailable."
                failures[candidate.id] = message
                tracker.finish(stage_name, "failed", message)
                continue
            tracker.finish(stage_name)
        return selected, failures

    def _apply_boundary_overrides(
        self,
        selected: list[Any],
        metadata: dict[str, Any],
        transcript_features: dict[str, Any],
        scenes: dict[str, Any],
        tracker: StageTracker,
    ) -> tuple[list[Any], dict[str, str]]:
        """Apply cached review edits without turning one invalid edit into a batch failure.

        The returned selection keeps its original IDs/order.  An invalid
        override deliberately leaves that item's original source range in
        place; the caller excludes it from transformation and records a
        candidate-owned failure, while every valid neighbour continues.
        """

        if not self.candidate_boundary_overrides:
            return selected, {}
        adjusted: list[Any] = []
        failures: dict[str, str] = {}
        for scored in selected:
            candidate_id = scored.candidate.id
            override = self.candidate_boundary_overrides.get(candidate_id)
            if not override:
                adjusted.append(scored)
                continue
            stage_name = f"boundary_override:{candidate_id}"

            def reject(message: str) -> None:
                tracker.start(stage_name, _hash({"candidate_id": candidate_id, "override": override}))
                tracker.finish(stage_name, "failed", message)
                failures[candidate_id] = message
                # Keep the pre-edit source range in draft-progress.json.  A
                # reversed/bounds-invalid override must never corrupt the
                # recovery binding for the rest of this batch.
                adjusted.append(scored)

            try:
                start, end = float(override["start"]), float(override["end"])
            except (KeyError, TypeError, ValueError):
                reject(f"Boundary override for {candidate_id} is invalid.")
                continue
            validation = validate_boundary_override(
                start, end,
                source_duration=float(metadata["duration"]) if metadata.get("duration") is not None else None,
                minimum_duration=self.config.candidate_generation.min_duration_seconds,
                maximum_duration=self.config.candidate_generation.max_duration_seconds,
                transcript_features=transcript_features,
                scenes=scenes,
            )
            if not validation["valid"]:
                reject(
                    f"Boundary override for {candidate_id} needs correction: "
                    f"{' '.join(validation['errors'])}"
                )
                continue
            copy = scored_from_dict(scored.to_dict())
            copy.candidate.start = start
            copy.candidate.end = end
            copy.candidate.boundary_diagnostics = {
                **copy.candidate.boundary_diagnostics,
                "review_override": validation,
                "candidate_boundary_fingerprint": _hash({
                    "candidate_id": candidate_id,
                    "start": validation["start"],
                    "end": validation["end"],
                    "analysis": self.expected_analysis_fingerprint,
                }),
            }
            self.warnings.extend(
                f"Boundary override {candidate_id}: {warning}" for warning in validation["warnings"]
            )
            adjusted.append(copy)
        return adjusted, failures

    def _finish_draft_preview(
        self,
        *,
        tracker: StageTracker,
        analysis: AnalysisArtifact,
        source: Source,
        source_data: dict[str, Any],
        metadata: dict[str, Any],
        transcript: dict[str, Any],
        content_profile: dict[str, Any],
        content_map: dict[str, Any],
        story_units: dict[str, Any],
        multimodal_timeline: dict[str, Any],
        visual_analysis: dict[str, Any],
        coverage_map: dict[str, Any],
        clip_count_recommendation: dict[str, Any],
        final_scored: list[Any],
        selected: list[Any],
        composition_vision: dict[str, dict[str, Any]],
        transformation: dict[str, Any],
        production: dict[str, Any],
        work_directory: Path,
        output_directory: Path,
    ) -> PipelineResult:
        """Create candidate-owned Creative Previews from the approved plan graph."""

        production = self._preflight_semantic_content(tracker, production)
        transformations = {
            str(item.get("candidate_id") or ""): item
            for item in transformation.get("items", []) if isinstance(item, dict)
        }
        plans = {
            str(item.get("candidate_id") or ""): item
            for item in production.get("items", []) if isinstance(item, dict)
        }
        selected_ranges = {
            item.candidate.id: (float(item.candidate.start), float(item.candidate.end))
            for item in selected
        }
        selected_by_id = {item.candidate.id: item for item in selected}
        tts = self._run_tts(tracker, production, work_directory, output_directory)
        audio = self._run_audio(
            tracker, production, tts, source, transcript, work_directory, output_directory,
            Path(str(metadata["audio_path"])) if metadata.get("audio_path") else None,
        )
        creative_previews = self._run_production_render(
            tracker, production, audio, source, transcript, work_directory, output_directory,
            visual_analysis,
            creative_candidates=selected,
            multimodal_timeline=multimodal_timeline,
            story_units=story_units,
            render_profile="creative_preview",
        )
        creative_preview_items = {
            str(item.get("candidate_id") or ""): item
            for item in creative_previews.get("items", []) if isinstance(item, dict)
        }
        progress_candidates = self._draft_progress_candidates(selected)
        reviewed: list[dict[str, Any]] = []
        preview_outputs: list[Path] = []
        for index, candidate_id in enumerate(self.selected_candidate_ids, start=1):
            suffix = safe_name(candidate_id, f"candidate-{index:02d}")
            transformation_item = transformations.get(candidate_id, {})
            plan_item = plans.get(candidate_id, {})
            final_script_path = output_directory / f"transformed-script-{suffix}.json"
            production_plan_path = output_directory / f"production-plan-{suffix}.json"
            base = {
                "candidate_id": candidate_id,
                "state": "draft_planning",
                "requested_index": index,
                "source_start_seconds": selected_ranges[candidate_id][0],
                "source_end_seconds": selected_ranges[candidate_id][1],
                "eligibility_decision": selected_by_id[candidate_id].candidate.eligibility_decision.to_dict(),
                "editorial_decision": selected_by_id[candidate_id].candidate.editorial_decision.to_dict(),
                "output_file": None,
                "final_script_ref": str(final_script_path) if final_script_path.is_file() else None,
                "production_plan_ref": str(production_plan_path) if production_plan_path.is_file() else None,
                "composition_vision_evidence": composition_vision.get(candidate_id),
            }
            if transformation_item.get("status") not in {"completed", "fallback"}:
                stage = str(transformation_item.get("stage") or f"transformation_result:{candidate_id}")
                reviewed.append({
                    **base, "state": "draft_failed",
                    "error": _candidate_stage_error(
                        transformation_item,
                        "Draft FinalScript was not created.",
                    ),
                    "stage": stage,
                })
                self._write_draft_progress(
                    output_directory=output_directory, analysis=analysis, source=source,
                    selected=selected, completed=reviewed, base_candidates=progress_candidates,
                )
                continue
            if plan_item.get("status") != "completed":
                stage = str(plan_item.get("stage") or f"production_plan:{candidate_id}")
                reviewed.append({
                    **base, "state": "draft_failed",
                    "error": str(plan_item.get("error") or plan_item.get("reason") or "Draft ProductionPlan was not created."),
                    "stage": stage,
                })
                self._write_draft_progress(
                    output_directory=output_directory, analysis=analysis, source=source,
                    selected=selected, completed=reviewed, base_candidates=progress_candidates,
                )
                continue
            try:
                plan = ProductionPlan.model_validate(plan_item.get("plan"))
            except Exception as error:
                safe = sanitize_api_error(error)
                reviewed.append({
                    **base,
                    "state": "draft_failed",
                    "error": safe,
                    "stage": f"creative_preview:{candidate_id}",
                })
                self._write_draft_progress(
                    output_directory=output_directory, analysis=analysis, source=source,
                    selected=selected, completed=reviewed, base_candidates=progress_candidates,
                )
                continue
            preview_item = creative_preview_items.get(candidate_id, {})
            preview_report = preview_item.get("report") if isinstance(preview_item, dict) else None
            preview_output = Path(str(preview_item.get("output_file") or "")) if isinstance(preview_item, dict) else Path()
            if (
                preview_item.get("status") not in {"completed", "warning"}
                or not isinstance(preview_report, dict)
                or not preview_output.is_file()
                or preview_report.get("render_profile") != "creative_preview"
                or not preview_report.get("compiled_plan_hash")
                or not preview_report.get("parity_signature")
            ):
                preview_error = _candidate_stage_error(
                    preview_item,
                    "Creative Preview не удалось подготовить. Исходный фрагмент остаётся доступным.",
                )
                reviewed.append({
                    **base, "state": "draft_failed", "error": preview_error,
                    "stage": str(preview_item.get("stage") or f"creative_preview:{candidate_id}"),
                    **{
                        key: preview_item[key]
                        for key in ("caption_feasibility_artifact", "pre_render_quality_gate")
                        if key in preview_item
                    },
                })
                self._write_draft_progress(
                    output_directory=output_directory, analysis=analysis, source=source,
                    selected=selected, completed=reviewed, base_candidates=progress_candidates,
                )
                continue
            creative_root = preview_output.parent
            preview = {
                "status": "draft_ready",
                "kind": "creative",
                "output_file": str(preview_output),
                "render_profile": "creative_preview",
                "compiled_plan_hash": str(preview_report["compiled_plan_hash"]),
                "parity_signature": str(preview_report["parity_signature"]),
                "parity_manifest_ref": str(creative_root / "parity-manifest.json"),
                "creative_identity_root": str(creative_root),
            }
            preview_outputs.append(preview_output)
            reviewed.append({
                **base,
                "state": "draft_ready",
                "candidate_boundary_fingerprint": _hash({
                    "candidate_id": candidate_id,
                    "source_range": _plan_source_range(plan),
                    "boundary": transformation_item.get("source_context", {}).get("candidate", {}),
                }),
                "transformation_fingerprint": str(transformation_item.get("transformation_fingerprint") or ""),
                "production_plan_fingerprint": str(plan_item.get("production_plan_fingerprint") or ""),
                "draft_final_script": transformation_item.get("final_script", {}),
                "draft_production_plan": plan.model_dump(mode="json"),
                "preview": preview,
                "creative_identity_root": str(creative_root),
                "output_file": str(preview_output),
                "hook": str(transformation_item.get("final_script", {}).get("hook") or ""),
                "development": str(transformation_item.get("final_script", {}).get("body") or ""),
                "payoff": str(transformation_item.get("final_script", {}).get("ending") or ""),
            })
            self._write_draft_progress(
                output_directory=output_directory, analysis=analysis, source=source,
                selected=selected, completed=reviewed, base_candidates=progress_candidates,
            )
        ready_count = sum(item.get("state") == "draft_ready" for item in reviewed)
        failed_count = len(reviewed) - ready_count
        if ready_count and failed_count:
            self.warnings.append(
                f"Draft preview partial success: {ready_count} of {len(reviewed)} selected candidates are ready; "
                f"{failed_count} can be retried individually."
            )
        artifact_status = "draft_ready" if ready_count == len(reviewed) and ready_count else "draft_partial"
        draft_fingerprint = _hash({
            "analysis": analysis.analysis_fingerprint,
            "candidate_ids": self.selected_candidate_ids,
            "plans": [item.get("draft_production_plan", {}).get("plan_id") for item in reviewed],
            "preview_version": "creative-preview-7G.3",
        })
        draft_id = f"draft-{draft_fingerprint[:16]}"
        draft_path = output_directory / "draft.json"
        draft = new_draft_artifact(
            draft_id=draft_id,
            analysis_id=analysis.analysis_id,
            analysis_fingerprint=analysis.analysis_fingerprint,
            analysis_artifact_path=str(self.analysis_artifact_path),
            project_id=self.project_id or analysis.project_id,
            source_fingerprint=source.id,
            candidates=reviewed,
            status=artifact_status,
            warnings=list(self.warnings),
            run_id=self.run_id,
            analysis_run_id=analysis.analysis_run_id or analysis.analysis_id,
            analysis_artifact_sha256=analysis.verified_sha256,
        )
        tracker.start("draft_artifact", draft_fingerprint)
        draft.write(draft_path)
        tracker.finish("draft_artifact")
        for stage in (*PRODUCTION_RENDER_STAGES, "render"):
            tracker.skip(stage, "Creative Preview is ready for review; final delivery was not started.")
        candidate_failures = [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "stage": str(item.get("stage") or ""),
                "error": str(item.get("error") or "Draft preview failed."),
                **{
                    key: item[key]
                    for key in ("caption_feasibility_artifact", "pre_render_quality_gate")
                    if key in item
                },
            }
            for item in reviewed if item.get("state") != "draft_ready"
        ]
        terminal_failure_message = "; ".join(
            f"{item['candidate_id']}: {item['error']}" for item in candidate_failures
        )
        terminal: dict[str, Any] = {
            "status": "draft_ready" if ready_count else "failed",
            "error_code": None if ready_count else "NO_DRAFT_PREVIEWS",
            "message": (
                "Draft previews are ready for user review."
                if ready_count else f"No candidate draft could be assembled. {terminal_failure_message}"
            ),
            "draft_id": draft_id,
            "candidate_failures": candidate_failures,
        }
        terminal_status = str(terminal["status"])
        terminal_message = str(terminal.get("message") or "")
        tracker.start("terminal")
        tracker.finish("terminal", terminal_status, terminal_message if terminal_status == "failed" else None)
        report_path = output_directory / "report.json"
        tracker.start("report", _hash({"draft": draft_fingerprint, "terminal": terminal}))
        tracker.finish("report")
        make_report(
            report_path, source_data, metadata, self.config, tracker.data, ready_count, len(final_scored),
            [str(path) for path in preview_outputs], self.warnings, self.errors, {},
            gpu_used=False, nvenc_used=False,
            clip_intelligence={
                "candidates": [item.to_dict() for item in final_scored],
                "analysis_artifact_ref": str(self.analysis_artifact_path),
                "analysis_id": analysis.analysis_id,
            },
            content_transformation=transformation,
            production_plan=production,
            tts=tts,
            audio=audio,
            production_render={"enabled": False, "status": "skipped", "reason": "awaiting_user_approval"},
            creative_preview=creative_previews,
            candidate_flow={
                "draft_candidates": reviewed,
                "production_allowed": False,
                "draft_summary": {
                    "requested": len(reviewed),
                    "ready": ready_count,
                    "failed": failed_count,
                },
            },
            terminal=terminal,
            content_understanding={
                "enabled": self.config.content_understanding.enabled, "profile": content_profile, "content_map": content_map,
                "story_units_ref": analysis.references["story_units"],
                "coverage_map_ref": analysis.references["coverage_map"],
                "clip_count_recommendation_ref": analysis.references["clip_count_recommendation"],
                "story_unit_count": len(story_units.get("story_units", [])),
                "coverage_map": coverage_map,
                "clip_count_recommendation": clip_count_recommendation,
            },
            run={
                "run_id": self.run_id, "project_id": self.project_id, "source_id": source.id,
                "run_directory": str(output_directory), "started_at": self.started_at,
                "analysis_id": analysis.analysis_id, "draft_id": draft_id,
                "draft_artifact_path": str(draft_path), "terminal_status": terminal["status"],
            },
        )
        return PipelineResult(
            work_directory=work_directory, output_directory=output_directory, report_path=report_path,
            selected_clips=ready_count, output_files=preview_outputs, warnings=self.warnings,
            terminal_status=terminal_status, error_code=terminal.get("error_code"),
            analysis_path=self.analysis_artifact_path, analysis_id=analysis.analysis_id,
            draft_path=draft_path, draft_id=draft_id,
        )

    def _draft_progress_candidates(self, selected: list[Any]) -> list[dict[str, Any]]:
        """Bind every requested draft to its reviewed source range before rendering.

        This is deliberately separate from discovery of media files: recovery may
        only accept a preview that is named by this binding.
        """

        return [
            {
                "candidate_id": item.candidate.id,
                "state": "draft_planning",
                "requested_index": index,
                "source_start_seconds": float(item.candidate.start),
                "source_end_seconds": float(item.candidate.end),
                "eligibility_decision": item.candidate.eligibility_decision.to_dict(),
                "editorial_decision": item.candidate.editorial_decision.to_dict(),
                "output_file": None,
            }
            for index, item in enumerate(selected, start=1)
        ]

    def _write_draft_progress(
        self,
        *,
        output_directory: Path,
        analysis: AnalysisArtifact,
        source: Source,
        selected: list[Any],
        completed: list[dict[str, Any]] | None = None,
        base_candidates: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Atomically persist candidate-owned draft progress for restart recovery."""

        base = base_candidates or self._draft_progress_candidates(selected)
        by_id = {
            str(item.get("candidate_id") or ""): dict(item)
            for item in completed or [] if isinstance(item, dict)
        }
        candidates = [
            by_id.get(str(item["candidate_id"]), dict(item))
            for item in base
        ]
        progress = new_draft_artifact(
            draft_id=f"draft-progress-{self.run_id}",
            analysis_id=analysis.analysis_id,
            analysis_fingerprint=analysis.analysis_fingerprint,
            analysis_artifact_path=str(self.analysis_artifact_path),
            project_id=self.project_id or analysis.project_id,
            source_fingerprint=source.id,
            candidates=candidates,
            status="draft_partial",
            warnings=list(self.warnings),
            run_id=self.run_id,
            analysis_run_id=analysis.analysis_run_id or analysis.analysis_id,
            analysis_artifact_sha256=analysis.verified_sha256,
        )
        path = output_directory / "draft-progress.json"
        progress.write(path)
        return path

    def _run_production_from_draft(
        self,
        *,
        tracker: StageTracker,
        source: Source,
        work_directory: Path,
        output_directory: Path,
    ) -> PipelineResult:
        """Run expensive delivery only for explicitly approved, draft-ready candidates."""

        assert self.draft_artifact_path is not None
        try:
            draft = DraftArtifact.read(self.draft_artifact_path)
            analysis = AnalysisArtifact.read_verified(
                Path(draft.analysis_artifact_path),
                expected_sha256=draft.analysis_artifact_sha256 or None,
            )
        except (DraftArtifactError, AnalysisArtifactError) as error:
            raise ClipEngineError(f"Draft hand-off cannot be used: {error}") from error
        self.warnings.extend(warning for warning in (*draft.warnings, *analysis.warnings) if warning not in self.warnings)
        if draft.project_id and self.project_id and draft.project_id != self.project_id:
            raise ClipEngineError("Draft artifact belongs to a different project.")
        if draft.source_fingerprint != source.id or analysis.source_fingerprint != source.id:
            raise ClipEngineError("The selected draft belongs to a different source file.")
        if (
            draft.analysis_id != analysis.analysis_id
            or draft.analysis_fingerprint != analysis.analysis_fingerprint
        ):
            raise ClipEngineError("Draft hand-off analysis identity mismatch.")
        expected_analysis_run_id = analysis.analysis_run_id or analysis.analysis_id
        if draft.analysis_run_id and draft.analysis_run_id != expected_analysis_run_id:
            raise ClipEngineError("Draft hand-off analysis run identity mismatch.")
        if analysis.schema_version == "1.0" and not Path(analysis.work_directory).resolve().is_relative_to(
            (self.root / "work").resolve()
        ):
            raise ClipEngineError("Legacy analysis work reference is outside this engine workspace.")
        if not self.selected_candidate_ids:
            raise ClipEngineError("Production render requires explicit approved candidate IDs.")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ClipEngineError("Approved candidate IDs must not contain duplicates.")
        by_id = {str(item.get("candidate_id") or ""): item for item in draft.candidates}
        requested_count = len(self.selected_candidate_ids)
        plans: list[dict[str, Any]] = []
        plan_outcomes: list[dict[str, Any]] = []
        candidate_creative_roots: dict[str, Path] = {}

        def fail_approved_plan(
            *, candidate_id: str, index: int, stage_name: str, reason: str, error: BaseException | str,
        ) -> None:
            safe = sanitize_api_error(error)
            tracker.finish(stage_name, "failed", safe)
            plan_outcomes.append({
                "candidate_id": candidate_id,
                "status": "failed",
                "reason": reason,
                "error": safe,
                "stage": stage_name,
                "requested_index": index,
            })
            self.errors.append(f"{stage_name}: {safe}")

        for index, candidate_id in enumerate(self.selected_candidate_ids, start=1):
            stage_name = f"approved_draft_plan:{candidate_id}"
            tracker.start(stage_name, _hash({"draft": draft.draft_id, "candidate_id": candidate_id}), cache_hit=True)
            record = by_id.get(candidate_id)
            if not isinstance(record, dict):
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_draft_candidate_missing",
                    error=f"Candidate is not present in the selected draft: {candidate_id}.",
                )
                continue
            if record.get("state") not in {"draft_ready", "selected"}:
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_draft_not_ready",
                    error=f"Production render requires a draft_ready candidate: {candidate_id}.",
                )
                continue
            plan = record.get("draft_production_plan")
            if not isinstance(plan, dict):
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_draft_plan_missing",
                    error=f"Draft ProductionPlan is missing for {candidate_id}.",
                )
                continue
            try:
                parsed = ProductionPlan.model_validate(plan)
            except Exception as error:
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_draft_plan_invalid",
                    error=f"Draft ProductionPlan is invalid for {candidate_id}: {sanitize_api_error(error)}",
                )
                continue
            try:
                self._assert_draft_plan_identity(parsed, draft, analysis, source, candidate_id)
            except ClipEngineError as error:
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_draft_plan_stale", error=error,
                )
                continue
            preview = record.get("preview")
            creative_root_value = record.get("creative_identity_root")
            if (
                not isinstance(preview, dict)
                or preview.get("kind") != "creative"
                or preview.get("render_profile") != "creative_preview"
                or not creative_root_value
            ):
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_creative_preview_missing",
                    error="Approved candidate has no verified Creative Preview.",
                )
                continue
            creative_root = Path(str(creative_root_value)).resolve()
            preview_output = Path(str(preview.get("output_file") or "")).resolve()
            if (
                preview_output.parent != creative_root
                or not preview_output.is_file()
                or not (creative_root / "parity-manifest.json").is_file()
                or not (creative_root / "compiled-render-plan.json").is_file()
            ):
                fail_approved_plan(
                    candidate_id=candidate_id, index=index, stage_name=stage_name,
                    reason="approved_creative_preview_invalid",
                    error="Approved Creative Preview is incomplete or unavailable.",
                )
                continue
            candidate_creative_roots[candidate_id] = creative_root
            source_range = _plan_source_range(parsed)
            outcome = {
                "candidate_id": candidate_id,
                "status": "completed",
                "plan": parsed.model_dump(mode="json"),
                "requested_index": index,
                "production_plan_id": parsed.plan_id,
                "source_start_seconds": source_range[0] if source_range else None,
                "source_end_seconds": source_range[1] if source_range else None,
            }
            tracker.finish(stage_name)
            plans.append(outcome)
            plan_outcomes.append(outcome)
        production: dict[str, Any] = {
            "enabled": True,
            "status": "completed" if len(plans) == len(plan_outcomes) else "partial" if plans else "failed",
            "items": plan_outcomes,
            "production_note": "Approved Draft ProductionPlan reused without analysis or draft reassembly.",
        }
        if plans:
            first_plan = plans[0]["plan"]
            timeline = first_plan["timeline"]
            production.update({
                "production_plan": first_plan,
                "segments": first_plan["segments"],
                "estimated_duration": timeline["estimated_duration_seconds"],
                "dialogue_count": timeline["dialogue_count"],
                "narration_count": timeline["narration_count"],
                "pause_count": timeline["pause_count"],
                "timeline_version": timeline["timeline_version"],
            })
        production = self._preflight_semantic_content(tracker, production)
        def load_reference(name: str) -> dict[str, Any]:
            try:
                return analysis.load_reference(name)
            except AnalysisArtifactError as error:
                raise ClipEngineError(f"Approved draft cannot use analysis data: {error}") from error

        source_data = load_reference("source")
        metadata = load_reference("metadata")
        transcript = load_reference("transcript")
        visual_analysis = load_reference("visual_analysis")
        content_profile = load_reference("content_profile")
        validate_video_content_profile(content_profile, expected_source_id=str(source_data.get("id") or ""))
        content_map = load_reference("content_map")
        story_units = load_reference("story_units")
        multimodal_timeline = load_reference("multimodal_timeline")
        coverage_map = load_reference("coverage_map")
        recommendation = load_reference("clip_count_recommendation")
        candidate_data = load_reference("candidate_data")
        final_scored = [
            scored_from_dict(item) for item in candidate_data.get("candidates", []) if isinstance(item, dict)
        ] if isinstance(candidate_data, dict) else []
        final_by_id = {item.candidate.id: item for item in final_scored}
        resolved_editorial_profile = resolve_editorial_profile(content_profile, source=source_data)
        editorial_quality_overrides: dict[str, dict[str, Any]] = {}
        for candidate_id in self.selected_candidate_ids:
            record = by_id.get(candidate_id)
            raw_eligibility = record.get("eligibility_decision") if isinstance(record, dict) else None
            raw_editorial = record.get("editorial_decision") if isinstance(record, dict) else None
            record_candidate_id = str(record.get("candidate_id") or "") if isinstance(record, dict) else ""
            editorial, handoff = build_editorial_final_handoff(
                raw_editorial,
                candidate_id=candidate_id,
                record_candidate_id=record_candidate_id,
                expected_profile=resolved_editorial_profile.to_dict(),
                draft_id=draft.draft_id,
                analysis_id=analysis.analysis_id,
                analysis_run_id=analysis.analysis_run_id or analysis.analysis_id,
                analysis_sha256=analysis.verified_sha256,
            )
            editorial_quality_overrides[candidate_id] = {
                "id": candidate_id,
                "candidate_id": record_candidate_id,
                "eligibility_decision": raw_eligibility,
                "editorial_decision": raw_editorial,
                "editorial_final_handoff": handoff,
            }
            final_item = final_by_id.get(candidate_id)
            if isinstance(raw_eligibility, dict) and final_item is not None:
                final_item.candidate.eligibility_decision = EligibilityDecision.from_dict(raw_eligibility)
            if editorial is not None and final_item is not None:
                final_item.candidate.editorial_decision = editorial
        selected_ids = set(self.selected_candidate_ids)
        tracker.start("approved_draft_handoff", _hash({"draft": draft.draft_id, "selected": self.selected_candidate_ids}), cache_hit=True)
        tracker.finish("approved_draft_handoff")
        tts = self._run_tts(tracker, production, work_directory, output_directory)
        audio = self._run_audio(
            tracker, production, tts, source, transcript, work_directory, output_directory,
            Path(str(metadata["audio_path"])) if metadata.get("audio_path") else None,
        )
        self._native_evidence_context = (final_scored, multimodal_timeline, story_units)
        production_render = self._run_production_render(
            tracker, production, audio, source, transcript, work_directory, output_directory, visual_analysis,
            candidate_creative_roots=candidate_creative_roots,
        )
        self.warnings.extend(production_render.get("warnings", []))
        render_settings_fingerprint = _hash({
            "production_render": self.config.production_render,
            "audio_composition": self.config.audio_composition,
            "tts": self.config.tts,
            "production": self.config.production,
        })
        production_render["render_settings_fingerprint"] = render_settings_fingerprint
        registry = primary_clip_results(production_render)
        registry, quality_reports = self._persist_quality_reports(
            output_directory=output_directory,
            registry=registry,
            source_data=source_data,
            production=production,
            audio=audio,
            production_render=production_render,
            final_scored=final_scored,
            diversity_decision=None,
            candidate_overrides=editorial_quality_overrides,
        )
        production_render["quality_reports"] = quality_reports
        self._assert_current_run_results(registry, output_directory)
        outputs = result_paths(registry, output_directory)
        # Draft FinalScripts were already validated before the user approved
        # this render.  Preserve that completed stage in the candidate flow
        # instead of treating it as missing merely because we intentionally do
        # not re-run transformation here.
        approved_transformation = {
            "items": [
                {"candidate_id": candidate_id, "status": "completed", "source": "approved_draft"}
                for candidate_id in self.selected_candidate_ids
            ],
        }
        candidate_flow = build_candidate_flow(
            final_scored, selected_ids, approved_transformation, production, production_render,
        )
        terminal = build_terminal_state(
            requested_count, outputs, candidate_flow, delivery_required=True, quality_reports=quality_reports,
        )
        tracker.start("terminal")
        tracker.finish("terminal", terminal["status"], terminal.get("message") if terminal["status"] == "failed" else None)
        if terminal["status"] == "failed":
            self.errors.append(f"{terminal['error_code']}: {terminal['message']}")
        elif terminal["status"] == "completed_with_warnings":
            self.warnings.append("Some approved drafts did not reach a final production render.")
        report_path = output_directory / "report.json"
        tracker.start("report", _hash({"draft": draft.draft_id, "selected": self.selected_candidate_ids, "terminal": terminal}))
        tracker.finish("report")
        report = make_report(
            report_path, source_data, metadata, self.config, tracker.data, requested_count, len(final_scored),
            [str(path) for path in outputs], self.warnings, self.errors,
            candidate_data.get("ai", {}) if isinstance(candidate_data, dict) else {},
            gpu_used=transcript.get("runtime", {}).get("device") == "cuda", nvenc_used=False,
            clip_intelligence={
                "candidates": [item.to_dict() for item in final_scored],
                "analysis_artifact_ref": str(draft.analysis_artifact_path),
                "analysis_id": analysis.analysis_id,
            },
            production_plan=production, tts=tts, audio=audio, production_render=production_render,
            candidate_flow=candidate_flow, terminal=terminal,
            content_understanding={
                "enabled": self.config.content_understanding.enabled, "profile": content_profile, "content_map": content_map,
                "story_units_ref": analysis.references["story_units"],
                "coverage_map_ref": analysis.references["coverage_map"],
                "clip_count_recommendation_ref": analysis.references["clip_count_recommendation"],
                "story_unit_count": len(story_units.get("story_units", [])), "coverage_map": coverage_map,
                "clip_count_recommendation": recommendation,
            },
            run={
                "run_id": self.run_id, "project_id": self.project_id, "source_id": source.id,
                "run_directory": str(output_directory), "started_at": self.started_at,
                "analysis_id": analysis.analysis_id, "draft_id": draft.draft_id,
                "selected_candidate_ids": list(self.selected_candidate_ids),
                "render_settings_fingerprint": render_settings_fingerprint,
                "terminal_status": terminal["status"], "error_code": terminal.get("error_code"),
            },
            primary_results=[item.to_dict() for item in registry],
            quality_gate=_quality_gate_summary(quality_reports),
        )
        manifest = write_run_manifest(
            output_directory / "manifest.json", run_id=self.run_id, source=source_data,
            started_at=self.started_at, requested_clip_count=requested_count, production_render=production_render,
            results=registry, run_directory=output_directory, project_id=self.project_id,
            content_understanding={
                "analysis_id": analysis.analysis_id, "draft_id": draft.draft_id,
                "draft_artifact_ref": str(self.draft_artifact_path), "selected_candidate_ids": list(self.selected_candidate_ids),
            }, terminal=terminal,
            quality_gate=_quality_gate_summary(quality_reports),
        )
        report["run"]["manifest_path"] = str(output_directory / "manifest.json")
        report["run"]["finished_at"] = manifest["finished_at"]
        write_json(report_path, report)
        return PipelineResult(
            work_directory=work_directory, output_directory=output_directory, report_path=report_path,
            selected_clips=requested_count, output_files=outputs, warnings=self.warnings,
            terminal_status=terminal["status"], error_code=terminal.get("error_code"),
            analysis_id=analysis.analysis_id, draft_path=self.draft_artifact_path, draft_id=draft.draft_id,
        )

    def _assert_draft_plan_identity(
        self,
        plan: ProductionPlan,
        draft: DraftArtifact,
        analysis: AnalysisArtifact,
        source: Source,
        candidate_id: str,
    ) -> None:
        """Reject a reused draft before TTS/audio/render can attach it elsewhere."""

        envelope = plan.envelope
        if envelope is None:
            raise ClipEngineError("EDIT_PLAN_SCHEMA_INVALID: Draft ProductionPlan has no envelope.")
        if envelope.compatibility_mode == "legacy_adapter":
            self.warnings.append(
                f"LEGACY_PLAN_ADAPTER: approved draft {plan.plan_id} is using the explicit 3A compatibility adapter."
            )
            return
        if envelope.input_fingerprints.analysis_sha256 != analysis.analysis_fingerprint:
            raise ClipEngineError(
                "STALE_INPUTS: Draft ProductionPlan analysis fingerprint does not match its approved analysis artifact."
            )
        expected = {
            "candidate_id": candidate_id,
            "source_id": source.id,
            "analysis_id": analysis.analysis_id,
            "project_id": self.project_id or analysis.project_id or f"project-{source.id}",
        }
        if draft.run_id:
            expected["run_id"] = draft.run_id
        for field, value in expected.items():
            actual = getattr(envelope.identity, field)
            if actual != value:
                raise ClipEngineError(
                    f"IDENTITY_MISMATCH: Draft ProductionPlan {field}={actual!r}; expected {value!r}."
                )

    def _source_stage(
        self, tracker: StageTracker, cache_tracker: StageTracker, artifact: Path, source: Source,
    ) -> dict[str, Any]:
        stored = read_json(artifact, {})
        if cache_tracker.completed("source", artifact, source.id) and stored.get("id") == source.id and stored.get("path") == str(source.path):
            tracker.start("source", source.id)
            tracker.finish("source")
            return stored
        tracker.start("source", source.id)
        cache_tracker.start("source", source.id)
        data = source.to_dict()
        write_json(artifact, data)
        cache_tracker.finish("source")
        tracker.finish("source")
        return data

    def _prepare_source(self, input_path: str | None, url: str | None) -> tuple[Source, Path, Path]:
        if input_path:
            source = local_source(input_path)
            run_key = f"{safe_name(source.display_name)}-{source.id[:12]}"
            work_directory = self.root / "work" / run_key
        else:
            assert url
            run_key = f"url-{stable_text_hash(url)[:16]}"
            work_directory = self.root / "work" / run_key
            old = read_json(work_directory / "source.json", {})
            old_path = Path(str(old.get("path", ""))) if old else None
            source = (
                Source(str(old["id"]), old_path, str(old["display_name"]), str(old["origin"]), bool(old.get("downloaded")))
                if old and old_path is not None and old_path.is_file()
                else url_source(url, work_directory / "download")
            )
        output_root = self.root / "output" / run_key
        output_directory = output_root / "runs" / self.run_id
        self.run_work_directory = work_directory / "runs" / self.run_id
        work_directory.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.run_work_directory.mkdir(parents=True, exist_ok=True)
        return source, work_directory, output_directory

    def _cached(
        self, tracker: StageTracker, stage: str, artifact: Path, fingerprint: Any,
        action: Callable[[], dict[str, Any]], *, cache_tracker: StageTracker | None = None,
        validator: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        if stage in INTELLIGENCE_STAGES:
            fingerprint = {"engine_version": INTELLIGENCE_ENGINE_VERSION, "input": fingerprint}
        cache_key = _hash(fingerprint)
        cache = cache_tracker or tracker
        if cache.completed(stage, artifact, cache_key):
            try:
                cached = read_json(artifact, {})
                if not isinstance(cached, dict):
                    raise ValueError("cached artifact is not an object")
                if validator is not None:
                    validator(cached)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # A completed index entry is not enough: a partial, corrupt or
                # identity-mismatched artifact must be rebuilt in place.
                cache.invalidate(f"Invalid cached artifact for {stage}.", (stage,))
            else:
                # Keep the per-run report truthful even when the source-level
                # cache supplied the artifact.  The cache index and the run state
                # are different files and both writes are process-locked by
                # write_json, so runs never replace one another's state.json.
                tracker.start(stage, cache_key, cache_hit=True)
                tracker.finish(stage)
                return cached
        tracker.start(stage, cache_key, cache_hit=False)
        if cache is not tracker:
            cache.start(stage, cache_key)
        try:
            data = action()
            if validator is not None:
                validator(data)
        except ClipEngineError as error:
            tracker.finish(stage, "failed", str(error))
            if cache is not tracker:
                cache.finish(stage, "failed", str(error))
            raise
        except Exception as error:
            message = f"Непредвиденная ошибка этапа {stage}: {error}"
            tracker.finish(stage, "failed", message)
            if cache is not tracker:
                cache.finish(stage, "failed", message)
            raise StageError(message) from error
        if cache is not tracker:
            cache.finish(stage)
        tracker.finish(stage)
        return data

    def _ai_rerank(self, candidates: list[Candidate], short_candidates: list[Candidate], transcript: dict[str, Any], path: Path) -> dict[str, Any]:
        semantic_mode = str(self.config.virality.semantic_ai_mode)
        reason = None
        if self.no_ai_rerank:
            reason = "no_ai_rerank_requested"
        elif not self.config.ai_reranking.enabled:
            reason = "ai_reranking_disabled"
        elif self.config.virality.enabled and semantic_mode == "off":
            reason = "semantic_ai_explicitly_off"
        if reason is not None:
            usage = _local_ai_usage(self.config.ai.provider)
            usage.update({
                "model": self.config.ai.model,
                "execution_state": "not_called",
                "reason": reason,
            })
            data: dict[str, Any] = {
                "candidates": [item.to_dict() for item in local_rank(candidates)],
                "ai": usage,
                "ai_reranking_used": False,
                "ai_fallback_used": False,
                "selection_mode": "local",
            }
            write_json(path, data)
            return data
        credential_issue = _ai_credential_issue(self.config, self.mock_ai)
        if credential_issue is not None:
            raise SemanticCredentialError(
                f"SEMANTIC_CREDENTIAL_{credential_issue.upper()}: AI API key is {credential_issue}. "
                "Откройте «Настройки», сохраните рабочий ключ и повторите анализ."
            )
        try:
            scorer = get_scorer(self.config, self.mock_ai)
        except Exception as error:
            semantic: list[ScoredCandidate] = []
            usage = _local_ai_usage(self.config.ai.provider, [sanitize_api_error(error)])
            usage.update({
                "model": self.config.ai.model,
                "execution_state": "degraded",
                "reason": "provider_temporarily_unavailable",
                "retryable": True,
            })
            ai_ok = False
        else:
            try:
                semantic, usage = scorer.score(short_candidates, transcript)
                usage = dict(usage)
                ai_ok = not usage.get("api_errors")
                usage.setdefault("provider", self.config.ai.provider)
                usage.setdefault("model", self.config.ai.model)
                if not ai_ok and usage.get("failure_kind") == "auth_rejected":
                    raise SemanticCredentialError(
                        "SEMANTIC_CREDENTIAL_AUTH_REJECTED: AI provider rejected the API key. "
                        "Откройте «Настройки», замените ключ и повторите анализ."
                    )
                usage["execution_state"] = "completed" if ai_ok else "degraded"
                usage["reason"] = (
                    "semantic_ai_completed" if ai_ok else "provider_temporarily_unavailable"
                )
                usage["retryable"] = not ai_ok
            except Exception as error:
                if isinstance(error, SemanticCredentialError):
                    raise
                semantic = []
                usage = _local_ai_usage(self.config.ai.provider, [sanitize_api_error(error)])
                usage.update({
                    "model": self.config.ai.model,
                    "execution_state": "degraded",
                    "reason": "provider_temporarily_unavailable",
                    "retryable": True,
                })
                ai_ok = False
        data = {
            "candidates": [item.to_dict() for item in merge_ai_ranking(candidates, semantic, ai_ok)],
            "ai": usage,
            "ai_reranking_used": ai_ok,
            "ai_fallback_used": not ai_ok,
            "selection_mode": "ai-reranked" if ai_ok else "local-fallback",
        }
        if not ai_ok:
            self.warnings.append(
                "Semantic AI временно недоступен. Локальные артефакты сохранены; "
                "результат помечен как degraded и анализ можно повторить позже."
            )
        write_json(path, data)
        return data

    def _finalize_deep_analysis(
        self, content_profile: dict[str, Any], source_metadata: dict[str, Any] | None = None,
    ) -> DeepAnalysisDecision:
        evidence = {**(source_metadata or {}), **content_profile}
        decision = resolve_deep_analysis(self.config.product_flow.deep_analysis_requested, evidence)
        self.config.optional_visual_features = decision.resolved
        self.config.product_flow.deep_analysis_resolved = decision.resolved
        self.config.product_flow.deep_analysis_reason = decision.reason
        return decision

    # Legacy helper retained for integrations that score an already-built candidate list.
    def _score_candidates(self, candidates: list[Candidate], transcript: dict[str, Any], path: Path) -> dict[str, Any]:
        if not candidates:
            data = {"candidates": [], "ai": _local_ai_usage("not-called")}
            write_json(path, data)
            return data
        semantic, usage = get_scorer(self.config, self.mock_ai).score(candidates, transcript)
        data = {"candidates": [item.to_dict() for item in semantic], "ai": usage}
        write_json(path, data)
        return data

    def _prepare_recommendation_candidates(
        self,
        scored: list,
        visual_analysis: dict[str, Any],
        *,
        content_profile: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        """Refresh deterministic gates after loading old ranking caches."""

        for item in scored:
            candidate = item.candidate
            candidate.eligibility_decision = resolve_eligibility_decision(
                candidate,
                candidate.feature_vector,
                config_version=self.config.scoring.candidate_quality_config_version,
                min_duration_seconds=self.config.candidate_generation.min_duration_seconds,
                max_duration_seconds=self.config.candidate_generation.max_duration_seconds,
                visual_analysis=visual_analysis,
                cached_eligibility=dict(item.virality.get("eligibility") or {}),
            )
            ensure_candidate_boundary_decision(candidate)
            candidate.editorial_decision = evaluate_editorial_candidate(
                candidate,
                content_profile,
                score=float(item.score),
                source=source,
            )

    def _production_feasibility(
        self,
        scored: list,
        *,
        source: Source,
        source_data: dict[str, Any],
        metadata: dict[str, Any],
        transcript: dict[str, Any],
        transcript_features: dict[str, Any],
        audio_features: dict[str, Any],
        scenes: dict[str, Any],
        multimodal_timeline: dict[str, Any],
        story_units: dict[str, Any],
        content_map: dict[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        analysis_fingerprint = _hash({
            "policy_version": PRODUCTION_FEASIBILITY_POLICY_VERSION,
            "source": source.id,
            "transcript": transcript,
            "candidates": [item.candidate.to_dict() for item in scored],
            "multimodal_timeline": multimodal_timeline,
            "story_units": story_units,
        })
        envelope_context = self._production_plan_envelope_context(
            source,
            transcript,
            analysis_id=f"analysis-feasibility-{analysis_fingerprint[:16]}",
            analysis_fingerprint=analysis_fingerprint,
        )
        data = resolve_recommendation_production_feasibility(
            scored,
            content_map=content_map,
            source=source_data,
            metadata=metadata,
            transcript=transcript,
            transcript_features=transcript_features,
            audio_features=audio_features,
            scenes=scenes,
            multimodal_timeline=multimodal_timeline,
            story_units=story_units,
            config=self.config,
            envelope_context=envelope_context,
        )
        write_json(path, data)
        return data

    def _final_selection(
        self,
        scored: list,
        path: Path,
        content_map: dict[str, Any] | None = None,
        production_feasibility: dict[str, Any] | None = None,
        *,
        transcript: dict[str, Any] | None = None,
        transcript_features: dict[str, Any] | None = None,
        scenes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if content_map is not None:
            selected, coverage = select_with_coverage(
                scored,
                self.config,
                content_map,
                production_feasibility=production_feasibility,
                content_profile=None,
            )
        else:
            selected = select_clips(
                scored,
                self.config,
                production_feasibility=production_feasibility,
            )
            coverage = {}
        story_expansions: list[dict[str, Any]] = []
        if content_map is not None and transcript is not None and transcript_features is not None and scenes is not None:
            story_expansions = expand_publishable_story_candidates(
                scored, content_map, transcript, transcript_features, scenes, self.config,
            )
        requested = min(self.config.max_clips, self.config.ai_reranking.final_clip_count)
        warnings: list[str] = []
        if len(selected) < requested:
            warnings.append(
                f"Найдено только {len(selected)} достаточно разных сильных фрагмента из запрошенных {requested}."
            )
        data = {
            "policy_version": "coverage-diversity-mmr-5B.2" if content_map is not None else "temporal-diversity-v2",
            "candidates": [item.to_dict() for item in scored],
            "selected_ids": [item.candidate.id for item in selected],
            "requested_count": requested,
            "warnings": warnings,
            "coverage": coverage,
            "diversity_decision": coverage.get("diversity_decision") if content_map is not None else None,
            "publishable_story_expansion": {
                "schema_version": PUBLISHABLE_STORY_EXPANSION_VERSION,
                "candidates": story_expansions,
            },
            "production_feasibility": {
                "policy_version": (
                    production_feasibility.get("policy_version")
                    if isinstance(production_feasibility, dict) else None
                ),
                "summary": (
                    production_feasibility.get("summary", {})
                    if isinstance(production_feasibility, dict) else {}
                ),
                "artifact_ref": str(path.with_name("production_feasibility.json")),
            },
        }
        write_json(path, data)
        return data

    def _render(self, source: Source, transcript: dict[str, Any], selected: list, output_directory: Path, path: Path) -> dict[str, Any]:
        if not selected:
            message = "Подходящих клипов не выбрано; MP4 не создан."
            self.warnings.append(message)
            data = {"output_files": [], "warnings": [message], "errors": [], "nvenc_used": False}
            write_json(path, data)
            return data
        files: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        used_nvenc = False
        for index, item in enumerate(selected, start=1):
            destination = output_directory / f"clip-{index:02d}-{item.candidate.id}.mp4"
            ass = create_ass(transcript, item.candidate, output_directory / f"clip-{index:02d}-{item.candidate.id}.ass", self.config.output_width, self.config.output_height) if self.config.subtitles_enabled else None
            try:
                rendered, nvenc, warning = render_clip(source.path, item, ass, destination, self.config)
                files.append(str(rendered)); used_nvenc = used_nvenc or nvenc
                if warning: warnings.append(warning)
            except ClipEngineError as error:
                errors.append(str(error))
        data = {"output_files": files, "warnings": warnings, "errors": errors, "nvenc_used": used_nvenc}
        write_json(path, data)
        return data

    def _skip_render_for_production_plan(self, tracker: StageTracker, _path: Path) -> dict[str, Any]:
        """Explicit CLI-only branch: Goal 3A creates no video, ASS, or audio."""

        reason = "Запрошен --production-plan-only: существующий render pipeline не запускался."
        # Do not overwrite render.json or its completed cache state: a later normal
        # process invocation must be able to reuse the old MP4 untouched.
        tracker.skip("render_skipped_production_plan", reason)
        data = {"output_files": [], "warnings": [reason], "errors": [], "nvenc_used": False}
        return data

    def _production_plan_envelope_context(
        self,
        source: Source,
        transcript: dict[str, Any],
        *,
        analysis_id: str,
        analysis_fingerprint: str,
    ) -> ProductionPlanEnvelopeContext:
        """Capture immutable identity and inputs once, without re-analysis."""

        render = self.config.production_render
        flow = self.config.product_flow
        return ProductionPlanEnvelopeContext(
            project_id=self.project_id or f"project-{source.id}",
            run_id=self.run_id,
            analysis_id=analysis_id,
            analysis_fingerprint=analysis_fingerprint,
            source_sha256=stable_file_hash(source.path),
            transcript_sha256=_hash(transcript),
            preset_id=flow.subtitle_preset,
            preset_version=flow.preset_version,
            platform=flow.platform,
            target_width=render.output_width,
            target_height=render.output_height,
            target_fps=render.output_fps,
        )

    def _build_production_plans(
        self, tracker: StageTracker, transformation: dict[str, Any],
        work_directory: Path, output_directory: Path,
        envelope_context: ProductionPlanEnvelopeContext | None = None,
    ) -> dict[str, Any]:
        enabled = self.config.production.enabled or self.production_plan_only
        items = [
            item for item in (transformation.get("items", []) if isinstance(transformation, dict) else [])
            if isinstance(item, dict) and item.get("status") in {"completed", "fallback"}
        ]
        if not enabled:
            tracker.skip("production_plan", "Production Plan отключён конфигурацией.")
            return {"enabled": False, "status": "skipped", "reason": "disabled", "items": []}
        if not items:
            tracker.skip("production_plan", "Нет FinalScript для Production Plan.")
            return {"enabled": True, "status": "skipped", "reason": "no_final_script", "items": []}
        outcomes: list[dict[str, Any]] = []
        artifacts: list[str] = []
        seen_candidate_ids: set[str] = set()
        seen_plan_ids: set[str] = set()
        seen_source_ranges: list[tuple[float, float]] = []
        for index, transformation_item in enumerate(items, start=1):
            raw_final = transformation_item.get("final_script", {})
            final = raw_final if isinstance(raw_final, dict) else {}
            candidate_id = str(final.get("candidate_id") or transformation_item.get("candidate_id") or f"candidate-{index:03d}")
            suffix = safe_name(candidate_id, f"clip-{index:02d}")
            artifact = work_directory / f"production-plan-{suffix}.json"
            stage_name = f"production_plan:{candidate_id}"
            outcome: dict[str, Any]
            try:
                cache_key = _hash({
                    "version": PRODUCTION_PLAN_VERSION,
                    "final_script": final,
                    "source_context": transformation_item.get("source_context", {}),
                    "semantic": transformation_item.get("semantic_representation", {}),
                    "production": self.config.production,
                    "envelope": envelope_context,
                })
                use_cache = self.config.production.cache_enabled and tracker.completed(
                    stage_name, artifact, cache_key,
                )
                final_validation = validate_final_script(
                    final,
                    transformation_item.get("source_context", {}),
                    transformation_item.get("semantic_representation", {}),
                    str(transformation_item.get("candidate_id") or ""),
                )
                if use_cache and not final_validation.passed:
                    tracker.invalidate("Cached ProductionPlan has an invalid FinalScript contract.", (stage_name,))
                    self.warnings.append(
                        f"Production plan cache for {candidate_id} was invalidated by FinalScript contract validation."
                    )
                    use_cache = False
                if use_cache:
                    plan_data = read_json(artifact, {})
                    try:
                        plan = ProductionPlan.model_validate(plan_data)
                    except Exception:
                        tracker.invalidate("Повреждён production plan cache.", (stage_name,))
                        use_cache = False
                    else:
                        outcome = {
                            "status": "completed", "candidate_id": candidate_id,
                            "plan": plan.model_dump(mode="json"), "cache_hit": True,
                        }
                if not use_cache:
                    tracker.start(stage_name, cache_key)
                    plan = build_production_plan(
                        transformation_item,
                        self.config.production,
                        envelope_context=envelope_context,
                    )
                    plan_data = plan.model_dump(mode="json")
                    write_json(artifact, plan_data)
                    tracker.finish(stage_name)
                    outcome = {
                        "status": "completed", "candidate_id": candidate_id,
                        "plan": plan_data, "cache_hit": False,
                    }
                outcome["production_plan_fingerprint"] = cache_key
                plan = ProductionPlan.model_validate(outcome["plan"])
                source_range = _plan_source_range(plan)
                duplicate_reason = _production_plan_duplicate_reason(
                    candidate_id, plan.plan_id, source_range,
                    seen_candidate_ids, seen_plan_ids, seen_source_ranges,
                )
                if duplicate_reason:
                    outcome.update({"status": "skipped", "reason": duplicate_reason})
                    self.warnings.append(
                        f"Production plan {candidate_id} исключён: он дублирует уже выбранный ролик ({duplicate_reason})."
                    )
                    tracker.finish(stage_name, "skipped", duplicate_reason)
                else:
                    outcome.update({
                        "requested_index": index,
                        "production_plan_id": plan.plan_id,
                        "source_start_seconds": source_range[0] if source_range else None,
                        "source_end_seconds": source_range[1] if source_range else None,
                    })
                    artifacts.extend(
                        self._write_production_artifacts(
                            output_directory, suffix, index, outcome["plan"],
                        )
                    )
                    seen_candidate_ids.add(candidate_id)
                    seen_plan_ids.add(plan.plan_id)
                    if source_range is not None:
                        seen_source_ranges.append(source_range)
            except Exception as error:
                safe = _finish_candidate_stage_failure(tracker, stage_name, error)
                outcome = {
                    "status": "failed", "candidate_id": candidate_id,
                    "requested_index": index, "stage": stage_name,
                    "error": safe, "errors": [safe], "cache_hit": False,
                }
                self.errors.append(f"production_plan:{candidate_id}: {safe}")
            outcomes.append(outcome)
        completed = [item for item in outcomes if item.get("status") == "completed"]
        if not completed:
            return {"enabled": True, "status": "failed", "items": outcomes, "artifacts": artifacts}
        first = completed[0]["plan"]
        timeline = first["timeline"]
        return {
            "enabled": True,
            "status": "completed" if len(completed) == len(outcomes) else "partial",
            "production_plan": first,
            "segments": first["segments"],
            "estimated_duration": timeline["estimated_duration_seconds"],
            "dialogue_count": timeline["dialogue_count"],
            "narration_count": timeline["narration_count"],
            "pause_count": timeline["pause_count"],
            "timeline_version": timeline["timeline_version"],
            "cache": {
                "enabled": self.config.production.cache_enabled,
                "hits": [bool(item.get("cache_hit", False)) for item in completed],
                "hit_count": sum(bool(item.get("cache_hit", False)) for item in completed),
            },
            "artifacts": artifacts,
            "items": outcomes,
            "production_note": "Production Plan is the future source of truth; no TTS, audio mix, subtitle render, ASS or video render was generated.",
        }

    def _preflight_semantic_content(
        self,
        tracker: StageTracker,
        production: dict[str, Any],
    ) -> dict[str, Any]:
        """Block unsafe exact dialogue mappings before any delivery work.

        This is a candidate fan-out stage: a blocked plan remains in the stage
        report with a typed reason while valid sibling plans stay renderable.
        Final Quality Gate calls the same policy function after render as a
        defence-in-depth check.
        """

        raw_items = production.get("items") if isinstance(production, dict) else None
        if not isinstance(raw_items, list) or not raw_items:
            return production
        outcomes: list[dict[str, Any]] = []
        for item in raw_items:
            if (
                not isinstance(item, dict)
                or item.get("status") != "completed"
                or not isinstance(item.get("plan"), dict)
            ):
                outcomes.append(item)
                continue
            candidate_id = str(item.get("candidate_id") or "candidate")
            plan = item["plan"]
            stage_name = f"semantic_content_preflight:{candidate_id}"
            tracker.start(stage_name, _hash({
                "policy": SPEECH_CLARITY_POLICY_VERSION,
                "dialogue_mappings": plan.get("dialogue_mappings"),
                "boundary_decision": plan.get("boundary_decision"),
                "continuity_decision": plan.get("continuity_decision"),
            }))
            blocker = exact_dialogue_semantic_blocker(plan)
            if blocker is None:
                tracker.finish(stage_name)
                outcomes.append(item)
                continue
            low_confidence = blocker["evidence"]["low_confidence_dialogue"]
            segment_ids = ", ".join(
                str(mapping.get("segment_id") or mapping.get("fact_id") or "unknown")
                for mapping in low_confidence
            )
            message = (
                f"{blocker['code']}: materially low-confidence exact dialogue "
                f"({blocker['measured_value']} below {blocker['threshold']}; {segment_ids})."
            )
            tracker.finish(stage_name, "failed", message)
            outcomes.append({
                **item,
                "status": "failed",
                "stage": stage_name,
                "reason": blocker["code"],
                "reason_code": blocker["code"],
                "error": message,
                "errors": [message],
                "semantic_blocker": blocker,
            })
            self.errors.append(f"{stage_name}: {message}")
        successful = [item for item in outcomes if isinstance(item, dict) and item.get("status") == "completed"]
        result = dict(production)
        result["items"] = outcomes
        result["status"] = (
            "completed" if successful and len(successful) == len(outcomes)
            else "partial" if successful
            else "failed"
        )
        return result

    def _run_tts(
        self, tracker: StageTracker, production: dict[str, Any],
        work_directory: Path, output_directory: Path,
    ) -> dict[str, Any]:
        """Run Goal 3B for every completed ProductionPlan without rebuilding it."""

        if not self.config.tts.enabled or self.disable_tts:
            tracker.skip("tts_generation", "TTS отключён конфигурацией или --disable-tts.")
            return {"enabled": False, "status": "skipped", "reason": "disabled"}
        plan_items = _production_items(production)
        if not plan_items:
            tracker.skip("tts_generation", "Нет ProductionPlan для TTS.")
            return {"enabled": True, "status": "skipped", "reason": "no_production_plan"}
        outcomes: list[dict[str, Any]] = []
        eligible: list[tuple[int, str, ProductionPlan, dict[str, Any]]] = []
        for default_index, item in enumerate(plan_items, start=1):
            candidate_id = str(item.get("candidate_id") or f"candidate-{default_index:03d}")
            stage_name = f"tts_generation:{candidate_id}"
            try:
                plan_data = item["plan"]
                index = int(item.get("requested_index") or default_index)
                plan = ProductionPlan.model_validate(plan_data)
                stage_name = f"tts_generation:{plan.plan_id}"
                allowed, reason = tts_eligibility(plan)
            except Exception as error:
                safe = _finish_candidate_stage_failure(tracker, stage_name, error)
                outcomes.append({
                    "candidate_id": candidate_id, "status": "failed",
                    "stage": stage_name, "error": safe, "errors": [safe],
                })
                self.errors.append(f"tts:{candidate_id}: {safe}")
                continue
            if not allowed:
                outcomes.append({
                    "candidate_id": candidate_id, "status": "skipped", "reason": reason,
                    "tts_invoked": False, "estimated_cost": 0.0, "actual_cost": 0.0,
                })
                continue
            eligible.append((index, candidate_id, plan, item))
        if not eligible:
            if any(item.get("status") == "failed" for item in outcomes):
                return _multi_stage_report("tts", outcomes)
            reason = str(outcomes[0].get("reason", "no_eligible_narration")) if outcomes else "no_eligible_narration"
            tracker.skip("tts_generation", f"TTS skipped: {reason}.")
            return {
                "enabled": True, "status": "skipped", "reason": reason,
                "tts_invoked": False, "estimated_cost": 0.0, "actual_cost": 0.0,
                "items": outcomes,
            }
        for index, candidate_id, plan, plan_item in eligible:
            stage_name = f"tts_generation:{plan.plan_id}"
            try:
                candidate_output = _candidate_output_directory(
                    output_directory, candidate_id, index,
                )
                tracker.start(stage_name, _hash({
                    "plan": plan.plan_id,
                    "tts": self.config.tts,
                    "recompute": self.recompute_tts,
                }))
                result = TTSService(self.root, self.config).generate(
                    plan, work_directory, candidate_output, force_recompute=self.recompute_tts,
                )
                tracker.finish(
                    stage_name,
                    "completed" if result.status in {"completed", "partial", "fallback"} else result.status,
                )
                report = tts_report_section(result)
                outcome = {
                    "candidate_id": candidate_id, "status": result.status,
                    "output_directory": str(candidate_output), "report": report,
                    "tts_invoked": bool(report.get("tts_invoked", True)),
                    **_production_item_identity(plan_item),
                }
                self.warnings.extend(result.warnings)
                self.errors.extend([f"tts:{candidate_id}: {entry.message}" for entry in result.api_errors])
            except Exception as error:
                safe = _finish_candidate_stage_failure(tracker, stage_name, error)
                outcome = {
                    "candidate_id": candidate_id, "status": "failed",
                    "stage": stage_name, "error": safe, "errors": [safe],
                    **_production_item_identity(plan_item),
                }
                self.errors.append(f"tts:{candidate_id}: {safe}")
            outcomes.append(outcome)
        return _multi_stage_report("tts", outcomes)

    def _run_audio(
        self, tracker: StageTracker, production: dict[str, Any], tts: dict[str, Any], source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path,
        prepared_source_audio_path: Path | None = None,
    ) -> dict[str, Any]:
        """Compose Audio Project after existing plan/TTS stages, never changing render inputs."""

        if not self.config.audio_composition.enabled or self.production_plan_only:
            reason = "disabled" if not self.config.audio_composition.enabled else "production_plan_only"
            tracker.skip("audio_composition", f"Audio Composition skipped: {reason}.")
            return {"enabled": False, "status": "skipped", "reason": reason}
        plan_items = _production_items(production)
        if not plan_items:
            tracker.skip("audio_composition", "Нет ProductionPlan для Audio Composition.")
            return {"enabled": True, "status": "skipped", "reason": "no_production_plan"}
        tts_items = {str(item.get("candidate_id")): item for item in tts.get("items", []) if isinstance(item, dict)}
        outcomes: list[dict[str, Any]] = []
        for default_index, item in enumerate(plan_items, start=1):
            candidate_id = str(item.get("candidate_id") or f"candidate-{default_index:03d}")
            stage_name = f"audio_composition:{candidate_id}"
            try:
                plan_data = item["plan"]
                index = int(item.get("requested_index") or default_index)
                plan = ProductionPlan.model_validate(plan_data)
                stage_name = f"audio_composition:{plan.plan_id}"
                tts_allowed, _reason = tts_eligibility(plan)
                tts_item = tts_items.get(candidate_id)
                if tts_allowed and (
                    not tts_item
                    or tts_item.get("status") not in {"completed", "partial", "fallback"}
                ):
                    upstream_stage = (
                        str(tts_item.get("stage") or f"tts_generation:{candidate_id}")
                        if isinstance(tts_item, dict)
                        else f"tts_generation:{candidate_id}"
                    )
                    outcomes.append({
                        "candidate_id": candidate_id, "status": "skipped",
                        "reason": "tts_unavailable",
                        "stage": upstream_stage,
                        "error": _candidate_stage_error(
                            tts_item,
                            "TTS output is unavailable for this candidate.",
                        ),
                    })
                    continue
                candidate_output_value = (
                    tts_item.get("output_directory") if isinstance(tts_item, dict) else None
                )
                candidate_output = (
                    Path(str(candidate_output_value))
                    if candidate_output_value
                    else _candidate_output_directory(output_directory, candidate_id, index)
                )
                tracker.start(stage_name, _hash({
                    "plan": plan.plan_id,
                    "audio": self.config.audio_composition,
                    "audio_mode": plan.audio_mode,
                    "tts_result": (
                        _file_fingerprint(candidate_output / "tts" / "tts-result.json")
                        if tts_allowed else None
                    ),
                    "recompute": self.recompute_audio,
                }))
                project = AudioCompositionService(self.root, self.config).compose(
                    plan, source, transcript, read_json(candidate_output / "tts" / "tts-result.json", {}) if tts_allowed else None,
                    work_directory, candidate_output, force_recompute=self.recompute_audio,
                    prepared_source_audio_path=prepared_source_audio_path,
                )
                tracker.finish(
                    stage_name,
                    "completed" if project.status in {"completed", "partial"} else project.status,
                )
                outcome = {
                    "candidate_id": candidate_id, "status": project.status,
                    "output_directory": str(candidate_output),
                    "report": audio_report_section(project),
                    **_production_item_identity(item),
                }
                self.warnings.extend(project.warnings)
                self.errors.extend([f"audio:{candidate_id}: {entry}" for entry in project.errors])
            except Exception as error:
                safe = _finish_candidate_stage_failure(tracker, stage_name, error)
                outcome = {
                    "candidate_id": candidate_id, "status": "failed",
                    "stage": stage_name, "error": safe, "errors": [safe],
                    **(
                        _audio_handoff_failure_details(error)
                        if isinstance(error, AudioCompositionError) else {}
                    ),
                    **_production_item_identity(item),
                }
                self.errors.append(f"audio:{candidate_id}: {safe}")
            outcomes.append(outcome)
        return _multi_stage_report("audio", outcomes)

    def _run_production_render(
        self, tracker: StageTracker, production: dict[str, Any], audio: dict[str, Any], source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path, visual_analysis: dict[str, Any] | None = None,
        *, creative_candidates: Iterable[Any] = (),
        multimodal_timeline: dict[str, Any] | None = None,
        story_units: dict[str, Any] | None = None,
        render_profile: Literal["creative_preview", "final"] = "final",
        candidate_creative_roots: dict[str, Path] | None = None,
    ) -> dict[str, Any]:
        """Run the Goal 3D executor after its upstream artifacts, never through legacy render."""

        if not self.config.production_render.enabled or self.disable_production_render or self.production_plan_only:
            if not self.config.production_render.enabled:
                reason = "disabled"
            elif self.disable_production_render:
                reason = "cli_disabled"
            else:
                reason = "production_plan_only"
            tracker.skip("production_render", f"Production render skipped: {reason}.")
            return {"enabled": False, "status": "skipped", "reason": reason}
        plan_items = _production_items(production)
        if not plan_items:
            tracker.skip("production_render", "Нет ProductionPlan для production render.")
            return {"enabled": True, "status": "skipped", "reason": "no_production_plan"}
        creative_candidates = tuple(creative_candidates)
        saved_context = getattr(self, "_native_evidence_context", None)
        if saved_context is not None and not creative_candidates:
            saved_candidates, saved_timeline, saved_stories = saved_context
            creative_candidates = saved_candidates
            multimodal_timeline = multimodal_timeline or saved_timeline
            story_units = story_units or saved_stories
        audio_items = {str(item.get("candidate_id")): item for item in audio.get("items", []) if isinstance(item, dict)}
        candidate_evidence = {
            str(record.get("id") or ""): record
            for item in creative_candidates
            for record in [item.to_dict() if hasattr(item, "to_dict") else item]
            if isinstance(record, dict)
        }
        outcomes: list[dict[str, Any]] = []
        candidate_creative_roots = candidate_creative_roots or {}
        for default_index, item in enumerate(plan_items, start=1):
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            identity = _production_item_identity(item)
            try:
                requested_index = int(identity.get("requested_index") or default_index)
            except (TypeError, ValueError) as error:
                safe = sanitize_api_error(error)
                outcomes.append({
                    "candidate_id": candidate_id, "status": "failed",
                    "stage": f"{render_profile}:{candidate_id}",
                    "error": safe, "errors": [safe], **identity,
                })
                self.errors.append(f"{render_profile}:{candidate_id}: {safe}")
                continue
            audio_item = audio_items.get(candidate_id)
            if not audio_item or audio_item.get("status") not in {"completed", "partial"}:
                upstream_stage = (
                    str(audio_item.get("stage") or f"audio_composition:{candidate_id}")
                    if isinstance(audio_item, dict)
                    else f"audio_composition:{candidate_id}"
                )
                outcomes.append({
                    "candidate_id": candidate_id,
                    "status": "skipped",
                    "reason": "audio_unavailable",
                    "stage": upstream_stage,
                    "error": _candidate_stage_error(
                        audio_item,
                        "Audio output is unavailable for this candidate.",
                    ),
                    **identity,
                })
                continue
            try:
                candidate_output = Path(str(audio_item["output_directory"]))
                plan = ProductionPlan.model_validate(plan_data)
                audio_project = AudioProject.model_validate(read_json(candidate_output / "audio" / "audio-project.json", {}))
            except Exception as error:
                safe = sanitize_api_error(error)
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "errors": [safe], **identity})
                self.errors.append(f"production_render:{candidate_id}: {safe}")
                continue
            compiled_plan: CompiledRenderPlan | None = None
            creative_intent: CreativeIntent | None = None
            creative_handoff: CandidateCreativeHandoff | None = None
            execution_status: str | None = None
            execution_reason_codes: tuple[str, ...] = ()
            execution_diagnostics: tuple[str, ...] = ()
            creative_root = candidate_creative_roots.get(candidate_id)
            if creative_root is not None:
                try:
                    try:
                        creative_intent, compiled_plan, creative_handoff, execution = load_candidate_creative_identity(
                            creative_root, plan,
                        )
                    except CreativeArtifactError:
                        compiled_plan = CompiledRenderPlan.model_validate(
                            read_json(creative_root / "compiled-render-plan.json", None)
                        )
                        if compiled_plan.compatibility_mode != "legacy_adapter":
                            raise
                        execution = None
                    if execution is not None:
                        execution_status = execution.execution_status
                        execution_reason_codes = execution.reason_codes
                        execution_diagnostics = execution.diagnostics
                    if render_profile == "final":
                        _copy_creative_preview_for_parity(creative_root, candidate_output / "creative-preview")
                except Exception as error:
                    safe = sanitize_api_error(error)
                    outcomes.append({
                        "candidate_id": candidate_id, "status": "failed", "errors": [safe], **identity,
                    })
                    self.errors.append(f"creative_preview:{candidate_id}: {safe}")
                    continue
            try:
                report = self._compose_production_render(
                    tracker, plan, audio_project, source, transcript, work_directory, candidate_output,
                    raise_on_error=False, visual_analysis=visual_analysis,
                    phase6_candidate=candidate_evidence.get(candidate_id),
                    multimodal_timeline=multimodal_timeline,
                    story_units=story_units,
                    render_profile=render_profile,
                    compiled_plan=compiled_plan,
                    creative_intent=creative_intent,
                    creative_handoff=creative_handoff,
                    execution_status=execution_status,
                    execution_reason_codes=execution_reason_codes,
                    execution_diagnostics=execution_diagnostics,
                )
                output_file = str(report.get("output_file") or "")
                if render_profile == "final" and report.get("status") in {"completed", "warning"} and output_file:
                    canonical = self._publish_run_result(Path(output_file), output_directory, requested_index)
                    report = dict(report)
                    report["intermediate_output_file"] = output_file
                    report["output_file"] = str(canonical)
                    output_file = str(canonical)
            except Exception as error:
                # A render is a candidate-scoped fan-out unit. Contract,
                # renderer, and publication failures remain local to it.
                safe = sanitize_api_error(error)
                outcomes.append({
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "stage": f"{render_profile}:{candidate_id}",
                    "error": safe,
                    "errors": [safe],
                    **identity,
                })
                self.errors.append(f"{render_profile}:{candidate_id}: {safe}")
                continue
            try:
                outcome = {
                    "clip_result_id": f"{candidate_id}:{plan.plan_id}",
                    "candidate_id": candidate_id,
                    "status": report.get("status", "failed"),
                    "output_directory": str(candidate_output),
                    "report": report,
                    "output_file": output_file,
                    "production_plan_id": plan.plan_id,
                    "source_start_seconds": identity.get("source_start_seconds"),
                    "source_end_seconds": identity.get("source_end_seconds"),
                    "source_fingerprint": _source_range_fingerprint(source.id, identity),
                    "content_fingerprint": _render_content_fingerprint(Path(output_file), report),
                    "run_id": self.run_id,
                    "revision_id": f"{self.run_id}:render-{requested_index:02d}",
                    **{
                        key: report[key]
                        for key in ("caption_feasibility_artifact", "pre_render_quality_gate")
                        if key in report
                    },
                    **identity,
                }
            except Exception as error:
                safe = sanitize_api_error(error)
                outcomes.append({
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "stage": f"{render_profile}:{candidate_id}",
                    "error": safe,
                    "errors": [safe],
                    **identity,
                })
                self.errors.append(f"{render_profile}:{candidate_id}: {safe}")
                continue
            outcomes.append(outcome)
        return _multi_stage_report(
            "production_render" if render_profile == "final" else "creative_preview",
            outcomes,
        )

    def _compose_production_render(
        self, tracker: StageTracker, plan: ProductionPlan, audio_project: AudioProject, source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path, raise_on_error: bool,
        visual_analysis: dict[str, Any] | None = None,
        phase6_candidate: dict[str, Any] | None = None,
        multimodal_timeline: dict[str, Any] | None = None,
        story_units: dict[str, Any] | None = None,
        compiled_plan: CompiledRenderPlan | None = None,
        creative_intent: CreativeIntent | None = None,
        creative_handoff: CandidateCreativeHandoff | None = None,
        execution_status: str | None = None,
        execution_reason_codes: Iterable[str] = (),
        execution_diagnostics: Iterable[str] = (),
        parent_compiled_plan_hash: str | None = None,
        parent_creative_intent_hash: str | None = None,
        style_revision: bool = False,
        allow_creative_revision: bool = False,
        render_profile: Literal["creative_preview", "final"] = "final",
    ) -> dict[str, Any]:
        stage_name = f"{'creative_preview' if render_profile == 'creative_preview' else 'production_render'}:{plan.plan_id}"
        tracker.start(stage_name, _hash({
            "plan": plan.plan_id, "audio_project": audio_project.project_id,
            "mixed_audio": _file_fingerprint(Path(audio_project.mix.mixed_audio_path or "")),
            "config": self.config.production_render, "recompute": self.recompute_production_render,
        }))
        try:
            project = VideoCompositionService(self.root, self.config).compose(
                plan, audio_project, source, transcript, work_directory, output_directory, visual_analysis=visual_analysis,
                force_recompute=self.recompute_production_render,
                phase6_candidate=phase6_candidate,
                phase6_multimodal_timeline=multimodal_timeline,
                phase6_story_units=story_units,
                compiled_plan=compiled_plan,
                creative_intent=creative_intent,
                creative_handoff=creative_handoff,
                execution_status=cast(Any, execution_status),
                execution_reason_codes=execution_reason_codes,
                execution_diagnostics=execution_diagnostics,
                parent_compiled_plan_hash=parent_compiled_plan_hash,
                parent_creative_intent_hash=parent_creative_intent_hash,
                style_revision=style_revision,
                allow_creative_revision=allow_creative_revision,
                render_profile=render_profile,
            )
            report = production_render_report_section(project)
            report["audio_mode"] = plan.audio_mode
            source_range = _plan_source_range(plan)
            identity = {
                "source_start_seconds": source_range[0] if source_range else None,
                "source_end_seconds": source_range[1] if source_range else None,
            }
            report.update({
                "clip_result_id": f"{plan.metadata.candidate_id}:{plan.plan_id}",
                "candidate_id": plan.metadata.candidate_id,
                "production_plan_id": plan.plan_id,
                **identity,
                "source_fingerprint": _source_range_fingerprint(source.id, identity),
                "content_fingerprint": _render_content_fingerprint(
                    Path(str(report.get("output_file") or "")), report,
                ),
                "primary": True,
            })
        except Exception as error:
            safe = sanitize_api_error(error)
            tracker.finish(stage_name, "failed", safe)
            self.errors.append(f"production_render: {safe}")
            lineage = _production_render_error_lineage(error)
            if raise_on_error:
                raise ProductionRenderError(
                    f"Production render did not complete: {safe}",
                    quality_gate_report=getattr(error, "quality_gate_report", None),
                    artifact_reference=getattr(error, "artifact_reference", None),
                ) from error
            return {
                "enabled": True, "status": "failed", "errors": [safe], "ai_called": False,
                **lineage,
            }
        tracker.finish(stage_name, "completed" if project.status in {"completed", "warning"} else project.status)
        self.warnings.extend(project.warnings)
        return report

    def _publish_run_result(self, source: Path, run_directory: Path, index: int) -> Path:
        """Publish an immutable canonical copy; renderer files stay intermediate-only."""

        if not source.is_file():
            raise ProductionRenderError(f"Production render did not produce an MP4: {source}")
        destination = run_directory / "results" / f"final-short-{index:02d}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        if not is_run_scoped_path(destination, run_directory):
            raise ProductionRenderError("Canonical result escaped current run directory.")
        return destination

    def _assert_current_run_results(self, results: list[ClipResult], run_directory: Path) -> None:
        for result in results:
            if result.run_id != self.run_id:
                raise ProductionRenderError(
                    f"Canonical ClipResult belongs to another run: {result.run_id or '<missing>'} != {self.run_id}."
                )
            if not result.revision_id:
                raise ProductionRenderError("Canonical ClipResult is missing revision_id.")
            if not is_run_scoped_path(Path(result.output_file), run_directory):
                raise ProductionRenderError(
                    f"Canonical ClipResult points outside current run directory: {result.output_file}"
                )

    def _persist_quality_reports(
        self,
        *,
        output_directory: Path,
        registry: list[ClipResult],
        source_data: dict[str, Any],
        production: dict[str, Any],
        audio: dict[str, Any],
        production_render: dict[str, Any],
        final_scored: list[Any],
        diversity_decision: dict[str, Any] | None,
        candidate_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[ClipResult], list[dict[str, Any]]]:
        """Persist one Final Quality Gate report for each canonical V2 MP4.

        All inputs are reports already produced by upstream stages.  This is an
        aggregation/persistence step only, so render-only recovery never causes
        a second ffprobe scan, media analysis, or expensive rerender.
        """

        plans = {
            str(item.get("candidate_id") or ""): item.get("plan")
            for item in production.get("items", []) if isinstance(item, dict) and isinstance(item.get("plan"), dict)
        }
        audio_items = {
            str(item.get("candidate_id") or ""): item.get("report")
            for item in audio.get("items", []) if isinstance(item, dict) and isinstance(item.get("report"), dict)
        }
        render_items = {
            str(item.get("candidate_id") or ""): item.get("report")
            for item in production_render.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("report"), dict)
        }
        candidates = {
            str(getattr(item, "candidate", item).id): item.to_dict()
            for item in final_scored if getattr(getattr(item, "candidate", item), "id", None)
        }
        for candidate_id, override in (candidate_overrides or {}).items():
            candidates[candidate_id] = {**candidates.get(candidate_id, {}), **override}
        persisted: list[ClipResult] = []
        references: list[dict[str, Any]] = []
        for index, result in enumerate(registry, start=1):
            render_report = render_items.get(result.candidate_id)
            if render_report is None and len(registry) == 1:
                render_report = production_render
            report = build_quality_report(
                artifact_path=Path(result.output_file),
                result=result,
                run_id=self.run_id,
                project_id=self.project_id,
                source=source_data,
                plan=plans.get(result.candidate_id),
                candidate=candidates.get(result.candidate_id),
                diversity_decision=diversity_decision,
                render_report=render_report,
                audio_report=audio_items.get(result.candidate_id),
                all_results=registry,
            )
            report_path = output_directory / "results" / f"quality-report-{index:02d}.json"
            write_json(report_path, report.to_dict())
            persisted.append(replace(
                result,
                artifact_id=report.artifact_id,
                artifact_checksum=report.artifact_sha256,
                quality_report_id=report.report_id,
                quality_report_path=str(report_path),
                quality_status=report.status,
            ))
            references.append(report.reference(report_path))
        return persisted, references

    def _run_tts_only(
        self, tracker: StageTracker, source: Source, work_directory: Path, output_directory: Path,
    ) -> PipelineResult:
        """Use an existing ProductionPlan only; no media preparation, render, or plan rebuild occurs."""

        plan_path = output_directory / "production-plan.json"
        if not plan_path.is_file():
            raise TTSError(
                "ProductionPlan не найден для --tts-only. Сначала создайте его через "
                "--production-plan-only --transform-script."
            )
        try:
            plan = ProductionPlan.model_validate(read_json(plan_path, {}))
        except Exception as error:
            raise TTSError(f"ProductionPlan для --tts-only невалиден: {sanitize_api_error(error)}") from error
        stage_name = f"tts_generation:{plan.plan_id}"
        tracker.start(stage_name, _hash({"plan": plan.plan_id, "tts": self.config.tts, "recompute": self.recompute_tts, "only": True}))
        if self.disable_tts:
            tracker.skip(stage_name, "Запрошен --disable-tts.")
            tts = {"enabled": False, "status": "skipped", "reason": "disabled"}
        else:
            result = TTSService(self.root, self.config).generate(
                plan, work_directory, output_directory, force_recompute=self.recompute_tts,
                enabled=self.config.tts.enabled,
            )
            tracker.finish(stage_name, "completed" if result.status in {"completed", "partial", "fallback"} else result.status)
            self.warnings.extend(result.warnings)
            self.errors.extend([f"tts: {item.message}" for item in result.api_errors])
            tts = tts_report_section(result)
        report_path = output_directory / "report.json"
        existing = read_json(report_path, {})
        if not isinstance(existing, dict) or not existing:
            existing = {
                "source": source.to_dict(), "source_duration_seconds": None,
                "selected_clips_count": 0, "candidates_count": 0, "output_files": [],
                "warnings": [], "errors": [], "production_plan": {"enabled": True, "status": "completed", "production_plan": plan.model_dump(mode="json")},
            }
        existing["tts"] = tts
        existing["stages"] = tracker.data.get("stages", {})
        existing["warnings"] = [*existing.get("warnings", []), *self.warnings]
        existing["errors"] = [*existing.get("errors", []), *self.errors]
        write_json(report_path, existing)
        old_outputs = [Path(value) for value in existing.get("output_files", []) if Path(value).is_file()]
        return PipelineResult(work_directory, output_directory, report_path, int(existing.get("selected_clips_count", 0) or 0), old_outputs, self.warnings)

    def _run_audio_only(
        self, tracker: StageTracker, source: Source, work_directory: Path, output_directory: Path,
    ) -> PipelineResult:
        """Compose only audio from existing artifacts; no transcribe, TTS, ASS, or video render."""

        plan_path = output_directory / "production-plan.json"
        if not plan_path.is_file():
            raise AudioCompositionError(
                "ProductionPlan не найден для --audio-only. Сначала создайте его через "
                "--production-plan-only --transform-script."
            )
        try:
            plan = ProductionPlan.model_validate(read_json(plan_path, {}))
        except Exception as error:
            raise AudioCompositionError(
                f"ProductionPlan для --audio-only невалиден: {sanitize_api_error(error)}"
            ) from error
        transcript = read_json(work_directory / "transcript.json", {})
        if not isinstance(transcript, dict):
            transcript = {}
        stage_name = f"audio_composition:{plan.plan_id}"
        tracker.start(stage_name, _hash({
            "plan": plan.plan_id, "audio": self.config.audio_composition,
            "tts_result": _file_fingerprint(output_directory / "tts" / "tts-result.json"),
            "recompute": self.recompute_audio, "only": True,
        }))
        try:
            project = AudioCompositionService(self.root, self.config).compose(
                plan, source, transcript, read_json(output_directory / "tts" / "tts-result.json", {}),
                work_directory, output_directory, force_recompute=self.recompute_audio,
                prepared_source_audio_path=_prepared_source_audio_path(work_directory),
            )
        except AudioCompositionError as error:
            safe = sanitize_api_error(error)
            tracker.finish(stage_name, "failed", safe)
            if hasattr(error, "code") and hasattr(error, "evidence"):
                raise error
            raise AudioCompositionError(f"Audio Composition не завершён: {safe}") from error
        tracker.finish(stage_name, "completed" if project.status in {"completed", "partial"} else project.status)
        self.warnings.extend(project.warnings)
        self.errors.extend([f"audio: {item}" for item in project.errors])
        audio = audio_report_section(project)
        report_path = output_directory / "report.json"
        existing = read_json(report_path, {})
        if not isinstance(existing, dict) or not existing:
            existing = {
                "source": source.to_dict(), "source_duration_seconds": None,
                "selected_clips_count": 0, "candidates_count": 0, "output_files": [],
                "warnings": [], "errors": [],
                "production_plan": {"enabled": True, "status": "completed", "production_plan": plan.model_dump(mode="json")},
                "tts": {"enabled": False, "status": "skipped", "reason": "not-run-by-audio-only"},
            }
        existing["audio"] = audio
        existing["stages"] = tracker.data.get("stages", {})
        existing["warnings"] = [*existing.get("warnings", []), *self.warnings]
        existing["errors"] = [*existing.get("errors", []), *self.errors]
        write_json(report_path, existing)
        old_outputs = [Path(value) for value in existing.get("output_files", []) if Path(value).is_file()]
        return PipelineResult(
            work_directory, output_directory, report_path,
            int(existing.get("selected_clips_count", 0) or 0), old_outputs, self.warnings,
        )

    def _run_production_render_only(
        self, tracker: StageTracker, source: Source, work_directory: Path, output_directory: Path,
    ) -> PipelineResult:
        """Execute only Goal 3D artifacts; this branch cannot invoke AI, TTS, audio mix, or legacy render."""

        if self.upstream_run_directory is not None:
            return self._run_candidate_production_rerender(
                tracker, source, work_directory, output_directory,
            )

        upstream_directory = self.upstream_run_directory or output_directory
        plan_path = upstream_directory / "production-plan.json"
        audio_path = upstream_directory / "audio" / "audio-project.json"
        if not plan_path.is_file():
            raise ProductionRenderError(
                "ProductionPlan не найден для --production-render-only. Сначала создайте его через "
                "--production-plan-only --transform-script."
            )
        if not audio_path.is_file():
            raise ProductionRenderError(
                "AudioProject не найден для --production-render-only. Сначала выполните --audio-only."
            )
        try:
            plan = ProductionPlan.model_validate(read_json(plan_path, {}))
            audio_project = AudioProject.model_validate(read_json(audio_path, {}))
        except Exception as error:
            raise ProductionRenderError(
                f"Upstream artifact для --production-render-only невалиден: {sanitize_api_error(error)}"
            ) from error
        transcript = read_json(work_directory / "transcript.json", {})
        if not isinstance(transcript, dict):
            transcript = {}
        report_path = output_directory / "report.json"
        existing = read_json(report_path, {})
        if not isinstance(existing, dict) or not existing:
            existing = {
                "source": source.to_dict(), "source_duration_seconds": None,
                "selected_clips_count": 0, "candidates_count": 0, "output_files": [],
                "warnings": [], "errors": [],
                "production_plan": {"enabled": True, "status": "completed", "production_plan": plan.model_dump(mode="json")},
                "audio": {"enabled": True, "status": audio_project.status},
                "tts": {"enabled": False, "status": "skipped", "reason": "not-run-by-production-render-only"},
            }
        try:
            production_render = self._compose_production_render(
                tracker, plan, audio_project, source, transcript, work_directory, output_directory, raise_on_error=True,
            )
        except ProductionRenderError as error:
            existing["production_render"] = {
                "enabled": True, "status": "failed", "ai_called": False,
                "tts_regenerated": False, "audio_remixed": False,
                "errors": [sanitize_api_error(error)], "warnings": [], "artifacts": [],
                **_production_render_error_lineage(error),
            }
            existing["stages"] = tracker.data.get("stages", {})
            existing["warnings"] = [*existing.get("warnings", []), *self.warnings]
            existing["errors"] = [*existing.get("errors", []), *self.errors]
            write_json(report_path, existing)
            raise
        existing["production_render"] = production_render
        intermediate = str(production_render.get("output_file") or "")
        if intermediate:
            canonical = self._publish_run_result(Path(intermediate), output_directory, 1)
            production_render = dict(production_render)
            production_render.update({
                "intermediate_output_file": intermediate,
                "output_file": str(canonical),
                "clip_result_id": f"{plan.metadata.candidate_id}:{plan.plan_id}",
                "candidate_id": plan.metadata.candidate_id,
                "production_plan_id": plan.plan_id,
                "run_id": self.run_id,
                "revision_id": f"{self.run_id}:render-01",
            })
            existing["production_render"] = production_render
        registry = primary_clip_results(production_render)
        registry, quality_reports = self._persist_quality_reports(
            output_directory=output_directory,
            registry=registry,
            source_data=source.to_dict(),
            production={"items": [{
                "candidate_id": plan.metadata.candidate_id,
                "plan": plan.model_dump(mode="json"),
            }]},
            audio={"items": [{
                "candidate_id": plan.metadata.candidate_id,
                "report": audio_report_section(audio_project),
            }]},
            production_render=production_render,
            final_scored=[],
            diversity_decision=None,
        )
        production_render["quality_reports"] = quality_reports
        self._assert_current_run_results(registry, output_directory)
        existing["primary_results"] = [item.to_dict() for item in registry]
        existing["produced_clips_count"] = len(registry)
        existing["output_files"] = [str(path) for path in result_paths(registry, output_directory) if path.is_file()]
        existing["stages"] = tracker.data.get("stages", {})
        existing["warnings"] = [*existing.get("warnings", []), *self.warnings]
        existing["errors"] = [*existing.get("errors", []), *self.errors]
        existing["run"] = {
            "run_id": self.run_id, "source_id": source.id, "run_directory": str(output_directory),
            "started_at": self.started_at or utc_now(), "manifest_path": str(output_directory / "manifest.json"),
        }
        existing["quality_gate"] = _quality_gate_summary(quality_reports)
        existing["terminal"] = build_terminal_state(
            1,
            [path for path in result_paths(registry, output_directory) if path.is_file()],
            {},
            delivery_required=True,
            quality_reports=quality_reports,
        )
        write_json(report_path, existing)
        write_run_manifest(
            output_directory / "manifest.json", run_id=self.run_id, source=source.to_dict(),
            started_at=self.started_at or utc_now(), requested_clip_count=1,
            production_render=production_render, results=registry, run_directory=output_directory, project_id=self.project_id,
            terminal=existing["terminal"], quality_gate=_quality_gate_summary(quality_reports),
        )
        output_files = [path for path in result_paths(registry, output_directory) if path.is_file()]
        return PipelineResult(
            work_directory, output_directory, report_path,
            int(existing.get("selected_clips_count", 0) or 0), output_files, self.warnings,
        )

    def _run_candidate_production_rerender(
        self, tracker: StageTracker, source: Source, work_directory: Path, output_directory: Path,
    ) -> PipelineResult:
        """Resolve every rerender input through its selected candidate identity."""

        assert self.upstream_run_directory is not None
        upstream = self.upstream_run_directory.resolve()
        parent_report = read_json(upstream / "report.json", None)
        if not isinstance(parent_report, dict):
            raise ProductionRenderError("RERENDER_PARENT_REPORT_INVALID: parent report is missing or corrupt.")
        raw_production_section = parent_report.get("production_plan")
        production_section: dict[str, Any] = (
            raw_production_section if isinstance(raw_production_section, dict) else {}
        )
        raw_render_section = parent_report.get("production_render")
        render_section: dict[str, Any] = (
            raw_render_section if isinstance(raw_render_section, dict) else {}
        )
        production_items = {
            str(item.get("candidate_id") or ""): item
            for item in production_section.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("plan"), dict)
        }
        render_items = {
            str(item.get("candidate_id") or ""): item
            for item in render_section.get("items", [])
            if isinstance(item, dict) and str(item.get("candidate_id") or "")
        }
        legacy_direct_parent = not production_items and isinstance(
            production_section.get("production_plan"), dict,
        )
        if legacy_direct_parent:
            raw = production_section["production_plan"]
            candidate_id = str(raw.get("metadata", {}).get("candidate_id") or "")
            if candidate_id:
                production_items[candidate_id] = {
                    "candidate_id": candidate_id, "plan": raw, "requested_index": 1,
                }
        candidate_ids = list(dict.fromkeys(self.selected_candidate_ids or render_items or production_items))
        if not candidate_ids:
            raise ProductionRenderError("RERENDER_CANDIDATE_MISSING: parent run has no candidate outputs.")

        transcript = read_json(work_directory / "transcript.json", {})
        if not isinstance(transcript, dict):
            transcript = {}
        tracker.skip("tts_generation", "Rerender reuses the selected candidate AudioProject.")
        tracker.skip("audio_composition", "Rerender reuses the selected candidate mixed audio.")
        outcomes: list[dict[str, Any]] = []
        plan_outcomes: list[dict[str, Any]] = []
        audio_outcomes: list[dict[str, Any]] = []

        for fallback_index, candidate_id in enumerate(candidate_ids, start=1):
            plan_item = production_items.get(candidate_id)
            render_item = render_items.get(candidate_id)
            failure: str | None = None
            plan: ProductionPlan | None = None
            audio_project: AudioProject | None = None
            if not isinstance(plan_item, dict) or not isinstance(plan_item.get("plan"), dict):
                failure = "RERENDER_PRODUCTION_PLAN_MISSING: selected candidate has no parent ProductionPlan."
            else:
                try:
                    plan = ProductionPlan.model_validate(plan_item["plan"])
                except Exception as error:
                    failure = f"RERENDER_PRODUCTION_PLAN_INVALID: {sanitize_api_error(error)}"
            if plan is not None and plan.metadata.candidate_id != candidate_id:
                failure = "RERENDER_CANDIDATE_IDENTITY_MISMATCH: ProductionPlan belongs to another candidate."
            requested_index = int((plan_item or {}).get("requested_index") or fallback_index)
            parent_output_value = render_item.get("output_directory") if isinstance(render_item, dict) else None
            parent_candidate_output = (
                Path(str(parent_output_value)).resolve()
                if isinstance(parent_output_value, str) and parent_output_value.strip()
                else (
                    upstream
                    if legacy_direct_parent else _candidate_output_directory(
                        upstream, candidate_id, requested_index,
                    ).resolve()
                )
            )
            if not parent_candidate_output.is_relative_to(upstream):
                failure = (
                    "RERENDER_PARENT_OUTPUT_UNSAFE: selected candidate output is outside "
                    "the parent run directory."
                )
            if failure is None:
                try:
                    audio_project = AudioProject.model_validate(
                        read_json(parent_candidate_output / "audio" / "audio-project.json", None)
                    )
                except Exception as error:
                    failure = f"RERENDER_AUDIO_PROJECT_INVALID: {sanitize_api_error(error)}"
            if (
                failure is None and plan is not None and audio_project is not None
                and audio_project.metadata.plan_reference is not None
                and audio_project.metadata.plan_reference != plan.reference()
            ):
                failure = "RERENDER_AUDIO_IDENTITY_MISMATCH: AudioProject belongs to another ProductionPlan."
            if failure is not None or plan is None or audio_project is None:
                outcomes.append({
                    "candidate_id": candidate_id, "status": "failed",
                    "errors": [failure or "RERENDER_PARENT_ARTIFACT_INVALID"],
                    "requested_index": requested_index,
                })
                continue

            compiled: CompiledRenderPlan | None = None
            intent: CreativeIntent | None = None
            handoff: CandidateCreativeHandoff | None = None
            execution_status: str | None = None
            reason_codes: tuple[str, ...] = ()
            diagnostics: tuple[str, ...] = ()
            parent_compiled_hash: str | None = None
            parent_intent_hash: str | None = None
            style_revision = False
            if plan.envelope is not None and plan.envelope.compatibility_mode == "native":
                try:
                    parent_intent, parent_compiled, handoff, parent_execution = load_candidate_creative_identity(
                        parent_candidate_output / "production-render", plan,
                    )
                except CreativeArtifactError as error:
                    outcomes.append({
                        "candidate_id": candidate_id, "status": "failed",
                        "errors": [str(error)], "requested_index": requested_index,
                    })
                    continue
                parent_compiled_hash = parent_compiled.plan_hash
                parent_intent_hash = parent_intent.canonical_hash()
                execution_status = parent_execution.execution_status
                reason_codes = parent_execution.reason_codes
                diagnostics = parent_execution.diagnostics
                if creative_policy_changed(parent_intent, self.config):
                    intent = revise_creative_intent(parent_intent, self.config)
                    style_revision = True
                else:
                    intent = parent_intent
                    compiled = parent_compiled

            candidate_output = _candidate_output_directory(
                output_directory, candidate_id, fallback_index,
            )
            report = self._compose_production_render(
                tracker, plan, audio_project, source, transcript, work_directory,
                candidate_output, raise_on_error=False,
                compiled_plan=compiled, creative_intent=intent, creative_handoff=handoff,
                execution_status=execution_status,
                execution_reason_codes=reason_codes,
                execution_diagnostics=diagnostics,
                parent_compiled_plan_hash=parent_compiled_hash,
                parent_creative_intent_hash=parent_intent_hash,
                style_revision=style_revision, allow_creative_revision=True,
            )
            output_file = str(report.get("output_file") or "")
            if report.get("status") in {"completed", "warning"} and output_file:
                canonical = self._publish_run_result(Path(output_file), output_directory, requested_index)
                report = dict(report)
                report.update({"intermediate_output_file": output_file, "output_file": str(canonical)})
                output_file = str(canonical)
            identity = _production_item_identity(plan_item or {})
            outcomes.append({
                "clip_result_id": f"{candidate_id}:{plan.plan_id}",
                "candidate_id": candidate_id, "status": report.get("status", "failed"),
                "output_directory": str(candidate_output), "report": report,
                "output_file": output_file, "production_plan_id": plan.plan_id,
                "source_start_seconds": identity.get("source_start_seconds"),
                "source_end_seconds": identity.get("source_end_seconds"),
                "source_fingerprint": _source_range_fingerprint(source.id, identity),
                "content_fingerprint": _render_content_fingerprint(Path(output_file), report),
                "run_id": self.run_id,
                "revision_id": f"{self.run_id}:render-{requested_index:02d}",
                "requested_index": requested_index,
            })
            plan_outcomes.append({
                "candidate_id": candidate_id, "status": "completed",
                "plan": plan.model_dump(mode="json"), "requested_index": requested_index,
            })
            audio_outcomes.append({
                "candidate_id": candidate_id, "status": audio_project.status,
                "output_directory": str(parent_candidate_output),
                "report": audio_report_section(audio_project),
            })

        production_render = _multi_stage_report("production_render", outcomes)
        production = _multi_stage_report("production_plan", plan_outcomes)
        audio = _multi_stage_report("audio", audio_outcomes)
        registry = primary_clip_results(production_render)
        registry, quality_reports = self._persist_quality_reports(
            output_directory=output_directory, registry=registry, source_data=source.to_dict(),
            production=production, audio=audio, production_render=production_render,
            final_scored=[], diversity_decision=None,
        )
        production_render["quality_reports"] = quality_reports
        self._assert_current_run_results(registry, output_directory)
        output_files = [path for path in result_paths(registry, output_directory) if path.is_file()]
        terminal = build_terminal_state(
            len(candidate_ids), output_files, {}, delivery_required=True,
            quality_reports=quality_reports,
        )
        report_path = output_directory / "report.json"
        report = {
            "source": source.to_dict(), "source_duration_seconds": None,
            "selected_clips_count": len(candidate_ids), "candidates_count": len(candidate_ids),
            "produced_clips_count": len(registry), "output_files": [str(path) for path in output_files],
            "warnings": list(self.warnings), "errors": list(self.errors),
            "stages": tracker.data.get("stages", {}), "production_plan": production,
            "tts": {"enabled": False, "status": "skipped", "reason": "rerender_reuses_parent_audio"},
            "audio": audio, "production_render": production_render,
            "primary_results": [item.to_dict() for item in registry],
            "quality_gate": _quality_gate_summary(quality_reports), "terminal": terminal,
            "run": {
                "run_id": self.run_id, "project_id": self.project_id, "source_id": source.id,
                "run_directory": str(output_directory), "started_at": self.started_at or utc_now(),
                "upstream_run_directory": str(upstream), "selected_candidate_ids": candidate_ids,
                "manifest_path": str(output_directory / "manifest.json"),
            },
        }
        write_json(report_path, report)
        write_run_manifest(
            output_directory / "manifest.json", run_id=self.run_id, source=source.to_dict(),
            started_at=self.started_at or utc_now(), requested_clip_count=len(candidate_ids),
            production_render=production_render, results=registry, run_directory=output_directory,
            project_id=self.project_id, terminal=terminal,
            quality_gate=_quality_gate_summary(quality_reports),
        )
        return PipelineResult(
            work_directory, output_directory, report_path, len(candidate_ids), output_files,
            self.warnings, terminal_status=str(terminal.get("status") or "failed"),
            error_code=terminal.get("error_code"),
        )

    def _write_production_artifacts(
        self, output_directory: Path, suffix: str, index: int, plan: dict[str, Any],
    ) -> list[str]:
        json_path = output_directory / f"production-plan-{suffix}.json"
        timeline_path = output_directory / f"timeline-{suffix}.json"
        summary_path = output_directory / f"production-summary-{suffix}.txt"
        write_json(json_path, plan)
        write_json(timeline_path, plan["timeline"])
        summary_path.write_text(production_summary(ProductionPlan.model_validate(plan)), encoding="utf-8")
        paths = [str(json_path), str(timeline_path), str(summary_path)]
        if index == 1:
            primary_json = output_directory / "production-plan.json"
            primary_timeline = output_directory / "timeline.json"
            primary_summary = output_directory / "production-summary.txt"
            write_json(primary_json, plan)
            write_json(primary_timeline, plan["timeline"])
            primary_summary.write_text(production_summary(ProductionPlan.model_validate(plan)), encoding="utf-8")
            paths.extend([str(primary_json), str(primary_timeline), str(primary_summary)])
        return paths

    def _transform_selected(
        self, tracker: StageTracker, source: dict[str, Any], metadata: dict[str, Any], selected: list,
        transcript: dict[str, Any], transcript_features: dict[str, Any], audio_features: dict[str, Any],
        scenes: dict[str, Any], work_directory: Path, output_directory: Path,
    ) -> dict[str, Any]:
        enabled = (self.config.transformation.enabled if self.transform_script is None else self.transform_script) or self.production_plan_only
        if not enabled:
            for name in TRANSFORMATION_STAGES:
                tracker.skip(name, "Transformation отключена конфигурацией или CLI.")
            return {"enabled": False, "status": "skipped", "reason": "disabled", "items": []}
        if not selected:
            for name in TRANSFORMATION_STAGES:
                tracker.skip(name, "Нет selected candidate для transformation.")
            return {"enabled": True, "status": "skipped", "reason": "no_selected_candidate", "items": []}
        provider = None
        provider_error: Exception | None = None
        if not self.no_ai_transformation and self.config.transformation.ai_strategy != "local_only":
            try:
                provider = get_transformer(self.config, self.mock_ai)
            except Exception as error:
                provider_error = error
        outcomes: list[dict[str, Any]] = []
        artifacts: list[str] = []
        for index, scored in enumerate(selected, start=1):
            candidate = getattr(scored, "candidate", None)
            candidate_id = str(getattr(candidate, "id", None) or f"candidate-{index:03d}")
            stage_name = f"transformation_result:{candidate_id}"
            try:
                if candidate is None:
                    raise ValueError("Selected item has no candidate payload.")
                context = build_source_context(
                    source, metadata, candidate, transcript, transcript_features, audio_features,
                    scenes, self.config.transformation,
                )
                suffix = safe_name(candidate_id, f"clip-{index:02d}")
                artifact = work_directory / f"transformation-{suffix}.json"
                cache_key = _hash({
                    "engine": TRANSFORMATION_ENGINE_VERSION,
                    "final_script_contract": FINAL_SCRIPT_CONTRACT_VERSION,
                    "source": source.get("id"),
                    "candidate": candidate.to_dict(),
                    "transcript": _hash(transcript),
                    "supporting_context": [item.to_dict() for item in context.supporting_context],
                    "transformation": self.config.transformation,
                    "provider": "mock" if self.mock_ai else self.config.ai.provider,
                    "model": self.config.ai.model,
                    "prompt_versions": PROMPT_VERSIONS,
                    "no_ai": self.no_ai_transformation,
                })
                use_cache = self.config.transformation.cache_enabled and tracker.completed(
                    stage_name, artifact, cache_key,
                )
                if use_cache:
                    outcome = read_json(artifact, {})
                    final_validation = validate_transformation_outcome(outcome, context)
                    if final_validation.passed:
                        outcome["cache_hit"] = True
                        outcome.setdefault("validation", {})["final_script"] = final_validation.to_dict()
                        outcome["final_script_source"] = "cache"
                    else:
                        tracker.invalidate("Cached FinalScript does not satisfy the current contract.", (stage_name,))
                        self.warnings.append(
                            f"Transformation cache for {candidate_id} was invalidated by FinalScript contract validation."
                        )
                        use_cache = False
                if not use_cache:
                    tracker.start(stage_name, cache_key)
                    actual_provider = provider
                    if provider_error is not None:
                        actual_provider = _UnavailableTransformer(provider_error)
                    outcome = run_content_transformation(
                        context, self.config.transformation, actual_provider,
                        force_local=self.no_ai_transformation,
                    )
                    outcome["cache_hit"] = False
                    write_json(artifact, outcome)
                    self._write_transformation_work_artifacts(work_directory, suffix, outcome)
                    outcome_status = str(outcome.get("status", "failed"))
                    tracker.finish(
                        stage_name,
                        outcome_status if outcome_status in {"completed", "fallback", "failed"} else "failed",
                        _outcome_detail(outcome) if outcome_status == "failed" else None,
                    )
                    self._record_transformation_substages(
                        tracker, candidate_id, cache_key, outcome,
                    )
                outcome["transformation_fingerprint"] = cache_key
                outcome_artifacts = self._write_transformation_artifacts(
                    output_directory, suffix, index, outcome,
                )
                usage = outcome.get("ai_usage", {})
                raw_errors = usage.get("api_errors", []) if isinstance(usage, dict) else []
                if raw_errors:
                    self.errors.extend([f"transformation: {sanitize_api_error(value)}" for value in raw_errors])
                normalization = outcome.get("normalization", {}) if isinstance(outcome.get("normalization"), dict) else {}
                for warning in normalization.get("warnings", []) if isinstance(normalization.get("warnings"), list) else []:
                    self.warnings.append(f"Transformation {candidate_id}: {warning}")
                if outcome.get("fallback", {}).get("used"):
                    reason = outcome.get("fallback", {}).get("reason")
                    self.warnings.append(
                        "Local-only transformation used conservative fallback."
                        if reason == "ai_disabled"
                        else "AI transformation failed -> local fallback used."
                    )
            except Exception as error:
                safe = _finish_candidate_stage_failure(tracker, stage_name, error)
                outcome = {
                    "enabled": True,
                    "status": "failed",
                    "candidate_id": candidate_id,
                    "stage": stage_name,
                    "error": safe,
                    "validation": {
                        "errors": [safe],
                        "final_script": {"passed": False, "errors": [safe]},
                    },
                    "final_script": {"production_ready_for_tts": False},
                    "cacheable": False,
                }
                outcome_artifacts = []
                self.errors.append(f"transformation:{candidate_id}: {safe}")
            outcomes.append(outcome)
            artifacts.extend(outcome_artifacts)
        outcomes, diversity_warnings = _deduplicate_transformation_outcomes(outcomes)
        statuses = [str(item.get("status", "failed")) for item in outcomes]
        overall = "failed" if all(item == "failed" for item in statuses) else "fallback" if "fallback" in statuses else "completed"
        first = outcomes[0]
        return {
            "enabled": True,
            "status": overall,
            "candidate_id": first.get("candidate_id"),
            "strategy": first.get("strategy"),
            "provider": first.get("provider"),
            "model": first.get("model"),
            "prompt_versions": PROMPT_VERSIONS,
            "source_context": first.get("source_context", {}),
            "semantic_representation": first.get("semantic_representation", {}),
            "narrative_plan": first.get("narrative_plan", {}),
            "draft_script": first.get("draft_script", {}),
            "validation": first.get("validation", {}),
            "repair_attempts": first.get("repair_attempts", []),
            "final_script": first.get("final_script", {}),
            "fallback": first.get("fallback", {}),
            "scores": first.get("scores", {}),
            "timings": {"items": [item.get("timings", {}) for item in outcomes]},
            "cache": {
                "enabled": self.config.transformation.cache_enabled,
                "hits": [bool(item.get("cache_hit", False)) for item in outcomes],
                "hit_count": sum(bool(item.get("cache_hit", False)) for item in outcomes),
                "items_cacheable": [bool(item.get("cacheable", True)) for item in outcomes],
            },
            "artifacts": artifacts,
            "items": outcomes,
            "warnings": diversity_warnings,
            "production_note": "Transformed script is separate; original audio and subtitles were not changed.",
        }

    def _record_transformation_substages(
        self, tracker: StageTracker, candidate_id: str, cache_key: str, outcome: dict[str, Any],
    ) -> None:
        """Expose typed intermediate stages in state.json without coupling them to render."""

        validation = outcome.get("validation", {}) if isinstance(outcome.get("validation"), dict) else {}
        final_validation = validation.get("final_script", {}) if isinstance(validation.get("final_script"), dict) else {}
        available = {
            "transformation_source_context": bool(outcome.get("source_context")),
            "transformation_semantic_representation": bool(outcome.get("semantic_representation")),
            "transformation_narrative_plan": bool(outcome.get("narrative_plan")),
            "transformation_script_draft": bool(outcome.get("draft_script")),
            "transformation_script_validation": bool(outcome.get("validation")),
            "transformation_final_script": bool(final_validation.get("passed")),
        }
        for name, exists in available.items():
            stage_name = f"{name}:{candidate_id}"
            tracker.start(stage_name, _hash({"parent": cache_key, "stage": name}))
            tracker.finish(stage_name, "completed" if exists else "failed")

    def _write_transformation_artifacts(
        self, output_directory: Path, suffix: str, index: int, outcome: dict[str, Any],
    ) -> list[str]:
        final = outcome.get("final_script", {})
        validation = outcome.get("validation", {}) if isinstance(outcome.get("validation"), dict) else {}
        final_validation = validation.get("final_script", {}) if isinstance(validation.get("final_script"), dict) else {}
        if not isinstance(final, dict) or not final.get("full_text") or not final_validation.get("passed"):
            return []
        json_path = output_directory / f"transformed-script-{suffix}.json"
        text_path = output_directory / f"transformed-script-{suffix}.txt"
        report_path = output_directory / f"transformation-report-{suffix}.json"
        write_json(json_path, final)
        text_path.write_text(str(final["full_text"]).strip() + "\n", encoding="utf-8")
        write_json(report_path, outcome)
        paths = [str(json_path), str(text_path), str(report_path)]
        if index == 1:
            primary_json = output_directory / "transformed-script.json"
            primary_text = output_directory / "transformed-script.txt"
            write_json(primary_json, final)
            primary_text.write_text(str(final["full_text"]).strip() + "\n", encoding="utf-8")
            original = output_directory / "original-transcript.txt"
            original.write_text(str(outcome.get("source_context", {}).get("transcript_text", "")).strip() + "\n", encoding="utf-8")
            paths.extend([str(primary_json), str(primary_text), str(original)])
        return paths

    def _write_transformation_work_artifacts(
        self, work_directory: Path, suffix: str, outcome: dict[str, Any],
    ) -> None:
        """Keep every typed stage inspectable and independently debuggable in work/."""

        stages = {
            "source_context": outcome.get("source_context", {}),
            "semantic_representation": outcome.get("semantic_representation", {}),
            "narrative_plan": outcome.get("narrative_plan", {}),
            "script_draft": outcome.get("draft_script", {}),
            "script_validation": outcome.get("validation", {}),
            "final_script": outcome.get("final_script", {}),
        }
        for name, data in stages.items():
            write_json(work_directory / f"{name}-{suffix}.json", data)

    def _finish_without_audio(self, tracker: StageTracker, source: dict[str, Any], metadata: dict[str, Any], work_directory: Path, output_directory: Path) -> PipelineResult:
        warning = str(metadata.get("warning", "В видео нет аудио."))
        self.warnings.append(warning)
        for stage in ("transcription", *(name for name in INTELLIGENCE_STAGES if name != "multimodal_timeline"), "production_plan"):
            tracker.skip(stage, warning)
        multimodal_timeline = read_json(work_directory / "multimodal_timeline.json", {})
        report_path = output_directory / "report.json"
        tracker.start("report"); tracker.finish("report")
        make_report(report_path, source, metadata, self.config, tracker.data, 0, 0, [], self.warnings, self.errors, _local_ai_usage("not-called"), False, False, clip_intelligence={"version": "1.6", "selection_mode": "no-audio"}, content_understanding={"enabled": True, "status": "insufficient_audio", "multimodal_timeline_ref": str(work_directory / "multimodal_timeline.json"), "multimodal_diagnostics": multimodal_timeline.get("diagnostics", {})}, content_transformation={"enabled": bool(self.config.transformation.enabled), "status": "skipped", "reason": "no-audio"}, production_plan={"enabled": bool(self.config.production.enabled), "status": "skipped", "reason": "no-audio"}, audio={"enabled": bool(self.config.audio_composition.enabled), "status": "skipped", "reason": "no-audio"}, production_render={"enabled": bool(self.config.production_render.enabled), "status": "skipped", "reason": "no-audio"})
        return PipelineResult(work_directory, output_directory, report_path, 0, [], self.warnings)


def _rebind_vision_artifact(
    path: Path,
    timeline: dict[str, Any],
    *,
    provider: str,
    model: str,
    processing_mode: str,
    prompt_version: str,
    schema_version: str,
) -> dict[str, Any] | None:
    """Reuse pre-Audio-v1 PASS 1 only when its complete frame plan is unchanged."""

    existing = read_json(path, {})
    if not isinstance(existing, dict) or existing.get("pass") != "pass1":
        return None
    provenance = existing.get("provenance")
    diagnostics = existing.get("diagnostics")
    if not isinstance(provenance, dict) or not isinstance(diagnostics, dict):
        return None
    if (
        provenance.get("timeline_schema_version") != "6A.1"
        or provenance.get("provider") != provider
        or provenance.get("model") != model
        or provenance.get("prompt_version") != prompt_version
        or provenance.get("schema_version") != schema_version
        or diagnostics.get("processing_mode") != processing_mode
    ):
        return None
    previous_frames = diagnostics.get("keyframes_found")
    current_frames = [
        {"keyframe_id": item["keyframe_id"], "timestamp": float(item["time_seconds"])}
        for item in timeline.get("keyframes", [])
    ]
    if previous_frames != current_frames:
        return None
    rebound = {
        **existing,
        "analysis_run_id": timeline["analysis_run_id"],
        "provenance": {
            **provenance,
            "timeline_schema_version": timeline["schema_version"],
            "audio_v1_migration": "identity_rebound_without_provider_call",
        },
    }
    try:
        return validate_vision_artifact(rebound, timeline)
    except (TypeError, ValueError):
        return None


def _hash(value: Any) -> str:
    def default(item: Any) -> Any:
        if is_dataclass(item):
            return asdict(cast(Any, item))
        return str(item)
    return stable_text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False, default=default))


def _read_candidate_composition_pass2(
    path: Path,
    *,
    expected: dict[str, Any],
    timeline: dict[str, Any],
) -> dict[str, Any] | None:
    """Read a candidate-owned PASS 2 artifact only when all lineage matches."""

    try:
        data = read_json(path, None)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION
        or data.get("cache_key") != expected["cache_key"]
        or not isinstance(data.get("lineage"), dict)
        or not isinstance(data.get("vision_pass2_evidence"), dict)
    ):
        return None
    lineage = data["lineage"]
    if any(lineage.get(key) != value for key, value in expected.items()):
        return None
    evidence = data["vision_pass2_evidence"]
    if evidence.get("schema_version") != PASS2_EVIDENCE_SCHEMA_VERSION:
        return None
    result = evidence.get("result")
    if result is None:
        return data if evidence.get("status") == "skipped" else None
    try:
        validate_pass2_request(result["request"], timeline)
        validate_pass2_result(
            result, timeline=timeline, request=expected["request"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        result.get("candidate_id") != expected["candidate_id"]
        or result.get("analysis_run_id") != expected["analysis_run_id"]
        or result.get("request") != expected["request"]
    ):
        return None
    return data


def _candidate_composition_pass2_summary(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    cache_hit: bool,
    usable: bool,
) -> dict[str, Any]:
    lineage = artifact["lineage"]
    evidence = artifact["vision_pass2_evidence"]
    result = evidence.get("result")
    observations = result.get("observations", []) if isinstance(result, dict) else []
    return {
        "schema_version": DRAFT_COMPOSITION_PASS2_SCHEMA_VERSION,
        "status": str(evidence.get("status") or "skipped"),
        "cache_hit": cache_hit,
        "usable_composition_evidence": usable,
        "artifact_ref": str(artifact_path),
        "lineage_id": str(lineage.get("lineage_id") or ""),
        "cache_key": str(artifact.get("cache_key") or ""),
        "model": str(lineage.get("model") or ""),
        "provider": str(lineage.get("provider") or ""),
        "source_range": dict(lineage.get("source_range") or {}),
        "bounded_frame_count": int(lineage.get("bounded_frame_count") or 0),
        "observation_count": len(observations) if isinstance(observations, list) else 0,
        "analysis_snapshot_mutated": False,
    }


def _file_fingerprint(path: Path) -> dict[str, Any] | None:
    """Small cache signature without reading or exposing TTS/artifact contents."""

    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _deduplicate_transformation_outcomes(
    outcomes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stop a transformed-script collapse before TTS/audio/render fan-out."""

    accepted: list[dict[str, Any]] = []
    warnings: list[str] = []
    for outcome in outcomes:
        if outcome.get("status") not in {"completed", "fallback"}:
            accepted.append(outcome)
            continue
        duplicate_of = next((
            chosen for chosen in accepted
            if chosen.get("status") in {"completed", "fallback"}
            and _transformation_duplicate(outcome, chosen)
        ), None)
        if duplicate_of is None:
            accepted.append(outcome)
            continue
        candidate_id = str(outcome.get("candidate_id") or "candidate")
        prior_id = str(duplicate_of.get("candidate_id") or "candidate")
        outcome.update({
            "status": "skipped",
            "reason": "transformation_duplicate",
            "duplicate_of_candidate_id": prior_id,
        })
        accepted.append(outcome)
        warnings.append(
            f"Фрагмент {candidate_id} исключён после transformation: он дублирует {prior_id}; готовых копий не будет."
        )
    return accepted, warnings


def _transformation_duplicate(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_range = _outcome_source_range(first)
    second_range = _outcome_source_range(second)
    if first_range is not None and second_range is not None:
        metrics = interval_metrics(*first_range, *second_range)
        if metrics.containment >= 0.98 or metrics.iou >= 0.90:
            return True
    first_context = first.get("source_context", {}) if isinstance(first.get("source_context"), dict) else {}
    second_context = second.get("source_context", {}) if isinstance(second.get("source_context"), dict) else {}
    if transcript_similarity(
        str(first_context.get("transcript_text") or ""),
        str(second_context.get("transcript_text") or ""),
    ) >= 0.94:
        return True
    first_final = first.get("final_script", {}) if isinstance(first.get("final_script"), dict) else {}
    second_final = second.get("final_script", {}) if isinstance(second.get("final_script"), dict) else {}
    return transcript_similarity(
        str(first_final.get("full_text") or ""), str(second_final.get("full_text") or ""),
    ) >= 0.96


def _outcome_source_range(outcome: dict[str, Any]) -> tuple[float, float] | None:
    context = outcome.get("source_context", {}) if isinstance(outcome.get("source_context"), dict) else {}
    candidates = (
        (context.get("start_time"), context.get("end_time")),
        (context.get("source_start_seconds"), context.get("source_end_seconds")),
        (outcome.get("source_start_seconds"), outcome.get("source_end_seconds")),
    )
    for start, end in candidates:
        if start is None or end is None:
            continue
        try:
            start_value, end_value = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end_value >= start_value:
            return start_value, end_value
    return None


def _plan_source_range(plan: ProductionPlan) -> tuple[float, float] | None:
    if not plan.dialogue_mappings:
        return None
    return (
        min(item.source_start_seconds for item in plan.dialogue_mappings),
        max(item.source_end_seconds for item in plan.dialogue_mappings),
    )


def _production_plan_duplicate_reason(
    candidate_id: str, plan_id: str, source_range: tuple[float, float] | None,
    seen_candidate_ids: set[str], seen_plan_ids: set[str], seen_source_ranges: list[tuple[float, float]],
) -> str | None:
    if candidate_id in seen_candidate_ids:
        return "candidate_id"
    if plan_id in seen_plan_ids:
        return "production_plan_id"
    if source_range is not None and any(
        abs(source_range[0] - prior[0]) <= 0.25 and abs(source_range[1] - prior[1]) <= 0.25
        for prior in seen_source_ranges
    ):
        return "source_range"
    return None


def _production_item_identity(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item[key]
        for key in ("requested_index", "production_plan_id", "source_start_seconds", "source_end_seconds")
        if key in item
    }


def _source_range_fingerprint(source_id: str, identity: dict[str, Any]) -> str:
    return _hash({
        "source_id": source_id,
        "start": identity.get("source_start_seconds"),
        "end": identity.get("source_end_seconds"),
    })


def _render_content_fingerprint(path: Path, report: dict[str, Any]) -> str:
    """Exact media fingerprint; a whole-file SHA is stronger than sparse frames."""

    if not path.is_file():
        return ""
    return _hash({"sha256": stable_file_hash(path), "duration": report.get("duration")})


def _production_items(production: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = production.get("items", []) if isinstance(production, dict) else []
    result: list[dict[str, Any]] = []
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict) or item.get("status") != "completed" or not isinstance(item.get("plan"), dict):
            continue
        result.append({
            "candidate_id": str(item.get("candidate_id") or "candidate"),
            "plan": item["plan"],
            **_production_item_identity(item),
        })
    # Old reports expose only the primary plan. Keep render-only and existing
    # cache layouts operational while new full runs fan out to every item.
    if (
        not result
        and not raw_items
        and isinstance(production, dict)
        and isinstance(production.get("production_plan"), dict)
    ):
        result.append({"candidate_id": "primary", "plan": production["production_plan"], "requested_index": 1})
    return result


def build_candidate_flow(
    final_scored: list[Any], selected_ids: set[str], transformation: dict[str, Any],
    production: dict[str, Any], production_render: dict[str, Any],
) -> dict[str, Any]:
    """Make every ranked candidate's path to a canonical clip explicit.

    A candidate may be intentionally rejected by selection, fail a later
    contract, or become a canonical rendered result.  Those are distinct
    states; never infer success merely from an intermediate transformation
    substage having run.
    """

    transformation_items = _items_by_candidate(transformation)
    production_items = _items_by_candidate(production)
    render_items = _items_by_candidate(production_render)
    items: list[dict[str, Any]] = []
    for scored in final_scored:
        candidate = getattr(scored, "candidate", None)
        candidate_id = str(getattr(candidate, "id", "") or "")
        if not candidate_id:
            continue
        if candidate_id not in selected_ids:
            items.append({
                "candidate_id": candidate_id,
                "outcome": "rejected",
                "reason": _selection_rejection_reason(scored),
                "message": _selection_rejection_message(scored),
            })
            continue
        transformed = transformation_items.get(candidate_id)
        if not transformed:
            items.append({
                "candidate_id": candidate_id,
                "outcome": "failed",
                "reason": "transformation_missing",
                "message": "Transformation outcome отсутствует.",
            })
            continue
        if str(transformed.get("status") or "failed") not in {"completed", "fallback"}:
            items.append({
                "candidate_id": candidate_id,
                "outcome": "failed",
                "reason": "transformation_failed",
                "message": _outcome_detail(transformed),
            })
            continue
        plan_item = production_items.get(candidate_id)
        if not plan_item or str(plan_item.get("status") or "failed") != "completed":
            item = {
                "candidate_id": candidate_id,
                "outcome": "failed",
                "reason": "production_plan_failed",
                "message": _outcome_detail(plan_item) if plan_item else "ProductionPlan не был создан.",
            }
            if isinstance(plan_item, dict) and plan_item.get("stage"):
                item["stage"] = str(plan_item["stage"])
            items.append(item)
            continue
        rendered = render_items.get(candidate_id)
        if rendered and str(rendered.get("status") or "failed") in {"completed", "warning"}:
            items.append({
                "candidate_id": candidate_id,
                "outcome": "selected",
                "reason": "rendered",
                "production_plan_id": plan_item.get("production_plan_id"),
                "clip_result_id": rendered.get("clip_result_id"),
            })
            continue
        if str(production_render.get("status") or "") == "skipped" and not production_render.get("enabled", False):
            items.append({
                "candidate_id": candidate_id,
                "outcome": "selected",
                "reason": "render_not_requested",
                "production_plan_id": plan_item.get("production_plan_id"),
            })
            continue
        items.append({
            "candidate_id": candidate_id,
            "outcome": "failed",
            "reason": "render_failed",
            "message": _outcome_detail(rendered) if rendered else "Render job не был создан.",
            "production_plan_id": plan_item.get("production_plan_id"),
        })
    return {
        "found": len(final_scored),
        "selected": len(selected_ids),
        "transformed": sum(
            str(item.get("status") or "") in {"completed", "fallback"}
            for item in transformation_items.values()
        ),
        "production_plans": sum(
            str(item.get("status") or "") == "completed" for item in production_items.values()
        ),
        "render_attempts": len(render_items),
        "rendered": sum(item["outcome"] == "selected" and item["reason"] == "rendered" for item in items),
        "rejected": sum(item["outcome"] == "rejected" for item in items),
        "failed": sum(item["outcome"] == "failed" for item in items),
        "items": items,
    }


def build_terminal_state(
    requested_clip_count: int,
    output_files: list[Path],
    candidate_flow: dict[str, Any],
    *,
    delivery_required: bool,
    quality_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a terminal contract after reportable artifacts already exist."""

    produced = len(output_files)
    details = {
        key: int(candidate_flow.get(key, 0) or 0)
        for key in ("found", "selected", "transformed", "production_plans", "render_attempts", "rendered", "rejected", "failed")
    }
    quality = _quality_gate_summary(quality_reports)
    if delivery_required and requested_clip_count > 0 and produced == 0:
        return {
            "status": "failed",
            "error_code": NO_RENDERABLE_CLIPS,
            "message": NO_RENDERABLE_CLIPS_MESSAGE,
            "requested_clip_count": requested_clip_count,
            "produced_clips_count": produced,
            "candidate_counts": details,
            "quality_gate": quality,
        }
    # ``None`` is the explicit legacy path.  An empty list is a V2 gate that
    # failed to persist a report and must never make an MP4 ready.
    if delivery_required and quality_reports is not None:
        if not quality_reports:
            return {
                "status": "failed",
                "error_code": "QUALITY_REPORT_MISSING",
                "message": "Final output has no persisted QualityReport.",
                "requested_clip_count": requested_clip_count,
                "produced_clips_count": produced,
                "candidate_counts": details,
                "quality_gate": quality,
            }
        if quality and quality["status"] == "BLOCKED":
            return {
                "status": "failed",
                "error_code": "QUALITY_GATE_BLOCKED",
                "message": "Final Quality Gate blocked the output; see QualityReport findings.",
                "requested_clip_count": requested_clip_count,
                "produced_clips_count": produced,
                "candidate_counts": details,
                "quality_gate": quality,
            }
    if produced and details["failed"]:
        return {
            "status": "completed_with_warnings",
            "error_code": None,
            "message": "Часть кандидатов не дошла до финального рендера.",
            "requested_clip_count": requested_clip_count,
            "produced_clips_count": produced,
            "candidate_counts": details,
            "quality_gate": quality,
        }
    return {
        "status": "completed",
        "error_code": None,
        "message": "",
        "requested_clip_count": requested_clip_count,
        "produced_clips_count": produced,
        "candidate_counts": details,
        "quality_gate": quality,
    }


def _quality_gate_summary(quality_reports: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Create a shared terminal/manifest/report reference, never a second gate."""

    if quality_reports is None:
        return None
    statuses = [str(item.get("status") or "") for item in quality_reports]
    status = "BLOCKED" if not statuses or "BLOCKED" in statuses else (
        "PASS_WITH_WARNINGS" if "PASS_WITH_WARNINGS" in statuses else "PASS"
    )
    return {
        "schema_version": "5G.0",
        "status": status,
        "reports": quality_reports,
    }


def _items_by_candidate(stage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_items = stage.get("items", []) if isinstance(stage, dict) else []
    result: dict[str, dict[str, Any]] = {}
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id:
            result[candidate_id] = item
    return result


def _outcome_detail(
    item: dict[str, Any] | None,
    fallback: str = "Неизвестная причина.",
) -> str:
    """Extract one candidate-owned failure reason from every stage result shape."""

    if not isinstance(item, dict):
        return fallback
    report = item.get("report")
    sources = (item, report if isinstance(report, dict) else {})
    for source in sources:
        error = str(source.get("error") or "").strip()
        if error:
            return error
        validation = source.get("validation")
        validation_sources = (
            (validation, validation.get("final_script", {}))
            if isinstance(validation, dict)
            else ()
        )
        for validation_source in validation_sources:
            if not isinstance(validation_source, dict):
                continue
            errors = validation_source.get("errors")
            if isinstance(errors, list) and errors:
                detail = "; ".join(str(value) for value in errors[:3] if str(value))
                if detail:
                    return detail
            if errors:
                return str(errors)
        errors = source.get("errors")
        if isinstance(errors, list) and errors:
            detail = "; ".join(str(value) for value in errors[:3] if str(value))
            if detail:
                return detail
        if errors:
            return str(errors)
        for key in ("reason", "message"):
            detail = str(source.get(key) or "").strip()
            if detail:
                return detail
    return fallback


def _selection_rejection_reason(scored: Any) -> str:
    candidate = getattr(scored, "candidate", None)
    boundary = getattr(candidate, "boundary_diagnostics", {}) if candidate is not None else {}
    if isinstance(boundary, dict) and boundary and not boundary.get("eligible", True):
        return "invalid_boundary"
    message = _selection_rejection_message(scored).lower()
    if "payoff" in message:
        return "missing_payoff"
    if "publish" in message or "quality" in message or "оценк" in message:
        return "not_publishable"
    return "not_selected"


def _selection_rejection_message(scored: Any) -> str:
    diagnostics = getattr(scored, "selection_diagnostics", {})
    if isinstance(diagnostics, dict) and diagnostics.get("reason"):
        return str(diagnostics["reason"])
    return str(getattr(scored, "rejection_reason", "") or getattr(scored, "selection_reason", "") or "Не прошёл selection.")


def _candidate_output_directory(root: Path, candidate_id: str, index: int) -> Path:
    return root if index == 1 else root / "candidates" / safe_name(candidate_id, f"clip-{index:02d}")


def _copy_creative_preview_for_parity(source_root: Path, destination_root: Path) -> None:
    """Stage the approved preview beside a new Final revision atomically."""

    source_root = source_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in ("creative-preview.mp4", "parity-manifest.json"):
        source = source_root / name
        if not source.is_file():
            raise CreativeArtifactError(f"CREATIVE_PREVIEW_ARTIFACT_MISSING:{name}")
        destination = destination_root / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)


def _multi_stage_report(stage: str, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in outcomes if item.get("status") in {"completed", "partial", "fallback", "warning"}]
    if not successful:
        return {"enabled": True, "status": "failed", "items": outcomes}
    primary = dict(successful[0].get("report") or {})
    status = "completed" if len(successful) == len(outcomes) else "warning" if stage == "production_render" else "partial"
    primary.update({"enabled": True, "status": status, "items": outcomes})
    if stage == "production_render":
        registry = primary_clip_results({"items": outcomes})
        primary["output_file"] = registry[0].output_file if registry else (successful[0].get("output_file") or primary.get("output_file"))
        primary["output_files"] = [item.output_file for item in registry]
        primary["clip_results"] = [item.to_dict() for item in registry]
        if len(registry) < len(successful):
            primary["status"] = "warning"
            primary.setdefault("warnings", []).append(
                f"Сохранено только {len(registry)} уникальных ролика из {len(successful)} завершённых; копии скрыты."
            )
        if len(successful) != len(outcomes):
            primary.setdefault("warnings", []).append("Не все ролики удалось экспортировать; готовые результаты сохранены.")
    return primary


def _candidate_stage_error(item: object, fallback: str) -> str:
    """Return the sanitized candidate-level failure carried by a stage item."""

    detail = _outcome_detail(item if isinstance(item, dict) else None, fallback)
    return sanitize_api_error(RuntimeError(detail))


def _production_render_error_lineage(error: BaseException) -> dict[str, Any]:
    """Carry a validated pre-render decision into candidate/run failure reports."""

    result: dict[str, Any] = {}
    artifact = getattr(error, "artifact_reference", None)
    quality_gate = getattr(error, "quality_gate_report", None)
    if isinstance(artifact, dict):
        result["caption_feasibility_artifact"] = dict(artifact)
    if isinstance(quality_gate, dict):
        result["pre_render_quality_gate"] = dict(quality_gate)
    return result


def _finish_candidate_stage_failure(
    tracker: StageTracker,
    stage_name: str,
    error: BaseException,
) -> str:
    """Close a candidate-owned stage after any ordinary unexpected exception."""

    safe = sanitize_api_error(error)
    stage = tracker.data.get("stages", {}).get(stage_name, {})
    if not isinstance(stage, dict) or stage.get("status") != "running":
        tracker.start(stage_name)
    tracker.finish(stage_name, "failed", safe)
    return safe


def _prepared_source_audio_path(work_directory: Path) -> Path | None:
    """Return only a local source-derived audio artifact; never download or transcribe here."""

    metadata = read_json(work_directory / "metadata.json", {})
    value = metadata.get("audio_path") if isinstance(metadata, dict) else None
    path = Path(str(value)) if value else work_directory / "audio_16khz_mono.wav"
    return path if path.is_file() else None


def _write(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    write_json(path, data); return data


def _write_analysis_snapshot(
    directory: Path,
    objects: dict[str, dict[str, Any]],
    producer: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Materialize immutable run lineage from analysis objects already in memory."""

    directory.mkdir(parents=True, exist_ok=True)
    references: dict[str, str] = {}
    integrity: dict[str, dict[str, Any]] = {}
    for name, value in objects.items():
        path = directory / f"{name}.json"
        if path.exists():
            if read_json(path, None) != value:
                raise ClipEngineError(
                    f"ANALYSIS_SNAPSHOT_IMMUTABLE: run snapshot member already differs: {name}."
                )
        else:
            write_json(path, value)
        references[name] = str(path)
        integrity[name] = {
            "sha256": stable_file_hash(path),
            "byte_size": path.stat().st_size,
            "producer": {**producer, "artifact_name": name},
        }
    return references, integrity


def _write_candidates(path: Path, candidates: list[Candidate]) -> dict[str, Any]:
    data = {"candidates": [candidate.to_dict() for candidate in candidates]}
    write_json(path, data)
    return data


def _write_generated_candidates(path: Path, generated: tuple[list[Candidate], int]) -> dict[str, Any]:
    candidates, before_deduplication = generated
    data = {
        "candidates": [candidate.to_dict() for candidate in candidates],
        "candidates_before_deduplication": before_deduplication,
        "candidates_generated": before_deduplication,
        "candidates_after_deduplication": len(candidates),
    }
    write_json(path, data)
    return data


def _local_ai_usage(provider: str, errors: list[str] | None = None) -> dict[str, Any]:
    return {"provider": provider, "model": None, "input_tokens": 0, "output_tokens": 0, "retries": 0, "api_errors": errors or []}


def _ai_credential_issue(config: AppConfig, force_mock: bool = False) -> str | None:
    if force_mock or config.mock_ai or config.ai.provider == "mock":
        return None
    variable = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}.get(config.ai.provider)
    if variable is None:
        return "missing"
    value = os.getenv(variable)
    if not value:
        return "missing"
    return "invalid" if validate_api_key(config.ai.provider, value) else None


def _audio_handoff_failure_details(error: AudioCompositionError) -> dict[str, Any]:
    """Keep pre-composition plan invariant failures structured in the stage artifact."""

    code = getattr(error, "code", None)
    evidence = getattr(error, "evidence", None)
    if isinstance(code, str) and isinstance(evidence, dict):
        return {"failure_code": code, "evidence": evidence}
    return {}


class _UnavailableTransformer:
    """Turns a missing key/dependency into the normal non-blocking fallback path."""

    name = "unavailable"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def transform_compact(self, context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        raise TransformationProviderError(sanitize_api_error(self.error))
