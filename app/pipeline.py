from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

from app.ai import get_scorer, get_transformer, sanitize_api_error
from app.analysis_artifact import AnalysisArtifact, AnalysisArtifactError, candidate_review_payload, new_analysis_artifact, potential_counts
from app.candidate_review import validate_boundary_override
from app.draft_artifact import DraftArtifact, DraftArtifactError, new_draft_artifact
from app.draft_preview import DraftPreviewService
from app.audio_modes import tts_eligibility
from app.clip_results import ClipResult, primary_clip_results, result_paths
from app.diversity import interval_metrics, transcript_similarity
from app.audio_features import analyse_audio
from app.audio_models import AudioProject
from app.audio_service import AudioCompositionService, audio_report_section
from app.config import AppConfig
from app.content_understanding import (
    CONTENT_STRATEGY_VERSION,
    build_coverage_map,
    build_global_content_map,
    build_video_content_profile,
    generate_semantic_candidates,
    recommend_clip_count,
    story_units_artifact,
)
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
    StageError,
    TTSError,
    TransformationProviderError,
)
from app.intelligence import intelligence_summary, local_rank, merge_ai_ranking, shortlist
from app.local_scoring import score_candidates
from app.media import prepare_media
from app.models import Candidate, candidate_from_dict, scored_from_dict
from app.rendering import render_clip
from app.reporting import make_report
from app.run_manifest import is_run_scoped_path, write_run_manifest
from app.production_models import ProductionPlan
from app.production_plan import PRODUCTION_PLAN_VERSION, build_production_plan, production_summary
from app.scene_detection import detect_scene_boundaries
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
from app.virality import apply_virality_ranking, build_virality_assessments


INTELLIGENCE_STAGES = (
    "transcript_features", "audio_features", "scene_detection", "candidates_v2",
    "local_scoring", "shortlist", "ai_ranking", "final_selection", "visual_analysis", "video_content_profile",
    "global_content_map", "story_units", "semantic_boundaries", "virality_profiles", "virality_ranking",
    "coverage_map", "clip_count_recommendation", "render", "report",
)
INTELLIGENCE_ENGINE_VERSION = "1.7.1"
TRANSFORMATION_STAGES = (
    "transformation_source_context", "transformation_semantic_representation",
    "transformation_narrative_plan", "transformation_script_draft",
    "transformation_script_validation", "transformation_final_script", "transformation_result",
)
PRODUCTION_PLAN_STAGES = ("production_plan",)
TTS_STAGES = ("tts_generation",)
AUDIO_COMPOSITION_STAGES = ("audio_composition",)
PRODUCTION_RENDER_STAGES = ("production_render",)


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


class StageTracker:
    def __init__(self, state_path: Path) -> None:
        self.path = state_path
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
        self.data["stages"][name] = {
            "status": "running", "started_at": utc_now(), "_started": time.perf_counter(),
            "cache_key": cache_key,
        }
        if cache_hit is not None:
            self.data["stages"][name]["cache_hit"] = cache_hit
        self._save()

    def finish(self, name: str, status: str = "completed", error: str | None = None) -> None:
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

    def run(self, input_path: str | None = None, url: str | None = None) -> PipelineResult:
        validate_source_arguments(input_path, url)
        self.started_at = utc_now()
        source, work_directory, output_directory = self._prepare_source(input_path, url)
        assert self.run_work_directory is not None
        tracker = StageTracker(self.run_work_directory / "state.json")
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
            return self._run_draft_preview(
                tracker=tracker,
                source=source,
                work_directory=work_directory,
                output_directory=output_directory,
            )
        if self.draft_artifact_path is not None:
            return self._run_production_from_draft(
                tracker=tracker,
                source=source,
                work_directory=work_directory,
                output_directory=output_directory,
            )
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
            return self._run_tts_only(tracker, source, work_directory, output_directory)
        if self.audio_only:
            return self._run_audio_only(tracker, source, work_directory, output_directory)
        if self.production_render_only:
            return self._run_production_render_only(tracker, source, work_directory, output_directory)
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
            return self._finish_without_audio(tracker, source_data, metadata, work_directory, output_directory)
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
        content_profile = self._cached(
            tracker, "video_content_profile", work_directory / "video_content_profile.json",
            {
                "source": source.id,
                "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features),
                "audio_features": _hash(audio_features),
                "scenes": _hash(scenes),
                "visual_analysis": _hash(visual_analysis),
                "strategy_version": self.config.content_understanding.strategy_version,
                "profile_schema_version": self.config.content_understanding.profile_schema_version,
                "implementation_version": CONTENT_STRATEGY_VERSION,
            },
            lambda: _write(
                work_directory / "video_content_profile.json",
                build_video_content_profile(
                    source_data, metadata, transcript, transcript_features, audio_features, scenes,
                    visual_analysis, self.config,
                ),
            ),
            cache_tracker=source_cache,
        )
        content_map = self._cached(
            tracker, "global_content_map", work_directory / "global_content_map.json",
            {
                "source": source.id,
                "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features),
                "audio_features": _hash(audio_features),
                "scenes": _hash(scenes),
                "visual_analysis": _hash(visual_analysis),
                "profile": _hash(content_profile),
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
                work_directory / "global_content_map.json",
                build_global_content_map(
                    source_data, metadata, transcript, transcript_features, audio_features, scenes,
                    visual_analysis, content_profile, self.config,
                ),
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
                "content_map": _hash(content_map), "transcript": _hash(transcript),
                "transcript_features": _hash(transcript_features), "scenes": _hash(scenes),
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
                work_directory / "semantic_boundaries.json",
                generate_semantic_candidates(content_map, transcript, transcript_features, scenes, self.config),
            ),
            cache_tracker=source_cache,
        )
        raw_candidates = self._cached(
            tracker, "candidates_v2", work_directory / "candidates_v2.json",
            {
                "semantic_boundaries": _hash(semantic_boundaries),
            },
            lambda: _write(work_directory / "candidates_v2.json", dict(semantic_boundaries)),
            cache_tracker=source_cache,
        )
        # Compatibility artifact retained for existing users of the pre-1.6 cache layout.
        write_json(work_directory / "candidates.raw.json", raw_candidates)
        candidates = [candidate_from_dict(item) for item in raw_candidates.get("candidates", [])]
        local_data = self._cached(
            tracker, "local_scoring", work_directory / "candidates.local.json",
            {"candidates": _hash(raw_candidates), "settings": self.config.scoring},
            lambda: _write_candidates(work_directory / "candidates.local.json", score_candidates(candidates, audio_features, scenes, self.config.scoring)),
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
        ai_data = self._cached(
            tracker, "ai_ranking", work_directory / "ai_ranking.json",
            {"shortlist": _hash(shortlist_data), "ai": self.config.ai, "reranking": self.config.ai_reranking, "mock": self.mock_ai, "disabled": self.no_ai_rerank},
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
                    "source": source.id, "candidates": _hash(local_data), "content_map": _hash(content_map),
                    "transcript_features": _hash(transcript_features), "audio_features": _hash(audio_features),
                    "visual_features": _hash(visual_analysis), "content_profile": _hash(content_profile),
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
                        [candidate_from_dict(item) for item in local_data.get("candidates", [])], content_map,
                        transcript_features, audio_features, visual_analysis, content_profile, self.config.virality,
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
        final_data = self._cached(
            tracker, "final_selection", work_directory / "final_selection.json",
            {
                "policy_version": "coverage-aware-virality-v2" if self.config.virality.enabled else "coverage-aware-v1",
                "scored": _hash(ranked_data), "content_map": _hash(content_map),
                "threshold": self.config.score_threshold, "overlap": self.config.overlap_threshold,
                "distance": self.config.min_selected_clip_distance_seconds, "limit": self.config.ai_reranking.final_clip_count,
                "coverage": {
                    "version": self.config.content_understanding.coverage_selection_version,
                    "weights": self.config.content_understanding.coverage_weights,
                    "strong_story_unit_threshold": self.config.content_understanding.strong_story_unit_threshold,
                    "semantic_duplicate_threshold": self.config.content_understanding.semantic_duplicate_threshold,
                    "coverage_min_quality_score": self.config.content_understanding.coverage_min_quality_score,
                },
                "virality": {
                    "enabled": self.config.virality.enabled,
                    "schema_version": self.config.virality.schema_version,
                    "minimum_quality_score": self.config.virality.minimum_quality_score,
                    "minimum_publishability_score": self.config.virality.minimum_publishability_score,
                },
            },
            lambda: self._final_selection(scored, work_directory / "final_selection.json", content_map),
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
            return self._finish_analysis_only(
                tracker=tracker,
                source=source,
                source_data=source_data,
                metadata=metadata,
                transcript=transcript,
                transcript_features=transcript_features,
                audio_features=audio_features,
                scenes=scenes,
                visual_analysis=visual_analysis,
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
                coverage_map=coverage_map,
                clip_count_recommendation=clip_count_recommendation,
                final_scored=final_scored,
                selected_ids=selected_ids,
                work_directory=work_directory,
                output_directory=output_directory,
            )
        transformation = self._transform_selected(
            tracker, source_data, metadata, selected, transcript, transcript_features,
            audio_features, scenes, work_directory, output_directory,
        )
        self.warnings.extend(transformation.get("warnings", []))
        production = self._build_production_plans(
            tracker, transformation, work_directory, output_directory,
        )
        tts = self._run_tts(tracker, production, work_directory, output_directory)
        audio = self._run_audio(
            tracker, production, tts, source, transcript, work_directory, output_directory,
            Path(str(metadata["audio_path"])) if metadata.get("audio_path") else None,
        )
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
        tracker.start("report", _hash({"final": final_data, "coverage": coverage_map, "recommendation": clip_count_recommendation, "render": render_data, "ai": ai_usage, "transformation": transformation, "production": production, "tts": tts, "audio": audio, "production_render": production_render}))
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
                "enabled": True,
                "profile": content_profile,
                "content_map": content_map,
                "story_units_ref": str(work_directory / "story_units.json"),
                "semantic_boundaries_ref": str(work_directory / "semantic_boundaries.json"),
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
                "story_units_ref": str(work_directory / "story_units.json"),
                "semantic_boundary_ref": str(work_directory / "semantic_boundaries.json"),
                "coverage_map_ref": str(work_directory / "coverage_map.json"),
                "clip_count_recommendation_ref": str(work_directory / "clip_count_recommendation.json"),
                "strategy_version": self.config.content_understanding.strategy_version,
                "analysis_fingerprint": _hash({"profile": content_profile, "map": content_map, "coverage": coverage_map}),
            },
            virality={
                **virality_report,
                "analysis_fingerprint": _hash({"profiles": virality_profiles, "ranking": virality_ranking}) if self.config.virality.enabled else None,
            },
            terminal=terminal,
        )
        report["run"]["manifest_path"] = str(output_directory / "manifest.json")
        report["run"]["finished_at"] = manifest["finished_at"]
        write_json(report_path, report)
        return PipelineResult(
            work_directory, output_directory, report_path, len(outputs), outputs, self.warnings,
            terminal_status=terminal["status"], error_code=terminal.get("error_code"),
        )

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
        coverage_map: dict[str, Any],
        clip_count_recommendation: dict[str, Any],
        final_scored: list[Any],
        selected_ids: set[str],
        work_directory: Path,
        output_directory: Path,
    ) -> PipelineResult:
        """Persist a reusable intelligence result without starting delivery work."""

        scored_records = [item.to_dict() for item in final_scored]
        candidate_data_path = work_directory / "candidates.scored.json"
        # The source cache remains the authoritative large-data store.  This
        # output-side artifact is the immutable hand-off for review and render.
        references = {
            "source": str(work_directory / "source.json"),
            "metadata": str(work_directory / "metadata.json"),
            "transcript": str(work_directory / "transcript.json"),
            "transcript_features": str(work_directory / "transcript_features.json"),
            "audio_features": str(work_directory / "audio_features.json"),
            "scene_boundaries": str(work_directory / "scene_boundaries.json"),
            "visual_analysis": str(work_directory / "visual_analysis.json"),
            "content_profile": str(work_directory / "video_content_profile.json"),
            "content_map": str(work_directory / "global_content_map.json"),
            "story_units": str(work_directory / "story_units.json"),
            "semantic_boundaries": str(work_directory / "semantic_boundaries.json"),
            "coverage_map": str(work_directory / "coverage_map.json"),
            "clip_count_recommendation": str(work_directory / "clip_count_recommendation.json"),
            "candidate_data": str(candidate_data_path),
        }
        if self.config.virality.enabled:
            references["virality_profiles"] = str(work_directory / "virality_profiles.json")
            references["virality_ranking"] = str(work_directory / "virality_ranking.json")
        analysis_fingerprint = _hash({
            "engine": INTELLIGENCE_ENGINE_VERSION,
            "source": source.id,
            "profile": content_profile,
            "content_map": content_map,
            "ranking": final_data,
            "coverage": coverage_map,
            "recommendation": clip_count_recommendation,
        })
        analysis_id = f"analysis-{analysis_fingerprint[:16]}"
        artifact_path = output_directory / "analysis.json"
        review_candidates = [candidate_review_payload(record, selected_ids) for record in scored_records]
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
            },
            summary={
                "candidate_count": len(final_scored),
                "recommended_count": len(selected_ids),
                "source_duration_seconds": metadata.get("duration"),
                "content_type": content_profile.get("detected_content_type"),
                "potential_counts": potential_counts(review_candidates),
            },
            content_profile={
                "detected_content_type": content_profile.get("detected_content_type"),
                "confidence": content_profile.get("confidence"),
                "strategy": content_profile.get("strategy"),
            },
            duration_seconds=float(metadata["duration"]) if metadata.get("duration") is not None else None,
            candidate_count=len(review_candidates),
            recommended_count={
                "min": int(clip_count_recommendation.get("estimated_publishable_clip_range", {}).get("min", len(selected_ids)) or 0),
                "max": int(clip_count_recommendation.get("estimated_publishable_clip_range", {}).get("max", len(selected_ids)) or 0),
                "default": len(selected_ids),
            },
            warnings=list(self.warnings),
        )
        tracker.start("analysis_artifact", analysis_fingerprint)
        artifact.write(artifact_path)
        tracker.finish("analysis_artifact")
        for stage in (
            *TRANSFORMATION_STAGES, *PRODUCTION_PLAN_STAGES, *TTS_STAGES,
            *AUDIO_COMPOSITION_STAGES, *PRODUCTION_RENDER_STAGES, "render",
        ):
            tracker.skip(stage, "Analysis-only run: delivery stage was not started.")
        terminal = {
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
                "enabled": True,
                "profile": content_profile,
                "content_map": content_map,
                "story_units_ref": str(work_directory / "story_units.json"),
                "semantic_boundaries_ref": str(work_directory / "semantic_boundaries.json"),
                "coverage_map_ref": str(work_directory / "coverage_map.json"),
                "clip_count_recommendation_ref": str(work_directory / "clip_count_recommendation.json"),
                "story_unit_count": len(story_units.get("story_units", [])),
                "coverage_map": coverage_map,
                "clip_count_recommendation": clip_count_recommendation,
                "strategy_version": self.config.content_understanding.strategy_version,
            },
            virality=virality_report,
            terminal=terminal,
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
            analysis = AnalysisArtifact.read(self.analysis_artifact_path)
        except AnalysisArtifactError as error:
            raise ClipEngineError(f"Analysis artifact cannot be used: {error}") from error
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

        analysis_work_directory = Path(analysis.work_directory).resolve()
        root_work_directory = (self.root / "work").resolve()
        if not analysis_work_directory.is_relative_to(root_work_directory):
            raise ClipEngineError("Analysis artifact work reference is outside this engine workspace.")

        def load_reference(name: str) -> dict[str, Any]:
            raw_path = analysis.references.get(name)
            if not raw_path:
                raise ClipEngineError(f"Analysis artifact is missing its {name} reference.")
            path = Path(raw_path).resolve()
            if not path.is_relative_to(analysis_work_directory) or not path.is_file():
                raise ClipEngineError(f"Analysis reference is unavailable or unsafe: {name}.")
            value = read_json(path, None)
            if not isinstance(value, dict):
                raise ClipEngineError(f"Analysis reference is corrupted: {name}.")
            return value

        candidate_path = Path(analysis.candidate_data_ref).resolve()
        if not candidate_path.is_relative_to(analysis_work_directory) or not candidate_path.is_file():
            raise ClipEngineError("Analysis candidate data is unavailable or unsafe.")
        candidate_data = read_json(candidate_path, None)
        if not isinstance(candidate_data, dict):
            raise ClipEngineError("Analysis candidate data is corrupted.")
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
        content_map = load_reference("content_map")
        coverage_map = load_reference("coverage_map")
        clip_count_recommendation = load_reference("clip_count_recommendation")
        story_units = load_reference("story_units")
        selected = self._apply_boundary_overrides(selected, metadata, transcript_features, scenes)

        tracker.start("analysis_handoff", analysis.analysis_fingerprint, cache_hit=True)
        tracker.finish("analysis_handoff")
        transformation = self._transform_selected(
            tracker, source_data, metadata, selected, transcript, transcript_features,
            audio_features, scenes, work_directory, output_directory,
        )
        self.warnings.extend(transformation.get("warnings", []))
        production = self._build_production_plans(tracker, transformation, work_directory, output_directory)
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
            coverage_map=coverage_map,
            clip_count_recommendation=clip_count_recommendation,
            final_scored=final_scored,
            transformation=transformation,
            production=production,
            work_directory=work_directory,
            output_directory=output_directory,
        )
    def _apply_boundary_overrides(
        self, selected: list[Any], metadata: dict[str, Any], transcript_features: dict[str, Any], scenes: dict[str, Any],
    ) -> list[Any]:
        """Apply persisted review edits using cached boundary evidence only."""

        if not self.candidate_boundary_overrides:
            return selected
        adjusted: list[Any] = []
        for scored in selected:
            candidate_id = scored.candidate.id
            override = self.candidate_boundary_overrides.get(candidate_id)
            if not override:
                adjusted.append(scored)
                continue
            try:
                start, end = float(override["start"]), float(override["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ClipEngineError(f"Boundary override is invalid for {candidate_id}.") from error
            validation = validate_boundary_override(
                start, end,
                source_duration=float(metadata["duration"]) if metadata.get("duration") is not None else None,
                minimum_duration=self.config.candidate_generation.min_duration_seconds,
                maximum_duration=self.config.candidate_generation.max_duration_seconds,
                transcript_features=transcript_features,
                scenes=scenes,
            )
            if not validation["valid"]:
                raise ClipEngineError(
                    f"Boundary override for {candidate_id} needs correction: {' '.join(validation['errors'])}"
                )
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
        return adjusted

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
        coverage_map: dict[str, Any],
        clip_count_recommendation: dict[str, Any],
        final_scored: list[Any],
        transformation: dict[str, Any],
        production: dict[str, Any],
        work_directory: Path,
        output_directory: Path,
    ) -> PipelineResult:
        """Create only fast review previews from Draft FinalScript/ProductionPlan."""

        transformations = {
            str(item.get("candidate_id") or ""): item
            for item in transformation.get("items", []) if isinstance(item, dict)
        }
        plans = {
            str(item.get("candidate_id") or ""): item
            for item in production.get("items", []) if isinstance(item, dict)
        }
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
                "final_script_ref": str(final_script_path) if final_script_path.is_file() else None,
                "production_plan_ref": str(production_plan_path) if production_plan_path.is_file() else None,
            }
            if transformation_item.get("status") not in {"completed", "fallback"}:
                reviewed.append({
                    **base, "state": "draft_failed",
                    "error": str(transformation_item.get("error") or "Draft FinalScript was not created."),
                })
                continue
            if plan_item.get("status") != "completed":
                reviewed.append({
                    **base, "state": "draft_failed",
                    "error": str(plan_item.get("error") or plan_item.get("reason") or "Draft ProductionPlan was not created."),
                })
                continue
            try:
                plan = ProductionPlan.model_validate(plan_item.get("plan"))
                tracker.start(f"draft_preview:{candidate_id}", _hash({
                    "analysis": analysis.analysis_fingerprint, "plan": plan.plan_id,
                    "preview": {"width": 540, "height": 960, "fps": 24, "version": "1.0"},
                }))
                preview = DraftPreviewService().render(plan, source, output_directory / "drafts" / f"{index:02d}-{suffix}")
                tracker.finish(f"draft_preview:{candidate_id}")
            except Exception as error:
                safe = sanitize_api_error(error)
                tracker.finish(f"draft_preview:{candidate_id}", "failed", safe)
                reviewed.append({**base, "state": "draft_failed", "error": safe})
                self.warnings.append(f"Draft preview {candidate_id} failed: {safe}")
                continue
            preview_outputs.append(preview.output_file)
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
                "preview": preview.to_dict(),
                "hook": str(transformation_item.get("final_script", {}).get("hook") or ""),
                "development": str(transformation_item.get("final_script", {}).get("body") or ""),
                "payoff": str(transformation_item.get("final_script", {}).get("ending") or ""),
            })
        ready_count = sum(item.get("state") == "draft_ready" for item in reviewed)
        artifact_status = "draft_ready" if ready_count == len(reviewed) and ready_count else "draft_partial"
        draft_fingerprint = _hash({
            "analysis": analysis.analysis_fingerprint,
            "candidate_ids": self.selected_candidate_ids,
            "plans": [item.get("draft_production_plan", {}).get("plan_id") for item in reviewed],
            "preview_version": "1.0",
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
        )
        tracker.start("draft_artifact", draft_fingerprint)
        draft.write(draft_path)
        tracker.finish("draft_artifact")
        for stage in (*TTS_STAGES, *AUDIO_COMPOSITION_STAGES, *PRODUCTION_RENDER_STAGES, "render"):
            tracker.skip(stage, "Draft preview is ready for review; production delivery was not started.")
        terminal = {
            "status": "draft_ready" if ready_count else "failed",
            "error_code": None if ready_count else "NO_DRAFT_PREVIEWS",
            "message": "Draft previews are ready for user review." if ready_count else "No candidate draft could be assembled.",
            "draft_id": draft_id,
        }
        tracker.start("terminal")
        tracker.finish("terminal", terminal["status"], terminal.get("message") if terminal["status"] == "failed" else None)
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
            production_render={"enabled": False, "status": "skipped", "reason": "awaiting_user_review"},
            candidate_flow={"draft_candidates": reviewed, "production_allowed": False},
            terminal=terminal,
            content_understanding={
                "enabled": True, "profile": content_profile, "content_map": content_map,
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
            terminal_status=terminal["status"], error_code=terminal.get("error_code"),
            analysis_path=self.analysis_artifact_path, analysis_id=analysis.analysis_id,
            draft_path=draft_path, draft_id=draft_id,
        )

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
            analysis = AnalysisArtifact.read(Path(draft.analysis_artifact_path))
        except (DraftArtifactError, AnalysisArtifactError) as error:
            raise ClipEngineError(f"Draft hand-off cannot be used: {error}") from error
        if draft.project_id and self.project_id and draft.project_id != self.project_id:
            raise ClipEngineError("Draft artifact belongs to a different project.")
        if draft.source_fingerprint != source.id or analysis.source_fingerprint != source.id:
            raise ClipEngineError("The selected draft belongs to a different source file.")
        if not self.selected_candidate_ids:
            raise ClipEngineError("Production render requires explicit approved candidate IDs.")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ClipEngineError("Approved candidate IDs must not contain duplicates.")
        by_id = {str(item.get("candidate_id") or ""): item for item in draft.candidates}
        missing = [candidate_id for candidate_id in self.selected_candidate_ids if candidate_id not in by_id]
        if missing:
            raise ClipEngineError(f"Candidates are not present in the selected draft: {', '.join(missing)}.")
        not_ready = [
            candidate_id for candidate_id in self.selected_candidate_ids
            if by_id[candidate_id].get("state") not in {"draft_ready", "selected"}
        ]
        if not_ready:
            raise ClipEngineError(f"Production render requires draft_ready candidates: {', '.join(not_ready)}.")
        plans: list[dict[str, Any]] = []
        for index, candidate_id in enumerate(self.selected_candidate_ids, start=1):
            record = by_id[candidate_id]
            plan = record.get("draft_production_plan")
            if not isinstance(plan, dict):
                raise ClipEngineError(f"Draft ProductionPlan is missing for {candidate_id}.")
            try:
                parsed = ProductionPlan.model_validate(plan)
            except Exception as error:
                raise ClipEngineError(f"Draft ProductionPlan is invalid for {candidate_id}: {sanitize_api_error(error)}") from error
            source_range = _plan_source_range(parsed)
            plans.append({
                "candidate_id": candidate_id,
                "status": "completed",
                "plan": parsed.model_dump(mode="json"),
                "requested_index": index,
                "production_plan_id": parsed.plan_id,
                "source_start_seconds": source_range[0] if source_range else None,
                "source_end_seconds": source_range[1] if source_range else None,
            })
        first_plan = plans[0]["plan"]
        timeline = first_plan["timeline"]
        production = {
            "enabled": True,
            "status": "completed",
            "items": plans,
            "production_plan": first_plan,
            "segments": first_plan["segments"],
            "estimated_duration": timeline["estimated_duration_seconds"],
            "dialogue_count": timeline["dialogue_count"],
            "narration_count": timeline["narration_count"],
            "pause_count": timeline["pause_count"],
            "timeline_version": timeline["timeline_version"],
            "production_note": "Approved Draft ProductionPlan reused without analysis or draft reassembly.",
        }
        analysis_work = Path(analysis.work_directory).resolve()
        root_work = (self.root / "work").resolve()
        if not analysis_work.is_relative_to(root_work):
            raise ClipEngineError("Analysis work reference is outside this engine workspace.")

        def load_reference(name: str) -> dict[str, Any]:
            raw = analysis.references.get(name)
            path = Path(str(raw or "")).resolve()
            if not raw or not path.is_relative_to(analysis_work) or not path.is_file():
                raise ClipEngineError(f"Approved draft is missing a safe {name} analysis reference.")
            value = read_json(path, None)
            if not isinstance(value, dict):
                raise ClipEngineError(f"Approved draft has corrupted {name} analysis data.")
            return value

        source_data = load_reference("source")
        metadata = load_reference("metadata")
        transcript = load_reference("transcript")
        visual_analysis = load_reference("visual_analysis")
        content_profile = load_reference("content_profile")
        content_map = load_reference("content_map")
        story_units = load_reference("story_units")
        coverage_map = load_reference("coverage_map")
        recommendation = load_reference("clip_count_recommendation")
        candidate_data_path = Path(analysis.candidate_data_ref).resolve()
        candidate_data = read_json(candidate_data_path, {}) if candidate_data_path.is_file() else {}
        final_scored = [
            scored_from_dict(item) for item in candidate_data.get("candidates", []) if isinstance(item, dict)
        ] if isinstance(candidate_data, dict) else []
        selected_ids = set(self.selected_candidate_ids)
        tracker.start("approved_draft_handoff", _hash({"draft": draft.draft_id, "selected": self.selected_candidate_ids}), cache_hit=True)
        tracker.finish("approved_draft_handoff")
        tts = self._run_tts(tracker, production, work_directory, output_directory)
        audio = self._run_audio(
            tracker, production, tts, source, transcript, work_directory, output_directory,
            Path(str(metadata["audio_path"])) if metadata.get("audio_path") else None,
        )
        production_render = self._run_production_render(
            tracker, production, audio, source, transcript, work_directory, output_directory, visual_analysis,
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
        terminal = build_terminal_state(len(plans), outputs, candidate_flow, delivery_required=True)
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
            report_path, source_data, metadata, self.config, tracker.data, len(plans), len(final_scored),
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
                "enabled": True, "profile": content_profile, "content_map": content_map,
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
        )
        manifest = write_run_manifest(
            output_directory / "manifest.json", run_id=self.run_id, source=source_data,
            started_at=self.started_at, requested_clip_count=len(plans), production_render=production_render,
            results=registry, run_directory=output_directory, project_id=self.project_id,
            content_understanding={
                "analysis_id": analysis.analysis_id, "draft_id": draft.draft_id,
                "draft_artifact_ref": str(self.draft_artifact_path), "selected_candidate_ids": list(self.selected_candidate_ids),
            }, terminal=terminal,
        )
        report["run"]["manifest_path"] = str(output_directory / "manifest.json")
        report["run"]["finished_at"] = manifest["finished_at"]
        write_json(report_path, report)
        return PipelineResult(
            work_directory=work_directory, output_directory=output_directory, report_path=report_path,
            selected_clips=len(plans), output_files=outputs, warnings=self.warnings,
            terminal_status=terminal["status"], error_code=terminal.get("error_code"),
            analysis_id=analysis.analysis_id, draft_path=self.draft_artifact_path, draft_id=draft.draft_id,
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
            source = Source(str(old["id"]), old_path, str(old["display_name"]), str(old["origin"]), bool(old.get("downloaded"))) if old and old_path.is_file() else url_source(url, work_directory / "download")
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
    ) -> dict[str, Any]:
        if stage in INTELLIGENCE_STAGES:
            fingerprint = {"engine_version": INTELLIGENCE_ENGINE_VERSION, "input": fingerprint}
        cache_key = _hash(fingerprint)
        cache = cache_tracker or tracker
        if cache.completed(stage, artifact, cache_key):
            # Keep the per-run report truthful even when the source-level
            # cache supplied the artifact.  The cache index and the run state
            # are different files and both writes are process-locked by
            # write_json, so runs never replace one another's state.json.
            tracker.start(stage, cache_key, cache_hit=True)
            tracker.finish(stage)
            return read_json(artifact, {})
        tracker.start(stage, cache_key, cache_hit=False)
        if cache is not tracker:
            cache.start(stage, cache_key)
        try:
            data = action()
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
        if self.no_ai_rerank or not self.config.ai_reranking.enabled or self.config.virality.enabled:
            reason = "virality_code_owned" if self.config.virality.enabled else "disabled"
            data = {"candidates": [item.to_dict() for item in local_rank(candidates)], "ai": _local_ai_usage("disabled"), "ai_reranking_used": False, "ai_fallback_used": False, "selection_mode": "local"}
            data["ai"]["reason"] = reason
            write_json(path, data)
            return data
        try:
            semantic, usage = get_scorer(self.config, self.mock_ai).score(short_candidates, transcript)
            ai_ok = not usage.get("api_errors")
        except Exception as error:
            semantic, usage, ai_ok = [], _local_ai_usage("fallback", [sanitize_api_error(error)]), False
        data = {
            "candidates": [item.to_dict() for item in merge_ai_ranking(candidates, semantic, ai_ok)],
            "ai": usage,
            "ai_reranking_used": ai_ok,
            "ai_fallback_used": not ai_ok,
            "selection_mode": "ai-reranked" if ai_ok else "local-fallback",
        }
        write_json(path, data)
        return data

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

    def _final_selection(self, scored: list, path: Path, content_map: dict[str, Any] | None = None) -> dict[str, Any]:
        selected = select_clips(scored, self.config, content_map)
        requested = min(self.config.max_clips, self.config.ai_reranking.final_clip_count)
        warnings: list[str] = []
        if len(selected) < requested:
            warnings.append(
                f"Найдено только {len(selected)} достаточно разных сильных фрагмента из запрошенных {requested}."
            )
        data = {
            "policy_version": "coverage-aware-v1" if content_map is not None else "temporal-diversity-v2",
            "candidates": [item.to_dict() for item in scored],
            "selected_ids": [item.candidate.id for item in selected],
            "requested_count": requested,
            "warnings": warnings,
            "coverage": build_coverage_map(content_map, scored, selected, self.config) if content_map is not None else {},
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

    def _build_production_plans(
        self, tracker: StageTracker, transformation: dict[str, Any],
        work_directory: Path, output_directory: Path,
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
            final = transformation_item.get("final_script", {}) if isinstance(transformation_item, dict) else {}
            candidate_id = str(final.get("candidate_id") or transformation_item.get("candidate_id") or f"candidate-{index:03d}")
            suffix = safe_name(candidate_id, f"clip-{index:02d}")
            artifact = work_directory / f"production-plan-{suffix}.json"
            cache_key = _hash({
                "version": PRODUCTION_PLAN_VERSION,
                "final_script": final,
                "source_context": transformation_item.get("source_context", {}),
                "semantic": transformation_item.get("semantic_representation", {}),
                "production": self.config.production,
            })
            stage_name = f"production_plan:{candidate_id}"
            use_cache = self.config.production.cache_enabled and tracker.completed(stage_name, artifact, cache_key)
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
                except Exception as error:
                    tracker.invalidate("Повреждён production plan cache.", (stage_name,))
                    use_cache = False
                    plan_data = {}
                else:
                    outcomes.append({"status": "completed", "candidate_id": candidate_id, "plan": plan.model_dump(mode="json"), "cache_hit": True})
            if not use_cache:
                tracker.start(stage_name, cache_key)
                try:
                    plan = build_production_plan(transformation_item, self.config.production)
                except (ProductionPlanError, ValueError) as error:
                    tracker.finish(stage_name, "failed", str(error))
                    outcomes.append({"status": "failed", "candidate_id": candidate_id, "error": str(error), "cache_hit": False})
                    continue
                plan_data = plan.model_dump(mode="json")
                write_json(artifact, plan_data)
                tracker.finish(stage_name)
                outcomes.append({"status": "completed", "candidate_id": candidate_id, "plan": plan_data, "cache_hit": False})
            outcome = outcomes[-1]
            if outcome.get("status") == "completed":
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
                    continue
                seen_candidate_ids.add(candidate_id)
                seen_plan_ids.add(plan.plan_id)
                if source_range is not None:
                    seen_source_ranges.append(source_range)
                outcome.update({
                    "requested_index": index,
                    "production_plan_id": plan.plan_id,
                    "source_start_seconds": source_range[0] if source_range else None,
                    "source_end_seconds": source_range[1] if source_range else None,
                })
                artifacts.extend(self._write_production_artifacts(output_directory, suffix, index, outcome["plan"]))
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
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            index = int(item.get("requested_index") or default_index)
            try:
                plan = ProductionPlan.model_validate(plan_data)
            except Exception as error:
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "error": sanitize_api_error(error)})
                continue
            allowed, reason = tts_eligibility(plan)
            if not allowed:
                outcomes.append({
                    "candidate_id": candidate_id, "status": "skipped", "reason": reason,
                    "tts_invoked": False, "estimated_cost": 0.0, "actual_cost": 0.0,
                })
                continue
            eligible.append((index, candidate_id, plan, item))
        if not eligible:
            reason = str(outcomes[0].get("reason", "no_eligible_narration")) if outcomes else "no_eligible_narration"
            tracker.skip("tts_generation", f"TTS skipped: {reason}.")
            return {
                "enabled": True, "status": "skipped", "reason": reason,
                "tts_invoked": False, "estimated_cost": 0.0, "actual_cost": 0.0,
                "items": outcomes,
            }
        for index, candidate_id, plan, plan_item in eligible:
            candidate_output = _candidate_output_directory(output_directory, candidate_id, index)
            stage_name = f"tts_generation:{plan.plan_id}"
            tracker.start(stage_name, _hash({"plan": plan.plan_id, "tts": self.config.tts, "recompute": self.recompute_tts}))
            try:
                result = TTSService(self.root, self.config).generate(
                    plan, work_directory, candidate_output, force_recompute=self.recompute_tts,
                )
            except TTSError as error:
                safe = sanitize_api_error(error)
                tracker.finish(stage_name, "failed", safe)
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "error": safe})
                self.errors.append(f"tts:{candidate_id}: {safe}")
                continue
            tracker.finish(stage_name, "completed" if result.status in {"completed", "partial", "fallback"} else result.status)
            report = tts_report_section(result)
            outcomes.append({
                "candidate_id": candidate_id, "status": result.status,
                "output_directory": str(candidate_output), "report": report,
                "tts_invoked": bool(report.get("tts_invoked", True)),
                **_production_item_identity(plan_item),
            })
            self.warnings.extend(result.warnings)
            self.errors.extend([f"tts:{candidate_id}: {entry.message}" for entry in result.api_errors])
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
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            index = int(item.get("requested_index") or default_index)
            try:
                plan = ProductionPlan.model_validate(plan_data)
            except Exception as error:
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "error": sanitize_api_error(error)})
                continue
            tts_allowed, _reason = tts_eligibility(plan)
            tts_item = tts_items.get(candidate_id)
            if tts_allowed and (not tts_item or tts_item.get("status") not in {"completed", "partial", "fallback"}):
                outcomes.append({"candidate_id": candidate_id, "status": "skipped", "reason": "tts_unavailable"})
                continue
            candidate_output_value = tts_item.get("output_directory") if isinstance(tts_item, dict) else None
            candidate_output = Path(str(candidate_output_value)) if candidate_output_value else _candidate_output_directory(output_directory, candidate_id, index)
            stage_name = f"audio_composition:{plan.plan_id}"
            tracker.start(stage_name, _hash({
                "plan": plan.plan_id,
                "audio": self.config.audio_composition,
                "audio_mode": plan.audio_mode,
                "tts_result": _file_fingerprint(candidate_output / "tts" / "tts-result.json") if tts_allowed else None,
                "recompute": self.recompute_audio,
            }))
            try:
                project = AudioCompositionService(self.root, self.config).compose(
                    plan, source, transcript, read_json(candidate_output / "tts" / "tts-result.json", {}) if tts_allowed else None,
                    work_directory, candidate_output, force_recompute=self.recompute_audio,
                    prepared_source_audio_path=prepared_source_audio_path,
                )
            except AudioCompositionError as error:
                safe = sanitize_api_error(error)
                tracker.finish(stage_name, "failed", safe)
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "error": safe})
                self.errors.append(f"audio:{candidate_id}: {safe}")
                continue
            tracker.finish(stage_name, "completed" if project.status in {"completed", "partial"} else project.status)
            outcomes.append({
                "candidate_id": candidate_id, "status": project.status,
                "output_directory": str(candidate_output), "report": audio_report_section(project),
                **_production_item_identity(item),
            })
            self.warnings.extend(project.warnings)
            self.errors.extend([f"audio:{candidate_id}: {entry}" for entry in project.errors])
        return _multi_stage_report("audio", outcomes)

    def _run_production_render(
        self, tracker: StageTracker, production: dict[str, Any], audio: dict[str, Any], source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path, visual_analysis: dict[str, Any] | None = None,
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
        audio_items = {str(item.get("candidate_id")): item for item in audio.get("items", []) if isinstance(item, dict)}
        outcomes: list[dict[str, Any]] = []
        for default_index, item in enumerate(plan_items, start=1):
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            identity = _production_item_identity(item)
            requested_index = int(identity.get("requested_index") or default_index)
            audio_item = audio_items.get(candidate_id)
            if not audio_item or audio_item.get("status") not in {"completed", "partial"}:
                outcomes.append({"candidate_id": candidate_id, "status": "skipped", "reason": "audio_unavailable", **identity})
                continue
            candidate_output = Path(str(audio_item["output_directory"]))
            try:
                plan = ProductionPlan.model_validate(plan_data)
                audio_project = AudioProject.model_validate(read_json(candidate_output / "audio" / "audio-project.json", {}))
            except Exception as error:
                safe = sanitize_api_error(error)
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "errors": [safe], **identity})
                self.errors.append(f"production_render:{candidate_id}: {safe}")
                continue
            report = self._compose_production_render(
                tracker, plan, audio_project, source, transcript, work_directory, candidate_output,
                raise_on_error=False, visual_analysis=visual_analysis,
            )
            output_file = str(report.get("output_file") or "")
            if report.get("status") in {"completed", "warning"} and output_file:
                canonical = self._publish_run_result(Path(output_file), output_directory, requested_index)
                report = dict(report)
                report["intermediate_output_file"] = output_file
                report["output_file"] = str(canonical)
                output_file = str(canonical)
            outcomes.append({
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
                **identity,
            })
        return _multi_stage_report("production_render", outcomes)

    def _compose_production_render(
        self, tracker: StageTracker, plan: ProductionPlan, audio_project: AudioProject, source: Source,
        transcript: dict[str, Any], work_directory: Path, output_directory: Path, raise_on_error: bool, visual_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage_name = f"production_render:{plan.plan_id}"
        tracker.start(stage_name, _hash({
            "plan": plan.plan_id, "audio_project": audio_project.project_id,
            "mixed_audio": _file_fingerprint(Path(audio_project.mix.mixed_audio_path or "")),
            "config": self.config.production_render, "recompute": self.recompute_production_render,
        }))
        try:
            project = VideoCompositionService(self.root, self.config).compose(
                plan, audio_project, source, transcript, work_directory, output_directory, visual_analysis=visual_analysis,
                force_recompute=self.recompute_production_render,
            )
        except ProductionRenderError as error:
            safe = sanitize_api_error(error)
            tracker.finish(stage_name, "failed", safe)
            self.errors.append(f"production_render: {safe}")
            if raise_on_error:
                raise ProductionRenderError(f"Production render не завершён: {safe}") from error
            return {"enabled": True, "status": "failed", "errors": [safe], "ai_called": False}
        tracker.finish(stage_name, "completed" if project.status in {"completed", "warning"} else project.status)
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
            "content_fingerprint": _render_content_fingerprint(Path(str(report.get("output_file") or "")), report),
            "primary": True,
        })
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
        write_json(report_path, existing)
        write_run_manifest(
            output_directory / "manifest.json", run_id=self.run_id, source=source.to_dict(),
            started_at=self.started_at or utc_now(), requested_clip_count=1,
            production_render=production_render, results=registry, run_directory=output_directory, project_id=self.project_id,
        )
        output_files = [path for path in result_paths(registry, output_directory) if path.is_file()]
        return PipelineResult(
            work_directory, output_directory, report_path,
            int(existing.get("selected_clips_count", 0) or 0), output_files, self.warnings,
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
            candidate = scored.candidate
            context = build_source_context(
                source, metadata, candidate, transcript, transcript_features, audio_features,
                scenes, self.config.transformation,
            )
            suffix = safe_name(candidate.id, f"clip-{index:02d}")
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
            stage_name = f"transformation_result:{candidate.id}"
            use_cache = self.config.transformation.cache_enabled and tracker.completed(stage_name, artifact, cache_key)
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
                        f"Transformation cache for {candidate.id} was invalidated by FinalScript contract validation."
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
                self._record_transformation_substages(tracker, candidate.id, cache_key, outcome)
            outcome["transformation_fingerprint"] = cache_key
            outcomes.append(outcome)
            outcome_artifacts = self._write_transformation_artifacts(output_directory, suffix, index, outcome)
            artifacts.extend(outcome_artifacts)
            usage = outcome.get("ai_usage", {})
            raw_errors = usage.get("api_errors", []) if isinstance(usage, dict) else []
            if raw_errors:
                self.errors.extend([f"transformation: {sanitize_api_error(value)}" for value in raw_errors])
            normalization = outcome.get("normalization", {}) if isinstance(outcome.get("normalization"), dict) else {}
            for warning in normalization.get("warnings", []) if isinstance(normalization.get("warnings"), list) else []:
                self.warnings.append(f"Transformation {candidate.id}: {warning}")
            if outcome.get("fallback", {}).get("used"):
                reason = outcome.get("fallback", {}).get("reason")
                self.warnings.append(
                    "Local-only transformation used conservative fallback."
                    if reason == "ai_disabled"
                    else "AI transformation failed -> local fallback used."
                )
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
        for stage in ("transcription", *INTELLIGENCE_STAGES, "production_plan"): tracker.skip(stage, warning)
        report_path = output_directory / "report.json"
        tracker.start("report"); tracker.finish("report")
        make_report(report_path, source, metadata, self.config, tracker.data, 0, 0, [], self.warnings, self.errors, _local_ai_usage("not-called"), False, False, clip_intelligence={"version": "1.6", "selection_mode": "no-audio"}, content_transformation={"enabled": bool(self.config.transformation.enabled), "status": "skipped", "reason": "no-audio"}, production_plan={"enabled": bool(self.config.production.enabled), "status": "skipped", "reason": "no-audio"}, audio={"enabled": bool(self.config.audio_composition.enabled), "status": "skipped", "reason": "no-audio"}, production_render={"enabled": bool(self.config.production_render.enabled), "status": "skipped", "reason": "no-audio"})
        return PipelineResult(work_directory, output_directory, report_path, 0, [], self.warnings)


def _hash(value: Any) -> str:
    def default(item: Any) -> Any:
        if is_dataclass(item):
            return asdict(item)
        return str(item)
    return stable_text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False, default=default))


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
        duplicate_of = next((chosen for chosen in accepted if _transformation_duplicate(outcome, chosen)), None)
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
    if not result and isinstance(production, dict) and isinstance(production.get("production_plan"), dict):
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
            items.append({
                "candidate_id": candidate_id,
                "outcome": "failed",
                "reason": "production_plan_failed",
                "message": _outcome_detail(plan_item) if plan_item else "ProductionPlan не был создан.",
            })
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
    requested_clip_count: int, output_files: list[Path], candidate_flow: dict[str, Any], *, delivery_required: bool,
) -> dict[str, Any]:
    """Return a terminal contract after reportable artifacts already exist."""

    produced = len(output_files)
    details = {
        key: int(candidate_flow.get(key, 0) or 0)
        for key in ("found", "selected", "transformed", "production_plans", "render_attempts", "rendered", "rejected", "failed")
    }
    if delivery_required and requested_clip_count > 0 and produced == 0:
        return {
            "status": "failed",
            "error_code": NO_RENDERABLE_CLIPS,
            "message": NO_RENDERABLE_CLIPS_MESSAGE,
            "requested_clip_count": requested_clip_count,
            "produced_clips_count": produced,
            "candidate_counts": details,
        }
    if produced and details["failed"]:
        return {
            "status": "completed_with_warnings",
            "error_code": None,
            "message": "Часть кандидатов не дошла до финального рендера.",
            "requested_clip_count": requested_clip_count,
            "produced_clips_count": produced,
            "candidate_counts": details,
        }
    return {
        "status": "completed",
        "error_code": None,
        "message": "",
        "requested_clip_count": requested_clip_count,
        "produced_clips_count": produced,
        "candidate_counts": details,
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


def _outcome_detail(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return "Неизвестная причина."
    error = str(item.get("error") or "").strip()
    if error:
        return error
    validation = item.get("validation", {})
    if isinstance(validation, dict):
        errors = validation.get("errors", [])
        if isinstance(errors, list) and errors:
            return "; ".join(str(value) for value in errors[:3] if str(value))
    errors = item.get("errors", [])
    if isinstance(errors, list) and errors:
        return "; ".join(str(value) for value in errors[:3] if str(value))
    reason = str(item.get("reason") or "").strip()
    return reason or "Неизвестная причина."


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


def _prepared_source_audio_path(work_directory: Path) -> Path | None:
    """Return only a local source-derived audio artifact; never download or transcribe here."""

    metadata = read_json(work_directory / "metadata.json", {})
    value = metadata.get("audio_path") if isinstance(metadata, dict) else None
    path = Path(str(value)) if value else work_directory / "audio_16khz_mono.wav"
    return path if path.is_file() else None


def _write(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    write_json(path, data); return data


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


class _UnavailableTransformer:
    """Turns a missing key/dependency into the normal non-blocking fallback path."""

    name = "unavailable"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def transform_compact(self, context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        raise TransformationProviderError(sanitize_api_error(self.error))
