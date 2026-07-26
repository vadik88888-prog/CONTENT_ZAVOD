from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

from app.ai import get_scorer, get_transformer, sanitize_api_error
from app.audio_features import analyse_audio
from app.audio_models import AudioProject
from app.audio_service import AudioCompositionService, audio_report_section
from app.config import AppConfig
from app.content_transformation import (
    TRANSFORMATION_ENGINE_VERSION,
    run_content_transformation,
    validate_transformation_outcome,
)
from app.errors import AudioCompositionError, ClipEngineError, ProductionPlanError, ProductionRenderError, StageError, TTSError, TransformationProviderError
from app.intelligence import intelligence_summary, local_rank, merge_ai_ranking, shortlist
from app.intelligence_candidates import generate_candidates_with_stats
from app.local_scoring import score_candidates
from app.media import prepare_media
from app.models import Candidate, candidate_from_dict, scored_from_dict
from app.rendering import render_clip
from app.reporting import make_report
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
from app.utils import read_json, safe_name, stable_text_hash, utc_now, write_json
from app.video_composition import VideoCompositionService, production_render_report_section
from app.visual_analysis import analyse_video_subjects


INTELLIGENCE_STAGES = (
    "transcript_features", "audio_features", "scene_detection", "candidates_v2",
    "local_scoring", "shortlist", "ai_ranking", "final_selection", "visual_analysis", "render", "report",
)
INTELLIGENCE_ENGINE_VERSION = "1.6.3"
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


class StageTracker:
    def __init__(self, state_path: Path) -> None:
        self.path = state_path
        self.data = read_json(state_path, {"created_at": utc_now(), "stages": {}})
        self.data.setdefault("stages", {})

    def completed(self, name: str, artifact: Path, cache_key: str | None = None) -> bool:
        stage = self.data["stages"].get(name, {})
        return (
            stage.get("status") == "completed"
            and artifact.exists()
            and (cache_key is None or stage.get("cache_key") == cache_key)
        )

    def start(self, name: str, cache_key: str | None = None) -> None:
        self.data["stages"][name] = {
            "status": "running", "started_at": utc_now(), "_started": time.perf_counter(),
            "cache_key": cache_key,
        }
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
        safe = json.loads(json.dumps(self.data))
        for stage in safe.get("stages", {}).values():
            stage.pop("_started", None)
        write_json(self.path, safe)


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
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def run(self, input_path: str | None = None, url: str | None = None) -> PipelineResult:
        validate_source_arguments(input_path, url)
        source, work_directory, output_directory = self._prepare_source(input_path, url)
        tracker = StageTracker(work_directory / "state.json")
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
        source_data = self._source_stage(tracker, work_directory / "source.json", source)
        metadata = self._cached(
            tracker, "metadata", work_directory / "metadata.json", {"source": source.id},
            lambda: prepare_media(source.path, work_directory),
        )
        if not metadata.get("audio_path"):
            return self._finish_without_audio(tracker, source_data, metadata, work_directory, output_directory)
        transcript = self._cached(
            tracker, "transcription", work_directory / "transcript.json",
            {"source": source.id, "whisper": self.config.whisper_model, "language": self.config.language, "device": self.config.device},
            lambda: transcribe(Path(str(metadata["audio_path"])), source.id, float(metadata["duration"]), self.config, work_directory / "transcript.json"),
        )
        transcript_features = self._cached(
            tracker, "transcript_features", work_directory / "transcript_features.json",
            {"transcript": _hash(transcript), "settings": self.config.transcript_features},
            lambda: _write(work_directory / "transcript_features.json", analyse_transcript(transcript, self.config.transcript_features)),
        )
        audio_features = self._cached(
            tracker, "audio_features", work_directory / "audio_features.json",
            {"audio": str(metadata["audio_path"]), "settings": self.config.audio_analysis},
            lambda: _write(work_directory / "audio_features.json", analyse_audio(Path(str(metadata["audio_path"])), self.config.audio_analysis)),
        )
        scenes = self._cached(
            tracker, "scene_detection", work_directory / "scene_boundaries.json",
            {"source": source.id, "settings": self.config.scene_detection},
            lambda: _write(work_directory / "scene_boundaries.json", detect_scene_boundaries(source.path, float(metadata["duration"]), self.config.scene_detection)),
        )
        visual_analysis = self._cached(
            tracker, "visual_analysis", work_directory / "visual_analysis.json",
            {"source": source.id, "duration": metadata.get("duration"), "enabled": self.config.optional_visual_features, "model": self.config.ai.model},
            lambda: _write(work_directory / "visual_analysis.json", analyse_video_subjects(source.path, float(metadata.get("duration") or 0), self.config)),
        )
        raw_candidates = self._cached(
            tracker, "candidates_v2", work_directory / "candidates_v2.json",
            {"transcript_features": _hash(transcript_features), "audio": _hash(audio_features), "scenes": _hash(scenes), "settings": self.config.candidate_generation},
            lambda: _write_generated_candidates(work_directory / "candidates_v2.json", generate_candidates_with_stats(transcript, transcript_features, audio_features, scenes, self.config.candidate_generation)),
        )
        # Compatibility artifact retained for existing users of the pre-1.6 cache layout.
        write_json(work_directory / "candidates.raw.json", raw_candidates)
        candidates = [candidate_from_dict(item) for item in raw_candidates.get("candidates", [])]
        local_data = self._cached(
            tracker, "local_scoring", work_directory / "candidates.local.json",
            {"candidates": _hash(raw_candidates), "settings": self.config.scoring},
            lambda: _write_candidates(work_directory / "candidates.local.json", score_candidates(candidates, audio_features, scenes, self.config.scoring)),
        )
        candidates = [candidate_from_dict(item) for item in local_data.get("candidates", [])]
        shortlist_data = self._cached(
            tracker, "shortlist", work_directory / "shortlist.json",
            {"candidates": _hash(local_data), "size": self.config.ai_reranking.shortlist_size},
            lambda: _write_candidates(work_directory / "shortlist.json", shortlist(candidates, self.config.ai_reranking.shortlist_size)),
        )
        short_candidates = [candidate_from_dict(item) for item in shortlist_data.get("candidates", [])]
        ai_data = self._cached(
            tracker, "ai_ranking", work_directory / "ai_ranking.json",
            {"shortlist": _hash(shortlist_data), "ai": self.config.ai, "reranking": self.config.ai_reranking, "mock": self.mock_ai, "disabled": self.no_ai_rerank},
            lambda: self._ai_rerank(candidates, short_candidates, transcript, work_directory / "ai_ranking.json"),
        )
        scored = [scored_from_dict(item) for item in ai_data.get("candidates", [])]
        final_data = self._cached(
            tracker, "final_selection", work_directory / "final_selection.json",
            {"scored": _hash(ai_data), "threshold": self.config.score_threshold, "overlap": self.config.overlap_threshold, "distance": self.config.min_selected_clip_distance_seconds, "limit": self.config.ai_reranking.final_clip_count},
            lambda: self._final_selection(scored, work_directory / "final_selection.json"),
        )
        selected_ids = set(final_data.get("selected_ids", []))
        final_scored = [scored_from_dict(item) for item in final_data.get("candidates", [])]
        write_json(
            work_directory / "candidates.scored.json",
            {"candidates": [item.to_dict() for item in final_scored], "ai": ai_data.get("ai", {})},
        )
        selected = [item for item in final_scored if item.candidate.id in selected_ids]
        transformation = self._transform_selected(
            tracker, source_data, metadata, selected, transcript, transcript_features,
            audio_features, scenes, work_directory, output_directory,
        )
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
        render_data = (
            self._skip_render_for_production_plan(tracker, work_directory / "render.json")
            if self.production_plan_only
            else self._cached(
                tracker, "render", work_directory / "render.json",
                {"selected": [(item.candidate.id, item.score) for item in selected], "render": self.config.render_mode, "dimensions": [self.config.output_width, self.config.output_height], "encoder": self.config.encoder_preference},
                lambda: self._render(source, transcript, selected, output_directory, work_directory / "render.json"),
            )
        )
        outputs = [Path(value) for value in render_data.get("output_files", []) if Path(value).is_file()]
        production_outputs = production_render.get("output_files", []) if isinstance(production_render, dict) else []
        for value in production_outputs if isinstance(production_outputs, list) else []:
            path = Path(str(value))
            if path.is_file() and path not in outputs:
                outputs.append(path)
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
        tracker.start("report", _hash({"final": final_data, "render": render_data, "ai": ai_usage, "transformation": transformation, "production": production, "tts": tts, "audio": audio, "production_render": production_render}))
        tracker.finish("report")
        make_report(
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
            },
            content_transformation=transformation,
            production_plan=production,
            tts=tts,
            audio=audio,
            production_render=production_render,
        )
        return PipelineResult(work_directory, output_directory, report_path, len(selected), outputs, self.warnings)

    def _source_stage(self, tracker: StageTracker, artifact: Path, source: Source) -> dict[str, Any]:
        stored = read_json(artifact, {})
        if tracker.completed("source", artifact) and stored.get("id") == source.id and stored.get("path") == str(source.path):
            return stored
        tracker.start("source", source.id)
        data = source.to_dict()
        write_json(artifact, data)
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
        output_directory = self.root / "output" / run_key
        work_directory.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=True, exist_ok=True)
        return source, work_directory, output_directory

    def _cached(self, tracker: StageTracker, stage: str, artifact: Path, fingerprint: Any, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if stage in INTELLIGENCE_STAGES:
            fingerprint = {"engine_version": INTELLIGENCE_ENGINE_VERSION, "input": fingerprint}
        cache_key = _hash(fingerprint)
        if tracker.completed(stage, artifact, cache_key):
            return read_json(artifact, {})
        tracker.start(stage, cache_key)
        try:
            data = action()
        except ClipEngineError as error:
            tracker.finish(stage, "failed", str(error))
            raise
        except Exception as error:
            message = f"Непредвиденная ошибка этапа {stage}: {error}"
            tracker.finish(stage, "failed", message)
            raise StageError(message) from error
        tracker.finish(stage)
        return data

    def _ai_rerank(self, candidates: list[Candidate], short_candidates: list[Candidate], transcript: dict[str, Any], path: Path) -> dict[str, Any]:
        if self.no_ai_rerank or not self.config.ai_reranking.enabled:
            data = {"candidates": [item.to_dict() for item in local_rank(candidates)], "ai": _local_ai_usage("disabled"), "ai_reranking_used": False, "ai_fallback_used": False, "selection_mode": "local"}
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

    def _final_selection(self, scored: list, path: Path) -> dict[str, Any]:
        selected = select_clips(scored, self.config)
        data = {"candidates": [item.to_dict() for item in scored], "selected_ids": [item.candidate.id for item in selected]}
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
        items = transformation.get("items", []) if isinstance(transformation, dict) else []
        if not enabled:
            tracker.skip("production_plan", "Production Plan отключён конфигурацией.")
            return {"enabled": False, "status": "skipped", "reason": "disabled", "items": []}
        if not items:
            tracker.skip("production_plan", "Нет FinalScript для Production Plan.")
            return {"enabled": True, "status": "skipped", "reason": "no_final_script", "items": []}
        outcomes: list[dict[str, Any]] = []
        artifacts: list[str] = []
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
            if outcomes[-1].get("status") == "completed":
                artifacts.extend(self._write_production_artifacts(output_directory, suffix, index, outcomes[-1]["plan"]))
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
        for index, item in enumerate(plan_items, start=1):
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            try:
                plan = ProductionPlan.model_validate(plan_data)
            except Exception as error:
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "error": sanitize_api_error(error)})
                continue
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
            outcomes.append({"candidate_id": candidate_id, "status": result.status, "output_directory": str(candidate_output), "report": report})
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
        for item in plan_items:
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            tts_item = tts_items.get(candidate_id)
            if not tts_item or tts_item.get("status") not in {"completed", "partial", "fallback"}:
                outcomes.append({"candidate_id": candidate_id, "status": "skipped", "reason": "tts_unavailable"})
                continue
            try:
                plan = ProductionPlan.model_validate(plan_data)
            except Exception as error:
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "error": sanitize_api_error(error)})
                continue
            candidate_output = Path(str(tts_item["output_directory"]))
            stage_name = f"audio_composition:{plan.plan_id}"
            tracker.start(stage_name, _hash({
                "plan": plan.plan_id,
                "audio": self.config.audio_composition,
                "tts_result": _file_fingerprint(candidate_output / "tts" / "tts-result.json"),
                "recompute": self.recompute_audio,
            }))
            try:
                project = AudioCompositionService(self.root, self.config).compose(
                    plan, source, transcript, read_json(candidate_output / "tts" / "tts-result.json", {}),
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
            outcomes.append({"candidate_id": candidate_id, "status": project.status, "output_directory": str(candidate_output), "report": audio_report_section(project)})
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
        for item in plan_items:
            candidate_id, plan_data = item["candidate_id"], item["plan"]
            audio_item = audio_items.get(candidate_id)
            if not audio_item or audio_item.get("status") not in {"completed", "partial"}:
                outcomes.append({"candidate_id": candidate_id, "status": "skipped", "reason": "audio_unavailable"})
                continue
            candidate_output = Path(str(audio_item["output_directory"]))
            try:
                plan = ProductionPlan.model_validate(plan_data)
                audio_project = AudioProject.model_validate(read_json(candidate_output / "audio" / "audio-project.json", {}))
            except Exception as error:
                safe = sanitize_api_error(error)
                outcomes.append({"candidate_id": candidate_id, "status": "failed", "errors": [safe]})
                self.errors.append(f"production_render:{candidate_id}: {safe}")
                continue
            report = self._compose_production_render(
                tracker, plan, audio_project, source, transcript, work_directory, candidate_output,
                raise_on_error=False, visual_analysis=visual_analysis,
            )
            outcomes.append({"candidate_id": candidate_id, "status": report.get("status", "failed"), "output_directory": str(candidate_output), "report": report, "output_file": report.get("output_file")})
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
        self.warnings.extend(project.warnings)
        return report

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

        plan_path = output_directory / "production-plan.json"
        audio_path = output_directory / "audio" / "audio-project.json"
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
        existing["stages"] = tracker.data.get("stages", {})
        existing["warnings"] = [*existing.get("warnings", []), *self.warnings]
        existing["errors"] = [*existing.get("errors", []), *self.errors]
        write_json(report_path, existing)
        final_output = production_render.get("output_file") if isinstance(production_render, dict) else None
        output_files = [Path(final_output)] if final_output and Path(final_output).is_file() else []
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
                )
                self._record_transformation_substages(tracker, candidate.id, cache_key, outcome)
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


def _production_items(production: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = production.get("items", []) if isinstance(production, dict) else []
    result: list[dict[str, Any]] = []
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict) or item.get("status") != "completed" or not isinstance(item.get("plan"), dict):
            continue
        result.append({"candidate_id": str(item.get("candidate_id") or "candidate"), "plan": item["plan"]})
    # Old reports expose only the primary plan. Keep render-only and existing
    # cache layouts operational while new full runs fan out to every item.
    if not result and isinstance(production, dict) and isinstance(production.get("production_plan"), dict):
        result.append({"candidate_id": "primary", "plan": production["production_plan"]})
    return result


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
        primary["output_file"] = successful[0].get("output_file") or primary.get("output_file")
        primary["output_files"] = [item["output_file"] for item in successful if isinstance(item.get("output_file"), str)]
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
