from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.audio_models import (
    AUDIO_PROJECT_SCHEMA_VERSION,
    AUDIO_TIMELINE_VERSION,
    AudioExport,
    AudioMetadata,
    AudioMix,
    AudioProject,
    AudioTimeline,
    AudioTrack,
    AudioValidation,
    DialogueClip,
    DuckingConfig,
    EffectClip,
    LoudnessConfig,
    MusicClip,
    NarrationClip,
    SilenceClip,
)
from app.config import AppConfig
from app.errors import AudioCompositionError, ProductionPlanHandoffError
from app.production_models import DialogueSegment, NarrationSegment, PauseSegment, ProductionPlan, validate_audio_handoff
from app.sources import Source
from app.subprocess_utils import UTF8_REPLACE_TEXT
from app.tts_models import TTSSegmentResult
from app.utils import read_json, stable_file_hash, stable_text_hash, utc_now, write_bytes_atomic, write_json

class AudioCompositionService:
    """Compose audio from existing production decisions without touching video or scripts."""

    def __init__(self, root: Path, config: AppConfig) -> None:
        self.root = root.resolve()
        self.config = config

    def compose(
        self, plan: ProductionPlan, source: Source, transcript: dict[str, Any],
        tts_result: dict[str, Any] | None, work_directory: Path, output_directory: Path,
        force_recompute: bool = False, prepared_source_audio_path: Path | None = None,
    ) -> AudioProject:
        handoff_failure = validate_audio_handoff(plan)
        if handoff_failure is not None:
            raise ProductionPlanHandoffError(handoff_failure.code, handoff_failure.evidence)
        if not source.path.is_file():
            raise AudioCompositionError("Исходный media file для Audio Project не найден.")
        started_at = utc_now()
        output_root = output_directory / "audio"
        segments_root = output_root / "segments"
        cache_root = self.root / "work" / "audio-cache"
        temp_root = work_directory / "audio-compose-tmp"
        tts_by_segment = _tts_segments(tts_result)
        cache_hits = {"dialogue": 0, "narration": 0, "source_bed": 0}
        cache_misses = {"dialogue": 0, "narration": 0, "source_bed": 0}
        warnings: list[str] = []
        errors: list[str] = []
        timeline_clips: list[Any] = []
        narration_clips: list[NarrationClip] = []
        dialogue_clips: list[DialogueClip] = []
        cursor = 0.0
        order = 1

        for segment in plan.segments:
            if isinstance(segment, NarrationSegment):
                tts = tts_by_segment.get(segment.segment_id)
                if tts is None:
                    warnings.append(f"Narration {segment.segment_id} skipped: no valid TTS artifact.")
                    continue
                try:
                    normalized_path, narration_cached = self._normalize_narration(
                        tts, cache_root / "narration", segments_root / "narration", force_recompute,
                    )
                    cache_hits["narration"] += int(narration_cached)
                    cache_misses["narration"] += int(not narration_cached)
                    duration = _probe_wav(normalized_path, self.config.audio_composition.sample_rate)["duration"]
                    bed_path, bed_cached = self._source_bed(
                        segment, source, transcript, cache_root / "source-bed", temp_root, force_recompute,
                        prepared_source_audio_path,
                    )
                    if bed_path is not None:
                        cache_hits["source_bed"] += int(bed_cached)
                        cache_misses["source_bed"] += int(not bed_cached)
                    rendered_path = segments_root / "narration" / f"{segment.segment_id}-{stable_text_hash(str(normalized_path))[:12]}.wav"
                    ducked_path = None
                    if bed_path is not None and self.config.audio_composition.ducking_enabled:
                        self._duck_and_mix(normalized_path, bed_path, rendered_path, duration)
                        ducked_path = str(bed_path)
                    else:
                        _copy_atomic(normalized_path, rendered_path)
                    actual = _probe_wav(rendered_path, self.config.audio_composition.sample_rate)["duration"]
                    clip = NarrationClip(
                        clip_id=f"audio-narration-{segment.segment_id}", order=order,
                        production_segment_id=segment.segment_id,
                        timeline_start_seconds=cursor, timeline_end_seconds=round(cursor + actual, 3),
                        duration_seconds=actual, audio_file_path=str(rendered_path),
                        checksum=stable_file_hash(rendered_path), status="ready",
                        cache_key=tts.cache_key, tts_segment_id=segment.segment_id,
                        normalized_tts_path=str(normalized_path), source_bed_path=str(bed_path) if bed_path else None,
                        ducked_source_bed_path=ducked_path, loudness_normalized=True,
                        target_lufs=self.config.audio_composition.narration_target_lufs,
                        duck_level=self.config.audio_composition.duck_level,
                        attack_seconds=self.config.audio_composition.duck_attack_seconds,
                        release_seconds=self.config.audio_composition.duck_release_seconds,
                    )
                    timeline_clips.append(clip)
                    narration_clips.append(clip)
                    cursor = clip.timeline_end_seconds
                    order += 1
                except Exception as error:
                    warnings.append(f"Narration {segment.segment_id} skipped: {_safe_error(error)}")
            elif isinstance(segment, DialogueSegment):
                try:
                    dialogue_path, cached, cache_key = self._dialogue_audio(
                        segment, source, cache_root / "dialogue", segments_root / "dialogue", force_recompute,
                        prepared_source_audio_path,
                    )
                    cache_hits["dialogue"] += int(cached)
                    cache_misses["dialogue"] += int(not cached)
                    duration = _probe_wav(dialogue_path, self.config.audio_composition.sample_rate)["duration"]
                    clip = DialogueClip(
                        clip_id=f"audio-dialogue-{segment.segment_id}", order=order,
                        production_segment_id=segment.segment_id,
                        timeline_start_seconds=cursor, timeline_end_seconds=round(cursor + duration, 3),
                        duration_seconds=duration, audio_file_path=str(dialogue_path),
                        checksum=stable_file_hash(dialogue_path), status="ready", cache_key=cache_key,
                        fact_id=segment.fact_id, transcript_segment_id=segment.transcript_segment_id,
                        source_start_seconds=segment.source_start_seconds,
                        source_end_seconds=segment.source_end_seconds, speaker=segment.speaker,
                    )
                    timeline_clips.append(clip)
                    dialogue_clips.append(clip)
                    cursor = clip.timeline_end_seconds
                    order += 1
                except Exception as error:
                    warnings.append(f"Dialogue {segment.segment_id} skipped: {_safe_error(error)}")
            elif isinstance(segment, PauseSegment) and segment.estimated_duration_seconds > 0:
                pause_path = segments_root / "silence" / f"{segment.segment_id}.wav"
                _make_silence(pause_path, segment.estimated_duration_seconds, self.config.audio_composition.sample_rate)
                duration = _probe_wav(pause_path, self.config.audio_composition.sample_rate)["duration"]
                clip = SilenceClip(
                    clip_id=f"audio-silence-{segment.segment_id}", order=order,
                    production_segment_id=segment.segment_id,
                    timeline_start_seconds=cursor, timeline_end_seconds=round(cursor + duration, 3),
                    duration_seconds=duration, audio_file_path=str(pause_path), checksum=stable_file_hash(pause_path),
                    status="ready", pause_reason=segment.reason,
                )
                timeline_clips.append(clip)
                cursor = clip.timeline_end_seconds
                order += 1

        timeline = AudioTimeline(
            timeline_version=AUDIO_TIMELINE_VERSION, clips=timeline_clips,
            duration_seconds=cursor,
        )
        mixed_path = output_root / "mixed_audio.wav"
        if timeline_clips:
            _concat_audio([Path(str(clip.audio_file_path)) for clip in timeline_clips if clip.audio_file_path], mixed_path, self.config.audio_composition.sample_rate)
            mixed_probe = _probe_wav(mixed_path, self.config.audio_composition.sample_rate)
            validation = AudioValidation(
                status="warning" if warnings else "valid", checked_clip_count=len(timeline_clips),
                failed_clip_count=0, mix_duration_seconds=mixed_probe["duration"], messages=warnings,
            )
            mix_status = "partial" if warnings else "completed"
            export_status = "completed"
            checksum = stable_file_hash(mixed_path)
            byte_size = mixed_path.stat().st_size
        else:
            validation = AudioValidation(
                status="not_applicable", checked_clip_count=0, failed_clip_count=0,
                messages=[*warnings, "No renderable audio clips were available."],
            )
            mix_status = "skipped"
            export_status = "skipped"
            checksum = None
            byte_size = 0
        tracks = [
            AudioTrack(track_id="track-narration", track_type="narration", clips=narration_clips, status="ready" if narration_clips else "empty"),
            AudioTrack(track_id="track-dialogue", track_type="dialogue", clips=dialogue_clips, status="ready" if dialogue_clips else "empty"),
            AudioTrack(
                track_id="track-music", track_type="music", clips=[], status="placeholder",
            ),
            AudioTrack(
                track_id="track-effects", track_type="effects", clips=[], status="placeholder",
            ),
        ]
        project_id = f"audio-{plan.plan_id}-{stable_text_hash(json.dumps([clip.clip_id for clip in timeline_clips]))[:12]}"
        project = AudioProject(
            project_id=project_id,
            audio_mode=plan.audio_mode,
            status=_project_status(mix_status, narration_clips, plan, warnings),
            timeline=timeline, tracks=tracks,
            mix=AudioMix(
                mixed_audio_path=str(mixed_path) if mixed_path.is_file() else None, checksum=checksum,
                duration_seconds=validation.mix_duration_seconds or 0,
                sample_rate=self.config.audio_composition.sample_rate,
                ducking=DuckingConfig(
                    enabled=self.config.audio_composition.ducking_enabled,
                    attack_seconds=self.config.audio_composition.duck_attack_seconds,
                    release_seconds=self.config.audio_composition.duck_release_seconds,
                    duck_level=self.config.audio_composition.duck_level,
                    preserve_original_events=self.config.audio_composition.preserve_original_events,
                    policy="audible_source_bed_with_smooth_ducking",
                ),
                loudness=LoudnessConfig(
                    narration_target_lufs=self.config.audio_composition.narration_target_lufs,
                    narration_true_peak_db=self.config.audio_composition.narration_true_peak_db,
                    narration_lra=self.config.audio_composition.narration_lra,
                    normalized_narration_count=len(narration_clips),
                ),
                status=mix_status,
            ),
            export=AudioExport(
                sample_rate=self.config.audio_composition.sample_rate,
                path=str(mixed_path) if mixed_path.is_file() else None,
                byte_size=byte_size, status=export_status,
            ),
            metadata=AudioMetadata(
                schema_version=AUDIO_PROJECT_SCHEMA_VERSION, audio_project_id=project_id,
                production_plan_id=plan.plan_id, source_id=source.id,
                plan_reference=plan.reference(),
                source_media_path=str(source.path),
                audio_mode=plan.audio_mode,
                tts_result_path=_tts_result_path(tts_result),
                transcript_path=str(work_directory / "transcript.json") if (work_directory / "transcript.json").is_file() else None,
                created_at=started_at, completed_at=utc_now(),
            ),
            validation=validation,
            cache={
                "enabled": self.config.audio_composition.cache_enabled,
                "dialogue_hit_count": cache_hits["dialogue"], "dialogue_miss_count": cache_misses["dialogue"],
                "narration_hit_count": cache_hits["narration"], "narration_miss_count": cache_misses["narration"],
                "source_bed_hit_count": cache_hits["source_bed"], "source_bed_miss_count": cache_misses["source_bed"],
            },
            warnings=warnings, errors=errors,
        )
        return self._write_artifacts(project, output_root)

    def _normalize_narration(
        self, tts: TTSSegmentResult, cache_directory: Path, output_directory: Path,
        force_recompute: bool,
    ) -> tuple[Path, bool]:
        assert tts.artifact and tts.artifact.audio_file_path and tts.artifact.checksum
        source = Path(tts.artifact.audio_file_path)
        if not source.is_file():
            raise AudioCompositionError("TTS artifact path is unavailable.")
        key = stable_text_hash(json.dumps({
            "tts_checksum": tts.artifact.checksum, "rate": self.config.audio_composition.sample_rate,
            "target_lufs": self.config.audio_composition.narration_target_lufs,
            "true_peak": self.config.audio_composition.narration_true_peak_db,
            "lra": self.config.audio_composition.narration_lra,
            "version": self.config.audio_composition.engine_version,
        }, sort_keys=True))
        cache_path = cache_directory / f"{key}.wav"
        cached = False
        if self.config.audio_composition.cache_enabled and not force_recompute and _is_valid_wav(cache_path, self.config.audio_composition.sample_rate):
            cached = True
        else:
            filter_chain = (
                f"loudnorm=I={self.config.audio_composition.narration_target_lufs}:"
                f"LRA={self.config.audio_composition.narration_lra}:"
                f"TP={self.config.audio_composition.narration_true_peak_db}:linear=true"
            )
            _ffmpeg_to_wav(
                ["-i", str(source), "-vn", "-af", filter_chain, "-ac", "1", "-ar", str(self.config.audio_composition.sample_rate)],
                cache_path,
            )
        output_path = output_directory / f"{tts.segment_id}-{key[:12]}.wav"
        _copy_atomic(cache_path, output_path)
        return output_path, cached

    def _dialogue_audio(
        self, segment: DialogueSegment, source: Source, cache_directory: Path, output_directory: Path,
        force_recompute: bool, prepared_source_audio_path: Path | None,
    ) -> tuple[Path, bool, str]:
        duration = segment.source_end_seconds - segment.source_start_seconds
        if duration <= 0:
            raise AudioCompositionError("Dialogue mapping has no positive source duration.")
        key = _source_cache_key(source, segment.source_start_seconds, segment.source_end_seconds, "dialogue", self.config)
        cache_path = cache_directory / f"{key}.wav"
        metadata = read_json(cache_directory / f"{key}.json", {})
        cached = (
            self.config.audio_composition.cache_enabled and not force_recompute
            and _is_valid_wav(cache_path, self.config.audio_composition.sample_rate)
            and isinstance(metadata, dict) and metadata.get("checksum") == stable_file_hash(cache_path)
        )
        if not cached:
            _extract_source_wav(
                source.path, segment.source_start_seconds, duration, cache_path,
                self.config.audio_composition.sample_rate, prepared_source_audio_path,
            )
            write_json(cache_directory / f"{key}.json", {"checksum": stable_file_hash(cache_path), "duration": _probe_wav(cache_path, self.config.audio_composition.sample_rate)["duration"]})
        output_path = output_directory / f"{segment.segment_id}-{key[:12]}.wav"
        _copy_atomic(cache_path, output_path)
        return output_path, cached, key

    def _source_bed(
        self, segment: NarrationSegment, source: Source, transcript: dict[str, Any],
        cache_directory: Path, temporary_directory: Path, force_recompute: bool,
        prepared_source_audio_path: Path | None,
    ) -> tuple[Path | None, bool]:
        ranges = _narration_source_ranges(segment, transcript)
        if not ranges:
            return None, False
        paths: list[Path] = []
        all_cached = True
        for index, (start, end) in enumerate(ranges, start=1):
            key = _source_cache_key(source, start, end, "source-bed", self.config)
            path = cache_directory / f"{key}.wav"
            cached = self.config.audio_composition.cache_enabled and not force_recompute and _is_valid_wav(path, self.config.audio_composition.sample_rate)
            if not cached:
                _extract_source_wav(
                    source.path, start, end - start, path,
                    self.config.audio_composition.sample_rate, prepared_source_audio_path,
                )
            all_cached = all_cached and cached
            paths.append(path)
        if len(paths) == 1:
            return paths[0], all_cached
        merged = temporary_directory / f"bed-{segment.segment_id}-{stable_text_hash('|'.join(str(path) for path in paths))[:12]}.wav"
        _concat_audio(paths, merged, self.config.audio_composition.sample_rate)
        return merged, all_cached

    def _duck_and_mix(self, narration: Path, bed: Path, destination: Path, duration: float) -> None:
        config = self.config.audio_composition
        attack = min(config.duck_attack_seconds, duration / 2)
        release = min(config.duck_release_seconds, max(0.0, duration - attack))
        return_start = max(attack, duration - release)
        if not config.preserve_original_events:
            expression = str(config.duck_level)
        elif attack <= 0 and release <= 0:
            expression = str(config.duck_level)
        else:
            expression = (
                f"if(lt(t,{attack:.6f}),1-(1-{config.duck_level:.6f})*t/{max(attack, 0.001):.6f},"
                f"if(gt(t,{return_start:.6f}),{config.duck_level:.6f}+(1-{config.duck_level:.6f})*(t-{return_start:.6f})/{max(release, 0.001):.6f},{config.duck_level:.6f}))"
            )
        filter_graph = (
            f"[1:a]apad=pad_dur={duration:.6f},atrim=duration={duration:.6f},"
            f"volume='{expression}':eval=frame[bed];[0:a][bed]amix=inputs=2:duration=first:normalize=0[mix]"
        )
        _ffmpeg_to_wav(
            ["-i", str(narration), "-i", str(bed), "-filter_complex", filter_graph, "-map", "[mix]", "-ac", "1", "-ar", str(config.sample_rate)],
            destination,
        )

    def _write_artifacts(self, project: AudioProject, output_root: Path) -> AudioProject:
        output_root.mkdir(parents=True, exist_ok=True)
        project_path = output_root / "audio-project.json"
        manifest_path = output_root / "audio-manifest.json"
        summary_path = output_root / "audio-summary.txt"
        artifacts = [str(project_path), str(manifest_path), str(summary_path)]
        if project.export.path:
            artifacts.append(project.export.path)
        complete = project.model_copy(update={
            "metadata": project.metadata.model_copy(update={"completed_at": utc_now()}),
            "artifacts": artifacts,
        })
        write_json(project_path, complete.model_dump(mode="json"))
        write_json(manifest_path, {
            "schema_version": AUDIO_PROJECT_SCHEMA_VERSION,
            "project_id": complete.project_id,
            "production_plan_id": complete.metadata.production_plan_id,
            "timeline": complete.timeline.model_dump(mode="json"),
            "tracks": [track.model_dump(mode="json") for track in complete.tracks],
            "mix": complete.mix.model_dump(mode="json"),
            "cache": complete.cache,
            "artifacts": artifacts,
        })
        write_bytes_atomic(summary_path, _summary(complete).encode("utf-8"))
        return complete


def audio_report_section(project: AudioProject) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": project.status,
        "audio_mode": project.audio_mode,
        "source_audio_present": bool(project.tracks[1].clips),
        "tracks": [track.model_dump(mode="json") for track in project.tracks],
        "ducking": project.mix.ducking.model_dump(mode="json"),
        "dialogue_count": len(project.tracks[1].clips),
        "narration_count": len(project.tracks[0].clips),
        "mix_duration": project.mix.duration_seconds,
        "cache": project.cache,
        "validation": project.validation.model_dump(mode="json"),
        "artifacts": project.artifacts,
        "warnings": project.warnings,
        "errors": project.errors,
    }


def _tts_segments(data: dict[str, Any] | None) -> dict[str, TTSSegmentResult]:
    if not isinstance(data, dict):
        return {}
    result: dict[str, TTSSegmentResult] = {}
    for raw in data.get("segments", []):
        try:
            item = TTSSegmentResult.model_validate(raw)
        except Exception:
            continue
        if item.status in {"generated", "cached"} and item.artifact and item.artifact.audio_file_path:
            result[item.segment_id] = item
    return result


def _tts_result_path(data: dict[str, Any] | None) -> str | None:
    if not isinstance(data, dict):
        return None
    artifacts = data.get("artifacts", [])
    if isinstance(artifacts, list):
        return next((str(path) for path in artifacts if str(path).endswith("tts-result.json")), None)
    return None


def _narration_source_ranges(segment: NarrationSegment, transcript: dict[str, Any]) -> list[tuple[float, float]]:
    if segment.source_ranges:
        return [
            (item.source_start_seconds, item.source_end_seconds)
            for item in segment.source_ranges
            if item.source_end_seconds > item.source_start_seconds
        ]
    raw_segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    by_id = {
        int(item.get("id", index)): item for index, item in enumerate(raw_segments)
        if isinstance(item, dict) and item.get("start") is not None and item.get("end") is not None
    }
    ranges: list[tuple[float, float]] = []
    for identifier in segment.source_segment_ids:
        item = by_id.get(identifier)
        if item is None:
            continue
        start, end = float(item["start"]), float(item["end"])
        if end > start:
            ranges.append((start, end))
    return ranges


def _source_cache_key(source: Source, start: float, end: float, purpose: str, config: AppConfig) -> str:
    return stable_text_hash(json.dumps({
        "source_id": source.id, "start": round(start, 6), "end": round(end, 6), "purpose": purpose,
        "sample_rate": config.audio_composition.sample_rate,
        "version": config.audio_composition.engine_version,
    }, sort_keys=True))


def _extract_source_wav(
    source: Path, start: float, duration: float, destination: Path, sample_rate: int,
    prepared_source_audio_path: Path | None = None,
) -> None:
    """Extract an exact local range, with a safe fallback for malformed media timestamps."""

    extraction = ["-i", str(source), "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-vn", "-ac", "1", "-ar", str(sample_rate)]
    direct_error: AudioCompositionError | None = None
    try:
        _ffmpeg_to_wav(extraction, destination)
        if _is_valid_wav(destination, sample_rate):
            return
        direct_error = AudioCompositionError("Direct source seek produced an empty or invalid WAV.")
    except AudioCompositionError as error:
        direct_error = error
    if prepared_source_audio_path is not None and prepared_source_audio_path.is_file() and prepared_source_audio_path != source:
        _ffmpeg_to_wav(
            ["-i", str(prepared_source_audio_path), "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-ac", "1", "-ar", str(sample_rate)],
            destination,
        )
        if _is_valid_wav(destination, sample_rate):
            return
    raise direct_error or AudioCompositionError("Source dialogue extraction did not create a valid WAV.")


def _make_silence(destination: Path, duration: float, sample_rate: int) -> None:
    _ffmpeg_to_wav(
        ["-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{duration:.6f}", "-ac", "1", "-ar", str(sample_rate)],
        destination,
    )


def _concat_audio(paths: list[Path], destination: Path, sample_rate: int) -> None:
    if not paths:
        raise AudioCompositionError("No audio clips to concatenate.")
    inputs: list[str] = []
    for path in paths:
        inputs.extend(["-i", str(path)])
    labels = "".join(f"[{index}:a]" for index in range(len(paths)))
    graph = f"{labels}concat=n={len(paths)}:v=0:a=1[mix]"
    _ffmpeg_to_wav([*inputs, "-filter_complex", graph, "-map", "[mix]", "-ac", "1", "-ar", str(sample_rate)], destination)


def _ffmpeg_to_wav(arguments: list[str], destination: Path) -> None:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AudioCompositionError("ffmpeg не найден для Audio Composition.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".wav") as temporary:
        temporary_path = Path(temporary.name)
    try:
        command = [executable, "-y", "-hide_banner", "-loglevel", "error", *arguments, "-c:a", "pcm_s16le", str(temporary_path)]
        result = subprocess.run(command, check=True, capture_output=True, timeout=600, **UTF8_REPLACE_TEXT)
        if result.returncode != 0 or temporary_path.stat().st_size <= 44:
            raise AudioCompositionError("FFmpeg produced an empty audio artifact.")
        temporary_path.replace(destination)
    except subprocess.TimeoutExpired as error:
        temporary_path.unlink(missing_ok=True)
        raise AudioCompositionError("FFmpeg audio composition timed out.") from error
    except (OSError, subprocess.CalledProcessError) as error:
        temporary_path.unlink(missing_ok=True)
        details = getattr(error, "stderr", "") or ""
        raise AudioCompositionError(f"FFmpeg audio composition failed: {details[-800:]}") from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _probe_wav(path: Path, sample_rate: int) -> dict[str, float]:
    executable = shutil.which("ffprobe")
    if not executable:
        raise AudioCompositionError("ffprobe не найден для Audio Composition.")
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, timeout=60, **UTF8_REPLACE_TEXT,
        )
        raw = json.loads(result.stdout)
        stream = next(item for item in raw.get("streams", []) if item.get("codec_type") == "audio")
        duration = float(raw.get("format", {}).get("duration") or stream.get("duration") or 0)
        if (
            duration <= 0 or stream.get("codec_name") != "pcm_s16le"
            or int(stream.get("sample_rate") or 0) != sample_rate or int(stream.get("channels") or 0) != 1
        ):
            raise AudioCompositionError("Audio artifact is not WAV PCM s16le mono at the configured sample rate.")
        return {"duration": round(duration, 3)}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration, TypeError, json.JSONDecodeError) as error:
        raise AudioCompositionError("ffprobe could not validate audio artifact.") from error


def _is_valid_wav(path: Path, sample_rate: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 44 and _probe_wav(path, sample_rate)["duration"] > 0
    except AudioCompositionError:
        return False


def _copy_atomic(source: Path, destination: Path) -> None:
    write_bytes_atomic(destination, source.read_bytes())


def _project_status(mix_status: str, narration: list[NarrationClip], plan: ProductionPlan, warnings: list[str]) -> str:
    if mix_status == "skipped":
        return "skipped"
    planned_narration = sum(isinstance(segment, NarrationSegment) for segment in plan.segments)
    if warnings or len(narration) < planned_narration:
        return "partial"
    return "completed"


def _safe_error(error: BaseException) -> str:
    return str(error).replace("\n", " ")[:800]


def _summary(project: AudioProject) -> str:
    narration = next(track for track in project.tracks if track.track_type == "narration")
    dialogue = next(track for track in project.tracks if track.track_type == "dialogue")
    return "\n".join([
        f"Audio Project: {project.project_id}",
        f"Status: {project.status}",
        f"Narration clips: {len(narration.clips)}",
        f"Dialogue clips: {len(dialogue.clips)}",
        f"Mix duration: {project.mix.duration_seconds:.3f} s",
        f"Ducking: enabled={project.mix.ducking.enabled}, level={project.mix.ducking.duck_level}",
        f"Dialogue cache hits: {project.cache.get('dialogue_hit_count', 0)}",
        "No video, subtitle sync, ASS, or video render was performed.",
        "",
    ])
