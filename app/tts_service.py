from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from app.ai import sanitize_api_error
from app.config import AppConfig
from app.production_models import NarrationSegment, ProductionPlan
from app.tts_models import (
    TTS_SCHEMA_VERSION,
    TTSAudioArtifact,
    TTSError,
    TTSGenerationResult,
    TTSMetadata,
    TTSProviderConfig,
    TTSRequest,
    TTSSegmentRequest,
    TTSSegmentResult,
    TTSUsage,
    TTSValidationResult,
    TTSVoiceConfig,
)
from app.tts_providers import TTSProvider, get_tts_provider
from app.utils import read_json, stable_file_hash, stable_text_hash, utc_now, write_bytes_atomic, write_json


TTS_ENGINE_VERSION = "3B.0"
SUPPORTED_LANGUAGES = {
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da", "nl", "en",
    "et", "fi", "fr", "gl", "de", "el", "he", "hi", "hu", "is", "id", "it", "ja", "kn", "kk",
    "ko", "lv", "lt", "mk", "ms", "mr", "mi", "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr",
    "sk", "sl", "es", "sw", "sv", "tl", "ta", "th", "tr", "uk", "ur", "vi", "cy",
}
VOICE_REGISTRY = {
    ("male", "calm"): "ash",
    ("male", "energetic"): "echo",
    ("male", "documentary"): "onyx",
    ("male", "conversational"): "echo",
    ("female", "calm"): "sage",
    ("female", "energetic"): "nova",
    ("female", "documentary"): "coral",
    ("female", "conversational"): "shimmer",
    ("neutral", "calm"): "sage",
    ("neutral", "energetic"): "marin",
    ("neutral", "documentary"): "cedar",
    ("neutral", "conversational"): "alloy",
}
SAFE_DEFAULT_VOICE = "cedar"


class TTSService:
    """Generate typed, independently cacheable narration audio from ProductionPlan only."""

    def __init__(
        self, root: Path, config: AppConfig,
        provider_factory: Callable[[AppConfig], TTSProvider] = get_tts_provider,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.provider_factory = provider_factory

    def generate(
        self, plan: ProductionPlan, work_directory: Path, output_directory: Path,
        force_recompute: bool = False, enabled: bool | None = None,
        provider: TTSProvider | None = None,
    ) -> TTSGenerationResult:
        started_at = utc_now()
        output_root = output_directory / "tts"
        segments_dir = output_root / "segments"
        cache_dir = self.root / "work" / "tts-cache"
        temporary_dir = work_directory / "tts-tmp"
        provider_config = self._provider_config()
        voice, voice_warning = resolve_voice(plan, self.config)
        batch = TTSRequest(
            production_plan_id=plan.plan_id, provider_config=provider_config, voice=voice, created_at=started_at,
        )
        requests, request_warnings = self._prepare_requests(plan, provider_config, voice, temporary_dir)
        warnings = ([voice_warning] if voice_warning else []) + request_warnings
        estimated_cost = round(sum(_cost(item, self.config) for item in requests), 8)

        if not requests:
            result = self._result(batch, started_at, "skipped", [], 0, warnings)
            return self._write_artifacts(result, output_root, segments_dir)

        active = self.config.tts.enabled if enabled is None else enabled
        cached: dict[str, TTSSegmentResult] = {}
        missing: list[TTSSegmentRequest] = []
        for request in requests:
            cached_result = None if force_recompute or not self.config.tts.cache_enabled else self._cache_hit(request, cache_dir, segments_dir)
            if cached_result is not None:
                cached[request.segment_id] = cached_result
            else:
                missing.append(request)

        results: list[TTSSegmentResult] = []
        if not active:
            results = [self._fallback(request, "ai_disabled") for request in missing]
        else:
            empty = [request for request in missing if not request.normalized_text.strip()]
            unsupported = [request for request in missing if request.language not in SUPPORTED_LANGUAGES and request not in empty]
            supported = [request for request in missing if request.language in SUPPORTED_LANGUAGES and request not in empty]
            results.extend(self._fallback(request, "empty_text") for request in empty)
            results.extend(self._fallback(request, "unsupported_language") for request in unsupported)
            pending_cost = round(sum(_cost(item, self.config) for item in supported), 8)
            if pending_cost > self.config.tts.budget_limit:
                results.extend(self._fallback(request, "budget_exceeded") for request in supported)
            else:
                resolved_provider = provider
                if supported and resolved_provider is None:
                    try:
                        resolved_provider = self.provider_factory(self.config)
                    except Exception as error:
                        reason = "missing_api_key" if "OPENAI_API_KEY" in str(error) else "provider_failure"
                        message = sanitize_api_error(error)
                        results.extend(self._fallback(request, reason, message) for request in supported)
                        resolved_provider = None
                if resolved_provider is not None:
                    for request in supported:
                        generated = resolved_provider.synthesize(request)
                        results.append(self._consume_provider_result(generated, request, cache_dir, segments_dir))

        by_id = {result.segment_id: result for result in [*results, *cached.values()]}
        ordered = [by_id[request.segment_id] for request in requests]
        billed_estimate = round(sum(
            _cost(request, self.config)
            for request in requests
            if by_id[request.segment_id].status == "generated"
        ), 8)
        result = self._result(batch, started_at, _status(ordered), ordered, billed_estimate, warnings)
        return self._write_artifacts(result, output_root, segments_dir)

    def _provider_config(self) -> TTSProviderConfig:
        tts = self.config.tts
        return TTSProviderConfig(
            provider=tts.provider, model=tts.model, timeout_seconds=tts.timeout_seconds,
            max_retries=tts.max_retries, speed=tts.speed, output_format="wav",
            sample_rate=tts.sample_rate, provider_config_version=tts.provider_config_version,
        )

    def _prepare_requests(
        self, plan: ProductionPlan, provider_config: TTSProviderConfig,
        voice: TTSVoiceConfig, temporary_dir: Path,
    ) -> tuple[list[TTSSegmentRequest], list[str]]:
        requests: list[TTSSegmentRequest] = []
        warnings: list[str] = []
        for segment in plan.segments:
            if not isinstance(segment, NarrationSegment):
                continue
            normalized = normalize_narration_text(segment.text)
            language = self.config.tts.language if self.config.tts.language != "auto" else plan.voice_profile.language
            language = str(language or "").lower().strip()
            cache_key = _cache_key(normalized, provider_config, voice, language)
            request = TTSSegmentRequest(
                production_plan_id=plan.plan_id,
                segment_id=segment.segment_id,
                narration_text=segment.text,
                normalized_text=normalized or " ",
                language=language or "unknown",
                voice=voice,
                provider_config=provider_config,
                estimated_duration_seconds=segment.estimated_duration_seconds,
                cache_key=cache_key,
                raw_output_path=str(temporary_dir / f"{segment.segment_id}-{cache_key[:12]}.raw.wav"),
            )
            if not normalized:
                warnings.append(f"Narration {segment.segment_id} became empty after technical whitespace normalization.")
            requests.append(request)
        return requests, warnings

    def _cache_hit(
        self, request: TTSSegmentRequest, cache_dir: Path, segments_dir: Path,
    ) -> TTSSegmentResult | None:
        audio_path = cache_dir / f"{request.cache_key}.wav"
        meta_path = cache_dir / f"{request.cache_key}.json"
        try:
            cached = TTSSegmentResult.model_validate(read_json(meta_path, {}))
            artifact = cached.artifact
            if (
                cached.status != "generated" or artifact is None or artifact.checksum is None
                or not audio_path.is_file() or audio_path.stat().st_size <= 44
                or stable_file_hash(audio_path) != artifact.checksum
            ):
                return None
            output_path = self._segment_output_path(segments_dir, request)
            _copy_atomic(audio_path, output_path)
            validation = validate_audio(output_path, request.estimated_duration_seconds, self.config)
            if validation.status == "invalid":
                return None
            return cached.model_copy(update={
                "status": "cached",
                "artifact": artifact.model_copy(update={
                    "audio_file_path": str(output_path), "raw_audio_file_path": None,
                    "origin": "cache", "byte_size": output_path.stat().st_size,
                    "actual_duration_seconds": validation.actual_duration_seconds,
                }),
                "validation": validation,
                "usage": cached.usage.model_copy(update={"api_call_count": 0, "retries": 0}),
                "created_at": utc_now(),
            })
        except Exception:
            return None

    def _consume_provider_result(
        self, result: TTSSegmentResult, request: TTSSegmentRequest,
        cache_dir: Path, segments_dir: Path,
    ) -> TTSSegmentResult:
        if result.status == "fallback":
            # The local provider is deliberately a typed no-audio placeholder,
            # not a provider failure and never a fake successful silent WAV.
            return result
        if result.status != "generated" or result.artifact is None or not result.artifact.audio_file_path:
            message = result.error.message if result.error else None
            return self._fallback(request, "provider_failure", message, result.usage)
        raw_path = Path(result.artifact.audio_file_path)
        if not raw_path.is_file() or raw_path.stat().st_size <= 44:
            return self._fallback(request, "invalid_audio", "Provider returned an empty audio artifact.", result.usage)
        cache_path = cache_dir / f"{request.cache_key}.wav"
        try:
            normalize_audio(raw_path, cache_path, self.config.tts.sample_rate)
            validation = validate_audio(cache_path, request.estimated_duration_seconds, self.config)
        except Exception as error:
            return self._fallback(request, "invalid_audio", sanitize_api_error(error), result.usage)
        if validation.status == "invalid":
            return self._fallback(request, "invalid_audio", validation.message, result.usage, validation)
        checksum = stable_file_hash(cache_path)
        output_path = self._segment_output_path(segments_dir, request)
        _copy_atomic(cache_path, output_path)
        artifact = TTSAudioArtifact(
            production_plan_id=request.production_plan_id,
            segment_id=request.segment_id,
            audio_file_path=str(output_path),
            raw_audio_file_path=str(raw_path),
            checksum=checksum,
            sample_rate=self.config.tts.sample_rate,
            byte_size=output_path.stat().st_size,
            estimated_duration_seconds=request.estimated_duration_seconds,
            actual_duration_seconds=validation.actual_duration_seconds,
            cache_key=request.cache_key,
            origin="provider",
            created_at=utc_now(),
        )
        finished = result.model_copy(update={"artifact": artifact, "validation": validation, "created_at": utc_now()})
        if self.config.tts.cache_enabled:
            cache_artifact = artifact.model_copy(update={"audio_file_path": str(cache_path), "raw_audio_file_path": None})
            write_json(cache_dir / f"{request.cache_key}.json", finished.model_copy(update={"artifact": cache_artifact}).model_dump(mode="json"))
        return finished

    def _fallback(
        self, request: TTSSegmentRequest, reason: str, message: str | None = None,
        usage: TTSUsage | None = None, validation: TTSValidationResult | None = None,
    ) -> TTSSegmentResult:
        safe_message = sanitize_api_error(message or reason)
        return TTSSegmentResult(
            production_plan_id=request.production_plan_id,
            segment_id=request.segment_id,
            status="fallback",
            provider=request.provider_config.provider,
            model=request.provider_config.model,
            voice=request.voice.voice,
            language=request.language,
            source_text=request.narration_text,
            cache_key=request.cache_key,
            artifact=TTSAudioArtifact(
                production_plan_id=request.production_plan_id, segment_id=request.segment_id,
                sample_rate=request.provider_config.sample_rate, byte_size=0,
                estimated_duration_seconds=request.estimated_duration_seconds,
                cache_key=request.cache_key, origin="silent_placeholder", created_at=utc_now(),
            ),
            validation=validation,
            usage=usage or TTSUsage(character_count=len(request.normalized_text), api_call_count=0, retries=0),
            fallback_reason=reason,
            error=TTSError(error_type="TTSFallback", message=safe_message, retryable=False, occurred_at=utc_now()),
            created_at=utc_now(),
        )

    def _segment_output_path(self, segments_dir: Path, request: TTSSegmentRequest) -> Path:
        return segments_dir / f"{request.segment_id}-{request.cache_key[:12]}.wav"

    def _result(
        self, request: TTSRequest, started_at: str, status: str,
        segments: list[TTSSegmentResult], estimated_cost: float, warnings: list[str],
    ) -> TTSGenerationResult:
        successful = [item for item in segments if item.status in {"generated", "cached"}]
        validations = [item.validation.status for item in successful if item.validation]
        validation_status = "invalid" if not successful else "warning" if "warning" in validations else "valid"
        if not successful:
            validation_status = "not_applicable"
        api_errors = [item.error for item in segments if item.error is not None]
        return TTSGenerationResult(
            request=request,
            metadata=TTSMetadata(
                schema_version=TTS_SCHEMA_VERSION, production_plan_id=request.production_plan_id,
                created_at=started_at, completed_at=utc_now(),
                normalized_sample_rate=request.provider_config.sample_rate,
            ),
            status=status,
            segments=segments,
            estimated_cost=estimated_cost,
            actual_cost=None,
            total_duration_seconds=round(sum(item.artifact.actual_duration_seconds or 0 for item in successful if item.artifact), 3),
            validation_status=validation_status,
            cache_hit_count=sum(item.status == "cached" for item in segments),
            generated_count=sum(item.status == "generated" for item in segments),
            fallback_count=sum(item.status == "fallback" for item in segments),
            api_call_count=sum(item.usage.api_call_count for item in segments),
            api_errors=[item for item in api_errors if item is not None],
            warnings=warnings,
        )

    def _write_artifacts(
        self, result: TTSGenerationResult, output_root: Path, segments_dir: Path,
    ) -> TTSGenerationResult:
        output_root.mkdir(parents=True, exist_ok=True)
        result_path = output_root / "tts-result.json"
        manifest_path = output_root / "tts-manifest.json"
        summary_path = output_root / "tts-summary.txt"
        audio_paths = [
            item.artifact.audio_file_path for item in result.segments
            if item.artifact and item.artifact.audio_file_path
        ]
        manifest = {
            "schema_version": TTS_SCHEMA_VERSION,
            "production_plan_id": result.request.production_plan_id,
            "provider": result.request.provider_config.provider,
            "model": result.request.provider_config.model,
            "voice": result.request.voice.voice,
            "segments": [item.model_dump(mode="json") for item in result.segments],
            "cache_hit_count": result.cache_hit_count,
        }
        paths = [str(result_path), str(manifest_path), str(summary_path), *[str(path) for path in audio_paths]]
        complete = result.model_copy(update={"artifacts": paths})
        write_json(result_path, complete.model_dump(mode="json"))
        write_json(manifest_path, manifest)
        write_bytes_atomic(summary_path, _summary(complete).encode("utf-8"))
        return complete


def resolve_voice(plan: ProductionPlan, config: AppConfig) -> tuple[TTSVoiceConfig, str | None]:
    profile = plan.voice_profile
    if config.tts.voice != "auto":
        voice, source, warning = config.tts.voice, "config_override", None
    else:
        voice = VOICE_REGISTRY.get((profile.gender, profile.style))
        source = "voice_registry" if voice else "safe_default"
        warning = None if voice else f"Unknown voice profile {profile.gender}/{profile.style}; using {SAFE_DEFAULT_VOICE}."
        voice = voice or SAFE_DEFAULT_VOICE
    style = profile.style if profile.style in {"calm", "energetic", "documentary", "conversational"} else "documentary"
    language = config.tts.language if config.tts.language != "auto" else profile.language
    return TTSVoiceConfig(
        voice=voice, gender=profile.gender, style=style, language=language or "unknown",
        instructions=f"Use a {style} narration style. Preserve the supplied wording exactly.",
        mapping_source=source,  # type: ignore[arg-type]
    ), warning


def normalize_narration_text(text: str) -> str:
    """Technical normalization only: no word replacement, translation, or paraphrasing."""

    cleaned = "".join(character for character in str(text).replace("\r\n", "\n").replace("\r", "\n") if ord(character) >= 32 or character in "\n\t")
    return " ".join(cleaned.split())


def validate_audio(path: Path, estimated_duration: float, config: AppConfig) -> TTSValidationResult:
    tts = config.tts
    if not path.is_file() or path.stat().st_size <= 44:
        return _invalid_validation(estimated_duration, tts.minimum_audio_duration, tts.maximum_segment_duration, "Audio file is empty.")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return _invalid_validation(estimated_duration, tts.minimum_audio_duration, tts.maximum_segment_duration, "ffprobe is unavailable.")
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        raw = json.loads(probe.stdout)
        stream = next(item for item in raw.get("streams", []) if item.get("codec_type") == "audio")
        actual = float(raw.get("format", {}).get("duration") or stream.get("duration") or 0)
        if stream.get("codec_name") != "pcm_s16le" or int(stream.get("sample_rate") or 0) != tts.sample_rate or int(stream.get("channels") or 0) != 1:
            return _invalid_validation(estimated_duration, tts.minimum_audio_duration, tts.maximum_segment_duration, "Audio is not normalized WAV PCM 16-bit mono.")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration, TypeError, json.JSONDecodeError):
        return _invalid_validation(estimated_duration, tts.minimum_audio_duration, tts.maximum_segment_duration, "ffprobe could not read normalized audio.")
    if actual < tts.minimum_audio_duration or actual > tts.maximum_segment_duration:
        return _invalid_validation(estimated_duration, tts.minimum_audio_duration, tts.maximum_segment_duration, "Audio duration is outside configured limits.", actual)
    difference = round(abs(actual - estimated_duration), 3)
    ratio = round(difference / estimated_duration, 4) if estimated_duration > 0 else 0.0
    if ratio > tts.duration_error_ratio:
        return TTSValidationResult(
            status="invalid", estimated_duration_seconds=estimated_duration, actual_duration_seconds=actual,
            difference_seconds=difference, difference_ratio=ratio,
            minimum_audio_duration=tts.minimum_audio_duration, maximum_segment_duration=tts.maximum_segment_duration,
            message="Audio duration differs from the Production Plan beyond error threshold.",
        )
    return TTSValidationResult(
        status="warning" if ratio > tts.duration_warning_ratio else "valid",
        estimated_duration_seconds=estimated_duration, actual_duration_seconds=actual,
        difference_seconds=difference, difference_ratio=ratio,
        minimum_audio_duration=tts.minimum_audio_duration, maximum_segment_duration=tts.maximum_segment_duration,
        message="Audio duration differs from the Production Plan warning threshold." if ratio > tts.duration_warning_ratio else None,
    )


def normalize_audio(source: Path, destination: Path, sample_rate: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".wav") as temporary:
        temporary_path = Path(temporary.name)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(temporary_path)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        if temporary_path.stat().st_size <= 44:
            raise RuntimeError("ffmpeg produced empty audio")
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def tts_report_section(result: TTSGenerationResult) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": result.status,
        "provider": result.request.provider_config.provider,
        "model": result.request.provider_config.model,
        "voice": result.request.voice.voice,
        "segment_count": len(result.segments),
        "generated_count": result.generated_count,
        "cache_hit_count": result.cache_hit_count,
        "fallback_count": result.fallback_count,
        "estimated_cost": result.estimated_cost,
        "actual_cost": result.actual_cost,
        "total_duration": result.total_duration_seconds,
        "validation_status": result.validation_status,
        "api_call_count": result.api_call_count,
        "api_errors": [item.model_dump(mode="json") for item in result.api_errors],
        "artifacts": result.artifacts,
        "cache": {"enabled": result.request.provider_config.provider != "local", "hit_count": result.cache_hit_count},
    }


def _cache_key(text: str, provider: TTSProviderConfig, voice: TTSVoiceConfig, language: str) -> str:
    return stable_text_hash(json.dumps({
        "schema": TTS_SCHEMA_VERSION, "text": text, "provider": provider.provider,
        "model": provider.model, "voice": voice.voice, "voice_style": voice.style,
        "voice_instructions": voice.instructions, "language": language,
        "speed": provider.speed, "format": provider.output_format, "sample_rate": provider.sample_rate,
        "provider_config_version": provider.provider_config_version,
    }, ensure_ascii=False, sort_keys=True))


def _cost(request: TTSSegmentRequest, config: AppConfig) -> float:
    return len(request.normalized_text) * config.tts.cost_per_1m_characters / 1_000_000


def _status(results: list[TTSSegmentResult]) -> str:
    if not results:
        return "skipped"
    successful = sum(item.status in {"generated", "cached"} for item in results)
    if successful == len(results):
        return "completed"
    if successful:
        return "partial"
    if all(item.status == "fallback" for item in results):
        return "fallback"
    return "failed"


def _invalid_validation(estimated: float, minimum: float, maximum: float, message: str, actual: float | None = None) -> TTSValidationResult:
    difference = round(abs(actual - estimated), 3) if actual is not None else None
    ratio = round(difference / estimated, 4) if difference is not None and estimated > 0 else None
    return TTSValidationResult(
        status="invalid", estimated_duration_seconds=estimated, actual_duration_seconds=actual,
        difference_seconds=difference, difference_ratio=ratio,
        minimum_audio_duration=minimum, maximum_segment_duration=maximum, message=message,
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    write_bytes_atomic(destination, source.read_bytes())


def _summary(result: TTSGenerationResult) -> str:
    return "\n".join([
        f"TTS plan: {result.request.production_plan_id}",
        f"Status: {result.status}",
        f"Provider/model/voice: {result.request.provider_config.provider} / {result.request.provider_config.model} / {result.request.voice.voice}",
        f"Narration segments: {len(result.segments)}",
        f"Generated: {result.generated_count}; cache hits: {result.cache_hit_count}; fallbacks: {result.fallback_count}",
        f"Estimated cost: ${result.estimated_cost:.8f}",
        f"Measured narration duration: {result.total_duration_seconds:.3f} s",
        "No mix, dialogue extraction, subtitle rendering, or video rendering was performed.",
        "",
    ])
