from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Protocol

from app.config import AppConfig
from app.errors import ClipEngineError
from app.models import AI_FIELDS, Candidate, ScoredCandidate

if TYPE_CHECKING:
    from app.transformation_models import SourceContext


# The SDK otherwise performs its own long connection retries inside each of
# the application's bounded attempts.  Manual QA measured 551.62 seconds for
# a failed compact transformation before the deterministic fallback ran.
TRANSFORMATION_REQUEST_TIMEOUT_SECONDS = 45.0

# Paid semantic-scoring calls use a deliberately small, evidence-only request
# contract.  Bump this whenever its shape or interpretation changes so the
# ai_ranking cache cannot reuse an assessment made from a different payload.
SEMANTIC_AI_PAYLOAD_VERSION = "semantic-score.3"

SEMANTIC_FACTOR_CONTRACT: dict[str, Any] = {
    "scale": {
        "minimum": 0,
        "maximum": 100,
        "judgment": (
            "Make an independent semantic assessment from the supplied evidence. "
            "Evidence values are inputs, not target scores; do not copy or mechanically "
            "rescale local/code scores."
        ),
    },
    "hook_score": {
        "zero": "The opening gives no credible reason to continue watching.",
        "hundred": (
            "The opening is an exceptionally strong, evidence-grounded reason "
            "to continue watching."
        ),
    },
    "completeness_score": {
        "zero": "The candidate has no coherent or resolved semantic unit.",
        "hundred": "The candidate delivers a fully coherent and resolved semantic unit.",
    },
    "emotional_score": {
        "zero": "The evidence supports no emotional or reaction impact.",
        "hundred": "The evidence supports exceptionally strong emotional or reaction impact.",
    },
    "clarity_score": {
        "zero": "The candidate is not understandable or internally coherent.",
        "hundred": "The candidate is immediately understandable and internally coherent.",
    },
    "context_dependency_score": {
        "zero": "The candidate is completely understandable on its own.",
        "hundred": "Understanding the candidate requires previous or external context.",
        "direction": (
            "Higher means more context dependency; it never means more context independence."
        ),
    },
}


class ClipScorer(Protocol):
    name: str

    def score(
        self, candidates: list[Candidate], transcript: dict[str, Any]
    ) -> tuple[list[ScoredCandidate], dict[str, Any]]:
        """Score candidates and return provider metadata safe for report.json."""


def _factor_score_property(field: str) -> dict[str, Any]:
    contract = SEMANTIC_FACTOR_CONTRACT[field]
    description = f"0: {contract['zero']} 100: {contract['hundred']}"
    if direction := contract.get("direction"):
        description = f"{description} {direction}"
    return {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "description": description,
    }


_SCORE_PROPERTIES: dict[str, Any] = {
    "candidate_id": {"type": "string"},
    "start": {"type": "number"},
    "end": {"type": "number"},
    "title": {"type": "string"},
    "hook": {"type": "string"},
    "summary": {"type": "string"},
    "score": {"type": "integer", "minimum": 0, "maximum": 100},
    "hook_score": _factor_score_property("hook_score"),
    "completeness_score": _factor_score_property("completeness_score"),
    "emotional_score": _factor_score_property("emotional_score"),
    "clarity_score": _factor_score_property("clarity_score"),
    "context_dependency_score": _factor_score_property("context_dependency_score"),
    "rejection_reason": {"type": ["string", "null"]},
    "selected": {"type": "boolean"},
}

# Responses Structured Outputs requires every property to be required and rejects
# unlisted fields.  This mirrors the existing ScoredCandidate result contract.
OPENAI_SCORE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": _SCORE_PROPERTIES,
                "required": ["candidate_id", *AI_FIELDS],
            },
        },
    },
    "required": ["candidates"],
}


def sanitize_api_error(error: BaseException | str, *secrets: str | None) -> str:
    """Return diagnostic text that cannot leak an API key into files or logs."""

    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", message)
    return message[:1000]


def _make_scored(candidate: Candidate, data: dict[str, Any]) -> ScoredCandidate:
    required = set(AI_FIELDS)
    if missing := required - set(data):
        raise ValueError(f"В ответе AI отсутствуют поля: {', '.join(sorted(missing))}")
    if abs(float(data["start"]) - candidate.start) > 1.5 or abs(float(data["end"]) - candidate.end) > 1.5:
        raise ValueError("AI изменил границы клипа больше допустимого.")
    numeric = (
        "score", "hook_score", "completeness_score", "emotional_score",
        "clarity_score", "context_dependency_score",
    )
    for field in numeric:
        if not 0 <= int(data[field]) <= 100:
            raise ValueError(f"Поле {field} должно быть от 0 до 100.")
    return ScoredCandidate(
        candidate=candidate,
        title=str(data["title"]).strip()[:140],
        hook=str(data["hook"]).strip()[:300],
        summary=str(data["summary"]).strip()[:600],
        score=int(data["score"]),
        hook_score=int(data["hook_score"]),
        completeness_score=int(data["completeness_score"]),
        emotional_score=int(data["emotional_score"]),
        clarity_score=int(data["clarity_score"]),
        context_dependency_score=int(data["context_dependency_score"]),
        rejection_reason=(
            str(data["rejection_reason"]).strip()
            if data["rejection_reason"] is not None
            else None
        ),
        selected=bool(data["selected"]),
    )


def _reject_candidates(
    candidates: list[Candidate], provider: str, error: str
) -> list[ScoredCandidate]:
    return [
        ScoredCandidate(
            candidate=candidate,
            title="",
            hook="",
            summary="",
            score=0,
            hook_score=0,
            completeness_score=0,
            emotional_score=0,
            clarity_score=0,
            context_dependency_score=100,
            rejection_reason=f"Ошибка {provider}: {error}",
            selected=False,
        )
        for candidate in candidates
    ]


def _usage(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    retries: int = 0,
    api_errors: list[str] | None = None,
    started: float | None = None,
    request_id: str | None = None,
    response_status: int | None = None,
) -> dict[str, Any]:
    data = {
        "provider": provider,
        "model": model,
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "cache_write_input_tokens": int(cache_write_input_tokens),
        "output_tokens": int(output_tokens),
        "retries": int(retries),
        "api_errors": api_errors or [],
        "processing_duration_seconds": (
            round(time.perf_counter() - started, 3) if started is not None else 0
        ),
    }
    if request_id:
        data["request_id"] = request_id
    if response_status is not None:
        data["response_status"] = response_status
    return data


def _openai_cached_input_tokens(usage: Any) -> int:
    details = getattr(usage, "input_tokens_details", None)
    if isinstance(details, dict):
        value = details.get("cached_tokens", 0)
    else:
        value = getattr(details, "cached_tokens", 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _openai_cache_write_input_tokens(usage: Any) -> int:
    details = getattr(usage, "input_tokens_details", None)
    if isinstance(details, dict):
        value = details.get("cache_write_tokens", 0)
    else:
        value = getattr(details, "cache_write_tokens", 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class MockProvider:
    config: AppConfig
    name: str = "mock"

    def score(
        self, candidates: list[Candidate], transcript: dict[str, Any]
    ) -> tuple[list[ScoredCandidate], dict[str, Any]]:
        scored: list[ScoredCandidate] = []
        for candidate in candidates:
            text = candidate.text.strip()
            word_count = len(text.split())
            ending_bonus = 10 if text.endswith((".", "!", "?", "…")) else 0
            target_distance = abs(candidate.duration - self.config.target_clip_duration)
            duration_score = max(0, 28 - round(target_distance))
            hook_score = min(100, 35 + min(45, word_count) + ending_bonus)
            completeness = min(100, 45 + duration_score + ending_bonus)
            clarity = min(100, 40 + min(50, word_count))
            emotional = min(75, 25 + (15 if "!" in text or "?" in text else 0))
            context = max(5, 70 - (10 if text.lower().startswith(("и ", "а ", "но ")) else 0))
            score = round(
                hook_score * 0.25 + completeness * 0.28 + clarity * 0.25
                + emotional * 0.12 + (100 - context) * 0.10
            )
            selected = score >= self.config.score_threshold
            scored.append(ScoredCandidate(
                candidate=candidate,
                title=_title(text),
                hook=_hook(text),
                summary=text[:600],
                score=score,
                hook_score=hook_score,
                completeness_score=completeness,
                emotional_score=emotional,
                clarity_score=clarity,
                context_dependency_score=context,
                rejection_reason=None if selected else "Оценка ниже настроенного порога.",
                selected=selected,
            ))
        return scored, _usage(self.name, self.config.ai.model)

    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: str,
        pass_kind: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Deterministic schema-valid fixture; it never performs external I/O."""

        observations = []
        for frame in frames:
            observations.append({
                "keyframe_id": str(frame["keyframe_id"]),
                "timestamp": float(frame["timestamp"]),
                "scene_type": "UNKNOWN",
                "primary_subject": "scene",
                "normalized_center_x": 0.5,
                "normalized_center_y": 0.5,
                "visible_face_count": 0,
                "action": "unknown",
                "reaction": "unknown",
                "payoff_signal": "unknown",
                "on_screen_text": "",
                "composition_risk": "unknown",
                "confidence": 0.25,
                "missing_evidence": ["action", "payoff", "reaction", "text"],
            })
        return {"observations": observations}, _usage(
            self.name, self.config.ai.model, response_status=200,
        )

    def transform_compact(self, context: "SourceContext") -> tuple[dict[str, Any], dict[str, Any]]:
        """Deterministic transformation fixture used by --mock-ai and local tests."""

        from app.errors import TransformationProviderError
        from app.narrative_planning import build_narrative_plan
        from app.script_generation import generate_script_draft
        from app.semantic_extraction import extract_semantic_representation

        mode = self.config.transformation.mock_mode
        if mode == "provider_error":
            raise TransformationProviderError("Mock transformation provider_error.")
        if mode == "malformed_json":
            raise TransformationProviderError("Mock transformation malformed_json.")
        semantic = extract_semantic_representation(context)
        plan = build_narrative_plan(semantic, self.config.transformation)
        draft = generate_script_draft(semantic, plan, self.config.transformation.target_words_per_second)
        if mode == "invalid_fact_id" and draft.sentences:
            draft.sentences[0].supported_by_fact_ids = ["fact-999"]
            draft.used_fact_ids = ["fact-999"]
        elif mode in {"unsupported_number", "repair_success"} and draft.sentences:
            draft.sentences[0].text = f"{draft.sentences[0].text} 999%"
        elif mode == "changed_negation" and draft.sentences:
            draft.sentences[0].text = re.sub(r"\bне\s+", "", draft.sentences[0].text, flags=re.I)
            draft.sentences[0].text = re.sub(r"\bnot\s+", "", draft.sentences[0].text, flags=re.I)
        elif mode in {"empty_script", "repair_failure"}:
            draft.sentences = []
            draft.used_fact_ids = []
            draft.full_text = ""
        return {
            "semantic_representation": semantic.to_dict(),
            "narrative_plan": plan.to_dict(),
            "script_draft": draft.to_dict(),
        }, _usage(self.name, self.config.ai.model, response_status=200)

    def repair_script(
        self, context: "SourceContext", semantic: dict[str, Any], plan: dict[str, Any],
        draft: dict[str, Any], validation_errors: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from app.errors import TransformationProviderError
        from app.narrative_planning import build_narrative_plan
        from app.script_generation import generate_script_draft
        from app.semantic_extraction import extract_semantic_representation

        if self.config.transformation.mock_mode == "repair_failure":
            raise TransformationProviderError("Mock transformation repair_failure.")
        local_semantic = extract_semantic_representation(context)
        local_plan = build_narrative_plan(local_semantic, self.config.transformation)
        repaired = generate_script_draft(local_semantic, local_plan, self.config.transformation.target_words_per_second)
        return repaired.to_dict(), _usage(self.name, self.config.ai.model, response_status=200)


# Backward-compatible import for existing integrations.
MockScorer = MockProvider


@dataclass(slots=True)
class OpenAIProvider:
    config: AppConfig
    api_key: str
    client: Any | None = None
    name: str = "openai"

    def score(
        self, candidates: list[Candidate], transcript: dict[str, Any]
    ) -> tuple[list[ScoredCandidate], dict[str, Any]]:
        client = self.client
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
        payload = build_openai_payload(candidates, transcript)
        errors: list[str] = []
        started = time.perf_counter()
        for attempt in range(self.config.ai.max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.config.ai.model,
                    instructions=(
                        "Оцени кандидатов для коротких вертикальных клипов. "
                        "Не меняй candidate_id, start/end. Верни оценку каждого кандидата в Structured Output."
                    ),
                    input=json.dumps(payload, ensure_ascii=False),
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "clip_candidate_scores",
                            "strict": True,
                            "schema": OPENAI_SCORE_RESPONSE_SCHEMA,
                        }
                    },
                )
                parsed = json.loads(response.output_text)
                items = parsed.get("candidates") if isinstance(parsed, dict) else None
                if not isinstance(items, list) or len(items) != len(candidates):
                    raise ValueError(
                        "Structured Output должен содержать candidates для каждого кандидата."
                    )
                for candidate, item in zip(candidates, items):
                    if str(item.get("candidate_id", "")) != candidate.id:
                        raise ValueError("AI вернул неизвестный или несоответствующий candidate_id.")
                scored = [_make_scored(candidate, item) for candidate, item in zip(candidates, items)]
                usage = getattr(response, "usage", None)
                return scored, _usage(
                    self.name,
                    self.config.ai.model,
                    getattr(usage, "input_tokens", 0) or 0,
                    getattr(usage, "output_tokens", 0) or 0,
                    cached_input_tokens=_openai_cached_input_tokens(usage),
                    cache_write_input_tokens=_openai_cache_write_input_tokens(usage),
                    retries=attempt,
                    api_errors=errors,
                    started=started,
                )
            except Exception as error:
                errors.append(sanitize_api_error(error, self.api_key))
        message = errors[-1] if errors else "Неизвестная ошибка OpenAI API."
        return _reject_candidates(candidates, self.name, message), _usage(
            self.name,
            self.config.ai.model,
            retries=self.config.ai.max_retries,
            api_errors=errors,
            started=started,
        )

    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: str,
        pass_kind: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make one budget-admitted Responses API request (no hidden retries)."""

        from app.vision_intelligence import VISION_RESPONSE_SCHEMA, VisionProviderCallError, vision_prompt

        client = self.client
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": vision_prompt(pass_kind)}]
        for frame in frames:
            content.append({
                "type": "input_text",
                "text": f"keyframe_id={frame['keyframe_id']} timestamp={float(frame['timestamp']):.3f}",
            })
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame['image_base64']}",
                "detail": detail,
            })
        started = time.perf_counter()
        response = client.responses.create(
            model=self.config.ai.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": f"vision_{pass_kind}_observations",
                    "strict": True,
                    "schema": VISION_RESPONSE_SCHEMA,
                }
            },
        )
        usage = getattr(response, "usage", None)
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        usage_data = _usage(
            self.name, self.config.ai.model,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=_openai_cached_input_tokens(usage),
            cache_write_input_tokens=_openai_cache_write_input_tokens(usage),
            started=started,
            request_id=str(request_id) if request_id else None,
            response_status=200,
        )
        try:
            parsed = json.loads(response.output_text)
        except (TypeError, json.JSONDecodeError) as error:
            raise VisionProviderCallError("Vision provider returned invalid or empty JSON.", usage_data) from error
        if not isinstance(parsed, dict):
            raise VisionProviderCallError("Vision Structured Output returned a non-object payload.", usage_data)
        return parsed, usage_data

    def transform_compact(self, context: "SourceContext") -> tuple[dict[str, Any], dict[str, Any]]:
        from app.errors import TransformationProviderError
        from app.transformation_prompts import (
            OPENAI_TRANSFORMATION_RESPONSE_SCHEMA,
            compact_instructions,
            compact_payload,
        )

        client = self.client
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                timeout=TRANSFORMATION_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        errors: list[str] = []
        started = time.perf_counter()
        payload = compact_payload(context, asdict(self.config.transformation))
        for attempt in range(self.config.ai.max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.config.ai.model,
                    instructions=compact_instructions(),
                    input=payload,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "grounded_content_transformation",
                            "strict": True,
                            "schema": OPENAI_TRANSFORMATION_RESPONSE_SCHEMA,
                        }
                    },
                )
                parsed = json.loads(response.output_text)
                if not isinstance(parsed, dict) or set(parsed) != {"semantic_representation", "narrative_plan", "script_draft"}:
                    raise ValueError("Structured Output transformation имеет неверную верхнеуровневую структуру.")
                if any(not isinstance(parsed[key], dict) for key in parsed):
                    raise ValueError("Structured Output transformation содержит не-объектный этап.")
                usage = getattr(response, "usage", None)
                request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
                return parsed, _usage(
                    self.name, self.config.ai.model,
                    getattr(usage, "input_tokens", 0) or 0,
                    getattr(usage, "output_tokens", 0) or 0,
                    cached_input_tokens=_openai_cached_input_tokens(usage),
                    cache_write_input_tokens=_openai_cache_write_input_tokens(usage),
                    retries=attempt, api_errors=errors, started=started,
                    request_id=str(request_id) if request_id else None, response_status=200,
                )
            except Exception as error:
                errors.append(sanitize_api_error(error, self.api_key))
        raise TransformationProviderError(errors[-1] if errors else "Неизвестная ошибка OpenAI transformation API.")

    def repair_script(
        self, context: "SourceContext", semantic: dict[str, Any], plan: dict[str, Any],
        draft: dict[str, Any], validation_errors: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from app.errors import TransformationProviderError
        from app.transformation_prompts import OPENAI_SCRIPT_DRAFT_SCHEMA, repair_instructions, repair_payload

        client = self.client
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                timeout=TRANSFORMATION_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        errors: list[str] = []
        started = time.perf_counter()
        payload = repair_payload(context, semantic, plan, draft, validation_errors, asdict(self.config.transformation))
        for attempt in range(self.config.ai.max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.config.ai.model,
                    instructions=repair_instructions(),
                    input=payload,
                    text={"format": {"type": "json_schema", "name": "grounded_script_repair", "strict": True, "schema": OPENAI_SCRIPT_DRAFT_SCHEMA}},
                )
                parsed = json.loads(response.output_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Structured Output repair вернул не JSON object.")
                response_usage = getattr(response, "usage", None)
                request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
                return parsed, _usage(
                    self.name, self.config.ai.model,
                    getattr(response_usage, "input_tokens", 0) or 0,
                    getattr(response_usage, "output_tokens", 0) or 0,
                    cached_input_tokens=_openai_cached_input_tokens(response_usage),
                    cache_write_input_tokens=_openai_cache_write_input_tokens(response_usage),
                    retries=attempt, api_errors=errors, started=started,
                    request_id=str(request_id) if request_id else None, response_status=200,
                )
            except Exception as error:
                errors.append(sanitize_api_error(error, self.api_key))
        raise TransformationProviderError(errors[-1] if errors else "Неизвестная ошибка OpenAI repair API.")


@dataclass(slots=True)
class GeminiProvider:
    config: AppConfig
    api_key: str
    name: str = "gemini"

    def score(
        self, candidates: list[Candidate], transcript: dict[str, Any]
    ) -> tuple[list[ScoredCandidate], dict[str, Any]]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        payload = build_gemini_payload(candidates, transcript)
        errors: list[str] = []
        started = time.perf_counter()
        for attempt in range(self.config.ai.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.config.ai.model,
                    contents=json.dumps(payload, ensure_ascii=False),
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                parsed = json.loads(response.text)
                if not isinstance(parsed, list) or len(parsed) != len(candidates):
                    raise ValueError("Ответ AI должен быть массивом по числу кандидатов.")
                scored = [_make_scored(candidate, item) for candidate, item in zip(candidates, parsed)]
                usage = getattr(response, "usage_metadata", None)
                return scored, _usage(
                    self.name,
                    self.config.ai.model,
                    getattr(usage, "prompt_token_count", 0) or 0,
                    getattr(usage, "candidates_token_count", 0) or 0,
                    retries=attempt,
                    api_errors=errors,
                    started=started,
                )
            except Exception as error:
                errors.append(sanitize_api_error(error, self.api_key))
        message = errors[-1] if errors else "Неизвестная ошибка Gemini API."
        return _reject_candidates(candidates, self.name, message), _usage(
            self.name,
            self.config.ai.model,
            retries=self.config.ai.max_retries,
            api_errors=errors,
            started=started,
        )

    def analyze_vision(
        self,
        frames: list[dict[str, Any]],
        *,
        detail: str,
        pass_kind: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make one budget-admitted Gemini request through the selected provider key."""

        from app.vision_intelligence import VISION_RESPONSE_SCHEMA, VisionProviderCallError, vision_prompt
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        contents: list[Any] = [vision_prompt(pass_kind)]
        for frame in frames:
            contents.append(f"keyframe_id={frame['keyframe_id']} timestamp={float(frame['timestamp']):.3f}")
            contents.append(types.Part.from_bytes(
                data=base64.b64decode(str(frame["image_base64"])), mime_type="image/jpeg",
            ))
        started = time.perf_counter()
        response = client.models.generate_content(
            model=self.config.ai.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=VISION_RESPONSE_SCHEMA,
                max_output_tokens=max_output_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        usage_data = _usage(
            self.name, self.config.ai.model,
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
            started=started, response_status=200,
        )
        try:
            parsed = json.loads(response.text)
        except (TypeError, json.JSONDecodeError) as error:
            raise VisionProviderCallError("Gemini vision returned invalid or empty JSON.", usage_data) from error
        if not isinstance(parsed, dict):
            raise VisionProviderCallError("Gemini vision returned a non-object payload.", usage_data)
        return parsed, usage_data

    def transform_compact(self, context: "SourceContext") -> tuple[dict[str, Any], dict[str, Any]]:
        from app.errors import TransformationProviderError
        from app.transformation_prompts import (
            OPENAI_TRANSFORMATION_RESPONSE_SCHEMA,
            compact_instructions,
            compact_payload,
        )
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        errors: list[str] = []
        started = time.perf_counter()
        payload = compact_payload(context, asdict(self.config.transformation))
        for attempt in range(self.config.ai.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.config.ai.model,
                    contents=[compact_instructions(), payload],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=OPENAI_TRANSFORMATION_RESPONSE_SCHEMA,
                    ),
                )
                parsed = json.loads(response.text)
                if not isinstance(parsed, dict):
                    raise ValueError("Gemini transformation вернул не JSON object.")
                usage = getattr(response, "usage_metadata", None)
                return parsed, _usage(
                    self.name, self.config.ai.model,
                    getattr(usage, "prompt_token_count", 0) or 0,
                    getattr(usage, "candidates_token_count", 0) or 0,
                    retries=attempt, api_errors=errors, started=started, response_status=200,
                )
            except Exception as error:
                errors.append(sanitize_api_error(error, self.api_key))
        raise TransformationProviderError(errors[-1] if errors else "Неизвестная ошибка Gemini transformation API.")

    def repair_script(
        self, context: "SourceContext", semantic: dict[str, Any], plan: dict[str, Any],
        draft: dict[str, Any], validation_errors: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from app.errors import TransformationProviderError
        from app.transformation_prompts import OPENAI_SCRIPT_DRAFT_SCHEMA, repair_instructions, repair_payload
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        errors: list[str] = []
        started = time.perf_counter()
        payload = repair_payload(context, semantic, plan, draft, validation_errors, asdict(self.config.transformation))
        for attempt in range(self.config.ai.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.config.ai.model,
                    contents=[repair_instructions(), payload],
                    config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=OPENAI_SCRIPT_DRAFT_SCHEMA),
                )
                parsed = json.loads(response.text)
                if not isinstance(parsed, dict):
                    raise ValueError("Gemini repair вернул не JSON object.")
                response_usage = getattr(response, "usage_metadata", None)
                return parsed, _usage(
                    self.name, self.config.ai.model,
                    getattr(response_usage, "prompt_token_count", 0) or 0,
                    getattr(response_usage, "candidates_token_count", 0) or 0,
                    retries=attempt, api_errors=errors, started=started, response_status=200,
                )
            except Exception as error:
                errors.append(sanitize_api_error(error, self.api_key))
        raise TransformationProviderError(errors[-1] if errors else "Неизвестная ошибка Gemini repair API.")


# Backward-compatible import for existing integrations.
GeminiScorer = GeminiProvider


def get_scorer(config: AppConfig, force_mock: bool = False) -> ClipScorer:
    if force_mock or config.mock_ai or config.ai.provider == "mock":
        return MockProvider(config)
    if config.ai.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ClipEngineError(
                "OPENAI_API_KEY не задан. Добавьте ключ в .env или запустите с --mock-ai."
            )
        try:
            import openai  # noqa: F401
        except ImportError as error:
            raise ClipEngineError(
                "Пакет openai не установлен. Выполните pip install -r requirements.txt."
            ) from error
        return OpenAIProvider(config, api_key)
    if config.ai.provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ClipEngineError(
                "GEMINI_API_KEY не задан. Добавьте ключ в .env или запустите с --mock-ai."
            )
        try:
            import google.genai  # noqa: F401
        except ImportError as error:
            raise ClipEngineError(
                "Пакет google-genai не установлен. Выполните pip install -r requirements.txt."
            ) from error
        return GeminiProvider(config, api_key)
    # AppConfig.validate catches this branch for normal CLI/config usage.
    raise ClipEngineError(f"Неподдерживаемый AI provider: {config.ai.provider}")


def get_transformer(config: AppConfig, force_mock: bool = False) -> Any:
    """Return the existing provider instance with its compact typed operation."""

    provider = get_scorer(config, force_mock)
    if not hasattr(provider, "transform_compact"):
        raise ClipEngineError(f"AI provider {provider.name} не поддерживает content transformation.")
    return provider


def get_vision_provider(config: AppConfig, force_mock: bool = False) -> Any:
    """Reuse the configured provider/key while exposing only the vision adapter."""

    provider = get_scorer(config, force_mock)
    if not hasattr(provider, "analyze_vision"):
        raise ClipEngineError(f"AI provider {provider.name} не поддерживает Vision Gateway.")
    return provider


_SEMANTIC_EVIDENCE_FIELDS = (
    "hook", "setup", "payoff", "ending", "completeness_score",
    "information_density",
)
_BOUNDARY_SIGNAL_FIELDS = (
    "word_integrity", "sentence_integrity", "semantic_completion",
    "head_naturalness", "tail_naturalness", "payoff_preserved",
    "continuation_risk",
)
_SPEECH_SIGNAL_FIELDS = (
    "transcript_confidence", "speech_density", "words_per_second",
)
_AUDIO_EVENT_FIELDS = (
    "event_type", "start_seconds", "end_seconds", "confidence",
)
_VISUAL_OBSERVATION_FIELDS = (
    "timestamp", "scene_type", "primary_subject", "action", "reaction",
    "payoff_signal", "on_screen_text", "composition_risk", "confidence",
    "missing_evidence",
)
_VISION_VERIFICATION_FIELDS = (
    "hook_visible", "action_visible", "reaction_visible", "payoff_visible",
    "continuity_risk", "confidence",
)
_MULTIMODAL_ANCHOR_FIELDS = ("hook", "action", "reaction", "payoff")


def _selected_fields(data: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {
        field: data[field]
        for field in fields
        if field in data and data[field] is not None
    }


def _compact_audio_events(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    events = provenance.get("audio_evidence", [])
    if not isinstance(events, list):
        return []
    return [
        compact
        for item in events
        if isinstance(item, dict)
        and (compact := _selected_fields(item, _AUDIO_EVENT_FIELDS))
    ]


def _compact_visual_observation(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    nested = item.get("observation")
    observation = nested if isinstance(nested, dict) else item
    compact = _selected_fields(observation, _VISUAL_OBSERVATION_FIELDS)
    if "timestamp" not in compact:
        timestamp = item.get("timestamp", item.get("start_seconds"))
        if timestamp is not None:
            compact["timestamp"] = timestamp
    return compact


def _compact_multimodal_signals(candidate: Candidate) -> dict[str, Any]:
    provenance = candidate.multimodal_provenance
    if not isinstance(provenance, dict):
        provenance = {}
    generation = provenance.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}

    pass2 = candidate.vision_pass2_evidence
    if not isinstance(pass2, dict):
        pass2 = {}
    result = pass2.get("result")
    if not isinstance(result, dict):
        result = {}
    pass2_observations = result.get("observations", [])
    pass1_observations = provenance.get("visual_evidence", [])
    if isinstance(pass2_observations, list) and pass2_observations:
        raw_observations = pass2_observations
        observation_source = "vision_pass2"
    elif isinstance(pass1_observations, list) and pass1_observations:
        raw_observations = pass1_observations
        observation_source = "vision_pass1"
    else:
        raw_observations = []
        observation_source = "none"

    return {
        "candidate_kind": candidate.candidate_kind,
        "anchors": _selected_fields(generation.get("anchors"), _MULTIMODAL_ANCHOR_FIELDS),
        "audio_events": _compact_audio_events(provenance),
        "visual_observation_source": observation_source,
        "visual_observations": [
            compact
            for item in raw_observations
            if (compact := _compact_visual_observation(item))
        ],
        "vision_pass2_status": str(pass2.get("status") or "not_available"),
        "vision_verification": _selected_fields(
            result.get("verification"), _VISION_VERIFICATION_FIELDS,
        ),
    }


def _compact_semantic_candidate(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "start": round(candidate.start, 3),
        "end": round(candidate.end, 3),
        "duration": round(candidate.duration, 3),
        "text": candidate.text,
        "core_idea": candidate.core_idea,
        "semantic_evidence": _selected_fields(
            candidate.semantic_evidence, _SEMANTIC_EVIDENCE_FIELDS,
        ),
        "boundary_signals": _selected_fields(
            candidate.boundary_diagnostics, _BOUNDARY_SIGNAL_FIELDS,
        ),
        "speech_signals": _selected_fields(
            candidate.feature_vector, _SPEECH_SIGNAL_FIELDS,
        ),
        "multimodal_signals": _compact_multimodal_signals(candidate),
    }


def _semantic_base_payload(candidates: list[Candidate], transcript: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_payload_version": SEMANTIC_AI_PAYLOAD_VERSION,
        "factor_contract": {
            field: dict(definition)
            for field, definition in SEMANTIC_FACTOR_CONTRACT.items()
        },
        "language": transcript.get("language"),
        "transcript": [
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment.get("text", "")),
            }
            for segment in transcript.get("segments", [])
            if "start" in segment and "end" in segment
        ],
        "candidates": [_compact_semantic_candidate(candidate) for candidate in candidates],
    }


def build_openai_payload(
    candidates: list[Candidate], transcript: dict[str, Any]
) -> dict[str, Any]:
    return {
        **_semantic_base_payload(candidates, transcript),
        "instruction": (
            "Assess every supplied candidate, not only a best-five list. Scores are integers 0..100. "
            "Apply factor_contract to all five factor fields. Make an independent assessment from the full "
            "transcript and the supplied semantic, speech, audio and Vision evidence; evidence values are not "
            "target scores and must not be copied or mechanically rescaled. The application ignores AI "
            "score/selected for final scoring, ranking and selection. Return only candidate_id values from the "
            "input and never change start/end."
        ),
    }


def build_gemini_payload(
    candidates: list[Candidate], transcript: dict[str, Any]
) -> dict[str, Any]:
    return {
        **_semantic_base_payload(candidates, transcript),
        "instruction": (
            "Assess every supplied candidate for a short vertical clip; do not return only a best-five list. "
            "Return only a JSON array. Every item must contain all fields: "
            + ", ".join(AI_FIELDS)
            + ". Scores are integer 0..100. Do not change start/end. Apply factor_contract to all five factor "
            "fields. Make an independent assessment from the full transcript and supplied semantic, speech, "
            "audio and Vision evidence; evidence values are not target scores and must not be copied or "
            "mechanically rescaled. The application ignores AI score/selected and owns final scoring, ranking "
            "and selection."
        ),
    }


def _title(text: str) -> str:
    words = text.replace("\n", " ").split()
    return " ".join(words[:8]).rstrip(".,!?") or "Фрагмент видео"


def _hook(text: str) -> str:
    return text.split(".")[0].strip()[:300] or text[:300]
