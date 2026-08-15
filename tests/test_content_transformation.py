from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai import MockProvider, OpenAIProvider, TRANSFORMATION_REQUEST_TIMEOUT_SECONDS
from app.config import AIConfig, AppConfig
from app.cli import _apply_transformation_arguments, build_parser
from app.content_transformation import run_content_transformation
from app.errors import NarrativePlanningError, SemanticExtractionError, TransformationProviderError
from app.models import Candidate, ScoredCandidate
from app.narrative_planning import build_narrative_plan
from app.pipeline import Pipeline, StageTracker, _deduplicate_transformation_outcomes
from app.production_plan import build_production_plan
from app.script_generation import generate_script_draft, recompute_script_metrics
from app.script_validation import score_script_quality, validate_script_grounding
from app.semantic_extraction import build_source_context, extract_semantic_representation
from app.transformation_fallback import build_local_fallback
from app.transformation_models import (
    EvidenceSegment,
    FactSourceScope,
    FactualityType,
    FallbackReason,
    FinalScript,
    NarrativePlan,
    ScriptSentence,
    SemanticFact,
    SemanticRepresentation,
    SentenceRole,
    validate_final_script,
)
from app.transformation_prompts import OPENAI_SCRIPT_DRAFT_SCHEMA, OPENAI_TRANSFORMATION_RESPONSE_SCHEMA


def _context(text: str = "This does not guarantee a result. This approach may help some clients when they measure the source data first.", language: str = "en"):
    candidate = Candidate("candidate-001", 10.0, 32.0, text, transcript_segment_ids=[0])
    transcript = {"language": language, "segments": [{"start": 10.0, "end": 32.0, "text": text}]}
    features = {"segments": [{
        "id": 0, "start": 10.0, "end": 32.0, "sentence_start": True, "sentence_end": True,
        "speech_density": 0.7, "pause_before_seconds": 0.5, "pause_after_seconds": 0.5,
        "filler_word_ratio": 0.0, "repetition_score": 0.0,
    }]}
    config = AppConfig().transformation
    return build_source_context(
        {"id": "source-1", "path": "source.mp4"}, {}, candidate, transcript,
        features, {}, {"boundaries": []}, config,
    )


def _draft(text: str | None = None):
    context = _context(text or "This does not guarantee a result. This approach may help some clients when they measure the source data first.")
    semantic = extract_semantic_representation(context)
    plan = build_narrative_plan(semantic, AppConfig().transformation)
    draft = generate_script_draft(semantic, plan, 2.4)
    return context, semantic, plan, draft


def _complete_boundary_context():
    segments = [
        {"id": 0, "start": 10.0, "end": 12.0, "text": "A complete opening establishes the question clearly."},
        {"id": 1, "start": 12.0, "end": 14.0, "text": "Several grounded details explain why the mechanism matters."},
        {"id": 2, "start": 14.0, "end": 16.0, "text": "The final complete answer resolves the question naturally."},
    ]
    text = " ".join(str(item["text"]) for item in segments)
    boundary_decision = {
        "schema_version": "5C.1",
        "decision_id": "boundary-candidate-complete",
        "candidate_id": "candidate-complete",
        "rough_range": {"start_seconds": 10.0, "end_seconds": 16.0},
        "refined_range": {"start_seconds": 10.0, "end_seconds": 16.0},
        "allowed_source_range": {"start_seconds": 10.0, "end_seconds": 16.0},
        "start_reason": "complete opening",
        "end_reason": "complete ending",
        "word_integrity": True,
        "sentence_integrity": True,
        "semantic_completion": True,
        "payoff_preserved": True,
        "continuation_risk": 0.1,
        "continuation_risk_threshold": 0.65,
        "pre_roll_seconds": 0.0,
        "post_roll_seconds": 0.0,
        "confidence": 0.95,
        "start_evidence": {"reason": "sentence_start"},
        "end_evidence": {"reason": "sentence_end"},
        "pause_evidence": {},
        "question_context": {"end_is_question": False, "answer_or_completion_included": True},
        "required_evidence": [
            {
                "requirement_type": "hook", "required": True,
                "source_range": {"start_seconds": 10.0, "end_seconds": 12.0},
                "transcript_segment_id": 0, "reason": "keep hook", "evidence": {},
            },
            {
                "requirement_type": "completion", "required": True,
                "source_range": {"start_seconds": 14.0, "end_seconds": 16.0},
                "transcript_segment_id": 2, "reason": "keep completion", "evidence": {},
            },
        ],
        "safe_start_points": [10.0, 12.0, 14.0],
        "safe_end_points": [12.0, 14.0, 16.0],
        "fallback_used": False,
    }
    candidate = Candidate(
        "candidate-complete", 10.0, 16.0, text,
        transcript_segment_ids=[0, 1, 2],
        boundary_diagnostics={"eligible": True, "boundary_decision": boundary_decision},
    )
    features = {"segments": [
        {
            **item,
            "sentence_start": True,
            "sentence_end": True,
            "speech_density": 0.7,
            "pause_before_seconds": 0.2,
            "pause_after_seconds": 0.2,
            "filler_word_ratio": 0.0,
            "repetition_score": 0.0,
        }
        for item in segments
    ]}
    return build_source_context(
        {"id": "source-1", "path": "source.mp4"}, {}, candidate,
        {"language": "en", "segments": segments}, features, {}, {"boundaries": []},
        AppConfig().transformation,
    )


def _set_sentence_text(draft, text: str) -> None:
    draft.sentences[0].text = text
    recompute_script_metrics(draft, 2.4)


def test_semantic_facts_preserve_segment_asr_confidence() -> None:
    candidate = Candidate(
        "candidate-low-asr", 10.0, 16.0, "Garbled semantic sentence.",
        transcript_segment_ids=[7],
    )
    context = build_source_context(
        {"id": "source-1", "path": "source.mp4"}, {}, candidate,
        {"language": "en", "segments": [{
            "id": 7, "start": 10.0, "end": 16.0,
            "text": "Garbled semantic sentence.",
        }]},
        {"segments": [{
            "id": 7, "transcript_confidence": 0.41,
            "sentence_start": True, "sentence_end": True,
        }]},
        {}, {"boundaries": []}, AppConfig().transformation,
    )

    semantic = extract_semantic_representation(context)

    assert context.primary_evidence[0].confidence == 0.41
    assert semantic.supporting_facts[0].confidence == 0.41
    assert semantic.confidence == 0.41


def test_transformation_duplicate_collapse_is_downgraded_before_production() -> None:
    first = {
        "candidate_id": "first", "status": "completed",
        "source_context": {"start_time": 10, "end_time": 30, "transcript_text": "A distinct source thought."},
        "final_script": {"full_text": "A transformed script for the first thought."},
    }
    second = {
        "candidate_id": "second", "status": "completed",
        "source_context": {"start_time": 80, "end_time": 100, "transcript_text": "A different source thought."},
        "final_script": {"full_text": "A transformed script for the first thought."},
    }

    outcomes, warnings = _deduplicate_transformation_outcomes([first, second])

    assert outcomes[0]["status"] == "completed"
    assert outcomes[1]["status"] == "skipped"
    assert outcomes[1]["reason"] == "transformation_duplicate"
    assert warnings and "копий" in warnings[0]


def test_failed_transformation_does_not_suppress_overlapping_success() -> None:
    failed = {
        "candidate_id": "failed-first",
        "status": "failed",
        "source_context": {
            "start_time": 10.0,
            "end_time": 30.0,
            "transcript_text": "The same candidate source range.",
        },
    }
    successful = {
        "candidate_id": "successful-second",
        "status": "completed",
        "source_context": {
            "start_time": 10.0,
            "end_time": 30.0,
            "transcript_text": "The same candidate source range.",
        },
        "final_script": {"full_text": "A valid completed candidate script."},
    }

    outcomes, warnings = _deduplicate_transformation_outcomes([failed, successful])

    assert [item["status"] for item in outcomes] == ["failed", "completed"]
    assert warnings == []


def test_semantic_facts_require_evidence_and_known_segments() -> None:
    context = _context()
    semantic = extract_semantic_representation(context)
    assert semantic.supporting_facts[0].evidence_segment_ids == [0]
    semantic.supporting_facts[0].evidence_segment_ids = []
    with pytest.raises(SemanticExtractionError, match="evidence"):
        semantic.validate(context)

    semantic = extract_semantic_representation(context)
    semantic.supporting_facts[0].evidence_segment_ids = [999]
    semantic.source_evidence_map[semantic.supporting_facts[0].fact_id] = [999]
    with pytest.raises(SemanticExtractionError, match="неизвест"):
        semantic.validate(context)


def test_supporting_context_cannot_become_primary_narrative_material() -> None:
    context = _context()
    context.supporting_context.append(EvidenceSegment(1, 0, 5, "Supporting only.", FactSourceScope.SUPPORTING_CONTEXT))
    fact = SemanticFact("fact-001", "Supporting only.", [1], "Supporting only.", 0, 5, 1, FactSourceScope.SUPPORTING_CONTEXT, FactualityType.EXPLICIT)
    semantic = SemanticRepresentation("candidate-001", "en", content_type=extract_semantic_representation(context).content_type, main_idea=fact.statement, core_claim=fact.statement, supporting_facts=[fact], source_evidence_map={"fact-001": [1]})
    semantic.validate(context)
    plan = NarrativePlan("candidate-001", build_narrative_plan(extract_semantic_representation(_context()), AppConfig().transformation).transformation_mode, 35, 84, "", "", [], "", "", None, [], [], ["fact-001"], "", "", "")
    with pytest.raises(NarrativePlanningError, match="supporting_context"):
        plan.validate(semantic, False)


def test_narrative_plan_uses_approved_ids_and_target_word_count() -> None:
    context, semantic, _, _ = _draft()
    config = AppConfig().transformation
    plan = build_narrative_plan(semantic, config)
    assert plan.target_word_count == round(config.target_duration_seconds * config.target_words_per_second)
    plan.required_fact_ids = ["unknown"]
    with pytest.raises(NarrativePlanningError, match="unknown"):
        plan.validate(semantic, False)
    plan = build_narrative_plan(semantic, config)
    plan.optional_cta = "Subscribe"
    with pytest.raises(NarrativePlanningError, match="CTA"):
        plan.validate(semantic, False)


@pytest.mark.parametrize(
    ("source", "replacement", "expected"),
    [
        ("The plan costs 10 dollars.", "The plan costs 99 dollars.", "numbers"),
        ("The conversion is 10%.", "The conversion is 25%.", "numbers"),
        ("The plan costs 10 dollars.", "The plan costs $20.", "currency"),
        ("Alice tested the plan.", "Bob tested the plan.", "entities"),
        ("This does not work.", "This works.", "отрицание"),
        ("This may help clients.", "This helps clients.", "модальность"),
    ],
)
def test_grounding_rejects_unsafe_surface_or_semantic_changes(source: str, replacement: str, expected: str) -> None:
    context, semantic, _, draft = _draft(source)
    _set_sentence_text(draft, replacement)
    result = validate_script_grounding(draft, semantic, context, False)
    assert not result.passed
    assert any(expected in item for item in result.errors)


def test_grounding_preserves_opinion_and_allows_source_absolute() -> None:
    context, semantic, _, draft = _draft("I think this always helps.")
    assert validate_script_grounding(draft, semantic, context, False).passed
    _set_sentence_text(draft, "This always helps.")
    result = validate_script_grounding(draft, semantic, context, False)
    assert not result.passed
    assert any("opinion" in item for item in result.errors)


def test_quality_scoring_is_deterministic_and_penalises_repetition_and_fillers() -> None:
    context, semantic, _, draft = _draft("Useful source fact. Useful source fact. Useful source fact.")
    grounding = validate_script_grounding(draft, semantic, context, False)
    first = score_script_quality(draft, semantic, grounding, AppConfig().transformation)
    second = score_script_quality(draft, semantic, grounding, AppConfig().transformation)
    assert first.to_dict() == second.to_dict()
    assert first.penalties["repetition_penalty"] > 0


def test_local_fallback_preserves_order_removes_duplicates_and_handles_empty() -> None:
    context, semantic, _, _ = _draft("Well, first fact is here. First fact is here. Final fact is here.")
    fallback = build_local_fallback(context, semantic, AppConfig().transformation, reason=FallbackReason.AI_DISABLED)
    assert fallback.full_text.index("first fact") < fallback.full_text.index("Final fact")
    assert fallback.full_text.lower().count("first fact") == 1
    empty = _context("")
    with pytest.raises(Exception):
        build_local_fallback(empty, extract_semantic_representation(empty), AppConfig().transformation, FallbackReason.EMPTY_RESULT)


def test_local_generation_keeps_complete_boundary_when_required_facts_exceed_word_budget() -> None:
    context = _complete_boundary_context()
    config = AppConfig()
    config.transformation.target_duration_seconds = 20.0
    config.transformation.target_words_per_second = 0.5

    semantic = extract_semantic_representation(context)
    fallback = build_local_fallback(
        context, semantic, config.transformation, FallbackReason.AI_DISABLED,
    )
    assert fallback.word_count > 10
    assert {
        source_id
        for sentence in fallback.sentences
        for source_id in sentence.source_segment_ids
    } == {0, 1, 2}

    result = run_content_transformation(
        context, config.transformation, None, force_local=True,
    )

    assert result["status"] == "fallback"
    assert result["final_script"]["word_count"] > 10
    assert result["final_script"]["ending"].endswith("naturally.")
    assert {
        source_id
        for sentence in result["final_script"]["sentences"]
        for source_id in sentence["source_segment_ids"]
    } == {0, 1, 2}
    plan = build_production_plan(result, config.production)
    assert {item.transcript_segment_id for item in plan.dialogue_mappings} == {0, 1, 2}


def test_local_fallback_does_not_deduplicate_required_completion_evidence() -> None:
    context = _complete_boundary_context()
    context.primary_evidence[-1].text = context.primary_evidence[0].text
    context.transcript_text = context.primary_text()
    semantic = extract_semantic_representation(context)

    fallback = build_local_fallback(
        context, semantic, AppConfig().transformation, FallbackReason.AI_DISABLED,
    )

    assert fallback.sentences[0].text == fallback.sentences[-1].text
    assert fallback.sentences[-1].source_segment_ids == [2]
    assert semantic.supporting_facts[-1].fact_id in fallback.used_fact_ids


def test_mock_modes_repair_and_provider_failure_are_safe_and_deterministic() -> None:
    config = AppConfig()
    context = _context()
    config.transformation.mock_mode = "repair_success"
    repaired = run_content_transformation(context, config.transformation, MockProvider(config))
    assert repaired["status"] == "completed"
    assert repaired["repair_attempts"]
    assert repaired["validation"]["grounding"]["passed"]

    config.transformation.mock_mode = "provider_error"
    fallback = run_content_transformation(context, config.transformation, MockProvider(config))
    assert fallback["status"] == "fallback"
    assert fallback["fallback"]["reason"] == "provider_failure"
    assert fallback["validation"]["grounding"]["passed"]


def test_failed_repair_never_returns_unsafe_script() -> None:
    config = AppConfig()
    config.transformation.mock_mode = "repair_failure"
    result = run_content_transformation(_context(), config.transformation, MockProvider(config))
    assert result["status"] == "fallback"
    # The malformed provider payload is discarded before the deterministic
    # fallback is built, so it is reported as a structured-output failure.
    assert result["fallback"]["reason"] == "invalid_structured_output"
    assert result["validation"]["grounding"]["passed"]
    assert result["validation"]["final_script"]["passed"]


class _StaticTransformer:
    name = "openai"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def transform_compact(self, _context):
        return deepcopy(self.payload), {"provider": self.name, "model": "test-model", "api_errors": []}


def _provider_payload() -> tuple[object, dict]:
    context, semantic, plan, draft = _draft()
    return context, {
        "semantic_representation": semantic.to_dict(),
        "narrative_plan": plan.to_dict(),
        "script_draft": draft.to_dict(),
    }


def test_failed_provider_draft_rebuilds_final_fallback_from_local_source_semantics() -> None:
    context = _context(
        "Original source premise is grounded. The final source sentence completes the idea."
    )
    provider_semantic = extract_semantic_representation(context)
    provider_semantic.supporting_facts[0].statement = "Zanzibar fabricated premise."
    provider_semantic.main_idea = provider_semantic.supporting_facts[0].statement
    provider_semantic.core_claim = provider_semantic.supporting_facts[0].statement
    provider_semantic.target_viewer_takeaway = provider_semantic.supporting_facts[0].statement
    provider_plan = build_narrative_plan(provider_semantic, AppConfig().transformation)
    provider_draft = generate_script_draft(provider_semantic, provider_plan, 2.4)
    payload = {
        "semantic_representation": provider_semantic.to_dict(),
        "narrative_plan": provider_plan.to_dict(),
        "script_draft": provider_draft.to_dict(),
    }

    result = run_content_transformation(
        context, AppConfig().transformation, _StaticTransformer(payload),
    )

    assert result["status"] == "fallback"
    assert result["fallback"]["reason"] == "repair_failed"
    assert result["repair_attempts"]
    assert result["validation"]["grounding"]["passed"]
    assert result["validation"]["final_script"]["passed"]
    assert "Zanzibar" not in result["final_script"]["full_text"]
    assert result["final_script"]["full_text"] == context.primary_text()
    assert all(
        fact["statement"] in context.primary_text()
        for fact in result["semantic_representation"]["supporting_facts"]
    )


@pytest.mark.parametrize("returned_candidate_id", ["", "candidate-other"])
def test_provider_candidate_identity_is_forced_to_current_candidate(returned_candidate_id: str) -> None:
    context, payload = _provider_payload()
    for item in payload.values():
        item["candidate_id"] = returned_candidate_id
    payload["semantic_representation"]["source_evidence_map"] = {"unrelated": [0]}

    result = run_content_transformation(context, AppConfig().transformation, _StaticTransformer(payload))

    assert result["status"] == "completed"
    assert result["final_script"]["candidate_id"] == context.candidate_id
    assert result["validation"]["final_script"]["passed"]
    assert result["semantic_representation"]["source_evidence_map"] == {
        item["fact_id"]: item["evidence_segment_ids"]
        for item in result["semantic_representation"]["supporting_facts"]
    }
    assert result["normalization"]["warnings"]


@pytest.mark.parametrize("invalid_part", ["narrative_plan", "script_draft"])
def test_invalid_provider_narrative_or_draft_uses_grounded_current_candidate_fallback(invalid_part: str) -> None:
    context, payload = _provider_payload()
    payload[invalid_part] = {}

    result = run_content_transformation(context, AppConfig().transformation, _StaticTransformer(payload))

    assert result["status"] == "fallback"
    assert result["fallback"]["reason"] == "invalid_structured_output"
    assert result["final_script"]["candidate_id"] == context.candidate_id
    assert result["final_script"]["sentences"]
    assert result["validation"]["final_script"]["passed"]
    primary_ids = {item.segment_id for item in context.primary_evidence}
    assert all(
        set(sentence["source_segment_ids"]).issubset(primary_ids)
        for sentence in result["final_script"]["sentences"]
    )


def test_empty_candidate_transcript_cannot_create_a_final_script() -> None:
    result = run_content_transformation(_context(""), AppConfig().transformation, None, force_local=True)

    assert result["status"] == "failed"
    assert not result["validation"]["final_script"]["passed"]
    assert not result["final_script"].get("sentences")


@pytest.mark.parametrize("mutation", ["candidate", "sentences", "text"])
def test_final_script_contract_rejects_missing_candidate_sentences_or_text(mutation: str) -> None:
    context, semantic, _plan, draft = _draft()
    final = FinalScript.from_draft(draft, "completed", True).to_dict()
    if mutation == "candidate":
        final["candidate_id"] = ""
    elif mutation == "sentences":
        final["sentences"] = []
    else:
        final["sentences"][0]["text"] = ""

    validation = validate_final_script(final, context, semantic, context.candidate_id)

    assert not validation.passed


@pytest.mark.parametrize("candidate_id", ["candidate-011", "candidate-023", "candidate-039"])
def test_final_script_contract_is_valid_for_each_selected_candidate(candidate_id: str) -> None:
    context = _context("The source statement remains grounded. The second statement is also grounded.")
    context.candidate_id = candidate_id

    result = run_content_transformation(context, AppConfig().transformation, None, force_local=True)

    assert result["final_script"]["candidate_id"] == candidate_id
    assert result["validation"]["final_script"]["passed"]


class _Responses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def test_openai_transformer_uses_responses_strict_schema() -> None:
    context, semantic, plan, draft = _draft()
    response = SimpleNamespace(
        output_text=json.dumps({
            "semantic_representation": semantic.to_dict(),
            "narrative_plan": plan.to_dict(),
            "script_draft": draft.to_dict(),
        }),
        usage=SimpleNamespace(input_tokens=7, output_tokens=9),
        _request_id="req-safe",
    )
    responses = _Responses(response)
    provider = OpenAIProvider(AppConfig(), "sk-test-secret", SimpleNamespace(responses=responses))
    data, usage = provider.transform_compact(context)
    assert data["semantic_representation"]["candidate_id"] == context.candidate_id
    assert usage["request_id"] == "req-safe"
    assert responses.calls[0]["text"] == {
        "format": {"type": "json_schema", "name": "grounded_content_transformation", "strict": True, "schema": OPENAI_TRANSFORMATION_RESPONSE_SCHEMA}
    }


def test_openai_transformation_connection_wait_is_bounded_without_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_options: list[dict[str, object]] = []

    class FailingResponses:
        @staticmethod
        def create(**_kwargs: object) -> object:
            raise ConnectionError("offline")

    def client(**options: object) -> object:
        constructor_options.append(options)
        return SimpleNamespace(responses=FailingResponses())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=0)), "sk-test-secret",
    )

    with pytest.raises(TransformationProviderError):
        provider.transform_compact(_context())

    assert constructor_options == [{
        "api_key": "sk-test-secret",
        "timeout": TRANSFORMATION_REQUEST_TIMEOUT_SECONDS,
        "max_retries": 0,
    }]


def test_openai_repair_uses_bounded_strict_script_schema() -> None:
    context, semantic, plan, draft = _draft()
    response = SimpleNamespace(output_text=json.dumps(draft.to_dict()), usage=SimpleNamespace(input_tokens=3, output_tokens=4))
    responses = _Responses(response)
    provider = OpenAIProvider(AppConfig(), "sk-test-secret", SimpleNamespace(responses=responses))
    repaired, usage = provider.repair_script(context, semantic.to_dict(), plan.to_dict(), draft.to_dict(), ["unsupported number"])
    assert repaired["candidate_id"] == context.candidate_id
    assert usage["input_tokens"] == 3
    assert responses.calls[0]["text"] == {
        "format": {"type": "json_schema", "name": "grounded_script_repair", "strict": True, "schema": OPENAI_SCRIPT_DRAFT_SCHEMA}
    }


def test_transformation_cache_and_artifacts_do_not_touch_render(tmp_path: Path) -> None:
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    pipeline = Pipeline(tmp_path, config, mock_ai=True, transform_script=True)
    tracker = StageTracker(tmp_path / "state.json")
    candidate = Candidate("candidate-001", 0, 20, "A complete source sentence. Another complete source sentence.", transcript_segment_ids=[0])
    scored = ScoredCandidate(candidate, "", "", "", 90, 90, 90, 60, 90, 10, None, True)
    transcript = {"language": "en", "segments": [{"start": 0, "end": 20, "text": candidate.text}]}
    features = {"segments": [{"id": 0, "start": 0, "end": 20, "sentence_start": True, "sentence_end": True, "speech_density": 0.5, "pause_before_seconds": 0, "pause_after_seconds": 0, "filler_word_ratio": 0, "repetition_score": 0}]}
    first = pipeline._transform_selected(tracker, {"id": "s", "path": "source.mp4"}, {}, [scored], transcript, features, {}, {"boundaries": []}, tmp_path / "work", tmp_path / "output")
    second = pipeline._transform_selected(tracker, {"id": "s", "path": "source.mp4"}, {}, [scored], transcript, features, {}, {"boundaries": []}, tmp_path / "work", tmp_path / "output")
    assert first["final_script"]["full_text"] == second["final_script"]["full_text"]
    assert second["cache"]["hit_count"] == 1
    assert (tmp_path / "output" / "transformed-script.txt").is_file()
    assert (tmp_path / "output" / "transformed-script.json").is_file()
    assert not list((tmp_path / "output").glob("*.ass"))


def test_legacy_cached_final_script_is_invalidated_before_production(tmp_path: Path) -> None:
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    pipeline = Pipeline(tmp_path, config, mock_ai=True, transform_script=True)
    tracker = StageTracker(tmp_path / "state.json")
    candidate = Candidate("candidate-001", 0, 20, "A complete source sentence. Another complete source sentence.", transcript_segment_ids=[0])
    scored = ScoredCandidate(candidate, "", "", "", 90, 90, 90, 60, 90, 10, None, True)
    transcript = {"language": "en", "segments": [{"start": 0, "end": 20, "text": candidate.text}]}
    features = {"segments": [{"id": 0, "start": 0, "end": 20, "sentence_start": True, "sentence_end": True, "speech_density": 0.5, "pause_before_seconds": 0, "pause_after_seconds": 0, "filler_word_ratio": 0, "repetition_score": 0}]}
    arguments = ({"id": "s", "path": "source.mp4"}, {}, [scored], transcript, features, {}, {"boundaries": []}, tmp_path / "work", tmp_path / "output")
    pipeline._transform_selected(tracker, *arguments)
    artifact = tmp_path / "work" / "transformation-candidate-001.json"
    cached = json.loads(artifact.read_text(encoding="utf-8"))
    cached["final_script"]["candidate_id"] = ""
    artifact.write_text(json.dumps(cached), encoding="utf-8")

    second = pipeline._transform_selected(tracker, *arguments)

    assert second["cache"]["hit_count"] == 0
    assert second["validation"]["final_script"]["passed"]
    assert second["final_script"]["candidate_id"] == candidate.id
    assert any("invalidated" in warning.lower() for warning in pipeline.warnings)


def test_provider_failure_is_not_cached_as_a_success(tmp_path: Path) -> None:
    config = AppConfig(score_threshold=0)
    config.transformation.enabled = True
    config.transformation.mock_mode = "provider_error"
    pipeline = Pipeline(tmp_path, config, mock_ai=True, transform_script=True)
    tracker = StageTracker(tmp_path / "state.json")
    candidate = Candidate("candidate-provider-error", 0, 20, "A complete source sentence.", transcript_segment_ids=[0])
    scored = ScoredCandidate(candidate, "", "", "", 90, 90, 90, 60, 90, 10, None, True)
    transcript = {"language": "en", "segments": [{"start": 0, "end": 20, "text": candidate.text}]}
    features = {"segments": [{"id": 0, "start": 0, "end": 20, "sentence_start": True, "sentence_end": True, "speech_density": 0.5, "pause_before_seconds": 0, "pause_after_seconds": 0, "filler_word_ratio": 0, "repetition_score": 0}]}
    kwargs = ({"id": "s", "path": "source.mp4"}, {}, [scored], transcript, features, {}, {"boundaries": []}, tmp_path / "work", tmp_path / "output")
    first = pipeline._transform_selected(tracker, *kwargs)
    second = pipeline._transform_selected(tracker, *kwargs)
    assert first["fallback"]["reason"] == "provider_failure"
    assert second["cache"]["hit_count"] == 0


def test_transformation_error_is_redacted_from_result() -> None:
    class BrokenProvider:
        name = "openai"

        def transform_compact(self, context):
            raise TransformationProviderError("Authorization: Bearer sk-super-secret")

    result = run_content_transformation(_context(), AppConfig().transformation, BrokenProvider())
    serialized = json.dumps(result)
    assert "sk-super-secret" not in serialized
    assert "[REDACTED]" in serialized


def test_transformation_cli_flags_apply_without_disabling_ai_reranking() -> None:
    arguments = build_parser().parse_args([
        "process", "--input", "source.mp4", "--transform-script", "--no-ai-transformation",
        "--transformation-mode", "hook_first", "--transformation-ai-strategy", "local_only",
        "--target-duration", "30", "--print-transformed-script", "--recompute-transformation",
    ])
    config = AppConfig()
    _apply_transformation_arguments(config, arguments)
    assert arguments.transform_script is True
    assert arguments.no_ai_transformation is True
    assert arguments.no_ai_rerank is False
    assert arguments.recompute_transformation is True
    assert config.transformation.mode == "hook_first"
    assert config.transformation.ai_strategy == "local_only"
    assert config.transformation.target_duration_seconds == 30
