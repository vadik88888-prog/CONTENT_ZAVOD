from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from app.ai import (
    GeminiProvider,
    MockProvider,
    OPENAI_SCORE_RESPONSE_SCHEMA,
    OpenAIProvider,
    SEMANTIC_CONNECT_TIMEOUT_SECONDS,
    SEMANTIC_POOL_TIMEOUT_SECONDS,
    SEMANTIC_READ_TIMEOUT_SECONDS,
    SEMANTIC_WRITE_TIMEOUT_SECONDS,
    get_scorer,
)
from app.config import AIConfig, AppConfig, load_config
from app.doctor import collect_checks
from app.errors import ClipEngineError
from app.models import AI_FIELDS, Candidate
from app.reporting import make_report
from app.vision_intelligence import VISION_RESPONSE_SCHEMA, VisionProviderCallError


def _candidate() -> Candidate:
    return Candidate("candidate-1", 10.0, 30.0, "Законченное высказывание для клипа.")


def _structured_item(candidate: Candidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.id,
        "start": candidate.start,
        "end": candidate.end,
        "title": "Заголовок",
        "hook": "Хук",
        "summary": "Краткое описание",
        "score": 88,
        "hook_score": 90,
        "completeness_score": 85,
        "emotional_score": 70,
        "clarity_score": 92,
        "context_dependency_score": 15,
        "rejection_reason": None,
        "selected": True,
    }


class _FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def test_openai_provider_maps_structured_response() -> None:
    candidate = _candidate()
    responses = _FakeResponses(SimpleNamespace(
        output_text=json.dumps({"candidates": [_structured_item(candidate)]}),
        usage=SimpleNamespace(
            input_tokens=123,
            output_tokens=45,
            input_tokens_details=SimpleNamespace(cached_tokens=80, cache_write_tokens=30),
        ),
    ))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(model="gpt-5-mini")),
        "sk-test-secret",
        SimpleNamespace(responses=responses),
    )

    scored, usage = provider.score([candidate], {"language": "ru", "segments": []})

    assert scored[0].title == "Заголовок"
    assert scored[0].score == 88
    assert usage["provider"] == "openai"
    assert usage["model"] == "gpt-5-mini"
    assert usage["input_tokens"] == 123
    assert usage["cached_input_tokens"] == 80
    assert usage["cache_write_input_tokens"] == 30
    assert usage["output_tokens"] == 45
    call = responses.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": "clip_candidate_scores",
            "strict": True,
            "schema": OPENAI_SCORE_RESPONSE_SCHEMA,
        }
    }
    item_schema = OPENAI_SCORE_RESPONSE_SCHEMA["properties"]["candidates"]["items"]
    assert set(item_schema["required"]) == {"candidate_id", *AI_FIELDS}


def test_openai_provider_rejects_unknown_candidate_id() -> None:
    candidate = _candidate()
    item = _structured_item(candidate)
    item["candidate_id"] = "not-in-shortlist"
    responses = _FakeResponses(SimpleNamespace(
        output_text=json.dumps({"candidates": [item]}),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    ))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=0)),
        "sk-test-secret",
        SimpleNamespace(responses=responses),
    )

    scored, usage = provider.score([candidate], {"segments": []})

    assert not scored[0].selected
    assert usage["api_errors"]
    assert "candidate_id" in usage["api_errors"][0]


def test_openai_provider_binds_reordered_assessments_by_candidate_id() -> None:
    first = _candidate()
    second = Candidate("candidate-2", 30.0, 55.0, "Another complete candidate.")
    first_item = _structured_item(first)
    second_item = _structured_item(second)
    first_item["score"] = 71
    second_item["score"] = 93
    responses = _FakeResponses(SimpleNamespace(
        output_text=json.dumps({"candidates": [second_item, first_item]}),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    ))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=0)),
        "sk-test-secret",
        SimpleNamespace(responses=responses),
    )

    scored, usage = provider.score([first, second], {"segments": []})

    assert [item.candidate.id for item in scored] == [first.id, second.id]
    assert [item.score for item in scored] == [71, 93]
    assert usage["api_errors"] == []


def test_openai_semantic_connection_wait_is_bounded_without_sdk_retries(
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

    scored, usage = provider.score([_candidate()], {"segments": []})

    assert len(constructor_options) == 1
    assert constructor_options[0]["api_key"] == "sk-test-secret"
    assert constructor_options[0]["max_retries"] == 0
    timeout = constructor_options[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.as_dict() == {
        "connect": SEMANTIC_CONNECT_TIMEOUT_SECONDS,
        "read": SEMANTIC_READ_TIMEOUT_SECONDS,
        "write": SEMANTIC_WRITE_TIMEOUT_SECONDS,
        "pool": SEMANTIC_POOL_TIMEOUT_SECONDS,
    }
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["api_errors"] == [
        "Semantic AI attempt 1/1 connect_failure "
        "(connect_timeout=10s; read_timeout=240s; sdk_retries=0): offline"
    ]
    assert "Semantic AI attempt 1/1" in (scored[0].rejection_reason or "")


def test_openai_semantic_retries_only_one_genuine_connection_failure() -> None:
    class FailingResponses:
        calls = 0

        @classmethod
        def create(cls, **_kwargs: object) -> object:
            cls.calls += 1
            raise ConnectionError("offline")

    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=2)),
        "sk-test-secret",
        SimpleNamespace(responses=FailingResponses()),
    )

    _, usage = provider.score([_candidate()], {"segments": []})

    assert FailingResponses.calls == 2
    assert usage["retries"] == 1
    assert ["connect_failure" in item for item in usage["api_errors"]] == [True, True]


def test_openai_semantic_retries_real_sdk_connection_exception_once() -> None:
    request = httpx.Request("POST", "https://api.openai.test/v1/responses")

    class FailingResponses:
        calls = 0

        @classmethod
        def create(cls, **_kwargs: object) -> object:
            cls.calls += 1
            raise APIConnectionError(request=request)

    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=2)),
        "sk-test-secret",
        SimpleNamespace(responses=FailingResponses()),
    )

    _, usage = provider.score([_candidate()], {"segments": []})

    assert FailingResponses.calls == 2
    assert usage["retries"] == 1
    assert ["connect_failure" in item for item in usage["api_errors"]] == [True, True]
    assert all("Connection error." in item for item in usage["api_errors"])


def test_openai_semantic_response_timeout_never_retries() -> None:
    request = httpx.Request("POST", "https://api.openai.test/v1/responses")

    class ReadTimingOutResponses:
        calls = 0

        @classmethod
        def create(cls, **_kwargs: object) -> object:
            cls.calls += 1
            raise httpx.ReadTimeout("generation deadline", request=request)

    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=2)),
        "sk-test-secret",
        SimpleNamespace(responses=ReadTimingOutResponses()),
    )

    _, usage = provider.score([_candidate()], {"segments": []})

    assert ReadTimingOutResponses.calls == 1
    assert usage["retries"] == 0
    assert "response_timeout" in usage["api_errors"][0]


def test_openai_vision_adapter_sends_real_frame_payload_once_with_strict_schema() -> None:
    frame = {
        "keyframe_id": "keyframe-1",
        "timestamp": 2.5,
        "image_base64": "anBlZy1ieXRlcw==",
    }
    observation = {
        "keyframe_id": "keyframe-1", "timestamp": 2.5,
        "scene_type": "TALKING_HEAD", "primary_subject": "face",
        "normalized_center_x": 0.5, "normalized_center_y": 0.4,
        "visible_face_count": 1, "action": "speaking", "reaction": "none",
        "payoff_signal": "none", "on_screen_text": "", "composition_risk": "none",
        "confidence": 0.9, "missing_evidence": ["text"],
    }
    responses = _FakeResponses(SimpleNamespace(
        output_text=json.dumps({"observations": [observation]}),
        usage=SimpleNamespace(
            input_tokens=321,
            output_tokens=87,
            input_tokens_details={"cached_tokens": 200, "cache_write_tokens": 50},
        ),
        request_id="request-vision-1",
    ))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(model="gpt-5-mini")),
        "sk-test-secret",
        SimpleNamespace(responses=responses),
    )

    payload, usage = provider.analyze_vision(
        [frame], detail="low", pass_kind="pass1", max_output_tokens=500,
    )

    assert payload["observations"][0]["keyframe_id"] == "keyframe-1"
    assert usage["input_tokens"] == 321
    assert usage["cached_input_tokens"] == 200
    assert usage["cache_write_input_tokens"] == 50
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["max_output_tokens"] == 500
    assert call["text"] == {
        "format": {
            "type": "json_schema", "name": "vision_pass1_observations",
            "strict": True, "schema": VISION_RESPONSE_SCHEMA,
        }
    }
    content = call["input"][0]["content"]
    assert any(item.get("type") == "input_image" and item.get("detail") == "low" for item in content)


def test_openai_vision_adapter_preserves_usage_when_output_json_is_empty() -> None:
    responses = _FakeResponses(SimpleNamespace(
        output_text="",
        usage=SimpleNamespace(input_tokens=555, output_tokens=333),
        request_id="request-empty-vision",
    ))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(model="gpt-5-mini")),
        "sk-test-secret",
        SimpleNamespace(responses=responses),
    )

    with pytest.raises(VisionProviderCallError) as captured:
        provider.analyze_vision(
            [{"keyframe_id": "keyframe-1", "timestamp": 2.5, "image_base64": "anBlZw=="}],
            detail="low", pass_kind="pass1", max_output_tokens=500,
        )

    assert captured.value.usage["input_tokens"] == 555
    assert captured.value.usage["output_tokens"] == 333


def test_mock_mode_never_selects_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    scorer = get_scorer(AppConfig(), force_mock=True)

    assert isinstance(scorer, MockProvider)


def test_missing_openai_key_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ClipEngineError, match="OPENAI_API_KEY"):
        get_scorer(AppConfig())


def test_doctor_checks_only_selected_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    checks = collect_checks(tmp_path, AppConfig())
    by_label = {check.label: check for check in checks}

    assert by_label["AI provider"].detail == "openai · gpt-5-mini"
    assert by_label["OpenAI API key"].status == "error"
    assert "Gemini API key" not in by_label


def test_gemini_provider_can_still_be_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    config = AppConfig(ai=AIConfig(provider="gemini", model="gemini-2.5-flash"))

    assert isinstance(get_scorer(config), GeminiProvider)


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "ai:\n  provider: unknown\n  model: gpt-5-mini\n  max_retries: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ClipEngineError, match="ai.provider"):
        load_config(config_path)


def test_secret_is_redacted_from_api_usage_and_report(tmp_path: Path) -> None:
    candidate = _candidate()
    secret = "sk-super-secret-value"
    responses = _FakeResponses(RuntimeError(f"API rejected Authorization: Bearer {secret}"))
    provider = OpenAIProvider(
        AppConfig(ai=AIConfig(max_retries=0)),
        secret,
        SimpleNamespace(responses=responses),
    )

    _, usage = provider.score([candidate], {"segments": []})
    report_path = tmp_path / "report.json"
    make_report(
        report_path,
        {},
        {},
        AppConfig(),
        {},
        0,
        1,
        [],
        [],
        [],
        usage,
        False,
        False,
    )

    serialized = report_path.read_text(encoding="utf-8")
    assert secret not in serialized
    assert "[REDACTED]" in serialized
