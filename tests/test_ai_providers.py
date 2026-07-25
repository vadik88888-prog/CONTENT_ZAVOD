from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai import (
    GeminiProvider,
    MockProvider,
    OPENAI_SCORE_RESPONSE_SCHEMA,
    OpenAIProvider,
    get_scorer,
)
from app.config import AIConfig, AppConfig, load_config
from app.doctor import collect_checks
from app.errors import ClipEngineError
from app.models import AI_FIELDS, Candidate
from app.reporting import make_report


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
        usage=SimpleNamespace(input_tokens=123, output_tokens=45),
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
    assert by_label["OpenAI API key"].status == "warn"
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
