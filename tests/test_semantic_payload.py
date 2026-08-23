from __future__ import annotations

import json

from app.ai import (
    OPENAI_SCORE_RESPONSE_SCHEMA,
    SEMANTIC_AI_PAYLOAD_VERSION,
    build_openai_payload,
)
from app.models import Candidate


def _serialized_size(value: object) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _technical_words(count: int = 400) -> list[dict[str, object]]:
    return [
        {
            "word": f"word-{index}",
            "start": index / 10,
            "end": index / 10 + 0.09,
            "probability": 0.99,
            "cache_locator": f"segments[0].words[{index}]",
        }
        for index in range(count)
    ]


def _evidence_heavy_candidate() -> Candidate:
    words = _technical_words()
    candidate = Candidate(
        "candidate-compact",
        10.125,
        34.875,
        "A complete setup explains the challenge, then the action delivers the payoff.",
        core_idea="A risky move succeeds after a clear setup.",
        candidate_kind="multimodal",
    )
    candidate.semantic_evidence = {
        "hook": "Can the risky move work?",
        "setup": "The player explains the challenge.",
        "payoff": "The move succeeds.",
        "ending": "The result is shown and explained.",
        "completeness_score": 0.92,
        "context_dependency_score": 0.08,
        "information_density": 0.81,
        "evidence": {"word_timestamps": words, "debug_trace": ["node"] * 400},
    }
    candidate.boundary_diagnostics = {
        "word_integrity": True,
        "sentence_integrity": True,
        "semantic_completion": 0.94,
        "context_independence": 0.91,
        "head_naturalness": 0.9,
        "tail_naturalness": 0.95,
        "payoff_preserved": True,
        "continuation_risk": "low",
        "safe_start_points": words,
        "boundary_decision": {"internal_tree": ["branch"] * 400},
    }
    candidate.feature_vector = {
        "transcript_confidence": 0.96,
        "speech_density": 0.78,
        "words_per_second": 2.7,
        "segment_ids": list(range(400)),
        "debug_cache": words,
    }
    candidate.multimodal_provenance = {
        "transcript_evidence": [{"words": words, "cache_key": "transcript-cache"}],
        "audio_evidence": [{
            "event_id": "audio-1",
            "event_type": "emphasis",
            "start_seconds": 30.1,
            "end_seconds": 30.3,
            "confidence": 0.88,
            "observation": {"normalized_loudness": 0.88, "threshold": 0.72},
            "provenance": [{"artifact": "audio_features.json", "locator": "energy[301]"}],
        }],
        "visual_evidence": [{
            "timestamp": 30.2,
            "scene_type": "GAMEPLAY",
            "primary_subject": "screen",
            "action": "interaction",
            "reaction": "surprise",
            "payoff_signal": "payoff",
            "on_screen_text": "WIN",
            "composition_risk": "none",
            "confidence": 0.9,
            "missing_evidence": [],
            "provenance": {"cache_key": "pass1-cache", "request_id": "request-pass1"},
        }],
        "keyframe_evidence": [{"time_seconds": 30.2, "provenance": words}],
        "generation": {
            "anchors": {"hook": 10.125, "action": 24.0, "reaction": 30.1, "payoff": 30.2},
            "initial_filter_score": 0.99,
            "reasons": ["internal decision tree"],
        },
        "analysis_run_id": "internal-run-id",
    }
    candidate.vision_pass2_evidence = {
        "status": "completed",
        "result": {
            "verification": {
                "hook_visible": True,
                "action_visible": True,
                "reaction_visible": True,
                "payoff_visible": True,
                "continuity_risk": "low",
                "confidence": 0.94,
                "internal_decision": "omit",
            },
            "observations": [{
                "timestamp": 30.2,
                "scene_type": "GAMEPLAY",
                "primary_subject": "screen",
                "action": "interaction",
                "reaction": "surprise",
                "payoff_signal": "payoff",
                "on_screen_text": "WIN",
                "composition_risk": "none",
                "confidence": 0.94,
                "missing_evidence": [],
                "normalized_center_x": 0.5,
                "provenance": {"cache_key": "pass2-cache", "request_id": "request-pass2"},
            }],
            "diagnostics": {"word_timestamps": words, "cache_key": "vision-cache"},
            "request": {"frames": words},
        },
    }
    candidate.content_signature = {"semantic_embedding_ref": "cache://embedding"}
    candidate.local_scores = {"weighted_score": 99, "debug_tree": words}
    candidate.composition_intent = {"internal_decision_tree": words}
    candidate.explanations = ["debug explanation"] * 400
    return candidate


def test_openai_semantic_payload_is_compact_allowlisted_and_keeps_global_context() -> None:
    candidate = _evidence_heavy_candidate()
    transcript = {
        "language": "en",
        "source_path": "C:/private/source.mp4",
        "segments": [
            {
                "start": 0,
                "end": 40,
                "text": "Global context before, during, and after the candidate.",
                "words": _technical_words(40),
            },
            {"start": 40, "end": 50, "text": "The complete transcript remains available."},
        ],
    }

    payload = build_openai_payload([candidate], transcript)
    compact = payload["candidates"][0]
    persisted = candidate.to_dict()
    legacy_payload = {
        "language": transcript["language"],
        "transcript": payload["transcript"],
        "candidates": [persisted],
        "instruction": payload["instruction"],
    }

    assert payload["semantic_payload_version"] == SEMANTIC_AI_PAYLOAD_VERSION
    assert payload["semantic_payload_version"] == "semantic-score.5"
    assert payload["transcript"] == [
        {
            "start": 0.0,
            "end": 40.0,
            "text": "Global context before, during, and after the candidate.",
        },
        {
            "start": 40.0,
            "end": 50.0,
            "text": "The complete transcript remains available.",
        },
    ]
    assert "source_path" not in payload
    assert set(compact) == {
        "candidate_id", "start", "end", "duration", "text", "core_idea",
        "semantic_evidence", "boundary_signals", "speech_signals",
        "multimodal_signals",
    }
    assert set(compact["semantic_evidence"]) == {
        "hook", "setup", "payoff", "ending", "completeness_score",
        "information_density",
    }
    assert "context_dependency_score" not in compact["semantic_evidence"]
    assert "context_independence" not in compact["boundary_signals"]
    assert compact["boundary_signals"]["payoff_preserved"] is True
    assert compact["speech_signals"] == {
        "transcript_confidence": 0.96,
        "speech_density": 0.78,
        "words_per_second": 2.7,
    }
    multimodal = compact["multimodal_signals"]
    assert multimodal["anchors"] == {
        "hook": 10.125, "action": 24.0, "reaction": 30.1, "payoff": 30.2,
    }
    assert set(multimodal["audio_events"][0]) == {
        "event_type", "start_seconds", "end_seconds", "confidence",
    }
    assert multimodal["visual_observation_source"] == "vision_pass2"
    assert set(multimodal["visual_observations"][0]) == {
        "timestamp", "scene_type", "primary_subject", "action", "reaction",
        "payoff_signal", "on_screen_text", "composition_risk", "confidence",
        "missing_evidence",
    }
    assert multimodal["vision_verification"]["payoff_visible"] is True

    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        "transcript_evidence", "keyframe_evidence", "multimodal_provenance",
        "feature_vector", "candidate_score_v2", "eligibility_decision",
        "composition_intent", "safe_start_points", "diagnostics", "provenance",
        "cache_key", "request_id", "internal_decision",
    ):
        assert forbidden not in serialized

    assert persisted["multimodal_provenance"]["transcript_evidence"]
    assert persisted["vision_pass2_evidence"]["result"]["diagnostics"]
    assert _serialized_size(legacy_payload) >= _serialized_size(payload) * 4


def test_context_dependency_factor_contract_has_one_noninverted_polarity() -> None:
    candidate = _evidence_heavy_candidate()
    payload = build_openai_payload(
        [candidate],
        {
            "language": "en",
            "segments": [
                {
                    "start": 0,
                    "end": 40,
                    "text": "The full transcript provides prior and external context.",
                }
            ],
        },
    )

    contract = payload["factor_contract"]
    assert set(contract) == {
        "scale", "hook_score", "completeness_score", "emotional_score",
        "clarity_score", "context_dependency_score",
    }
    assert contract["scale"]["minimum"] == 0
    assert contract["scale"]["maximum"] == 100
    dependency = contract["context_dependency_score"]
    assert dependency["zero"] == "The candidate is completely understandable on its own."
    assert dependency["hundred"] == (
        "Understanding the candidate requires previous or external context."
    )
    assert dependency["direction"] == (
        "Higher means more context dependency; it never means more context independence."
    )

    compact = payload["candidates"][0]
    assert "context_dependency_score" not in compact["semantic_evidence"]
    assert "context_independence" not in compact["boundary_signals"]
    assert "independent assessment" in payload["instruction"]
    assert "must not be copied or mechanically rescaled" in payload["instruction"]

    response_property = OPENAI_SCORE_RESPONSE_SCHEMA["properties"]["candidates"][
        "items"
    ]["properties"]["context_dependency_score"]
    assert response_property["minimum"] == 0
    assert response_property["maximum"] == 100
    assert dependency["zero"] in response_property["description"]
    assert dependency["hundred"] in response_property["description"]
    assert dependency["direction"] in response_property["description"]


def test_semantic_payload_keeps_concise_pass1_signals_when_pass2_is_unavailable() -> None:
    candidate = _evidence_heavy_candidate()
    candidate.vision_pass2_evidence = {"status": "skipped", "reason": "provider unavailable"}

    payload = build_openai_payload([candidate], {"language": "en", "segments": []})
    multimodal = payload["candidates"][0]["multimodal_signals"]

    assert multimodal["vision_pass2_status"] == "skipped"
    assert multimodal["vision_verification"] == {}
    assert multimodal["visual_observation_source"] == "vision_pass1"
    assert multimodal["visual_observations"][0]["payoff_signal"] == "payoff"
    assert "provenance" not in multimodal["visual_observations"][0]
