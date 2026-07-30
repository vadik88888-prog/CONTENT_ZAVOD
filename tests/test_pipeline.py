import json

from app.config import AppConfig
from app.pipeline import Pipeline, RunHeartbeat, StageTracker


def test_no_candidates_do_not_require_a_gemini_key(tmp_path) -> None:
    path = tmp_path / "candidates.scored.json"
    data = Pipeline(tmp_path, AppConfig())._score_candidates([], {"language": "ru"}, path)

    assert data["candidates"] == []
    assert data["ai"]["provider"] == "not-called"
    assert path.exists()


def test_state_invalidation_marks_completed_steps_pending(tmp_path) -> None:
    state = StageTracker(tmp_path / "state.json")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    state.start("metadata")
    state.finish("metadata")

    assert state.completed("metadata", artifact)
    state.invalidate("source changed")

    assert not state.completed("metadata", artifact)
    assert state.data["stages"]["metadata"]["status"] == "pending"


def test_state_can_persist_configuration_signature(tmp_path) -> None:
    state = StageTracker(tmp_path / "state.json")

    state.set_config_signature("abc123")

    restored = StageTracker(tmp_path / "state.json")
    assert restored.data["config_signature"] == "abc123"


def test_stage_tracker_updates_engine_heartbeat(tmp_path) -> None:
    heartbeat = RunHeartbeat(tmp_path / "heartbeat.json")
    heartbeat.start()
    try:
        state = StageTracker(tmp_path / "state.json", heartbeat=heartbeat)
        state.start("transcription")

        payload = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
        assert payload["stage"] == "transcription"
        assert payload["pid"] > 0
    finally:
        heartbeat.stop()
