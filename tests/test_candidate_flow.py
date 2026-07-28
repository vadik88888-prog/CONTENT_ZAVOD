from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.cli import main
from app.config import AppConfig
from app.content_transformation import run_content_transformation
from app.models import Candidate
from app.pipeline import Pipeline, PipelineResult, StageTracker, build_candidate_flow, build_terminal_state
from app.semantic_extraction import build_source_context
from app.utils import write_json


GAMEPLAY_IDS = (
    "candidate-chapter-004-story-001",
    "candidate-chapter-008-story-002",
    "candidate-chapter-021-story-001",
)


def _gameplay_transformation(candidate_id: str, start: float, text: str) -> tuple[Candidate, dict]:
    config = AppConfig()
    config.transformation.ai_strategy = "local_only"
    candidate = Candidate(candidate_id, start, start + 18, text, transcript_segment_ids=[0])
    transcript = {"language": "en", "segments": [{"start": start, "end": start + 18, "text": text}]}
    features = {"segments": [{
        "id": 0, "start": start, "end": start + 18, "sentence_start": True, "sentence_end": True,
        "speech_density": 0.7, "pause_before_seconds": 0.2, "pause_after_seconds": 0.3,
        "filler_word_ratio": 0.0, "repetition_score": 0.0,
    }]}
    context = build_source_context(
        {"id": "pubg-source", "path": "pubg_source.webm"}, {}, candidate, transcript,
        features, {}, {"boundaries": []}, config.transformation,
    )
    outcome = run_content_transformation(context, config.transformation, None, force_local=True)
    assert outcome["status"] in {"completed", "fallback"}
    return candidate, outcome


def test_gameplay_storyunit_outcomes_propagate_to_distinct_production_plans(tmp_path: Path) -> None:
    """Gameplay candidates retain their StoryUnit IDs through plan fan-out."""

    values = (
        (GAMEPLAY_IDS[0], 20.0, "We hear a team near the bridge, so we take cover before crossing the road."),
        (GAMEPLAY_IDS[1], 80.0, "The safe zone closes behind us, so we rotate early and keep the vehicle ready."),
        (GAMEPLAY_IDS[2], 140.0, "The final fight starts after we hold the high ground and wait for the last team."),
    )
    prepared = [_gameplay_transformation(*value) for value in values]
    candidates = [value[0] for value in prepared]
    transformation = {"items": [value[1] for value in prepared]}
    pipeline = Pipeline(tmp_path, AppConfig())
    production = pipeline._build_production_plans(
        StageTracker(tmp_path / "state.json"), transformation, tmp_path / "work", tmp_path / "output",
    )

    plans = [item for item in production["items"] if item["status"] == "completed"]
    assert production["status"] == "completed"
    assert [item["candidate_id"] for item in plans] == list(GAMEPLAY_IDS)
    assert len({item["production_plan_id"] for item in plans}) == 3

    rendered = {
        "enabled": True,
        "status": "completed",
        "items": [
            {"candidate_id": item["candidate_id"], "status": "completed", "clip_result_id": f"result-{index}"}
            for index, item in enumerate(plans, start=1)
        ],
    }
    flow = build_candidate_flow(
        [SimpleNamespace(candidate=candidate) for candidate in candidates], set(GAMEPLAY_IDS),
        transformation, production, rendered,
    )

    assert flow["transformed"] == flow["production_plans"] == flow["rendered"] == 3
    assert [item["candidate_id"] for item in flow["items"]] == list(GAMEPLAY_IDS)
    assert all(item["outcome"] == "selected" for item in flow["items"])


def test_candidate_flow_records_every_lost_transformation_before_zero_output() -> None:
    candidates = [Candidate(candidate_id, index * 20, index * 20 + 18, "A complete gameplay moment.") for index, candidate_id in enumerate(GAMEPLAY_IDS)]
    transformation = {
        "items": [
            {"candidate_id": candidate.id, "status": "completed", "final_script": {"candidate_id": candidate.id}}
            for candidate in candidates
        ]
    }
    flow = build_candidate_flow(
        [SimpleNamespace(candidate=candidate) for candidate in candidates], set(GAMEPLAY_IDS),
        transformation, {"enabled": True, "status": "failed", "items": []},
        {"enabled": True, "status": "skipped", "reason": "no_production_plan", "items": []},
    )
    terminal = build_terminal_state(3, [], flow, delivery_required=True)

    assert flow["selected"] == flow["transformed"] == 3
    assert flow["production_plans"] == flow["rendered"] == 0
    assert [item["reason"] for item in flow["items"]] == ["production_plan_failed"] * 3
    assert terminal["status"] == "failed"
    assert terminal["error_code"] == "NO_RENDERABLE_CLIPS"


def test_partial_candidate_batch_keeps_rendered_clips_and_terminal_warning(tmp_path: Path) -> None:
    candidates = [Candidate(candidate_id, index * 20, index * 20 + 18, "A complete gameplay moment.") for index, candidate_id in enumerate(GAMEPLAY_IDS)]
    transformation = {"items": [{"candidate_id": candidate.id, "status": "completed"} for candidate in candidates]}
    production = {
        "enabled": True,
        "status": "completed",
        "items": [
            {"candidate_id": candidate.id, "status": "completed", "production_plan_id": f"plan-{index}"}
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    rendered = {
        "enabled": True,
        "status": "warning",
        "items": [
            {"candidate_id": GAMEPLAY_IDS[0], "status": "completed", "clip_result_id": "result-1"},
            {"candidate_id": GAMEPLAY_IDS[1], "status": "completed", "clip_result_id": "result-2"},
            {"candidate_id": GAMEPLAY_IDS[2], "status": "failed", "errors": ["simulated render failure"]},
        ],
    }
    flow = build_candidate_flow(
        [SimpleNamespace(candidate=candidate) for candidate in candidates], set(GAMEPLAY_IDS), transformation, production, rendered,
    )
    terminal = build_terminal_state(3, [tmp_path / "one.mp4", tmp_path / "two.mp4"], flow, delivery_required=True)

    assert flow["rendered"] == 2 and flow["failed"] == 1
    assert terminal["status"] == "completed_with_warnings"
    assert terminal["error_code"] is None


def test_cli_returns_nonzero_for_reported_zero_output(tmp_path: Path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "report.json"
    write_json(report_path, {
        "terminal": {
            "status": "failed",
            "error_code": "NO_RENDERABLE_CLIPS",
            "message": "Не удалось подготовить ни одного ролика к созданию.",
        },
    })
    result = PipelineResult(tmp_path, tmp_path, report_path, 0, [], [], "failed", "NO_RENDERABLE_CLIPS")

    class FakePipeline:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, **_kwargs) -> PipelineResult:
            return result

    monkeypatch.setattr("app.cli.load_config", lambda _path: AppConfig())
    monkeypatch.setattr("app.cli.Pipeline", FakePipeline)

    assert main(["process", "--input", str(tmp_path / "pubg_source.webm")]) == 2
    assert "NO_RENDERABLE_CLIPS" in capsys.readouterr().err
