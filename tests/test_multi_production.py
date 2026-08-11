from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import AppConfig
from app.pipeline import Pipeline, StageTracker
from app.production_models import ProductionPlan
from app.sources import Source
from app.utils import write_json
from tests.test_audio_composition import _plan


def _production_with_two_plans() -> dict:
    items: list[dict] = []
    for number, candidate_id in enumerate(("clip-one", "clip-two"), start=1):
        raw = _plan().model_dump(mode="json")
        raw["plan_id"] = f"fanout-plan-{number}"
        raw["metadata"]["candidate_id"] = candidate_id
        plan = ProductionPlan.model_validate(raw)
        items.append({"candidate_id": candidate_id, "status": "completed", "plan": plan.model_dump(mode="json")})
    return {"enabled": True, "status": "completed", "items": items}


@pytest.mark.parametrize("fail_second", [False, True])
def test_every_production_plan_gets_an_isolated_output_and_partial_results_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_second: bool,
) -> None:
    """The production path must not collapse a multi-clip selection to its first plan."""

    config = AppConfig()
    config.tts.enabled = True
    config.audio_composition.enabled = True
    config.production_render.enabled = True
    pipeline = Pipeline(tmp_path, config)
    tracker = StageTracker(tmp_path / "state.json")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source("source", source_path, "source", "local")
    production = _production_with_two_plans()
    output_directory = tmp_path / "output"

    class FakeTTSService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def generate(self, _plan: ProductionPlan, _work: Path, destination: Path, **_kwargs: object) -> SimpleNamespace:
            (destination / "tts").mkdir(parents=True, exist_ok=True)
            write_json(destination / "tts" / "tts-result.json", {"status": "completed"})
            return SimpleNamespace(status="completed", warnings=[], api_errors=[])

    class FakeAudioService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def compose(self, _plan: ProductionPlan, *_args: object, output_directory: Path | None = None, **_kwargs: object) -> SimpleNamespace:
            # The production runner reads this path positionally in the real service.
            # Resolve it from positional arguments as well to match that public API.
            destination = output_directory
            if destination is None:
                destination = _args[4]
            assert isinstance(destination, Path)
            (destination / "audio").mkdir(parents=True, exist_ok=True)
            write_json(destination / "audio" / "audio-project.json", {"candidate": _plan.metadata.candidate_id})
            return SimpleNamespace(status="completed", warnings=[], errors=[])

    class FakeAudioProject:
        @classmethod
        def model_validate(cls, data: dict) -> dict:
            return data

    def fake_compose(
        _self: Pipeline, _tracker: StageTracker, plan: ProductionPlan, _audio: dict,
        _source: Source, _transcript: dict, _work: Path, destination: Path, **_kwargs: object,
    ) -> dict:
        if fail_second and plan.metadata.candidate_id == "clip-two":
            return {"enabled": True, "status": "failed", "errors": ["simulated render failure"]}
        artifact = destination / "production-render" / f"{plan.metadata.candidate_id}.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"mp4:{plan.metadata.candidate_id}".encode("utf-8"))
        return {"enabled": True, "status": "completed", "output_file": str(artifact)}

    monkeypatch.setattr("app.pipeline.TTSService", FakeTTSService)
    monkeypatch.setattr("app.pipeline.tts_report_section", lambda _result: {"enabled": True, "status": "completed"})
    monkeypatch.setattr("app.pipeline.AudioCompositionService", FakeAudioService)
    monkeypatch.setattr("app.pipeline.AudioProject", FakeAudioProject)
    monkeypatch.setattr("app.pipeline.audio_report_section", lambda _project: {"enabled": True, "status": "completed"})
    monkeypatch.setattr(Pipeline, "_compose_production_render", fake_compose)

    tts = pipeline._run_tts(tracker, production, tmp_path / "work", output_directory)
    audio = pipeline._run_audio(tracker, production, tts, source, {}, tmp_path / "work", output_directory)
    rendered = pipeline._run_production_render(tracker, production, audio, source, {}, tmp_path / "work", output_directory)

    assert len(tts["items"]) == 2
    assert len(audio["items"]) == 2
    assert (output_directory / "tts" / "tts-result.json").is_file()
    assert (output_directory / "candidates" / "clip-two" / "tts" / "tts-result.json").is_file()
    assert len(rendered["output_files"]) == (1 if fail_second else 2)
    assert len(rendered["clip_results"]) == (1 if fail_second else 2)
    assert all(Path(value).is_file() for value in rendered["output_files"])
    assert rendered["status"] == ("warning" if fail_second else "completed")


def test_unexpected_candidate_render_failure_does_not_stop_later_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.production_render.enabled = True
    pipeline = Pipeline(tmp_path, config, run_id="isolated-render-run")
    tracker = StageTracker(tmp_path / "state.json")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source("source", source_path, "source", "local")
    production = _production_with_two_plans()
    output_directory = tmp_path / "output"
    audio_items = []
    for index, candidate_id in enumerate(("clip-one", "clip-two"), start=1):
        destination = output_directory if index == 1 else output_directory / "candidates" / candidate_id
        (destination / "audio").mkdir(parents=True, exist_ok=True)
        write_json(destination / "audio" / "audio-project.json", {"candidate": candidate_id})
        audio_items.append({
            "candidate_id": candidate_id,
            "status": "completed",
            "output_directory": str(destination),
        })

    class FakeAudioProject:
        @classmethod
        def model_validate(cls, data: dict) -> dict:
            return data

    calls: list[str] = []

    def fake_compose(
        _self: Pipeline, _tracker: StageTracker, plan: ProductionPlan, _audio: dict,
        _source: Source, _transcript: dict, _work: Path, destination: Path, **_kwargs: object,
    ) -> dict:
        candidate_id = plan.metadata.candidate_id
        calls.append(candidate_id)
        if candidate_id == "clip-one":
            raise RuntimeError("candidate-one contract exploded")
        artifact = destination / "production-render" / "clip-two.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"mp4:clip-two")
        return {"enabled": True, "status": "completed", "output_file": str(artifact)}

    monkeypatch.setattr("app.pipeline.AudioProject", FakeAudioProject)
    monkeypatch.setattr(Pipeline, "_compose_production_render", fake_compose)

    rendered = pipeline._run_production_render(
        tracker,
        production,
        {"enabled": True, "status": "completed", "items": audio_items},
        source,
        {},
        tmp_path / "work",
        output_directory,
        render_profile="creative_preview",
    )

    assert calls == ["clip-one", "clip-two"]
    assert rendered["status"] == "partial"
    assert rendered["items"][0]["candidate_id"] == "clip-one"
    assert rendered["items"][0]["status"] == "failed"
    assert "candidate-one contract exploded" in rendered["items"][0]["error"]
    assert rendered["items"][1]["candidate_id"] == "clip-two"
    assert rendered["items"][1]["status"] == "completed"


def test_post_render_fingerprint_failure_does_not_stop_later_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.production_render.enabled = True
    pipeline = Pipeline(tmp_path, config, run_id="fingerprint-isolation-run")
    tracker = StageTracker(tmp_path / "state.json")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source("source", source_path, "source", "local")
    production = _production_with_two_plans()
    output_directory = tmp_path / "output"
    audio_items = []
    for index, candidate_id in enumerate(("clip-one", "clip-two"), start=1):
        destination = output_directory if index == 1 else output_directory / "candidates" / candidate_id
        (destination / "audio").mkdir(parents=True, exist_ok=True)
        write_json(destination / "audio" / "audio-project.json", {"candidate": candidate_id})
        audio_items.append({
            "candidate_id": candidate_id, "status": "completed",
            "output_directory": str(destination),
        })

    class FakeAudioProject:
        @classmethod
        def model_validate(cls, data: dict) -> dict:
            return data

    def fake_compose(
        _self: Pipeline, _tracker: StageTracker, plan: ProductionPlan, _audio: dict,
        _source: Source, _transcript: dict, _work: Path, destination: Path, **_kwargs: object,
    ) -> dict:
        artifact = destination / "production-render" / f"{plan.metadata.candidate_id}.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"mp4")
        return {"enabled": True, "status": "completed", "output_file": str(artifact)}

    fingerprint_calls = 0

    def fingerprint(_path: Path, _report: dict) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        if fingerprint_calls == 1:
            raise OSError("candidate-one output disappeared during fingerprint")
        return "second-candidate-fingerprint"

    monkeypatch.setattr("app.pipeline.AudioProject", FakeAudioProject)
    monkeypatch.setattr(Pipeline, "_compose_production_render", fake_compose)
    monkeypatch.setattr("app.pipeline._render_content_fingerprint", fingerprint)

    rendered = pipeline._run_production_render(
        tracker, production,
        {"enabled": True, "status": "completed", "items": audio_items},
        source, {}, tmp_path / "work", output_directory,
        render_profile="creative_preview",
    )

    assert [item["candidate_id"] for item in rendered["items"]] == ["clip-one", "clip-two"]
    assert rendered["items"][0]["status"] == "failed"
    assert "disappeared during fingerprint" in rendered["items"][0]["error"]
    assert rendered["items"][1]["status"] == "completed"
    assert rendered["items"][1]["content_fingerprint"] == "second-candidate-fingerprint"


def test_unexpected_composition_exception_finishes_candidate_stage_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.production_render.enabled = True
    pipeline = Pipeline(tmp_path, config, run_id="stage-failure-run")
    tracker = StageTracker(tmp_path / "state.json")
    plan = _plan()
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source("source", source_path, "source", "local")
    audio = SimpleNamespace(
        project_id="audio-project",
        mix=SimpleNamespace(mixed_audio_path=str(tmp_path / "mix.wav")),
    )

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unexpected renderer adapter failure")

    monkeypatch.setattr("app.pipeline.VideoCompositionService.compose", explode)

    report = pipeline._compose_production_render(
        tracker, plan, audio, source, {}, tmp_path / "work", tmp_path / "output",
        raise_on_error=False, render_profile="creative_preview",
    )

    stage = tracker.data["stages"][f"creative_preview:{plan.plan_id}"]
    assert report["status"] == "failed"
    assert "unexpected renderer adapter failure" in report["errors"][0]
    assert stage["status"] == "failed"
    assert "unexpected renderer adapter failure" in stage["error"]


def _assert_no_running_stages(tracker: StageTracker) -> None:
    assert not [
        name for name, stage in tracker.data["stages"].items()
        if stage.get("status") == "running"
    ]


def test_unexpected_transformation_failure_isolated_from_later_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.transformation.enabled = True
    pipeline = Pipeline(tmp_path, config, no_ai_transformation=True)
    tracker = StageTracker(tmp_path / "state.json")
    work = tmp_path / "work"
    output = tmp_path / "output"
    work.mkdir()
    output.mkdir()

    class FakeCandidate:
        def __init__(self, candidate_id: str) -> None:
            self.id = candidate_id

        def to_dict(self) -> dict:
            return {"id": self.id}

    selected = [
        SimpleNamespace(candidate=FakeCandidate(candidate_id))
        for candidate_id in ("clip-one", "clip-two")
    ]
    calls: list[str] = []

    def context(*args: object, **_kwargs: object) -> SimpleNamespace:
        candidate = args[2]
        return SimpleNamespace(
            candidate_id=candidate.id,
            supporting_context=[],
        )

    def transform(source_context: SimpleNamespace, *_args: object, **_kwargs: object) -> dict:
        candidate_id = source_context.candidate_id
        calls.append(candidate_id)
        if candidate_id == "clip-one":
            raise RuntimeError("unexpected first transformation failure")
        return {
            "status": "completed",
            "candidate_id": candidate_id,
            "source_context": {"transcript_text": "distinct second candidate source"},
            "final_script": {
                "candidate_id": candidate_id,
                "full_text": "A complete second candidate script.",
            },
            "validation": {"final_script": {"passed": True}},
            "fallback": {"used": False},
        }

    monkeypatch.setattr("app.pipeline.build_source_context", context)
    monkeypatch.setattr("app.pipeline.run_content_transformation", transform)

    result = pipeline._transform_selected(
        tracker, {"id": "source"}, {}, selected, {}, {}, {}, {}, work, output,
    )

    assert calls == ["clip-one", "clip-two"]
    assert [item["status"] for item in result["items"]] == ["failed", "completed"]
    assert "unexpected first transformation failure" in result["items"][0]["error"]
    assert tracker.data["stages"]["transformation_result:clip-one"]["status"] == "failed"
    assert tracker.data["stages"]["transformation_result:clip-two"]["status"] == "completed"
    _assert_no_running_stages(tracker)


def test_unexpected_production_plan_failure_isolated_from_later_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.production.enabled = True
    pipeline = Pipeline(tmp_path, config)
    tracker = StageTracker(tmp_path / "state.json")
    work = tmp_path / "work"
    output = tmp_path / "output"
    work.mkdir()
    output.mkdir()
    transformations = {
        "items": [
            {
                "candidate_id": candidate_id,
                "status": "completed",
                "final_script": {"candidate_id": candidate_id},
            }
            for candidate_id in ("clip-one", "clip-two")
        ],
    }
    calls: list[str] = []

    def build(item: dict, *_args: object, **_kwargs: object) -> ProductionPlan:
        candidate_id = item["candidate_id"]
        calls.append(candidate_id)
        if candidate_id == "clip-one":
            raise RuntimeError("unexpected first production-plan failure")
        raw = _plan().model_dump(mode="json")
        raw["plan_id"] = "isolated-plan-two"
        raw["metadata"]["candidate_id"] = candidate_id
        return ProductionPlan.model_validate(raw)

    monkeypatch.setattr(
        "app.pipeline.validate_final_script",
        lambda *_args, **_kwargs: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr("app.pipeline.build_production_plan", build)

    result = pipeline._build_production_plans(
        tracker, transformations, work, output,
    )

    assert calls == ["clip-one", "clip-two"]
    assert [item["status"] for item in result["items"]] == ["failed", "completed"]
    assert "unexpected first production-plan failure" in result["items"][0]["error"]
    assert tracker.data["stages"]["production_plan:clip-one"]["status"] == "failed"
    assert tracker.data["stages"]["production_plan:clip-two"]["status"] == "completed"
    _assert_no_running_stages(tracker)


def test_unexpected_tts_failure_isolated_from_later_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.tts.enabled = True
    pipeline = Pipeline(tmp_path, config)
    tracker = StageTracker(tmp_path / "state.json")
    production = _production_with_two_plans()
    calls: list[str] = []

    class FakeTTSService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def generate(
            self, plan: ProductionPlan, *_args: object, **_kwargs: object,
        ) -> SimpleNamespace:
            candidate_id = plan.metadata.candidate_id
            calls.append(candidate_id)
            if candidate_id == "clip-one":
                raise RuntimeError("unexpected first tts failure")
            return SimpleNamespace(status="completed", warnings=[], api_errors=[])

    monkeypatch.setattr("app.pipeline.TTSService", FakeTTSService)
    monkeypatch.setattr(
        "app.pipeline.tts_report_section",
        lambda _result: {"enabled": True, "status": "completed"},
    )

    result = pipeline._run_tts(
        tracker, production, tmp_path / "work", tmp_path / "output",
    )

    assert calls == ["clip-one", "clip-two"]
    assert [item["status"] for item in result["items"]] == ["failed", "completed"]
    assert "unexpected first tts failure" in result["items"][0]["error"]
    assert tracker.data["stages"]["tts_generation:fanout-plan-1"]["status"] == "failed"
    assert tracker.data["stages"]["tts_generation:fanout-plan-2"]["status"] == "completed"
    _assert_no_running_stages(tracker)


def test_unexpected_audio_failure_isolated_from_later_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.audio_composition.enabled = True
    pipeline = Pipeline(tmp_path, config)
    tracker = StageTracker(tmp_path / "state.json")
    production = _production_with_two_plans()
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = Source("source", source_path, "source", "local")
    tts_items: list[dict] = []
    for index, candidate_id in enumerate(("clip-one", "clip-two"), start=1):
        destination = (
            tmp_path / "output"
            if index == 1 else tmp_path / "output" / "candidates" / candidate_id
        )
        (destination / "tts").mkdir(parents=True, exist_ok=True)
        write_json(destination / "tts" / "tts-result.json", {"status": "completed"})
        tts_items.append({
            "candidate_id": candidate_id,
            "status": "completed",
            "output_directory": str(destination),
        })
    calls: list[str] = []

    class FakeAudioService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def compose(
            self, plan: ProductionPlan, *_args: object, **_kwargs: object,
        ) -> SimpleNamespace:
            candidate_id = plan.metadata.candidate_id
            calls.append(candidate_id)
            if candidate_id == "clip-one":
                raise RuntimeError("unexpected first audio failure")
            return SimpleNamespace(status="completed", warnings=[], errors=[])

    monkeypatch.setattr("app.pipeline.AudioCompositionService", FakeAudioService)
    monkeypatch.setattr(
        "app.pipeline.audio_report_section",
        lambda _project: {"enabled": True, "status": "completed"},
    )

    result = pipeline._run_audio(
        tracker, production,
        {"enabled": True, "status": "completed", "items": tts_items},
        source, {}, tmp_path / "work", tmp_path / "output",
    )

    assert calls == ["clip-one", "clip-two"]
    assert [item["status"] for item in result["items"]] == ["failed", "completed"]
    assert "unexpected first audio failure" in result["items"][0]["error"]
    assert tracker.data["stages"]["audio_composition:fanout-plan-1"]["status"] == "failed"
    assert tracker.data["stages"]["audio_composition:fanout-plan-2"]["status"] == "completed"
    _assert_no_running_stages(tracker)
