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
        artifact.write_bytes(b"mp4")
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
