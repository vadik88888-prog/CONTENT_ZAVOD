from __future__ import annotations

from pathlib import Path

import pytest

from app.clip_results import ClipResult
from app.config import AppConfig
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.pipeline import Pipeline
from app.run_manifest import write_run_manifest
from app.utils import write_json


def _result(candidate_id: str, output_file: Path, run_id: str, index: int = 1) -> ClipResult:
    return ClipResult(
        candidate_id,
        str(output_file),
        clip_result_id=f"{candidate_id}:plan-{index}",
        production_plan_id=f"plan-{index}",
        run_id=run_id,
        revision_id=f"{run_id}:render-{index:02d}",
    )


def test_same_source_creates_distinct_run_directories_and_state_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    first = Pipeline(tmp_path, AppConfig(), run_id="run-a")
    second = Pipeline(tmp_path, AppConfig(), run_id="run-b")
    _source_a, work_a, output_a = first._prepare_source(str(source), None)
    _source_b, work_b, output_b = second._prepare_source(str(source), None)

    assert work_a == work_b
    assert output_a != output_b
    assert output_a.name == "run-a" and output_b.name == "run-b"
    assert first.run_work_directory and second.run_work_directory
    assert first.run_work_directory != second.run_work_directory


def _write_scoped_report(run_directory: Path, run_id: str, result: ClipResult) -> PreparedPipelineRun:
    report = run_directory / "report.json"
    write_json(report, {
        "production_render": {"status": "completed", "output_file": result.output_file},
        "primary_results": [result.to_dict()], "warnings": [], "ai": {}, "tts": {},
    })
    write_run_manifest(
        run_directory / "manifest.json", run_id=run_id, source={"id": "source"}, started_at="2026-01-01T00:00:00+00:00",
        requested_clip_count=1, production_render={"enabled": True, "status": "completed"},
        results=[result], run_directory=run_directory, project_id="project-scoped",
    )
    return PreparedPipelineRun(
        program="python", arguments=[], working_directory=run_directory, state_path=run_directory / "state.json",
        report_path=report, output_directory=run_directory, runtime_config_path=run_directory / "runtime.yaml",
        run_id=run_id, manifest_path=run_directory / "manifest.json",
    )


def test_gui_completion_reads_only_selected_run_manifest(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "output" / "source" / "runs"
    run_a = root / "run-a"; run_b = root / "run-b"
    for directory in (run_a / "results", run_b / "results"):
        directory.mkdir(parents=True)
    file_a = run_a / "results" / "final-short-01.mp4"; file_a.write_bytes(b"a")
    file_b = run_b / "results" / "final-short-01.mp4"; file_b.write_bytes(b"b")
    prepared_a = _write_scoped_report(run_a, "run-a", _result("candidate-a", file_a, "run-a"))
    prepared_b = _write_scoped_report(run_b, "run-b", _result("candidate-b", file_b, "run-b"))
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    facade = PipelineFacade(tmp_path)
    completion = facade.completion(prepared_a)
    completion_b = facade.completion(prepared_b)

    assert completion.output_files == [file_a]
    assert file_b not in completion.output_files
    assert completion_b.output_files == [file_b]
    assert file_a not in completion_b.output_files


def test_run_manifest_persists_project_identity(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "run-a"
    (run_directory / "results").mkdir(parents=True)
    output = run_directory / "results" / "final-short-01.mp4"
    output.write_bytes(b"video")

    manifest = write_run_manifest(
        run_directory / "manifest.json", run_id="run-a", source={"id": "source"},
        started_at="2026-01-01T00:00:00+00:00", requested_clip_count=1,
        production_render={
            "enabled": True, "status": "completed", "audio_mode": "original",
            "items": [{
                "candidate_id": "candidate-a",
                "report": {
                    "subtitles_enabled": True, "subtitle_cue_count": 2,
                    "quality": {"status": "passed"},
                    "subtitle_layout": {"cues": [{"fallback_used": False}, {"fallback_used": True}]},
                },
            }],
        },
        results=[_result("candidate-a", output, "run-a")],
        run_directory=run_directory, project_id="project-scoped",
    )

    assert manifest["project_id"] == "project-scoped"
    assert manifest["audio_mode"] == "original"
    assert manifest["subtitle_status"] == {
        "enabled": True, "validation": "passed",
        "items": [{"candidate_id": "candidate-a", "enabled": True, "cue_count": 2, "validation": "passed", "fallback_cue_count": 1}],
    }


def test_manifest_path_outside_current_run_is_rejected(tmp_path: Path, monkeypatch) -> None:
    run_a = tmp_path / "runs" / "run-a"; run_b = tmp_path / "runs" / "run-b"
    (run_a / "results").mkdir(parents=True); (run_b / "results").mkdir(parents=True)
    foreign = run_b / "results" / "final-short-01.mp4"; foreign.write_bytes(b"foreign")
    report = run_a / "report.json"
    result = _result("candidate-b", foreign, "run-a")
    write_json(report, {"production_render": {"status": "completed", "output_file": str(foreign)}, "primary_results": [result.to_dict()]})
    write_json(run_a / "manifest.json", {"run_id": "run-a", "primary_results": [result.to_dict()]})
    prepared = PreparedPipelineRun(
        program="python", arguments=[], working_directory=run_a, state_path=run_a / "state.json",
        report_path=report, output_directory=run_a, runtime_config_path=run_a / "runtime.yaml",
        run_id="run-a", manifest_path=run_a / "manifest.json",
    )
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))

    completion = PipelineFacade(tmp_path).completion(prepared)

    assert completion.error_summary == "Manifest содержит путь вне текущего запуска."


def test_manifest_rejects_result_owned_by_another_run(tmp_path: Path) -> None:
    run_a = tmp_path / "runs" / "run-a"
    (run_a / "results").mkdir(parents=True)
    output = run_a / "results" / "final-short-01.mp4"
    output.write_bytes(b"video")

    with pytest.raises(ValueError, match="run_id does not match"):
        write_run_manifest(
            run_a / "manifest.json", run_id="run-a", source={"id": "source"},
            started_at="2026-01-01T00:00:00+00:00", requested_clip_count=1,
            production_render={"enabled": True, "status": "completed"},
            results=[_result("candidate-b", output, "run-b")], run_directory=run_a,
        )


def test_recovery_uses_only_its_manifest_and_never_falls_back_to_neighbour_run(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "output" / "source" / "runs"
    run_a = root / "run-a"; run_b = root / "run-b"
    for directory in (run_a / "results", run_b / "results"):
        directory.mkdir(parents=True)
    file_a = run_a / "results" / "final-short-01.mp4"; file_a.write_bytes(b"a")
    file_b = run_b / "results" / "final-short-01.mp4"; file_b.write_bytes(b"b")
    prepared_a = _write_scoped_report(run_a, "run-a", _result("candidate-a", file_a, "run-a"))
    _write_scoped_report(run_b, "run-b", _result("candidate-b", file_b, "run-b"))
    monkeypatch.setattr(PipelineFacade, "_validate_final_mp4", staticmethod(lambda _path: None))
    facade = PipelineFacade(tmp_path)

    recovered = facade.recovery_completion(prepared_a, "2026-01-01T00:00:00+00:00")

    assert recovered is not None and recovered.output_files == [file_a]
    write_json(run_a / "manifest.json", {"run_id": "run-b", "primary_results": []})
    assert facade.recovery_completion(prepared_a, "2026-01-01T00:00:00+00:00") is None


def test_three_result_manifest_uses_unique_canonical_names_inside_one_run(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "run-a"
    results_directory = run_directory / "results"
    results_directory.mkdir(parents=True)
    results = [
        _result(f"candidate-{index:03d}", results_directory / f"final-short-{index:02d}.mp4", "run-a", index)
        for index in range(1, 4)
    ]
    for result in results:
        Path(result.output_file).write_bytes(result.candidate_id.encode("utf-8"))

    manifest = write_run_manifest(
        run_directory / "manifest.json", run_id="run-a", source={"id": "source"},
        started_at="2026-01-01T00:00:00+00:00", requested_clip_count=3,
        production_render={"enabled": True, "status": "completed"}, results=results,
        run_directory=run_directory,
    )

    assert manifest["completed_clip_count"] == 3
    assert [Path(value).name for value in manifest["result_paths"]] == [
        "final-short-01.mp4", "final-short-02.mp4", "final-short-03.mp4",
    ]
    assert all(Path(value).resolve().is_relative_to(run_directory.resolve()) for value in manifest["result_paths"])
    assert len({item["clip_result_id"] for item in manifest["primary_results"]}) == 3
    assert len({item["revision_id"] for item in manifest["primary_results"]}) == 3
