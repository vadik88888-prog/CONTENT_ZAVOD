from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.gui.screens.project_screen import ProjectScreen
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.pipeline_facade import PipelineFacade, PreparedPipelineRun
from app.gui.services.run_history_store import RunHistoryStore


def _prepared(*, render_only: bool = False) -> PreparedPipelineRun:
    root = Path.cwd()
    return PreparedPipelineRun(
        program="python",
        arguments=[],
        working_directory=root,
        state_path=root / "state.json",
        report_path=root / "report.json",
        output_directory=root,
        runtime_config_path=root / "runtime-config.yaml",
        runtime_flags={"render_only": "true"} if render_only else {},
    )


def _provider_telemetry(*, semantic_requests: int = 1, vision_requests: int = 0) -> dict:
    return {
        "ai_cost": {
            "total_cost_usd": 0.1275,
            "provenance": {"cost_basis": "provider_reported_token_usage"},
            "semantic": {"provider": "openai", "request_count": semantic_requests},
            "vision": {"provider": "openai", "request_count": vision_requests},
        },
    }


def test_actual_cost_accepts_only_current_provider_usage_telemetry() -> None:
    report = _provider_telemetry(semantic_requests=0, vision_requests=2)

    assert PipelineFacade._actual_ai_cost_from_report(report, _prepared()) == 0.1275
    assert PipelineFacade._actual_ai_cost_from_report(report, _prepared(render_only=True)) is None
    assert PipelineFacade._actual_ai_cost_from_report(
        _provider_telemetry(semantic_requests=0, vision_requests=0), _prepared(),
    ) is None


def test_ai_cost_text_is_honest_for_expected_actual_and_local_runs() -> None:
    paid = SimpleNamespace(
        estimated_ai_cost_min=0.012,
        estimated_ai_cost_max=0.037,
        cost_note="Диапазон рассчитан по тарифам.",
    )
    local = SimpleNamespace(
        estimated_ai_cost_min=None,
        estimated_ai_cost_max=None,
        cost_note="Платные обращения в этом запуске не используются.",
    )

    assert ProjectScreen._estimate_ai_cost_text(paid) == "Ожидаемая стоимость AI ≈ $0.01–$0.04"
    assert ProjectScreen._estimate_ai_cost_text(local) == "AI не используется"
    assert ProjectScreen._actual_ai_cost_text(0.1275) == "Фактическая стоимость AI: $0.13"
    assert ProjectScreen._actual_ai_cost_text(None) == "AI не используется"


def test_actual_cost_survives_run_history_restart(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    projects = DesktopProjectStore(tmp_path / "desktop-data")
    project = projects.create(source)
    runs = RunHistoryStore(projects)
    run = runs.create(project, {}, {}, "0.1.0")
    run.actual_cost = 0.1275
    runs.save(run)

    restored = RunHistoryStore(projects).load(project.project_id, run.run_id)

    assert restored.actual_cost == 0.1275
