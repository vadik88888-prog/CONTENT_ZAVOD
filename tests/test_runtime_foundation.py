from __future__ import annotations

import os
from pathlib import Path
import sys

from app import frozen_entrypoint
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.runtime import DATA_DIRECTORY_ENV, INTERNAL_CLI_SWITCH, RuntimeLayout, default_data_directory


def test_source_layout_does_not_depend_on_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    layout = RuntimeLayout.detect(data=tmp_path / "data")

    assert layout.frozen is False
    assert layout.program == Path(sys.executable).resolve()
    assert layout.resources == Path(__file__).resolve().parents[1]
    assert layout.data == (tmp_path / "data").resolve()
    assert layout.tools == layout.resources / "tools"
    command = layout.internal_cli_command(["doctor"])
    assert command.arguments == ("-u", "-m", "app", "doctor")
    assert command.working_directory == layout.resources


def test_frozen_layout_uses_bundle_resources_and_writable_data(
    tmp_path: Path, monkeypatch,
) -> None:
    program = tmp_path / "portable" / "ContentFactory.exe"
    resources = program.parent / "_internal"
    data = tmp_path / "profile" / "ContentFactoryData"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(resources), raising=False)
    monkeypatch.setattr(sys, "executable", str(program))

    layout = RuntimeLayout.detect(data=data)

    assert layout == RuntimeLayout.for_frozen(
        program=program, resources=resources, data=data,
    )
    assert layout.resources != layout.program.parent
    command = layout.internal_cli_command(["process", "--run-id", "run-1"])
    assert command.program == program.resolve()
    assert command.arguments == (
        INTERNAL_CLI_SWITCH, "process", "--run-id", "run-1",
    )
    assert command.working_directory == data.resolve()


def test_runtime_tools_are_prepended_without_losing_inherited_path(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    tools = resources / "tools"
    tools.mkdir(parents=True)
    layout = RuntimeLayout.for_frozen(
        program=tmp_path / "ContentFactory.exe",
        resources=resources,
        data=tmp_path / "data",
    )

    environment = layout.process_environment({"PATH": os.pathsep.join(["A", "B"]), "TOKEN": "kept"})

    assert environment["PATH"].split(os.pathsep) == [str(tools.resolve()), "A", "B"]
    assert environment["TOKEN"] == "kept"
    assert environment[DATA_DIRECTORY_ENV] == str(layout.data)


def test_default_data_directory_prefers_windows_local_app_data(tmp_path: Path) -> None:
    assert default_data_directory({"LOCALAPPDATA": str(tmp_path), "APPDATA": "ignored"}) == (
        tmp_path / "ContentFactoryData"
    )


def test_default_data_directory_honors_desktop_worker_override(tmp_path: Path) -> None:
    data = tmp_path / "canonical-data"

    assert default_data_directory({DATA_DIRECTORY_ENV: str(data), "LOCALAPPDATA": "ignored"}) == data


def test_desktop_services_keep_resources_separate_from_writable_data(tmp_path: Path) -> None:
    resources = Path(__file__).resolve().parents[1]
    layout = RuntimeLayout.for_source(resources, data=tmp_path / "data")

    services = DesktopServices.create(layout)

    assert services.engine_root == layout.data
    assert services.resources_root == layout.resources
    assert services.settings_store.bootstrap_directory == layout.data
    assert services.pipeline.engine_root == layout.data
    assert services.pipeline.resources_root == layout.resources


def test_frozen_dispatch_routes_private_worker_without_importing_desktop(
    tmp_path: Path, monkeypatch,
) -> None:
    layout = RuntimeLayout.for_frozen(
        program=tmp_path / "ContentFactory.exe",
        resources=tmp_path / "_internal",
        data=tmp_path / "data",
    )
    calls: list[tuple[str, list[str], RuntimeLayout]] = []
    monkeypatch.setattr(
        frozen_entrypoint,
        "_run_internal_cli",
        lambda args, active: calls.append(("cli", args, active)) or 23,
    )
    monkeypatch.setattr(
        frozen_entrypoint,
        "_run_desktop",
        lambda args, active: calls.append(("desktop", args, active)) or 29,
    )

    assert frozen_entrypoint.main(
        [INTERNAL_CLI_SWITCH, "analyze", "--run-id", "run-2"], layout=layout,
    ) == 23
    assert calls == [("cli", ["analyze", "--run-id", "run-2"], layout)]


def test_frozen_dispatch_starts_desktop_by_default(tmp_path: Path, monkeypatch) -> None:
    layout = RuntimeLayout.for_frozen(
        program=tmp_path / "ContentFactory.exe",
        resources=tmp_path / "_internal",
        data=tmp_path / "data",
    )
    calls: list[tuple[list[str], RuntimeLayout]] = []
    monkeypatch.setattr(
        frozen_entrypoint,
        "_run_desktop",
        lambda args, active: calls.append((args, active)) or 31,
    )

    assert frozen_entrypoint.main([], layout=layout) == 31
    assert calls == [([], layout)]


def test_pipeline_facade_uses_frozen_internal_dispatch(tmp_path: Path) -> None:
    layout = RuntimeLayout.for_frozen(
        program=tmp_path / "ContentFactory.exe",
        resources=tmp_path / "_internal",
        data=tmp_path / "data",
    )

    prepared = PipelineFacade(layout)._pending_prepared(
        ["process", "--run-id", "run-3"],
        tmp_path / "source.mp4",
        tmp_path / "runtime-config.yaml",
        "run-3",
        "project-1",
        {},
    )

    assert prepared.program == str(layout.program)
    assert prepared.arguments == [
        INTERNAL_CLI_SWITCH, "process", "--run-id", "run-3",
    ]
    assert prepared.working_directory == layout.data
    assert prepared.artifact_metadata_path is not None
    assert prepared.artifact_metadata_path.is_relative_to(layout.data)


def test_internal_cli_receives_runtime_data_root(tmp_path: Path, monkeypatch) -> None:
    from app import cli

    layout = RuntimeLayout.for_frozen(
        program=tmp_path / "ContentFactory.exe",
        resources=tmp_path / "_internal",
        data=tmp_path / "data",
    )
    received: list[tuple[list[str], Path | None]] = []
    monkeypatch.setattr(
        cli,
        "main",
        lambda args, runtime_root=None: received.append((args, runtime_root)) or 37,
    )

    assert frozen_entrypoint._run_internal_cli(["doctor"], layout) == 37
    assert received == [(["doctor"], layout.data)]


def test_cli_uses_explicit_runtime_root_instead_of_current_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    from app import cli

    data = tmp_path / "runtime-data"
    current = tmp_path / "unrelated-cwd"
    current.mkdir()
    seen: list[Path] = []
    monkeypatch.chdir(current)
    monkeypatch.setattr(cli, "_load_dotenv", lambda root: seen.append(root))
    monkeypatch.setattr(cli, "load_config", lambda _path: object())
    monkeypatch.setattr(cli, "collect_checks", lambda root, _config: seen.append(root) or [])
    monkeypatch.setattr(cli, "format_report", lambda _checks: "ok")

    assert cli.main(["doctor"], runtime_root=data) == 0
    assert seen == [data.resolve(), data.resolve()]


def test_windows_spec_preserves_worker_stdio_and_early_freeze_support() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = (root / "packaging" / "windows" / "ContentFactory.spec").read_text(encoding="utf-8")
    entrypoint = (root / "packaging" / "windows" / "desktop_entrypoint.py").read_text(encoding="utf-8")

    assert "console=True" in spec
    assert 'hide_console="hide-early"' in spec
    assert entrypoint.index("freeze_support()") < entrypoint.index("from app.frozen_entrypoint import main")
