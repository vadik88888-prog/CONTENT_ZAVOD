from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app import cli, frozen_entrypoint
from app.gui.main_window import MainWindow
from app.gui.models import DesktopSettings
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.licensing import ActivationService, create_signed_license, generate_signing_seed, public_key_from_seed
from app.runtime import INTERNAL_CLI_SWITCH, RuntimeLayout


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires QApplication")
    return QApplication.instance() or QApplication([])


def _missing_cli_arguments(command: str) -> list[str]:
    source = "source.mp4"
    if command == "process":
        return [command, "--input", source]
    if command == "analyze":
        return [command, "--input", source]
    if command == "draft":
        return [command, "--input", source, "--analysis", "analysis.json", "--candidate-id", "candidate-1", "--project-id", "project-1"]
    return [command, "--input", source, "--draft", "draft.json", "--candidate-id", "candidate-1", "--project-id", "project-1", "--confirm-production"]


@pytest.mark.parametrize("command", ["process", "analyze", "draft", "render"])
def test_cli_processing_commands_stop_at_the_canonical_license_gate(tmp_path: Path, monkeypatch, command: str) -> None:
    monkeypatch.setattr(cli, "Pipeline", lambda *_args, **_kwargs: pytest.fail("Pipeline must not start without a licence"))

    assert cli.main(_missing_cli_arguments(command), runtime_root=tmp_path) == 3


def test_frozen_internal_cli_is_gated_before_pipeline_creation(tmp_path: Path, monkeypatch) -> None:
    layout = RuntimeLayout.for_frozen(
        program=tmp_path / "ContentFactory.exe", resources=tmp_path / "_internal", data=tmp_path / "data",
    )
    monkeypatch.setattr(cli, "Pipeline", lambda *_args, **_kwargs: pytest.fail("Pipeline must not start without a licence"))

    assert frozen_entrypoint.main(
        [INTERNAL_CLI_SWITCH, *_missing_cli_arguments("analyze")], layout=layout,
    ) == 3


def _services(tmp_path: Path, activation: ActivationService) -> tuple[DesktopServices, object]:
    data = tmp_path / "data"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    settings = DesktopSettings.defaults(data)
    settings.onboarding_completed = True
    return DesktopServices(
        engine_root=tmp_path,
        settings_store=SettingsStore(data),
        settings=settings,
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(tmp_path),
        system=SystemService(tmp_path),
        activation=activation,
    ), project


def test_expired_license_keeps_desktop_projects_readable_and_reactivation_restores_controls(tmp_path: Path) -> None:
    application = _application()
    seed = generate_signing_seed()
    activation = ActivationService(
        tmp_path / "data", public_key_b64=base64.b64encode(public_key_from_seed(seed)).decode("ascii"),
    )
    expired = create_signed_license(seed, activation.device_code, datetime.now(timezone.utc) - timedelta(seconds=1))
    activation.license_path.parent.mkdir(parents=True, exist_ok=True)
    activation.license_path.write_text(json.dumps(expired), encoding="utf-8")
    services, project = _services(tmp_path, activation)

    window = MainWindow(services)
    try:
        window.show()
        application.processEvents()
        window.show_projects()
        assert services.projects.load(project.project_id).project_id == project.project_id
        assert "Срок доступа закончился" in window.activation_title.text()
        assert window.activation_button.isVisible()
        assert not window.new_button.isEnabled()

        renewed = create_signed_license(seed, activation.device_code, datetime.now(timezone.utc) + timedelta(days=7))
        assert activation.install_license(json.dumps(renewed).encode("utf-8")).active
        window._refresh_activation_state()
        assert services.processing_available
        assert "Лицензия активна" in window.activation_title.text()
        assert not window.activation_button.isVisible()
        assert window.new_button.isEnabled()
    finally:
        window.close()
        application.processEvents()
        window.deleteLater()
