"""Show the native Windows activation screen and assert its device-bound state."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtWidgets import QApplication

from app.gui.screens.activation_screen import ActivationDialog
from app.gui.main_window import MainWindow
from app.gui.models import DesktopSettings
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService
from app.licensing import ActivationService, create_signed_license, generate_signing_seed, public_key_from_seed


def main() -> int:
    application = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="content-factory-activation-qa-") as directory:
        seed = generate_signing_seed()
        public_key = base64.b64encode(public_key_from_seed(seed)).decode("ascii")
        activation = ActivationService(Path(directory), public_key_b64=public_key)
        dialog = ActivationDialog(activation)
        dialog.show()
        application.processEvents()
        handle = dialog.windowHandle()
        assert dialog.isVisible() and handle is not None and int(handle.winId()) != 0
        assert activation.status().code == "missing"
        assert activation.device_code in dialog.device_code.text()
        data = Path(directory) / "desktop-data"
        source = Path(directory) / "source.mp4"
        source.write_bytes(b"source")
        projects = DesktopProjectStore(data)
        project = projects.create(source)
        settings = DesktopSettings.defaults(data)
        settings.onboarding_completed = True
        services = DesktopServices(
            engine_root=Path(directory), settings_store=SettingsStore(data), settings=settings,
            projects=projects, runs=RunHistoryStore(projects), pipeline=PipelineFacade(Path(directory)),
            system=SystemService(Path(directory)), activation=activation,
        )
        window = MainWindow(services)
        window.show()
        application.processEvents()
        assert window.isVisible() and window.activation_button.isVisible()
        assert not window.new_button.isEnabled()
        read_only_before_activation = not window.new_button.isEnabled()
        license_data = create_signed_license(seed, activation.device_code, datetime.now(timezone.utc) + timedelta(days=1))
        assert activation.install_license(json.dumps(license_data).encode("utf-8")).active
        window._refresh_activation_state()
        assert window.new_button.isEnabled() and window.activation_button.isHidden()
        print(json.dumps({
            "window_visible": window.isVisible(),
            "native_handle": int(handle.winId()),
            "license_status_after_activation": activation.status().code,
            "device_code_visible": activation.device_code in dialog.device_code.text(),
            "read_only_before_activation": read_only_before_activation,
            "processing_available_after_activation": services.processing_available,
        }))
        window.close()
        dialog.close()
        application.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
