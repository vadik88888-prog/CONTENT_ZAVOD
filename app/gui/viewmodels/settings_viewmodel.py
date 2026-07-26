from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from app.gui.models import DesktopSettings
from app.gui.services.desktop_services import DesktopServices


class SettingsViewModel(QObject):
    settings_changed = Signal(object)
    diagnostics_ready = Signal(list)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services

    @property
    def settings(self) -> DesktopSettings:
        return self.services.settings

    def save(self) -> None:
        self.services.save_settings()
        self.settings_changed.emit(self.services.settings)

    def set_data_directory(self, directory: str) -> None:
        self.services.reconfigure_data_directory(Path(directory))
        self.settings_changed.emit(self.services.settings)

    def diagnostics(self) -> None:
        self.diagnostics_ready.emit(self.services.system.checks(self.services.settings))
