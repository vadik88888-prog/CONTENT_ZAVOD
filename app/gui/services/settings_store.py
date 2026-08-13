from __future__ import annotations

from pathlib import Path

from app.gui.models import DesktopSettings
from app.runtime import default_data_directory
from app.utils import read_json, write_json


class SettingsStore:
    """Persists non-secret desktop preferences with atomic replacement."""

    def __init__(self, bootstrap_directory: Path | None = None) -> None:
        self.bootstrap_directory = (bootstrap_directory or default_data_directory()).resolve()
        self.path = self.bootstrap_directory / "settings.json"

    def load(self) -> DesktopSettings:
        if not self.path.exists():
            return DesktopSettings.defaults(self.bootstrap_directory)
        try:
            raw = read_json(self.path)
            if not isinstance(raw, dict):
                raise ValueError("Settings JSON root is not an object.")
            return DesktopSettings.from_dict(raw)
        except (OSError, ValueError, KeyError, TypeError):
            # A corrupt preference file must not prevent the desktop shell opening.
            return DesktopSettings.defaults(self.bootstrap_directory)

    def save(self, settings: DesktopSettings) -> None:
        write_json(self.path, settings.to_dict())
