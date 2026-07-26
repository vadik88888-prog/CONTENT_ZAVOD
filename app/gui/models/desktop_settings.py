from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DesktopSettings:
    schema_version: int = 1
    data_directory: str = ""
    config_path: str | None = None
    local_test_mode: bool = False
    device_preference: str = "auto"
    theme: str = "dark"
    onboarding_completed: bool = False
    window_geometry: str | None = None

    @classmethod
    def defaults(cls, data_directory: Path) -> "DesktopSettings":
        return cls(data_directory=str(data_directory))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported settings schema version.")
        if not self.data_directory.strip():
            raise ValueError("Data directory is required.")
        if self.theme != "dark":
            raise ValueError("Only the dark theme is currently supported.")
        if self.device_preference not in {"auto", "cuda", "cpu"}:
            raise ValueError("Unsupported device preference.")
        if not isinstance(self.local_test_mode, bool) or not isinstance(self.onboarding_completed, bool):
            raise ValueError("Settings booleans are invalid.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DesktopSettings":
        settings = cls(
            schema_version=int(value.get("schema_version", 1)),
            data_directory=str(value["data_directory"]),
            config_path=str(value["config_path"]) if value.get("config_path") else None,
            local_test_mode=bool(value.get("local_test_mode", False)),
            device_preference=str(value.get("device_preference", "auto")),
            theme=str(value.get("theme", "dark")),
            onboarding_completed=bool(value.get("onboarding_completed", False)),
            window_geometry=str(value["window_geometry"]) if value.get("window_geometry") else None,
        )
        settings.validate()
        return settings
