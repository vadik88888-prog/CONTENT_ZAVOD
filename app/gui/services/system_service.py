from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig, load_config
from app.doctor import Check, collect_checks
from app.errors import ClipEngineError
from app.gui.models import DesktopSettings
from app.runtime import RuntimeLayout
from app.secure_secrets import load_runtime_secrets


@dataclass(slots=True)
class SystemService:
    runtime: RuntimeLayout | Path

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeLayout):
            root = Path(self.runtime)
            self.runtime = RuntimeLayout.for_source(root, data=root)

    @property
    def resources_root(self) -> Path:
        assert isinstance(self.runtime, RuntimeLayout)
        return self.runtime.resources

    @property
    def data_root(self) -> Path:
        assert isinstance(self.runtime, RuntimeLayout)
        return self.runtime.data

    def config(self, settings: DesktopSettings) -> AppConfig:
        config_path = (
            Path(settings.config_path)
            if settings.config_path
            else self.resources_root / "config.example.yaml"
        )
        if settings.config_path and not config_path.is_file():
            raise ClipEngineError("Configured engine file is unavailable.")
        config = load_config(config_path if config_path.is_file() else None)
        if settings.local_test_mode:
            config.ai.provider = "mock"
        return config

    def ai_provider(self, settings: DesktopSettings) -> str | None:
        try:
            return self.config(settings).ai.provider
        except (ClipEngineError, OSError, UnicodeError):
            return None

    def checks(self, settings: DesktopSettings) -> list[Check]:
        load_runtime_secrets(self.data_root)
        try:
            config = self.config(settings)
        except (ClipEngineError, OSError, UnicodeError):
            return [Check(
                "Конфигурация",
                "error",
                "Файл конфигурации не удалось проверить.",
                "Выберите корректный config.yaml или верните стандартную конфигурацию.",
            )]
        return collect_checks(self.runtime, config)
