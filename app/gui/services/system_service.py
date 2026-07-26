from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import load_config
from app.cli import _load_dotenv
from app.doctor import Check, collect_checks
from app.gui.models import DesktopSettings


@dataclass(slots=True)
class SystemService:
    engine_root: Path

    def checks(self, settings: DesktopSettings) -> list[Check]:
        _load_dotenv(self.engine_root)
        config_path = Path(settings.config_path) if settings.config_path else self.engine_root / "config.example.yaml"
        config = load_config(config_path if config_path.is_file() else None)
        if settings.local_test_mode:
            config.ai.provider = "mock"
        return collect_checks(self.engine_root, config)
