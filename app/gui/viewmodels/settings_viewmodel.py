from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal, Slot

from app.gui.models import DesktopSettings
from app.feedback_export import FeedbackExportResult
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.diagnostic_task import DiagnosticTask
from app.secure_secrets import ApiKeySaveResult, save_api_key


class SettingsViewModel(QObject):
    settings_changed = Signal(object)
    diagnostics_started = Signal()
    diagnostics_ready = Signal(list)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self._diagnostic_task: DiagnosticTask | None = None
        self._diagnostics_pending = False

    @property
    def settings(self) -> DesktopSettings:
        return self.services.settings

    def save(self) -> None:
        self.services.save_settings()
        self.settings_changed.emit(self.services.settings)
        QTimer.singleShot(0, self.diagnostics)

    def set_data_directory(self, directory: str) -> None:
        self.services.reconfigure_data_directory(Path(directory))
        self.settings_changed.emit(self.services.settings)
        QTimer.singleShot(0, self.diagnostics)

    def diagnostics(self) -> None:
        if self._diagnostic_task is not None:
            self._diagnostics_pending = True
            return
        snapshot = replace(self.services.settings)
        task = DiagnosticTask(lambda: self.services.system.checks(snapshot))
        task.signals.completed.connect(self._diagnostics_finished)
        self._diagnostic_task = task
        self.diagnostics_started.emit()
        QThreadPool.globalInstance().start(task)

    def ai_provider(self) -> str | None:
        return self.services.system.ai_provider(self.services.settings)

    def save_api_key(self, value: str) -> ApiKeySaveResult:
        provider = self.ai_provider()
        if provider not in {"openai", "gemini"}:
            return ApiKeySaveResult(False, "Для текущего AI-провайдера ключ не требуется.")
        result = save_api_key(provider, value, self.services.system.data_root)
        if result.saved:
            self.settings_changed.emit(self.services.settings)
            QTimer.singleShot(0, self.diagnostics)
        return result

    def export_feedback(self) -> FeedbackExportResult:
        """Create the local, sendable Friend Beta feedback archive."""

        return self.services.export_feedback_data()

    @Slot(object)
    def _diagnostics_finished(self, checks) -> None:
        self._diagnostic_task = None
        self.diagnostics_ready.emit(list(checks))
        if self._diagnostics_pending:
            self._diagnostics_pending = False
            QTimer.singleShot(0, self.diagnostics)
