from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from app.doctor import Check


class DiagnosticTaskSignals(QObject):
    completed = Signal(object)


class DiagnosticTask(QRunnable):
    """Run the synchronous system probe away from the Qt event loop."""

    def __init__(self, callback) -> None:
        super().__init__()
        self.callback = callback
        self.signals = DiagnosticTaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            checks = self.callback()
        except Exception:
            # Diagnostics are best-effort, but an unknown result cannot let
            # first-run claim readiness.  Never surface exception text here:
            # provider failures can contain paths or credential-shaped data.
            checks = [Check(
                "Диагностика",
                "error",
                "Проверка системы завершилась непредвиденной ошибкой.",
                "Повторите проверку. Если ошибка сохраняется, переустановите portable-сборку.",
            )]
        try:
            self.signals.completed.emit(checks)
        except RuntimeError:
            # The owning window may have closed while the bounded probe was
            # finishing.  Diagnostics must never keep shutdown alive.
            pass


__all__ = ["DiagnosticTask"]
