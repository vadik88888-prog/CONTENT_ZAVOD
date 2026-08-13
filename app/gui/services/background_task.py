from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _WorkerSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class _CallableRunnable(QRunnable):
    def __init__(self, operation: Callable[[], Any], signals: _WorkerSignals) -> None:
        super().__init__()
        self._operation = operation
        self._signals = signals

    @Slot()
    def run(self) -> None:
        try:
            result = self._operation()
        except Exception as error:
            self._signals.failed.emit(error)
        else:
            self._signals.succeeded.emit(result)
        finally:
            self._signals.finished.emit()


# A runnable may outlive the screen that requested it. Keeping the dispatch
# object alive until its worker exits prevents Qt signal objects from being
# deleted underneath an in-flight ffprobe/hash/copy operation. Connections to
# a deleted receiver are still removed automatically by Qt.
_ACTIVE_TASKS: set["BackgroundTask"] = set()


class BackgroundTask(QObject):
    """Deliver one blocking callable back to Qt without blocking its GUI loop."""

    result_ready = Signal(object)
    error_raised = Signal(object)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self._signals = _WorkerSignals()
        self._runnable = _CallableRunnable(operation, self._signals)
        self._cancelled = False
        self._started = False
        self._signals.succeeded.connect(self._deliver_result)
        self._signals.failed.connect(self._deliver_error)
        self._signals.finished.connect(self._worker_finished)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Background task has already been started.")
        self._started = True
        _ACTIVE_TASKS.add(self)
        QThreadPool.globalInstance().start(self._runnable)

    def cancel_delivery(self) -> None:
        """Ignore a read-only result that is no longer owned by the UI flow."""

        self._cancelled = True

    @Slot(object)
    def _deliver_result(self, result: object) -> None:
        if not self._cancelled:
            self.result_ready.emit(result)

    @Slot(object)
    def _deliver_error(self, error: object) -> None:
        if not self._cancelled:
            self.error_raised.emit(error)

    @Slot()
    def _worker_finished(self) -> None:
        self.finished.emit()
        _ACTIVE_TASKS.discard(self)


def run_in_background(operation: Callable[[], Any]) -> BackgroundTask:
    task = BackgroundTask(operation)
    task.start()
    return task
