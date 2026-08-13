from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from app.gui.models import DesktopProject
from app.gui.services.background_task import BackgroundTask
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.error_mapping import map_error
from app.gui.services.url_source_service import URLSourceService


class ProjectsViewModel(QObject):
    projects_changed = Signal(list)
    project_created = Signal(object)
    error_occurred = Signal(object)
    url_busy_changed = Signal(bool)
    source_busy_changed = Signal(bool)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self._source_task: BackgroundTask | None = None
        self._url_busy = False
        self.url_source = URLSourceService(self)
        self.url_source.metadata_ready.connect(self._url_metadata_ready)
        self.url_source.failed.connect(lambda message: self.error_occurred.emit(map_error(message)))
        self.url_source.busy_changed.connect(self._url_busy_changed)

    def refresh(self) -> None:
        self.projects_changed.emit(self.services.list_projects())

    def create(self, path: str) -> None:
        if self._source_task is not None or self._url_busy:
            return
        task = BackgroundTask(lambda: self.services.validate_source(path))
        task.result_ready.connect(self._source_validated)
        task.error_raised.connect(self._source_validation_failed)
        self._source_task = task
        self.source_busy_changed.emit(True)
        task.start()

    def _source_validated(self, source: object) -> None:
        try:
            project = self.services.create_validated_project(source)  # type: ignore[arg-type]
        except Exception as error:
            self.error_occurred.emit(map_error(error))
        else:
            self.project_created.emit(project)
            self.refresh()
        finally:
            self._finish_source_task()

    def _source_validation_failed(self, error: object) -> None:
        self.error_occurred.emit(map_error(error if isinstance(error, Exception) else str(error)))
        self._finish_source_task()

    def _finish_source_task(self) -> None:
        self._source_task = None
        self.source_busy_changed.emit(self._url_busy)

    def create_from_url(self, url: str) -> None:
        if self._source_task is not None:
            return
        if not url.strip():
            self.error_occurred.emit(map_error("Укажите ссылку на видео."))
            return
        self.url_source.inspect(url)

    def _url_metadata_ready(self, metadata: dict) -> None:
        try:
            project = self.services.create_url_project(str(metadata["url"]), metadata)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return
        self.project_created.emit(project)
        self.refresh()

    def _url_busy_changed(self, busy: bool) -> None:
        self._url_busy = busy
        self.url_busy_changed.emit(busy)
        self.source_busy_changed.emit(busy or self._source_task is not None)
