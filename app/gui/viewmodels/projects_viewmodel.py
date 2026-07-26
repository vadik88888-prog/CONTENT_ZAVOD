from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from app.gui.models import DesktopProject
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.error_mapping import map_error
from app.gui.services.url_source_service import URLSourceService


class ProjectsViewModel(QObject):
    projects_changed = Signal(list)
    project_created = Signal(object)
    error_occurred = Signal(object)
    url_busy_changed = Signal(bool)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self.url_source = URLSourceService(self)
        self.url_source.metadata_ready.connect(self._url_metadata_ready)
        self.url_source.failed.connect(lambda message: self.error_occurred.emit(map_error(message)))
        self.url_source.busy_changed.connect(self.url_busy_changed)

    def refresh(self) -> None:
        self.projects_changed.emit(self.services.list_projects())

    def create(self, path: str) -> None:
        try:
            project = self.services.create_project(path)
        except Exception as error:
            self.error_occurred.emit(map_error(error))
            return
        self.project_created.emit(project)
        self.refresh()

    def create_from_url(self, url: str) -> None:
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
