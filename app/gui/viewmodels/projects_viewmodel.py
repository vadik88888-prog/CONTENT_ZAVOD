from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from app.gui.models import DesktopProject
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.error_mapping import map_error


class ProjectsViewModel(QObject):
    projects_changed = Signal(list)
    project_created = Signal(object)
    error_occurred = Signal(object)

    def __init__(self, services: DesktopServices, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.services = services

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
