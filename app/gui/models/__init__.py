from app.gui.models.desktop_project import DesktopProject, ProjectOptions, ProjectStatus, SetupState
from app.gui.models.desktop_settings import DesktopSettings
from app.gui.models.processing_state import ProcessingPhase, ProcessingSnapshot
from app.gui.models.project_presentation import ProjectPresentation, derive_project_presentation
from app.gui.models.project_run import ProjectRun, RunKind, RunStatus

__all__ = [
    "DesktopProject",
    "DesktopSettings",
    "ProcessingPhase",
    "ProcessingSnapshot",
    "ProjectOptions",
    "ProjectPresentation",
    "ProjectRun",
    "RunKind",
    "SetupState",
    "ProjectStatus",
    "RunStatus",
    "derive_project_presentation",
]
