from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from app.gui.components import VideoDropZone
from app.gui.models import DesktopProject
from app.gui.services.error_mapping import map_error
from app.gui.viewmodels import ProjectsViewModel


_STATUS = {
    "draft": "Черновик", "ready": "Готов", "processing": "Создаём ролик",
    "completed": "Готово", "completed_with_warnings": "Готово с предупреждениями",
    "failed": "Ошибка", "cancelled": "Отменено", "interrupted": "Прервано", "queued": "Ожидает",
}


class ProjectsScreen(QWidget):
    project_opened = Signal(object)

    def __init__(self, viewmodel: ProjectsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 30, 34, 30)
        header = QHBoxLayout()
        title = QLabel("Проекты")
        title.setObjectName("title")
        self.new_button = QPushButton("Новый проект")
        self.new_button.setObjectName("primary")
        self.new_button.clicked.connect(self.choose_file)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.new_button)
        root.addLayout(header)
        hint = QLabel("Выберите локальное видео — приложение создаст проект, но не начнёт обработку без вашего подтверждения.")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.drop_zone = VideoDropZone()
        self.drop_zone.file_dropped.connect(self.viewmodel.create)
        root.addWidget(self.drop_zone)
        self.empty = QLabel("Сохранённых проектов пока нет.")
        self.empty.setObjectName("muted")
        root.addWidget(self.empty)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.list_host = QWidget()
        self.list_layout = QVBoxLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(10)
        self.list_layout.addStretch()
        self.scroll.setWidget(self.list_host)
        root.addWidget(self.scroll, 1)
        self.viewmodel.projects_changed.connect(self._render)
        self.viewmodel.project_created.connect(self.project_opened)
        self.viewmodel.error_occurred.connect(self._show_error)

    def refresh(self) -> None:
        self.viewmodel.refresh()

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Видео (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)"
        )
        if path:
            self.viewmodel.create(path)

    def _render(self, projects: list[DesktopProject]) -> None:
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty.setVisible(not projects)
        for project in projects:
            self.list_layout.insertWidget(self.list_layout.count() - 1, self._card(project))

    def _card(self, project: DesktopProject) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        text = QVBoxLayout()
        name = QLabel(project.name)
        name.setStyleSheet("font-size: 16px; font-weight: 600;")
        detail = QLabel(f"{Path(project.source_path).name} · изменён {project.updated_at[:16].replace('T', ' ')}")
        detail.setObjectName("muted")
        text.addWidget(name)
        text.addWidget(detail)
        status = QLabel(_STATUS.get(project.status, project.status))
        status.setObjectName("status")
        open_button = QPushButton("Открыть")
        open_button.clicked.connect(lambda: self.project_opened.emit(project))
        folder_button = QPushButton("Папка")
        folder_button.setToolTip("Открыть папку проекта")
        folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(project.project_directory)))
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("danger")
        delete_button.clicked.connect(lambda: self._delete(project))
        layout.addLayout(text, 1)
        layout.addWidget(status)
        layout.addWidget(open_button)
        layout.addWidget(folder_button)
        layout.addWidget(delete_button)
        return card

    def _delete(self, project: DesktopProject) -> None:
        answer = QMessageBox.question(
            self, "Удалить проект?", "Будут удалены только данные проекта и история запусков. Исходное видео останется на диске.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.viewmodel.services.delete_project(project.project_id)
            except Exception as error:
                self._show_error(map_error(error))
                return
            self.refresh()

    def _show_error(self, error) -> None:
        QMessageBox.warning(self, error.title, error.user_message)
