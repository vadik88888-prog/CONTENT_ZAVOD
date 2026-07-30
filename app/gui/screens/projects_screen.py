from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from app.gui.components import VideoDropZone
from app.gui.models import DesktopProject
from app.gui.services.error_mapping import map_error
from app.gui.viewmodels import ProjectsViewModel


_STATUS = {
    "new": "Источник выбран", "source_ready": "Готов к настройке", "analyzing": "Ищем моменты",
    "analysis_ready": "Моменты готовы", "reviewing_candidates": "Выбор моментов",
    "rendering_selected": "Создаём готовые ролики", "partially_rendered": "Готово частично",
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
        self.new_button = QPushButton("Выбрать файл")
        self.new_button.setObjectName("primary")
        self.new_button.clicked.connect(self.choose_file)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.new_button)
        root.addLayout(header)
        hint = QLabel("Выберите видео с компьютера или вставьте открытую ссылку. Дальше вы отдельно скачаете видео и настроите обработку.")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)
        url_row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Вставить ссылку на открытое видео")
        self.url_input.returnPressed.connect(self._create_url)
        self.url_button = QPushButton("Продолжить")
        self.url_button.clicked.connect(self._create_url)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.url_button)
        root.addLayout(url_row)
        public_note = QLabel("Подойдут только видео, которые открываются без входа и оплаты.")
        public_note.setObjectName("muted")
        public_note.setWordWrap(True)
        root.addWidget(public_note)
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
        self.viewmodel.url_busy_changed.connect(self._url_busy_changed)

    def refresh(self) -> None:
        self.viewmodel.refresh()

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Видео (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)"
        )
        if path:
            self.viewmodel.create(path)

    def _create_url(self) -> None:
        self.viewmodel.create_from_url(self.url_input.text())

    def _url_busy_changed(self, busy: bool) -> None:
        self.url_input.setDisabled(busy)
        self.url_button.setDisabled(busy)
        self.url_button.setText("Проверяем ссылку…" if busy else "Продолжить")

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
        name.setWordWrap(True)
        source_name = (
            project.source.name if project.source_spec.is_ready
            else str(project.source_metadata.get("title") or "Видео по ссылке")
        )
        detail = QLabel(f"{source_name} · изменён {project.updated_at[:16].replace('T', ' ')}")
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        text.addWidget(name)
        text.addWidget(detail)
        status = QLabel(_STATUS.get(project.status, project.status))
        status.setObjectName("status")
        status.setWordWrap(True)
        open_button = QPushButton("Открыть")
        open_button.clicked.connect(lambda: self.project_opened.emit(project))
        folder_button = QPushButton("Папка")
        folder_button.setToolTip("Открыть папку проекта")
        folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(project.project_directory)))
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("danger")
        delete_button.clicked.connect(lambda: self._delete(project))
        actions = QVBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(status)
        action_row = QHBoxLayout()
        action_row.addWidget(open_button)
        action_row.addWidget(folder_button)
        action_row.addWidget(delete_button)
        actions.addLayout(action_row)
        layout.addLayout(text, 1)
        layout.addLayout(actions)
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
