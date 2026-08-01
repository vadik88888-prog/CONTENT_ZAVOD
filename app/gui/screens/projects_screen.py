from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gui.components import VideoDropZone
from app.gui.models import DesktopProject
from app.gui.services.error_mapping import dialog_message, map_error
from app.gui.viewmodels import ProjectsViewModel


_STATUS = {
    "new": "Источник ждёт загрузки",
    "source_ready": "Готов к настройке",
    "analyzing": "Ищем моменты",
    "analysis_ready": "Моменты готовы",
    "reviewing_candidates": "Выберите моменты",
    "rendering_selected": "Создаём ролики",
    "partially_rendered": "Готово частично",
    "draft": "Черновик",
    "ready": "Готов",
    "processing": "Создаём ролик",
    "completed": "Ролики готовы",
    "completed_with_warnings": "Готово с замечаниями",
    "failed": "Нужно внимание",
    "cancelled": "Остановлено",
    "interrupted": "Прервано",
    "queued": "Ожидает",
}


class ProjectsScreen(QWidget):
    """Source onboarding and recent projects without a second navigation state."""

    project_opened = Signal(object)

    def __init__(self, viewmodel: ProjectsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        self._projects: list[DesktopProject] = []
        self._rendered_columns = 0
        self._reflow_pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 22)
        root.setSpacing(0)

        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # ``scroll`` used to name the projects list. Keep the public attribute
        # while making the whole compact workspace vertically scrollable.
        self.scroll = self.content_scroll
        host = QWidget()
        content = QVBoxLayout(host)
        content.setContentsMargins(0, 0, 0, 4)
        content.setSpacing(16)

        top = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(3)
        eyebrow = QLabel("CONTENT FACTORY")
        eyebrow.setObjectName("muted")
        eyebrow.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #FF7900;")
        title = QLabel("Новый проект")
        title.setObjectName("title")
        subtitle = QLabel("Создайте короткие ролики из длинного видео — всё останется на вашем компьютере.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        titles.addWidget(eyebrow)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        top.addLayout(titles, 1)
        local_note = QLabel("●  Локальная работа")
        local_note.setObjectName("status")
        top.addWidget(local_note, 0, Qt.AlignmentFlag.AlignTop)
        content.addLayout(top)

        source_card = QFrame()
        source_card.setObjectName("card")
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(18, 17, 18, 17)
        source_layout.setSpacing(11)
        source_heading = QLabel("Добавьте исходное видео")
        source_heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        source_hint = QLabel("Перетащите один файл или выберите его на компьютере. Поддерживаются длинные видео и высокое разрешение.")
        source_hint.setObjectName("muted")
        source_hint.setWordWrap(True)
        source_layout.addWidget(source_heading)
        source_layout.addWidget(source_hint)

        self.drop_zone = VideoDropZone()
        self.drop_zone.file_dropped.connect(self.viewmodel.create)
        self.drop_zone.setMinimumHeight(126)
        source_layout.addWidget(self.drop_zone)
        self.file_button = QPushButton("Выбрать видео")
        self.file_button.setObjectName("primary")
        self.file_button.setMinimumHeight(38)
        self.file_button.clicked.connect(self.choose_file)
        # Compatibility with the original screen and any integrations that use
        # its button directly.
        self.new_button = self.file_button
        source_layout.addWidget(self.file_button, 0, Qt.AlignmentFlag.AlignHCenter)
        formats = QLabel("MP4, MOV, MKV, AVI, WebM и M4V")
        formats.setObjectName("muted")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source_layout.addWidget(formats)

        divider = QHBoxLayout()
        divider.setSpacing(9)
        left_line = QFrame()
        left_line.setFrameShape(QFrame.Shape.HLine)
        left_line.setStyleSheet("color: #272D36;")
        divider_label = QLabel("или вставьте публичную ссылку")
        divider_label.setObjectName("muted")
        right_line = QFrame()
        right_line.setFrameShape(QFrame.Shape.HLine)
        right_line.setStyleSheet("color: #272D36;")
        divider.addWidget(left_line, 1)
        divider.addWidget(divider_label)
        divider.addWidget(right_line, 1)
        source_layout.addLayout(divider)

        url_row = QHBoxLayout()
        url_row.setSpacing(8)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Вставьте ссылку на открытое видео")
        self.url_input.returnPressed.connect(self._create_url)
        self.url_button = QPushButton("Добавить видео")
        self.url_button.clicked.connect(self._create_url)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.url_button)
        source_layout.addLayout(url_row)
        public_note = QLabel("Подойдут только видео, доступные без входа, оплаты и других ограничений.")
        public_note.setObjectName("muted")
        public_note.setWordWrap(True)
        source_layout.addWidget(public_note)
        content.addWidget(source_card)

        recent_header = QHBoxLayout()
        recent_title = QLabel("Недавние проекты")
        recent_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.recent_count = QLabel()
        self.recent_count.setObjectName("muted")
        recent_header.addWidget(recent_title)
        recent_header.addWidget(self.recent_count)
        recent_header.addStretch()
        content.addLayout(recent_header)

        self.empty = QLabel("Сохранённых проектов пока нет. Начните с видео выше.")
        self.empty.setObjectName("muted")
        self.empty.setWordWrap(True)
        content.addWidget(self.empty)
        self.list_host = QWidget()
        self.list_layout = QGridLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setHorizontalSpacing(12)
        self.list_layout.setVerticalSpacing(12)
        content.addWidget(self.list_host)
        content.addStretch()

        self.content_scroll.setWidget(host)
        root.addWidget(self.content_scroll, 1)

        self.viewmodel.projects_changed.connect(self._render)
        self.viewmodel.project_created.connect(self.project_opened)
        self.viewmodel.error_occurred.connect(self._show_error)
        self.viewmodel.url_busy_changed.connect(self._url_busy_changed)

    def refresh(self) -> None:
        self.viewmodel.refresh()

    def focus_source(self) -> None:
        """Present the source choice after global “New project” navigation."""

        self.content_scroll.verticalScrollBar().setValue(0)
        self.file_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Видео (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)",
        )
        if path:
            self.viewmodel.create(path)

    def _create_url(self) -> None:
        self.viewmodel.create_from_url(self.url_input.text())

    def _url_busy_changed(self, busy: bool) -> None:
        self.url_input.setDisabled(busy)
        self.url_button.setDisabled(busy)
        self.url_button.setText("Проверяем ссылку…" if busy else "Добавить видео")

    def _render(self, projects: list[DesktopProject]) -> None:
        self._projects = list(projects)
        self.recent_count.setText(f"{len(projects)}" if projects else "")
        self.empty.setVisible(not projects)
        self._render_cards()

    def _render_cards(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        columns = self._recent_columns()
        self._rendered_columns = columns
        for index, project in enumerate(self._projects):
            row, column = divmod(index, columns)
            self.list_layout.addWidget(self._card(project), row, column)
        if self._projects:
            self.list_layout.setRowStretch((len(self._projects) - 1) // columns + 1, 1)
        for column in range(columns):
            self.list_layout.setColumnStretch(column, 1)

    def _recent_columns(self) -> int:
        # Cards collapse before they become cramped on a 1280 px laptop once
        # the shell sidebar and Windows scaling have taken their share.
        available = max(0, self.width() - 52)
        if available >= 960:
            return 3
        if available >= 620:
            return 2
        return 1

    def _card(self, project: DesktopProject) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 13, 14, 13)
        layout.setSpacing(7)
        top = QHBoxLayout()
        name = QLabel(project.name)
        name.setStyleSheet("font-size: 15px; font-weight: 600;")
        name.setWordWrap(True)
        status = QLabel(_STATUS.get(project.status, "В работе"))
        status.setObjectName("status")
        status.setWordWrap(True)
        top.addWidget(name, 1)
        top.addWidget(status, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)

        source_name = (
            project.source.name if project.source_spec.is_ready
            else str(project.source_metadata.get("title") or "Видео по ссылке")
        )
        source = QLabel(source_name)
        source.setObjectName("muted")
        source.setWordWrap(True)
        updated = QLabel(f"Изменён {project.updated_at[:16].replace('T', ' ')}")
        updated.setObjectName("muted")
        layout.addWidget(source)
        layout.addWidget(updated)

        actions = QHBoxLayout()
        actions.setSpacing(7)
        open_button = QPushButton("Открыть")
        open_button.clicked.connect(lambda _checked=False, value=project: self.project_opened.emit(value))
        folder_button = QPushButton("Папка")
        folder_button.setToolTip("Открыть папку проекта")
        folder_button.clicked.connect(
            lambda _checked=False, path=project.project_directory: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        )
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("danger")
        delete_button.clicked.connect(lambda _checked=False, value=project: self._delete(value))
        actions.addWidget(open_button)
        actions.addWidget(folder_button)
        actions.addStretch()
        actions.addWidget(delete_button)
        layout.addLayout(actions)
        return card

    def _delete(self, project: DesktopProject) -> None:
        answer = QMessageBox.question(
            self,
            "Удалить проект?",
            "Будут удалены только данные проекта и история запусков. Исходное видео останется на диске.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.viewmodel.services.delete_project(project.project_id)
            except Exception as error:
                self._show_error(map_error(error))
                return
            self.refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._projects or self._recent_columns() == self._rendered_columns or self._reflow_pending:
            return
        self._reflow_pending = True
        QTimer.singleShot(0, self._finish_reflow)

    def _finish_reflow(self) -> None:
        self._reflow_pending = False
        self._render_cards()

    def _show_error(self, error) -> None:
        QMessageBox.warning(self, error.title, dialog_message(error))
