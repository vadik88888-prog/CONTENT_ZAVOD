from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from app.gui.components import ProcessingProgress, VideoPreview
from app.gui.models import DesktopProject, ProcessingSnapshot, ProjectRun
from app.gui.viewmodels import ProjectViewModel
from app.utils import format_seconds


_STATUS = {
    "draft": "Черновик", "ready": "Готов", "queued": "Ожидает", "processing": "Создаём ролик",
    "completed": "Готово", "completed_with_warnings": "Готово с предупреждениями",
    "failed": "Ошибка", "cancelled": "Отменено", "interrupted": "Прервано",
}


class ProjectScreen(QWidget):
    back_requested = Signal()

    def __init__(self, viewmodel: ProjectViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        self.project: DesktopProject | None = None
        self.runs: list[ProjectRun] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(34, 26, 34, 30)
        header = QHBoxLayout()
        back = QPushButton("← Проекты")
        back.clicked.connect(self.back_requested)
        self.title = QLabel("Проект")
        self.title.setObjectName("title")
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.folder = QPushButton("Открыть папку")
        self.folder.clicked.connect(self._open_project_folder)
        header.addWidget(back)
        header.addWidget(self.title)
        header.addStretch()
        header.addWidget(self.status)
        header.addWidget(self.folder)
        root.addLayout(header)
        self.autosave = QLabel("Изменения сохраняются автоматически")
        self.autosave.setObjectName("muted")
        root.addWidget(self.autosave)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.preview = VideoPreview()
        left.addWidget(self.preview)
        self.metadata = self._card("Сведения о видео")
        left.addWidget(self.metadata)
        self.progress = ProcessingProgress()
        self.progress.cancel_requested.connect(self.viewmodel.cancel)
        left.addWidget(self.progress)
        history_title = QLabel("История запусков")
        history_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        left.addWidget(history_title)
        self.history = QScrollArea()
        self.history.setWidgetResizable(True)
        self.history_host = QWidget()
        self.history_layout = QVBoxLayout(self.history_host)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.addStretch()
        self.history.setWidget(self.history_host)
        left.addWidget(self.history, 1)
        body.addLayout(left, 3)
        panel = QFrame()
        panel.setObjectName("card")
        panel.setMaximumWidth(300)
        settings = QVBoxLayout(panel)
        settings.setContentsMargins(18, 18, 18, 18)
        heading = QLabel("Настройки ролика")
        heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        settings.addWidget(heading)
        settings.addWidget(QLabel("Субтитры"))
        self.subtitles = QCheckBox("Показывать субтитры")
        self.subtitles.toggled.connect(lambda value: self.viewmodel.save_options(subtitles_enabled=value))
        settings.addWidget(self.subtitles)
        settings.addWidget(QLabel("Стиль субтитров"))
        self.subtitle_style = QComboBox()
        self.subtitle_style.addItems(["documentary", "clean", "minimal", "dynamic"])
        self.subtitle_style.currentTextChanged.connect(lambda value: self.viewmodel.save_options(subtitle_style=value))
        settings.addWidget(self.subtitle_style)
        settings.addWidget(QLabel("Кодирование"))
        self.encoder = QComboBox()
        self.encoder.addItems(["auto", "cpu", "nvenc"])
        self.encoder.currentTextChanged.connect(lambda value: self.viewmodel.save_options(encoder=value))
        settings.addWidget(self.encoder)
        self.cache = QCheckBox("Использовать сохранённые данные")
        self.cache.toggled.connect(lambda value: self.viewmodel.save_options(use_cache=value))
        settings.addWidget(self.cache)
        self.recompute = QCheckBox("Полностью пересчитать")
        self.recompute.setToolTip("Использует существующие безопасные флаги пересчёта pipeline.")
        self.recompute.toggled.connect(lambda value: self.viewmodel.save_options(recompute_all=value))
        settings.addWidget(self.recompute)
        settings.addStretch()
        self.run_button = QPushButton("Создать ролик")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.viewmodel.start)
        settings.addWidget(self.run_button)
        body.addWidget(panel)
        root.addLayout(body, 1)
        self.viewmodel.project_changed.connect(self._project_changed)
        self.viewmodel.runs_changed.connect(self._runs_changed)
        self.viewmodel.processing_changed.connect(self._processing_changed)
        self.viewmodel.error_occurred.connect(self._error)

    def open(self, project: DesktopProject) -> None:
        self.viewmodel.open(project)

    def _project_changed(self, project: DesktopProject) -> None:
        self.project = project
        self.title.setText(project.name)
        self.status.setText(_STATUS.get(project.status, "Неизвестно"))
        self.preview.set_file(project.source_path)
        source = project.source_metadata
        duration = format_seconds(source.get("duration")) if source else "н/д"
        resolution = f"{source.get('width', '—')} × {source.get('height', '—')}" if source else "н/д"
        fps = source.get("fps", "н/д") if source else "н/д"
        self._replace_card_text(self.metadata, [
            f"Файл: {Path(project.source_path).name}", f"Длительность: {duration}",
            f"Разрешение: {resolution}", f"FPS: {fps}",
        ])
        self.subtitles.blockSignals(True); self.subtitles.setChecked(project.settings.subtitles_enabled); self.subtitles.blockSignals(False)
        self.subtitle_style.blockSignals(True); self.subtitle_style.setCurrentText(project.settings.subtitle_style); self.subtitle_style.blockSignals(False)
        self.encoder.blockSignals(True); self.encoder.setCurrentText(project.settings.encoder); self.encoder.blockSignals(False)
        self.cache.blockSignals(True); self.cache.setChecked(project.settings.use_cache); self.cache.blockSignals(False)
        self.recompute.blockSignals(True); self.recompute.setChecked(project.settings.recompute_all); self.recompute.blockSignals(False)

    def _runs_changed(self, runs: list[ProjectRun]) -> None:
        self.runs = runs
        while self.history_layout.count() > 1:
            item = self.history_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for run in runs:
            frame = QFrame(); frame.setObjectName("card")
            layout = QHBoxLayout(frame); layout.setContentsMargins(12, 10, 12, 10)
            text = QLabel(f"{run.started_at[:16].replace('T', ' ')} · {run.status.replace('_', ' ')}")
            text.setWordWrap(True)
            layout.addWidget(text, 1)
            result = next((Path(item) for item in run.artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()), None)
            if result:
                button = QPushButton("Открыть ролик")
                button.clicked.connect(lambda _, path=result: self._open_file(path))
                layout.addWidget(button)
            folder = QPushButton("Папка")
            folder.clicked.connect(lambda _, path=Path(run.log_path).parent if run.log_path else None: self._open_folder(path))
            layout.addWidget(folder)
            self.history_layout.insertWidget(self.history_layout.count() - 1, frame)
        if runs:
            result = next((Path(item) for item in runs[0].artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()), None)
            if result: self.preview.set_file(result)

    def _processing_changed(self, snapshot: ProcessingSnapshot) -> None:
        active = snapshot.phase in {"preparing", "running", "cancelling"}
        if active:
            self.progress.set_running(snapshot.stage_label, f"Прошло {format_seconds(snapshot.elapsed_seconds)}")
        else:
            self.progress.set_finished(snapshot.message)
        self.run_button.setDisabled(active)
        for widget in (self.subtitles, self.subtitle_style, self.encoder, self.cache, self.recompute):
            widget.setDisabled(active)

    def _open_project_folder(self) -> None:
        if self.project: self._open_folder(Path(self.project.project_directory))

    @staticmethod
    def _open_folder(path: Path | None) -> None:
        if path and path.is_dir(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _open_file(path: Path) -> None:
        if path.is_file(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _card(title: str) -> QFrame:
        card = QFrame(); card.setObjectName("card")
        layout = QVBoxLayout(card); layout.setContentsMargins(14, 12, 14, 12)
        label = QLabel(title); label.setStyleSheet("font-weight: 600;")
        layout.addWidget(label)
        return card

    @staticmethod
    def _replace_card_text(card: QFrame, values: list[str]) -> None:
        layout = card.layout()
        while layout.count() > 1:
            item = layout.takeAt(1)
            if item.widget(): item.widget().deleteLater()
        for value in values:
            label = QLabel(value); label.setObjectName("muted")
            layout.addWidget(label)

    def _error(self, error) -> None:
        QMessageBox.warning(self, error.title, error.user_message)
