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
        self.estimate = self._card("Предварительная оценка")
        left.addWidget(self.estimate)
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
        heading = QLabel("Как подготовить ролики")
        heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        settings.addWidget(heading)
        settings.addWidget(QLabel("Режим обработки"))
        self.processing_mode = QComboBox()
        self.processing_mode.addItem("Быстро", "fast")
        self.processing_mode.addItem("Стандарт", "standard")
        self.processing_mode.addItem("Максимальное качество", "maximum")
        self.processing_mode.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(processing_mode=str(self.processing_mode.currentData()))
        )
        settings.addWidget(self.processing_mode)
        settings.addWidget(QLabel("Глубокий анализ видео"))
        self.deep_analysis = QComboBox()
        self.deep_analysis.addItem("Автоматически", "auto")
        self.deep_analysis.addItem("Включён", "on")
        self.deep_analysis.addItem("Выключен", "off")
        self.deep_analysis.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(deep_analysis=str(self.deep_analysis.currentData()))
        )
        settings.addWidget(self.deep_analysis)
        settings.addWidget(QLabel("Площадка"))
        self.platform = QComboBox()
        self.platform.addItem("TikTok", "tiktok")
        self.platform.addItem("Instagram Reels", "reels")
        self.platform.addItem("YouTube Shorts", "shorts")
        self.platform.addItem("Универсальный вертикальный", "universal")
        self.platform.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(platform=str(self.platform.currentData()))
        )
        settings.addWidget(self.platform)
        settings.addWidget(QLabel("Количество роликов"))
        self.clip_count = QComboBox()
        for label, value in (("Авто", "auto"), ("1 ролик", "1"), ("3 ролика", "3"), ("5 роликов", "5")):
            self.clip_count.addItem(label, value)
        self.clip_count.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(clip_count=str(self.clip_count.currentData()))
        )
        settings.addWidget(self.clip_count)
        settings.addWidget(QLabel("Субтитры"))
        self.subtitles = QCheckBox("Показывать субтитры")
        self.subtitles.toggled.connect(lambda value: self.viewmodel.save_options(subtitles_enabled=value))
        settings.addWidget(self.subtitles)
        settings.addWidget(QLabel("Стиль субтитров"))
        self.subtitle_style = QComboBox()
        self.subtitle_style.addItem("Документальный", "documentary")
        self.subtitle_style.addItem("Чистый", "clean")
        self.subtitle_style.addItem("Минималистичный", "minimal")
        self.subtitle_style.addItem("Динамичный", "dynamic")
        self.subtitle_style.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(subtitle_style=str(self.subtitle_style.currentData()))
        )
        settings.addWidget(self.subtitle_style)
        self.cache = QCheckBox("Использовать готовый анализ, если он есть")
        self.cache.toggled.connect(lambda value: self.viewmodel.save_options(use_cache=value))
        settings.addWidget(self.cache)
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
        if project.source_spec.is_ready:
            self.preview.set_file(str(project.source))
        source = project.source_metadata
        duration = format_seconds(source.get("duration")) if source else "н/д"
        resolution = f"{source.get('width', '—')} × {source.get('height', '—')}" if source else "н/д"
        fps = source.get("fps", "н/д") if source else "н/д"
        size = self._format_file_size(source.get("size_bytes") or source.get("estimated_size_bytes")) if source else "н/д"
        source_name = project.source.name if project.source_spec.is_ready else str(source.get("title") or "Видео по ссылке")
        source_kind = "Ссылка на видео" if project.source_spec.kind == "url" else "Файл"
        self._replace_card_text(self.metadata, [
            f"{source_kind}: {source_name}", f"Длительность: {duration}",
            f"Разрешение: {resolution}", f"Размер: {size}", f"FPS: {fps}",
        ])
        self._update_estimate(project)
        self._set_combo_data(self.processing_mode, project.settings.processing_mode)
        self._set_combo_data(self.deep_analysis, project.settings.deep_analysis)
        self._set_combo_data(self.platform, project.settings.platform)
        self._set_combo_data(self.clip_count, str(project.settings.clip_count))
        self.subtitles.blockSignals(True); self.subtitles.setChecked(project.settings.subtitles_enabled); self.subtitles.blockSignals(False)
        self._set_combo_data(self.subtitle_style, project.settings.subtitle_style)
        self.cache.blockSignals(True); self.cache.setChecked(project.settings.use_cache); self.cache.blockSignals(False)

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
            results = [Path(item) for item in run.artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()]
            for index, result in enumerate(results, start=1):
                button = QPushButton("Открыть ролик" if len(results) == 1 else f"Ролик {index}")
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
            activity = snapshot.last_activity_at or "ожидаем запуск"
            transfer = ""
            if snapshot.transfer_speed:
                transfer = f" · Скорость: {snapshot.transfer_speed}"
            if snapshot.eta_seconds is not None:
                transfer += f" · Осталось: {format_seconds(snapshot.eta_seconds)}"
            self.progress.set_running(
                snapshot.stage_label,
                f"Прошло {format_seconds(snapshot.elapsed_seconds)} · Активность: {activity}{transfer}",
                snapshot.progress_fraction,
            )
        else:
            self.progress.set_finished(snapshot.message)
        self.run_button.setDisabled(active)
        for widget in (
            self.processing_mode, self.deep_analysis, self.platform, self.clip_count,
            self.subtitles, self.subtitle_style, self.cache,
        ):
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

    def _update_estimate(self, project: DesktopProject) -> None:
        try:
            estimate = self.viewmodel.services.processing_estimate(project)
            minutes = (
                f"около {max(1, round(estimate.estimated_seconds_min / 60))}–"
                f"{max(1, round(estimate.estimated_seconds_max / 60))} мин"
            )
            cost = (
                "без платного AI"
                if estimate.estimated_ai_cost_min is None
                else f"ориентировочно ${estimate.estimated_ai_cost_min:.2f}–${estimate.estimated_ai_cost_max:.2f}"
            )
            analysis = "будет использован" if estimate.deep_analysis_resolved else "не потребуется"
            self._replace_card_text(self.estimate, [
                f"Время: {minutes}",
                f"Результат: примерно {estimate.estimated_clips_min}–{estimate.estimated_clips_max} ролика(ов)",
                f"Глубокий анализ: {analysis}",
                f"Стоимость AI: {cost}",
            ])
        except Exception:
            self._replace_card_text(self.estimate, ["Оценка появится после проверки настроек."])

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        combo.blockSignals(True)
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @staticmethod
    def _format_file_size(value: object) -> str:
        try:
            size = float(value)
        except (TypeError, ValueError):
            return "н/д"
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if size < 1024 or unit == "ГБ":
                return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
            size /= 1024
        return "н/д"

    def _error(self, error) -> None:
        QMessageBox.warning(self, error.title, error.user_message)
