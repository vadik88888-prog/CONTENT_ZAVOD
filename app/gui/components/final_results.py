from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from app.gui.components.candidate_thumbnail import CandidateThumbnailLoader
from app.gui.components.video_preview import VideoPreview


@dataclass(frozen=True, slots=True)
class FinalOutput:
    """One validated, metadata-bound production output shown to a person."""

    result_id: str
    candidate_id: str
    path: Path
    title: str
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    status: str = "completed"


class _FinalOutputCard(QFrame):
    activated = Signal(str)

    def __init__(self, result_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result_id = result_id
        self.setObjectName("finalOutputCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.result_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.activated.emit(self.result_id)
            event.accept()
            return
        super().keyPressEvent(event)


class FinalResultsWorkspace(QWidget):
    """The final-result viewer; it never infers output paths from filenames."""

    output_selected = Signal(str)
    create_more_requested = Signal()
    projects_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("finalResultsWorkspace")
        self._outputs: list[FinalOutput] = []
        self._active_id: str | None = None
        self._signature: tuple[tuple[str, str], ...] = ()
        self._project_directory: Path | None = None
        self._warnings: list[str] = []
        self._cards: dict[str, _FinalOutputCard] = {}
        self._thumbnail_labels: dict[str, QLabel] = {}
        self._thumbnail_loader = CandidateThumbnailLoader(self)
        self._thumbnail_loader.thumbnail_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.thumbnail_unavailable.connect(self._thumbnail_unavailable)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        heading = QLabel("Готовые ролики")
        heading.setObjectName("title")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        titles.addWidget(heading)
        titles.addWidget(self.subtitle)
        header.addLayout(titles)
        header.addStretch()
        root.addLayout(header)

        self.stepper = QFrame()
        self.stepper.setObjectName("resultsStepper")
        stepper_layout = QHBoxLayout(self.stepper)
        stepper_layout.setContentsMargins(14, 10, 14, 10)
        for index, label in enumerate(("Источник", "Настройка", "Обработка", "4  Результаты")):
            item = QLabel(("✓  " if index < 3 else "") + label)
            item.setObjectName("resultsStepActive" if index == 3 else "resultsStep")
            stepper_layout.addWidget(item)
            if index < 3:
                divider = QLabel("—")
                divider.setObjectName("muted")
                stepper_layout.addWidget(divider)
        stepper_layout.addStretch()
        root.addWidget(self.stepper)

        self.summary = QFrame()
        self.summary.setObjectName("finalSummary")
        summary_layout = QHBoxLayout(self.summary)
        summary_layout.setContentsMargins(18, 12, 18, 12)
        self.summary_values: dict[str, QLabel] = {}
        for key, caption in (
            ("count", "Роликов создано"), ("duration", "Общая длительность"),
            ("format", "Формат"), ("size", "Общий размер"),
        ):
            column = QVBoxLayout()
            value = QLabel("—")
            value.setObjectName("finalSummaryValue")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption_label = QLabel(caption)
            caption_label.setObjectName("muted")
            caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(value)
            column.addWidget(caption_label)
            summary_layout.addLayout(column, 1)
            self.summary_values[key] = value
        root.addWidget(self.summary)

        body_host = QWidget()
        # Keep the result action bar in the initial viewport even when the
        # project has real warning text. The warning list itself scrolls.
        body_host.setMaximumHeight(590)
        body = QHBoxLayout(body_host)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)

        list_panel = QFrame()
        list_panel.setObjectName("finalOutputList")
        list_panel.setMinimumWidth(300)
        list_panel.setMaximumWidth(360)
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(10, 10, 10, 10)
        list_heading = QLabel("Готовые ролики")
        list_heading.setStyleSheet("font-weight: 600;")
        list_layout.addWidget(list_heading)
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_host = QWidget()
        self.list_items = QVBoxLayout(self.list_host)
        self.list_items.setContentsMargins(0, 0, 0, 0)
        self.list_items.setSpacing(8)
        self.list_items.addStretch()
        self.list_scroll.setWidget(self.list_host)
        list_layout.addWidget(self.list_scroll, 1)
        self.create_more_button = QPushButton("＋  Создать ещё ролики")
        self.create_more_button.setObjectName("secondaryAction")
        self.create_more_button.setToolTip("Вернуться к моментам и выбрать другие ролики без повторного анализа.")
        self.create_more_button.clicked.connect(self.create_more_requested)
        list_layout.addWidget(self.create_more_button)
        body.addWidget(list_panel, 1)

        self.preview = VideoPreview()
        self.preview.set_vertical_frame_size(247, 440)
        self.preview.open_button.hide()
        body.addWidget(self.preview, 2, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        info_panel = QFrame()
        info_panel.setObjectName("finalInfo")
        info_panel.setMinimumWidth(270)
        info_panel.setMaximumWidth(350)
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_heading = QLabel("Информация о ролике")
        info_heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        info_layout.addWidget(info_heading)
        self.info_grid = QGridLayout()
        self.info_grid.setHorizontalSpacing(14)
        self.info_grid.setVerticalSpacing(10)
        self.info_values: dict[str, QLabel] = {}
        for row, (key, label) in enumerate((
            ("source", "Исходный момент"), ("duration", "Длительность"),
            ("format", "Формат"), ("size", "Размер файла"), ("created", "Создан"),
        )):
            name = QLabel(label)
            name.setObjectName("muted")
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setWordWrap(True)
            self.info_grid.addWidget(name, row, 0)
            self.info_grid.addWidget(value, row, 1)
            self.info_values[key] = value
        info_layout.addLayout(self.info_grid)

        self.warning_box = QFrame()
        self.warning_box.setObjectName("finalWarnings")
        warning_layout = QVBoxLayout(self.warning_box)
        warning_layout.setContentsMargins(10, 10, 10, 10)
        warning_heading = QLabel("Готово с предупреждениями")
        warning_heading.setObjectName("warning")
        warning_layout.addWidget(warning_heading)
        warning_scroll = QScrollArea()
        warning_scroll.setWidgetResizable(True)
        warning_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        warning_scroll.setFixedHeight(108)
        warning_host = QWidget()
        warning_host_layout = QVBoxLayout(warning_host)
        warning_host_layout.setContentsMargins(0, 0, 0, 0)
        self.warning_text = QLabel()
        self.warning_text.setWordWrap(True)
        self.warning_text.setObjectName("muted")
        warning_host_layout.addWidget(self.warning_text)
        warning_host_layout.addStretch()
        warning_scroll.setWidget(warning_host)
        warning_layout.addWidget(warning_scroll)
        self.warning_box.hide()
        info_layout.addWidget(self.warning_box)
        info_layout.addStretch()
        actions_heading = QLabel("Действия")
        actions_heading.setStyleSheet("font-weight: 600;")
        info_layout.addWidget(actions_heading)
        self.open_video_button = QPushButton("▷  Открыть видео")
        self.open_video_button.setObjectName("primary")
        self.open_video_button.clicked.connect(self._open_active_video)
        self.show_folder_button = QPushButton("▢  Показать в папке")
        self.show_folder_button.clicked.connect(self._show_active_folder)
        info_layout.addWidget(self.open_video_button)
        info_layout.addWidget(self.show_folder_button)
        body.addWidget(info_panel, 1)
        root.addWidget(body_host, 1)

        bottom = QFrame()
        bottom.setObjectName("finalActionBar")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(12, 10, 12, 10)
        self.project_folder_button = QPushButton("▢  Открыть папку проекта")
        self.project_folder_button.clicked.connect(self._open_project_folder)
        self.projects_button = QPushButton("К проектам")
        self.projects_button.setObjectName("primary")
        self.projects_button.clicked.connect(self.projects_requested)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.project_folder_button)
        bottom_layout.addWidget(self.projects_button)
        root.addWidget(bottom)

    @property
    def active_output_id(self) -> str | None:
        return self._active_id

    def set_results(
        self, outputs: list[FinalOutput], *, selected_id: str | None, project_directory: Path,
        warnings: list[str] | None = None,
    ) -> None:
        signature = tuple((output.result_id, str(output.path)) for output in outputs)
        changed = signature != self._signature
        self._outputs = list(outputs)
        self._project_directory = project_directory
        self._warnings = list(warnings or [])
        self.subtitle.setText(f"{len(outputs)} {self._plural(len(outputs), 'ролик готов к публикации', 'ролика готовы к публикации', 'роликов готовы к публикации')}")
        self._update_summary()
        if changed:
            self._signature = signature
            self._rebuild_list()
        target = selected_id if any(output.result_id == selected_id for output in outputs) else (outputs[0].result_id if outputs else None)
        if target and (changed or target != self._active_id):
            self._activate(target, emit=False, warnings=self._warnings)
        elif target:
            self._render_details(self._output_for(target), self._warnings)
            self._mark_active()

    def _rebuild_list(self) -> None:
        while self.list_items.count() > 1:
            item = self.list_items.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = {}
        self._thumbnail_labels = {}
        for number, output in enumerate(self._outputs, start=1):
            card = _FinalOutputCard(output.result_id)
            card.activated.connect(lambda result_id: self._activate(result_id, emit=True, warnings=self._warnings))
            layout = QHBoxLayout(card)
            layout.setContentsMargins(9, 9, 9, 9)
            thumbnail = QLabel("Первый кадр\nзагружается")
            thumbnail.setObjectName("finalThumbnail")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setFixedSize(72, 128)
            thumbnail.setWordWrap(True)
            layout.addWidget(thumbnail)
            self._thumbnail_labels[output.result_id] = thumbnail
            text = QVBoxLayout()
            title = QLabel(f"{number}. {output.title}")
            title.setStyleSheet("font-weight: 600;")
            title.setWordWrap(True)
            state = QLabel("● Готово" if output.status == "completed" else "● Готово с предупреждениями")
            state.setObjectName("finalReady" if output.status == "completed" else "warning")
            detail = QLabel(f"{self._seconds(output.duration_seconds)}  ·  {self._size(self._file_size(output.path))}")
            detail.setObjectName("muted")
            detail.setWordWrap(True)
            text.addWidget(title)
            text.addWidget(state)
            text.addWidget(detail)
            text.addStretch()
            layout.addLayout(text, 1)
            self.list_items.insertWidget(self.list_items.count() - 1, card)
            self._cards[output.result_id] = card
            if self._project_directory and output.path.is_file():
                self._thumbnail_loader.request(
                    cache_directory=self._project_directory / "final-thumbnails",
                    analysis_id="final-results",
                    candidate_id=output.result_id,
                    source_path=output.path,
                    timestamp_seconds=0.05,
                )

    def _activate(self, result_id: str, *, emit: bool, warnings: list[str]) -> None:
        output = self._output_for(result_id)
        if output is None:
            return
        self._active_id = output.result_id
        self._render_details(output, warnings)
        self._mark_active()
        # Update card/details synchronously, then let VideoPreview queue the
        # backend source handoff.  A slow multimedia backend must never delay
        # the visible selection change.
        self.preview.show_final(output.path, output.title)
        if emit:
            self.output_selected.emit(output.result_id)

    def _render_details(self, output: FinalOutput | None, warnings: list[str]) -> None:
        if output is None:
            return
        source = "Не указан в metadata"
        if output.source_start_seconds is not None and output.source_end_seconds is not None:
            source = f"{self._seconds(output.source_start_seconds)} – {self._seconds(output.source_end_seconds)}"
        size = self._file_size(output.path)
        dimensions = (
            f"{output.width}×{output.height} (9:16)"
            if output.width and output.height else "Не удалось определить"
        )
        created = "—"
        try:
            created = datetime.fromtimestamp(output.path.stat().st_mtime).strftime("%d.%m.%Y, %H:%M")
        except OSError:
            pass
        values = {
            "source": source,
            "duration": self._seconds(output.duration_seconds),
            "format": dimensions,
            "size": self._size(size),
            "created": created,
        }
        for key, value in values.items():
            self.info_values[key].setText(value)
        all_warnings = list(dict.fromkeys(warning for warning in warnings if warning.strip()))
        if output.status == "warning" and not all_warnings:
            all_warnings.append("Файл создан с предупреждением, указанным в metadata результата.")
        self.warning_text.setText("\n".join(f"• {warning}" for warning in all_warnings))
        self.warning_box.setVisible(bool(all_warnings))
        available = output.path.is_file()
        self.open_video_button.setEnabled(available)
        self.show_folder_button.setEnabled(available)

    def _mark_active(self) -> None:
        for result_id, card in self._cards.items():
            card.setProperty("activeFinalOutput", result_id == self._active_id)
            card.style().unpolish(card)
            card.style().polish(card)

    def _thumbnail_ready(self, result_id: str, path: str) -> None:
        label = self._thumbnail_labels.get(result_id)
        if label is None:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        label.setText("")
        label.setPixmap(pixmap.scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _thumbnail_unavailable(self, result_id: str) -> None:
        label = self._thumbnail_labels.get(result_id)
        if label is not None:
            label.setText("Первый кадр\nнедоступен")

    def _open_active_video(self) -> None:
        output = self._output_for(self._active_id)
        if output and output.path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.path)))

    def _show_active_folder(self) -> None:
        output = self._output_for(self._active_id)
        if output and output.path.parent.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.path.parent)))

    def _open_project_folder(self) -> None:
        if self._project_directory and self._project_directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project_directory)))

    def _output_for(self, result_id: str | None) -> FinalOutput | None:
        return next((output for output in self._outputs if output.result_id == result_id), None)

    def _update_summary(self) -> None:
        durations = [value.duration_seconds for value in self._outputs if value.duration_seconds is not None]
        sizes = [self._file_size(value.path) for value in self._outputs]
        formats = {(value.width, value.height) for value in self._outputs if value.width and value.height}
        format_text = "—"
        if len(formats) == 1:
            width, height = next(iter(formats))
            format_text = f"{width}×{height}"
        self.summary_values["count"].setText(str(len(self._outputs)))
        self.summary_values["duration"].setText(self._seconds(sum(durations)) if durations else "—")
        self.summary_values["format"].setText(format_text)
        self.summary_values["size"].setText(self._size(sum(sizes)))

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _seconds(value: float | None) -> str:
        if value is None:
            return "—"
        whole = max(0, round(value))
        return f"{whole // 60:02d}:{whole % 60:02d}"

    @staticmethod
    def _size(value: int) -> str:
        if value < 1024 * 1024:
            return f"{value / 1024:.0f} КБ"
        if value < 1024 * 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} МБ"
        return f"{value / (1024 * 1024 * 1024):.2f} ГБ"

    @staticmethod
    def _plural(value: int, one: str, few: str, many: str) -> str:
        if value % 10 == 1 and value % 100 != 11:
            return one
        if value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
            return few
        return many
