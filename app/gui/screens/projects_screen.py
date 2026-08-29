from __future__ import annotations

import os
import stat
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.gui.components import ProjectPosterLoader, VideoDropZone
from app.gui.components.project_poster import project_poster_has_input, project_poster_path
from app.gui.models import DesktopProject, ProjectPresentation, RunStatus
from app.gui.responsive import set_responsive_text
from app.gui.services.error_mapping import dialog_message, map_error
from app.gui.viewmodels import ProjectsViewModel


class ProjectsScreen(QWidget):
    """Source onboarding and recent projects without a second navigation state."""

    project_opened = Signal(object)

    def __init__(self, viewmodel: ProjectsViewModel, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("screen")
        self.viewmodel = viewmodel
        self._projects: list[DesktopProject] = []
        self._presentations: dict[str, ProjectPresentation] = {}
        self._active_project_id: str | None = None
        self._rendered_columns = 0
        self._render_signature: tuple[object, ...] | None = None
        self._reflow_pending = False
        self._refresh_pending = False
        self._dirty = True
        self._compact_source_layout: bool | None = None
        self._thumbnail_labels: dict[str, list[QLabel]] = {}
        self._thumbnail_paths: dict[str, Path] = {}
        self._thumbnail_apply_queue: list[tuple[str, Path]] = []
        self._thumbnail_apply_timer = QTimer(self)
        self._thumbnail_apply_timer.setSingleShot(True)
        self._thumbnail_apply_timer.timeout.connect(self._flush_thumbnail_applies)
        self._thumbnail_loader = ProjectPosterLoader(self)
        self._thumbnail_loader.poster_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.poster_unavailable.connect(self._thumbnail_unavailable)

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 22)
        root.setSpacing(0)

        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # ``scroll`` used to name the projects list. Keep the public attribute
        # while making the whole compact workspace vertically scrollable.
        # QWidget already exposes a ``scroll`` method.  Preserve the legacy
        # instance attribute without shadowing that method in static typing.
        setattr(self, "scroll", self.content_scroll)
        host = QWidget()
        content = QVBoxLayout(host)
        content.setContentsMargins(0, 0, 0, 4)
        content.setSpacing(16)

        top = QHBoxLayout()
        self._top_layout = top
        titles = QVBoxLayout()
        titles.setSpacing(3)
        eyebrow = QLabel("CONTENT FACTORY")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Новый проект")
        title.setObjectName("title")
        subtitle = QLabel("Создайте короткие ролики из длинного видео — всё останется на вашем компьютере.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        titles.addWidget(eyebrow)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        top.addLayout(titles, 1)
        local_note = QLabel("▣  Исходники и проекты хранятся локально")
        local_note.setObjectName("muted")
        self.local_note = local_note
        titles.addWidget(local_note)
        local_note.setWordWrap(True)
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
        source_methods = QHBoxLayout()
        self._source_methods_layout = source_methods
        source_methods.setSpacing(20)

        file_panel = QWidget()
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(8)
        file_layout.addWidget(self.drop_zone)
        self.file_button = QPushButton("Выбрать видео")
        self.file_button.setObjectName("primary")
        self.file_button.setMinimumHeight(38)
        self.file_button.clicked.connect(self.choose_file)
        # Compatibility with the original screen and any integrations that use
        # its button directly.
        self.new_button = self.file_button
        file_layout.addWidget(self.file_button, 0, Qt.AlignmentFlag.AlignHCenter)
        formats = QLabel("MP4, MOV, MKV, AVI, WebM и M4V")
        formats.setObjectName("muted")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(formats)

        self.source_divider = QWidget()
        source_divider_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self.source_divider)
        self._source_divider_layout = source_divider_layout
        source_divider_layout.setContentsMargins(0, 8, 0, 8)
        source_divider_layout.setSpacing(8)
        self.source_divider_before = QFrame()
        self.source_divider_before.setObjectName("sourceDivider")
        self.source_divider_after = QFrame()
        self.source_divider_after.setObjectName("sourceDivider")
        self.source_divider_label = QLabel("или")
        self.source_divider_label.setObjectName("muted")
        self.source_divider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source_divider_layout.addWidget(self.source_divider_before, 1)
        source_divider_layout.addWidget(self.source_divider_label)
        source_divider_layout.addWidget(self.source_divider_after, 1)

        self.url_panel = QWidget()
        url_panel_layout = QVBoxLayout(self.url_panel)
        url_panel_layout.setContentsMargins(12, 8, 0, 8)
        url_panel_layout.setSpacing(10)
        url_panel_layout.addStretch()
        url_heading = QLabel("Вставьте ссылку на видео")
        url_heading.setStyleSheet("font-size: 15px; font-weight: 600;")
        url_panel_layout.addWidget(url_heading)

        url_row = QHBoxLayout()
        self._url_row_layout = url_row
        url_row.setSpacing(8)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Вставьте ссылку на открытое видео")
        self.url_input.returnPressed.connect(self._create_url)
        self.url_button = QPushButton("Добавить видео")
        self.url_button.clicked.connect(self._create_url)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.url_button)
        url_panel_layout.addLayout(url_row)
        public_note = QLabel("Подойдут только видео, доступные без входа, оплаты и других ограничений.")
        public_note.setObjectName("muted")
        public_note.setWordWrap(True)
        url_panel_layout.addWidget(public_note)
        url_panel_layout.addStretch()

        source_methods.addWidget(file_panel, 1)
        source_methods.addWidget(self.source_divider)
        source_methods.addWidget(self.url_panel, 1)
        source_layout.addLayout(source_methods)
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
        self.list_layout.setHorizontalSpacing(10)
        self.list_layout.setVerticalSpacing(10)
        content.addWidget(self.list_host)
        content.addStretch()

        self.content_scroll.setWidget(host)
        root.addWidget(self.content_scroll, 1)

        self.viewmodel.projects_changed.connect(self._render)
        self.viewmodel.project_created.connect(self.project_opened)
        self.viewmodel.error_occurred.connect(self._show_error)
        self.viewmodel.url_busy_changed.connect(self._url_busy_changed)
        self.viewmodel.source_busy_changed.connect(self._source_busy_changed)
        self._apply_responsive_layout(force=True)

    def refresh(self) -> None:
        self._dirty = False
        self.viewmodel.refresh()

    def mark_dirty(self) -> None:
        """Invalidate recent projects without rebuilding a hidden screen."""

        self._dirty = True
        if self.isVisible():
            self._queue_refresh()

    def refresh_if_dirty(self) -> None:
        if self._dirty:
            self._queue_refresh()

    def _queue_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._flush_refresh)

    def _flush_refresh(self) -> None:
        self._refresh_pending = False
        if not self.isVisible():
            return
        if self._dirty:
            self.refresh()

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

    def _source_busy_changed(self, busy: bool) -> None:
        self.drop_zone.setDisabled(busy)
        self.file_button.setDisabled(busy)
        self.url_input.setDisabled(busy)
        self.url_button.setDisabled(busy)
        self.file_button.setText("Проверяем видео…" if busy else "Выбрать видео")

    def _render(self, projects: list[DesktopProject]) -> None:
        signature = self._projection_signature(projects)
        if signature == self._render_signature:
            return
        self._thumbnail_loader.replace_pending()
        self._thumbnail_apply_timer.stop()
        self._thumbnail_apply_queue.clear()
        self._projects = list(projects)
        self._presentations = {}
        active_project_id: str | None = None
        for project in self._projects:
            runs = self.viewmodel.services.runs_for(project)
            self._presentations[project.project_id] = self.viewmodel.services.presentation(
                project, runs=runs,
            )
            if active_project_id is None and (
                project.source_spec.download_state == "downloading"
                or any(run.status in RunStatus.ACTIVE for run in runs)
            ):
                active_project_id = project.project_id
        self._active_project_id = active_project_id
        self.recent_count.setText(f"{len(projects)}" if projects else "")
        self.empty.setVisible(not projects)
        self._render_cards()
        self._render_signature = signature

    def _projection_signature(self, projects: list[DesktopProject]) -> tuple[object, ...]:
        """Cheaply identify the exact persisted card projection.

        ``updated_at`` changes on every project save.  Run records have their
        own lifecycle, so include their file revisions without parsing every
        JSON payload.  This keeps an unchanged Projects refresh cheap while a
        status transition, recovery update, or restart still invalidates the
        presentation and active-job projection.
        """

        revisions: list[tuple[object, ...]] = []
        for project in projects:
            run_revisions: list[tuple[str, int, int]] = []
            runs_directory = project.directory / "runs"
            try:
                with os.scandir(runs_directory) as entries:
                    for run_directory in entries:
                        if not run_directory.is_dir(follow_symlinks=False):
                            continue
                        run_path = Path(run_directory.path) / "run.json"
                        try:
                            run_stat = run_path.stat()
                        except FileNotFoundError:
                            continue
                        except OSError:
                            run_revisions.append((run_directory.name, -1, -1))
                            continue
                        if not stat.S_ISREG(run_stat.st_mode):
                            continue
                        run_revisions.append((
                            run_directory.name,
                            run_stat.st_size,
                            run_stat.st_mtime_ns,
                        ))
            except FileNotFoundError:
                pass
            except OSError:
                # Preserve a distinct revision for an inaccessible run store;
                # a later successful stat cannot reuse this projection.
                run_revisions.append(("unavailable", -1, -1))
            revisions.append((
                project.project_id,
                project.updated_at,
                project.status,
                project.latest_run_id,
                project.thumbnail_path,
                project.source_spec.download_state,
                tuple(sorted(run_revisions)),
            ))
        return (self._recent_columns(), tuple(revisions))

    def _render_cards(self) -> None:
        self._thumbnail_apply_timer.stop()
        self._thumbnail_apply_queue.clear()
        # QGridLayout retains stretch factors after its widgets are removed.
        # Reset them so a wide -> compact resize cannot leave empty historical
        # columns consuming part of the new one-column workspace.
        for row in range(self.list_layout.rowCount()):
            self.list_layout.setRowStretch(row, 0)
        for column in range(self.list_layout.columnCount()):
            self.list_layout.setColumnStretch(column, 0)
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        self._thumbnail_labels = {}
        self._thumbnail_paths = {}
        columns = self._recent_columns()
        self._rendered_columns = columns
        for index, project in enumerate(self._projects):
            row, column = divmod(index, columns)
            self.list_layout.addWidget(self._card(project, self._active_project_id), row, column)
        if self._projects:
            self.list_layout.setRowStretch((len(self._projects) - 1) // columns + 1, 1)
        for column in range(columns):
            self.list_layout.setColumnStretch(column, 1)
        # QScrollArea can retain the old multi-column widget width for one
        # layout pass after the grid minimum shrinks.  Synchronise on the next
        # event-loop turn so a wide -> compact transition cannot expose a
        # stale hidden horizontal range.
        QTimer.singleShot(0, self._sync_content_width)

    def _sync_content_width(self) -> None:
        host = self.content_scroll.widget()
        viewport = self.content_scroll.viewport()
        if host is None or viewport.width() <= 0:
            return
        self.list_layout.activate()
        host_layout = host.layout()
        if host_layout is not None:
            host_layout.activate()
        if host.minimumSizeHint().width() <= viewport.width() and host.width() != viewport.width():
            host.resize(viewport.width(), host.height())

    def _recent_columns(self) -> int:
        # Use the real scroll viewport instead of the outer screen width.  A
        # project card owns three actions and therefore needs substantially
        # more room than its title alone suggests.  The previous thresholds
        # placed two or three real persisted cards into columns that could not
        # contain their minimum action row, while the hidden horizontal
        # scrollbar made the resulting clipping look like a successful fit.
        viewport_width = self.content_scroll.viewport().width()
        available = viewport_width if viewport_width > 0 else max(0, self.width() - 52)
        # Dense recent cards mirror the approved Source composition: four on a
        # wide desktop, then 3/2/1 without hiding an action rail.  Thresholds
        # retain a scrollbar-sized margin for a refresh that grows vertically.
        if available >= 1_120:
            return 4
        if available >= 840:
            return 3
        if available >= 600:
            return 2
        return 1

    def _card(self, project: DesktopProject, active_project_id: str | None) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 11, 12, 11)
        layout.setSpacing(6)

        poster = QLabel("Готовим кадр…")
        poster.setObjectName("projectPoster")
        poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        poster.setMinimumHeight(92)
        poster.setMaximumHeight(124)
        # Ignore the pixmap's native size hint; card width owns the crop.
        poster.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(poster)
        self._thumbnail_labels.setdefault(project.project_id, []).append(poster)
        expected = project_poster_path(project).resolve(strict=False)
        persisted = Path(project.thumbnail_path).resolve(strict=False) if project.thumbnail_path else None
        if persisted == expected and expected.is_file():
            self._thumbnail_paths[project.project_id] = expected
            self._queue_thumbnail_apply(project.project_id, expected)
        elif project_poster_has_input(project):
            destination = self._thumbnail_loader.request(project)
            self._thumbnail_paths[project.project_id] = destination.resolve(strict=False)
        else:
            poster.setText("Видео будет доступно после загрузки")
        top = QHBoxLayout()
        name = QLabel()
        name.setStyleSheet("font-size: 15px; font-weight: 600;")
        name.setWordWrap(True)
        name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        set_responsive_text(name, project.name.replace("_", " "))
        name.setToolTip(project.name)
        presentation = self._presentations.get(project.project_id)
        if presentation is None:
            runs = self.viewmodel.services.runs_for(project)
            presentation = self.viewmodel.services.presentation(project, runs=runs)
        status = QLabel()
        status.setObjectName("status")
        status.setWordWrap(True)
        # Keep the compact state badge visible while the project title absorbs
        # the flexible width. Ignored + zero stretch collapses this label to
        # zero even though the card itself fits.
        status.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        status.setMaximumWidth(156)
        set_responsive_text(status, presentation.status_label)
        top.addWidget(name, 1)
        top.addWidget(status, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)

        source_name = (
            project.source.name if project.source_spec.is_ready
            else str(project.source_metadata.get("title") or "Видео по ссылке")
        )
        source = QLabel()
        source.setObjectName("muted")
        source.setWordWrap(True)
        source.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        set_responsive_text(source, source_name)
        updated = QLabel(f"Изменён {project.updated_at[:16].replace('T', ' ')}")
        updated.setObjectName("muted")
        updated.setWordWrap(True)
        updated.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(source)
        layout.addWidget(updated)

        actions = QGridLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setHorizontalSpacing(7)
        actions.setVerticalSpacing(6)
        open_button = QPushButton("Открыть")
        open_button.clicked.connect(lambda _checked=False, value=project: self.project_opened.emit(value))
        folder_button = QPushButton("Папка")
        folder_button.setToolTip("Открыть папку проекта")
        folder_button.clicked.connect(
            lambda _checked=False, path=project.project_directory: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        )
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("danger")
        active = active_project_id == project.project_id
        delete_button.setDisabled(active)
        if active:
            delete_button.setToolTip("Проект нельзя удалить, пока его обработка не завершена или не остановлена.")
        delete_button.clicked.connect(lambda _checked=False, value=project: self._delete(value))
        # Keep every existing action directly available, but avoid making
        # three padded buttons the minimum width of the whole recent card.
        # The approved dense grid gives the primary open action its own row;
        # filesystem and destructive utilities share the compact second row.
        actions.addWidget(open_button, 0, 0, 1, 2)
        actions.addWidget(folder_button, 1, 0)
        actions.addWidget(delete_button, 1, 1)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)
        layout.addLayout(actions)
        return card

    def _thumbnail_ready(self, project_id: str, path: str) -> None:
        expected = self._thumbnail_paths.get(project_id)
        actual = Path(path).resolve(strict=False)
        if expected is None or expected != actual:
            return
        self._apply_thumbnail(project_id, actual)
        project = next((item for item in self._projects if item.project_id == project_id), None)
        if project is not None and project.thumbnail_path != str(actual):
            try:
                self.viewmodel.services.update_project_thumbnail(project, actual)
            except Exception:
                # The real cached frame is already visible. Persistence can be
                # retried by the next normal project refresh.
                pass

    def _queue_thumbnail_apply(self, project_id: str, path: Path) -> None:
        """Decode cached posters in small GUI batches instead of one stall."""

        self._thumbnail_apply_queue.append((project_id, path))
        if not self._thumbnail_apply_timer.isActive():
            self._thumbnail_apply_timer.start(0)

    def _flush_thumbnail_applies(self) -> None:
        batch = self._thumbnail_apply_queue[:4]
        del self._thumbnail_apply_queue[:4]
        for project_id, path in batch:
            if self._thumbnail_paths.get(project_id) == path:
                self._apply_thumbnail(project_id, path)
        if self._thumbnail_apply_queue:
            self._thumbnail_apply_timer.start(16)

    def _thumbnail_unavailable(self, project_id: str, path: str) -> None:
        expected = self._thumbnail_paths.get(project_id)
        if expected is None or expected != Path(path).resolve(strict=False):
            return
        for label in self._thumbnail_labels.get(project_id, []):
            try:
                label.setText("Кадр недоступен\nВидео можно открыть")
            except RuntimeError:
                continue

    def _apply_thumbnail(self, project_id: str, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        for label in self._thumbnail_labels.get(project_id, []):
            try:
                label.setText("")
                label.setPixmap(pixmap.scaled(
                    max(1, label.width()), max(1, label.height()),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            except RuntimeError:
                continue

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
        self._apply_responsive_layout()
        if not self._projects or self._recent_columns() == self._rendered_columns or self._reflow_pending:
            return
        self._reflow_pending = True
        QTimer.singleShot(0, self._finish_reflow)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.refresh_if_dirty()
        # The first refresh normally happens before QScrollArea has a real
        # viewport. Re-evaluate once the native window publishes that width so
        # a wide first show does not remain stuck in the fallback one column.
        if self._projects and not self._reflow_pending:
            self._reflow_pending = True
            QTimer.singleShot(0, self._finish_reflow)

    def _finish_reflow(self) -> None:
        self._reflow_pending = False
        columns = self._recent_columns()
        if columns != self._rendered_columns:
            self._render_cards()
        if self._render_signature is not None:
            # The projection itself did not change; only its responsive grid
            # did.  Record the new column count so the next warm refresh can
            # reuse the exact cards instead of rebuilding them a second time.
            self._render_signature = (
                columns,
                self._render_signature[1],
            )

    def _apply_responsive_layout(self, *, force: bool = False) -> None:
        """Reflow source onboarding before a scaled laptop viewport clips it.

        A 1280 px display at 150% scaling leaves this screen with roughly
        600 logical pixels after the shell sidebar. Keeping the local-work
        status beside the title makes the scroll host claim a wider minimum
        than its viewport, which only hides the horizontal scrollbar. The
        compact composition keeps the approved content, stacked in order.
        """

        compact = self.width() < 900
        if not force and compact == self._compact_source_layout:
            return
        self._compact_source_layout = compact
        self._top_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        self._top_layout.setSpacing(6 if compact else 0)
        self._top_layout.setAlignment(
            self.local_note,
            Qt.AlignmentFlag.AlignLeft if compact else Qt.AlignmentFlag.AlignTop,
        )
        self._source_methods_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        self._source_methods_layout.setSpacing(12 if compact else 20)
        divider_shape = QFrame.Shape.HLine if compact else QFrame.Shape.VLine
        self.source_divider_before.setFrameShape(divider_shape)
        self.source_divider_after.setFrameShape(divider_shape)
        self._source_divider_layout.setDirection(
            QBoxLayout.Direction.LeftToRight
            if compact
            else QBoxLayout.Direction.TopToBottom
        )
        self.source_divider.setSizePolicy(
            QSizePolicy.Policy.Expanding if compact else QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed if compact else QSizePolicy.Policy.Expanding,
        )
        self.url_panel.layout().setContentsMargins(0, 8, 0, 8)
        # Give the URL field and its CTA independent full rows in the same
        # compact profile. This avoids relying on a few spare pixels that can
        # disappear with Windows font scaling.
        self._url_row_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if compact
            else QBoxLayout.Direction.LeftToRight
        )
        self._url_row_layout.setSpacing(8)
        self.updateGeometry()

    def _show_error(self, error) -> None:
        QMessageBox.warning(self, error.title, dialog_message(error))
