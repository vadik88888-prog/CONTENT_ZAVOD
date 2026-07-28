from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from app.gui.components import CandidateThumbnailLoader, ProcessingProgress, VideoPreview
from app.gui.models import DesktopProject, ProcessingSnapshot, ProjectRun
from app.gui.viewmodels import ProjectViewModel
from app.utils import format_seconds, read_json


_STATUS = {
    "new": "Новый проект", "source_ready": "Источник готов", "analyzing": "Анализируем",
    "analysis_ready": "Анализ готов", "reviewing_candidates": "Проверка кандидатов",
    "rendering_selected": "Создаём выбранные ролики", "partially_rendered": "Готово частично",
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
        self._active_candidate_id: str | None = None
        self._candidate_thumbnail_labels: dict[str, list[QLabel]] = {}
        self._thumbnail_loader = CandidateThumbnailLoader(self)
        self._thumbnail_loader.thumbnail_ready.connect(self._thumbnail_ready)
        self._thumbnail_loader.thumbnail_unavailable.connect(self._thumbnail_unavailable)
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
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_host = QWidget()
        left = QVBoxLayout(self.content_host)
        left.setContentsMargins(0, 0, 0, 0)
        self.preview = VideoPreview()
        left.addWidget(self.preview)
        self.metadata = self._card("Сведения о видео")
        left.addWidget(self.metadata)
        self.estimate = self._card("Предварительная оценка")
        left.addWidget(self.estimate)
        self.content_summary = self._card("Что найдено в видео")
        self._replace_card_text(self.content_summary, ["Рекомендация появится после завершения анализа."])
        left.addWidget(self.content_summary)
        self.candidate_detail = self._card("Просмотр момента")
        self._replace_card_text(self.candidate_detail, ["Выберите момент в списке, чтобы просмотреть исходный фрагмент."])
        left.addWidget(self.candidate_detail)
        self.candidate_review = self._card("Кандидаты и черновики")
        # The primary action scrolls directly to this workspace.  Make the
        # destination focusable as well, so keyboard focus follows the review
        # action instead of remaining on the header button.
        self.candidate_review.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.candidate_review_layout = self.candidate_review.layout()
        self._candidate_checks: dict[str, QCheckBox] = {}
        self._candidate_filter = "all"
        self._candidate_sort = "recommendation"
        self._replace_card_text(self.candidate_review, ["После анализа здесь появятся моменты для будущих роликов."])
        self.draft_button = QPushButton("Собрать черновик")
        self.draft_button.clicked.connect(self._draft_action)
        self.production_button = QPushButton("Создать выбранные ролики")
        self.production_button.setObjectName("primary")
        self.production_button.clicked.connect(self._confirm_production_render)
        self.candidate_review_layout.addWidget(self.draft_button)
        self.candidate_review_layout.addWidget(self.production_button)
        left.addWidget(self.candidate_review)
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
        self.history.setMinimumHeight(180)
        left.addWidget(self.history, 1)
        self.content_scroll.setWidget(self.content_host)
        body.addWidget(self.content_scroll, 3)
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
        settings.addWidget(QLabel("Аудио"))
        self.audio_mode = QComboBox()
        self.audio_mode.addItem("Исходная речь", "original")
        self.audio_mode.addItem("Исходная речь, улучшить звук", "original_enhanced")
        self.audio_mode.addItem("Озвучка", "voiceover")
        self.audio_mode.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(audio_mode=str(self.audio_mode.currentData()))
        )
        settings.addWidget(self.audio_mode)
        settings.addWidget(QLabel("Композиция кадра"))
        self.composition_strategy = QComboBox()
        self.composition_strategy.addItem("Авто: сохранить важное", "safe_auto")
        self.composition_strategy.addItem("По центру", "center_crop")
        self.composition_strategy.addItem("С размытым фоном", "fit_blur_background")
        self.composition_strategy.addItem("С однотонным фоном", "fit_solid_background")
        self.composition_strategy.addItem("Верхняя часть кадра", "top_crop")
        self.composition_strategy.currentIndexChanged.connect(
            lambda _index: self.viewmodel.save_options(composition_strategy=str(self.composition_strategy.currentData()))
        )
        settings.addWidget(self.composition_strategy)
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
        self.run_button = QPushButton("Анализировать видео")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._primary_action)
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
        review_ready = bool(project.analysis_artifact_path) and project.status in {
            "analysis_ready", "reviewing_candidates", "partially_rendered", "completed",
        }
        self.run_button.setText("Посмотреть найденные моменты" if review_ready else "Анализировать видео")
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
        self._set_combo_data(self.audio_mode, project.settings.audio_mode)
        self._set_combo_data(self.composition_strategy, project.settings.composition_strategy)
        self.subtitles.blockSignals(True); self.subtitles.setChecked(project.settings.subtitles_enabled); self.subtitles.blockSignals(False)
        self._set_combo_data(self.subtitle_style, project.settings.subtitle_style)
        self.cache.blockSignals(True); self.cache.setChecked(project.settings.use_cache); self.cache.blockSignals(False)
        self._update_candidate_review(project)
        self._refresh_active_candidate_detail(project)

    def _primary_action(self) -> None:
        if not self.project:
            return
        if self.project.analysis_artifact_path and self.project.status in {
            "analysis_ready", "reviewing_candidates", "partially_rendered", "completed",
        }:
            self.content_scroll.ensureWidgetVisible(self.candidate_review, 0, 16)
            self.candidate_review.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        self.viewmodel.start_analysis()

    def _update_candidate_review(self, project: DesktopProject) -> None:
        layout = self.candidate_review_layout
        while layout.count() > 1:
            item = layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._candidate_checks = {}
        self._candidate_thumbnail_labels = {}
        analysis_path = Path(project.analysis_artifact_path) if project.analysis_artifact_path else None
        analysis = read_json(analysis_path, {}) if analysis_path and analysis_path.is_file() else {}
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        previews: dict[str, dict] = {}
        for candidate_id, artifact_path in project.candidate_draft_artifacts.items():
            path = Path(artifact_path)
            draft = read_json(path, {}) if path.is_file() else {}
            if not isinstance(draft, dict):
                continue
            candidate = next(
                (item for item in draft.get("candidates", [])
                 if isinstance(item, dict) and str(item.get("candidate_id") or "") == candidate_id),
                None,
            )
            if isinstance(candidate, dict):
                previews[candidate_id] = candidate
        if not candidates:
            self._replace_card_text(self.candidate_review, ["После анализа здесь появятся моменты для будущих роликов."])
            layout.addWidget(self.draft_button); layout.addWidget(self.production_button)
            self.draft_button.setDisabled(True); self.production_button.setDisabled(True)
            return
        draftable_exists = any(
            project.candidate_states.get(candidate_id) not in {"draft_ready", "selected"}
            for candidate_id in project.review_selected_candidate_ids
        )
        potential_counts = {"high": 0, "medium": 0, "low": 0}
        for item in candidates:
            if isinstance(item, dict):
                level = str(item.get("potential") or "low")
                potential_counts[level if level in potential_counts else "low"] += 1
        recommended_count = sum(
            bool(item.get("recommended", item.get("selected_by_recommendation")))
            for item in candidates if isinstance(item, dict)
        )
        rendered_count = sum(state == "rendered" for state in project.candidate_states.values())
        processing_count = sum(state in {"draft_planning", "production_rendering"} for state in project.candidate_states.values())
        error_count = len({
            *[candidate_id for candidate_id, state in project.candidate_states.items() if state == "draft_failed"],
            *project.candidate_errors,
        })
        selection_toolbar = QFrame()
        toolbar_layout = QHBoxLayout(selection_toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        summary = QLabel(
            f"Найдено: {len(candidates)} · Рекомендуется: {recommended_count} · "
            f"Высокий: {potential_counts['high']} · Средний: {potential_counts['medium']} · "
            f"Выбрано: {len(project.review_selected_candidate_ids)} · Готово: {rendered_count} · "
            f"В процессе: {processing_count} · Ошибки: {error_count}"
        )
        summary.setWordWrap(True)
        toolbar_layout.addWidget(summary, 1)
        recommended_button = QPushButton("Выбрать рекомендованные")
        recommended_button.clicked.connect(self._select_recommended)
        clear_button = QPushButton("Снять выбор")
        clear_button.clicked.connect(self._clear_review_selection)
        toolbar_layout.addWidget(recommended_button)
        toolbar_layout.addWidget(clear_button)
        layout.addWidget(selection_toolbar)
        filter_toolbar = QFrame()
        filter_layout = QHBoxLayout(filter_toolbar)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("Показать"))
        filter_box = QComboBox()
        for label, value in (
            ("Все", "all"), ("Рекомендуемые", "recommended"), ("Высокий потенциал", "high"),
            ("Средний потенциал", "medium"), ("Не выбраны", "unselected"),
        ):
            filter_box.addItem(label, value)
        filter_box.setCurrentIndex(max(0, filter_box.findData(self._candidate_filter)))
        filter_box.currentIndexChanged.connect(lambda _index, box=filter_box: self._change_candidate_filter(str(box.currentData())))
        filter_layout.addWidget(filter_box)
        filter_layout.addWidget(QLabel("Сортировка"))
        sort_box = QComboBox()
        for label, value in (("По рекомендации", "recommendation"), ("По времени", "time"), ("По потенциалу", "potential")):
            sort_box.addItem(label, value)
        sort_box.setCurrentIndex(max(0, sort_box.findData(self._candidate_sort)))
        sort_box.currentIndexChanged.connect(lambda _index, box=sort_box: self._change_candidate_sort(str(box.currentData())))
        filter_layout.addWidget(sort_box)
        filter_layout.addStretch()
        layout.addWidget(filter_toolbar)
        visible_candidates = self._filtered_candidates(candidates, project)
        for item in visible_candidates:
            if not isinstance(item, dict) or not item.get("candidate_id"):
                continue
            candidate_id = str(item["candidate_id"])
            state = project.candidate_states.get(candidate_id, str(item.get("recommendation_status") or "analyzed"))
            override = project.candidate_boundary_overrides.get(candidate_id, {})
            start_value = override.get("start", item.get("start")) if isinstance(override, dict) else item.get("start")
            end_value = override.get("end", item.get("end")) if isinstance(override, dict) else item.get("end")
            start, end = format_seconds(start_value), format_seconds(end_value)
            potential = {"high": "Высокий потенциал", "medium": "Средний потенциал", "low": "Низкий потенциал"}.get(
                str(item.get("potential") or "low"), "Предварительная оценка",
            )
            confidence = float(item.get("confidence") or 0)
            status_label = {
                "analyzed": "готов к просмотру", "draft_planning": "собираем черновик",
                "draft_ready": "черновик готов", "draft_failed": "черновик не готов",
                "selected": "подтверждён", "production_rendering": "создаём ролик", "rendered": "готово",
            }.get(state, "готов к просмотру")
            frame = QFrame(); frame.setObjectName("card")
            row = QHBoxLayout(frame); row.setContentsMargins(10, 8, 10, 8)
            thumbnail = QLabel("Кадр\nзагружается")
            thumbnail.setObjectName("muted")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setFixedSize(112, 64)
            thumbnail.setStyleSheet("border: 1px solid #303640; border-radius: 4px;")
            row.addWidget(thumbnail)
            self._candidate_thumbnail_labels.setdefault(candidate_id, []).append(thumbnail)
            try:
                start_seconds = float(start_value)
                end_seconds = float(end_value)
            except (TypeError, ValueError):
                start_seconds, end_seconds = 0.0, 0.0
            preview_contract = item.get("preview") if isinstance(item.get("preview"), dict) else {}
            thumbnail_contract = preview_contract.get("thumbnail") if isinstance(preview_contract.get("thumbnail"), dict) else {}
            thumbnail_time = thumbnail_contract.get("timestamp_seconds", start_seconds + max(0.0, min(1.0, (end_seconds - start_seconds) / 2)))
            try:
                thumbnail_seconds = float(thumbnail_time)
            except (TypeError, ValueError):
                thumbnail_seconds = start_seconds
            if project.source.is_file():
                self._thumbnail_loader.request(
                    cache_directory=project.directory / "candidate-thumbnails",
                    analysis_id=project.analysis_id or "analysis",
                    candidate_id=candidate_id,
                    source_path=project.source,
                    timestamp_seconds=thumbnail_seconds,
                )
            information = QVBoxLayout()
            title = QLabel(str(item.get("title") or item.get("core_idea") or "Момент из видео"))
            title.setStyleSheet("font-weight: 600;")
            title.setWordWrap(True)
            information.addWidget(title)
            recommended = " · рекомендуем" if item.get("recommended", item.get("selected_by_recommendation")) else ""
            check = QCheckBox(f"{start}–{end} · {potential} · уверенность {confidence * 100:.0f}%{recommended}")
            check.setToolTip(f"{status_label}. {item.get('title') or item.get('text') or item.get('reason') or ''}")
            check.setChecked(candidate_id in project.review_selected_candidate_ids)
            check.toggled.connect(lambda _checked=False: self._persist_review_selection())
            self._candidate_checks[candidate_id] = check
            information.addWidget(check)
            reasons = [str(value) for value in item.get("reasons", []) if str(value)]
            if reasons:
                reason = QLabel("Почему: " + reasons[0])
                reason.setObjectName("muted")
                reason.setWordWrap(True)
                information.addWidget(reason)
            row.addLayout(information, 1)
            actions = QVBoxLayout()
            status = QLabel(status_label)
            status.setObjectName("muted")
            candidate_error = project.candidate_errors.get(candidate_id)
            if candidate_error:
                status.setText("есть ошибка")
                status.setToolTip(candidate_error)
            status.setWordWrap(True)
            actions.addWidget(status)
            source_preview = QPushButton("Просмотреть")
            source_preview.clicked.connect(lambda _checked=False, value=dict(item): self._preview_candidate(value))
            actions.addWidget(source_preview)
            preview = previews.get(candidate_id, {}).get("preview", {}) if isinstance(previews.get(candidate_id), dict) else {}
            preview_file = Path(str(preview.get("output_file") or "")) if isinstance(preview, dict) else None
            if preview_file and preview_file.is_file():
                button = QPushButton("Смотреть черновик")
                button.clicked.connect(lambda _checked=False, path=preview_file: self.preview.set_file(str(path)))
                actions.addWidget(button)
            row.addLayout(actions)
            layout.addWidget(frame)
        selected_count = len(project.review_selected_candidate_ids)
        self.draft_button.setText(
            f"Собрать {selected_count} черновик(а)" if draftable_exists else f"Подтвердить {selected_count} черновик(а)"
        )
        self.draft_button.setDisabled(not bool(project.review_selected_candidate_ids))
        selected_drafts_exist = all(
            Path(project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
            for candidate_id in project.selected_candidate_ids
        )
        self.production_button.setText(f"Создать {len(project.selected_candidate_ids)} выбранных ролика(ов)")
        self.production_button.setDisabled(not bool(project.selected_candidate_ids) or not selected_drafts_exist)
        layout.addWidget(self.draft_button); layout.addWidget(self.production_button)

    def _checked_candidate_ids(self) -> list[str]:
        return [candidate_id for candidate_id, checkbox in self._candidate_checks.items() if checkbox.isChecked()]

    def _draft_action(self) -> None:
        if not self.project:
            return
        self._persist_review_selection()
        candidate_ids = list(self.project.review_selected_candidate_ids)
        if not candidate_ids:
            return
        needs_draft = [
            candidate_id for candidate_id in candidate_ids
            if self.project.candidate_states.get(candidate_id) not in {"draft_ready", "selected"}
        ]
        if needs_draft:
            self.viewmodel.build_drafts(needs_draft)
        else:
            self.viewmodel.select_drafts(candidate_ids)

    def _persist_review_selection(self) -> None:
        if self.project:
            # Filters may hide already selected cards.  Update only the
            # visible subset so a click never silently discards a selection
            # made in another filter view.
            selected = set(self.project.review_selected_candidate_ids)
            for candidate_id, checkbox in self._candidate_checks.items():
                if checkbox.isChecked():
                    selected.add(candidate_id)
                else:
                    selected.discard(candidate_id)
            self.viewmodel.set_review_selection([
                candidate_id for candidate_id in self.project.review_selected_candidate_ids
                if candidate_id in selected
            ] + [candidate_id for candidate_id in self._candidate_checks if candidate_id in selected and candidate_id not in self.project.review_selected_candidate_ids])

    def _select_recommended(self) -> None:
        if not self.project:
            return
        path = Path(self.project.analysis_artifact_path) if self.project.analysis_artifact_path else None
        analysis = read_json(path, {}) if path and path.is_file() else {}
        candidate_ids = [
            str(item.get("candidate_id")) for item in analysis.get("candidates", [])
            if isinstance(item, dict) and item.get("recommended", item.get("selected_by_recommendation")) and item.get("candidate_id")
        ] if isinstance(analysis, dict) else []
        self.viewmodel.set_review_selection(candidate_ids)

    def _clear_review_selection(self) -> None:
        self.viewmodel.set_review_selection([])

    def _change_candidate_filter(self, value: str) -> None:
        self._candidate_filter = value
        if self.project:
            self._update_candidate_review(self.project)

    def _change_candidate_sort(self, value: str) -> None:
        self._candidate_sort = value
        if self.project:
            self._update_candidate_review(self.project)

    def _filtered_candidates(self, candidates: list[object], project: DesktopProject) -> list[dict]:
        values = [dict(item) for item in candidates if isinstance(item, dict) and item.get("candidate_id")]
        if self._candidate_filter == "recommended":
            values = [item for item in values if item.get("recommended", item.get("selected_by_recommendation"))]
        elif self._candidate_filter in {"high", "medium"}:
            values = [item for item in values if item.get("potential") == self._candidate_filter]
        elif self._candidate_filter == "unselected":
            values = [item for item in values if item.get("candidate_id") not in project.review_selected_candidate_ids]
        potential_rank = {"high": 2, "medium": 1, "low": 0}
        if self._candidate_sort == "time":
            values.sort(key=lambda item: float(item.get("start_seconds", item.get("start", 0)) or 0))
        elif self._candidate_sort == "potential":
            values.sort(key=lambda item: (
                potential_rank.get(str(item.get("potential")), 0),
                float(item.get("confidence") or 0), float(item.get("score") or 0),
            ), reverse=True)
        else:
            values.sort(key=lambda item: (
                bool(item.get("recommended", item.get("selected_by_recommendation"))),
                potential_rank.get(str(item.get("potential")), 0),
                float(item.get("score") or 0),
            ), reverse=True)
        return values

    def _preview_candidate(self, candidate: dict) -> None:
        if not self.project:
            return
        try:
            start, end = self._candidate_range(candidate)
        except (TypeError, ValueError):
            return
        self.preview.set_range(self.project.source, start, end)
        self._active_candidate_id = str(candidate.get("candidate_id") or "") or None
        self._show_candidate_detail(candidate, start, end)

    def _candidate_range(self, candidate: dict) -> tuple[float, float]:
        candidate_id = str(candidate.get("candidate_id") or "")
        override = self.project.candidate_boundary_overrides.get(candidate_id, {}) if self.project else {}
        start = float(override.get("start", candidate.get("start_seconds", candidate.get("start", 0))))
        end = float(override.get("end", candidate.get("end_seconds", candidate.get("end", start))))
        return start, max(start, end)

    def _show_candidate_detail(self, candidate: dict, start: float, end: float) -> None:
        if not self.project:
            return
        potential = {"high": "Высокий", "medium": "Средний", "low": "Низкий"}.get(str(candidate.get("potential")), "Предварительный")
        reasons = [str(item) for item in candidate.get("reasons", []) if str(item)]
        risks = [str(item) for item in candidate.get("risks", []) if str(item)]
        lines = [
            str(candidate.get("title") or "Момент"),
            f"{format_seconds(start)}–{format_seconds(end)} · {format_seconds(end - start)}",
            f"Потенциал: {potential} · Уверенность: {float(candidate.get('confidence') or 0) * 100:.0f}%",
            f"Идея: {candidate.get('core_idea') or candidate.get('summary') or '—'}",
            f"Начало: {candidate.get('hook_summary') or '—'}",
            f"Финал: {candidate.get('payoff_summary') or '—'}",
            "Оценки: " + " · ".join((
                f"удержание {self._score_text(candidate.get('retention_score'))}",
                f"публикация {self._score_text(candidate.get('publishability_score'))}",
                f"потенциал {self._score_text(candidate.get('viral_score'))}",
            )),
        ]
        excerpt = str(candidate.get("transcript_excerpt") or candidate.get("text") or "").strip()
        if excerpt:
            lines.append("Текст: " + excerpt)
        if reasons:
            lines.append("Почему рекомендуем: " + " ".join(reasons[:2]))
        if risks:
            lines.append("Риск: " + " ".join(risks[:2]))
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id and self.project.candidate_errors.get(candidate_id):
            lines.append("Причина ошибки: " + self.project.candidate_errors[candidate_id])
        self._replace_card_text(self.candidate_detail, lines)
        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 4, 0, 0)
        for index, (text, boundary, delta) in enumerate((
            ("Начало −1 с", "start", -1.0), ("Начало −0.5 с", "start", -0.5),
            ("Начало +0.5 с", "start", 0.5), ("Начало +1 с", "start", 1.0),
            ("Конец −1 с", "end", -1.0), ("Конец −0.5 с", "end", -0.5),
            ("Конец +0.5 с", "end", 0.5), ("Конец +1 с", "end", 1.0),
        )):
            button = QPushButton(text)
            button.setToolTip("Проверит только сохранённые границы речи и сцены; повторный анализ не нужен.")
            button.clicked.connect(
                lambda _checked=False, cid=candidate_id, name=boundary, value=delta: self._adjust_candidate_boundary(cid, name, value)
            )
            grid.addWidget(button, index // 4, index % 4)
        self.candidate_detail.layout().addWidget(controls)

    @staticmethod
    def _score_text(value: object) -> str:
        try:
            return f"{float(value):.0f}/100"
        except (TypeError, ValueError):
            return "—"

    def _adjust_candidate_boundary(self, candidate_id: str, boundary: str, delta_seconds: float) -> None:
        if candidate_id:
            self.viewmodel.adjust_candidate_boundary(candidate_id, boundary, delta_seconds)

    def _refresh_active_candidate_detail(self, project: DesktopProject) -> None:
        if not self._active_candidate_id or not project.analysis_artifact_path:
            return
        analysis = read_json(Path(project.analysis_artifact_path), {})
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        candidate = next(
            (item for item in candidates if isinstance(item, dict) and item.get("candidate_id") == self._active_candidate_id),
            None,
        )
        if isinstance(candidate, dict):
            try:
                start, end = self._candidate_range(candidate)
            except (TypeError, ValueError):
                return
            self._show_candidate_detail(candidate, start, end)

    def _thumbnail_ready(self, candidate_id: str, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        for label in self._candidate_thumbnail_labels.get(candidate_id, []):
            try:
                label.setText("")
                label.setPixmap(pixmap.scaled(
                    label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
                ))
            except RuntimeError:
                continue

    def _thumbnail_unavailable(self, candidate_id: str) -> None:
        for label in self._candidate_thumbnail_labels.get(candidate_id, []):
            try:
                label.setText("Кадр\nнедоступен")
            except RuntimeError:
                continue

    def _confirm_production_render(self) -> None:
        if not self.project or not self.project.selected_candidate_ids:
            return
        answer = QMessageBox.question(
            self, "Создать итоговые ролики",
            "Запустить тяжёлый production render 1080×1920 только для подтверждённых черновиков?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.viewmodel.render_selected()

    def _runs_changed(self, runs: list[ProjectRun]) -> None:
        self.runs = runs
        self._update_content_summary(runs)
        while self.history_layout.count() > 1:
            item = self.history_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for run in runs:
            frame = QFrame(); frame.setObjectName("card")
            layout = QHBoxLayout(frame); layout.setContentsMargins(12, 10, 12, 10)
            kind = {
                "analysis": "Анализ видео", "draft": "Подготовка черновика",
                "selected_render": "Создание роликов", "render_revision": "Создать заново",
                "full": "Полный запуск",
            }.get(run.run_kind, "Запуск")
            text = QLabel(f"{run.started_at[:16].replace('T', ' ')} · {kind} · {run.status.replace('_', ' ')}")
            text.setWordWrap(True)
            layout.addWidget(text, 1)
            results = [Path(item) for item in run.artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()]
            for index, result in enumerate(results, start=1):
                button = QPushButton("Открыть ролик" if len(results) == 1 else f"Ролик {index}")
                button.clicked.connect(lambda _, path=result: self._open_file(path))
                layout.addWidget(button)
            if run.status in {"completed", "completed_with_warnings"}:
                rerender = QPushButton("Создать заново")
                rerender.setToolTip("Повторно экспортировать с текущими стилем, платформой и качеством без нового AI-анализа.")
                rerender.clicked.connect(lambda _, parent_run=run: self.viewmodel.rerender(parent_run))
                layout.addWidget(rerender)
            folder = QPushButton("Папка")
            folder.clicked.connect(lambda _, path=Path(run.log_path).parent if run.log_path else None: self._open_folder(path))
            layout.addWidget(folder)
            self.history_layout.insertWidget(self.history_layout.count() - 1, frame)
        if runs:
            result = next((Path(item) for item in runs[0].artifact_paths if Path(item).suffix.lower() == ".mp4" and Path(item).is_file()), None)
            if result: self.preview.set_file(result)

    def _update_content_summary(self, runs: list[ProjectRun]) -> None:
        for run in runs:
            if not run.report_path:
                continue
            report = read_json(Path(run.report_path), {})
            understanding = report.get("content_understanding", {}) if isinstance(report, dict) else {}
            if not isinstance(understanding, dict) or not understanding.get("enabled"):
                continue
            profile = understanding.get("profile", {})
            content_map = understanding.get("content_map", {})
            recommendation = understanding.get("clip_count_recommendation", {})
            coverage = understanding.get("coverage_map", understanding.get("coverage", {}))
            if not all(isinstance(item, dict) for item in (profile, content_map, recommendation, coverage)):
                continue
            clip_range = recommendation.get("estimated_publishable_clip_range", {})
            lower = clip_range.get("min", "—")
            upper = clip_range.get("max", "—")
            selected_chapters = coverage.get("selected_chapters", [])
            coverage_status = (
                "Подборка охватывает разные части видео."
                if isinstance(selected_chapters, list) and len(selected_chapters) > 1
                else "Подборка покрывает найденные самостоятельные фрагменты."
            )
            lines = [
                f"Тип: {profile.get('detected_content_type', 'не определён')}",
                f"Смысловых частей: {len(content_map.get('chapters', []))}",
                f"Самостоятельных историй: {recommendation.get('estimated_story_count', understanding.get('story_unit_count', 0))}",
                f"Рекомендуем создать: {lower}–{upper} ролика(ов)",
                coverage_status,
            ]
            lines.extend(self._virality_summary_lines(report))
            self._replace_card_text(self.content_summary, lines)
            return
        self._replace_card_text(self.content_summary, ["Рекомендация появится после завершения анализа."])

    @staticmethod
    def _virality_summary_lines(report: dict) -> list[str]:
        """Keep Goal 5B reasons short and human-readable; never expose its formula."""

        virality = report.get("virality", {}) if isinstance(report, dict) else {}
        intelligence = report.get("clip_intelligence", {}) if isinstance(report, dict) else {}
        candidates = intelligence.get("candidates", []) if isinstance(intelligence, dict) else []
        if not isinstance(virality, dict) or not virality.get("enabled") or not isinstance(candidates, list):
            return []
        chosen = next(
            (item for item in candidates if isinstance(item, dict) and item.get("selected") and isinstance(item.get("virality"), dict)),
            next((item for item in candidates if isinstance(item, dict) and isinstance(item.get("virality"), dict)), None),
        )
        if not isinstance(chosen, dict):
            return []
        details = chosen.get("virality", {})
        potential = details.get("viral_potential", {}) if isinstance(details, dict) else {}
        publishability = details.get("publishability", {}) if isinstance(details, dict) else {}
        retention = details.get("retention_profile", {}) if isinstance(details, dict) else {}
        if not all(isinstance(item, dict) for item in (potential, publishability, retention)):
            return []
        level = {
            "weak": "Низкий", "moderate": "Средний", "strong": "Высокий", "excellent": "Очень высокий",
        }.get(str(potential.get("level")), "Предварительный")
        lines = [f"Потенциал: {level}"]
        factors = potential.get("strongest_factors", [])
        if isinstance(factors, list):
            labels = {
                "hook": "Сильное начало", "curiosity": "Интрига раскрывается", "emotion": "Эмоциональная развязка",
                "payoff": "Самостоятельный вывод", "retention": "Хороший шанс удержания",
                "publishability": "Готов к публикации", "quotability": "Запоминающаяся фраза",
                "usefulness": "Практическая ценность", "momentum": "Мысль развивается",
            }
            for factor in factors:
                label = labels.get(str(factor))
                if label and label not in lines:
                    lines.append(label)
                if len(lines) >= 4:
                    break
        eligibility = details.get("eligibility", {}) if isinstance(details, dict) else {}
        status = eligibility.get("status") if isinstance(eligibility, dict) else ""
        if status == "publishable_now" and "Готов к публикации" not in lines:
            lines.append("Готов к публикации")
        elif status in {"needs_reconstruction", "publishable_with_minor_adjustment"}:
            lines.append("Лучше доработать перед публикацией")
        confidence = potential.get("confidence", {}) if isinstance(potential, dict) else {}
        if isinstance(confidence, dict) and confidence.get("warnings"):
            lines.append("Предварительная оценка: недостаточно визуальных данных")
        return lines[:5]

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
        self.draft_button.setDisabled(active or not (self.project and self.project.review_selected_candidate_ids))
        selected_drafts_exist = bool(self.project and self.project.selected_candidate_ids) and all(
            Path(self.project.candidate_draft_artifacts.get(candidate_id, "")).is_file()
            for candidate_id in self.project.selected_candidate_ids
        ) if self.project else False
        self.production_button.setDisabled(active or not selected_drafts_exist)
        for widget in (
            self.processing_mode, self.deep_analysis, self.platform, self.clip_count,
            self.audio_mode, self.composition_strategy, self.subtitles, self.subtitle_style, self.cache,
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
