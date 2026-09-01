from __future__ import annotations

from PySide6.QtCore import QByteArray, QPoint, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.doctor import DoctorReadiness, summarize_checks
from app.gui.screens import OnboardingDialog, ProjectScreen, ProjectsScreen, SettingsScreen
from app.gui.services.desktop_services import DesktopServices
from app.gui.viewmodels import ProjectViewModel, ProjectsViewModel, SettingsViewModel


class MainWindow(QMainWindow):
    """The durable desktop shell; project flow itself stays in the existing screens."""

    def __init__(self, services: DesktopServices) -> None:
        super().__init__()
        self.services = services
        self.projects_viewmodel = ProjectsViewModel(services, self)
        self.project_viewmodel = ProjectViewModel(services, self)
        self.settings_viewmodel = SettingsViewModel(services, self)
        self._onboarding_dialog: OnboardingDialog | None = None
        self._screen_signal_connected = False
        self.setWindowTitle("Content Factory")
        # Leave room for the native title bar at 1280×720 / 150%.  Every page
        # owns vertical scrolling and sticky actions, so the shell itself does
        # not need to claim the entire logical screen.
        self.setMinimumSize(720, 380)
        self.resize(1320, 840)
        if services.settings.window_geometry:
            geometry = QByteArray.fromBase64(services.settings.window_geometry.encode("ascii", errors="ignore"))
            if not geometry.isEmpty():
                self.restoreGeometry(geometry)

        shell = QWidget()
        shell.setObjectName("appShell")
        self.setCentralWidget(shell)
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(224)
        nav = QVBoxLayout(self.sidebar)
        nav.setContentsMargins(16, 20, 16, 18)
        nav.setSpacing(7)

        brand = QFrame()
        self.brand = brand
        brand.setObjectName("brand")
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(2, 0, 2, 10)
        brand_layout.setSpacing(0)
        self.brand_content = QLabel("CONTENT")
        self.brand_content.setObjectName("brandContent")
        self.brand_content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.brand_content.setToolTip("CONTENT FACTORY")
        self.brand_factory = QLabel("FACTORY")
        self.brand_factory.setObjectName("brandFactory")
        self.brand_factory.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.brand_factory.setToolTip("CONTENT FACTORY")
        brand_layout.addWidget(self.brand_content)
        brand_layout.addWidget(self.brand_factory)
        nav.addWidget(brand)
        nav.addSpacing(18)

        self.new_button = self._nav_button("＋  Новый проект")
        self.new_button.setObjectName("navNewProject")
        self.new_button.setMinimumHeight(38)
        self.projects_button = self._nav_button("▢  Проекты")
        self.settings_button = self._nav_button("⚙  Настройки")
        self.projects_button.clicked.connect(self.show_projects)
        self.new_button.clicked.connect(self._new_project)
        self.settings_button.clicked.connect(self.show_settings)
        nav.addWidget(self.new_button)
        nav.addSpacing(7)
        nav.addWidget(self.projects_button)
        nav.addWidget(self.settings_button)
        nav.addStretch()

        self.system_status = QFrame()
        self.system_status.setObjectName("card")
        status_layout = QVBoxLayout(self.system_status)
        status_layout.setContentsMargins(12, 11, 12, 11)
        status_layout.setSpacing(4)
        self.system_status_title = QLabel("●  Проверка…")
        self.system_status_title.setObjectName("systemStatusTitle")
        self.system_status_title.setWordWrap(True)
        self.system_status_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.system_status_title.setToolTip("Проверяется состояние системы")
        self.system_status_detail = QLabel("Диагностика запускается в фоне")
        self.system_status_detail.setObjectName("muted")
        self.system_status_detail.setWordWrap(True)
        self.system_status_detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        status_layout.addWidget(self.system_status_title)
        status_layout.addWidget(self.system_status_detail)
        nav.addWidget(self.system_status)

        self.help_button = QPushButton("?  Помощь и поддержка")
        self.help_button.setObjectName("nav")
        self.help_button.clicked.connect(self._open_support)
        nav.addWidget(self.help_button)
        self.version = QLabel("Локальная версия")
        self.version.setObjectName("muted")
        self.version.setWordWrap(True)
        self.version.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.version.setToolTip("Локальная версия")
        nav.addWidget(self.version)

        # A narrow shell should not truncate a primary navigation action.
        # Retain the descriptive name as a tooltip and use progressively
        # shorter labels before Windows scaling can squeeze button contents.
        self._sidebar_full_labels = {
            self.new_button: "＋  Новый проект",
            self.projects_button: "▣  Проекты",
            self.settings_button: "⚙  Настройки",
            self.help_button: "?  Помощь и поддержка",
        }
        for button, label in self._sidebar_full_labels.items():
            button.setToolTip(label)

        layout.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        self.stack.setObjectName("contentStack")
        self.projects_screen = ProjectsScreen(self.projects_viewmodel)
        self.project_screen = ProjectScreen(self.project_viewmodel)
        self.settings_screen = SettingsScreen(self.settings_viewmodel)
        self.projects_index = self.stack.addWidget(self.projects_screen)
        self.project_index = self.stack.addWidget(self.project_screen)
        self.settings_index = self.stack.addWidget(self.settings_screen)
        self.projects_screen.project_opened.connect(self.show_project)
        self.project_screen.back_requested.connect(self.show_projects)
        self.project_viewmodel.project_persisted.connect(lambda _project_id: self.projects_screen.mark_dirty())
        self.settings_viewmodel.diagnostics_started.connect(self._diagnostics_started)
        self.settings_viewmodel.diagnostics_ready.connect(self._diagnostics_ready)
        layout.addWidget(self.stack, 1)
        self._restore_last_screen()
        self._apply_sidebar_layout()
        QTimer.singleShot(0, self._fit_to_available_screen)
        QTimer.singleShot(0, self._maybe_onboard)
        if self.services.settings.onboarding_completed:
            QTimer.singleShot(0, self.settings_viewmodel.diagnostics)

    def show_projects(self, *, remember: bool = True) -> None:
        self.stack.setCurrentIndex(self.projects_index)
        self._set_selected(self.projects_button)
        self.projects_screen.refresh_if_dirty()
        if remember:
            self.services.settings.last_screen = "projects"

    def show_project(self, project, *, remember: bool = True) -> None:
        self.project_screen.open(project)
        self.stack.setCurrentIndex(self.project_index)
        self._set_selected(None)
        if remember:
            self.services.settings.last_screen = "project"
            self.services.settings.last_open_project_id = project.project_id

    def show_settings(self) -> None:
        self.stack.setCurrentIndex(self.settings_index)
        self._set_selected(self.settings_button)
        self.services.settings.last_screen = "settings"

    def _new_project(self) -> None:
        # ProjectsScreen is the source onboarding workspace.  Do not open a
        # file dialog automatically: it would skip the source choice and make
        # a public URL harder to discover.
        self.show_projects()
        self._set_selected(self.new_button)
        self.projects_screen.focus_source()

    def _restore_last_screen(self) -> None:
        if self.services.settings.last_screen == "project" and self.services.settings.last_open_project_id:
            try:
                project = self.services.projects.load(self.services.settings.last_open_project_id)
            except Exception:
                self.show_projects(remember=False)
                return
            self.show_project(project, remember=False)
            return
        if self.services.settings.last_screen == "settings":
            self.stack.setCurrentIndex(self.settings_index)
            self._set_selected(self.settings_button)
            return
        self.show_projects(remember=False)

    def _maybe_onboard(self) -> None:
        if self.services.settings.onboarding_completed:
            return
        # This callback may be queued more than once while a window is being
        # restored.  Keep one parented dialog instead of stacking modal
        # onboarding windows behind each other.
        if self._onboarding_dialog is not None:
            try:
                if self._onboarding_dialog.isVisible():
                    return
            except RuntimeError:
                self._onboarding_dialog = None
        dialog = OnboardingDialog(self.settings_viewmodel, self)
        self._onboarding_dialog = dialog
        try:
            result = dialog.exec()
        finally:
            self._onboarding_dialog = None
        if result == QDialog.DialogCode.Rejected and not self.services.settings.onboarding_completed:
            self.close()
            return
        self.show_projects()

    def _set_selected(self, button: QPushButton | None) -> None:
        for item in (self.projects_button, self.new_button, self.settings_button):
            item.setChecked(item is button)

    @staticmethod
    def _open_support() -> None:
        QDesktopServices.openUrl(QUrl("https://t.me/rezvis"))

    def _diagnostics_started(self) -> None:
        self.system_status_title.setText("●  Проверка…")
        self.system_status_title.setToolTip("Проверяется состояние системы")
        self.system_status_detail.setText("Диагностика выполняется в фоне")

    def _diagnostics_ready(self, checks) -> None:
        summary = summarize_checks(checks)
        prefix = {
            DoctorReadiness.READY: "●  ",
            DoctorReadiness.LIMITED: "▲  ",
            DoctorReadiness.SETUP_REQUIRED: "■  ",
        }[summary.readiness]
        self.system_status_title.setText(prefix + summary.title)
        actionable = next((item for item in checks if item.blocking), None)
        if actionable is None:
            actionable = next((item for item in checks if item.warning), None)
        if actionable is not None:
            detail = f"{actionable.label}: {actionable.action}"
            tooltip = f"{actionable.detail}\n{actionable.action}"
        else:
            detail = "Система готова к обработке видео"
            tooltip = summary.detail
        self.system_status_title.setToolTip(tooltip)
        self.system_status_detail.setText(detail)
        self.system_status_detail.setToolTip(tooltip)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Keep a useful work area on compact laptops while retaining a fixed,
        # recognisable creator-tool sidebar at normal desktop widths.
        if not hasattr(self, "sidebar"):
            return
        self._apply_sidebar_layout()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        handle = self.windowHandle()
        if handle is not None and not self._screen_signal_connected:
            handle.screenChanged.connect(lambda _screen: QTimer.singleShot(0, self._fit_to_available_screen))
            self._screen_signal_connected = True
        QTimer.singleShot(0, self._fit_to_available_screen)

    def _fit_to_available_screen(self) -> None:
        """Clamp restored/default geometry to the current logical work area."""

        if self.isMaximized() or self.isFullScreen():
            return
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame_extra_width = max(0, frame.width() - self.width())
        frame_extra_height = max(0, frame.height() - self.height())
        maximum_width = max(1, available.width() - frame_extra_width)
        maximum_height = max(1, available.height() - frame_extra_height)
        target_width = min(self.width(), maximum_width)
        target_height = min(self.height(), maximum_height)
        if maximum_width >= self.minimumWidth():
            target_width = max(self.minimumWidth(), target_width)
        if maximum_height >= self.minimumHeight():
            target_height = max(self.minimumHeight(), target_height)
        if (target_width, target_height) != (self.width(), self.height()):
            self.resize(target_width, target_height)
            frame = self.frameGeometry()

        maximum_x = available.right() - frame.width() + 1
        maximum_y = available.bottom() - frame.height() + 1
        target_x = min(max(frame.x(), available.left()), max(available.left(), maximum_x))
        target_y = min(max(frame.y(), available.top()), max(available.top(), maximum_y))
        if (target_x, target_y) != (frame.x(), frame.y()):
            self.move(self.pos() + QPoint(target_x - frame.x(), target_y - frame.y()))

    def _apply_sidebar_layout(self) -> None:
        """Use compact, tooltip-backed navigation before labels can clip."""

        if not hasattr(self, "sidebar"):
            return
        width = self.width()
        profile = "compact" if width < 920 else "medium" if width < 1120 else "wide"
        # Compact and medium both retain the established 156 px rail.  At the
        # 760 px shell minimum this leaves the Project workspace at its tested
        # 604 px compact breakpoint; icon-only labels keep the rail itself safe.
        target_width = 156 if profile != "wide" else 224
        if self.sidebar.width() != target_width:
            self.sidebar.setFixedWidth(target_width)
        compact_brand = profile != "wide"
        for brand_label in (self.brand_content, self.brand_factory):
            if brand_label.property("compactBrand") != compact_brand:
                brand_label.setProperty("compactBrand", compact_brand)
                brand_label.style().unpolish(brand_label)
                brand_label.style().polish(brand_label)
        self.brand_content.setText("CF" if compact_brand else "CONTENT")
        self.brand_factory.setVisible(not compact_brand)
        if profile != "wide":
            labels = {
                self.new_button: "＋",
                self.projects_button: "▣",
                self.settings_button: "⚙",
                self.help_button: "?",
            }
        else:
            labels = {
                self.new_button: "＋  Новый проект",
                self.projects_button: "▣  Проекты",
                self.settings_button: "⚙  Настройки",
                self.help_button: "?  Помощь и поддержка",
            }
        for button, button_text in labels.items():
            if button.text() != button_text:
                button.setText(button_text)
        show_wide_footer = profile == "wide" and self.height() >= 480
        self.system_status.setVisible(show_wide_footer)
        self.version.setVisible(show_wide_footer)

    @staticmethod
    def _nav_button(text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("nav")
        button.setCheckable(True)
        return button

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.project_viewmodel.active:
            owner = self.project_viewmodel.active_project_name or "открытом проекте"
            answer = QMessageBox.question(
                self,
                "Закрыть приложение?",
                f"Сейчас идёт работа в проекте «{owner}». При закрытии запуск будет отмечен как прерванный и его можно будет начать снова.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.project_viewmodel.cancel()
        self.services.settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self.services.save_settings()
        event.accept()
