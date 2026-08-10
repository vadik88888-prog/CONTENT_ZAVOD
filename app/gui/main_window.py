from __future__ import annotations

from PySide6.QtCore import QByteArray, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
        self.setWindowTitle("Content Factory")
        # A smaller, practical lower bound lets Windows use the application at
        # 1280×720 and elevated scaling without making the shell itself clip.
        self.setMinimumSize(760, 480)
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
        brand.setObjectName("brand")
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(2, 0, 2, 10)
        brand_layout.setSpacing(0)
        self.brand_content = QLabel("CONTENT")
        self.brand_content.setStyleSheet(
            "font-size: 20px; font-weight: 700; letter-spacing: 1px; color: #F4F6F8;"
        )
        self.brand_factory = QLabel("FACTORY")
        self.brand_factory.setStyleSheet(
            "font-size: 20px; font-weight: 700; letter-spacing: 1px; color: #101216; "
            "background: #FF7900; padding: 0 4px;"
        )
        self.brand_factory.setMaximumWidth(108)
        brand_layout.addWidget(self.brand_content)
        brand_layout.addWidget(self.brand_factory)
        nav.addWidget(brand)
        nav.addSpacing(18)

        self.new_button = self._nav_button("＋  Новый проект")
        self.new_button.setObjectName("primary")
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
        status_title = QLabel("●  Система готова")
        status_title.setStyleSheet("font-weight: 600; color: #DCE4DD;")
        status_detail = QLabel("Локальная обработка\nВсе проекты остаются здесь")
        status_detail.setObjectName("muted")
        status_detail.setWordWrap(True)
        status_layout.addWidget(status_title)
        status_layout.addWidget(status_detail)
        nav.addWidget(self.system_status)

        self.help_button = QPushButton("?  Помощь и поддержка")
        self.help_button.setObjectName("nav")
        self.help_button.clicked.connect(self._show_help)
        nav.addWidget(self.help_button)
        self.version = QLabel("Локальная версия")
        self.version.setObjectName("muted")
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
        self.project_viewmodel.project_persisted.connect(lambda _project_id: self.projects_screen.refresh())
        layout.addWidget(self.stack, 1)
        self._restore_last_screen()
        self._apply_sidebar_layout()
        QTimer.singleShot(0, self._maybe_onboard)

    def show_projects(self, *, remember: bool = True) -> None:
        self.stack.setCurrentIndex(self.projects_index)
        self._set_selected(self.projects_button)
        self.projects_screen.refresh()
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
            dialog.exec()
        finally:
            self._onboarding_dialog = None
        self.show_projects()

    def _set_selected(self, button: QPushButton | None) -> None:
        for item in (self.projects_button, self.new_button, self.settings_button):
            item.setChecked(item is button)

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "Помощь Content Factory",
            "Выберите длинное видео или публичную ссылку, настройте обработку и подтвердите лучшие моменты. "
            "Все данные и ролики остаются на этом компьютере.",
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Keep a useful work area on compact laptops while retaining a fixed,
        # recognisable creator-tool sidebar at normal desktop widths.
        if not hasattr(self, "sidebar"):
            return
        self._apply_sidebar_layout()

    def _apply_sidebar_layout(self) -> None:
        """Use compact, tooltip-backed navigation before labels can clip."""

        if not hasattr(self, "sidebar"):
            return
        width = self.width()
        target_width = 156 if width < 920 else 184 if width < 1120 else 224
        if self.sidebar.width() != target_width:
            self.sidebar.setFixedWidth(target_width)
        if width < 920:
            labels = {
                self.new_button: "＋",
                self.projects_button: "▣",
                self.settings_button: "⚙",
                self.help_button: "?",
            }
        else:
            labels = {
                self.new_button: "＋  Новый",
                self.projects_button: "▣  Проекты",
                self.settings_button: "⚙  Настройки",
                self.help_button: "?  Помощь",
            }
        for button, label in labels.items():
            if button.text() != label:
                button.setText(label)
        compact = width < 920
        self.system_status.setVisible(not compact)
        self.version.setVisible(not compact)

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
