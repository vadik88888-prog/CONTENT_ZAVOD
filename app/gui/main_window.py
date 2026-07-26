from __future__ import annotations

from PySide6.QtCore import QByteArray, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from app.gui.screens import OnboardingDialog, ProjectScreen, ProjectsScreen, SettingsScreen
from app.gui.services.desktop_services import DesktopServices
from app.gui.viewmodels import ProjectViewModel, ProjectsViewModel, SettingsViewModel


class MainWindow(QMainWindow):
    def __init__(self, services: DesktopServices) -> None:
        super().__init__()
        self.services = services
        self.projects_viewmodel = ProjectsViewModel(services, self)
        self.project_viewmodel = ProjectViewModel(services, self)
        self.settings_viewmodel = SettingsViewModel(services, self)
        self.setWindowTitle("Content Factory")
        self.setMinimumSize(960, 680)
        self.resize(1320, 840)
        if services.settings.window_geometry:
            geometry = QByteArray.fromBase64(services.settings.window_geometry.encode("ascii", errors="ignore"))
            if not geometry.isEmpty():
                self.restoreGeometry(geometry)
        shell = QWidget(); shell.setObjectName("appShell"); self.setCentralWidget(shell)
        layout = QHBoxLayout(shell); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(220)
        nav = QVBoxLayout(sidebar); nav.setContentsMargins(14, 20, 14, 20)
        brand = QPushButton("CONTENT\nFACTORY")
        brand.setEnabled(False); brand.setStyleSheet("text-align: left; font-size: 18px; font-weight: 700; border: 0; background: transparent;")
        nav.addWidget(brand)
        nav.addSpacing(20)
        self.projects_button = self._nav_button("Проекты")
        self.new_button = self._nav_button("Новый проект")
        self.settings_button = self._nav_button("Настройки")
        self.projects_button.clicked.connect(self.show_projects)
        self.new_button.clicked.connect(self._new_project)
        self.settings_button.clicked.connect(self.show_settings)
        nav.addWidget(self.projects_button); nav.addWidget(self.new_button); nav.addWidget(self.settings_button); nav.addStretch()
        footer = QPushButton("Локально · без облака")
        footer.setEnabled(False); footer.setStyleSheet("text-align: left; color: #737D8C; border: 0; background: transparent;")
        nav.addWidget(footer)
        layout.addWidget(sidebar)
        self.stack = QStackedWidget(); self.stack.setObjectName("contentStack")
        self.projects_screen = ProjectsScreen(self.projects_viewmodel)
        self.project_screen = ProjectScreen(self.project_viewmodel)
        self.settings_screen = SettingsScreen(self.settings_viewmodel)
        self.projects_index = self.stack.addWidget(self.projects_screen)
        self.project_index = self.stack.addWidget(self.project_screen)
        self.settings_index = self.stack.addWidget(self.settings_screen)
        self.projects_screen.project_opened.connect(self.show_project)
        self.project_screen.back_requested.connect(self.show_projects)
        layout.addWidget(self.stack, 1)
        self.show_projects()
        QTimer.singleShot(0, self._maybe_onboard)

    def show_projects(self) -> None:
        self.stack.setCurrentIndex(self.projects_index)
        self._set_selected(self.projects_button)
        self.projects_screen.refresh()

    def show_project(self, project) -> None:
        self.project_screen.open(project)
        self.stack.setCurrentIndex(self.project_index)
        self._set_selected(None)

    def show_settings(self) -> None:
        self.stack.setCurrentIndex(self.settings_index)
        self._set_selected(self.settings_button)

    def _new_project(self) -> None:
        self.show_projects()
        self.projects_screen.choose_file()

    def _maybe_onboard(self) -> None:
        if not self.services.settings.onboarding_completed:
            OnboardingDialog(self.settings_viewmodel, self).exec()
            self.show_projects()

    def _set_selected(self, button: QPushButton | None) -> None:
        for item in (self.projects_button, self.new_button, self.settings_button):
            item.setChecked(item is button)

    @staticmethod
    def _nav_button(text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("nav")
        button.setCheckable(True)
        return button

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.project_viewmodel.active:
            answer = QMessageBox.question(
                self, "Закрыть приложение?",
                "Сейчас создаётся ролик. При закрытии запуск будет отмечен как прерванный и его можно будет начать снова.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.project_viewmodel.runner.cancel()
        self.services.settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self.services.save_settings()
        event.accept()
