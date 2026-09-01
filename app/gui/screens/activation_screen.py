from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from app.gui.responsive import make_label_shrinkable
from app.licensing import ActivationService


class ActivationDialog(QDialog):
    """A blocking, local activation screen shown before the desktop shell."""

    def __init__(self, activation: ActivationService, parent=None) -> None:
        super().__init__(parent)
        self.activation = activation
        self.setWindowTitle("Активация Friend Beta")
        self.setModal(True)
        self.setMinimumWidth(510)
        self.resize(620, 340)

        layout = QVBoxLayout(self)
        title = QLabel("Активация Friend Beta")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)
        description = QLabel(
            "Эта копия приложения привязана к одному устройству. Скопируйте код ниже, "
            "получите файл лицензии у администратора и выберите его здесь."
        )
        description.setWordWrap(True)
        make_label_shrinkable(description)
        layout.addWidget(description)

        self.device_code = QLabel()
        self.device_code.setObjectName("activationDeviceCode")
        self.device_code.setTextInteractionFlags(self.device_code.textInteractionFlags())
        self.device_code.setWordWrap(True)
        make_label_shrinkable(self.device_code)
        layout.addWidget(self.device_code)

        copy_button = QPushButton("Скопировать код устройства")
        copy_button.clicked.connect(self._copy_device_code)
        layout.addWidget(copy_button)
        self.status_label = QLabel()
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        make_label_shrinkable(self.status_label)
        layout.addWidget(self.status_label)
        layout.addStretch()

        actions = QHBoxLayout()
        self.install_button = QPushButton("Выбрать файл лицензии…")
        self.install_button.setObjectName("primary")
        self.install_button.clicked.connect(self._install_license)
        exit_button = QPushButton("Закрыть")
        exit_button.clicked.connect(self.reject)
        actions.addWidget(self.install_button)
        actions.addStretch()
        actions.addWidget(exit_button)
        layout.addLayout(actions)
        self._render()

    def _render(self, message: str | None = None) -> None:
        try:
            code = self.activation.device_code
        except Exception:
            code = "Недоступен"
        self.device_code.setText(f"Код устройства\n{code}")
        self.device_code.setToolTip(code)
        status = self.activation.status()
        self.status_label.setText(message or status.message)

    def _copy_device_code(self) -> None:
        try:
            code = self.activation.device_code
        except Exception as error:
            self._render(str(error))
            return
        QGuiApplication.clipboard().setText(code)
        self._render("Код устройства скопирован в буфер обмена.")

    def _install_license(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Выберите лицензию Friend Beta", "", "Лицензия Friend Beta (*.json);;Все файлы (*)",
        )
        if not path:
            return
        result = self.activation.install_license(path)
        if result.active:
            self.accept()
            return
        self._render(result.message)
