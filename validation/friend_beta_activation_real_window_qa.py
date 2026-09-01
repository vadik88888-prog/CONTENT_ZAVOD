"""Show the native Windows activation screen and assert its device-bound state."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PySide6.QtWidgets import QApplication

from app.gui.screens.activation_screen import ActivationDialog
from app.licensing import ActivationService, generate_signing_seed, public_key_from_seed


def main() -> int:
    application = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="content-factory-activation-qa-") as directory:
        seed = generate_signing_seed()
        public_key = base64.b64encode(public_key_from_seed(seed)).decode("ascii")
        activation = ActivationService(Path(directory), public_key_b64=public_key)
        dialog = ActivationDialog(activation)
        dialog.show()
        application.processEvents()
        handle = dialog.windowHandle()
        assert dialog.isVisible() and handle is not None and int(handle.winId()) != 0
        assert activation.status().code == "missing"
        assert activation.device_code in dialog.device_code.text()
        print(json.dumps({
            "window_visible": dialog.isVisible(),
            "native_handle": int(handle.winId()),
            "license_status": activation.status().code,
            "device_code_visible": activation.device_code in dialog.device_code.text(),
        }))
        dialog.close()
        application.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
