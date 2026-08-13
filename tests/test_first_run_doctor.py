from __future__ import annotations

import os
import threading
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QObject, Signal
from PySide6.QtWidgets import QApplication

from app import cli
import app.doctor as doctor_module
from app.doctor import (
    Check,
    CredentialProbeStatus,
    DoctorReadiness,
    _ai_provider_checks,
    _probe_api_credential,
    format_report,
    summarize_checks,
)
from app.gui.models import DesktopSettings
from app.gui.screens.onboarding_screen import OnboardingDialog
from app.gui.services.system_service import SystemService
from app.gui.viewmodels.settings_viewmodel import SettingsViewModel
from app.runtime import RuntimeLayout
from app.secure_secrets import ApiKeySaveResult, api_key_state, load_runtime_secrets, save_api_key


def _application() -> QApplication:
    existing = QCoreApplication.instance()
    if existing is not None and not isinstance(existing, QApplication):
        pytest.skip("requires QApplication")
    return QApplication.instance() or QApplication([])


def test_doctor_summary_separates_blocking_warning_and_ready() -> None:
    warning = Check("yt-dlp", "warn", "Не найден.", "Используйте локальный файл.")
    blocking = Check("FFmpeg", "error", "Не найден.", "Переустановите portable-сборку.")

    assert summarize_checks([]).readiness == DoctorReadiness.READY
    assert summarize_checks([warning]).readiness == DoctorReadiness.LIMITED
    summary = summarize_checks([warning, blocking])
    assert summary.readiness == DoctorReadiness.SETUP_REQUIRED
    assert summary.blocking_count == 1 and summary.warning_count == 1
    report = format_report([warning, blocking])
    assert "WARNING yt-dlp" in report
    assert "BLOCKING FFmpeg" in report
    assert "Что сделать:" in report


def test_required_tool_is_blocking_while_optional_url_tool_is_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeLayout.for_source(tmp_path, data=tmp_path)
    monkeypatch.setattr(doctor_module, "_find_executable", lambda _runtime, _command: None)

    ffmpeg = doctor_module._tool_check(runtime, "ffmpeg", "FFmpeg", blocking=True)
    ytdlp = doctor_module._tool_check(runtime, "yt-dlp", "yt-dlp", blocking=False)

    assert ffmpeg.blocking
    assert ytdlp.warning and not ytdlp.blocking


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ([Check("GPU", "warn", "CPU fallback.", "Продолжайте на CPU.")], 0),
        ([Check("FFmpeg", "error", "Missing.", "Install FFmpeg.")], 2),
    ],
)
def test_cli_doctor_exit_code_is_nonzero_only_for_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checks: list[Check], expected: int,
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda _path: object())
    monkeypatch.setattr(cli, "collect_checks", lambda _root, _config: checks)
    monkeypatch.setattr(cli, "format_report", lambda _checks: "doctor")

    assert cli.main(["doctor"], runtime_root=tmp_path) == expected


def test_api_key_store_validates_and_never_returns_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secret = "sk-" + "a" * 40

    invalid = save_api_key("openai", "not-a-key", tmp_path)
    assert not invalid.saved and secret not in invalid.message
    assert not (tmp_path / ".env").exists()

    saved = save_api_key("openai", secret, tmp_path)
    assert saved.saved
    assert secret not in saved.message
    assert api_key_state("openai", tmp_path) == "configured"
    assert secret not in repr(saved)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_runtime_secrets(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == secret


def test_doctor_report_never_contains_configured_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-" + "z" * 40
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    runtime = RuntimeLayout.for_source(tmp_path, data=tmp_path)
    monkeypatch.setattr(
        doctor_module,
        "_tool_check",
        lambda _runtime, _command, label, *, blocking: Check(label, "ok", "Доступен."),
    )
    monkeypatch.setattr(doctor_module, "_nvidia_checks", lambda: [])
    monkeypatch.setattr(doctor_module, "_cuda_check", lambda: Check("CUDA", "ok", "Доступна."))
    monkeypatch.setattr(
        doctor_module,
        "_probe_api_credential",
        lambda _provider, _secret: CredentialProbeStatus.CONFIGURED,
    )
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        doctor_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=20 * 1024**3),
    )

    report = format_report(doctor_module.collect_checks(runtime))

    assert secret not in report
    assert "OpenAI API key: Ключ подтверждён провайдером; значение скрыто." in report


def _credential_checks(
    tmp_path: Path,
    *,
    provider: str = "openai",
) -> list[Check]:
    config = doctor_module.AppConfig()
    config.ai.provider = provider
    runtime = RuntimeLayout.for_source(tmp_path, data=tmp_path)
    return _ai_provider_checks(runtime, config)


@pytest.mark.parametrize(
    ("provider", "variable", "label"),
    [
        ("openai", "OPENAI_API_KEY", "OpenAI API key"),
        ("gemini", "GEMINI_API_KEY", "Gemini API key"),
    ],
)
def test_real_provider_without_key_is_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    variable: str,
    label: str,
) -> None:
    monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        doctor_module,
        "_probe_api_credential",
        lambda _provider, _secret: pytest.fail("missing key must not be probed"),
    )

    credential = _credential_checks(tmp_path, provider=provider)[-1]

    assert credential.label == label
    assert credential.blocking
    assert "не настроен" in credential.detail


def test_real_provider_locally_invalid_key_is_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key-format")
    monkeypatch.setattr(
        doctor_module,
        "_probe_api_credential",
        lambda _provider, _secret: pytest.fail("invalid key must not be sent"),
    )

    credential = _credential_checks(tmp_path)[-1]

    assert credential.blocking
    assert "локальную проверку формата" in credential.detail


@pytest.mark.parametrize("status", [401, 403])
def test_real_provider_confirmed_auth_rejection_is_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int,
) -> None:
    secret = "sk-" + "r" * 40
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def reject(_request, *, timeout):
        assert timeout == 5
        raise urllib.error.HTTPError(
            "https://provider.invalid/models", status, "rejected", None, None,
        )

    monkeypatch.setattr(doctor_module.urllib.request, "urlopen", reject)

    assert _probe_api_credential("openai", secret) == CredentialProbeStatus.AUTH_REJECTED
    credential = _credential_checks(tmp_path)[-1]
    assert credential.blocking
    assert "401/403" in credential.detail
    assert secret not in credential.detail
    assert secret not in credential.action


def test_real_provider_network_outage_with_credential_is_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-" + "n" * 40
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        doctor_module.urllib.request,
        "urlopen",
        lambda _request, *, timeout: (_ for _ in ()).throw(
            urllib.error.URLError("temporary audit outage")
        ),
    )

    credential = _credential_checks(tmp_path)[-1]

    assert credential.warning and not credential.blocking
    assert "временно недоступен" in credential.detail
    assert secret not in credential.detail
    assert secret not in credential.action


def test_explicit_mock_mode_without_key_is_nonblocking_but_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        doctor_module,
        "_probe_api_credential",
        lambda _provider, _secret: pytest.fail("mock mode must not probe a key"),
    )

    checks = _credential_checks(tmp_path, provider="mock")
    summary = summarize_checks(checks)

    assert not any(item.blocking for item in checks)
    assert any(item.label == "Локальный тестовый режим" and item.warning for item in checks)
    assert summary.readiness == DoctorReadiness.LIMITED


def test_explicit_missing_config_is_blocking_and_does_not_escape_runtime_layout(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    runtime = RuntimeLayout.for_source(resources, data=tmp_path / "data")
    settings = DesktopSettings.defaults(runtime.data)
    settings.config_path = str(tmp_path / "missing-config.yaml")
    service = SystemService(runtime)

    checks = service.checks(settings)

    assert service.data_root == runtime.data
    assert service.resources_root == runtime.resources
    assert len(checks) == 1 and checks[0].blocking
    assert checks[0].label == "Конфигурация"
    assert service.ai_provider(settings) is None


def test_async_diagnostics_returns_before_slow_probe_finishes(tmp_path: Path) -> None:
    application = _application()
    entered = threading.Event()
    release = threading.Event()
    received: list[list[Check]] = []

    class SlowSystem:
        data_root = tmp_path

        @staticmethod
        def checks(_settings):
            entered.set()
            release.wait(2)
            return [Check("Runtime", "ok", "Готов.")]

        @staticmethod
        def ai_provider(_settings):
            return "mock"

    services = SimpleNamespace(
        settings=DesktopSettings.defaults(tmp_path),
        system=SlowSystem(),
    )
    viewmodel = SettingsViewModel(services)
    viewmodel.diagnostics_ready.connect(received.append)

    started_at = time.perf_counter()
    viewmodel.diagnostics()
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.1
    assert entered.wait(1)
    assert not received
    release.set()
    deadline = time.monotonic() + 3
    while not received and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert received and received[0][0].label == "Runtime"


class _FakeOnboardingViewModel(QObject):
    settings_changed = Signal(object)
    diagnostics_started = Signal()
    diagnostics_ready = Signal(list)

    def __init__(self, data_root: Path) -> None:
        super().__init__()
        self.settings = DesktopSettings.defaults(data_root)
        self.saved = 0
        self.diagnostic_requests = 0

    def diagnostics(self) -> None:
        self.diagnostic_requests += 1
        self.diagnostics_started.emit()

    def ai_provider(self) -> str:
        return "mock" if self.settings.local_test_mode else "openai"

    @staticmethod
    def save_api_key(_value: str) -> ApiKeySaveResult:
        return ApiKeySaveResult(False, "Ключ не сохранён.")

    def save(self) -> None:
        self.saved += 1


def test_onboarding_cannot_finish_with_blocking_but_warning_is_allowed(tmp_path: Path) -> None:
    _application()
    viewmodel = _FakeOnboardingViewModel(tmp_path)
    dialog = OnboardingDialog(viewmodel)  # type: ignore[arg-type]
    blocking = Check("FFmpeg", "error", "Не найден.", "Установите FFmpeg.")
    warning = Check("API key", "warn", "Не настроен.", "Добавьте ключ позже.")

    try:
        viewmodel.diagnostics_ready.emit([blocking])
        assert not dialog.continue_button.isEnabled()
        dialog._finish()
        assert not viewmodel.settings.onboarding_completed
        assert viewmodel.saved == 0

        viewmodel.diagnostics_ready.emit([warning])
        assert dialog.continue_button.isEnabled()
        dialog._finish()
        assert viewmodel.settings.onboarding_completed
        assert viewmodel.saved == 1
    finally:
        dialog.deleteLater()


def test_onboarding_offers_explicit_local_test_escape_from_real_provider_block(
    tmp_path: Path,
) -> None:
    _application()
    viewmodel = _FakeOnboardingViewModel(tmp_path)
    dialog = OnboardingDialog(viewmodel)  # type: ignore[arg-type]
    local_warning = Check(
        "Локальный тестовый режим",
        "warn",
        "Mock mode.",
        "Настройте real provider для production.",
    )

    try:
        viewmodel.diagnostics_ready.emit([
            Check(
                "OpenAI API key",
                "error",
                "Ключ не настроен.",
                "Добавьте ключ или включите локальный тестовый режим.",
            )
        ])
        assert not dialog.continue_button.isEnabled()

        requests_before_toggle = viewmodel.diagnostic_requests
        dialog.local_test.setChecked(True)
        assert viewmodel.settings.local_test_mode
        assert viewmodel.ai_provider() == "mock"
        assert viewmodel.diagnostic_requests == requests_before_toggle + 1
        assert dialog.api_setup_button.isHidden()

        viewmodel.diagnostics_ready.emit([local_warning])
        assert dialog.continue_button.isEnabled()
        assert summarize_checks([local_warning]).readiness == DoctorReadiness.LIMITED
    finally:
        dialog.deleteLater()
