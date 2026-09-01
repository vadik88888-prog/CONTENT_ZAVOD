from app.config import AppConfig
from app.doctor import Check, _cuda_check, collect_checks, format_report


def test_cuda_check_warns_when_cuda_device_has_an_incomplete_runtime(monkeypatch) -> None:
    from app.cuda_runtime import CudaRuntimeProbe

    monkeypatch.setattr(
        "app.doctor.probe_cuda_runtime",
        lambda: CudaRuntimeProbe(1, False, "CUDA runtime incomplete: required cublas64_12.dll is unavailable"),
    )

    check = _cuda_check()

    assert check.status == "warn"
    assert "cublas64_12.dll" in check.detail


def test_doctor_report_is_encodable_in_cp1251() -> None:
    report = format_report([Check("Python", "ok", "3.11")])

    assert "OK Python: 3.11" in report
    report.encode("cp1251")


def test_cuda_check_has_a_nonempty_user_facing_result() -> None:
    check = _cuda_check()

    assert check.label == "CUDA"
    assert check.status in {"ok", "warn"}
    assert check.detail


def test_doctor_checks_openai_tts_key_only_when_tts_is_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = AppConfig()
    config.tts.enabled = True
    checks = collect_checks(tmp_path, config)
    tts = next(item for item in checks if item.label == "OpenAI TTS API key")
    assert tts.status == "warn"
    config.tts.provider = "mock"
    checks = collect_checks(tmp_path, config)
    assert not any(item.label == "OpenAI TTS API key" for item in checks)
