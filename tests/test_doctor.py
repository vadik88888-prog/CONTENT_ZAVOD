from app.doctor import Check, _cuda_check, format_report


def test_doctor_report_is_encodable_in_cp1251() -> None:
    report = format_report([Check("Python", "ok", "3.11")])

    assert "OK Python: 3.11" in report
    report.encode("cp1251")


def test_cuda_check_has_a_nonempty_user_facing_result() -> None:
    check = _cuda_check()

    assert check.label == "CUDA"
    assert check.status in {"ok", "warn"}
    assert check.detail
