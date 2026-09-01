from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import sys

import pytest

from app.licensing import (
    ActivationService,
    LicensingError,
    create_signed_license,
    device_code_from_secret,
    generate_signing_seed,
    public_key_from_seed,
    sign_message,
    verify_message,
)
from app.gui.models import DesktopSettings
from app.gui.services.desktop_project_store import DesktopProjectStore
from app.gui.services.desktop_services import DesktopServices
from app.gui.services.pipeline_facade import PipelineFacade
from app.gui.services.run_history_store import RunHistoryStore
from app.gui.services.settings_store import SettingsStore
from app.gui.services.system_service import SystemService


def _activation(tmp_path, name: str, public_key: bytes) -> ActivationService:
    return ActivationService(
        tmp_path / name,
        public_key_b64=base64.b64encode(public_key).decode("ascii"),
    )


def test_ed25519_matches_rfc8032_test_vector() -> None:
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    public_key = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    expected = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    assert public_key_from_seed(seed) == public_key
    assert sign_message(seed, b"") == expected
    assert verify_message(public_key, b"", expected)
    assert not verify_message(public_key, b"changed", expected)


def test_device_bound_license_survives_restart_and_rejects_second_pc(tmp_path) -> None:
    seed = generate_signing_seed()
    public_key = public_key_from_seed(seed)
    pc_a = _activation(tmp_path, "pc-a", public_key)
    pc_b = _activation(tmp_path, "pc-b", public_key)
    code_a = pc_a.device_code
    assert code_a == ActivationService(pc_a.data_directory, public_key_b64=base64.b64encode(public_key).decode("ascii")).device_code
    if sys.platform == "win32":
        assert pc_a.identity_path.read_bytes() != pc_a._device_secret()  # Windows DPAPI blob, not raw identity

    license_data = create_signed_license(seed, code_a, datetime.now(timezone.utc) + timedelta(days=7))
    license_bytes = __import__("json").dumps(license_data).encode("utf-8")
    assert pc_a.install_license(license_bytes).active
    assert pc_a.status().active  # restart reads the same DPAPI-protected identity
    rejected = pc_b.install_license(license_bytes)
    assert not rejected.active and rejected.code == "wrong_device"
    assert pc_b.status().code == "missing"
    with pytest.raises(LicensingError):
        pc_b.require_processing()


def test_expired_and_tampered_licenses_block_processing_without_deleting_projects(tmp_path) -> None:
    seed = generate_signing_seed()
    public_key = public_key_from_seed(seed)
    activation = _activation(tmp_path, "pc", public_key)
    expired = create_signed_license(seed, activation.device_code, datetime.now(timezone.utc) - timedelta(seconds=1))
    assert not activation.install_license(__import__("json").dumps(expired).encode("utf-8")).active
    assert not activation.license_path.exists()  # invalid replacement does not leave a bad licence behind
    activation.license_path.parent.mkdir(parents=True, exist_ok=True)
    activation.license_path.write_text(__import__("json").dumps(expired), encoding="utf-8")
    with pytest.raises(LicensingError, match="истёк"):
        activation.require_processing()

    active = create_signed_license(seed, activation.device_code, datetime.now(timezone.utc) + timedelta(days=1))
    active["payload"]["device_code"] = device_code_from_secret(b"x" * 32)
    assert not activation.install_license(__import__("json").dumps(active).encode("utf-8")).active
    assert activation.status().code == "expired"  # the previous licence was retained, never project data


def test_every_new_processing_entrypoint_checks_license_before_project_mutation(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    data = tmp_path / "data"
    projects = DesktopProjectStore(data)
    project = projects.create(source)
    seed = generate_signing_seed()
    services = DesktopServices(
        engine_root=tmp_path,
        settings_store=SettingsStore(data),
        settings=DesktopSettings.defaults(data),
        projects=projects,
        runs=RunHistoryStore(projects),
        pipeline=PipelineFacade(tmp_path),
        system=SystemService(tmp_path),
        activation=_activation(data, "activation", public_key_from_seed(seed)),
    )
    entrypoints = (
        lambda: services.prepare_run(project),
        lambda: services.prepare_analysis(project),
        lambda: services.prepare_draft(project, []),
        lambda: services.prepare_selected_render(project),
        lambda: services.prepare_render_revision(project, None),  # type: ignore[arg-type]
    )
    for entrypoint in entrypoints:
        with pytest.raises(LicensingError):
            entrypoint()
    assert services.runs.list(project.project_id) == []
