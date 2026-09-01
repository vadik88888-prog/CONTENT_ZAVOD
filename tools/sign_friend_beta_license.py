"""Local-only Friend Beta licence signer; intentionally excluded from packages."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, time, timezone
import json
from pathlib import Path
import sys

from app.licensing import (
    PUBLIC_VERIFICATION_KEY_B64,
    LicensingError,
    _write_private_file,
    create_signed_license,
    default_admin_key_path,
    generate_signing_seed,
    normalize_device_code,
    public_key_from_seed,
)


def _read_seed(path: Path) -> bytes:
    try:
        seed = base64.b64decode(path.read_text(encoding="ascii").strip(), validate=True)
    except (OSError, UnicodeError, ValueError) as error:
        raise LicensingError("Не удалось прочитать приватный signing key.") from error
    if len(seed) != 32:
        raise LicensingError("Приватный signing key имеет неверный формат.")
    return seed


def _create_key(path: Path) -> bytes:
    if path.exists():
        raise LicensingError(f"Signing key уже существует: {path}")
    seed = generate_signing_seed()
    _write_private_file(path, base64.b64encode(seed) + b"\n")
    return seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a signed, device-bound Friend Beta licence.")
    parser.add_argument("--device-code", help="Код устройства из экрана активации")
    parser.add_argument("--expires", help="Срок в UTC: YYYY-MM-DD (включительно)")
    parser.add_argument("--output", type=Path, help="Путь к создаваемому .json файлу лицензии")
    parser.add_argument("--private-key", type=Path, default=default_admin_key_path())
    parser.add_argument("--create-key", action="store_true", help="Создать локальный key вне репозитория и вывести его public key")
    args = parser.parse_args(argv)

    try:
        if args.create_key:
            seed = _create_key(args.private_key)
            print("Private key created at:", args.private_key)
            print("Public verification key:", base64.b64encode(public_key_from_seed(seed)).decode("ascii"))
            return 0
        if not args.device_code or not args.expires or args.output is None:
            parser.error("Для подписи нужны --device-code, --expires и --output.")
        output = args.output.expanduser().resolve()
        repository = Path(__file__).resolve().parents[1]
        if output.is_relative_to(repository):
            raise LicensingError("Лицензию нельзя записывать внутрь репозитория.")
        expiry = datetime.combine(datetime.strptime(args.expires, "%Y-%m-%d").date(), time.max, tzinfo=timezone.utc)
        seed = _read_seed(args.private_key.expanduser())
        if base64.b64encode(public_key_from_seed(seed)).decode("ascii") != PUBLIC_VERIFICATION_KEY_B64:
            raise LicensingError("Этот signing key не соответствует public verification key приложения.")
        license_data = create_signed_license(seed, normalize_device_code(args.device_code), expiry)
        _write_private_file(output, (json.dumps(license_data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print("Signed licence created:", output)
        return 0
    except (LicensingError, ValueError) as error:
        print("Error:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
