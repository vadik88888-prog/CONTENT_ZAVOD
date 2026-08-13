"""Compatibility imports for the core local secret store."""

from app.secure_secrets import (
    ApiKeySaveResult,
    api_key_state,
    key_configured,
    load_runtime_secrets,
    save_api_key,
    validate_api_key,
)

__all__ = [
    "ApiKeySaveResult", "api_key_state", "key_configured", "load_runtime_secrets",
    "save_api_key", "validate_api_key",
]
