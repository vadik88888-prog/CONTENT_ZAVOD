"""Durable source descriptors shared by desktop projects and download services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SOURCE_KINDS = frozenset({"local_file", "url"})
DOWNLOAD_STATES = frozenset({"not_required", "pending", "metadata_ready", "downloading", "downloaded", "failed", "cancelled"})


@dataclass(slots=True)
class SourceSpec:
    kind: str = "local_file"
    original_url: str | None = None
    downloaded_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    download_state: str = "not_required"
    error_message: str | None = None

    def validate(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError("Unsupported source kind.")
        if self.download_state not in DOWNLOAD_STATES:
            raise ValueError("Unsupported download state.")
        if self.kind == "url" and not isinstance(self.original_url, str):
            raise ValueError("URL source needs its original URL.")
        if self.kind == "local_file" and self.download_state != "not_required":
            raise ValueError("Local sources cannot have a download state.")
        if self.kind == "url" and self.download_state == "downloaded" and not self.downloaded_path:
            raise ValueError("Downloaded URL source needs a local path.")

    @property
    def is_ready(self) -> bool:
        return self.kind == "local_file" or self.download_state == "downloaded"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def local(cls, path: str, metadata: dict[str, Any] | None = None) -> "SourceSpec":
        return cls(kind="local_file", downloaded_path=path, metadata=dict(metadata or {}))

    @classmethod
    def url(cls, url: str, metadata: dict[str, Any] | None = None) -> "SourceSpec":
        return cls(kind="url", original_url=url, metadata=dict(metadata or {}), download_state="metadata_ready")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None, *, fallback_path: str = "", fallback_metadata: dict[str, Any] | None = None) -> "SourceSpec":
        if not value:
            return cls.local(fallback_path, fallback_metadata)
        if not isinstance(value, dict):
            raise ValueError("Source specification is corrupted.")
        source = cls(
            kind=str(value.get("kind", "local_file")),
            original_url=str(value["original_url"]) if value.get("original_url") else None,
            downloaded_path=str(value["downloaded_path"]) if value.get("downloaded_path") else None,
            metadata=dict(value.get("metadata") or fallback_metadata or {}),
            download_state=str(value.get("download_state", "not_required")),
            error_message=str(value["error_message"]) if value.get("error_message") else None,
        )
        source.validate()
        return source
