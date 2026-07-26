from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Iterable

from app.gui.models import DesktopProject, ProjectOptions, ProjectStatus
from app.source_models import SourceSpec
from app.utils import read_json, safe_name, utc_now, write_json


SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})


class PersistenceError(RuntimeError):
    pass


class InputValidationError(ValueError):
    pass


def validate_video_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise InputValidationError("Исходный видеофайл не найден.")
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        suffixes = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise InputValidationError(f"Поддерживаются видеофайлы: {suffixes}.")
    return path


class DesktopProjectStore:
    """Owns project metadata.  Generated source media is never copied here."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory.expanduser().resolve()
        self.projects_directory = self.data_directory / "projects"

    def project_directory(self, project_id: str) -> Path:
        if not project_id or any(part in project_id for part in ("/", "\\", "..")):
            raise PersistenceError("Некорректный идентификатор проекта.")
        return self.projects_directory / project_id

    def project_path(self, project_id: str) -> Path:
        return self.project_directory(project_id) / "project.json"

    def create(
        self, source_path: str | Path, *, name: str | None = None,
        options: ProjectOptions | None = None, source_metadata: dict | None = None,
    ) -> DesktopProject:
        source = validate_video_path(source_path)
        project_id = uuid.uuid4().hex
        directory = self.project_directory(project_id)
        directory.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        project = DesktopProject(
            project_id=project_id,
            name=(name or safe_name(source.name)).strip() or "Видео",
            created_at=now,
            updated_at=now,
            source_path=str(source),
            project_directory=str(directory),
            status=ProjectStatus.READY,
            settings=options or ProjectOptions(),
            source_metadata=dict(source_metadata or {"size_bytes": source.stat().st_size}),
            source_spec=SourceSpec.local(str(source), source_metadata),
        )
        self.save(project)
        return project

    def create_url(
        self, url: str, metadata: dict, *, name: str | None = None, options: ProjectOptions | None = None,
    ) -> DesktopProject:
        """Create a pending URL project; its source is downloaded only on confirmation."""

        project_id = uuid.uuid4().hex
        directory = self.project_directory(project_id)
        directory.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        title = name or str(metadata.get("title") or "Видео по ссылке")
        project = DesktopProject(
            project_id=project_id,
            name=safe_name(title, "Видео по ссылке"),
            created_at=now,
            updated_at=now,
            source_path="",
            project_directory=str(directory),
            status=ProjectStatus.READY,
            settings=options or ProjectOptions(),
            source_metadata=dict(metadata),
            source_spec=SourceSpec.url(url, metadata),
        )
        self.save(project)
        return project

    def save(self, project: DesktopProject) -> None:
        expected = self.project_directory(project.project_id).resolve()
        if Path(project.project_directory).resolve() != expected:
            raise PersistenceError("Проект пытается записаться за пределы каталога приложения.")
        project.touch()
        write_json(expected / "project.json", project.to_dict())

    def load(self, project_id: str) -> DesktopProject:
        path = self.project_path(project_id)
        try:
            raw = read_json(path)
            if not isinstance(raw, dict):
                raise ValueError("Project JSON root is not an object.")
            project = DesktopProject.from_dict(raw)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PersistenceError("Не удалось открыть сохранённый проект.") from error
        if project.project_id != project_id:
            raise PersistenceError("Идентификатор проекта не совпадает с его каталогом.")
        return project

    def list(self) -> list[DesktopProject]:
        if not self.projects_directory.exists():
            return []
        projects: list[DesktopProject] = []
        for path in self.projects_directory.iterdir():
            if not path.is_dir() or path.is_symlink() or self._is_junction(path):
                continue
            try:
                projects.append(self.load(path.name))
            except PersistenceError:
                # A single corrupt project must not hide the remaining workspace.
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def delete(self, project_id: str) -> None:
        target = self.project_directory(project_id)
        root = self.projects_directory.resolve()
        if not target.exists():
            return
        if target.is_symlink() or self._is_junction(target) or not target.resolve().is_relative_to(root):
            raise PersistenceError("Небезопасный каталог проекта не будет удалён.")
        for item in target.rglob("*"):
            if item.is_symlink() or self._is_junction(item):
                raise PersistenceError("Проект содержит ссылку; автоматическое удаление отменено.")
        shutil.rmtree(target)

    @staticmethod
    def _is_junction(path: Path) -> bool:
        probe = getattr(path, "is_junction", None)
        return bool(probe and probe())
