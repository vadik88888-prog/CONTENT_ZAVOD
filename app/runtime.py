from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


INTERNAL_CLI_SWITCH = "--content-factory-internal-cli"
DATA_DIRECTORY_ENV = "CONTENT_FACTORY_DATA_DIRECTORY"


def default_data_directory(environ: Mapping[str, str] | None = None) -> Path:
    """Return the writable per-user root shared by source and frozen desktop runs."""

    values = os.environ if environ is None else environ
    explicit = values.get(DATA_DIRECTORY_ENV)
    if explicit:
        return Path(explicit).expanduser()
    root = values.get("LOCALAPPDATA") or values.get("APPDATA")
    return Path(root) / "ContentFactoryData" if root else Path.home() / ".content-factory"


@dataclass(frozen=True, slots=True)
class RuntimeCommand:
    """One shell-free child-process invocation resolved from a runtime layout."""

    program: Path
    arguments: tuple[str, ...]
    working_directory: Path


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    """All location decisions that differ between a checkout and an onedir app.

    ``program`` is the executable to launch, ``resources`` is read-only bundled
    content, ``data`` is the writable engine/project root, and ``tools`` is the
    directory containing bundled command-line tools such as ffmpeg/ffprobe.
    """

    program: Path
    resources: Path
    data: Path
    tools: Path
    frozen: bool

    @classmethod
    def for_source(
        cls,
        resources: Path,
        *,
        data: Path | None = None,
        program: Path | None = None,
        tools: Path | None = None,
    ) -> "RuntimeLayout":
        resource_root = resources.expanduser().resolve()
        return cls(
            program=(program or Path(sys.executable)).expanduser().resolve(),
            resources=resource_root,
            data=(data or default_data_directory()).expanduser().resolve(),
            tools=(tools or resource_root / "tools").expanduser().resolve(),
            frozen=False,
        )

    @classmethod
    def for_frozen(
        cls,
        *,
        program: Path,
        resources: Path,
        data: Path | None = None,
        tools: Path | None = None,
    ) -> "RuntimeLayout":
        resource_root = resources.expanduser().resolve()
        return cls(
            program=program.expanduser().resolve(),
            resources=resource_root,
            data=(data or default_data_directory()).expanduser().resolve(),
            tools=(tools or resource_root / "tools").expanduser().resolve(),
            frozen=True,
        )

    @classmethod
    def detect(cls, *, data: Path | None = None) -> "RuntimeLayout":
        """Resolve the active checkout or PyInstaller layout without using cwd."""

        if bool(getattr(sys, "frozen", False)):
            bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
            return cls.for_frozen(
                program=Path(sys.executable), resources=bundle_root, data=data,
            )
        return cls.for_source(
            Path(__file__).resolve().parents[1], data=data, program=Path(sys.executable),
        )

    def internal_cli_command(self, arguments: Sequence[str]) -> RuntimeCommand:
        """Build the engine command for this runtime without invoking a shell."""

        cli_arguments = tuple(str(argument) for argument in arguments)
        if self.frozen:
            return RuntimeCommand(
                program=self.program,
                arguments=(INTERNAL_CLI_SWITCH, *cli_arguments),
                working_directory=self.data,
            )
        return RuntimeCommand(
            program=self.program,
            arguments=("-u", "-m", "app", *cli_arguments),
            working_directory=self.resources,
        )

    def process_environment(
        self, environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return an inherited environment with bundled tools taking precedence."""

        values = dict(os.environ if environ is None else environ)
        # Source workers run with the checkout as cwd so ``python -m app`` is
        # importable.  Pass the data root separately: cwd is an engine root,
        # never an activation-state location.
        values[DATA_DIRECTORY_ENV] = str(self.data)
        if not self.tools.is_dir():
            return values
        current = values.get("PATH", "")
        entries = [entry for entry in current.split(os.pathsep) if entry]
        tool_key = os.path.normcase(str(self.tools))
        entries = [entry for entry in entries if os.path.normcase(entry) != tool_key]
        values["PATH"] = os.pathsep.join([str(self.tools), *entries])
        return values

    def activate(self) -> None:
        """Expose bundled tools to this process and every child it launches."""

        os.environ.update(self.process_environment())
