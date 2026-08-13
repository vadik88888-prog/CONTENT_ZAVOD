from __future__ import annotations

import sys
from typing import Sequence

from app.runtime import INTERNAL_CLI_SWITCH, RuntimeLayout


def _run_internal_cli(arguments: list[str], layout: RuntimeLayout) -> int:
    from app.cli import main as cli_main

    return cli_main(arguments, runtime_root=layout.data)


def _run_desktop(arguments: list[str], layout: RuntimeLayout) -> int:
    from app.gui.application import run

    return run([str(layout.program), *arguments], runtime=layout)


def main(
    argv: Sequence[str] | None = None, *, layout: RuntimeLayout | None = None,
) -> int:
    """Dispatch one frozen executable to either the GUI or its internal worker."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    runtime = layout or RuntimeLayout.detect()
    runtime.activate()
    if arguments[:1] == [INTERNAL_CLI_SWITCH]:
        return _run_internal_cli(arguments[1:], runtime)
    return _run_desktop(arguments, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
