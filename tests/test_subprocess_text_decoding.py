from __future__ import annotations

import sys

from app.media import run_checked


def test_text_subprocess_output_replaces_bytes_invalid_for_windows_charmap() -> None:
    """A malformed tool byte must reach the caller instead of killing a reader thread."""

    result = run_checked([
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'progress: \\x98\\n'); sys.stderr.buffer.write(b'error: \\x98\\n')",
    ])

    assert result.stdout == "progress: �\n"
    assert result.stderr == "error: �\n"
