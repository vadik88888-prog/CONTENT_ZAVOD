"""Safe text-decoding defaults for every child process launched by the app."""

from __future__ import annotations


# Windows otherwise decodes subprocess pipes with the active ANSI code page.
# Tool output is commonly UTF-8, and a single incompatible byte can crash the
# reader thread before stderr reaches the caller. Replacement preserves the
# rest of the output for the run log and the user-facing error.
UTF8_REPLACE_TEXT: dict[str, str | bool] = {
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
}
