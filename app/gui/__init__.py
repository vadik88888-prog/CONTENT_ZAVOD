"""Desktop application layer for the local Content Factory engine.

The package intentionally stays outside the engine modules.  Widgets talk to
desktop services; only the facade owns the CLI/process integration.
"""

from __future__ import annotations
