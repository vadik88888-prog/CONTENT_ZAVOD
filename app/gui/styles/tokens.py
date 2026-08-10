from __future__ import annotations

from types import MappingProxyType


# Approved Desktop creator-tech palette.  QSS and any runtime colour checks
# resolve through this single source so screenshots cannot drift independently
# from the documented visual contract.
THEME_TOKENS = MappingProxyType({
    "APP_BG": "#0D0F13",
    "SIDEBAR": "#11141A",
    "MAIN": "#151922",
    "ELEVATED": "#1B202C",
    "BORDER": "#2A303D",
    "PRIMARY": "#FF6A00",
    "PRIMARY_HOVER": "#FF7F33",
    "ACCENT": "#252A4A",
    "SUCCESS": "#56D6A0",
    "WARNING": "#D7A95B",
    "ERROR": "#E46B78",
    "PRIMARY_TEXT": "#F3F4F7",
    "SECONDARY": "#9299AA",
    "MUTED": "#676E7F",
})
