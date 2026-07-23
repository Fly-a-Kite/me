"""Oracle-Carrying Semantic Contrast (OSC) implementation.

This package is intentionally independent from :mod:`datadiff`.  Migration
adapters live at explicit compatibility boundaries; the OSC semantic core never
uses candidate, family, or root strings to decide an oracle verdict.
"""

from datadiff_osc.api import *  # noqa: F401,F403
from datadiff_osc.api import __all__ as _PUBLIC_API

__all__ = list(_PUBLIC_API)

__version__ = "0.1.0"
