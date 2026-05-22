from __future__ import annotations

try:
    import pysqlite3 as sqlite3

    SQLITE_RUNTIME = "pysqlite3"
except ImportError:  # pragma: no cover - depends on optional latest-SQLite wheel.
    import sqlite3

    SQLITE_RUNTIME = "stdlib"

SQLITE_VERSION = sqlite3.sqlite_version
