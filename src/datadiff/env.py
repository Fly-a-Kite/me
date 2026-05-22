from __future__ import annotations

import platform
import sys
from functools import lru_cache
from importlib import metadata

from datadiff.sqlite_runtime import SQLITE_RUNTIME, SQLITE_VERSION


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


@lru_cache(maxsize=1)
def _collect_environment_cached() -> tuple[tuple[str, str], ...]:
    data = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": package_version("pandas"),
        "polars": package_version("polars"),
        "duckdb": package_version("duckdb"),
        "sqlite": SQLITE_VERSION,
        "sqlite_runtime": SQLITE_RUNTIME,
        "pysqlite3_binary": package_version("pysqlite3-binary"),
        "datadiff_fuzz_lab": package_version("datadiff-fuzz-lab"),
    }
    return tuple(data.items())


def collect_environment() -> dict[str, str]:
    return dict(_collect_environment_cached())
