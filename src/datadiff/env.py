from __future__ import annotations

import platform
import sys
from hashlib import sha256
from functools import lru_cache
from importlib import metadata
from pathlib import Path
import json

from datadiff.sqlite_runtime import SQLITE_RUNTIME, SQLITE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


@lru_cache(maxsize=1)
def source_tree_sha256() -> str:
    """Fingerprint executable project sources for experiment reuse checks."""

    rows = []
    for root in (PROJECT_ROOT / "src" / "datadiff", PROJECT_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                rows.append((path.relative_to(PROJECT_ROOT).as_posix(), sha256(path.read_bytes()).hexdigest()))
    return sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _collect_environment_cached() -> tuple[tuple[str, str], ...]:
    data = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": package_version("pandas"),
        "polars": package_version("polars"),
        "duckdb": package_version("duckdb"),
        "pyarrow": package_version("pyarrow"),
        "datafusion": package_version("datafusion"),
        "chdb": package_version("chdb"),
        "sqlite": SQLITE_VERSION,
        "sqlite_runtime": SQLITE_RUNTIME,
        "pysqlite3_binary": package_version("pysqlite3-binary"),
        "datadiff_fuzz_lab": package_version("datadiff-fuzz-lab"),
        "source_tree_sha256": source_tree_sha256(),
    }
    return tuple(data.items())


def collect_environment() -> dict[str, str]:
    return dict(_collect_environment_cached())
