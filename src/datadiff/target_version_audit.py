from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import urllib.request

from datadiff.env import package_version
from datadiff.util import dump_json, utc_now


TARGET_VERSION_AUDIT_SCHEMA_VERSION = "target-version-audit-v1"


@dataclass(frozen=True, slots=True)
class TargetPackage:
    package: str
    target: str
    role: str


DEFAULT_TARGET_PACKAGES: tuple[TargetPackage, ...] = (
    TargetPackage("pandas", "pandas", "implemented target"),
    TargetPackage("polars", "polars", "implemented target for polars/polars_lazy/polars_streaming"),
    TargetPackage("duckdb", "duckdb", "implemented target for duckdb/duckdb_persistent"),
    TargetPackage("pyarrow", "pyarrow", "implemented target"),
    TargetPackage("datafusion", "datafusion", "implemented target"),
    TargetPackage("chdb", "chdb", "experimental optional target"),
    TargetPackage("pysqlite3-binary", "sqlite", "sqlite runtime package"),
)


def build_target_version_audit(
    *,
    packages: list[str] | tuple[str, ...] | None = None,
    latest_versions: Mapping[str, str] | None = None,
    query_latest: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    selected = _target_packages(packages)
    latest_map = {str(key): str(value) for key, value in (latest_versions or {}).items()}
    rows = [
        _package_row(target_package, latest_map=latest_map, query_latest=query_latest)
        for target_package in selected
    ]
    outdated = [
        row
        for row in rows
        if row["latest_version"] and row["installed_version"] != row["latest_version"]
    ]
    unknown_latest = [row for row in rows if not row["latest_version"]]
    return {
        "schema_version": TARGET_VERSION_AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now(),
        "python": sys.version.replace("\n", " "),
        "target_packages": rows,
        "summary": {
            "target_package_count": len(rows),
            "up_to_date_target_package_count": sum(1 for row in rows if row["up_to_date"] is True),
            "outdated_target_package_count": len(outdated),
            "unknown_latest_target_package_count": len(unknown_latest),
            "all_target_packages_up_to_date": not outdated and not unknown_latest,
            "outdated_target_packages": outdated,
            "unknown_latest_target_packages": unknown_latest,
        },
        "methodology_claim": (
            "Fresh latest-version bug claims must record installed target package versions "
            "and compare them against the latest public package versions before final artifact reporting."
        ),
    }


def write_target_version_audit(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    dump_json(dict(payload), path)
    return path


def parse_latest_version_overrides(value: str | None) -> dict[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return {}
    if raw.startswith("@"):
        return _latest_map_from_json(json.loads(Path(raw[1:]).read_text(encoding="utf-8")))
    if raw.startswith("{"):
        return _latest_map_from_json(json.loads(raw))
    result: dict[str, str] = {}
    for item in raw.split(","):
        name, sep, version = item.partition("=")
        if sep and name.strip() and version.strip():
            result[name.strip()] = version.strip()
    return result


def _target_packages(packages: list[str] | tuple[str, ...] | None) -> list[TargetPackage]:
    if not packages:
        return list(DEFAULT_TARGET_PACKAGES)
    defaults = {target.package: target for target in DEFAULT_TARGET_PACKAGES}
    selected: list[TargetPackage] = []
    for package in packages:
        name = str(package).strip()
        if not name:
            continue
        selected.append(defaults.get(name, TargetPackage(name, name, "extra audited package")))
    return selected


def _package_row(
    target_package: TargetPackage,
    *,
    latest_map: Mapping[str, str],
    query_latest: bool,
) -> dict[str, Any]:
    installed = package_version(target_package.package)
    latest = latest_map.get(target_package.package, "")
    latest_source = "override" if latest else ""
    latest_error = ""
    if not latest and query_latest:
        try:
            latest = _pypi_latest_version(target_package.package)
            latest_source = "pypi"
        except Exception as exc:  # noqa: BLE001
            latest_error = f"{type(exc).__name__}: {exc}"
            latest_source = "unavailable"
    elif not latest:
        latest_source = "not_checked"
    up_to_date = installed == latest if latest else None
    return {
        "package": target_package.package,
        "target": target_package.target,
        "role": target_package.role,
        "installed_version": installed,
        "latest_version": latest,
        "up_to_date": up_to_date,
        "latest_source": latest_source,
        "latest_error": latest_error,
        "source": f"https://pypi.org/project/{target_package.package}/",
    }


def _pypi_latest_version(package: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=15) as response:
        payload = json.load(response)
    return str(payload["info"]["version"])


def _latest_map_from_json(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("latest version overrides must be a JSON object")
    return {str(key): str(value) for key, value in payload.items() if str(key).strip()}
