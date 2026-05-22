from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from datadiff.dsl import Case
from datadiff.fixtures import build_fixture_case, load_fixture_table
from datadiff.util import load_json


def load_fixture_replay_spec(path: str | Path) -> dict[str, Any]:
    spec = load_json(Path(path))
    if not isinstance(spec, dict):
        raise ValueError("fixture replay spec must be a JSON object")
    return spec


def build_fixture_replay_case(spec: dict[str, Any], fixture_path: str | Path) -> Case:
    fixture_path = Path(fixture_path)
    fixture = spec.get("fixture", {})
    if not isinstance(fixture, dict):
        raise ValueError("fixture replay spec field 'fixture' must be an object")

    expected_sha256 = str(fixture.get("sha256") or "")
    actual_sha256 = fixture_sha256(fixture_path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"fixture sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    operations = spec.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("fixture replay spec field 'operations' must be a non-empty list")

    case_id = str(spec.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("fixture replay spec field 'case_id' is required")
    seed = int(spec.get("seed", 0))
    columns = fixture.get("columns")
    if columns is not None and not isinstance(columns, list):
        raise ValueError("fixture replay spec field 'fixture.columns' must be a list")
    max_rows = fixture.get("max_rows")

    table = load_fixture_table(
        fixture_path,
        name=str(fixture.get("name") or "t0"),
        columns=columns,
        max_rows=int(max_rows) if max_rows is not None else None,
    )
    metadata = dict(spec.get("metadata", {}))
    metadata.setdefault("fixture_source", str(fixture.get("source") or ""))
    metadata["fixture_path"] = str(fixture_path)
    metadata["fixture_sha256"] = actual_sha256
    return build_fixture_case(
        case_id=case_id,
        seed=seed,
        table=table,
        operations=operations,
        metadata=metadata,
    )


def fixture_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
