from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


EXECUTION_COST_MODEL_SCHEMA_VERSION = "execution-cost-model-v1"
DEFAULT_PROFILE_PATH = Path("experiments/revision_cost_model_v2/result.json")
DEFAULT_BACKEND_COST_ALIASES = {
    # The persistent target is an execution mode of the same DuckDB package.
    # Prefer a direct measurement when one exists; otherwise inherit DuckDB's
    # measured process cost instead of making the target unrunnable.
    "duckdb_persistent": "duckdb",
}


@dataclass(frozen=True, slots=True)
class BackendCostModel:
    source_path: str
    source_profile_id: str
    source_result_digest: str
    metric: str
    raw_costs: dict[str, float]
    normalized_costs: dict[str, float]
    digest: str
    backend_aliases: dict[str, str] = field(default_factory=dict)

    def cost(self, backend: str) -> float:
        key = str(backend)
        resolved_key = self.backend_aliases.get(key, key)
        if resolved_key not in self.normalized_costs:
            raise KeyError(f"backend cost is undeclared: {key}")
        value = float(self.normalized_costs[resolved_key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"backend cost must be positive and finite: {key}={value}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_COST_MODEL_SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_profile_id": self.source_profile_id,
            "source_result_digest": self.source_result_digest,
            "metric": self.metric,
            "raw_costs": dict(sorted(self.raw_costs.items())),
            "normalized_costs": dict(sorted(self.normalized_costs.items())),
            "backend_aliases": dict(sorted(self.backend_aliases.items())),
            "normalization": "divide_by_arithmetic_mean_of_available_backends",
            "digest": self.digest,
        }


def load_backend_cost_model(
    repo_root: Path,
    path: Path = DEFAULT_PROFILE_PATH,
) -> BackendCostModel | None:
    target = path if path.is_absolute() else repo_root / path
    if not target.is_file():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = payload.get("cost_tables", {}).get("backend", [])
    raw_costs: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        backend = str(row.get("backend", ""))
        if not backend:
            continue
        if backend in raw_costs:
            raise ValueError(f"backend cost profile contains duplicate row: {backend}")
        value = float(row.get("isolate_process_cpu_ms_per_case", 0.0) or 0.0)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"backend cost profile contains non-positive cost: {backend}={value}"
            )
        raw_costs[backend] = value
    if not raw_costs:
        raise ValueError("backend cost profile contains no positive backend costs")
    mean_cost = sum(raw_costs.values()) / len(raw_costs)
    normalized = {
        backend: cost / mean_cost
        for backend, cost in raw_costs.items()
    }
    backend_aliases = {
        alias: target
        for alias, target in DEFAULT_BACKEND_COST_ALIASES.items()
        if alias not in normalized and target in normalized
    }
    material = {
        "schema_version": EXECUTION_COST_MODEL_SCHEMA_VERSION,
        "source_path": target.relative_to(repo_root).as_posix(),
        "source_profile_id": str(payload.get("profile_id", "")),
        "source_result_digest": str(payload.get("result_digest", "")),
        "metric": "isolate_process_cpu_ms_per_case",
        "raw_costs": dict(sorted(raw_costs.items())),
        "normalized_costs": dict(sorted(normalized.items())),
        "backend_aliases": dict(sorted(backend_aliases.items())),
    }
    digest = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return BackendCostModel(
        source_path=material["source_path"],
        source_profile_id=material["source_profile_id"],
        source_result_digest=material["source_result_digest"],
        metric=material["metric"],
        raw_costs=raw_costs,
        normalized_costs=normalized,
        digest=f"cost-model-{digest}",
        backend_aliases=backend_aliases,
    )
