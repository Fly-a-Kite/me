# Reproducing the OSC pre-implementation review

Run all commands from `/data1/lbw/xjx/datadiff_fuzz_lab` with the current
frozen dirty-worktree snapshot. These commands are read-only.

## Validate the authority artifact

```bash
jq empty experiments/osc_v1/preimplementation_credibility_review.json
jq '{review_status, authority, preimplementation_checks}' \
  experiments/osc_v1/preimplementation_credibility_review.json
```

## Recompute family, cell, edge, pair and tile denominators

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from itertools import combinations

from datadiff.family_witness_registry import (
    GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE,
    global_family_witness_registrations,
)
from datadiff.semantic_family_universe_v3 import semantic_family_v3_definitions

definitions = {item.family_id: item for item in semantic_family_v3_definitions()}
registrations = global_family_witness_registrations(
    GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE
)
fresh = [
    item
    for item in registrations
    if definitions[item.family_id].source == "coverage_expansion"
]
regression = [
    item
    for item in registrations
    if definitions[item.family_id].source == "confirmed_root"
]

edges = 0
tiles = 0
backend_executions = 0
for registration in fresh:
    dimensions = [len(values) for _, values in registration.axes]
    for index, size in enumerate(dimensions):
        count = size - 1
        for other, other_size in enumerate(dimensions):
            if other != index:
                count *= other_size
        edges += count
    family_tiles = 0
    for left, right in combinations(range(len(dimensions)), 2):
        count = (dimensions[left] - 1) * (dimensions[right] - 1)
        for index, size in enumerate(dimensions):
            if index not in {left, right}:
                count *= size
        family_tiles += count
    tiles += family_tiles
    backend_executions += 4 * family_tiles * len(registration.backends)

pairs = sum(
    registration.cell_count
    * len(definitions[registration.family_id].control_backends)
    for registration in fresh
)
print(
    {
        "fresh_families": len(fresh),
        "fresh_cells": sum(item.cell_count for item in fresh),
        "regression_families": len(regression),
        "regression_cells": sum(item.cell_count for item in regression),
        "edges": edges,
        "pairs": pairs,
        "shadow_tiles": tiles,
        "tile_endpoints_without_reuse": 4 * tiles,
        "tile_backend_executions_without_reuse": backend_executions,
    }
)
PY
```

Expected values are `16`, `232`, `9`, `144`, `384`, `502`, `202`, `808`,
and `2224`, respectively.

## Recompute the offline component-planner estimate

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import glob
import hashlib
import json

files = sorted(
    glob.glob(
        "experiments/next_architecture_capability_audit_v6/"
        "shard_results/*.json"
    )
)
groups = backend_results = mismatch_groups = 0
full_bytes = exact_bytes = compact_bytes = 0

for path in files:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    for row in payload["case_rows"]:
        groups += 1
        normalized = row.get("normalized", {})
        mismatch = (
            str(
                row.get("disagreement_descriptor", {}).get(
                    "mismatch_class", "none"
                )
            )
            != "none"
        )
        mismatch_groups += int(mismatch)
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        full_bytes += len(encoded)
        exact_bytes += len(encoded) if mismatch else 0
        view = str(row.get("semantic_comparison_profile", {}).get("view", "unknown"))
        for backend, result in normalized.items():
            backend_results += 1
            schema = json.dumps(
                [result.get("columns", []), result.get("column_types", [])],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            component = json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            compact = {
                "backend": backend,
                "status": result.get("status"),
                "schema_digest": hashlib.sha256(schema).hexdigest(),
                "row_count": result.get("row_count", len(result.get("rows", []))),
                "observer": view,
                "contract_digest": "0" * 64,
                "component_digest": hashlib.sha256(component).hexdigest(),
                "hash": "sha256-v1",
            }
            compact_bytes += len(
                json.dumps(
                    compact, sort_keys=True, separators=(",", ":")
                ).encode()
            )

print(
    json.dumps(
        {
            "source_files": len(files),
            "groups": groups,
            "backend_results": backend_results,
            "mismatch_groups": mismatch_groups,
            "full_materialized_bytes": full_bytes,
            "exact_escalation_bytes": exact_bytes,
            "materialized_reduction_ratio": (full_bytes - exact_bytes)
            / full_bytes,
            "compact_fingerprint_transport_bytes_estimate": compact_bytes,
            "total_transport_reduction_ratio_estimate": (
                full_bytes - (compact_bytes + exact_bytes)
            )
            / full_bytes,
        },
        indent=2,
    )
)
PY
```

Expected core values: 9 files, 156 groups, 312 backend results, 33 escalated
groups, 492,081 full bytes, 62,499 exact-escalation bytes, 87.299% avoided
materialization and an estimated 65.420% total transport reduction.

This calculation is a structural shadow estimate. It is not the required
100,000-group correctness benchmark or paired throughput benchmark.
