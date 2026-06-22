from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plan_20plus_bug_triage.py"
    spec = importlib.util.spec_from_file_location("plan_20plus_bug_triage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rank_candidates_prioritizes_non_datafusion_single_backend_and_excludes_confirmed():
    module = _module()
    final = {
        "summary": {
            "rewardable_live_candidate_families": {
                "groupby_aggregation@duckdb": 21,
                "running_sum_precision@duckdb": 200,
                "groupby_aggregation@datafusion": 50,
                "coalesce_null_semantics@pyarrow": 8,
            },
            "known_saturated_live_candidate_families": {},
        }
    }
    bug_status = {
        "summary": {
            "recorded_fresh_candidate_families": {
                "groupby_aggregation@duckdb": 21,
                "coalesce_null_semantics@pyarrow": 8,
            }
        }
    }
    confirmations = {"confirmations": [{"family": "groupby_aggregation@datafusion"}]}

    rows = module.rank_candidates(final, bug_status=bug_status, confirmations=confirmations, limit=4)

    assert rows[0]["family"] == "groupby_aggregation@duckdb"
    assert rows[0]["priority"] == "P0"
    confirmed = next(row for row in rows if row["family"] == "groupby_aggregation@datafusion")
    assert confirmed["priority"] == "exclude_confirmed"
    noisy = next(row for row in rows if row["family"] == "running_sum_precision@duckdb")
    assert noisy["priority"] == "deprioritize"


def test_split_family_handles_multi_backend():
    module = _module()

    root, backends = module.split_family("coalesce_null_semantics@duckdb,pyarrow")

    assert root == "coalesce_null_semantics"
    assert backends == ["duckdb", "pyarrow"]
