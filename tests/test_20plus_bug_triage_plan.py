from __future__ import annotations

import importlib.util
import json
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
                "stable_value_contract@duckdb": 21,
                "running_sum_precision@duckdb": 200,
                "groupby_aggregation@datafusion": 50,
                "coalesce_null_semantics@pyarrow": 8,
                "similar_existing@pandas": 4,
            },
            "known_saturated_live_candidate_families": {},
        }
    }
    bug_status = {
        "summary": {
            "recorded_fresh_candidate_families": {
                "stable_value_contract@duckdb": 21,
                "coalesce_null_semantics@pyarrow": 8,
            }
        }
    }
    confirmations = {
        "confirmations": [
            {
                "family": "groupby_aggregation@datafusion",
                "discovery_credit": "datadiff_submitted",
            },
            {
                "family": "similar_existing@pandas",
                "discovery_credit": "similar_existing",
            },
        ]
    }

    rows = module.rank_candidates(final, bug_status=bug_status, confirmations=confirmations, limit=5)

    assert rows[0]["family"] == "coalesce_null_semantics@pyarrow"
    assert rows[0]["priority"] == "P0"
    stable = next(row for row in rows if row["family"] == "stable_value_contract@duckdb")
    assert stable["priority"] == "P0"
    confirmed = next(row for row in rows if row["family"] == "groupby_aggregation@datafusion")
    assert confirmed["priority"] == "exclude_confirmed"
    similar = next(row for row in rows if row["family"] == "similar_existing@pandas")
    assert similar["priority"] != "exclude_confirmed"
    noisy = next(row for row in rows if row["family"] == "running_sum_precision@duckdb")
    assert noisy["priority"] == "deprioritize"


def test_split_family_handles_multi_backend():
    module = _module()

    root, backends = module.split_family("coalesce_null_semantics@duckdb,pyarrow")

    assert root == "coalesce_null_semantics"
    assert backends == ["duckdb", "pyarrow"]


def test_rank_candidates_falls_back_to_bug_status_fresh_families_without_final_readiness():
    module = _module()
    bug_status = {
        "summary": {
            "recorded_fresh_candidate_families": {
                "coalesce_null_semantics@pyarrow": 9,
                "conditional_expression@duckdb": 500,
            }
        },
        "fresh_candidate_evidence": [
            {
                    "path": "new_issue/generated/fresh.json",
                    "source_run_file": "runs/run-a.jsonl.gz",
                    "generated_at": "2026-07-04T00:00:00Z",
                    "fresh_candidate_bug_families": {"coalesce_null_semantics@pyarrow": 9},
                }
            ],
        }

    rows = module.rank_candidates(
        {},
        bug_status=bug_status,
        confirmations={"confirmations": []},
        limit=2,
        candidate_source="auto",
    )

    assert rows[0]["family"] == "coalesce_null_semantics@pyarrow"
    assert rows[0]["priority"] == "P0"
    assert rows[0]["evidence_sources"][0]["path"] == "new_issue/generated/fresh.json"
    assert rows[1]["family"] == "conditional_expression@duckdb"
    assert rows[1]["priority"] == "deprioritize"


def test_rank_candidates_dynamically_downgrades_from_candidate_row_evidence(tmp_path):
    module = _module()
    module.PROJECT_ROOT = tmp_path
    evidence_path = tmp_path / "new_issue" / "generated" / "fresh.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "candidate_rows": [
                    _candidate_row(
                        "duplicate_schema@duckdb",
                        mismatch_class="value",
                        duplicate_schema=True,
                    ),
                    _candidate_row("row_order_only@duckdb", mismatch_class="row_order"),
                    _candidate_row(
                        "not_reproduced@duckdb",
                        mismatch_class="value",
                        candidate_recheck={"non_reproduced_keys": ["not_reproduced@duckdb"]},
                    ),
                    _candidate_row(
                        "source_issue_only@pandas",
                        mismatch_class="value",
                        source_issue="https://github.com/example/project/issues/1",
                    ),
                    _candidate_row(
                        "nullable_join_boundary@pandas",
                        mismatch_class="row_count",
                        nullable_join=True,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    bug_status = {
        "summary": {
            "recorded_fresh_candidate_families": {
                "duplicate_schema@duckdb": 2,
                "row_order_only@duckdb": 2,
                "not_reproduced@duckdb": 2,
                "source_issue_only@pandas": 2,
                "nullable_join_boundary@pandas": 2,
            }
        },
        "fresh_candidate_evidence": [
            {
                "path": "new_issue/generated/fresh.json",
                "fresh_candidate_bug_families": {
                    "duplicate_schema@duckdb": 2,
                    "row_order_only@duckdb": 2,
                    "not_reproduced@duckdb": 2,
                    "source_issue_only@pandas": 2,
                    "nullable_join_boundary@pandas": 2,
                },
            }
        ],
    }

    rows = module.rank_candidates(
        {},
        bug_status=bug_status,
        confirmations={"confirmations": []},
        limit=5,
        candidate_source="bug_status",
    )
    by_family = {row["family"]: row for row in rows}

    assert by_family["duplicate_schema@duckdb"]["priority"] == "deprioritize"
    assert "current_evidence_duplicate_column_schema_only" in by_family["duplicate_schema@duckdb"]["risks"]
    assert "unique-column schemas" in by_family["duplicate_schema@duckdb"]["next_action"]
    assert by_family["row_order_only@duckdb"]["priority"] == "deprioritize"
    assert "current_evidence_row_order_only_without_final_order_contract" in by_family["row_order_only@duckdb"]["risks"]
    assert by_family["not_reproduced@duckdb"]["priority"] == "deprioritize"
    assert "current_evidence_recheck_not_reproduced" in by_family["not_reproduced@duckdb"]["risks"]
    assert "current_evidence_source_issue_only" in by_family["source_issue_only@pandas"]["risks"]
    assert by_family["nullable_join_boundary@pandas"]["priority"] == "deprioritize"
    assert "current_evidence_nullable_join_key_boundary_only" in by_family["nullable_join_boundary@pandas"]["risks"]
    assert "nullable join-key" in by_family["nullable_join_boundary@pandas"]["next_action"]


def test_rank_candidates_dynamically_downgrades_from_outcome_ledger(tmp_path):
    module = _module()
    module.PROJECT_ROOT = tmp_path
    ledger_path = tmp_path / "reports" / "candidate-family-outcomes-current.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "candidate-family-outcomes-v1",
                "outcomes": [
                    {
                        "family": "arbitrary_adapter_family@duckdb",
                        "outcome": "local_adapter_false_positive",
                        "active": True,
                        "reason": "verified local lowering defect",
                    },
                    {
                        "family": "arbitrary_similar_family@pandas",
                        "outcome": "similar_existing_upstream_issue",
                        "active": True,
                        "upstream_issues": ["https://github.com/example/project/issues/1"],
                    },
                    {
                        "family": "inactive_family@polars",
                        "outcome": "latest_recheck_no_longer_reproduces",
                        "active": False,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    final = {
        "summary": {
            "rewardable_live_candidate_families": {
                "arbitrary_adapter_family@duckdb": 10,
                "arbitrary_similar_family@pandas": 10,
                "inactive_family@polars": 10,
            },
            "known_saturated_live_candidate_families": {},
        }
    }
    outcomes = module.load_candidate_outcomes(ledger_path)

    rows = module.rank_candidates(
        final,
        bug_status={},
        confirmations={"confirmations": []},
        candidate_outcomes=outcomes,
        limit=3,
    )
    by_family = {row["family"]: row for row in rows}

    adapter = by_family["arbitrary_adapter_family@duckdb"]
    assert adapter["priority"] == "deprioritize"
    assert "outcome_local_adapter_false_positive" in adapter["risks"]
    assert "local adapter false positive" in adapter["next_action"]
    similar = by_family["arbitrary_similar_family@pandas"]
    assert similar["priority"] == "deprioritize"
    assert "outcome_similar_existing_upstream_issue" in similar["risks"]
    assert "duplicate" in similar["next_action"]
    assert "outcome_latest_recheck_no_longer_reproduces" not in by_family["inactive_family@polars"]["risks"]


def _candidate_row(
    family: str,
    *,
    mismatch_class: str,
    duplicate_schema: bool = False,
    nullable_join: bool = False,
    source_issue: str = "",
    candidate_recheck: dict[str, object] | None = None,
) -> dict[str, object]:
    root, backends = family.split("@", 1)
    columns = [{"name": "x", "type": "int", "nullable": nullable_join}]
    if duplicate_schema:
        columns.append({"name": "x", "type": "int", "nullable": False})
    tables = [{"name": "t0", "columns": columns, "rows": [{"x": 1}]}]
    operations = []
    if nullable_join:
        tables.append(
            {
                "name": "t1",
                "columns": [{"name": "x", "type": "int", "nullable": True}],
                "rows": [{"x": 1}],
            }
        )
        operations.append({"op": "join", "table": "t1", "left_on": "x", "right_on": "x", "how": "inner"})
    return {
        "case": {
            "case_id": f"case-{root}",
            "seed": 1,
            "tables": tables,
            "program": {"program_id": f"prog-{root}", "seed": 1, "operations": operations},
        },
        "findings": [
            {
                "root_cause": root,
                "suspicious_backends": backends.split(","),
                "mismatch_class": mismatch_class,
                "source_issue": source_issue,
            }
        ],
        "candidate_recheck": candidate_recheck or {},
    }
