from copy import deepcopy

from datadiff import runner
from datadiff.run_signatures import (
    behavior_signature,
    coverage_discovery_signature,
    coverage_signal_signature,
    discovery_signature,
    row_count_bucket,
    signal_signature,
)


def _row(*, rows=None, finding_root="topk_ordering"):
    return {
        "case": {
            "program": {
                "operations": [
                    {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
                    {"op": "sort", "columns": ["x"], "ascending": True},
                    {"op": "limit", "n": 2},
                ]
            }
        },
        "normalized": {
            "pandas": {
                "status": "ok",
                "columns": ["x"],
                "rows": rows if rows is not None else [{"x": 1}, {"x": 2}],
            },
            "duckdb": {
                "status": "ok",
                "columns": ["x"],
                "rows": rows if rows is not None else [{"x": 1}, {"x": 2}],
            },
        },
        "findings": [
            {
                "kind": "semantic_output_mismatch",
                "root_cause": finding_root,
                "suspicious_backends": ["duckdb"],
            }
        ],
    }


def test_run_signatures_are_stable_and_runner_reexports_them():
    row = _row()

    assert behavior_signature(row) == behavior_signature(row)
    assert discovery_signature(row) == discovery_signature(row)
    assert signal_signature(row) == signal_signature(row)
    assert runner.behavior_signature is behavior_signature
    assert runner.discovery_signature is discovery_signature
    assert runner.signal_signature is signal_signature
    assert runner._row_count_bucket is row_count_bucket


def test_behavior_signature_tracks_result_sample_but_signal_signature_is_coarser():
    left = _row(rows=[{"x": 1}, {"x": 2}])
    right = _row(rows=[{"x": 1}, {"x": 99}])

    assert behavior_signature(left) != behavior_signature(right)
    assert discovery_signature(left) == discovery_signature(right)
    assert signal_signature(left) == signal_signature(right)


def test_discovery_signature_tracks_row_count_bucket_and_root_bucket():
    base = _row(rows=[{"x": 1}])
    same_bucket = _row(rows=[{"x": 1}, {"x": 2}, {"x": 3}])
    different_bucket = _row(rows=[{"x": index} for index in range(8)])
    different_root = _row(rows=[{"x": 1}], finding_root="join_null_semantics")

    assert row_count_bucket(0) == "0"
    assert row_count_bucket(1) == "1"
    assert row_count_bucket(3) == "2-3"
    assert row_count_bucket(8) == "8-15"
    assert discovery_signature(base) != discovery_signature(same_bucket)
    assert discovery_signature(base) != discovery_signature(different_bucket)
    assert discovery_signature(base) != discovery_signature(different_root)


def test_coverage_signatures_do_not_treat_backend_rotation_as_new_behavior():
    subset = _row()
    full = deepcopy(subset)
    full["normalized"]["sqlite"] = deepcopy(full["normalized"]["pandas"])

    assert discovery_signature(subset) != discovery_signature(full)
    assert signal_signature(subset) != signal_signature(full)
    assert coverage_discovery_signature(subset) == coverage_discovery_signature(full)
    assert coverage_signal_signature(subset) == coverage_signal_signature(full)
