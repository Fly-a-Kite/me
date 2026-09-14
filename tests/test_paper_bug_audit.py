"""Regression test for the paper-facing bug audit builder.

The audit joins the confirmation ledger, the canonical confirmed-bug corpus, and
the curated audit metadata. These assertions pin the paper-reported invariants so
the table cannot silently change when the inputs are edited.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts/paper/build_bug_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_bug_audit", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bug_audit_counts_and_fields() -> None:
    module = _load_module()
    audit = module.build(
        REPO_ROOT / "experiments/latest_confirmations.json",
        REPO_ROOT / "experiments/canonical_confirmed_bug_corpus/v2/manifest.json",
        REPO_ROOT / "experiments/bug_audit_metadata.json",
    )
    summary = audit["summary"]

    assert summary["confirmed_roots"] == 9
    assert summary["still_affected_on_latest"] == 4
    assert summary["pending_roots"] == 1
    assert summary["excluded_records"] == 1
    assert summary["distinct_root_cause_groups"] == 7
    assert summary["severity_weighted_score"] == 25
    assert summary["by_backend_family"] == {
        "query engine": 5,
        "DataFrame API": 2,
        "Arrow compute": 1,
        "embedded SQL": 1,
    }

    for root in audit["roots"]:
        assert root["severity"] in {"high", "medium", "low"}
        assert root["impact_class"]
        assert root["root_cause_group"]
        assert root["first_violated_component"]
        assert root["issue_url"].startswith("https://github.com/")
        # every confirmed root must carry an upstream acknowledgement or a fix
        assert root["upstream_status"] in {
            "upstream_labeled_bug",
            "fixed_upstream",
        }
        if root["still_affected_on_latest"] is False:
            assert root["fixed_by"] or root["fixed_versions"]


def test_bug_audit_excludes_invalid_and_pending() -> None:
    module = _load_module()
    audit = module.build(
        REPO_ROOT / "experiments/latest_confirmations.json",
        REPO_ROOT / "experiments/canonical_confirmed_bug_corpus/v2/manifest.json",
        REPO_ROOT / "experiments/bug_audit_metadata.json",
    )
    excluded = {row["family"] for row in audit["excluded"]}
    assert "polars_vector_division_rounding@polars" in excluded
    pending = {row["root_id"] for row in audit["pending"]}
    assert "datafusion-order-by-offset-subquery-groupby-001" in pending
    for row in audit["roots"]:
        assert row["root_id"] not in pending
