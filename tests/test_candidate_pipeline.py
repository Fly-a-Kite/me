import json

from datadiff import candidate_pipeline, issue_readiness
from datadiff.candidate_pipeline import build_candidate_pipeline
from datadiff.config import DiscoveryBias
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.util import dump_json


def _issue_body(*, family: str) -> str:
    return f"""# Existing Draft

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `{family}` |
| First DataDiffFuzz signal | 2026-05-31T00:00:00Z |
| How found | DataDiffFuzz |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: local

## Reproducer

```python
print("repro")
```

## Expected Output

```text
expected
```

## Actual Output

```text
actual
```

## DataDiffFuzz Evidence

- Bug artifact: `bugs/bug_sig-fresh`
"""


def test_build_candidate_pipeline_freezes_rechecks_reduces_and_projects_issue_readiness(tmp_path, monkeypatch):
    new_issue_dir = tmp_path / "new_issue"
    generated_issue_dir = new_issue_dir / "generated"
    old_issue_dir = tmp_path / "old_issue"
    bugs_dir = tmp_path / "bugs"
    for path in (new_issue_dir, generated_issue_dir, old_issue_dir, bugs_dir):
        path.mkdir(parents=True, exist_ok=True)

    family = "fresh_family@duckdb"
    (new_issue_dir / "existing.md").write_text(_issue_body(family=family), encoding="utf-8")
    dump_json({"confirmations": []}, tmp_path / "experiments" / "latest_confirmations.json")

    case = Case(
        "case-fresh",
        7,
        [TableData("t0", [ColumnSpec("x", "int", nullable=False)], [{"x": 1}, {"x": 2}])],
        Program("prog-fresh", 7, [{"op": "select", "columns": ["x"]}]),
    )
    case.metadata["mutation"] = {
        "operator": "ir_pushdown_filter",
        "detail": "ir_pushdown_filter:test",
    }
    finding = {
        "finding_id": "f-1",
        "kind": "semantic_output_mismatch",
        "severity": "medium",
        "suspicious_backends": ["duckdb"],
        "evidence": "duckdb diverged",
        "signature": "sig-fresh",
        "root_cause": "fresh_family",
        "triage_verdict": "candidate_implementation_bug",
        "paper_status": "candidate_bug_needs_external_confirmation",
        "triage_confidence": "high",
        "false_positive": False,
        "mismatch_class": "row_order",
    }
    evidence_file = generated_issue_dir / "fresh-candidates.json"
    dump_json(
        {
            "schema_version": "discovery-run-fresh-candidates-v1",
            "generated_at": "2026-05-31T00:00:00Z",
            "generated_by": "datadiff discovery-run",
            "source_run_file": "runs/run-fresh.jsonl.gz",
            "fresh_candidate_bug_families": {family: 1},
            "candidate_row_count": 1,
            "candidate_rows": [
                {
                    "case": case.to_dict(),
                    "findings": [finding],
                    "mutation": {
                        "operator": "ir_pushdown_filter",
                        "detail": "ir_pushdown_filter:test",
                    },
                    "semantic_contract_lattice": {
                        "schema_version": "semantic-contract-lattice-v1",
                        "case_id": "case-fresh",
                        "boundary_axes": ["ordering"],
                        "strict_axes": ["null", "nan", "dtype_coercion"],
                        "contract_tags": ["ordering_boundary"],
                        "operation_contracts": [{"operation": "limit"}],
                    },
                    "normalized": {
                        "pandas": {"backend": "pandas", "status": "ok", "columns": ["x"], "rows": [[1], [2]]},
                        "duckdb": {"backend": "duckdb", "status": "ok", "columns": ["x"], "rows": [[1], [3]]},
                    },
                    "raw_results": {
                        "pandas": {"status": "ok", "rows": [[1], [2]]},
                        "duckdb": {"status": "ok", "rows": [[1], [3]]},
                    },
                    "config": {"generator_profile": "discovery_fresh"},
                    "candidate_recheck": {"enabled": True, "attempts": 2, "non_reproduced_keys": []},
                }
            ],
        },
        evidence_file,
    )

    monkeypatch.setattr(candidate_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(issue_readiness, "PROJECT_ROOT", tmp_path)

    def fake_save_bug_artifact(case, *, raw_results, normalized, findings, config=None):
        bug_dir = bugs_dir / "bug_sig-fresh"
        bug_dir.mkdir(parents=True, exist_ok=True)
        dump_json(case.to_dict(), bug_dir / "case.json")
        dump_json(raw_results, bug_dir / "results.json")
        dump_json(normalized, bug_dir / "normalized.json")
        dump_json([item.to_dict() for item in findings], bug_dir / "findings.json")
        dump_json(config or {}, bug_dir / "config.json")
        dump_json({"python": "3.12"}, bug_dir / "environment.json")
        (bug_dir / "reproduce.py").write_text("print('repro')\n", encoding="utf-8")
        return bug_dir

    monkeypatch.setattr(candidate_pipeline, "save_bug_artifact", fake_save_bug_artifact)

    def fake_run_loaded_case(case, *, backends, config, save_artifact=False):
        return {
            "status": "bug",
            "findings": [dict(finding)],
        }

    monkeypatch.setattr(candidate_pipeline, "run_loaded_case", fake_run_loaded_case)
    monkeypatch.setattr(candidate_pipeline, "reduce_case", lambda case, **kwargs: case)

    manifest = build_candidate_pipeline(
        evidence_files=[evidence_file],
        output_dir=generated_issue_dir / "candidate-pipelines",
        latest_confirmation_files=[tmp_path / "experiments" / "latest_confirmations.json"],
        new_issue_dir=new_issue_dir,
        old_issue_dir=old_issue_dir,
        generated_issue_dir=generated_issue_dir,
    )

    assert manifest["summary"]["candidate_count"] == 1
    assert manifest["summary"]["processed_candidate_count"] == 1
    assert manifest["summary"]["skipped_duplicate_candidate_count"] == 0
    assert manifest["bug_discovery_system"]["schema_version"] == "bug-discovery-system-v1"
    assert manifest["summary"]["reproduced_count"] == 1
    assert manifest["summary"]["reduced_count"] == 1
    assert manifest["summary"]["needs_dedup_check_count"] == 1
    assert manifest["summary"]["semantic_contract_candidate_count"] == 1
    assert manifest["summary"]["semantic_contract_boundary_axes"] == ["ordering"]
    assert manifest["summary"]["semantic_contract_matched_boundary_axes"] == ["ordering"]
    assert manifest["summary"]["ir_rewrite_candidate_count"] == 1
    assert manifest["summary"]["ir_rewrite_rules"] == ["ir.rewrite.filter_pushdown"]
    assert manifest["pipeline_issue_readiness_summary"]["needs_dedup_check_count"] == 1
    candidate = manifest["candidates"][0]
    assert candidate["candidate_acquisition"]["rewardable_candidate_finding_count"] == 1
    assert candidate["semantic_contract_evidence"]["matched_boundary_axes"] == ["ordering"]
    assert candidate["ir_rewrite_evidence"]["operators"] == ["ir_pushdown_filter"]
    assert candidate["dedup"]["status"] == "duplicate_local_family"
    assert candidate["recheck"]["reproduced"] is True
    assert candidate["triage"]["verdict"] == "candidate_implementation_bug"
    assert candidate["issue_readiness"]["readiness_status"] == "needs_dedup_check"
    assert candidate["strategy_learning_path"].endswith("candidate-pipeline-learning.json")
    assert (tmp_path / manifest["manifest_path"]).is_file()
    assert (tmp_path / manifest["markdown_path"]).is_file()
    assert (tmp_path / manifest["frozen_candidates_path"]).is_file()
    assert (tmp_path / manifest["strategy_snapshot_path"]).is_file()
    assert (tmp_path / candidate["issue_draft"]["path"]).is_file()
    frozen = json.loads((tmp_path / manifest["frozen_candidates_path"]).read_text(encoding="utf-8"))
    assert frozen["candidates"][0]["semantic_contract_evidence"]["boundary_axes"] == ["ordering"]
    assert frozen["candidates"][0]["ir_rewrite_evidence"]["rule_ids"] == ["ir.rewrite.filter_pushdown"]
    assert (bugs_dir / "bug_sig-fresh" / "reduced_case.json").is_file()
    stored_config = json.loads((bugs_dir / "bug_sig-fresh" / "config.json").read_text(encoding="utf-8"))
    assert stored_config["freeze_strategy_snapshot"] is True
    assert stored_config["strategy_snapshot_path"]

    rendered = (tmp_path / manifest["markdown_path"]).read_text(encoding="utf-8")
    assert "## Candidates" in rendered
    assert "True bug probability" in rendered
    assert "Skipped duplicate candidates" in rendered
    assert "Semantic-contract candidates" in rendered
    assert "IR rewrite candidates" in rendered
    assert "needs_dedup_check" in rendered
    issue_text = (tmp_path / candidate["issue_draft"]["path"]).read_text(encoding="utf-8")
    assert "Semantic contract boundary axes" in issue_text
    assert "IR rewrite rules" in issue_text


def test_candidate_pipeline_artifact_config_rehydrates_discovery_biases(tmp_path):
    bug_dir = tmp_path / "bugs" / "bug_sig-fresh"
    bug_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        {
            "generator_profile": "common_api_workflow",
            "discovery_biases": [
                {
                    "targets": ["common_api_workflow"],
                    "feature_prefixes": ["common_api_template:"],
                    "score_bonus": 0.5,
                    "novelty_bonus": 0.25,
                    "contribution_bonus": 0.2,
                    "candidate_pool_bonus": 0.1,
                    "keep_in_pool": True,
                }
            ],
        },
        bug_dir / "config.json",
    )

    config = candidate_pipeline._artifact_config_or_default(bug_dir, {})

    assert config.discovery_biases
    assert isinstance(config.discovery_biases[0], DiscoveryBias)
    assert config.discovery_biases[0].targets == ["common_api_workflow"]


def test_candidate_pipeline_ranks_high_proof_fresh_candidates_before_duplicates(tmp_path, monkeypatch):
    new_issue_dir = tmp_path / "new_issue"
    generated_issue_dir = new_issue_dir / "generated"
    old_issue_dir = tmp_path / "old_issue"
    for path in (new_issue_dir, generated_issue_dir, old_issue_dir):
        path.mkdir(parents=True, exist_ok=True)

    duplicate_family = "duplicate_family@duckdb"
    (new_issue_dir / "duplicate.md").write_text(_issue_body(family=duplicate_family), encoding="utf-8")
    dump_json({"confirmations": []}, tmp_path / "latest-confirmations.json")

    fresh_case = Case(
        "case-fresh",
        11,
        [TableData("t0", [ColumnSpec("x", "int", nullable=False)], [{"x": 1}, {"x": 2}, {"x": 3}])],
        Program("prog-fresh", 11, [{"op": "filter", "predicate": {"column": "x", "op": ">", "value": 1}}]),
    )
    duplicate_case = Case(
        "case-duplicate",
        12,
        [TableData("t0", [ColumnSpec("x", "int", nullable=False)], [{"x": 1}])],
        Program("prog-duplicate", 12, [{"op": "select", "columns": ["x"]}]),
    )
    fresh_finding = {
        "finding_id": "fresh",
        "kind": "semantic_output_mismatch",
        "suspicious_backends": ["duckdb"],
        "signature": "sig-fresh",
        "root_cause": "fresh_family",
        "triage_verdict": "candidate_implementation_bug",
        "false_positive": False,
    }
    duplicate_finding = {
        "finding_id": "duplicate",
        "kind": "semantic_output_mismatch",
        "suspicious_backends": ["duckdb"],
        "signature": "sig-duplicate",
        "root_cause": "duplicate_family",
        "triage_verdict": "candidate_implementation_bug",
        "false_positive": False,
    }
    evidence_file = generated_issue_dir / "fresh-candidates.json"
    dump_json(
        {
            "schema_version": "discovery-run-fresh-candidates-v1",
            "source_run_file": "runs/run-fresh.jsonl.gz",
            "fresh_candidate_bug_families": {"fresh_family@duckdb": 1, duplicate_family: 1},
            "candidate_row_count": 2,
            "candidate_rows": [
                {
                    "case": duplicate_case.to_dict(),
                    "findings": [duplicate_finding],
                    "candidate_recheck": {"attempts": 1, "non_reproduced_keys": [duplicate_family]},
                },
                {
                    "case": fresh_case.to_dict(),
                    "findings": [fresh_finding],
                    "normalized": {
                        "pandas": {"backend": "pandas", "status": "ok", "columns": ["x"], "rows": [[1], [2]]},
                        "duckdb": {"backend": "duckdb", "status": "ok", "columns": ["x"], "rows": [[1], [3]]},
                    },
                    "raw_results": {
                        "pandas": {"status": "ok", "rows": [[1], [2]]},
                        "duckdb": {"status": "ok", "rows": [[1], [3]]},
                    },
                    "bug_dir": "bugs/bug_sig-fresh",
                    "candidate_recheck": {"attempts": 2, "reproduced": True},
                },
            ],
        },
        evidence_file,
    )

    monkeypatch.setattr(candidate_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(issue_readiness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        candidate_pipeline,
        "_process_candidate",
        lambda **kwargs: {
            "candidate_id": kwargs["candidate_id"],
            "primary_family": kwargs["row"]["families"][0],
            "families": kwargs["row"]["families"],
            "candidate_acquisition": kwargs["row"]["candidate_acquisition"],
            "recheck": {"attempts": 0, "reproduced": False},
            "reduction": {},
            "triage": {},
            "dedup": {"status": "needs_final_upstream_dedup"},
            "issue_draft": {},
            "issue_readiness": {},
        },
    )

    manifest = build_candidate_pipeline(
        evidence_files=[evidence_file],
        output_dir=generated_issue_dir / "candidate-pipelines",
        latest_confirmation_files=[tmp_path / "latest-confirmations.json"],
        new_issue_dir=new_issue_dir,
        old_issue_dir=old_issue_dir,
        generated_issue_dir=generated_issue_dir,
        recheck_attempts=0,
        reduce_artifacts=False,
        standalone_reproducer=False,
    )

    assert [candidate["primary_family"] for candidate in manifest["candidates"]] == [
        "fresh_family@duckdb",
        duplicate_family,
    ]
    assert manifest["candidates"][0]["candidate_acquisition"]["acquisition_score"] > (
        manifest["candidates"][1]["candidate_acquisition"]["acquisition_score"]
    )
    frozen = json.loads((tmp_path / manifest["frozen_candidates_path"]).read_text(encoding="utf-8"))
    assert [candidate["families"][0] for candidate in frozen["candidates"]] == [
        "fresh_family@duckdb",
        duplicate_family,
    ]
    assert frozen["bug_discovery_system"]["schema_version"] == "bug-discovery-system-v1"


def test_candidate_pipeline_skips_duplicate_family_after_actionable_representative(tmp_path, monkeypatch):
    new_issue_dir, generated_issue_dir, old_issue_dir, evidence_file, family = _duplicate_family_pipeline_fixture(
        tmp_path,
        row_count=4,
    )
    processed_ids: list[str] = []

    def fake_process_candidate(**kwargs):
        processed_ids.append(kwargs["candidate_id"])
        row = kwargs["row"]
        return _fake_processed_candidate(
            candidate_id=kwargs["candidate_id"],
            row=row,
            actionable=True,
        )

    monkeypatch.setattr(candidate_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(issue_readiness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(candidate_pipeline, "_process_candidate", fake_process_candidate)

    manifest = build_candidate_pipeline(
        evidence_files=[evidence_file],
        output_dir=generated_issue_dir / "candidate-pipelines",
        latest_confirmation_files=[tmp_path / "latest-confirmations.json"],
        new_issue_dir=new_issue_dir,
        old_issue_dir=old_issue_dir,
        generated_issue_dir=generated_issue_dir,
        recheck_attempts=0,
        reduce_artifacts=False,
        standalone_reproducer=False,
    )

    assert len(processed_ids) == 1
    assert manifest["summary"]["candidate_count"] == 4
    assert manifest["summary"]["processed_candidate_count"] == 1
    assert manifest["summary"]["skipped_duplicate_candidate_count"] == 3
    assert [candidate["primary_family"] for candidate in manifest["candidates"]] == [family] * 4
    skipped = manifest["candidates"][1:]
    assert all(candidate["pipeline_processing"]["status"] == "skipped_duplicate_family" for candidate in skipped)
    assert {candidate["pipeline_processing"]["reason"] for candidate in skipped} == {
        "actionable_family_representative_exists"
    }
    assert all(candidate["dedup"]["pipeline_duplicate_of"] == processed_ids[0] for candidate in skipped)
    assert all(candidate["recheck"]["attempts"] == 0 for candidate in skipped)


def test_candidate_pipeline_caps_duplicate_family_expensive_processing(tmp_path, monkeypatch):
    new_issue_dir, generated_issue_dir, old_issue_dir, evidence_file, family = _duplicate_family_pipeline_fixture(
        tmp_path,
        row_count=5,
    )
    processed_ids: list[str] = []

    def fake_process_candidate(**kwargs):
        processed_ids.append(kwargs["candidate_id"])
        return _fake_processed_candidate(
            candidate_id=kwargs["candidate_id"],
            row=kwargs["row"],
            actionable=False,
        )

    monkeypatch.setattr(candidate_pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(issue_readiness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(candidate_pipeline, "CANDIDATE_PIPELINE_MAX_EXPENSIVE_ROWS_PER_FAMILY", 2)
    monkeypatch.setattr(candidate_pipeline, "_process_candidate", fake_process_candidate)

    manifest = build_candidate_pipeline(
        evidence_files=[evidence_file],
        output_dir=generated_issue_dir / "candidate-pipelines",
        latest_confirmation_files=[tmp_path / "latest-confirmations.json"],
        new_issue_dir=new_issue_dir,
        old_issue_dir=old_issue_dir,
        generated_issue_dir=generated_issue_dir,
        recheck_attempts=0,
        reduce_artifacts=False,
        standalone_reproducer=False,
    )

    assert len(processed_ids) == 2
    assert manifest["summary"]["candidate_count"] == 5
    assert manifest["summary"]["processed_candidate_count"] == 2
    assert manifest["summary"]["skipped_duplicate_candidate_count"] == 3
    assert [candidate["primary_family"] for candidate in manifest["candidates"]] == [family] * 5
    skipped = manifest["candidates"][2:]
    assert {candidate["pipeline_processing"]["reason"] for candidate in skipped} == {
        "family_expensive_processing_cap_reached"
    }
    assert all(candidate["dedup"]["pipeline_duplicate_of"] == processed_ids[0] for candidate in skipped)


def _duplicate_family_pipeline_fixture(tmp_path, *, row_count: int):
    new_issue_dir = tmp_path / "new_issue"
    generated_issue_dir = new_issue_dir / "generated"
    old_issue_dir = tmp_path / "old_issue"
    for path in (new_issue_dir, generated_issue_dir, old_issue_dir):
        path.mkdir(parents=True, exist_ok=True)
    dump_json({"confirmations": []}, tmp_path / "latest-confirmations.json")

    family_root = "duplicate_family"
    family = f"{family_root}@duckdb"
    rows = []
    for index in range(row_count):
        case = Case(
            f"case-duplicate-{index}",
            index,
            [TableData("t0", [ColumnSpec("x", "int", nullable=False)], [{"x": index}, {"x": index + 1}])],
            Program(f"prog-duplicate-{index}", index, [{"op": "select", "columns": ["x"]}]),
        )
        rows.append(
            {
                "case": case.to_dict(),
                "findings": [
                    {
                        "finding_id": f"duplicate-{index}",
                        "kind": "semantic_output_mismatch",
                        "suspicious_backends": ["duckdb"],
                        "signature": f"sig-duplicate-{index}",
                        "root_cause": family_root,
                        "triage_verdict": "candidate_implementation_bug",
                        "paper_status": "candidate_bug_needs_external_confirmation",
                        "triage_confidence": "high",
                        "false_positive": False,
                    }
                ],
                "normalized": {
                    "pandas": {"backend": "pandas", "status": "ok", "columns": ["x"], "rows": [[index]]},
                    "duckdb": {"backend": "duckdb", "status": "ok", "columns": ["x"], "rows": [[index + 1]]},
                },
                "raw_results": {
                    "pandas": {"status": "ok", "rows": [[index]]},
                    "duckdb": {"status": "ok", "rows": [[index + 1]]},
                },
                "candidate_recheck": {"attempts": 1, "reproduced": True},
            }
        )

    evidence_file = generated_issue_dir / "fresh-candidates.json"
    dump_json(
        {
            "schema_version": "discovery-run-fresh-candidates-v1",
            "source_run_file": "runs/run-duplicates.jsonl.gz",
            "fresh_candidate_bug_families": {family: row_count},
            "candidate_row_count": row_count,
            "candidate_rows": rows,
        },
        evidence_file,
    )
    return new_issue_dir, generated_issue_dir, old_issue_dir, evidence_file, family


def _fake_processed_candidate(*, candidate_id: str, row: dict, actionable: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "primary_family": row["families"][0],
        "families": row["families"],
        "candidate_acquisition": dict(row.get("candidate_acquisition", {}) or {}),
        "source_evidence_file": str(row.get("source_evidence_file", "")),
        "source_run_file": str(row.get("source_run_file", "")),
        "case_id": row.get("case", {}).get("case_id", ""),
        "semantic_contract_evidence": {},
        "ir_rewrite_evidence": {},
        "bug_dir": "",
        "artifact_created": False,
        "initial_candidate_recheck": dict(row.get("candidate_recheck", {}) or {}),
        "recheck": {"attempts": 1, "reproduced": False},
        "reduction": {"requested": False, "performed": False},
        "triage": {"verdict": "candidate_implementation_bug" if actionable else ""},
        "dedup": {"status": "needs_final_upstream_dedup", "pipeline_duplicate_of": ""},
        "issue_draft": {"path": f"new_issue/generated/{candidate_id}.md"} if actionable else {},
        "strategy_learning_path": "",
        "issue_readiness": {},
    }
