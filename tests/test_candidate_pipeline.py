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
    }
    evidence_file = generated_issue_dir / "fresh-candidates.json"
    dump_json(
        {
            "schema_version": "bug-hunt-fresh-candidates-v1",
            "generated_at": "2026-05-31T00:00:00Z",
            "generated_by": "datadiff bug-hunt",
            "source_run_file": "runs/run-fresh.jsonl.gz",
            "fresh_candidate_bug_families": {family: 1},
            "candidate_row_count": 1,
            "candidate_rows": [
                {
                    "case": case.to_dict(),
                    "findings": [finding],
                    "normalized": {
                        "pandas": {"backend": "pandas", "status": "ok", "columns": ["x"], "rows": [[1], [2]]},
                        "duckdb": {"backend": "duckdb", "status": "ok", "columns": ["x"], "rows": [[1], [3]]},
                    },
                    "raw_results": {
                        "pandas": {"status": "ok", "rows": [[1], [2]]},
                        "duckdb": {"status": "ok", "rows": [[1], [3]]},
                    },
                    "config": {"generator_profile": "bughunt_fresh"},
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
    assert manifest["summary"]["reproduced_count"] == 1
    assert manifest["summary"]["reduced_count"] == 1
    assert manifest["summary"]["needs_dedup_check_count"] == 1
    assert manifest["pipeline_issue_readiness_summary"]["needs_dedup_check_count"] == 1
    candidate = manifest["candidates"][0]
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
    assert (bugs_dir / "bug_sig-fresh" / "reduced_case.json").is_file()
    stored_config = json.loads((bugs_dir / "bug_sig-fresh" / "config.json").read_text(encoding="utf-8"))
    assert stored_config["freeze_strategy_snapshot"] is True
    assert stored_config["strategy_snapshot_path"]

    rendered = (tmp_path / manifest["markdown_path"]).read_text(encoding="utf-8")
    assert "## Candidates" in rendered
    assert "needs_dedup_check" in rendered


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
