import os
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

from datadiff.reproducer_scripts import write_evidence_queue_reproducer, write_reduced_reproducer


def test_write_reduced_reproducer_writes_executable_script(tmp_path):
    write_reduced_reproducer(tmp_path, ["duckdb", "sqlite"])

    script = tmp_path / "reproduce_reduced.py"
    text = script.read_text(encoding="utf-8")

    assert script.is_file()
    assert os.access(script, os.X_OK)
    assert "Case.from_dict" in text
    assert "run_loaded_case" in text
    assert "backends=['duckdb', 'sqlite']" in text


def test_write_evidence_queue_reproducer_writes_executable_batch_validator(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps(_queue_payload(expected_root="union_all_row_append")), encoding="utf-8")

    script = write_evidence_queue_reproducer(queue)
    text = script.read_text(encoding="utf-8")

    assert script.is_file()
    assert os.access(script, os.X_OK)
    assert "Batch-rerun DataDiffFuzz evidence-queue candidates" in text
    assert "run_loaded_case" in text
    assert "evidence-queue-reproducer-results-v1" in text


def test_generated_evidence_queue_reproducer_validates_expected_finding_keys(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps(_queue_payload(expected_root="union_all_row_append")), encoding="utf-8")
    script = write_evidence_queue_reproducer(queue)
    module = _import_script(script)

    def fake_run_loaded_case(case, *, backends, config, save_artifact):
        return {
            "status": "bug_candidate",
            "normalized": {
                "duckdb": {"status": "ok", "columns": ["x"], "rows": [[1]]},
                "sqlite": {"status": "ok", "columns": ["x"], "rows": [[2]]},
            },
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "union_all_row_append",
                    "suspicious_backends": ["duckdb"],
                    "mismatch_class": "value",
                }
            ],
        }

    module.run_loaded_case = fake_run_loaded_case
    output = tmp_path / "validation.json"

    rc = module.run_with_args(Namespace(queue=str(queue), family=[], limit=0, output_json=str(output), allow_nonreproduced=False))

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["passed_count"] == 1
    assert payload["summary"]["witness_passed_count"] == 1
    assert payload["results"][0]["reproduced_finding_keys"] == [
        "semantic_output_mismatch:union_all_row_append@duckdb:value"
    ]
    assert payload["results"][0]["witness_validation"]["status"] == "passed"


def test_generated_evidence_queue_reproducer_fails_when_candidate_does_not_reproduce(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps(_queue_payload(expected_root="union_all_row_append")), encoding="utf-8")
    script = write_evidence_queue_reproducer(queue)
    module = _import_script(script)
    module.run_loaded_case = lambda case, *, backends, config, save_artifact: {"status": "ok", "findings": []}

    output = tmp_path / "validation.json"
    rc = module.run_with_args(Namespace(queue=str(queue), family=[], limit=0, output_json=str(output), allow_nonreproduced=False))

    assert rc == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["failed_count"] == 1
    assert payload["results"][0]["missing_finding_keys"] == [
        "semantic_output_mismatch:union_all_row_append@duckdb:value"
    ]


def test_generated_evidence_queue_reproducer_fails_when_witness_does_not_reproduce(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps(_queue_payload(expected_root="union_all_row_append")), encoding="utf-8")
    script = write_evidence_queue_reproducer(queue)
    module = _import_script(script)

    def fake_run_loaded_case(case, *, backends, config, save_artifact):
        return {
            "status": "bug_candidate",
            "normalized": {
                "duckdb": {"status": "ok", "columns": ["x"], "rows": [[2]]},
                "sqlite": {"status": "ok", "columns": ["x"], "rows": [[2]]},
            },
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "union_all_row_append",
                    "suspicious_backends": ["duckdb"],
                    "mismatch_class": "value",
                }
            ],
        }

    module.run_loaded_case = fake_run_loaded_case
    output = tmp_path / "validation.json"

    rc = module.run_with_args(
        Namespace(queue=str(queue), family=[], limit=0, output_json=str(output), allow_nonreproduced=False)
    )

    assert rc == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["failed_count"] == 1
    assert payload["summary"]["witness_failed_count"] == 1
    assert payload["results"][0]["witness_validation"]["missing_expected_failures"] == ["duckdb"]


def _import_script(path: Path):
    spec = importlib.util.spec_from_file_location("generated_queue_reproducer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _queue_payload(*, expected_root: str) -> dict[str, object]:
    return {
        "schema_version": "reproducer-queue-v1",
        "families": {
            f"{expected_root}@duckdb": [
                {
                    "family": f"{expected_root}@duckdb",
                    "case_id": "case-1",
                    "case": {
                        "case_id": "case-1",
                        "seed": 1,
                        "tables": [],
                        "program": {"program_id": "program-1", "seed": 1, "operations": []},
                    },
                    "config": {"candidate_recheck_count": 2},
                    "verification_backends": ["duckdb", "sqlite"],
                    "source_artifact": "rows/family.jsonl",
                    "source_row_index": 7,
                    "expected_finding_keys": [f"semantic_output_mismatch:{expected_root}@duckdb:value"],
                    "witness_plan": {
                        "schema_version": "datadiff-reference-witness-plan-v1",
                        "status": "available",
                        "contract": {
                            "schema_version": "datadiff-witness-contract-v1",
                            "kind": "row_containment",
                            "row": {"x": 2},
                            "backends": ["duckdb", "sqlite"],
                            "reason": "reference backends agree this output row should be present and suspicious backend omits it",
                            "source": "reference_consensus",
                        },
                        "satisfied_reference_backends": ["sqlite"],
                        "failing_suspicious_backends": ["duckdb"],
                    },
                }
            ]
        },
    }
