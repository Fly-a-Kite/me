from __future__ import annotations

from pathlib import Path


def write_reduced_reproducer(bug_dir: Path, backends: list[str]) -> None:
    repro = f'''#!/usr/bin/env python3
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.runner import run_loaded_case
from datadiff.util import load_json

here = __import__("pathlib").Path(__file__).parent
case = Case.from_dict(load_json(here / "reduced_case.json"))
config_data = load_json(here / "config.json")
config = ExperimentConfig.from_payload(config_data)
result = run_loaded_case(case, backends={backends!r}, config=config, save_artifact=False)
print(result["status"])
for finding in result["findings"]:
    print(finding)
'''
    path = bug_dir / "reproduce_reduced.py"
    path.write_text(repro, encoding="utf-8")
    path.chmod(0o755)


def write_evidence_queue_reproducer(queue_json_path: Path, script_path: Path | None = None) -> Path:
    """Write a batch rerunner for a reproducer-queue JSON artifact."""
    queue_json_path = Path(queue_json_path)
    script_path = Path(script_path) if script_path is not None else queue_json_path.with_name(
        f"{queue_json_path.stem}_reproduce.py"
    )
    if queue_json_path.parent.resolve() == script_path.parent.resolve():
        default_queue_expr = f'Path(__file__).with_name({queue_json_path.name!r})'
    else:
        default_queue_expr = f"Path({str(queue_json_path)!r})"
    repro = f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.run_findings import _finding_recheck_key, _format_recheck_key
from datadiff.runner import run_loaded_case
from datadiff.witness_oracle import evaluate_witness_contract


DEFAULT_QUEUE = {default_queue_expr}


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    queue_path = Path(str(args.queue))
    payload = load_json(queue_path)
    rows = selected_rows(payload, families=set(args.family or []), limit=int(args.limit or 0))
    results = [validate_row(row) for row in rows]
    summary = summarize(results)
    output = {{
        "schema_version": "evidence-queue-reproducer-results-v1",
        "generated_at": utc_now_iso(),
        "queue": str(queue_path),
        "filters": {{
            "families": sorted(set(args.family or [])),
            "limit": int(args.limit or 0),
        }},
        "summary": summary,
        "results": results,
    }}
    output_json = Path(str(args.output_json)) if str(args.output_json or "").strip() else queue_path.with_name(
        f"{{queue_path.stem}}-validation.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(
        "validated {{candidate_count}} queue rows: {{passed}} passed, {{failed}} failed, {{errors}} errors".format(
            candidate_count=summary["candidate_count"],
            passed=summary["passed_count"],
            failed=summary["failed_count"],
            errors=summary["error_count"],
        )
    )
    print(f"validation json: {{output_json}}")
    if args.allow_nonreproduced:
        return 0
    return 1 if summary["failed_count"] or summary["error_count"] else 0


def selected_rows(payload: dict[str, Any], *, families: set[str], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    grouped = payload.get("families", {{}})
    if not isinstance(grouped, dict):
        return result
    for family, rows in grouped.items():
        family_text = str(family)
        if families and family_text not in families:
            continue
        family_rows = [row for row in rows or [] if isinstance(row, dict)]
        if limit > 0:
            family_rows = family_rows[:limit]
        result.extend(family_rows)
    return result


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    base = {{
        "family": str(row.get("family", "")),
        "case_id": str(row.get("case_id", "")),
        "source_artifact": str(row.get("source_artifact", "")),
        "source_row_index": row.get("source_row_index", ""),
        "verification_backends": list(row.get("verification_backends", []) or []),
        "expected_finding_keys": list(row.get("expected_finding_keys", []) or []),
    }}
    try:
        case_data = row.get("case", {{}})
        if not isinstance(case_data, dict) or not case_data:
            return {{**base, "status": "error", "error": "missing case payload"}}
        backends = [str(item) for item in row.get("verification_backends", []) or [] if str(item)]
        if not backends:
            return {{**base, "status": "error", "error": "missing verification backends"}}
        config_data = dict(row.get("config", {{}}) if isinstance(row.get("config", {{}}), dict) else {{}})
        config_data["candidate_recheck_count"] = 0
        config_data["enable_artifact"] = False
        case = Case.from_dict(case_data)
        attach_witness_plan(case, row)
        result = run_loaded_case(
            case,
            backends=backends,
            config=ExperimentConfig.from_payload(config_data),
            save_artifact=False,
        )
        findings = [finding for finding in result.get("findings", []) or [] if isinstance(finding, dict)]
        actual_keys = sorted({{_format_recheck_key(_finding_recheck_key(finding)) for finding in findings}})
        expected_keys = list(base["expected_finding_keys"])
        reproduced = sorted(set(expected_keys).intersection(actual_keys))
        missing = sorted(set(expected_keys) - set(actual_keys))
        witness_validation = validate_witness_plan(case, row, result)
        finding_passed = bool(expected_keys) and not missing
        witness_passed = witness_validation.get("status") in {{"not_available", "passed"}}
        status = "passed" if finding_passed and witness_passed else "failed"
        return {{
            **base,
            "status": status,
            "run_status": str(result.get("status", "")),
            "finding_count": len(findings),
            "actual_finding_keys": actual_keys,
            "reproduced_finding_keys": reproduced,
            "missing_finding_keys": missing,
            "witness_validation": witness_validation,
        }}
    except Exception as exc:
        return {{
            **base,
            "status": "error",
            "error": f"{{type(exc).__name__}}: {{exc}}",
            "traceback": traceback.format_exc(limit=12),
        }}


def attach_witness_plan(case: Case, row: dict[str, Any]) -> None:
    witness_plan = row.get("witness_plan", {{}})
    if not isinstance(witness_plan, dict) or witness_plan.get("status") != "available":
        return
    contract = witness_plan.get("contract", {{}})
    if not isinstance(contract, dict) or not contract:
        return
    metadata = dict(case.metadata or {{}})
    metadata["witness_contract"] = dict(contract)
    case.metadata = metadata


def validate_witness_plan(case: Case, row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    witness_plan = row.get("witness_plan", {{}})
    if not isinstance(witness_plan, dict) or witness_plan.get("status") != "available":
        return {{"status": "not_available", "reason": "queue row has no available witness plan"}}
    normalized = result.get("normalized", {{}})
    if not isinstance(normalized, dict) or not normalized:
        return {{"status": "failed", "reason": "rerun result has no normalized backend outputs"}}
    witness_result = evaluate_witness_contract(case, normalized, enabled=False).to_dict()
    expected_failing = sorted(str(item) for item in witness_plan.get("failing_suspicious_backends", []) or [])
    actual_failing = sorted(str(item) for item in witness_result.get("failing_backends", []) or [])
    expected_satisfied = sorted(str(item) for item in witness_plan.get("satisfied_reference_backends", []) or [])
    actual_satisfied = sorted(str(item) for item in witness_result.get("satisfied_backends", []) or [])
    missing_failures = sorted(set(expected_failing) - set(actual_failing))
    missing_satisfied = sorted(set(expected_satisfied) - set(actual_satisfied))
    status = "passed" if expected_failing and not missing_failures and not missing_satisfied else "failed"
    return {{
        "status": status,
        "expected_failing_suspicious_backends": expected_failing,
        "actual_failing_backends": actual_failing,
        "expected_satisfied_reference_backends": expected_satisfied,
        "actual_satisfied_backends": actual_satisfied,
        "missing_expected_failures": missing_failures,
        "missing_expected_satisfied": missing_satisfied,
        "contract": witness_result.get("contract", {{}}),
    }}


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    return {{
        "candidate_count": len(results),
        "passed_count": sum(1 for row in results if row.get("status") == "passed"),
        "failed_count": sum(1 for row in results if row.get("status") == "failed"),
        "error_count": sum(1 for row in results if row.get("status") == "error"),
        "family_count": len({{str(row.get("family", "")) for row in results if row.get("family")}}),
        "witness_passed_count": sum(
            1
            for row in results
            if isinstance(row.get("witness_validation"), dict)
            and row.get("witness_validation", {{}}).get("status") == "passed"
        ),
        "witness_failed_count": sum(
            1
            for row in results
            if isinstance(row.get("witness_validation"), dict)
            and row.get("witness_validation", {{}}).get("status") == "failed"
        ),
    }}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {{path}}")
    return data


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-rerun DataDiffFuzz evidence-queue candidates.")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="optional per-family candidate limit; 0 means all")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--allow-nonreproduced", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
'''
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(repro, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path
