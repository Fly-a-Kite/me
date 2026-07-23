from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SOURCE_DIR = Path(
    "experiments/publication_baseline/v1/rlcmf_large96_v1"
)
DEFAULT_OUTPUT_DIR = Path("experiments/p7_rlcmf_baseline_v1")

SOURCE_ROLES = {
    "manifest": "manifest.json",
    "selection": "selection.json",
    "trace": "trace.jsonl",
    "result": "result.json",
    "analysis_original": "analysis.json",
    "analysis_recomputed": "analysis_recomputed.json",
    "report_original": "report.md",
    "runtime": "runtime.txt",
}

OUTPUT_FILES = (
    "baseline_index.json",
    "diagnostic_decomposition.json",
    "report.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _counter_rows(counter: Counter[Any], *, key_name: str) -> list[dict[str, Any]]:
    return [
        {key_name: key, "count": int(count)}
        for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
    ]


def _source_entries(repo_root: Path, source_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for role, name in SOURCE_ROLES.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            display_path = path.relative_to(repo_root).as_posix()
        except ValueError:
            display_path = path.as_posix()
        entries.append(
            {
                "role": role,
                "path": display_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(entries, key=lambda row: (row["role"], row["path"]))


def _arm_call_costs(result: Mapping[str, Any]) -> dict[str, int]:
    policy = result["policy"]
    backend_count = len(policy["backends"])
    low_backend_count = int(policy["low_backend_sample_size"])
    low_variant_count = int(policy["low_metamorphic_variant_limit"])
    reference_variant_count = int(policy["reference_metamorphic_variant_limit"])
    return {
        "low": low_backend_count * (1 + low_variant_count),
        "backend": backend_count * (1 + low_variant_count),
        "relation": low_backend_count * (1 + reference_variant_count),
        "joint": backend_count * (1 + reference_variant_count),
        "full_reference": backend_count * (1 + reference_variant_count),
    }


def _decompose_calls(
    case_results: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    arm_costs = _arm_call_costs(result)
    executed_cases: Counter[str] = Counter()
    selected_cases: Counter[str] = Counter()
    derived_cases: Counter[str] = Counter()
    calls_by_axis: Counter[str] = Counter()
    low_calls = 0
    audit_calls = 0
    reference_calls = 0
    reused_reference_cases = 0
    separate_reference_cases = 0
    separate_reference_calls = 0
    low_plus_joint_overlap_calls = 0

    for case in case_results:
        adaptive = case["adaptive"]
        observed_low_calls = int(adaptive["low_backend_calls"])
        if observed_low_calls != arm_costs["low"]:
            raise ValueError("low-path call count does not match the frozen policy")
        low_calls += observed_low_calls

        expected_audit_calls = 0
        for axis, metadata in adaptive["audit_arms"].items():
            if metadata.get("selected") is True:
                selected_cases[axis] += 1
            if metadata.get("derived_from_axis"):
                derived_cases[axis] += 1
            if metadata.get("executed") is True:
                if axis not in {"backend", "relation", "joint"}:
                    raise ValueError(f"unexpected historical audit axis: {axis}")
                executed_cases[axis] += 1
                expected_audit_calls += arm_costs[axis]
                calls_by_axis[axis] += arm_costs[axis]

        observed_audit_calls = int(adaptive["audit_backend_calls"])
        if observed_audit_calls != expected_audit_calls:
            raise ValueError(
                "audit call count cannot be reconciled from the frozen backend/MR policy"
            )
        audit_calls += observed_audit_calls

        observed_reference_calls = int(case["full_reference_accounting"]["backend_calls"])
        if observed_reference_calls != arm_costs["full_reference"]:
            raise ValueError("full-reference call count does not match the frozen policy")
        reference_calls += observed_reference_calls

        if case["full_reference_reused_from_joint_audit"] is True:
            if case["adaptive"]["audit_arms"]["joint"].get("executed") is not True:
                raise ValueError("full-reference reuse lacks an executed joint audit")
            reused_reference_cases += 1
            low_plus_joint_overlap_calls += observed_low_calls
        else:
            separate_reference_cases += 1
            separate_reference_calls += observed_reference_calls

    adaptive_calls = low_calls + audit_calls
    recorded_metrics = result["metrics"]
    if adaptive_calls != int(recorded_metrics["adaptive_backend_calls"]):
        raise ValueError("adaptive backend-call total does not reconcile")
    if reference_calls != int(recorded_metrics["full_reference_backend_calls"]):
        raise ValueError("reference backend-call total does not reconcile")
    if separate_reference_calls != int(
        recorded_metrics["validation_extra_backend_calls_excluded_from_adaptive"]
    ):
        raise ValueError("validation-excluded backend calls do not reconcile")

    reuse_savings = reused_reference_cases * arm_costs["full_reference"]
    return {
        "arm_call_formula": {
            "base_execution_per_backend": 1,
            "low_backend_count": int(result["policy"]["low_backend_sample_size"]),
            "reference_backend_count": len(result["policy"]["backends"]),
            "low_metamorphic_variant_limit": int(
                result["policy"]["low_metamorphic_variant_limit"]
            ),
            "reference_metamorphic_variant_limit": int(
                result["policy"]["reference_metamorphic_variant_limit"]
            ),
            "calls_per_arm": arm_costs,
        },
        "adaptive": {
            "low_path_calls": low_calls,
            "audit_calls": audit_calls,
            "combined_calls": adaptive_calls,
            "audit_calls_by_executed_axis": dict(sorted(calls_by_axis.items())),
            "selected_cases_by_axis": dict(sorted(selected_cases.items())),
            "executed_cases_by_axis": dict(sorted(executed_cases.items())),
            "derived_cases_by_axis": dict(sorted(derived_cases.items())),
            "low_path_calls_stacked_on_joint_full_audit": low_plus_joint_overlap_calls,
        },
        "reference": {
            "full_static_calls": reference_calls,
            "reused_from_joint_cases": reused_reference_cases,
            "separately_executed_validation_cases": separate_reference_cases,
            "validation_calls_excluded_from_adaptive": separate_reference_calls,
            "reference_calls_avoided_by_joint_reuse": reuse_savings,
        },
        "paired_harness": {
            "actual_executed_calls": adaptive_calls + separate_reference_calls,
            "calls_if_reference_never_reused": adaptive_calls + reference_calls,
            "calls_saved_by_reference_reuse": reuse_savings,
        },
        "reconciliation": {
            "adaptive_equals_low_plus_audits": adaptive_calls == low_calls + audit_calls,
            "adaptive_matches_frozen_result": adaptive_calls
            == int(recorded_metrics["adaptive_backend_calls"]),
            "reference_matches_frozen_result": reference_calls
            == int(recorded_metrics["full_reference_backend_calls"]),
            "all_pass": True,
        },
    }


def _decompose_time(
    case_results: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    low_backend_ms = sum(
        float(case["adaptive"]["low_backend_reported_ms"]) for case in case_results
    )
    audit_backend_ms = sum(
        float(case["adaptive"]["audit_backend_reported_ms"])
        for case in case_results
    )
    low_wall_ms = sum(float(case["adaptive"]["low_wall_ms"]) for case in case_results)
    audit_wall_ms = sum(
        float(case["adaptive"]["audit_wall_ms"]) for case in case_results
    )
    derivation_wall_ms = sum(
        float(case["adaptive"]["audit_derivation_wall_ms"])
        for case in case_results
    )
    arm_execution_wall_ms: Counter[str] = Counter()
    for case in case_results:
        for axis, metadata in case["adaptive"]["audit_arms"].items():
            if metadata.get("executed") is True:
                arm_execution_wall_ms[axis] += float(metadata.get("duration_ms", 0.0))

    joint_backend_ms = sum(
        float(case["full_reference_accounting"]["backend_reported_ms"])
        for case in case_results
        if case["adaptive"]["audit_arms"]["joint"].get("executed") is True
    )
    non_joint_backend_ms = audit_backend_ms - joint_backend_ms
    reference_backend_ms = sum(
        float(case["full_reference_accounting"]["backend_reported_ms"])
        for case in case_results
    )
    reference_wall_ms = sum(
        float(case["full_reference_accounting"]["wall_ms"]) for case in case_results
    )
    separate_reference_backend_ms = sum(
        float(case["full_reference_accounting"]["backend_reported_ms"])
        for case in case_results
        if case["full_reference_reused_from_joint_audit"] is not True
    )
    separate_reference_wall_ms = sum(
        float(case["full_reference_accounting"]["wall_ms"])
        for case in case_results
        if case["full_reference_reused_from_joint_audit"] is not True
    )

    metrics = result["metrics"]
    checks = {
        "adaptive_backend_reported_ms": _close(
            low_backend_ms + audit_backend_ms,
            metrics["adaptive_backend_reported_ms"],
        ),
        "adaptive_wall_ms": _close(
            low_wall_ms + audit_wall_ms,
            metrics["adaptive_accounted_wall_ms"],
        ),
        "reference_backend_reported_ms": _close(
            reference_backend_ms,
            metrics["full_reference_backend_reported_ms"],
        ),
        "reference_wall_ms": _close(
            reference_wall_ms,
            metrics["full_reference_wall_ms"],
        ),
        "validation_backend_reported_ms": _close(
            separate_reference_backend_ms,
            metrics["validation_extra_backend_reported_ms_excluded_from_adaptive"],
        ),
        "validation_wall_ms": _close(
            separate_reference_wall_ms,
            metrics["validation_extra_wall_ms_excluded_from_adaptive"],
        ),
        "audit_wall_components": _close(
            sum(arm_execution_wall_ms.values()) + derivation_wall_ms,
            audit_wall_ms,
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"historical time accounting does not reconcile: {checks}")

    return {
        "measurement_semantics": {
            "wall_ms": "accounted elapsed duration; not additive across concurrency in general",
            "backend_reported_ms": "sum of backend-reported operation durations; CPU proxy only",
            "process_cpu": "not recorded by the historical 96-case run",
        },
        "adaptive": {
            "low_backend_reported_ms": low_backend_ms,
            "audit_backend_reported_ms": audit_backend_ms,
            "joint_audit_backend_reported_ms": joint_backend_ms,
            "non_joint_audit_backend_reported_ms": non_joint_backend_ms,
            "combined_backend_reported_ms": low_backend_ms + audit_backend_ms,
            "low_wall_ms": low_wall_ms,
            "audit_wall_ms": audit_wall_ms,
            "audit_derivation_wall_ms": derivation_wall_ms,
            "audit_execution_wall_ms_by_axis": dict(sorted(arm_execution_wall_ms.items())),
            "combined_wall_ms": low_wall_ms + audit_wall_ms,
        },
        "reference": {
            "backend_reported_ms": reference_backend_ms,
            "wall_ms": reference_wall_ms,
            "validation_backend_reported_ms_excluded_from_adaptive": (
                separate_reference_backend_ms
            ),
            "validation_wall_ms_excluded_from_adaptive": separate_reference_wall_ms,
        },
        "process_cpu": {
            "available": False,
            "adaptive_process_cpu_s": None,
            "reference_process_cpu_s": None,
            "ratio": None,
            "reason": "historical source artifacts did not record process-tree CPU",
        },
        "reconciliation": {**checks, "all_pass": all(checks.values())},
    }


def _decompose_controller(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    states_before: Counter[str] = Counter()
    states_after: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    state_propensity: Counter[str] = Counter()
    requested_propensity: dict[str, Counter[str]] = {
        "backend": Counter(),
        "relation": Counter(),
        "joint": Counter(),
    }
    selected: Counter[str] = Counter()
    created_debt: Counter[str] = Counter()
    forced: Counter[str] = Counter()
    reasons_before: Counter[str] = Counter()
    reasons_after: Counter[str] = Counter()

    for case in case_results:
        before = str(case["safety_state_before"])
        after = str(case["safety_state_after"])
        states_before[before] += 1
        states_after[after] += 1
        transitions[f"{before}->{after}"] += 1
        for reason in case.get("safety_reasons_before", []):
            reasons_before[str(reason)] += 1
        for reason in case.get("safety_reasons_after", []):
            reasons_after[str(reason)] += 1

        plans = case["audit_plans"]
        propensity_key = ",".join(
            f"{axis}={plans[axis]['decision']['requested_propensity_exact']}"
            for axis in ("backend", "relation", "joint")
        )
        state_propensity[f"{before}|{propensity_key}"] += 1
        for axis, plan in plans.items():
            decision = plan["decision"]
            requested_propensity[axis][str(decision["requested_propensity_exact"])] += 1
            if plan["selected"] is True:
                selected[axis] += 1
            if plan.get("created_debt_id"):
                created_debt[axis] += 1
            if decision.get("forced") is True:
                forced[axis] += 1

    return {
        "state_before_cases": dict(sorted(states_before.items())),
        "state_after_cases": dict(sorted(states_after.items())),
        "state_transitions": dict(sorted(transitions.items())),
        "state_and_requested_propensity_cases": dict(sorted(state_propensity.items())),
        "requested_propensity_cases_by_axis": {
            axis: dict(sorted(rows.items()))
            for axis, rows in sorted(requested_propensity.items())
        },
        "selected_cases_by_axis": dict(sorted(selected.items())),
        "created_debt_records_by_axis": dict(sorted(created_debt.items())),
        "forced_audits_by_axis": {
            axis: int(forced.get(axis, 0)) for axis in ("backend", "relation", "joint")
        },
        "safety_reasons_before": dict(sorted(reasons_before.items())),
        "safety_reasons_after": dict(sorted(reasons_after.items())),
        "diagnosis": {
            "calibration_full_cases": int(states_before.get("calibration_full", 0)),
            "caution_cases": int(states_before.get("caution", 0)),
            "caution_half_propensity_cases": int(
                state_propensity.get("caution|backend=0.5,relation=0.5,joint=0.5", 0)
            ),
            "caution_full_propensity_cases": int(
                state_propensity.get("caution|backend=1,relation=1,joint=1", 0)
            ),
            "forced_debt_audits": int(sum(forced.values())),
            "interpretation": (
                "after calibration, caution used 0.5 propensities for only one "
                "eight-case epoch, then returned all three axes to propensity 1.0 "
                "while the low path continued to execute"
            ),
        },
    }


def build_diagnostic_decomposition(
    *,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    case_results = result.get("case_results")
    if not isinstance(case_results, list) or len(case_results) != 96:
        raise ValueError("P7.1 requires the frozen 96-case RLCMF result")
    manifest_sha256 = result["manifest_sha256"]
    if manifest_sha256 != analysis["manifest_sha256"]:
        raise ValueError("result/analysis manifest hash mismatch")
    if manifest["manifest_id"] != result["manifest_id"]:
        raise ValueError("manifest/result identifier mismatch")

    call_decomposition = _decompose_calls(case_results, result)
    time_decomposition = _decompose_time(case_results, result)
    controller_decomposition = _decompose_controller(case_results)
    metrics = analysis["metrics"]
    candidate = analysis["candidate_evidence"]

    zero_root_boundary = {
        "adaptive_candidate_family_count": len(
            result["metrics"]["adaptive_candidate_families"]
        ),
        "adaptive_candidate_root_count": len(
            result["metrics"]["adaptive_candidate_roots"]
        ),
        "full_reference_candidate_family_count": len(
            result["metrics"]["full_reference_candidate_families"]
        ),
        "full_reference_candidate_root_count": len(
            result["metrics"]["full_reference_candidate_roots"]
        ),
        "independently_confirmed_unique_real_roots_per_cpu_hour": result["metrics"][
            "independently_confirmed_unique_real_roots_per_cpu_hour"
        ],
        "analysis_candidate_evidence": candidate,
        "claim": (
            "zero candidate families/roots and unavailable independently confirmed "
            "root efficiency are preserved; no bug-yield claim is made"
        ),
    }

    return {
        "schema_version": "p7-rlcmf-negative-baseline-diagnostic-v1",
        "baseline_id": "p7-rlcmf-large96-negative-baseline-v1",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256,
        "case_count": len(case_results),
        "measurement_scope": result["measurement_scope"],
        "call_decomposition": call_decomposition,
        "time_decomposition": time_decomposition,
        "controller_decomposition": controller_decomposition,
        "paired_regression": {
            metric: {
                "adaptive_total": row["adaptive_total"],
                "reference_total": row["reference_total"],
                "ratio_of_totals": row["ratio_of_totals"],
                "ratio_confidence_interval": row["ratio_confidence_interval"],
                "decision": row["decision"],
            }
            for metric, row in sorted(metrics.items())
        },
        "negative_evidence_boundary": zero_root_boundary,
        "root_cause_diagnosis": {
            "primary": (
                "calibration/caution drove 94 of 96 cases through joint full audit "
                "while every case still paid the low-path cost"
            ),
            "secondary": (
                "two cases without joint audit required 98 separate full-reference "
                "validation calls; these were excluded from adaptive metrics"
            ),
            "not_established": [
                "process-CPU regression because process-tree CPU was not recorded",
                "candidate-axis omission effects because no common-batch executor existed",
                "independently confirmed real-root yield",
            ],
        },
        "holdout_use_restriction": (
            "diagnostic negative control only; do not tune P7 policies on these outcomes "
            "or reuse this trace as a positive promotion holdout"
        ),
    }


def _render_report(diagnostic: Mapping[str, Any]) -> str:
    calls = diagnostic["call_decomposition"]
    time = diagnostic["time_decomposition"]
    controller = diagnostic["controller_decomposition"]
    regressions = diagnostic["paired_regression"]
    boundary = diagnostic["negative_evidence_boundary"]
    audit_axis = calls["adaptive"]["audit_calls_by_executed_axis"]
    return f"""# P7.1 RLCMF 96-case negative baseline

This report is a deterministic diagnostic over the immutable publication
baseline. It is a negative control, not a tuning or promotion holdout.

## Reconciled calls

- Adaptive: {calls['adaptive']['combined_calls']:,} =
  {calls['adaptive']['low_path_calls']:,} low-path +
  {calls['adaptive']['audit_calls']:,} audit calls.
- Audit calls: {audit_axis.get('joint', 0):,} joint,
  {audit_axis.get('backend', 0):,} backend-only, and
  {audit_axis.get('relation', 0):,} relation-only.
- Full-static reference: {calls['reference']['full_static_calls']:,} calls.
- Joint/reference reuse avoided {calls['reference']['reference_calls_avoided_by_joint_reuse']:,}
  additional calls, but the two non-joint cases required
  {calls['reference']['validation_calls_excluded_from_adaptive']:,} separate
  validation calls.
- All reconciliation checks pass: {str(calls['reconciliation']['all_pass']).lower()}.

## Why adaptive regressed

The controller spent {controller['diagnosis']['calibration_full_cases']} cases in
`calibration_full`. In `caution`, only
{controller['diagnosis']['caution_half_propensity_cases']} cases used propensity
0.5; the remaining {controller['diagnosis']['caution_full_propensity_cases']}
used propensity 1.0. Consequently, joint/full audit executed on
{calls['adaptive']['executed_cases_by_axis']['joint']} of 96 cases while every
case still paid the low path. The low path stacked on joint/full audit accounts
for {calls['adaptive']['low_path_calls_stacked_on_joint_full_audit']:,} calls.

## Paired cost result

- Backend calls ratio: {regressions['backend_calls']['ratio_of_totals']:.6f},
  95% moving-block CI
  [{regressions['backend_calls']['ratio_confidence_interval']['lower']:.6f},
  {regressions['backend_calls']['ratio_confidence_interval']['upper']:.6f}].
- Backend-reported duration ratio:
  {regressions['backend_reported_ms']['ratio_of_totals']:.6f}, CI
  [{regressions['backend_reported_ms']['ratio_confidence_interval']['lower']:.6f},
  {regressions['backend_reported_ms']['ratio_confidence_interval']['upper']:.6f}].
- Accounted wall ratio: {regressions['wall_ms']['ratio_of_totals']:.6f}, CI
  [{regressions['wall_ms']['ratio_confidence_interval']['lower']:.6f},
  {regressions['wall_ms']['ratio_confidence_interval']['upper']:.6f}].
- Historical process CPU is unavailable. Backend-reported duration remains a
  work proxy and is not relabeled as process CPU.

## Evidence boundary

- Adaptive candidate families/roots:
  {boundary['adaptive_candidate_family_count']}/
  {boundary['adaptive_candidate_root_count']}.
- Full-reference candidate families/roots:
  {boundary['full_reference_candidate_family_count']}/
  {boundary['full_reference_candidate_root_count']}.
- Independently confirmed roots/CPU-hour: unavailable.
- Candidate/common-batch effects are not identified by this historical executor.

P7 must therefore add common-batch candidate auditing and exact process-tree CPU
accounting before a new, outcome-blind promotion experiment.
"""


def build_p7_rlcmf_baseline(
    *,
    repo_root: Path,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source_dir = (source_dir or repo_root / DEFAULT_SOURCE_DIR).resolve()
    output_dir = (output_dir or repo_root / DEFAULT_OUTPUT_DIR).resolve()
    source_entries = _source_entries(repo_root, source_dir)

    manifest = _read_json(source_dir / SOURCE_ROLES["manifest"])
    result = _read_json(source_dir / SOURCE_ROLES["result"])
    analysis = _read_json(source_dir / SOURCE_ROLES["analysis_recomputed"])
    diagnostic = build_diagnostic_decomposition(
        manifest=manifest,
        result=result,
        analysis=analysis,
    )

    source_archive_sha256 = hashlib.sha256(
        _canonical_json_bytes(source_entries)
    ).hexdigest()
    index = {
        "schema_version": "p7-rlcmf-negative-baseline-index-v1",
        "baseline_id": diagnostic["baseline_id"],
        "source_mode": "immutable_reference_to_publication_baseline",
        "source_directory": source_dir.relative_to(repo_root).as_posix(),
        "source_archive_sha256": source_archive_sha256,
        "source_files": source_entries,
        "generated_files": list(OUTPUT_FILES),
        "case_count": diagnostic["case_count"],
        "primary_decision": analysis["primary_decision"],
        "process_cpu_available": False,
        "candidate_common_batch_available": False,
        "zero_root_facts_preserved": True,
        "holdout_use_restriction": diagnostic["holdout_use_restriction"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "diagnostic_decomposition.json", diagnostic)
    _write_json(output_dir / "baseline_index.json", index)
    (output_dir / "report.md").write_text(
        _render_report(diagnostic),
        encoding="utf-8",
    )

    checksum_rows = source_entries + [
        {
            "role": f"generated_{name}",
            "path": (output_dir / name).relative_to(repo_root).as_posix(),
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in OUTPUT_FILES
    ]
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows),
        encoding="utf-8",
    )
    return index


def verify_p7_rlcmf_baseline(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = (output_dir or repo_root / DEFAULT_OUTPUT_DIR).resolve()
    index = _read_json(output_dir / "baseline_index.json")
    source_dir = (repo_root / index["source_directory"]).resolve()
    source_entries = _source_entries(repo_root, source_dir)
    if source_entries != index["source_files"]:
        raise ValueError("P7 source index differs from immutable baseline files")
    archive_digest = hashlib.sha256(_canonical_json_bytes(source_entries)).hexdigest()
    if archive_digest != index["source_archive_sha256"]:
        raise ValueError("P7 source archive digest mismatch")

    manifest = _read_json(source_dir / SOURCE_ROLES["manifest"])
    result = _read_json(source_dir / SOURCE_ROLES["result"])
    analysis = _read_json(source_dir / SOURCE_ROLES["analysis_recomputed"])
    expected_diagnostic = build_diagnostic_decomposition(
        manifest=manifest,
        result=result,
        analysis=analysis,
    )
    observed_diagnostic = _read_json(output_dir / "diagnostic_decomposition.json")
    if expected_diagnostic != observed_diagnostic:
        raise ValueError("P7 diagnostic decomposition is not reproducible")
    expected_report = _render_report(expected_diagnostic)
    if (output_dir / "report.md").read_text(encoding="utf-8") != expected_report:
        raise ValueError("P7 report is not reproducible")

    checksum_lines = (output_dir / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    checksums = {}
    for line in checksum_lines:
        digest, relative_path = line.split("  ", 1)
        checksums[relative_path] = digest
    expected_paths = {row["path"] for row in source_entries}
    expected_paths.update((output_dir / name).relative_to(repo_root).as_posix() for name in OUTPUT_FILES)
    if set(checksums) != expected_paths:
        raise ValueError("P7 checksum manifest path set mismatch")
    for relative_path, digest in checksums.items():
        if sha256_file(repo_root / relative_path) != digest:
            raise ValueError(f"P7 checksum mismatch: {relative_path}")
    return {
        "schema_version": "p7-rlcmf-negative-baseline-verification-v1",
        "baseline_id": index["baseline_id"],
        "source_file_count": len(source_entries),
        "generated_file_count": len(OUTPUT_FILES),
        "all_pass": True,
    }
