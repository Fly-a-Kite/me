from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.declared_subset_corpus import build_declared_subset_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.env import collect_environment
from datadiff.execution import BackendExecutionSession
from datadiff.experiment_manifest import stable_digest
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding, evaluate_case
from datadiff.targets import TARGETS, production_target_ids
from datadiff.util import JsonlWriter, dump_json, utc_now


LARGE_SCALE_AUDIT_SCHEMA_VERSION = "large-scale-audit-v1"
GENERAL_EXPECTATION = "agreement"
FAULT_EXPECTATION = "seeded_fault_detection"


@dataclass(frozen=True, slots=True)
class AuditCohort:
    name: str
    targets: tuple[str, ...]
    case_count: int
    seed_start: int
    generator: str
    expectation: str = GENERAL_EXPECTATION
    fault_target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": list(self.targets),
            "case_count": self.case_count,
            "seed_start": self.seed_start,
            "generator": self.generator,
            "expectation": self.expectation,
            "fault_target": self.fault_target,
        }


def build_audit_plan(
    *,
    seed_start: int = 3_000_000,
    production_cases: int = 10_000,
    storage_cases: int = 2_000,
    numpy_cases: int = 5_000,
    fault_cases: int = 1_000,
) -> tuple[AuditCohort, ...]:
    counts = [production_cases, storage_cases, numpy_cases, fault_cases]
    if any(int(value) < 0 for value in counts):
        raise ValueError("audit case counts must be non-negative")
    offset = int(seed_start)
    cohorts: list[AuditCohort] = []

    cohorts.append(
        AuditCohort(
            "production_generalization",
            production_target_ids(),
            int(production_cases),
            offset,
            "typed_grammar",
        )
    )
    offset += int(production_cases)
    cohorts.append(
        AuditCohort(
            "duckdb_storage_generalization",
            ("pandas", "duckdb", "duckdb_persistent"),
            int(storage_cases),
            offset,
            "typed_grammar",
        )
    )
    offset += int(storage_cases)
    cohorts.append(
        AuditCohort(
            "numpy_declared_subset",
            ("pandas", "numpy"),
            int(numpy_cases),
            offset,
            "declared_subset",
        )
    )
    offset += int(numpy_cases)
    for fault in ("filter", "groupby", "join", "mutate"):
        target = f"buggy_{fault}"
        cohorts.append(
            AuditCohort(
                f"seeded_{fault}_detection",
                ("pandas", target),
                int(fault_cases),
                offset,
                f"seeded_{fault}",
                expectation=FAULT_EXPECTATION,
                fault_target=target,
            )
        )
        offset += int(fault_cases)
    return tuple(cohorts)


def run_large_scale_audit(
    output_dir: Path,
    *,
    cohorts: Sequence[AuditCohort],
    batch_size: int = 50,
    parallel_backends: bool = True,
    minimum_cases_per_target: int = 1_000,
    anomaly_sample_limit_per_cohort: int = 100,
    baseline_log: Path | None = None,
    baseline_runtime_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = output_dir.resolve()
    result_path = output / "result.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(result_path)
    output.mkdir(parents=True, exist_ok=True)
    anomaly_path = output / "anomalies.jsonl"
    if overwrite and anomaly_path.exists():
        anomaly_path.unlink()

    resolved_cohorts = tuple(cohort for cohort in cohorts if cohort.case_count > 0)
    _validate_plan(resolved_cohorts)
    protocol = {
        "schema_version": LARGE_SCALE_AUDIT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "batch_size": max(1, int(batch_size)),
        "parallel_backends": bool(parallel_backends),
        "minimum_cases_per_target": max(0, int(minimum_cases_per_target)),
        "anomaly_sample_limit_per_cohort": max(0, int(anomaly_sample_limit_per_cohort)),
        "cohorts": [cohort.to_dict() for cohort in resolved_cohorts],
    }
    dump_json(protocol, output / "protocol.json")

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_io = _process_io()
    aggregate_targets: dict[str, Counter[str]] = defaultdict(Counter)
    aggregate_durations: dict[str, list[float]] = defaultdict(list)
    cohort_results: list[dict[str, Any]] = []
    normal_digest = hashlib.sha256()
    config = _audit_config(parallel_backends=parallel_backends)

    with JsonlWriter(
        anomaly_path,
        mode="wt",
        sort_keys=True,
        buffer_lines=16,
    ) as anomaly_writer:
        for cohort in resolved_cohorts:
            cohort_results.append(
                _run_cohort(
                    cohort,
                    config=config,
                    batch_size=max(1, int(batch_size)),
                    anomaly_writer=anomaly_writer,
                    anomaly_limit=max(0, int(anomaly_sample_limit_per_cohort)),
                    aggregate_targets=aggregate_targets,
                    aggregate_durations=aggregate_durations,
                    normal_digest=normal_digest,
                )
            )

    runtime_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    process_io = _counter_delta(_process_io(), started_io)
    total_cases = sum(cohort.case_count for cohort in resolved_cohorts)
    total_backend_calls = sum(int(row["backend_calls"]) for row in cohort_results)
    target_rows = _aggregate_target_rows(aggregate_targets, aggregate_durations)
    assessment = _build_assessment(
        cohorts=resolved_cohorts,
        cohort_results=cohort_results,
        target_rows=target_rows,
        minimum_cases_per_target=max(0, int(minimum_cases_per_target)),
    )
    baseline = _baseline_metrics(
        baseline_log,
        baseline_runtime_seconds=baseline_runtime_seconds,
    )
    anomaly_bytes = anomaly_path.stat().st_size if anomaly_path.exists() else 0
    io_comparison = _io_comparison(
        baseline,
        anomaly_bytes=anomaly_bytes,
        total_cases=total_cases,
        cohort_results=cohort_results,
    )
    result: dict[str, Any] = {
        "schema_version": LARGE_SCALE_AUDIT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "protocol_digest": stable_digest("large-scale-audit-protocol", protocol),
        "environment": collect_environment(),
        "summary": {
            "cohort_count": len(resolved_cohorts),
            "target_count": len(target_rows),
            "case_count": total_cases,
            "backend_calls": total_backend_calls,
            "runtime_seconds": runtime_seconds,
            "cpu_seconds": cpu_seconds,
            "cases_per_second": total_cases / runtime_seconds if runtime_seconds else 0.0,
            "backend_calls_per_second": total_backend_calls / runtime_seconds if runtime_seconds else 0.0,
            "normal_stream_digest": normal_digest.hexdigest(),
        },
        "process": {
            "io_delta": process_io,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "current_rss_kib": _current_rss_kib(),
        },
        "cohorts": cohort_results,
        "targets": target_rows,
        "assessment": assessment,
        "baseline": baseline,
        "io_comparison": io_comparison,
    }
    result["result_digest"] = stable_digest("large-scale-audit-result", result)
    dump_json(result, result_path)
    (output / "report.md").write_text(_render_report(result), encoding="utf-8")
    _update_artifact_metrics(result, output)
    dump_json(result, result_path)
    (output / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def _run_cohort(
    cohort: AuditCohort,
    *,
    config: ExperimentConfig,
    batch_size: int,
    anomaly_writer: JsonlWriter,
    anomaly_limit: int,
    aggregate_targets: dict[str, Counter[str]],
    aggregate_durations: dict[str, list[float]],
    normal_digest: Any,
) -> dict[str, Any]:
    status_counts = {target: Counter() for target in cohort.targets}
    durations = {target: [] for target in cohort.targets}
    finding_kinds: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    anomaly_cases = 0
    persisted_anomalies = 0
    priority_anomalies = 0
    fault_detections = 0
    cohort_digest = hashlib.sha256()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_io = _process_io()

    session = BackendExecutionSession(
        list(cohort.targets),
        cache_limit=0,
        cache_max_bytes=0,
    )
    try:
        for batch_start in range(0, cohort.case_count, batch_size):
            cases = [
                _build_case(cohort, index)
                for index in range(batch_start, min(cohort.case_count, batch_start + batch_size))
            ]
            batch_results = session.execute_cases(
                cases,
                list(cohort.targets),
                config,
            )
            for case, (raw_results, normalized) in zip(cases, batch_results, strict=True):
                findings = evaluate_case(case, normalized)
                annotate_findings(
                    case,
                    findings,
                    normalized,
                    raw_results,
                    config.to_dict(),
                    list(cohort.targets),
                )
                case_statuses = {}
                for target in cohort.targets:
                    raw = raw_results[target]
                    status = str(raw.get("status", "unknown"))
                    case_statuses[target] = status
                    status_counts[target][status] += 1
                    aggregate_targets[target][status] += 1
                    duration = float(raw.get("duration_ms", 0.0) or 0.0)
                    durations[target].append(duration)
                    aggregate_durations[target].append(duration)
                for finding in findings:
                    finding_kinds[finding.kind] += 1
                    verdicts[finding.triage_verdict] += 1

                detected = _fault_detected(cohort, findings)
                fault_detections += int(detected)
                is_anomaly = bool(findings) or any(status != "ok" for status in case_statuses.values())
                if is_anomaly:
                    anomaly_cases += 1
                    priority = _is_priority_anomaly(
                        cohort,
                        findings,
                        case_statuses,
                    )
                    if priority:
                        priority_anomalies += 1
                    if priority or persisted_anomalies < anomaly_limit:
                        anomaly_writer.write(
                            _anomaly_record(
                                cohort,
                                case,
                                raw_results,
                                normalized,
                                findings,
                                fault_detected=detected,
                            )
                        )
                        persisted_anomalies += 1
                digest_payload = {
                    "case": stable_digest("audit-case", case.to_dict()),
                    "statuses": case_statuses,
                    "findings": [
                        {
                            "kind": finding.kind,
                            "signature": finding.signature,
                            "verdict": finding.triage_verdict,
                        }
                        for finding in findings
                    ],
                }
                digest_bytes = json.dumps(
                    digest_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                cohort_digest.update(digest_bytes)
                normal_digest.update(digest_bytes)
    finally:
        session.close()

    runtime_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    target_rows = [
        _target_row(target, status_counts[target], durations[target])
        for target in cohort.targets
    ]
    return {
        **cohort.to_dict(),
        "runtime_seconds": runtime_seconds,
        "cpu_seconds": cpu_seconds,
        "cases_per_second": cohort.case_count / runtime_seconds if runtime_seconds else 0.0,
        "backend_calls": session.backend_calls,
        "backend_calls_per_second": session.backend_calls / runtime_seconds if runtime_seconds else 0.0,
        "process_io_delta": _counter_delta(_process_io(), started_io),
        "finding_count": sum(finding_kinds.values()),
        "finding_kinds": dict(sorted(finding_kinds.items())),
        "verdict_counts": dict(sorted(verdicts.items())),
        "anomaly_case_count": anomaly_cases,
        "persisted_anomaly_case_count": persisted_anomalies,
        "priority_anomaly_case_count": priority_anomalies,
        "omitted_anomaly_case_count": max(0, anomaly_cases - persisted_anomalies),
        "fault_detection_count": fault_detections,
        "fault_detection_rate": (
            fault_detections / cohort.case_count
            if cohort.expectation == FAULT_EXPECTATION and cohort.case_count
            else None
        ),
        "stream_digest": cohort_digest.hexdigest(),
        "targets": target_rows,
        "execution_session": session.summary(),
    }


def _audit_config(*, parallel_backends: bool) -> ExperimentConfig:
    return ExperimentConfig(
        method_arm="contract_ccs_obligations_cartesian",
        enable_metamorphic_oracle=False,
        enable_feedback=False,
        enable_artifact=False,
        enable_parallel_backend_execution=parallel_backends,
        enable_backend_sampling=False,
        enable_adaptive_candidate_pool=False,
        candidate_recheck_count=0,
    )


def _build_case(cohort: AuditCohort, index: int) -> Case:
    seed = cohort.seed_start + index
    if cohort.generator == "typed_grammar":
        case = generate_case(seed, profile="typed_grammar")
        case.metadata.update(
            {
                "large_scale_audit_cohort": cohort.name,
                "large_scale_audit_index": index,
            }
        )
        return case
    if cohort.generator == "declared_subset":
        return build_declared_subset_case(
            seed=seed,
            index=index,
            backends=cohort.targets,
        )
    if cohort.generator.startswith("seeded_"):
        return _seeded_fault_case(cohort.generator.removeprefix("seeded_"), seed, index)
    raise ValueError(f"unknown audit generator: {cohort.generator}")


def _seeded_fault_case(fault: str, seed: int, index: int) -> Case:
    row_count = 5 + index % 5
    if fault == "filter":
        table = TableData(
            "t0",
            [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
            [{"id": row, "x": row * 3 - index % 7} for row in range(row_count)],
        )
        operations = [
            {"op": "filter", "column": "id", "cmp": ">=", "value": index % 3},
            {"op": "select", "columns": ["id", "x"]},
        ]
    elif fault == "groupby":
        table = TableData(
            "t0",
            [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
            [
                {"g": f"g{row % 3}", "x": row + 1 + index % 4}
                for row in range(row_count)
            ],
        )
        operations = [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
            }
        ]
    elif fault == "join":
        table = TableData(
            "t0",
            [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
            [{"id": row, "x": row + index % 11} for row in range(row_count)],
        )
        right = TableData(
            "t1",
            [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
            [{"id": row, "j": 100 + row + index % 13} for row in range(row_count)],
        )
        operations = [
            {
                "op": "join",
                "table": "t1",
                "left_on": "id",
                "right_on": "id",
                "how": "left",
            }
        ]
        return Case(
            f"large-audit-{fault}-{index:06d}",
            seed,
            [table, right],
            Program(f"large-audit-program-{fault}-{index:06d}", seed, operations),
            metadata={"large_scale_audit_fault": fault, "large_scale_audit_index": index},
        )
    elif fault == "mutate":
        table = TableData(
            "t0",
            [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
            [{"id": row, "x": row - index % 5} for row in range(row_count)],
        )
        operations = [
            {
                "op": "mutate",
                "column": "m",
                "expr": {
                    "kind": "arith_const",
                    "source": "x",
                    "op": "mul",
                    "value": 2 + index % 3,
                },
            }
        ]
    else:
        raise ValueError(f"unknown seeded fault: {fault}")
    return Case(
        f"large-audit-{fault}-{index:06d}",
        seed,
        [table],
        Program(f"large-audit-program-{fault}-{index:06d}", seed, operations),
        metadata={"large_scale_audit_fault": fault, "large_scale_audit_index": index},
    )


def _fault_detected(cohort: AuditCohort, findings: Sequence[Finding]) -> bool:
    if cohort.expectation != FAULT_EXPECTATION:
        return False
    return any(
        finding.triage_verdict == "candidate_implementation_bug"
        and cohort.fault_target in finding.suspicious_backends
        for finding in findings
    )


def _is_priority_anomaly(
    cohort: AuditCohort,
    findings: Sequence[Finding],
    statuses: dict[str, str],
) -> bool:
    if cohort.expectation != GENERAL_EXPECTATION:
        return False
    if any(status != "ok" for status in statuses.values()):
        return True
    return any(
        finding.triage_verdict
        in {
            "candidate_implementation_bug",
            "needs_manual_confirmation",
            "semantic_divergence_needs_confirmation",
            "unclassified",
        }
        for finding in findings
    )


def _anomaly_record(
    cohort: AuditCohort,
    case: Case,
    raw_results: dict[str, dict[str, Any]],
    normalized: dict[str, NormalizedResult],
    findings: Sequence[Finding],
    *,
    fault_detected: bool,
) -> dict[str, Any]:
    return {
        "schema_version": LARGE_SCALE_AUDIT_SCHEMA_VERSION,
        "cohort": cohort.name,
        "expectation": cohort.expectation,
        "fault_target": cohort.fault_target,
        "fault_detected": fault_detected,
        "case": case.to_dict(),
        "raw_results": raw_results,
        "normalized": {
            target: {
                "status": result.status,
                "columns": result.columns,
                "row_count": result.row_count,
                "rows": result.rows[:20],
                "rows_truncated": result.row_count > 20,
                "error_type": result.error_type,
                "error": result.error,
                "comparison_key": result.comparison_key,
            }
            for target, result in normalized.items()
        },
        "findings": [finding.to_dict() for finding in findings],
    }


def _target_row(
    target: str,
    statuses: Counter[str],
    durations: Sequence[float],
) -> dict[str, Any]:
    case_count = sum(statuses.values())
    return {
        "target": target,
        "case_count": case_count,
        "status_counts": dict(sorted(statuses.items())),
        "ok_count": statuses["ok"],
        "error_count": statuses["error"],
        "timeout_count": statuses["timeout"],
        "missing_count": statuses["missing"],
        "ok_rate": statuses["ok"] / case_count if case_count else 0.0,
        "duration_ms": _duration_summary(durations),
    }


def _aggregate_target_rows(
    statuses: dict[str, Counter[str]],
    durations: dict[str, list[float]],
) -> list[dict[str, Any]]:
    return [
        _target_row(target, statuses[target], durations[target])
        for target in TARGETS
        if target in statuses
    ]


def _duration_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _build_assessment(
    *,
    cohorts: Sequence[AuditCohort],
    cohort_results: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]],
    minimum_cases_per_target: int,
) -> dict[str, Any]:
    planned_targets = {target for cohort in cohorts for target in cohort.targets}
    missing_targets = sorted(set(TARGETS) - planned_targets)
    under_minimum = sorted(
        row["target"]
        for row in target_rows
        if int(row["case_count"]) < minimum_cases_per_target
    )
    general_rows = [row for row in cohort_results if row["expectation"] == GENERAL_EXPECTATION]
    fault_rows = [row for row in cohort_results if row["expectation"] == FAULT_EXPECTATION]
    unexpected_statuses = sum(
        int(target["case_count"]) - int(target["ok_count"])
        for row in general_rows
        for target in row["targets"]
    )
    candidate_cases = sum(
        int(row["verdict_counts"].get("candidate_implementation_bug", 0))
        for row in general_rows
    )
    manual_cases = sum(
        sum(
            int(count)
            for verdict, count in row["verdict_counts"].items()
            if verdict in {"needs_manual_confirmation", "semantic_divergence_needs_confirmation"}
        )
        for row in general_rows
    )
    seeded_misses = sum(
        int(row["case_count"]) - int(row["fault_detection_count"])
        for row in fault_rows
    )
    concerns = []
    if missing_targets:
        concerns.append(f"registered targets not covered: {', '.join(missing_targets)}")
    if under_minimum:
        concerns.append(f"targets below minimum case count: {', '.join(under_minimum)}")
    if unexpected_statuses:
        concerns.append(f"general cohorts produced {unexpected_statuses} non-ok target executions")
    if candidate_cases:
        concerns.append(f"general cohorts produced {candidate_cases} candidate implementation findings")
    if manual_cases:
        concerns.append(f"general cohorts produced {manual_cases} unresolved findings")
    if seeded_misses:
        concerns.append(f"seeded fault cohorts missed {seeded_misses} injected faults")
    return {
        "target_coverage_complete": not missing_targets,
        "missing_targets": missing_targets,
        "minimum_cases_per_target": minimum_cases_per_target,
        "targets_under_minimum": under_minimum,
        "general_non_ok_execution_count": unexpected_statuses,
        "general_candidate_finding_count": candidate_cases,
        "general_unresolved_finding_count": manual_cases,
        "seeded_fault_miss_count": seeded_misses,
        "implementation_concerns": concerns,
        "passed": not concerns,
    }


def _validate_plan(cohorts: Sequence[AuditCohort]) -> None:
    names = [cohort.name for cohort in cohorts]
    if len(names) != len(set(names)):
        raise ValueError("audit cohort names must be unique")
    for cohort in cohorts:
        unknown = sorted(set(cohort.targets) - set(TARGETS))
        if unknown:
            raise ValueError(f"cohort {cohort.name} has unknown targets: {', '.join(unknown)}")
        if cohort.expectation not in {GENERAL_EXPECTATION, FAULT_EXPECTATION}:
            raise ValueError(f"cohort {cohort.name} has unknown expectation: {cohort.expectation}")
        if cohort.expectation == FAULT_EXPECTATION and cohort.fault_target not in cohort.targets:
            raise ValueError(f"cohort {cohort.name} does not include fault target {cohort.fault_target}")


def _process_io() -> dict[str, int]:
    path = Path("/proc/self/io")
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if separator:
            try:
                values[key.strip()] = int(raw.strip())
            except ValueError:
                continue
    return values


def _counter_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    return {
        key: int(current.get(key, 0)) - int(previous.get(key, 0))
        for key in sorted(set(current) | set(previous))
    }


def _current_rss_kib() -> int:
    path = Path("/proc/self/status")
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return 0
    return 0


def _baseline_metrics(
    baseline_log: Path | None,
    *,
    baseline_runtime_seconds: float | None,
) -> dict[str, Any] | None:
    if baseline_log is None:
        return None
    path = baseline_log.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    case_count = _line_count(path)
    size_bytes = path.stat().st_size
    runtime = float(baseline_runtime_seconds or 0.0)
    return {
        "path": str(path),
        "case_count": case_count,
        "size_bytes": size_bytes,
        "bytes_per_case": size_bytes / case_count if case_count else 0.0,
        "runtime_seconds": runtime,
        "cases_per_second": case_count / runtime if runtime else None,
    }


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def _io_comparison(
    baseline: dict[str, Any] | None,
    *,
    anomaly_bytes: int,
    total_cases: int,
    cohort_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    bytes_per_case = anomaly_bytes / total_cases if total_cases else 0.0
    production = next(
        (row for row in cohort_results if row["name"] == "production_generalization"),
        None,
    )
    output: dict[str, Any] = {
        "anomaly_stream_bytes": anomaly_bytes,
        "anomaly_stream_bytes_per_case": bytes_per_case,
        "normal_case_rows_persisted": 0,
    }
    if baseline is None:
        return output
    baseline_bytes_per_case = float(baseline["bytes_per_case"] or 0.0)
    output["baseline_to_anomaly_stream_size_ratio"] = (
        baseline_bytes_per_case / bytes_per_case
        if bytes_per_case
        else None
    )
    output["log_bytes_per_case_reduction_percent"] = (
        100.0 * (1.0 - bytes_per_case / baseline_bytes_per_case)
        if baseline_bytes_per_case
        else None
    )
    baseline_rate = baseline.get("cases_per_second")
    if production is not None and baseline_rate:
        output["production_throughput_speedup"] = (
            float(production["cases_per_second"]) / float(baseline_rate)
        )
    return output


def _update_artifact_metrics(result: dict[str, Any], output: Path) -> None:
    artifact_bytes = sum(
        path.stat().st_size
        for path in output.iterdir()
        if path.is_file()
    )
    case_count = int(result["summary"]["case_count"])
    result["io_comparison"]["total_artifact_bytes"] = artifact_bytes
    result["io_comparison"]["total_artifact_bytes_per_case"] = (
        artifact_bytes / case_count if case_count else 0.0
    )


def _render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    assessment = result["assessment"]
    lines = [
        "# Large-Scale Generality and Throughput Audit",
        "",
        f"- Cases: `{summary['case_count']}` across `{summary['cohort_count']}` cohorts.",
        f"- Backend calls: `{summary['backend_calls']}`.",
        f"- Runtime: `{summary['runtime_seconds']:.3f}` seconds.",
        f"- Throughput: `{summary['cases_per_second']:.3f}` cases/s, `{summary['backend_calls_per_second']:.3f}` backend calls/s.",
        f"- Acceptance passed: `{str(assessment['passed']).lower()}`.",
        "",
        "## Cohorts",
        "",
        "| cohort | cases | calls | cases/s | findings | anomalies saved | fault detection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort in result["cohorts"]:
        detection = cohort["fault_detection_rate"]
        detection_text = "n/a" if detection is None else f"{100.0 * detection:.2f}%"
        lines.append(
            f"| {cohort['name']} | {cohort['case_count']} | {cohort['backend_calls']} | "
            f"{cohort['cases_per_second']:.3f} | {cohort['finding_count']} | "
            f"{cohort['persisted_anomaly_case_count']} | {detection_text} |"
        )
    lines.extend(
        [
            "",
            "## Targets",
            "",
            "| target | cases | ok | error | timeout | missing | mean ms | p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in result["targets"]:
        duration = target["duration_ms"]
        lines.append(
            f"| {target['target']} | {target['case_count']} | {target['ok_count']} | "
            f"{target['error_count']} | {target['timeout_count']} | {target['missing_count']} | "
            f"{duration['mean']:.3f} | {duration['p95']:.3f} |"
        )
    lines.extend(["", "## Assessment", ""])
    if assessment["implementation_concerns"]:
        lines.extend(f"- {concern}" for concern in assessment["implementation_concerns"])
    else:
        lines.append("- No unresolved implementation concern was observed under the registered audit gates.")
    io_comparison = result["io_comparison"]
    lines.extend(
        [
            "",
            "## I/O",
            "",
            f"- Anomaly stream: `{io_comparison['anomaly_stream_bytes']}` bytes; normal rows are aggregate-only.",
            f"- Process write bytes: `{result['process']['io_delta'].get('write_bytes', 0)}`.",
        ]
    )
    if io_comparison.get("log_bytes_per_case_reduction_percent") is not None:
        lines.append(
            "- Log bytes per case reduction versus baseline: "
            f"`{io_comparison['log_bytes_per_case_reduction_percent']:.3f}%`."
        )
    if io_comparison.get("production_throughput_speedup") is not None:
        lines.append(
            "- Production cohort throughput speedup versus baseline: "
            f"`{io_comparison['production_throughput_speedup']:.3f}x`."
        )
    lines.append("")
    return "\n".join(lines)
