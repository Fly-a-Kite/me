from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, Program
from datadiff.execution_accounting import (
    backend_calls_from_raw,
    backend_reported_ms_from_raw,
    execution_cache_hits_from_raw,
)
from datadiff.operation_semantics import op_kind
from datadiff.oracle import Finding, evaluate_case


LOCALIZATION_SCHEMA_VERSION = "prefix-first-divergence-localization-v1"
ExecutePrefixFn = Callable[..., tuple[dict[str, dict[str, Any]], dict[str, Any]]]
DivergenceFn = Callable[[Case, dict[str, Any], str], bool]
ClockFn = Callable[[], float]


@dataclass(frozen=True, slots=True)
class PrefixObservation:
    prefix_length: int
    operation_index: int | None
    operation_kind: str
    divergent: bool
    backend_statuses: dict[str, str]
    finding_count: int
    plan_fingerprints: dict[str, str]
    plan_operators: dict[str, tuple[str, ...]]
    backend_calls: int
    backend_reported_ms: float
    cache_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_length": self.prefix_length,
            "operation_index": self.operation_index,
            "operation_kind": self.operation_kind,
            "divergent": self.divergent,
            "backend_statuses": dict(self.backend_statuses),
            "finding_count": self.finding_count,
            "plan_fingerprints": dict(self.plan_fingerprints),
            "plan_operators": {
                backend: list(operators)
                for backend, operators in self.plan_operators.items()
            },
            "backend_calls": self.backend_calls,
            "backend_reported_ms": self.backend_reported_ms,
            "cache_hits": self.cache_hits,
        }


def localize_first_divergence(
    case: Case,
    *,
    backends: list[str],
    config: ExperimentConfig,
    raw_results: Mapping[str, Any],
    normalized: dict[str, Any],
    findings: Sequence[Finding],
    execute_prefix_fn: ExecutePrefixFn,
    backend_instances: dict[str, Any] | None = None,
    divergence_fn: DivergenceFn | None = None,
    perf_counter_fn: ClockFn = time.perf_counter,
    process_cpu_fn: ClockFn = time.process_time,
) -> dict[str, Any]:
    started = perf_counter_fn()
    process_started = process_cpu_fn()
    operation_count = len(case.program.operations)
    suspicious_backends = _diagnostic_backends(backends, findings)
    comparison_mode = config.method_policy.semantic.comparison_mode
    diagnostic_config = _diagnostic_config(config)
    detector = divergence_fn or _default_divergence
    cache: dict[int, PrefixObservation] = {}
    extra_backend_calls = 0
    extra_backend_reported_ms = 0.0
    extra_cache_hits = 0

    def observe(prefix_length: int, *, force_plan: bool = False) -> PrefixObservation:
        nonlocal extra_backend_calls, extra_backend_reported_ms, extra_cache_hits
        length = max(0, min(operation_count, int(prefix_length)))
        existing = cache.get(length)
        if existing is not None and (not force_plan or existing.plan_fingerprints):
            return existing
        prefix_case = _prefix_case(case, length)
        prefix_raw, prefix_normalized = execute_prefix_fn(
            prefix_case,
            suspicious_backends,
            diagnostic_config,
            backend_instances=backend_instances,
        )
        prefix_findings = (
            evaluate_case(
                prefix_case,
                prefix_normalized,
                comparison_mode=comparison_mode,
            )
            if divergence_fn is None
            else []
        )
        divergent = (
            bool(prefix_findings)
            if divergence_fn is None
            else bool(detector(prefix_case, prefix_normalized, comparison_mode))
        )
        calls = backend_calls_from_raw(prefix_raw)
        reported_ms = backend_reported_ms_from_raw(prefix_raw)
        cache_hits = execution_cache_hits_from_raw(prefix_raw)
        extra_backend_calls += calls
        extra_backend_reported_ms += reported_ms
        extra_cache_hits += cache_hits
        observation = PrefixObservation(
            prefix_length=length,
            operation_index=(length - 1 if length else None),
            operation_kind=(
                op_kind(case.program.operations[length - 1]) if length else "source"
            ),
            divergent=divergent,
            backend_statuses={
                str(backend): str(raw.get("status", ""))
                for backend, raw in prefix_raw.items()
                if isinstance(raw, Mapping)
            },
            finding_count=len(prefix_findings),
            plan_fingerprints=_physical_plan_fingerprints(prefix_raw),
            plan_operators=_physical_plan_operators(prefix_raw),
            backend_calls=calls,
            backend_reported_ms=reported_ms,
            cache_hits=cache_hits,
        )
        cache[length] = observation
        return observation

    base_differential = bool(
        detector(case, normalized, comparison_mode)
        if divergence_fn is not None
        else _default_divergence(case, normalized, comparison_mode)
    )
    skip_reason = ""
    first_divergent_prefix: int | None = None
    if not findings:
        skip_reason = "no_finding"
    elif operation_count <= 0:
        skip_reason = "empty_program"
    elif not base_differential:
        skip_reason = "finding_not_observable_in_base_prefixes"
    else:
        base_observation = PrefixObservation(
            prefix_length=operation_count,
            operation_index=operation_count - 1,
            operation_kind=op_kind(case.program.operations[-1]),
            divergent=True,
            backend_statuses={
                str(backend): str(raw.get("status", ""))
                for backend, raw in raw_results.items()
                if isinstance(raw, Mapping)
            },
            finding_count=sum(
                1 for finding in findings if finding.oracle in {"differential", "witness"}
            ),
            plan_fingerprints=_physical_plan_fingerprints(raw_results),
            plan_operators=_physical_plan_operators(raw_results),
            backend_calls=0,
            backend_reported_ms=0.0,
            cache_hits=0,
        )
        cache[operation_count] = base_observation
        low = 0
        low_observation = observe(0)
        if low_observation.divergent:
            first_divergent_prefix = 0
        else:
            high = operation_count
            while high - low > 1:
                middle = (low + high) // 2
                observation = observe(middle)
                if observation.divergent:
                    high = middle
                else:
                    low = middle
            first_divergent_prefix = high
            # Prefix divergence is not guaranteed monotone. Audit all earlier
            # unobserved prefixes before accepting the binary-search boundary.
            for length in range(1, high):
                if observe(length).divergent:
                    first_divergent_prefix = length
                    break
        if first_divergent_prefix is not None:
            observe(max(0, first_divergent_prefix - 1), force_plan=True)
            observe(first_divergent_prefix, force_plan=True)

    plan_transitions = _prefix_plan_transitions(cache)
    culprit_ranking = _rank_culprits(
        case,
        findings,
        first_divergent_prefix=first_divergent_prefix,
        plan_transitions=plan_transitions,
    )
    nodes = [cache[length].to_dict() for length in sorted(cache)]
    tested_lengths = sorted(cache)
    dag_edges = [
        {"from_prefix": left, "to_prefix": right}
        for left, right in zip(tested_lengths, tested_lengths[1:])
    ]
    wall_ms = max(0.0, (perf_counter_fn() - started) * 1000.0)
    process_cpu_ms = max(0.0, (process_cpu_fn() - process_started) * 1000.0)
    return {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "enabled": True,
        "attempted": bool(findings and operation_count and base_differential),
        "skip_reason": skip_reason,
        "diagnostic_backends": suspicious_backends,
        "base_differential_observable": base_differential,
        "first_divergent_prefix": first_divergent_prefix,
        "culprit_operation_index": (
            first_divergent_prefix - 1
            if first_divergent_prefix is not None and first_divergent_prefix > 0
            else None
        ),
        "prefix_result_dag": {
            "nodes": nodes,
            "edges": dag_edges,
        },
        "plan_transitions": plan_transitions,
        "culprit_ranking": culprit_ranking,
        "top1_operation_index": (
            culprit_ranking[0]["operation_index"] if culprit_ranking else None
        ),
        "topk_operation_indexes": [
            item["operation_index"] for item in culprit_ranking[:3]
        ],
        "extra_backend_calls": extra_backend_calls,
        "extra_backend_reported_ms": extra_backend_reported_ms,
        "extra_execution_cache_hits": extra_cache_hits,
        "wall_ms": wall_ms,
        "process_cpu_ms": process_cpu_ms,
        "diagnostic_plan_collection": True,
    }


def evaluate_localization_accuracy(
    reports: Sequence[Mapping[str, Any]],
    expected_operation_indexes: Sequence[int],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    if len(reports) != len(expected_operation_indexes):
        raise ValueError("reports and expected indexes must have equal length")
    top1 = 0
    topk = 0
    attempted = 0
    calls = 0
    wall_ms = 0.0
    for report, expected in zip(reports, expected_operation_indexes, strict=True):
        if not bool(report.get("attempted")):
            continue
        attempted += 1
        ranking = list(report.get("culprit_ranking", ()) or ())
        indexes = [int(item.get("operation_index", -1)) for item in ranking]
        top1 += int(bool(indexes) and indexes[0] == int(expected))
        topk += int(int(expected) in indexes[: max(1, int(top_k))])
        calls += max(0, int(report.get("extra_backend_calls", 0) or 0))
        wall_ms += max(0.0, float(report.get("wall_ms", 0.0) or 0.0))
    return {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "case_count": len(reports),
        "attempted_count": attempted,
        "top1_correct": top1,
        "topk_correct": topk,
        "top1_accuracy": top1 / attempted if attempted else 0.0,
        "topk_accuracy": topk / attempted if attempted else 0.0,
        "extra_backend_calls": calls,
        "wall_ms": wall_ms,
    }


def _diagnostic_config(config: ExperimentConfig) -> ExperimentConfig:
    return config.for_evidence_tier("finding")


def _prefix_case(case: Case, prefix_length: int) -> Case:
    metadata = dict(case.metadata or {})
    for key in (
        "disagreement_descriptor",
        "case_fingerprint",
        "interaction_descriptor",
        "semantic_contract_lattice",
    ):
        metadata.pop(key, None)
    return Case(
        case_id=f"{case.case_id}-prefix-{prefix_length}",
        seed=case.seed,
        tables=case.tables,
        program=Program(
            program_id=f"{case.program.program_id}-prefix-{prefix_length}",
            seed=case.program.seed,
            operations=list(case.program.operations[:prefix_length]),
        ),
        metadata=metadata,
    )


def _default_divergence(
    case: Case,
    normalized: dict[str, Any],
    comparison_mode: str,
) -> bool:
    return bool(evaluate_case(case, normalized, comparison_mode=comparison_mode))


def _diagnostic_backends(
    backends: list[str],
    findings: Sequence[Finding],
) -> list[str]:
    suspicious = {
        backend
        for finding in findings
        for backend in finding.suspicious_backends
        if backend in backends
    }
    if not suspicious:
        return list(backends)
    reference = next((backend for backend in backends if backend not in suspicious), None)
    selected = [backend for backend in backends if backend in suspicious]
    if reference is not None:
        selected.insert(0, reference)
    return selected if len(selected) >= 2 else list(backends)


def _physical_plan_fingerprints(raw_results: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for backend, raw in raw_results.items():
        plan = raw.get("physical_plan") if isinstance(raw, Mapping) else None
        observations = plan.get("observations", ()) if isinstance(plan, Mapping) else ()
        physical = next(
            (
                item for item in observations or ()
                if isinstance(item, Mapping)
                and item.get("status") == "ok"
                and item.get("plan_kind") == "physical"
            ),
            None,
        )
        if physical is not None and physical.get("fingerprint"):
            out[str(backend)] = str(physical["fingerprint"])
    return out


def _physical_plan_operators(
    raw_results: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for backend, raw in raw_results.items():
        plan = raw.get("physical_plan") if isinstance(raw, Mapping) else None
        observations = plan.get("observations", ()) if isinstance(plan, Mapping) else ()
        physical = next(
            (
                item for item in observations or ()
                if isinstance(item, Mapping)
                and item.get("status") == "ok"
                and item.get("plan_kind") == "physical"
            ),
            None,
        )
        if physical is not None:
            out[str(backend)] = tuple(
                str(item) for item in physical.get("operator_tokens", ()) or ()
            )
    return out


def _prefix_plan_transitions(
    observations: Mapping[int, PrefixObservation],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for length in sorted(observations):
        if length <= 0 or length - 1 not in observations:
            continue
        left = observations[length - 1]
        right = observations[length]
        for backend in sorted(set(left.plan_operators) | set(right.plan_operators)):
            left_counts = Counter(left.plan_operators.get(backend, ()))
            right_counts = Counter(right.plan_operators.get(backend, ()))
            added = tuple(sorted((right_counts - left_counts).elements()))
            removed = tuple(sorted((left_counts - right_counts).elements()))
            left_fingerprint = left.plan_fingerprints.get(backend, "")
            right_fingerprint = right.plan_fingerprints.get(backend, "")
            transitions.append(
                {
                    "backend": backend,
                    "operation_index": length - 1,
                    "operation_kind": right.operation_kind,
                    "left_fingerprint": left_fingerprint,
                    "right_fingerprint": right_fingerprint,
                    "changed": bool(
                        left_fingerprint
                        and right_fingerprint
                        and left_fingerprint != right_fingerprint
                    ),
                    "added_operators": list(added),
                    "removed_operators": list(removed),
                }
            )
    return transitions


def _rank_culprits(
    case: Case,
    findings: Sequence[Finding],
    *,
    first_divergent_prefix: int | None,
    plan_transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    transition_by_index: dict[int, list[Mapping[str, Any]]] = {}
    for transition in plan_transitions:
        transition_by_index.setdefault(
            int(transition.get("operation_index", -1)), []
        ).append(transition)
    root_text = " ".join(
        f"{finding.root_cause} {finding.kind} {finding.mismatch_class}"
        for finding in findings
    ).lower()
    first_index = (
        first_divergent_prefix - 1
        if first_divergent_prefix is not None and first_divergent_prefix > 0
        else None
    )
    ranking: list[dict[str, Any]] = []
    for index, operation in enumerate(case.program.operations):
        kind = op_kind(operation)
        score = 0.0
        reasons: list[str] = []
        if first_index is not None:
            distance = abs(index - first_index)
            score += max(0.0, 100.0 - 20.0 * distance)
            if index == first_index:
                reasons.append("first_disagreeing_prefix")
        transitions = transition_by_index.get(index, [])
        changed_transitions = [item for item in transitions if item.get("changed")]
        if changed_transitions:
            score += 15.0
            reasons.append("physical_plan_transition")
        if any(item.get("added_operators") or item.get("removed_operators") for item in transitions):
            score += 5.0
            reasons.append("physical_operator_delta")
        if kind in root_text or any(alias in root_text for alias in _kind_aliases(kind)):
            score += 12.0
            reasons.append("finding_root_cause_match")
        if kind in {"sort", "limit", "offset", "groupby", "aggregate", "join", "semi_join", "anti_join", "running_sum"}:
            score += 2.0
        ranking.append(
            {
                "operation_index": index,
                "operation_kind": kind,
                "score": round(score, 6),
                "reasons": reasons,
                "plan_transitions": [dict(item) for item in transitions],
            }
        )
    ranking.sort(key=lambda item: (-float(item["score"]), int(item["operation_index"])))
    return ranking


def _kind_aliases(kind: str) -> tuple[str, ...]:
    aliases = {
        "groupby": ("aggregate", "aggregation", "grouped"),
        "aggregate": ("aggregation", "grouped"),
        "sort": ("order", "ordering", "topk"),
        "limit": ("topk", "pushdown"),
        "offset": ("skip",),
        "join": ("membership", "cardinality"),
        "semi_join": ("membership", "join"),
        "anti_join": ("membership", "join"),
        "mutate": ("cast", "expression", "arithmetic"),
        "running_sum": ("window", "cumulative"),
    }
    return aliases.get(kind, ())
