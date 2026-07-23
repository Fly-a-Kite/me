"""Runtime adapter that makes the Coordinator own candidate admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datadiff.coordinator import (
    CandidateSource,
    CapabilityStatus,
    Coordinator,
    FamilyLifecycle,
    ScheduledCase,
    Utility,
)
from datadiff.dsl import Case
from datadiff.finding_outcomes import candidate_issue_family_keys
from datadiff.operation_semantics import (
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    condition_cmp,
    condition_column,
    expr_kind,
    op_column,
    op_columns,
    op_kind,
    op_source,
    operation_names,
)
from datadiff.semantic_novelty import (
    build_post_execution_semantic_novelty_descriptor,
    build_semantic_novelty_descriptor,
    freeze_semantic_novelty_batch,
)


@dataclass(frozen=True, slots=True)
class CoordinatedCandidates:
    candidates: tuple[Case, ...]
    scheduled_by_case_id: dict[int, ScheduledCase]


def coordinate_candidates(
    coordinator: Coordinator,
    candidates: Sequence[Case],
    candidate_meta: Mapping[int, Mapping[str, Any]],
    *,
    target_shard: str,
) -> CoordinatedCandidates:
    if not candidates:
        raise ValueError("coordinator requires at least one candidate")
    scheduled_by_case: dict[int, ScheduledCase] = {}
    prepared: list[tuple[Case, Mapping[str, Any], tuple[str, ...], Any]] = []
    for case in candidates:
        metadata = candidate_meta.get(id(case), {})
        units = capability_units_for_case(case, target_shard=target_shard)
        descriptor = build_semantic_novelty_descriptor(
            case,
            target_keys=tuple(
                part for part in str(target_shard).split(",") if part
            ),
            candidate_metadata=metadata,
        )
        prepared.append((case, metadata, units, descriptor))
    frozen_batch = freeze_semantic_novelty_batch(
        coordinator.semantic_novelty_ledger,
        tuple(item[3] for item in prepared),
    )
    for index, (case, metadata, units, descriptor) in enumerate(prepared):
        novelty_preview = frozen_batch.preview_for(index)
        scheduled = ScheduledCase(
            case_id=case.case_id,
            case_index=case.seed,
            source=_source(
                metadata.get("source", "generated"),
                metadata=metadata,
                target_shard=target_shard,
                case=case,
            ),
            derived_seed=case.seed,
            target_shard=target_shard,
            capability_units=units,
            utility=_utility(case),
            semantic_novelty=descriptor,
            semantic_novelty_preview=novelty_preview,
            family_lifecycle=_family_lifecycle(metadata),
        )
        novelty_selection = {
            **descriptor.compact_dict(),
            **novelty_preview.to_dict(),
            "selection_metric": round(
                (
                    coordinator.policy.semantic_novelty_weight
                    * novelty_preview.rarity_score
                    + coordinator.policy.semantic_depth_weight
                    * descriptor.depth_score
                )
                if coordinator.policy.semantic_novelty_enabled
                else 0.0,
                6,
            ),
        }
        case._runtime_cache["semantic_novelty_selection"] = novelty_selection
        if isinstance(metadata, dict):
            metadata["semantic_novelty"] = novelty_selection
        scheduled_by_case[id(case)] = scheduled
    selected = coordinator.rank(scheduled_by_case.values())
    if selected is None:  # pragma: no cover - guarded by the input check
        raise RuntimeError("coordinator did not select a candidate")
    ordered = sorted(
        candidates,
        key=lambda case: (
            scheduled_by_case[id(case)].schedule_key != selected.schedule_key,
            case.seed,
            case.case_id,
        ),
    )
    return CoordinatedCandidates(tuple(ordered), scheduled_by_case)


def record_coordinator_outcome(
    coordinator: Coordinator,
    scheduled: ScheduledCase,
    *,
    row: Mapping[str, Any],
    causal_signatures: Sequence[str],
    witness_id: str,
) -> None:
    findings = [item for item in row.get("findings", ()) if isinstance(item, Mapping)]
    candidate = any(
        str(finding.get("triage_verdict", "") or "") == "candidate_implementation_bug"
        for finding in findings
    )
    cpu_ms = sum(
        _number(result.get("duration_ms", 0.0))
        for result in _mapping(row.get("raw_results")).values()
        if isinstance(result, Mapping)
    )
    io_bytes = sum(
        _number(result.get("io_read_bytes", 0.0)) + _number(result.get("io_write_bytes", 0.0))
        for result in _mapping(row.get("raw_results")).values()
        if isinstance(result, Mapping)
    )
    coordinator.record_outcome(
        scheduled,
        candidate=candidate,
        causal_signatures=causal_signatures,
        observed_families=candidate_issue_family_keys(findings),
        cpu_ms=cpu_ms,
        io_bytes=io_bytes,
    )
    if scheduled.semantic_novelty is not None:
        post_descriptor = build_post_execution_semantic_novelty_descriptor(
            scheduled.semantic_novelty,
            raw_results=_mapping(row.get("raw_results")),
            findings=findings,
            causal_signatures=causal_signatures,
            independent_reproduction=_independent_reproduction_state(row),
        )
        coordinator.semantic_novelty_ledger.record_post_execution(post_descriptor)
    evidence_units = set(scheduled.capability_units)
    evidence_units.update(_row_capability_units(row))
    for backend, raw in _mapping(row.get("raw_results")).items():
        if not isinstance(raw, Mapping):
            continue
        for unit in _units_for_backend(evidence_units, str(backend)):
            status = _capability_status(raw)
            if status is CapabilityStatus.WITNESSED:
                coordinator.update_capability(
                    unit,
                    status=status,
                    minimum_witness=witness_id,
                )
            elif status is CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE:
                coordinator.update_capability(
                    unit,
                    status=status,
                    evidence_id=witness_id,
                )
            elif status is CapabilityStatus.BLOCKED:
                coordinator.update_capability(unit, status=status)


def _source(
    value: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    target_shard: str = "",
    case: Case | None = None,
) -> CandidateSource:
    context = [str(value or ""), str(target_shard or "")]
    if metadata:
        context.extend(str(item) for item in metadata.values() if isinstance(item, (str, int, float, bool)))
    if case is not None and isinstance(case.metadata, Mapping):
        context.extend(
            str(item)
            for item in case.metadata.values()
            if isinstance(item, (str, int, float, bool))
        )
    source = " ".join(context).lower()
    if "seeded" in source or "fault_sensitivity" in source or "buggy_" in source:
        return CandidateSource.SEEDED_SENSITIVITY
    if "metamorphic" in source or "semantic" in source:
        return CandidateSource.SEMANTIC_METAMORPHIC
    if "known_regression" in source or "known replay" in source:
        return CandidateSource.KNOWN_REGRESSION
    if "historical" in source or "exact_reproducer" in source or "replay" in source:
        return CandidateSource.HISTORICAL_REPLAY
    if "regression" in source or "known" in source:
        return CandidateSource.KNOWN_REGRESSION
    if "feedback" in source or "mutation" in source:
        return CandidateSource.FEEDBACK_MUTATION
    return CandidateSource.FRESH_GRAMMAR


def _family_lifecycle(metadata: Mapping[str, Any]) -> FamilyLifecycle:
    raw = (
        metadata.get("family_lifecycle")
        or metadata.get("candidate_status")
        or metadata.get("bug_status")
        or "novel"
    )
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return FamilyLifecycle(normalized)
    except ValueError:
        return FamilyLifecycle.NOVEL


def _independent_reproduction_state(row: Mapping[str, Any]) -> str:
    for key in ("independent_reproduction", "reproduction_state", "recheck_status"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    recheck = row.get("recheck")
    if isinstance(recheck, Mapping):
        if bool(recheck.get("survived", False)):
            return "locally_confirmed"
        if recheck:
            return "rechecked_not_confirmed"
    return "unverified"


def capability_units_for_case(case: Case, *, target_shard: str) -> tuple[str, ...]:
    """Return compact capability-matrix cells for a concrete case.

    An operation token alone cannot tell whether its nullable, typed or
    order-sensitive variants were exercised.  The cells deliberately carry
    those axes while retaining the plain operation cell as a coarse support
    witness.  This gives the coordinator a useful debt signal without
    conflating a successful integer/filter case with every filter contract.
    """

    operations = list(case.program.operations)
    names = sorted(set(operation_names(operations, default="unknown")))
    column_types = {
        str(column.name): str(column.type)
        for table in case.tables
        for column in table.columns
        if str(column.name) and str(column.type)
    }
    types = sorted(set(column_types.values())) or ["unknown"]
    has_null = any(
        column.nullable or any(row.get(column.name) is None for row in table.rows)
        for table in case.tables
        for column in table.columns
    )
    order = "observing" if any(name in {"sort", "limit", "offset", "row_number_filter", "running_sum", "sortedness_check"} for name in names) else "agnostic"
    layout = "multi_table" if len(case.tables) > 1 else "single_table"
    table_capability = "multi" if len(case.tables) > 1 else "single"
    targets = tuple(sorted(part for part in str(target_shard).split(",") if part)) or ("unconfigured",)
    units: set[str] = set()
    units.add(f"table:{table_capability}")
    units.update(f"type:{logical_type}" for logical_type in types)
    if has_null:
        units.add("nulls")
    for target in targets:
        for operation in operations:
            name = op_kind(operation, default="unknown")
            units.add(f"op:{name}")
            units.add(f"target={target}/op={name}")
            for logical_type in _operation_types(operation, column_types):
                units.add(
                    f"target={target}/op={name}/type={logical_type}"
                    f"/null={'present' if has_null else 'absent'}/order={order}/layout={layout}"
                )
            if name == "filter":
                comparator = condition_cmp(operation)
                if comparator in {"str_contains", "str_starts_with", "str_ends_with"}:
                    units.add(f"filter:{comparator}")
            if name == "mutate":
                expression = expr_kind(operation)
                if expression:
                    units.add(f"expr:{expression}")
            if name in {"aggregate", "groupby"}:
                for aggregate in aggregate_specs(operation):
                    function = aggregate_func(aggregate)
                    if function:
                        units.add(f"agg:{function}")
            if name == "running_sum" and operation.get("partition_by"):
                units.add("running:partition_by")
    return tuple(sorted(units))


def _utility(case: Case) -> Utility:
    operations = operation_names(case.program.operations, default="unknown")
    row_count = sum(len(table.rows) for table in case.tables)
    return Utility(
        predicted_cpu_cost=0.05 * len(operations),
        predicted_io_cost=0.001 * row_count,
    )


def _units_for_backend(units: Sequence[str], backend: str) -> tuple[str, ...]:
    prefix = f"target={backend}/"
    return tuple(
        unit
        for unit in units
        if unit.startswith(prefix) or not unit.startswith("target=")
    )


def _operation_types(operation: Mapping[str, Any], column_types: Mapping[str, str]) -> tuple[str, ...]:
    """Return only input types actually consumed by one operation.

    A case may carry int, float, bool and str columns while a filter touches
    only one of them.  Recording all four as witnesses would make the
    capability matrix claim evidence it does not have.
    """

    columns = {
        value
        for value in (
            op_column(operation),
            condition_column(operation),
            op_source(operation),
            *op_columns(operation),
        )
        if value
    }
    for aggregate in aggregate_specs(operation):
        column = aggregate_column(aggregate)
        if column:
            columns.add(column)
    if op_kind(operation) == "coalesce":
        columns.update(str(value) for value in operation.get("sources", ()) if str(value))
    return tuple(sorted({column_types[column] for column in columns if column in column_types}))


def _row_capability_units(row: Mapping[str, Any]) -> tuple[str, ...]:
    program_ir = row.get("program_ir", {})
    if not isinstance(program_ir, Mapping):
        return ()
    return tuple(
        sorted(
            {
                str(unit)
                for unit in program_ir.get("required_capabilities", ())
                if str(unit)
            }
        )
    )


def _capability_status(raw: Mapping[str, Any]) -> CapabilityStatus | None:
    status = str(raw.get("status", "") or "").lower()
    error_type = str(raw.get("error_type", "") or "").lower()
    error = str(raw.get("error", "") or "").lower()
    if status == "ok":
        return CapabilityStatus.WITNESSED
    if status in {"unsupported", "not_supported"} or "unsupported" in error_type:
        return CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE
    if "harnessloweringerror" in error_type or "harnessloweringerror" in error:
        return CapabilityStatus.BLOCKED
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0
