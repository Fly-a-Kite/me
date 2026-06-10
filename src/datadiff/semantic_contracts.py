from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case, normalize_sort_keys
from datadiff.operation_semantics import (
    aggregate_func,
    aggregate_specs,
    condition_cmp,
    condition_value,
    expr_kind,
    expr_operator,
    groupby_keys,
    join_how,
    op_column,
    op_columns,
    op_kind,
)

SEMANTIC_CONTRACT_LATTICE_SCHEMA_VERSION = "semantic-contract-lattice-v1"
CONTRACT_AXES: tuple[str, ...] = (
    "ordering",
    "null",
    "nan",
    "dtype_coercion",
    "error_equivalence",
    "layout_sensitivity",
    "determinism",
)
CONTRACT_POLICIES: tuple[str, ...] = (
    "strict",
    "canonicalized",
    "tolerated",
    "boundary",
    "probe",
)
BOUNDARY_POLICIES = frozenset({"boundary", "probe"})
BUG_POLICIES = frozenset({"strict", "canonicalized", "tolerated"})
_POLICY_RANK = {policy: index for index, policy in enumerate(CONTRACT_POLICIES)}

_ROOT_CAUSE_AXES: dict[str, tuple[str, ...]] = {
    "ordering_or_limit": ("ordering",),
    "grouped_topk_null_sort_key": ("ordering", "null"),
    "distinct_null_topk": ("ordering", "null"),
    "topk_filter_pushdown": ("ordering",),
    "joined_order_offset_projection": ("ordering",),
    "ordered_topk_projection": ("ordering",),
    "nan_inf_semantics": ("nan",),
    "float_group_key_instability": ("nan", "dtype_coercion"),
    "negative_zero_comparison": ("nan",),
    "running_sum_precision": ("ordering", "nan"),
    "arithmetic_expression": ("nan", "dtype_coercion", "error_equivalence"),
    "reverse_division_operand_order": ("nan", "dtype_coercion"),
    "null_semantics": ("null",),
    "coalesce_null_semantics": ("null",),
    "fill_null_null_semantics": ("null",),
    "drop_nulls_null_filter": ("null",),
    "tuple_absence_null_filter": ("null",),
    "outer_join_truth_filter": ("null",),
    "nullable_boolean_expression": ("null",),
    "groupby_aggregation": ("null", "nan", "dtype_coercion", "ordering"),
    "join_semantics": ("null", "ordering", "layout_sensitivity"),
    "semi_join_membership": ("null", "ordering"),
    "anti_join_exclusion": ("null", "ordering"),
    "type_cast": ("dtype_coercion", "error_equivalence"),
    "datetime_expression": ("dtype_coercion", "error_equivalence"),
    "exception_taxonomy": ("error_equivalence",),
    "schema_projection": ("dtype_coercion", "layout_sensitivity"),
    "unicode_case_mapping": ("dtype_coercion", "error_equivalence"),
}
_MISMATCH_CLASS_AXES: dict[str, tuple[str, ...]] = {
    "row_order": ("ordering",),
    "order": ("ordering",),
    "status": ("error_equivalence",),
    "exception": ("error_equivalence",),
    "schema": ("dtype_coercion", "layout_sensitivity"),
    "columns": ("dtype_coercion", "layout_sensitivity"),
}


@dataclass(frozen=True, slots=True)
class ContractAxisPolicy:
    axis: str
    policy: str
    reason: str
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "policy": self.policy,
            "reason": self.reason,
            "signals": list(self.signals),
        }


@dataclass(frozen=True, slots=True)
class OperationContract:
    operation_index: int
    operation: str
    axes: tuple[ContractAxisPolicy, ...]
    contract_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_index": self.operation_index,
            "operation": self.operation,
            "axes": {axis.axis: axis.to_dict() for axis in self.axes},
            "contract_tags": list(self.contract_tags),
        }


@dataclass(frozen=True, slots=True)
class SemanticContractLattice:
    case_id: str
    operation_contracts: tuple[OperationContract, ...]
    axes: dict[str, ContractAxisPolicy]
    contract_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        boundary_axes = [
            axis
            for axis, policy in self.axes.items()
            if policy.policy in BOUNDARY_POLICIES
        ]
        strict_axes = [
            axis
            for axis, policy in self.axes.items()
            if policy.policy in BUG_POLICIES
        ]
        return {
            "schema_version": SEMANTIC_CONTRACT_LATTICE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "axes": {axis: policy.to_dict() for axis, policy in self.axes.items()},
            "boundary_axes": boundary_axes,
            "strict_axes": strict_axes,
            "operation_contracts": [contract.to_dict() for contract in self.operation_contracts],
            "contract_tags": list(self.contract_tags),
            "oracle_guidance": {
                "boundary_policies": sorted(BOUNDARY_POLICIES),
                "bug_policies": sorted(BUG_POLICIES),
                "decision_rule": (
                    "Downgrade a finding to a semantic boundary only when its root_cause "
                    "or mismatch_class maps to an axis whose joined policy is boundary/probe."
                ),
            },
            "lattice_order": list(CONTRACT_POLICIES),
        }


def semantic_contract_lattice(case: Case) -> SemanticContractLattice:
    operation_contracts = tuple(_operation_contracts(case))
    axis_rows: dict[str, list[ContractAxisPolicy]] = {
        axis: [_strict_axis(axis)] for axis in CONTRACT_AXES
    }
    for axis in _case_level_axes(case):
        axis_rows.setdefault(axis.axis, []).append(axis)
    tags: list[str] = []
    for contract in operation_contracts:
        tags.extend(contract.contract_tags)
        for axis in contract.axes:
            axis_rows.setdefault(axis.axis, []).append(axis)
    axes = {
        axis: _join_axis_policies(axis, axis_rows.get(axis, []))
        for axis in CONTRACT_AXES
    }
    return SemanticContractLattice(
        case_id=case.case_id,
        operation_contracts=operation_contracts,
        axes=axes,
        contract_tags=tuple(_unique(tags)),
    )


def semantic_contract_lattice_payload(case: Case) -> dict[str, Any]:
    return semantic_contract_lattice(case).to_dict()


def finding_contract_axes(finding: Any) -> tuple[str, ...]:
    root = _finding_value(finding, "root_cause", "")
    mismatch = _finding_value(finding, "mismatch_class", "")
    axes: list[str] = []
    axes.extend(_ROOT_CAUSE_AXES.get(root, ()))
    axes.extend(_MISMATCH_CLASS_AXES.get(mismatch, ()))
    root_lower = root.lower()
    if "order" in root_lower or "topk" in root_lower:
        axes.append("ordering")
    if "null" in root_lower or "bool" in root_lower:
        axes.append("null")
    if "nan" in root_lower or "float" in root_lower or "precision" in root_lower:
        axes.append("nan")
    if "dtype" in root_lower or "type" in root_lower or "cast" in root_lower:
        axes.append("dtype_coercion")
    if "error" in root_lower or "exception" in root_lower:
        axes.append("error_equivalence")
    if any(token in root_lower for token in ("layout", "schema", "projection", "partition", "chunk", "sliced")):
        axes.append("layout_sensitivity")
    if any(token in root_lower for token in ("random", "inplace", "alias")):
        axes.append("determinism")
    return tuple(axis for axis in _unique(axes) if axis in CONTRACT_AXES)


def finding_matches_contract_boundary(
    case: Case,
    finding: Any,
    config: Mapping[str, Any] | None = None,
) -> bool:
    del config
    axes = finding_contract_axes(finding)
    if not axes:
        return False
    lattice = semantic_contract_lattice(case)
    return any(lattice.axes[axis].policy in BOUNDARY_POLICIES for axis in axes)


def _operation_contracts(case: Case) -> Iterable[OperationContract]:
    order_defined = False
    for index, operation in enumerate(case.program.operations):
        kind = op_kind(operation, "unknown")
        axes = list(_axes_for_operation(case, operation, order_defined=order_defined))
        tags = _tags_for_axes(kind, axes)
        if axes:
            yield OperationContract(
                operation_index=index,
                operation=kind,
                axes=tuple(_dedupe_axis_policies(axes)),
                contract_tags=tuple(tags),
            )
        order_defined = _next_order_defined(order_defined, operation)


def _axes_for_operation(
    case: Case,
    operation: Mapping[str, Any],
    *,
    order_defined: bool,
) -> Iterable[ContractAxisPolicy]:
    kind = op_kind(operation)
    if kind in {"groupby", "aggregate", "distinct", "union_all"}:
        yield _axis("ordering", "canonicalized", f"{kind} output row order is canonicalized before comparison.")
    if kind in {"limit", "offset"} and not order_defined:
        yield _axis("ordering", "boundary", f"{kind} observes input row order without a prior order contract.")
    if kind == "sort":
        yield _axis("ordering", "canonicalized", "Explicit sort defines the observable row order contract.")
        try:
            keys = normalize_sort_keys(operation)
        except ValueError:
            keys = []
        if keys and any(_column_nullable(case, key.column) for key in keys):
            yield _axis("null", "boundary", "Sort key contains nullable data; NULL placement is backend-sensitive.")
    if kind in {"running_sum", "row_number_filter", "sortedness_check"}:
        yield _axis("ordering", "boundary", f"{kind} observes ordered input/window semantics.")
        yield _axis("null", "boundary", f"{kind} may observe NULL placement in ordered partitions.")
    if kind in {"join", "semi_join", "anti_join", "tuple_absence_filter"}:
        yield _axis("ordering", "canonicalized", f"{kind} output order is canonicalized before comparison.")
        yield _axis("null", "boundary", f"{kind} compares nullable membership/key semantics across backends.")
        yield _axis("layout_sensitivity", "tolerated", f"{kind} may alter physical/schema layout without value mismatch.")
    if kind == "filter":
        cmp = condition_cmp(operation)
        value = condition_value(operation)
        if value is None or cmp.startswith("bool_") or "null" in cmp:
            yield _axis("null", "boundary", "Filter predicate observes NULL/unknown truth semantics.")
        if _is_special_float(value):
            yield _axis("nan", "boundary", "Filter predicate compares against NaN/Inf literal.")
    if kind in {"drop_nulls", "fill_null", "coalesce", "case_when"}:
        yield _axis("null", "boundary", f"{kind} explicitly rewrites or observes NULL values.")
    if kind in {"groupby", "aggregate"}:
        funcs = [aggregate_func(aggregate) for aggregate in aggregate_specs(operation)]
        if funcs:
            yield _axis("null", "boundary", "Aggregate functions differ on NULL skipping and empty-group behavior.")
            yield _axis("dtype_coercion", "boundary", "Aggregate output dtype coercion is backend-family dependent.")
        if _case_contains_special_float(case) and any(func in {"sum", "mean", "min", "max"} for func in funcs):
            yield _axis("nan", "boundary", "Floating aggregate observes NaN/Inf propagation.")
    if kind == "mutate":
        yield from _axes_for_mutate(case, operation)
    if kind == "select":
        yield _axis("layout_sensitivity", "canonicalized", "Projection layout is normalized before oracle comparison.")
    if kind.endswith("_probe") or kind in {"random_case_probe", "sortedness_check"}:
        yield from _axes_for_probe(kind)


def _axes_for_mutate(case: Case, operation: Mapping[str, Any]) -> Iterable[ContractAxisPolicy]:
    kind = expr_kind(operation)
    if kind in {"cast", "date_part"}:
        yield _axis("dtype_coercion", "boundary", f"{kind} uses backend-specific conversion rules.")
        yield _axis("error_equivalence", "boundary", f"{kind} may raise equivalent errors with different taxonomies.")
    if kind in {"arith_const", "add_const", "reverse_division_columns", "abs", "clip"}:
        yield _axis("dtype_coercion", "tolerated", f"{kind} numeric result dtype is normalized when possible.")
        if _case_contains_special_float(case) or expr_operator(operation) in {"div", "mod"}:
            yield _axis("nan", "boundary", f"{kind} may observe NaN/Inf or signed-zero numeric semantics.")
        if expr_operator(operation) in {"div", "mod"}:
            yield _axis("error_equivalence", "boundary", f"{kind} may hit divide/modulo edge errors.")
    if kind in {"bool_not", "string_null_if_empty"}:
        yield _axis("null", "boundary", f"{kind} observes nullable scalar semantics.")
    if kind in {"string_lower", "string_upper", "string_replace", "string_slice", "string_split_part"}:
        yield _axis("error_equivalence", "tolerated", f"{kind} string API errors are normalized when equivalent.")


def _axes_for_probe(kind: str) -> Iterable[ContractAxisPolicy]:
    lower = kind.lower()
    if any(token in lower for token in ("order", "topk", "sorted", "pivot")):
        yield _axis("ordering", "probe", f"{kind} intentionally probes ordering-sensitive behavior.")
    if "null" in lower or "bool" in lower:
        yield _axis("null", "probe", f"{kind} intentionally probes nullable/boolean boundary behavior.")
    if any(token in lower for token in ("float", "round", "precision", "quantile")):
        yield _axis("nan", "probe", f"{kind} intentionally probes floating-point boundary behavior.")
        yield _axis("dtype_coercion", "probe", f"{kind} intentionally probes numeric dtype coercion.")
    if any(token in lower for token in ("schema", "layout", "sliced", "partition", "run_end", "large_string", "list")):
        yield _axis("layout_sensitivity", "probe", f"{kind} intentionally probes layout-sensitive behavior.")
    if any(token in lower for token in ("random", "inplace", "alias")):
        yield _axis("determinism", "probe", f"{kind} intentionally probes determinism/aliasing behavior.")
    yield _axis("error_equivalence", "probe", f"{kind} is a targeted boundary probe.")


def _case_level_axes(case: Case) -> Iterable[ContractAxisPolicy]:
    if _case_contains_special_float(case):
        yield _axis("nan", "boundary", "Input data contains NaN/Inf values.")


def _next_order_defined(order_defined: bool, operation: Mapping[str, Any]) -> bool:
    kind = op_kind(operation)
    if kind in {"sort", "running_sum", "row_number_filter", "sortedness_check"}:
        return True
    if kind in {
        "filter",
        "tuple_absence_filter",
        "drop_nulls",
        "semi_join",
        "anti_join",
        "fill_null",
        "coalesce",
        "case_when",
        "select",
        "mutate",
        "limit",
        "offset",
    }:
        return order_defined
    return False


def _column_nullable(case: Case, column: str) -> bool:
    for table in case.tables:
        for spec in table.columns:
            if spec.name == column and spec.nullable:
                return True
        for row in table.rows:
            if row.get(column) is None and column in row:
                return True
    return False


def _case_contains_special_float(case: Case) -> bool:
    return any(_is_special_float(value) for table in case.tables for row in table.rows for value in row.values())


def _is_special_float(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def _axis(axis: str, policy: str, reason: str, *signals: str) -> ContractAxisPolicy:
    return ContractAxisPolicy(axis=axis, policy=policy, reason=reason, signals=tuple(_unique(signals)))


def _strict_axis(axis: str) -> ContractAxisPolicy:
    return _axis(axis, "strict", "No operation declares a semantic boundary for this axis.")


def _join_axis_policies(axis: str, policies: list[ContractAxisPolicy]) -> ContractAxisPolicy:
    if not policies:
        return _strict_axis(axis)
    selected = max(policies, key=lambda row: _POLICY_RANK.get(row.policy, 0))
    signals = tuple(_unique(signal for policy in policies for signal in policy.signals))
    return ContractAxisPolicy(axis=axis, policy=selected.policy, reason=selected.reason, signals=signals)


def _dedupe_axis_policies(policies: Iterable[ContractAxisPolicy]) -> list[ContractAxisPolicy]:
    by_axis: dict[str, list[ContractAxisPolicy]] = {}
    for policy in policies:
        if policy.axis in CONTRACT_AXES and policy.policy in CONTRACT_POLICIES:
            by_axis.setdefault(policy.axis, []).append(policy)
    return [_join_axis_policies(axis, rows) for axis, rows in by_axis.items()]


def _tags_for_axes(kind: str, axes: Iterable[ContractAxisPolicy]) -> list[str]:
    tags = [f"op:{kind}"] if kind else []
    tags.extend(f"contract:{axis.axis}:{axis.policy}" for axis in axes)
    return _unique(tags)


def _finding_value(finding: Any, key: str, default: str = "") -> str:
    if isinstance(finding, Mapping):
        return str(finding.get(key, default) or default)
    return str(getattr(finding, key, default) or default)


def _unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
