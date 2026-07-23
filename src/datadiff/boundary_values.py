from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.operation_semantics import (
    expr_input_domain,
    expr_kind,
    expr_source,
    op_kind,
)


BOUNDARY_LIBRARY_SCHEMA_VERSION = "fault-model-boundary-library-v1"


@dataclass(frozen=True, slots=True)
class BoundaryProfile:
    profile_id: str
    logical_types: tuple[str, ...]
    operation_kinds: tuple[str, ...]
    fault_models: tuple[str, ...]
    values_by_type: dict[str, tuple[Any, ...]]
    compatible_input_domains: tuple[str, ...] = ()
    layout_mode: str = ""
    row_shape: str = "preserve"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BOUNDARY_LIBRARY_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "logical_types": list(self.logical_types),
            "operation_kinds": list(self.operation_kinds),
            "fault_models": list(self.fault_models),
            "values_by_type": {
                key: [_value_label(value) for value in values]
                for key, values in sorted(self.values_by_type.items())
            },
            "compatible_input_domains": list(self.compatible_input_domains),
            "layout_mode": self.layout_mode,
            "row_shape": self.row_shape,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class BoundaryApplication:
    case: Case
    trace: dict[str, Any]


_PROFILES: tuple[BoundaryProfile, ...] = (
    BoundaryProfile(
        profile_id="null_truth_table",
        logical_types=("int", "float", "str", "bool"),
        operation_kinds=(
            "filter",
            "mutate",
            "case_when",
            "join",
            "semi_join",
            "anti_join",
            "tuple_absence_filter",
            "groupby",
            "aggregate",
            "sort",
            "sortedness_check",
        ),
        fault_models=("null_semantics", "three_valued_logic", "join_null_semantics"),
        values_by_type={
            "int": (None, 0, 1, -1),
            "float": (None, 0.0, -0.0, 1.0),
            "str": (None, "", "missing", " "),
            "bool": (None, True, False),
        },
        rationale="Exercise NULL/unknown truth tables only where the program consumes null-sensitive state.",
    ),
    BoundaryProfile(
        profile_id="signed_zero_nan_infinity",
        logical_types=("float",),
        operation_kinds=("filter", "sort", "groupby", "aggregate", "mutate", "running_sum"),
        fault_models=("numeric_precision", "negative_zero", "nan_semantics", "ordering_stability"),
        values_by_type={
            "float": (-0.0, 0.0, float("nan"), float("inf"), float("-inf"), 5e-324),
        },
        rationale="Target IEEE-754 equality, ordering, grouping and accumulation boundaries.",
    ),
    BoundaryProfile(
        profile_id="int64_overflow_edges",
        logical_types=("int",),
        operation_kinds=("filter", "mutate", "aggregate", "groupby", "running_sum"),
        fault_models=("numeric_precision", "overflow", "cast_semantics"),
        values_by_type={
            "int": (
                -(2**63),
                -(2**63) + 1,
                -(2**53) - 1,
                2**53 + 1,
                2**63 - 2,
                2**63 - 1,
            )
        },
        rationale="Exercise integer overflow and the IEEE-754 exactness boundary without producing out-of-domain int64 values.",
    ),
    BoundaryProfile(
        profile_id="decimal_scale_edges",
        logical_types=("float",),
        operation_kinds=("mutate", "filter", "groupby", "aggregate", "sort"),
        fault_models=("decimal_scale", "rounding", "numeric_precision"),
        values_by_type={
            "float": (0.005, 0.015, 1.005, -1.005, 999999.995, -999999.995),
        },
        rationale="Approximate decimal scale/rounding boundaries in the current float DSL; native decimal remains an explicit P6 capability extension.",
    ),
    BoundaryProfile(
        profile_id="timezone_transition_strings",
        logical_types=("str",),
        operation_kinds=("mutate", "filter", "sort", "groupby"),
        fault_models=("timezone", "timestamp_precision", "cast_semantics"),
        values_by_type={
            "str": (
                "1969-12-31T23:59:59.999999999Z",
                "1970-01-01T00:00:00Z",
                "2024-03-10T01:59:59-08:00",
                "2024-03-10T03:00:00-07:00",
                "2024-11-03T01:30:00-07:00",
                "2024-11-03T01:30:00-08:00",
            )
        },
        compatible_input_domains=("datetime_string",),
        rationale="Target epoch, precision and daylight-saving transition boundaries for date/timestamp expressions.",
    ),
    BoundaryProfile(
        profile_id="dictionary_chunk_slice_layout",
        logical_types=("str", "bool"),
        operation_kinds=("filter", "distinct", "groupby", "join", "semi_join", "anti_join"),
        fault_models=("dictionary_encoding", "chunk_boundary", "slice_offset", "layout_semantics"),
        values_by_type={
            "str": ("alpha", "alpha", "", None, "omega"),
            "bool": (True, True, False, None),
        },
        layout_mode="dictionary_chunked_slice",
        rationale="Preserve semantic values while exposing dictionary, multi-chunk and non-zero slice offsets in provenance.",
    ),
    BoundaryProfile(
        profile_id="run_end_encoding_layout",
        logical_types=("bool", "int"),
        operation_kinds=("filter", "groupby", "aggregate", "distinct"),
        fault_models=("run_end_encoding", "null_bitmap", "layout_semantics"),
        values_by_type={
            "bool": (True, True, True, None, False, False),
            "int": (1, 1, 1, 1, 0, 0),
        },
        layout_mode="run_end_encoded",
        rationale="Create long runs and nullable transitions suitable for run-end encoding checks.",
    ),
    BoundaryProfile(
        profile_id="empty_singleton_transition",
        logical_types=("int", "float", "str", "bool"),
        operation_kinds=("union_all", "groupby", "aggregate", "distinct", "limit", "offset"),
        fault_models=("empty_input_handling", "singleton_cardinality", "union_all_semantics"),
        values_by_type={},
        row_shape="empty_or_singleton",
        rationale="Target zero-to-one row cardinality transitions only for cardinality-sensitive operations.",
    ),
    BoundaryProfile(
        profile_id="duplicate_heavy_keys",
        logical_types=("int", "str", "bool"),
        operation_kinds=("join", "semi_join", "anti_join", "union_all", "distinct", "groupby"),
        fault_models=("duplicate_semantics", "join_cardinality", "set_vs_bag"),
        values_by_type={
            "int": (0, 0, 0, 1, 1),
            "str": ("dup", "dup", "dup", "other"),
            "bool": (True, True, True, False),
        },
        row_shape="duplicate_heavy",
        rationale="Increase multiplicity only when bag/set or join cardinality is semantically relevant.",
    ),
)


def boundary_profiles() -> tuple[BoundaryProfile, ...]:
    return _PROFILES


def boundary_profile(profile_id: str) -> BoundaryProfile:
    return next(profile for profile in _PROFILES if profile.profile_id == profile_id)


def applicable_boundary_profiles(
    case: Case,
    *,
    fault_models: tuple[str, ...] | list[str] = (),
) -> tuple[BoundaryProfile, ...]:
    logical_types = {column.type for table in case.tables for column in table.columns}
    operation_kinds = {op_kind(operation) for operation in case.program.operations}
    requested_faults = {str(item) for item in fault_models if str(item)}
    return tuple(
        profile
        for profile in _PROFILES
        if logical_types.intersection(profile.logical_types)
        and operation_kinds.intersection(profile.operation_kinds)
        and (
            not requested_faults
            or requested_faults.intersection(profile.fault_models)
        )
    )


def apply_targeted_boundary_profile(
    case: Case,
    *,
    seed: int,
    fault_models: tuple[str, ...] | list[str] = (),
    preferred_profile_ids: tuple[str, ...] | list[str] = (),
    _trial_copy: bool = False,
) -> BoundaryApplication:
    structurally_applicable = list(
        applicable_boundary_profiles(case, fault_models=())
    )
    applicable = list(
        applicable_boundary_profiles(case, fault_models=fault_models)
    )
    preferred = [
        profile
        for profile_id in preferred_profile_ids
        for profile in structurally_applicable
        if profile.profile_id == profile_id
    ]
    candidates = preferred or applicable
    if not candidates:
        result = _clone_boundary_case(case, trial_copy=_trial_copy)
        trace = {
            "schema_version": BOUNDARY_LIBRARY_SCHEMA_VERSION,
            "applied": False,
            "profile_id": "",
            "skip_reason": "no_applicable_profile",
            "fault_models": sorted(set(str(item) for item in fault_models)),
        }
        result.metadata = dict(result.metadata or {})
        result.metadata["boundary_application"] = trace
        return BoundaryApplication(
            case=result,
            trace=trace,
        )
    profile = candidates[int(seed) % len(candidates)]
    result = _clone_boundary_case(case, trial_copy=_trial_copy)
    changed_cells = _inject_profile_values(result, profile)
    row_shape_changes = _apply_row_shape(result, profile, seed=seed)
    layout_changes = _apply_layout_mode(result, profile)
    applied = bool(changed_cells or row_shape_changes or layout_changes)
    trace = {
        "schema_version": BOUNDARY_LIBRARY_SCHEMA_VERSION,
        "applied": applied,
        "profile_id": profile.profile_id,
        "selection_basis": "preferred_profile_id" if preferred else "fault_model_match",
        "fault_models": list(profile.fault_models),
        "applicability": {
            "logical_types": list(profile.logical_types),
            "operation_kinds": list(profile.operation_kinds),
        },
        "changed_cells": changed_cells,
        "row_shape_changes": row_shape_changes,
        "layout_changes": layout_changes,
        "strict_native_value_labels": {
            key: [_value_label(value) for value in values]
            for key, values in profile.values_by_type.items()
        },
        "skip_reason": "" if applied else "applicable_profile_had_no_mutable_target",
    }
    result.metadata = dict(result.metadata or {})
    result.metadata["boundary_application"] = trace
    return BoundaryApplication(case=result, trace=trace)


def _clone_boundary_case(case: Case, *, trial_copy: bool) -> Case:
    # Rows must always be copied explicitly: ``TableData.to_dict`` exposes its
    # row list directly, so a Case.from_dict(case.to_dict()) round trip aliases
    # the source and makes duplicate-heavy trials grow it on every attempt.
    # Internal activation trials can share the read-only program and columns;
    # the public/default path preserves full object independence.
    return Case(
        case_id=case.case_id,
        seed=case.seed,
        tables=[
            TableData(
                name=table.name,
                columns=(
                    table.columns
                    if trial_copy
                    else [
                        ColumnSpec(column.name, column.type, column.nullable)
                        for column in table.columns
                    ]
                ),
                rows=[dict(row) for row in table.rows],
            )
            for table in case.tables
        ],
        program=(
            case.program
            if trial_copy
            else Program(
                case.program.program_id,
                case.program.seed,
                [operation.copy() for operation in case.program.operations],
            )
        ),
        metadata=(
            dict(case.metadata or {})
            if trial_copy
            else copy.deepcopy(case.metadata or {})
        ),
    )


def _inject_profile_values(case: Case, profile: BoundaryProfile) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    restricted_domains = _restricted_input_domains(case)
    compatible_domains = set(profile.compatible_input_domains)
    for table in case.tables:
        candidates = [
            column
            for column in table.columns
            if column.type in profile.values_by_type
            and column.name not in {"id", "row_id", "sample_id"}
            and _profile_matches_column_domain(
                column.name,
                restricted_domains,
                compatible_domains,
            )
        ]
        if not candidates:
            candidates = [
                column
                for column in table.columns
                if column.type in profile.values_by_type
                and _profile_matches_column_domain(
                    column.name,
                    restricted_domains,
                    compatible_domains,
                )
            ]
        for column in candidates:
            values = profile.values_by_type.get(column.type, ())
            for index, value in enumerate(values):
                if index >= len(table.rows):
                    break
                if value is None and not column.nullable:
                    continue
                table.rows[index][column.name] = value
                changes.append(
                    {
                        "table": table.name,
                        "row": index,
                        "column": column.name,
                        "value": _value_label(value),
                    }
                )
    return changes


def _restricted_input_domains(case: Case) -> dict[str, set[str]]:
    restricted: dict[str, set[str]] = {}
    for operation in case.program.operations:
        if op_kind(operation) != "mutate":
            continue
        kind = expr_kind(operation)
        source = expr_source(operation)
        if not source:
            continue
        if kind == "date_part":
            restricted.setdefault(source, set()).add("datetime_string")
        elif kind == "cast":
            domain = expr_input_domain(operation)
            if domain in {"integer_string", "numeric_string"}:
                restricted.setdefault(source, set()).add(domain)
    return restricted


def _profile_matches_column_domain(
    column: str,
    restricted_domains: dict[str, set[str]],
    compatible_domains: set[str],
) -> bool:
    required = restricted_domains.get(column, set())
    return not required or bool(required.intersection(compatible_domains))


def _apply_row_shape(
    case: Case,
    profile: BoundaryProfile,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    if profile.row_shape == "empty_or_singleton" and case.tables:
        table = case.tables[0]
        before = len(table.rows)
        table.rows = [] if int(seed) % 2 == 0 else table.rows[:1]
        changes.append(
            {"table": table.name, "before": before, "after": len(table.rows)}
        )
    elif profile.row_shape == "duplicate_heavy":
        for table in case.tables:
            if not table.rows:
                continue
            before = len(table.rows)
            source = dict(table.rows[0])
            while len(table.rows) < max(8, before + 3):
                table.rows.append(dict(source))
            changes.append(
                {"table": table.name, "before": before, "after": len(table.rows)}
            )
    return changes


def _apply_layout_mode(case: Case, profile: BoundaryProfile) -> list[dict[str, Any]]:
    if not profile.layout_mode:
        return []
    case.metadata = dict(case.metadata or {})
    layouts = dict(case.metadata.get("input_layouts", {}) or {})
    changes: list[dict[str, Any]] = []
    for table in case.tables:
        dictionary_columns = [
            column.name for column in table.columns if column.type in {"str", "bool"}
        ]
        layout = dict(layouts.get(table.name, {}) or {})
        layout.update(
            {
                "representation": profile.layout_mode,
                "chunk_count": 3,
                "dictionary_columns": dictionary_columns,
                "attributes": {
                    "slice_offset": 1 if len(table.rows) > 1 else 0,
                    "boundary_profile": profile.profile_id,
                },
            }
        )
        layouts[table.name] = layout
        changes.append({"table": table.name, **layout})
    case.metadata["input_layouts"] = layouts
    return changes


def _value_label(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
        if value == 0.0:
            return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    return repr(value)
