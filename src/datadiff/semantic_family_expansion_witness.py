"""Bounded deterministic witnesses for the six family-universe expansions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.family_witness_registry import family_witness_registration
from datadiff.operation_semantics import op_kind
from datadiff.semantic_family_universe import semantic_family_definition


SEMANTIC_FAMILY_EXPANSION_SCHEMA_VERSION = "semantic-family-expansion-witness-v1"

PANDAS_NULLABLE_BOOL_FAMILY_ID = "pandas_nullable_bool_reduction"
SQLITE_AFFINITY_FAMILY_ID = "sqlite_affinity_null_ordered_cut"
POLARS_LAZY_OPTIMIZER_FAMILY_ID = "polars_lazy_filter_groupby_window"
POLARS_LAZY_TEMPORAL_FAMILY_ID = "polars_lazy_temporal_cast_boundary"
PYARROW_ENCODED_NESTED_FAMILY_ID = "pyarrow_encoded_nested_compute"
DATAFUSION_WINDOW_JOIN_FAMILY_ID = "datafusion_window_order_join_interaction"


@dataclass(frozen=True, slots=True)
class SemanticFamilyExpansionGenerationResult:
    case: Case
    trace: dict[str, Any]


def generate_pandas_nullable_bool_witness_case(
    seed: int, *, profile: str = ""
) -> SemanticFamilyExpansionGenerationResult:
    return _generate_case(PANDAS_NULLABLE_BOOL_FAMILY_ID, seed, profile=profile)


def generate_sqlite_affinity_witness_case(
    seed: int, *, profile: str = ""
) -> SemanticFamilyExpansionGenerationResult:
    return _generate_case(SQLITE_AFFINITY_FAMILY_ID, seed, profile=profile)


def generate_polars_lazy_optimizer_witness_case(
    seed: int, *, profile: str = ""
) -> SemanticFamilyExpansionGenerationResult:
    return _generate_case(POLARS_LAZY_OPTIMIZER_FAMILY_ID, seed, profile=profile)


def generate_polars_lazy_temporal_witness_case(
    seed: int, *, profile: str = ""
) -> SemanticFamilyExpansionGenerationResult:
    return _generate_case(POLARS_LAZY_TEMPORAL_FAMILY_ID, seed, profile=profile)


def generate_pyarrow_encoded_nested_witness_case(
    seed: int, *, profile: str = ""
) -> SemanticFamilyExpansionGenerationResult:
    return _generate_case(PYARROW_ENCODED_NESTED_FAMILY_ID, seed, profile=profile)


def generate_datafusion_window_join_witness_case(
    seed: int, *, profile: str = ""
) -> SemanticFamilyExpansionGenerationResult:
    return _generate_case(DATAFUSION_WINDOW_JOIN_FAMILY_ID, seed, profile=profile)


def semantic_family_expansion_static_preconditions(
    family_id: str,
    axes: Mapping[str, Any],
    case: Case,
) -> bool:
    """Validate one family cell using only its program, data, and metadata."""

    try:
        registration = family_witness_registration(family_id)
        definition = semantic_family_definition(family_id)
    except KeyError:
        return False
    normalized_axes = {str(key): str(value) for key, value in axes.items()}
    if set(normalized_axes) != set(registration.axis_names):
        return False
    if any(
        normalized_axes[name] not in values
        for name, values in registration.axes
    ):
        return False
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("semantic_family_expansion_witness", {})
    family_witness = metadata.get("family_witness", {})
    if not isinstance(witness, Mapping) or not isinstance(family_witness, Mapping):
        return False
    if (
        str(witness.get("family_id", "")) != family_id
        or str(witness.get("target_backend", "")) != definition.target_backend
        or list(witness.get("root_ids", []) or []) != [definition.root_id]
        or bool(witness.get("canonical_case_replay", True))
        or bool(witness.get("runtime_corpus_io", True))
        or family_witness.get("axes") != normalized_axes
    ):
        return False
    kinds = tuple(op_kind(operation) for operation in case.program.operations)
    if tuple(witness.get("operation_kinds", ()) or ()) != kinds:
        return False
    if family_id == PANDAS_NULLABLE_BOOL_FAMILY_ID:
        return _pandas_preconditions(normalized_axes, case)
    if family_id == SQLITE_AFFINITY_FAMILY_ID:
        return _sqlite_preconditions(normalized_axes, case)
    if family_id == POLARS_LAZY_OPTIMIZER_FAMILY_ID:
        return _polars_lazy_optimizer_preconditions(normalized_axes, case)
    if family_id == POLARS_LAZY_TEMPORAL_FAMILY_ID:
        return _polars_lazy_temporal_preconditions(normalized_axes, case)
    if family_id == PYARROW_ENCODED_NESTED_FAMILY_ID:
        return _pyarrow_encoded_preconditions(normalized_axes, case)
    if family_id == DATAFUSION_WINDOW_JOIN_FAMILY_ID:
        return _datafusion_window_join_preconditions(normalized_axes, case)
    return False


def _generate_case(
    family_id: str,
    seed: int,
    *,
    profile: str,
) -> SemanticFamilyExpansionGenerationResult:
    registration = family_witness_registration(family_id)
    definition = semantic_family_definition(family_id)
    cell_index, axes = registration.cell_for_seed(seed)
    tables, operations, extra_metadata = _build_payload(family_id, axes)
    operation_kinds = [op_kind(operation) for operation in operations]
    family_witness = {
        "schema_version": "family-witness-case-v1",
        "family_id": family_id,
        "generation_mode": registration.generation_mode,
        "goal_id": registration.goal_id,
        "root_ids": [definition.root_id],
        "cell_index": cell_index,
        "axes": dict(axes),
        "palette_dimensions": [axes[name] for name in registration.axis_names],
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    witness = {
        "schema_version": SEMANTIC_FAMILY_EXPANSION_SCHEMA_VERSION,
        "family_id": family_id,
        "mechanism_id": definition.mechanism_id,
        "goal_id": registration.goal_id,
        "root_ids": [definition.root_id],
        "target_backend": definition.target_backend,
        "control_backends": list(definition.control_backends),
        "cell_index": cell_index,
        "axes": dict(axes),
        "operation_kinds": operation_kinds,
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "bounded_for_cache_reuse": True,
        **extra_metadata,
    }
    trace = {
        "schema_version": SEMANTIC_FAMILY_EXPANSION_SCHEMA_VERSION,
        "generation_mode": registration.generation_mode,
        "selected_goal": {
            "goal_id": registration.goal_id,
            "mechanism_id": definition.mechanism_id,
            "target_backend": definition.target_backend,
        },
        "selection_reason": "family_universe_v1_axis_product_cycle",
        "builder_variant": {
            "variant_id": ":".join(axes[name] for name in registration.axis_names),
            "variant_family": "semantic_family_universe_expansion_v1",
            "cell_index": cell_index,
            "construction_seed": cell_index,
            "case_seed": int(seed),
            "root_guidance": [definition.root_id],
            "axes": dict(axes),
        },
        "requested_profile": str(profile or ""),
        "activation_requirement": {
            "required": True,
            "policy": "deterministic_family_universe_witness",
        },
        "semantic_family_expansion_witness": witness,
        "family_witness": family_witness,
        "valid": True,
        "constructible": True,
        "skip_reason": "",
    }
    metadata: dict[str, Any] = {
        "generator_profile": f"{family_id}_witness",
        "generation_mode": registration.generation_mode,
        "goal_id": registration.goal_id,
        "goal_fault_models": [definition.mechanism_id],
        "semantic_activation_syntactic_reached": True,
        "goal_first_generation": trace,
        "goal_builder_variant": dict(trace["builder_variant"]),
        "semantic_witness_builder_variant": trace["builder_variant"]["variant_id"],
        "semantic_witness_data_pattern": {
            "pattern_id": axes.get("data_pattern", axes.get("boundary", "")),
        },
        "semantic_family_expansion_witness": witness,
        "family_witness": family_witness,
        **extra_metadata,
    }
    case = Case(
        case_id=f"case-{int(seed):08d}-{family_id}-{cell_index:03d}",
        seed=int(seed),
        tables=tables,
        program=Program(
            f"prog-{int(seed):08d}-{family_id}-{cell_index:03d}",
            int(seed),
            operations,
        ),
        metadata=metadata,
    )
    from datadiff.semantic_core.activation import evaluate_semantic_activation

    activation = evaluate_semantic_activation(
        case,
        goal_id=registration.goal_id,
        syntactic_reached=True,
    ).to_dict()
    case.metadata["semantic_activation"] = activation
    trace["semantic_activation"] = activation
    return SemanticFamilyExpansionGenerationResult(case=case, trace=trace)


def _build_payload(
    family_id: str,
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    if family_id == PANDAS_NULLABLE_BOOL_FAMILY_ID:
        return _pandas_payload(axes)
    if family_id == SQLITE_AFFINITY_FAMILY_ID:
        return _sqlite_payload(axes)
    if family_id == POLARS_LAZY_OPTIMIZER_FAMILY_ID:
        return _polars_lazy_optimizer_payload(axes)
    if family_id == POLARS_LAZY_TEMPORAL_FAMILY_ID:
        return _polars_lazy_temporal_payload(axes)
    if family_id == PYARROW_ENCODED_NESTED_FAMILY_ID:
        return _pyarrow_encoded_payload(axes)
    if family_id == DATAFUSION_WINDOW_JOIN_FAMILY_ID:
        return _datafusion_window_join_payload(axes)
    raise KeyError(family_id)


def _pandas_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    values = {
        "true_null": [True, None],
        "false_null": [False, None],
        "all_null": [None, None],
    }[axes["data_pattern"]]
    rows = [
        {"id": index + 1, "g": "a", "flag": value}
        for index, value in enumerate(values)
    ]
    probe_kind = {
        "frame": "bool_reduction_skipna_probe",
        "groupby": "arrow_bool_groupby_reduction_probe",
    }[axes["context"]]
    operation = {
        "op": probe_kind,
        "as": "nullable_bool_mismatch",
        "family_id": PANDAS_NULLABLE_BOOL_FAMILY_ID,
        "context": axes["context"],
        "reduction": axes["reduction"],
        "data_pattern": axes["data_pattern"],
        "values": values,
    }
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    return [table], [operation], {"nullable_bool_values": values}


def _sqlite_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    pattern = axes["data_pattern"]
    cast_mode = axes["cast_mode"]
    if cast_mode == "numeric_text_to_int":
        values = {
            "boundary": ["-10", "0", "2", "10"],
            "duplicates": ["2", "2", "10", "10"],
            "nullable": [None, "0", "2", "10"],
        }[pattern]
        column_type = "str"
        expression = {
            "kind": "cast",
            "source": "raw",
            "to": "int",
            "input_domain": "integer_string",
        }
    else:
        values = {
            "boundary": [-10, 0, 2, 10],
            "duplicates": [2, 2, 10, 10],
            "nullable": [None, 0, 2, 10],
        }[pattern]
        column_type = "int"
        expression = {"kind": "cast", "source": "raw", "to": "str"}
    rows = [{"id": index + 1, "raw": value} for index, value in enumerate(values)]
    operations = [
        {"op": "mutate", "column": "casted", "expr": expression},
        {
            "op": "sort",
            "keys": [
                {
                    "column": "casted",
                    "ascending": True,
                    "nulls": axes["null_order"],
                },
                {"column": "id", "ascending": True, "nulls": "last"},
            ],
        },
        {"op": "offset", "n": 1},
        {"op": "limit", "n": 2},
        {"op": "select", "columns": ["id", "raw", "casted"]},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("raw", column_type, nullable=pattern == "nullable"),
        ],
        rows,
    )
    return [table], operations, {"cast_values": values}


def _polars_lazy_optimizer_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    rows = {
        "balanced": [
            {"id": 1, "g": "a", "x": 1.0},
            {"id": 2, "g": "a", "x": 2.0},
            {"id": 3, "g": "b", "x": -1.0},
            {"id": 4, "g": "b", "x": 3.0},
        ],
        "duplicate_order": [
            {"id": 1, "g": "a", "x": 1.0},
            {"id": 1, "g": "a", "x": 2.0},
            {"id": 2, "g": "b", "x": 1.0},
            {"id": 2, "g": "b", "x": 3.0},
        ],
        "nullable": [
            {"id": 1, "g": "a", "x": None},
            {"id": 2, "g": "a", "x": 2.0},
            {"id": 3, "g": "b", "x": 0.0},
            {"id": 4, "g": "b", "x": None},
        ],
    }[axes["data_pattern"]]
    comparator, value = {
        "ge_zero": (">=", 0.0),
        "le_two": ("<=", 2.0),
    }[axes["filter_mode"]]
    ascending = axes["window_order"] == "asc"
    operations = [
        {"op": "filter", "column": "x", "cmp": comparator, "value": value},
        {
            "op": "running_sum",
            "source": "x",
            "column": "run_x",
            "partition_by": ["g"],
            "order_by": [
                {"column": "id", "ascending": ascending, "nulls": "last"},
            ],
            "input_dtype": "float64",
        },
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [{"column": "run_x", "func": "max", "as": "max_run_x"}],
        },
        {
            "op": "sort",
            "keys": [
                {"column": "g", "ascending": True, "nulls": "last"},
                {"column": "max_run_x", "ascending": False, "nulls": "last"},
            ],
        },
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "float", nullable=True),
        ],
        rows,
    )
    return [table], operations, {"filter_value": value}


def _polars_lazy_temporal_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    probe_kind = {
        "precision_cast": "timestamp_precision_filter_probe",
        "timezone_convert": "polars_timezone_filter_probe",
    }[axes["temporal_op"]]
    operation = {
        "op": probe_kind,
        "as": "temporal_cast_mismatch",
        "family_id": POLARS_LAZY_TEMPORAL_FAMILY_ID,
        "temporal_op": axes["temporal_op"],
        "direction": axes["direction"],
        "boundary": axes["boundary"],
    }
    table = TableData(
        "t0",
        [ColumnSpec("id", "int", nullable=False)],
        [{"id": 1}],
    )
    return [table], [operation], {}


def _pyarrow_encoded_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    probe_kind = {
        "run_end_null": "run_end_null_compute_probe",
        "list_parent": "list_flatten_parent_indices_probe",
        "large_string_partition": "large_string_partition_probe",
    }[axes["probe"]]
    rows = {
        "null_heavy": [
            {"id": 1, "label": "a", "flag": None},
            {"id": 2, "label": None, "flag": False},
            {"id": 3, "label": "b", "flag": None},
        ],
        "duplicate_boundary": [
            {"id": 1, "label": "a", "flag": True},
            {"id": 2, "label": "a", "flag": False},
            {"id": 3, "label": "b", "flag": None},
        ],
    }[axes["data_pattern"]]
    operation = {
        "op": probe_kind,
        "as": "encoded_nested_mismatch",
        "family_id": PYARROW_ENCODED_NESTED_FAMILY_ID,
        "probe": axes["probe"],
        "layout": axes["layout"],
        "data_pattern": axes["data_pattern"],
    }
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("label", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    input_layouts = {
        "t0": {
            "representation": axes["layout"],
            "chunk_count": 2 if axes["layout"] == "chunked" else 1,
        }
    }
    return [table], [operation], {"input_layouts": input_layouts}


def _datafusion_window_join_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    left_rows = {
        "balanced": [
            {"id": 1, "g": "a", "x": 1.0},
            {"id": 2, "g": "a", "x": 2.0},
            {"id": 3, "g": "b", "x": 3.0},
            {"id": 4, "g": "b", "x": 4.0},
        ],
        "duplicate_join": [
            {"id": 1, "g": "a", "x": 1.0},
            {"id": 2, "g": "a", "x": 2.0},
            {"id": 2, "g": "b", "x": 3.0},
            {"id": 3, "g": "b", "x": 4.0},
        ],
        "nullable_payload": [
            {"id": 1, "g": "a", "x": None},
            {"id": 2, "g": "a", "x": 2.0},
            {"id": 3, "g": "b", "x": 3.0},
            {"id": 4, "g": "b", "x": None},
        ],
    }[axes["data_pattern"]]
    right_rows = [
        {"rid": 1, "weight": 10.0},
        {"rid": 2, "weight": 20.0},
        {"rid": 3, "weight": 30.0},
    ]
    if axes["data_pattern"] == "duplicate_join":
        right_rows.append({"rid": 2, "weight": 21.0})
    ascending = axes["order_mode"] == "asc"
    operations = [
        {
            "op": "join",
            "table": "t1",
            "left_on": "id",
            "right_on": "rid",
            "how": axes["join_mode"],
        },
        {
            "op": "running_sum",
            "source": "x",
            "column": "run_x",
            "partition_by": ["g"],
            "order_by": [
                {"column": "id", "ascending": ascending, "nulls": "last"},
                {"column": "weight", "ascending": True, "nulls": "last"},
            ],
            "input_dtype": "float64",
        },
        {
            "op": "row_number_filter",
            "partition_by": ["g"],
            "order_by": [
                {"column": "run_x", "ascending": False, "nulls": "last"},
                {"column": "id", "ascending": True, "nulls": "last"},
                {"column": "weight", "ascending": True, "nulls": "last"},
            ],
            "cmp": "<=",
            "value": 2,
        },
        {
            "op": "sort",
            "keys": [
                {"column": "g", "ascending": True, "nulls": "last"},
                {"column": "id", "ascending": ascending, "nulls": "last"},
                {"column": "weight", "ascending": True, "nulls": "last"},
            ],
        },
        {"op": "offset", "n": 1},
        {"op": "limit", "n": 3},
    ]
    tables = [
        TableData(
            "t0",
            [
                ColumnSpec("id", "int", nullable=False),
                ColumnSpec("g", "str", nullable=False),
                ColumnSpec("x", "float", nullable=True),
            ],
            left_rows,
        ),
        TableData(
            "t1",
            [
                ColumnSpec("rid", "int", nullable=False),
                ColumnSpec("weight", "float", nullable=False),
            ],
            right_rows,
        ),
    ]
    return tables, operations, {}


def _pandas_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    if len(case.program.operations) != 1:
        return False
    operation = case.program.operations[0]
    expected = {
        "frame": "bool_reduction_skipna_probe",
        "groupby": "arrow_bool_groupby_reduction_probe",
    }[axes["context"]]
    values = [row.get("flag") for row in case.tables[0].rows]
    expected_values = {
        "true_null": [True, None],
        "false_null": [False, None],
        "all_null": [None, None],
    }[axes["data_pattern"]]
    return bool(
        op_kind(operation) == expected
        and str(operation.get("reduction", "")) == axes["reduction"]
        and str(operation.get("data_pattern", "")) == axes["data_pattern"]
        and values == expected_values
    )


def _sqlite_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = tuple(op_kind(item) for item in case.program.operations)
    sort = case.program.operations[1] if len(case.program.operations) > 1 else {}
    keys = list(sort.get("keys", []) or [])
    return bool(
        kinds == ("mutate", "sort", "offset", "limit", "select")
        and case.program.operations[2].get("n") == 1
        and case.program.operations[3].get("n") == 2
        and keys
        and keys[0].get("nulls") == axes["null_order"]
    )


def _polars_lazy_optimizer_preconditions(
    axes: Mapping[str, str], case: Case
) -> bool:
    kinds = tuple(op_kind(item) for item in case.program.operations)
    running = case.program.operations[1] if len(case.program.operations) > 1 else {}
    order_by = list(running.get("order_by", []) or [])
    return bool(
        kinds == ("filter", "running_sum", "groupby", "sort")
        and running.get("partition_by") == ["g"]
        and order_by
        and bool(order_by[0].get("ascending", True))
        == (axes["window_order"] == "asc")
    )


def _polars_lazy_temporal_preconditions(
    axes: Mapping[str, str], case: Case
) -> bool:
    expected = {
        "precision_cast": "timestamp_precision_filter_probe",
        "timezone_convert": "polars_timezone_filter_probe",
    }[axes["temporal_op"]]
    operation = case.program.operations[0] if case.program.operations else {}
    return bool(
        len(case.program.operations) == 1
        and op_kind(operation) == expected
        and str(operation.get("direction", "")) == axes["direction"]
        and str(operation.get("boundary", "")) == axes["boundary"]
    )


def _pyarrow_encoded_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    expected = {
        "run_end_null": "run_end_null_compute_probe",
        "list_parent": "list_flatten_parent_indices_probe",
        "large_string_partition": "large_string_partition_probe",
    }[axes["probe"]]
    layouts = case.metadata.get("input_layouts", {})
    layout = layouts.get("t0", {}) if isinstance(layouts, Mapping) else {}
    operation = case.program.operations[0] if case.program.operations else {}
    return bool(
        len(case.program.operations) == 1
        and op_kind(operation) == expected
        and isinstance(layout, Mapping)
        and str(layout.get("representation", "")) == axes["layout"]
    )


def _datafusion_window_join_preconditions(
    axes: Mapping[str, str], case: Case
) -> bool:
    kinds = tuple(op_kind(item) for item in case.program.operations)
    join = case.program.operations[0] if case.program.operations else {}
    running = case.program.operations[1] if len(case.program.operations) > 1 else {}
    row_number = case.program.operations[2] if len(case.program.operations) > 2 else {}
    final_sort = case.program.operations[3] if len(case.program.operations) > 3 else {}
    running_order = list(running.get("order_by", []) or [])
    row_number_order = list(row_number.get("order_by", []) or [])
    final_sort_keys = list(final_sort.get("keys", []) or [])
    return bool(
        kinds
        == ("join", "running_sum", "row_number_filter", "sort", "offset", "limit")
        and join.get("how") == axes["join_mode"]
        and running.get("partition_by") == ["g"]
        and [item.get("column") for item in running_order] == ["id", "weight"]
        and [item.get("column") for item in row_number_order]
        == ["run_x", "id", "weight"]
        and [item.get("column") for item in final_sort_keys]
        == ["g", "id", "weight"]
        and case.program.operations[4].get("n") == 1
        and case.program.operations[5].get("n") == 3
    )
