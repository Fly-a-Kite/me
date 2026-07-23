"""Bounded deterministic witnesses for semantic-family universe v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.family_witness_registry import family_witness_registration
from datadiff.operation_semantics import (
    aggregate_functions,
    expr_kinds,
    op_kind,
)
from datadiff.semantic_family_universe_v2 import semantic_family_v2_definition


SEMANTIC_FAMILY_EXPANSION_V2_SCHEMA_VERSION = (
    "semantic-family-expansion-witness-v2"
)

PANDAS_STRING_FAMILY_ID = "pandas_nullable_string_normalization"
PYARROW_STRING_LAYOUT_FAMILY_ID = "pyarrow_string_layout_predicate"
POLARS_ARITHMETIC_FAMILY_ID = "polars_arithmetic_cast_sortedness"
POLARS_LAZY_MEMBERSHIP_FAMILY_ID = "polars_lazy_case_string_membership"
DUCKDB_UNION_AGGREGATE_FAMILY_ID = "duckdb_union_duplicate_global_aggregate"
SQLITE_MEMBERSHIP_FAMILY_ID = "sqlite_three_valued_membership"
DATAFUSION_NULL_SETOP_FAMILY_ID = "datafusion_null_setop_aggregate"

EXPANSION_V2_FAMILY_IDS = frozenset(
    {
        PANDAS_STRING_FAMILY_ID,
        PYARROW_STRING_LAYOUT_FAMILY_ID,
        POLARS_ARITHMETIC_FAMILY_ID,
        POLARS_LAZY_MEMBERSHIP_FAMILY_ID,
        DUCKDB_UNION_AGGREGATE_FAMILY_ID,
        SQLITE_MEMBERSHIP_FAMILY_ID,
        DATAFUSION_NULL_SETOP_FAMILY_ID,
    }
)


@dataclass(frozen=True, slots=True)
class SemanticFamilyExpansionV2GenerationResult:
    case: Case
    trace: dict[str, Any]


def generate_pandas_nullable_string_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(PANDAS_STRING_FAMILY_ID, seed, profile=profile)


def generate_pyarrow_string_layout_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(PYARROW_STRING_LAYOUT_FAMILY_ID, seed, profile=profile)


def generate_polars_arithmetic_sortedness_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(POLARS_ARITHMETIC_FAMILY_ID, seed, profile=profile)


def generate_polars_lazy_string_membership_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(POLARS_LAZY_MEMBERSHIP_FAMILY_ID, seed, profile=profile)


def generate_duckdb_union_aggregate_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(DUCKDB_UNION_AGGREGATE_FAMILY_ID, seed, profile=profile)


def generate_sqlite_three_valued_membership_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(SQLITE_MEMBERSHIP_FAMILY_ID, seed, profile=profile)


def generate_datafusion_null_setop_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV2GenerationResult:
    return _generate_case(DATAFUSION_NULL_SETOP_FAMILY_ID, seed, profile=profile)


def semantic_family_expansion_v2_static_preconditions(
    family_id: str,
    axes: Mapping[str, Any],
    case: Case,
) -> bool:
    """Validate a v2 expansion cell using only static case evidence."""

    if family_id not in EXPANSION_V2_FAMILY_IDS:
        return False
    try:
        registration = family_witness_registration(family_id)
        definition = semantic_family_v2_definition(family_id)
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
    kinds = tuple(op_kind(operation) for operation in case.program.operations)
    if (
        str(witness.get("schema_version", ""))
        != SEMANTIC_FAMILY_EXPANSION_V2_SCHEMA_VERSION
        or str(witness.get("family_id", "")) != family_id
        or str(witness.get("target_backend", "")) != definition.target_backend
        or list(witness.get("root_ids", []) or []) != [definition.root_id]
        or bool(witness.get("canonical_case_replay", True))
        or bool(witness.get("runtime_corpus_io", True))
        or family_witness.get("axes") != normalized_axes
        or tuple(witness.get("operation_kinds", ()) or ()) != kinds
    ):
        return False
    validators = {
        PANDAS_STRING_FAMILY_ID: _pandas_preconditions,
        PYARROW_STRING_LAYOUT_FAMILY_ID: _pyarrow_preconditions,
        POLARS_ARITHMETIC_FAMILY_ID: _polars_preconditions,
        POLARS_LAZY_MEMBERSHIP_FAMILY_ID: _polars_lazy_preconditions,
        DUCKDB_UNION_AGGREGATE_FAMILY_ID: _duckdb_preconditions,
        SQLITE_MEMBERSHIP_FAMILY_ID: _sqlite_preconditions,
        DATAFUSION_NULL_SETOP_FAMILY_ID: _datafusion_preconditions,
    }
    return validators[family_id](normalized_axes, case)


def _generate_case(
    family_id: str,
    seed: int,
    *,
    profile: str,
) -> SemanticFamilyExpansionV2GenerationResult:
    registration = family_witness_registration(family_id)
    definition = semantic_family_v2_definition(family_id)
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
        "schema_version": SEMANTIC_FAMILY_EXPANSION_V2_SCHEMA_VERSION,
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
        "schema_version": SEMANTIC_FAMILY_EXPANSION_V2_SCHEMA_VERSION,
        "generation_mode": registration.generation_mode,
        "selected_goal": {
            "goal_id": registration.goal_id,
            "mechanism_id": definition.mechanism_id,
            "target_backend": definition.target_backend,
        },
        "selection_reason": "family_universe_v2_axis_product_cycle",
        "builder_variant": {
            "variant_id": ":".join(axes[name] for name in registration.axis_names),
            "variant_family": "semantic_family_universe_expansion_v2",
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
        "semantic_witness_builder_variant": trace["builder_variant"][
            "variant_id"
        ],
        "semantic_witness_data_pattern": {
            "pattern_id": axes.get("data_pattern", ""),
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
    return SemanticFamilyExpansionV2GenerationResult(case=case, trace=trace)


def _build_payload(
    family_id: str,
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    builders = {
        PANDAS_STRING_FAMILY_ID: _pandas_payload,
        PYARROW_STRING_LAYOUT_FAMILY_ID: _pyarrow_payload,
        POLARS_ARITHMETIC_FAMILY_ID: _polars_payload,
        POLARS_LAZY_MEMBERSHIP_FAMILY_ID: _polars_lazy_payload,
        DUCKDB_UNION_AGGREGATE_FAMILY_ID: _duckdb_payload,
        SQLITE_MEMBERSHIP_FAMILY_ID: _sqlite_payload,
        DATAFUSION_NULL_SETOP_FAMILY_ID: _datafusion_payload,
    }
    try:
        return builders[family_id](axes)
    except KeyError as exc:
        raise KeyError(family_id) from exc


def _pandas_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    rows = {
        "unicode_empty": [
            {"id": 1, "g": "a", "s": " Café ", "alt": None},
            {"id": 2, "g": "a", "s": "", "alt": "fallback"},
            {"id": 3, "g": "b", "s": "ßeta", "alt": None},
            {"id": 4, "g": "b", "s": None, "alt": "missing"},
        ],
        "null_duplicates": [
            {"id": 1, "g": "a", "s": None, "alt": "A"},
            {"id": 2, "g": "a", "s": " Foo ", "alt": None},
            {"id": 3, "g": "b", "s": " Foo ", "alt": "B"},
            {"id": 4, "g": "b", "s": None, "alt": None},
        ],
        "whitespace_case": [
            {"id": 1, "g": "a", "s": " Alpha ", "alt": None},
            {"id": 2, "g": "a", "s": "alpha", "alt": None},
            {"id": 3, "g": "b", "s": "B-ETA", "alt": None},
            {"id": 4, "g": "b", "s": "  ", "alt": "blank"},
        ],
    }[axes["data_pattern"]]
    if axes["transform"] == "lower_strip":
        operations = [
            {
                "op": "mutate",
                "column": "s_step",
                "expr": {"kind": "string_strip", "source": "s"},
            },
            {
                "op": "mutate",
                "column": "s_norm",
                "expr": {"kind": "string_lower", "source": "s_step"},
            },
        ]
    else:
        operations = [
            {
                "op": "mutate",
                "column": "s_step",
                "expr": {
                    "kind": "string_replace",
                    "source": "s",
                    "old": "-",
                    "new": "_",
                },
            },
            {
                "op": "mutate",
                "column": "s_norm",
                "expr": {"kind": "string_upper", "source": "s_step"},
            },
        ]
    operations.append(
        {
            "op": "mutate",
            "column": "s_nonempty",
            "expr": {"kind": "string_null_if_empty", "source": "s_norm"},
        }
    )
    if axes["null_policy"] == "coalesce":
        operations.append(
            {
                "op": "coalesce",
                "columns": ["s_nonempty", "alt"],
                "as": "s_key",
                "fallback": "missing",
            }
        )
    else:
        operations.extend(
            [
                {"op": "fill_null", "column": "s_nonempty", "value": "missing"},
                {
                    "op": "mutate",
                    "column": "s_key",
                    "expr": {"kind": "string_lower", "source": "s_nonempty"},
                },
            ]
        )
    operations.extend(
        [
            {
                "op": "groupby",
                "keys": ["s_key"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "g", "func": "nunique", "as": "nunique_g"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_key", "ascending": True, "nulls": "last"}
                ],
            },
        ]
    )
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("alt", "str", nullable=True),
        ],
        rows,
    )
    return [table], operations, {}


def _pyarrow_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    rows = {
        "ascii_nullable": [
            {"id": 1, "s": " alpha "},
            {"id": 2, "s": "beta"},
            {"id": 3, "s": None},
            {"id": 4, "s": "alphabet"},
        ],
        "unicode_duplicates": [
            {"id": 1, "s": " café "},
            {"id": 2, "s": "café"},
            {"id": 3, "s": "βeta"},
            {"id": 4, "s": "café"},
        ],
    }[axes["data_pattern"]]
    predicate_kind = {
        "contains": "string_contains",
        "starts_with": "string_starts_with",
        "ends_with": "string_ends_with",
    }[axes["predicate"]]
    operations = [
        {
            "op": "mutate",
            "column": "s_trim",
            "expr": {"kind": "string_strip", "source": "s"},
        },
        {
            "op": "mutate",
            "column": "matched",
            "expr": {
                "kind": predicate_kind,
                "source": "s_trim",
                "needle": "a",
            },
        },
        {
            "op": "case_when",
            "as": "bucket",
            "condition": {
                "column": "matched",
                "cmp": "bool_is_true",
                "value": None,
            },
            "then": "hit",
            "else": "miss",
        },
        {
            "op": "groupby",
            "keys": ["bucket"],
            "aggs": [
                {"column": "id", "func": "count", "as": "count_id"},
                {"column": "s_trim", "func": "nunique", "as": "nunique_s"},
            ],
        },
        {
            "op": "sort",
            "keys": [
                {"column": "bucket", "ascending": True, "nulls": "last"}
            ],
        },
    ]
    layout = axes["layout"]
    input_layouts = {
        "t0": {
            "representation": layout,
            "chunk_count": 2 if layout == "chunked" else 1,
            "dictionary_columns": ["s"] if layout == "dictionary" else [],
        }
    }
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    return [table], operations, {"input_layouts": input_layouts}


def _polars_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    rows = {
        "boundary": [
            {"id": 1, "x": -5.0},
            {"id": 2, "x": -1.0},
            {"id": 3, "x": 0.0},
            {"id": 4, "x": 6.0},
        ],
        "duplicates": [
            {"id": 1, "x": -2.0},
            {"id": 2, "x": -2.0},
            {"id": 3, "x": 2.0},
            {"id": 4, "x": 2.0},
        ],
        "nullable": [
            {"id": 1, "x": None},
            {"id": 2, "x": -3.0},
            {"id": 3, "x": 1.0},
            {"id": 4, "x": None},
        ],
    }[axes["data_pattern"]]
    if axes["arithmetic"] == "add_clip":
        operations = [
            {
                "op": "mutate",
                "column": "x_step",
                "expr": {"kind": "add_const", "source": "x", "value": 1.0},
            },
            {
                "op": "mutate",
                "column": "x_out",
                "expr": {
                    "kind": "clip",
                    "source": "x_step",
                    "lower": -2.0,
                    "upper": 3.0,
                },
            },
        ]
    else:
        operations = [
            {
                "op": "mutate",
                "column": "x_step",
                "expr": {"kind": "abs", "source": "x"},
            },
            {
                "op": "mutate",
                "column": "x_out",
                "expr": {"kind": "cast", "source": "x_step", "to": "str"},
            },
        ]
    ascending = axes["order_mode"].startswith("asc")
    nulls = "last" if axes["order_mode"].endswith("last") else "first"
    operations.extend(
        [
            {
                "op": "sort",
                "keys": [
                    {
                        "column": "x_out",
                        "ascending": ascending,
                        "nulls": nulls,
                    },
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {
                "op": "sortedness_check",
                "column": "x_out",
                "ascending": ascending,
                "nulls": nulls,
                "as": "is_sorted",
            },
        ]
    )
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("x", "float", nullable=True),
        ],
        rows,
    )
    return [table], operations, {}


def _polars_lazy_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    rows = {
        "duplicate_keys": [
            {"id": 1, "s": " Alpha ", "suffix": "x", "flag": True},
            {"id": 2, "s": "alpha", "suffix": "x", "flag": False},
            {"id": 3, "s": "Beta", "suffix": "y", "flag": True},
            {"id": 4, "s": "Gamma", "suffix": "z", "flag": False},
        ],
        "null_keys": [
            {"id": 1, "s": None, "suffix": "x", "flag": True},
            {"id": 2, "s": "Alpha", "suffix": None, "flag": False},
            {"id": 3, "s": "Beta", "suffix": "y", "flag": None},
            {"id": 4, "s": "Gamma", "suffix": "z", "flag": True},
        ],
        "unicode": [
            {"id": 1, "s": " Café ", "suffix": "x", "flag": True},
            {"id": 2, "s": "CAFÉ", "suffix": "x", "flag": False},
            {"id": 3, "s": "βeta", "suffix": "y", "flag": True},
            {"id": 4, "s": "delta", "suffix": "z", "flag": False},
        ],
    }[axes["data_pattern"]]
    if axes["string_expr"] == "lower_concat":
        operations = [
            {
                "op": "mutate",
                "column": "s_step",
                "expr": {"kind": "string_strip", "source": "s"},
            },
            {
                "op": "mutate",
                "column": "s_norm",
                "expr": {"kind": "string_lower", "source": "s_step"},
            },
            {
                "op": "mutate",
                "column": "member_key",
                "expr": {
                    "kind": "string_concat",
                    "source": "s_norm",
                    "other": "suffix",
                    "sep": ":",
                },
            },
        ]
        lookup_values = ["alpha:x", "beta:y", "café:x"]
    else:
        operations = [
            {
                "op": "mutate",
                "column": "s_step",
                "expr": {"kind": "string_strip", "source": "s"},
            },
            {
                "op": "mutate",
                "column": "s_slice",
                "expr": {
                    "kind": "string_slice",
                    "source": "s_step",
                    "start": 0,
                    "length": 4,
                },
            },
            {
                "op": "mutate",
                "column": "member_key",
                "expr": {
                    "kind": "string_replace",
                    "source": "s_slice",
                    "old": " ",
                    "new": "",
                },
            },
        ]
        lookup_values = ["Alph", "Beta", "Café"]
    operations.extend(
        [
            {
                "op": "case_when",
                "as": "bucket",
                "condition": {
                    "column": "flag",
                    "cmp": "bool_is_true",
                    "value": None,
                },
                "then": "true",
                "else": "other",
            },
            {
                "op": axes["membership"],
                "table": "t1",
                "left_on": "member_key",
                "right_on": "lookup_key",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "bucket", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "member_key", "bucket"]},
        ]
    )
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("suffix", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    right = TableData(
        "t1",
        [ColumnSpec("lookup_key", "str", nullable=False)],
        [{"lookup_key": value} for value in lookup_values],
    )
    return [left, right], operations, {}


def _duckdb_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    left_rows, right_rows = {
        "duplicates": (
            [
                {"id": 1, "g": "a", "x": 1.0},
                {"id": 2, "g": "a", "x": 2.0},
            ],
            [
                {"id": 2, "g": "a", "x": 2.0},
                {"id": 3, "g": "b", "x": 3.0},
            ],
        ),
        "nulls": (
            [
                {"id": 1, "g": "a", "x": None},
                {"id": 2, "g": None, "x": 2.0},
            ],
            [
                {"id": 2, "g": None, "x": 2.0},
                {"id": 3, "g": "b", "x": None},
            ],
        ),
        "boundary": (
            [
                {"id": 1, "g": "a", "x": -10.0},
                {"id": 2, "g": "a", "x": 0.0},
            ],
            [
                {"id": 3, "g": "b", "x": 10.0},
                {"id": 4, "g": "b", "x": 100.0},
            ],
        ),
    }[axes["data_pattern"]]
    operations: list[dict[str, Any]] = [{"op": "union_all", "table": "t1"}]
    if axes["dedupe"] == "distinct_union":
        operations.append({"op": "distinct", "columns": ["id", "g", "x"]})
    aggregates = (
        [
            {"column": "id", "func": "count", "as": "count_id"},
            {"column": "x", "func": "sum", "as": "sum_x"},
        ]
        if axes["aggregate_pair"] == "count_sum"
        else [
            {"column": "x", "func": "mean", "as": "mean_x"},
            {"column": "g", "func": "nunique", "as": "nunique_g"},
        ]
    )
    operations.append({"op": "aggregate", "aggs": aggregates})
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("x", "float", nullable=True),
    ]
    return (
        [
            TableData("t0", columns, left_rows),
            TableData("t1", columns, right_rows),
        ],
        operations,
        {},
    )


def _sqlite_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    left_rows = {
        "right_null": [
            {"id": 1, "k1": 1, "k2": "a", "x": 10},
            {"id": 2, "k1": 2, "k2": "b", "x": 20},
            {"id": 3, "k1": 3, "k2": "c", "x": 30},
            {"id": 4, "k1": 4, "k2": "d", "x": 40},
        ],
        "left_null": [
            {"id": 1, "k1": 1, "k2": "a", "x": 10},
            {"id": 2, "k1": None, "k2": "b", "x": 20},
            {"id": 3, "k1": 3, "k2": None, "x": 30},
            {"id": 4, "k1": 4, "k2": "d", "x": 40},
        ],
    }[axes["data_pattern"]]
    right_rows = {
        "right_null": [
            {"r1": 1, "r2": "a"},
            {"r1": None, "r2": "b"},
            {"r1": 3, "r2": None},
        ],
        "left_null": [
            {"r1": 1, "r2": "a"},
            {"r1": 2, "r2": "b"},
            {"r1": 4, "r2": "d"},
        ],
    }[axes["data_pattern"]]
    left_keys: str | list[str]
    right_keys: str | list[str]
    if axes["key_shape"] == "single":
        left_keys = "k1"
        right_keys = "r1"
    else:
        left_keys = ["k1", "k2"]
        right_keys = ["r1", "r2"]
    if axes["membership"] == "tuple_absence_filter":
        membership = {
            "op": "tuple_absence_filter",
            "table": "t1",
            "columns": [left_keys] if isinstance(left_keys, str) else left_keys,
            "right_columns": (
                [right_keys] if isinstance(right_keys, str) else right_keys
            ),
        }
    else:
        membership = {
            "op": axes["membership"],
            "table": "t1",
            "left_on": left_keys,
            "right_on": right_keys,
        }
    operations = [
        membership,
        {
            "op": "case_when",
            "as": "bucket",
            "condition": {"column": "x", "cmp": ">=", "value": 25},
            "then": "high",
            "else": "low",
        },
        {
            "op": "groupby",
            "keys": ["bucket"],
            "aggs": [{"column": "id", "func": "count", "as": "count_id"}],
        },
        {
            "op": "sort",
            "keys": [
                {"column": "bucket", "ascending": True, "nulls": "last"}
            ],
        },
    ]
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("k1", "int", nullable=True),
            ColumnSpec("k2", "str", nullable=True),
            ColumnSpec("x", "int", nullable=False),
        ],
        left_rows,
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("r1", "int", nullable=True),
            ColumnSpec("r2", "str", nullable=True),
        ],
        right_rows,
    )
    return [left, right], operations, {}


def _datafusion_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], list[dict[str, Any]], dict[str, Any]]:
    left_rows, right_rows = {
        "all_null_group": (
            [
                {"id": 1, "g": "a", "x": None, "y": 1.0},
                {"id": 2, "g": "a", "x": None, "y": 2.0},
            ],
            [
                {"id": 3, "g": "b", "x": 3.0, "y": None},
                {"id": 4, "g": "b", "x": None, "y": 4.0},
            ],
        ),
        "duplicate_null": (
            [
                {"id": 1, "g": "a", "x": 1.0, "y": None},
                {"id": 2, "g": "a", "x": None, "y": 2.0},
            ],
            [
                {"id": 2, "g": "a", "x": None, "y": 2.0},
                {"id": 3, "g": "b", "x": 3.0, "y": None},
            ],
        ),
    }[axes["data_pattern"]]
    operations: list[dict[str, Any]] = [{"op": "union_all", "table": "t1"}]
    aggregate_source = "x"
    if axes["null_op"] == "drop_nulls":
        operations.append({"op": "drop_nulls", "columns": ["x"]})
    elif axes["null_op"] == "fill_null":
        operations.append({"op": "fill_null", "column": "x", "value": 0.0})
    else:
        operations.append(
            {
                "op": "coalesce",
                "columns": ["x", "y"],
                "as": "value",
                "fallback": 0.0,
            }
        )
        aggregate_source = "value"
    aggregates = (
        [
            {
                "column": aggregate_source,
                "func": "sum",
                "as": "sum_value",
            },
            {
                "column": aggregate_source,
                "func": "min",
                "as": "min_value",
            },
        ]
        if axes["aggregate_pair"] == "sum_min"
        else [
            {
                "column": aggregate_source,
                "func": "mean",
                "as": "mean_value",
            },
            {"column": "id", "func": "count", "as": "count_id"},
        ]
    )
    operations.extend(
        [
            {"op": "groupby", "keys": ["g"], "aggs": aggregates},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"}
                ],
            },
        ]
    )
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=False),
        ColumnSpec("x", "float", nullable=True),
        ColumnSpec("y", "float", nullable=True),
    ]
    return (
        [
            TableData("t0", columns, left_rows),
            TableData("t1", columns, right_rows),
        ],
        operations,
        {},
    )


def _pandas_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = [op_kind(operation) for operation in case.program.operations]
    expressions = expr_kinds(case.program.operations)
    expected_transform = (
        {"string_strip", "string_lower"}
        if axes["transform"] == "lower_strip"
        else {"string_replace", "string_upper"}
    )
    expected_null_op = "coalesce" if axes["null_policy"] == "coalesce" else "fill_null"
    return bool(
        kinds[-2:] == ["groupby", "sort"]
        and expected_null_op in kinds
        and "string_null_if_empty" in expressions
        and expected_transform <= expressions
        and _aggregate_set(case) == {"count", "nunique"}
        and len(case.tables) == 1
    )


def _pyarrow_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = [op_kind(operation) for operation in case.program.operations]
    expressions = expr_kinds(case.program.operations)
    expected_predicate = {
        "contains": "string_contains",
        "starts_with": "string_starts_with",
        "ends_with": "string_ends_with",
    }[axes["predicate"]]
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    layouts = metadata.get("input_layouts", {})
    table_layout = layouts.get("t0", {}) if isinstance(layouts, Mapping) else {}
    return bool(
        kinds == ["mutate", "mutate", "case_when", "groupby", "sort"]
        and expressions == {"string_strip", expected_predicate}
        and _aggregate_set(case) == {"count", "nunique"}
        and isinstance(table_layout, Mapping)
        and str(table_layout.get("representation", "")) == axes["layout"]
        and (
            int(table_layout.get("chunk_count", 0) or 0) == 2
            if axes["layout"] == "chunked"
            else int(table_layout.get("chunk_count", 0) or 0) == 1
        )
    )


def _polars_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = [op_kind(operation) for operation in case.program.operations]
    expressions = expr_kinds(case.program.operations)
    expected = (
        {"add_const", "clip"}
        if axes["arithmetic"] == "add_clip"
        else {"abs", "cast"}
    )
    sort = case.program.operations[-2]
    check = case.program.operations[-1]
    sort_keys = list(sort.get("keys", []) or [])
    ascending = axes["order_mode"].startswith("asc")
    nulls = "last" if axes["order_mode"].endswith("last") else "first"
    return bool(
        kinds == ["mutate", "mutate", "sort", "sortedness_check"]
        and expressions == expected
        and [item.get("column") for item in sort_keys] == ["x_out", "id"]
        and bool(check.get("ascending")) == ascending
        and str(check.get("nulls", "")) == nulls
    )


def _polars_lazy_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = [op_kind(operation) for operation in case.program.operations]
    expressions = expr_kinds(case.program.operations)
    expected = (
        {"string_strip", "string_lower", "string_concat"}
        if axes["string_expr"] == "lower_concat"
        else {"string_strip", "string_slice", "string_replace"}
    )
    sort = case.program.operations[-2]
    return bool(
        len(case.tables) == 2
        and kinds[-4:] == ["case_when", axes["membership"], "sort", "select"]
        and expressions == expected
        and [item.get("column") for item in sort.get("keys", [])]
        == ["bucket", "id"]
    )


def _duckdb_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = [op_kind(operation) for operation in case.program.operations]
    expected_aggs = (
        {"count", "sum"}
        if axes["aggregate_pair"] == "count_sum"
        else {"mean", "nunique"}
    )
    distinct_expected = axes["dedupe"] == "distinct_union"
    return bool(
        len(case.tables) == 2
        and kinds[0] == "union_all"
        and kinds[-1] == "aggregate"
        and ("distinct" in kinds) == distinct_expected
        and _aggregate_set(case) == expected_aggs
    )


def _sqlite_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    operations = list(case.program.operations)
    kinds = [op_kind(operation) for operation in operations]
    membership = operations[0]
    key_count = 1 if axes["key_shape"] == "single" else 2
    if axes["membership"] == "tuple_absence_filter":
        observed_key_count = len(list(membership.get("columns", []) or []))
    else:
        left_on = membership.get("left_on")
        observed_key_count = 1 if isinstance(left_on, str) else len(left_on or [])
    return bool(
        len(case.tables) == 2
        and kinds == [axes["membership"], "case_when", "groupby", "sort"]
        and observed_key_count == key_count
        and _aggregate_set(case) == {"count"}
        and _case_contains_null(case)
    )


def _datafusion_preconditions(axes: Mapping[str, str], case: Case) -> bool:
    kinds = [op_kind(operation) for operation in case.program.operations]
    expected_aggs = (
        {"sum", "min"}
        if axes["aggregate_pair"] == "sum_min"
        else {"mean", "count"}
    )
    return bool(
        len(case.tables) == 2
        and kinds == ["union_all", axes["null_op"], "groupby", "sort"]
        and _aggregate_set(case) == expected_aggs
        and _case_contains_null(case)
        and [
            item.get("column")
            for item in case.program.operations[-1].get("keys", [])
        ]
        == ["g"]
    )


def _aggregate_set(case: Case) -> set[str]:
    return {
        function
        for operation in case.program.operations
        for function in aggregate_functions(operation)
    }


def _case_contains_null(case: Case) -> bool:
    return any(
        value is None
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )
