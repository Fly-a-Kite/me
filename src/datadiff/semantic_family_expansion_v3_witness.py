"""Issue-risk-class witnesses for semantic-family universe v3.

The builders deliberately abstract operation mechanisms from upstream issues.
They use only in-memory generated tables, never replay an upstream reproducer,
and keep the family palette small enough for low-I/O paired screening.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.boundary_values import apply_targeted_boundary_profile
from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.family_witness_registry import family_witness_registration
from datadiff.operation_semantics import aggregate_functions, expr_kinds, op_kind
from datadiff.semantic_family_universe_v3 import (
    CROSS_BACKEND_RISK_FAMILY_ID,
    EXACT_DTYPE_FAMILY_ID,
    PYARROW_LAYOUT_RISK_FAMILY_ID,
    issue_risk_boundary_profile_ids,
    pipeline_evidence_spec,
    semantic_family_v3_definition,
    semantic_family_v3_witness_spec,
    semantic_family_v3_witness_specs,
)


SEMANTIC_FAMILY_EXPANSION_V3_SCHEMA_VERSION = (
    "semantic-family-expansion-witness-v3"
)

EXPANSION_V3_FAMILY_IDS = frozenset(
    item.family_id for item in semantic_family_v3_witness_specs()
)


@dataclass(frozen=True, slots=True)
class SemanticFamilyExpansionV3GenerationResult:
    case: Case
    trace: dict[str, Any]


def generate_cross_backend_issue_risk_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV3GenerationResult:
    return _generate_case(CROSS_BACKEND_RISK_FAMILY_ID, seed, profile=profile)


def generate_cross_backend_exact_dtype_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV3GenerationResult:
    return _generate_case(EXACT_DTYPE_FAMILY_ID, seed, profile=profile)


def generate_pyarrow_layout_interaction_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> SemanticFamilyExpansionV3GenerationResult:
    return _generate_case(PYARROW_LAYOUT_RISK_FAMILY_ID, seed, profile=profile)


def semantic_family_expansion_v3_static_preconditions(
    family_id: str,
    axes: Mapping[str, Any],
    case: Case,
) -> bool:
    if family_id not in EXPANSION_V3_FAMILY_IDS:
        return False
    try:
        registration = family_witness_registration(family_id)
        definition = semantic_family_v3_definition(family_id)
        family_spec = semantic_family_v3_witness_spec(family_id)
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
        != SEMANTIC_FAMILY_EXPANSION_V3_SCHEMA_VERSION
        or str(witness.get("family_id", "")) != family_id
        or str(witness.get("target_backend", "")) != definition.target_backend
        or list(witness.get("root_ids", []) or []) != [definition.root_id]
        or bool(witness.get("canonical_case_replay", True))
        or bool(witness.get("runtime_corpus_io", True))
        or family_witness.get("axes") != normalized_axes
        or tuple(witness.get("operation_kinds", ()) or ()) != kinds
        or str(witness.get("source_profile", "")) != normalized_axes["pipeline"]
        or any(key in metadata for key in ("source_issue", "source_issue_alt"))
    ):
        return False
    try:
        evidence = pipeline_evidence_spec(normalized_axes["pipeline"])
    except KeyError:
        return False
    operations = set(kinds)
    expressions = expr_kinds(case.program.operations)
    aggregates = {
        function
        for operation in case.program.operations
        for function in aggregate_functions(operation)
    }
    if (
        tuple(registration.axes) != family_spec.axes
        or tuple(registration.backends) != family_spec.execution_backends
        or not set(evidence.required_operations) <= operations
        or not set(evidence.required_expressions) <= expressions
        or not set(evidence.required_aggregates) <= aggregates
        or (
            evidence.aggregate_any_of
            and not set(evidence.aggregate_any_of).intersection(aggregates)
        )
        or any(
            not _is_subsequence(chain, kinds)
            for chain in evidence.required_operation_chains
        )
        or any(kind.endswith("_probe") for kind in operations)
    ):
        return False
    if normalized_axes["data_pattern"] in family_spec.boundary_data_patterns:
        boundary = metadata.get("boundary_application", {})
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("applied") is not True
            or str(boundary.get("profile_id", ""))
            not in issue_risk_boundary_profile_ids(evidence.risk_class)
        ):
            return False
    if family_spec.comparison_view:
        comparison = metadata.get("semantic_comparison", {})
        if (
            not isinstance(comparison, Mapping)
            or comparison.get("view") != family_spec.comparison_view
        ):
            return False
    if family_spec.layouts:
        layout_map = metadata.get("input_layouts", {})
        if not isinstance(layout_map, Mapping) or len(layout_map) != len(case.tables):
            return False
        expected_layout = normalized_axes["layout"]
        if any(
            not isinstance(raw, Mapping)
            or str(raw.get("representation", "")) != expected_layout
            for raw in layout_map.values()
        ):
            return False
    return bool(case.tables and case.program.operations)


def _generate_case(
    family_id: str,
    seed: int,
    *,
    profile: str,
) -> SemanticFamilyExpansionV3GenerationResult:
    registration = family_witness_registration(family_id)
    definition = semantic_family_v3_definition(family_id)
    family_spec = semantic_family_v3_witness_spec(family_id)
    cell_index, axes = registration.cell_for_seed(seed)
    construction_registration = registration
    construction_spec = family_spec
    if registration.cache_reuse_source_family_id:
        construction_registration = family_witness_registration(
            registration.cache_reuse_source_family_id
        )
        construction_spec = semantic_family_v3_witness_spec(
            construction_registration.family_id
        )
    construction_axes = {
        name: axes[name]
        for name in construction_registration.axis_names
    }
    construction_cell_index = construction_registration.cell_index_for_axes(
        construction_axes
    )
    source_seed = 94_000_000 + construction_cell_index
    case = _source_case(axes["pipeline"], source_seed)
    evidence = pipeline_evidence_spec(axes["pipeline"])
    if axes["data_pattern"] in construction_spec.boundary_data_patterns:
        case = apply_targeted_boundary_profile(
            case,
            seed=source_seed,
            preferred_profile_ids=issue_risk_boundary_profile_ids(
                evidence.risk_class
            ),
        ).case
    case = _apply_data_pattern(case, axes["data_pattern"])
    if family_spec.comparison_view:
        case.metadata["semantic_comparison"] = {
            "view": family_spec.comparison_view
        }
    if family_spec.layouts:
        case.metadata["input_layouts"] = _input_layouts(case, axes["layout"])

    for key in ("source_issue", "source_issue_alt", "issue_inspiration"):
        case.metadata.pop(key, None)
    case.case_id = f"case-{int(seed):08d}-{family_id}-{cell_index:03d}"
    case.seed = int(seed)
    case.program.program_id = f"prog-{int(seed):08d}-{family_id}-{cell_index:03d}"
    case.program.seed = int(seed)
    operation_kinds = [op_kind(operation) for operation in case.program.operations]
    family_witness = {
        "schema_version": "family-witness-case-v1",
        "family_id": family_id,
        "generation_mode": registration.generation_mode,
        "goal_id": registration.goal_id,
        "root_ids": [definition.root_id],
        "cell_index": cell_index,
        "axes": dict(axes),
        "palette_dimensions": [axes[name] for name in registration.axis_names],
        "cache_reuse_source_family_id": registration.cache_reuse_source_family_id,
        "cache_reuse_axes": list(registration.cache_reuse_axes),
        "cache_reuse_source_cell_index": (
            construction_cell_index
            if registration.cache_reuse_source_family_id
            else None
        ),
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    witness = {
        "schema_version": SEMANTIC_FAMILY_EXPANSION_V3_SCHEMA_VERSION,
        "family_id": family_id,
        "mechanism_id": definition.mechanism_id,
        "goal_id": registration.goal_id,
        "root_ids": [definition.root_id],
        "target_backend": definition.target_backend,
        "control_backends": list(definition.control_backends),
        "cell_index": cell_index,
        "axes": dict(axes),
        "source_profile": axes["pipeline"],
        "risk_class": evidence.risk_class,
        "boundary_profile": str(
            (case.metadata.get("boundary_application", {}) or {}).get(
                "profile_id",
                "",
            )
        ),
        "execution_backends": list(family_spec.execution_backends),
        "operation_kinds": operation_kinds,
        "expression_kinds": sorted(expr_kinds(case.program.operations)),
        "aggregate_functions": sorted(
            {
                function
                for operation in case.program.operations
                for function in aggregate_functions(operation)
            }
        ),
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "issue_risk_abstraction": True,
        "required_operation_chains": [
            list(chain) for chain in evidence.required_operation_chains
        ],
        "bounded_for_cache_reuse": True,
        "cache_reuse_source_family_id": registration.cache_reuse_source_family_id,
        "cache_reuse_axes": list(registration.cache_reuse_axes),
        "cache_reuse_source_cell_index": (
            construction_cell_index
            if registration.cache_reuse_source_family_id
            else None
        ),
    }
    trace = {
        "schema_version": SEMANTIC_FAMILY_EXPANSION_V3_SCHEMA_VERSION,
        "generation_mode": registration.generation_mode,
        "selected_goal": {
            "goal_id": registration.goal_id,
            "mechanism_id": definition.mechanism_id,
            "target_backend": definition.target_backend,
        },
        "selection_reason": "issue_risk_class_axis_product_cycle",
        "builder_variant": {
            "variant_id": ":".join(axes[name] for name in registration.axis_names),
            "variant_family": "semantic_family_universe_expansion_v3",
            "cell_index": cell_index,
            "construction_seed": source_seed,
            "construction_family_id": construction_registration.family_id,
            "construction_cell_index": construction_cell_index,
            "case_seed": int(seed),
            "root_guidance": [definition.root_id],
            "axes": dict(axes),
        },
        "requested_profile": str(profile or ""),
        "activation_requirement": {
            "required": True,
            "policy": "deterministic_issue_risk_family_witness",
        },
        "semantic_family_expansion_witness": witness,
        "family_witness": family_witness,
        "valid": True,
        "constructible": True,
        "skip_reason": "",
    }
    case.metadata.update(
        {
            "generator_profile": f"{family_id}_witness",
            "source_generator_profile": axes["pipeline"],
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
                "pattern_id": axes["data_pattern"]
            },
            "semantic_family_expansion_witness": witness,
            "family_witness": family_witness,
            "canonical_case_replay": False,
            "runtime_corpus_io": False,
        }
    )
    from datadiff.semantic_core.activation import evaluate_semantic_activation

    activation = evaluate_semantic_activation(
        case,
        goal_id=registration.goal_id,
        syntactic_reached=True,
    ).to_dict()
    case.metadata["semantic_activation"] = activation
    trace["semantic_activation"] = activation
    return SemanticFamilyExpansionV3GenerationResult(case=case, trace=trace)


def _is_subsequence(
    expected: tuple[str, ...],
    observed: tuple[str, ...],
) -> bool:
    if not expected:
        return True
    index = 0
    for item in observed:
        if item == expected[index]:
            index += 1
            if index == len(expected):
                return True
    return False


def _source_case(pipeline: str, seed: int) -> Case:
    if pipeline == "aggregate_mean_matrix":
        return _aggregate_mean_case(seed)
    if pipeline == "scalar_abs_upper_filter":
        return _scalar_abs_upper_case(seed)
    return generate_case(seed, profile=pipeline)


def _apply_data_pattern(case: Case, pattern: str) -> Case:
    result = copy.deepcopy(case)
    if pattern == "baseline":
        return result
    if pattern != "reordered_duplicate":
        raise KeyError(pattern)
    for table in result.tables:
        table.rows = list(reversed(table.rows))
    primary = result.tables[0]
    if primary.rows:
        duplicate = dict(primary.rows[0])
        identifier = _identifier_column(primary)
        if identifier:
            values = [row.get(identifier) for row in primary.rows]
            integers = [value for value in values if isinstance(value, int)]
            duplicate[identifier] = (max(integers) + 1) if integers else len(values)
        primary.rows.append(duplicate)
    result.metadata["input_reordered"] = True
    result.metadata["boundary_duplicate_added"] = True
    return result


def _identifier_column(table: TableData) -> str:
    columns = {column.name: column for column in table.columns}
    for name in ("row_id", "id", "seq"):
        column = columns.get(name)
        if column is not None and column.type == "int" and not column.nullable:
            return name
    for column in table.columns:
        if column.type == "int" and not column.nullable:
            return column.name
    return ""


def _input_layouts(case: Case, layout: str) -> dict[str, dict[str, Any]]:
    return {
        table.name: {
            "representation": layout,
            "chunk_count": 2 if layout == "chunked" else 1,
            "dictionary_columns": [
                column.name for column in table.columns if column.type == "str"
            ]
            if layout == "dictionary"
            else [],
        }
        for table in case.tables
    }


def _aggregate_mean_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "float", nullable=True),
        ],
        [
            {"id": 1, "g": "a", "x": 1.0},
            {"id": 2, "g": "a", "x": None},
            {"id": 3, "g": "b", "x": -2.5},
            {"id": 4, "g": "b", "x": 4.5},
            {"id": 5, "g": None, "x": 0.0},
        ],
    )
    operations = [
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [
                {"column": "x", "func": "mean", "as": "mean_x"},
                {"column": "x", "func": "min", "as": "min_x"},
                {"column": "id", "func": "count", "as": "count_id"},
            ],
        },
        {
            "op": "sort",
            "keys": [{"column": "g", "ascending": True, "nulls": "last"}],
        },
    ]
    return Case(
        case_id=f"case-{seed:08d}-aggregate-mean-matrix",
        seed=seed,
        tables=[table],
        program=Program(f"prog-{seed:08d}-aggregate-mean-matrix", seed, operations),
        metadata={"generator_profile": "aggregate_mean_matrix"},
    )


def _scalar_abs_upper_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("x", "float", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        [
            {"id": 1, "x": -4.0, "s": "alpha"},
            {"id": 2, "x": -0.0, "s": "Beta"},
            {"id": 3, "x": None, "s": None},
            {"id": 4, "x": 2.5, "s": " café "},
        ],
    )
    operations = [
        {"op": "fill_null", "column": "x", "value": 0.0},
        {
            "op": "mutate",
            "column": "abs_x",
            "expr": {"kind": "abs", "source": "x"},
        },
        {
            "op": "mutate",
            "column": "x_plus",
            "expr": {"kind": "add_const", "source": "abs_x", "value": 1.0},
        },
        {
            "op": "mutate",
            "column": "s_upper",
            "expr": {"kind": "string_upper", "source": "s"},
        },
        {"op": "filter", "column": "x_plus", "cmp": ">", "value": 1.0},
        {
            "op": "sort",
            "keys": [
                {"column": "x_plus", "ascending": True, "nulls": "last"},
                {"column": "id", "ascending": True, "nulls": "last"},
            ],
        },
        {"op": "select", "columns": ["id", "x_plus", "s_upper"]},
    ]
    return Case(
        case_id=f"case-{seed:08d}-scalar-abs-upper-filter",
        seed=seed,
        tables=[table],
        program=Program(f"prog-{seed:08d}-scalar-abs-upper-filter", seed, operations),
        metadata={"generator_profile": "scalar_abs_upper_filter"},
    )
