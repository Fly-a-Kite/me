"""Bounded in-memory witnesses for every confirmed root not covered by v4-v6.

The builders in this module deliberately do not read the canonical corpus.
They encode small mechanism-preserving palettes and emit one Boolean native
probe.  The target adapter evaluates the native contract; every comparison
adapter returns ``False`` as a control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.family_witness_registry import family_witness_registration


CONFIRMED_ROOT_WITNESS_SCHEMA_VERSION = "confirmed-root-witness-v1"
CONFIRMED_ROOT_PROBE_KIND = "confirmed_root_witness_probe"

DATAFUSION_LIMIT_OFFSET_FAMILY_ID = "datafusion_limit_offset_pushdown"
DATAFUSION_NEGATIVE_ZERO_FAMILY_ID = "datafusion_negative_zero_comparison"
DATAFUSION_DISTINCT_NULL_TOPK_FAMILY_ID = "datafusion_distinct_null_topk"
DATAFUSION_LIMIT_IDEMPOTENCE_FAMILY_ID = "datafusion_ordered_limit_idempotence"
POLARS_GROUPED_MAX_METADATA_FAMILY_ID = "polars_grouped_max_sort_metadata"
DUCKDB_JOIN_FILTER_FAMILY_ID = "duckdb_join_filter_pushdown_limit"


@dataclass(frozen=True, slots=True)
class ConfirmedRootWitnessGenerationResult:
    case: Case
    trace: dict[str, Any]


_FAMILY_SPECS: dict[str, dict[str, str]] = {
    DATAFUSION_LIMIT_OFFSET_FAMILY_ID: {
        "root_id": "datafusion-limit-offset-pushdown-001",
        "target_backend": "datafusion",
        "root_cause": "joined_order_offset_projection",
        "fault_model": "handled_outer_offset_leaks_into_inner_limit",
        "target_contract": "inner_ordered_limit_does_not_inherit_outer_offset",
    },
    DATAFUSION_NEGATIVE_ZERO_FAMILY_ID: {
        "root_id": "datafusion-negative-zero-comparison-001",
        "target_backend": "datafusion",
        "root_cause": "negative_zero_comparison",
        "fault_model": "total_order_used_at_sql_signed_zero_boundary",
        "target_contract": "negative_zero_compares_equal_to_positive_zero",
    },
    DATAFUSION_DISTINCT_NULL_TOPK_FAMILY_ID: {
        "root_id": "datafusion-distinct-null-topk-001",
        "target_backend": "datafusion",
        "root_cause": "distinct_null_topk",
        "fault_model": "distinct_topk_drops_null_heap_entry",
        "target_contract": "limited_distinct_order_matches_full_order_prefix",
    },
    DATAFUSION_LIMIT_IDEMPOTENCE_FAMILY_ID: {
        "root_id": "datafusion-ordered-limit-idempotence-001",
        "target_backend": "datafusion",
        "root_cause": "datafusion_limit_idempotence",
        "fault_model": "duplicate_limit_merges_with_stale_offset",
        "target_contract": "repeated_identical_ordered_limit_is_idempotent",
    },
    POLARS_GROUPED_MAX_METADATA_FAMILY_ID: {
        "root_id": "polars-grouped-max-sort-metadata-001",
        "target_backend": "polars",
        "root_cause": "groupby_aggregation",
        "fault_model": "stale_sortedness_metadata_after_grouped_max",
        "target_contract": "grouped_max_output_does_not_reuse_input_sort_flags",
    },
    DUCKDB_JOIN_FILTER_FAMILY_ID: {
        "root_id": "duckdb-join-filter-pushdown-limit-001",
        "target_backend": "duckdb",
        "root_cause": "topk_filter_pushdown",
        "fault_model": "runtime_join_filter_crosses_ordered_offset_boundary",
        "target_contract": "membership_filter_does_not_cross_ordered_cut",
    },
}


def generate_datafusion_limit_offset_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ConfirmedRootWitnessGenerationResult:
    return _generate_case(DATAFUSION_LIMIT_OFFSET_FAMILY_ID, seed, profile=profile)


def generate_datafusion_negative_zero_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ConfirmedRootWitnessGenerationResult:
    return _generate_case(DATAFUSION_NEGATIVE_ZERO_FAMILY_ID, seed, profile=profile)


def generate_datafusion_distinct_null_topk_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ConfirmedRootWitnessGenerationResult:
    return _generate_case(DATAFUSION_DISTINCT_NULL_TOPK_FAMILY_ID, seed, profile=profile)


def generate_datafusion_limit_idempotence_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ConfirmedRootWitnessGenerationResult:
    return _generate_case(DATAFUSION_LIMIT_IDEMPOTENCE_FAMILY_ID, seed, profile=profile)


def generate_polars_grouped_max_metadata_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ConfirmedRootWitnessGenerationResult:
    return _generate_case(POLARS_GROUPED_MAX_METADATA_FAMILY_ID, seed, profile=profile)


def generate_duckdb_join_filter_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ConfirmedRootWitnessGenerationResult:
    return _generate_case(DUCKDB_JOIN_FILTER_FAMILY_ID, seed, profile=profile)


def confirmed_root_witness_static_preconditions(
    root_id: str,
    axes: Mapping[str, Any],
    probe_spec: Mapping[str, Any],
) -> bool:
    """Return whether a witness uses one complete registered palette cell."""

    family_id = next(
        (
            family
            for family, spec in _FAMILY_SPECS.items()
            if spec["root_id"] == str(root_id)
        ),
        "",
    )
    if not family_id:
        return False
    registration = family_witness_registration(family_id)
    normalized_axes = {str(key): str(value) for key, value in axes.items()}
    if set(normalized_axes) != set(registration.axis_names):
        return False
    if any(
        normalized_axes[name] not in values
        for name, values in registration.axes
    ):
        return False
    spec = _FAMILY_SPECS[family_id]
    return bool(
        str(probe_spec.get("root_id", "")) == spec["root_id"]
        and str(probe_spec.get("target_backend", "")) == spec["target_backend"]
        and str(probe_spec.get("root_cause", "")) == spec["root_cause"]
    )


def _generate_case(
    family_id: str,
    seed: int,
    *,
    profile: str,
) -> ConfirmedRootWitnessGenerationResult:
    registration = family_witness_registration(family_id)
    cell_index, axes = registration.cell_for_seed(seed)
    spec = _FAMILY_SPECS[family_id]
    tables, native_parameters = _build_payload(family_id, axes)
    probe_spec = {
        **spec,
        "family_id": family_id,
        "axes": dict(axes),
        "native_parameters": native_parameters,
    }
    operation = {
        "op": CONFIRMED_ROOT_PROBE_KIND,
        "as": "confirmed_root_mismatch",
        **probe_spec,
    }
    program = Program(
        f"prog-{int(seed):08d}-{family_id}-{cell_index:03d}",
        int(seed),
        [operation],
    )
    family_witness = {
        "schema_version": "family-witness-case-v1",
        "family_id": family_id,
        "generation_mode": registration.generation_mode,
        "goal_id": registration.goal_id,
        "root_ids": [registration.root_id],
        "cell_index": cell_index,
        "axes": dict(axes),
        "palette_dimensions": [axes[name] for name in registration.axis_names],
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    witness = {
        "schema_version": CONFIRMED_ROOT_WITNESS_SCHEMA_VERSION,
        "family_id": family_id,
        "goal_id": registration.goal_id,
        "root_ids": [registration.root_id],
        "target_backend": spec["target_backend"],
        "root_cause": spec["root_cause"],
        "fault_model": spec["fault_model"],
        "cell_index": cell_index,
        "axes": dict(axes),
        "native_parameters": native_parameters,
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "bounded_for_cache_reuse": True,
    }
    trace = {
        "schema_version": CONFIRMED_ROOT_WITNESS_SCHEMA_VERSION,
        "generation_mode": registration.generation_mode,
        "selected_goal": {
            "goal_id": registration.goal_id,
            "target_contract": spec["target_contract"],
            "fault_models": [spec["fault_model"]],
        },
        "selection_reason": "registered_root_axis_product_cycle_v1",
        "builder_variant": {
            "variant_id": ":".join(axes[name] for name in registration.axis_names),
            "variant_family": "global_confirmed_roots_v1",
            "cell_index": cell_index,
            "construction_seed": cell_index,
            "case_seed": int(seed),
            "root_guidance": [registration.root_id],
            "axes": dict(axes),
        },
        "requested_profile": str(profile or ""),
        "activation_requirement": {
            "required": True,
            "policy": "deterministic_registered_confirmed_root_witness",
        },
        "confirmed_root_witness": witness,
        "family_witness": family_witness,
        "valid": True,
        "constructible": True,
        "skip_reason": "",
    }
    case = Case(
        case_id=f"case-{int(seed):08d}-{family_id}-{cell_index:03d}",
        seed=int(seed),
        tables=tables,
        program=program,
        metadata={
            "generator_profile": f"{family_id}_witness",
            "generation_mode": registration.generation_mode,
            "goal_id": registration.goal_id,
            "goal_fault_models": [spec["fault_model"]],
            "semantic_activation_syntactic_reached": True,
            "goal_first_generation": trace,
            "goal_builder_variant": dict(trace["builder_variant"]),
            "semantic_witness_builder_variant": trace["builder_variant"][
                "variant_id"
            ],
            "semantic_witness_data_pattern": {
                "pattern_id": axes.get("data_pattern", ""),
            },
            "confirmed_root_witness": witness,
            "family_witness": family_witness,
        },
    )
    from datadiff.semantic_core.activation import evaluate_semantic_activation

    activation = evaluate_semantic_activation(
        case,
        goal_id=registration.goal_id,
        syntactic_reached=True,
    ).to_dict()
    case.metadata["semantic_activation"] = activation
    trace["semantic_activation"] = activation
    return ConfirmedRootWitnessGenerationResult(case=case, trace=trace)


def _build_payload(
    family_id: str,
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    if family_id == DATAFUSION_LIMIT_OFFSET_FAMILY_ID:
        return _datafusion_limit_offset_payload(axes)
    if family_id == DATAFUSION_NEGATIVE_ZERO_FAMILY_ID:
        return _datafusion_negative_zero_payload(axes)
    if family_id == DATAFUSION_DISTINCT_NULL_TOPK_FAMILY_ID:
        return _datafusion_distinct_null_payload(axes)
    if family_id == DATAFUSION_LIMIT_IDEMPOTENCE_FAMILY_ID:
        return _datafusion_limit_idempotence_payload(axes)
    if family_id == POLARS_GROUPED_MAX_METADATA_FAMILY_ID:
        return _polars_grouped_max_payload(axes)
    if family_id == DUCKDB_JOIN_FILTER_FAMILY_ID:
        return _duckdb_join_filter_payload(axes)
    raise KeyError(family_id)


def _datafusion_limit_offset_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    pattern = axes["data_pattern"]
    left_rows = [{"id": value} for value in range(4)]
    right_rows = {
        "four_groups": [{"id": 1, "j": 1}, {"id": 3, "j": 3}],
        "duplicate_join_match": [
            {"id": 1, "j": 1},
            {"id": 1, "j": 2},
            {"id": 3, "j": 3},
        ],
        "nullable_join_payload": [
            {"id": 1, "j": None},
            {"id": 2, "j": 2},
            {"id": 3, "j": 3},
        ],
    }[pattern]
    return (
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False)],
                left_rows,
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int", nullable=True),
                ],
                right_rows,
            ),
        ],
        {
            "inner_limit": int(axes["inner_limit"].removeprefix("limit")),
            "outer_offset": int(axes["outer_offset"].removeprefix("offset")),
            "data_pattern": pattern,
        },
    )


def _datafusion_negative_zero_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    rows = {
        "singleton_zero": [{"id": 1, "y": 0.0}],
        "duplicate_zero": [{"id": 1, "y": 0.0}, {"id": 2, "y": 0.0}],
        "zero_with_positive_control": [
            {"id": 1, "y": 0.0},
            {"id": 2, "y": 1.0},
        ],
    }[axes["data_pattern"]]
    return (
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("y", "float", nullable=False),
                ],
                rows,
            )
        ],
        dict(axes),
    )


def _datafusion_distinct_null_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    values = {
        "null_empty_text": [None, "", "a"],
        "duplicate_null": [None, None, "a"],
        "mixed_text": [None, "b", "a", ""],
    }[axes["data_pattern"]]
    return (
        [
            TableData(
                "t0",
                [ColumnSpec("v", "str", nullable=True)],
                [{"v": value} for value in values],
            )
        ],
        dict(axes),
    )


_LIMIT_IDEMPOTENCE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": 0, "g": "a", "x": None, "z": 8, "s": "A"},
    {"id": 1, "g": "b", "x": 10, "z": None, "s": "space value"},
    {"id": 2, "g": "c", "x": -10, "z": -3, "s": ""},
    {"id": 3, "g": "b", "x": -2, "z": 0, "s": "space value"},
    {"id": 4, "g": "b", "x": 10, "z": None, "s": "a"},
    {"id": 5, "g": "a", "x": None, "z": 0, "s": ""},
    {"id": 6, "g": None, "x": None, "z": 0, "s": "space value"},
    {"id": 7, "g": "a", "x": 0, "z": 1, "s": "a"},
    {"id": 8, "g": "c", "x": 0, "z": 0, "s": "A"},
    {"id": 9, "g": "c", "x": -10, "z": 8, "s": "A"},
    {"id": 10, "g": "c", "x": -10, "z": -3, "s": "space value"},
    {"id": 11, "g": "b", "x": -10, "z": 8, "s": "A"},
    {"id": 12, "g": "a", "x": -2, "z": -3, "s": ""},
    {"id": 13, "g": "a", "x": 2, "z": 1, "s": "space value"},
)


def _datafusion_limit_idempotence_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    pattern = axes["data_pattern"]
    rows = [dict(row) for row in _LIMIT_IDEMPOTENCE_ROWS]
    if pattern == "reversed_input":
        rows.reverse()
    elif pattern == "duplicate_boundary":
        rows.append({"id": 14, "g": "c", "x": -10, "z": -3, "s": ""})
    return (
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                    ColumnSpec("z", "int", nullable=True),
                    ColumnSpec("s", "str", nullable=True),
                ],
                rows,
            )
        ],
        {
            "limit_n": int(axes["limit_n"].removeprefix("limit")),
            "offset_n": int(axes["offset_n"].removeprefix("offset")),
            "data_pattern": pattern,
        },
    )


def _polars_grouped_max_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    group_shape = axes["group_shape"]
    pattern = axes["data_pattern"]
    if group_shape == "two_groups":
        groups = ["a", "b", "b", "b", "a"]
    else:
        groups = ["a", "b", "c", "b", "c", "a"]
    values = {
        "canonical_counts": [1, 2, 0, None, 3, 1],
        "duplicate_max": [2, 2, 0, None, 2, 1],
        "null_heavy": [1, None, 0, None, 3, None],
    }[pattern][: len(groups)]
    rows = [{"s": group, "z": value} for group, value in zip(groups, values, strict=True)]
    return (
        [
            TableData(
                "t0",
                [
                    ColumnSpec("s", "str", nullable=False),
                    ColumnSpec("z", "int", nullable=True),
                ],
                rows,
            )
        ],
        {**dict(axes), "rows": rows},
    )


def _duckdb_join_filter_payload(
    axes: Mapping[str, str],
) -> tuple[list[TableData], dict[str, Any]]:
    rows: list[tuple[int, bool | None]] = [
        (1, False),
        (1, True),
        (1, False),
        (0, True),
        (1, True),
        (0, None),
        (0, True),
        (2, False),
        (1, False),
        (0, True),
        (1, True),
        (24, False),
        (1, True),
        (2, False),
    ]
    if axes["data_pattern"] == "duplicate_low_groups":
        rows.extend(((0, True), (0, None)))
    elif axes["data_pattern"] == "duplicate_selected_group":
        rows.extend(((24, False), (24, False)))
    mirrored = axes["flag_mode"] == "mirrored"
    if mirrored:
        rows = [(key, None if flag is None else not flag) for key, flag in rows]
    selected_flag = axes["cut_mode"] == "offset1_desc"
    if mirrored:
        selected_flag = not selected_flag
    return (
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("flag", "bool", nullable=True),
                ],
                [{"id": key, "flag": flag} for key, flag in rows],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("z", "float", nullable=False),
                    ColumnSpec("tag", "str", nullable=False),
                ],
                [
                    {"id": 0, "z": 1.0, "tag": "BkcDeJw"},
                    {"id": 24, "z": 24.0, "tag": "tag_24"},
                ],
            ),
            TableData(
                "keys",
                [ColumnSpec("flag_key", "bool", nullable=False)],
                [{"flag_key": selected_flag}],
            ),
        ],
        {**dict(axes), "selected_flag": selected_flag, "threads": 1},
    )

