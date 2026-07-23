"""Deterministic, diversity-preserving builders for semantic witness v2.

Each variant keeps the static activation contract of its semantic goal while
changing a real program path and a constrained data shape.  The variants are
backend-outcome independent and are selected only by the seed and goal id.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.profile_generators import (
    generate_empty_then_union_groupby_case,
    generate_empty_union_window_aggregate_case,
    generate_join_ordered_agg_topk_case,
    generate_large_int_text_membership_window_case,
    generate_multi_key_anti_join_null_guard_case,
    generate_nested_topk_offset_aggregate_case,
    generate_null_groupby_topk_case,
    generate_topk_resort_case,
    generate_union_distinct_anti_running_sum_case,
)


SEMANTIC_WITNESS_VARIANT_SCHEMA_VERSION = "semantic-witness-builder-variants-v2"
ROOT_GUIDED_WITNESS_VARIANT_SCHEMA_VERSION = "semantic-witness-builder-variants-v3"
SEMANTIC_WITNESS_DATA_PATTERN_COUNT = 3
GoalBuilder = Callable[[int], Case]


@dataclass(frozen=True, slots=True)
class WitnessBuilderVariant:
    variant_id: str
    builder: GoalBuilder
    program_strategy: str
    data_strategy: str
    diversity_axes: tuple[str, ...]
    schema_version: str = SEMANTIC_WITNESS_VARIANT_SCHEMA_VERSION
    root_guidance: tuple[str, ...] = ()
    interaction_family: str = ""
    trigger_dimensions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "variant_id": self.variant_id,
            "program_strategy": self.program_strategy,
            "data_strategy": self.data_strategy,
            "diversity_axes": list(self.diversity_axes),
        }
        if self.root_guidance:
            payload["root_guidance"] = list(self.root_guidance)
        if self.interaction_family:
            payload["interaction_family"] = self.interaction_family
        if self.trigger_dimensions:
            payload["trigger_dimensions"] = list(self.trigger_dimensions)
        return payload


def witness_builder_variants(
    goal_id: str,
    *,
    family: str = "v2",
) -> tuple[WitnessBuilderVariant, ...]:
    try:
        registry = _VARIANT_FAMILIES[str(family)]
    except KeyError as exc:
        raise ValueError(f"unknown semantic witness variant family: {family!r}") from exc
    return registry[str(goal_id)]


def all_witness_builder_variants(goal_id: str) -> tuple[WitnessBuilderVariant, ...]:
    observed: dict[str, WitnessBuilderVariant] = {}
    for registry in _VARIANT_FAMILIES.values():
        for variant in registry[str(goal_id)]:
            observed.setdefault(variant.variant_id, variant)
    return tuple(observed.values())


def registered_witness_variant_count(*, family: str = "v2") -> int:
    try:
        registry = _VARIANT_FAMILIES[str(family)]
    except KeyError as exc:
        raise ValueError(f"unknown semantic witness variant family: {family!r}") from exc
    return sum(len(variants) for variants in registry.values())


def _order_nested_double_topk(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=0, flavor=0)
    case = generate_nested_topk_offset_aggregate_case(pattern_seed)
    _vary_order_rows(case, pattern_seed, flavor=0)
    return _mark(case, "nested_double_topk", seed, pattern_seed)


def _order_offset_before_limit(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=0, flavor=1)
    case = generate_nested_topk_offset_aggregate_case(pattern_seed)
    operations = _operation_dicts(case)
    # Keep one ordered cut before the grouped observation, but exercise the
    # OFFSET -> LIMIT path instead of the original nested LIMIT -> OFFSET path.
    _replace_operations(
        case,
        [
            operations[0],
            operations[1],
            operations[3],
            operations[2],
            operations[6],
            operations[7],
        ],
    )
    _vary_order_rows(case, pattern_seed, flavor=1)
    return _mark(case, "offset_before_limit", seed, pattern_seed)


def _order_resort_then_group(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=0, flavor=2)
    case = generate_topk_resort_case(pattern_seed)
    operations = _operation_dicts(case)
    operations[1]["n"] = max(2, int(operations[1].get("n", 0) or 0))
    operations[3]["n"] = 1
    operations.extend(
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "max", "as": "max_x"},
                    {"column": "z", "func": "sum", "as": "sum_z"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_x", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
        ]
    )
    _replace_operations(case, operations)
    return _mark(case, "resort_then_group", seed, pattern_seed)


def _order_join_cut_membership(seed: int) -> Case:
    """Exercise a join/runtime-filter boundary across an ordered cardinality cut."""

    pattern_seed = _builder_pattern_seed(seed, goal_slot=0, flavor=2)
    case = generate_join_ordered_agg_topk_case(pattern_seed)
    _vary_join_cut_rows(case, pattern_seed)
    pattern_index = int(pattern_seed) // 6
    allowed_groups = (
        ("a", "b"),
        ("b", "c"),
        ("a", "d"),
    )[pattern_index]
    case.tables.append(
        TableData(
            "t_allowed_groups",
            [ColumnSpec("g_key", "str", nullable=False)],
            [{"g_key": value} for value in allowed_groups],
        )
    )
    _replace_operations(
        case,
        [
            {
                "op": "join",
                "table": "t1",
                "left_on": "id",
                "right_on": "id",
                "how": "left",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": 1},
            {"op": "limit", "n": 4 + pattern_index},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "max", "as": "max_x"},
                    {"column": "z", "func": "max", "as": "max_z"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_z", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
            {
                "op": "semi_join",
                "table": "t_allowed_groups",
                "left_on": "g",
                "right_on": "g_key",
            },
        ],
    )
    case.metadata = dict(case.metadata or {})
    case.metadata["root_guided_interaction"] = {
        "root_ids": ["duckdb-join-filter-pushdown-limit-001"],
        "interaction_family": "join_ordered_cut_pushdown",
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    return _mark(
        case,
        "join_cut_membership",
        seed,
        pattern_seed,
        schema_version=ROOT_GUIDED_WITNESS_VARIANT_SCHEMA_VERSION,
    )


def _membership_guarded_case(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=1, flavor=0)
    case = generate_multi_key_anti_join_null_guard_case(pattern_seed)
    _vary_membership_rows(case, pattern_seed, flavor=0)
    return _mark(case, "guarded_membership", seed, pattern_seed)


def _membership_guarded_distinct(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=1, flavor=1)
    case = generate_multi_key_anti_join_null_guard_case(pattern_seed)
    operations = _operation_dicts(case)
    operations.insert(
        2,
        {
            "op": "distinct",
            "columns": ["row_id", "k1", "k2", "x", "flag"],
        },
    )
    _replace_operations(case, operations)
    _vary_membership_rows(case, pattern_seed, flavor=1)
    return _mark(case, "guarded_distinct_membership", seed, pattern_seed)


def _membership_projected_observation(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=1, flavor=2)
    case = generate_multi_key_anti_join_null_guard_case(pattern_seed)
    operations = _operation_dicts(case)
    operations[0], operations[1] = operations[1], operations[0]
    operations.insert(
        3,
        {
            "op": "select",
            "columns": ["row_id", "k1", "k2", "x", "flag"],
        },
    )
    _replace_operations(case, operations)
    _vary_membership_rows(case, pattern_seed, flavor=2)
    return _mark(case, "projected_membership_observation", seed, pattern_seed)


def _membership_ordered_cut(seed: int) -> Case:
    """Keep the nullable membership witness while cutting its survivor stream."""

    pattern_seed = _builder_pattern_seed(seed, goal_slot=1, flavor=2)
    case = generate_multi_key_anti_join_null_guard_case(pattern_seed)
    operations = _operation_dicts(case)
    pattern_index = int(pattern_seed) // 6
    operations[4:4] = [
        {
            "op": "sort",
            "keys": [
                {"column": "x", "ascending": False, "nulls": "last"},
                {"column": "row_id", "ascending": True, "nulls": "last"},
            ],
        },
        {"op": "offset", "n": 1},
        {"op": "limit", "n": 3 + pattern_index},
    ]
    _replace_operations(case, operations)
    _vary_membership_rows(case, pattern_seed, flavor=2)
    case.metadata = dict(case.metadata or {})
    case.metadata["root_guided_interaction"] = {
        "root_ids": ["duckdb-join-filter-pushdown-limit-001"],
        "interaction_family": "membership_ordered_cut_pushdown",
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    return _mark(
        case,
        "ordered_cut_membership",
        seed,
        pattern_seed,
        schema_version=ROOT_GUIDED_WITNESS_VARIANT_SCHEMA_VERSION,
    )


def _union_materialized_window(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=2, flavor=0)
    case = generate_union_distinct_anti_running_sum_case(pattern_seed)
    _vary_union_rows(case, pattern_seed, flavor=0)
    return _mark(case, "materialized_anti_window", seed, pattern_seed)


def _union_filter_before_distinct(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=2, flavor=1)
    case = generate_union_distinct_anti_running_sum_case(pattern_seed)
    operations = _operation_dicts(case)
    filter_operation = operations.pop(2)
    operations.insert(1, filter_operation)
    _replace_operations(case, operations)
    _vary_union_rows(case, pattern_seed, flavor=1)
    return _mark(case, "filter_before_distinct_window", seed, pattern_seed)


def _union_case_before_anti(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=2, flavor=2)
    case = generate_union_distinct_anti_running_sum_case(pattern_seed)
    operations = _operation_dicts(case)
    case_when = operations.pop(4)
    operations.insert(3, case_when)
    _replace_operations(case, operations)
    _vary_union_rows(case, pattern_seed, flavor=2)
    return _mark(case, "case_before_anti_window", seed, pattern_seed)


def _large_cast_membership_window(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=3, flavor=0)
    case = generate_large_int_text_membership_window_case(pattern_seed)
    _vary_large_integer_rows(case, pattern_seed, flavor=0)
    return _mark(case, "cast_membership_window", seed, pattern_seed)


def _large_cast_distinct_window(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=3, flavor=1)
    case = generate_large_int_text_membership_window_case(pattern_seed)
    operations = _operation_dicts(case)
    operations.insert(
        3,
        {
            "op": "distinct",
            "columns": ["id", "acct", "num_s", "num_value", "delta"],
        },
    )
    _replace_operations(case, operations)
    _vary_large_integer_rows(case, pattern_seed, flavor=1)
    return _mark(case, "cast_distinct_membership_window", seed, pattern_seed)


def _large_cast_classify_then_rank(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=3, flavor=2)
    case = generate_large_int_text_membership_window_case(pattern_seed)
    operations = _operation_dicts(case)
    case_when = operations.pop(5)
    operations.insert(2, case_when)
    window_index = next(
        index for index, operation in enumerate(operations) if operation["op"] == "running_sum"
    )
    operations.insert(
        window_index + 1,
        {
            "op": "row_number_filter",
            "partition_by": ["acct", "magnitude_bucket"],
            "order_by": [
                {"column": "run_delta", "ascending": False, "nulls": "last"},
                {"column": "id", "ascending": True, "nulls": "last"},
            ],
            "cmp": "<=",
            "value": 3,
        },
    )
    _replace_operations(case, operations)
    _vary_large_integer_rows(case, pattern_seed, flavor=2)
    return _mark(
        case,
        "classify_membership_ranked_window",
        seed,
        pattern_seed,
    )


def _empty_union_grouped(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=4, flavor=0)
    case = generate_empty_then_union_groupby_case(pattern_seed)
    _vary_empty_union_rows(case, pattern_seed, flavor=0)
    return _mark(case, "empty_union_grouped", seed, pattern_seed)


def _empty_union_window_grouped(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=4, flavor=1)
    case = generate_empty_union_window_aggregate_case(pattern_seed)
    _vary_empty_union_rows(case, pattern_seed, flavor=1)
    return _mark(case, "empty_union_window_grouped", seed, pattern_seed)


def _empty_union_distinct_grouped(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=4, flavor=2)
    case = generate_empty_then_union_groupby_case(pattern_seed)
    operations = _operation_dicts(case)
    group_index = next(
        index for index, operation in enumerate(operations) if operation["op"] == "groupby"
    )
    operations.insert(
        group_index,
        {"op": "distinct", "columns": ["id", "g", "x", "s"]},
    )
    _replace_operations(case, operations)
    _vary_empty_union_rows(case, pattern_seed, flavor=2)
    return _mark(case, "empty_union_distinct_grouped", seed, pattern_seed)


def _nullable_grouped_topk(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=5, flavor=0)
    case = generate_null_groupby_topk_case(pattern_seed)
    _vary_nullable_topk_rows(case, pattern_seed, flavor=0)
    return _mark(case, "nullable_grouped_topk", seed, pattern_seed)


def _nullable_direct_grouped_topk(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=5, flavor=1)
    case = generate_null_groupby_topk_case(pattern_seed)
    operations = _operation_dicts(case)
    _replace_operations(case, operations[1:])
    _vary_nullable_topk_rows(case, pattern_seed, flavor=1)
    return _mark(case, "nullable_direct_grouped_topk", seed, pattern_seed)


def _nullable_filtered_distinct_topk(seed: int) -> Case:
    pattern_seed = _builder_pattern_seed(seed, goal_slot=5, flavor=2)
    case = generate_null_groupby_topk_case(pattern_seed)
    operations = _operation_dicts(case)
    operations[:0] = [
        {"op": "filter", "column": "s", "cmp": "is_not_null", "value": None},
        {"op": "distinct", "columns": ["x", "s"]},
    ]
    _replace_operations(case, operations)
    _vary_nullable_topk_rows(case, pattern_seed, flavor=2)
    return _mark(
        case,
        "nullable_filtered_distinct_topk",
        seed,
        pattern_seed,
    )


def _vary_order_rows(case: Case, seed: int, *, flavor: int) -> None:
    rows = case.tables[0].rows
    next_id = max((int(row.get("id", -1)) for row in rows), default=-1) + 1
    count = 1 + ((int(seed) // 6 + flavor) % 3)
    for index in range(count):
        rows.append(
            {
                "id": next_id + index,
                "grp": f"seed_group_{(seed + flavor + index) % 5}",
                "score": [10, 7, 5][(seed + flavor + index) % 3],
                "x": None if (seed + index + flavor) % 3 == 0 else (seed + index) % 11 - 5,
                "flag": [True, False, None][(seed + flavor + index) % 3],
            }
        )


def _vary_join_cut_rows(case: Case, seed: int) -> None:
    left, right = case.tables[:2]
    pattern_index = int(seed) // 6
    tied_id = 1 + pattern_index
    left.rows.extend(
        [
            {
                "id": tied_id,
                "g": f"root_group_{pattern_index}",
                "x": None if pattern_index == 1 else 7 - pattern_index,
            },
            {
                "id": tied_id,
                "g": f"root_group_{pattern_index}",
                "x": -7 + pattern_index,
            },
        ]
    )
    right.rows.append(
        {
            "id": tied_id,
            "j": 20 + pattern_index,
            "z": None if pattern_index == 2 else 10 - pattern_index,
            "tag": f"root_tag_{pattern_index}",
        }
    )


def _vary_membership_rows(case: Case, seed: int, *, flavor: int) -> None:
    left, right = case.tables[:2]
    next_id = max((int(row.get("row_id", -1)) for row in left.rows), default=-1) + 1
    duplicate_source = left.rows[(seed + flavor) % 3]
    duplicate_count = 1 + ((seed // 6 + flavor) % 2)
    for index in range(duplicate_count):
        row = dict(duplicate_source)
        row["row_id"] = next_id + index
        row["x"] = None if (seed + index) % 2 == 0 else (seed + flavor + index) % 9
        left.rows.append(row)
    right.rows.append(
        {
            "rk1": 20 + (seed + flavor) % 7,
            "rk2": f"seed_lookup_{(seed // 6 + flavor) % 5}",
        }
    )


def _vary_union_rows(case: Case, seed: int, *, flavor: int) -> None:
    base, append = case.tables[:2]
    append.rows.append(dict(base.rows[(seed + flavor) % len(base.rows)]))
    next_id = max(
        (int(row.get("sample_id", -1)) for table in case.tables for row in table.rows),
        default=-1,
    ) + 1
    count = 1 + ((seed // 6 + flavor) % 3)
    for index in range(count):
        append.rows.append(
            {
                "sample_id": next_id + index,
                "grp": f"seed_partition_{(seed + flavor) % 4}",
                "seq": 20 + index,
                "x": None if (seed + flavor + index) % 2 == 0 else float((seed + index) % 13 - 6),
                "flag": [True, False, None][(seed + flavor + index) % 3],
            }
        )


def _vary_large_integer_rows(case: Case, seed: int, *, flavor: int) -> None:
    high = 9_007_199_254_740_992
    rows = case.tables[0].rows
    next_id = max((int(row.get("id", -1)) for row in rows), default=-1) + 1
    count = 1 + ((seed // 6 + flavor) % 3)
    for index in range(count):
        signed = -1 if (seed + flavor + index) % 2 else 1
        rows.append(
            {
                "id": next_id + index,
                "acct": f"seed_acct_{(seed + flavor) % 4}",
                "num_s": str(signed * (high + 2 + ((seed + index) % 5))),
                "delta": None if (seed + index) % 3 == 0 else signed * (index + 1),
            }
        )


def _vary_empty_union_rows(case: Case, seed: int, *, flavor: int) -> None:
    append = case.tables[1]
    id_column = "id"
    next_id = max((int(row.get(id_column, -1)) for row in append.rows), default=-1) + 1
    count = 1 + ((seed // 6 + flavor) % 3)
    amount_column = "x" if any(column.name == "x" for column in append.columns) else "amount"
    for index in range(count):
        row = {column.name: None for column in append.columns}
        row.update(
            {
                id_column: next_id + index,
                "g" if any(column.name == "g" for column in append.columns) else "grp": (
                    None if (seed + index + flavor) % 4 == 0 else f"seed_group_{(seed + flavor + index) % 6}"
                ),
                amount_column: None if (seed + index) % 3 == 0 else (seed + index) % 9 - 4,
            }
        )
        if "s" in row:
            row["s"] = None if index % 2 == 0 else f"seed_text_{(seed + flavor) % 5}"
        if "seq" in row:
            row["seq"] = 20 + index
        if "flag" in row:
            row["flag"] = [True, False, None][(seed + flavor + index) % 3]
        append.rows.append(row)


def _vary_nullable_topk_rows(case: Case, seed: int, *, flavor: int) -> None:
    rows = case.tables[0].rows
    count = 1 + ((seed // 6 + flavor) % 3)
    for index in range(count):
        rows.append(
            {
                "x": None if (seed + flavor + index) % 3 == 0 else (seed + index) % 17 - 8,
                "s": f"seed_topk_{(seed + flavor + index) % 7}",
            }
        )


def _operation_dicts(case: Case) -> list[dict[str, Any]]:
    return [operation.to_dict() for operation in case.program.operations]


def _replace_operations(case: Case, operations: list[dict[str, Any]]) -> None:
    case.program = Program(case.program.program_id, case.program.seed, operations)


def _builder_pattern_seed(seed: int, *, goal_slot: int, flavor: int) -> int:
    pattern_index = (int(seed) // 18 + int(flavor)) % SEMANTIC_WITNESS_DATA_PATTERN_COUNT
    return int(goal_slot) + 6 * pattern_index


def _mark(
    case: Case,
    variant_id: str,
    seed: int,
    pattern_seed: int,
    *,
    schema_version: str = SEMANTIC_WITNESS_VARIANT_SCHEMA_VERSION,
) -> Case:
    case.seed = int(seed)
    case.program.seed = int(seed)
    case.case_id = f"case-{int(seed):08d}-{variant_id}"
    case.program.program_id = f"prog-{int(seed):08d}-{variant_id}"
    case.metadata = dict(case.metadata or {})
    case.metadata["semantic_witness_builder_variant"] = variant_id
    case.metadata["semantic_witness_data_pattern"] = {
        "schema_version": schema_version,
        "pattern_count": SEMANTIC_WITNESS_DATA_PATTERN_COUNT,
        "pattern_seed": int(pattern_seed),
        "pattern_index": int(pattern_seed) // 6,
        "bounded_for_cache_reuse": True,
    }
    return case


_VARIANTS_BY_GOAL_V2: dict[str, tuple[WitnessBuilderVariant, ...]] = {
    "order_offset_aggregate": (
        WitnessBuilderVariant(
            "nested_double_topk",
            _order_nested_double_topk,
            "nested LIMIT/OFFSET and resort before grouped observation",
            "seeded tied rows, nullable payloads, and group-count variation",
            ("cut_order", "resort_depth", "row_count", "group_cardinality"),
        ),
        WitnessBuilderVariant(
            "offset_before_limit",
            _order_offset_before_limit,
            "OFFSET before LIMIT with a single ordered cut",
            "seeded tied rows, nullable payloads, and group-count variation",
            ("cut_order", "pipeline_depth", "row_count", "group_cardinality"),
        ),
        WitnessBuilderVariant(
            "resort_then_group",
            _order_resort_then_group,
            "independent top-k resort builder followed by grouped observation",
            "random-seeded multi-column values and nullable groups",
            ("sort_keys", "resort_depth", "data_domain", "row_count"),
        ),
    ),
    "nullable_membership_join": (
        WitnessBuilderVariant(
            "guarded_membership",
            _membership_guarded_case,
            "explicit null guards before multi-key membership",
            "seeded duplicate keys and unmatched lookup rows",
            ("join_kind", "duplicate_pressure", "lookup_cardinality"),
        ),
        WitnessBuilderVariant(
            "guarded_distinct_membership",
            _membership_guarded_distinct,
            "guarded projection distinct before membership",
            "seeded duplicate keys and unmatched lookup rows",
            ("materialization", "join_kind", "duplicate_pressure", "row_count"),
        ),
        WitnessBuilderVariant(
            "projected_membership_observation",
            _membership_projected_observation,
            "reversed guards and post-membership projection before case/group observation",
            "seeded duplicate keys and unmatched lookup rows",
            ("guard_order", "projection", "join_kind", "lookup_cardinality"),
        ),
    ),
    "union_distinct_window": (
        WitnessBuilderVariant(
            "materialized_anti_window",
            _union_materialized_window,
            "UNION/DISTINCT feeding anti membership and running window",
            "cross-branch duplicates and seeded partition-size variation",
            ("bag_set_transition", "partition_cardinality", "null_density"),
        ),
        WitnessBuilderVariant(
            "filter_before_distinct_window",
            _union_filter_before_distinct,
            "null filter before the UNION/DISTINCT materialization boundary",
            "cross-branch duplicates and seeded partition-size variation",
            ("filter_position", "bag_set_transition", "partition_cardinality"),
        ),
        WitnessBuilderVariant(
            "case_before_anti_window",
            _union_case_before_anti,
            "classification before anti membership and running window",
            "cross-branch duplicates and seeded partition-size variation",
            ("case_position", "partition_cardinality", "null_density"),
        ),
    ),
    "large_integer_cast_window": (
        WitnessBuilderVariant(
            "cast_membership_window",
            _large_cast_membership_window,
            "exact integer-string cast feeding membership and running window",
            "seeded signed values beyond the adjacent precision boundary",
            ("cast_boundary", "signed_domain", "row_count", "partition_cardinality"),
        ),
        WitnessBuilderVariant(
            "cast_distinct_membership_window",
            _large_cast_distinct_window,
            "post-membership distinct before running window",
            "seeded signed values beyond the adjacent precision boundary",
            ("materialization", "cast_boundary", "row_count", "partition_cardinality"),
        ),
        WitnessBuilderVariant(
            "classify_membership_ranked_window",
            _large_cast_classify_then_rank,
            "pre-membership classification and post-window row-number observation",
            "seeded signed values beyond the adjacent precision boundary",
            ("case_position", "window_observation", "signed_domain", "row_count"),
        ),
    ),
    "empty_union_groupby": (
        WitnessBuilderVariant(
            "empty_union_grouped",
            _empty_union_grouped,
            "empty filtered branch unioned directly into grouped observation",
            "seeded append rows, group counts, and nullable payloads",
            ("row_count", "group_cardinality", "null_density"),
        ),
        WitnessBuilderVariant(
            "empty_union_window_grouped",
            _empty_union_window_grouped,
            "empty union feeding running/window rank before grouped observation",
            "seeded append rows, group counts, and nullable payloads",
            ("window_depth", "row_count", "group_cardinality", "null_density"),
        ),
        WitnessBuilderVariant(
            "empty_union_distinct_grouped",
            _empty_union_distinct_grouped,
            "empty union materialized through distinct before grouped observation",
            "seeded append rows, group counts, and nullable payloads",
            ("materialization", "row_count", "group_cardinality", "null_density"),
        ),
    ),
    "nullable_grouped_topk": (
        WitnessBuilderVariant(
            "nullable_grouped_topk",
            _nullable_grouped_topk,
            "derived-key grouped top-k repaired to nullable aggregate observation",
            "seeded row, group, and nullable-value variation",
            ("derived_key", "row_count", "group_cardinality", "null_density"),
        ),
        WitnessBuilderVariant(
            "nullable_direct_grouped_topk",
            _nullable_direct_grouped_topk,
            "direct grouped top-k without the derived-key prelude",
            "seeded row, group, and nullable-value variation",
            ("pipeline_depth", "row_count", "group_cardinality", "null_density"),
        ),
        WitnessBuilderVariant(
            "nullable_filtered_distinct_topk",
            _nullable_filtered_distinct_topk,
            "filter/distinct materialization before nullable grouped top-k",
            "seeded row, group, and nullable-value variation",
            ("filtering", "materialization", "row_count", "group_cardinality"),
        ),
    ),
}


_ROOT_GUIDED_ORDER_VARIANT = WitnessBuilderVariant(
    "join_cut_membership",
    _order_join_cut_membership,
    "left join before ordered OFFSET/LIMIT and grouped observation, followed by membership",
    "three bounded join-multiplicity, nullable-payload, and allowed-group patterns",
    ("join_cardinality", "cut_order", "runtime_filter", "group_cardinality"),
    schema_version=ROOT_GUIDED_WITNESS_VARIANT_SCHEMA_VERSION,
    root_guidance=("duckdb-join-filter-pushdown-limit-001",),
    interaction_family="join_ordered_cut_pushdown",
    trigger_dimensions=("join_before_cut", "cut_before_group", "membership_after_group"),
)

_ROOT_GUIDED_MEMBERSHIP_VARIANT = WitnessBuilderVariant(
    "ordered_cut_membership",
    _membership_ordered_cut,
    "guarded multi-key anti membership followed by ordered OFFSET/LIMIT before grouping",
    "three bounded survivor-cut patterns with matched, unmatched, duplicate, and null keys",
    ("join_cardinality", "cut_order", "runtime_filter", "null_density"),
    schema_version=ROOT_GUIDED_WITNESS_VARIANT_SCHEMA_VERSION,
    root_guidance=("duckdb-join-filter-pushdown-limit-001",),
    interaction_family="membership_ordered_cut_pushdown",
    trigger_dimensions=("membership_before_cut", "positive_offset", "cut_before_group"),
)

_VARIANTS_BY_GOAL_V3: dict[str, tuple[WitnessBuilderVariant, ...]] = {
    **_VARIANTS_BY_GOAL_V2,
    "order_offset_aggregate": (
        _VARIANTS_BY_GOAL_V2["order_offset_aggregate"][0],
        _VARIANTS_BY_GOAL_V2["order_offset_aggregate"][1],
        _ROOT_GUIDED_ORDER_VARIANT,
    ),
    "nullable_membership_join": (
        _VARIANTS_BY_GOAL_V2["nullable_membership_join"][0],
        _VARIANTS_BY_GOAL_V2["nullable_membership_join"][1],
        _ROOT_GUIDED_MEMBERSHIP_VARIANT,
    ),
}

_VARIANT_FAMILIES: dict[
    str,
    dict[str, tuple[WitnessBuilderVariant, ...]],
] = {
    "v2": _VARIANTS_BY_GOAL_V2,
    "v3": _VARIANTS_BY_GOAL_V3,
}
