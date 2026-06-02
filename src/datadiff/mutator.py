from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from datadiff.datagen import repair_operations
from datadiff.dsl import Case, ColumnSpec, IRNode, Program, TableData, normalize_sort_keys
from datadiff.identifiers import make_safe_output_name
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import aggregate_alias, aggregate_column, aggregate_func, aggregate_specs, op_ascending, op_column, op_kind, op_n, operation_count
from datadiff.program_state import ProgramState, state_after_operations
from datadiff.util import unique_preserve_order

INHERITED_METADATA_KEYS = (
    "generator_profile",
    "mixed_generator_profile",
    "source_issue",
    "source_issue_alt",
)

BOOLEAN_PROBE_OUTPUT_PREFIXES = (
    "sorted_ok_",
    "unexpected_else_seen",
    "quantile_key_mismatch",
    "scalar_subquery_mismatch",
    "window_avg_mismatch",
    "struct_distinct_mismatch",
    "bit_compare_mismatch",
    "round_even_mismatch",
    "series_rtruediv_mismatch",
    "uint64_isin_mismatch",
    "tuple_anti_null_mismatch",
    "setop_all_duplicate_mismatch",
    "json_predicate_order_mismatch",
    "sparse_mask_mismatch",
    "float_wrap_mismatch",
    "index_bool_mismatch",
    "empty_literal_groupby_mismatch",
    "arrow_string_eq_sum_mismatch",
    "arrow_timestamp_loc_slice_mismatch",
    "arrow_timestamp_index_attr_mismatch",
    "eval_inplace_alias_mismatch",
    "bool_reduction_skipna_mismatch",
    "dataset_isin_all_match_mismatch",
    "run_end_null_compute_mismatch",
    "large_string_partition_mismatch",
    "hash_pivot_wider_mismatch",
    "list_flatten_parent_indices_mismatch",
    "rolling_mean_by_null_count_mismatch",
)

IMMUTABLE_VALUE_TYPES = (str, bytes, int, float, bool, type(None))


@dataclass(slots=True)
class MutationResult:
    case: Case
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True, init=False)
class MutationOperator:
    name: str
    apply: Callable[[list[TableData], list[dict[str, Any]], random.Random], str]
    semantic_family_affinity: tuple[str, ...]
    semantic_signal_affinity: tuple[str, ...]

    def __init__(
        self,
        name: str,
        apply: Callable[[list[TableData], list[dict[str, Any]], random.Random], str],
        semantic_family_affinity: tuple[str, ...] = (),
        semantic_signal_affinity: tuple[str, ...] = (),
        *,
        semantic_affinity: tuple[str, ...] | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "apply", apply)
        family_affinity = semantic_family_affinity if semantic_affinity is None else semantic_affinity
        object.__setattr__(self, "semantic_family_affinity", tuple(family_affinity))
        object.__setattr__(self, "semantic_signal_affinity", tuple(semantic_signal_affinity))

    @property
    def semantic_affinity(self) -> tuple[str, ...]:
        return self.semantic_family_affinity


@dataclass(frozen=True, slots=True)
class MutationSchema:
    columns: list[str]
    column_types: dict[str, str]
    nullable_columns: set[str]

    @property
    def available(self) -> list[str]:
        return list(self.columns)

    def column_type(self, name: str) -> str:
        if name in self.column_types:
            return self.column_types[name]
        return _fallback_column_type(name)

    def column_has_null(self, name: str) -> bool:
        return name in self.nullable_columns


def mutate_case(case: Case, seed: int) -> Case:
    return mutate_case_with_metadata(case, seed).case


def mutate_case_with_metadata(
    case: Case,
    seed: int,
    *,
    allow_probe_operators: bool = True,
    operator_scores: Mapping[str, float] | None = None,
) -> MutationResult:
    rnd = random.Random(seed * 104729 + case.seed)
    operator_pool = MUTATION_OPERATORS if allow_probe_operators else DISCOVERY_MUTATION_OPERATORS
    attempt_order = _mutation_attempt_order(operator_pool, rnd, operator_scores=operator_scores)
    fallback: tuple[list[TableData], list[dict[str, Any]], MutationOperator, str, bool] | None = None
    selected: tuple[list[TableData], list[dict[str, Any]], MutationOperator, str, bool] | None = None
    original_tables = case.tables
    original_ops = case.program.operations
    for operator in attempt_order:
        tables = _clone_tables(case.tables)
        table = tables[0]
        operations = _clone_operations(case.program.operations)
        detail = operator.apply(tables, operations, rnd)
        operations = repair_operations(table, operations, extra_tables=tables[1:])
        if not operations:
            operations = [{"op": "limit", "n": len(table.rows)}]
        changed = tables != original_tables or operations != original_ops
        attempt = (tables, operations, operator, detail, changed)
        fallback = attempt
        if changed and not _mutation_detail_is_unproductive(detail):
            selected = attempt
            break
    if selected is None:
        if fallback is None:
            raise RuntimeError("mutation operator pool is empty")
        selected = fallback
    tables, operations, operator, detail, changed = selected
    table = tables[0]
    choice = operator.name
    parent_lineage = case.metadata.get("seed_lineage", {}) if isinstance(case.metadata, dict) else {}
    root_seed = parent_lineage.get("root_seed", case.seed)
    depth = int(parent_lineage.get("depth", 0) or 0) + 1
    metadata = _inherited_metadata(case.metadata)
    metadata.update(
        {
            "candidate_source": "feedback_mutation",
            "seed_lineage": {
                "root_seed": root_seed,
                "parent_seed": case.seed,
                "parent_case_id": case.case_id,
                "mutation_seed": seed,
                "depth": depth,
            },
            "mutation": {
                "operator": choice,
                "detail": detail,
                "changed": changed,
            },
        }
    )
    program = Program(
        program_id=f"{case.program.program_id}-mut-{seed}",
        seed=seed,
        operations=operations,
    )
    mutated = Case(
        f"{case.case_id}-mut-{seed}",
        seed,
        tables,
        program,
        metadata=metadata,
    )
    return MutationResult(mutated, metadata)


def _mutation_attempt_order(
    operator_pool: tuple[MutationOperator, ...],
    rnd: random.Random,
    *,
    operator_scores: Mapping[str, float] | None = None,
) -> list[MutationOperator]:
    order = rnd.sample(list(operator_pool), k=len(operator_pool))
    if not operator_scores:
        return order
    untried_score = float(operator_scores.get("__untried__", 0.0))
    return sorted(
        order,
        key=lambda operator: (
            float(operator_scores.get(operator.name, untried_score)),
            rnd.random(),
        ),
        reverse=True,
    )


def mutation_operator_profiles(
    *,
    allow_probe_operators: bool = True,
) -> Mapping[str, MutationOperator]:
    if allow_probe_operators:
        return ALL_MUTATION_OPERATOR_PROFILES
    return DISCOVERY_MUTATION_OPERATOR_PROFILES


def _mutation_detail_is_unproductive(detail: str) -> bool:
    tokens = detail.split(":")[1:]
    if not tokens:
        return False
    first = tokens[0]
    return (
        first == "none"
        or first.startswith("no-")
        or first.startswith("not-enough")
        or first in {"too-few-columns", "duplicate-keys"}
    )


def _mutation_schema(tables: list[TableData], operations: list[dict[str, Any]]) -> MutationSchema:
    if not tables:
        return MutationSchema([], {}, set())
    state = state_after_operations(
        tables[0],
        operations,
        extra_tables=tables[1:],
    )
    nullable_columns = set(state.nullable_columns)
    for name in state.columns:
        if _base_table_column_has_null(tables, name):
            nullable_columns.add(name)
    return MutationSchema(list(state.columns), dict(state.column_types), nullable_columns)


def _base_table_column_has_null(tables: list[TableData], name: str) -> bool:
    return any(name in row and row.get(name) is None for table in tables for row in table.rows)


def _fallback_column_type(name: str) -> str:
    if name.startswith("sorted_ok_"):
        return "bool"
    if any(name == prefix or name.startswith(f"{prefix}_") for prefix in BOOLEAN_PROBE_OUTPUT_PREFIXES):
        return "bool"
    if name.startswith(("any_", "all_")):
        return "bool"
    return "float" if name.startswith(("m_", "sum_", "min_", "max_", "run_")) else "int"


def _inherited_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {key: _clone_value(metadata[key]) for key in INHERITED_METADATA_KEYS if key in metadata}


def _clone_tables(tables: list[TableData]) -> list[TableData]:
    return [_clone_table(table) for table in tables]


def _clone_table(table: TableData) -> TableData:
    return TableData(
        table.name,
        [ColumnSpec(column.name, column.type, nullable=column.nullable) for column in table.columns],
        [_clone_row(row) for row in table.rows],
    )


def _clone_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_value(value) for key, value in row.items()}


def _clone_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clone_operation(operation) for operation in operations]


def _clone_operation(operation: dict[str, Any]) -> dict[str, Any]:
    if isinstance(operation, IRNode):
        return operation.copy()
    return {
        key: _clone_value(value)
        for key, value in operation.items()
    }


def _clone_value(value: Any) -> Any:
    if isinstance(value, IMMUTABLE_VALUE_TYPES):
        return value
    if isinstance(value, IRNode):
        return value.copy()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return {_clone_value(item) for item in value}
    return copy.deepcopy(value)


def _mutate_scalar_value(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables:
        return "value:none"
    return _mutate_value(tables[0], rnd)


def _mutate_value(table: TableData, rnd: random.Random) -> str:
    if not table.rows:
        return "value:none"
    col = rnd.choice(table.columns)
    row = rnd.choice(table.rows)
    current = row.get(col.name)
    if current is None:
        if col.type == "str" and _looks_like_date_column(col.name):
            row[col.name] = rnd.choice(["2024-01-03", "2024-02-14T08:30:00", "2025-12-31", "2026-01-01"])
        else:
            row[col.name] = _literal_for_type(col.type, rnd)
        return f"value:{col.type}:fill-null"
    if col.type == "int":
        row[col.name] = int(current) + rnd.choice([-10, -1, 0, 1, 10])
    elif col.type == "float":
        if isinstance(current, float) and (math.isnan(current) or math.isinf(current)):
            row[col.name] = 0.0
        else:
            row[col.name] = float(current) + rnd.choice([-1.0, -0.5, 0.5, 1.0])
    elif col.type == "bool":
        row[col.name] = not bool(current)
    elif col.type == "str":
        if _looks_like_date_column(col.name):
            row[col.name] = rnd.choice(["2024-01-03", "2024-02-14T08:30:00", "2025-12-31", "2026-01-01"])
        else:
            row[col.name] = rnd.choice(["", "alpha", "ALPHA", "中文", str(current) + "_x"])
    return f"value:{col.type}:{col.name}"


def _nullify_value(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "nullify:none"
    table = tables[0]
    nullable = [col for col in table.columns if col.nullable]
    if not nullable:
        return "nullify:no-nullable-column"
    candidates = [
        (row, col)
        for row in table.rows
        for col in nullable
        if row.get(col.name) is not None
    ]
    if not candidates:
        candidates = [(row, col) for row in table.rows for col in nullable]
    row, col = rnd.choice(candidates)
    row[col.name] = None
    return f"nullify:{col.type}:{col.name}"


def _duplicate_row(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "duplicate_row:none"
    tables[0].rows.append(_clone_row(rnd.choice(tables[0].rows)))
    return f"duplicate_row:{tables[0].name}"


def _drop_row(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "drop_row:none"
    removed_index = rnd.randrange(len(tables[0].rows))
    tables[0].rows.pop(removed_index)
    return f"drop_row:{tables[0].name}:{removed_index}"


def _shuffle_rows(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or len(tables[0].rows) < 2:
        return "shuffle_rows:none"
    before = list(tables[0].rows)
    rnd.shuffle(tables[0].rows)
    if tables[0].rows == before:
        tables[0].rows.reverse()
    return f"shuffle_rows:{tables[0].name}"


def _append_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    op = _random_operation(tables, operations, rnd)
    if op is not None:
        operations.append(op)
        return f"append:{op_kind(op, 'unknown')}"
    return "append:none"


def _drop_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del tables
    if len(operations) <= 1:
        return "drop:none"
    removed = operations.pop(rnd.randrange(len(operations)))
    return f"drop:{op_kind(removed, 'unknown')}"


def _tweak_random_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not operations:
        return "tweak:none"
    op = operations[rnd.randrange(len(operations))]
    return _tweak_operation(tables, op, rnd)


def _append_order_projection_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_order_projection:none"
    available = _available_columns(tables, operations)
    if len(available) < 2:
        return "append_order_projection:too-few-columns"
    primary = rnd.choice(available)
    selected_candidates = [column for column in available if column != primary]
    selected_count = rnd.randint(1, min(3, len(selected_candidates)))
    selected = sorted(rnd.sample(selected_candidates, selected_count))
    sort_columns = [primary] + sorted(column for column in available if column != primary)
    operations.append(
        {
            "op": "sort",
            "keys": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": rnd.choice(["first", "last"]),
                }
                for column in sort_columns
            ],
        }
    )
    operations.append({"op": "select", "columns": selected})
    if rnd.random() < 0.75:
        if rnd.random() < 0.70:
            operations.append({"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))})
            tail = "limit"
        else:
            operations.append({"op": "offset", "n": rnd.randint(0, 2)})
            tail = "offset"
    else:
        tail = "none"
    return f"append_order_projection:{primary}:tail={tail}"


def _append_truth_filter_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_truth_filter:none"
    available = _available_columns(tables, operations)
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric:
        return "append_truth_filter:no-numeric-column"
    column = rnd.choice(numeric)
    comparator = rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"])
    operations.append(
        {
            "op": "filter",
            "column": column,
            "cmp": comparator,
            "value": _literal_for_type(_column_type(tables, column), rnd),
        }
    )
    return f"append_truth_filter:{column}:{comparator}"


def _append_boolean_predicate_filter_probe(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_predicate_filter:none"
    available = _available_columns(tables, operations)
    boolean_columns = [column for column in available if _column_type(tables, column) == "bool"]
    if not boolean_columns:
        return "append_boolean_predicate_filter:no-bool-column"
    column = rnd.choice(boolean_columns)
    comparator = rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"])
    operations.append({"op": "filter", "column": column, "cmp": comparator, "value": None})
    return f"append_boolean_predicate_filter:{column}:{comparator}"


def _append_range_filter_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_range_filter:none"
    available = _available_columns(tables, operations)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric_columns:
        return "append_range_filter:no-numeric-column"
    column = rnd.choice(numeric_columns)
    values = sorted(rnd.sample(_literal_list_for_type(_column_type(tables, column), rnd), 2))
    operations.append({"op": "filter", "column": column, "cmp": "range_closed", "value": values})
    return f"append_range_filter:{column}:{values[0]}:{values[1]}"


def _append_tuple_absence_filter_probe(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if len(tables) < 2:
        return "append_tuple_absence_filter:no-right-table"
    available = _available_columns(tables, operations)
    right = rnd.choice(tables[1:])
    right_columns = [column.name for column in right.columns]
    pairs = [
        (left, right_column)
        for left in available
        for right_column in right_columns
        if _column_type(tables, left) == right.column_type(right_column)
    ]
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for left, right_column in rnd.sample(pairs, k=len(pairs)):
        if left in left_seen or right_column in right_seen:
            continue
        chosen.append((left, right_column))
        left_seen.add(left)
        right_seen.add(right_column)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return "append_tuple_absence_filter:no-compatible-pairs"
    left_columns = [left for left, _ in chosen]
    selected_right_columns = [right_column for _, right_column in chosen]
    operations.append(
        {
            "op": "tuple_absence_filter",
            "columns": left_columns,
            "table": right.name,
            "right_columns": selected_right_columns,
        }
    )
    return f"append_tuple_absence_filter:{','.join(left_columns)}:{right.name}:{','.join(selected_right_columns)}"


def _append_row_value_absence_filter(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    detail = _append_tuple_absence_filter_probe(tables, operations, rnd)
    if detail.startswith("append_tuple_absence_filter:"):
        return "append_row_value_absence_filter:" + detail.split(":", 1)[1]
    return detail.replace("append_tuple_absence_filter", "append_row_value_absence_filter", 1)


def _append_running_sum_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_running_sum:none"
    available = _available_columns(tables, operations)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric_columns or not available:
        return "append_running_sum:no-numeric-column"
    source = rnd.choice(numeric_columns)
    order_column = rnd.choice(available)
    order_columns = [order_column] + sorted(column for column in available if column != order_column)
    used = set(available)
    output_column = make_safe_output_name(f"run_{source}", used=used)
    operations.append(
        {
            "op": "running_sum",
            "source": source,
            "column": output_column,
            "order_by": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": rnd.choice(["first", "last"]),
                }
                for column in order_columns
            ],
            "input_dtype": "float32" if _column_type(tables, source) == "float" or rnd.random() < 0.5 else "float64",
        }
    )
    return f"append_running_sum:{source}:order={','.join(order_columns)}:out={output_column}"


def _append_sortedness_check_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_sortedness_check:none"
    available = _available_columns(tables, operations)
    candidates = [
        column
        for column in available
        if _column_type(tables, column) in {"int", "float", "str", "bool"}
    ]
    if not candidates:
        return "append_sortedness_check:no-comparable-column"
    null_candidates = [column for column in candidates if _column_has_null(tables, column)]
    column = rnd.choice(null_candidates or candidates)
    sort_nulls = rnd.choice(["first", "last"])
    check_nulls = "last" if sort_nulls == "first" else "first"
    ascending = rnd.choice([True, False])
    alias = make_safe_output_name(f"sorted_ok_{column}", used=set(available))
    operations.extend(
        [
            {
                "op": "sort",
                "keys": [{"column": column, "ascending": ascending, "nulls": sort_nulls}],
            },
            {
                "op": "sortedness_check",
                "column": column,
                "as": alias,
                "ascending": ascending,
                "nulls": check_nulls,
            },
        ]
    )
    return f"append_sortedness_check:{column}:sort_nulls={sort_nulls}:check_nulls={check_nulls}:out={alias}"


def _append_random_case_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_random_case_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("unexpected_else_seen", used=set(available))
    rows = rnd.choice([50_000, 100_000, 150_000])
    branches = rnd.choice([3, 4])
    operations.append({"op": "random_case_probe", "as": alias, "rows": rows, "branches": branches})
    return f"append_random_case_probe:rows={rows}:branches={branches}:out={alias}"


def _append_group_quantile_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_group_quantile_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("quantile_key_mismatch", used=set(available))
    operations.append(
        {
            "op": "group_quantile_probe",
            "as": alias,
            "values": [1, 2, 3],
            "quantiles": [0.0, 0.5, 1.0],
        }
    )
    return f"append_group_quantile_probe:out={alias}"


def _append_scalar_subquery_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_scalar_subquery_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("scalar_subquery_mismatch", used=set(available))
    operations.append({"op": "scalar_subquery_probe", "as": alias})
    return f"append_scalar_subquery_probe:out={alias}"


def _append_window_avg_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_window_avg_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("window_avg_mismatch", used=set(available))
    operations.append({"op": "window_avg_probe", "as": alias})
    return f"append_window_avg_probe:out={alias}"


def _append_struct_distinct_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_struct_distinct_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("struct_distinct_mismatch", used=set(available))
    operations.append({"op": "struct_distinct_probe", "as": alias})
    return f"append_struct_distinct_probe:out={alias}"


def _append_bit_compare_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_bit_compare_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("bit_compare_mismatch", used=set(available))
    operations.append({"op": "bit_compare_probe", "as": alias})
    return f"append_bit_compare_probe:out={alias}"


def _append_round_even_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_round_even_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("round_even_mismatch", used=set(available))
    operations.append({"op": "round_even_probe", "as": alias})
    return f"append_round_even_probe:out={alias}"


def _append_series_rtruediv_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_series_rtruediv_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("series_rtruediv_mismatch", used=set(available))
    operations.append({"op": "series_rtruediv_probe", "as": alias})
    return f"append_series_rtruediv_probe:out={alias}"


def _append_uint64_isin_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_uint64_isin_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("uint64_isin_mismatch", used=set(available))
    operations.append({"op": "uint64_isin_probe", "as": alias})
    return f"append_uint64_isin_probe:out={alias}"


def _append_tuple_anti_null_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_tuple_anti_null_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("tuple_anti_null_mismatch", used=set(available))
    operations.append({"op": "tuple_anti_null_probe", "as": alias})
    return f"append_tuple_anti_null_probe:out={alias}"


def _append_setop_all_duplicate_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_setop_all_duplicate_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("setop_all_duplicate_mismatch", used=set(available))
    operations.append({"op": "setop_all_duplicate_probe", "as": alias})
    return f"append_setop_all_duplicate_probe:out={alias}"


def _append_json_predicate_order_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_json_predicate_order_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("json_predicate_order_mismatch", used=set(available))
    operations.append({"op": "json_predicate_order_probe", "as": alias})
    return f"append_json_predicate_order_probe:out={alias}"


def _append_sparse_mask_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_sparse_mask_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("sparse_mask_mismatch", used=set(available))
    operations.append({"op": "sparse_mask_probe", "as": alias})
    return f"append_sparse_mask_probe:out={alias}"


def _append_float_wrap_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_float_wrap_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("float_wrap_mismatch", used=set(available))
    operations.append({"op": "float_wrap_probe", "as": alias})
    return f"append_float_wrap_probe:out={alias}"


def _append_index_bool_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_index_bool_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("index_bool_mismatch", used=set(available))
    operations.append({"op": "index_bool_probe", "as": alias})
    return f"append_index_bool_probe:out={alias}"


def _append_empty_literal_groupby_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_empty_literal_groupby_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("empty_literal_groupby_mismatch", used=set(available))
    operations.append({"op": "empty_literal_groupby_probe", "as": alias})
    return f"append_empty_literal_groupby_probe:out={alias}"


def _append_arrow_string_eq_sum_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_arrow_string_eq_sum_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("arrow_string_eq_sum_mismatch", used=set(available))
    operations.append({"op": "arrow_string_eq_sum_probe", "as": alias})
    return f"append_arrow_string_eq_sum_probe:out={alias}"


def _append_arrow_timestamp_loc_slice_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_arrow_timestamp_loc_slice_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("arrow_timestamp_loc_slice_mismatch", used=set(available))
    operations.append({"op": "arrow_timestamp_loc_slice_probe", "as": alias})
    return f"append_arrow_timestamp_loc_slice_probe:out={alias}"


def _append_arrow_timestamp_index_attr_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_arrow_timestamp_index_attr_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("arrow_timestamp_index_attr_mismatch", used=set(available))
    operations.append({"op": "arrow_timestamp_index_attr_probe", "as": alias})
    return f"append_arrow_timestamp_index_attr_probe:out={alias}"


def _append_eval_inplace_alias_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_eval_inplace_alias_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("eval_inplace_alias_mismatch", used=set(available))
    operations.append({"op": "eval_inplace_alias_probe", "as": alias})
    return f"append_eval_inplace_alias_probe:out={alias}"


def _append_bool_reduction_skipna_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_bool_reduction_skipna_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("bool_reduction_skipna_mismatch", used=set(available))
    operations.append({"op": "bool_reduction_skipna_probe", "as": alias})
    return f"append_bool_reduction_skipna_probe:out={alias}"


def _append_dataset_isin_all_match_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_dataset_isin_all_match_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("dataset_isin_all_match_mismatch", used=set(available))
    operations.append({"op": "dataset_isin_all_match_probe", "as": alias})
    return f"append_dataset_isin_all_match_probe:out={alias}"


def _append_run_end_null_compute_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_run_end_null_compute_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("run_end_null_compute_mismatch", used=set(available))
    operations.append({"op": "run_end_null_compute_probe", "as": alias})
    return f"append_run_end_null_compute_probe:out={alias}"


def _append_large_string_partition_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_large_string_partition_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("large_string_partition_mismatch", used=set(available))
    operations.append({"op": "large_string_partition_probe", "as": alias})
    return f"append_large_string_partition_probe:out={alias}"


def _append_hash_pivot_wider_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_hash_pivot_wider_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("hash_pivot_wider_mismatch", used=set(available))
    operations.append({"op": "hash_pivot_wider_probe", "as": alias})
    return f"append_hash_pivot_wider_probe:out={alias}"


def _append_list_flatten_parent_indices_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_list_flatten_parent_indices_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("list_flatten_parent_indices_mismatch", used=set(available))
    operations.append({"op": "list_flatten_parent_indices_probe", "as": alias})
    return f"append_list_flatten_parent_indices_probe:out={alias}"


def _append_rolling_mean_by_null_count_probe(
    tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random
) -> str:
    if not tables:
        return "append_rolling_mean_by_null_count_probe:none"
    available = _available_columns(tables, operations)
    alias = make_safe_output_name("rolling_mean_by_null_count_mismatch", used=set(available))
    operations.append({"op": "rolling_mean_by_null_count_probe", "as": alias})
    return f"append_rolling_mean_by_null_count_probe:out={alias}"


def _append_grouped_topk_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_grouped_topk:none"
    available = _available_columns(tables, operations)
    if not available:
        return "append_grouped_topk:no-columns"
    numeric = [
        column
        for column in available
        if _column_type(tables, column) in {"int", "float"}
        or column.startswith(("m_", "sum_", "min_", "max_", "count_", "nunique_", "uniq_"))
    ]
    if not numeric:
        return "append_grouped_topk:no-numeric-column"
    key = rnd.choice(numeric)
    value = rnd.choice([column for column in numeric if column != key] or numeric)
    alias = make_safe_output_name(f"count_{value}", used={key})
    operations.extend(
        [
            {
                "op": "groupby",
                "keys": [key],
                "aggs": [{"column": value, "func": "count", "as": alias}],
            },
            {"op": "select", "columns": [key]},
            {
                "op": "sort",
                "keys": [
                    {
                        "column": key,
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                ],
            },
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
        ]
    )
    return f"append_grouped_topk:{key}:{value}"


def _append_groupby_fractional_membership_filter(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_groupby_fractional_membership_filter:none"
    available = set(_available_columns(tables, operations))
    candidates: list[tuple[str, str]] = []
    for op in operations:
        if op_kind(op) != "groupby":
            continue
        for agg in aggregate_specs(op):
            alias = aggregate_alias(agg)
            source = aggregate_column(agg)
            if (
                alias in available
                and aggregate_func(agg) in {"min", "max"}
                and _column_type(tables, source) == "int"
            ):
                candidates.append((alias, source))
    if not candidates:
        return "append_groupby_fractional_membership_filter:no-int-aggregate"
    alias, source = rnd.choice(candidates)
    source_values = [
        int(row[source])
        for table in tables
        for row in table.rows
        if source in row and isinstance(row.get(source), int) and not isinstance(row.get(source), bool)
    ]
    non_negative_values = [value for value in unique_preserve_order(source_values) if value >= 0]
    if not non_negative_values:
        return "append_groupby_fractional_membership_filter:no-nonnegative-value"
    target = rnd.choice(non_negative_values)
    operations.append(
        {
            "op": "filter",
            "column": alias,
            "cmp": "in_set",
            "value": [float(target) + 0.5, -999.0, 999.0],
        }
    )
    return f"append_groupby_fractional_membership_filter:{alias}:{target}"


def _append_normalized_string_membership(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_normalized_string_membership:none"
    available = _available_columns(tables, operations)
    string_columns = [column for column in available if _column_type(tables, column) == "str"]
    if not string_columns:
        return "append_normalized_string_membership:no-string-column"
    source = rnd.choice(string_columns)
    normalized_values = unique_preserve_order(
        value.strip().lower()
        for table in tables
        for row in table.rows
        for value in [row.get(source)]
        if isinstance(value, str)
    )
    if not normalized_values:
        return f"append_normalized_string_membership:no-values:{source}"
    if len(normalized_values) == 1:
        selected_values = normalized_values
        join_kind = "semi_join"
    else:
        width = rnd.randint(1, max(1, len(normalized_values) - 1))
        selected_values = sorted(rnd.sample(normalized_values, width))
        join_kind = rnd.choice(["semi_join", "anti_join"])
    table_name = make_safe_output_name("t_string_membership_mut", used={table.name for table in tables})
    tables.append(
        TableData(
            table_name,
            [ColumnSpec("s_key", "str", nullable=False)],
            [{"s_key": value} for value in selected_values],
        )
    )
    used_columns = set(available)
    clean_col = make_safe_output_name(f"{source}_clean", used=used_columns)
    used_columns.add(clean_col)
    key_col = make_safe_output_name(f"{source}_key", used=used_columns)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    operations.extend(
        [
            {"op": "mutate", "column": clean_col, "expr": {"kind": "string_strip", "source": source}},
            {"op": "mutate", "column": key_col, "expr": {"kind": "string_lower", "source": clean_col}},
            {"op": "filter", "column": key_col, "cmp": "is_not_null", "value": None},
            {"op": join_kind, "table": table_name, "left_on": key_col, "right_on": "s_key"},
        ]
    )
    if join_kind == "anti_join":
        count_source = numeric_columns[0] if numeric_columns else source
        operations.extend(
            [
                {
                    "op": "groupby",
                    "keys": [key_col],
                    "aggs": [{"column": count_source, "func": "count", "as": "count_rows"}],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": key_col, "ascending": True, "nulls": "last"},
                        {"column": "count_rows", "ascending": False, "nulls": "last"},
                    ],
                },
            ]
        )
    else:
        sort_keys = [{"column": key_col, "ascending": True, "nulls": "last"}]
        if numeric_columns:
            sort_keys.append({"column": numeric_columns[0], "ascending": False, "nulls": "last"})
        selected = unique_preserve_order([key_col, source, *numeric_columns[:1]])
        operations.extend(
            [
                {"op": "sort", "keys": sort_keys},
                {"op": "select", "columns": selected},
                {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
            ]
        )
    return f"append_normalized_string_membership:{join_kind}:{source}:{','.join(map(str, selected_values))}"


def _append_sql_distinct_null_topk(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_sql_distinct_null_topk:none"
    available = _available_columns(tables, operations)
    strings = [column for column in available if _column_type(tables, column) == "str"]
    if not strings:
        return "append_sql_distinct_null_topk:no-string-column"
    used_columns = set(available)
    source = rnd.choice(strings)
    nonempty_col = make_safe_output_name(f"{source}_nonempty", used=used_columns)
    used_columns.add(nonempty_col)
    label_col = nonempty_col
    operations.append(
        {"op": "mutate", "column": nonempty_col, "expr": {"kind": "string_null_if_empty", "source": source}}
    )
    coalesce_sources = [column for column in strings if column != source]
    if coalesce_sources:
        label_col = make_safe_output_name(f"{source}_label", used=used_columns)
        used_columns.add(label_col)
        operations.append(
            {
                "op": "coalesce",
                "columns": [nonempty_col, rnd.choice(coalesce_sources)],
                "as": label_col,
                "fallback": "missing",
            }
        )
    distinct_cols = [label_col]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    distinct_cols.extend((bools + numeric)[:2])
    distinct_cols = unique_preserve_order(distinct_cols)
    sort_keys = [
        {"column": distinct_cols[0], "ascending": True, "nulls": "first"},
        *[
            {
                "column": column,
                "ascending": False if _column_type(tables, column) != "str" else True,
                "nulls": "first" if idx == 0 else "last",
            }
            for idx, column in enumerate(distinct_cols[1:])
        ],
    ]
    operations.extend(
        [
            {"op": "distinct", "columns": distinct_cols},
            {"op": "sort", "keys": sort_keys},
            {"op": "offset", "n": rnd.randint(0, 1)},
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
        ]
    )
    return f"append_sql_distinct_null_topk:{source}:label={label_col}:cols={','.join(distinct_cols)}"


def _append_left_join_coalesce_membership(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_coalesce_membership:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    string_candidates = [column for column in available if _column_type(tables, column) == "str"]
    if not key_candidates or not string_candidates:
        return "append_left_join_coalesce_membership:no-key-or-string-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    key_type = base_columns[key].type
    key_values = unique_preserve_order(
        row.get(key)
        for row in tables[0].rows
        if row.get(key) is not None
    )
    if not key_values:
        return f"append_left_join_coalesce_membership:no-key-values:{key}"
    used_table_names = {table.name for table in tables}
    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_membership_mut", used=used_table_names)
    membership_table = make_safe_output_name("t_segment_membership_mut", used=used_table_names | {lookup_table})
    right_key = "lookup_key"
    label_col = make_safe_output_name("lookup_label", used=used_columns)
    used_columns.add(label_col)
    value_col = make_safe_output_name("lookup_j", used=used_columns)
    used_columns.add(value_col)
    segment_col = make_safe_output_name("segment_key", used=used_columns)
    labels = ["dim-a", "dim-b", "space value", None]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, key_type, nullable=False),
                ColumnSpec(label_col, "str", nullable=True),
                ColumnSpec(value_col, "int", nullable=True),
            ],
            [
                {
                    right_key: value,
                    label_col: labels[idx % len(labels)],
                    value_col: None if idx % 4 == 0 else rnd.choice([-2, 0, 1, 3, 8]),
                }
                for idx, value in enumerate(key_values[:8])
            ],
        )
    )
    tables.append(
        TableData(
            membership_table,
            [ColumnSpec(segment_col, "str", nullable=False)],
            [{"segment_key": value} if segment_col == "segment_key" else {segment_col: value} for value in labels if value],
        )
    )
    source_string = rnd.choice(string_candidates)
    join_kind = rnd.choice(["semi_join", "anti_join"])
    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {"op": "coalesce", "columns": [label_col, source_string], "as": segment_col, "fallback": "missing"},
            {"op": "fill_null", "column": value_col, "value": 0},
            {"op": join_kind, "table": membership_table, "left_on": segment_col, "right_on": segment_col},
        ]
    )
    if join_kind == "semi_join":
        operations.extend(
            [
                {
                    "op": "groupby",
                    "keys": [segment_col],
                    "aggs": [
                        {"column": key, "func": "count", "as": "count_key"},
                        {"column": value_col, "func": "sum", "as": "sum_lookup_j"},
                    ],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": segment_col, "ascending": True, "nulls": "last"},
                        {"column": "count_key", "ascending": False, "nulls": "last"},
                    ],
                },
            ]
        )
    else:
        operations.extend(
            [
                {
                    "op": "sort",
                    "keys": [
                        {"column": segment_col, "ascending": True, "nulls": "first"},
                        {"column": value_col, "ascending": False, "nulls": "last"},
                        {"column": key, "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": [key, segment_col, value_col]},
                {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
            ]
        )
    return f"append_left_join_coalesce_membership:{join_kind}:{key}:segment={segment_col}"


def _append_left_join_case_membership(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_case_membership:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    if not key_candidates:
        return "append_left_join_case_membership:no-key-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    key_type = base_columns[key].type
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_case_membership:no-key-values:{key}"
    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_case_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    tag_col = make_safe_output_name("lookup_tag", used=used_columns)
    used_columns.add(tag_col)
    value_col = make_safe_output_name("lookup_j", used=used_columns)
    used_columns.add(value_col)
    bucket_col = make_safe_output_name("tag_bucket", used=used_columns)
    used_columns.add(bucket_col)
    labels = ["dim-a", "dim-b", "space value", None]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, key_type, nullable=False),
                ColumnSpec(tag_col, "str", nullable=True),
                ColumnSpec(value_col, "int", nullable=True),
            ],
            [
                {
                    right_key: value,
                    tag_col: labels[idx % len(labels)],
                    value_col: None if idx % 4 == 0 else rnd.choice([-2, 0, 1, 3, 8]),
                }
                for idx, value in enumerate(key_values[:8])
            ],
        )
    )
    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    sum_alias = make_safe_output_name("sum_lookup_j", used=used_columns)
    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": tag_col, "cmp": "in_set", "value": ["dim-a", "space value"]},
                "then": "interesting",
                "else": "other_or_null",
            },
            {"op": "fill_null", "column": value_col, "value": 0},
            {
                "op": "groupby",
                "keys": [bucket_col],
                "aggs": [
                    {"column": key, "func": "count", "as": count_alias},
                    {"column": value_col, "func": "sum", "as": sum_alias},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_left_join_case_membership:{key}:bucket={bucket_col}"


def _append_empty_filter_global_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_empty_filter_global_aggregate:none"
    available = _available_columns(tables, operations)
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not numeric:
        return "append_empty_filter_global_aggregate:no-numeric-column"
    source = rnd.choice(numeric)
    source_type = _column_type(tables, source)
    used_columns = set(available)
    operations.append(
        {
            "op": "filter",
            "column": source,
            "cmp": ">",
            "value": 10**12 if source_type == "int" else 1.0e12,
        }
    )
    aggs: list[dict[str, Any]] = []
    count_alias = make_safe_output_name(f"count_{source}_empty", used=used_columns)
    used_columns.add(count_alias)
    aggs.append({"column": source, "func": "count", "as": count_alias})
    if numeric:
        numeric_source = source if source in numeric else rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}_empty", used=used_columns)
        used_columns.add(sum_alias)
        mean_alias = make_safe_output_name(f"mean_{numeric_source}_empty", used=used_columns)
        used_columns.add(mean_alias)
        aggs.extend(
            [
                {"column": numeric_source, "func": "sum", "as": sum_alias},
                {"column": numeric_source, "func": "mean", "as": mean_alias},
            ]
        )
    if bools:
        bool_source = source if source in bools else rnd.choice(bools)
        any_alias = make_safe_output_name(f"any_{bool_source}_empty", used=used_columns)
        used_columns.add(any_alias)
        all_alias = make_safe_output_name(f"all_{bool_source}_empty", used=used_columns)
        aggs.extend(
            [
                {"column": bool_source, "func": "any", "as": any_alias},
                {"column": bool_source, "func": "all", "as": all_alias},
            ]
        )
    operations.append({"op": "aggregate", "aggs": aggs})
    return f"append_empty_filter_global_aggregate:{source}:{','.join(agg['as'] for agg in aggs)}"


def _append_sql_union_coalesce_distinct_topk(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_sql_union_coalesce_distinct_topk:none"
    available = _available_columns(tables, operations)
    strings = [column for column in available if _column_type(tables, column) == "str"]
    if not strings:
        return "append_sql_union_coalesce_distinct_topk:no-string-column"
    table_name = make_safe_output_name("t_union_distinct_mut", used={table.name for table in tables})
    columns = [ColumnSpec(column, _column_type(tables, column), nullable=True) for column in available]
    row_count = rnd.randint(2, 5)
    rows = []
    for row_idx in range(row_count):
        row: dict[str, Any] = {}
        for column in columns:
            row[column.name] = None if row_idx % 4 == 0 else _literal_for_type(column.type, rnd)
        rows.append(row)
    tables.append(TableData(table_name, columns, rows))

    used_columns = set(available)
    source = rnd.choice(strings)
    nonempty_col = make_safe_output_name(f"{source}_nonempty", used=used_columns)
    used_columns.add(nonempty_col)
    label_col = nonempty_col
    operations.extend(
        [
            {"op": "union_all", "table": table_name},
            {"op": "mutate", "column": nonempty_col, "expr": {"kind": "string_null_if_empty", "source": source}},
        ]
    )
    coalesce_sources = [column for column in strings if column != source]
    if coalesce_sources:
        label_col = make_safe_output_name(f"{source}_label", used=used_columns)
        used_columns.add(label_col)
        operations.append(
            {
                "op": "coalesce",
                "columns": [nonempty_col, rnd.choice(coalesce_sources)],
                "as": label_col,
                "fallback": "missing",
            }
        )
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    distinct_cols = unique_preserve_order([label_col, *(bools + numeric)[:2]])
    operations.extend(
        [
            {"op": "distinct", "columns": distinct_cols},
            {
                "op": "sort",
                "keys": [
                    {"column": distinct_cols[0], "ascending": True, "nulls": "first"},
                    *[
                        {
                            "column": column,
                            "ascending": False,
                            "nulls": "first" if idx == 0 else "last",
                        }
                        for idx, column in enumerate(distinct_cols[1:])
                    ],
                ],
            },
            {"op": "offset", "n": rnd.randint(0, 2)},
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + row_count + 2, 8)))},
        ]
    )
    return f"append_sql_union_coalesce_distinct_topk:{table_name}:{source}:label={label_col}"


def _append_boolean_membership_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_membership_case_aggregate:none"
    available = _available_columns(tables, operations)
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not bools:
        return "append_boolean_membership_case_aggregate:no-bool-column"
    source = "flag" if "flag" in bools else rnd.choice(bools)
    present_values = unique_preserve_order(
        bool(row.get(source))
        for table in tables
        for row in table.rows
        if source in row and isinstance(row.get(source), bool)
    )
    if not present_values:
        present_values = [True]
    selected_value = rnd.choice(present_values)
    table_name = make_safe_output_name("t_bool_membership_mut", used={table.name for table in tables})
    right_key = "flag_key"
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(right_key, "bool", nullable=False)],
            [{right_key: selected_value}],
        )
    )

    used_columns = set(available)
    bucket_col = make_safe_output_name(f"{source}_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": source, "func": "count", "as": count_alias},
        {"column": source, "func": "any", "as": any_alias},
        {"column": source, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "filter", "column": source, "cmp": "bool_is_not_unknown", "value": None},
            {"op": "semi_join", "table": table_name, "left_on": source, "right_on": right_key},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": source, "cmp": "bool_is_true", "value": None},
                "then": "true_member",
                "else": "false_member",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_boolean_membership_case_aggregate:{source}:member={selected_value}:bucket={bucket_col}"


def _append_left_join_boolean_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_boolean_case_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not key_candidates or not bools:
        return "append_left_join_boolean_case_aggregate:no-key-or-bool-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    bool_source = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_boolean_case_aggregate:no-key-values:{key}"

    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_bool_case_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    tag_col = make_safe_output_name("lookup_tag", used=used_columns)
    used_columns.add(tag_col)
    missing_col = make_safe_output_name("dim_missing", used=used_columns)
    used_columns.add(missing_col)
    selected_values = key_values[: max(0, len(key_values) - 1)]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, base_columns[key].type, nullable=False),
                ColumnSpec(tag_col, "str", nullable=True),
            ],
            [
                {
                    right_key: value,
                    tag_col: None if idx % 3 == 0 else rnd.choice(["matched", "space value"]),
                }
                for idx, value in enumerate(selected_values)
            ],
        )
    )

    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{bool_source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{bool_source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": key, "func": "count", "as": count_alias},
        {"column": bool_source, "func": "any", "as": any_alias},
        {"column": bool_source, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"} and column != key]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {
                "op": "case_when",
                "as": missing_col,
                "condition": {"column": tag_col, "cmp": "is_null", "value": None},
                "then": True,
                "else": False,
            },
            {"op": "groupby", "keys": [missing_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": missing_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_left_join_boolean_case_aggregate:{key}:bool={bool_source}:missing={missing_col}"


def _append_left_join_boolean_coalesce_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_boolean_coalesce_case_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not key_candidates or not bools:
        return "append_left_join_boolean_coalesce_case_aggregate:no-key-or-bool-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    fallback_bool = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_boolean_coalesce_case_aggregate:no-key-values:{key}"

    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_bool_coalesce_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    dim_flag_col = make_safe_output_name("dim_flag", used=used_columns)
    used_columns.add(dim_flag_col)
    effective_col = make_safe_output_name("flag_effective", used=used_columns)
    used_columns.add(effective_col)
    bucket_col = make_safe_output_name("flag_bucket", used=used_columns)
    used_columns.add(bucket_col)
    selected_values = key_values[: max(0, len(key_values) - 1)]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, base_columns[key].type, nullable=False),
                ColumnSpec(dim_flag_col, "bool", nullable=True),
            ],
            [
                {
                    right_key: value,
                    dim_flag_col: None if idx % 3 == 0 else rnd.choice([True, False]),
                }
                for idx, value in enumerate(selected_values)
            ],
        )
    )

    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{effective_col}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{effective_col}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": key, "func": "count", "as": count_alias},
        {"column": effective_col, "func": "any", "as": any_alias},
        {"column": effective_col, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"} and column != key]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {"op": "coalesce", "columns": [dim_flag_col, fallback_bool], "as": effective_col, "fallback": False},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": effective_col, "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false_or_missing",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_left_join_boolean_coalesce_case_aggregate:{key}:fallback={fallback_bool}:effective={effective_col}"


def _append_boolean_antijoin_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_antijoin_case_aggregate:none"
    available = _available_columns(tables, operations)
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not bools:
        return "append_boolean_antijoin_case_aggregate:no-bool-column"
    source = "flag" if "flag" in bools else rnd.choice(bools)
    present_values = unique_preserve_order(
        bool(row.get(source))
        for table in tables
        for row in table.rows
        if source in row and isinstance(row.get(source), bool)
    )
    if not present_values:
        present_values = [True]
    excluded_value = rnd.choice(present_values)
    table_name = make_safe_output_name("t_bool_antijoin_mut", used={table.name for table in tables})
    right_key = "flag_key"
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(right_key, "bool", nullable=False)],
            [{right_key: excluded_value}],
        )
    )

    used_columns = set(available)
    bucket_col = make_safe_output_name(f"{source}_anti_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": source, "func": "count", "as": count_alias},
        {"column": source, "func": "any", "as": any_alias},
        {"column": source, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "filter", "column": source, "cmp": "bool_is_not_unknown", "value": None},
            {"op": "anti_join", "table": table_name, "left_on": source, "right_on": right_key},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": source, "cmp": "bool_is_true", "value": None},
                "then": "true_non_member",
                "else": "false_non_member",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_boolean_antijoin_case_aggregate:{source}:excluded={excluded_value}:bucket={bucket_col}"


def _append_left_join_boolean_coalesce_filter_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_left_join_boolean_coalesce_filter_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    key_candidates = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "str"}
    ]
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not key_candidates or not bools:
        return "append_left_join_boolean_coalesce_filter_aggregate:no-key-or-bool-column"
    key = "id" if "id" in key_candidates else rnd.choice(key_candidates)
    fallback_bool = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(row.get(key) for row in tables[0].rows if row.get(key) is not None)
    if not key_values:
        return f"append_left_join_boolean_coalesce_filter_aggregate:no-key-values:{key}"

    used_columns = set(available)
    lookup_table = make_safe_output_name("t_left_join_bool_filter_mut", used={table.name for table in tables})
    right_key = "lookup_key"
    dim_flag_col = make_safe_output_name("dim_flag", used=used_columns)
    used_columns.add(dim_flag_col)
    effective_col = make_safe_output_name("flag_effective", used=used_columns)
    used_columns.add(effective_col)
    bucket_col = make_safe_output_name("flag_filter_bucket", used=used_columns)
    used_columns.add(bucket_col)
    selected_values = key_values[: max(0, len(key_values) - 1)]
    tables.append(
        TableData(
            lookup_table,
            [
                ColumnSpec(right_key, base_columns[key].type, nullable=False),
                ColumnSpec(dim_flag_col, "bool", nullable=True),
            ],
            [
                {
                    right_key: value,
                    dim_flag_col: None if idx % 4 == 0 else rnd.choice([True, False]),
                }
                for idx, value in enumerate(selected_values)
            ],
        )
    )

    filter_value = rnd.choice([True, False])
    count_alias = make_safe_output_name("count_key", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{effective_col}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{effective_col}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": key, "func": "count", "as": count_alias},
        {"column": effective_col, "func": "any", "as": any_alias},
        {"column": effective_col, "func": "all", "as": all_alias},
    ]
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"} and column != key]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    operations.extend(
        [
            {"op": "join", "table": lookup_table, "left_on": key, "right_on": right_key, "how": "left"},
            {"op": "coalesce", "columns": [dim_flag_col, fallback_bool], "as": effective_col, "fallback": False},
            {"op": "filter", "column": effective_col, "cmp": "==", "value": filter_value},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": effective_col, "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return (
        "append_left_join_boolean_coalesce_filter_aggregate:"
        f"{key}:fallback={fallback_bool}:filter={filter_value}:effective={effective_col}"
    )


def _append_numeric_text_boolean_antijoin_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_numeric_text_boolean_antijoin_case_aggregate:none"
    available = _available_columns(tables, operations)
    bools = [column for column in available if _column_type(tables, column) == "bool"]
    if not bools:
        return "append_numeric_text_boolean_antijoin_case_aggregate:no-bool-column"
    used_columns = set(available)
    numeric_text = _ensure_numeric_string_column(tables, available, rnd, used_columns)
    if numeric_text is None:
        return "append_numeric_text_boolean_antijoin_case_aggregate:no-numeric-string-column"
    if numeric_text not in available:
        available.append(numeric_text)
    used_columns.add(numeric_text)

    bool_source = "flag" if "flag" in bools else rnd.choice(bools)
    num_col = make_safe_output_name("num_value", used=used_columns)
    used_columns.add(num_col)
    table_name = make_safe_output_name("t_numeric_antijoin_mut", used={table.name for table in tables})
    parsed_values = []
    for row in tables[0].rows:
        value = row.get(numeric_text)
        if value is None:
            continue
        try:
            parsed_values.append(int(value))
        except (TypeError, ValueError):
            continue
    parsed_values = unique_preserve_order(parsed_values)
    if not parsed_values:
        parsed_values = [0]
    excluded_values = parsed_values[: max(1, min(len(parsed_values), 3))]
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(num_col, "int", nullable=False)],
            [{num_col: value} for value in excluded_values],
        )
    )

    bucket_col = make_safe_output_name(f"{bool_source}_num_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    sum_alias = make_safe_output_name(f"sum_{num_col}", used=used_columns)
    used_columns.add(sum_alias)
    any_alias = make_safe_output_name(f"any_{bool_source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{bool_source}", used=used_columns)

    operations.extend(
        [
            {
                "op": "mutate",
                "column": num_col,
                "expr": {"kind": "cast", "source": numeric_text, "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": num_col, "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": bool_source, "cmp": "bool_is_not_false", "value": None},
            {"op": "anti_join", "table": table_name, "left_on": num_col, "right_on": num_col},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": bool_source, "cmp": "bool_is_true", "value": None},
                "then": "true_unmatched_number",
                "else": "null_or_false_unmatched_number",
            },
            {
                "op": "groupby",
                "keys": [bucket_col],
                "aggs": [
                    {"column": bool_source, "func": "count", "as": count_alias},
                    {"column": num_col, "func": "sum", "as": sum_alias},
                    {"column": bool_source, "func": "any", "as": any_alias},
                    {"column": bool_source, "func": "all", "as": all_alias},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return (
        "append_numeric_text_boolean_antijoin_case_aggregate:"
        f"{numeric_text}:bool={bool_source}:excluded={excluded_values}"
    )


def _append_multi_key_membership_case_aggregate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_multi_key_membership_case_aggregate:none"
    available = _available_columns(tables, operations)
    base_columns = {column.name: column for column in tables[0].columns}
    int_keys = [
        column
        for column in available
        if column in base_columns and base_columns[column].type == "int"
    ]
    string_keys = [
        column
        for column in available
        if column in base_columns and base_columns[column].type == "str"
    ]
    bools = [
        column
        for column in available
        if column in base_columns and base_columns[column].type == "bool"
    ]
    if not int_keys or not string_keys or not bools:
        return "append_multi_key_membership_case_aggregate:no-key-or-bool-column"

    left_keys = ["id" if "id" in int_keys else rnd.choice(int_keys), "g" if "g" in string_keys else rnd.choice(string_keys)]
    if left_keys[0] == left_keys[1]:
        return "append_multi_key_membership_case_aggregate:duplicate-keys"
    bool_source = "flag" if "flag" in bools else rnd.choice(bools)
    key_values = unique_preserve_order(
        tuple(row.get(column) for column in left_keys)
        for row in tables[0].rows
        if all(row.get(column) is not None for column in left_keys)
    )
    if len(key_values) < 2:
        return "append_multi_key_membership_case_aggregate:not-enough-key-values"

    table_name = make_safe_output_name("t_multi_key_membership_mut", used={table.name for table in tables})
    selected_values = key_values[: max(1, len(key_values) // 2)]
    tables.append(
        TableData(
            table_name,
            [ColumnSpec(column, base_columns[column].type, nullable=False) for column in left_keys],
            [dict(zip(left_keys, values)) for values in selected_values],
        )
    )

    used_columns = set(available)
    bucket_col = make_safe_output_name(f"{bool_source}_multi_key_bucket", used=used_columns)
    used_columns.add(bucket_col)
    count_alias = make_safe_output_name("count_rows", used=used_columns)
    used_columns.add(count_alias)
    any_alias = make_safe_output_name(f"any_{bool_source}", used=used_columns)
    used_columns.add(any_alias)
    all_alias = make_safe_output_name(f"all_{bool_source}", used=used_columns)
    used_columns.add(all_alias)
    aggs: list[dict[str, Any]] = [
        {"column": left_keys[0], "func": "count", "as": count_alias},
        {"column": bool_source, "func": "any", "as": any_alias},
        {"column": bool_source, "func": "all", "as": all_alias},
    ]
    numeric = [
        column
        for column in available
        if column in base_columns and base_columns[column].type in {"int", "float"} and column not in left_keys
    ]
    if numeric:
        numeric_source = rnd.choice(numeric)
        sum_alias = make_safe_output_name(f"sum_{numeric_source}", used=used_columns)
        aggs.append({"column": numeric_source, "func": "sum", "as": sum_alias})

    join_kind = rnd.choice(["semi_join", "anti_join"])
    operations.extend(
        [
            {"op": "filter", "column": left_keys[1], "cmp": "is_not_null", "value": None},
            {"op": join_kind, "table": table_name, "left_on": left_keys, "right_on": left_keys},
            {
                "op": "case_when",
                "as": bucket_col,
                "condition": {"column": bool_source, "cmp": "bool_is_true", "value": None},
                "then": f"{join_kind}_true",
                "else": f"{join_kind}_false_or_null",
            },
            {"op": "groupby", "keys": [bucket_col], "aggs": aggs},
            {
                "op": "sort",
                "keys": [
                    {"column": bucket_col, "ascending": True, "nulls": "last"},
                    {"column": count_alias, "ascending": False, "nulls": "last"},
                ],
            },
        ]
    )
    return f"append_multi_key_membership_case_aggregate:{join_kind}:{','.join(left_keys)}"


def _random_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> dict[str, Any] | None:
    table = tables[0]
    available = _available_columns(tables, operations)
    if not available:
        return None
    numeric = [
        c
        for c in available
        if _column_type(tables, c) in {"int", "float"}
        or c.startswith(("m_", "sum_", "min_", "max_", "count_", "nunique_", "uniq_"))
    ]
    bools = [
        c
        for c in available
        if _column_type(tables, c) == "bool" or c.startswith(("any_", "all_"))
    ]
    strings = [c for c in available if _column_type(tables, c) == "str"]
    date_strings = [c for c in strings if _looks_like_date_column(c)]
    numeric_strings = [c for c in strings if _looks_like_numeric_string_column(c)]
    choices = ["filter", "select", "sort", "limit"]
    if numeric or strings:
        choices.append("mutate")
    if numeric or bools:
        choices.append("groupby")
        choices.append("aggregate")
    kind = rnd.choice(choices)
    if kind == "filter":
        col = rnd.choice(available)
        typ = _column_type(tables, col)
        cmp = rnd.choice(["==", "!="] if typ in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="])
        if typ in {"int", "float"} and rnd.random() < 0.20:
            cmp = rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"])
        if rnd.random() < 0.15:
            cmp = rnd.choice(["in_set", "not_in_set"])
        if rnd.random() < 0.12:
            cmp = rnd.choice(["is_null", "is_not_null"])
        if typ == "bool" and rnd.random() < 0.25:
            cmp = rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"])
        if typ in {"int", "float"} and rnd.random() < 0.18:
            cmp = "range_closed"
        if cmp in {"in_set", "not_in_set"}:
            value = _literal_list_for_type(typ, rnd)
        elif cmp == "range_closed":
            value = sorted(rnd.sample(_literal_list_for_type(typ, rnd), 2))
        elif cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
            value = None
        else:
            value = _literal_for_type(typ, rnd)
        return {"op": "filter", "column": col, "cmp": cmp, "value": value}
    if kind == "select":
        count = rnd.randint(1, len(available))
        return {"op": "select", "columns": sorted(rnd.sample(available, count))}
    if kind == "sort":
        first = rnd.choice(available)
        cols = [first] + sorted(c for c in available if c != first)
        if len(cols) > 1 and rnd.random() < 0.35:
            return {
                "op": "sort",
                "keys": [
                    {
                        "column": column,
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                    for column in cols
                ],
            }
        return {"op": "sort", "columns": cols, "ascending": rnd.choice([True, False])}
    if kind == "limit":
        return {"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 3))}
    if kind == "mutate" and (numeric or strings):
        if numeric_strings and rnd.random() < 0.20:
            src = rnd.choice(numeric_strings)
            expr = {
                "kind": "cast",
                "source": src,
                "to": rnd.choice(["int", "float"]),
                "input_domain": "integer_string",
            }
        elif date_strings and rnd.random() < 0.25:
            src = rnd.choice(date_strings)
            expr = {"kind": "date_part", "source": src, "part": rnd.choice(["year", "month", "day"])}
        elif strings and (not numeric or rnd.random() < 0.3):
            src = rnd.choice(strings)
            string_exprs = [
                {"kind": "string_length", "source": src},
                {"kind": "string_lower", "source": src},
                {"kind": "string_upper", "source": src},
                {"kind": "string_strip", "source": src},
                {"kind": "string_null_if_empty", "source": src},
                {"kind": "string_replace", "source": src, "old": " ", "new": "_"},
                {"kind": "string_slice", "source": src, "start": 0, "length": rnd.randint(1, 3)},
                {"kind": "string_split_part", "source": src, "sep": rnd.choice([" ", "-", "_"]), "index": 0},
                {"kind": "string_basename", "source": src},
                {"kind": "string_contains", "source": src, "needle": rnd.choice(["a", "A", "space", "pad"])},
                {"kind": "string_starts_with", "source": src, "needle": rnd.choice(["a", "A", "space", "pad"])},
                {"kind": "string_ends_with", "source": src, "needle": rnd.choice(["a", "A", "e", "d"])},
            ]
            if len(strings) > 1:
                other = rnd.choice([column for column in strings if column != src])
                string_exprs.append({"kind": "string_concat", "source": src, "other": other, "sep": "-"})
            expr = rnd.choice(string_exprs)
        else:
            src = rnd.choice(numeric)
            if len(numeric) > 1 and rnd.random() < 0.20:
                numerator = rnd.choice([column for column in numeric if column != src])
                expr = {"kind": "reverse_division_columns", "source": src, "numerator": numerator}
            elif rnd.random() < 0.25:
                if _column_type(tables, src) == "int":
                    expr = {"kind": "cast", "source": src, "to": rnd.choice(["float", "str"])}
                else:
                    expr = {"kind": "cast", "source": src, "to": "float"}
            else:
                expr = {"kind": "add_const", "source": src, "value": rnd.choice([-10, -1, 0, 1, 10])}
        return {
            "op": "mutate",
            "column": f"m_{operation_count(operations, 'mutate')}",
            "expr": expr,
        }
    if kind == "groupby" and (numeric or bools):
        keys = [rnd.choice(available)]
        val = rnd.choice(numeric + bools)
        if val in bools:
            func = rnd.choice(["any", "all", "min", "max", "count", "nunique"])
        else:
            func = rnd.choice(["sum", "min", "max", "count", "nunique"])
        alias = make_safe_output_name(f"{func}_{val}", used=set(keys))
        return {"op": "groupby", "keys": keys, "aggs": [{"column": val, "func": func, "as": alias}]}
    if kind == "aggregate" and (numeric or bools):
        val = rnd.choice(numeric + bools)
        if val in bools:
            func = rnd.choice(["any", "all", "min", "max", "count", "nunique"])
        else:
            func = rnd.choice(["sum", "min", "max", "count", "nunique"])
        alias = make_safe_output_name(f"{func}_{val}_all")
        return {"op": "aggregate", "aggs": [{"column": val, "func": func, "as": alias}]}
    return None


def _tweak_operation(tables: list[TableData], op: dict[str, Any], rnd: random.Random) -> str:
    kind = op_kind(op)
    if kind == "filter":
        op["cmp"] = rnd.choice([">", ">=", "<", "<=", "==", "!="])
        op["value"] = _literal_for_type(_column_type(tables, op["column"]), rnd)
        return "tweak:filter"
    elif kind == "sort":
        if "keys" in op:
            keys = [key.to_dict() for key in normalize_sort_keys(op)]
            if not keys:
                return "tweak:sort:no-keys"
            key = keys[rnd.randrange(len(keys))]
            if rnd.random() < 0.5:
                key["ascending"] = not bool(key.get("ascending", True))
            else:
                key["nulls"] = "first" if key.get("nulls", "last") == "last" else "last"
            op["keys"] = keys
        else:
            op["ascending"] = not op_ascending(op, True)
        return "tweak:sort"
    elif kind == "limit":
        op["n"] = max(0, op_n(op, 0) + rnd.choice([-2, -1, 1, 2]))
        return "tweak:limit"
    elif kind == "mutate":
        if "value" in op["expr"]:
            op["expr"]["value"] = op["expr"].get("value", 0) + rnd.choice([-2, -1, 1, 2])
            return "tweak:mutate"
    return f"tweak:{kind or 'unknown'}:noop"


def _looks_like_date_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"dt", "date", "event_date", "timestamp"} or lowered.endswith(("_dt", "_date"))


def _looks_like_numeric_string_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"num_s", "number_text", "numeric_text"} or lowered.endswith(("_num_s", "_number_text", "_numeric_text"))


def _ensure_numeric_string_column(
    tables: list[TableData],
    available: list[str],
    rnd: random.Random,
    used_columns: set[str],
) -> str | None:
    numeric_strings = [
        column
        for column in available
        if _column_type(tables, column) == "str" and _looks_like_numeric_string_column(column)
        and _column_has_only_integer_strings(tables, column)
    ]
    if numeric_strings:
        return rnd.choice(numeric_strings)
    base_ints = [column.name for column in tables[0].columns if column.type == "int"]
    if not base_ints:
        return None
    source = "id" if "id" in base_ints else rnd.choice(base_ints)
    column = make_safe_output_name(f"{source}_num_s", used=used_columns)
    tables[0].columns.append(ColumnSpec(column, "str", nullable=True))
    for idx, row in enumerate(tables[0].rows):
        value = row.get(source)
        if value is None or idx % 5 == 0:
            row[column] = None
        else:
            row[column] = str(int(value))
    return column


def _column_has_only_integer_strings(tables: list[TableData], column: str) -> bool:
    seen = False
    for table in tables:
        for row in table.rows:
            if column not in row:
                continue
            value = row.get(column)
            if value is None:
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                return False
            seen = True
    return seen


def _available_columns(tables: list[TableData], operations: list[dict[str, Any]]) -> list[str]:
    return _mutation_schema(tables, operations).available


def _column_type(tables: list[TableData], name: str) -> str:
    return _mutation_schema(tables, []).column_type(name)


def _column_has_null(tables: list[TableData], name: str) -> bool:
    return _mutation_schema(tables, []).column_has_null(name)


def _literal_for_type(typ: str, rnd: random.Random) -> Any:
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _literal_list_for_type(typ: str, rnd: random.Random) -> list[Any]:
    if typ == "int":
        return rnd.sample([-10, -1, 0, 1, 2, 10], k=3)
    if typ == "float":
        return rnd.sample([-1.0, 0.0, 0.5, 1.0, 10.0], k=3)
    if typ == "bool":
        return rnd.sample([True, False], k=rnd.randint(1, 2))
    return rnd.sample(["", "alpha", "beta", "中文", "missing"], k=3)


MUTATION_OPERATORS: tuple[MutationOperator, ...] = (
    MutationOperator("value", _mutate_scalar_value),
    MutationOperator("nullify_value", _nullify_value),
    MutationOperator("duplicate_row", _duplicate_row),
    MutationOperator("drop_row", _drop_row),
    MutationOperator("shuffle_rows", _shuffle_rows),
    MutationOperator("append_op", _append_operation),
    MutationOperator("append_order_projection", _append_order_projection_probe),
    MutationOperator("append_truth_filter", _append_truth_filter_probe),
    MutationOperator("append_boolean_predicate_filter", _append_boolean_predicate_filter_probe),
    MutationOperator("append_range_filter", _append_range_filter_probe),
    MutationOperator("append_tuple_absence_filter", _append_tuple_absence_filter_probe),
    MutationOperator(
        "append_row_value_absence_filter",
        _append_row_value_absence_filter,
        semantic_family_affinity=("null_semantics", "join_membership"),
        semantic_signal_affinity=("row_value_absence_filter",),
    ),
    MutationOperator("append_running_sum", _append_running_sum_probe),
    MutationOperator("append_sortedness_check", _append_sortedness_check_probe),
    MutationOperator("append_random_case_probe", _append_random_case_probe),
    MutationOperator("append_group_quantile_probe", _append_group_quantile_probe),
    MutationOperator("append_scalar_subquery_probe", _append_scalar_subquery_probe),
    MutationOperator("append_window_avg_probe", _append_window_avg_probe),
    MutationOperator("append_struct_distinct_probe", _append_struct_distinct_probe),
    MutationOperator("append_bit_compare_probe", _append_bit_compare_probe),
    MutationOperator("append_round_even_probe", _append_round_even_probe),
    MutationOperator("append_series_rtruediv_probe", _append_series_rtruediv_probe),
    MutationOperator("append_uint64_isin_probe", _append_uint64_isin_probe),
    MutationOperator("append_tuple_anti_null_probe", _append_tuple_anti_null_probe),
    MutationOperator("append_setop_all_duplicate_probe", _append_setop_all_duplicate_probe),
    MutationOperator("append_json_predicate_order_probe", _append_json_predicate_order_probe),
    MutationOperator("append_sparse_mask_probe", _append_sparse_mask_probe),
    MutationOperator("append_float_wrap_probe", _append_float_wrap_probe),
    MutationOperator("append_index_bool_probe", _append_index_bool_probe),
    MutationOperator("append_empty_literal_groupby_probe", _append_empty_literal_groupby_probe),
    MutationOperator("append_arrow_string_eq_sum_probe", _append_arrow_string_eq_sum_probe),
    MutationOperator("append_arrow_timestamp_loc_slice_probe", _append_arrow_timestamp_loc_slice_probe),
    MutationOperator("append_arrow_timestamp_index_attr_probe", _append_arrow_timestamp_index_attr_probe),
    MutationOperator("append_eval_inplace_alias_probe", _append_eval_inplace_alias_probe),
    MutationOperator("append_bool_reduction_skipna_probe", _append_bool_reduction_skipna_probe),
    MutationOperator("append_dataset_isin_all_match_probe", _append_dataset_isin_all_match_probe),
    MutationOperator("append_run_end_null_compute_probe", _append_run_end_null_compute_probe),
    MutationOperator("append_large_string_partition_probe", _append_large_string_partition_probe),
    MutationOperator("append_hash_pivot_wider_probe", _append_hash_pivot_wider_probe),
    MutationOperator("append_list_flatten_parent_indices_probe", _append_list_flatten_parent_indices_probe),
    MutationOperator("append_rolling_mean_by_null_count_probe", _append_rolling_mean_by_null_count_probe),
    MutationOperator("append_grouped_topk", _append_grouped_topk_probe),
    MutationOperator(
        "append_groupby_fractional_membership_filter",
        _append_groupby_fractional_membership_filter,
        semantic_family_affinity=("aggregation_cardinality", "join_membership", "type_coercion"),
        semantic_signal_affinity=("distinct_count_aggregation",),
    ),
    MutationOperator(
        "append_normalized_string_membership",
        _append_normalized_string_membership,
        semantic_family_affinity=("join_membership", "string_semantics"),
        semantic_signal_affinity=("normalized_string_membership_key", "chained_string_normalized_membership_key"),
    ),
    MutationOperator(
        "append_sql_distinct_null_topk",
        _append_sql_distinct_null_topk,
        semantic_family_affinity=("set_semantics", "topk_ordering", "null_semantics", "string_semantics"),
        semantic_signal_affinity=("sql_distinct_null_topk", "coalesced_distinct_topk"),
    ),
    MutationOperator(
        "append_left_join_coalesce_membership",
        _append_left_join_coalesce_membership,
        semantic_family_affinity=("join_membership", "null_semantics", "string_semantics"),
        semantic_signal_affinity=("left_join_coalesce_membership", "join_coalesce_membership"),
    ),
    MutationOperator(
        "append_left_join_case_membership",
        _append_left_join_case_membership,
        semantic_family_affinity=("join_membership", "conditional_semantics", "string_semantics"),
        semantic_signal_affinity=("left_join_case_when_membership", "join_case_when_membership"),
    ),
    MutationOperator(
        "append_empty_filter_global_aggregate",
        _append_empty_filter_global_aggregate,
        semantic_family_affinity=("aggregation_cardinality", "null_semantics"),
        semantic_signal_affinity=("filtered_global_aggregation",),
    ),
    MutationOperator(
        "append_sql_union_coalesce_distinct_topk",
        _append_sql_union_coalesce_distinct_topk,
        semantic_family_affinity=("set_semantics", "topk_ordering", "null_semantics"),
        semantic_signal_affinity=("union_coalesce_distinct_topk",),
    ),
    MutationOperator(
        "append_boolean_membership_case_aggregate",
        _append_boolean_membership_case_aggregate,
        semantic_family_affinity=(
            "boolean_logic",
            "join_membership",
            "conditional_semantics",
            "aggregation_cardinality",
        ),
        semantic_signal_affinity=("boolean_membership_case_aggregation", "boolean_case_when_predicate"),
    ),
    MutationOperator(
        "append_left_join_boolean_case_aggregate",
        _append_left_join_boolean_case_aggregate,
        semantic_family_affinity=(
            "boolean_logic",
            "join_membership",
            "conditional_semantics",
            "aggregation_cardinality",
        ),
        semantic_signal_affinity=("left_join_boolean_case_aggregation", "boolean_case_when_aggregation"),
    ),
    MutationOperator(
        "append_left_join_boolean_coalesce_case_aggregate",
        _append_left_join_boolean_coalesce_case_aggregate,
        semantic_family_affinity=("boolean_logic", "join_membership", "conditional_semantics", "null_semantics"),
        semantic_signal_affinity=("left_join_boolean_coalesce_aggregation", "boolean_coalesce_case_aggregation"),
    ),
    MutationOperator(
        "append_boolean_antijoin_case_aggregate",
        _append_boolean_antijoin_case_aggregate,
        semantic_family_affinity=(
            "boolean_logic",
            "join_membership",
            "conditional_semantics",
            "aggregation_cardinality",
        ),
        semantic_signal_affinity=("boolean_membership_case_aggregation", "boolean_case_when_predicate"),
    ),
    MutationOperator(
        "append_left_join_boolean_coalesce_filter_aggregate",
        _append_left_join_boolean_coalesce_filter_aggregate,
        semantic_family_affinity=("boolean_logic", "join_membership", "null_semantics", "aggregation_cardinality"),
        semantic_signal_affinity=("left_join_boolean_coalesce_filter_aggregation", "boolean_coalesce_filter_aggregation"),
    ),
    MutationOperator(
        "append_numeric_text_boolean_antijoin_case_aggregate",
        _append_numeric_text_boolean_antijoin_case_aggregate,
        semantic_family_affinity=("type_coercion", "boolean_logic", "join_membership", "conditional_semantics"),
        semantic_signal_affinity=("numeric_text_cast_membership_aggregation", "boolean_membership_case_aggregation"),
    ),
    MutationOperator(
        "append_multi_key_membership_case_aggregate",
        _append_multi_key_membership_case_aggregate,
        semantic_family_affinity=("join_membership", "conditional_semantics", "aggregation_cardinality"),
        semantic_signal_affinity=("multi_key_membership_aggregation", "multi_key_semi_anti_join"),
    ),
    MutationOperator("drop_op", _drop_operation),
    MutationOperator("tweak_op", _tweak_random_operation),
)
PROBE_MUTATION_OPERATOR_NAMES = frozenset(
    operator.name
    for operator in MUTATION_OPERATORS
    if operator.name.endswith("_probe")
)
SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES = frozenset(
    {
        "append_grouped_topk",
        "append_running_sum",
        "append_sortedness_check",
        "append_tuple_absence_filter",
    }
)
ROOT_TARGETED_MUTATION_OPERATOR_NAMES = SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES
DISCOVERY_MUTATION_OPERATORS: tuple[MutationOperator, ...] = tuple(
    operator
    for operator in MUTATION_OPERATORS
    if operator.name not in PROBE_MUTATION_OPERATOR_NAMES
    and operator.name not in SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES
)
ALL_MUTATION_OPERATOR_PROFILES: Mapping[str, MutationOperator] = MappingProxyType(
    {operator.name: operator for operator in MUTATION_OPERATORS}
)
DISCOVERY_MUTATION_OPERATOR_PROFILES: Mapping[str, MutationOperator] = MappingProxyType(
    {operator.name: operator for operator in DISCOVERY_MUTATION_OPERATORS}
)
MUTATION_OPERATOR_NAMES = tuple(operator.name for operator in MUTATION_OPERATORS)
DISCOVERY_MUTATION_OPERATOR_NAMES = tuple(operator.name for operator in DISCOVERY_MUTATION_OPERATORS)
