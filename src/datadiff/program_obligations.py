from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datadiff.canonicalization import canonical_key
from datadiff.dsl import Case, sort_columns
from datadiff.filtering import evaluate_filter_predicate
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.operation_semantics import (
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    expr_source,
    groupby_keys,
    has_order_observer,
    op_column,
    op_columns,
    op_comparator,
    op_kind,
    op_n,
    op_table,
    op_value,
)
from datadiff.program_state import state_before_operation


PROGRAM_OBLIGATION_SCOPES = {
    "input_relation",
    "operation_sequence",
    "program_tail",
    "whole_program",
}


@dataclass(frozen=True, slots=True)
class ProgramObligationIR:
    """A metamorphic obligation whose validity depends on the whole program.

    Node contracts retain operation-local laws. This representation covers
    input rewrites, interactions between operations, and tail transformations
    without assigning them to an arbitrary relational node.
    """

    obligation_id: str
    scope: str
    transformation: str
    expected_relation: str
    preconditions: tuple[str, ...]
    oracle: str
    target_fault_models: tuple[str, ...]
    anchor_node_ids: tuple[str, ...]
    source_relation_ids: tuple[str, ...]
    rationale: str
    estimated_cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.obligation_id:
            raise ValueError("program obligation id must not be empty")
        if self.scope not in PROGRAM_OBLIGATION_SCOPES:
            raise ValueError(f"unsupported program obligation scope: {self.scope}")
        if self.estimated_cost < 0.0:
            raise ValueError("program obligation cost must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "kind": "metamorphic_relation",
            "scope": self.scope,
            "transformation": self.transformation,
            "expected_relation": self.expected_relation,
            "preconditions": list(self.preconditions),
            "oracle": self.oracle,
            "target_fault_models": list(self.target_fault_models),
            "anchor_node_ids": list(self.anchor_node_ids),
            "source_relation_ids": list(self.source_relation_ids),
            "rationale": self.rationale,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramObligationIR":
        return cls(
            obligation_id=str(data.get("obligation_id", "")),
            scope=str(data.get("scope", "whole_program")),
            transformation=str(data.get("transformation", "")),
            expected_relation=str(data.get("expected_relation", "contract_equivalent")),
            preconditions=tuple(str(item) for item in data.get("preconditions", ())),
            oracle=str(data.get("oracle", "metamorphic")),
            target_fault_models=tuple(
                str(item) for item in data.get("target_fault_models", ())
            ),
            anchor_node_ids=tuple(
                str(item) for item in data.get("anchor_node_ids", ())
            ),
            source_relation_ids=tuple(
                str(item) for item in data.get("source_relation_ids", ())
            ),
            rationale=str(data.get("rationale", "")),
            estimated_cost=float(data.get("estimated_cost", 1.0)),
        )


def infer_program_obligations(
    case: Case,
    *,
    node_ids: Sequence[str],
    source_relation_ids: Sequence[str],
) -> tuple[ProgramObligationIR, ...]:
    """Compile independently inferred whole-program metamorphic laws."""

    if len(node_ids) != len(case.program.operations):
        raise ValueError("program obligation anchors must align with operations")
    if not case.tables or not source_relation_ids:
        return ()
    return _ProgramObligationCompiler(
        case,
        node_ids=node_ids,
        source_relation_ids=source_relation_ids,
    ).compile()


class _ProgramObligationCompiler:
    def __init__(
        self,
        case: Case,
        *,
        node_ids: Sequence[str],
        source_relation_ids: Sequence[str],
    ) -> None:
        self.case = case
        self.operations = list(case.program.operations)
        self.kinds = [
            op_kind(operation, default="unknown")
            for operation in self.operations
        ]
        self.node_ids = tuple(node_ids)
        self.primary_source_ids = tuple(source_relation_ids[:1])
        self.all_source_ids = tuple(source_relation_ids)
        self.order_observed = bool(
            case.program.order_sensitive or has_order_observer(case.program)
        )
        self.declarations: list[ProgramObligationIR] = []

    def compile(self) -> tuple[ProgramObligationIR, ...]:
        self._declare_input_materialization()
        self._declare_input_invariants()
        self._declare_filter_and_join_invariants()
        self._declare_aggregation_invariants()
        self._declare_sequence_rewrites()
        self._declare_tail_rewrites()
        return tuple(self.declarations)

    def _anchors(self, indices: Sequence[int]) -> tuple[str, ...]:
        return tuple(
            self.node_ids[index]
            for index in dict.fromkeys(indices)
            if 0 <= index < len(self.node_ids)
        )

    def _declare(
        self,
        relation: str,
        *,
        scope: str,
        anchor_indices: Sequence[int] = (),
        source_relation_ids: Sequence[str] | None = None,
        rationale: str,
        fault_models: Sequence[str] = ("optimizer_equivalence",),
        cost: float = 1.0,
        secondary_relation_required: bool = False,
    ) -> None:
        if any(
            item.obligation_id == relation for item in self.declarations
        ):
            return
        preconditions = ["input_schema_valid", "program_pattern_matched"]
        if secondary_relation_required:
            preconditions.append("secondary_relation_available")
        self.declarations.append(
            ProgramObligationIR(
                obligation_id=relation,
                scope=scope,
                transformation=relation,
                expected_relation="contract_equivalent",
                preconditions=tuple(preconditions),
                oracle="metamorphic",
                target_fault_models=tuple(fault_models),
                anchor_node_ids=self._anchors(anchor_indices),
                source_relation_ids=tuple(
                    source_relation_ids
                    if source_relation_ids is not None
                    else self.primary_source_ids
                ),
                rationale=rationale,
                estimated_cost=cost,
            )
        )

    def _declare_input_materialization(self) -> None:
        first_kind = self.kinds[0] if self.kinds else ""
        first_anchor = (0,) if self.operations else ()
        if first_kind == "filter" and _filter_materialization_applicable(self.case):
            self._declare(
                "filter_input_materialization",
                scope="input_relation",
                anchor_indices=first_anchor,
                rationale="A leading filter can be materialized into the primary input.",
                fault_models=("filter_pushdown", "input_materialization"),
                cost=1.25,
            )
        cleanup_applicable = {
            "drop_nulls": _drop_nulls_materialization_applicable,
            "fill_null": _fill_null_materialization_applicable,
            "distinct": _distinct_materialization_applicable,
        }
        if first_kind in cleanup_applicable and cleanup_applicable[first_kind](self.case):
            self._declare(
                f"{first_kind}_input_materialization",
                scope="input_relation",
                anchor_indices=first_anchor,
                rationale="A leading cleanup operation can be materialized into its input.",
                fault_models=("null_semantics", "input_materialization"),
                cost=1.25,
            )

    def _declare_input_invariants(self) -> None:
        primary = self.case.tables[0]
        if not self.order_observed and len(primary.rows) >= 2:
            self._declare(
                "row_permutation",
                scope="input_relation",
                rationale="Bag semantics permit permutation of primary input rows.",
                fault_models=("ordering_stability", "bag_semantics"),
                cost=1.5,
            )
            if all(
                table.name != "t_partition_tail"
                for table in self.case.tables
            ):
                self._declare(
                    "input_partition_union_all",
                    scope="input_relation",
                    rationale="Partitioning and reassembling the primary input preserves its bag.",
                    fault_models=("partitioning", "union_all_semantics"),
                    cost=1.5,
                )
        if all(table.name != "t_empty_union" for table in self.case.tables):
            self._declare(
                "union_all_empty_append",
                scope="input_relation",
                rationale="Union with an empty relation is the identity transformation.",
                fault_models=("union_all_semantics", "empty_input_handling"),
            )
        if not self.order_observed and any(
            column.type in {"int", "float"} for column in primary.columns
        ):
            self._declare(
                "mutate_add_zero_insertion",
                scope="whole_program",
                rationale="Adding zero to a numeric input column is a neutral prefix rewrite.",
                fault_models=("numeric_precision", "optimizer_equivalence"),
            )
        if _has_lower_normalized_string_column(self.case):
            self._declare(
                "string_lower_normalized_column",
                scope="whole_program",
                rationale="Lowercasing an already normalized string column is a no-op.",
                fault_models=("string_semantics", "input_normalization"),
            )
        if not self.order_observed and _has_integral_id_domain(self.case):
            self._declare(
                "filter_tautology_insertion",
                scope="whole_program",
                rationale="A predicate bounded by the observed integer id domain is tautological.",
                fault_models=("predicate_simplification", "filter_pushdown"),
            )

    def _declare_filter_and_join_invariants(self) -> None:
        filter_indices = self._indices("filter")
        if (
            filter_indices
            and not self.order_observed
            and _filter_rejecting_row_applicable(self.case)
        ):
            self._declare(
                "filter_rejecting_row_injection",
                scope="input_relation",
                anchor_indices=filter_indices,
                rationale="Rows rejected by an existing filter must not affect the result.",
                fault_models=("predicate_semantics", "input_cardinality"),
                cost=1.25,
            )

        join_indices = self._indices("join")
        unmatched_join_index = _join_unmatched_dimension_index(self.case)
        if unmatched_join_index is not None:
            self._declare(
                "join_unmatched_dimension_injection",
                scope="input_relation",
                anchor_indices=(unmatched_join_index,),
                source_relation_ids=self.all_source_ids,
                rationale="An unmatched dimension row must not change an eligible join result.",
                fault_models=("join_cardinality", "join_null_semantics"),
                cost=1.5,
                secondary_relation_required=True,
            )
        equivalent_join_index = _join_inner_left_equivalence_index(self.case)
        if equivalent_join_index is not None:
            self._declare(
                "join_inner_left_equivalence",
                scope="whole_program",
                anchor_indices=(equivalent_join_index,),
                source_relation_ids=self.all_source_ids,
                rationale="Covered non-null join keys make inner and left join equivalent.",
                fault_models=("join_cardinality", "join_null_semantics"),
                cost=1.5,
                secondary_relation_required=True,
            )
        if not self.order_observed and any(
            len(table.rows) >= 2 for table in self.case.tables[1:]
        ):
            self._declare(
                "join_table_permutation",
                scope="input_relation",
                anchor_indices=join_indices,
                source_relation_ids=self.all_source_ids,
                rationale="Bag semantics permit permutation of secondary input rows.",
                fault_models=("join_ordering", "bag_semantics"),
                cost=1.5,
                secondary_relation_required=True,
            )

        semi_anti_indices = self._indices("semi_join", "anti_join")
        for relation, scope, rationale in (
            (
                "semi_anti_join_rewrite",
                "operation_sequence",
                "Existence joins can be lowered to an equivalent materialized-key plan.",
            ),
            (
                "semi_anti_join_right_duplicate_injection",
                "input_relation",
                "Right-side duplicates do not alter existence-join membership.",
            ),
            (
                "semi_anti_join_unmatched_right_injection",
                "input_relation",
                "An unmatched right row does not alter existence-join membership.",
            ),
        ):
            if semi_anti_indices and (
                relation != "semi_anti_join_right_duplicate_injection"
                or _semi_anti_duplicate_applicable(self.case)
            ):
                self._declare(
                    relation,
                    scope=scope,
                    anchor_indices=semi_anti_indices,
                    source_relation_ids=self.all_source_ids,
                    rationale=rationale,
                    fault_models=("join_cardinality", "join_null_semantics"),
                    cost=1.5,
                    secondary_relation_required=True,
                )

    def _declare_aggregation_invariants(self) -> None:
        groupby_indices = self._indices("groupby")
        if groupby_indices and not self.order_observed:
            self._declare(
                "groupby_neutral_mutation",
                scope="operation_sequence",
                anchor_indices=groupby_indices,
                rationale="A neutral numeric mutation may be inserted before an aggregation.",
                fault_models=("aggregation_null_semantics", "numeric_precision"),
                cost=1.25,
            )
            sorted_groupby_indices = _groupby_sorted_input_indices(self.case)
            if sorted_groupby_indices:
                self._declare(
                    "groupby_sorted_input",
                    scope="operation_sequence",
                    anchor_indices=sorted_groupby_indices,
                    rationale="Eligible exact aggregations are invariant to a deterministic input sort.",
                    fault_models=("aggregation_strategy", "ordering_stability"),
                    cost=1.25,
                )
        aggregate_permutation_indices = [
            index
            for index in groupby_indices
            if len(aggregate_specs(self.operations[index])) >= 2
        ]
        if aggregate_permutation_indices and not self.order_observed:
            self._declare(
                "groupby_aggregation_permutation",
                scope="operation_sequence",
                anchor_indices=aggregate_permutation_indices,
                rationale="Independent aggregate outputs may be requested in a different order.",
                fault_models=("aggregation_projection", "schema_ordering"),
            )

    def _declare_sequence_rewrites(self) -> None:
        for relation, pairs, rationale in (
            (
                "filter_mutate_commutation",
                {("filter", "mutate"), ("mutate", "filter")},
                "Independent adjacent filter and mutation operations may commute.",
            ),
            (
                "filter_commutativity",
                {("filter", "filter")},
                "Adjacent filters may commute under conjunction semantics.",
            ),
            (
                "sort_select_commutation",
                {("sort", "select"), ("select", "sort")},
                "A projection retaining all sort keys may commute with sorting.",
            ),
            (
                "offset_limit_fusion",
                {("offset", "limit")},
                "Adjacent offset and limit operations admit an equivalent fused rewrite.",
            ),
            (
                "limit_offset_fusion",
                {("limit", "offset")},
                "Adjacent limit and offset operations admit an equivalent fused rewrite.",
            ),
        ):
            indices = (
                _filter_mutate_commutation_indices(self.operations)
                if relation == "filter_mutate_commutation"
                else _sort_select_commutation_indices(self.operations)
                if relation == "sort_select_commutation"
                else _adjacent_pattern_indices(self.kinds, pairs)
            )
            if indices:
                self._declare(
                    relation,
                    scope="operation_sequence",
                    anchor_indices=indices,
                    rationale=rationale,
                    fault_models=("optimizer_equivalence", "operation_reordering"),
                )

        pushdown_indices = _join_filter_pushdown_indices(self.case)
        if pushdown_indices:
            self._declare(
                "join_filter_pushdown",
                scope="operation_sequence",
                anchor_indices=pushdown_indices,
                source_relation_ids=self.all_source_ids,
                rationale="An independent left-side filter may be pushed below a join.",
                fault_models=("filter_pushdown", "join_cardinality"),
                secondary_relation_required=True,
            )

    def _declare_tail_rewrites(self) -> None:
        ordering_indices = self._indices("sort", "limit", "offset")
        if ordering_indices and not _ends_in_zero_offset(self.operations):
            self._declare(
                "offset_zero_insertion",
                scope="program_tail",
                anchor_indices=ordering_indices,
                rationale="Appending offset zero preserves an order-observing result.",
                fault_models=("offset_semantics", "ordering_stability"),
            )
        tail_limit_index = _tail_limit_index(self.kinds)
        if tail_limit_index is not None:
            self._declare(
                "offset_zero_at_tail_after_limit",
                scope="program_tail",
                anchor_indices=(tail_limit_index,),
                rationale="Offset zero after a tail limit preserves the exact row sequence.",
                fault_models=("limit_pushdown", "offset_semantics"),
            )
        large_limit_index = _no_op_limit_index(self.case, self.kinds)
        if large_limit_index is not None:
            self._declare(
                "limit_above_data_no_op",
                scope="program_tail",
                anchor_indices=(large_limit_index,),
                rationale="A limit already above the data cardinality may be enlarged safely.",
                fault_models=("limit_overflow", "limit_pushdown"),
            )

    def _indices(self, *kinds: str) -> tuple[int, ...]:
        wanted = set(kinds)
        return tuple(
            index
            for index, kind in enumerate(self.kinds)
            if kind in wanted
        )


def _adjacent_pattern_indices(
    kinds: Sequence[str],
    patterns: set[tuple[str, str]],
) -> tuple[int, ...]:
    for index in range(len(kinds) - 1):
        if (kinds[index], kinds[index + 1]) in patterns:
            return (index, index + 1)
    return ()


def _filter_materialization_applicable(case: Case) -> bool:
    if not case.tables or not case.program.operations:
        return False
    if case.program.order_sensitive or has_order_observer(case.program):
        return False
    operation = case.program.operations[0]
    if op_kind(operation) != "filter":
        return False
    base = case.tables[0]
    column = op_column(operation)
    comparator = op_comparator(operation)
    if column not in {spec.name for spec in base.columns} or comparator is None:
        return False
    try:
        filtered = [
            row
            for row in base.rows
            if evaluate_filter_predicate(
                row.get(column), comparator, op_value(operation)
            )
        ]
    except (TypeError, ValueError):
        return False
    return len(filtered) != len(base.rows)


def _drop_nulls_materialization_applicable(case: Case) -> bool:
    operation = case.program.operations[0]
    base = case.tables[0]
    columns = op_columns(operation)
    if not columns or any(
        column not in {spec.name for spec in base.columns} for column in columns
    ):
        return False
    return any(any(row.get(column) is None for column in columns) for row in base.rows)


def _fill_null_materialization_applicable(case: Case) -> bool:
    operation = case.program.operations[0]
    base = case.tables[0]
    column = op_column(operation)
    if column not in {spec.name for spec in base.columns} or op_value(operation) is None:
        return False
    return any(row.get(column) is None for row in base.rows)


def _distinct_materialization_applicable(case: Case) -> bool:
    operation = case.program.operations[0]
    base = case.tables[0]
    columns = op_columns(operation)
    if not columns or any(
        column not in {spec.name for spec in base.columns} for column in columns
    ):
        return False
    if len(columns) < len(base.columns):
        return True
    signatures = [
        canonical_key([row.get(column) for column in columns])
        for row in base.rows
    ]
    return len(signatures) != len(set(signatures))


def _filter_rejecting_row_applicable(case: Case) -> bool:
    columns = {column.name: column.type for column in case.tables[0].columns}
    for operation in case.program.operations:
        kind = op_kind(operation)
        if kind in {"limit", "offset", "groupby"}:
            return False
        if kind != "filter":
            continue
        column_type = columns.get(op_column(operation))
        comparator = op_comparator(operation)
        value = op_value(operation)
        if value is None or column_type is None:
            continue
        if column_type == "bool" and comparator in {"==", "!="}:
            return True
        if column_type == "str" and isinstance(value, str) and comparator in {"==", "!="}:
            return True
        if (
            column_type in {"int", "float"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and comparator in {">", ">=", "<", "<=", "==", "!="}
        ):
            return True
    return False


def _filter_mutate_commutation_indices(
    operations: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    for index in range(len(operations) - 1):
        first = operations[index]
        second = operations[index + 1]
        if op_kind(first) == "filter" and op_kind(second) == "mutate":
            filter_op, mutate_op = first, second
        elif op_kind(first) == "mutate" and op_kind(second) == "filter":
            filter_op, mutate_op = second, first
        else:
            continue
        filter_column = op_column(filter_op)
        if (
            filter_column
            and filter_column != op_column(mutate_op)
            and filter_column != expr_source(mutate_op)
        ):
            return (index, index + 1)
    return ()


def _sort_select_commutation_indices(
    operations: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    for index in range(len(operations) - 1):
        first, second = operations[index], operations[index + 1]
        if op_kind(first) == "select" and op_kind(second) == "sort":
            selected, ordering = set(op_columns(first)), second
        elif op_kind(first) == "sort" and op_kind(second) == "select":
            selected, ordering = set(op_columns(second)), first
        else:
            continue
        try:
            ordered_columns = sort_columns(ordering)
        except ValueError:
            continue
        if ordered_columns and set(ordered_columns).issubset(selected):
            return (index, index + 1)
    return ()


_EXACT_GROUPBY_SORT_FUNCS = {"count", "nunique", "min", "max", "any", "all"}
_GROUPBY_OUTPUT_ORDER_OBSERVERS = {
    "limit",
    "offset",
    "row_number_filter",
    "running_sum",
    "sortedness_check",
}


def _groupby_sorted_input_indices(case: Case) -> tuple[int, ...]:
    if has_order_observer(case.program):
        return ()
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) != "groupby":
            continue
        aggregates = list(aggregate_specs(operation))
        if not aggregates or any(
            aggregate_func(aggregate) not in _EXACT_GROUPBY_SORT_FUNCS
            for aggregate in aggregates
        ):
            continue
        if not _groupby_tail_safe(case.program.operations[index + 1 :]):
            continue
        columns = state_before_operation(case, index).columns
        if len(columns) < 2:
            continue
        keys = [key for key in groupby_keys(operation) if key in columns]
        if keys or columns:
            return (index,)
    return ()


def _groupby_tail_safe(tail: Sequence[Mapping[str, Any]]) -> bool:
    for operation in tail:
        kind = op_kind(operation)
        if kind == "sort":
            return True
        if kind in _GROUPBY_OUTPUT_ORDER_OBSERVERS:
            return False
    return True


def _join_inner_left_equivalence_index(case: Case) -> int | None:
    if has_order_observer(case.program) or len(case.tables) < 2:
        return None
    primary = case.tables[0]
    tables = {table.name: table for table in case.tables}
    mutated: set[str] = set()
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) == "mutate":
            mutated.add(op_column(operation))
            continue
        if op_kind(operation) != "join":
            continue
        left_keys, right_keys = join_key_pairs(operation)
        if len(left_keys) != 1 or len(right_keys) != 1:
            return None
        left_key, right_key = left_keys[0], right_keys[0]
        right = tables.get(op_table(operation))
        if right is None or left_key in mutated:
            return None
        left_values = [row.get(left_key) for row in primary.rows]
        if not left_values or any(_nullish(value) for value in left_values):
            return None
        right_values = {row.get(right_key) for row in right.rows}
        if not set(left_values).issubset(right_values):
            return None
        return index if _multiset_stable_tail(case, index + 1) else None
    return None


def _join_unmatched_dimension_index(case: Case) -> int | None:
    if has_order_observer(case.program) or len(case.tables) < 2:
        return None
    primary = case.tables[0]
    tables = {table.name: table for table in case.tables}
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) != "join":
            continue
        right = tables.get(op_table(operation))
        if right is None or any(
            op_table(later) == right.name
            for later in case.program.operations[index + 1 :]
        ):
            continue
        left_keys, right_keys = join_key_pairs(operation)
        if len(left_keys) != 1 or len(right_keys) != 1:
            continue
        right_type = next(
            (
                column.type
                for column in right.columns
                if column.name == right_keys[0]
            ),
            "",
        )
        if right_type not in {"int", "float", "str"}:
            continue
        left_values = {row.get(left_keys[0]) for row in primary.rows}
        if any(isinstance(value, (list, dict, set)) for value in left_values):
            continue
        return index
    return None


def _join_filter_pushdown_indices(case: Case) -> tuple[int, ...]:
    if has_order_observer(case.program):
        return ()
    primary_columns = {column.name for column in case.tables[0].columns}
    mutated: set[str] = set()
    operations = case.program.operations
    for join_index, operation in enumerate(operations):
        if op_kind(operation) == "mutate":
            mutated.add(op_column(operation))
            continue
        if op_kind(operation) != "join":
            continue
        between_mutated: set[str] = set()
        for filter_index in range(join_index + 1, len(operations)):
            candidate = operations[filter_index]
            kind = op_kind(candidate)
            if kind in {"groupby", "limit", "offset", "join", "tuple_absence_filter"}:
                break
            if kind == "mutate":
                between_mutated.add(op_column(candidate))
                continue
            if kind != "filter":
                continue
            column = op_column(candidate)
            if (
                column in primary_columns
                and column not in mutated
                and column not in between_mutated
            ):
                return (join_index, filter_index)
    return ()


def _multiset_stable_tail(case: Case, start_index: int) -> bool:
    tail = list(case.program.operations[start_index:])
    if has_order_observer(tail):
        return False
    for relative_index, operation in enumerate(tail):
        if op_kind(operation) != "groupby":
            continue
        column_types = state_before_operation(
            case, start_index + relative_index
        ).column_types
        for aggregate in aggregate_specs(operation):
            function = aggregate_func(aggregate)
            source_type = column_types.get(aggregate_column(aggregate))
            if function in {"mean", "any", "all"}:
                return False
            if function == "sum" and source_type == "float":
                return False
    return True


def _semi_anti_duplicate_applicable(case: Case) -> bool:
    tables = {table.name: table for table in case.tables}
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) not in {"semi_join", "anti_join"}:
            continue
        right = tables.get(op_table(operation))
        if right is None:
            continue
        if any(
            other_index != index and op_table(other) == right.name
            for other_index, other in enumerate(case.program.operations)
        ):
            continue
        _, right_keys = join_key_pairs(operation)
        if right_keys and any(
            join_key_value(row, right_keys) is not None for row in right.rows
        ):
            return True
    return False


def _nullish(value: Any) -> bool:
    return value is None or isinstance(value, float) and math.isnan(value)


def _has_lower_normalized_string_column(case: Case) -> bool:
    for column in case.tables[0].columns:
        if column.type != "str":
            continue
        values = [row.get(column.name) for row in case.tables[0].rows]
        if values and not any(
            isinstance(value, str) and value != value.lower() for value in values
        ):
            return True
    return False


def _has_integral_id_domain(case: Case) -> bool:
    id_column = next(
        (
            column
            for column in case.tables[0].columns
            if column.name == "id" and column.type == "int"
        ),
        None,
    )
    if id_column is None:
        return False
    values = [row.get("id") for row in case.tables[0].rows]
    return bool(values) and all(type(value) is int for value in values)


def _ends_in_zero_offset(operations: Sequence[Mapping[str, Any]]) -> bool:
    if not operations or op_kind(operations[-1]) != "offset":
        return False
    try:
        return op_n(operations[-1]) == 0
    except (TypeError, ValueError):
        return False


def _tail_limit_index(kinds: Sequence[str]) -> int | None:
    for index in range(len(kinds) - 1, -1, -1):
        kind = kinds[index]
        if kind == "limit":
            return index
        if kind not in {"select", "mutate"}:
            return None
    return None


def _no_op_limit_index(case: Case, kinds: Sequence[str]) -> int | None:
    total_rows = sum(len(table.rows) for table in case.tables)
    if total_rows <= 0:
        return None
    for index in range(len(kinds) - 1, -1, -1):
        kind = kinds[index]
        if kind == "limit":
            try:
                return index if op_n(case.program.operations[index]) >= total_rows else None
            except (TypeError, ValueError):
                return None
        if kind not in {"offset", "select", "mutate"}:
            return None
    return None
