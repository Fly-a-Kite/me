from __future__ import annotations

import json
import math
from dataclasses import replace
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

from datadiff.canonicalization import (
    compare_result_batch,
    dedupe_by_canonical_key,
    short_canonical_hash,
)
from datadiff.contract_comparison import (
    ContractComparisonProfile,
    compare_results_under_contract,
    comparison_payload_for_case,
    comparison_profile_for_case,
)
from datadiff.dsl import Case, ColumnSpec, Program, TableData, sort_columns
from datadiff.operation_type_semantics import case_when_output_type
from datadiff.expression_semantics import aggregate_output_type, cast_output_type, expr_output_type, literal_output_type
from datadiff.filtering import evaluate_filter_predicate
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.mutator_ir import metamorphic_rewrite_rule_metadata
from datadiff.normalizer import NormalizedResult, _norm_value
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    condition_column,
    expr_index,
    expr_kind,
    expr_length,
    expr_new,
    expr_old,
    expr_separator,
    expr_source,
    expr_start,
    groupby_keys,
    has_order_observer,
    join_how,
    op_column,
    op_comparator,
    op_columns,
    op_kind,
    op_n,
    op_output_alias,
    op_table,
    op_value,
    operation_names,
)
from datadiff.oracle import Finding
from datadiff.program_state import state_before_operation


@dataclass(slots=True)
class MetamorphicVariant:
    name: str
    relation: str
    case: Case


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


def build_metamorphic_variants(
    case: Case,
    limit: int = 4,
    *,
    relation_order: Iterable[Any] | None = None,
) -> list[MetamorphicVariant]:
    return select_metamorphic_variants(
        all_metamorphic_variants(case),
        limit=limit,
        relation_order=relation_order,
    )


def ir_rewrite_metamorphic_rule_registry() -> dict[str, Any]:
    rules = metamorphic_rewrite_rule_metadata()
    return {
        "schema_version": "ir-rewrite-metamorphic-registry-v1",
        "rules": rules,
        "relations": {str(rule.get("relation", "")): rule for rule in rules},
        "methodology_claim": (
            "Metamorphic IR rewrite variants use the same typed rewrite rule source "
            "as mutation and reducer explanations; only semantics-preserving rules "
            "are eligible as equality oracles."
        ),
    }


def all_metamorphic_variants(case: Case) -> list[MetamorphicVariant]:
    return [
        variant
        for _relations, builder in _metamorphic_builder_specs()
        for variant in builder(case)
    ]


def constructible_metamorphic_relations(case: Case) -> frozenset[str]:
    """Return relations for which the canonical builder produced a variant."""

    return frozenset(variant.relation for variant in all_metamorphic_variants(case))


def select_metamorphic_variants(
    variants: Iterable[MetamorphicVariant],
    *,
    limit: int,
    relation_order: Iterable[Any] | None = None,
) -> list[MetamorphicVariant]:
    rows = list(variants)
    if limit <= 0:
        return []
    order = _relation_order_index(relation_order)
    if not order:
        return rows[:limit]
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (order.get(item[1].relation, len(order)), item[0]),
    )
    return [variant for _index, variant in ranked[:limit]]


@lru_cache(maxsize=1)
def metamorphic_relation_builder_registry(
) -> dict[str, tuple[Callable[[Case], list[MetamorphicVariant]], ...]]:
    """Executable builders used by CCS-guided obligation selection."""

    registry: dict[str, list[Callable[[Case], list[MetamorphicVariant]]]] = {}
    for relations, builder in _metamorphic_builder_specs():
        for relation in relations:
            registry.setdefault(relation, []).append(builder)
    return {
        relation: tuple(builders)
        for relation, builders in registry.items()
    }


def _metamorphic_builder_specs() -> tuple[
    tuple[tuple[str, ...], Callable[[Case], list[MetamorphicVariant]]], ...
]:
    """Single ordered source of truth for complete and guided construction."""

    return (
        (("filter_input_materialization",), _filter_input_materialization_variants),
        (
            (
                "drop_nulls_input_materialization",
                "fill_null_input_materialization",
                "distinct_input_materialization",
            ),
            _cleanup_input_materialization_variants,
        ),
        (("input_partition_union_all",), _input_partition_union_all_variants),
        (("filter_rejecting_row_injection",), _filter_rejecting_row_injection_variants),
        (("join_unmatched_dimension_injection",), _join_unmatched_dimension_injection_variants),
        (("join_inner_left_equivalence",), _join_inner_left_equivalence_variants),
        (("join_filter_pushdown",), _join_filter_pushdown_variants),
        (("semi_anti_join_rewrite",), _semi_anti_join_rewrite_variants),
        (("groupby_sorted_input",), _groupby_sorted_input_variants),
        (("groupby_neutral_mutation",), _groupby_neutral_mutation_variants),
        (("mutate_add_zero_insertion",), _mutate_add_zero_insertion_variants),
        (("filter_mutate_commutation",), _filter_mutate_commutation_variants),
        (("string_lower_normalized_column",), _string_lower_normalized_column_variants),
        (("string_lower_idempotence",), _string_lower_idempotence_variants),
        (("string_upper_idempotence",), _string_upper_idempotence_variants),
        (("string_strip_idempotence",), _string_strip_idempotence_variants),
        (("string_null_if_empty_idempotence",), _string_null_if_empty_idempotence_variants),
        (("string_replace_idempotence",), _string_replace_idempotence_variants),
        (("string_slice_prefix_idempotence",), _string_slice_prefix_idempotence_variants),
        (("string_split_part_idempotence",), _string_split_part_idempotence_variants),
        (("filter_tautology_insertion",), _filter_tautology_insertion_variants),
        (("offset_limit_fusion",), _offset_limit_fusion_variants),
        (("limit_offset_fusion",), _limit_offset_fusion_variants),
        (("offset_zero_insertion",), _offset_zero_insertion_variants),
        (("groupby_aggregation_permutation",), _groupby_aggregation_permutation_variants),
        (("sort_select_commutation",), _sort_select_commutation_variants),
        (("sort_idempotence",), _sort_idempotence_variants),
        (("row_permutation",), _row_permutation_variants),
        (("join_table_permutation",), _join_table_permutation_variants),
        (("filter_idempotence",), _filter_idempotence_variants),
        (("union_all_empty_append",), _union_all_empty_append_variants),
        (("drop_nulls_idempotence",), _drop_nulls_idempotence_variants),
        (
            ("semi_anti_join_unmatched_right_injection",),
            _semi_anti_join_unmatched_right_injection_variants,
        ),
        (
            ("semi_anti_join_right_duplicate_injection",),
            _semi_anti_join_right_duplicate_variants,
        ),
        (("distinct_idempotence",), _distinct_idempotence_variants),
        (("fill_null_idempotence",), _fill_null_idempotence_variants),
        (("coalesce_idempotence",), _coalesce_idempotence_variants),
        (("case_when_idempotence",), _case_when_idempotence_variants),
        (("limit_idempotence",), _limit_idempotence_variants),
        (("groupby_key_permutation",), _groupby_key_permutation_variants),
        (("filter_commutativity",), _filter_commutativity_variants),
        (("select_idempotence",), _select_idempotence_variants),
        (("limit_above_data_no_op",), _limit_above_data_no_op_variants),
        (("offset_zero_at_tail_after_limit",), _offset_zero_at_tail_after_limit_variants),
        (("filter_idempotence_immediate",), _filter_idempotence_immediate_variants),
    )


def build_metamorphic_variants_for_relations(
    case: Case,
    relations: Iterable[str],
) -> list[MetamorphicVariant]:
    registry = metamorphic_relation_builder_registry()
    requested = tuple(dict.fromkeys(str(item) for item in relations if str(item)))
    requested_set = set(requested)
    builders: list[Callable[[Case], list[MetamorphicVariant]]] = []
    for relation in requested:
        for builder in registry.get(relation, ()):
            if builder not in builders:
                builders.append(builder)
    variants = [variant for builder in builders for variant in builder(case)]
    return [variant for variant in variants if variant.relation in requested_set]


def _relation_order_index(relation_order: Iterable[Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in relation_order or []:
        relation = str(value).strip()
        if relation and relation not in out:
            out[relation] = len(out)
    return out


def evaluate_metamorphic_variants(
    case: Case,
    base: dict[str, NormalizedResult],
    variants: dict[str, dict[str, NormalizedResult]],
    *,
    comparison_mode: str = "contract",
) -> list[Finding]:
    if comparison_mode not in {"legacy", "contract"}:
        raise ValueError(f"unsupported comparison mode: {comparison_mode}")
    findings: list[Finding] = []
    base_comparison_profile = comparison_profile_for_case(case)
    for variant_name, normalized in variants.items():
        relation = variant_name.split(":", 1)[0]
        comparison_profile = _metamorphic_comparison_profile(
            base_comparison_profile,
            relation,
        )
        for backend, base_result in base.items():
            variant_result = normalized.get(backend)
            if variant_result is None:
                continue
            contract_comparison = None
            if comparison_mode == "legacy":
                if base_result.comparison_key == variant_result.comparison_key:
                    continue
                if _relaxed_float_payload(base_result) == _relaxed_float_payload(variant_result):
                    continue
                comparison = compare_result_batch([base_result, variant_result])
            else:
                contract_comparison = compare_results_under_contract(
                    case,
                    [base_result, variant_result],
                    profile=comparison_profile,
                )
                if not contract_comparison.has_mismatch:
                    continue
                comparison = contract_comparison.comparison
            mismatch_class = comparison.mismatch_class
            sig = _signature(
                case,
                backend,
                variant_name,
                base_result,
                variant_result,
                profile=comparison_profile,
                comparison_mode=comparison_mode,
            )
            comparison_detail = (
                f"mismatch_class={mismatch_class}"
                if comparison_mode == "legacy"
                else f"view={contract_comparison.profile.view}; mismatch_class={mismatch_class}"
            )
            findings.append(
                Finding(
                    finding_id=f"finding-{sig}",
                    kind=f"metamorphic_{relation}_violation",
                    severity="high",
                    suspicious_backends=[backend],
                    evidence=(
                        f"Backend {backend} violates metamorphic relation {variant_name}; "
                        f"base_status={base_result.status} variant_status={variant_result.status}; "
                        f"{comparison_detail}"
                    ),
                    signature=sig,
                    root_cause=f"metamorphic_{relation}",
                    oracle="metamorphic",
                    confidence="medium",
                    mismatch_class=mismatch_class,
                )
            )
    return findings


def _metamorphic_comparison_profile(
    profile: ContractComparisonProfile,
    relation: str,
) -> ContractComparisonProfile:
    if relation == "semi_anti_join_rewrite" and profile.view == "ordered_value":
        # The rewrite checks relational membership equivalence. Replacing an
        # existence join with an inner/left join does not promise that the
        # physical join operator preserves an upstream row order.
        return replace(profile, view="bag_value")
    return profile


def _row_permutation_variants(case: Case) -> list[MetamorphicVariant]:
    if has_order_observer(case.program):
        return []
    table = case.tables[0]
    if len(table.rows) < 2:
        return []
    permuted_table = TableData(table.name, table.columns, list(reversed(table.rows)))
    variant_tables = [permuted_table] + list(case.tables[1:])
    variant = Case(
        case_id=f"{case.case_id}-mr-row-permutation",
        seed=case.seed,
        tables=variant_tables,
        program=case.program,
    )
    return [MetamorphicVariant("row_permutation:reverse", "row_permutation", variant)]


def _filter_input_materialization_variants(case: Case) -> list[MetamorphicVariant]:
    """Replace a leading filter with an equivalent pre-filtered input table."""

    if not case.tables or not case.program.operations:
        return []
    if case.program.order_sensitive or has_order_observer(case.program):
        return []
    filter_op = case.program.operations[0]
    if op_kind(filter_op) != "filter":
        return []
    base = case.tables[0]
    column = op_column(filter_op)
    if column not in {spec.name for spec in base.columns}:
        return []
    comparator = op_comparator(filter_op)
    if comparator is None:
        return []
    try:
        filtered_rows = [
            dict(row)
            for row in base.rows
            if evaluate_filter_predicate(row.get(column), comparator, op_value(filter_op))
        ]
    except (TypeError, ValueError):
        return []
    if filtered_rows == [dict(row) for row in base.rows]:
        return []

    materialized = TableData(base.name, list(base.columns), filtered_rows)
    program = Program(
        program_id=f"{case.program.program_id}-mr-filter-input-materialization",
        seed=case.program.seed,
        operations=list(case.program.operations[1:]),
    )
    return [
        MetamorphicVariant(
            f"filter_input_materialization:{column}-{len(base.rows)}to{len(filtered_rows)}",
            "filter_input_materialization",
            Case(
                f"{case.case_id}-mr-filter-input-materialization",
                case.seed,
                [materialized, *case.tables[1:]],
                program,
                metadata=dict(case.metadata),
            ),
        )
    ]


def _cleanup_input_materialization_variants(case: Case) -> list[MetamorphicVariant]:
    """Replace a leading cleanup/table-shaping op with an equivalent input table."""

    if not case.tables or not case.program.operations:
        return []
    op = case.program.operations[0]
    kind = op_kind(op)
    if kind == "drop_nulls":
        return _drop_nulls_input_materialization_variants(case, op)
    if kind == "fill_null":
        return _fill_null_input_materialization_variants(case, op)
    if kind == "distinct":
        return _distinct_input_materialization_variants(case, op)
    return []


def _drop_nulls_input_materialization_variants(case: Case, op: dict[str, Any]) -> list[MetamorphicVariant]:
    base = case.tables[0]
    base_columns = {spec.name for spec in base.columns}
    columns = op_columns(op)
    if not columns or any(column not in base_columns for column in columns):
        return []
    rows = [
        dict(row)
        for row in base.rows
        if all(evaluate_filter_predicate(row.get(column), "is_not_null", None) for column in columns)
    ]
    if rows == [dict(row) for row in base.rows]:
        return []
    materialized = TableData(base.name, list(base.columns), rows)
    program = Program(
        program_id=f"{case.program.program_id}-mr-drop-nulls-input-materialization",
        seed=case.program.seed,
        operations=list(case.program.operations[1:]),
    )
    return [
        MetamorphicVariant(
            f"drop_nulls_input_materialization:{'-'.join(columns)}-{len(base.rows)}to{len(rows)}",
            "drop_nulls_input_materialization",
            Case(
                f"{case.case_id}-mr-drop-nulls-input-materialization",
                case.seed,
                [materialized, *case.tables[1:]],
                program,
                metadata=dict(case.metadata),
            ),
        )
    ]


def _fill_null_input_materialization_variants(case: Case, op: dict[str, Any]) -> list[MetamorphicVariant]:
    base = case.tables[0]
    column = op_column(op)
    fill_value = op_value(op)
    if column not in {spec.name for spec in base.columns} or fill_value is None:
        return []
    rows: list[dict[str, Any]] = []
    changed = False
    for row in base.rows:
        copied = dict(row)
        if evaluate_filter_predicate(copied.get(column), "is_null", None):
            copied[column] = fill_value
            changed = True
        rows.append(copied)
    if not changed:
        return []
    materialized = TableData(base.name, list(base.columns), rows)
    program = Program(
        program_id=f"{case.program.program_id}-mr-fill-null-input-materialization",
        seed=case.program.seed,
        operations=list(case.program.operations[1:]),
    )
    return [
        MetamorphicVariant(
            f"fill_null_input_materialization:{column}",
            "fill_null_input_materialization",
            Case(
                f"{case.case_id}-mr-fill-null-input-materialization",
                case.seed,
                [materialized, *case.tables[1:]],
                program,
                metadata=dict(case.metadata),
            ),
        )
    ]


def _distinct_input_materialization_variants(case: Case, op: dict[str, Any]) -> list[MetamorphicVariant]:
    base = case.tables[0]
    spec_by_name = {spec.name: spec for spec in base.columns}
    columns = op_columns(op)
    if not columns or any(column not in spec_by_name for column in columns):
        return []
    rows = dedupe_by_canonical_key(
        ({column: row.get(column) for column in columns} for row in base.rows),
        payload=lambda row: [ _norm_value(row.get(column)) for column in columns ],
    )
    if rows == [{column: row.get(column) for column in columns} for row in base.rows] and len(columns) == len(base.columns):
        return []
    materialized = TableData(base.name, [spec_by_name[column] for column in columns], rows)
    program = Program(
        program_id=f"{case.program.program_id}-mr-distinct-input-materialization",
        seed=case.program.seed,
        operations=list(case.program.operations[1:]),
    )
    return [
        MetamorphicVariant(
            f"distinct_input_materialization:{'-'.join(columns)}-{len(base.rows)}to{len(rows)}",
            "distinct_input_materialization",
            Case(
                f"{case.case_id}-mr-distinct-input-materialization",
                case.seed,
                [materialized, *case.tables[1:]],
                program,
                metadata=dict(case.metadata),
            ),
        )
    ]


def _filter_rejecting_row_injection_variants(case: Case) -> list[MetamorphicVariant]:
    """Inject a domain row that should be removed by an existing cleaning filter."""

    if has_order_observer(case.program):
        return []
    primary = case.tables[0]
    columns = {col.name: col for col in primary.columns}
    for idx, op in enumerate(case.program.operations):
        kind = op_kind(op)
        if kind in {"limit", "offset", "groupby"}:
            return []
        if kind != "filter":
            continue
        column = op_column(op)
        spec = columns.get(column)
        if spec is None:
            continue
        rejecting = _rejecting_value(spec.type, op_comparator(op), op_value(op))
        if rejecting is _NO_VALUE:
            continue
        injected = _default_row(primary)
        injected[column] = rejecting
        variant_tables = [
            TableData(primary.name, primary.columns, list(primary.rows) + [injected]),
            *case.tables[1:],
        ]
        variant = Case(
            f"{case.case_id}-mr-filter-reject-row-{idx}",
            case.seed,
            variant_tables,
            case.program,
        )
        return [
            MetamorphicVariant(
                f"filter_rejecting_row_injection:{column}-{idx}",
                "filter_rejecting_row_injection",
                variant,
            )
        ]
    return []


def _join_unmatched_dimension_injection_variants(case: Case) -> list[MetamorphicVariant]:
    """Add a dimension row with no matching fact key; inner/left enrichment output is unchanged."""

    if has_order_observer(case.program):
        return []
    primary = case.tables[0]
    table_by_name = {table.name: table for table in case.tables}
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "join":
            continue
        right = table_by_name.get(op_table(op))
        if right is None:
            continue
        if any(op_table(later_op) == right.name for later_op in case.program.operations[idx + 1 :]):
            continue
        left_keys, right_keys = join_key_pairs(op)
        if len(left_keys) != 1 or len(right_keys) != 1:
            continue
        left_on = left_keys[0]
        right_on = right_keys[0]
        left_values = {row.get(left_on) for row in primary.rows}
        right_spec = _column_spec(right, right_on)
        if right_spec is None:
            continue
        unmatched = _fresh_unmatched_value(right_spec.type, left_values)
        if unmatched is _NO_VALUE:
            continue
        injected = _default_row(right)
        injected[right_on] = unmatched
        variant_tables = []
        for table in case.tables:
            if table.name == right.name:
                variant_tables.append(TableData(table.name, table.columns, list(table.rows) + [injected]))
            else:
                variant_tables.append(table)
        variant = Case(
            f"{case.case_id}-mr-unmatched-dimension-{idx}",
            case.seed,
            variant_tables,
            case.program,
        )
        return [
            MetamorphicVariant(
                f"join_unmatched_dimension_injection:{right.name}-{idx}",
                "join_unmatched_dimension_injection",
                variant,
            )
        ]
    return []


def _join_inner_left_equivalence_variants(case: Case) -> list[MetamorphicVariant]:
    """Flip inner/left join mode when every base left key has a matching right key."""

    if has_order_observer(case.program):
        return []
    primary = case.tables[0]
    table_by_name = {table.name: table for table in case.tables}
    mutated: set[str] = set()
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) == "mutate":
            mutated.add(op_column(op))
            continue
        if op_kind(op) != "join":
            continue
        left_keys, right_keys = join_key_pairs(op)
        if len(left_keys) != 1 or len(right_keys) != 1:
            return []
        left_on = left_keys[0]
        right_on = right_keys[0]
        right = table_by_name.get(op_table(op))
        if right is None or not left_on or not right_on or left_on in mutated:
            return []
        left_raw_values = [row.get(left_on) for row in primary.rows]
        if any(_is_nullish_join_key(value) for value in left_raw_values):
            return []
        left_values = set(left_raw_values)
        if not left_values:
            return []
        right_values = {row.get(right_on) for row in right.rows}
        if not left_values.issubset(right_values):
            return []
        if not _multiset_stable_tail(case, idx + 1):
            return []
        replacement = _as_plain_mapping(op)
        replacement["how"] = "inner" if join_how(op) == "left" else "left"
        program = Program(
            program_id=f"{case.program.program_id}-mr-join-left-inner-equivalence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[:idx] + [replacement] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"join_inner_left_equivalence:{idx}",
                "join_inner_left_equivalence",
                Case(f"{case.case_id}-mr-join-left-inner-equivalence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _multiset_stable_tail(case: Case, start_index: int) -> bool:
    tail = list(case.program.operations[start_index:])
    if has_order_observer(tail):
        return False
    for relative_index, op in enumerate(tail):
        kind = op_kind(op)
        if kind != "groupby":
            continue
        col_types = _column_types_before(case, start_index + relative_index)
        for agg in aggregate_specs(op):
            func = aggregate_func(agg)
            source_type = col_types.get(aggregate_column(agg))
            if func in {"mean", "any", "all"}:
                return False
            if func == "sum" and source_type == "float":
                return False
    return True


def _is_nullish_join_key(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _join_filter_pushdown_variants(case: Case) -> list[MetamorphicVariant]:
    """Move a left-table filter before a join when intervening ops are independent."""

    if has_order_observer(case.program):
        return []
    primary_columns = {col.name for col in case.tables[0].columns}
    mutated: set[str] = set()
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) == "mutate":
            mutated.add(op_column(op))
            continue
        if op_kind(op) != "join":
            continue
        between_mutated: set[str] = set()
        for filter_idx in range(idx + 1, len(ops)):
            candidate = ops[filter_idx]
            candidate_kind = op_kind(candidate)
            if candidate_kind in {"groupby", "limit", "offset", "join", "tuple_absence_filter"}:
                break
            if candidate_kind == "mutate":
                between_mutated.add(op_column(candidate))
                continue
            if candidate_kind != "filter":
                continue
            filter_column = op_column(candidate)
            if (
                filter_column not in primary_columns
                or filter_column in mutated
                or filter_column in between_mutated
            ):
                continue
            rewritten = (
                ops[:idx]
                + [candidate, op]
                + ops[idx + 1 : filter_idx]
                + ops[filter_idx + 1 :]
            )
            program = Program(
                program_id=f"{case.program.program_id}-mr-join-filter-pushdown-{idx}",
                seed=case.program.seed,
                operations=rewritten,
            )
            return [
                MetamorphicVariant(
                    f"join_filter_pushdown:{filter_column}-{idx}-{filter_idx}",
                    "join_filter_pushdown",
                    Case(f"{case.case_id}-mr-join-filter-pushdown-{idx}", case.seed, case.tables, program),
                )
            ]
    return []


def _groupby_neutral_mutation_variants(case: Case) -> list[MetamorphicVariant]:
    """Aggregate over a +0 mirror of a numeric column while keeping aliases unchanged."""

    if has_order_observer(case.program):
        return []
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "groupby":
            continue
        col_types = _column_types_before(case, idx)
        used_columns = set(col_types)
        aggs = list(aggregate_specs(op))
        for agg_index, agg in enumerate(aggs):
            source = aggregate_column(agg)
            if col_types.get(source) not in {"int", "float"}:
                continue
            mirror = _fresh_column_name(used_columns, f"mr_{source}_plus0")
            mutated_agg = _as_plain_mapping(agg)
            mutated_agg["column"] = mirror
            replacement = _as_plain_mapping(op)
            replacement["aggs"] = aggs[:agg_index] + [mutated_agg] + aggs[agg_index + 1 :]
            neutral_mutate = {
                "op": "mutate",
                "column": mirror,
                "expr": {"kind": "add_const", "source": source, "value": 0},
            }
            program = Program(
                program_id=f"{case.program.program_id}-mr-groupby-neutral-mutation-{idx}",
                seed=case.program.seed,
                operations=case.program.operations[:idx]
                + [neutral_mutate, replacement]
                + case.program.operations[idx + 1 :],
            )
            return [
                MetamorphicVariant(
                    f"groupby_neutral_mutation:{source}-{idx}",
                    "groupby_neutral_mutation",
                    Case(f"{case.case_id}-mr-groupby-neutral-mutation-{idx}", case.seed, case.tables, program),
                )
            ]
    return []


def _groupby_sorted_input_variants(case: Case) -> list[MetamorphicVariant]:
    """Pre-sort exact groupby inputs to exercise sorted aggregation paths."""

    if has_order_observer(case.program):
        return []
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "groupby":
            continue
        aggs = list(aggregate_specs(op))
        if not aggs or any(aggregate_func(agg) not in _EXACT_GROUPBY_SORT_FUNCS for agg in aggs):
            continue
        if not _groupby_sorted_input_tail_safe(case.program.operations[idx + 1 :]):
            continue
        current_columns = _column_names_before(case, idx)
        if len(current_columns) < 2:
            continue
        keys = [key for key in groupby_keys(op) if key in current_columns]
        sort_columns_for_variant = keys + [column for column in current_columns if column not in set(keys)]
        sort_columns_for_variant = sort_columns_for_variant[: min(3, len(sort_columns_for_variant))]
        if not sort_columns_for_variant:
            continue
        sort_op = {
            "op": "sort",
            "keys": [
                {
                    "column": column,
                    "ascending": index % 2 == 0,
                    "nulls": "first" if index % 2 else "last",
                }
                for index, column in enumerate(sort_columns_for_variant)
            ],
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-groupby-sorted-input-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[:idx] + [sort_op, _as_plain_mapping(op)] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"groupby_sorted_input:{'-'.join(sort_columns_for_variant)}-{idx}",
                "groupby_sorted_input",
                Case(
                    f"{case.case_id}-mr-groupby-sorted-input-{idx}",
                    case.seed,
                    case.tables,
                    program,
                    metadata=dict(case.metadata),
                ),
            )
        ]
    return []


_EXACT_GROUPBY_SORT_FUNCS = {"count", "nunique", "min", "max", "any", "all"}
_GROUPBY_OUTPUT_ORDER_OBSERVERS = {"limit", "offset", "row_number_filter", "running_sum", "sortedness_check"}


def _groupby_sorted_input_tail_safe(tail: list[dict[str, Any]]) -> bool:
    for op in tail:
        kind = op_kind(op)
        if kind == "sort":
            return True
        if kind in _GROUPBY_OUTPUT_ORDER_OBSERVERS:
            return False
    return True


def _mutate_add_zero_insertion_variants(case: Case) -> list[MetamorphicVariant]:
    if has_order_observer(case.program):
        return []
    primary = case.tables[0]
    source = next((col.name for col in primary.columns if col.type == "int"), None)
    if source is None:
        source = next((col.name for col in primary.columns if col.type == "float"), None)
    if source is None:
        return []
    neutral = {
        "op": "mutate",
        "column": source,
        "expr": {"kind": "add_const", "source": source, "value": 0},
    }
    program = Program(
        program_id=f"{case.program.program_id}-mr-mutate-add-zero",
        seed=case.program.seed,
        operations=[neutral] + list(case.program.operations),
    )
    return [
        MetamorphicVariant(
            f"mutate_add_zero_insertion:{source}",
            "mutate_add_zero_insertion",
            Case(f"{case.case_id}-mr-mutate-add-zero", case.seed, case.tables, program),
        )
    ]


def _filter_mutate_commutation_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx in range(len(ops) - 1):
        first = ops[idx]
        second = ops[idx + 1]
        if op_kind(first) == "filter" and op_kind(second) == "mutate":
            if not _filter_and_mutate_are_independent(first, second):
                continue
            swapped = list(ops)
            swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
            program = Program(
                program_id=f"{case.program.program_id}-mr-filter-mutate-commute-{idx}",
                seed=case.program.seed,
                operations=swapped,
            )
            return [
                MetamorphicVariant(
                    f"filter_mutate_commutation:swap-{idx}-{idx + 1}",
                    "filter_mutate_commutation",
                    Case(f"{case.case_id}-mr-filter-mutate-commute-{idx}", case.seed, case.tables, program),
                )
            ]
        if op_kind(first) == "mutate" and op_kind(second) == "filter":
            if not _filter_and_mutate_are_independent(second, first):
                continue
            swapped = list(ops)
            swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
            program = Program(
                program_id=f"{case.program.program_id}-mr-mutate-filter-commute-{idx}",
                seed=case.program.seed,
                operations=swapped,
            )
            return [
                MetamorphicVariant(
                    f"filter_mutate_commutation:swap-{idx}-{idx + 1}",
                    "filter_mutate_commutation",
                    Case(f"{case.case_id}-mr-mutate-filter-commute-{idx}", case.seed, case.tables, program),
                )
            ]
    return []


def _filter_and_mutate_are_independent(filter_op: dict[str, Any], mutate_op: dict[str, Any]) -> bool:
    filter_column = op_column(filter_op)
    mutate_column = op_column(mutate_op)
    mutate_source = expr_source(mutate_op)
    return bool(filter_column) and filter_column != mutate_column and filter_column != mutate_source


def _string_lower_normalized_column_variants(case: Case) -> list[MetamorphicVariant]:
    primary = case.tables[0]
    for spec in primary.columns:
        if spec.type != "str":
            continue
        values = [row.get(spec.name) for row in primary.rows]
        if not values or any(isinstance(value, str) and value != value.lower() for value in values):
            continue
        neutral = {
            "op": "mutate",
            "column": spec.name,
            "expr": {"kind": "string_lower", "source": spec.name},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-lower-normalized-{spec.name}",
            seed=case.program.seed,
            operations=[neutral] + list(case.program.operations),
        )
        return [
            MetamorphicVariant(
                f"string_lower_normalized_column:{spec.name}",
                "string_lower_normalized_column",
                Case(f"{case.case_id}-mr-string-lower-normalized-{spec.name}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_lower_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_lower":
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_lower", "source": column},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-lower-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_lower_idempotence:repeat-{idx}",
                "string_lower_idempotence",
                Case(f"{case.case_id}-mr-string-lower-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_upper_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_upper":
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_upper", "source": column},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-upper-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_upper_idempotence:repeat-{idx}",
                "string_upper_idempotence",
                Case(f"{case.case_id}-mr-string-upper-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_strip_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_strip":
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_strip", "source": column},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-strip-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_strip_idempotence:repeat-{idx}",
                "string_strip_idempotence",
                Case(f"{case.case_id}-mr-string-strip-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_null_if_empty_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_null_if_empty":
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_null_if_empty", "source": column},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-null-if-empty-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_null_if_empty_idempotence:repeat-{idx}",
                "string_null_if_empty_idempotence",
                Case(f"{case.case_id}-mr-string-null-if-empty-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_replace_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_replace":
            continue
        old = expr_old(op)
        new = expr_new(op)
        if not isinstance(old, str) or old == "" or not isinstance(new, str) or old in new:
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_replace", "source": column, "old": old, "new": new},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-replace-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_replace_idempotence:repeat-{idx}",
                "string_replace_idempotence",
                Case(f"{case.case_id}-mr-string-replace-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_slice_prefix_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_slice":
            continue
        start = expr_start(op)
        length = expr_length(op)
        if start != 0 or type(length) is not int or length < 0:
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_slice", "source": column, "start": 0, "length": length},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-slice-prefix-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_slice_prefix_idempotence:repeat-{idx}",
                "string_slice_prefix_idempotence",
                Case(f"{case.case_id}-mr-string-slice-prefix-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _string_split_part_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) != "mutate" or expr_kind(op) != "string_split_part":
            continue
        sep = expr_separator(op)
        index = expr_index(op)
        if not isinstance(sep, str) or sep == "" or index != 0:
            continue
        column = op_column(op)
        if not column:
            continue
        repeated = {
            "op": "mutate",
            "column": column,
            "expr": {"kind": "string_split_part", "source": column, "sep": sep, "index": 0},
        }
        program = Program(
            program_id=f"{case.program.program_id}-mr-string-split-part-idempotence-{idx}",
            seed=case.program.seed,
            operations=case.program.operations[: idx + 1] + [repeated] + case.program.operations[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"string_split_part_idempotence:repeat-{idx}",
                "string_split_part_idempotence",
                Case(f"{case.case_id}-mr-string-split-part-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _filter_tautology_insertion_variants(case: Case) -> list[MetamorphicVariant]:
    if has_order_observer(case.program):
        return []
    primary = case.tables[0]
    id_spec = _column_spec(primary, "id")
    if id_spec is None or id_spec.type != "int":
        return []
    if _sorts_by_mean_aggregation(case.program.operations):
        return []
    id_values = []
    for row in primary.rows:
        value = row.get("id")
        if not isinstance(value, int) or isinstance(value, bool):
            return []
        id_values.append(value)
    if not id_values:
        return []
    tautology = {"op": "filter", "column": "id", "cmp": ">=", "value": min(id_values)}
    program = Program(
        program_id=f"{case.program.program_id}-mr-filter-tautology",
        seed=case.program.seed,
        operations=[tautology] + list(case.program.operations),
    )
    return [
        MetamorphicVariant(
            "filter_tautology_insertion:id-min",
            "filter_tautology_insertion",
            Case(f"{case.case_id}-mr-filter-tautology", case.seed, case.tables, program),
        )
    ]


def _sorts_by_mean_aggregation(ops: list[dict[str, Any]]) -> bool:
    mean_aliases: set[str] = set()
    for op in ops:
        kind = op_kind(op)
        if kind == "groupby":
            for agg in aggregate_specs(op):
                if aggregate_func(agg) == "mean" and aggregate_alias(agg):
                    mean_aliases.add(aggregate_alias(agg))
        elif kind == "sort" and mean_aliases:
            try:
                sorted_columns = sort_columns(op)
            except ValueError:
                continue
            if any(column in mean_aliases for column in sorted_columns):
                return True
    return False


def _offset_zero_insertion_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    if not any(kind in {"sort", "limit", "offset"} for kind in operation_names(ops)):
        return []
    if ops and op_kind(ops[-1]) == "offset" and op_n(ops[-1]) == 0:
        return []
    program = Program(
        program_id=f"{case.program.program_id}-mr-offset-zero",
        seed=case.program.seed,
        operations=list(ops) + [{"op": "offset", "n": 0}],
    )
    return [
        MetamorphicVariant(
            "offset_zero_insertion:tail",
            "offset_zero_insertion",
            Case(f"{case.case_id}-mr-offset-zero", case.seed, case.tables, program),
        )
    ]


def _offset_limit_fusion_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx in range(len(ops) - 1):
        first = ops[idx]
        second = ops[idx + 1]
        if op_kind(first) != "offset" or op_kind(second) != "limit":
            continue
        try:
            offset = op_n(first)
            limit = op_n(second)
        except (TypeError, ValueError):
            continue
        if offset < 0 or limit < 0:
            continue
        rewritten = ops[:idx] + [{"op": "limit", "n": offset + limit}, {"op": "offset", "n": offset}] + ops[idx + 2 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-offset-limit-fusion-{idx}",
            seed=case.program.seed,
            operations=rewritten,
        )
        return [
            MetamorphicVariant(
                f"offset_limit_fusion:swap-{idx}-{idx + 1}",
                "offset_limit_fusion",
                Case(f"{case.case_id}-mr-offset-limit-fusion-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _limit_offset_fusion_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx in range(len(ops) - 1):
        first = ops[idx]
        second = ops[idx + 1]
        if op_kind(first) != "limit" or op_kind(second) != "offset":
            continue
        try:
            limit = op_n(first)
            offset = op_n(second)
        except (TypeError, ValueError):
            continue
        if offset < 0 or limit < 0:
            continue
        rewritten = ops[:idx] + [{"op": "offset", "n": offset}, {"op": "limit", "n": max(0, limit - offset)}] + ops[idx + 2 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-limit-offset-fusion-{idx}",
            seed=case.program.seed,
            operations=rewritten,
        )
        return [
            MetamorphicVariant(
                f"limit_offset_fusion:swap-{idx}-{idx + 1}",
                "limit_offset_fusion",
                Case(f"{case.case_id}-mr-limit-offset-fusion-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _groupby_aggregation_permutation_variants(case: Case) -> list[MetamorphicVariant]:
    if has_order_observer(case.program):
        return []
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "groupby":
            continue
        aggs = list(aggregate_specs(op))
        if len(aggs) < 2:
            continue
        permuted = _as_plain_mapping(op)
        permuted["aggs"] = list(reversed(aggs))
        program = Program(
            program_id=f"{case.program.program_id}-mr-groupby-agg-permutation-{idx}",
            seed=case.program.seed,
            operations=ops[:idx] + [permuted] + ops[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"groupby_aggregation_permutation:reverse-{idx}",
                "groupby_aggregation_permutation",
                Case(f"{case.case_id}-mr-groupby-agg-permutation-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _join_table_permutation_variants(case: Case) -> list[MetamorphicVariant]:
    if has_order_observer(case.program) or len(case.tables) < 2:
        return []
    for index, table in enumerate(case.tables[1:], start=1):
        if len(table.rows) < 2:
            continue
        variant_tables = list(case.tables)
        variant_tables[index] = TableData(table.name, table.columns, list(reversed(table.rows)))
        variant = Case(
            case_id=f"{case.case_id}-mr-join-table-permutation-{index}",
            seed=case.seed,
            tables=variant_tables,
            program=case.program,
        )
        return [
            MetamorphicVariant(
                f"join_table_permutation:reverse-{table.name}",
                "join_table_permutation",
                variant,
            )
        ]
    return []


def _sort_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "sort":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-sort-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"sort_idempotence:duplicate-{idx}",
                "sort_idempotence",
                Case(f"{case.case_id}-mr-sort-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


_LIMIT_HUGE_THRESHOLD = 1_000_000_000  # 1B rows — far above any generated case


def _case_total_input_row_count(case: Case) -> int:
    return sum(len(table.rows) for table in case.tables)


def _offset_zero_at_tail_after_limit_variants(case: Case) -> list[MetamorphicVariant]:
    """Appending offset(0) after an existing limit must be a no-op.

    The original program already has a deterministic limit; offset(0) cannot
    drop rows and must return the same row sequence. Catches backend bugs
    where offset 0 is incorrectly applied (e.g. consumes a row, or interacts
    badly with limit pushdown).
    """

    ops = case.program.operations
    if not ops:
        return []
    has_tail_limit = False
    for op in reversed(ops):
        kind = op_kind(op)
        if kind == "limit":
            has_tail_limit = True
            break
        if kind in {"select", "mutate"}:
            continue
        break
    if not has_tail_limit:
        return []
    rewritten = list(ops) + [{"op": "offset", "n": 0}]
    program = Program(
        program_id=f"{case.program.program_id}-mr-offset-zero-tail",
        seed=case.program.seed,
        operations=rewritten,
    )
    return [
        MetamorphicVariant(
            "offset_zero_at_tail_after_limit:append",
            "offset_zero_at_tail_after_limit",
            Case(
                f"{case.case_id}-mr-offset-zero-tail",
                case.seed,
                case.tables,
                program,
            ),
        )
    ]


def _filter_idempotence_immediate_variants(case: Case) -> list[MetamorphicVariant]:
    """Duplicating any filter op immediately must be a no-op.

    `filter(p) ≡ filter(p) + filter(p)` is a textbook idempotence. Catches
    optimizer bugs that re-evaluate the predicate against an already-filtered
    intermediate or drop rows under double evaluation.
    """

    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "filter":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-filter-idempotence-immediate-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"filter_idempotence_immediate:duplicate-{idx}",
                "filter_idempotence_immediate",
                Case(
                    f"{case.case_id}-mr-filter-idempotence-immediate-{idx}",
                    case.seed,
                    case.tables,
                    program,
                ),
            )
        ]
    return []


def _limit_above_data_no_op_variants(case: Case) -> list[MetamorphicVariant]:
    """Replace a tail limit(N) with limit(BIG) when BIG >> total input rows.

    Tests backend handling of huge LIMIT values (overflow paths, signed/unsigned
    conversions, optimizer short-circuits). Result must be unchanged because the
    original limit is also a no-op when N exceeds the realised row count.
    """

    ops = case.program.operations
    if not ops:
        return []
    total_rows = _case_total_input_row_count(case)
    if total_rows == 0:
        return []
    tail_limit_idx = None
    for idx in range(len(ops) - 1, -1, -1):
        kind = op_kind(ops[idx])
        if kind == "limit":
            tail_limit_idx = idx
            break
        if kind in {"offset", "select", "mutate"}:
            continue
        break
    if tail_limit_idx is None:
        return []
    try:
        existing_n = op_n(ops[tail_limit_idx])
    except (TypeError, ValueError):
        return []
    # Only valid when the existing limit is already a no-op vs realised data.
    if existing_n is None or existing_n < total_rows:
        return []
    huge_op = _as_plain_mapping(ops[tail_limit_idx])
    huge_op["n"] = max(int(existing_n), _LIMIT_HUGE_THRESHOLD)
    rewritten = list(ops)
    rewritten[tail_limit_idx] = huge_op
    program = Program(
        program_id=f"{case.program.program_id}-mr-limit-huge-{tail_limit_idx}",
        seed=case.program.seed,
        operations=rewritten,
    )
    return [
        MetamorphicVariant(
            f"limit_above_data_no_op:{existing_n}-to-{huge_op['n']}",
            "limit_above_data_no_op",
            Case(f"{case.case_id}-mr-limit-huge-{tail_limit_idx}", case.seed, case.tables, program),
        )
    ]


def _distinct_after_groupby_no_op_variants(case: Case) -> list[MetamorphicVariant]:
    """Appending distinct() after groupby+aggregate is a no-op.

    GroupBy already emits one row per distinct key combination, so an extra
    distinct must not change the result. Catches optimizer bugs that re-shuffle
    rows or drop them under double dedup.
    """

    ops = case.program.operations
    if not ops:
        return []
    last_groupby_idx = None
    for idx in range(len(ops) - 1, -1, -1):
        kind = op_kind(ops[idx])
        if kind == "groupby":
            last_groupby_idx = idx
            break
        # Tolerate trailing harmless ops between groupby and tail, but not
        # any op that mutates the row identity.
        if kind in {"select", "sort"}:
            continue
        break
    if last_groupby_idx is None:
        return []
    if last_groupby_idx == len(ops) - 1:
        appended_ops = list(ops) + [{"op": "distinct"}]
    else:
        appended_ops = list(ops[: last_groupby_idx + 1]) + [{"op": "distinct"}] + list(ops[last_groupby_idx + 1 :])
    program = Program(
        program_id=f"{case.program.program_id}-mr-distinct-after-groupby-{last_groupby_idx}",
        seed=case.program.seed,
        operations=appended_ops,
    )
    return [
        MetamorphicVariant(
            f"distinct_after_groupby_no_op:idx-{last_groupby_idx}",
            "distinct_after_groupby_no_op",
            Case(
                f"{case.case_id}-mr-distinct-after-groupby-{last_groupby_idx}",
                case.seed,
                case.tables,
                program,
            ),
        )
    ]


def _filter_constant_true_at_tail_variants(case: Case) -> list[MetamorphicVariant]:
    """Tail-append a tautological filter on a column known to be all non-null.

    Equivalent to no-op; tests backend handling of trivial predicate planners
    and 'always-true' constant-folding paths, which is where DataFusion and
    Polars optimizers have historically had bugs.
    """

    ops = case.program.operations
    if not ops:
        return []
    primary = case.tables[0]
    # Pick a column that is present in the base table and has at least one
    # non-null entry to keep the filter safe regardless of intermediate ops.
    candidate_col = None
    for col in primary.columns:
        non_nulls = [row.get(col.name) for row in primary.rows]
        if any(value is not None for value in non_nulls) and col.type in {"int", "float", "bool", "str"}:
            candidate_col = col.name
            break
    if candidate_col is None:
        return []
    tautology = {
        "op": "filter",
        "column": candidate_col,
        "condition": {"cmp": "is_not_null", "column": candidate_col},
    }
    # Only append when no later op mutates the column or removes the table.
    rewritten = list(ops) + [tautology]
    program = Program(
        program_id=f"{case.program.program_id}-mr-filter-constant-true-tail",
        seed=case.program.seed,
        operations=rewritten,
    )
    return [
        MetamorphicVariant(
            f"filter_constant_true_tail:{candidate_col}",
            "filter_constant_true_tail",
            Case(
                f"{case.case_id}-mr-filter-constant-true-tail",
                case.seed,
                case.tables,
                program,
            ),
        )
    ]


def _union_all_self_then_distinct_variants(case: Case) -> list[MetamorphicVariant]:
    """UNION ALL(t, t) then DISTINCT == DISTINCT(t).

    By replacing the primary input with `t UNION ALL t` and then forcing a
    distinct at the tail, the result must equal the original program with the
    tail distinct. Catches set-operation duplicate-counting bugs (DataFusion
    has had these) and distinct-after-set-op planner bugs.
    """

    ops = case.program.operations
    if not ops or len(case.tables) < 1:
        return []
    if case.program.order_sensitive or has_order_observer(case.program):
        return []
    primary = case.tables[0]
    if len(primary.rows) == 0 or len(primary.rows) > 32:
        return []
    # New table = primary doubled via union-all in-source.
    doubled = TableData(primary.name, list(primary.columns), list(primary.rows) + list(primary.rows))
    new_tables = [doubled, *case.tables[1:]]
    appended_ops = list(ops) + [{"op": "distinct"}]
    program = Program(
        program_id=f"{case.program.program_id}-mr-union-all-self-distinct",
        seed=case.program.seed,
        operations=appended_ops,
    )
    return [
        MetamorphicVariant(
            f"union_all_self_distinct:{primary.name}-rows{len(primary.rows)}",
            "union_all_self_distinct",
            Case(
                f"{case.case_id}-mr-union-all-self-distinct",
                case.seed,
                new_tables,
                program,
            ),
        )
    ]


def _input_partition_union_all_variants(case: Case) -> list[MetamorphicVariant]:
    """Rebuild the primary input through a UNION ALL partition boundary."""

    if not case.tables or case.program.order_sensitive or has_order_observer(case.program):
        return []
    base = case.tables[0]
    if len(base.rows) < 2:
        return []
    existing_names = {table.name for table in case.tables}
    if "t_partition_tail" in existing_names:
        return []
    split = len(base.rows) // 2
    if split <= 0 or split >= len(base.rows):
        return []

    head = TableData(base.name, list(base.columns), list(base.rows[:split]))
    tail = TableData("t_partition_tail", list(base.columns), list(base.rows[split:]))
    program = Program(
        program_id=f"{case.program.program_id}-mr-input-partition-union-all",
        seed=case.program.seed,
        operations=[{"op": "union_all", "table": tail.name}, *case.program.operations],
    )
    return [
        MetamorphicVariant(
            f"input_partition_union_all:split-{split}-{len(base.rows) - split}",
            "input_partition_union_all",
            Case(
                f"{case.case_id}-mr-input-partition-union-all",
                case.seed,
                [head, *case.tables[1:], tail],
                program,
                metadata=dict(case.metadata),
            ),
        )
    ]


def _sort_select_commutation_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx in range(len(ops) - 1):
        first = ops[idx]
        second = ops[idx + 1]
        if op_kind(first) == "select" and op_kind(second) == "sort":
            selected = set(op_columns(first))
            columns = sort_columns(second)
            if columns and set(columns).issubset(selected):
                swapped = list(ops)
                swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
                program = Program(
                    program_id=f"{case.program.program_id}-mr-sort-select-commute-{idx}",
                    seed=case.program.seed,
                    operations=swapped,
                )
                return [
                    MetamorphicVariant(
                        f"sort_select_commutation:swap-{idx}-{idx + 1}",
                        "sort_select_commutation",
                        Case(f"{case.case_id}-mr-sort-select-commute-{idx}", case.seed, case.tables, program),
                    )
                ]
        if op_kind(first) == "sort" and op_kind(second) == "select":
            selected = set(op_columns(second))
            columns = sort_columns(first)
            if columns and set(columns).issubset(selected):
                swapped = list(ops)
                swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
                program = Program(
                    program_id=f"{case.program.program_id}-mr-select-sort-commute-{idx}",
                    seed=case.program.seed,
                    operations=swapped,
                )
                return [
                    MetamorphicVariant(
                        f"sort_select_commutation:swap-{idx}-{idx + 1}",
                        "sort_select_commutation",
                        Case(f"{case.case_id}-mr-select-sort-commute-{idx}", case.seed, case.tables, program),
                    )
                ]
    return []


def _filter_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "filter":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-filter-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"filter_idempotence:duplicate-{idx}",
                "filter_idempotence",
                Case(f"{case.case_id}-mr-filter-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _distinct_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "distinct":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-distinct-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"distinct_idempotence:duplicate-{idx}",
                "distinct_idempotence",
                Case(f"{case.case_id}-mr-distinct-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _drop_nulls_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "drop_nulls":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-drop-nulls-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"drop_nulls_idempotence:duplicate-{idx}",
                "drop_nulls_idempotence",
                Case(f"{case.case_id}-mr-drop-nulls-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _union_all_empty_append_variants(case: Case) -> list[MetamorphicVariant]:
    if not case.tables or any(table.name == "t_empty_union" for table in case.tables):
        return []
    base = case.tables[0]
    empty = TableData("t_empty_union", list(base.columns), [])
    program = Program(
        program_id=f"{case.program.program_id}-mr-union-all-empty-append",
        seed=case.program.seed,
        operations=[{"op": "union_all", "table": empty.name}, *case.program.operations],
    )
    return [
        MetamorphicVariant(
            "union_all_empty_append:prepend-empty",
            "union_all_empty_append",
            Case(
                f"{case.case_id}-mr-union-all-empty-append",
                case.seed,
                [*case.tables, empty],
                program,
                metadata=dict(case.metadata),
            ),
        )
    ]


def _semi_anti_join_rewrite_variants(case: Case) -> list[MetamorphicVariant]:
    """Rewrite existence joins into equivalent materialized-key join plans."""

    if not case.tables:
        return []
    tables = {table.name: table for table in case.tables}
    table_names = set(tables)
    all_column_names = {column.name for table in case.tables for column in table.columns}
    for idx, op in enumerate(case.program.operations):
        kind = op_kind(op)
        if kind not in {"semi_join", "anti_join"}:
            continue
        right = tables.get(op_table(op))
        if right is None:
            continue
        left_keys, right_keys = join_key_pairs(op)
        if not left_keys or len(left_keys) != len(right_keys) or len(set(right_keys)) != len(right_keys):
            continue
        current_columns = _column_names_before(case, idx)
        if not current_columns or any(left_key not in current_columns for left_key in left_keys):
            continue
        marker = None
        if kind == "anti_join":
            marker = _fresh_column_name(all_column_names | set(current_columns), "__datadiff_mr_match")
        key_table_name = _fresh_table_name(table_names, f"t_mr_{kind}_keys_{idx}")
        key_table = _semi_anti_rewrite_key_table(key_table_name, right, right_keys, marker)
        if key_table is None:
            continue
        rewritten_join = {
            "op": "join",
            "table": key_table.name,
            "left_on": list(left_keys),
            "right_on": list(right_keys),
            "how": "inner" if kind == "semi_join" else "left",
        }
        rewritten_ops: list[dict[str, Any]] = [_as_plain_mapping(operation) for operation in case.program.operations[:idx]]
        rewritten_ops.append(rewritten_join)
        if kind == "anti_join":
            if marker is None:
                continue
            rewritten_ops.extend(
                [
                    {"op": "filter", "column": marker, "cmp": "is_null", "value": None},
                    {"op": "select", "columns": list(current_columns)},
                ]
            )
        rewritten_ops.extend(_as_plain_mapping(operation) for operation in case.program.operations[idx + 1 :])
        program = Program(
            program_id=f"{case.program.program_id}-mr-semi-anti-rewrite-{idx}",
            seed=case.program.seed,
            operations=rewritten_ops,
        )
        return [
            MetamorphicVariant(
                f"semi_anti_join_rewrite:{kind}-op-{idx}",
                "semi_anti_join_rewrite",
                Case(
                    f"{case.case_id}-mr-semi-anti-rewrite-{idx}",
                    case.seed,
                    [*case.tables, key_table],
                    program,
                    metadata=dict(case.metadata),
                ),
            )
        ]
    return []


def _semi_anti_rewrite_key_table(
    name: str,
    right: TableData,
    right_keys: list[str],
    marker: str | None,
) -> TableData | None:
    columns: list[ColumnSpec] = []
    for key in right_keys:
        spec = _column_spec(right, key)
        if spec is None:
            return None
        columns.append(ColumnSpec(key, spec.type, nullable=False))
    if marker is not None:
        columns.append(ColumnSpec(marker, "int", nullable=False))

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in right.rows:
        key = join_key_value(row, right_keys)
        if key is None or key in seen:
            continue
        seen.add(key)
        out = {right_key: row.get(right_key) for right_key in right_keys}
        if marker is not None:
            out[marker] = 1
        rows.append(out)
    return TableData(name, columns, rows)


def _semi_anti_join_right_duplicate_variants(case: Case) -> list[MetamorphicVariant]:
    tables = {table.name: table for table in case.tables}
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) not in {"semi_join", "anti_join"}:
            continue
        right = tables.get(op_table(op))
        if right is None:
            continue
        if any(
            other_index != idx and op_table(other) == right.name
            for other_index, other in enumerate(case.program.operations)
        ):
            # The transformation mutates the input table globally. Reusing
            # that table in another operation (for example an earlier inner
            # join) can legitimately change the program before the target
            # existence join, invalidating the MR.
            continue
        _, right_keys = join_key_pairs(op)
        if not right_keys:
            continue
        duplicate = next((row for row in right.rows if join_key_value(row, right_keys) is not None), None)
        if duplicate is None:
            continue
        cloned_tables = []
        for table in case.tables:
            rows = [dict(row) for row in table.rows]
            if table.name == right.name:
                rows.append(dict(duplicate))
            cloned_tables.append(TableData(table.name, list(table.columns), rows))
        program = Program(
            program_id=f"{case.program.program_id}-mr-semi-anti-right-duplicate-{idx}",
            seed=case.program.seed,
            operations=[_as_plain_mapping(operation) for operation in case.program.operations],
        )
        return [
            MetamorphicVariant(
                f"semi_anti_join_right_duplicate_injection:op-{idx}",
                "semi_anti_join_right_duplicate_injection",
                Case(
                    f"{case.case_id}-mr-semi-anti-right-duplicate-{idx}",
                    case.seed,
                    cloned_tables,
                    program,
                    metadata=dict(case.metadata),
                ),
            )
        ]
    return []


def _semi_anti_join_unmatched_right_injection_variants(case: Case) -> list[MetamorphicVariant]:
    tables = {table.name: table for table in case.tables}
    primary = case.tables[0]
    for idx, op in enumerate(case.program.operations):
        if op_kind(op) not in {"semi_join", "anti_join"}:
            continue
        right = tables.get(op_table(op))
        if right is None:
            continue
        left_keys, right_keys = join_key_pairs(op)
        if not left_keys or len(left_keys) != len(right_keys):
            continue
        injected = _default_row(right)
        if not _fill_unmatched_join_key(injected, right, right_keys, primary, left_keys):
            continue
        cloned_tables = []
        for table in case.tables:
            rows = [dict(row) for row in table.rows]
            if table.name == right.name:
                rows.append(injected)
            cloned_tables.append(TableData(table.name, list(table.columns), rows))
        program = Program(
            program_id=f"{case.program.program_id}-mr-semi-anti-right-unmatched-{idx}",
            seed=case.program.seed,
            operations=[_as_plain_mapping(operation) for operation in case.program.operations],
        )
        return [
            MetamorphicVariant(
                f"semi_anti_join_unmatched_right_injection:op-{idx}",
                "semi_anti_join_unmatched_right_injection",
                Case(
                    f"{case.case_id}-mr-semi-anti-right-unmatched-{idx}",
                    case.seed,
                    cloned_tables,
                    program,
                    metadata=dict(case.metadata),
                ),
            )
        ]
    return []


def _fill_unmatched_join_key(
    injected: dict[str, Any],
    right: TableData,
    right_keys: list[str],
    primary: TableData,
    left_keys: list[str],
) -> bool:
    left_values_by_key = {
        left_key: {row.get(left_key) for row in primary.rows}
        for left_key in left_keys
    }
    for left_key, right_key in zip(left_keys, right_keys):
        right_spec = _column_spec(right, right_key)
        if right_spec is None:
            return False
        existing_values = left_values_by_key.get(left_key, set())
        fresh_value = _fresh_unmatched_value(right_spec.type, existing_values)
        if fresh_value is _NO_VALUE:
            continue
        injected[right_key] = fresh_value
        for other_right_key in right_keys:
            if other_right_key != right_key and injected.get(other_right_key) is None:
                other_spec = _column_spec(right, other_right_key)
                if other_spec is None:
                    return False
                injected[other_right_key] = _default_value(other_spec.type)
        return True
    return False


def _fill_null_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "fill_null":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-fill-null-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"fill_null_idempotence:duplicate-{idx}",
                "fill_null_idempotence",
                Case(f"{case.case_id}-mr-fill-null-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _coalesce_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "coalesce":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-coalesce-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"coalesce_idempotence:duplicate-{idx}",
                "coalesce_idempotence",
                Case(f"{case.case_id}-mr-coalesce-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _case_when_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "case_when":
            continue
        if condition_column(op) == op_output_alias(op):
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-case-when-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"case_when_idempotence:duplicate-{idx}",
                "case_when_idempotence",
                Case(f"{case.case_id}-mr-case-when-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _limit_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "limit":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-limit-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"limit_idempotence:duplicate-{idx}",
                "limit_idempotence",
                Case(f"{case.case_id}-mr-limit-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _groupby_key_permutation_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "groupby":
            continue
        keys = list(groupby_keys(op))
        if len(keys) < 2:
            continue
        permuted = _as_plain_mapping(op)
        permuted["keys"] = list(reversed(keys))
        program = Program(
            program_id=f"{case.program.program_id}-mr-groupby-key-permutation-{idx}",
            seed=case.program.seed,
            operations=ops[:idx] + [permuted] + ops[idx + 1 :],
        )
        return [
            MetamorphicVariant(
                f"groupby_key_permutation:reverse-{idx}",
                "groupby_key_permutation",
                Case(f"{case.case_id}-mr-groupby-key-permutation-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


def _filter_commutativity_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    variants: list[MetamorphicVariant] = []
    for idx in range(len(ops) - 1):
        if op_kind(ops[idx]) != "filter" or op_kind(ops[idx + 1]) != "filter":
            continue
        swapped = list(ops)
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        program = Program(
            program_id=f"{case.program.program_id}-mr-filter-commute-{idx}",
            seed=case.program.seed,
            operations=swapped,
        )
        variants.append(
            MetamorphicVariant(
                f"filter_commutativity:swap-{idx}-{idx + 1}",
                "filter_commutativity",
                Case(f"{case.case_id}-mr-filter-commute-{idx}", case.seed, case.tables, program),
            )
        )
        break
    return variants


def _select_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op_kind(op) != "select":
            continue
        duplicated = ops[: idx + 1] + [_as_plain_mapping(op)] + ops[idx + 1 :]
        program = Program(
            program_id=f"{case.program.program_id}-mr-select-idempotence-{idx}",
            seed=case.program.seed,
            operations=duplicated,
        )
        return [
            MetamorphicVariant(
                f"select_idempotence:duplicate-{idx}",
                "select_idempotence",
                Case(f"{case.case_id}-mr-select-idempotence-{idx}", case.seed, case.tables, program),
            )
        ]
    return []


_NO_VALUE = object()


def _column_spec(table: TableData, column: str):
    for spec in table.columns:
        if spec.name == column:
            return spec
    return None


def _column_names_before(case: Case, op_index: int) -> list[str]:
    return state_before_operation(case, op_index).columns


def _column_types_before(case: Case, op_index: int) -> dict[str, str]:
    return state_before_operation(case, op_index).column_types


def _case_when_output_type(then_value: Any, else_value: Any) -> str | None:
    return case_when_output_type(then_value, else_value)


def _literal_output_type(value: Any) -> str | None:
    return literal_output_type(value)


def _aggregate_output_type(source_type: str | None, func: str) -> str:
    return aggregate_output_type(source_type, func)


def _expr_output_type(expr: dict[str, Any], col_types: dict[str, str]) -> str | None:
    return expr_output_type(expr, col_types)


def _cast_output_type(source_type: str, expr: dict[str, Any]) -> str | None:
    return cast_output_type(source_type, expr)


def _fresh_column_name(existing: set[str], stem: str) -> str:
    candidate = stem
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    return candidate


def _fresh_table_name(existing: set[str], stem: str) -> str:
    candidate = stem
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    return candidate


def _default_row(table: TableData) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column in table.columns:
        row[column.name] = _default_value(column.type)
    return row


def _default_value(column_type: str) -> Any:
    if column_type == "int":
        return 0
    if column_type == "float":
        return 0.0
    if column_type == "bool":
        return False
    return ""


def _rejecting_value(column_type: str, comparator: Any, value: Any) -> Any:
    if value is None:
        return _NO_VALUE
    if column_type == "bool":
        if comparator == "==":
            return not bool(value)
        if comparator == "!=":
            return bool(value)
        return _NO_VALUE
    if column_type == "str":
        if not isinstance(value, str):
            return _NO_VALUE
        if comparator == "==":
            return "__datadiff_rejected__" if value != "__datadiff_rejected__" else "__datadiff_other__"
        if comparator == "!=":
            return value
        return _NO_VALUE
    if column_type in {"int", "float"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return _NO_VALUE
        if comparator == ">":
            return value
        if comparator == ">=":
            return value - 1
        if comparator == "<":
            return value
        if comparator == "<=":
            return value + 1
        if comparator == "==":
            return value + 1
        if comparator == "!=":
            return value
    return _NO_VALUE


def _fresh_unmatched_value(column_type: str, existing: set[Any]) -> Any:
    if column_type == "int":
        numeric = [value for value in existing if isinstance(value, int) and not isinstance(value, bool)]
        candidate = (max(numeric) if numeric else 0) + 1_000_003
        while candidate in existing:
            candidate += 1
        return candidate
    if column_type == "float":
        numeric = [value for value in existing if isinstance(value, (int, float)) and not isinstance(value, bool)]
        candidate = float(max(numeric) if numeric else 0.0) + 1_000_003.0
        while candidate in existing:
            candidate += 1.0
        return candidate
    if column_type == "str":
        candidate = "__datadiff_unmatched_dimension__"
        suffix = 0
        while candidate in existing:
            suffix += 1
            candidate = f"__datadiff_unmatched_dimension_{suffix}__"
        return candidate
    return _NO_VALUE


def _payload(
    case: Case,
    norm: NormalizedResult,
    *,
    profile: ContractComparisonProfile,
    comparison_mode: str,
) -> dict[str, Any]:
    if comparison_mode == "legacy":
        return {
            "status": norm.status,
            "columns": norm.columns,
            "rows": norm.rows,
            "error_type": norm.error_type,
        }
    return comparison_payload_for_case(case, norm, profile=profile)


def _relaxed_float_payload(norm: NormalizedResult) -> dict[str, Any]:
    return {
        "status": norm.status,
        "columns": norm.columns,
        "rows": [
            [_norm_value(value, preserve_float_precision=False) for value in row]
            for row in norm.rows
        ],
        "error_type": norm.error_type,
    }


def _signature(
    case: Case,
    backend: str,
    variant_name: str,
    base_result: NormalizedResult,
    variant_result: NormalizedResult,
    *,
    profile: ContractComparisonProfile,
    comparison_mode: str = "contract",
) -> str:
    payload = {
        "case_id": case.case_id,
        "backend": backend,
        "variant": variant_name,
        "base": _payload(
            case,
            base_result,
            profile=profile,
            comparison_mode=comparison_mode,
        ),
        "variant_result": _payload(
            case,
            variant_result,
            profile=profile,
            comparison_mode=comparison_mode,
        ),
    }
    return short_canonical_hash(payload, 16)
