from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case, Program, TableData, sort_columns
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding


@dataclass(slots=True)
class MetamorphicVariant:
    name: str
    relation: str
    case: Case


def build_metamorphic_variants(case: Case, limit: int = 4) -> list[MetamorphicVariant]:
    variants: list[MetamorphicVariant] = []
    variants.extend(_filter_rejecting_row_injection_variants(case))
    variants.extend(_join_unmatched_dimension_injection_variants(case))
    variants.extend(_join_inner_left_equivalence_variants(case))
    variants.extend(_join_filter_pushdown_variants(case))
    variants.extend(_groupby_neutral_mutation_variants(case))
    variants.extend(_mutate_add_zero_insertion_variants(case))
    variants.extend(_filter_mutate_commutation_variants(case))
    variants.extend(_string_lower_normalized_column_variants(case))
    variants.extend(_string_lower_idempotence_variants(case))
    variants.extend(_filter_tautology_insertion_variants(case))
    variants.extend(_groupby_aggregation_permutation_variants(case))
    variants.extend(_sort_select_commutation_variants(case))
    variants.extend(_sort_idempotence_variants(case))
    variants.extend(_row_permutation_variants(case))
    variants.extend(_join_table_permutation_variants(case))
    variants.extend(_filter_idempotence_variants(case))
    variants.extend(_limit_idempotence_variants(case))
    variants.extend(_groupby_key_permutation_variants(case))
    variants.extend(_filter_commutativity_variants(case))
    variants.extend(_select_idempotence_variants(case))
    return variants[:limit]


def evaluate_metamorphic_variants(
    case: Case,
    base: dict[str, NormalizedResult],
    variants: dict[str, dict[str, NormalizedResult]],
) -> list[Finding]:
    findings: list[Finding] = []
    for variant_name, normalized in variants.items():
        for backend, base_result in base.items():
            variant_result = normalized.get(backend)
            if variant_result is None:
                continue
            if _payload(base_result) == _payload(variant_result):
                continue
            relation = variant_name.split(":", 1)[0]
            sig = _signature(case, backend, variant_name, base_result, variant_result)
            findings.append(
                Finding(
                    finding_id=f"finding-{sig}",
                    kind=f"metamorphic_{relation}_violation",
                    severity="high",
                    suspicious_backends=[backend],
                    evidence=(
                        f"Backend {backend} violates metamorphic relation {variant_name}; "
                        f"base_status={base_result.status} variant_status={variant_result.status}"
                    ),
                    signature=sig,
                    root_cause=f"metamorphic_{relation}",
                    oracle="metamorphic",
                    confidence="medium",
                )
            )
    return findings


def _row_permutation_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.op_sequence()
    if "limit" in ops or "offset" in ops:
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


def _filter_rejecting_row_injection_variants(case: Case) -> list[MetamorphicVariant]:
    """Inject a domain row that should be removed by an existing cleaning filter."""

    primary = case.tables[0]
    columns = {col.name: col for col in primary.columns}
    for idx, op in enumerate(case.program.operations):
        kind = op.get("op")
        if kind in {"limit", "offset", "groupby"}:
            return []
        if kind != "filter":
            continue
        column = str(op.get("column", ""))
        spec = columns.get(column)
        if spec is None:
            continue
        rejecting = _rejecting_value(spec.type, op.get("cmp"), op.get("value"))
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

    primary = case.tables[0]
    table_by_name = {table.name: table for table in case.tables}
    for idx, op in enumerate(case.program.operations):
        if op.get("op") != "join":
            continue
        right = table_by_name.get(str(op.get("table", "")))
        if right is None:
            continue
        left_on = str(op.get("left_on", ""))
        right_on = str(op.get("right_on", ""))
        if not left_on or not right_on:
            continue
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

    primary = case.tables[0]
    table_by_name = {table.name: table for table in case.tables}
    mutated: set[str] = set()
    for idx, op in enumerate(case.program.operations):
        if op.get("op") == "mutate":
            mutated.add(str(op.get("column", "")))
            continue
        if op.get("op") != "join":
            continue
        left_on = str(op.get("left_on", ""))
        right_on = str(op.get("right_on", ""))
        right = table_by_name.get(str(op.get("table", "")))
        if right is None or not left_on or not right_on or left_on in mutated:
            return []
        left_values = {row.get(left_on) for row in primary.rows}
        if not left_values:
            return []
        right_values = {row.get(right_on) for row in right.rows}
        if not left_values.issubset(right_values):
            return []
        replacement = dict(op)
        replacement["how"] = "inner" if op.get("how") == "left" else "left"
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


def _join_filter_pushdown_variants(case: Case) -> list[MetamorphicVariant]:
    """Move a left-table filter before a join when intervening ops are independent."""

    primary_columns = {col.name for col in case.tables[0].columns}
    mutated: set[str] = set()
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op.get("op") == "mutate":
            mutated.add(str(op.get("column", "")))
            continue
        if op.get("op") != "join":
            continue
        between_mutated: set[str] = set()
        for filter_idx in range(idx + 1, len(ops)):
            candidate = ops[filter_idx]
            if candidate.get("op") in {"groupby", "limit", "offset", "join"}:
                break
            if candidate.get("op") == "mutate":
                between_mutated.add(str(candidate.get("column", "")))
                continue
            if candidate.get("op") != "filter":
                continue
            filter_column = str(candidate.get("column", ""))
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

    for idx, op in enumerate(case.program.operations):
        if op.get("op") != "groupby":
            continue
        col_types = _column_types_before(case, idx)
        used_columns = set(col_types)
        aggs = list(op.get("aggs", []))
        for agg_index, agg in enumerate(aggs):
            source = str(agg.get("column", ""))
            if col_types.get(source) not in {"int", "float"}:
                continue
            mirror = _fresh_column_name(used_columns, f"mr_{source}_plus0")
            mutated_agg = dict(agg)
            mutated_agg["column"] = mirror
            replacement = dict(op)
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


def _mutate_add_zero_insertion_variants(case: Case) -> list[MetamorphicVariant]:
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
        if first.get("op") == "filter" and second.get("op") == "mutate":
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
        if first.get("op") == "mutate" and second.get("op") == "filter":
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
    filter_column = str(filter_op.get("column", ""))
    mutate_column = str(mutate_op.get("column", ""))
    mutate_source = str(mutate_op.get("expr", {}).get("source", ""))
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
        expr = op.get("expr", {})
        if op.get("op") != "mutate" or expr.get("kind") != "string_lower":
            continue
        column = str(op.get("column", ""))
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


def _filter_tautology_insertion_variants(case: Case) -> list[MetamorphicVariant]:
    primary = case.tables[0]
    id_spec = _column_spec(primary, "id")
    if id_spec is None or id_spec.type != "int":
        return []
    id_values = [
        row.get("id")
        for row in primary.rows
        if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)
    ]
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


def _groupby_aggregation_permutation_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op.get("op") != "groupby":
            continue
        aggs = list(op.get("aggs", []))
        if len(aggs) < 2:
            continue
        permuted = dict(op)
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
    ops = case.program.op_sequence()
    if "limit" in ops or "offset" in ops or len(case.tables) < 2:
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
        if op.get("op") != "sort":
            continue
        duplicated = ops[: idx + 1] + [dict(op)] + ops[idx + 1 :]
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


def _sort_select_commutation_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx in range(len(ops) - 1):
        first = ops[idx]
        second = ops[idx + 1]
        if first.get("op") == "select" and second.get("op") == "sort":
            selected = set(first.get("columns", []))
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
        if first.get("op") == "sort" and second.get("op") == "select":
            selected = set(second.get("columns", []))
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
        if op.get("op") != "filter":
            continue
        duplicated = ops[: idx + 1] + [dict(op)] + ops[idx + 1 :]
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


def _limit_idempotence_variants(case: Case) -> list[MetamorphicVariant]:
    ops = case.program.operations
    for idx, op in enumerate(ops):
        if op.get("op") != "limit":
            continue
        duplicated = ops[: idx + 1] + [dict(op)] + ops[idx + 1 :]
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
        if op.get("op") != "groupby":
            continue
        keys = list(op.get("keys", []))
        if len(keys) < 2:
            continue
        permuted = dict(op)
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
        if ops[idx].get("op") != "filter" or ops[idx + 1].get("op") != "filter":
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
        if op.get("op") != "select":
            continue
        duplicated = ops[: idx + 1] + [dict(op)] + ops[idx + 1 :]
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


def _column_types_before(case: Case, op_index: int) -> dict[str, str]:
    table_by_name = {table.name: table for table in case.tables}
    col_types = {col.name: col.type for col in case.tables[0].columns}
    for op in case.program.operations[:op_index]:
        kind = op.get("op")
        if kind == "join":
            right = table_by_name.get(str(op.get("table", "")))
            if right is None:
                continue
            right_on = str(op.get("right_on", ""))
            for col in right.columns:
                if col.name == right_on or col.name in col_types:
                    continue
                col_types[col.name] = col.type
        elif kind == "select":
            selected = set(op.get("columns", []))
            col_types = {name: typ for name, typ in col_types.items() if name in selected}
        elif kind == "mutate":
            out_type = _expr_output_type(op.get("expr", {}), col_types)
            if out_type is not None:
                col_types[str(op.get("column", ""))] = out_type
        elif kind == "groupby":
            next_types: dict[str, str] = {}
            for key in op.get("keys", []):
                if key in col_types:
                    next_types[str(key)] = col_types[str(key)]
            for agg in op.get("aggs", []):
                source = str(agg.get("column", ""))
                alias = str(agg.get("as", ""))
                if not alias:
                    continue
                next_types[alias] = "int" if agg.get("func") == "count" else col_types.get(source, "float")
            col_types = next_types
    return col_types


def _expr_output_type(expr: dict[str, Any], col_types: dict[str, str]) -> str | None:
    source = str(expr.get("source", ""))
    source_type = col_types.get(source)
    if source_type is None:
        return None
    kind = expr.get("kind")
    if kind == "add_const":
        return source_type if source_type in {"int", "float"} else None
    if kind == "arith_const":
        op = expr.get("op")
        if source_type not in {"int", "float"} or op not in {"sub", "mul", "div", "mod"}:
            return None
        return "float" if op == "div" or source_type == "float" else source_type
    if kind == "cast" and expr.get("to") == "float":
        return "float" if source_type in {"int", "float"} else None
    if kind == "string_length":
        return "int" if source_type == "str" else None
    if kind == "string_lower":
        return "str" if source_type == "str" else None
    return None


def _fresh_column_name(existing: set[str], stem: str) -> str:
    candidate = stem
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    return candidate


def _default_row(table: TableData) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column in table.columns:
        if column.type == "int":
            row[column.name] = 0
        elif column.type == "float":
            row[column.name] = 0.0
        elif column.type == "bool":
            row[column.name] = False
        else:
            row[column.name] = ""
    return row


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


def _payload(norm: NormalizedResult) -> dict[str, Any]:
    return {
        "status": norm.status,
        "columns": norm.columns,
        "rows": norm.rows,
        "error_type": norm.error_type,
    }


def _signature(
    case: Case,
    backend: str,
    variant_name: str,
    base_result: NormalizedResult,
    variant_result: NormalizedResult,
) -> str:
    payload = {
        "case_id": case.case_id,
        "backend": backend,
        "variant": variant_name,
        "base": _payload(base_result),
        "variant_result": _payload(variant_result),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
