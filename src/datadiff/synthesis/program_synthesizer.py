from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datadiff.dsl import Program, TableData
from datadiff.identifiers import make_safe_output_name
from datadiff.synthesis.grammar import (
    GrammarRegistry,
    ProductionRule,
    transition_by_program_state,
)
from datadiff.synthesis.typed_state import (
    TableSchema,
    TypedProgramState,
    compatible_join_key_pairs,
)
from datadiff.reproducibility import canonical_candidates
from datadiff.util import unique_preserve_order


@dataclass(frozen=True, slots=True)
class SynthesisContext:
    primary_table: TableData
    extra_tables: tuple[TableData, ...] = ()

    @property
    def table_by_name(self) -> dict[str, TableData]:
        return {table.name: table for table in (self.primary_table, *self.extra_tables)}


def synthesize_program(
    seed: int,
    table: TableData,
    *,
    max_ops: int = 6,
    extra_tables: Sequence[TableData] | None = None,
    weights: Mapping[str, float] | None = None,
    registry: GrammarRegistry | None = None,
) -> Program:
    rnd = random.Random(seed * 1_000_003 + 0x71)
    context = SynthesisContext(table, tuple(extra_tables or ()))
    state = TypedProgramState.from_table(table, extra_tables=context.extra_tables)
    grammar = registry or default_grammar_registry()
    op_count = rnd.randint(2, max(2, int(max_ops)))
    operations: list[dict[str, Any]] = []
    for _ in range(op_count):
        if state.is_grouped:
            step_weights = {"filter": 0.4, "sort": 1.5, "limit": 1.4, "offset": 0.6, "select": 1.1}
            if weights:
                step_weights.update(weights)
        else:
            step_weights = weights
        resolved_weights = _adaptive_step_weights(state, step_weights)
        try:
            operation, state = grammar.synthesize_step(rnd, state, context, weights=resolved_weights)
        except ValueError:
            break
        operations.append(operation)
    if not operations:
        operations.append({"op": "limit", "n": len(table.rows)})
    return Program(program_id=f"prog-{seed:08d}-typed-grammar", seed=seed, operations=operations)


def default_grammar_registry() -> GrammarRegistry:
    return GrammarRegistry(
        [
            ProductionRule("filter", 1.35, lambda state: bool(state.comparable), _gen_filter, transition_by_program_state),
            ProductionRule("drop_nulls", 0.45, lambda state: bool(state.columns), _gen_drop_nulls, transition_by_program_state),
            ProductionRule("select", 0.70, lambda state: bool(state.columns), _gen_select, transition_by_program_state),
            ProductionRule("distinct", 0.40, lambda state: bool(state.columns), _gen_distinct, transition_by_program_state),
            ProductionRule("fill_null", 0.45, lambda state: bool(state.columns), _gen_fill_null, transition_by_program_state),
            ProductionRule("coalesce", 0.55, _has_coalesce_group, _gen_coalesce, transition_by_program_state),
            ProductionRule("case_when", 0.70, lambda state: bool(state.comparable), _gen_case_when, transition_by_program_state),
            ProductionRule("mutate", 1.20, _has_mutate_source, _gen_mutate, transition_by_program_state),
            ProductionRule("sort", 0.70, lambda state: bool(state.columns), _gen_sort, transition_by_program_state),
            ProductionRule("limit", 0.55, lambda state: True, _gen_limit, transition_by_program_state),
            ProductionRule("offset", 0.35, lambda state: True, _gen_offset, transition_by_program_state),
            ProductionRule("groupby", 0.85, _can_groupby, _gen_groupby, transition_by_program_state),
            ProductionRule("join", 0.75, lambda state: bool(state.compatible_join_tables()), _gen_join, transition_by_program_state),
        ]
    )


def _adaptive_step_weights(
    state: TypedProgramState,
    weights: Mapping[str, float] | None,
) -> dict[str, float] | None:
    resolved = dict(weights or {})
    if state.validity_score < 0.55:
        resolved["join"] = resolved.get("join", 1.0) * 0.55
        resolved["groupby"] = resolved.get("groupby", 1.0) * 0.70
    if state.expandability_score < 0.25:
        resolved["limit"] = resolved.get("limit", 1.0) * 0.60
        resolved["offset"] = resolved.get("offset", 1.0) * 0.60
        resolved["select"] = resolved.get("select", 1.0) * 0.75
    if "aggregation_boundary" not in state.structural_risk_tags:
        resolved["groupby"] = resolved.get("groupby", 1.0) * 1.20
    if "join_cardinality" not in state.structural_risk_tags and state.compatible_join_tables():
        resolved["join"] = resolved.get("join", 1.0) * 1.20
    if "shape:ordered" not in state.coverage_axes:
        resolved["sort"] = resolved.get("sort", 1.0) * 1.15
    return resolved or None


def _gen_filter(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    column = rnd.choice(canonical_candidates(state.comparable))
    column_type = state.column_types.get(column, "int")
    if column in state.nullable_columns and rnd.random() < 0.20:
        return {"op": "filter", "column": column, "cmp": rnd.choice(["is_null", "is_not_null"]), "value": None}
    if column_type == "bool" and rnd.random() < 0.35:
        return {
            "op": "filter",
            "column": column,
            "cmp": rnd.choice(["bool_is_true", "bool_is_false", "bool_is_not_unknown"]),
            "value": None,
        }
    if column_type == "str" and rnd.random() < 0.30:
        return {
            "op": "filter",
            "column": column,
            "cmp": rnd.choice(["str_contains", "str_starts_with", "str_ends_with"]),
            "value": rnd.choice(["a", "A", "space", "pad", "0"]),
        }
    comparators = ["==", "!="] if column_type in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
    return {
        "op": "filter",
        "column": column,
        "cmp": rnd.choice(comparators),
        "value": _literal_for_state_column(rnd, state, context.primary_table, column),
    }


def _gen_drop_nulls(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    columns = canonical_candidates(state.nullable_columns & state.available) or canonical_candidates(state.columns)
    width = rnd.randint(1, min(3, len(columns)))
    return {"op": "drop_nulls", "columns": sorted(rnd.sample(columns, width))}


def _gen_select(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    columns = canonical_candidates(state.columns)
    width = rnd.randint(1, len(columns))
    return {"op": "select", "columns": sorted(rnd.sample(columns, width))}


def _gen_distinct(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    columns = canonical_candidates(state.columns)
    width = rnd.randint(1, min(3, len(columns)))
    return {"op": "distinct", "columns": sorted(rnd.sample(columns, width))}


def _gen_fill_null(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    column = rnd.choice(
        canonical_candidates(state.nullable_columns & state.available)
        or canonical_candidates(state.columns)
    )
    return {"op": "fill_null", "column": column, "value": _literal_for_type(rnd, state.column_types.get(column, "int"))}


def _has_coalesce_group(state: TypedProgramState) -> bool:
    return any(len(columns) >= 2 for _type, columns in _same_type_column_groups(state))


def _gen_coalesce(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    output_type, columns = rnd.choice(_same_type_column_groups(state))
    width = rnd.randint(2, min(3, len(columns)))
    selected = rnd.sample(columns, width)
    alias = _safe_alias("co", selected[0], state)
    operation: dict[str, Any] = {"op": "coalesce", "columns": selected, "as": alias}
    if rnd.random() < 0.75:
        operation["fallback"] = _literal_for_type(rnd, output_type)
    return operation


def _gen_case_when(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    column = rnd.choice(canonical_candidates(state.comparable))
    column_type = state.column_types.get(column, "int")
    cmp = rnd.choice(["==", "!="] if column_type in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="])
    if column in state.nullable_columns and rnd.random() < 0.25:
        cmp = rnd.choice(["is_null", "is_not_null"])
        value = None
    else:
        value = _literal_for_state_column(rnd, state, context.primary_table, column)
    output_type = rnd.choice(["str", "int", "bool"])
    then_value, else_value = _case_literals(rnd, output_type)
    return {
        "op": "case_when",
        "as": _safe_alias("cw", str(len(state.operations_so_far)), state),
        "condition": {"column": column, "cmp": cmp, "value": value},
        "then": then_value,
        "else": else_value,
    }


def _has_mutate_source(state: TypedProgramState) -> bool:
    return bool(state.numeric or state.strings or state.booleans)


def _gen_mutate(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    choices: list[str] = []
    if state.numeric:
        choices.extend(["arith_const", "abs", "cast_float", "cast_string"])
    if state.strings:
        choices.extend(["string_length", "string_lower", "string_upper", "string_contains"])
    if state.booleans:
        choices.append("bool_not")
    kind = rnd.choice(choices)
    if kind == "arith_const":
        source = rnd.choice(canonical_candidates(state.numeric))
        expr = {
            "kind": "arith_const",
            "source": source,
            "op": rnd.choice(["sub", "mul", "div"]),
            "value": rnd.choice([2, 3, 5, 10]),
        }
    elif kind == "abs":
        expr = {"kind": "abs", "source": rnd.choice(canonical_candidates(state.numeric))}
    elif kind == "cast_float":
        expr = {"kind": "cast", "source": rnd.choice(canonical_candidates(state.numeric)), "to": "float"}
    elif kind == "cast_string":
        expr = {"kind": "cast", "source": rnd.choice(canonical_candidates(state.numeric)), "to": "str"}
    elif kind == "string_length":
        expr = {"kind": "string_length", "source": rnd.choice(canonical_candidates(state.strings))}
    elif kind == "string_lower":
        expr = {"kind": "string_lower", "source": rnd.choice(canonical_candidates(state.strings))}
    elif kind == "string_upper":
        expr = {"kind": "string_upper", "source": rnd.choice(canonical_candidates(state.strings))}
    elif kind == "string_contains":
        expr = {"kind": "string_contains", "source": rnd.choice(canonical_candidates(state.strings)), "needle": rnd.choice(["a", "A", "0"])}
    else:
        expr = {"kind": "bool_not", "source": rnd.choice(canonical_candidates(state.booleans))}
    return {"op": "mutate", "column": _safe_alias("m", str(len(state.operations_so_far)), state), "expr": expr}


def _gen_sort(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    columns = canonical_candidates(state.comparable or state.columns)
    first = rnd.choice(columns)
    width = rnd.randint(1, min(3, len(columns)))
    selected = unique_preserve_order([first, *rnd.sample([column for column in columns if column != first], max(0, width - 1))])
    return {
        "op": "sort",
        "keys": [
            {"column": column, "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])}
            for column in selected
        ],
    }


def _gen_limit(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    return {"op": "limit", "n": rnd.randint(0, max(1, state.row_count_estimate + 2))}


def _gen_offset(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    return {"op": "offset", "n": rnd.randint(0, max(1, state.row_count_estimate + 2))}


def _can_groupby(state: TypedProgramState) -> bool:
    return bool(not state.is_grouped and state.columns and (state.numeric or state.booleans))


def _gen_groupby(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    key_candidates = canonical_candidates(
        column for column in state.columns if state.column_types.get(column) in {"int", "str", "bool"}
    )
    keys = sorted(rnd.sample(key_candidates or canonical_candidates(state.columns), 1))
    aggregate_candidates = canonical_candidates(state.numeric or state.booleans)
    width = rnd.randint(1, min(3, len(aggregate_candidates)))
    used_aliases = set(keys)
    aggs: list[dict[str, Any]] = []
    for column in rnd.sample(aggregate_candidates, width):
        funcs = ["sum", "mean", "min", "max", "count", "nunique"]
        if state.column_types.get(column) == "bool":
            funcs = ["any", "all", "min", "max", "count", "nunique"]
        func = rnd.choice(funcs)
        alias = make_safe_output_name(f"{func}_{column}", used=used_aliases)
        used_aliases.add(alias)
        aggs.append({"column": column, "func": func, "as": alias})
    return {"op": "groupby", "keys": keys, "aggs": aggs}


def _gen_join(rnd: random.Random, state: TypedProgramState, context: SynthesisContext) -> dict[str, Any]:
    table = rnd.choice(canonical_candidates(state.compatible_join_tables(), key=lambda value: value.name))
    pairs = canonical_candidates(compatible_join_key_pairs(state, table))
    left, right = rnd.choice(pairs)
    return {
        "op": "join",
        "table": table.name,
        "left_on": left,
        "right_on": right,
        "how": rnd.choice(["inner", "left"]),
    }


def _same_type_column_groups(state: TypedProgramState) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for column in state.columns:
        column_type = state.column_types.get(column)
        if column_type:
            groups.setdefault(column_type, []).append(column)
    return [
        (column_type, canonical_candidates(columns))
        for column_type, columns in sorted(groups.items())
        if len(columns) >= 2
    ]


def _safe_alias(prefix: str, suffix: str, state: TypedProgramState) -> str:
    return make_safe_output_name(f"{prefix}_{suffix}", used=set(state.columns))


def _literal_for_state_column(
    rnd: random.Random,
    state: TypedProgramState,
    table: TableData,
    column: str,
) -> Any:
    values = [row.get(column) for row in table.rows if row.get(column) is not None]
    if values and rnd.random() < 0.65:
        value = rnd.choice(values)
        if isinstance(value, float) and value != value:
            return 0.0
        return value
    return _literal_for_type(rnd, state.column_types.get(column, "int"))


def _literal_for_type(rnd: random.Random, column_type: str | None) -> Any:
    if column_type == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if column_type == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if column_type == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "space value", "missing"])


def _case_literals(rnd: random.Random, output_type: str) -> tuple[Any, Any]:
    if output_type == "int":
        return rnd.choice([0, 1, 2]), rnd.choice([-1, 0, 1])
    if output_type == "bool":
        return True, False
    return rnd.choice(["hit", "yes", "A"]), rnd.choice(["miss", "no", "B"])
