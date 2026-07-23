from __future__ import annotations

import copy
from collections import UserDict
from dataclasses import asdict, dataclass, field
from keyword import kwlist as PYTHON_KEYWORDS
from typing import Any, Literal, Mapping

ColumnType = Literal["int", "float", "bool", "str"]
SortNulls = Literal["first", "last"]
Value = Any


@dataclass(slots=True)
class ColumnSpec:
    name: str
    type: ColumnType
    nullable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnSpec":
        return cls(**data)


@dataclass(slots=True)
class TableData:
    name: str
    columns: list[ColumnSpec]
    rows: list[dict[str, Value]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableData":
        return cls(
            name=data["name"],
            columns=[ColumnSpec.from_dict(c) for c in data["columns"]],
            rows=data["rows"],
        )

    def column_type(self, name: str) -> ColumnType:
        for c in self.columns:
            if c.name == name:
                return c.type
        raise KeyError(name)

    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.type in {"int", "float"}]

    def comparable_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.type in {"int", "float", "str", "bool"}]


@dataclass(slots=True)
class Program:
    program_id: str
    seed: int
    operations: list["Operation"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.operations = [coerce_operation(operation) for operation in self.operations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "seed": self.seed,
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        return cls(**data)

    @property
    def order_sensitive(self) -> bool:
        from datadiff.operation_semantics import op_kind

        # Internal order observers affect which values a program computes even
        # when a later operation (for example, a join or groupby) destroys the
        # final row order.  Keep this broad predicate for mutation/metamorphic
        # safety and use ``output_order_sensitive`` when comparing final rows.
        if any(op_kind(op) in {"running_sum", "row_number_filter", "sortedness_check"} for op in self.operations):
            return True
        return self.output_order_sensitive

    @property
    def output_order_sensitive(self) -> bool:
        """Whether the final operation chain defines observable row order."""

        from datadiff.operation_semantics import op_kind

        row_order_preserving = {
            "filter",
            "tuple_absence_filter",
            "drop_nulls",
            "semi_join",
            "anti_join",
            "coalesce",
            "select",
            "fill_null",
            "case_when",
            "mutate",
            "limit",
            "offset",
        }
        for op in reversed(self.operations):
            kind = op_kind(op)
            if kind in {"sort", "running_sum", "row_number_filter", "sortedness_check"}:
                return True
            if kind in row_order_preserving:
                continue
            return False
        return False

    def op_sequence(self) -> list[str]:
        from datadiff.operation_semantics import operation_names

        return operation_names(self.operations, default="unknown")


@dataclass(frozen=True, slots=True)
class SortKey:
    column: str
    ascending: bool = True
    nulls: SortNulls = "last"

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "ascending": self.ascending, "nulls": self.nulls}


class IRNode(UserDict):
    _child_roles: dict[str, type["IRNode"]] = {}
    _list_child_roles: dict[str, type["IRNode"]] = {}
    _reserved_property_names = frozenset(
        {
            *[name for name in dir(dict) if not name.startswith("_")],
            *[name for name in dir(UserDict) if not name.startswith("_")],
            *PYTHON_KEYWORDS,
        }
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        conflicts = sorted(
            name
            for name, value in cls.__dict__.items()
            if isinstance(value, property) and name in cls._reserved_property_names
        )
        if conflicts:
            names = ", ".join(conflicts)
            raise TypeError(
                f"{cls.__name__} defines reserved IR property names: {names}. "
                "Use semantic aliases that do not shadow mapping methods or Python keywords."
            )

    def __init__(self, initialdata: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
        payload: dict[str, Any] = {}
        if initialdata is not None:
            payload.update(dict(initialdata))
        if kwargs:
            payload.update(kwargs)
        super().__init__()
        for key, value in payload.items():
            self.data[str(key)] = self._coerce_key_value(str(key), value)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[str(key)] = self._coerce_key_value(str(key), value)

    def _coerce_key_value(self, key: str, value: Any) -> Any:
        if isinstance(value, IRNode):
            return value
        role = self._child_roles.get(key)
        if role is not None:
            return _coerce_typed_child(role, value)
        list_role = self._list_child_roles.get(key)
        if list_role is not None and isinstance(value, list):
            return [_coerce_typed_child(list_role, item) for item in value]
        if key in {"keys", "order_by"} and _looks_like_sort_key_list(value):
            return [_coerce_typed_child(SortKeySpec, item) for item in value]
        if isinstance(value, Mapping):
            return IRNode(value)
        if isinstance(value, list):
            return [self._coerce_list_item(item) for item in value]
        return value

    def _coerce_list_item(self, value: Any) -> Any:
        if isinstance(value, IRNode):
            return value
        if isinstance(value, Mapping):
            return IRNode(value)
        if isinstance(value, list):
            return [self._coerce_list_item(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {key: _unwrap_ir_value(value) for key, value in self.data.items()}

    def copy(self) -> "IRNode":
        return type(self)(self.to_dict())

    def __deepcopy__(self, memo: dict[int, Any]) -> "IRNode":
        node = type(self)(copy.deepcopy(self.to_dict(), memo))
        memo[id(self)] = node
        return node


class Expression(IRNode):
    @property
    def kind(self) -> str:
        return str(self.get("kind", ""))

    @property
    def source(self) -> str:
        return str(self.get("source", ""))


class StringLowerExpr(Expression):
    pass


class StringUpperExpr(Expression):
    pass


class StringStripExpr(Expression):
    pass


class StringReplaceExpr(Expression):
    @property
    def old(self) -> Any:
        return self.get("old")

    @property
    def new(self) -> Any:
        return self.get("new")


class StringSliceExpr(Expression):
    @property
    def start(self) -> Any:
        return self.get("start")

    @property
    def length(self) -> Any:
        return self.get("length")


class StringSplitPartExpr(Expression):
    @property
    def separator(self) -> Any:
        return self.get("sep")

    @property
    def index(self) -> Any:
        return self.get("index")


class StringBasenameExpr(Expression):
    pass


class StringConcatExpr(Expression):
    @property
    def other(self) -> str:
        return str(self.get("other", ""))

    @property
    def separator(self) -> str:
        return str(self.get("sep", ""))


class StringContainsExpr(Expression):
    @property
    def needle(self) -> Any:
        return self.get("needle")


class StringStartsWithExpr(Expression):
    @property
    def needle(self) -> Any:
        return self.get("needle")


class StringEndsWithExpr(Expression):
    @property
    def needle(self) -> Any:
        return self.get("needle")


class StringLengthExpr(Expression):
    pass


class StringNullIfEmptyExpr(Expression):
    pass


class AddConstExpr(Expression):
    @property
    def value(self) -> Any:
        return self.get("value")


class ArithConstExpr(Expression):
    @property
    def operator(self) -> str:
        return str(self.get("op", ""))

    @property
    def value(self) -> Any:
        return self.get("value")


class CastExpr(Expression):
    @property
    def target_type(self) -> str:
        return str(self.get("to", ""))

    @property
    def input_domain(self) -> str:
        return str(self.get("input_domain", ""))


class AbsExpr(Expression):
    pass


class ClipExpr(Expression):
    @property
    def lower(self) -> Any:
        return self.get("lower")

    @property
    def upper(self) -> Any:
        return self.get("upper")


class BoolNotExpr(Expression):
    pass


class DatePartExpr(Expression):
    @property
    def part(self) -> str:
        return str(self.get("part", ""))


class ReverseDivisionColumnsExpr(Expression):
    @property
    def numerator(self) -> str:
        return str(self.get("numerator", ""))


class Condition(IRNode):
    @property
    def comparator(self) -> str:
        return str(self.get("cmp", ""))

    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def value(self) -> Any:
        return self.get("value")


class BooleanPredicateCondition(Condition):
    pass


class AggregateSpec(IRNode):
    @property
    def func(self) -> str:
        return str(self.get("func", ""))

    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def alias(self) -> str:
        return str(self.get("as", ""))

    @property
    def output_name(self) -> str:
        return self.alias


class SortKeySpec(IRNode):
    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def ascending(self) -> bool:
        value = self.get("ascending", True)
        return value if isinstance(value, bool) else bool(value)

    @property
    def nulls(self) -> SortNulls:
        value = str(self.get("nulls", "last"))
        return "first" if value == "first" else "last"


def _normalized_column_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    column = str(value or "")
    return [column] if column else []


class Operation(IRNode):
    _child_roles = {
        "expr": Expression,
        "condition": Condition,
    }
    _list_child_roles = {
        "aggs": AggregateSpec,
    }

    @property
    def kind(self) -> str:
        return str(self.get("op", ""))

    @property
    def output_alias(self) -> str:
        return str(self.get("as", ""))

    @property
    def table(self) -> str:
        return str(self.get("table", ""))


class FilterOp(Operation):
    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def comparator(self) -> str:
        return str(self.get("cmp", ""))


class TupleAbsenceFilterOp(Operation):
    @property
    def columns(self) -> list[str]:
        return _normalized_column_list(self.get("columns", []))

    @property
    def right_columns(self) -> list[str]:
        return _normalized_column_list(self.get("right_columns", []))


class UnionAllOp(Operation):
    pass


class DropNullsOp(Operation):
    @property
    def columns(self) -> list[str]:
        return _normalized_column_list(self.get("columns", []))


class FillNullOp(Operation):
    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def value(self) -> Any:
        return self.get("value")


class CoalesceOp(Operation):
    @property
    def sources(self) -> list[str]:
        return _normalized_column_list(self.get("columns", []))

    @property
    def fallback(self) -> Any:
        return self.get("fallback")


class CaseWhenOp(Operation):
    _child_roles = {
        **Operation._child_roles,
        "condition": Condition,
    }

    @property
    def then_value(self) -> Any:
        return self.get("then")

    @property
    def else_value(self) -> Any:
        return self.get("else")


class MutateOp(Operation):
    _child_roles = {
        **Operation._child_roles,
        "expr": Expression,
    }

    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def expression(self) -> Expression | None:
        expr = self.get("expr")
        return expr if isinstance(expr, Expression) else None


class RowNumberFilterOp(Operation):
    @property
    def partition_columns(self) -> list[str]:
        return _normalized_column_list(self.get("partition_by", []))

    @property
    def order_keys(self) -> list[SortKey]:
        return normalize_sort_keys({"keys": self.get("order_by", [])})

    @property
    def comparator(self) -> str:
        return str(self.get("cmp", ""))

    @property
    def value(self) -> Any:
        return self.get("value")


class RunningSumOp(Operation):
    @property
    def source(self) -> str:
        return str(self.get("source", ""))

    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def partition_columns(self) -> list[str]:
        return _normalized_column_list(self.get("partition_by", []))

    @property
    def order_keys(self) -> list[SortKey]:
        return normalize_sort_keys({"keys": self.get("order_by", [])})


class SortednessCheckOp(Operation):
    @property
    def column(self) -> str:
        return str(self.get("column", ""))

    @property
    def ascending(self) -> Any:
        return self.get("ascending", True)

    @property
    def nulls(self) -> str:
        return str(self.get("nulls", "last"))


class ProbeOp(Operation):
    @property
    def rows(self) -> Any:
        return self.get("rows")

    @property
    def branches(self) -> Any:
        return self.get("branches")

    @property
    def probe_values(self) -> list[Any]:
        value = self.get("values", [])
        return list(value) if isinstance(value, (list, tuple)) else ([] if value is None else [value])

    @property
    def quantiles(self) -> list[Any]:
        value = self.get("quantiles", [])
        return list(value) if isinstance(value, (list, tuple)) else ([] if value is None else [value])

    @property
    def literal(self) -> Any:
        return self.get("literal")


class JoinLikeOp(Operation):
    @property
    def left_keys(self) -> list[str]:
        return _normalized_column_list(self.get("left_on"))

    @property
    def right_keys(self) -> list[str]:
        return _normalized_column_list(self.get("right_on"))


class JoinOp(JoinLikeOp):
    @property
    def how(self) -> str:
        return str(self.get("how", ""))


class SemiJoinOp(JoinLikeOp):
    pass


class AntiJoinOp(JoinLikeOp):
    pass


class DistinctOp(Operation):
    @property
    def columns(self) -> list[str]:
        return _normalized_column_list(self.get("columns", []))


class SelectOp(Operation):
    @property
    def columns(self) -> list[str]:
        return _normalized_column_list(self.get("columns", []))


class GroupByOp(Operation):
    _list_child_roles = {
        **Operation._list_child_roles,
        "aggs": AggregateSpec,
    }

    @property
    def group_keys(self) -> list[str]:
        return [str(key) for key in self.get("keys", [])]

    @property
    def aggregates(self) -> list[AggregateSpec]:
        return [agg for agg in self.get("aggs", []) if isinstance(agg, AggregateSpec)]

    @property
    def aggregate_aliases(self) -> list[str]:
        return [agg.alias for agg in self.aggregates if agg.alias]


class AggregateOp(Operation):
    _list_child_roles = {
        **Operation._list_child_roles,
        "aggs": AggregateSpec,
    }

    @property
    def aggregates(self) -> list[AggregateSpec]:
        return [agg for agg in self.get("aggs", []) if isinstance(agg, AggregateSpec)]

    @property
    def aggregate_aliases(self) -> list[str]:
        return [agg.alias for agg in self.aggregates if agg.alias]

    @property
    def columns(self) -> list[str]:
        return [agg.column for agg in self.aggregates if agg.column]


class SortOp(Operation):
    @property
    def sort_keys(self) -> list[SortKey]:
        return normalize_sort_keys(self)


class LimitOp(Operation):
    @property
    def n(self) -> Any:
        return self.get("n")


class OffsetOp(Operation):
    @property
    def n(self) -> Any:
        return self.get("n")


EXPRESSION_NODE_TYPES: dict[str, type[Expression]] = {
    "string_lower": StringLowerExpr,
    "string_upper": StringUpperExpr,
    "string_strip": StringStripExpr,
    "string_replace": StringReplaceExpr,
    "string_slice": StringSliceExpr,
    "string_split_part": StringSplitPartExpr,
    "string_basename": StringBasenameExpr,
    "string_concat": StringConcatExpr,
    "string_contains": StringContainsExpr,
    "string_starts_with": StringStartsWithExpr,
    "string_ends_with": StringEndsWithExpr,
    "string_length": StringLengthExpr,
    "string_null_if_empty": StringNullIfEmptyExpr,
    "add_const": AddConstExpr,
    "arith_const": ArithConstExpr,
    "cast": CastExpr,
    "abs": AbsExpr,
    "clip": ClipExpr,
    "bool_not": BoolNotExpr,
    "date_part": DatePartExpr,
    "reverse_division_columns": ReverseDivisionColumnsExpr,
}


OPERATION_NODE_TYPES: dict[str, type[Operation]] = {
    "filter": FilterOp,
    "tuple_absence_filter": TupleAbsenceFilterOp,
    "union_all": UnionAllOp,
    "drop_nulls": DropNullsOp,
    "fill_null": FillNullOp,
    "coalesce": CoalesceOp,
    "case_when": CaseWhenOp,
    "mutate": MutateOp,
    "row_number_filter": RowNumberFilterOp,
    "running_sum": RunningSumOp,
    "sortedness_check": SortednessCheckOp,
    "random_case_probe": ProbeOp,
    "group_quantile_probe": ProbeOp,
    "scalar_subquery_probe": ProbeOp,
    "window_avg_probe": ProbeOp,
    "struct_distinct_probe": ProbeOp,
    "bit_compare_probe": ProbeOp,
    "round_even_probe": ProbeOp,
    "float_literal_precision_probe": ProbeOp,
    "timestamp_precision_filter_probe": ProbeOp,
    "series_rtruediv_probe": ProbeOp,
    "series_reflected_arithmetic_probe": ProbeOp,
    "datafusion_grouped_null_topk_probe": ProbeOp,
    "confirmed_root_witness_probe": ProbeOp,
    "uint64_isin_probe": ProbeOp,
    "tuple_anti_null_probe": ProbeOp,
    "setop_all_duplicate_probe": ProbeOp,
    "json_predicate_order_probe": ProbeOp,
    "sparse_mask_probe": ProbeOp,
    "float_wrap_probe": ProbeOp,
    "index_bool_probe": ProbeOp,
    "empty_literal_groupby_probe": ProbeOp,
    "arrow_string_eq_sum_probe": ProbeOp,
    "arrow_timestamp_loc_slice_probe": ProbeOp,
    "arrow_timestamp_index_attr_probe": ProbeOp,
    "eval_inplace_alias_probe": ProbeOp,
    "bool_reduction_skipna_probe": ProbeOp,
    "dataset_isin_all_match_probe": ProbeOp,
    "run_end_null_compute_probe": ProbeOp,
    "large_string_partition_probe": ProbeOp,
    "hash_pivot_wider_probe": ProbeOp,
    "list_flatten_parent_indices_probe": ProbeOp,
    "rolling_mean_by_null_count_probe": ProbeOp,
    "csv_long_numeric_roundtrip_probe": ProbeOp,
    "join": JoinOp,
    "semi_join": SemiJoinOp,
    "anti_join": AntiJoinOp,
    "distinct": DistinctOp,
    "select": SelectOp,
    "groupby": GroupByOp,
    "aggregate": AggregateOp,
    "sort": SortOp,
    "limit": LimitOp,
    "offset": OffsetOp,
}


def _looks_like_sort_key_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, Mapping) and "column" in item
        for item in value
    )


def _unwrap_ir_value(value: Any) -> Any:
    if isinstance(value, IRNode):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap_ir_value(item) for item in value]
    return value


def _coerce_typed_child(role: type[IRNode], value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if issubclass(role, Expression):
        return coerce_expression(value)
    if issubclass(role, Condition):
        return coerce_condition(value)
    if issubclass(role, AggregateSpec):
        return AggregateSpec(value)
    if issubclass(role, SortKeySpec):
        return SortKeySpec(value)
    return role(value)


def coerce_expression(expression: Mapping[str, Any] | Expression) -> Expression:
    if isinstance(expression, Expression):
        return expression
    if not isinstance(expression, Mapping):
        raise TypeError(f"expression must be a mapping, got {type(expression).__name__}")
    kind = str(expression.get("kind", ""))
    expr_type = EXPRESSION_NODE_TYPES.get(kind, Expression)
    return expr_type(expression)


def coerce_condition(condition: Mapping[str, Any] | Condition) -> Condition:
    if isinstance(condition, Condition):
        return condition
    if not isinstance(condition, Mapping):
        raise TypeError(f"condition must be a mapping, got {type(condition).__name__}")
    cmp = str(condition.get("cmp", ""))
    if cmp.startswith("bool_"):
        return BooleanPredicateCondition(condition)
    return Condition(condition)


def coerce_operation(operation: Mapping[str, Any] | Operation) -> Operation:
    if isinstance(operation, Operation):
        return operation
    if not isinstance(operation, Mapping):
        raise TypeError(f"operation must be a mapping, got {type(operation).__name__}")
    payload = dict(operation)
    if not payload.get("op") and payload.get("kind"):
        payload["op"] = payload["kind"]
    kind = str(payload.get("op", ""))
    op_type = OPERATION_NODE_TYPES.get(kind, Operation)
    return op_type(payload)


def normalize_sort_keys(op: Mapping[str, Any]) -> list[SortKey]:
    """Return the canonical per-column sort keys for old and new sort ops."""

    if "keys" in op:
        keys = []
        for item in op.get("keys", []):
            if not isinstance(item, Mapping):
                raise ValueError(f"sort key must be a mapping: {item!r}")
            column = item.get("column")
            ascending = item.get("ascending", True)
            nulls = item.get("nulls", "last")
            if not isinstance(column, str) or not column:
                raise ValueError(f"sort key column must be a non-empty string: {column!r}")
            if not isinstance(ascending, bool):
                raise ValueError(f"sort key ascending must be a boolean: {ascending!r}")
            if nulls not in {"first", "last"}:
                raise ValueError(f"sort key nulls must be 'first' or 'last': {nulls!r}")
            keys.append(SortKey(column=column, ascending=ascending, nulls=nulls))
        return keys

    columns = op.get("columns", [])
    ascending = op.get("ascending", True)
    if not isinstance(ascending, bool):
        raise ValueError(f"sort ascending must be a boolean: {ascending!r}")
    keys = []
    for column in columns:
        if not isinstance(column, str) or not column:
            raise ValueError(f"sort column must be a non-empty string: {column!r}")
        keys.append(SortKey(column=column, ascending=ascending, nulls="last"))
    return keys


def sort_columns(op: Mapping[str, Any]) -> list[str]:
    return [key.column for key in normalize_sort_keys(op)]


@dataclass(slots=True)
class Case:
    case_id: str
    seed: int
    tables: list[TableData]
    program: Program
    metadata: dict[str, Any] = field(default_factory=dict)
    _runtime_cache: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "seed": self.seed,
            "tables": [t.to_dict() for t in self.tables],
            "program": self.program.to_dict(),
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Case":
        return cls(
            case_id=data["case_id"],
            seed=data["seed"],
            tables=[TableData.from_dict(t) for t in data["tables"]],
            program=Program.from_dict(data["program"]),
            metadata=dict(data.get("metadata", {})),
        )
