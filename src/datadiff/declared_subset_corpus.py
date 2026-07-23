from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.targets import target_context


OperationBuilder = Callable[[int], list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class DeclaredSubsetTemplate:
    name: str
    required_capabilities: frozenset[str]
    build_operations: OperationBuilder


def _filter_operations(column: str, comparator: str, value: Any) -> list[dict[str, Any]]:
    selected = ["id"] if column == "id" else ["id", column]
    return [
        {"op": "filter", "column": column, "cmp": comparator, "value": value},
        {"op": "select", "columns": selected},
    ]


def _mutate_operations(output: str, expression: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"op": "mutate", "column": output, "expr": expression},
        {"op": "select", "columns": ["id", output]},
    ]


DECLARED_SUBSET_TEMPLATES: tuple[DeclaredSubsetTemplate, ...] = (
    DeclaredSubsetTemplate(
        "projection",
        frozenset({"table:single", "op:select", "type:int", "type:str"}),
        lambda _index: [{"op": "select", "columns": ["id", "label"]}],
    ),
    DeclaredSubsetTemplate(
        "numeric_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:int", "type:float"}),
        lambda index: [
            {"op": "filter", "column": "id", "cmp": ">=", "value": index % 5},
            {"op": "select", "columns": ["id", "value"]},
        ],
    ),
    DeclaredSubsetTemplate(
        "set_membership_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:int"}),
        lambda index: _filter_operations("id", "in_set", [index % 4, (index + 2) % 7]),
    ),
    DeclaredSubsetTemplate(
        "set_non_membership_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:int"}),
        lambda index: _filter_operations("id", "not_in_set", [index % 3, (index + 1) % 5]),
    ),
    DeclaredSubsetTemplate(
        "null_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:float", "nulls"}),
        lambda _index: _filter_operations("value", "is_null", None),
    ),
    DeclaredSubsetTemplate(
        "not_null_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:str", "nulls"}),
        lambda _index: _filter_operations("label", "is_not_null", None),
    ),
    DeclaredSubsetTemplate(
        "closed_range_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:float"}),
        lambda index: _filter_operations("value", "range_closed", [-2.5, float(1 + index % 4)]),
    ),
    DeclaredSubsetTemplate(
        "string_contains_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:str"}),
        lambda _index: _filter_operations("label", "str_contains", "a"),
    ),
    DeclaredSubsetTemplate(
        "string_starts_with_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:str"}),
        lambda _index: _filter_operations("label", "str_starts_with", " "),
    ),
    DeclaredSubsetTemplate(
        "string_ends_with_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:str"}),
        lambda _index: _filter_operations("label", "str_ends_with", " "),
    ),
    *(
        DeclaredSubsetTemplate(
            f"boolean_{truth_test}_filter",
            frozenset({"table:single", "op:filter", "op:select", "type:bool", "nulls"}),
            lambda _index, comparator=f"bool_{truth_test}": _filter_operations(
                "flag",
                comparator,
                None,
            ),
        )
        for truth_test in (
            "is_true",
            "is_not_true",
            "is_false",
            "is_not_false",
            "is_unknown",
            "is_not_unknown",
        )
    ),
    DeclaredSubsetTemplate(
        "comparison_truth_filter",
        frozenset({"table:single", "op:filter", "op:select", "type:float", "nulls"}),
        lambda _index: _filter_operations("value", "gt_is_unknown", 0.0),
    ),
    DeclaredSubsetTemplate(
        "add_constant",
        frozenset({"table:single", "op:mutate", "op:select", "expr:add_const", "type:float"}),
        lambda index: _mutate_operations(
            "value_plus",
            {"kind": "add_const", "source": "value", "value": (index % 5) - 2},
        ),
    ),
    *(
        DeclaredSubsetTemplate(
            f"arithmetic_{operator}",
            frozenset({"table:single", "op:mutate", "op:select", "expr:arith_const", "type:float"}),
            lambda index, current_operator=operator: _mutate_operations(
                f"value_{current_operator}",
                {
                    "kind": "arith_const",
                    "source": "value",
                    "op": current_operator,
                    "value": 2 + index % 3,
                },
            ),
        )
        for operator in ("sub", "mul", "div", "mod")
    ),
    DeclaredSubsetTemplate(
        "numeric_abs",
        frozenset({"table:single", "op:mutate", "op:select", "expr:abs", "type:float"}),
        lambda _index: [
            {
                "op": "mutate",
                "column": "magnitude",
                "expr": {"kind": "abs", "source": "value"},
            },
            {"op": "select", "columns": ["id", "magnitude"]},
        ],
    ),
    DeclaredSubsetTemplate(
        "boolean_not",
        frozenset({"table:single", "op:mutate", "op:select", "expr:bool_not", "type:bool", "nulls"}),
        lambda _index: _mutate_operations(
            "not_flag",
            {"kind": "bool_not", "source": "flag"},
        ),
    ),
    DeclaredSubsetTemplate(
        "bounded_slice",
        frozenset({"table:single", "op:offset", "op:limit", "op:select", "type:int", "type:str"}),
        lambda index: [
            {"op": "offset", "n": index % 3},
            {"op": "limit", "n": 3},
            {"op": "select", "columns": ["id", "label"]},
        ],
    ),
    DeclaredSubsetTemplate(
        "string_lower",
        frozenset({"table:single", "op:mutate", "op:select", "expr:string_lower", "type:str"}),
        lambda _index: _mutate_operations(
            "lowered",
            {"kind": "string_lower", "source": "label"},
        ),
    ),
    DeclaredSubsetTemplate(
        "string_upper",
        frozenset({"table:single", "op:mutate", "op:select", "expr:string_upper", "type:str"}),
        lambda _index: _mutate_operations(
            "uppered",
            {"kind": "string_upper", "source": "label"},
        ),
    ),
    DeclaredSubsetTemplate(
        "string_strip",
        frozenset({"table:single", "op:mutate", "op:select", "expr:string_strip", "type:str"}),
        lambda _index: [
            {
                "op": "mutate",
                "column": "clean",
                "expr": {"kind": "string_strip", "source": "label"},
            },
            {"op": "select", "columns": ["id", "clean"]},
        ],
    ),
)


def applicable_declared_subset_templates(
    backends: Sequence[str],
) -> tuple[DeclaredSubsetTemplate, ...]:
    common_capabilities = set(target_context(backends).common_capabilities)
    applicable = tuple(
        template
        for template in DECLARED_SUBSET_TEMPLATES
        if template.required_capabilities <= common_capabilities
    )
    if not applicable:
        raise ValueError(
            "selected targets have no executable declared-subset template: "
            + ", ".join(backends)
        )
    return applicable


def build_declared_subset_case(
    *,
    seed: int,
    index: int,
    backends: Sequence[str],
) -> Case:
    templates = applicable_declared_subset_templates(backends)
    template = templates[index % len(templates)]
    label_values = (
        " alpha ",
        "BETA ",
        " a ",
        "delta ",
        " mixed Value ",
    )
    row_count = 6 + seed % 5
    rows = []
    for row_id in range(row_count):
        value = None
        if (row_id + seed) % 5 != 0:
            scale = 0.5 + (seed % 7) / 10.0
            value = round((row_id - row_count / 2) * scale, 3)
        label = None
        if (row_id * 3 + seed) % 7 != 0:
            base = label_values[(row_id + seed) % len(label_values)]
            label = f"{base}{seed % 97}-{row_id} "
        flag = None if (row_id + seed) % 4 == 0 else (row_id + seed) % 2 == 0
        rows.append(
            {
                "id": row_id,
                "value": value,
                "label": label,
                "flag": flag,
            }
        )
    return Case(
        f"declared-subset-{index:04d}",
        seed,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("value", "float"),
                    ColumnSpec("label", "str"),
                    ColumnSpec("flag", "bool"),
                ],
                rows,
            )
        ],
        Program(
            f"declared-subset-program-{index:04d}",
            seed,
            template.build_operations(index),
        ),
        metadata={
            "corpus": "declared_subset",
            "template": template.name,
            "required_capabilities": sorted(template.required_capabilities),
            "selected_backends": list(backends),
        },
    )
